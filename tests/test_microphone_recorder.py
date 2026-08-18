import wave
from pathlib import Path

import pytest

from src.agent.interview_coordinator import InterviewCoordinator
from src.agent.interviewer_agent import InterviewerDecision, QuestionType
from src.job.job_profile import JobProfile, JobRequirement
from src.planning.interview_plan import InterviewPlan, InterviewTarget
from src.profile.candidate_profile import CandidateProfile
from src.speech.audio_recorder import (
    AudioRecorder,
    InvalidChannelCountError,
    InvalidRecordingDurationError,
    InvalidSampleRateError,
    LocalMicrophoneRecorder,
    MicrophonePermissionError,
    MicrophoneUnavailableError,
    RecordedAudio,
    RecordedAudioFileEmptyError,
    RecordedAudioFileMissingError,
    RecordingBackendError,
)
from src.speech.speech_to_text import (
    CandidateAnswerTranscriber,
    TranscriptionProviderError,
    TranscriptionResult,
)


class FakeCaptureBackend:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def capture(self, *, duration_seconds, sample_rate, channels):
        self.calls.append(
            {
                "duration_seconds": duration_seconds,
                "sample_rate": sample_rate,
                "channels": channels,
            }
        )
        if self.error is not None:
            raise self.error
        frames = round(duration_seconds * sample_rate)
        return b"\x00\x00" * frames * channels


def test_successful_fake_recording_creates_nonempty_wav(tmp_path):
    backend = FakeCaptureBackend()
    output = tmp_path / "candidate.wav"
    recorder = LocalMicrophoneRecorder(
        backend=backend,
        default_duration_seconds=1.25,
        sample_rate=16_000,
        channels=1,
    )

    recording = recorder.record(output_path=output)

    assert isinstance(recorder, AudioRecorder)
    assert recording == RecordedAudio(
        path=output,
        format="wav",
        duration_seconds=1.25,
        sample_rate=16_000,
        channels=1,
    )
    assert output.is_file()
    assert output.stat().st_size > 44
    with wave.open(str(output), "rb") as audio:
        assert audio.getframerate() == 16_000
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2


def test_duration_sample_rate_and_channels_are_configurable(tmp_path):
    backend = FakeCaptureBackend()
    recorder = LocalMicrophoneRecorder(
        backend=backend,
        default_duration_seconds=2,
        max_duration_seconds=10,
        sample_rate=44_100,
        channels=2,
    )

    recording = recorder.record(
        duration_seconds=0.5,
        output_path=tmp_path / "stereo.wav",
    )

    assert recording.duration_seconds == 0.5
    assert recording.sample_rate == 44_100
    assert recording.channels == 2
    assert backend.calls == [
        {
            "duration_seconds": 0.5,
            "sample_rate": 44_100,
            "channels": 2,
        }
    ]


@pytest.mark.parametrize("duration", [0, -1, -0.1])
def test_nonpositive_duration_is_rejected(duration):
    recorder = LocalMicrophoneRecorder(backend=FakeCaptureBackend())

    with pytest.raises(InvalidRecordingDurationError):
        recorder.record(duration_seconds=duration)


def test_duration_above_maximum_is_rejected():
    recorder = LocalMicrophoneRecorder(
        backend=FakeCaptureBackend(),
        default_duration_seconds=5,
        max_duration_seconds=10,
    )

    with pytest.raises(InvalidRecordingDurationError, match="cannot exceed 10"):
        recorder.record(duration_seconds=10.1)


@pytest.mark.parametrize("sample_rate", [0, -1, 16_000.5, True])
def test_invalid_sample_rate_is_rejected(sample_rate):
    with pytest.raises(InvalidSampleRateError):
        LocalMicrophoneRecorder(
            backend=FakeCaptureBackend(),
            sample_rate=sample_rate,
        )


@pytest.mark.parametrize("channels", [0, -1, 1.5, True])
def test_invalid_channel_count_is_rejected(channels):
    with pytest.raises(InvalidChannelCountError):
        LocalMicrophoneRecorder(
            backend=FakeCaptureBackend(),
            channels=channels,
        )


def test_backend_initialization_failure_is_wrapped_with_cause():
    original = RuntimeError("PortAudio initialization failed")

    def fail():
        raise original

    recorder = LocalMicrophoneRecorder(backend_factory=fail)

    with pytest.raises(MicrophoneUnavailableError) as captured:
        recorder.record(duration_seconds=0.1)

    assert captured.value.__cause__ is original


def test_backend_recording_failure_is_wrapped_with_cause():
    original = RuntimeError("input device disconnected")
    recorder = LocalMicrophoneRecorder(
        backend=FakeCaptureBackend(error=original)
    )

    with pytest.raises(RecordingBackendError) as captured:
        recorder.record(duration_seconds=0.1)

    assert captured.value.__cause__ is original


def test_microphone_permission_failure_is_wrapped_with_cause():
    original = PermissionError("microphone permission denied")
    recorder = LocalMicrophoneRecorder(
        backend=FakeCaptureBackend(error=original)
    )

    with pytest.raises(MicrophonePermissionError) as captured:
        recorder.record(duration_seconds=0.1)

    assert captured.value.__cause__ is original


class MissingFileRecorder(LocalMicrophoneRecorder):
    def _write_wav(self, path, pcm_data):
        path.unlink(missing_ok=True)


class EmptyFileRecorder(LocalMicrophoneRecorder):
    def _write_wav(self, path, pcm_data):
        path.write_bytes(b"")


def test_missing_generated_file_is_rejected():
    recorder = MissingFileRecorder(backend=FakeCaptureBackend())

    with pytest.raises(RecordedAudioFileMissingError):
        recorder.record(duration_seconds=0.1)


def test_empty_generated_file_is_rejected():
    recorder = EmptyFileRecorder(backend=FakeCaptureBackend())

    with pytest.raises(RecordedAudioFileEmptyError):
        recorder.record(duration_seconds=0.1)


def test_cleanup_removes_temporary_recording():
    recording = LocalMicrophoneRecorder(
        backend=FakeCaptureBackend()
    ).record(duration_seconds=0.1)

    assert recording.path.exists()
    recording.cleanup()
    assert not recording.path.exists()
    recording.cleanup()


class FakeAudioRecorder:
    def __init__(self, recording):
        self.recording = recording

    def record(self, duration_seconds=None, output_path=None):
        return self.recording


def test_fake_recorder_dependency_injection(tmp_path):
    path = tmp_path / "injected.wav"
    path.write_bytes(b"audio")
    expected = RecordedAudio(path, "wav", 1.0, 16_000, 1)
    recorder: AudioRecorder = FakeAudioRecorder(expected)

    assert recorder.record() is expected


class FakeSpeechProvider:
    def __init__(self, text="Spoken answer", error=None):
        self.text = text
        self.error = error
        self.paths = []

    def transcribe(self, audio_input):
        self.paths.append(Path(audio_input))
        if self.error is not None:
            raise self.error
        return TranscriptionResult(text=self.text)


class DecisionGenerator:
    def __init__(self):
        self.answers = []

    def __call__(self, engine, jd, resume, answer, correction):
        self.answers.append(answer)
        return InterviewerDecision(
            next_question="Describe your Python work.",
            question_type=QuestionType.NEW_TARGET,
            target_competency="Python",
            competency_updates={},
        )


def make_coordinator(generator):
    plan = InterviewPlan(
        role="Engineer",
        targets=[
            InterviewTarget("Python", "HIGH", "Required")
        ],
    )
    return InterviewCoordinator(
        candidate_profile=CandidateProfile(name="Candidate"),
        job_profile=JobProfile(
            role="Engineer",
            requirements=[JobRequirement("Python", "HIGH")],
        ),
        job_description="Python engineer",
        decision_generator=generator,
        interview_plan=plan,
    )


def test_recorder_itself_does_not_mutate_interview_state(tmp_path):
    coordinator = make_coordinator(DecisionGenerator())
    before = coordinator.to_session_dict()

    recording = LocalMicrophoneRecorder(
        backend=FakeCaptureBackend()
    ).record(duration_seconds=0.1, output_path=tmp_path / "answer.wav")

    assert recording.path.exists()
    assert coordinator.to_session_dict() == before


def test_fake_microphone_flows_through_existing_stt(tmp_path):
    recording = LocalMicrophoneRecorder(
        backend=FakeCaptureBackend()
    ).record(duration_seconds=0.1, output_path=tmp_path / "answer.wav")
    provider = FakeSpeechProvider("I built Python services.")

    transcription = CandidateAnswerTranscriber(provider).transcribe(
        recording.path
    )

    assert transcription.text == "I built Python services."
    assert provider.paths == [recording.path]


def test_fake_microphone_transcript_drives_coordinator_turn(tmp_path):
    decisions = DecisionGenerator()
    coordinator = make_coordinator(decisions)
    recording = LocalMicrophoneRecorder(
        backend=FakeCaptureBackend()
    ).record(duration_seconds=0.1, output_path=tmp_path / "answer.wav")
    transcription = CandidateAnswerTranscriber(
        FakeSpeechProvider("I built Python services.")
    ).transcribe(recording.path)

    turn = coordinator.submit_answer(transcription.text)

    assert decisions.answers == ["I built Python services."]
    assert turn.next_question == "Describe your Python work."


def test_recording_failure_leaves_coordinator_unchanged():
    coordinator = make_coordinator(DecisionGenerator())
    before = coordinator.to_session_dict()
    recorder = LocalMicrophoneRecorder(
        backend=FakeCaptureBackend(error=RuntimeError("device failure"))
    )

    with pytest.raises(RecordingBackendError):
        recorder.record(duration_seconds=0.1)

    assert coordinator.to_session_dict() == before


def test_stt_failure_after_recording_leaves_coordinator_unchanged(tmp_path):
    coordinator = make_coordinator(DecisionGenerator())
    before = coordinator.to_session_dict()
    recording = LocalMicrophoneRecorder(
        backend=FakeCaptureBackend()
    ).record(duration_seconds=0.1, output_path=tmp_path / "answer.wav")
    transcriber = CandidateAnswerTranscriber(
        FakeSpeechProvider(error=RuntimeError("STT failure"))
    )

    with pytest.raises(TranscriptionProviderError):
        transcriber.transcribe(recording.path)

    assert recording.path.exists()
    assert coordinator.to_session_dict() == before
