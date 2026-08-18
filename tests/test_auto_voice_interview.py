import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.agent.interview_coordinator import InterviewCoordinator
from src.agent.interviewer_agent import InterviewerDecision, QuestionType
from src.job.job_profile import JobProfile, JobRequirement
from src.planning.interview_plan import InterviewPlan, InterviewTarget
from src.profile.candidate_profile import CandidateProfile
from src.speech.audio_playback import PlaybackResult
from src.speech.audio_recorder import (
    AutomaticRecordingConfig,
    LocalMicrophoneRecorder,
    RecordingStopReason,
)
from src.speech.speech_to_text import TranscriptionResult
from src.speech.text_to_speech import SynthesizedSpeech
from src.speech.voice_activity import RmsVoiceActivityDetector
from src.speech.voice_interview_runner import VoiceInterviewRunner


SAMPLE_RATE = 1000


def pcm(amplitude):
    return np.full(100, amplitude, dtype="<i2").tobytes()


SILENCE = pcm(0)
SPEECH = pcm(1200)


class DecisionGenerator:
    def __init__(self):
        self.answers = []

    def __call__(self, engine, jd, resume, answer, correction):
        self.answers.append(answer)
        return InterviewerDecision(
            next_question="Describe your Python experience.",
            question_type=QuestionType.NEW_TARGET,
            target_competency="Python",
            competency_updates={},
        )


def make_coordinator(generator):
    plan = InterviewPlan(
        role="Engineer",
        targets=[InterviewTarget("Python", "HIGH", "Required")],
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


def temporary_audio(suffix, content):
    output = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    output.write(content)
    path = Path(output.name)
    output.close()
    return path


class FakeSynthesizer:
    def __init__(self, events, unchanged):
        self.events = events
        self.unchanged = unchanged
        self.path = None

    def synthesize(self, question):
        self.events.append("tts")
        self.unchanged()
        self.path = temporary_audio(".mp3", b"question")
        return SynthesizedSpeech(self.path, "mp3", "fake")


class FakePlayer:
    def __init__(self, events, unchanged):
        self.events = events
        self.unchanged = unchanged

    def play(self, path):
        self.events.append("playback")
        self.unchanged()
        return PlaybackResult(Path(path), True)


class ChunkBackend:
    def __init__(self, chunks, events, error=None):
        self.chunks = chunks
        self.events = events
        self.error = error

    def capture(self, **kwargs):
        raise AssertionError("auto mode must not use fixed capture")

    def capture_chunks(self, **kwargs):
        self.events.append("record")
        for chunk in self.chunks:
            yield chunk
        if self.error:
            raise self.error


class LoggingDetector:
    def __init__(self, events, error=None):
        self.events = events
        self.error = error
        self.detector = RmsVoiceActivityDetector(500)

    def is_speech(self, chunk, *, channels):
        self.events.append("vad")
        if self.error:
            raise self.error
        return self.detector.is_speech(chunk, channels=channels)


class TrackingRecorder(LocalMicrophoneRecorder):
    def _resolve_output_path(self, output_path):
        path = super()._resolve_output_path(output_path)
        self.last_path = path
        return path


class FakeTranscriber:
    def __init__(self, events, unchanged, error=None):
        self.events = events
        self.unchanged = unchanged
        self.error = error

    def transcribe(self, path):
        self.events.append("stt")
        self.unchanged()
        if self.error:
            raise self.error
        return TranscriptionResult("I built Python services.")


class TrackingCoordinator:
    def __init__(self, coordinator, events, initial):
        self.coordinator = coordinator
        self.events = events
        self.initial = initial

    @property
    def current_question(self):
        return self.coordinator.current_question

    def submit_answer(self, answer):
        self.events.append("submit")
        assert self.coordinator.to_session_dict() == self.initial
        return self.coordinator.submit_answer(answer)


def build_auto_runner(
    chunks,
    *,
    detector_error=None,
    microphone_error=None,
    stt_error=None,
):
    events = []
    decisions = DecisionGenerator()
    coordinator = make_coordinator(decisions)
    initial = coordinator.to_session_dict()

    def unchanged():
        assert coordinator.to_session_dict() == initial

    synthesizer = FakeSynthesizer(events, unchanged)
    backend = ChunkBackend(chunks, events, microphone_error)
    recorder = TrackingRecorder(
        backend=backend,
        sample_rate=SAMPLE_RATE,
        activity_detector=LoggingDetector(events, detector_error),
        automatic_config=AutomaticRecordingConfig(
            speech_energy_threshold=500,
            initial_speech_timeout_seconds=0.5,
            end_silence_seconds=0.3,
            minimum_speech_seconds=0.2,
            max_answer_duration_seconds=2,
            chunk_duration_seconds=0.1,
        ),
    )
    runner = VoiceInterviewRunner(
        coordinator=TrackingCoordinator(coordinator, events, initial),
        synthesizer=synthesizer,
        player=FakePlayer(events, unchanged),
        recorder=recorder,
        transcriber=FakeTranscriber(events, unchanged, stt_error),
        recording_mode="auto",
        before_recording=lambda: events.append("listen"),
        on_speech_detected=lambda: events.append("speech"),
        after_recording=lambda recording: events.append("captured"),
    )
    return runner, coordinator, initial, decisions, events, synthesizer, recorder


def test_offline_auto_voice_turn_detects_silence_and_submits():
    chunks = [SILENCE] * 2 + [SPEECH] * 2 + [SILENCE] + [SPEECH] + [SILENCE] * 3
    runner, coordinator, initial, decisions, events, speech, recorder = (
        build_auto_runner(chunks)
    )

    result = runner.run_turn()

    assert events[:4] == ["tts", "playback", "listen", "record"]
    assert events[-3:] == ["captured", "stt", "submit"]
    assert events.index("speech") < events.index("captured")
    assert result.recording.stop_reason == RecordingStopReason.SILENCE
    assert result.transcript == "I built Python services."
    assert result.next_question == "Describe your Python experience."
    assert decisions.answers == ["I built Python services."]
    assert coordinator.to_session_dict() != initial
    assert not speech.path.exists()
    assert not recorder.last_path.exists()


@pytest.mark.parametrize(
    ("failure", "chunks", "kwargs"),
    [
        ("no speech", [SILENCE] * 5, {}),
        (
            "vad",
            [SPEECH],
            {"detector_error": RuntimeError("VAD failure")},
        ),
        (
            "microphone",
            [SPEECH],
            {"microphone_error": RuntimeError("device failure")},
        ),
        ("empty", [b""], {}),
        (
            "stt",
            [SPEECH] * 2 + [SILENCE] * 3,
            {"stt_error": RuntimeError("STT failure")},
        ),
    ],
)
def test_auto_voice_failures_are_atomic_and_cleanup(failure, chunks, kwargs):
    runner, coordinator, initial, decisions, _, speech, recorder = (
        build_auto_runner(chunks, **kwargs)
    )

    with pytest.raises(RuntimeError):
        runner.run_turn()

    assert coordinator.to_session_dict() == initial
    assert decisions.answers == []
    assert not speech.path.exists()
    assert not recorder.last_path.exists()
