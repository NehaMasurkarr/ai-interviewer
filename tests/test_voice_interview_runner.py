import tempfile
from pathlib import Path

import pytest

from src.agent.interview_coordinator import InterviewCoordinator
from src.agent.interviewer_agent import InterviewerDecision, QuestionType
from src.job.job_profile import JobProfile, JobRequirement
from src.planning.interview_plan import InterviewPlan, InterviewTarget
from src.profile.candidate_profile import CandidateProfile
from src.speech.audio_playback import PlaybackResult
from src.speech.audio_recorder import RecordedAudio
from src.speech.speech_to_text import TranscriptionResult
from src.speech.text_to_speech import SynthesizedSpeech
from src.speech.voice_interview_runner import (
    InvalidPlaybackResultError,
    VoiceInterviewRunner,
)


class DecisionGenerator:
    def __init__(self, error=None):
        self.error = error
        self.answers = []

    def __call__(self, engine, jd, resume, answer, correction):
        self.answers.append(answer)
        if self.error is not None:
            raise self.error
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
    def __init__(self, events, check_state, error=None):
        self.events = events
        self.check_state = check_state
        self.error = error
        self.path = None

    def synthesize(self, question):
        self.events.append("tts")
        self.check_state()
        if self.error:
            raise self.error
        self.path = temporary_audio(".mp3", b"question audio")
        return SynthesizedSpeech(self.path, "mp3", "fake")


class FakePlayer:
    def __init__(self, events, check_state, error=None, completed=True):
        self.events = events
        self.check_state = check_state
        self.error = error
        self.completed = completed

    def play(self, path):
        self.events.append("playback")
        self.check_state()
        assert Path(path).exists()
        if self.error:
            raise self.error
        return PlaybackResult(Path(path), self.completed)


class FakeRecorder:
    def __init__(self, events, check_state, error=None):
        self.events = events
        self.check_state = check_state
        self.error = error
        self.path = None
        self.durations = []

    def record(self, duration_seconds=None, output_path=None):
        self.events.append("recording")
        self.check_state()
        self.durations.append(duration_seconds)
        if self.error:
            raise self.error
        self.path = temporary_audio(".wav", b"candidate audio")
        return RecordedAudio(self.path, "wav", 1.0, 16_000, 1)


class FakeTranscriber:
    def __init__(self, events, check_state, error=None):
        self.events = events
        self.check_state = check_state
        self.error = error

    def transcribe(self, path):
        self.events.append("stt")
        self.check_state()
        assert Path(path).exists()
        if self.error:
            raise self.error
        return TranscriptionResult("I built Python services.")


class TrackingCoordinator:
    def __init__(self, coordinator, events, initial_state):
        self.coordinator = coordinator
        self.events = events
        self.initial_state = initial_state
        self.submit_calls = 0

    @property
    def current_question(self):
        return self.coordinator.current_question

    def submit_answer(self, transcript):
        self.events.append("submit")
        assert self.coordinator.to_session_dict() == self.initial_state
        self.submit_calls += 1
        return self.coordinator.submit_answer(transcript)


def build_runner(*, fail_stage=None, retain_audio=False, completed=True):
    events = []
    decision_error = RuntimeError("coordinator failure") if fail_stage == "submit" else None
    decisions = DecisionGenerator(error=decision_error)
    actual = make_coordinator(decisions)
    initial = actual.to_session_dict()
    tracked = TrackingCoordinator(actual, events, initial)

    def check_state():
        assert actual.to_session_dict() == initial

    synthesizer = FakeSynthesizer(
        events,
        check_state,
        RuntimeError("TTS failure") if fail_stage == "tts" else None,
    )
    player = FakePlayer(
        events,
        check_state,
        RuntimeError("playback failure") if fail_stage == "playback" else None,
        completed=completed,
    )
    recorder = FakeRecorder(
        events,
        check_state,
        RuntimeError("recording failure") if fail_stage == "recording" else None,
    )
    transcriber = FakeTranscriber(
        events,
        check_state,
        RuntimeError("STT failure") if fail_stage == "stt" else None,
    )
    runner = VoiceInterviewRunner(
        coordinator=tracked,
        synthesizer=synthesizer,
        player=player,
        recorder=recorder,
        transcriber=transcriber,
        retain_audio=retain_audio,
    )
    return runner, actual, tracked, decisions, events, synthesizer, recorder


def test_successful_spoken_turn_has_exact_order_and_single_mutation():
    (
        runner,
        coordinator,
        tracked,
        decisions,
        events,
        synthesizer,
        recorder,
    ) = build_runner()

    result = runner.run_turn(recording_duration=12)

    assert events == ["tts", "playback", "recording", "stt", "submit"]
    assert tracked.submit_calls == 1
    assert decisions.answers == ["I built Python services."]
    assert result.transcript == "I built Python services."
    assert result.next_question == "Describe your Python experience."
    assert recorder.durations == [12]
    assert not synthesizer.path.exists()
    assert not recorder.path.exists()
    assert len(coordinator.engine.memory) == 1


def test_incomplete_playback_prevents_recording_and_cleans_audio():
    runner, coordinator, tracked, _, events, synthesizer, recorder = (
        build_runner(completed=False)
    )
    before = coordinator.to_session_dict()

    with pytest.raises(InvalidPlaybackResultError):
        runner.run_turn()

    assert events == ["tts", "playback"]
    assert recorder.path is None
    assert not synthesizer.path.exists()
    assert tracked.submit_calls == 0
    assert coordinator.to_session_dict() == before


@pytest.mark.parametrize(
    ("stage", "expected_events"),
    [
        ("tts", ["tts"]),
        ("playback", ["tts", "playback"]),
        ("recording", ["tts", "playback", "recording"]),
        ("stt", ["tts", "playback", "recording", "stt"]),
        ("submit", ["tts", "playback", "recording", "stt", "submit"]),
    ],
)
def test_failure_stages_are_atomic_and_cleanup_artifacts(stage, expected_events):
    runner, coordinator, tracked, _, events, synthesizer, recorder = (
        build_runner(fail_stage=stage)
    )
    before = coordinator.to_session_dict()

    with pytest.raises(RuntimeError):
        runner.run_turn()

    assert events == expected_events
    assert coordinator.to_session_dict() == before
    assert tracked.submit_calls == (1 if stage == "submit" else 0)
    if synthesizer.path is not None:
        assert not synthesizer.path.exists()
    if recorder.path is not None:
        assert not recorder.path.exists()


def test_retain_audio_keeps_both_artifacts():
    runner, _, _, _, _, synthesizer, recorder = build_runner(
        retain_audio=True
    )

    result = runner.run_turn()

    assert result.audio_retained
    assert synthesizer.path.exists()
    assert recorder.path.exists()
    result.recording.cleanup()
    result.speech.cleanup()


def test_typed_coordinator_usage_remains_unchanged():
    decisions = DecisionGenerator()
    coordinator = make_coordinator(decisions)

    result = coordinator.submit_answer("Typed answer")

    assert decisions.answers == ["Typed answer"]
    assert result.next_question == "Describe your Python experience."
