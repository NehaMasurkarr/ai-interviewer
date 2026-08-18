from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.interview_coordinator import InterviewCoordinator
from src.agent.interviewer_agent import InterviewerDecision, QuestionType
from src.job.job_profile import JobProfile, JobRequirement
from src.planning.interview_plan import InterviewPlan, InterviewTarget
from src.profile.candidate_profile import CandidateProfile
from src.speech.openai_provider import OpenAISpeechToTextProvider
from src.speech.speech_to_text import (
    AudioFileEmptyError,
    AudioFileNotFoundError,
    CandidateAnswerTranscriber,
    EmptyTranscriptionError,
    InvalidTranscriptionResponseError,
    TranscriptionProviderError,
    TranscriptionResult,
    UnsupportedAudioFormatError,
)


def audio_file(tmp_path: Path, suffix: str = ".wav") -> Path:
    path = tmp_path / f"answer{suffix}"
    path.write_bytes(b"fake audio bytes")
    return path


class FakeProvider:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.paths = []

    def transcribe(self, audio_input):
        self.paths.append(audio_input)
        if self.error is not None:
            raise self.error
        return self.result


def test_valid_audio_file_returns_normalized_result(tmp_path):
    path = audio_file(tmp_path)
    provider = FakeProvider(
        TranscriptionResult(
            text="  I built\n a Python   service.  ",
            language=" en ",
            duration_seconds=4,
        )
    )

    result = CandidateAnswerTranscriber(provider).transcribe(path)

    assert result == TranscriptionResult(
        text="I built a Python service.",
        language="en",
        duration_seconds=4.0,
    )
    assert provider.paths == [path]


@pytest.mark.parametrize("suffix", [".wav", ".mp3", ".m4a"])
def test_common_interview_audio_formats_are_accepted(tmp_path, suffix):
    path = audio_file(tmp_path, suffix)
    provider = FakeProvider(TranscriptionResult(text="Candidate answer"))

    result = CandidateAnswerTranscriber(provider).transcribe(path)

    assert result.text == "Candidate answer"


def test_missing_audio_file_is_rejected_before_provider(tmp_path):
    provider = FakeProvider(TranscriptionResult(text="unused"))

    with pytest.raises(AudioFileNotFoundError, match="Audio file not found"):
        CandidateAnswerTranscriber(provider).transcribe(
            tmp_path / "missing.wav"
        )

    assert provider.paths == []


def test_empty_audio_file_is_rejected_before_provider(tmp_path):
    path = tmp_path / "empty.mp3"
    path.touch()
    provider = FakeProvider(TranscriptionResult(text="unused"))

    with pytest.raises(AudioFileEmptyError, match="Audio file is empty"):
        CandidateAnswerTranscriber(provider).transcribe(path)

    assert provider.paths == []


def test_unsupported_audio_format_is_rejected(tmp_path):
    path = audio_file(tmp_path, ".txt")
    provider = FakeProvider(TranscriptionResult(text="unused"))

    with pytest.raises(
        UnsupportedAudioFormatError,
        match="Unsupported audio format",
    ):
        CandidateAnswerTranscriber(provider).transcribe(path)

    assert provider.paths == []


def test_empty_transcript_is_rejected(tmp_path):
    provider = FakeProvider(TranscriptionResult(text=" \n\t "))

    with pytest.raises(EmptyTranscriptionError):
        CandidateAnswerTranscriber(provider).transcribe(audio_file(tmp_path))


def test_invalid_provider_result_is_rejected(tmp_path):
    provider = FakeProvider({"text": "not a domain result"})

    with pytest.raises(InvalidTranscriptionResponseError):
        CandidateAnswerTranscriber(provider).transcribe(audio_file(tmp_path))


def test_provider_failure_is_wrapped_and_preserves_cause(tmp_path):
    original = ConnectionError("network unavailable")
    provider = FakeProvider(error=original)

    with pytest.raises(TranscriptionProviderError) as captured:
        CandidateAnswerTranscriber(provider).transcribe(audio_file(tmp_path))

    assert captured.value.__cause__ is original


class FakeOpenAITranscriptions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def fake_openai_client(transcriptions):
    return SimpleNamespace(
        audio=SimpleNamespace(transcriptions=transcriptions)
    )


def test_openai_provider_normalizes_sdk_response_without_network(tmp_path):
    api = FakeOpenAITranscriptions(
        response=SimpleNamespace(
            text="  Spoken   candidate answer. ",
            language="en",
            duration=2.5,
        )
    )
    provider = OpenAISpeechToTextProvider(client=fake_openai_client(api))

    result = provider.transcribe(audio_file(tmp_path, ".m4a"))

    assert result == TranscriptionResult(
        text="Spoken candidate answer.",
        language="en",
        duration_seconds=2.5,
    )
    assert api.calls[0]["model"] == "gpt-4o-mini-transcribe"
    assert api.calls[0]["response_format"] == "json"


def test_openai_provider_wraps_sdk_failure(tmp_path):
    original = RuntimeError("provider outage")
    api = FakeOpenAITranscriptions(error=original)
    provider = OpenAISpeechToTextProvider(client=fake_openai_client(api))

    with pytest.raises(TranscriptionProviderError) as captured:
        provider.transcribe(audio_file(tmp_path))

    assert captured.value.__cause__ is original


def make_coordinator(decision_generator):
    candidate = CandidateProfile(name="Candidate", skills=["Python"])
    job = JobProfile(
        role="Engineer",
        requirements=[JobRequirement(name="Python", priority="HIGH")],
    )
    plan = InterviewPlan(
        role="Engineer",
        targets=[
            InterviewTarget(
                competency="Python",
                priority="HIGH",
                reason="Required",
            )
        ],
    )
    return InterviewCoordinator(
        candidate_profile=candidate,
        job_profile=job,
        job_description="Python engineer",
        decision_generator=decision_generator,
        interview_plan=plan,
    )


class OneDecisionGenerator:
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


def test_transcription_does_not_mutate_interview_state(tmp_path):
    coordinator = make_coordinator(OneDecisionGenerator())
    before = coordinator.to_session_dict()
    transcriber = CandidateAnswerTranscriber(
        FakeProvider(TranscriptionResult(text="Spoken answer"))
    )

    result = transcriber.transcribe(audio_file(tmp_path))

    assert result.text == "Spoken answer"
    assert coordinator.to_session_dict() == before


def test_transcript_feeds_normal_atomic_interview_turn(tmp_path):
    decisions = OneDecisionGenerator()
    coordinator = make_coordinator(decisions)
    transcriber = CandidateAnswerTranscriber(
        FakeProvider(TranscriptionResult(text="I built Python services."))
    )

    transcription = transcriber.transcribe(audio_file(tmp_path, ".mp3"))
    turn = coordinator.submit_answer(transcription.text)

    assert decisions.answers == ["I built Python services."]
    assert turn.next_question == "Describe your Python experience."
    assert coordinator.engine.memory.get_history()[0]["answer"] == (
        "I built Python services."
    )


def test_failed_transcription_leaves_interview_state_unchanged(tmp_path):
    coordinator = make_coordinator(OneDecisionGenerator())
    before = coordinator.to_session_dict()
    transcriber = CandidateAnswerTranscriber(
        FakeProvider(error=RuntimeError("transcription failed"))
    )

    with pytest.raises(TranscriptionProviderError):
        transcriber.transcribe(audio_file(tmp_path))

    assert coordinator.to_session_dict() == before
