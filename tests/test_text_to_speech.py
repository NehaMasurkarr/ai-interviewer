import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.interview_coordinator import InterviewCoordinator
from src.agent.interviewer_agent import InterviewerDecision, QuestionType
from src.job.job_profile import JobProfile, JobRequirement
from src.planning.interview_plan import InterviewPlan, InterviewTarget
from src.profile.candidate_profile import CandidateProfile
from src.speech.audio_recorder import LocalMicrophoneRecorder
from src.speech.openai_tts_provider import OpenAITextToSpeechProvider
from src.speech.speech_to_text import (
    CandidateAnswerTranscriber,
    TranscriptionResult,
)
from src.speech.text_to_speech import (
    InterviewerSpeechSynthesizer,
    InvalidSpeechTextError,
    InvalidSpeechVoiceError,
    SpeechSynthesisProviderError,
    SynthesizedAudioEmptyError,
    SynthesizedAudioMissingError,
    SynthesizedSpeech,
    UnsupportedSpeechFormatError,
)


class FakeTextToSpeechProvider:
    def __init__(self, *, audio=b"fake mp3 audio", error=None):
        self.audio = audio
        self.error = error
        self.calls = []

    def synthesize(self, text, *, output_path=None):
        self.calls.append({"text": text, "output_path": output_path})
        if self.error is not None:
            raise self.error

        if output_path is None:
            temporary = tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            )
            path = Path(temporary.name)
            temporary.close()
        else:
            path = Path(output_path)

        path.write_bytes(self.audio)
        return SynthesizedSpeech(path, "mp3", "fake-voice")


def test_valid_text_synthesis_creates_nonempty_audio(tmp_path):
    provider = FakeTextToSpeechProvider()
    output = tmp_path / "question.mp3"

    speech = InterviewerSpeechSynthesizer(provider).synthesize(
        "What project are you most proud of?",
        output_path=output,
    )

    assert speech.path == output
    assert speech.format == "mp3"
    assert speech.voice == "fake-voice"
    assert output.is_file()
    assert output.stat().st_size > 0


def test_only_surrounding_whitespace_is_normalized(tmp_path):
    provider = FakeTextToSpeechProvider()
    text = "  Explain your design.\nThen discuss its tradeoffs.  "

    InterviewerSpeechSynthesizer(provider).synthesize(
        text, output_path=tmp_path / "question.mp3"
    )

    assert provider.calls[0]["text"] == (
        "Explain your design.\nThen discuss its tradeoffs."
    )


@pytest.mark.parametrize("text", ["", "  \n\t "])
def test_empty_text_is_rejected_before_provider(text):
    provider = FakeTextToSpeechProvider()

    with pytest.raises(InvalidSpeechTextError):
        InterviewerSpeechSynthesizer(provider).synthesize(text)

    assert provider.calls == []


def test_unreasonable_text_length_is_rejected():
    provider = FakeTextToSpeechProvider()
    synthesizer = InterviewerSpeechSynthesizer(
        provider, max_text_length=10
    )

    with pytest.raises(InvalidSpeechTextError, match="cannot exceed 10"):
        synthesizer.synthesize("This question is too long")


@pytest.mark.parametrize("output_format", ["exe", "m4a", "", None])
def test_invalid_output_format_is_rejected(output_format):
    with pytest.raises(UnsupportedSpeechFormatError):
        OpenAITextToSpeechProvider(
            client=object(), output_format=output_format
        )


@pytest.mark.parametrize("voice", ["", "   ", None])
def test_invalid_voice_is_rejected(voice):
    with pytest.raises(InvalidSpeechVoiceError):
        OpenAITextToSpeechProvider(client=object(), voice=voice)


class MissingOutputProvider:
    def synthesize(self, text, *, output_path=None):
        return SynthesizedSpeech(
            Path("/tmp/ai-interviewer-missing-speech.mp3"),
            "mp3",
        )


def test_missing_provider_output_is_rejected():
    with pytest.raises(SynthesizedAudioMissingError):
        InterviewerSpeechSynthesizer(
            MissingOutputProvider()
        ).synthesize("Question?")


def test_empty_provider_output_is_rejected(tmp_path):
    provider = FakeTextToSpeechProvider(audio=b"")

    with pytest.raises(SynthesizedAudioEmptyError):
        InterviewerSpeechSynthesizer(provider).synthesize(
            "Question?", output_path=tmp_path / "empty.mp3"
        )


def test_provider_failure_is_wrapped_and_preserves_cause():
    original = ConnectionError("provider unavailable")
    provider = FakeTextToSpeechProvider(error=original)

    with pytest.raises(SpeechSynthesisProviderError) as captured:
        InterviewerSpeechSynthesizer(provider).synthesize("Question?")

    assert captured.value.__cause__ is original


def test_fake_provider_satisfies_dependency_injection(tmp_path):
    provider = FakeTextToSpeechProvider()
    result = InterviewerSpeechSynthesizer(provider).synthesize(
        "Question?", output_path=tmp_path / "fake.mp3"
    )

    assert result.path.read_bytes() == b"fake mp3 audio"


class FakeBinarySpeechResponse:
    def __init__(self, audio=b"openai speech bytes"):
        self.audio = audio
        self.paths = []

    def write_to_file(self, path):
        path = Path(path)
        self.paths.append(path)
        path.write_bytes(self.audio)


class FakeOpenAISpeechEndpoint:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def fake_openai_client(endpoint):
    return SimpleNamespace(audio=SimpleNamespace(speech=endpoint))


def test_openai_client_injection_passes_model_voice_format_and_exact_text(
    tmp_path,
):
    response = FakeBinarySpeechResponse()
    endpoint = FakeOpenAISpeechEndpoint(response=response)
    provider = OpenAITextToSpeechProvider(
        client=fake_openai_client(endpoint),
        model="tts-1-hd",
        voice="cedar",
        output_format="wav",
    )
    output = tmp_path / "question.wav"

    speech = provider.synthesize(
        "Explain your architecture—and its tradeoffs.",
        output_path=output,
    )

    assert endpoint.calls == [
        {
            "model": "tts-1-hd",
            "voice": "cedar",
            "input": "Explain your architecture—and its tradeoffs.",
            "response_format": "wav",
        }
    ]
    assert speech.path == output
    assert speech.voice == "cedar"
    assert output.read_bytes() == b"openai speech bytes"


def test_openai_provider_failure_preserves_cause(tmp_path):
    original = RuntimeError("OpenAI outage")
    endpoint = FakeOpenAISpeechEndpoint(error=original)
    provider = OpenAITextToSpeechProvider(
        client=fake_openai_client(endpoint)
    )

    with pytest.raises(SpeechSynthesisProviderError) as captured:
        provider.synthesize(
            "Question?", output_path=tmp_path / "question.mp3"
        )

    assert captured.value.__cause__ is original


def test_cleanup_removes_temporary_speech_and_is_idempotent():
    speech = InterviewerSpeechSynthesizer(
        FakeTextToSpeechProvider()
    ).synthesize("Question?")

    assert speech.path.exists()
    speech.cleanup()
    assert not speech.path.exists()
    speech.cleanup()


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


def test_current_question_synthesis_does_not_mutate_coordinator(tmp_path):
    coordinator = make_coordinator(DecisionGenerator())
    before = coordinator.to_session_dict()
    provider = FakeTextToSpeechProvider()

    speech = InterviewerSpeechSynthesizer(provider).synthesize(
        coordinator.current_question,
        output_path=tmp_path / "opening.mp3",
    )

    assert provider.calls[0]["text"] == coordinator.current_question
    assert speech.path.exists()
    assert coordinator.to_session_dict() == before


def test_tts_failure_leaves_coordinator_unchanged():
    coordinator = make_coordinator(DecisionGenerator())
    before = coordinator.to_session_dict()
    synthesizer = InterviewerSpeechSynthesizer(
        FakeTextToSpeechProvider(error=RuntimeError("TTS failure"))
    )

    with pytest.raises(SpeechSynthesisProviderError):
        synthesizer.synthesize(coordinator.current_question)

    assert coordinator.to_session_dict() == before


class FakeCaptureBackend:
    def capture(self, *, duration_seconds, sample_rate, channels):
        frames = round(duration_seconds * sample_rate)
        return b"\x00\x00" * frames * channels


class FakeSpeechToTextProvider:
    def transcribe(self, audio_input):
        return TranscriptionResult(text="I built Python services.")


def test_complete_offline_voice_boundary(tmp_path):
    decisions = DecisionGenerator()
    coordinator = make_coordinator(decisions)
    initial_state = coordinator.to_session_dict()

    spoken_question = InterviewerSpeechSynthesizer(
        FakeTextToSpeechProvider()
    ).synthesize(
        coordinator.current_question,
        output_path=tmp_path / "spoken-question.mp3",
    )
    assert spoken_question.path.exists()
    assert coordinator.to_session_dict() == initial_state

    recording = LocalMicrophoneRecorder(
        backend=FakeCaptureBackend()
    ).record(
        duration_seconds=0.1,
        output_path=tmp_path / "candidate-answer.wav",
    )
    assert coordinator.to_session_dict() == initial_state

    transcript = CandidateAnswerTranscriber(
        FakeSpeechToTextProvider()
    ).transcribe(recording.path)
    assert coordinator.to_session_dict() == initial_state

    turn = coordinator.submit_answer(transcript.text)

    assert decisions.answers == ["I built Python services."]
    assert turn.next_question == "Describe your Python experience."
    assert coordinator.to_session_dict() != initial_state
