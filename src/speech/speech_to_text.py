from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Union, runtime_checkable


AudioPath = Union[str, Path]

SUPPORTED_AUDIO_EXTENSIONS = frozenset(
    {
        ".m4a",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".mpga",
        ".wav",
        ".webm",
    }
)


class SpeechToTextError(RuntimeError):
    """Base error for normalized speech-to-text failures."""


class AudioFileNotFoundError(SpeechToTextError):
    """Raised when an audio path does not identify a file."""


class AudioFileEmptyError(SpeechToTextError):
    """Raised when an audio file contains no bytes."""


class UnsupportedAudioFormatError(SpeechToTextError):
    """Raised when an audio extension is not supported."""


class TranscriptionProviderError(SpeechToTextError):
    """Raised when the configured provider fails."""


class InvalidTranscriptionResponseError(SpeechToTextError):
    """Raised when a provider returns an invalid result."""


class EmptyTranscriptionError(SpeechToTextError):
    """Raised when no meaningful speech was transcribed."""


@dataclass(frozen=True)
class TranscriptionResult:
    """Provider-neutral transcription returned to the application."""

    text: str
    language: Optional[str] = None
    duration_seconds: Optional[float] = None


@runtime_checkable
class SpeechToTextProvider(Protocol):
    """Interface implemented by speech transcription providers."""

    def transcribe(self, audio_input: AudioPath) -> TranscriptionResult:
        ...


class CandidateAnswerTranscriber:
    """Validate candidate audio and normalize provider output."""

    def __init__(self, provider: SpeechToTextProvider):
        if not hasattr(provider, "transcribe") or not callable(
            provider.transcribe
        ):
            raise TypeError("provider must implement transcribe().")

        self.provider = provider

    def transcribe(self, audio_input: AudioPath) -> TranscriptionResult:
        audio_path = validate_audio_file(audio_input)

        try:
            result = self.provider.transcribe(audio_path)
        except SpeechToTextError:
            raise
        except Exception as error:
            raise TranscriptionProviderError(
                "Speech-to-text provider failed."
            ) from error

        return normalize_transcription_result(result)


def validate_audio_file(audio_input: AudioPath) -> Path:
    """Return a validated audio file path."""

    if audio_input is None or not str(audio_input).strip():
        raise AudioFileNotFoundError("Audio file path is required.")

    path = Path(audio_input)

    if not path.is_file():
        raise AudioFileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(
            sorted(extension.lstrip(".") for extension in SUPPORTED_AUDIO_EXTENSIONS)
        )
        raise UnsupportedAudioFormatError(
            f"Unsupported audio format: {path.suffix or '(none)'}. "
            f"Supported formats: {supported}."
        )

    if path.stat().st_size == 0:
        raise AudioFileEmptyError(f"Audio file is empty: {path}")

    return path


def normalize_transcription_result(result) -> TranscriptionResult:
    """Validate and normalize a provider-neutral result."""

    if not isinstance(result, TranscriptionResult):
        raise InvalidTranscriptionResponseError(
            "Provider did not return a TranscriptionResult."
        )

    if not isinstance(result.text, str):
        raise InvalidTranscriptionResponseError(
            "Transcription text must be a string."
        )

    text = " ".join(result.text.split())

    if not text:
        raise EmptyTranscriptionError(
            "Speech transcription produced no meaningful text."
        )

    language = result.language
    if language is not None:
        if not isinstance(language, str):
            raise InvalidTranscriptionResponseError(
                "Transcription language must be a string or null."
            )
        language = language.strip() or None

    duration = result.duration_seconds
    if duration is not None:
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise InvalidTranscriptionResponseError(
                "Transcription duration must be numeric or null."
            )
        if duration < 0:
            raise InvalidTranscriptionResponseError(
                "Transcription duration cannot be negative."
            )
        duration = float(duration)

    return TranscriptionResult(
        text=text,
        language=language,
        duration_seconds=duration,
    )
