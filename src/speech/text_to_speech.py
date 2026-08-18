from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Union, runtime_checkable


SpeechOutputPath = Union[str, Path]
SUPPORTED_SPEECH_FORMATS = frozenset(
    {"aac", "flac", "mp3", "opus", "pcm", "wav"}
)
DEFAULT_MAX_SPEECH_TEXT_LENGTH = 4096


class TextToSpeechError(RuntimeError):
    """Base error for normalized text-to-speech failures."""


class InvalidSpeechTextError(TextToSpeechError):
    """Raised when speech input text is missing or unsafe."""


class UnsupportedSpeechFormatError(TextToSpeechError):
    """Raised when an output audio format is unsupported."""


class InvalidSpeechVoiceError(TextToSpeechError):
    """Raised when voice configuration is empty or invalid."""


class InvalidSpeechOutputPathError(TextToSpeechError):
    """Raised when a requested output path cannot be used safely."""


class SpeechSynthesisProviderError(TextToSpeechError):
    """Raised when the configured synthesis provider fails."""


class InvalidSpeechSynthesisResponseError(TextToSpeechError):
    """Raised when a provider returns an invalid domain result."""


class SynthesizedAudioMissingError(TextToSpeechError):
    """Raised when synthesis did not create an audio file."""


class SynthesizedAudioEmptyError(TextToSpeechError):
    """Raised when synthesis created an empty audio file."""


@dataclass(frozen=True)
class SynthesizedSpeech:
    """Provider-neutral metadata for generated interviewer speech."""

    path: Path
    format: str
    voice: Optional[str] = None
    duration_seconds: Optional[float] = None

    def cleanup(self) -> None:
        """Explicitly remove generated speech if it still exists."""

        self.path.unlink(missing_ok=True)


@runtime_checkable
class TextToSpeechProvider(Protocol):
    """Interface implemented by speech-synthesis providers."""

    def synthesize(
        self,
        text: str,
        *,
        output_path: Optional[SpeechOutputPath] = None,
    ) -> SynthesizedSpeech:
        ...


class InterviewerSpeechSynthesizer:
    """Validate question text and normalize provider output."""

    def __init__(
        self,
        provider: TextToSpeechProvider,
        *,
        max_text_length: int = DEFAULT_MAX_SPEECH_TEXT_LENGTH,
    ):
        if not hasattr(provider, "synthesize") or not callable(
            provider.synthesize
        ):
            raise TypeError("provider must implement synthesize().")

        if (
            isinstance(max_text_length, bool)
            or not isinstance(max_text_length, int)
            or max_text_length < 1
        ):
            raise ValueError("max_text_length must be a positive integer.")

        self.provider = provider
        self.max_text_length = max_text_length

    def synthesize(
        self,
        text: str,
        *,
        output_path: Optional[SpeechOutputPath] = None,
    ) -> SynthesizedSpeech:
        normalized_text = validate_speech_text(text, self.max_text_length)

        try:
            result = self.provider.synthesize(
                normalized_text,
                output_path=output_path,
            )
        except TextToSpeechError:
            raise
        except Exception as error:
            raise SpeechSynthesisProviderError(
                "Text-to-speech provider failed."
            ) from error

        return validate_synthesized_speech(result)


def validate_speech_text(
    text: str,
    max_text_length: int = DEFAULT_MAX_SPEECH_TEXT_LENGTH,
) -> str:
    """Validate text while preserving its wording and punctuation."""

    if not isinstance(text, str):
        raise InvalidSpeechTextError("Speech text must be a string.")

    text = text.strip()

    if not text:
        raise InvalidSpeechTextError("Speech text cannot be empty.")

    if len(text) > max_text_length:
        raise InvalidSpeechTextError(
            f"Speech text cannot exceed {max_text_length} characters."
        )

    return text


def validate_speech_format(output_format: str) -> str:
    if not isinstance(output_format, str) or not output_format.strip():
        raise UnsupportedSpeechFormatError(
            "Speech output format is required."
        )

    output_format = output_format.strip().lower().lstrip(".")

    if output_format not in SUPPORTED_SPEECH_FORMATS:
        raise UnsupportedSpeechFormatError(
            f"Unsupported speech output format: {output_format}."
        )

    return output_format


def validate_speech_voice(voice: str) -> str:
    if not isinstance(voice, str) or not voice.strip():
        raise InvalidSpeechVoiceError("Speech voice is required.")
    return voice.strip()


def validate_synthesized_speech(result) -> SynthesizedSpeech:
    """Validate provider-neutral output and its retained audio file."""

    if not isinstance(result, SynthesizedSpeech):
        raise InvalidSpeechSynthesisResponseError(
            "Provider did not return SynthesizedSpeech."
        )

    path = Path(result.path)
    output_format = validate_speech_format(result.format)

    if path.suffix.lower() != f".{output_format}":
        raise InvalidSpeechSynthesisResponseError(
            "Synthesized file extension does not match its format."
        )

    if not path.is_file():
        raise SynthesizedAudioMissingError(
            f"Synthesized audio file was not created: {path}"
        )

    if path.stat().st_size == 0:
        raise SynthesizedAudioEmptyError(
            f"Synthesized audio file is empty: {path}"
        )

    voice = result.voice
    if voice is not None:
        voice = validate_speech_voice(voice)

    duration = result.duration_seconds
    if duration is not None:
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise InvalidSpeechSynthesisResponseError(
                "Speech duration must be numeric or null."
            )
        if duration < 0:
            raise InvalidSpeechSynthesisResponseError(
                "Speech duration cannot be negative."
            )
        duration = float(duration)

    return SynthesizedSpeech(
        path=path,
        format=output_format,
        voice=voice,
        duration_seconds=duration,
    )
