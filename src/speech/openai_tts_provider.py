import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from src.speech.text_to_speech import (
    InvalidSpeechOutputPathError,
    SpeechOutputPath,
    SpeechSynthesisProviderError,
    SynthesizedSpeech,
    TextToSpeechError,
    validate_speech_format,
    validate_speech_text,
    validate_speech_voice,
    validate_synthesized_speech,
)


DEFAULT_SPEECH_MODEL = "tts-1"
DEFAULT_SPEECH_VOICE = "alloy"
DEFAULT_SPEECH_FORMAT = "mp3"


class OpenAITextToSpeechProvider:
    """OpenAI speech endpoint adapter behind the TTS protocol."""

    def __init__(
        self,
        client: Optional[Any] = None,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_SPEECH_MODEL,
        voice: str = DEFAULT_SPEECH_VOICE,
        output_format: str = DEFAULT_SPEECH_FORMAT,
        max_text_length: int = 4096,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Speech model is required.")

        if (
            isinstance(max_text_length, bool)
            or not isinstance(max_text_length, int)
            or max_text_length < 1
        ):
            raise ValueError("max_text_length must be a positive integer.")

        self.model = model.strip()
        self.voice = validate_speech_voice(voice)
        self.output_format = validate_speech_format(output_format)
        self.max_text_length = max_text_length

        if client is None:
            from dotenv import load_dotenv
            from openai import OpenAI

            load_dotenv()
            resolved_key = api_key or os.getenv("OPENAI_API_KEY")

            if not resolved_key:
                raise SpeechSynthesisProviderError(
                    "OPENAI_API_KEY is required for OpenAI speech synthesis."
                )

            client = OpenAI(api_key=resolved_key)

        self.client = client

    def synthesize(
        self,
        text: str,
        *,
        output_path: Optional[SpeechOutputPath] = None,
    ) -> SynthesizedSpeech:
        text = validate_speech_text(text, self.max_text_length)
        path = self._resolve_output_path(output_path)

        try:
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format=self.output_format,
            )

            write_to_file = getattr(response, "write_to_file", None)
            if not callable(write_to_file):
                raise SpeechSynthesisProviderError(
                    "OpenAI returned unusable speech audio."
                )

            write_to_file(path)

            result = SynthesizedSpeech(
                path=path,
                format=self.output_format,
                voice=self.voice,
                duration_seconds=None,
            )
            return validate_synthesized_speech(result)
        except TextToSpeechError:
            path.unlink(missing_ok=True)
            raise
        except Exception as error:
            path.unlink(missing_ok=True)
            raise SpeechSynthesisProviderError(
                "OpenAI speech synthesis failed."
            ) from error

    def _resolve_output_path(
        self,
        output_path: Optional[SpeechOutputPath],
    ) -> Path:
        if output_path is None:
            temporary = tempfile.NamedTemporaryFile(
                prefix="ai-interviewer-question-",
                suffix=f".{self.output_format}",
                delete=False,
            )
            path = Path(temporary.name)
            temporary.close()
            return path

        if not str(output_path).strip():
            raise InvalidSpeechOutputPathError(
                "Speech output path is required."
            )

        path = Path(output_path)

        if path.suffix.lower() != f".{self.output_format}":
            raise InvalidSpeechOutputPathError(
                "Speech output extension must match configured format "
                f".{self.output_format}."
            )

        if not path.parent.is_dir():
            raise InvalidSpeechOutputPathError(
                f"Speech output directory does not exist: {path.parent}"
            )

        if path.exists():
            raise InvalidSpeechOutputPathError(
                f"Speech output path already exists: {path}"
            )

        return path
