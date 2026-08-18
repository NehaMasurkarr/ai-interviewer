import os
from typing import Any, Optional

from src.speech.speech_to_text import (
    InvalidTranscriptionResponseError,
    SpeechToTextError,
    TranscriptionProviderError,
    TranscriptionResult,
    normalize_transcription_result,
    validate_audio_file,
)


DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"


class OpenAISpeechToTextProvider:
    """OpenAI audio-transcription adapter behind the STT protocol."""

    def __init__(
        self,
        client: Optional[Any] = None,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_TRANSCRIPTION_MODEL,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Transcription model is required.")

        if client is None:
            from dotenv import load_dotenv
            from openai import OpenAI

            load_dotenv()
            resolved_key = api_key or os.getenv("OPENAI_API_KEY")

            if not resolved_key:
                raise TranscriptionProviderError(
                    "OPENAI_API_KEY is required for OpenAI transcription."
                )

            client = OpenAI(api_key=resolved_key)

        self.client = client
        self.model = model.strip()

    def transcribe(self, audio_input) -> TranscriptionResult:
        audio_path = validate_audio_file(audio_input)

        try:
            with audio_path.open("rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    response_format="json",
                )
        except SpeechToTextError:
            raise
        except Exception as error:
            raise TranscriptionProviderError(
                "OpenAI audio transcription failed."
            ) from error

        return normalize_transcription_result(
            self._convert_response(response)
        )

    @staticmethod
    def _convert_response(response: Any) -> TranscriptionResult:
        if isinstance(response, str):
            return TranscriptionResult(text=response)

        if isinstance(response, dict):
            text = response.get("text")
            language = response.get("language")
            duration = response.get("duration")
        else:
            text = getattr(response, "text", None)
            language = getattr(response, "language", None)
            duration = getattr(response, "duration", None)

        if not isinstance(text, str):
            raise InvalidTranscriptionResponseError(
                "OpenAI returned an invalid transcription response."
            )

        return TranscriptionResult(
            text=text,
            language=language,
            duration_seconds=duration,
        )
