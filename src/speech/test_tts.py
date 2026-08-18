import argparse

from src.speech.openai_tts_provider import OpenAITextToSpeechProvider
from src.speech.text_to_speech import (
    InterviewerSpeechSynthesizer,
    TextToSpeechError,
)


DEFAULT_QUESTION = (
    "Tell me about yourself and the experience most relevant to this role."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an interviewer-question audio file."
    )
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--voice", default="alloy")
    parser.add_argument("--format", default="mp3")
    arguments = parser.parse_args()

    try:
        provider = OpenAITextToSpeechProvider(
            voice=arguments.voice,
            output_format=arguments.format,
        )
        speech = InterviewerSpeechSynthesizer(provider).synthesize(
            arguments.question
        )
    except TextToSpeechError as error:
        print(f"Speech synthesis failed: {error}")
        raise SystemExit(1) from error

    print(f"Output path: {speech.path}")
    print(f"Format: {speech.format}")
    print(f"Voice: {speech.voice}")
    print(f"Duration: {speech.duration_seconds or 'not provided'}")
    print("The file was not played automatically.")
    print("Delete it when finished with speech.cleanup() or rm <path>.")


if __name__ == "__main__":
    main()
