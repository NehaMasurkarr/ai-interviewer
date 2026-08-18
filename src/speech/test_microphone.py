import argparse

from src.speech.audio_recorder import (
    AudioRecordingError,
    LocalMicrophoneRecorder,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a local microphone sample as WAV."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds (default: 5).",
    )
    arguments = parser.parse_args()

    print("Microphone access will be requested by macOS.")
    print(f"Recording for {arguments.duration:g} seconds...")

    try:
        recording = LocalMicrophoneRecorder().record(
            duration_seconds=arguments.duration
        )
    except AudioRecordingError as error:
        print(f"Recording failed: {error}")
        raise SystemExit(1) from error

    print(f"Saved WAV: {recording.path}")
    print(f"Duration: {recording.duration_seconds:g} seconds")
    print(f"Sample rate: {recording.sample_rate} Hz")
    print(f"Channels: {recording.channels}")
    print("No transcription API was called.")
    print("Delete it when finished with: recording.cleanup() or rm <path>.")


if __name__ == "__main__":
    main()
