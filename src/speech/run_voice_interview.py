import argparse
from pathlib import Path

from src.agent.interview_coordinator import live_decision_generator
from src.pipeline.interview_preparation import prepare_interview_coordinator
from src.speech.audio_playback import MacOSAfplayAudioPlayer
from src.speech.audio_recorder import LocalMicrophoneRecorder
from src.speech.audio_recorder import AutomaticRecordingConfig
from src.speech.openai_provider import OpenAISpeechToTextProvider
from src.speech.openai_tts_provider import OpenAITextToSpeechProvider
from src.speech.speech_to_text import CandidateAnswerTranscriber
from src.speech.text_to_speech import InterviewerSpeechSynthesizer
from src.speech.voice_interview_runner import VoiceInterviewRunner


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a local spoken AI interview."
    )
    parser.add_argument("--resume", required=True)
    parser.add_argument("--job-description-file", required=True)
    parser.add_argument("--recording-duration", type=float, default=30.0)
    parser.add_argument(
        "--recording-mode",
        choices=["auto", "fixed"],
        default="auto",
    )
    parser.add_argument("--silence-seconds", type=float, default=2.0)
    parser.add_argument("--speech-threshold", type=float, default=500.0)
    parser.add_argument("--speech-start-timeout", type=float, default=8.0)
    parser.add_argument("--max-answer-duration", type=float, default=120.0)
    parser.add_argument("--voice", default="alloy")
    parser.add_argument("--retain-audio", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    job_path = Path(arguments.job_description_file)

    if not job_path.is_file():
        raise SystemExit(f"Job description file not found: {job_path}")

    job_description = job_path.read_text(encoding="utf-8")

    print("Preparing interview from resume and job description...")

    try:
        coordinator = prepare_interview_coordinator(
            resume_path=arguments.resume,
            job_description=job_description,
            decision_generator=live_decision_generator,
        )
        runner = VoiceInterviewRunner(
            coordinator=coordinator,
            synthesizer=InterviewerSpeechSynthesizer(
                OpenAITextToSpeechProvider(voice=arguments.voice)
            ),
            player=MacOSAfplayAudioPlayer(),
            recorder=LocalMicrophoneRecorder(
                default_duration_seconds=arguments.recording_duration,
                automatic_config=AutomaticRecordingConfig(
                    speech_energy_threshold=arguments.speech_threshold,
                    initial_speech_timeout_seconds=(
                        arguments.speech_start_timeout
                    ),
                    end_silence_seconds=arguments.silence_seconds,
                    max_answer_duration_seconds=(
                        arguments.max_answer_duration
                    ),
                ),
            ),
            transcriber=CandidateAnswerTranscriber(
                OpenAISpeechToTextProvider()
            ),
            retain_audio=arguments.retain_audio,
            recording_mode=arguments.recording_mode,
            before_recording=lambda: print("Listening..."),
            on_speech_detected=lambda: print("Speech detected..."),
            after_recording=lambda recording: print(
                f"Answer captured ({recording.stop_reason.value})."
            ),
            before_transcription=lambda: print("Transcribing..."),
            after_transcription=lambda text: print(
                f"Candidate transcript: {text}"
            ),
        )
    except Exception as error:
        print(f"Unable to initialize voice interview: {error}")
        raise SystemExit(1) from error

    print("Interview ready. Press Enter for each spoken turn, or q to quit.")

    while not coordinator.is_complete:
        print(f"\nInterviewer: {coordinator.current_question}")
        command = input("Press Enter to play/record, or q to quit: ").strip()

        if command.lower() in {"q", "quit", "exit"}:
            break

        try:
            print("AI speaking...")
            result = runner.run_turn(
                recording_duration=arguments.recording_duration
            )
        except Exception as error:
            print(f"Voice turn failed; interview state was not advanced: {error}")
            retry = input("Press Enter to retry, or q to quit: ").strip()
            if retry.lower() in {"q", "quit", "exit"}:
                break
            continue

        if arguments.retain_audio:
            print(f"Retained interviewer audio: {result.speech.path}")
            print(f"Retained candidate audio: {result.recording.path}")

        if result.next_question:
            print(f"Next interviewer question: {result.next_question}")

    print("\nInterview complete." if coordinator.is_complete else "\nInterview ended.")


if __name__ == "__main__":
    main()
