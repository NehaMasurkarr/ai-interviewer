from dataclasses import dataclass
from typing import Callable, Optional

from src.agent.interview_coordinator import CoordinatorTurnResult
from src.speech.audio_playback import (
    AudioPlayer,
    PlaybackResult,
)
from src.speech.audio_recorder import AudioRecorder, RecordedAudio
from src.speech.speech_to_text import (
    CandidateAnswerTranscriber,
    TranscriptionResult,
)
from src.speech.text_to_speech import (
    InterviewerSpeechSynthesizer,
    SynthesizedSpeech,
)


class VoiceInterviewError(RuntimeError):
    """Base error for voice-turn orchestration failures."""


class InvalidPlaybackResultError(VoiceInterviewError):
    """Raised when an audio player violates its blocking contract."""


class VoiceArtifactCleanupError(VoiceInterviewError):
    """Raised when successful turn artifacts cannot be cleaned up."""


class InvalidRecordingModeError(VoiceInterviewError):
    """Raised when the requested voice recording mode is unsupported."""


@dataclass(frozen=True)
class VoiceTurnResult:
    """Result of one complete spoken interview turn."""

    question: str
    speech: SynthesizedSpeech
    playback: PlaybackResult
    recording: RecordedAudio
    transcription: TranscriptionResult
    coordinator_result: CoordinatorTurnResult
    audio_retained: bool

    @property
    def transcript(self) -> str:
        return self.transcription.text

    @property
    def next_question(self) -> Optional[str]:
        return self.coordinator_result.next_question


class VoiceInterviewRunner:
    """Compose existing voice components for one atomic interview turn."""

    def __init__(
        self,
        coordinator,
        synthesizer: InterviewerSpeechSynthesizer,
        player: AudioPlayer,
        recorder: AudioRecorder,
        transcriber: CandidateAnswerTranscriber,
        *,
        retain_audio: bool = False,
        before_recording: Optional[Callable[[], None]] = None,
        after_transcription: Optional[Callable[[str], None]] = None,
        recording_mode: str = "fixed",
        on_speech_detected: Optional[Callable[[], None]] = None,
        after_recording: Optional[Callable[[RecordedAudio], None]] = None,
        before_transcription: Optional[Callable[[], None]] = None,
    ):
        if recording_mode not in {"fixed", "auto"}:
            raise InvalidRecordingModeError(
                "recording_mode must be 'fixed' or 'auto'."
            )

        self.coordinator = coordinator
        self.synthesizer = synthesizer
        self.player = player
        self.recorder = recorder
        self.transcriber = transcriber
        self.retain_audio = retain_audio
        self.before_recording = before_recording
        self.after_transcription = after_transcription
        self.recording_mode = recording_mode
        self.on_speech_detected = on_speech_detected
        self.after_recording = after_recording
        self.before_transcription = before_transcription

    def run_turn(
        self,
        *,
        recording_duration: Optional[float] = None,
    ) -> VoiceTurnResult:
        """Run TTS → playback → recording → STT → submission."""

        question = self.coordinator.current_question
        speech: Optional[SynthesizedSpeech] = None
        recording: Optional[RecordedAudio] = None
        primary_error = None

        try:
            speech = self.synthesizer.synthesize(question)
            playback = self.player.play(speech.path)

            if not isinstance(playback, PlaybackResult) or not playback.completed:
                raise InvalidPlaybackResultError(
                    "Audio player did not confirm completed playback."
                )

            if self.before_recording is not None:
                self.before_recording()

            if self.recording_mode == "auto":
                record_until_silence = getattr(
                    self.recorder,
                    "record_until_silence",
                    None,
                )
                if not callable(record_until_silence):
                    raise InvalidRecordingModeError(
                        "Configured recorder does not support auto mode."
                    )
                recording = record_until_silence(
                    on_speech_detected=self.on_speech_detected
                )
            else:
                recording = self.recorder.record(
                    duration_seconds=recording_duration
                )

            if self.after_recording is not None:
                self.after_recording(recording)

            if self.before_transcription is not None:
                self.before_transcription()

            transcription = self.transcriber.transcribe(recording.path)

            if self.after_transcription is not None:
                self.after_transcription(transcription.text)

            coordinator_result = self.coordinator.submit_answer(
                transcription.text
            )

            return VoiceTurnResult(
                question=question,
                speech=speech,
                playback=playback,
                recording=recording,
                transcription=transcription,
                coordinator_result=coordinator_result,
                audio_retained=self.retain_audio,
            )
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if not self.retain_audio:
                cleanup_errors = []

                for artifact in (recording, speech):
                    if artifact is None:
                        continue
                    try:
                        artifact.cleanup()
                    except Exception as error:
                        cleanup_errors.append(error)

                if cleanup_errors:
                    if primary_error is not None:
                        primary_error.add_note(
                            "Temporary voice artifact cleanup also failed: "
                            f"{cleanup_errors[0]}"
                        )
                    else:
                        raise VoiceArtifactCleanupError(
                            "Temporary voice artifact cleanup failed."
                        ) from cleanup_errors[0]
