import tempfile
import wave
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol, Union, runtime_checkable

from src.speech.voice_activity import (
    RmsVoiceActivityDetector,
    VoiceActivityDetectionError,
    VoiceActivityDetector,
)


DEFAULT_DURATION_SECONDS = 30.0
DEFAULT_MAX_DURATION_SECONDS = 300.0
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2


class RecordingStopReason(str, Enum):
    FIXED_DURATION = "FIXED_DURATION"
    SILENCE = "SILENCE"
    MAX_DURATION = "MAX_DURATION"


@dataclass(frozen=True)
class AutomaticRecordingConfig:
    """Central configuration for silence-detected answer recording."""

    speech_energy_threshold: float = 500.0
    initial_speech_timeout_seconds: float = 8.0
    end_silence_seconds: float = 2.0
    minimum_speech_seconds: float = 0.4
    max_answer_duration_seconds: float = 120.0
    chunk_duration_seconds: float = 0.1

    def __post_init__(self):
        values = {
            "speech_energy_threshold": self.speech_energy_threshold,
            "initial_speech_timeout_seconds": self.initial_speech_timeout_seconds,
            "end_silence_seconds": self.end_silence_seconds,
            "minimum_speech_seconds": self.minimum_speech_seconds,
            "max_answer_duration_seconds": self.max_answer_duration_seconds,
            "chunk_duration_seconds": self.chunk_duration_seconds,
        }
        for name, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise InvalidAutomaticRecordingConfigError(
                    f"{name} must be a positive number."
                )

        if self.minimum_speech_seconds > self.max_answer_duration_seconds:
            raise InvalidAutomaticRecordingConfigError(
                "minimum_speech_seconds cannot exceed maximum answer duration."
            )

OutputPath = Union[str, Path]


class AudioRecordingError(RuntimeError):
    """Base error for local microphone recording failures."""


class InvalidRecordingDurationError(AudioRecordingError):
    """Raised when a requested duration is invalid or unsafe."""


class InvalidSampleRateError(AudioRecordingError):
    """Raised when the configured sample rate is invalid."""


class InvalidChannelCountError(AudioRecordingError):
    """Raised when the configured channel count is invalid."""


class MicrophoneUnavailableError(AudioRecordingError):
    """Raised when no usable local microphone backend is available."""


class MicrophonePermissionError(AudioRecordingError):
    """Raised when microphone access is denied."""


class RecordingBackendError(AudioRecordingError):
    """Raised when the microphone backend cannot capture audio."""


class RecordedAudioFileMissingError(AudioRecordingError):
    """Raised when recording did not produce the requested WAV file."""


class RecordedAudioFileEmptyError(AudioRecordingError):
    """Raised when recording produced an empty WAV file."""


class InvalidAutomaticRecordingConfigError(AudioRecordingError):
    """Raised when automatic recording configuration is invalid."""


class SpeechStartTimeoutError(AudioRecordingError):
    """Raised when meaningful candidate speech does not begin in time."""


class NoSpeechDetectedError(AudioRecordingError):
    """Raised when captured activity does not meet the speech minimum."""


class VoiceActivityRecordingError(AudioRecordingError):
    """Raised when automatic recording cannot evaluate speech activity."""


@dataclass(frozen=True)
class RecordedAudio:
    """Metadata for a WAV recording retained on disk."""

    path: Path
    format: str
    duration_seconds: float
    sample_rate: int
    channels: int
    stop_reason: RecordingStopReason = RecordingStopReason.FIXED_DURATION

    def cleanup(self) -> None:
        """Explicitly remove the recording if it still exists."""

        self.path.unlink(missing_ok=True)


@runtime_checkable
class AudioRecorder(Protocol):
    """Provider-independent microphone recorder contract."""

    def record(
        self,
        duration_seconds: Optional[float] = None,
        output_path: Optional[OutputPath] = None,
    ) -> RecordedAudio:
        ...


@runtime_checkable
class MicrophoneCaptureBackend(Protocol):
    """Low-level backend that returns signed 16-bit PCM bytes."""

    def capture(
        self,
        *,
        duration_seconds: float,
        sample_rate: int,
        channels: int,
    ) -> bytes:
        ...


@runtime_checkable
class ChunkedMicrophoneCaptureBackend(Protocol):
    """Backend extension yielding sequential signed 16-bit PCM chunks."""

    def capture_chunks(
        self,
        *,
        chunk_duration_seconds: float,
        sample_rate: int,
        channels: int,
    ) -> Iterable[bytes]:
        ...


class SoundDeviceMicrophoneBackend:
    """Local microphone capture using the optional sounddevice package."""

    def __init__(self, sounddevice_module=None):
        if sounddevice_module is None:
            try:
                import sounddevice as sounddevice_module
            except Exception as error:
                raise MicrophoneUnavailableError(
                    "The sounddevice package is required for local "
                    "microphone recording."
                ) from error

        self.sounddevice = sounddevice_module

        try:
            self.sounddevice.query_devices(kind="input")
        except Exception as error:
            raise MicrophoneUnavailableError(
                "No usable input microphone was found."
            ) from error

    def capture(
        self,
        *,
        duration_seconds: float,
        sample_rate: int,
        channels: int,
    ) -> bytes:
        frame_count = round(duration_seconds * sample_rate)
        recording = self.sounddevice.rec(
            frame_count,
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
        )
        self.sounddevice.wait()
        return recording.tobytes()

    def capture_chunks(
        self,
        *,
        chunk_duration_seconds: float,
        sample_rate: int,
        channels: int,
    ) -> Iterable[bytes]:
        frames_per_chunk = max(1, round(chunk_duration_seconds * sample_rate))

        with self.sounddevice.RawInputStream(
            samplerate=sample_rate,
            blocksize=frames_per_chunk,
            channels=channels,
            dtype="int16",
        ) as stream:
            while True:
                data, _overflowed = stream.read(frames_per_chunk)
                yield bytes(data)


class LocalMicrophoneRecorder:
    """Capture bounded local microphone audio into a retained WAV file."""

    def __init__(
        self,
        backend: Optional[MicrophoneCaptureBackend] = None,
        *,
        backend_factory: Callable[[], MicrophoneCaptureBackend] = (
            SoundDeviceMicrophoneBackend
        ),
        default_duration_seconds: float = DEFAULT_DURATION_SECONDS,
        max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        activity_detector: Optional[VoiceActivityDetector] = None,
        automatic_config: Optional[AutomaticRecordingConfig] = None,
    ):
        self.default_duration_seconds = _validate_positive_number(
            default_duration_seconds,
            "default duration",
        )
        self.max_duration_seconds = _validate_positive_number(
            max_duration_seconds,
            "maximum duration",
        )

        if self.default_duration_seconds > self.max_duration_seconds:
            raise InvalidRecordingDurationError(
                "Default recording duration cannot exceed maximum duration."
            )

        self.sample_rate = _validate_sample_rate(sample_rate)
        self.channels = _validate_channels(channels)
        self._backend = backend
        self._backend_factory = backend_factory
        self.activity_detector = activity_detector
        self.automatic_config = (
            automatic_config
            if automatic_config is not None
            else AutomaticRecordingConfig()
        )

    def record(
        self,
        duration_seconds: Optional[float] = None,
        output_path: Optional[OutputPath] = None,
    ) -> RecordedAudio:
        duration = (
            self.default_duration_seconds
            if duration_seconds is None
            else _validate_positive_number(duration_seconds, "duration")
        )

        if duration > self.max_duration_seconds:
            raise InvalidRecordingDurationError(
                "Recording duration cannot exceed "
                f"{self.max_duration_seconds:g} seconds."
            )

        path = self._resolve_output_path(output_path)

        try:
            backend = self._get_backend()
            pcm_data = backend.capture(
                duration_seconds=duration,
                sample_rate=self.sample_rate,
                channels=self.channels,
            )

            if not isinstance(pcm_data, (bytes, bytearray, memoryview)):
                raise RecordingBackendError(
                    "Recording backend returned invalid PCM audio."
                )

            pcm_data = bytes(pcm_data)

            if not pcm_data:
                raise RecordedAudioFileEmptyError(
                    "Recording backend produced no audio data."
                )

            frame_width = SAMPLE_WIDTH_BYTES * self.channels
            if len(pcm_data) % frame_width != 0:
                raise RecordingBackendError(
                    "Recording backend returned misaligned PCM audio."
                )

            self._write_wav(path, pcm_data)
            self._validate_output(path)
        except AudioRecordingError:
            self._remove_failed_output(path)
            raise
        except PermissionError as error:
            self._remove_failed_output(path)
            raise MicrophonePermissionError(
                "Microphone access was denied."
            ) from error
        except Exception as error:
            self._remove_failed_output(path)
            if _looks_like_permission_error(error):
                raise MicrophonePermissionError(
                    "Microphone access was denied."
                ) from error
            raise RecordingBackendError(
                "Microphone recording failed."
            ) from error

        return RecordedAudio(
            path=path,
            format="wav",
            duration_seconds=(
                len(pcm_data)
                / (self.sample_rate * self.channels * SAMPLE_WIDTH_BYTES)
            ),
            sample_rate=self.sample_rate,
            channels=self.channels,
            stop_reason=RecordingStopReason.FIXED_DURATION,
        )

    def record_until_silence(
        self,
        output_path: Optional[OutputPath] = None,
        *,
        config: Optional[AutomaticRecordingConfig] = None,
        on_speech_detected: Optional[Callable[[], None]] = None,
    ) -> RecordedAudio:
        """Capture chunks until meaningful speech is followed by silence."""

        active_config = config or self.automatic_config

        if not isinstance(active_config, AutomaticRecordingConfig):
            raise InvalidAutomaticRecordingConfigError(
                "config must be an AutomaticRecordingConfig."
            )

        detector = self.activity_detector or RmsVoiceActivityDetector(
            active_config.speech_energy_threshold
        )
        path = self._resolve_output_path(output_path)
        captured_chunks = []
        elapsed_seconds = 0.0
        answer_seconds = 0.0
        speech_seconds = 0.0
        silence_seconds = 0.0
        first_activity_seen = False
        meaningful_speech = False
        stop_reason = None

        try:
            backend = self._get_backend()
            capture_chunks = getattr(backend, "capture_chunks", None)

            if not callable(capture_chunks):
                raise RecordingBackendError(
                    "Microphone backend does not support chunked capture."
                )

            chunks = capture_chunks(
                chunk_duration_seconds=active_config.chunk_duration_seconds,
                sample_rate=self.sample_rate,
                channels=self.channels,
            )

            for raw_chunk in chunks:
                chunk = self._validate_pcm_chunk(raw_chunk)
                chunk_seconds = self._pcm_duration(chunk)
                elapsed_seconds += chunk_seconds

                try:
                    contains_speech = detector.is_speech(
                        chunk,
                        channels=self.channels,
                    )
                except VoiceActivityDetectionError:
                    raise
                except Exception as error:
                    raise VoiceActivityDetectionError(
                        "Voice activity detector failed."
                    ) from error

                if contains_speech:
                    if not first_activity_seen:
                        first_activity_seen = True
                    speech_seconds += chunk_seconds
                    silence_seconds = 0.0
                elif first_activity_seen:
                    silence_seconds += chunk_seconds

                if first_activity_seen:
                    captured_chunks.append(chunk)
                    answer_seconds += chunk_seconds

                if (
                    not meaningful_speech
                    and speech_seconds >= active_config.minimum_speech_seconds
                ):
                    meaningful_speech = True
                    if on_speech_detected is not None:
                        on_speech_detected()

                if not meaningful_speech and elapsed_seconds >= (
                    active_config.initial_speech_timeout_seconds
                ):
                    raise SpeechStartTimeoutError(
                        "No meaningful candidate speech was detected before "
                        "the initial speech timeout."
                    )

                if meaningful_speech and answer_seconds >= (
                    active_config.max_answer_duration_seconds
                ):
                    stop_reason = RecordingStopReason.MAX_DURATION
                    break

                if meaningful_speech and silence_seconds >= (
                    active_config.end_silence_seconds
                ):
                    stop_reason = RecordingStopReason.SILENCE
                    break
            else:
                if not meaningful_speech:
                    raise NoSpeechDetectedError(
                        "Audio source ended before meaningful speech was captured."
                    )
                raise RecordingBackendError(
                    "Chunked microphone source ended before a stop condition."
                )

            pcm_data = b"".join(captured_chunks)
            if not pcm_data:
                raise RecordedAudioFileEmptyError(
                    "Automatic recording produced no audio data."
                )

            self._write_wav(path, pcm_data)
            self._validate_output(path)
        except AudioRecordingError:
            self._remove_failed_output(path)
            raise
        except VoiceActivityDetectionError as error:
            self._remove_failed_output(path)
            raise VoiceActivityRecordingError(
                "Voice activity detection failed."
            ) from error
        except PermissionError as error:
            self._remove_failed_output(path)
            raise MicrophonePermissionError(
                "Microphone access was denied."
            ) from error
        except Exception as error:
            self._remove_failed_output(path)
            if _looks_like_permission_error(error):
                raise MicrophonePermissionError(
                    "Microphone access was denied."
                ) from error
            raise RecordingBackendError(
                "Automatic microphone recording failed."
            ) from error

        return RecordedAudio(
            path=path,
            format="wav",
            duration_seconds=self._pcm_duration(pcm_data),
            sample_rate=self.sample_rate,
            channels=self.channels,
            stop_reason=stop_reason,
        )

    def _validate_pcm_chunk(self, value) -> bytes:
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise RecordingBackendError(
                "Chunked backend returned invalid PCM audio."
            )
        value = bytes(value)
        frame_width = SAMPLE_WIDTH_BYTES * self.channels
        if not value or len(value) % frame_width != 0:
            raise RecordedAudioFileEmptyError(
                "Chunked backend returned empty or misaligned PCM audio."
            )
        return value

    def _pcm_duration(self, pcm_data: bytes) -> float:
        return len(pcm_data) / (
            self.sample_rate * self.channels * SAMPLE_WIDTH_BYTES
        )

    def _get_backend(self) -> MicrophoneCaptureBackend:
        if self._backend is not None:
            return self._backend

        try:
            self._backend = self._backend_factory()
        except AudioRecordingError:
            raise
        except Exception as error:
            raise MicrophoneUnavailableError(
                "Microphone backend initialization failed."
            ) from error

        if not hasattr(self._backend, "capture") or not callable(
            self._backend.capture
        ):
            raise MicrophoneUnavailableError(
                "Microphone backend must implement capture()."
            )

        return self._backend

    @staticmethod
    def _resolve_output_path(output_path: Optional[OutputPath]) -> Path:
        if output_path is None:
            temporary = tempfile.NamedTemporaryFile(
                prefix="ai-interviewer-",
                suffix=".wav",
                delete=False,
            )
            path = Path(temporary.name)
            temporary.close()
            return path

        if not str(output_path).strip():
            raise RecordedAudioFileMissingError(
                "Recording output path is required."
            )

        path = Path(output_path)

        if path.suffix.lower() != ".wav":
            raise RecordedAudioFileMissingError(
                "Microphone recordings must use a .wav output path."
            )

        if not path.parent.is_dir():
            raise RecordedAudioFileMissingError(
                f"Recording output directory does not exist: {path.parent}"
            )

        return path

    def _write_wav(self, path: Path, pcm_data: bytes) -> None:
        with wave.open(str(path), "wb") as output:
            output.setnchannels(self.channels)
            output.setsampwidth(SAMPLE_WIDTH_BYTES)
            output.setframerate(self.sample_rate)
            output.writeframes(pcm_data)

    @staticmethod
    def _validate_output(path: Path) -> None:
        if not path.is_file():
            raise RecordedAudioFileMissingError(
                f"Recording output file was not created: {path}"
            )

        if path.stat().st_size == 0:
            raise RecordedAudioFileEmptyError(
                f"Recording output file is empty: {path}"
            )

    @staticmethod
    def _remove_failed_output(path: Path) -> None:
        path.unlink(missing_ok=True)


def _validate_positive_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidRecordingDurationError(
            f"Recording {label} must be a number."
        )

    value = float(value)
    if value <= 0:
        raise InvalidRecordingDurationError(
            f"Recording {label} must be greater than zero."
        )

    return value


def _validate_sample_rate(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidSampleRateError(
            "Recording sample rate must be a positive integer."
        )
    return value


def _validate_channels(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidChannelCountError(
            "Recording channel count must be a positive integer."
        )
    return value


def _looks_like_permission_error(error: Exception) -> bool:
    message = str(error).lower()
    return "permission" in message or "not authorized" in message
