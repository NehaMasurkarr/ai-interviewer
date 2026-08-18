import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Union, runtime_checkable


PlaybackPath = Union[str, Path]
SUPPORTED_PLAYBACK_EXTENSIONS = frozenset({".mp3", ".wav"})
DEFAULT_AFPLAY_PATH = Path("/usr/bin/afplay")


class AudioPlaybackError(RuntimeError):
    """Base error for local audio playback failures."""


class PlaybackFileNotFoundError(AudioPlaybackError):
    """Raised when playback input does not identify a file."""


class PlaybackFileEmptyError(AudioPlaybackError):
    """Raised when playback input contains no audio bytes."""


class UnsupportedPlaybackFormatError(AudioPlaybackError):
    """Raised when the local player does not support a file format."""


class PlaybackBackendUnavailableError(AudioPlaybackError):
    """Raised when the configured playback executable is unavailable."""


class PlaybackBackendError(AudioPlaybackError):
    """Raised when a playback process fails."""


@dataclass(frozen=True)
class PlaybackResult:
    """Result of synchronous audio playback."""

    path: Path
    completed: bool


@runtime_checkable
class AudioPlayer(Protocol):
    """Framework-independent blocking audio-player contract."""

    def play(self, audio_path: PlaybackPath) -> PlaybackResult:
        ...


class MacOSAfplayAudioPlayer:
    """Blocking local playback through the macOS afplay executable."""

    def __init__(
        self,
        executable: PlaybackPath = DEFAULT_AFPLAY_PATH,
        *,
        process_runner: Callable = subprocess.run,
    ):
        self.executable = Path(executable)
        self.process_runner = process_runner

    def play(self, audio_path: PlaybackPath) -> PlaybackResult:
        path = validate_playback_file(audio_path)
        self._validate_backend()

        try:
            self.process_runner(
                [str(self.executable), str(path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as error:
            raise PlaybackBackendUnavailableError(
                f"Playback backend was not found: {self.executable}"
            ) from error
        except Exception as error:
            raise PlaybackBackendError(
                f"Audio playback failed for: {path}"
            ) from error

        # process_runner is synchronous; reaching here means afplay exited
        # successfully and playback has finished.
        return PlaybackResult(path=path, completed=True)

    def _validate_backend(self) -> None:
        if not self.executable.is_file() or not os.access(
            self.executable, os.X_OK
        ):
            raise PlaybackBackendUnavailableError(
                f"Playback backend is unavailable: {self.executable}"
            )


def validate_playback_file(audio_path: PlaybackPath) -> Path:
    """Validate an existing local audio file before playback."""

    if audio_path is None or not str(audio_path).strip():
        raise PlaybackFileNotFoundError("Playback file path is required.")

    path = Path(audio_path)

    if not path.is_file():
        raise PlaybackFileNotFoundError(
            f"Playback file not found: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_PLAYBACK_EXTENSIONS:
        raise UnsupportedPlaybackFormatError(
            f"Unsupported playback format: {path.suffix or '(none)'}."
        )

    if path.stat().st_size == 0:
        raise PlaybackFileEmptyError(f"Playback file is empty: {path}")

    return path
