from typing import Protocol, runtime_checkable

import numpy as np


class VoiceActivityDetectionError(RuntimeError):
    """Raised when PCM speech activity cannot be evaluated."""


class InvalidVoiceActivityConfigurationError(VoiceActivityDetectionError):
    """Raised when detector configuration is invalid."""


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """Hardware-independent detector for signed 16-bit PCM chunks."""

    def is_speech(self, pcm_chunk: bytes, *, channels: int) -> bool:
        ...


class RmsVoiceActivityDetector:
    """Classify speech using deterministic PCM root-mean-square energy."""

    def __init__(self, energy_threshold: float = 500.0):
        if (
            isinstance(energy_threshold, bool)
            or not isinstance(energy_threshold, (int, float))
            or energy_threshold <= 0
        ):
            raise InvalidVoiceActivityConfigurationError(
                "Speech energy threshold must be a positive number."
            )

        self.energy_threshold = float(energy_threshold)

    def is_speech(self, pcm_chunk: bytes, *, channels: int) -> bool:
        return self.rms_energy(pcm_chunk, channels=channels) >= (
            self.energy_threshold
        )

    @staticmethod
    def rms_energy(pcm_chunk: bytes, *, channels: int = 1) -> float:
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise VoiceActivityDetectionError(
                "PCM channel count must be a positive integer."
            )

        if not isinstance(pcm_chunk, (bytes, bytearray, memoryview)):
            raise VoiceActivityDetectionError("PCM chunk must contain bytes.")

        pcm_chunk = bytes(pcm_chunk)
        frame_width = 2 * channels

        if not pcm_chunk or len(pcm_chunk) % frame_width != 0:
            raise VoiceActivityDetectionError(
                "PCM chunk is empty or not aligned to 16-bit frames."
            )

        samples = np.frombuffer(pcm_chunk, dtype="<i2").astype(np.float64)
        return float(np.sqrt(np.mean(np.square(samples))))
