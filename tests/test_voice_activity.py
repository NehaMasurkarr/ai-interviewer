import numpy as np
import pytest

from src.speech.voice_activity import (
    InvalidVoiceActivityConfigurationError,
    RmsVoiceActivityDetector,
    VoiceActivityDetectionError,
)


def pcm(amplitude, frames=100, channels=1):
    return np.full(
        frames * channels,
        amplitude,
        dtype="<i2",
    ).tobytes()


def test_silence_is_classified_as_silence():
    detector = RmsVoiceActivityDetector(energy_threshold=500)

    assert detector.rms_energy(pcm(0), channels=1) == 0
    assert not detector.is_speech(pcm(0), channels=1)


def test_speech_like_pcm_is_classified_as_speech():
    detector = RmsVoiceActivityDetector(energy_threshold=500)

    assert detector.rms_energy(pcm(1000), channels=1) == 1000
    assert detector.is_speech(pcm(1000), channels=1)


def test_threshold_is_configurable_and_deterministic():
    chunk = pcm(600)

    assert RmsVoiceActivityDetector(500).is_speech(chunk, channels=1)
    assert not RmsVoiceActivityDetector(700).is_speech(chunk, channels=1)


@pytest.mark.parametrize("threshold", [0, -1, True, "500", None])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(InvalidVoiceActivityConfigurationError):
        RmsVoiceActivityDetector(threshold)


def test_stereo_energy_uses_all_channels():
    detector = RmsVoiceActivityDetector(500)
    stereo = np.tile(np.array([1000, 0], dtype="<i2"), 100).tobytes()

    assert detector.rms_energy(stereo, channels=2) == pytest.approx(
        707.106, rel=1e-3
    )
    assert detector.is_speech(stereo, channels=2)


def test_invalid_pcm_is_rejected_without_hardware():
    detector = RmsVoiceActivityDetector(500)

    with pytest.raises(VoiceActivityDetectionError):
        detector.is_speech(b"", channels=1)

    with pytest.raises(VoiceActivityDetectionError):
        detector.is_speech(b"\x00", channels=1)
