import wave
from pathlib import Path

import numpy as np
import pytest

from src.speech.audio_recorder import (
    AutomaticRecordingConfig,
    InvalidAutomaticRecordingConfigError,
    LocalMicrophoneRecorder,
    NoSpeechDetectedError,
    RecordingBackendError,
    RecordingStopReason,
    SpeechStartTimeoutError,
    VoiceActivityRecordingError,
)


SAMPLE_RATE = 1000
CHUNK_SECONDS = 0.1


def pcm(amplitude, channels=1):
    frames = round(SAMPLE_RATE * CHUNK_SECONDS)
    return np.full(
        frames * channels,
        amplitude,
        dtype="<i2",
    ).tobytes()


SILENCE = pcm(0)
SPEECH = pcm(1200)
TINY_NOISE = pcm(100)


class FakeChunkBackend:
    def __init__(self, chunks=None, error=None, events=None):
        self.chunks = chunks or []
        self.error = error
        self.events = events
        self.calls = []

    def capture(self, **kwargs):
        raise AssertionError("Fixed capture should not be used in auto mode.")

    def capture_chunks(self, **kwargs):
        self.calls.append(kwargs)
        if self.events is not None:
            self.events.append("record")
        for chunk in self.chunks:
            if self.events is not None:
                self.events.append("vad")
            yield chunk
        if self.error is not None:
            raise self.error


def config(**overrides):
    values = {
        "speech_energy_threshold": 500,
        "initial_speech_timeout_seconds": 0.5,
        "end_silence_seconds": 0.3,
        "minimum_speech_seconds": 0.2,
        "max_answer_duration_seconds": 2.0,
        "chunk_duration_seconds": CHUNK_SECONDS,
    }
    values.update(overrides)
    return AutomaticRecordingConfig(**values)


def recorder(chunks, **config_overrides):
    backend = FakeChunkBackend(chunks)
    return (
        LocalMicrophoneRecorder(
            backend=backend,
            sample_rate=SAMPLE_RATE,
            channels=1,
            automatic_config=config(**config_overrides),
        ),
        backend,
    )


@pytest.mark.parametrize(
    "override",
    [
        {"speech_energy_threshold": 0},
        {"initial_speech_timeout_seconds": -1},
        {"end_silence_seconds": 0},
        {"minimum_speech_seconds": 0},
        {"max_answer_duration_seconds": 0},
        {"chunk_duration_seconds": 0},
        {
            "minimum_speech_seconds": 2,
            "max_answer_duration_seconds": 1,
        },
    ],
)
def test_invalid_automatic_recording_configuration(override):
    with pytest.raises(InvalidAutomaticRecordingConfigError):
        config(**override)


def test_waits_initial_silence_and_short_pause_does_not_stop(tmp_path):
    chunks = (
        [SILENCE, SILENCE]
        + [SPEECH] * 3
        + [SILENCE]
        + [SPEECH] * 2
        + [SILENCE] * 3
        + [SPEECH]
    )
    subject, backend = recorder(chunks)
    detected = []

    result = subject.record_until_silence(
        tmp_path / "answer.wav",
        on_speech_detected=lambda: detected.append(True),
    )

    assert result.stop_reason == RecordingStopReason.SILENCE
    assert result.duration_seconds == pytest.approx(0.9)
    assert detected == [True]
    assert len(backend.calls) == 1
    with wave.open(str(result.path), "rb") as audio:
        assert audio.getframerate() == SAMPLE_RATE
        assert audio.getnchannels() == 1
        assert audio.getnframes() == 900


def test_long_silence_stops_before_later_chunks(tmp_path):
    subject, _ = recorder([SPEECH] * 2 + [SILENCE] * 3 + [SPEECH] * 5)

    result = subject.record_until_silence(tmp_path / "answer.wav")

    assert result.stop_reason == RecordingStopReason.SILENCE
    assert result.duration_seconds == pytest.approx(0.5)


def test_no_speech_triggers_initial_timeout_and_removes_output(tmp_path):
    subject, _ = recorder([SILENCE] * 10)
    output = tmp_path / "answer.wav"

    with pytest.raises(SpeechStartTimeoutError):
        subject.record_until_silence(output)

    assert not output.exists()


def test_tiny_noise_does_not_satisfy_minimum_speech(tmp_path):
    subject, _ = recorder([TINY_NOISE, SPEECH] + [SILENCE] * 5)

    with pytest.raises(SpeechStartTimeoutError):
        subject.record_until_silence(tmp_path / "answer.wav")


def test_source_end_without_speech_is_clear_error(tmp_path):
    subject, _ = recorder([SILENCE, SILENCE])

    with pytest.raises(NoSpeechDetectedError):
        subject.record_until_silence(tmp_path / "answer.wav")


def test_maximum_duration_stops_valid_speech_safely(tmp_path):
    subject, _ = recorder(
        [SPEECH] * 20,
        max_answer_duration_seconds=0.5,
    )

    result = subject.record_until_silence(tmp_path / "answer.wav")

    assert result.stop_reason == RecordingStopReason.MAX_DURATION
    assert result.duration_seconds == pytest.approx(0.5)
    assert result.path.stat().st_size > 44


class FailingDetector:
    def __init__(self, error):
        self.error = error

    def is_speech(self, chunk, *, channels):
        raise self.error


def test_detector_failure_is_wrapped_and_preserves_cause(tmp_path):
    original = RuntimeError("detector failed")
    subject, _ = recorder([SPEECH])
    subject.activity_detector = FailingDetector(original)

    with pytest.raises(VoiceActivityRecordingError) as captured:
        subject.record_until_silence(tmp_path / "answer.wav")

    assert captured.value.__cause__.__cause__ is original


def test_chunked_microphone_failure_is_wrapped_and_preserves_cause(tmp_path):
    original = RuntimeError("microphone disconnected")
    backend = FakeChunkBackend([SPEECH], error=original)
    subject = LocalMicrophoneRecorder(
        backend=backend,
        sample_rate=SAMPLE_RATE,
        automatic_config=config(minimum_speech_seconds=0.3),
    )

    with pytest.raises(RecordingBackendError) as captured:
        subject.record_until_silence(tmp_path / "answer.wav")

    assert captured.value.__cause__ is original


def test_auto_recording_cleanup_is_unchanged():
    subject, _ = recorder([SPEECH] * 2 + [SILENCE] * 3)

    result = subject.record_until_silence()

    assert result.path.exists()
    result.cleanup()
    assert not result.path.exists()
    result.cleanup()
