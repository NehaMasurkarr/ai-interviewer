import subprocess
from pathlib import Path

import pytest

from src.agent.interview_coordinator import InterviewCoordinator
from src.job.job_profile import JobProfile, JobRequirement
from src.planning.interview_plan import InterviewPlan, InterviewTarget
from src.profile.candidate_profile import CandidateProfile
from src.speech.audio_playback import (
    AudioPlayer,
    MacOSAfplayAudioPlayer,
    PlaybackBackendError,
    PlaybackBackendUnavailableError,
    PlaybackFileEmptyError,
    PlaybackFileNotFoundError,
    PlaybackResult,
    UnsupportedPlaybackFormatError,
)


def audio_file(tmp_path: Path, suffix=".mp3") -> Path:
    path = tmp_path / f"question{suffix}"
    path.write_bytes(b"fake audio")
    return path


def executable(tmp_path: Path) -> Path:
    path = tmp_path / "afplay"
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o700)
    return path


class FakeProcessRunner:
    def __init__(self, error=None):
        self.error = error
        self.calls = []
        self.returned = False

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if self.error is not None:
            raise self.error
        self.returned = True


@pytest.mark.parametrize("suffix", [".mp3", ".wav"])
def test_successful_blocking_playback_accepts_mp3_and_wav(tmp_path, suffix):
    path = audio_file(tmp_path, suffix)
    runner = FakeProcessRunner()
    player = MacOSAfplayAudioPlayer(
        executable(tmp_path), process_runner=runner
    )

    result = player.play(path)

    assert isinstance(player, AudioPlayer)
    assert runner.returned
    assert result == PlaybackResult(path=path, completed=True)
    command, options = runner.calls[0]
    assert command == [str(player.executable), str(path)]
    assert options["check"] is True


def test_missing_playback_file_is_rejected(tmp_path):
    player = MacOSAfplayAudioPlayer(executable(tmp_path))

    with pytest.raises(PlaybackFileNotFoundError):
        player.play(tmp_path / "missing.mp3")


def test_empty_playback_file_is_rejected(tmp_path):
    path = tmp_path / "empty.wav"
    path.touch()
    player = MacOSAfplayAudioPlayer(executable(tmp_path))

    with pytest.raises(PlaybackFileEmptyError):
        player.play(path)


def test_unsupported_playback_format_is_rejected(tmp_path):
    player = MacOSAfplayAudioPlayer(executable(tmp_path))

    with pytest.raises(UnsupportedPlaybackFormatError):
        player.play(audio_file(tmp_path, ".aac"))


def test_backend_unavailable_is_rejected(tmp_path):
    player = MacOSAfplayAudioPlayer(tmp_path / "missing-afplay")

    with pytest.raises(PlaybackBackendUnavailableError):
        player.play(audio_file(tmp_path))


def test_backend_execution_failure_wraps_and_preserves_cause(tmp_path):
    original = subprocess.CalledProcessError(1, ["afplay"])
    runner = FakeProcessRunner(error=original)
    player = MacOSAfplayAudioPlayer(
        executable(tmp_path), process_runner=runner
    )

    with pytest.raises(PlaybackBackendError) as captured:
        player.play(audio_file(tmp_path))

    assert captured.value.__cause__ is original


class FakeAudioPlayer:
    def play(self, audio_path):
        return PlaybackResult(Path(audio_path), True)


def test_fake_player_dependency_injection(tmp_path):
    player: AudioPlayer = FakeAudioPlayer()
    path = audio_file(tmp_path)

    assert player.play(path) == PlaybackResult(path, True)


def test_playback_does_not_mutate_coordinator_state(tmp_path):
    plan = InterviewPlan(
        role="Engineer",
        targets=[InterviewTarget("Python", "HIGH", "Required")],
    )
    coordinator = InterviewCoordinator(
        candidate_profile=CandidateProfile(name="Candidate"),
        job_profile=JobProfile(
            role="Engineer",
            requirements=[JobRequirement("Python", "HIGH")],
        ),
        job_description="Python engineer",
        decision_generator=lambda *args: None,
        interview_plan=plan,
    )
    before = coordinator.to_session_dict()

    FakeAudioPlayer().play(audio_file(tmp_path))

    assert coordinator.to_session_dict() == before
