import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pytest

from fgc_detector.cli import _pump, main
from fgc_detector.confirmer import Confirmer, ConfirmerConfig
from fgc_detector.detectors.registry import NullDetector, register
from fgc_detector.types import Frame, Game

TS = datetime(2026, 7, 21, 20, 0, 0, tzinfo=timezone.utc)
IMAGE = np.zeros((1080, 1920, 3), dtype=np.uint8)


def test_no_subcommand_prints_usage_and_fails():
    with pytest.raises(SystemExit):
        main([])


def test_unknown_subcommand_fails():
    with pytest.raises(SystemExit):
        main(["frobnicate"])


def test_replay_on_a_missing_video_returns_nonzero(tmp_path, capsys):
    register(NullDetector(Game.SF6))
    code = main(["replay", "--video", str(tmp_path / "absent.mp4"), "--game", "sf6"])
    captured = capsys.readouterr()
    assert code == 2
    assert "could not open video" in captured.err


def test_roi_on_a_missing_sample_returns_nonzero(tmp_path, capsys):
    register(NullDetector(Game.SF6))
    code = main(["roi", "--game", "sf6", "--sample", str(tmp_path / "absent.png")])
    captured = capsys.readouterr()
    assert code == 2
    assert "could not read sample image" in captured.err


def test_replay_contains_a_mid_vod_failure_and_exits_nonzero(tmp_path, capsys, monkeypatch):
    """A decode/detector failure partway through a VOD must produce a
    readable stderr message and exit 2, not a raw traceback."""
    register(NullDetector(Game.SF6))
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"not really a video, just needs to exist")

    def boom(*_args, **_kwargs):
        raise RuntimeError("decode exploded")

    monkeypatch.setattr("fgc_detector.cli.run_offline", boom)

    code = main(["replay", "--video", str(video_path), "--game", "sf6"])
    captured = capsys.readouterr()
    assert code == 2
    assert "decode exploded" in captured.err


# --- _pump: the live pipeline's per-frame loop, exercised without OBS/asyncio server ---


class FakeSource:
    """A minimal FrameSource-like stand-in: a generator of frames plus a
    `connected` flag, exactly what `_pump` reads from its source."""

    def __init__(self, frame_factory):
        self.connected = True
        self._frame_factory = frame_factory

    def frames(self):
        return self._frame_factory()


@dataclass
class StatusSnapshot:
    game: Game


class RecordingServer:
    """Stands in for EventServer: records every status/event broadcast."""

    def __init__(self, confirmer):
        self.confirmer = confirmer
        self.broadcasts = []

    def status_event(self, now):
        return StatusSnapshot(game=self.confirmer.game)

    async def broadcast(self, event):
        self.broadcasts.append(event)


def test_pump_broadcasts_status_when_only_the_game_changes():
    """Finding 1: an IDLE-to-IDLE set_game (the normal between-matches case)
    must still trigger a status broadcast, so the dashboard doesn't keep
    showing the previous game."""
    register(NullDetector(Game.SF6))
    register(NullDetector(Game.TEKKEN8))
    confirmer = Confirmer(Game.SF6, ConfirmerConfig())

    def make_frames():
        yield Frame(image=IMAGE, captured_at=TS)
        # Simulates a dashboard-issued set_game landing between frames while
        # the confirmer was already IDLE: state/armed/connected are unchanged.
        confirmer.set_game(Game.TEKKEN8)
        yield Frame(image=IMAGE, captured_at=TS)

    source = FakeSource(make_frames)
    server = RecordingServer(confirmer)

    asyncio.run(_pump(source, confirmer, server, recorder=None))

    statuses = [b for b in server.broadcasts if isinstance(b, StatusSnapshot)]
    assert [s.game for s in statuses] == [Game.SF6, Game.TEKKEN8]


def test_pump_contains_a_per_frame_exception_and_keeps_pumping(caplog):
    """Finding 2b: a per-frame failure (here, UnknownGameError because no
    detector is registered for the confirmer's game) must be logged and must
    not stop the pump loop from consuming subsequent frames."""
    confirmer = Confirmer(Game.SF6, ConfirmerConfig())  # no detector registered
    pulled = []

    def make_frames():
        for index in range(2):
            pulled.append(index)
            yield Frame(image=IMAGE, captured_at=TS)

    source = FakeSource(make_frames)
    server = RecordingServer(confirmer)

    with caplog.at_level("ERROR"):
        asyncio.run(_pump(source, confirmer, server, recorder=None))

    assert pulled == [0, 1], "both frames must have been pulled despite the failure on each"
    assert caplog.text.count("pump iteration failed") == 2
