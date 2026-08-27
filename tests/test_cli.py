import asyncio
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pytest

from fgc_detector import cli as cli_module
from fgc_detector.cli import _pump, _retune_capture, main
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


# --- _cmd_run: resource guard around serve_ui (Finding 1) ---


def _write_minimal_config(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        game = "sf6"

        [obs]
        source_name = "capture"

        [server]
        """
    )
    return config_path


def test_failing_serve_ui_still_closes_the_obs_source(tmp_path, monkeypatch):
    """Finding 1: serve_ui is called after `source` (an ObsFrameSource) is
    constructed but before the try/finally that closes it. If serve_ui raises
    - most plausibly because ui_port is already bound - source must still be
    closed rather than leaked. Reverting the guard in _cmd_run must make this
    fail."""
    config_path = _write_minimal_config(tmp_path)
    register(NullDetector(Game.SF6))

    closed = []
    real_close = cli_module.ObsFrameSource.close
    monkeypatch.setattr(
        cli_module.ObsFrameSource,
        "close",
        lambda self: (closed.append(True), real_close(self))[1],
    )
    monkeypatch.setattr(
        cli_module,
        "serve_ui",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ui_port in use")),
    )

    with pytest.raises(RuntimeError, match="ui_port in use"):
        main(["run", "--config", str(config_path)])

    assert closed == [True], "source.close() must run even when serve_ui raises"


# --- _pump: the live pipeline's per-frame loop, exercised without OBS/asyncio server ---


class FakeSource:
    """A minimal FrameSource-like stand-in: a generator of frames plus a
    `connected` flag, exactly what `_pump` reads from its source."""

    def __init__(self, frame_factory):
        self.connected = True
        self._frame_factory = frame_factory

    def frames(self):
        return self._frame_factory()

    def stop(self) -> None:
        pass


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


# --- _pump: shutdown-ordering (Finding 1) ---
#
# `_pump` reads frames via `loop.run_in_executor(None, next, frames, ...)`.
# When OBS is down, `next()` blocks *inside that executor thread*, looping on
# backoff sleeps with no `yield` in between — a single call can run for a
# long time. `asyncio.run()`'s internal shutdown sequence is:
#   cancel all tasks -> shutdown_asyncgens() -> shutdown_default_executor()
# and that last step blocks waiting for any in-flight executor call to
# finish. If nothing tells the blocked frame loop to stop *before* that wait
# begins, `asyncio.run()` hangs until the frame source gives up on its own
# (which, against real OBS, it may never do).
#
# `BlockingFrameSource.frames()` mimics that shape: it loops on tiny sleeps,
# checking `_stopped`, and never yields a frame — so `next()` on it blocks
# in the executor thread exactly like the real generator does when OBS is
# unreachable.


class BlockingFrameSource:
    """A FrameSource stand-in whose frames() blocks in the executor thread
    until stop() is called, without ever producing a frame. Simulates OBS
    being unreachable: the target scenario for the Ctrl-C hang."""

    def __init__(self) -> None:
        self.connected = False
        self._stopped = False
        self.entered_loop = threading.Event()

    def stop(self) -> None:
        self._stopped = True

    def frames(self):
        while not self._stopped:
            self.entered_loop.set()
            time.sleep(0.01)
        return
        yield  # pragma: no cover - unreachable; makes this a generator function


def test_cancelling_the_pump_lets_asyncio_run_shut_down_promptly():
    """Finding 1: reproduces asyncio.run()'s real internal shutdown ordering
    (cancel tasks, then shutdown_default_executor()) against a source that
    never stops on its own. If the pump does not signal source.stop() from
    inside the async context (e.g. before it returns/raises on
    cancellation), the still-running executor thread keeps the frame loop
    alive forever and asyncio.run() never returns — shutdown_default_executor
    would wait for a thread that will never finish.

    Run in a background thread with a bounded join so a regression fails
    the test instead of hanging the suite.

    NOTE for future maintainers: this test exercises a real shutdown race by
    design, and a hang IS the failure signal for that race - it is not
    something to "fix" by simplifying the test. The bounded `runner.join`
    below converts a regression into a clean assertion failure for the main
    test process, but the daemon thread itself is left running (Python
    cannot forcibly kill a thread) if the code under test really does hang.
    That means a regression here can still leave a stuck background thread
    inside the pytest worker process after the test "fails" - if a mutation
    or CI run leaves a worker that needs a manual kill, that is this test
    catching a real regression, not a flaky test to retry."""
    confirmer = Confirmer(Game.SF6, ConfirmerConfig())
    source = BlockingFrameSource()
    server = RecordingServer(confirmer)

    async def main_async() -> None:
        task = asyncio.create_task(_pump(source, confirmer, server, recorder=None))
        loop = asyncio.get_running_loop()
        # Wait until the frame loop has actually started running in its
        # executor thread, so the cancellation below lands mid-block —
        # exactly the moment a Ctrl-C during OBS-down retry would land.
        entered = await loop.run_in_executor(None, source.entered_loop.wait, 2)
        assert entered, "background frame loop never started"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    result: dict[str, float | BaseException | None] = {"duration": None, "error": None}

    def run_it() -> None:
        start = time.monotonic()
        try:
            asyncio.run(main_async())
        except BaseException as exc:  # noqa: BLE001 - surfaced to the main thread below
            result["error"] = exc
        finally:
            result["duration"] = time.monotonic() - start

    runner = threading.Thread(target=run_it, daemon=True)
    runner.start()
    # asyncio.run()'s shutdown_default_executor() has no timeout of its own;
    # a generous bound distinguishes "shut down promptly" from "hung" without
    # making the test flaky.
    runner.join(timeout=5)

    assert not runner.is_alive(), (
        "asyncio.run(main_async()) did not return within 5s: the frame "
        "source was never told to stop before shutdown_default_executor() "
        "started waiting on its (still-running) executor thread"
    )
    if result["error"] is not None:
        raise result["error"]
    assert source._stopped is True
    assert result["duration"] < 2.0


# --- live capture retuning --------------------------------------------------


class RecordingSource:
    """Captures what reconfigure() was asked to change."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def reconfigure(self, **kwargs) -> None:
        self.calls.append(kwargs)


def _obs(**overrides):
    from fgc_detector.config import ObsConfig

    base = {
        "source_name": "Game Capture",
        "host": "localhost",
        "port": 4455,
        "password": "",
        "poll_hz": 5.0,
    }
    return ObsConfig(**{**base, **overrides})


def test_retune_capture_does_nothing_when_nothing_changed():
    """A set_game or event-filter edit persists the whole config, so this is
    called on every change. Rebuilding the OBS client each time would drop
    the connection for edits that have nothing to do with capture."""
    source = RecordingSource()
    _retune_capture(source, _obs(), _obs())
    assert source.calls == []


def test_retune_capture_passes_a_new_poll_rate_without_reconnecting():
    source = RecordingSource()
    _retune_capture(source, _obs(), _obs(poll_hz=9.0))
    assert source.calls == [{"poll_hz": 9.0}]


def test_retune_capture_passes_a_new_source_name_without_reconnecting():
    source = RecordingSource()
    _retune_capture(source, _obs(), _obs(source_name="Other"))
    assert source.calls == [{"source_name": "Other"}]


@pytest.mark.parametrize(
    "change", [{"host": "10.0.0.2"}, {"port": 4456}, {"password": "hunter2"}]
)
def test_a_changed_connection_detail_rebuilds_the_client(change):
    source = RecordingSource()
    _retune_capture(source, _obs(), _obs(**change))
    assert list(source.calls[0]) == ["client_factory"]


def test_the_rebuilt_client_factory_uses_the_new_connection_details(monkeypatch):
    built = {}

    def fake_factory(host, port, password):
        built.update(host=host, port=port, password=password)
        return lambda: object()

    monkeypatch.setattr(cli_module, "default_client_factory", fake_factory)
    source = RecordingSource()
    _retune_capture(source, _obs(), _obs(host="10.0.0.2", port=4456, password="hunter2"))
    assert built == {"host": "10.0.0.2", "port": 4456, "password": "hunter2"}
