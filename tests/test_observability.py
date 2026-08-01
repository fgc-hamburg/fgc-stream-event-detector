import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from fgc_detector.events import MatchEndEvent
from fgc_detector.observability import FireRecorder
from fgc_detector.types import Frame, Game, Observation, Screen, Side

TS = datetime(2026, 7, 21, 20, 15, 30, tzinfo=timezone.utc)


def _frame() -> Frame:
    return Frame(image=np.full((1080, 1920, 3), 77, dtype=np.uint8), captured_at=TS)


def _event() -> MatchEndEvent:
    return MatchEndEvent(game=Game.SF6, winner=Side.P1, confidence=0.87, ts=TS)


def _observation() -> Observation:
    return Observation(
        screen=Screen.MATCH_END,
        winner=Side.P1,
        confidence=0.87,
        debug={"p1_marker_2": 0.94, "p2_marker_2": 0.03},
    )


def test_writes_both_a_png_and_a_json_sidecar(tmp_path):
    recorder = FireRecorder(tmp_path)
    png_path = recorder.record(_event(), _frame(), _observation())
    assert png_path.exists()
    assert png_path.with_suffix(".json").exists()


def test_png_contains_the_triggering_frame(tmp_path):
    png_path = FireRecorder(tmp_path).record(_event(), _frame(), _observation())
    image = cv2.imread(str(png_path))
    assert image.shape == (1080, 1920, 3)
    assert int(image[0, 0, 0]) == 77


def test_sidecar_records_the_full_debug_mapping(tmp_path):
    png_path = FireRecorder(tmp_path).record(_event(), _frame(), _observation())
    sidecar = json.loads(png_path.with_suffix(".json").read_text())
    assert sidecar["event"]["winner"] == "p1"
    assert sidecar["debug"] == {"p1_marker_2": 0.94, "p2_marker_2": 0.03}
    assert sidecar["screen"] == "MATCH_END"


def test_filename_is_sortable_and_identifies_the_call(tmp_path):
    png_path = FireRecorder(tmp_path).record(_event(), _frame(), _observation())
    assert png_path.name.startswith("2026-07-21T20-15-30")
    assert "sf6" in png_path.name
    assert "p1" in png_path.name


def test_creates_the_output_directory_if_absent(tmp_path):
    target = tmp_path / "nested" / "fires"
    FireRecorder(target).record(_event(), _frame(), _observation())
    assert target.is_dir()


def test_two_fires_in_the_same_second_do_not_collide(tmp_path):
    recorder = FireRecorder(tmp_path)
    first = recorder.record(_event(), _frame(), _observation())
    second = recorder.record(_event(), _frame(), _observation())
    assert first != second


def test_two_recorder_instances_do_not_collide_across_a_restart(tmp_path):
    # Simulates a process restart: a brand-new FireRecorder, with its own
    # fresh in-memory counter, pointed at a directory that already holds
    # evidence from a previous instance with the same event timestamp.
    first = FireRecorder(tmp_path).record(_event(), _frame(), _observation())
    second = FireRecorder(tmp_path).record(_event(), _frame(), _observation())

    assert first is not None
    assert second is not None
    assert first != second
    assert first.exists()
    assert second.exists()
    assert first.with_suffix(".json").exists()
    assert second.with_suffix(".json").exists()


def test_failed_image_write_returns_none_and_logs_without_raising(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(cv2, "imwrite", lambda *args, **kwargs: False)
    with caplog.at_level("ERROR"):
        result = FireRecorder(tmp_path).record(_event(), _frame(), _observation())

    assert result is None
    assert any(
        record.levelname == "ERROR" for record in caplog.records
    ), "expected an ERROR log on write failure"
    # No PNG or sidecar left behind (nothing to leave, imwrite reported failure).
    assert list(tmp_path.iterdir()) == []


def test_unwritable_output_directory_returns_none_and_does_not_raise(
    tmp_path, caplog
):
    target = tmp_path / "fires"
    target.mkdir()
    target.chmod(0o400)  # read-only: mkdir(exist_ok=True) succeeds, writes fail
    try:
        with caplog.at_level("ERROR"):
            result = FireRecorder(target).record(_event(), _frame(), _observation())
    finally:
        target.chmod(0o700)  # restore so tmp_path cleanup can remove it

    assert result is None
    assert any(record.levelname == "ERROR" for record in caplog.records)


def test_sidecar_write_failure_removes_the_orphaned_png(tmp_path, monkeypatch, caplog):
    """Finding 2: if the PNG write succeeds but the sidecar write then fails
    (e.g. disk fills between the two), the PNG must not be left behind with
    no metadata to explain it — evidence must stay all-or-nothing."""
    real_write_text = Path.write_text

    def boom(self, *args, **kwargs):
        if self.suffix == ".json":
            raise OSError("disk full")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", boom)

    with caplog.at_level("ERROR"):
        result = FireRecorder(tmp_path).record(_event(), _frame(), _observation())

    assert result is None
    assert list(tmp_path.iterdir()) == [], (
        "the PNG must be removed when its sidecar fails to write, "
        "so no orphaned evidence file is left with no metadata"
    )
    assert any(record.levelname == "ERROR" for record in caplog.records)


def test_sidecar_write_failure_and_unremovable_png_still_does_not_raise(
    tmp_path, monkeypatch, caplog
):
    """The cleanup itself must be contained: if even removing the orphaned
    PNG fails, record() must still return None instead of raising into the
    event path."""
    real_write_text = Path.write_text

    def boom_write_text(self, *args, **kwargs):
        if self.suffix == ".json":
            raise OSError("disk full")
        return real_write_text(self, *args, **kwargs)

    def boom_unlink(self, *args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "write_text", boom_write_text)
    monkeypatch.setattr(Path, "unlink", boom_unlink)

    with caplog.at_level("ERROR"):
        result = FireRecorder(tmp_path).record(_event(), _frame(), _observation())

    assert result is None
    assert any(record.levelname == "ERROR" for record in caplog.records)
