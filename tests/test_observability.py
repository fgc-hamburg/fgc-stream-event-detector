import json
from datetime import datetime, timezone

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
