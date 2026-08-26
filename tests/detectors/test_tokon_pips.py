"""Unit tests for TokonPipDetector on synthetic frames.

Each frame is painted at the *measured* ROIs (see the 2026-08-26 TOKON
calibration report), so these tests pin the detector's decision rules -- pale
core => empty, disc over background => lit, neither => ambiguous -- without
depending on the real corpus.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from fgc_detector.detectors.registry import get_detector, register
from fgc_detector.detectors.roi import Roi
from fgc_detector.detectors.tokon import (
    CANONICAL_SIZE,
    P1_PIP_CENTRES,
    P2_PIP_CENTRES,
    TokonPipDetector,
    _background,
    _core,
    _icon,
)
from fgc_detector.types import (
    DETAIL_P1_ROUNDS,
    DETAIL_P2_ROUNDS,
    EventType,
    Frame,
    Game,
    Screen,
    Side,
)

STAGE = (120, 60, 40)     # BGR: a saturated blue-ish stage behind the HUD
NEAR_WHITE = (245, 245, 245)
ICON = (40, 90, 235)      # BGR: an orange disc, like the real P/V icons


def _blank() -> np.ndarray:
    return np.full((CANONICAL_SIZE[1], CANONICAL_SIZE[0], 3), STAGE, dtype=np.uint8)


def _paint(img: np.ndarray, roi: Roi, bgr: tuple[int, int, int]) -> None:
    img[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = bgr


def _frame(img: np.ndarray) -> Frame:
    return Frame(image=img, captured_at=datetime.now(timezone.utc))


def _draw_empty(img: np.ndarray, cx: int) -> None:
    """An empty slot: stage everywhere, plus the small near-white circle."""
    _paint(img, _core(cx), NEAR_WHITE)


def _draw_lit(img: np.ndarray, cx: int) -> None:
    """A lit slot: an opaque icon disc over the whole marker band."""
    _paint(img, _icon(cx), ICON)
    _paint(img, _core(cx), ICON)


def _board(p1_lit: int, p2_lit: int) -> np.ndarray:
    """Pips fill centre-outward, so the innermost `n` slots are the lit ones."""
    img = _blank()
    for i, cx in enumerate(reversed(P1_PIP_CENTRES)):
        (_draw_lit if i < p1_lit else _draw_empty)(img, cx)
    for i, cx in enumerate(P2_PIP_CENTRES):
        (_draw_lit if i < p2_lit else _draw_empty)(img, cx)
    return img


# --- registry / protocol ---------------------------------------------------


def test_detector_is_registered_for_tokon() -> None:
    detector = TokonPipDetector()
    register(detector)  # autouse clean_registry fixture clears the import-time one
    assert get_detector(Game.TOKON) is detector


def test_declares_canonical_size_and_supported_events() -> None:
    detector = TokonPipDetector()
    assert detector.canonical_size == (1920, 1080)
    assert detector.supported_events() == frozenset({EventType.MATCH_END})


def test_rois_expose_every_slot_and_stay_inside_the_canonical_frame() -> None:
    rois = TokonPipDetector().rois()
    assert len(rois) == 18  # six slots x (core, icon, background)
    for name, roi in rois.items():
        assert roi.x + roi.w <= CANONICAL_SIZE[0], name
        assert roi.y + roi.h <= CANONICAL_SIZE[1], name


def test_background_roi_never_overlaps_its_own_icon_roi() -> None:
    for cx in P1_PIP_CENTRES + P2_PIP_CENTRES:
        assert _background(cx).y + _background(cx).h <= _icon(cx).y


# --- scores ----------------------------------------------------------------


@pytest.mark.parametrize(
    "p1,p2",
    [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1), (2, 1), (1, 2), (2, 2)],
)
def test_undecided_scores_read_in_match_with_the_right_counts(p1: int, p2: int) -> None:
    observation = TokonPipDetector().observe(_frame(_board(p1, p2)))
    assert observation.screen is Screen.IN_MATCH
    assert observation.winner is None
    assert observation.details[DETAIL_P1_ROUNDS] == str(p1)
    assert observation.details[DETAIL_P2_ROUNDS] == str(p2)


@pytest.mark.parametrize("p2", [0, 1, 2])
def test_three_p1_pips_is_a_p1_match_end(p2: int) -> None:
    observation = TokonPipDetector().observe(_frame(_board(3, p2)))
    assert observation.screen is Screen.MATCH_END
    assert observation.winner is Side.P1
    assert observation.details[DETAIL_P1_ROUNDS] == "3"
    assert observation.confidence > 0.0


@pytest.mark.parametrize("p1", [0, 1, 2])
def test_three_p2_pips_is_a_p2_match_end(p1: int) -> None:
    observation = TokonPipDetector().observe(_frame(_board(p1, 3)))
    assert observation.screen is Screen.MATCH_END
    assert observation.winner is Side.P2
    assert observation.details[DETAIL_P2_ROUNDS] == "3"


def test_both_sides_reading_three_refuses_to_guess_a_winner() -> None:
    """Impossible in a real match, so it is a misread: fail safe, no winner."""
    observation = TokonPipDetector().observe(_frame(_board(3, 3)))
    assert observation.screen is Screen.IN_MATCH
    assert observation.winner is None


# --- ambiguity / fail-safe -------------------------------------------------


def test_bare_stage_with_no_hud_reads_unknown() -> None:
    """No circle and no disc anywhere: every slot is ambiguous."""
    observation = TokonPipDetector().observe(_frame(_blank()))
    assert observation.screen is Screen.UNKNOWN
    assert observation.winner is None
    assert DETAIL_P1_ROUNDS not in observation.details


def test_one_ambiguous_slot_makes_the_whole_frame_unknown() -> None:
    """A would-be 3-0: the outer slot's circle is scrubbed but no disc replaces
    it, which is exactly the sprite-occlusion hazard. It must not read 3-0."""
    img = _board(2, 0)
    outer = P1_PIP_CENTRES[0]
    _paint(img, _core(outer), STAGE)  # circle gone, nothing drawn over it
    observation = TokonPipDetector().observe(_frame(img))
    assert observation.screen is Screen.UNKNOWN
    assert observation.winner is None


def test_a_disc_matching_its_own_background_is_not_read_as_lit() -> None:
    """Positive icon evidence means *differing from the local stage*: an icon
    band painted the same colour as the stage above it is not evidence."""
    img = _board(2, 0)
    outer = P1_PIP_CENTRES[0]
    _paint(img, _core(outer), STAGE)
    _paint(img, _icon(outer), STAGE)
    observation = TokonPipDetector().observe(_frame(img))
    assert observation.screen is not Screen.MATCH_END


def test_observe_is_pure() -> None:
    detector = TokonPipDetector()
    frame = _frame(_board(3, 1))
    assert detector.observe(frame).payload == detector.observe(frame).payload
