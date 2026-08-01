from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from fgc_detector.detectors.avatar import (
    AvatarPipDetector,
    CANONICAL_SIZE,
    EMBLEM_ROI,
    P1_PIP_1,
    P1_PIP_2,
    P2_PIP_1,
    P2_PIP_2,
)
from fgc_detector.detectors.registry import get_detector, register
from fgc_detector.detectors.roi import Roi
from fgc_detector.types import (
    DETAIL_P1_ROUNDS,
    DETAIL_P2_ROUNDS,
    EventType,
    Frame,
    Game,
    Screen,
    Side,
)

RED = (0, 0, 255)     # BGR
BLUE = (255, 0, 0)    # BGR
DARK = (20, 20, 20)


def _blank() -> np.ndarray:
    # Mid-grey background that is neither a lit pip nor the dark emblem.
    return np.full((CANONICAL_SIZE[1], CANONICAL_SIZE[0], 3), 90, dtype=np.uint8)


def _paint(img: np.ndarray, roi: Roi, bgr: tuple[int, int, int]) -> None:
    img[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = bgr


def _frame(img: np.ndarray) -> Frame:
    return Frame(image=img, captured_at=datetime.now(timezone.utc))


def _in_match(p1_lit: int, p2_lit: int) -> np.ndarray:
    img = _blank()
    _paint(img, EMBLEM_ROI, DARK)  # HUD present
    for i, roi in enumerate((P1_PIP_1, P1_PIP_2)):
        _paint(img, roi, RED if i < p1_lit else DARK)
    for i, roi in enumerate((P2_PIP_1, P2_PIP_2)):
        _paint(img, roi, BLUE if i < p2_lit else DARK)
    return img


def test_no_emblem_reads_unknown() -> None:
    img = _blank()  # emblem region left mid-grey: HUD absent
    obs = AvatarPipDetector().observe(_frame(img))
    assert obs.screen is Screen.UNKNOWN


def test_zero_zero_is_in_match_with_zero_rounds() -> None:
    obs = AvatarPipDetector().observe(_frame(_in_match(0, 0)))
    assert obs.screen is Screen.IN_MATCH
    assert obs.details[DETAIL_P1_ROUNDS] == "0"
    assert obs.details[DETAIL_P2_ROUNDS] == "0"
    assert obs.winner is None


def test_one_each_is_in_match_no_winner() -> None:
    obs = AvatarPipDetector().observe(_frame(_in_match(1, 1)))
    assert obs.screen is Screen.IN_MATCH
    assert obs.details[DETAIL_P1_ROUNDS] == "1"
    assert obs.details[DETAIL_P2_ROUNDS] == "1"
    assert obs.winner is None


def test_two_pips_p1_is_match_end_p1() -> None:
    obs = AvatarPipDetector().observe(_frame(_in_match(2, 1)))
    assert obs.screen is Screen.MATCH_END
    assert obs.winner is Side.P1
    assert obs.details[DETAIL_P1_ROUNDS] == "2"


def test_two_pips_p2_is_match_end_p2() -> None:
    obs = AvatarPipDetector().observe(_frame(_in_match(0, 2)))
    assert obs.screen is Screen.MATCH_END
    assert obs.winner is Side.P2


def test_both_two_refuses_to_guess() -> None:
    obs = AvatarPipDetector().observe(_frame(_in_match(2, 2)))
    assert obs.screen is Screen.IN_MATCH
    assert obs.winner is None


def test_observe_is_pure() -> None:
    frame = _frame(_in_match(2, 0))
    d = AvatarPipDetector()
    assert d.observe(frame).payload == d.observe(frame).payload


def test_rois_within_canonical_bounds() -> None:
    w, h = CANONICAL_SIZE
    for roi in AvatarPipDetector().rois().values():
        assert 0 <= roi.x and 0 <= roi.y
        assert roi.x + roi.w <= w and roi.y + roi.h <= h


def test_supported_events_is_match_end_only() -> None:
    assert AvatarPipDetector().supported_events() == frozenset({EventType.MATCH_END})


def test_registered_for_avatar() -> None:
    d = AvatarPipDetector()
    register(d)  # autouse clean_registry fixture clears the import-time registration
    assert get_detector(Game.AVATAR) is d
