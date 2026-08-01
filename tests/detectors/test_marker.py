from datetime import datetime, timezone

import numpy as np
import pytest

from fgc_detector.detectors.marker import MarkerLayout, MarkerRoundDetector
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

CANONICAL = (1920, 1080)

P1_MARKERS = (Roi(100, 100, 20, 20), Roi(130, 100, 20, 20))
P2_MARKERS = (Roi(1800, 100, 20, 20), Roi(1770, 100, 20, 20))
HEALTH_BAR = Roi(200, 60, 400, 20)
CHAR_SELECT = Roi(900, 500, 40, 40)

LAYOUT = MarkerLayout(
    game=Game.SF6,
    rounds_to_win=2,
    p1_markers=P1_MARKERS,
    p2_markers=P2_MARKERS,
    health_bar=HEALTH_BAR,
    char_select_marker=CHAR_SELECT,
)


def _blank() -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _light(image: np.ndarray, roi: Roi) -> np.ndarray:
    image[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = 255
    return image


def _frame(image: np.ndarray) -> Frame:
    return Frame(image=image, captured_at=datetime.now(timezone.utc))


def _in_match_image(p1_lit: int = 0, p2_lit: int = 0) -> np.ndarray:
    image = _light(_blank(), HEALTH_BAR)
    for roi in P1_MARKERS[:p1_lit]:
        _light(image, roi)
    for roi in P2_MARKERS[:p2_lit]:
        _light(image, roi)
    return image


@pytest.fixture
def detector() -> MarkerRoundDetector:
    return MarkerRoundDetector(LAYOUT)


def test_exposes_the_layouts_game(detector):
    assert detector.game is Game.SF6


def test_canonical_size_is_1080p(detector):
    assert detector.canonical_size == CANONICAL


def test_supported_events_is_match_end_only(detector):
    assert detector.supported_events() == frozenset({EventType.MATCH_END})


def test_health_bar_visible_with_no_markers_is_in_match(detector):
    assert detector.observe(_frame(_in_match_image())).screen is Screen.IN_MATCH


def test_partial_markers_is_still_in_match(detector):
    observation = detector.observe(_frame(_in_match_image(p1_lit=1, p2_lit=1)))
    assert observation.screen is Screen.IN_MATCH
    assert observation.winner is None


def test_p1_reaching_the_round_count_ends_the_match(detector):
    observation = detector.observe(_frame(_in_match_image(p1_lit=2, p2_lit=1)))
    assert observation.screen is Screen.MATCH_END
    assert observation.winner is Side.P1


def test_p2_reaching_the_round_count_ends_the_match(detector):
    observation = detector.observe(_frame(_in_match_image(p1_lit=1, p2_lit=2)))
    assert observation.screen is Screen.MATCH_END
    assert observation.winner is Side.P2


def test_both_sides_full_is_not_a_match_end(detector):
    # Impossible in a real game; means the ROIs are misreading. Refuse to
    # guess a winner rather than picking one arbitrarily.
    observation = detector.observe(_frame(_in_match_image(p1_lit=2, p2_lit=2)))
    assert observation.screen is Screen.IN_MATCH
    assert observation.winner is None


def test_no_health_bar_is_unknown(detector):
    assert detector.observe(_frame(_blank())).screen is Screen.UNKNOWN


def test_char_select_marker_wins_over_everything(detector):
    # Character select is checked first: it is the Confirmer's only cooldown
    # exit, so a frame that looks like both must resolve to CHAR_SELECT.
    image = _light(_in_match_image(p1_lit=2), CHAR_SELECT)
    assert detector.observe(_frame(image)).screen is Screen.CHAR_SELECT


def test_debug_carries_every_named_roi_score(detector):
    observation = detector.observe(_frame(_in_match_image(p1_lit=2)))
    assert set(observation.debug) == set(detector.rois())
    assert observation.debug["p1_round_1"] == pytest.approx(1.0)
    assert observation.debug["p2_round_1"] == pytest.approx(0.0)


def test_confidence_on_match_end_is_the_weakest_winning_marker(detector):
    image = _in_match_image(p1_lit=2)
    # Dim the second marker to roughly 70% coverage.
    roi = P1_MARKERS[1]
    image[roi.y + 14 : roi.y + roi.h, roi.x : roi.x + roi.w] = 0
    observation = detector.observe(_frame(image))
    assert observation.screen is Screen.MATCH_END
    assert observation.confidence == pytest.approx(0.7, abs=0.05)


def test_rois_are_named_and_within_canonical_bounds(detector):
    width, height = detector.canonical_size
    names = set(detector.rois())
    assert names == {
        "p1_round_1", "p1_round_2", "p2_round_1", "p2_round_2",
        "health_bar", "char_select_marker",
    }
    for name, roi in detector.rois().items():
        assert roi.x + roi.w <= width, name
        assert roi.y + roi.h <= height, name


def test_detector_is_pure(detector):
    frame = _frame(_in_match_image(p1_lit=2))
    assert detector.observe(frame) == detector.observe(frame)


def test_layout_rejects_a_marker_count_that_disagrees_with_rounds_to_win():
    with pytest.raises(ValueError, match="rounds_to_win"):
        MarkerLayout(
            game=Game.SF6,
            rounds_to_win=3,
            p1_markers=P1_MARKERS,
            p2_markers=P2_MARKERS,
            health_bar=HEALTH_BAR,
            char_select_marker=CHAR_SELECT,
        )


def test_layout_rejects_lopsided_marker_counts():
    with pytest.raises(ValueError, match="same number"):
        MarkerLayout(
            game=Game.SF6,
            rounds_to_win=2,
            p1_markers=P1_MARKERS,
            p2_markers=P2_MARKERS[:1],
            health_bar=HEALTH_BAR,
            char_select_marker=CHAR_SELECT,
        )


def test_in_match_publishes_round_counts_under_shared_constants(detector):
    # Guards the producer/consumer contract with the Confirmer: if this
    # detector ever published under a drifted string literal instead of the
    # shared constants imported from types.py, this test would fail even
    # though every other test here still passes.
    observation = detector.observe(_frame(_in_match_image(p1_lit=1, p2_lit=0)))
    assert observation.screen is Screen.IN_MATCH
    assert observation.details[DETAIL_P1_ROUNDS] == "1"
    assert observation.details[DETAIL_P2_ROUNDS] == "0"


def test_match_end_also_publishes_round_counts_under_shared_constants(detector):
    observation = detector.observe(_frame(_in_match_image(p1_lit=2, p2_lit=1)))
    assert observation.screen is Screen.MATCH_END
    assert observation.details[DETAIL_P1_ROUNDS] == "2"
    assert observation.details[DETAIL_P2_ROUNDS] == "1"


def test_zero_zero_round_counts_are_published_as_strings(detector):
    # This is the primary cooldown-release signal for rematches that skip
    # character select: the Confirmer checks for the literal strings "0"
    # and "0", not the integer 0. If details ever carried ints instead of
    # strings, this would fail.
    observation = detector.observe(_frame(_in_match_image(p1_lit=0, p2_lit=0)))
    assert observation.details[DETAIL_P1_ROUNDS] == "0"
    assert observation.details[DETAIL_P2_ROUNDS] == "0"
