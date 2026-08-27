"""Corpus-driven and confirmer-integration tests for TokonPipDetector.

Part A parametrizes over the real, labelled `samples/tokon/*.png` corpus built
by scripts/build_tokon_corpus.py: every filename is hand-verified ground truth
for both players' round-pip counts, and (per the 2026-08-26 calibration report
and the fail-safe rule) a frame reads MATCH_END with a winner exactly when one
side's count is 3, never otherwise. `occluded_*` frames are HUD-degraded but
still readable and are held to the same exact ground truth.

`sprite_*` frames are the documented adversarial case: a character sprite or a
pip-slide animation completely covers one *empty* slot, so the frame is not
readable and the detector over-reads it by one pip. Those cases are strict
xfails -- the corpus label stays truthful and the test fails loudly if the
behaviour ever changes -- and section 6 of the calibration report shows why
they are safe in the pipeline (each is a single 5Hz sample against the
Confirmer's three-sample agreement requirement).

Part B covers the `between_*` corpus (no HUD at all). Like Avatar, a frame with
no HUD legitimately reads IN_MATCH 0-0 whenever all six slots happen to look
pale -- harmless (no pips are drawn, so both sides read 0) and it is exactly
the marker Confirmer's 0-0 cooldown-release signal. So this test asserts the
safety invariant that actually matters: no between frame ever reports a nonzero
pip count or fires a match end.

Part C proves the detector's contract (DETAIL_P1_ROUNDS/DETAIL_P2_ROUNDS +
Screen.MATCH_END + winner) drives the reused marker Confirmer end-to-end, using
synthetic Observations (no images, no real clock).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import pytest

from fgc_detector.confirmer import Confirmer, ConfirmerConfig
from fgc_detector.detectors.tokon import TokonPipDetector
from fgc_detector.types import (
    DETAIL_P1_ROUNDS,
    DETAIL_P2_ROUNDS,
    Frame,
    Game,
    Observation,
    Screen,
    Side,
)

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "samples" / "tokon"

_SCORED_PATTERN = re.compile(r"^(in_match|occluded|sprite)_p1-(\d)_p2-(\d)_\d+\.png$")

#: A sprite covers P1's empty outer slot in these frames, so it reads lit: the
#: detector over-reads 2 pips as 3. Kept as strict xfail so the corpus label
#: stays honest and any change in behaviour is caught.
_SPRITE_XFAIL = "a sprite covers P1's empty outer slot; the frame is unreadable"

_SCORED_CASES: list = []
_BETWEEN_CASES: list[str] = []
for path in sorted(CORPUS_DIR.glob("*.png")):
    match = _SCORED_PATTERN.match(path.name)
    if match:
        case = (path.name, match.group(2), match.group(3))
        if match.group(1) == "sprite":
            _SCORED_CASES.append(
                pytest.param(
                    *case, marks=pytest.mark.xfail(strict=True, reason=_SPRITE_XFAIL)
                )
            )
        else:
            _SCORED_CASES.append(pytest.param(*case))
    elif path.name.startswith("between_"):
        _BETWEEN_CASES.append(path.name)

assert _SCORED_CASES, "corpus must contain scored frames"
assert _BETWEEN_CASES, "corpus must contain between_* frames"


def _frame(name: str) -> Frame:
    image = cv2.imread(str(CORPUS_DIR / name))
    assert image is not None, f"failed to load corpus frame {name}"
    return Frame(image=image, captured_at=datetime.now(timezone.utc))


@pytest.fixture(scope="module")
def detector() -> TokonPipDetector:
    return TokonPipDetector()


# --- Part A: scored ground truth -------------------------------------------


@pytest.mark.parametrize("name,p1,p2", _SCORED_CASES)
def test_scored_frames_read_the_correct_pip_counts(
    detector: TokonPipDetector, name: str, p1: str, p2: str
) -> None:
    observation = detector.observe(_frame(name))
    assert observation.details.get(DETAIL_P1_ROUNDS) == p1
    assert observation.details.get(DETAIL_P2_ROUNDS) == p2

    if p1 == "3" and p2 != "3":
        assert observation.screen is Screen.MATCH_END
        assert observation.winner is Side.P1
    elif p2 == "3" and p1 != "3":
        assert observation.screen is Screen.MATCH_END
        assert observation.winner is Side.P2
    else:
        assert observation.screen is Screen.IN_MATCH
        assert observation.winner is None


def test_corpus_covers_both_winners_and_a_full_score_ladder() -> None:
    """Guards against a corpus that only proves the easy cases."""
    labels = {name.rsplit("_", 1)[0] for name, _, _ in
              (c.values for c in _SCORED_CASES)}
    assert "in_match_p1-3_p2-0" in labels  # P1 sweep
    assert "in_match_p1-0_p2-3" in labels  # P2 sweep
    assert "in_match_p1-3_p2-2" in labels  # P1 wins a decider
    assert "in_match_p1-2_p2-3" in labels  # P2 wins a decider
    assert "in_match_p1-0_p2-0" in labels
    assert "in_match_p1-2_p2-2" in labels


# --- Part B: between_* safety invariant ------------------------------------


@pytest.mark.parametrize("name", _BETWEEN_CASES)
def test_between_frames_never_report_a_pip_or_a_match_end(
    detector: TokonPipDetector, name: str
) -> None:
    observation = detector.observe(_frame(name))

    assert observation.winner is None
    assert observation.screen is not Screen.MATCH_END
    if DETAIL_P1_ROUNDS in observation.details:
        assert observation.details[DETAIL_P1_ROUNDS] == "0"
    if DETAIL_P2_ROUNDS in observation.details:
        assert observation.details[DETAIL_P2_ROUNDS] == "0"


def test_between_corpus_contains_both_unknown_and_zero_zero_readings(
    detector: TokonPipDetector,
) -> None:
    """Guards against the Part B assertion being vacuously true.

    If every between frame read UNKNOWN, the "0"/"0" branches above would never
    execute. The corpus deliberately contains both kinds (calibration report
    section 6): cut-scenes where some slot is ambiguous -> UNKNOWN, and pale
    cut-scenes where all six read as empty circles -> IN_MATCH 0-0.
    """
    screens = {detector.observe(_frame(name)).screen for name in _BETWEEN_CASES}
    assert Screen.UNKNOWN in screens
    assert Screen.IN_MATCH in screens


# --- Part C: confirmer integration -----------------------------------------


def test_pip_sequence_fires_one_match_end_for_p2() -> None:
    """A realistic TOKON observation sequence drives the marker Confirmer.

    IN_MATCH 2-2 for a while, then the detector agrees on MATCH_END 2-3 (P2)
    for >= agreement_frames consecutive frames: exactly one MatchEndEvent
    should fire, naming P2. Synthetic observations only -- no images, no real
    clock (a fixed base datetime plus injected offsets stands in for it).
    """
    confirmer = Confirmer(Game.TOKON, ConfirmerConfig(agreement_frames=3))
    confirmer.arm()
    base = datetime(2026, 8, 26, tzinfo=timezone.utc)

    def obs(screen: Screen, p1: int, p2: int, winner: Side | None = None) -> Observation:
        return Observation(
            screen=screen,
            winner=winner,
            details={DETAIL_P1_ROUNDS: str(p1), DETAIL_P2_ROUNDS: str(p2)},
            confidence=1.0,
        )

    sequence = (
        [obs(Screen.IN_MATCH, 2, 2)] * 3
        + [obs(Screen.MATCH_END, 2, 3, Side.P2)] * 5
    )

    events = []
    for i, observation in enumerate(sequence):
        event = confirmer.observe(observation, base + timedelta(seconds=i * 0.2))
        if event is not None:
            events.append(event)

    assert len(events) == 1
    assert events[0].winner is Side.P2


def test_a_single_stray_three_pip_reading_never_fires() -> None:
    """The measured sprite-occlusion misread is one 5Hz sample long.

    This is the temporal half of the fail-safe argument in section 6 of the
    calibration report: one stray MATCH_END surrounded by IN_MATCH readings
    must not reach agreement_frames.
    """
    confirmer = Confirmer(Game.TOKON, ConfirmerConfig(agreement_frames=3))
    confirmer.arm()
    base = datetime(2026, 8, 26, tzinfo=timezone.utc)

    def obs(screen: Screen, p1: int, p2: int, winner: Side | None = None) -> Observation:
        return Observation(
            screen=screen,
            winner=winner,
            details={DETAIL_P1_ROUNDS: str(p1), DETAIL_P2_ROUNDS: str(p2)},
            confidence=1.0,
        )

    sequence = (
        [obs(Screen.IN_MATCH, 2, 0)] * 4
        + [obs(Screen.MATCH_END, 3, 0, Side.P1)]  # the sprite-occluded frame
        + [obs(Screen.IN_MATCH, 2, 0)] * 4
    )

    events = [
        event
        for i, observation in enumerate(sequence)
        if (event := confirmer.observe(observation, base + timedelta(seconds=i * 0.2)))
        is not None
    ]
    assert events == []
