"""Corpus-driven and confirmer-integration tests for AvatarPipDetector.

Part A parametrizes over the real, labelled `samples/avatar/in_match_*.png`
corpus (modeled on tests/detectors/test_sf6_counter.py): every filename is
ground truth for both players' round-pip counts, and (per the 2026-07-30
calibration report and the plan's fail-safe rule) a match reads MATCH_END
with a winner exactly when one side's count is 2, never otherwise.

Part B covers the `between_*.png` corpus. This detector's HUD-present gate is
emblem darkness (see docs/superpowers/reports/2026-07-30-avatar-calibration.md
section 4): a *dark* transition frame (black KO wipe, title card) has no
readable HUD but is not bright either, so it legitimately reads IN_MATCH 0-0
-- harmless (no pips are ever drawn during those frames, so both sides read 0)
and it is exactly the marker Confirmer's 0-0 cooldown-release signal. Only
*bright* sustained between-screens (story dialogue, results menu) clear
EMBLEM_DARK_MAX and read UNKNOWN. So this test does not assert every between
frame reads UNKNOWN -- it asserts the safety invariant that actually matters:
no between frame ever reports a nonzero pip count or fires a match end. That
invariant is meaningful (not vacuous): it would fail immediately if the
detector ever misread stage-colour bleed or a transition flash as a lit pip.

Part C proves the detector's contract (DETAIL_P1_ROUNDS/DETAIL_P2_ROUNDS +
Screen.MATCH_END + winner) actually drives the reused marker Confirmer
end-to-end, using synthetic Observations (no images, no real clock).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import pytest

from fgc_detector.confirmer import Confirmer, ConfirmerConfig
from fgc_detector.detectors.avatar import AvatarPipDetector
from fgc_detector.types import (
    DETAIL_P1_ROUNDS,
    DETAIL_P2_ROUNDS,
    Frame,
    Game,
    Observation,
    Screen,
    Side,
)

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "samples" / "avatar"

_IN_MATCH_PATTERN = re.compile(r"^in_match_p1-(\d)_p2-(\d)_\d+\.png$")

_IN_MATCH_CASES: list[tuple[str, str, str]] = []
_BETWEEN_CASES: list[str] = []
for path in sorted(CORPUS_DIR.glob("*.png")):
    match = _IN_MATCH_PATTERN.match(path.name)
    if match:
        _IN_MATCH_CASES.append((path.name, match.group(1), match.group(2)))
    elif path.name.startswith("between_"):
        _BETWEEN_CASES.append(path.name)

assert _IN_MATCH_CASES, "corpus must contain in_match_* frames"
assert _BETWEEN_CASES, "corpus must contain between_* frames"


def _frame(name: str) -> Frame:
    image = cv2.imread(str(CORPUS_DIR / name))
    assert image is not None, f"failed to load corpus frame {name}"
    return Frame(image=image, captured_at=datetime.now(timezone.utc))


@pytest.fixture(scope="module")
def detector() -> AvatarPipDetector:
    return AvatarPipDetector()


# --- Part A: in_match_* ground truth --------------------------------------


@pytest.mark.parametrize("name,p1,p2", _IN_MATCH_CASES)
def test_in_match_frames_read_the_correct_pip_counts(
    detector: AvatarPipDetector, name: str, p1: str, p2: str
) -> None:
    observation = detector.observe(_frame(name))
    assert observation.details[DETAIL_P1_ROUNDS] == p1
    assert observation.details[DETAIL_P2_ROUNDS] == p2

    if p1 == "2" and p2 != "2":
        assert observation.screen is Screen.MATCH_END
        assert observation.winner is Side.P1
    elif p2 == "2" and p1 != "2":
        assert observation.screen is Screen.MATCH_END
        assert observation.winner is Side.P2
    else:
        assert observation.screen is Screen.IN_MATCH
        assert observation.winner is None


# --- Part B: between_* safety invariant ------------------------------------
#
# See the module docstring: the emblem-darkness gate reads dark transition
# frames (KO wipes, title cards) as IN_MATCH 0-0 by design, so UNKNOWN is not
# asserted here. What must always hold for every between frame -- dark or
# bright -- is that it never reports a nonzero pip count and never fires a
# match end.


@pytest.mark.parametrize("name", _BETWEEN_CASES)
def test_between_frames_never_report_a_pip_or_a_match_end(
    detector: AvatarPipDetector, name: str
) -> None:
    observation = detector.observe(_frame(name))

    assert observation.winner is None
    assert observation.screen is not Screen.MATCH_END
    if DETAIL_P1_ROUNDS in observation.details:
        assert observation.details[DETAIL_P1_ROUNDS] == "0"
    if DETAIL_P2_ROUNDS in observation.details:
        assert observation.details[DETAIL_P2_ROUNDS] == "0"


def test_between_corpus_contains_both_unknown_and_dark_in_match_readings(
    detector: AvatarPipDetector,
) -> None:
    """Guards against the Part B assertion being vacuously true.

    If every between frame happened to read UNKNOWN, the "0"/"0" branches
    above would never execute and the nonzero-pip check would be untested.
    The corpus is known (calibration report §6) to contain both dialogue
    frames (bright -> UNKNOWN) and dark wipes/title card (dark -> IN_MATCH
    0-0), so assert both screens actually appear.
    """
    screens = {detector.observe(_frame(name)).screen for name in _BETWEEN_CASES}
    assert Screen.UNKNOWN in screens
    assert Screen.IN_MATCH in screens


def test_observe_is_pure(detector: AvatarPipDetector) -> None:
    frame = _frame(_IN_MATCH_CASES[0][0])
    first = detector.observe(frame)
    second = detector.observe(frame)
    assert first.payload == second.payload


# --- Part C: confirmer integration -----------------------------------------


def test_pip_sequence_fires_one_match_end_for_p1() -> None:
    """A realistic Avatar observation sequence drives the marker Confirmer.

    IN_MATCH 1-1 for a while, then the detector agrees on MATCH_END 2-1 (P1)
    for >= agreement_frames consecutive frames: exactly one MatchEndEvent
    should fire, naming P1. Synthetic observations only -- no images, no
    real clock (a fixed base datetime plus injected offsets stands in for
    the real clock).
    """
    confirmer = Confirmer(Game.AVATAR, ConfirmerConfig(agreement_frames=3))
    confirmer.arm()
    base = datetime(2026, 7, 30, tzinfo=timezone.utc)

    def obs(screen: Screen, p1: int, p2: int, winner: Side | None = None) -> Observation:
        return Observation(
            screen=screen,
            winner=winner,
            details={DETAIL_P1_ROUNDS: str(p1), DETAIL_P2_ROUNDS: str(p2)},
            confidence=1.0,
        )

    sequence = (
        [obs(Screen.IN_MATCH, 1, 1)] * 3
        + [obs(Screen.MATCH_END, 2, 1, Side.P1)] * 4
    )

    events = []
    for i, observation in enumerate(sequence):
        event = confirmer.observe(observation, base + timedelta(seconds=i))
        if event is not None:
            events.append(event)

    assert len(events) == 1
    assert events[0].winner is Side.P1
