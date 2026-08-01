"""Avatar Legends round-pip detector.

Avatar shows each player's round wins as two angular bars flanking the central
clock emblem: P1's two bars (left) fill red, P2's two (right) fill blue. Two
filled bars = that side won the match (best of 3). This reads the four bars by
colour in one frame and reports what it sees; the decision that a match ended
belongs to the marker Confirmer, not this pure detector.

Pips fill with a saturated colour (not brightness) and empty pips carry bright
cyan/gold outlines, so they are read with color_fill_ratio rather than the
brightness-based fill_ratio -- see docs/superpowers/specs/2026-07-30-avatar-
legends-detector.md and the 2026-07-30 calibration report for why these
constants, and how they were measured (never guessed).

Character select is intentionally NOT handled here: calibration (Task 3)
traced ~505s of real footage end-to-end and never observed a character-select
screen (matches go results-menu -> title card -> next match directly), so its
ROI cannot be measured and guessing it is forbidden. This is safe because the
marker Confirmer's cooldown is also released by a fresh, agreeing 0-0 reading
between matches (this detector publishes DETAIL_P1_ROUNDS/DETAIL_P2_ROUNDS on
every IN_MATCH observation, including 0-0), so CHAR_SELECT is not required for
Avatar's cooldown to clear.
"""

from __future__ import annotations

import cv2

from ..types import (
    DETAIL_P1_ROUNDS,
    DETAIL_P2_ROUNDS,
    EventType,
    Frame,
    Game,
    Observation,
    Screen,
    Side,
)
from .registry import register
from .roi import Roi, color_fill_ratio

#: Canonical resolution these ROIs are expressed in. Frames are normalised to
#: this before observe() runs.
CANONICAL_SIZE = (1920, 1080)

# --- MEASURED CONSTANTS: transcribed verbatim from the Task 3 calibration
# --- report (docs/superpowers/reports/2026-07-30-avatar-calibration.md).
EMBLEM_ROI = Roi(930, 45, 60, 20)   # narrow band inside the dark hexagon,
                                    # above the countdown digits; HUD-present anchor
P1_PIP_1 = Roi(892, 85, 12, 18)
P1_PIP_2 = Roi(892, 115, 12, 18)
P2_PIP_1 = Roi(1014, 83, 12, 18)
P2_PIP_2 = Roi(1014, 113, 12, 18)

RED_HUE = (170, 20)      # (hue_lo, hue_hi), wrap-around near 0/179
BLUE_HUE = (95, 140)     # (hue_lo, hue_hi), tight band excluding cyan ~90
SAT_MIN = 60
VAL_MIN = 150
#: pip fill-ratio at or above which a bar counts as lit.
PIP_LIT = 0.4

#: emblem mean-grayscale at or below which the HUD is considered present.
EMBLEM_DARK_MAX = 80.0

ROUNDS_TO_WIN = 2


class AvatarPipDetector:
    """Counts lit round pips by colour. Stateless and pure."""

    canonical_size = CANONICAL_SIZE
    game = Game.AVATAR

    def rois(self) -> dict[str, Roi]:
        return {
            "emblem": EMBLEM_ROI,
            "p1_pip_1": P1_PIP_1,
            "p1_pip_2": P1_PIP_2,
            "p2_pip_1": P2_PIP_1,
            "p2_pip_2": P2_PIP_2,
        }

    def supported_events(self) -> frozenset[EventType]:
        return frozenset({EventType.MATCH_END})

    def _lit(self, image, rois, hue) -> tuple[int, list[float]]:
        ratios = [
            color_fill_ratio(
                image, roi, hue_lo=hue[0], hue_hi=hue[1],
                sat_min=SAT_MIN, val_min=VAL_MIN,
            )
            for roi in rois
        ]
        return sum(1 for r in ratios if r >= PIP_LIT), ratios

    def _emblem_mean(self, image) -> float:
        patch = EMBLEM_ROI.crop(image)
        if patch.size == 0:
            return 255.0
        return float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).mean())

    def observe(self, frame: Frame) -> Observation:
        image = frame.image

        emblem_mean = self._emblem_mean(image)
        if emblem_mean > EMBLEM_DARK_MAX:
            return Observation(screen=Screen.UNKNOWN, debug={"emblem_mean": emblem_mean})

        p1_lit, p1_ratios = self._lit(image, (P1_PIP_1, P1_PIP_2), RED_HUE)
        p2_lit, p2_ratios = self._lit(image, (P2_PIP_1, P2_PIP_2), BLUE_HUE)
        debug = {
            "emblem_mean": emblem_mean,
            "p1_ratios": p1_ratios,
            "p2_ratios": p2_ratios,
        }
        details = {DETAIL_P1_ROUNDS: str(p1_lit), DETAIL_P2_ROUNDS: str(p2_lit)}

        p1_won = p1_lit >= ROUNDS_TO_WIN
        p2_won = p2_lit >= ROUNDS_TO_WIN
        if p1_won == p2_won:
            # Neither done, or both read done (impossible in a real match ->
            # a misread). Refuse to guess a winner.
            return Observation(screen=Screen.IN_MATCH, details=details, debug=debug)

        winner = Side.P1 if p1_won else Side.P2
        winner_ratios = p1_ratios if p1_won else p2_ratios
        return Observation(
            screen=Screen.MATCH_END,
            winner=winner,
            confidence=min(winner_ratios),
            details=details,
            debug=debug,
        )


register(AvatarPipDetector())
