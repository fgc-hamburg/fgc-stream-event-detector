"""Marvel TOKON round-pip detector.

TOKON shows each player's round wins as three small markers flanking the
central match clock: P1's three sit left of it, P2's three right of it, and
both sides fill centre-outward. First to three rounds wins the match.

Unlike Avatar, the two marker states are not "dark vs bright" and not "grey vs
coloured" -- so neither MarkerRoundDetector's brightness fill_ratio nor
color_fill_ratio can read them:

  empty  a small (~10px) near-white circle drawn over the live stage, so the
         slot is ~90% see-through background;
  lit    an opaque ~27px disc bearing a character icon, in one of several
         colours (orange, yellow, ...) with a dark star and a white P/V badge.

The icon colour is not side-coded and varies by character, so the sides are
separated by position only, and "lit" cannot be a hue test.

Reading "the white circle is gone" alone would be an inverted, fail-open test:
anything that covers an *empty* slot -- a character sprite, a super flash, the
HUD vanishing for a KO cinematic -- would read as lit, and three of those would
name a false winner. So a slot is only called lit on *positive* evidence of a
disc: the pale core must be gone AND the icon band must differ from the stage
directly above that same slot. Any slot that satisfies neither test is
ambiguous, and one ambiguous slot makes the whole frame UNKNOWN -- a missed
match end is recoverable, a false one is not.

`region_difference` is used rather than an absolute colour threshold precisely
because the stage behind the pips is arbitrary: comparing a slot against its
own local background is what makes the reading background-independent (an
early hue/saturation candidate was rejected after blue-sky stages bled through
empty pips -- see the 2026-08-26 calibration report, section 4).

Character select is intentionally NOT handled here: calibration traced ~683s of
real footage end-to-end and never observed a character/team-select screen, so
its ROI cannot be measured and guessing it is forbidden. This is safe because
the marker Confirmer's cooldown is also released by a fresh, agreeing 0-0
reading between matches, which this detector publishes on every readable
IN_MATCH frame.
"""

from __future__ import annotations

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
from .roi import Roi, pale_fill_ratio, region_difference

#: Canonical resolution these ROIs are expressed in. Frames are normalised to
#: this before observe() runs.
CANONICAL_SIZE = (1920, 1080)

# --- MEASURED CONSTANTS: transcribed verbatim from the calibration report
# --- (docs/superpowers/reports/2026-08-26-tokon-calibration.md).

#: Pip-slot centre x, outermost first per side. Measured from the near-white
#: empty circles over 94 frames across five stages; pitch is a uniform 38px and
#: the six slots are symmetric about x=960.
P1_PIP_CENTRES = (747, 785, 823)
P2_PIP_CENTRES = (1097, 1135, 1173)

#: All six markers share this vertical geometry: the empty circle occupies
#: y 44..53 and the lit disc y 35..61.
_CORE_Y, _CORE_SIZE = 45, 6      # inside the empty circle, clear of its edge
_ICON_Y, _ICON_H = 36, 15        # upper half of the lit disc, above its badge
_ICON_W = 24
_BG_Y, _BG_H = 18, 12            # stage directly above the slot, no HUD there


def _core(cx: int) -> Roi:
    return Roi(cx - _CORE_SIZE // 2, _CORE_Y, _CORE_SIZE, _CORE_SIZE)


def _icon(cx: int) -> Roi:
    return Roi(cx - _ICON_W // 2, _ICON_Y, _ICON_W, _ICON_H)


def _background(cx: int) -> Roi:
    return Roi(cx - _ICON_W // 2, _BG_Y, _ICON_W, _BG_H)


#: HSV bounds for "this is the near-white empty circle". Deliberately loose:
#: the circle is drawn semi-transparently over the stage.
CORE_SAT_MAX = 90
CORE_VAL_MIN = 150

#: Pale-core fraction at or above which a slot is definitely empty. Measured on
#: the corpus: every empty slot scores >= 0.972, every lit slot scores 0.000.
EMPTY_CORE_PALE_MIN = 0.50
#: ...and at or below which the circle is definitely gone.
LIT_CORE_PALE_MAX = 0.10

#: Icon-band vs local-background difference at or above which a slot carries a
#: disc. Measured on the corpus: clean lit slots >= 0.134, clean empty slots
#: <= 0.109. The threshold sits high in that gap because over-reading a lit pip
#: is the dangerous direction.
LIT_ICON_DIFF_MIN = 0.128

ROUNDS_TO_WIN = 3

_EMPTY, _LIT, _AMBIGUOUS = 0, 1, -1


class TokonPipDetector:
    """Counts lit round pips by icon presence. Stateless and pure."""

    canonical_size = CANONICAL_SIZE
    game = Game.TOKON

    def rois(self) -> dict[str, Roi]:
        out: dict[str, Roi] = {}
        for side, centres in (("p1", P1_PIP_CENTRES), ("p2", P2_PIP_CENTRES)):
            for i, cx in enumerate(centres, start=1):
                out[f"{side}_core_{i}"] = _core(cx)
                out[f"{side}_icon_{i}"] = _icon(cx)
                out[f"{side}_bg_{i}"] = _background(cx)
        return out

    def supported_events(self) -> frozenset[EventType]:
        return frozenset({EventType.MATCH_END})

    def _slot(self, image, cx: int) -> tuple[int, float, float]:
        """Classify one pip slot as empty, lit or ambiguous."""
        pale = pale_fill_ratio(
            image, _core(cx), sat_max=CORE_SAT_MAX, val_min=CORE_VAL_MIN
        )
        if pale >= EMPTY_CORE_PALE_MIN:
            return _EMPTY, pale, 0.0
        diff = region_difference(image, _icon(cx), _background(cx))
        if pale <= LIT_CORE_PALE_MAX and diff >= LIT_ICON_DIFF_MIN:
            return _LIT, pale, diff
        return _AMBIGUOUS, pale, diff

    def observe(self, frame: Frame) -> Observation:
        image = frame.image
        slots = {
            f"{side}_{i}": self._slot(image, cx)
            for side, centres in (("p1", P1_PIP_CENTRES), ("p2", P2_PIP_CENTRES))
            for i, cx in enumerate(centres, start=1)
        }
        debug = {name: {"pale": round(p, 3), "diff": round(d, 3)}
                 for name, (_, p, d) in slots.items()}

        if any(state is _AMBIGUOUS for state, _, _ in slots.values()):
            # At least one slot is neither a white circle nor a disc: the HUD is
            # absent, mid-transition or covered. Refuse to read the frame.
            return Observation(screen=Screen.UNKNOWN, debug=debug)

        p1_lit = sum(1 for n, (s, _, _) in slots.items() if n.startswith("p1") and s is _LIT)
        p2_lit = sum(1 for n, (s, _, _) in slots.items() if n.startswith("p2") and s is _LIT)
        details = {DETAIL_P1_ROUNDS: str(p1_lit), DETAIL_P2_ROUNDS: str(p2_lit)}

        p1_won = p1_lit >= ROUNDS_TO_WIN
        p2_won = p2_lit >= ROUNDS_TO_WIN
        if p1_won == p2_won:
            # Neither done, or both read done (impossible in a real match ->
            # a misread). Refuse to guess a winner.
            return Observation(screen=Screen.IN_MATCH, details=details, debug=debug)

        winner = Side.P1 if p1_won else Side.P2
        prefix = "p1" if p1_won else "p2"
        confidence = min(
            d for n, (_, _, d) in slots.items() if n.startswith(prefix)
        )
        return Observation(
            screen=Screen.MATCH_END,
            winner=winner,
            confidence=confidence,
            details=details,
            debug=debug,
        )


register(TokonPipDetector())
