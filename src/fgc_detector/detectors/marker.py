"""Round-marker detection, written once for every game that works this way.

Every fighting game in scope is read the same way: count the lit round-win
markers beside each health bar, and if one side has reached its round count the
match is over. Only coordinates, thresholds and round count differ, so a game
contributes a MarkerLayout — data, not code.

Markers are position-fixed and language-independent, so unlike an OCR approach
this imposes no requirement on the game's display language.

A game whose HUD does not fit this shape should implement the Detector protocol
directly rather than bending this class.
"""

from __future__ import annotations

from dataclasses import dataclass

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
from .roi import Roi, fill_ratio


@dataclass(frozen=True)
class MarkerLayout:
    """Everything that differs between games."""

    game: Game
    rounds_to_win: int
    p1_markers: tuple[Roi, ...]
    p2_markers: tuple[Roi, ...]
    health_bar: Roi
    char_select_marker: Roi
    #: Fill ratio at or above which a round marker counts as lit.
    marker_filled: float = 0.60
    #: Fill ratio below which we assume no match HUD is on screen at all.
    health_bar_present: float = 0.30
    #: Fill ratio at or above which the character-select screen is showing.
    char_select_present: float = 0.50

    def __post_init__(self) -> None:
        if self.rounds_to_win < 1:
            raise ValueError(f"rounds_to_win must be >= 1, got {self.rounds_to_win}")
        if len(self.p1_markers) != len(self.p2_markers):
            raise ValueError(
                "both sides must have the same number of markers, got "
                f"{len(self.p1_markers)} and {len(self.p2_markers)}"
            )
        if len(self.p1_markers) != self.rounds_to_win:
            raise ValueError(
                f"rounds_to_win is {self.rounds_to_win} but {len(self.p1_markers)} "
                "markers were given per side"
            )


class MarkerRoundDetector:
    """Classifies a frame by counting lit round markers. Stateless and pure."""

    canonical_size = (1920, 1080)

    def __init__(self, layout: MarkerLayout) -> None:
        self._layout = layout
        self.game = layout.game

    def rois(self) -> dict[str, Roi]:
        layout = self._layout
        named: dict[str, Roi] = {
            "health_bar": layout.health_bar,
            "char_select_marker": layout.char_select_marker,
        }
        for index, roi in enumerate(layout.p1_markers, start=1):
            named[f"p1_round_{index}"] = roi
        for index, roi in enumerate(layout.p2_markers, start=1):
            named[f"p2_round_{index}"] = roi
        return named

    def supported_events(self) -> frozenset[EventType]:
        return frozenset({EventType.MATCH_END})

    def observe(self, frame: Frame) -> Observation:
        layout = self._layout
        image = frame.image
        scores = {name: fill_ratio(image, roi) for name, roi in self.rois().items()}

        # Checked first: character select is the Confirmer's only cooldown exit,
        # so a frame that could read as either must resolve to CHAR_SELECT.
        if scores["char_select_marker"] >= layout.char_select_present:
            return Observation(
                screen=Screen.CHAR_SELECT,
                confidence=scores["char_select_marker"],
                debug=scores,
            )

        if scores["health_bar"] < layout.health_bar_present:
            return Observation(screen=Screen.UNKNOWN, debug=scores)

        p1_lit = self._lit(scores, Side.P1)
        p2_lit = self._lit(scores, Side.P2)

        # Published on every IN_MATCH and MATCH_END observation under the
        # shared constants the Confirmer reads: this is the primary cooldown
        # release path for rematches that skip character select, so the two
        # files must never drift onto different string literals for this key.
        details = {
            DETAIL_P1_ROUNDS: str(p1_lit),
            DETAIL_P2_ROUNDS: str(p2_lit),
        }

        p1_won = p1_lit >= layout.rounds_to_win
        p2_won = p2_lit >= layout.rounds_to_win
        if p1_won == p2_won:
            # Neither side is done, or both read as done — the latter is
            # impossible in a real game and means the ROIs are misreading.
            # Refuse to guess a winner.
            return Observation(screen=Screen.IN_MATCH, debug=scores, details=details)

        winner = Side.P1 if p1_won else Side.P2
        marker_scores = [
            scores[f"{winner.value}_round_{index}"]
            for index in range(1, layout.rounds_to_win + 1)
        ]
        return Observation(
            screen=Screen.MATCH_END,
            winner=winner,
            confidence=min(marker_scores),
            debug=scores,
            details=details,
        )

    def _lit(self, scores: dict[str, float], side: Side) -> int:
        return sum(
            1
            for index in range(1, self._layout.rounds_to_win + 1)
            if scores[f"{side.value}_round_{index}"] >= self._layout.marker_filled
        )
