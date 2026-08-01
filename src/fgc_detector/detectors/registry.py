"""The detector seam.

Detectors are stateless and pure: `observe()` classifies exactly one frame and
may not keep history, read a clock, or do I/O. All temporal reasoning lives in
the Confirmer. This is what makes adding a third game cheap — a new detector is
"read some pixels, report what you see" and inherits debounce, arming and
cooldown for free.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import EventType, Frame, Game, Observation, Screen
from .roi import Roi


class UnknownGameError(LookupError):
    pass


@runtime_checkable
class Detector(Protocol):
    game: Game
    canonical_size: tuple[int, int]

    def observe(self, frame: Frame) -> Observation:
        """Classify a single frame. Pure: same frame in, same observation out."""
        ...

    def rois(self) -> dict[str, Roi]:
        """The detector's named sampling rectangles, for the `roi` CLI preview."""
        ...

    def supported_events(self) -> frozenset[EventType]:
        """Which event types this detector can produce. Drives the config UI."""
        ...


_REGISTRY: dict[Game, Detector] = {}


def register(detector: Detector) -> None:
    if detector.game in _REGISTRY:
        raise ValueError(f"a detector for {detector.game.value} is already registered")
    _REGISTRY[detector.game] = detector


def get_detector(game: Game) -> Detector:
    try:
        return _REGISTRY[game]
    except KeyError as exc:
        raise UnknownGameError(f"no detector registered for {game.value}") from exc


def available_games() -> list[Game]:
    """Every game with a registered detector, in stable display order."""
    return sorted(_REGISTRY, key=lambda game: game.value)


class NullDetector:
    """Reports UNKNOWN for every frame. Used by tests and as a safe default."""

    canonical_size = (1920, 1080)

    def __init__(self, game: Game) -> None:
        self.game = game

    def observe(self, frame: Frame) -> Observation:
        return Observation(screen=Screen.UNKNOWN)

    def rois(self) -> dict[str, Roi]:
        return {}

    def supported_events(self) -> frozenset[EventType]:
        return frozenset({EventType.MATCH_END})
