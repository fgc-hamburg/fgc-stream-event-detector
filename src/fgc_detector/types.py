"""Core value types.

Everything in a closed set is an enum. Values that cross the WebSocket are
StrEnum so serialization is a `.value` lookup and deserialization is a
constructor call that raises on anything unrecognized. Screen is deliberately
NOT a StrEnum: it is internal to detection and must never be confused with a
wire value.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, StrEnum, auto
from types import MappingProxyType
from typing import Mapping

import numpy as np


class Game(StrEnum):
    SF6 = "sf6"
    TEKKEN8 = "tekken8"
    AVATAR = "avatar"
    TOKON = "tokon"


class Side(StrEnum):
    P1 = "p1"
    P2 = "p2"


class EventType(StrEnum):
    MATCH_END = "match_end"
    STATUS = "status"
    CONFIG = "config"


class Command(StrEnum):
    ARM = "arm"
    DISARM = "disarm"
    SET_GAME = "set_game"
    GET_CONFIG = "get_config"
    SET_ENABLED_GAMES = "set_enabled_games"
    SET_ENABLED_EVENTS = "set_enabled_events"
    SET_CAPTURE = "set_capture"
    SET_CONFIRMER = "set_confirmer"


#: Event types that carry detection results and may therefore be filtered.
#: STATUS and CONFIG are transport bookkeeping and are always delivered.
FILTERABLE_EVENTS: frozenset[EventType] = frozenset({EventType.MATCH_END})


class ConfirmerState(StrEnum):
    IDLE = "idle"
    LIVE = "live"
    COOLDOWN = "cooldown"


class Screen(Enum):
    """What the detector believes is on screen right now."""

    UNKNOWN = auto()
    CHAR_SELECT = auto()
    IN_MATCH = auto()
    MATCH_END = auto()


_EMPTY_STR_MAP: Mapping[str, str] = MappingProxyType({})
_EMPTY_NUM_MAP: Mapping[str, float] = MappingProxyType({})

#: Keys under which detectors publish per-side round-win counts in Observation.details.
DETAIL_P1_ROUNDS = "p1_rounds"
DETAIL_P2_ROUNDS = "p2_rounds"

#: Keys under which a counter detector publishes per-side games-won-in-set counts.
DETAIL_P1_GAMES = "p1_games"
DETAIL_P2_GAMES = "p2_games"


@dataclass(frozen=True)
class Frame:
    """A single captured image, already normalized to a detector's canonical size."""

    image: np.ndarray  # BGR, uint8
    captured_at: datetime


@dataclass(frozen=True)
class Observation:
    """A detector's read of exactly one frame. Carries no history."""

    screen: Screen
    winner: Side | None = None
    details: Mapping[str, str] = _EMPTY_STR_MAP
    confidence: float = 0.0
    debug: Mapping[str, float] = _EMPTY_NUM_MAP

    @property
    def payload(self) -> tuple[Screen, Side | None, tuple[tuple[str, str], ...]]:
        """The facts that N-frame agreement compares.

        Deliberately excludes confidence and debug, which jitter frame to frame
        and would otherwise prevent any two observations from ever agreeing.
        Includes `details` so future event types (character lock) inherit the
        agreement rule with no change to the Confirmer.
        """
        return (self.screen, self.winner, tuple(sorted(self.details.items())))


@dataclass(frozen=True)
class RuntimeSettings:
    """What the operator has selected. Validated on construction.

    Only one game is on screen at a time, so `enabled_games` is the roster the
    operator picks from, not a set of concurrently running detectors.
    """

    active_game: Game
    enabled_games: frozenset[Game]
    enabled_events: frozenset[EventType]

    def __post_init__(self) -> None:
        if not self.enabled_games:
            raise ValueError("enabled_games must contain at least one game")
        if self.active_game not in self.enabled_games:
            raise ValueError(
                f"active game {self.active_game.value} is not in enabled_games"
            )
        unfilterable = self.enabled_events - FILTERABLE_EVENTS
        if unfilterable:
            names = ", ".join(sorted(item.value for item in unfilterable))
            raise ValueError(f"these event types cannot be filtered: {names}")

    def allows(self, event_type: EventType) -> bool:
        """Whether an event of this type should be delivered."""
        if event_type not in FILTERABLE_EVENTS:
            return True
        return event_type in self.enabled_events
