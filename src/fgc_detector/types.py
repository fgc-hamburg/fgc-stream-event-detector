"""Core value types.

Everything in a closed set is an enum. Values that cross the WebSocket are
StrEnum so serialization is a `.value` lookup and deserialization is a
constructor call that raises on anything unrecognized. Screen is deliberately
NOT a StrEnum: it is internal to detection and must never be confused with a
wire value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, StrEnum, auto
from types import MappingProxyType
from typing import Mapping

import numpy as np


class Game(StrEnum):
    SF6 = "sf6"
    TEKKEN8 = "tekken8"


class Side(StrEnum):
    P1 = "p1"
    P2 = "p2"


class EventType(StrEnum):
    MATCH_END = "match_end"
    STATUS = "status"


class Command(StrEnum):
    ARM = "arm"
    DISARM = "disarm"
    SET_GAME = "set_game"


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
