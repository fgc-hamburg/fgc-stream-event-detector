"""The JSON boundary.

This is the only module allowed to turn an enum into a string or a string into
an enum. Everything inward of here is typed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar

from .types import Command, ConfirmerState, EventType, Game, Side


class CommandError(ValueError):
    """An inbound message could not be understood. Never fatal — reply and continue."""


def _iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        raise ValueError(f"timestamp must be timezone-aware, got naive {ts!r}")
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class MatchEndEvent:
    game: Game
    winner: Side
    confidence: float
    ts: datetime

    TYPE: ClassVar[EventType] = EventType.MATCH_END

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.TYPE.value,
            "game": self.game.value,
            "winner": self.winner.value,
            "confidence": round(self.confidence, 4),
            "ts": _iso(self.ts),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass(frozen=True)
class StatusEvent:
    game: Game
    armed: bool
    state: ConfirmerState
    obs_connected: bool
    ts: datetime

    TYPE: ClassVar[EventType] = EventType.STATUS

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.TYPE.value,
            "game": self.game.value,
            "armed": self.armed,
            "state": self.state.value,
            "obs_connected": self.obs_connected,
            "ts": _iso(self.ts),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


Event = MatchEndEvent | StatusEvent


@dataclass(frozen=True)
class ArmCommand:
    pass


@dataclass(frozen=True)
class DisarmCommand:
    pass


@dataclass(frozen=True)
class SetGameCommand:
    game: Game


ParsedCommand = ArmCommand | DisarmCommand | SetGameCommand


def parse_command(raw: str) -> ParsedCommand:
    """Parse an inbound dashboard message, rejecting anything unrecognized.

    Raises CommandError rather than returning a sentinel: an unknown command is
    a bug in the caller, and swallowing it silently is exactly the failure mode
    enums exist to prevent.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CommandError(f"not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise CommandError(f"expected a JSON object, got {type(payload).__name__}")

    raw_cmd = payload.get("cmd")
    try:
        command = Command(raw_cmd)
    except ValueError as exc:
        raise CommandError(f"unknown command: {raw_cmd!r}") from exc

    match command:
        case Command.ARM:
            return ArmCommand()
        case Command.DISARM:
            return DisarmCommand()
        case Command.SET_GAME:
            raw_game = payload.get("game")
            if raw_game is None:
                raise CommandError("set_game requires a 'game' field")
            try:
                return SetGameCommand(Game(raw_game))
            except ValueError as exc:
                raise CommandError(f"unknown game: {raw_game!r}") from exc
