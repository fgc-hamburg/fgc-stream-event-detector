"""The JSON boundary.

This is the only module allowed to turn an enum into a string or a string into
an enum. Everything inward of here is typed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar

from .types import Command, ConfirmerState, EventType, Game, RuntimeSettings, Side


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


@dataclass(frozen=True)
class ConfigEvent:
    """Current selections plus what is available to select. Drives the UI."""

    settings: RuntimeSettings
    available_games: list[Game]
    supported_events: frozenset[EventType]
    ts: datetime

    TYPE: ClassVar[EventType] = EventType.CONFIG

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.TYPE.value,
            "active_game": self.settings.active_game.value,
            "enabled_games": sorted(item.value for item in self.settings.enabled_games),
            "enabled_events": sorted(
                item.value for item in self.settings.enabled_events
            ),
            "available_games": [item.value for item in self.available_games],
            "supported_events": sorted(item.value for item in self.supported_events),
            "ts": _iso(self.ts),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


Event = MatchEndEvent | StatusEvent | ConfigEvent


@dataclass(frozen=True)
class ArmCommand:
    pass


@dataclass(frozen=True)
class DisarmCommand:
    pass


@dataclass(frozen=True)
class SetGameCommand:
    game: Game


@dataclass(frozen=True)
class GetConfigCommand:
    pass


@dataclass(frozen=True)
class SetEnabledGamesCommand:
    games: frozenset[Game]


@dataclass(frozen=True)
class SetEnabledEventsCommand:
    events: frozenset[EventType]


ParsedCommand = (
    ArmCommand
    | DisarmCommand
    | SetGameCommand
    | GetConfigCommand
    | SetEnabledGamesCommand
    | SetEnabledEventsCommand
)


def _parse_enum_list(payload: dict, key: str, enum_type, label: str) -> frozenset:
    raw = payload.get(key)
    if not isinstance(raw, list):
        raise CommandError(f"'{key}' must be a list, got {type(raw).__name__}")
    parsed = set()
    for item in raw:
        try:
            parsed.add(enum_type(item))
        except ValueError as exc:
            raise CommandError(f"unknown {label}: {item!r}") from exc
    return frozenset(parsed)


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
        case Command.GET_CONFIG:
            return GetConfigCommand()
        case Command.SET_ENABLED_GAMES:
            return SetEnabledGamesCommand(
                _parse_enum_list(payload, "games", Game, "game")
            )
        case Command.SET_ENABLED_EVENTS:
            return SetEnabledEventsCommand(
                _parse_enum_list(payload, "events", EventType, "event")
            )
        case _:
            raise AssertionError(f"unhandled command {command}")
