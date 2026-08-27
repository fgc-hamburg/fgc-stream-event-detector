"""The JSON boundary.

This is the only module allowed to turn an enum into a string or a string into
an enum. Everything inward of here is typed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, ClassVar

from .types import Command, ConfirmerState, EventType, Game, RuntimeSettings, Side

if TYPE_CHECKING:  # imported for typing only: both modules import this one.
    from .config import ObsConfig
    from .confirmer import ConfirmerConfig


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
    obs: "ObsConfig"
    confirmer: "ConfirmerConfig"
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
            "obs": {
                "source_name": self.obs.source_name,
                "host": self.obs.host,
                "port": self.obs.port,
                "poll_hz": self.obs.poll_hz,
                # The password itself is never published: this event goes to
                # every connected client, and the page only needs to know
                # whether one is stored.
                "password_set": bool(self.obs.password),
            },
            "confirmer": {
                "agreement_frames": self.confirmer.agreement_frames,
                "cooldown_max_seconds": self.confirmer.cooldown_max_seconds,
                "streak_staleness_seconds": self.confirmer.streak_staleness_seconds,
            },
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


@dataclass(frozen=True)
class SetCaptureCommand:
    """A partial edit of the OBS capture settings.

    Every field is optional and `None` means "leave it as it is", so the page
    can send only what the operator changed. `password` therefore has three
    states: `None` (absent -- keep the stored one), `""` (clear it), or a new
    value. That is what lets the password stay write-only: the page is never
    told the current password, so it could not echo one back.
    """

    source_name: str | None = None
    host: str | None = None
    port: int | None = None
    poll_hz: float | None = None
    password: str | None = None


@dataclass(frozen=True)
class SetConfirmerCommand:
    """A partial edit of the confirmation thresholds. `None` means unchanged."""

    agreement_frames: int | None = None
    cooldown_max_seconds: float | None = None
    streak_staleness_seconds: float | None = None


ParsedCommand = (
    ArmCommand
    | DisarmCommand
    | SetGameCommand
    | GetConfigCommand
    | SetEnabledGamesCommand
    | SetEnabledEventsCommand
    | SetCaptureCommand
    | SetConfirmerCommand
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


def _optional_str(payload: dict, key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CommandError(f"'{key}' must be a string, got {type(value).__name__}")
    return value


def _optional_number(payload: dict, key: str, kind: type) -> int | float | None:
    """Read an optional numeric field, rejecting anything not of `kind`.

    A float field accepts a JSON integer (`"poll_hz": 2` is a perfectly good
    2.0), but an integer field rejects a float: silently truncating
    `agreement_frames: 2.5` would change the meaning of the setting. `bool` is
    excluded explicitly because it subclasses `int`: without that check
    `{"port": true}` would quietly become port 1.
    """
    value = payload.get(key)
    if value is None:
        return None
    accepted: tuple[type, ...] = (int, float) if kind is float else (kind,)
    if isinstance(value, bool) or not isinstance(value, accepted):
        raise CommandError(
            f"'{key}' must be {'an integer' if kind is int else 'a number'}, "
            f"got {type(value).__name__}"
        )
    return kind(value)


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
        case Command.SET_CAPTURE:
            return SetCaptureCommand(
                source_name=_optional_str(payload, "source_name"),
                host=_optional_str(payload, "host"),
                port=_optional_number(payload, "port", int),
                poll_hz=_optional_number(payload, "poll_hz", float),
                password=_optional_str(payload, "password"),
            )
        case Command.SET_CONFIRMER:
            return SetConfirmerCommand(
                agreement_frames=_optional_number(payload, "agreement_frames", int),
                cooldown_max_seconds=_optional_number(
                    payload, "cooldown_max_seconds", float
                ),
                streak_staleness_seconds=_optional_number(
                    payload, "streak_staleness_seconds", float
                ),
            )
        case _:
            raise AssertionError(f"unhandled command {command}")
