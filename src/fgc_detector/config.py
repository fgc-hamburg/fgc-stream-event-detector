"""TOML configuration loading, with every failure reported as ConfigError."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import tomli_w

from .confirmer import ConfirmerConfig
from .events import SetCaptureCommand, SetConfirmerCommand
from .types import EventType, FILTERABLE_EVENTS, Game, RuntimeSettings


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ObsConfig:
    source_name: str
    host: str = "localhost"
    port: int = 4455
    password: str = ""
    poll_hz: float = 5.0

    def __post_init__(self) -> None:
        # Validated here rather than only at load time because the control
        # panel builds these at runtime: a bad value has to be refused before
        # it reaches the live frame source, not after.
        if not self.source_name:
            raise ValueError("obs.source_name must not be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError(f"obs.port must be in 1..65535, got {self.port}")
        if self.poll_hz <= 0:
            raise ValueError(f"obs.poll_hz must be > 0, got {self.poll_hz}")


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 6600
    ui_port: int = 6601


@dataclass(frozen=True)
class AppConfig:
    game: Game
    obs: ObsConfig
    server: ServerConfig
    confirmer: ConfirmerConfig
    runtime: RuntimeSettings

    def with_runtime(self, runtime: RuntimeSettings) -> "AppConfig":
        return replace(self, game=runtime.active_game, runtime=runtime)


def apply_capture(obs: ObsConfig, command: SetCaptureCommand) -> ObsConfig:
    """Fold a partial capture edit onto the current settings.

    A `None` field means the operator did not touch it. `password` is the one
    field where that matters for correctness: the page is never told the
    stored password, so "absent" must mean "keep it" and only an explicit
    empty string clears it.
    """
    return ObsConfig(
        source_name=(
            obs.source_name if command.source_name is None else command.source_name
        ),
        host=obs.host if command.host is None else command.host,
        port=obs.port if command.port is None else command.port,
        password=obs.password if command.password is None else command.password,
        poll_hz=obs.poll_hz if command.poll_hz is None else command.poll_hz,
    )


def apply_confirmer(
    confirmer: ConfirmerConfig, command: SetConfirmerCommand
) -> ConfirmerConfig:
    """Fold a partial confirmer edit onto the current thresholds."""
    return ConfirmerConfig(
        agreement_frames=(
            confirmer.agreement_frames
            if command.agreement_frames is None
            else command.agreement_frames
        ),
        cooldown_max_seconds=(
            confirmer.cooldown_max_seconds
            if command.cooldown_max_seconds is None
            else command.cooldown_max_seconds
        ),
        streak_staleness_seconds=(
            confirmer.streak_staleness_seconds
            if command.streak_staleness_seconds is None
            else command.streak_staleness_seconds
        ),
    )


def _require_table(raw: dict[str, Any], section: str) -> dict[str, Any]:
    """Return raw[section] as a table, or raise ConfigError naming the section."""
    value = raw.get(section, {})
    if not isinstance(value, dict):
        raise ConfigError(
            f"config section {section!r} must be a table, got {type(value).__name__}"
        )
    return value


def load_config(path: Path) -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    try:
        raw: dict[str, Any] = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config could not be parsed: {exc}") from exc

    if "game" not in raw:
        raise ConfigError(
            "game is required; expected one of: "
            + ", ".join(item.value for item in Game)
        )

    try:
        game = Game(raw.get("game"))
    except ValueError as exc:
        raise ConfigError(f"unknown game: {raw.get('game')!r}") from exc

    obs_section = _require_table(raw, "obs")
    source_name = obs_section.get("source_name")
    if not source_name:
        raise ConfigError("obs.source_name is required")

    server_section = _require_table(raw, "server")
    confirmer_section = _require_table(raw, "confirmer")

    try:
        obs = ObsConfig(
            source_name=source_name,
            host=obs_section.get("host", "localhost"),
            port=int(obs_section.get("port", 4455)),
            password=obs_section.get("password", ""),
            poll_hz=float(obs_section.get("poll_hz", 5.0)),
        )
        server = ServerConfig(
            host=server_section.get("host", "127.0.0.1"),
            port=int(server_section.get("port", 6600)),
            ui_port=int(server_section.get("ui_port", 6601)),
        )
        confirmer = ConfirmerConfig(
            agreement_frames=int(confirmer_section.get("agreement_frames", 3)),
            cooldown_max_seconds=float(
                confirmer_section.get("cooldown_max_seconds", 180.0)
            ),
            streak_staleness_seconds=float(
                confirmer_section.get("streak_staleness_seconds", 3.0)
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid config value: {exc}") from exc

    runtime_section = _require_table(raw, "runtime")
    try:
        enabled_games = (
            frozenset(Game(item) for item in runtime_section["enabled_games"])
            if "enabled_games" in runtime_section
            else frozenset(Game)
        )
        enabled_events = (
            frozenset(EventType(item) for item in runtime_section["enabled_events"])
            if "enabled_events" in runtime_section
            else frozenset(FILTERABLE_EVENTS)
        )
        runtime = RuntimeSettings(
            active_game=game,
            enabled_games=enabled_games,
            enabled_events=enabled_events,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid [runtime] section: {exc}") from exc

    return AppConfig(
        game=game, obs=obs, server=server, confirmer=confirmer, runtime=runtime
    )


def save_config(path: Path, config: AppConfig) -> None:
    """Write the whole config back to disk. The file stays hand-editable."""
    document = {
        "game": config.runtime.active_game.value,
        "obs": {
            "source_name": config.obs.source_name,
            "host": config.obs.host,
            "port": config.obs.port,
            "password": config.obs.password,
            "poll_hz": config.obs.poll_hz,
        },
        "server": {
            "host": config.server.host,
            "port": config.server.port,
            "ui_port": config.server.ui_port,
        },
        "confirmer": {
            "agreement_frames": config.confirmer.agreement_frames,
            "cooldown_max_seconds": config.confirmer.cooldown_max_seconds,
            "streak_staleness_seconds": config.confirmer.streak_staleness_seconds,
        },
        "runtime": {
            "enabled_games": sorted(item.value for item in config.runtime.enabled_games),
            "enabled_events": sorted(
                item.value for item in config.runtime.enabled_events
            ),
        },
    }
    Path(path).write_text(tomli_w.dumps(document))
