"""TOML configuration loading, with every failure reported as ConfigError."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .confirmer import ConfirmerConfig
from .types import Game


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ObsConfig:
    source_name: str
    host: str = "localhost"
    port: int = 4455
    password: str = ""
    poll_hz: float = 5.0


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 6600


@dataclass(frozen=True)
class AppConfig:
    game: Game
    obs: ObsConfig
    server: ServerConfig
    confirmer: ConfirmerConfig


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

    return AppConfig(game=game, obs=obs, server=server, confirmer=confirmer)
