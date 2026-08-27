"""WebSocket server: the detector's only contact with the outside world.

Emits events, accepts arm/disarm/set_game commands, and pushes a status event on
connect and on every state change so a freshly-connected dashboard never has to
poll. The server knows nothing about brackets, sets, or scores.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable

import websockets

from .config import AppConfig, apply_capture, apply_confirmer
from .confirmation import ConfirmerLike
from .detectors.registry import available_games, get_detector
from .events import (
    ArmCommand,
    CommandError,
    ConfigEvent,
    DisarmCommand,
    Event,
    GetConfigCommand,
    SetCaptureCommand,
    SetConfirmerCommand,
    SetEnabledEventsCommand,
    SetEnabledGamesCommand,
    SetGameCommand,
    StatusEvent,
    parse_command,
)
from .types import RuntimeSettings

log = logging.getLogger(__name__)


class EventServer:
    def __init__(
        self,
        confirmer: ConfirmerLike,
        host: str,
        port: int,
        obs_connected_getter: Callable[[], bool],
        config: AppConfig,
        on_config_changed: Callable[[AppConfig], None],
    ) -> None:
        self.confirmer = confirmer
        self._host = host
        self._port = port
        self._obs_connected = obs_connected_getter
        self.config = config
        self._on_config_changed = on_config_changed
        self._clients: set[Any] = set()

    @property
    def settings(self) -> RuntimeSettings:
        """The operator's selections. Read-only: go through `_apply` to change
        them, so the confirmer and the config file cannot drift from them."""
        return self.config.runtime

    def status_event(self, now: datetime) -> StatusEvent:
        return StatusEvent(
            game=self.confirmer.game,
            armed=self.confirmer.armed,
            state=self.confirmer.state,
            obs_connected=self._obs_connected(),
            ts=now,
        )

    def config_event(self, now: datetime) -> ConfigEvent:
        games = available_games()
        supported_events = (
            frozenset().union(*(get_detector(game).supported_events() for game in games))
            if games
            else frozenset()
        )
        return ConfigEvent(
            settings=self.settings,
            available_games=games,
            supported_events=supported_events,
            obs=self.config.obs,
            confirmer=self.config.confirmer,
            ts=now,
        )

    def _apply_runtime(self, settings: RuntimeSettings) -> None:
        """Adopt new operator selections and sync the confirmer.

        `set_game` only mutates the existing confirmer's `game` attribute; it
        does not rebuild it via `make_confirmer`. If `active_game` switches to
        a game whose confirmation strategy differs from the one already
        constructed (e.g. SF6's counter-based confirmer vs. a future
        marker-based game), the confirmer silently keeps running the wrong
        strategy until restart. Not yet handled -- see docs/TODO.md.
        """
        self._apply(self.config.with_runtime(settings))
        if self.confirmer.game is not settings.active_game:
            self.confirmer.set_game(settings.active_game)

    def _apply(self, config: AppConfig) -> None:
        """Adopt a new whole-config and hand it to the owner to act on.

        The single write path: everything that changes settings builds the
        new `AppConfig` first, so the callback (which persists it to disk and
        retunes the live frame source) always sees the complete picture
        rather than one section at a time. A rejected edit raises before
        reaching here, so nothing is adopted or persisted.
        """
        self.config = config
        self._on_config_changed(config)

    async def broadcast(self, event: Event) -> None:
        if not self.settings.allows(event.TYPE):
            log.debug("suppressing %s: disabled by runtime settings", event.TYPE.value)
            return
        message = event.to_json()
        for client in list(self._clients):
            try:
                await client.send(message)
            except Exception as exc:
                log.info("dropping client after send failure: %s", exc)
                self._clients.discard(client)

    async def _send_status(self, socket: Any) -> None:
        await socket.send(self.status_event(datetime.now(timezone.utc)).to_json())

    async def handle_client(self, socket: Any) -> None:
        self._clients.add(socket)
        try:
            await self._send_status(socket)
            await socket.send(self.config_event(datetime.now(timezone.utc)).to_json())
            async for raw in socket:
                await self._handle_message(socket, raw)
        finally:
            self._clients.discard(socket)

    async def _handle_message(self, socket: Any, raw: str) -> None:
        try:
            command = parse_command(raw)
        except CommandError as exc:
            log.warning("rejected inbound message: %s", exc)
            await socket.send(json.dumps({"error": str(exc)}))
            return

        now = datetime.now(timezone.utc)
        try:
            match command:
                case ArmCommand():
                    self.confirmer.arm()
                case DisarmCommand():
                    self.confirmer.disarm()
                case SetGameCommand(game=game):
                    self._apply_runtime(replace(self.settings, active_game=game))
                case GetConfigCommand():
                    pass
                case SetEnabledGamesCommand(games=games):
                    self._apply_runtime(replace(self.settings, enabled_games=games))
                case SetEnabledEventsCommand(events=events):
                    self._apply_runtime(replace(self.settings, enabled_events=events))
                case SetCaptureCommand():
                    self._apply(
                        replace(self.config, obs=apply_capture(self.config.obs, command))
                    )
                case SetConfirmerCommand():
                    # Validation happens inside apply_confirmer, before either
                    # the live confirmer or the stored config is touched.
                    confirmer_config = apply_confirmer(self.config.confirmer, command)
                    self.confirmer.configure(confirmer_config)
                    self._apply(replace(self.config, confirmer=confirmer_config))
        except ValueError as exc:
            await socket.send(json.dumps({"error": str(exc)}))
            return

        await self._send_status(socket)
        await socket.send(self.config_event(now).to_json())

    async def serve(self) -> None:
        log.info("event server listening on ws://%s:%s", self._host, self._port)
        async with websockets.serve(self.handle_client, self._host, self._port):
            await asyncio.Future()  # run forever
