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

from .confirmer import Confirmer
from .detectors.registry import available_games, get_detector
from .events import (
    ArmCommand,
    CommandError,
    ConfigEvent,
    DisarmCommand,
    Event,
    GetConfigCommand,
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
        confirmer: Confirmer,
        host: str,
        port: int,
        obs_connected_getter: Callable[[], bool],
        settings: RuntimeSettings,
        on_settings_changed: Callable[[RuntimeSettings], None],
    ) -> None:
        self.confirmer = confirmer
        self._host = host
        self._port = port
        self._obs_connected = obs_connected_getter
        self.settings = settings
        self._on_settings_changed = on_settings_changed
        self._clients: set[Any] = set()

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
            ts=now,
        )

    def _apply(self, settings: RuntimeSettings) -> None:
        """Adopt new settings, sync the confirmer, and persist."""
        self.settings = settings
        if self.confirmer.game is not settings.active_game:
            self.confirmer.set_game(settings.active_game)
        self._on_settings_changed(settings)

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
                    self._apply(replace(self.settings, active_game=game))
                case GetConfigCommand():
                    pass
                case SetEnabledGamesCommand(games=games):
                    self._apply(replace(self.settings, enabled_games=games))
                case SetEnabledEventsCommand(events=events):
                    self._apply(replace(self.settings, enabled_events=events))
        except ValueError as exc:
            await socket.send(json.dumps({"error": str(exc)}))
            return

        await self._send_status(socket)
        await socket.send(self.config_event(now).to_json())

    async def serve(self) -> None:
        log.info("event server listening on ws://%s:%s", self._host, self._port)
        async with websockets.serve(self.handle_client, self._host, self._port):
            await asyncio.Future()  # run forever
