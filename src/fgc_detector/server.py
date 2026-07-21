"""WebSocket server: the detector's only contact with the outside world.

Emits events, accepts arm/disarm/set_game commands, and pushes a status event on
connect and on every state change so a freshly-connected dashboard never has to
poll. The server knows nothing about brackets, sets, or scores.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable

import websockets

from .confirmer import Confirmer
from .events import (
    ArmCommand,
    CommandError,
    DisarmCommand,
    Event,
    SetGameCommand,
    StatusEvent,
    parse_command,
)

log = logging.getLogger(__name__)


class EventServer:
    def __init__(
        self,
        confirmer: Confirmer,
        host: str,
        port: int,
        obs_connected_getter: Callable[[], bool],
    ) -> None:
        self.confirmer = confirmer
        self._host = host
        self._port = port
        self._obs_connected = obs_connected_getter
        self._clients: set[Any] = set()

    def status_event(self, now: datetime) -> StatusEvent:
        return StatusEvent(
            game=self.confirmer.game,
            armed=self.confirmer.armed,
            state=self.confirmer.state,
            obs_connected=self._obs_connected(),
            ts=now,
        )

    async def broadcast(self, event: Event) -> None:
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

        match command:
            case ArmCommand():
                self.confirmer.arm()
            case DisarmCommand():
                self.confirmer.disarm()
            case SetGameCommand(game=game):
                self.confirmer.set_game(game)

        await self._send_status(socket)

    async def serve(self) -> None:
        log.info("event server listening on ws://%s:%s", self._host, self._port)
        async with websockets.serve(self.handle_client, self._host, self._port):
            await asyncio.Future()  # run forever
