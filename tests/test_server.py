import json
from datetime import datetime, timezone

import pytest

from fgc_detector.confirmer import Confirmer, ConfirmerConfig
from fgc_detector.server import EventServer
from fgc_detector.types import ConfirmerState, EventType, Game, RuntimeSettings, Side

TS = datetime(2026, 7, 21, 20, 0, 0, tzinfo=timezone.utc)


class FakeSocket:
    """Minimal stand-in for a websockets server connection."""

    def __init__(self, inbound=()):
        self.sent: list[str] = []
        self._inbound = list(inbound)

    async def send(self, message: str) -> None:
        self.sent.append(message)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._inbound:
            raise StopAsyncIteration
        return self._inbound.pop(0)

    def payloads(self) -> list[dict]:
        return [json.loads(message) for message in self.sent]


@pytest.fixture
def server():
    confirmer = Confirmer(Game.SF6, ConfirmerConfig())
    settings = RuntimeSettings(
        active_game=Game.SF6,
        enabled_games=frozenset({Game.SF6, Game.TEKKEN8}),
        enabled_events=frozenset({EventType.MATCH_END}),
    )
    return EventServer(
        confirmer=confirmer,
        host="127.0.0.1",
        port=0,
        obs_connected_getter=lambda: True,
        settings=settings,
        on_settings_changed=lambda _s: None,
    )


@pytest.mark.asyncio
async def test_new_client_immediately_receives_status(server):
    socket = FakeSocket()
    await server.handle_client(socket)
    first = socket.payloads()[0]
    assert first["type"] == "status"
    assert first["game"] == "sf6"
    assert first["armed"] is False


@pytest.mark.asyncio
async def test_arm_command_arms_the_confirmer_and_echoes_status(server):
    socket = FakeSocket(['{"cmd":"arm"}'])
    await server.handle_client(socket)
    assert server.confirmer.armed is True
    status = [item for item in socket.payloads() if item["type"] == "status"][-1]
    assert status["armed"] is True


@pytest.mark.asyncio
async def test_disarm_command_disarms(server):
    server.confirmer.arm()
    socket = FakeSocket(['{"cmd":"disarm"}'])
    await server.handle_client(socket)
    assert server.confirmer.armed is False


@pytest.mark.asyncio
async def test_set_game_command_switches_game(server):
    socket = FakeSocket(['{"cmd":"set_game","game":"tekken8"}'])
    await server.handle_client(socket)
    assert server.confirmer.game is Game.TEKKEN8
    status = [item for item in socket.payloads() if item["type"] == "status"][-1]
    assert status["game"] == "tekken8"


@pytest.mark.asyncio
async def test_bad_command_returns_an_error_and_keeps_the_connection(server):
    socket = FakeSocket(['{"cmd":"nonsense"}', '{"cmd":"arm"}'])
    await server.handle_client(socket)
    payloads = socket.payloads()
    assert any(item.get("error") for item in payloads)
    assert server.confirmer.armed is True, "connection must survive a bad command"


@pytest.mark.asyncio
async def test_broadcast_reaches_every_connected_client(server):
    first, second = FakeSocket(), FakeSocket()
    server._clients.update({first, second})
    await server.broadcast(MatchEndEventFactory())
    assert json.loads(first.sent[-1])["type"] == "match_end"
    assert json.loads(second.sent[-1])["type"] == "match_end"


@pytest.mark.asyncio
async def test_broadcast_drops_a_client_that_errors(server):
    class Broken(FakeSocket):
        async def send(self, message):
            raise ConnectionResetError

    broken, healthy = Broken(), FakeSocket()
    server._clients.update({broken, healthy})
    await server.broadcast(MatchEndEventFactory())
    assert broken not in server._clients
    assert healthy in server._clients


def MatchEndEventFactory():
    from fgc_detector.events import MatchEndEvent

    return MatchEndEvent(game=Game.SF6, winner=Side.P1, confidence=0.9, ts=TS)


def test_status_event_reflects_confirmer_state(server):
    server.confirmer.arm()
    status = server.status_event(TS)
    assert status.armed is True
    assert status.state is ConfirmerState.IDLE
    assert status.obs_connected is True
