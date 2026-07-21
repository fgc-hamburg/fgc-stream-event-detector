import json
from datetime import datetime, timezone

import pytest

from fgc_detector.confirmer import Confirmer, ConfirmerConfig
from fgc_detector.detectors.registry import NullDetector, register
from fgc_detector.events import MatchEndEvent
from fgc_detector.server import EventServer
from fgc_detector.types import EventType, Game, RuntimeSettings, Side

TS = datetime(2026, 7, 21, 20, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _detectors():
    # tests/conftest.py's clean_registry autouse fixture clears and restores
    # the registry around every test; this just populates it for this module.
    register(NullDetector(Game.SF6))
    register(NullDetector(Game.TEKKEN8))


class FakeSocket:
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


def _settings(**overrides) -> RuntimeSettings:
    base = {
        "active_game": Game.SF6,
        "enabled_games": frozenset({Game.SF6, Game.TEKKEN8}),
        "enabled_events": frozenset({EventType.MATCH_END}),
    }
    return RuntimeSettings(**{**base, **overrides})


def _server(settings=None, saves=None):
    confirmer = Confirmer(Game.SF6, ConfirmerConfig())
    return EventServer(
        confirmer=confirmer,
        host="127.0.0.1",
        port=0,
        obs_connected_getter=lambda: True,
        settings=settings or _settings(),
        on_settings_changed=(saves.append if saves is not None else lambda _s: None),
    )


async def test_new_client_receives_config_after_status():
    server = _server()
    socket = FakeSocket()
    await server.handle_client(socket)
    kinds = [item["type"] for item in socket.payloads()]
    assert kinds[:2] == ["status", "config"]


async def test_config_event_lists_available_games_from_the_registry():
    server = _server()
    socket = FakeSocket(['{"cmd":"get_config"}'])
    await server.handle_client(socket)
    config = [item for item in socket.payloads() if item["type"] == "config"][-1]
    assert config["available_games"] == ["sf6", "tekken8"]
    assert config["supported_events"] == ["match_end"]


async def test_set_enabled_games_updates_and_persists():
    saves = []
    server = _server(saves=saves)
    socket = FakeSocket(['{"cmd":"set_enabled_games","games":["sf6"]}'])
    await server.handle_client(socket)
    assert server.settings.enabled_games == frozenset({Game.SF6})
    assert saves[-1].enabled_games == frozenset({Game.SF6})


async def test_disabling_the_active_game_is_rejected_not_applied():
    # Dropping the active game from the roster would leave the detector
    # sampling a game the operator says is not in use.
    saves = []
    server = _server(saves=saves)
    socket = FakeSocket(['{"cmd":"set_enabled_games","games":["tekken8"]}'])
    await server.handle_client(socket)
    assert any(item.get("error") for item in socket.payloads())
    assert server.settings.enabled_games == frozenset({Game.SF6, Game.TEKKEN8})
    assert saves == []


async def test_set_game_to_a_disabled_game_is_rejected():
    server = _server(settings=_settings(enabled_games=frozenset({Game.SF6})))
    socket = FakeSocket(['{"cmd":"set_game","game":"tekken8"}'])
    await server.handle_client(socket)
    assert any(item.get("error") for item in socket.payloads())
    assert server.confirmer.game is Game.SF6


async def test_set_game_to_an_enabled_game_updates_and_persists():
    saves = []
    server = _server(saves=saves)
    socket = FakeSocket(['{"cmd":"set_game","game":"tekken8"}'])
    await server.handle_client(socket)
    assert server.confirmer.game is Game.TEKKEN8
    assert saves[-1].active_game is Game.TEKKEN8


async def test_set_enabled_events_updates_and_persists():
    saves = []
    server = _server(saves=saves)
    socket = FakeSocket(['{"cmd":"set_enabled_events","events":[]}'])
    await server.handle_client(socket)
    assert server.settings.enabled_events == frozenset()
    assert saves[-1].enabled_events == frozenset()


async def test_disabled_event_type_is_not_broadcast():
    server = _server(settings=_settings(enabled_events=frozenset()))
    socket = FakeSocket()
    server._clients.add(socket)
    await server.broadcast(
        MatchEndEvent(game=Game.SF6, winner=Side.P1, confidence=0.9, ts=TS)
    )
    assert [item["type"] for item in socket.payloads()] == []


async def test_status_is_broadcast_even_with_all_events_disabled():
    server = _server(settings=_settings(enabled_events=frozenset()))
    socket = FakeSocket()
    server._clients.add(socket)
    await server.broadcast(server.status_event(TS))
    assert socket.payloads()[0]["type"] == "status"
