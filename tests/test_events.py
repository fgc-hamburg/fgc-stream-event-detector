import json
from datetime import datetime, timezone

import pytest

from fgc_detector.events import (
    ArmCommand,
    CommandError,
    DisarmCommand,
    MatchEndEvent,
    SetGameCommand,
    StatusEvent,
    parse_command,
)
from fgc_detector.types import ConfirmerState, Game, Side

TS = datetime(2026, 7, 21, 10, 40, 0, tzinfo=timezone.utc)


def test_match_end_event_serializes_enum_values_not_repr():
    event = MatchEndEvent(game=Game.SF6, winner=Side.P1, confidence=0.9412, ts=TS)
    assert event.to_dict() == {
        "type": "match_end",
        "game": "sf6",
        "winner": "p1",
        "confidence": 0.9412,
        "ts": "2026-07-21T10:40:00Z",
    }


def test_match_end_event_json_round_trips_to_expected_keys():
    payload = json.loads(MatchEndEvent(Game.TEKKEN8, Side.P2, 0.8, TS).to_json())
    assert payload["type"] == "match_end"
    assert payload["game"] == "tekken8"
    assert payload["winner"] == "p2"


def test_status_event_serializes():
    event = StatusEvent(
        game=Game.SF6, armed=True, state=ConfirmerState.LIVE, obs_connected=True, ts=TS
    )
    assert event.to_dict() == {
        "type": "status",
        "game": "sf6",
        "armed": True,
        "state": "live",
        "obs_connected": True,
        "ts": "2026-07-21T10:40:00Z",
    }


def test_naive_timestamps_are_rejected():
    with pytest.raises(ValueError):
        MatchEndEvent(Game.SF6, Side.P1, 0.9, datetime(2026, 7, 21, 10, 40, 0)).to_dict()


def test_parse_arm_and_disarm():
    assert parse_command('{"cmd":"arm"}') == ArmCommand()
    assert parse_command('{"cmd":"disarm"}') == DisarmCommand()


def test_parse_set_game():
    assert parse_command('{"cmd":"set_game","game":"tekken8"}') == SetGameCommand(Game.TEKKEN8)


def test_unknown_command_is_rejected():
    with pytest.raises(CommandError, match="unknown command"):
        parse_command('{"cmd":"self_destruct"}')


def test_unknown_game_is_rejected():
    with pytest.raises(CommandError, match="unknown game"):
        parse_command('{"cmd":"set_game","game":"smash"}')


def test_set_game_without_game_is_rejected():
    with pytest.raises(CommandError, match="requires a 'game'"):
        parse_command('{"cmd":"set_game"}')


def test_malformed_json_is_rejected():
    with pytest.raises(CommandError, match="not valid JSON"):
        parse_command("{not json")


def test_non_object_json_is_rejected():
    with pytest.raises(CommandError, match="JSON object"):
        parse_command('["arm"]')
