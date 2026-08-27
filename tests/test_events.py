import json
from datetime import datetime, timezone

import pytest

from fgc_detector.events import (
    ArmCommand,
    CommandError,
    DisarmCommand,
    MatchEndEvent,
    SetCaptureCommand,
    SetConfirmerCommand,
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


# --- set_capture / set_confirmer -------------------------------------------


def test_parse_set_capture_with_every_field():
    command = parse_command(
        '{"cmd":"set_capture","source_name":"Cap","host":"10.0.0.2",'
        '"port":4456,"poll_hz":8.5,"password":"hunter2"}'
    )
    assert command == SetCaptureCommand(
        source_name="Cap", host="10.0.0.2", port=4456, poll_hz=8.5, password="hunter2"
    )


def test_omitted_capture_fields_parse_as_none_meaning_unchanged():
    command = parse_command('{"cmd":"set_capture","poll_hz":2}')
    assert command == SetCaptureCommand(poll_hz=2.0)


def test_an_omitted_password_is_distinct_from_an_empty_one():
    """Absent means 'keep the stored password'; empty means 'clear it'. The
    UI never receives the current password, so it cannot echo one back, and
    conflating these would wipe the password on every unrelated edit."""
    assert parse_command('{"cmd":"set_capture","host":"a"}').password is None
    assert parse_command('{"cmd":"set_capture","password":""}').password == ""


def test_non_numeric_capture_values_are_rejected():
    with pytest.raises(CommandError):
        parse_command('{"cmd":"set_capture","port":"4455"}')
    with pytest.raises(CommandError):
        parse_command('{"cmd":"set_capture","poll_hz":"fast"}')


def test_non_string_capture_text_is_rejected():
    with pytest.raises(CommandError):
        parse_command('{"cmd":"set_capture","host":5}')


def test_booleans_are_not_accepted_as_numbers():
    # bool is an int subclass in Python; a naive isinstance check would let
    # {"port": true} through as port 1.
    with pytest.raises(CommandError):
        parse_command('{"cmd":"set_capture","port":true}')


def test_parse_set_confirmer():
    command = parse_command(
        '{"cmd":"set_confirmer","agreement_frames":5,'
        '"cooldown_max_seconds":90,"streak_staleness_seconds":2.5}'
    )
    assert command == SetConfirmerCommand(
        agreement_frames=5, cooldown_max_seconds=90.0, streak_staleness_seconds=2.5
    )


def test_omitted_confirmer_fields_parse_as_none_meaning_unchanged():
    command = parse_command('{"cmd":"set_confirmer","agreement_frames":4}')
    assert command == SetConfirmerCommand(agreement_frames=4)


def test_non_integer_agreement_frames_is_rejected():
    with pytest.raises(CommandError):
        parse_command('{"cmd":"set_confirmer","agreement_frames":2.5}')
