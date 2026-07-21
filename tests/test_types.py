import pytest

from fgc_detector.types import (
    Command,
    ConfirmerState,
    EventType,
    Game,
    Observation,
    Screen,
    Side,
)


def test_wire_enums_have_stable_string_values():
    assert Game.SF6.value == "sf6"
    assert Game.TEKKEN8.value == "tekken8"
    assert Side.P1.value == "p1"
    assert Side.P2.value == "p2"
    assert EventType.MATCH_END.value == "match_end"
    assert EventType.STATUS.value == "status"
    assert Command.ARM.value == "arm"
    assert Command.SET_GAME.value == "set_game"
    assert ConfirmerState.IDLE.value == "idle"


def test_wire_enums_parse_from_their_value():
    assert Game("sf6") is Game.SF6
    assert Side("p2") is Side.P2


def test_unknown_enum_value_raises():
    with pytest.raises(ValueError):
        Game("smash")


def test_screen_is_not_a_string_enum():
    # Screen never crosses the wire, so it must not be silently comparable to a string.
    assert Screen.IN_MATCH != "in_match"


def test_observation_defaults_are_empty_and_shared():
    a = Observation(screen=Screen.UNKNOWN)
    b = Observation(screen=Screen.UNKNOWN)
    assert a.details == {}
    assert a == b


def test_payload_ignores_confidence_and_debug():
    # Two observations that agree on the facts must compare equal for N-frame
    # agreement, even when their confidence scores differ frame to frame.
    a = Observation(Screen.MATCH_END, winner=Side.P1, confidence=0.91, debug={"x": 1.0})
    b = Observation(Screen.MATCH_END, winner=Side.P1, confidence=0.97, debug={"x": 2.0})
    assert a.payload == b.payload
    assert a != b


def test_payload_differs_on_winner():
    a = Observation(Screen.MATCH_END, winner=Side.P1)
    b = Observation(Screen.MATCH_END, winner=Side.P2)
    assert a.payload != b.payload
