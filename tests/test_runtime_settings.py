import pytest

from fgc_detector.config import load_config, save_config
from fgc_detector.detectors.registry import NullDetector, available_games, register
from fgc_detector.events import (
    ConfigEvent,
    CommandError,
    GetConfigCommand,
    SetEnabledEventsCommand,
    SetEnabledGamesCommand,
    parse_command,
)
from fgc_detector.types import EventType, Game, RuntimeSettings

from datetime import datetime, timezone

TS = datetime(2026, 7, 21, 20, 0, 0, tzinfo=timezone.utc)

VALID = """
game = "sf6"

[obs]
source_name = "Game Capture"

[runtime]
enabled_games = ["sf6", "tekken8"]
enabled_events = ["match_end"]
"""

# Registry cleanup is provided by the autouse `clean_registry` fixture in
# tests/conftest.py.


def _write(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_new_enum_members_have_stable_values():
    assert EventType.CONFIG.value == "config"


def test_available_games_lists_registered_detectors_only():
    assert available_games() == []
    register(NullDetector(Game.TEKKEN8))
    register(NullDetector(Game.SF6))
    # Sorted for a stable UI ordering, not registration order.
    assert available_games() == [Game.SF6, Game.TEKKEN8]


def test_null_detector_declares_the_events_it_can_produce():
    assert NullDetector(Game.SF6).supported_events() == frozenset({EventType.MATCH_END})


def test_runtime_settings_rejects_an_active_game_not_in_the_roster():
    with pytest.raises(ValueError, match="not in enabled_games"):
        RuntimeSettings(
            active_game=Game.SF6,
            enabled_games=frozenset({Game.TEKKEN8}),
            enabled_events=frozenset({EventType.MATCH_END}),
        )


def test_runtime_settings_rejects_an_empty_roster():
    with pytest.raises(ValueError, match="at least one"):
        RuntimeSettings(
            active_game=Game.SF6,
            enabled_games=frozenset(),
            enabled_events=frozenset({EventType.MATCH_END}),
        )


def test_status_is_not_a_filterable_event():
    # Status is transport bookkeeping; letting an operator disable it would
    # leave the dashboard blind with no way to recover.
    with pytest.raises(ValueError, match="cannot be filtered"):
        RuntimeSettings(
            active_game=Game.SF6,
            enabled_games=frozenset({Game.SF6}),
            enabled_events=frozenset({EventType.STATUS}),
        )


def test_config_event_serializes_capabilities_and_selections():
    register(NullDetector(Game.SF6))
    register(NullDetector(Game.TEKKEN8))
    event = ConfigEvent(
        settings=RuntimeSettings(
            active_game=Game.SF6,
            enabled_games=frozenset({Game.SF6, Game.TEKKEN8}),
            enabled_events=frozenset({EventType.MATCH_END}),
        ),
        available_games=[Game.SF6, Game.TEKKEN8],
        supported_events=frozenset({EventType.MATCH_END}),
        ts=TS,
    )
    assert event.to_dict() == {
        "type": "config",
        "active_game": "sf6",
        "enabled_games": ["sf6", "tekken8"],
        "enabled_events": ["match_end"],
        "available_games": ["sf6", "tekken8"],
        "supported_events": ["match_end"],
        "ts": "2026-07-21T20:00:00Z",
    }


def test_parse_get_config():
    assert parse_command('{"cmd":"get_config"}') == GetConfigCommand()


def test_parse_set_enabled_games():
    command = parse_command('{"cmd":"set_enabled_games","games":["sf6"]}')
    assert command == SetEnabledGamesCommand(frozenset({Game.SF6}))


def test_parse_set_enabled_events():
    command = parse_command('{"cmd":"set_enabled_events","events":["match_end"]}')
    assert command == SetEnabledEventsCommand(frozenset({EventType.MATCH_END}))


def test_set_enabled_games_rejects_an_unknown_game():
    with pytest.raises(CommandError, match="unknown game"):
        parse_command('{"cmd":"set_enabled_games","games":["smash"]}')


def test_set_enabled_games_rejects_a_non_list():
    with pytest.raises(CommandError, match="list"):
        parse_command('{"cmd":"set_enabled_games","games":"sf6"}')


def test_set_enabled_events_rejects_an_unknown_event():
    with pytest.raises(CommandError, match="unknown event"):
        parse_command('{"cmd":"set_enabled_events","events":["explode"]}')


def test_config_loads_the_runtime_section(tmp_path):
    config = load_config(_write(tmp_path, VALID))
    assert config.runtime.active_game is Game.SF6
    assert config.runtime.enabled_games == frozenset({Game.SF6, Game.TEKKEN8})
    assert config.runtime.enabled_events == frozenset({EventType.MATCH_END})


def test_runtime_section_defaults_to_every_game_and_event(tmp_path):
    minimal = 'game = "sf6"\n\n[obs]\nsource_name = "Capture"\n'
    config = load_config(_write(tmp_path, minimal))
    assert config.runtime.enabled_games == frozenset(Game)
    assert config.runtime.enabled_events == frozenset({EventType.MATCH_END})


def test_save_config_round_trips(tmp_path):
    path = _write(tmp_path, VALID)
    config = load_config(path)
    updated = config.with_runtime(
        RuntimeSettings(
            active_game=Game.TEKKEN8,
            enabled_games=frozenset({Game.TEKKEN8}),
            enabled_events=frozenset({EventType.MATCH_END}),
        )
    )
    save_config(path, updated)
    reloaded = load_config(path)
    assert reloaded.runtime.active_game is Game.TEKKEN8
    assert reloaded.runtime.enabled_games == frozenset({Game.TEKKEN8})
    assert reloaded.obs.source_name == "Game Capture", "unrelated settings preserved"


def test_ui_port_survives_save_config_round_trip(tmp_path):
    """Guards against a field added to ServerConfig, load_config, and
    config.example.toml but forgotten in save_config's document dict (or vice
    versa) — this project has already shipped that exact bug once. If
    save_config's `server` table drops `ui_port`, the reload below falls back
    to the dataclass default (6601) instead of the non-default value set
    here, and this assertion fails.
    """
    from dataclasses import replace

    path = _write(tmp_path, VALID)
    config = load_config(path)
    updated = replace(config, server=replace(config.server, ui_port=9999))
    save_config(path, updated)
    assert load_config(path).server.ui_port == 9999


def test_save_config_escapes_awkward_strings(tmp_path):
    # Built via dataclasses.replace rather than interpolated into raw TOML
    # text: a naive f-string writer would produce invalid TOML for a value
    # containing an unescaped quote and backslash, so this only proves
    # anything if save_config does real escaping (e.g. via tomli_w).
    from dataclasses import replace

    path = _write(tmp_path, VALID)
    config = load_config(path)
    awkward = replace(
        config, obs=replace(config.obs, source_name='Weird "Name" \\ Here')
    )
    save_config(path, awkward)
    assert load_config(path).obs.source_name == 'Weird "Name" \\ Here'
