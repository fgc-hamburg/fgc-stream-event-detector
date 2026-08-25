import re
from pathlib import Path

import pytest

from fgc_detector.config import ConfigError, load_config
from fgc_detector.types import EventType, Game

EXAMPLE_CONFIG = Path(__file__).resolve().parent.parent / "config.example.toml"

VALID = """
game = "sf6"

[obs]
host = "localhost"
port = 4455
password = "secret"
source_name = "Game Capture"
poll_hz = 5.0

[server]
host = "127.0.0.1"
port = 6600

[confirmer]
agreement_frames = 3
cooldown_max_seconds = 180.0
"""


def _write(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_loads_a_valid_config(tmp_path):
    config = load_config(_write(tmp_path, VALID))
    assert config.game is Game.SF6
    assert config.obs.source_name == "Game Capture"
    assert config.obs.poll_hz == 5.0
    assert config.server.port == 6600
    assert config.confirmer.agreement_frames == 3
    # Runtime settings default to the full roster and event set when the
    # config has no [runtime] section.
    assert config.runtime.active_game is Game.SF6
    assert config.runtime.enabled_games == frozenset(Game)
    assert config.runtime.enabled_events == frozenset({EventType.MATCH_END})


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.toml")


def test_unknown_game_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="unknown game"):
        load_config(_write(tmp_path, VALID.replace('"sf6"', '"smash"')))


def test_missing_source_name_raises_config_error(tmp_path):
    text = VALID.replace('source_name = "Game Capture"\n', "")
    with pytest.raises(ConfigError, match="source_name"):
        load_config(_write(tmp_path, text))


def test_defaults_are_applied_for_optional_fields(tmp_path):
    minimal = """
game = "tekken8"

[obs]
source_name = "Capture"
"""
    config = load_config(_write(tmp_path, minimal))
    assert config.obs.host == "localhost"
    assert config.obs.port == 4455
    assert config.server.port == 6600
    assert config.confirmer.agreement_frames == 3
    assert config.runtime.enabled_games == frozenset(Game)
    assert config.runtime.enabled_events == frozenset({EventType.MATCH_END})


def test_malformed_toml_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="could not be parsed"):
        load_config(_write(tmp_path, "this is [not toml"))


def test_missing_game_key_raises_config_error(tmp_path):
    text = VALID.replace('game = "sf6"\n', "")
    with pytest.raises(ConfigError, match="game is required"):
        load_config(_write(tmp_path, text))


@pytest.mark.parametrize("section", ["obs", "server", "confirmer"])
def test_non_table_section_raises_config_error(tmp_path, section):
    # Drop the [section] table entirely and give the same key a scalar value
    # instead, so it is a genuinely non-table value at the top level.
    text = re.sub(rf"\[{section}\]\n(?:[^\[\n]*\n)*", "", VALID)
    text = f'{section} = "oops"\n' + text
    with pytest.raises(ConfigError, match=section):
        load_config(_write(tmp_path, text))


def test_example_config_loads_cleanly_with_documented_defaults():
    config = load_config(EXAMPLE_CONFIG)
    assert config.game is Game.SF6
    assert config.obs.source_name == "Game Capture"
    assert config.obs.host == "localhost"
    assert config.obs.port == 4455
    assert config.obs.password == ""
    assert config.obs.poll_hz == 5.0
    assert config.server.host == "127.0.0.1"
    assert config.server.port == 6600
    assert config.server.ui_port == 6601
    assert config.confirmer.agreement_frames == 3
    assert config.confirmer.cooldown_max_seconds == 180.0
    assert config.confirmer.streak_staleness_seconds == 3.0
    assert config.runtime.active_game is Game.SF6
    assert config.runtime.enabled_games == frozenset(
        {Game.SF6, Game.TEKKEN8, Game.AVATAR, Game.TOKON}
    )
    assert config.runtime.enabled_events == frozenset({EventType.MATCH_END})
