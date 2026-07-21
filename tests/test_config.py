import pytest

from fgc_detector.config import ConfigError, load_config
from fgc_detector.types import Game

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


def test_malformed_toml_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="could not be parsed"):
        load_config(_write(tmp_path, "this is [not toml"))
