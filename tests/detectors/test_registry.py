import pytest

from fgc_detector.detectors.registry import (
    NullDetector,
    UnknownGameError,
    get_detector,
    register,
)
from fgc_detector.types import Game, Screen


@pytest.fixture(autouse=True)
def _clean_registry():
    from fgc_detector.detectors import registry

    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)


def test_null_detector_always_reports_unknown():
    detector = NullDetector(Game.SF6)
    observation = detector.observe(frame=None)
    assert observation.screen is Screen.UNKNOWN
    assert observation.winner is None


def test_register_then_get_returns_the_same_instance():
    detector = NullDetector(Game.TEKKEN8)
    register(detector)
    assert get_detector(Game.TEKKEN8) is detector


def test_get_unregistered_game_raises_with_a_useful_message():
    with pytest.raises(UnknownGameError, match="no detector registered"):
        get_detector(Game.SF6)


def test_registering_twice_for_one_game_raises():
    register(NullDetector(Game.SF6))
    with pytest.raises(ValueError, match="already registered"):
        register(NullDetector(Game.SF6))
