import pytest


@pytest.fixture(autouse=True)
def clean_registry():
    """Every test starts and ends with an empty detector registry.

    Shared here (rather than duplicated per-module) because both
    tests/detectors/test_registry.py and tests/test_cli.py register stub
    detectors and must not leak them into other tests.
    """
    from fgc_detector.detectors import registry

    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)
