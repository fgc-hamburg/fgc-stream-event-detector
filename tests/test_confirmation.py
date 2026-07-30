"""make_confirmer picks the right strategy per game and both strategies expose
the interface the pipeline relies on (observe/arm/disarm/set_game/state/armed/
game).
"""

from fgc_detector.confirmation import make_confirmer
from fgc_detector.confirmer import Confirmer, ConfirmerConfig
from fgc_detector.set_score_confirmer import SetScoreConfirmer
from fgc_detector.types import Game

_INTERFACE_ATTRS = ("observe", "arm", "disarm", "set_game", "state", "armed", "game")


def test_sf6_gets_set_score_confirmer() -> None:
    """Fails if make_confirmer stops special-casing SF6 (e.g. the mapping is
    removed or SF6 falls through to the default branch)."""
    confirmer = make_confirmer(Game.SF6, ConfirmerConfig())
    assert isinstance(confirmer, SetScoreConfirmer)


def test_non_sf6_gets_default_confirmer() -> None:
    """Fails if a non-SF6 game is incorrectly routed to SetScoreConfirmer, or
    if the default branch is removed/broken."""
    confirmer = make_confirmer(Game.TEKKEN8, ConfirmerConfig())
    assert isinstance(confirmer, Confirmer)
    assert not isinstance(confirmer, SetScoreConfirmer)


def test_set_score_confirmer_exposes_pipeline_interface() -> None:
    """Fails if SetScoreConfirmer is missing (or renames) any method/property
    the pipeline calls on a confirmer."""
    confirmer = make_confirmer(Game.SF6, ConfirmerConfig())
    for attr in _INTERFACE_ATTRS:
        assert hasattr(confirmer, attr), f"missing {attr}"


def test_default_confirmer_exposes_pipeline_interface() -> None:
    """Fails if Confirmer is missing (or renames) any method/property the
    pipeline calls on a confirmer."""
    confirmer = make_confirmer(Game.TEKKEN8, ConfirmerConfig())
    for attr in _INTERFACE_ATTRS:
        assert hasattr(confirmer, attr), f"missing {attr}"


def test_avatar_uses_the_marker_confirmer() -> None:
    from fgc_detector.confirmation import make_confirmer
    from fgc_detector.confirmer import Confirmer, ConfirmerConfig
    from fgc_detector.types import Game
    confirmer = make_confirmer(Game.AVATAR, ConfirmerConfig())
    assert isinstance(confirmer, Confirmer)
    assert confirmer.game is Game.AVATAR
