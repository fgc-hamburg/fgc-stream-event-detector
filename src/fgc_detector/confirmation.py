"""Picks the confirmation strategy for a game.

Two strategies exist today, both exposing the same interface (observe/arm/
disarm/set_game/state/armed/game): `Confirmer` (the marker/round path, the
default) and `SetScoreConfirmer` (the SF6 games-won-in-set counter path).
This is a deliberately small, explicit map -- generalizing this into a
detector-declared strategy is a follow-up (see docs/TODO.md), not something
to build here.
"""

from __future__ import annotations

from .confirmer import Confirmer, ConfirmerConfig
from .set_score_confirmer import SetScoreConfirmer
from .types import Game


def make_confirmer(game: Game, config: ConfirmerConfig) -> Confirmer | SetScoreConfirmer:
    """Return the confirmation strategy for `game`.

    SF6 uses the set-score counter path; every other game uses the default
    marker/round path.
    """
    if game is Game.SF6:
        return SetScoreConfirmer(game, config)
    return Confirmer(game, config)
