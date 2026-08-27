"""Picks the confirmation strategy for a game.

Two strategies exist today, both exposing the same interface (observe/arm/
disarm/set_game/configure/state/armed/game): `Confirmer` (the marker/round path, the
default) and `SetScoreConfirmer` (the SF6 games-won-in-set counter path).
This is a deliberately small, explicit map -- generalizing this into a
detector-declared strategy is a follow-up (see docs/TODO.md), not something
to build here.
"""

from __future__ import annotations

from .confirmer import Confirmer, ConfirmerConfig
from .set_score_confirmer import SetScoreConfirmer
from .types import Game

# Both strategies expose the same interface (observe/arm/disarm/set_game/
# configure/state/armed/game); consumers that hold "a confirmer" without caring which strategy
# it is should type-hint with this alias rather than the concrete `Confirmer`.
ConfirmerLike = Confirmer | SetScoreConfirmer


def make_confirmer(game: Game, config: ConfirmerConfig) -> ConfirmerLike:
    """Return the confirmation strategy for `game`.

    SF6 uses the set-score counter path; every other game uses the default
    marker/round path.

    Called once at CLI startup. The runtime `set_game` command (see
    `server.py`'s `EventServer._apply`) mutates an existing confirmer's game
    in place; it does not call back into this factory, so it cannot swap
    strategies. That's harmless today (only SF6 is registered) but will
    silently stop detection if a game is registered whose strategy differs
    from whatever confirmer is already live -- see docs/TODO.md.
    """
    if game is Game.SF6:
        return SetScoreConfirmer(game, config)
    return Confirmer(game, config)
