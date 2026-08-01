"""Temporal confirmation for counter-based games-won-in-set tracking.

Mirrors `Confirmer`: all temporal logic for turning per-frame counter readings
into `match_end` events lives here, so the counter detector that feeds it
(`Sf6CounterDetector`, built separately) stays pure and stateless. This is a
parallel, independent state machine -- it does not touch or subclass the
existing marker-based `Confirmer`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from .confirmer import ConfirmerConfig
from .events import MatchEndEvent
from .types import (
    DETAIL_P1_GAMES,
    DETAIL_P2_GAMES,
    ConfirmerState,
    Game,
    Observation,
    Screen,
    Side,
)

log = logging.getLogger(__name__)

Score = tuple[int, int]


class SetScoreConfirmer:
    """Turns per-frame (p1_games, p2_games) counter readings into match_end events.

    Exposes the same public interface as `Confirmer` (observe/arm/disarm/
    set_game/state/armed/game) so the pipeline treats both confirmers
    identically. There is no cooldown concept here: a fired event's new
    baseline itself prevents a re-fire on a sustained score (see `observe`).
    """

    def __init__(self, game: Game, config: ConfirmerConfig) -> None:
        self._game = game
        self._config = config
        self._armed = False
        self._baseline: Score | None = None
        self._streak_score: Score | None = None
        self._streak_count: int = 0
        self._streak_last_ts: datetime | None = None
        self._streak_confidence: float = 0.0

    @property
    def state(self) -> ConfirmerState:
        if not self._armed or self._baseline is None:
            return ConfirmerState.IDLE
        return ConfirmerState.LIVE

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def game(self) -> Game:
        return self._game

    def arm(self) -> None:
        """Arm and reset. The dashboard calls this when a set is loaded."""
        self._armed = True
        self._reset()

    def disarm(self) -> None:
        self._armed = False
        self._reset()

    def set_game(self, game: Game) -> None:
        self._game = game
        self._reset()

    def _reset(self) -> None:
        self._baseline = None
        self._clear_streak()

    def _clear_streak(self) -> None:
        self._streak_score = None
        self._streak_count = 0
        self._streak_last_ts = None
        self._streak_confidence = 0.0

    @staticmethod
    def _read(observation: Observation) -> Score | None:
        """Parse a valid (p1, p2) reading, or None if this frame has no reading.

        Never raises: unparseable or missing values are indistinguishable
        from "no reading" (UNKNOWN, between-game transitions).
        """
        if observation.screen is not Screen.IN_MATCH:
            return None
        details = observation.details
        p1_raw = details.get(DETAIL_P1_GAMES)
        p2_raw = details.get(DETAIL_P2_GAMES)
        if p1_raw is None or p2_raw is None:
            return None
        try:
            return int(p1_raw), int(p2_raw)
        except (ValueError, TypeError):
            return None

    def observe(self, observation: Observation, now: datetime) -> MatchEndEvent | None:
        """Feed one observation. Returns an event only when one is confirmed."""
        if not self._armed:
            return None

        reading = self._read(observation)

        if reading is None:
            # UNKNOWN (or any non-reading) is expected between games: it must
            # neither extend nor break a partial streak, but a stale partial
            # streak is still discarded so it can't combine with a fresh run
            # much later.
            self._discard_if_stale(now)
            return None

        stale = self._streak_last_ts is not None and (
            now - self._streak_last_ts
            > timedelta(seconds=self._config.streak_staleness_seconds)
        )
        if self._streak_score is None or reading != self._streak_score or stale:
            self._streak_score = reading
            self._streak_count = 1
        else:
            self._streak_count += 1
        self._streak_last_ts = now
        self._streak_confidence = observation.confidence

        if self._streak_count < self._config.agreement_frames:
            return None

        return self._confirm(reading, now)

    def _discard_if_stale(self, now: datetime) -> None:
        if self._streak_last_ts is None:
            return
        stale = now - self._streak_last_ts > timedelta(
            seconds=self._config.streak_staleness_seconds
        )
        if stale:
            self._clear_streak()

    def _confirm(self, score: Score, now: datetime) -> MatchEndEvent | None:
        """A reading has reached agreement_frames. Decide whether to fire."""
        if self._baseline is None:
            # First sighting: adopt it as the baseline, never retro-fire past
            # games. This also handles arming mid-set (e.g. at 1-0).
            self._baseline = score
            return None

        if score == self._baseline:
            return None

        b1, b2 = self._baseline
        winner: Side | None = None
        if score == (b1 + 1, b2):
            winner = Side.P1
        elif score == (b1, b2 + 1):
            winner = Side.P2

        if winner is None:
            if score[0] <= b1 and score[1] <= b2:
                # Reset: a decrease on both sides, or (0, 0) after a nonzero
                # baseline. Treat as a genuine new starting point and
                # re-baseline without firing.
                log.info(
                    "set reset observed game=%s baseline=%s reading=%s; "
                    "re-baselined without firing",
                    self._game.value,
                    self._baseline,
                    score,
                )
                self._baseline = score
                return None

            # Implausible higher jump: some component is greater than the
            # baseline but it isn't a single-side +1 (e.g. +2 on one side,
            # both sides +1, or a non-adjacent jump). Hold the baseline
            # unchanged and fire nothing -- a transient misread should not
            # be able to corrupt tracking and cause a later real win to be
            # silently swallowed or misattributed. Worst case here is the
            # detector staying blind until the next arm, which is a visible
            # failure (stalled scoreboard) rather than a silent wrong winner.
            log.info(
                "implausible set-score transition ignored game=%s baseline=%s "
                "reading=%s; baseline held",
                self._game.value,
                self._baseline,
                score,
            )
            return None

        self._baseline = score
        log.info(
            "confirmed set-score increment game=%s winner=%s score=%s confidence=%.4f",
            self._game.value,
            winner.value,
            score,
            self._streak_confidence,
        )
        return MatchEndEvent(
            game=self._game,
            winner=winner,
            confidence=self._streak_confidence,
            ts=now,
        )
