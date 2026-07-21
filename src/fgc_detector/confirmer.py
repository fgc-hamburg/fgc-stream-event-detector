"""The state machine that turns per-frame observations into events.

All temporal logic lives here and nowhere else, so per-game detectors stay pure
and every false-positive defence is tested in one place against synthetic
observation sequences — no images, no OBS, no real clock.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .events import MatchEndEvent
from .types import ConfirmerState, Game, Observation, Screen, Side

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfirmerConfig:
    agreement_frames: int = 3
    cooldown_max_seconds: float = 180.0

    def __post_init__(self) -> None:
        if self.agreement_frames < 1:
            raise ValueError(
                f"agreement_frames must be >= 1, got {self.agreement_frames}"
            )
        if self.cooldown_max_seconds <= 0:
            raise ValueError(
                f"cooldown_max_seconds must be > 0, got {self.cooldown_max_seconds}"
            )


class Confirmer:
    def __init__(self, game: Game, config: ConfirmerConfig) -> None:
        self._game = game
        self._config = config
        self._armed = False
        self._state = ConfirmerState.IDLE
        self._streak: list[Observation] = []
        self._cooldown_started: datetime | None = None

    @property
    def state(self) -> ConfirmerState:
        return self._state

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
        self._state = ConfirmerState.IDLE
        self._streak.clear()
        self._cooldown_started = None

    def observe(self, observation: Observation, now: datetime) -> MatchEndEvent | None:
        """Feed one observation. Returns an event only when one is confirmed."""
        if not self._armed:
            return None

        match self._state:
            case ConfirmerState.IDLE:
                return self._observe_idle(observation)
            case ConfirmerState.LIVE:
                return self._observe_live(observation, now)
            case ConfirmerState.COOLDOWN:
                return self._observe_cooldown(observation, now)

    def _observe_idle(self, observation: Observation) -> None:
        if observation.screen is Screen.IN_MATCH:
            self._state = ConfirmerState.LIVE
            self._streak.clear()
        return None

    def _observe_live(
        self, observation: Observation, now: datetime
    ) -> MatchEndEvent | None:
        if observation.screen is Screen.CHAR_SELECT:
            # The match was abandoned or we misread; start over cleanly.
            self._reset()
            return None

        if observation.screen is Screen.IN_MATCH:
            self._streak.clear()
            return None

        if observation.screen is not Screen.MATCH_END or observation.winner is None:
            # UNKNOWN frames are common (transitions, flashes) and must not
            # break a run of agreeing MATCH_END frames, but they don't extend
            # one either.
            return None

        if self._streak and self._streak[-1].payload != observation.payload:
            self._streak.clear()
        self._streak.append(observation)

        if len(self._streak) < self._config.agreement_frames:
            return None

        winner = observation.winner
        confidence = min(item.confidence for item in self._streak)
        self._state = ConfirmerState.COOLDOWN
        self._cooldown_started = now
        self._streak.clear()
        log.info(
            "confirmed match_end game=%s winner=%s confidence=%.4f",
            self._game.value,
            winner.value,
            confidence,
        )
        return MatchEndEvent(
            game=self._game, winner=winner, confidence=confidence, ts=now
        )

    def _observe_cooldown(self, observation: Observation, now: datetime) -> None:
        """Hold until a definitive between-games screen.

        Deliberately does not exit on elapsed time alone: the post-match replay
        shows real gameplay and a real KO, so a time-based exit would re-arm
        mid-replay and fire on the replayed KO. CHAR_SELECT is the only signal
        that unambiguously means "the previous game is over". The safety valve
        below prevents a missed CHAR_SELECT from wedging the detector forever.
        """
        if observation.screen is Screen.CHAR_SELECT:
            self._reset()
            return None

        if self._cooldown_started is not None:
            elapsed = now - self._cooldown_started
            if elapsed > timedelta(seconds=self._config.cooldown_max_seconds):
                log.warning(
                    "cooldown safety valve released after %.0fs without CHAR_SELECT; "
                    "the character-select ROI may be miscalibrated",
                    elapsed.total_seconds(),
                )
                self._reset()
        return None
