"""The state machine that turns per-frame observations into events.

All temporal logic lives here and nowhere else, so per-game detectors stay pure
and every false-positive defence is tested in one place against synthetic
observation sequences — no images, no OBS, no real clock.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from .events import MatchEndEvent
from .types import (
    DETAIL_P1_ROUNDS,
    DETAIL_P2_ROUNDS,
    ConfirmerState,
    Game,
    Observation,
    Screen,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfirmerConfig:
    agreement_frames: int = 3
    cooldown_max_seconds: float = 180.0
    streak_staleness_seconds: float = 3.0

    def __post_init__(self) -> None:
        if self.agreement_frames < 1:
            raise ValueError(
                f"agreement_frames must be >= 1, got {self.agreement_frames}"
            )
        if self.cooldown_max_seconds <= 0:
            raise ValueError(
                f"cooldown_max_seconds must be > 0, got {self.cooldown_max_seconds}"
            )
        if self.streak_staleness_seconds <= 0:
            raise ValueError(
                f"streak_staleness_seconds must be > 0, got "
                f"{self.streak_staleness_seconds}"
            )


class Confirmer:
    def __init__(self, game: Game, config: ConfirmerConfig) -> None:
        self._game = game
        self._config = config
        self._armed = False
        self._state = ConfirmerState.IDLE
        self._streak: list[Observation] = []
        self._streak_last_ts: datetime | None = None
        self._cooldown_started: datetime | None = None
        self._zero_streak: list[Observation] = []
        self._zero_streak_last_ts: datetime | None = None

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

    def configure(self, config: ConfirmerConfig) -> None:
        """Adopt new thresholds without restarting.

        Any partial streak is discarded: it was gathered under the old
        thresholds, and letting it count towards a bar the operator just
        raised is exactly what they were trying to prevent. Armedness and an
        active cooldown survive -- neither is a threshold, and dropping the
        cooldown would let the match that just ended fire a second event.
        """
        self._config = config
        self._clear_streaks()

    def _clear_streaks(self) -> None:
        self._streak.clear()
        self._streak_last_ts = None
        self._zero_streak.clear()
        self._zero_streak_last_ts = None

    def _reset(self) -> None:
        self._state = ConfirmerState.IDLE
        self._cooldown_started = None
        self._clear_streaks()

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
            self._streak_last_ts = None
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
            self._streak_last_ts = None
            return None

        if observation.screen is not Screen.MATCH_END or observation.winner is None:
            # UNKNOWN frames are common (transitions, flashes) and must not
            # break a run of agreeing MATCH_END frames, but they don't extend
            # one either.
            return None

        if self._streak:
            stale = (
                self._streak_last_ts is not None
                and now - self._streak_last_ts
                > timedelta(seconds=self._config.streak_staleness_seconds)
            )
            if self._streak[-1].payload != observation.payload or stale:
                self._streak.clear()
        self._streak.append(observation)
        self._streak_last_ts = now

        if len(self._streak) < self._config.agreement_frames:
            return None

        winner = observation.winner
        confidence = min(item.confidence for item in self._streak)
        self._state = ConfirmerState.COOLDOWN
        self._cooldown_started = now
        self._streak.clear()
        self._streak_last_ts = None
        # Explicit, not incidental: every other route into COOLDOWN goes
        # through _reset(), which also clears this. State this invariant
        # here too so it doesn't depend on that being the only entry point.
        self._zero_streak.clear()
        self._zero_streak_last_ts = None
        log.info(
            "confirmed match_end game=%s winner=%s confidence=%.4f",
            self._game.value,
            winner.value,
            confidence,
        )
        return MatchEndEvent(
            game=self._game, winner=winner, confidence=confidence, ts=now
        )

    @staticmethod
    def _is_fresh_game_start(observation: Observation) -> bool:
        """True if this frame reports both round-win counters reset to 0-0.

        Round counters reset to 0-0 at the start of every game, including
        rematches that skip character select entirely. Missing or
        unparseable values are treated as "not a fresh game" rather than
        raising: a detector that never publishes these keys (NullDetector)
        must behave exactly as it did before this exit existed.
        """
        if observation.screen is not Screen.IN_MATCH:
            return False
        details = observation.details
        p1_raw = details.get(DETAIL_P1_ROUNDS)
        p2_raw = details.get(DETAIL_P2_ROUNDS)
        if p1_raw is None or p2_raw is None:
            return False
        try:
            return int(p1_raw) == 0 and int(p2_raw) == 0
        except (ValueError, TypeError):
            return False

    def _observe_cooldown(self, observation: Observation, now: datetime) -> None:
        """Hold until a definitive between-games signal.

        Deliberately does not exit on elapsed time alone: the post-match replay
        shows real gameplay and a real KO, so a time-based exit would re-arm
        mid-replay and fire on the replayed KO. Two signals are treated as
        meaning "the previous game is over": CHAR_SELECT, and round counters
        reading 0-0 for `agreement_frames` consecutive IN_MATCH observations
        (players routinely rematch without ever passing through character
        select, so CHAR_SELECT alone would wedge the detector until the
        safety valve).

        The 0-0 exit is a deliberate trade-off, not a fully safe signal: it
        defends against a replay of a *decisive* round, whose round counters
        read the end-of-game state (e.g. 2-1), not 0-0. A replay that happens
        to show round 1 of the set also reads 0-0 and will release cooldown,
        which can let a replayed KO fire as a phantom event. This is accepted
        because the alternative -- not releasing on 0-0 -- misses game 2 of
        every set whose rematch skips character select, which is the more
        damaging and more common failure. The safety valve below prevents a
        missed signal from wedging the detector forever.

        There is a second, distinct way this exit can misfire, separate from
        the round-1-replay case above: if a detector misreads the still-
        displayed win screen as IN_MATCH with 0-0 markers for
        `agreement_frames` consecutive frames, cooldown releases on a screen
        that was never a fresh game at all. The next IN_MATCH promotes to
        LIVE, and the win screen's MATCH_END frames (still on screen) then
        confirm a duplicate event for the *same* game. This is more likely at
        this operator's events than a round-1 replay -- a win screen is
        guaranteed after every game, a replay is not. The primary defence is
        at the detector level: a correctly-implemented detector reports
        UNKNOWN on a win screen (it has no health bar to match against), not
        IN_MATCH. Agreement over `agreement_frames` frames plus the
        MATCH_END-clears-the-counter rule below are secondary defences, not
        a guarantee. This Confirmer deliberately does not try to close this
        gap further: doing so would re-introduce the missed-game-2 failure
        the 0-0 exit exists to fix.

        Like the MATCH_END streak in _observe_live, this counter treats
        UNKNOWN observations as neutral -- they neither extend nor break a
        run of 0-0 frames, since transitions and flashes are common -- and is
        bounded by `streak_staleness_seconds` so a stale partial run (e.g.
        either side of a capture stall) cannot combine with a fresh one to
        release cooldown.
        """
        if observation.screen is Screen.CHAR_SELECT:
            self._reset()
            return None

        if observation.screen is Screen.UNKNOWN:
            pass  # neutral: neither extends nor breaks the run, like _streak.
        elif self._is_fresh_game_start(observation):
            stale = (
                self._zero_streak_last_ts is not None
                and now - self._zero_streak_last_ts
                > timedelta(seconds=self._config.streak_staleness_seconds)
            )
            if stale:
                self._zero_streak.clear()
            self._zero_streak.append(observation)
            self._zero_streak_last_ts = now
            if len(self._zero_streak) >= self._config.agreement_frames:
                self._reset()
                return None
        else:
            # Positive evidence the previous game is still on screen (e.g. a
            # MATCH_END frame, or IN_MATCH with nonzero/missing round data).
            self._zero_streak.clear()
            self._zero_streak_last_ts = None

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
