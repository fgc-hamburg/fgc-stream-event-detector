from datetime import datetime, timedelta, timezone

import pytest

from fgc_detector.confirmer import ConfirmerConfig
from fgc_detector.set_score_confirmer import SetScoreConfirmer
from fgc_detector.types import (
    DETAIL_P1_GAMES,
    DETAIL_P2_GAMES,
    ConfirmerState,
    Game,
    Observation,
    Screen,
    Side,
)

START = datetime(2026, 7, 21, 20, 0, 0, tzinfo=timezone.utc)


def _score(p1: int, p2: int, confidence: float = 0.9) -> Observation:
    return Observation(
        screen=Screen.IN_MATCH,
        details={DETAIL_P1_GAMES: str(p1), DETAIL_P2_GAMES: str(p2)},
        confidence=confidence,
    )


def _unknown() -> Observation:
    return Observation(screen=Screen.UNKNOWN)


class Driver:
    """Feeds observations to a SetScoreConfirmer on a deterministic clock."""

    def __init__(self, confirmer: SetScoreConfirmer, step: float = 0.2) -> None:
        self.confirmer = confirmer
        self.now = START
        self.step = timedelta(seconds=step)
        self.events = []

    def feed(self, observation: Observation, times: int = 1):
        for _ in range(times):
            event = self.confirmer.observe(observation, self.now)
            if event is not None:
                self.events.append(event)
            self.now += self.step
        return self

    def advance(self, seconds: float):
        self.now += timedelta(seconds=seconds)
        return self


@pytest.fixture
def driver():
    confirmer = SetScoreConfirmer(Game.SF6, ConfirmerConfig(agreement_frames=3))
    confirmer.arm()
    return Driver(confirmer)


def test_disarmed_confirmer_never_fires():
    confirmer = SetScoreConfirmer(Game.SF6, ConfirmerConfig(agreement_frames=3))
    driver = Driver(confirmer)  # never armed
    driver.feed(_score(0, 0), 5).feed(_score(1, 0), 10)
    assert driver.events == []


def test_first_confident_reading_adopts_baseline_and_fires_nothing(driver):
    driver.feed(_score(0, 0), 3)
    assert driver.events == []
    assert driver.confirmer.state is ConfirmerState.LIVE


def test_single_side_increment_sustained_fires_exactly_one_event(driver):
    driver.feed(_score(0, 0), 3)
    driver.feed(_score(1, 0), 3)
    assert len(driver.events) == 1
    assert driver.events[0].winner is Side.P1
    assert driver.events[0].game is Game.SF6


def test_does_not_fire_before_agreement_frames(driver):
    driver.feed(_score(0, 0), 3)
    driver.feed(_score(1, 0), 2)  # agreement_frames - 1
    assert driver.events == []


def test_one_frame_digit_misread_does_not_fire(driver):
    driver.feed(_score(0, 0), 3)
    # A single stray misread frame amid a stable 0-0 score restarts the streak
    # rather than confirming, so 2 more 0-0 frames (only 3 total after the
    # misread) must not be enough to re-confirm anything new, and the misread
    # itself must never confirm.
    driver.feed(_score(0, 1), 1)
    driver.feed(_score(0, 0), 2)
    assert driver.events == []


def test_full_clip_sequence_fires_three_events_in_order(driver):
    """Headline regression: 0-0 -> 1-0 -> 1-1 -> 2-1, UNKNOWN between games.

    If the fire condition were changed to also match same-score baselines, or
    if UNKNOWN broke the streak instead of being neutral, or if incoherent
    jumps were mis-detected as coherent, this would fire the wrong count or
    wrong winners.
    """
    driver.feed(_score(0, 0), 3)
    driver.feed(_unknown(), 5)
    driver.feed(_score(1, 0), 3)
    driver.feed(_unknown(), 5)
    driver.feed(_score(1, 1), 3)
    driver.feed(_unknown(), 5)
    driver.feed(_score(2, 1), 3)

    assert len(driver.events) == 3
    assert [e.winner for e in driver.events] == [Side.P1, Side.P2, Side.P1]

    # A fourth step: 2-1 -> 3-1 adds a fourth P1 win.
    driver.feed(_unknown(), 5)
    driver.feed(_score(3, 1), 3)
    assert len(driver.events) == 4
    assert [e.winner for e in driver.events] == [Side.P1, Side.P2, Side.P1, Side.P1]


def test_between_game_unknown_frames_do_not_break_tracking_or_fire(driver):
    driver.feed(_score(0, 0), 3)
    driver.feed(_unknown(), 20)
    assert driver.events == []
    assert driver.confirmer.state is ConfirmerState.LIVE
    driver.feed(_score(1, 0), 3)
    assert len(driver.events) == 1


def test_reset_to_zero_zero_after_nonzero_baseline_rebaselines_and_fires_nothing(driver):
    driver.feed(_score(1, 1), 3)  # armed mid-set baseline
    assert driver.events == []
    driver.feed(_score(0, 0), 3)
    assert driver.events == []
    # after re-baselining to 0-0, a genuine +1 should fire normally
    driver.feed(_score(1, 0), 3)
    assert len(driver.events) == 1
    assert driver.events[0].winner is Side.P1


def test_armed_mid_set_first_reading_is_one_zero_fires_nothing(driver):
    driver.feed(_score(1, 0), 3)
    assert driver.events == []
    assert driver.confirmer.state is ConfirmerState.LIVE


def test_incoherent_jump_both_sides_rebaselines_and_fires_nothing(driver):
    driver.feed(_score(0, 0), 3)
    driver.feed(_score(1, 1), 3)
    assert driver.events == []
    # subsequent genuine +1 from the new baseline fires normally
    driver.feed(_score(2, 1), 3)
    assert len(driver.events) == 1
    assert driver.events[0].winner is Side.P1


def test_incoherent_jump_of_two_rebaselines_and_fires_nothing(driver):
    driver.feed(_score(0, 0), 3)
    driver.feed(_score(2, 0), 3)
    assert driver.events == []


def test_stale_partial_streak_is_discarded_and_new_frame_counts_as_one(driver):
    driver.feed(_score(0, 0), 3)
    driver.feed(_score(1, 0), 2)
    driver.advance(600)  # well past streak_staleness_seconds
    driver.feed(_score(1, 0), 1)  # discarded streak counts as 1, not 3
    assert driver.events == []
    driver.feed(_score(1, 0), 2)  # 2 more needed to reach agreement_frames=3
    assert len(driver.events) == 1


def test_streak_within_staleness_window_still_fires(driver):
    driver.feed(_score(0, 0), 3)
    driver.feed(_score(1, 0), 2)
    driver.advance(1)  # within the default 3s staleness window
    driver.feed(_score(1, 0), 1)
    assert len(driver.events) == 1


def test_missing_or_unparseable_details_treated_as_no_reading_and_never_raises(driver):
    driver.feed(_score(0, 0), 3)
    driver.feed(Observation(screen=Screen.IN_MATCH), 5)  # no details at all
    assert driver.events == []
    driver.feed(
        Observation(
            screen=Screen.IN_MATCH,
            details={DETAIL_P1_GAMES: "oops", DETAIL_P2_GAMES: "0"},
        ),
        5,
    )
    assert driver.events == []
    assert driver.confirmer.state is ConfirmerState.LIVE


def test_arm_resets_baseline_after_tracking(driver):
    driver.feed(_score(0, 0), 3)
    driver.feed(_score(1, 0), 3)
    assert driver.confirmer.state is ConfirmerState.LIVE
    driver.confirmer.arm()
    assert driver.confirmer.state is ConfirmerState.IDLE


def test_long_game_does_not_double_fire(driver):
    driver.feed(_score(0, 0), 3)
    driver.feed(_score(1, 0), 60)  # sustained long game at the new score
    assert len(driver.events) == 1


def test_fired_event_carries_game_and_side(driver):
    driver.feed(_score(0, 0), 3)
    driver.feed(_score(0, 1), 3)
    assert len(driver.events) == 1
    assert driver.events[0].game is Game.SF6
    assert driver.events[0].winner is Side.P2


def test_disarm_and_rearm_clears_streak(driver):
    driver.feed(_score(0, 0), 3)
    driver.feed(_score(1, 0), 2)
    driver.confirmer.disarm()
    driver.confirmer.arm()
    driver.feed(_score(1, 0), 2)  # only 2 after rearm; would need 3 to fire
    assert driver.events == []


def test_set_game_changes_reported_game_and_resets(driver):
    driver.feed(_score(0, 0), 3)
    driver.confirmer.set_game(Game.TEKKEN8)
    assert driver.confirmer.state is ConfirmerState.IDLE
    driver.feed(_score(0, 0), 3)
    driver.feed(_score(1, 0), 3)
    assert driver.events[0].game is Game.TEKKEN8


def test_reported_confidence_is_the_confirming_frames_confidence(driver):
    driver.feed(_score(0, 0), 3)
    driver.feed(_score(1, 0, confidence=0.5), 2)
    driver.feed(_score(1, 0, confidence=0.77), 1)
    assert driver.events[0].confidence == pytest.approx(0.77)
