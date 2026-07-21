from datetime import datetime, timedelta, timezone

import pytest

from fgc_detector.confirmer import Confirmer, ConfirmerConfig
from fgc_detector.types import ConfirmerState, Game, Observation, Screen, Side

START = datetime(2026, 7, 21, 20, 0, 0, tzinfo=timezone.utc)


def _in_match() -> Observation:
    return Observation(screen=Screen.IN_MATCH)


def _match_end(winner: Side, confidence: float = 0.9) -> Observation:
    return Observation(screen=Screen.MATCH_END, winner=winner, confidence=confidence)


def _char_select() -> Observation:
    return Observation(screen=Screen.CHAR_SELECT)


class Driver:
    """Feeds observations to a Confirmer on a deterministic clock."""

    def __init__(self, confirmer: Confirmer, step: float = 0.2) -> None:
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
    confirmer = Confirmer(Game.SF6, ConfirmerConfig(agreement_frames=3))
    confirmer.arm()
    return Driver(confirmer)


def test_starts_idle_and_disarmed():
    confirmer = Confirmer(Game.SF6, ConfirmerConfig())
    assert confirmer.state is ConfirmerState.IDLE
    assert confirmer.armed is False


def test_fires_after_n_agreeing_match_end_frames(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    assert len(driver.events) == 1
    assert driver.events[0].winner is Side.P1
    assert driver.events[0].game is Game.SF6


def test_does_not_fire_before_n_frames(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 2)
    assert driver.events == []


def test_does_not_fire_when_frames_disagree_on_winner(driver):
    driver.feed(_in_match(), 5)
    driver.feed(_match_end(Side.P1), 2).feed(_match_end(Side.P2), 2)
    assert driver.events == []


def test_disagreement_restarts_the_streak_rather_than_resetting_to_zero(driver):
    driver.feed(_in_match(), 5)
    driver.feed(_match_end(Side.P1), 2).feed(_match_end(Side.P2), 3)
    assert len(driver.events) == 1
    assert driver.events[0].winner is Side.P2


def test_confidence_jitter_does_not_break_agreement(driver):
    driver.feed(_in_match(), 5)
    for confidence in (0.81, 0.93, 0.88):
        driver.feed(_match_end(Side.P1, confidence))
    assert len(driver.events) == 1


def test_reported_confidence_is_the_minimum_of_the_streak(driver):
    driver.feed(_in_match(), 5)
    for confidence in (0.9, 0.7, 0.95):
        driver.feed(_match_end(Side.P1, confidence))
    assert driver.events[0].confidence == pytest.approx(0.7)


def test_cannot_fire_from_idle_without_seeing_a_live_match(driver):
    driver.feed(_match_end(Side.P1), 10)
    assert driver.events == []


def test_in_match_frames_interrupt_a_partial_streak(driver):
    driver.feed(_in_match(), 5)
    driver.feed(_match_end(Side.P1), 2).feed(_in_match(), 1).feed(_match_end(Side.P1), 2)
    assert driver.events == []


def test_fires_only_once_for_a_sustained_match_end_screen(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 60)
    assert len(driver.events) == 1


def test_post_match_replay_does_not_fire_a_phantom_second_event(driver):
    """The single most important regression test in the suite.

    After a set, SF6 and Tekken show a replay: real gameplay, real HUD, and a
    real KO at the end of it. A detector without cooldown reports that replayed
    KO as a second game.
    """
    driver.feed(_in_match(), 10).feed(_match_end(Side.P1), 3)
    assert len(driver.events) == 1

    driver.advance(10)  # replay begins
    driver.feed(_in_match(), 30).feed(_match_end(Side.P1), 10)
    assert len(driver.events) == 1, "replayed KO must not fire a second event"


def test_char_select_ends_cooldown_and_the_next_match_can_fire(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    driver.feed(_char_select(), 5)
    assert driver.confirmer.state is ConfirmerState.IDLE
    driver.feed(_in_match(), 5).feed(_match_end(Side.P2), 3)
    assert len(driver.events) == 2
    assert driver.events[1].winner is Side.P2


def test_safety_valve_releases_cooldown_if_char_select_is_never_seen(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    assert driver.confirmer.state is ConfirmerState.COOLDOWN
    driver.advance(181).feed(Observation(Screen.UNKNOWN))
    assert driver.confirmer.state is ConfirmerState.IDLE


def test_disarmed_confirmer_never_fires():
    confirmer = Confirmer(Game.SF6, ConfirmerConfig(agreement_frames=3))
    driver = Driver(confirmer)  # never armed
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 10)
    assert driver.events == []


def test_disarming_mid_streak_discards_it(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 2)
    driver.confirmer.disarm()
    driver.confirmer.arm()
    driver.feed(_match_end(Side.P1), 2)
    assert driver.events == []


def test_arming_resets_state_to_idle(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    assert driver.confirmer.state is ConfirmerState.COOLDOWN
    driver.confirmer.arm()
    assert driver.confirmer.state is ConfirmerState.IDLE


def test_set_game_changes_the_reported_game_and_resets(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 2)
    driver.confirmer.set_game(Game.TEKKEN8)
    assert driver.confirmer.state is ConfirmerState.IDLE
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    assert driver.events[0].game is Game.TEKKEN8


def test_unknown_screens_do_not_disturb_a_live_match(driver):
    driver.feed(_in_match(), 5).feed(Observation(Screen.UNKNOWN), 3)
    assert driver.confirmer.state is ConfirmerState.LIVE
    driver.feed(_match_end(Side.P1), 3)
    assert len(driver.events) == 1


def test_match_end_without_a_winner_is_ignored(driver):
    driver.feed(_in_match(), 5).feed(Observation(Screen.MATCH_END, winner=None), 10)
    assert driver.events == []


def test_agreement_frames_must_be_positive():
    with pytest.raises(ValueError):
        ConfirmerConfig(agreement_frames=0)
