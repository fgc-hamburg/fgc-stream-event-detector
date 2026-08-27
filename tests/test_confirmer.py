from datetime import datetime, timedelta, timezone

import pytest

from fgc_detector.confirmer import Confirmer, ConfirmerConfig
from fgc_detector.types import (
    DETAIL_P1_ROUNDS,
    DETAIL_P2_ROUNDS,
    ConfirmerState,
    Game,
    Observation,
    Screen,
    Side,
)

START = datetime(2026, 7, 21, 20, 0, 0, tzinfo=timezone.utc)


def _in_match() -> Observation:
    return Observation(screen=Screen.IN_MATCH)


def _match_end(winner: Side, confidence: float = 0.9) -> Observation:
    return Observation(screen=Screen.MATCH_END, winner=winner, confidence=confidence)


def _char_select() -> Observation:
    return Observation(screen=Screen.CHAR_SELECT)


def _in_match_rounds(p1: str, p2: str) -> Observation:
    return Observation(
        screen=Screen.IN_MATCH,
        details={DETAIL_P1_ROUNDS: p1, DETAIL_P2_ROUNDS: p2},
    )


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

    The replay is of the *decisive* (final) round of the set, so a real
    detector's round-counter fields read the end-of-game state -- e.g. 2-1,
    not 0-0. This must not be confused with the start of a fresh game, so it
    must not release cooldown either. If this test only passed because the
    replay frames omitted round details entirely, it would be worthless: real
    detectors (e.g. MarkerRoundDetector) publish round counters on every
    IN_MATCH observation, including during a replay.
    """
    driver.feed(_in_match(), 10).feed(_match_end(Side.P1), 3)
    assert len(driver.events) == 1

    driver.advance(10)  # replay begins
    driver.feed(_in_match_rounds("2", "1"), 30).feed(_match_end(Side.P1), 10)
    assert len(driver.events) == 1, "replayed KO must not fire a second event"


def test_a_replay_starting_from_round_one_does_release_cooldown_known_limitation(driver):
    """Known, accepted limitation: a replay of round 1 reads as a fresh game.

    Round counters read 0-0 at the start of round 1, and the cooldown's 0-0
    exit cannot distinguish "round 1 is genuinely starting" from "we are
    replaying a set's very first round, which also displayed 0-0". If a
    replay happens to show round 1 (rather than the more typical final-round
    replay), cooldown releases early and a replayed KO could fire as a
    phantom second event.

    This trade-off is deliberate, not a bug: without the 0-0 exit, the
    detector wedges through every rematch that skips CHAR_SELECT and misses
    game 2 of the set entirely -- a far more damaging and far more common
    failure than the rare case of a round-1 replay. This test documents that
    reality so nobody "fixes" it by accident; it is not asserting desired
    behavior.
    """
    driver.feed(_in_match(), 10).feed(_match_end(Side.P1), 3)
    assert driver.confirmer.state is ConfirmerState.COOLDOWN

    driver.feed(_in_match_rounds("0", "0"), 3)  # replay happens to show round 1
    assert driver.confirmer.state is ConfirmerState.IDLE, (
        "documented limitation: a round-1 replay releases cooldown"
    )

    # The replay is still playing out on screen: its gameplay promotes back
    # to LIVE, and its (replayed) KO then confirms a second, phantom event
    # for what is really still the same game.
    driver.feed(_in_match(), 3)
    driver.feed(_match_end(Side.P1), 3)
    assert len(driver.events) == 2, (
        "documented limitation: the released cooldown lets the replayed KO "
        "fire a phantom duplicate event"
    )
    assert driver.events[1].winner is Side.P1


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


def test_unknown_frames_interleaved_within_a_streak_do_not_break_it(driver):
    """UNKNOWN must be ignored even mid-streak, not just before one starts.

    Feeding UNKNOWN only before the streak begins can't distinguish "UNKNOWN is
    ignored" from "UNKNOWN clears an (already empty) streak". Interleaving it
    between agreeing MATCH_END frames is the real test.
    """
    driver.feed(_in_match(), 5)
    driver.feed(_match_end(Side.P1), 1)
    driver.feed(Observation(Screen.UNKNOWN), 1)
    driver.feed(_match_end(Side.P1), 1)
    driver.feed(Observation(Screen.UNKNOWN), 1)
    driver.feed(_match_end(Side.P1), 1)
    assert len(driver.events) == 1


def test_missed_game_2_regression_cooldown_releases_on_zero_zero_rounds(driver):
    """The bug this fix exists for: rematches skip CHAR_SELECT.

    Game 1 ends, cooldown begins. The operator confirmed CHAR_SELECT is often
    never seen between games of a set. Once round counters agree at 0-0 for
    `agreement_frames` consecutive IN_MATCH frames, cooldown must release so
    game 2 can be detected -- without this, the detector wedges until the
    180s safety valve and misses game 2 entirely.
    """
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    assert driver.confirmer.state is ConfirmerState.COOLDOWN

    driver.feed(_in_match_rounds("0", "0"), 3)
    assert driver.confirmer.state is ConfirmerState.IDLE

    driver.feed(_in_match(), 5).feed(_match_end(Side.P2), 3)
    assert len(driver.events) == 2
    assert driver.events[1].winner is Side.P2


def test_single_zero_zero_frame_does_not_release_cooldown(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    assert driver.confirmer.state is ConfirmerState.COOLDOWN

    driver.feed(_in_match_rounds("0", "0"), 1)
    assert driver.confirmer.state is ConfirmerState.COOLDOWN


def test_nonzero_rounds_do_not_release_cooldown(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    assert driver.confirmer.state is ConfirmerState.COOLDOWN

    driver.feed(_in_match_rounds("2", "0"), 5)
    assert driver.confirmer.state is ConfirmerState.COOLDOWN


def test_missing_or_unparseable_round_details_do_not_release_cooldown_or_raise(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    assert driver.confirmer.state is ConfirmerState.COOLDOWN

    driver.feed(_in_match(), 5)  # no details at all (e.g. NullDetector)
    assert driver.confirmer.state is ConfirmerState.COOLDOWN

    driver.feed(
        Observation(
            screen=Screen.IN_MATCH,
            details={DETAIL_P1_ROUNDS: "oops", DETAIL_P2_ROUNDS: "0"},
        ),
        5,
    )
    assert driver.confirmer.state is ConfirmerState.COOLDOWN


def test_stale_partial_streak_does_not_fire(driver):
    driver.feed(_in_match(), 5)
    driver.feed(_match_end(Side.P1), 2)
    driver.advance(600)  # ten minutes later, well past streak_staleness_seconds
    driver.feed(_match_end(Side.P1), 1)
    assert driver.events == []


def test_streak_within_staleness_window_still_fires(driver):
    driver.feed(_in_match(), 5)
    driver.feed(_match_end(Side.P1), 2)
    driver.advance(1)  # within the default 3s staleness window
    driver.feed(_match_end(Side.P1), 1)
    assert len(driver.events) == 1


def test_streak_staleness_seconds_must_be_positive():
    with pytest.raises(ValueError):
        ConfirmerConfig(streak_staleness_seconds=0)


def test_match_end_without_a_winner_is_ignored(driver):
    driver.feed(_in_match(), 5).feed(Observation(Screen.MATCH_END, winner=None), 10)
    assert driver.events == []


def test_agreement_frames_must_be_positive():
    with pytest.raises(ValueError):
        ConfirmerConfig(agreement_frames=0)


def test_unknown_frames_interleaved_within_a_zero_streak_still_release_cooldown(driver):
    """Finding 1's missed-detection regression.

    Before this fix, _observe_cooldown treated ANY non-fresh-game observation
    -- including UNKNOWN -- as clearing _zero_streak, unlike the symmetric
    treatment UNKNOWN already got in _observe_live. A single UNKNOWN flicker
    at the start of game 2 (transitions and flashes are common there) would
    then prevent cooldown from ever releasing on the 0-0 path, wedging the
    detector until the 180s safety valve -- by which point game 2 may be
    over. UNKNOWN must be neutral here exactly as it is for the MATCH_END
    streak.
    """
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    assert driver.confirmer.state is ConfirmerState.COOLDOWN

    driver.feed(_in_match_rounds("0", "0"), 1)
    driver.feed(Observation(Screen.UNKNOWN), 1)
    driver.feed(_in_match_rounds("0", "0"), 1)
    driver.feed(Observation(Screen.UNKNOWN), 1)
    driver.feed(_in_match_rounds("0", "0"), 1)
    assert driver.confirmer.state is ConfirmerState.IDLE


def test_zero_streak_spanning_a_gap_past_staleness_does_not_release_cooldown(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    assert driver.confirmer.state is ConfirmerState.COOLDOWN

    driver.feed(_in_match_rounds("0", "0"), 2)
    driver.advance(10)  # past streak_staleness_seconds, well short of the 180s safety valve
    driver.feed(_in_match_rounds("0", "0"), 1)
    assert driver.confirmer.state is ConfirmerState.COOLDOWN


def test_zero_streak_within_staleness_window_still_releases_cooldown(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    assert driver.confirmer.state is ConfirmerState.COOLDOWN

    driver.feed(_in_match_rounds("0", "0"), 2)
    driver.advance(1)  # within the default 3s staleness window
    driver.feed(_in_match_rounds("0", "0"), 1)
    assert driver.confirmer.state is ConfirmerState.IDLE


def test_zero_streak_staleness_boundary_is_inclusive():
    """A gap of exactly streak_staleness_seconds must not discard the streak.

    Mirrors the MATCH_END streak's `>` (not `>=`) staleness comparison: only
    a gap strictly greater than the configured window counts as stale. Uses
    a zero-step driver so the boundary gap is exact, not fudged by the
    fixture's per-frame step.
    """
    confirmer = Confirmer(Game.SF6, ConfirmerConfig(agreement_frames=3))
    confirmer.arm()
    driver = Driver(confirmer, step=0.0)
    driver.feed(_in_match(), 1).feed(_match_end(Side.P1), 3)
    assert driver.confirmer.state is ConfirmerState.COOLDOWN

    driver.feed(_in_match_rounds("0", "0"), 2)
    driver.advance(3.0)  # exactly streak_staleness_seconds
    driver.feed(_in_match_rounds("0", "0"), 1)
    assert driver.confirmer.state is ConfirmerState.IDLE


def test_match_end_streak_staleness_boundary_is_inclusive():
    """A gap of exactly streak_staleness_seconds must not discard the streak."""
    confirmer = Confirmer(Game.SF6, ConfirmerConfig(agreement_frames=3))
    confirmer.arm()
    driver = Driver(confirmer, step=0.0)
    driver.feed(_in_match(), 1)
    driver.feed(_match_end(Side.P1), 2)
    driver.advance(3.0)  # exactly streak_staleness_seconds
    driver.feed(_match_end(Side.P1), 1)
    assert len(driver.events) == 1


def test_match_end_streak_after_stale_discard_counts_as_one_not_zero(driver):
    """A mutant that clears _streak without appending the new frame would
    pass the existing staleness tests (they only assert non-firing) but
    would require a 4th agreeing frame here instead of the 3rd. Pin the
    count, not just the non-firing behavior.
    """
    driver.feed(_in_match(), 5)
    driver.feed(_match_end(Side.P1), 2)
    driver.advance(600)  # discard: well past staleness
    driver.feed(_match_end(Side.P1), 3)  # the discarded frame counts as 1, so 3 more fire
    assert len(driver.events) == 1


def test_zero_streak_after_stale_discard_counts_as_one_not_zero(driver):
    """Same pin as the MATCH_END streak, for _zero_streak."""
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    assert driver.confirmer.state is ConfirmerState.COOLDOWN

    driver.feed(_in_match_rounds("0", "0"), 2)
    driver.advance(10)  # discard: past staleness, well short of the 180s safety valve
    driver.feed(_in_match_rounds("0", "0"), 3)  # discarded frame counts as 1, so 3 more release
    assert driver.confirmer.state is ConfirmerState.IDLE


def test_detector_misreading_win_screen_as_fresh_game_fires_phantom_duplicate(driver):
    """Finding 2: an undocumented spurious release path, distinct from the
    round-1-replay limitation above.

    If a detector misreads the still-displayed win screen itself as IN_MATCH
    with 0-0 markers (rather than genuinely replaying gameplay), the same
    0-0 exit releases cooldown on a screen that was never a fresh game. The
    next IN_MATCH observation (the same misread win screen) promotes back to
    LIVE, and the win screen's MATCH_END frames then confirm a duplicate
    event for the *same* game.

    This is documented, not fixed, here: the correct fix is at the detector
    level (a win screen has no health bar, so a correct detector reports
    UNKNOWN, not IN_MATCH). Agreement over `agreement_frames` frames and the
    MATCH_END-clears-the-counter rule are secondary defences only.
    """
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    assert len(driver.events) == 1
    assert driver.confirmer.state is ConfirmerState.COOLDOWN

    # The win screen is still on screen, but a detector bug reports it as a
    # fresh IN_MATCH 0-0 for `agreement_frames` consecutive frames.
    driver.feed(_in_match_rounds("0", "0"), 3)
    assert driver.confirmer.state is ConfirmerState.IDLE

    # The still-displayed win screen promotes back to LIVE...
    driver.feed(_in_match(), 3)
    # ...and its MATCH_END frames confirm a phantom duplicate for the same game.
    driver.feed(_match_end(Side.P1), 3)
    assert len(driver.events) == 2
    assert driver.events[1].winner is Side.P1


# --- live reconfiguration ---------------------------------------------------


def test_configure_raises_the_bar_mid_streak():
    """A streak gathered under the old threshold must not be allowed to fire
    against the new one: the operator raised the bar because they did not
    trust those frames."""
    confirmer = Confirmer(Game.AVATAR, ConfirmerConfig(agreement_frames=3))
    confirmer.arm()
    driver = Driver(confirmer)
    driver.feed(_in_match()).feed(_match_end(Side.P1), times=2)

    confirmer.configure(ConfirmerConfig(agreement_frames=3))
    driver.feed(_match_end(Side.P1))
    assert driver.events == []


def test_configure_takes_effect_for_the_next_streak():
    confirmer = Confirmer(Game.AVATAR, ConfirmerConfig(agreement_frames=3))
    confirmer.arm()
    confirmer.configure(ConfirmerConfig(agreement_frames=2))
    driver = Driver(confirmer)
    driver.feed(_in_match()).feed(_match_end(Side.P1), times=2)
    assert len(driver.events) == 1


def test_configure_keeps_the_confirmer_armed():
    """Retuning a threshold mid-set must not silently take the detector
    off-air; only arm/disarm change armedness."""
    confirmer = Confirmer(Game.AVATAR, ConfirmerConfig())
    confirmer.arm()
    confirmer.configure(ConfirmerConfig(agreement_frames=4))
    assert confirmer.armed is True


def test_configure_does_not_release_an_active_cooldown():
    """Cooldown is what stops a second event firing on the same match end.
    Dropping it because a threshold was edited would re-fire the match."""
    confirmer = Confirmer(Game.AVATAR, ConfirmerConfig(agreement_frames=2))
    confirmer.arm()
    driver = Driver(confirmer)
    driver.feed(_in_match()).feed(_match_end(Side.P1), times=2)
    assert confirmer.state is ConfirmerState.COOLDOWN

    confirmer.configure(ConfirmerConfig(agreement_frames=2))
    assert confirmer.state is ConfirmerState.COOLDOWN
    driver.feed(_match_end(Side.P1), times=3)
    assert len(driver.events) == 1
