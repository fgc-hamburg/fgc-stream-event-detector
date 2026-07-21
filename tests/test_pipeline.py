from datetime import datetime, timedelta, timezone

import numpy as np

from fgc_detector.confirmer import Confirmer, ConfirmerConfig
from fgc_detector.pipeline import run_offline
from fgc_detector.types import Frame, Game, Observation, Screen, Side

START = datetime(2026, 7, 21, 20, 0, 0, tzinfo=timezone.utc)


class ScriptedDetector:
    """Returns a pre-written observation per frame, ignoring pixels."""

    game = Game.SF6
    canonical_size = (1920, 1080)

    def __init__(self, script):
        self._script = list(script)
        self._index = 0

    def observe(self, frame):
        observation = self._script[self._index]
        self._index += 1
        return observation


class ScriptedSource:
    def __init__(self, count):
        self._count = count

    def frames(self):
        for index in range(self._count):
            yield Frame(
                image=np.zeros((1080, 1920, 3), dtype=np.uint8),
                captured_at=START + timedelta(seconds=index * 0.2),
            )

    def close(self):
        pass


def test_pipeline_emits_one_event_for_a_clean_match():
    script = [Observation(Screen.IN_MATCH)] * 5 + [
        Observation(Screen.MATCH_END, winner=Side.P1, confidence=0.9)
    ] * 3
    confirmer = Confirmer(Game.SF6, ConfirmerConfig(agreement_frames=3))
    confirmer.arm()
    events = run_offline(ScriptedSource(len(script)), ScriptedDetector(script), confirmer)
    assert len(events) == 1
    assert events[0].winner is Side.P1


def test_pipeline_emits_nothing_when_disarmed():
    script = [Observation(Screen.IN_MATCH)] * 5 + [
        Observation(Screen.MATCH_END, winner=Side.P1, confidence=0.9)
    ] * 3
    confirmer = Confirmer(Game.SF6, ConfirmerConfig(agreement_frames=3))
    events = run_offline(ScriptedSource(len(script)), ScriptedDetector(script), confirmer)
    assert events == []


def test_pipeline_uses_the_frame_timestamp_not_wall_clock():
    script = [Observation(Screen.IN_MATCH)] * 2 + [
        Observation(Screen.MATCH_END, winner=Side.P2, confidence=0.8)
    ] * 3
    confirmer = Confirmer(Game.SF6, ConfirmerConfig(agreement_frames=3))
    confirmer.arm()
    events = run_offline(ScriptedSource(len(script)), ScriptedDetector(script), confirmer)
    assert events[0].ts == START + timedelta(seconds=0.8)


def test_pipeline_records_evidence_when_a_recorder_is_supplied(tmp_path):
    from fgc_detector.observability import FireRecorder

    script = [Observation(Screen.IN_MATCH)] * 2 + [
        Observation(Screen.MATCH_END, winner=Side.P1, confidence=0.9)
    ] * 3
    confirmer = Confirmer(Game.SF6, ConfirmerConfig(agreement_frames=3))
    confirmer.arm()
    run_offline(
        ScriptedSource(len(script)),
        ScriptedDetector(script),
        confirmer,
        recorder=FireRecorder(tmp_path),
    )
    assert len(list(tmp_path.glob("*.png"))) == 1
