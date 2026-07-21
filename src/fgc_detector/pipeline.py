"""Wiring: frames in, events out.

Time comes from the frame's own timestamp rather than the wall clock, so a VOD
replayed at many times realtime produces exactly the events it would have
produced live.
"""

from __future__ import annotations

import logging

from .confirmer import Confirmer
from .detectors.registry import Detector
from .events import MatchEndEvent
from .frames.source import FrameSource
from .observability import FireRecorder

log = logging.getLogger(__name__)


def run_offline(
    source: FrameSource,
    detector: Detector,
    confirmer: Confirmer,
    recorder: FireRecorder | None = None,
) -> list[MatchEndEvent]:
    """Drive the pipeline to exhaustion, returning every confirmed event."""
    events: list[MatchEndEvent] = []
    try:
        for frame in source.frames():
            observation = detector.observe(frame)
            event = confirmer.observe(observation, frame.captured_at)
            if event is None:
                continue
            events.append(event)
            if recorder is not None:
                recorder.record(event, frame, observation)
    finally:
        source.close()
    return events
