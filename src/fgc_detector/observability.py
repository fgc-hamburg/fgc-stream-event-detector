"""Recording every fire, so silent detector drift becomes visible.

A HUD restyle after a game patch breaks fixed ROIs without raising anything.
The detector will keep firing, confidently and wrongly. The triggering frame
plus the raw per-ROI scores are the evidence needed to spot that happening.
"""

from __future__ import annotations

import json
import logging
from itertools import count
from pathlib import Path

import cv2

from .events import MatchEndEvent
from .types import Frame, Observation

log = logging.getLogger(__name__)


class FireRecorder:
    def __init__(self, output_dir: Path) -> None:
        self._dir = Path(output_dir)
        self._counter = count()

    def record(
        self, event: MatchEndEvent, frame: Frame, observation: Observation
    ) -> Path:
        """Write the triggering frame and its scores. Returns the PNG's path."""
        self._dir.mkdir(parents=True, exist_ok=True)

        stamp = event.ts.strftime("%Y-%m-%dT%H-%M-%S")
        name = f"{stamp}_{event.game.value}_{event.winner.value}_{next(self._counter):03d}"
        png_path = self._dir / f"{name}.png"

        cv2.imwrite(str(png_path), frame.image)
        sidecar = {
            "event": event.to_dict(),
            "screen": observation.screen.name,
            "confidence": observation.confidence,
            "details": dict(observation.details),
            "debug": dict(observation.debug),
        }
        png_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
        log.info("recorded fire evidence: %s", png_path)
        return png_path
