"""Recording every fire, so silent detector drift becomes visible.

A HUD restyle after a game patch breaks fixed ROIs without raising anything.
The detector will keep firing, confidently and wrongly. The triggering frame
plus the raw per-ROI scores are the evidence needed to spot that happening.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2

from .events import MatchEndEvent
from .types import Frame, Observation

log = logging.getLogger(__name__)


class FireRecorder:
    def __init__(self, output_dir: Path) -> None:
        self._dir = Path(output_dir)

    def record(
        self, event: MatchEndEvent, frame: Frame, observation: Observation
    ) -> Path | None:
        """Write the triggering frame and its scores.

        Returns the PNG's path, or None if evidence could not be written.
        Recording evidence must never break event emission, so every failure
        here is caught and logged rather than raised.
        """
        try:
            self._dir.mkdir(parents=True, exist_ok=True)

            stamp = event.ts.strftime("%Y-%m-%dT%H-%M-%S")
            base = f"{stamp}_{event.game.value}_{event.winner.value}"
            png_path = self._unused_path(base)

            try:
                wrote = cv2.imwrite(str(png_path), frame.image)
            except Exception:
                log.error(
                    "failed to write fire evidence image %s", png_path, exc_info=True
                )
                return None
            if not wrote:
                log.error("cv2.imwrite reported failure writing %s", png_path)
                return None

            sidecar = {
                "event": event.to_dict(),
                "screen": observation.screen.name,
                "confidence": observation.confidence,
                "details": dict(observation.details),
                "debug": dict(observation.debug),
            }
            png_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
        except OSError:
            log.error(
                "failed to record fire evidence in %s", self._dir, exc_info=True
            )
            return None

        log.info("recorded fire evidence: %s", png_path)
        return png_path

    def _unused_path(self, base: str) -> Path:
        """Pick a PNG path under base that does not already exist on disk.

        A counter kept on the instance is not enough: a fresh FireRecorder
        (e.g. after a process restart) starts counting from zero again, so it
        could pick a name a previous instance already used and silently
        overwrite that earlier evidence. Checking the filesystem holds across
        separate instances pointed at the same directory.
        """
        n = 0
        while True:
            candidate = self._dir / f"{base}_{n:03d}.png"
            if not candidate.exists():
                return candidate
            n += 1
