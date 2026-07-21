"""Offline frame sources: a folder of stills, and a recorded video.

These are what make per-game detector work tractable — a detector can be tuned
against last week's VOD at many times realtime instead of in front of a console.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import cv2

from ..types import Frame
from .normalize import normalize

log = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class FolderFrameSource:
    """Yields every readable image in a directory, sorted by filename."""

    def __init__(self, path: Path, canonical: tuple[int, int]) -> None:
        self._path = Path(path)
        self._canonical = canonical

    def frames(self) -> Iterator[Frame]:
        paths = sorted(
            entry
            for entry in self._path.iterdir()
            if entry.is_file() and entry.suffix.lower() in _IMAGE_SUFFIXES
        )
        for entry in paths:
            image = cv2.imread(str(entry))
            if image is None:
                log.warning("skipping unreadable image: %s", entry)
                continue
            normalized = normalize(image, self._canonical)
            if normalized is None:
                log.warning("skipping wrong-aspect image: %s %s", entry, image.shape)
                continue
            yield Frame(image=normalized, captured_at=datetime.now(timezone.utc))

    def close(self) -> None:
        return None


class VideoFrameSource:
    """Yields every `sample_every`-th frame of a video file."""

    def __init__(
        self, path: Path, canonical: tuple[int, int], sample_every: int = 1
    ) -> None:
        if sample_every < 1:
            raise ValueError(f"sample_every must be >= 1, got {sample_every}")
        self._path = Path(path)
        self._canonical = canonical
        self._sample_every = sample_every
        self._capture: cv2.VideoCapture | None = None

    def frames(self) -> Iterator[Frame]:
        if not self._path.exists():
            raise FileNotFoundError(self._path)
        self._capture = cv2.VideoCapture(str(self._path))
        if not self._capture.isOpened():
            raise FileNotFoundError(f"could not open video: {self._path}")

        index = 0
        try:
            while True:
                ok, image = self._capture.read()
                if not ok:
                    return
                if index % self._sample_every == 0:
                    normalized = normalize(image, self._canonical)
                    if normalized is not None:
                        yield Frame(
                            image=normalized, captured_at=datetime.now(timezone.utc)
                        )
                index += 1
        finally:
            self.close()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
