"""Offline frame sources: a folder of stills, and a recorded video.

These are what make per-game detector work tractable — a detector can be tuned
against last week's VOD at many times realtime instead of in front of a console.

`VideoFrameSource` stamps frames with a timestamp derived from their position
in the video (start_time + frame_index/fps) rather than wall-clock time. This
lets the `replay` CLI's timeline be checked against the VOD by eye, and lets
downstream time-based logic (e.g. the Confirmer's cooldown) see the same
elapsed time it would during a live stream, even though replay itself runs
much faster than realtime. `FolderFrameSource` has no inherent timeline, so it
keeps stamping wall-clock time.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
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


_DEFAULT_FPS = 30.0
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class VideoFrameSource:
    """Yields every `sample_every`-th frame of a video file.

    Frame timestamps are positional: `captured_at` for frame N is
    `start_time + N/fps`, where N is the decoded frame index (it advances on
    every frame read, not just sampled ones). This makes replay timelines
    line up with the VOD and lets time-based downstream logic behave as it
    would live, regardless of how fast replay actually runs.
    """

    def __init__(
        self,
        path: Path,
        canonical: tuple[int, int],
        sample_every: int = 1,
        start_time: datetime | None = None,
    ) -> None:
        if sample_every < 1:
            raise ValueError(f"sample_every must be >= 1, got {sample_every}")
        if start_time is not None and start_time.tzinfo is None:
            raise ValueError("start_time must be timezone-aware")
        self._path = Path(path)
        self._canonical = canonical
        self._sample_every = sample_every
        self._start_time = start_time if start_time is not None else _EPOCH
        self._capture: cv2.VideoCapture | None = None

    def frames(self) -> Iterator[Frame]:
        if not self._path.exists():
            raise FileNotFoundError(self._path)
        self._capture = cv2.VideoCapture(str(self._path))
        if not self._capture.isOpened():
            raise FileNotFoundError(f"could not open video: {self._path}")

        fps = self._capture.get(cv2.CAP_PROP_FPS)
        if not fps or math.isnan(fps) or fps <= 0:
            log.warning(
                "video reports invalid fps (%r), falling back to %.1f: %s",
                fps,
                _DEFAULT_FPS,
                self._path,
            )
            fps = _DEFAULT_FPS

        index = 0
        try:
            while True:
                ok, image = self._capture.read()
                if not ok:
                    return
                if index % self._sample_every == 0:
                    normalized = normalize(image, self._canonical)
                    if normalized is not None:
                        captured_at = self._start_time + timedelta(
                            seconds=index / fps
                        )
                        yield Frame(image=normalized, captured_at=captured_at)
                    else:
                        log.warning("skipping wrong-aspect video frame: %s [frame %d] %s", self._path, index, image.shape)
                index += 1
        finally:
            self.close()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
