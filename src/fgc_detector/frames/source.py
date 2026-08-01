"""The frame-source seam.

Three implementations exist: a folder of PNGs and a video file (both offline,
used by tests and by ROI tuning) and OBS (live). Everything downstream is
written against this protocol so tuning a detector never requires OBS running.
"""

from __future__ import annotations

from typing import Iterator, Protocol

from ..types import Frame


class FrameSource(Protocol):
    def frames(self) -> Iterator[Frame]:
        """Yield frames until exhausted or closed."""
        ...

    def close(self) -> None:
        """Release any underlying resource. Safe to call more than once."""
        ...
