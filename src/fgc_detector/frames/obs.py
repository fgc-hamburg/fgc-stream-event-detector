"""Live frames from OBS via obs-websocket v5.

Targets a named source (the game capture) rather than program output, so
overlays, commentator cams and transition stingers can never contaminate the
ROIs a detector samples.
"""

from __future__ import annotations

import base64
import logging
import time
from datetime import datetime, timezone
from typing import Callable, Iterator

import cv2
import numpy as np

from ..types import Frame
from .normalize import normalize

log = logging.getLogger(__name__)

_MAX_BACKOFF_SECONDS = 10.0
_DATA_URI_PREFIX = "base64,"

#: Screenshots are requested as JPEG, not PNG. Lossless PNG compression of a
#: noisy game frame is expensive *inside OBS*, and it dominates the capture
#: cost while OBS is also busy decoding: measured against a live 1080p60
#: source, PNG at 1280x720 cost 1128ms per frame against JPEG q=80's 104ms.
#: That is the difference between ~1Hz and ~10Hz, and at ~1Hz a match end
#: whose marker is on screen for under a second cannot be confirmed at all.
#:
#: q=80 was checked against every committed corpus (TOKON, Avatar, SF6) by
#: re-encoding each frame and re-running its detector: no frame changed its
#: reported screen, winner or counts. See the 2026-08-27 TOKON report.
_DEFAULT_FORMAT = "jpg"
_DEFAULT_QUALITY = 80

#: Successful captures to observe before reporting the rate actually achieved.
_RATE_REPORT_AFTER = 20


class ObsFrameSource:
    def __init__(
        self,
        client_factory: Callable[[], object],
        source_name: str,
        canonical: tuple[int, int],
        poll_hz: float = 5.0,
        request_size: tuple[int, int] = (1280, 720),
        sleeper: Callable[[float], None] = time.sleep,
        image_format: str = _DEFAULT_FORMAT,
        image_quality: int = _DEFAULT_QUALITY,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if poll_hz <= 0:
            raise ValueError(f"poll_hz must be > 0, got {poll_hz}")
        self._client_factory = client_factory
        self._source_name = source_name
        self._canonical = canonical
        self._poll_hz = poll_hz
        self._interval = 1.0 / poll_hz
        self._request_size = request_size
        self._sleep = sleeper
        self._image_format = image_format
        self._image_quality = image_quality
        self._clock = clock
        self._captures = 0
        self._first_capture_at: float | None = None
        self._client: object | None = None
        self._connected = False
        self._backoff = 0.5
        self._stopped = False

    @property
    def connected(self) -> bool:
        return self._connected

    def _ensure_client(self) -> object:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def _attempt_once(self) -> Frame | None:
        """One capture attempt. Returns None on any failure, never raises."""
        try:
            client = self._ensure_client()
            response = client.get_source_screenshot(
                self._source_name,
                self._image_format,
                *self._request_size,
                self._image_quality,
            )
            image = self._decode(response.image_data)
        except Exception as exc:  # obsws raises a wide variety; none are fatal here
            log.warning("OBS capture failed: %s", exc)
            self._connected = False
            stale_client = self._client
            self._client = None
            if stale_client is not None:
                try:
                    stale_client.disconnect()
                except Exception as disconnect_exc:
                    log.debug("error disconnecting stale OBS client: %s", disconnect_exc)
            return None

        self._connected = True
        self._backoff = 0.5
        if image is None:
            return None
        normalized = normalize(image, self._canonical)
        if normalized is None:
            log.warning("OBS returned a wrong-aspect image: %s", image.shape)
            return None
        return Frame(image=normalized, captured_at=datetime.now(timezone.utc))

    @staticmethod
    def _decode(image_data: str) -> np.ndarray | None:
        _, _, payload = image_data.partition(_DATA_URI_PREFIX)
        if not payload:
            log.warning("unexpected screenshot payload format")
            return None
        raw = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)

    def stop(self) -> None:
        """Signal frames() to stop iterating after its current attempt.

        Cooperative: the generator only checks this between attempts, so it
        won't interrupt a capture or sleep already in progress. That's the
        best a synchronous, uninterruptible loop can offer, but it's enough
        to let the loop actually end instead of running forever.
        """
        self._stopped = True

    def _pace(self, cycle_started: float) -> None:
        """Sleep only the part of the poll interval not already spent.

        Sleeping the whole interval *after* a capture makes the achieved
        period `capture_latency + 1/poll_hz`, so poll_hz silently becomes an
        upper bound the source never reaches. Pacing against the start of the
        cycle keeps the requested rate whenever the work fits inside it, and
        degrades to "as fast as possible" when it does not.
        """
        remaining = self._interval - (self._clock() - cycle_started)
        if remaining > 0:
            self._sleep(remaining)

    def _record_capture(self, at: float) -> None:
        """Report the rate actually achieved, once, after enough samples.

        A source that cannot keep up is otherwise invisible: every frame looks
        fine individually, and only a missed event reveals the shortfall.
        """
        self._captures += 1
        if self._first_capture_at is None:
            self._first_capture_at = at
            return
        if self._captures != _RATE_REPORT_AFTER:
            return
        elapsed = at - self._first_capture_at
        if elapsed <= 0:
            return
        achieved = (self._captures - 1) / elapsed
        if achieved < self._poll_hz * 0.5:
            log.warning(
                "OBS capture is achieving %.1f Hz against a configured poll_hz "
                "of %.1f; a match end whose marker is brief may be missed. Try "
                "a smaller obs request size, or check what else is loading OBS.",
                achieved,
                self._poll_hz,
            )
        else:
            log.info(
                "OBS capture achieving %.1f Hz (poll_hz=%.1f)",
                achieved,
                self._poll_hz,
            )

    def frames(self) -> Iterator[Frame]:
        while not self._stopped:
            cycle_started = self._clock()
            frame = self._attempt_once()
            if frame is not None:
                self._record_capture(self._clock())
                yield frame
                self._pace(cycle_started)
                continue
            if self._connected:
                # Connected but this frame was unusable — keep the normal cadence.
                self._pace(cycle_started)
            else:
                self._sleep(self._backoff)
                self._backoff = min(self._backoff * 2, _MAX_BACKOFF_SECONDS)

    def close(self) -> None:
        self.stop()
        client = self._client
        self._client = None
        self._connected = False
        if client is not None:
            try:
                client.disconnect()
            except Exception as exc:
                log.debug("error disconnecting OBS client: %s", exc)


def default_client_factory(host: str, port: int, password: str) -> Callable[[], object]:
    """Build a factory that opens a fresh obsws ReqClient on demand."""

    def factory() -> object:
        import obsws_python

        return obsws_python.ReqClient(host=host, port=port, password=password)

    return factory
