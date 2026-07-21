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


class ObsFrameSource:
    def __init__(
        self,
        client_factory: Callable[[], object],
        source_name: str,
        canonical: tuple[int, int],
        poll_hz: float = 5.0,
        request_size: tuple[int, int] = (1280, 720),
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if poll_hz <= 0:
            raise ValueError(f"poll_hz must be > 0, got {poll_hz}")
        self._client_factory = client_factory
        self._source_name = source_name
        self._canonical = canonical
        self._interval = 1.0 / poll_hz
        self._request_size = request_size
        self._sleep = sleeper
        self._client: object | None = None
        self._connected = False
        self._backoff = 0.5

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
                self._source_name, "png", *self._request_size, -1
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

    def frames(self) -> Iterator[Frame]:
        while True:
            frame = self._attempt_once()
            if frame is not None:
                yield frame
                self._sleep(self._interval)
                continue
            if self._connected:
                # Connected but this frame was unusable — keep the normal cadence.
                self._sleep(self._interval)
            else:
                self._sleep(self._backoff)
                self._backoff = min(self._backoff * 2, _MAX_BACKOFF_SECONDS)

    def close(self) -> None:
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
