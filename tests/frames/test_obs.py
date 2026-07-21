import base64

import cv2
import numpy as np
import pytest

from fgc_detector.frames.obs import ObsFrameSource

CANONICAL = (1920, 1080)


def _data_uri(width: int = 1280, height: int = 720, value: int = 42) -> str:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode()


class FakeResponse:
    def __init__(self, image_data: str) -> None:
        self.image_data = image_data


class FakeClient:
    """Stands in for obsws_python.ReqClient."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.disconnected = False

    def get_source_screenshot(self, name, img_format, width, height, quality):
        self.calls.append((name, img_format, width, height))
        if not self._responses:
            raise StopIteration
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return FakeResponse(response)

    def disconnect(self):
        self.disconnected = True


def _source(responses, **kwargs):
    client = FakeClient(responses)
    source = ObsFrameSource(
        client_factory=lambda: client,
        source_name="Game Capture",
        canonical=CANONICAL,
        sleeper=lambda _seconds: None,
        **kwargs,
    )
    return source, client


def test_decodes_screenshot_into_normalized_frame():
    source, _ = _source([_data_uri()])
    frame = next(source.frames())
    assert frame.image.shape == (1080, 1920, 3)
    assert frame.captured_at.tzinfo is not None


def test_requests_the_configured_source_by_name():
    source, client = _source([_data_uri()])
    next(source.frames())
    assert client.calls[0][0] == "Game Capture"


def test_marks_connected_after_a_successful_capture():
    source, _ = _source([_data_uri()])
    next(source.frames())
    assert source.connected is True


def test_survives_a_transient_capture_error_and_recovers():
    source, _ = _source([ConnectionError("boom"), _data_uri()])
    frames = source.frames()
    frame = next(frames)
    assert frame.image.shape == (1080, 1920, 3)


def test_marks_disconnected_while_erroring():
    source, _ = _source([ConnectionError("boom"), _data_uri()])
    frames = source.frames()
    # Drive one failing attempt without consuming a frame.
    source._attempt_once()
    assert source.connected is False


def test_wrong_aspect_screenshot_yields_no_frame():
    source, _ = _source([_data_uri(width=1024, height=768), _data_uri()])
    frame = next(source.frames())
    assert frame.image.shape == (1080, 1920, 3)


def test_close_disconnects_the_client():
    source, client = _source([_data_uri()])
    next(source.frames())
    source.close()
    assert client.disconnected is True


def test_invalid_poll_rate_rejected():
    with pytest.raises(ValueError):
        ObsFrameSource(
            client_factory=lambda: FakeClient([]),
            source_name="x",
            canonical=CANONICAL,
            poll_hz=0,
        )
