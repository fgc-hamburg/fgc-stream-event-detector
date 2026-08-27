import base64

import cv2
import numpy as np
import pytest

from fgc_detector.frames.obs import _MAX_BACKOFF_SECONDS, ObsFrameSource

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
        # `responses` is a shared, mutable list: when a failure causes
        # ObsFrameSource to discard this client and build a new one via the
        # factory, the replacement client must continue consuming the same
        # underlying queue rather than starting over.
        self._responses = responses
        self.calls = []
        self.disconnected = False

    def get_source_screenshot(self, name, img_format, width, height, quality):
        self.calls.append((name, img_format, width, height, quality))
        if not self._responses:
            raise StopIteration
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return FakeResponse(response)

    def disconnect(self):
        self.disconnected = True


class RecordingClientFactory:
    """A client_factory that builds a fresh FakeClient each call and records
    every client it has produced, so tests can observe reconnection directly
    (invocation count, and which discarded client got disconnect()ed)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.clients: list[FakeClient] = []

    def __call__(self) -> FakeClient:
        client = FakeClient(self._responses)
        self.clients.append(client)
        return client

    @property
    def call_count(self) -> int:
        return len(self.clients)


def _source(responses, sleeper=None, **kwargs):
    factory = RecordingClientFactory(responses)
    source = ObsFrameSource(
        client_factory=factory,
        source_name="Game Capture",
        canonical=CANONICAL,
        sleeper=sleeper if sleeper is not None else (lambda _seconds: None),
        **kwargs,
    )
    return source, factory


def test_decodes_screenshot_into_normalized_frame():
    source, _ = _source([_data_uri()])
    frame = next(source.frames())
    assert frame.image.shape == (1080, 1920, 3)
    assert frame.captured_at.tzinfo is not None


def test_requests_the_configured_source_by_name():
    source, factory = _source([_data_uri()])
    next(source.frames())
    assert factory.clients[0].calls[0][0] == "Game Capture"


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
    source, factory = _source([_data_uri()])
    next(source.frames())
    source.close()
    assert factory.clients[0].disconnected is True


def test_failed_attempt_rebuilds_client_via_factory():
    """After a failed attempt, the next attempt must build a fresh client
    through client_factory rather than silently reusing the dead one."""
    source, factory = _source([ConnectionError("boom"), _data_uri()])
    frames = source.frames()
    next(frames)
    assert factory.call_count == 2


def test_failed_attempt_disconnects_the_discarded_client():
    """The client abandoned after a failed attempt must be disconnect()ed,
    not merely dropped — otherwise its socket/thread leaks."""
    source, factory = _source([ConnectionError("boom"), _data_uri()])
    frames = source.frames()
    next(frames)
    assert len(factory.clients) == 2
    stale_client, fresh_client = factory.clients
    assert stale_client.disconnected is True
    assert fresh_client.disconnected is False


def test_backoff_grows_and_is_capped_across_consecutive_failures():
    # frames() only yields control back to the caller on a *successful*
    # capture — a run of failures is retried internally within a single
    # next(frames) call. So seven failures followed by one success, driven
    # by a single next(), lets us observe every backoff sleep in sequence.
    sleeps: list[float] = []
    source, _ = _source(
        [ConnectionError("boom")] * 7 + [_data_uri()],
        sleeper=lambda seconds: sleeps.append(seconds),
    )
    frame = next(source.frames())
    assert frame.image.shape == (1080, 1920, 3)

    assert sleeps == [0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 10.0]
    assert sleeps[-1] == _MAX_BACKOFF_SECONDS


def test_backoff_resets_after_a_successful_capture():
    sleeps: list[float] = []
    source, _ = _source(
        [
            ConnectionError("boom"),
            ConnectionError("boom"),
            _data_uri(),
            ConnectionError("boom"),
            _data_uri(),
        ],
        sleeper=lambda seconds: sleeps.append(seconds),
    )
    frames = source.frames()
    next(frames)  # succeeds on the third attempt, after two failures growing backoff to 2.0
    assert sleeps == [0.5, 1.0]

    sleeps.clear()
    next(frames)  # normal-cadence sleep, then one failure, then a fresh success
    # sleeps[0] is the post-success normal-cadence sleep (poll interval).
    # sleeps[1] is the backoff used for the failure right after that success:
    # if backoff had not reset, it would continue growing from 2.0 instead.
    assert sleeps[1] == 0.5


def test_stop_ends_the_frame_generator():
    source, _ = _source([_data_uri(), _data_uri(), _data_uri()])
    frames = source.frames()
    next(frames)
    source.stop()
    with pytest.raises(StopIteration):
        next(frames)


def test_close_also_stops_the_frame_generator():
    """Regression: close() used to only drop the client, which
    _ensure_client() would silently rebuild on the generator's next loop
    iteration — so close() never actually stopped frames()."""
    source, _ = _source([_data_uri(), _data_uri()])
    frames = source.frames()
    next(frames)
    source.close()
    with pytest.raises(StopIteration):
        next(frames)


def test_invalid_poll_rate_rejected():
    with pytest.raises(ValueError):
        ObsFrameSource(
            client_factory=lambda: FakeClient([]),
            source_name="x",
            canonical=CANONICAL,
            poll_hz=0,
        )


# --- capture format and pacing ---------------------------------------------


def test_screenshots_are_requested_as_jpeg_by_default():
    """PNG is what makes OBS slow, not the transfer.

    Lossless compression of a noisy game frame costs OBS ~1.1s per 1280x720
    frame while it is also decoding, against ~0.1s for JPEG. At ~1Hz a brief
    match-end marker can never accumulate the confirmer's agreeing frames, so
    the default has to be the cheap one.
    """
    source, factory = _source([_data_uri()])
    next(source.frames())
    _, img_format, _, _, quality = factory.clients[0].calls[0]
    assert img_format == "jpg"
    assert quality == 80


def test_capture_format_and_quality_are_overridable():
    """Calibration measures ROIs off these frames, so it needs them lossless."""
    source, factory = _source([_data_uri()], image_format="png", image_quality=-1)
    next(source.frames())
    _, img_format, _, _, quality = factory.clients[0].calls[0]
    assert img_format == "png"
    assert quality == -1


def test_pacing_sleeps_only_the_unused_remainder_of_the_interval():
    """A capture that takes half the interval must be followed by half a sleep.

    Sleeping the full interval after the capture would make the achieved
    period `latency + 1/poll_hz`, so poll_hz would be a rate the source can
    never actually reach.
    """
    sleeps: list[float] = []
    now = [0.0]

    def clock() -> float:
        return now[0]

    def capture_takes_40ms(_seconds: float) -> None:
        sleeps.append(_seconds)

    source, _ = _source(
        [_data_uri(), _data_uri()],
        sleeper=capture_takes_40ms,
        poll_hz=10.0,  # 0.1s interval
        clock=clock,
    )
    frames = source.frames()
    next(frames)
    now[0] += 0.04  # the capture and the consumer's work took 40ms
    next(frames)

    assert sleeps == [pytest.approx(0.06)]


def test_a_capture_slower_than_the_interval_does_not_sleep_at_all():
    """Falling behind must degrade to 'as fast as possible', never to a
    negative sleep or a backwards deadline."""
    sleeps: list[float] = []
    now = [0.0]

    source, _ = _source(
        [_data_uri(), _data_uri()],
        sleeper=lambda seconds: sleeps.append(seconds),
        poll_hz=10.0,
        clock=lambda: now[0],
    )
    frames = source.frames()
    next(frames)
    now[0] += 0.85  # OBS took far longer than the 0.1s interval
    next(frames)

    assert sleeps == []
