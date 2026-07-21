from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from fgc_detector.frames.offline import FolderFrameSource, VideoFrameSource

CANONICAL = (1920, 1080)


@pytest.fixture
def png_dir(tmp_path):
    for index, value in enumerate([10, 20, 30]):
        image = np.full((720, 1280, 3), value, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"frame_{index:04d}.png"), image)
    return tmp_path


def test_folder_source_yields_frames_in_filename_order(png_dir):
    source = FolderFrameSource(png_dir, CANONICAL)
    frames = list(source.frames())
    assert len(frames) == 3
    assert [int(frame.image[0, 0, 0]) for frame in frames] == [10, 20, 30]


def test_folder_source_normalizes_to_canonical(png_dir):
    frames = list(FolderFrameSource(png_dir, CANONICAL).frames())
    assert all(frame.image.shape == (1080, 1920, 3) for frame in frames)


def test_folder_source_frames_are_timezone_aware(png_dir):
    frame = next(iter(FolderFrameSource(png_dir, CANONICAL).frames()))
    assert frame.captured_at.tzinfo is not None


def test_folder_source_skips_unreadable_and_wrong_aspect_files(tmp_path):
    cv2.imwrite(str(tmp_path / "a_good.png"), np.full((720, 1280, 3), 5, dtype=np.uint8))
    cv2.imwrite(str(tmp_path / "b_bad_aspect.png"), np.full((768, 1024, 3), 5, dtype=np.uint8))
    (tmp_path / "c_not_an_image.png").write_text("garbage")
    frames = list(FolderFrameSource(tmp_path, CANONICAL).frames())
    assert len(frames) == 1


def test_folder_source_on_empty_directory_yields_nothing(tmp_path):
    assert list(FolderFrameSource(tmp_path, CANONICAL).frames()) == []


def test_video_source_yields_every_nth_frame(tmp_path):
    path = tmp_path / "clip.mp4"
    # Use MJPG codec which is available on all platforms; mp4v not available in standard OpenCV wheels
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (1280, 720))
    for index in range(10):
        writer.write(np.full((720, 1280, 3), index * 10, dtype=np.uint8))
    writer.release()

    frames = list(VideoFrameSource(path, CANONICAL, sample_every=5).frames())
    assert len(frames) == 2


def test_video_source_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(VideoFrameSource(tmp_path / "nope.mp4", CANONICAL).frames())


def test_video_source_skips_wrong_aspect_frames_with_warning(tmp_path, caplog):
    path = tmp_path / "wrong_aspect.mp4"
    # Create a video with 1024x768 aspect (4:3) instead of canonical 1920x1080 (16:9)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (1024, 768))
    for index in range(3):
        writer.write(np.full((768, 1024, 3), index * 10, dtype=np.uint8))
    writer.release()

    frames = list(VideoFrameSource(path, CANONICAL).frames())
    assert len(frames) == 0
    assert "skipping wrong-aspect video frame" in caplog.text
    assert str(path) in caplog.text


def _write_video(path, fps=30.0, count=10, size=(1280, 720)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for index in range(count):
        writer.write(np.full((size[1], size[0], 3), index * 10, dtype=np.uint8))
    writer.release()


def test_video_source_timestamps_advance_by_one_over_fps(tmp_path):
    path = tmp_path / "clip.mp4"
    _write_video(path, fps=30.0, count=5)

    frames = list(VideoFrameSource(path, CANONICAL).frames())
    assert len(frames) == 5
    deltas = [
        (b.captured_at - a.captured_at).total_seconds()
        for a, b in zip(frames, frames[1:])
    ]
    for delta in deltas:
        assert delta == pytest.approx(1.0 / 30.0, abs=1e-6)


def test_video_source_sample_every_reflects_true_video_position(tmp_path):
    path = tmp_path / "clip.mp4"
    _write_video(path, fps=30.0, count=10)

    frames = list(VideoFrameSource(path, CANONICAL, sample_every=5).frames())
    assert len(frames) == 2
    delta = (frames[1].captured_at - frames[0].captured_at).total_seconds()
    assert delta == pytest.approx(5 / 30.0, abs=1e-6)


def test_video_source_default_start_time_is_unix_epoch(tmp_path):
    path = tmp_path / "clip.mp4"
    _write_video(path, fps=30.0, count=3)

    frames = list(VideoFrameSource(path, CANONICAL).frames())
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    # frame 0 is exactly at the epoch (offset 0 seconds into the video)
    assert frames[0].captured_at == epoch
    # frame index 1's offset into the video is 1/fps seconds
    assert frames[1].captured_at == epoch + timedelta(seconds=1.0 / 30.0)


def test_video_source_explicit_start_time_is_honoured(tmp_path):
    path = tmp_path / "clip.mp4"
    _write_video(path, fps=30.0, count=3)

    start_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    frames = list(VideoFrameSource(path, CANONICAL, start_time=start_time).frames())
    assert frames[0].captured_at == start_time
    assert frames[1].captured_at == start_time + timedelta(seconds=1.0 / 30.0)


def test_video_source_naive_start_time_raises():
    with pytest.raises(ValueError):
        VideoFrameSource(Path("unused.mp4"), CANONICAL, start_time=datetime(2026, 1, 1))


def test_video_source_bogus_fps_falls_back_to_default_with_warning(tmp_path, caplog):
    path = tmp_path / "clip.mp4"
    _write_video(path, fps=30.0, count=2)

    source = VideoFrameSource(path, CANONICAL)
    real_capture_cls = cv2.VideoCapture

    class StubCapture:
        def __init__(self, filename):
            self._inner = real_capture_cls(filename)

        def isOpened(self):
            return self._inner.isOpened()

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 0.0
            return self._inner.get(prop)

        def read(self):
            return self._inner.read()

        def release(self):
            return self._inner.release()

    original_video_capture = cv2.VideoCapture
    cv2.VideoCapture = lambda filename: StubCapture(filename)
    try:
        with caplog.at_level("WARNING"):
            frames = list(source.frames())
    finally:
        cv2.VideoCapture = original_video_capture

    assert len(frames) == 2
    assert frames[1].captured_at - frames[0].captured_at == timedelta(seconds=1.0 / 30.0)
    assert "invalid fps" in caplog.text
