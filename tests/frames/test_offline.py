from datetime import datetime, timezone

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
