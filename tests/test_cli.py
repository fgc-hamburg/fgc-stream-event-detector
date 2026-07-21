import pytest

from fgc_detector.cli import main


def test_no_subcommand_prints_usage_and_fails():
    with pytest.raises(SystemExit):
        main([])


def test_unknown_subcommand_fails():
    with pytest.raises(SystemExit):
        main(["frobnicate"])


def test_replay_on_a_missing_video_returns_nonzero(tmp_path, capsys):
    code = main(["replay", "--video", str(tmp_path / "absent.mp4"), "--game", "sf6"])
    assert code != 0


def test_roi_on_a_missing_sample_returns_nonzero(tmp_path):
    code = main(["roi", "--game", "sf6", "--sample", str(tmp_path / "absent.png")])
    assert code != 0
