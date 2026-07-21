import numpy as np
import pytest

from fgc_detector.detectors.roi import Roi, fill_ratio, match_template, mean_color


def _black(width: int = 100, height: int = 100) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_crop_returns_the_requested_rectangle():
    image = _black()
    image[10:20, 30:50] = 255
    cropped = Roi(30, 10, 20, 10).crop(image)
    assert cropped.shape == (10, 20, 3)
    assert cropped.min() == 255


def test_roi_rejects_non_positive_size():
    with pytest.raises(ValueError):
        Roi(0, 0, 0, 10)


def test_fill_ratio_all_dark_is_zero():
    assert fill_ratio(_black(), Roi(0, 0, 10, 10)) == 0.0


def test_fill_ratio_all_bright_is_one():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert fill_ratio(image, Roi(0, 0, 10, 10)) == 1.0


def test_fill_ratio_half_bright_is_half():
    image = _black()
    image[0:5, 0:10] = 255
    assert fill_ratio(image, Roi(0, 0, 10, 10)) == pytest.approx(0.5)


def test_fill_ratio_respects_threshold():
    image = np.full((100, 100, 3), 100, dtype=np.uint8)
    assert fill_ratio(image, Roi(0, 0, 10, 10), threshold=50) == 1.0
    assert fill_ratio(image, Roi(0, 0, 10, 10), threshold=150) == 0.0


def test_roi_outside_image_bounds_returns_zero_not_a_crash():
    # A resolution change can push an ROI off the frame. Degrade to "saw
    # nothing" rather than raising in the middle of a live match.
    assert fill_ratio(_black(50, 50), Roi(40, 40, 100, 100)) == 0.0


def test_mean_color_is_bgr_order():
    image = _black()
    image[0:10, 0:10] = (255, 0, 0)  # pure blue in BGR
    blue, green, red = mean_color(image, Roi(0, 0, 10, 10))
    assert (blue, green, red) == pytest.approx((255.0, 0.0, 0.0))


def test_match_template_identical_region_scores_one():
    image = _black()
    image[10:30, 10:30] = 200
    template = image[10:30, 10:30].copy()
    assert match_template(image, Roi(10, 10, 20, 20), template) == pytest.approx(1.0, abs=1e-3)


def test_match_template_mismatched_region_scores_low():
    image = _black()
    template = np.full((20, 20, 3), 255, dtype=np.uint8)
    template[0:10] = 0  # give the template variance so correlation is defined
    assert match_template(image, Roi(10, 10, 20, 20), template) < 0.5


def test_match_template_size_mismatch_raises():
    with pytest.raises(ValueError):
        match_template(_black(), Roi(0, 0, 20, 20), np.zeros((5, 5, 3), dtype=np.uint8))
