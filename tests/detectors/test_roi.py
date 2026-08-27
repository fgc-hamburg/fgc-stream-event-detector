import cv2
import numpy as np
import pytest

from fgc_detector.detectors.roi import (
    Roi,
    fill_ratio,
    match_template,
    pale_fill_ratio,
    region_difference,
)


def _black(width: int = 100, height: int = 100) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_crop_returns_the_requested_rectangle():
    image = _black()
    image[10:20, 30:50] = 255
    cropped = Roi(30, 10, 20, 10).crop(image)
    assert cropped.shape == (10, 20, 3)
    assert cropped.min() == 255


def test_crop_at_exact_right_edge_returns_full_rectangle():
    # x + w == width is in-bounds: the strict `>` check must accept it. If
    # the bounds check ever flips to `>=`, this must fail.
    image = _black(width=100, height=100)
    image[10:20, 80:100] = 255
    cropped = Roi(80, 10, 20, 10).crop(image)
    assert cropped.shape == (10, 20, 3)
    assert cropped.min() == 255


def test_crop_at_exact_bottom_edge_returns_full_rectangle():
    # y + h == height is in-bounds, mirroring the right-edge case above.
    image = _black(width=100, height=100)
    image[80:100, 10:20] = 255
    cropped = Roi(10, 80, 10, 20).crop(image)
    assert cropped.shape == (20, 10, 3)
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


def _solid_hsv(h: int, s: int, v: int, size: int = 20) -> np.ndarray:
    """A `size`x`size` BGR image of one HSV colour."""
    hsv = np.full((size, size, 3), (h, s, v), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_pale_fill_ratio_counts_white_pixels() -> None:
    """White is bright and colourless: saturation under the ceiling, value over
    the floor. This is the marker `color_fill_ratio` structurally cannot find."""
    image = _solid_hsv(0, 0, 255)

    assert pale_fill_ratio(image, Roi(0, 0, 20, 20), sat_max=60, val_min=150) == 1.0


def test_pale_fill_ratio_rejects_a_vivid_colour() -> None:
    """A saturated icon fails the saturation ceiling however bright it is."""
    image = _solid_hsv(15, 200, 255)

    assert pale_fill_ratio(image, Roi(0, 0, 20, 20), sat_max=60, val_min=150) == 0.0


def test_pale_fill_ratio_rejects_a_dark_grey() -> None:
    """Colourless but dim: passes the saturation ceiling, fails the value floor."""
    image = _solid_hsv(0, 0, 40)

    assert pale_fill_ratio(image, Roi(0, 0, 20, 20), sat_max=60, val_min=150) == 0.0


def test_pale_fill_ratio_is_a_fraction_not_a_flag() -> None:
    """Half white, half vivid reads 0.5 -- the caller thresholds it."""
    image = _solid_hsv(15, 200, 255)
    image[:10, :] = _solid_hsv(0, 0, 255, size=20)[:10, :]

    assert pale_fill_ratio(image, Roi(0, 0, 20, 20), sat_max=60, val_min=150) == 0.5


def test_pale_fill_ratio_degrades_on_an_out_of_frame_roi() -> None:
    """Like every primitive here: a neutral reading, never a crash mid-match."""
    image = _solid_hsv(0, 0, 255)

    assert pale_fill_ratio(image, Roi(15, 15, 20, 20), sat_max=60, val_min=150) == 0.0


def test_region_difference_is_zero_for_identical_regions() -> None:
    """Two patches of the same colour are indistinguishable."""
    image = _solid_hsv(100, 180, 200, size=40)

    assert region_difference(image, Roi(0, 0, 10, 10), Roi(20, 20, 10, 10)) == 0.0


def test_region_difference_is_one_for_black_versus_white() -> None:
    """The scale is normalized so 1.0 is maximally different."""
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:20, :] = 255

    assert region_difference(image, Roi(0, 0, 10, 10), Roi(0, 25, 10, 10)) == 1.0


def test_region_difference_ignores_a_tint_applied_to_both_regions() -> None:
    """The point of the primitive: a stage that tints the whole HUD equally
    cancels out, so the reading survives a background an absolute threshold
    would be fooled by."""
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:20, :] = (0, 0, 200)   # region A: red-ish
    image[25:, :] = (0, 0, 100)   # region B: darker red
    plain = region_difference(image, Roi(0, 0, 10, 10), Roi(0, 25, 10, 10))

    tinted = image.astype(np.int16) + np.array([40, 40, 40], dtype=np.int16)
    tinted = np.clip(tinted, 0, 255).astype(np.uint8)

    assert region_difference(tinted, Roi(0, 0, 10, 10), Roi(0, 25, 10, 10)) == plain


def test_region_difference_degrades_on_an_out_of_frame_roi() -> None:
    image = _solid_hsv(0, 0, 255, size=20)

    assert region_difference(image, Roi(0, 0, 10, 10), Roi(15, 15, 20, 20)) == 0.0
