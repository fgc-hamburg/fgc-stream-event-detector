"""Unit tests for the colour-aware fill primitive.

OpenCV BGR->HSV reference hues (0-179 scale): pure red (0,0,255)->H=0,
pure blue (255,0,0)->H=120, cyan (255,255,0)->H=90. These are the colours
Avatar's pips (red P1, blue P2) and empty outlines (cyan) actually use, so
the tests assert the primitive separates them.
"""
from __future__ import annotations

import numpy as np

from fgc_detector.detectors.roi import Roi, color_fill_ratio

# OpenCV hue bands (0-179). Red wraps around 0. Blue is a tight band that must
# exclude cyan (H~90).
RED = dict(hue_lo=170, hue_hi=10, sat_min=80, val_min=60)
BLUE = dict(hue_lo=105, hue_hi=135, sat_min=80, val_min=60)


def _solid(bgr: tuple[int, int, int], w: int = 20, h: int = 20) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


def test_red_patch_reads_full_under_red_band() -> None:
    img = _solid((0, 0, 255))  # BGR red
    assert color_fill_ratio(img, Roi(0, 0, 20, 20), **RED) > 0.99


def test_blue_patch_reads_zero_under_red_band() -> None:
    img = _solid((255, 0, 0))  # BGR blue
    assert color_fill_ratio(img, Roi(0, 0, 20, 20), **RED) == 0.0


def test_blue_patch_reads_full_under_blue_band() -> None:
    img = _solid((255, 0, 0))
    assert color_fill_ratio(img, Roi(0, 0, 20, 20), **BLUE) > 0.99


def test_cyan_patch_reads_zero_under_blue_band() -> None:
    img = _solid((255, 255, 0))  # BGR cyan, H~90 -- the empty-pip outline colour
    assert color_fill_ratio(img, Roi(0, 0, 20, 20), **BLUE) == 0.0


def test_dark_patch_reads_zero_even_if_hue_matches() -> None:
    img = _solid((40, 0, 0))  # dark blue-ish: low value, below val_min
    assert color_fill_ratio(img, Roi(0, 0, 20, 20), **BLUE) == 0.0


def test_out_of_frame_roi_degrades_to_zero() -> None:
    img = _solid((0, 0, 255))
    assert color_fill_ratio(img, Roi(30, 30, 20, 20), **RED) == 0.0
