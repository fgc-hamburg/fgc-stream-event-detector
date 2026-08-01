"""Fixed-region sampling primitives.

Round-win markers are position-fixed and language-independent, which is why
fill-ratio sampling is preferred over OCR: no game-language requirement, and a
threshold is far cheaper than a text recognizer.

Every function degrades to a neutral reading rather than raising when an ROI
falls outside the frame. A crash mid-match is worse than a missed detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Roi:
    """A rectangle in canonical-resolution pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"ROI must have positive size, got {self.w}x{self.h}")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"ROI origin must be non-negative, got ({self.x}, {self.y})")

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Return the ROI's pixels, or an empty array if it falls outside `image`."""
        height, width = image.shape[:2]
        if self.x + self.w > width or self.y + self.h > height:
            return np.empty((0, 0, 3), dtype=image.dtype)
        return image[self.y : self.y + self.h, self.x : self.x + self.w]


def fill_ratio(image: np.ndarray, roi: Roi, threshold: int = 128) -> float:
    """Fraction of the ROI's pixels brighter than `threshold` after grayscaling.

    This is the workhorse for round-win markers: an unfilled marker is dark, a
    filled one is bright, and the ratio between them is a wide, stable gap.
    """
    patch = roi.crop(image)
    if patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(np.count_nonzero(gray > threshold) / gray.size)


def match_template(image: np.ndarray, roi: Roi, template: np.ndarray) -> float:
    """Normalized correlation of the ROI against `template`, in 0.0–1.0.

    The template must be exactly the ROI's size; this is a fixed-position match,
    not a search, because the HUD element's location is already known.
    """
    if template.shape[:2] != (roi.h, roi.w):
        raise ValueError(
            f"template is {template.shape[:2]}, ROI is {(roi.h, roi.w)}; they must match"
        )
    patch = roi.crop(image)
    if patch.size == 0:
        return 0.0
    score = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
    return float(max(0.0, score[0][0]))


def color_fill_ratio(
    image: np.ndarray,
    roi: Roi,
    *,
    hue_lo: int,
    hue_hi: int,
    sat_min: int,
    val_min: int,
) -> float:
    """Fraction of the ROI's pixels that are a vivid instance of one colour.

    A pixel counts when its HSV hue is in ``[hue_lo, hue_hi]`` (OpenCV's 0-179
    scale) AND its saturation >= ``sat_min`` AND its value >= ``val_min``. When
    ``hue_lo > hue_hi`` the band wraps around 0 (e.g. red: ``hue_lo=170,
    hue_hi=10`` matches both H~179 and H~0).

    The colour analogue of ``fill_ratio``: it reads a pip that fills with a
    saturated colour (red, blue) rather than with brightness, and rejects both
    a dark empty interior (fails ``val_min``/``sat_min``) and a bright outline
    of a different hue (fails the hue band). Degrades to 0.0 on an out-of-frame
    ROI, like every primitive here.
    """
    patch = roi.crop(image)
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    if hue_lo <= hue_hi:
        hue_mask = (hue >= hue_lo) & (hue <= hue_hi)
    else:  # wrap-around band around 0 (red)
        hue_mask = (hue >= hue_lo) | (hue <= hue_hi)
    mask = hue_mask & (sat >= sat_min) & (val >= val_min)
    return float(np.count_nonzero(mask) / mask.size)
