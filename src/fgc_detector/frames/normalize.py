"""Resolution normalization.

Detectors sample fixed pixel rectangles. Those rectangles are only meaningful at
one aspect ratio, so a frame whose aspect does not match is rejected outright
rather than squashed to fit — a squashed frame produces confident garbage, which
is worse than no reading at all.
"""

from __future__ import annotations

import cv2
import numpy as np


def normalize(
    image: np.ndarray,
    canonical: tuple[int, int],
    aspect_tolerance: float = 0.02,
) -> np.ndarray | None:
    """Scale `image` to `canonical` (width, height), or return None if it can't be.

    Returns None for empty images and for any aspect ratio differing from the
    canonical one by more than `aspect_tolerance` (relative).
    """
    if image.size == 0 or image.ndim != 3:
        return None

    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return None

    target_width, target_height = canonical
    if (width, height) == (target_width, target_height):
        return image

    target_aspect = target_width / target_height
    actual_aspect = width / height
    if abs(actual_aspect - target_aspect) / target_aspect > aspect_tolerance:
        return None

    # INTER_AREA for downscale preserves the flat colour regions that fill-ratio
    # sampling depends on; INTER_LINEAR is the right choice going up.
    interpolation = cv2.INTER_AREA if width > target_width else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)
