"""SF6 games-won-in-set counter detector.

Street Fighter 6 shows each player's games-won-in-set as a single digit inside
a light parallelogram box beside their name: P1's digit sits left-of-centre,
P2's right-of-centre (the two boxes are mirror images of each other, not
translates of one another). This module reads those two digits from one
frame and reports what it sees; it never decides that a game ended -- that
temporal judgement belongs to a stateful confirmer, not this pure detector.

Recognition is glyph-normalised so the same reference masks read both boxes
despite their mirrored layout:

1. Locate the box: threshold the grayscale ROI at a fixed brightness (the
   box interior is a light gray/white panel, well above the background
   behind it) and take the largest resulting contour's bounding box.
2. Search a fixed central fraction of that box for the digit. The box is a
   slanted parallelogram, so its axis-aligned bounding rectangle always
   includes two background-coloured triangular corners; insetting misses
   them entirely without needing to know the box's true polygon.
3. Within that search window, take max(saturation, 255-gray) per pixel. The
   digit is reliably picked out by *one* of those two signals even when the
   other is washed out by motion blur or sits close to the box's own
   brightness -- across the corpus, no single channel alone was enough, but
   the pointwise max is a strict improvement over either.
4. Otsu-threshold that combined channel (computed fresh per frame -- a fixed
   threshold does not survive lighting variation across the corpus) and take
   the largest resulting contour as the digit's blob.
5. Crop to the digit's bounding box, resize to a fixed canonical mask, and
   `matchTemplate` (TM_CCOEFF_NORMED) against reference masks for 0-3.

See docs/superpowers/specs/2026-07-22-sf6-counter-detector.md for the design
this implements, and .superpowers/sdd/task-B-report.md for the calibration
log (why these constants, what the corpus showed, how the digit references
were produced -- including the synthetic "3", which no corpus frame or clip
frame shows).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..types import (
    DETAIL_P1_GAMES,
    DETAIL_P2_GAMES,
    EventType,
    Frame,
    Game,
    Observation,
    Screen,
)
from .registry import register
from .roi import Roi

#: Canonical resolution this detector's ROIs are expressed in.
CANONICAL_SIZE = (1920, 1080)

#: The two counter boxes. Not symmetric: P1's digit sits left-of-centre in
#: its box, P2's right-of-centre in its (mirrored) box -- these were located
#: by scanning the real corpus, not assumed from the design doc's estimate.
P1_ROI = Roi(605, 3, 100, 62)
P2_ROI = Roi(1220, 3, 100, 62)

#: Grayscale value above which a pixel is considered part of the box's light
#: interior panel (as opposed to the darker background around it).
BOX_BRIGHT_THRESHOLD = 158

#: Minimum plausible box size, in pixels, before we trust the contour found
#: by BOX_BRIGHT_THRESHOLD is really the counter box and not a stray bright
#: patch of background.
MIN_BOX_SIZE = (20, 10)  # (width, height)

#: Central fraction of the box's bounding rectangle searched for the digit.
#: The box is a slanted parallelogram, so its axis-aligned bbox always
#: includes two background-coloured triangular corners; this inset clears
#: them without needing the true polygon.
DIGIT_SEARCH_X = (0.22, 0.78)
DIGIT_SEARCH_Y = (0.10, 0.95)

#: Fixed size (width, height) the extracted digit glyph is resized to before
#: template matching. Arbitrary but must match the committed reference masks.
MASK_SIZE = (24, 36)

#: Mean grayscale of the ROI below which we assume no counter box is on
#: screen at all (between games, transitions) and skip digit extraction.
BOX_PRESENT_MEAN = 95.0

#: Matched-template score below which a digit read is untrustworthy. Real
#: corpus reads land at 0.54-1.00; the corpus's few genuine near-misses
#: (heavy motion blur) land at or below 0.47. 0.5 sits in that gap.
CONFIDENT_MATCH = 0.5

_REFS_DIR = Path(__file__).resolve().parents[3] / "samples" / "sf6" / "refs"
_DIGITS = ("0", "1", "2", "3")


def _load_reference_masks() -> dict[str, np.ndarray]:
    refs: dict[str, np.ndarray] = {}
    for digit in _DIGITS:
        path = _REFS_DIR / f"digit_{digit}.npy"
        refs[digit] = np.load(path)
    return refs


def _extract_digit_mask(roi_image: np.ndarray) -> np.ndarray | None:
    """Locate and normalise the digit glyph in one counter box's pixels.

    Returns a MASK_SIZE binary mask (digit = white), or None if no box /
    no plausible digit contour was found. Pure: same input, same output.
    """
    if roi_image.size == 0:
        return None
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    # Whichever signal shows the digit best, pixel by pixel: some frames read
    # cleanly on inverted brightness, others (heavy motion blur washing out
    # luminance contrast) only show it in saturation. See module docstring.
    combined = np.maximum(saturation, 255 - gray)

    _, bright = cv2.threshold(gray, BOX_BRIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    box = max(contours, key=cv2.contourArea)
    box_x, box_y, box_w, box_h = cv2.boundingRect(box)
    min_w, min_h = MIN_BOX_SIZE
    if box_w < min_w or box_h < min_h:
        return None

    x0, x1 = DIGIT_SEARCH_X
    y0, y1 = DIGIT_SEARCH_Y
    search_x = box_x + int(box_w * x0)
    search_w = max(1, int(box_w * (x1 - x0)))
    search_y = box_y + int(box_h * y0)
    search_h = max(1, int(box_h * (y1 - y0)))
    search_window = combined[search_y : search_y + search_h, search_x : search_x + search_w]
    if search_window.size < 10:
        return None

    otsu_threshold, _ = cv2.threshold(
        search_window.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    _, digit_mask = cv2.threshold(search_window, otsu_threshold, 255, cv2.THRESH_BINARY)
    digit_contours, _ = cv2.findContours(
        digit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not digit_contours:
        return None
    digit = max(digit_contours, key=cv2.contourArea)
    dx, dy, dw, dh = cv2.boundingRect(digit)
    if dw < 3 or dh < 5:
        return None
    glyph = digit_mask[dy : dy + dh, dx : dx + dw]
    return cv2.resize(glyph, MASK_SIZE, interpolation=cv2.INTER_AREA)


def _match_digit(
    mask: np.ndarray, references: dict[str, np.ndarray]
) -> tuple[int, float]:
    """Best-matching digit and its score against the reference masks."""
    query = mask.astype(np.float32)
    best_digit = "0"
    best_score = -1.0
    for digit, reference in references.items():
        score = float(
            cv2.matchTemplate(query, reference.astype(np.float32), cv2.TM_CCOEFF_NORMED)[
                0
            ][0]
        )
        if score > best_score:
            best_digit, best_score = digit, score
    return int(best_digit), best_score


class Sf6CounterDetector:
    """Reads SF6's per-side games-won-in-set digit. Stateless and pure."""

    canonical_size = CANONICAL_SIZE
    game = Game.SF6

    def __init__(self) -> None:
        self._references = _load_reference_masks()

    def rois(self) -> dict[str, Roi]:
        return {"p1_games": P1_ROI, "p2_games": P2_ROI}

    def supported_events(self) -> frozenset[EventType]:
        return frozenset({EventType.MATCH_END})

    def observe(self, frame: Frame) -> Observation:
        image = frame.image
        p1_patch = P1_ROI.crop(image)
        p2_patch = P2_ROI.crop(image)

        p1_mean = float(cv2.cvtColor(p1_patch, cv2.COLOR_BGR2GRAY).mean()) if p1_patch.size else 0.0
        p2_mean = float(cv2.cvtColor(p2_patch, cv2.COLOR_BGR2GRAY).mean()) if p2_patch.size else 0.0

        debug = {"p1_mean": p1_mean, "p2_mean": p2_mean}

        if p1_mean < BOX_PRESENT_MEAN or p2_mean < BOX_PRESENT_MEAN:
            return Observation(screen=Screen.UNKNOWN, debug=debug)

        p1_mask = _extract_digit_mask(p1_patch)
        p2_mask = _extract_digit_mask(p2_patch)
        if p1_mask is None or p2_mask is None:
            return Observation(screen=Screen.UNKNOWN, debug=debug)

        p1_digit, p1_score = _match_digit(p1_mask, self._references)
        p2_digit, p2_score = _match_digit(p2_mask, self._references)
        debug = {**debug, "p1_score": p1_score, "p2_score": p2_score}

        if p1_score < CONFIDENT_MATCH or p2_score < CONFIDENT_MATCH:
            return Observation(screen=Screen.UNKNOWN, debug=debug)

        return Observation(
            screen=Screen.IN_MATCH,
            winner=None,
            details={
                DETAIL_P1_GAMES: str(p1_digit),
                DETAIL_P2_GAMES: str(p2_digit),
            },
            confidence=min(p1_score, p2_score),
            debug=debug,
        )


register(Sf6CounterDetector())
