#!/usr/bin/env python3
"""Extend the Marvel TOKON golden corpus with native 1920x1080 frames.

The original corpus (scripts/build_tokon_corpus.py) was cut from a 1280x714
crop of a VOD. That single source is what let the first calibration ship a lit
test that was blind on real full-resolution capture: every constant sat inside
one recording's framing and tone curve. These frames come from a second,
independent recording made through OBS at canonical resolution, so the corpus
now pins the detector against two unrelated captures.

Labels and file naming match the original builder exactly, so
tests/detectors/test_tokon_corpus.py picks these up with no change:

  in_match_p1-<n>_p2-<n>   clean HUD, both counts hand-verified
  sprite_p1-<n>_p2-<n>     a character sprite covers a slot; unreadable frame
                           kept as adversarial ground truth (strict xfail)
  between                  no HUD at all (KO flash, cinematic, results screen)

Indices start at 100 to stay clear of the original corpus's numbering.

Ground truth was read off rendered HUD strips by eye (see the 2026-08-27 TOKON
recalibration report); regenerate with:
    python scripts/build_tokon_native_corpus.py ~/repos/tokon/tokon-3.mp4
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

from fgc_detector.frames.normalize import normalize

CANONICAL = (1920, 1080)
OUT_DIR = Path(__file__).resolve().parent.parent / "samples" / "tokon"

# (second, label_prefix, p1_pips, p2_pips). Every scored entry was verified by
# eye against a zoomed HUD strip: an empty pip is a small pale circle, a lit
# one a gold disc bearing a white P/V badge.
#
# The clip runs 917.7s at 60fps and contains four match ends. Three are
# observable and are the corpus's MATCH_END frames -- P2 wins all three, at
# 145.9s (0-3), 675.1s (2-3) and 914.0s (1-3). The fourth, at ~348.7s, is
# deliberately absent: the HUD cuts from 2-1 straight to the results screen
# and the winning third pip is never drawn, so there is nothing to label.
FRAMES = [
    # --- fresh matches: 0-0, four different stages ------------------------
    *[(s, "in_match", 0, 0) for s in (300, 400, 700)],
    # --- one side ahead ---------------------------------------------------
    *[(s, "in_match", 1, 0) for s in (200, 470)],
    (60, "in_match", 0, 2),
    (780, "in_match", 0, 1),
    (860, "in_match", 1, 2),
    (540, "in_match", 1, 1),
    # --- 2-1, the state either side of the sprite frame below -------------
    *[(s, "in_match", 2, 1) for s in (348.65, 575.5, 580, 600)],
    # --- the three observable match ends, all P2 --------------------------
    (145.9, "in_match", 0, 3),
    (675.1, "in_match", 2, 3),
    (914.0, "in_match", 1, 3),
    # --- adversarial: an orange sprite covers P1's slots while the HUD is
    # --- up. The true state is 2-1, confirmed at 575.5s and 580.0s either
    # --- side of it. The retired mean-difference lit test read this as a
    # --- sustained 3-1 and would have named a false P1 winner.
    (576.9, "sprite", 2, 1),
    # --- no HUD at all: super flashes, KO cinematics, results screens. Each
    # --- of these is a moment where "the white circle is gone" is true of
    # --- every slot, i.e. the fail-open case the lit test exists to refuse.
    *[
        (s, "between", None, None)
        for s in (104.6, 132.3, 321.4, 347.0, 386.9, 436.3, 439.1, 672.7, 688.5, 887.4)
    ],
]


def main(video: str) -> int:
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        print(f"could not open video: {video}", file=sys.stderr)
        return 2
    fps = capture.get(cv2.CAP_PROP_FPS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    for index, (second, prefix, p1, p2) in enumerate(FRAMES, start=100):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
        ok, image = capture.read()
        if not ok:
            print(f"could not read frame at {second}s", file=sys.stderr)
            continue
        frame = normalize(image, CANONICAL)
        if frame is None:
            print(f"wrong-aspect frame at {second}s: {image.shape}", file=sys.stderr)
            continue
        label = prefix if p1 is None else f"{prefix}_p1-{p1}_p2-{p2}"
        path = OUT_DIR / f"{label}_{index:04d}.png"
        cv2.imwrite(str(path), frame)
        written += 1
    capture.release()
    print(f"wrote {written} frames to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
