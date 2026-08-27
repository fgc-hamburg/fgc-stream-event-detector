#!/usr/bin/env python3
"""Extract a labelled Marvel TOKON golden corpus from a clean game-capture VOD.

Frames are named `<label>_<idx>.png` and written to samples/tokon/, where
`<label>` is one of:

  in_match_p1-<n>_p2-<n>   clean HUD, both sides' round-pip counts hand-verified
  occluded_p1-<n>_p2-<n>   HUD present but degraded (flash wash-out, an effect
                           over a pip) and still readable -- robustness cases
  sprite_p1-<n>_p2-<n>     a character sprite or a pip-slide animation fully
                           covers one empty pip slot, so the frame is genuinely
                           unreadable; kept as adversarial ground truth
  between                  no HUD at all (KO cinematic, cut-scene, results)

Ground-truth labels come from a hand-verified clip (see the 2026-08-26 TOKON
calibration report); regenerate with:
    python scripts/build_tokon_corpus.py <clean_tokon.mp4>
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

from fgc_detector.frames.normalize import normalize

# (second, label_prefix, p1_pips, p2_pips) -- every entry was read off a
# rendered HUD strip by eye against ~/repos/tokon/TOKON.mp4 (1280x714,
# 29.97fps, ~683s). The clip contains five matches; P1 wins three of them
# (111.1s, 470.1s, 667.3s) and P2 two (288.1s, 582.6s).
FRAMES = [
    # --- 0-0, five different stages plus the start of the sixth match -----
    *[(s, "in_match", 0, 0) for s in (30, 130, 310, 500, 600, 677, 680)],
    # --- P1 ahead ---------------------------------------------------------
    *[(s, "in_match", 1, 0) for s in (60, 165, 620)],
    *[(s, "in_match", 2, 0) for s in (85, 645)],
    # --- P2 ahead ---------------------------------------------------------
    *[(s, "in_match", 0, 1) for s in (328, 520)],
    *[(s, "in_match", 0, 2) for s in (370, 555)],
    # --- split scores -----------------------------------------------------
    (195, "in_match", 1, 1),
    (230, "in_match", 2, 1),
    *[(s, "in_match", 2, 2) for s in (258, 445)],
    # --- match ends: all three pips lit on the winning side ---------------
    # The HUD only comes back for ~1s after the KO cinematic, so these are
    # the whole visible window of each win (see report section 5).
    *[(s, "in_match", 3, 0) for s in (111.3, 111.7, 667.5, 667.9)],
    *[(s, "in_match", 3, 2) for s in (470.4, 471.2)],
    *[(s, "in_match", 0, 3) for s in (583.0, 583.6)],
    *[(s, "in_match", 2, 3) for s in (288.3, 288.8)],
    # --- degraded but readable -------------------------------------------
    (45.6, "occluded", 1, 0),  # whole HUD washed out by a super flash
    (313.0, "occluded", 0, 0),  # HUD faded to near-transparent on a transition
    (405.0, "occluded", 1, 2),  # a move effect sits over P1's lit inner pip
    # --- unreadable: a sprite covers an *empty* slot ----------------------
    # These are the fail-safe hazard for a detector that reads "the white
    # circle is gone" as lit; they are expected over-reads (report section 6).
    *[(s, "sprite", 2, 0) for s in (101.51, 107.52)],
    *[(s, "sprite", 2, 2) for s in (461.91, 466.51)],
    # --- no HUD: KO cinematics, cut-scenes, results and title cards -------
    *[
        (s, "between", None, None)
        for s in (
            5, 15, 73.5, 95.2, 118, 222, 249, 295, 338.5, 391.5,
            423.5, 452, 461, 468, 474, 508.5, 548.5, 634, 668.8, 672,
        )
    ],
]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: build_tokon_corpus.py <clean_tokon.mp4>", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    out = Path(__file__).parent.parent / "samples" / "tokon"
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    counts: dict[str, int] = {}
    for sec, state, p1, p2 in FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(sec * fps)))
        ok, img = cap.read()
        if not ok:
            print("read fail", sec)
            continue
        f = normalize(img, (1920, 1080))
        if f is None:
            print("normalize rejected", sec, img.shape)
            continue
        label = state if p1 is None else f"{state}_p1-{p1}_p2-{p2}"
        counts[label] = counts.get(label, 0) + 1
        cv2.imwrite(str(out / f"{label}_{counts[label]:04d}.png"), f)
    print("wrote", sum(counts.values()), "frames:", dict(sorted(counts.items())))
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
