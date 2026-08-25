#!/usr/bin/env python3
"""Extract a labelled Avatar Legends golden corpus from a clean game-capture VOD.

Frames are named `<state>[_p1-<n>_p2-<n>]_<idx>.png` and written to samples/avatar/.
Ground-truth labels come from a hand-verified clip (see the 2026-07-30 calibration
report); regenerate with:  python scripts/build_avatar_corpus.py <clean_avatar.mp4>
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

from fgc_detector.frames.normalize import normalize

# (second, state, p1_pips, p2_pips) -- stable moments, ground truth hand-verified
# against ~/repos/avatar.mp4 (1280x714, 29.97fps, ~505s). See the 2026-07-30
# calibration report for how each state/second was derived.
FRAMES = [
    # match 2 (gray/rocky stage), round 1: 0-0
    *[(s, "in_match", 0, 0) for s in (145, 150, 155, 160, 165, 170)],
    # match 3 (orange/fire stage), round 1: 0-0 -- a second stage/background
    # for the same score, to prove the pip ROIs generalise across stages.
    *[(s, "in_match", 0, 0) for s in (245, 255, 260, 270)],
    # match 2, round 2 (P2/blue won round 1): 0-1
    *[(s, "in_match", 0, 1) for s in (211, 215, 219)],
    # match 1 (orange/fire stage), round 3 (1-1 after a split round 1/round 2)
    *[(s, "in_match", 1, 1) for s in (100, 105, 110, 120, 130)],
    # match 3 finish: P1/red sweeps 2-0 (match_end, P1 winner). Verified
    # frame-by-frame: the 2nd pip only finishes lighting at ~329s (325-328
    # are mid-animation with pip 2 still empty), so use 329-333.
    *[(s, "in_match", 2, 0) for s in (329, 330, 332, 333)],
    # match 2 finish: P2/blue sweeps 2-0 (match_end, P2 winner)
    *[(s, "in_match", 0, 2) for s in (229, 230, 232)],
    # between-match / HUD-absent transitions: story dialogue, black KO wipe
    # (incl. the bar-glitch wipe ~85s), title card, results/menu screen --
    # must never read as an in-match pip count.
    *[(s, "between", None, None) for s in (0, 2, 85, 139, 141, 235, 335)],
]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: build_avatar_corpus.py <clean_avatar.mp4>", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    out = Path(__file__).parent.parent / "samples" / "avatar"
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    counts: dict[str, int] = {}
    for sec, state, p1, p2 in FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
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
