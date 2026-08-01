#!/usr/bin/env python3
"""Extract a labelled SF6 golden corpus from a clean game-capture VOD.

Frames are named `<state>[_p1-<n>_p2-<n>]_<idx>.png` and written to samples/sf6/.
Ground-truth labels come from a hand-verified clip (see the sf6-counter-calibration
memory); regenerate with:  python scripts/build_sf6_corpus.py <clean_sf6.mp4>
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
from fgc_detector.frames.normalize import normalize

# (second, state, p1_games, p2_games) — stable mid-round moments, verified truth.
FRAMES = [
    # game 1: set score 0-0
    *[(s, "in_match", 0, 0) for s in (45, 50, 60, 68, 72)],
    # game 2: 1-0 (P1 won g1 ~75s)
    *[(s, "in_match", 1, 0) for s in (90, 99, 115, 120, 125)],
    # game 3: 1-1 (P2 won g2 ~131s)
    *[(s, "in_match", 1, 1) for s in (150, 168, 186, 200, 222)],
    # game 4: 2-1 (P1 won g3 ~244s)
    *[(s, "in_match", 2, 1) for s in (255, 260, 275, 285, 288)],
    # between-game transitions: HUD absent -> UNKNOWN, must never read a score
    *[(s, "between", None, None) for s in (81, 82, 134, 136, 247, 248, 292)],
]

def main() -> int:
    if len(sys.argv) != 2:
        print("usage: build_sf6_corpus.py <clean_sf6.mp4>", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    out = Path(__file__).parent.parent / "samples" / "sf6"
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    counts: dict[str, int] = {}
    for sec, state, p1, p2 in FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps)); ok, img = cap.read()
        if not ok: print("read fail", sec); continue
        f = normalize(img, (1920, 1080))
        if f is None: print("normalize rejected", sec, img.shape); continue
        label = state if p1 is None else f"{state}_p1-{p1}_p2-{p2}"
        counts[label] = counts.get(label, 0) + 1
        cv2.imwrite(str(out / f"{label}_{counts[label]:04d}.png"), f)
    print("wrote", sum(counts.values()), "frames:", dict(sorted(counts.items())))
    cap.release(); return 0

if __name__ == "__main__":
    raise SystemExit(main())
