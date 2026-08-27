# Marvel TOKON Pip Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Marvel TOKON detector that fires one `match_end` naming the winner when one side's three round pips are all lit, reading the pips in a way that does not depend on their colour.

**Architecture:** A new pure detector module `detectors/tokon.py` implementing the `Detector` protocol, modeled on `detectors/avatar.py`. It reuses the existing marker `Confirmer` for all temporal logic. Because a lit pip is a **white circle replaced by an icon of unknown colour**, the lit test is two-sided and colour-agnostic: the white circle must be gone *and* positive evidence of an icon must be present. Two new general primitives are added to `roi.py` to express that; which one carries the "white circle gone" half is decided by a measured bake-off, not by this plan.

**Tech Stack:** Python ≥ 3.12, uv, OpenCV (`cv2`), NumPy, pytest.

**Spec:** [`docs/superpowers/specs/2026-08-25-tokon-pip-detector.md`](../specs/2026-08-25-tokon-pip-detector.md)

## Global Constraints

- **Branch:** work on `feat/tokon-detector` (already created, based on merged `main`). Never commit to `main`. Never force-push.
- **Run everything with uv from the repo root:** `uv run pytest`, `uv run python …`, `uv run fgc-detect …`.
- **Canonical resolution is 1920×1080.** Every ROI is expressed there; frames are `normalize()`d before `observe()` runs.
- **Never guess ROIs or thresholds.** Every committed constant is measured from `~/repos/tokon/TOKON.mp4` and transcribed verbatim from the calibration report. The coordinates that appear in this plan's scan scripts are **eyeballed starting points for measurement only** and must never reach `tokon.py`.
- **Detectors are pure.** No history, no clock, no I/O inside `observe()`.
- **Fail safe.** An ambiguous reading resolves to *not lit* / `IN_MATCH` / no event, never a guessed winner. A missed match end is recoverable; a false one corrupts the scoreboard.
- **`lit` requires positive evidence of an icon** — never merely the failure to find the white circle. This is the constraint that keeps an occluded pip (super flash, hit spark) from reading as lit.
- **Every closed set is an enum.** Cross-boundary strings use the `DETAIL_*` constants from `types.py`, never bare literals.
- **Tests must be hermetic and able to fail.** No OBS, GPU, network, or real clock. No test may load the `.mp4`; committed corpus PNGs only.
- **`uv run pytest` must be green before and after every task.** Baseline at the start of this plan: 310 passed, 1 xfailed.
- **Reference media** (outside the repo, supplied by the user): `~/repos/tokon/TOKON.mp4` (1280×714, 29.97 fps, ~683 s, contains both a P1 and a P2 win), plus `vlcsnap-00001.png` (0-0), `vlcsnap-00002.png` (P1 2, P2 1), `vlcsnap-00003-p1-win.png` (P1 3, P2 2), `vlcsnap-00004-p2-win.png` (P1 2, P2 3).

---

### Task 1: `Game.TOKON` and the game roster

Adds the enum value and threads it through the roster so the config UI offers it and the confirmer factory hands it the marker strategy. No detection yet.

**Files:**
- Modify: `src/fgc_detector/types.py` (the `Game` enum)
- Modify: `config.example.toml` (`enabled_games`)
- Test: `tests/test_types.py`, `tests/test_config.py`, `tests/test_confirmation.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Game.TOKON` with value `"tokon"`, used by every later task.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_types.py`:

```python
def test_tokon_is_a_game() -> None:
    """TOKON round-trips through the wire value, like every other game."""
    assert Game("tokon") is Game.TOKON
    assert Game.TOKON.value == "tokon"
```

Append to `tests/test_confirmation.py`:

```python
def test_tokon_uses_the_marker_confirmer() -> None:
    """TOKON counts round pips, so it gets the default marker strategy --
    not SF6's set-score counter."""
    confirmer = make_confirmer(Game.TOKON, ConfirmerConfig())

    assert isinstance(confirmer, Confirmer)
    assert confirmer.game is Game.TOKON
```

In `tests/test_config.py`, change the expected roster at line 120 from
`{Game.SF6, Game.TEKKEN8, Game.AVATAR}` to `{Game.SF6, Game.TEKKEN8, Game.AVATAR, Game.TOKON}`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_types.py::test_tokon_is_a_game tests/test_confirmation.py::test_tokon_uses_the_marker_confirmer tests/test_config.py -v`

Expected: FAIL — `AttributeError: TOKON` on the enum tests, and the config test failing on the roster set.

- [ ] **Step 3: Add the enum value**

In `src/fgc_detector/types.py`, add to the `Game` enum:

```python
class Game(StrEnum):
    SF6 = "sf6"
    TEKKEN8 = "tekken8"
    AVATAR = "avatar"
    TOKON = "tokon"
```

- [ ] **Step 4: Add it to the example roster**

In `config.example.toml`, change the `enabled_games` line to:

```toml
enabled_games = ["sf6", "tekken8", "avatar", "tokon"]
```

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS — 312 passed, 1 xfailed.

- [ ] **Step 6: Commit**

```bash
git add src/fgc_detector/types.py config.example.toml tests/test_types.py tests/test_config.py tests/test_confirmation.py
git commit -m "feat: add Game.TOKON enum value and roster entry"
```

---

### Task 2: Two colour-agnostic sampling primitives in `roi.py`

`roi.py` today offers `fill_ratio` (brightness), `color_fill_ratio` (a *vivid* colour — saturation above a floor), and `match_template`. None can express "is there a **white** marker here?", because that needs saturation *below* a ceiling, and none can compare a region against its own background. Both gaps are general, so both fixes go in `roi.py` as general primitives, not TOKON helpers.

**Files:**
- Modify: `src/fgc_detector/detectors/roi.py` (append after `color_fill_ratio`)
- Test: `tests/detectors/test_roi.py`

**Interfaces:**
- Consumes: `Roi` from `roi.py`.
- Produces:
  - `pale_fill_ratio(image: np.ndarray, roi: Roi, *, sat_max: int, val_min: int) -> float`
  - `region_difference(image: np.ndarray, a: Roi, b: Roi) -> float`

- [ ] **Step 1: Write the failing tests**

In `tests/detectors/test_roi.py`, add `import cv2` at the top (it is not imported yet) and extend the
import line to `from fgc_detector.detectors.roi import (Roi, fill_ratio, match_template,
pale_fill_ratio, region_difference)`. Then append:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/detectors/test_roi.py -v`
Expected: FAIL — `ImportError: cannot import name 'pale_fill_ratio'`.

- [ ] **Step 3: Implement both primitives**

Append to `src/fgc_detector/detectors/roi.py`:

```python
def pale_fill_ratio(
    image: np.ndarray,
    roi: Roi,
    *,
    sat_max: int,
    val_min: int,
) -> float:
    """Fraction of the ROI's pixels that are pale: bright and near-colourless.

    A pixel counts when its HSV saturation <= ``sat_max`` AND its value >=
    ``val_min``. This is the reading ``color_fill_ratio`` structurally cannot
    make: that function requires saturation *above* a floor, so it can find a
    vivid marker but never a white one.

    Written for markers whose signal is their *absence* of colour -- a white
    dot, circle, or outline that some other state replaces. Degrades to 0.0 on
    an out-of-frame ROI, like every primitive here.
    """
    patch = roi.crop(image)
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] <= sat_max) & (hsv[:, :, 2] >= val_min)
    return float(np.count_nonzero(mask) / mask.size)


def region_difference(image: np.ndarray, a: Roi, b: Roi) -> float:
    """How different two regions look, as 0.0 (identical) to 1.0 (black vs white).

    Compares the mean BGR colour of each region and returns the mean absolute
    channel difference, normalized by 255. The two ROIs need not be the same
    size or adjacent.

    This is the background-*independent* primitive: a stage or overlay that
    tints both regions equally cancels out, where any absolute threshold would
    be fooled. Use it to ask "does this spot differ from its own surroundings?"
    -- e.g. is a marker drawn here, or is this just the stage showing through?
    Degrades to 0.0 when either ROI falls outside the frame.
    """
    patch_a = a.crop(image)
    patch_b = b.crop(image)
    if patch_a.size == 0 or patch_b.size == 0:
        return 0.0
    mean_a = patch_a.reshape(-1, patch_a.shape[-1]).mean(axis=0)
    mean_b = patch_b.reshape(-1, patch_b.shape[-1]).mean(axis=0)
    return float(np.abs(mean_a - mean_b).mean() / 255.0)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/detectors/test_roi.py -v`
Expected: PASS — all nine new tests green.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS — 321 passed, 1 xfailed.

- [ ] **Step 6: Commit**

```bash
git add src/fgc_detector/detectors/roi.py tests/detectors/test_roi.py
git commit -m "feat: add pale_fill_ratio and region_difference sampling primitives"
```

---

### Task 3: Locate the TOKON ROIs and build the labelled corpus

The measurement task. Produces exact ROI coordinates and a committed corpus of labelled real frames. Nothing here is guessed.

**Files:**
- Create: `scripts/build_tokon_corpus.py`
- Create: `samples/tokon/*.png` (the corpus)
- Create: `/tmp/tokon_locate.py` (throwaway analysis, not committed)

**Interfaces:**
- Consumes: `Game.TOKON` (Task 1), `normalize` from `fgc_detector.frames.normalize`.
- Produces: `samples/tokon/in_match_p1-<n>_p2-<n>_<idx>.png` and `samples/tokon/between_<idx>.png`, plus a written table of ROI coordinates handed to Task 4.

- [ ] **Step 1: Find candidate seconds for every pip state**

Write `/tmp/tokon_locate.py` (throwaway):

```python
import sys
import cv2
import numpy as np

sys.path.insert(0, "src")
from fgc_detector.frames.normalize import normalize

# EYEBALLED starting points, for finding frames only. Never commit these.
P1X = [747, 785, 824]
P2X = [1098, 1136, 1175]
Y = 48

cap = cv2.VideoCapture("/Users/renatomrcosta/repos/tokon/TOKON.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
for sec in range(0, 683, 2):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
    ok, img = cap.read()
    if not ok:
        continue
    frame = normalize(img, (1920, 1080))
    if frame is None:
        continue
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    scores = []
    for xs in (P1X, P2X):
        for x in xs:
            patch = hsv[Y - 6 : Y + 6, x - 6 : x + 6]
            pale = ((patch[:, :, 1] <= 60) & (patch[:, :, 2] >= 150)).mean()
            scores.append(round(float(pale), 3))
    print(sec, scores)
cap.release()
```

Run: `uv run python /tmp/tokon_locate.py > /tmp/tokon_scan.txt`

A **high** pale score means the white circle is still there (empty); a **low** one means something replaced it (lit). Use this to shortlist seconds for each state.

- [ ] **Step 2: Verify every shortlisted second by eye**

For each candidate second, dump the HUD strip and look at it:

```bash
uv run python -c "
import sys, cv2
sys.path.insert(0, 'src')
from fgc_detector.frames.normalize import normalize
cap = cv2.VideoCapture('/Users/renatomrcosta/repos/tokon/TOKON.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
for sec in [28, 42, 126]:  # replace with your shortlist
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec*fps)); ok, img = cap.read()
    f = normalize(img, (1920, 1080))
    cv2.imwrite(f'/tmp/strip_{sec}.png', cv2.resize(f[20:95, 700:1250], None, fx=2.2, fy=2.2, interpolation=cv2.INTER_NEAREST))
"
```

Open each `/tmp/strip_<sec>.png` and record the true `(p1_lit, p2_lit)`. **The label is what you see, not what the scan scored.** Do not proceed with a second you could not read confidently.

- [ ] **Step 3: Measure the exact ROIs by diffing known states**

Using one verified 0-0 frame and one verified frame where a given pip is lit, find the pip's true bounding box:

```bash
uv run python -c "
import sys, cv2, numpy as np
sys.path.insert(0, 'src')
from fgc_detector.frames.normalize import normalize
cap = cv2.VideoCapture('/Users/renatomrcosta/repos/tokon/TOKON.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
def grab(sec):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec*fps)); ok, img = cap.read()
    return normalize(img, (1920, 1080))
empty, lit = grab(28), grab(42)   # replace with your verified seconds
d = np.abs(empty[20:95, 700:1250].astype(int) - lit[20:95, 700:1250].astype(int)).sum(axis=2)
ys, xs = np.where(d > 60)
print('changed rows', ys.min()+20, ys.max()+20, 'cols', xs.min()+700, xs.max()+700)
cv2.imwrite('/tmp/diff.png', (d.clip(0,255)).astype('uint8'))
"
```

Record, in a scratch notes file you keep open for Task 4:
- `P1_PIP_1..3` and `P2_PIP_1..3` — the icon-sized box for each pip (~22×22 px)
- `P1_CORE_1..3` and `P2_CORE_1..3` — a tight box on the white circle at each pip's centre (~12×12 px)
- `P1_BG_1..3` and `P2_BG_1..3` — a background reference box immediately above each pip, outside the icon's reach, same size as the pip box
- Candidate **HUD-gate** ROIs: the green segmented sub-bar under each health bar, and the clock-digit band
- A candidate **char-select** ROI, if and only if the team-select screen occurs between matches (check the seconds around every match boundary, not just t≈0)

- [ ] **Step 4: Write the corpus builder**

Create `scripts/build_tokon_corpus.py`, replacing the `FRAMES` table below with **your verified seconds and labels**:

```python
#!/usr/bin/env python3
"""Extract a labelled Marvel TOKON golden corpus from a clean game-capture VOD.

Frames are named `<state>[_p1-<n>_p2-<n>]_<idx>.png` and written to samples/tokon/.
Ground-truth labels come from a hand-verified clip (see the 2026-08-26 calibration
report); regenerate with:  python scripts/build_tokon_corpus.py <TOKON.mp4>
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

from fgc_detector.frames.normalize import normalize

# (second, state, p1_pips, p2_pips) -- stable moments, ground truth hand-verified
# against ~/repos/tokon/TOKON.mp4 (1280x714, 29.97fps, ~683s). Every second here
# was read off the HUD strip by eye; see the 2026-08-26 calibration report.
FRAMES = [
    # --- REPLACE EVERYTHING BELOW WITH YOUR VERIFIED SECONDS ---
    # 0-0 on at least two visually different stages
    *[(s, "in_match", 0, 0) for s in (126, 128)],
    # mixed scores
    *[(s, "in_match", 2, 1) for s in ()],
    # P1 takes it 3-x, and P2 takes it x-3
    *[(s, "in_match", 3, 2) for s in ()],
    *[(s, "in_match", 2, 3) for s in ()],
    # ADVERSARIAL, required by the spec:
    #   blue-sky stage, all pips empty -- defeats a hue-agnostic saturation test
    *[(s, "in_match", 0, 0) for s in (28,)],
    #   effect-heavy frames (super flash / hit spark) over pips of a KNOWN state
    *[(s, "in_match", 0, 0) for s in ()],
    # between-match / HUD-absent: team select, K.O. banner, results, title card
    *[(s, "between", None, None) for s in (0,)],
]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: build_tokon_corpus.py <TOKON.mp4>", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    out = Path(__file__).parent.parent / "samples" / "tokon"
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
```

Coverage the corpus must have, per the spec: 0-0, at least two mixed scores, a 3-x P1 win, an x-3 P2 win, between-match frames, the blue-sky empty-pip frames, at least two visually different stages, every distinct icon colour the footage contains, and effect-heavy frames over pips of known state. Keep the total to **25–35 frames**.

- [ ] **Step 5: Build the corpus and check it**

Run: `uv run python scripts/build_tokon_corpus.py ~/repos/tokon/TOKON.mp4`
Expected: `wrote 25..35 frames: {...}` with every state present.

Then open several written PNGs and confirm the filename label matches what the HUD shows.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_tokon_corpus.py samples/tokon
git commit -m "data: labelled Marvel TOKON golden corpus from real footage"
```

---

### Task 4: The bake-off — choose the lit test and the HUD gate, write the calibration report

Measures every candidate against the corpus, picks the winners on margin, and records the constants. **This task produces a decision document, not detector code.**

**Files:**
- Create: `docs/superpowers/reports/2026-08-26-tokon-calibration.md`
- Create: `/tmp/tokon_bakeoff.py` (throwaway, not committed)

**Interfaces:**
- Consumes: `samples/tokon/*.png` (Task 3), `pale_fill_ratio` / `region_difference` / `color_fill_ratio` / `match_template` from `roi.py` (Task 2).
- Produces: a copy-paste constants block naming `PIP_ROIS`, `CORE_ROIS`, `BG_ROIS`, `HUD_GATE_ROI`, `WHITE_ABSENT_MAX`, `ICON_PRESENT_MIN`, `PALE_SAT_MAX`, `PALE_VAL_MIN`, `HUD_*` thresholds, and (if built) `CHAR_SELECT_ROI` / `CHAR_SELECT_PRESENT`. Task 5 transcribes these verbatim.

- [ ] **Step 1: Score every candidate over every labelled pip**

Write `/tmp/tokon_bakeoff.py`. Fill `PIP`, `CORE`, `BG` from your Task 3 measurements:

```python
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "src")
from fgc_detector.detectors.roi import (
    Roi, pale_fill_ratio, region_difference, color_fill_ratio,
)

PIP = {}   # e.g. ("p1", 1): Roi(x, y, w, h)  -- from Task 3
CORE = {}  # ("p1", 1): Roi(...)  tight box on the white circle
BG = {}    # ("p1", 1): Roi(...)  background reference above the pip

CORPUS = Path("samples/tokon")
PATTERN = re.compile(r"^in_match_p1-(\d)_p2-(\d)_\d+\.png$")

rows = []
for path in sorted(CORPUS.glob("in_match_*.png")):
    m = PATTERN.match(path.name)
    if not m:
        continue
    image = cv2.imread(str(path))
    counts = {"p1": int(m.group(1)), "p2": int(m.group(2))}
    for side in ("p1", "p2"):
        for index in (1, 2, 3):
            # Pips fill centre-outward, so pip `index` counting from the
            # centre is lit exactly when the side's count reaches it.
            label = 1 if index <= counts[side] else 0
            rows.append({
                "file": path.name, "side": side, "index": index, "label": label,
                "pale_core": pale_fill_ratio(image, CORE[(side, index)], sat_max=60, val_min=150),
                "pale_pip": pale_fill_ratio(image, PIP[(side, index)], sat_max=60, val_min=150),
                "icon_diff": region_difference(image, PIP[(side, index)], BG[(side, index)]),
            })

for metric, lit_is in (("pale_core", "low"), ("pale_pip", "low"), ("icon_diff", "high")):
    lit = [r[metric] for r in rows if r["label"] == 1]
    emp = [r[metric] for r in rows if r["label"] == 0]
    gap = (min(emp) - max(lit)) if lit_is == "low" else (min(lit) - max(emp))
    print(f"{metric:10s} lit=[{min(lit):.3f},{max(lit):.3f}] "
          f"empty=[{min(emp):.3f},{max(emp):.3f}] margin={gap:+.3f}")
    for r in rows:
        ok = (r[metric] < min(emp)) if (lit_is == "low" and r["label"] == 1) else True
        if not ok:
            print("   overlap:", r["file"], r["side"], r["index"], round(r[metric], 3))
```

Run: `uv run python /tmp/tokon_bakeoff.py`

- [ ] **Step 2: Pick the two-part rule and place the thresholds**

The committed rule is fixed by the spec's fail-safe constraint and is **two-sided**:

```
lit = (white-circle score <= WHITE_ABSENT_MAX) AND (icon-present score >= ICON_PRESENT_MIN)
```

- The **white-circle** half is whichever of `pale_core` or `pale_pip` shows the wider margin. If neither separates, add the third candidate — `match_template` against a committed empty-circle template crop — and measure it the same way.
- The **icon-present** half is `icon_diff` (`region_difference`). This half is what makes an occluded pip read *not lit*, so it is not optional however wide the other margin is.
- Place each threshold in the **middle of its measured gap**, and write both the gap and the chosen value into the report.

**Gate — stop and report if either holds:**
- No white-circle candidate separates the labelled states at all.
- A margin is so thin that a threshold cannot sit clear of both ranges.

Neither is something to tune around. Bring it back to the user with the numbers.

- [ ] **Step 3: Choose the HUD-present gate**

Score both candidates over `in_match_*.png` (must read present) and `between_*.png` (must read absent):

```python
GATE_CANDIDATES = {
    "green_subbar": (Roi(0, 0, 0, 0), "color"),   # from Task 3
    "clock_digits": (Roi(0, 0, 0, 0), "pale"),    # from Task 3
}
for name, (roi, kind) in GATE_CANDIDATES.items():
    scores = {}
    for group in ("in_match", "between"):
        vals = []
        for path in sorted(CORPUS.glob(f"{group}_*.png")):
            image = cv2.imread(str(path))
            vals.append(
                color_fill_ratio(image, roi, hue_lo=35, hue_hi=85, sat_min=60, val_min=90)
                if kind == "color"
                else pale_fill_ratio(image, roi, sat_max=60, val_min=150)
            )
        scores[group] = (min(vals), max(vals))
    print(name, scores, "margin=", round(scores["in_match"][0] - scores["between"][1], 3))
```

Take the wider margin. The gate must read *present* on the match-deciding frames — check those specifically, since the losing side's health bar goes dark exactly then.

- [ ] **Step 4: Decide character select**

Check the seconds around every match boundary in the VOD. If the team-select screen appears between matches, measure its ROI and record `CHAR_SELECT_ROI` / `CHAR_SELECT_PRESENT`. If it appears only at session start, record the decision to **omit** it and the reason: the marker `Confirmer` also releases cooldown on an agreeing fresh 0-0 reading, which TOKON publishes at the start of every match's first round. Either answer is written into the report and the module docstring.

- [ ] **Step 5: Write the calibration report**

Create `docs/superpowers/reports/2026-08-26-tokon-calibration.md` containing:

1. **Source clip** — path, resolution, fps, duration.
2. **Method** — how each ROI was located (state diffing), how each frame was labelled (by eye off the HUD strip).
3. **Bake-off table** — every candidate, its lit range, its empty range, its margin, and which won.
4. **The rejected approaches** — the warm hue band and hue-agnostic saturation, with the blue-sky frames named, so nobody re-proposes them.
5. **Copy-paste constants block** — exactly the Python that goes into `tokon.py`.
6. **Ground truth** — the `(second, winner)` list of every match end in the clip, for Task 7's replay validation.
7. **Open item** — the live-1080p ROI verification (see Task 8).

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/reports/2026-08-26-tokon-calibration.md
git commit -m "docs: Marvel TOKON calibration report and lit-test bake-off"
```

---

### Task 5: `TokonPipDetector`

The detector module. Every constant is transcribed verbatim from Task 4's report.

**Files:**
- Create: `src/fgc_detector/detectors/tokon.py`
- Modify: `src/fgc_detector/detectors/__init__.py`
- Test: `tests/detectors/test_tokon_pips.py`

**Interfaces:**
- Consumes: `pale_fill_ratio`, `region_difference`, `color_fill_ratio`, `Roi` (Task 2); `Game.TOKON` (Task 1); the constants from Task 4's report.
- Produces: `TokonPipDetector` with `game = Game.TOKON`, `canonical_size = (1920, 1080)`, `observe(frame) -> Observation`, `rois() -> dict[str, Roi]`, `supported_events() -> frozenset[EventType]`; module constants `CANONICAL_SIZE`, `ROUNDS_TO_WIN`, `P1_PIPS`, `P2_PIPS`, `P1_CORES`, `P2_CORES`, `P1_BGS`, `P2_BGS`, `HUD_GATE_ROI`, `WHITE_ABSENT_MAX`, `ICON_PRESENT_MIN`, `PALE_SAT_MAX`, `PALE_VAL_MIN`.

- [ ] **Step 1: Write the failing tests**

Create `tests/detectors/test_tokon_pips.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np

from fgc_detector.detectors.registry import get_detector, register
from fgc_detector.detectors.roi import Roi
from fgc_detector.detectors.tokon import (
    CANONICAL_SIZE,
    HUD_GATE_ROI,
    P1_BGS,
    P1_CORES,
    P1_PIPS,
    P2_BGS,
    P2_CORES,
    P2_PIPS,
    ROUNDS_TO_WIN,
    TokonPipDetector,
)
from fgc_detector.types import (
    DETAIL_P1_ROUNDS,
    DETAIL_P2_ROUNDS,
    EventType,
    Frame,
    Game,
    Screen,
    Side,
)

WHITE = (255, 255, 255)
STAGE = (120, 100, 90)   # a flat, unremarkable background


def _hsv_bgr(h: int, s: int, v: int) -> tuple[int, int, int]:
    pixel = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(pixel[0]), int(pixel[1]), int(pixel[2])


def _blank() -> np.ndarray:
    return np.full((CANONICAL_SIZE[1], CANONICAL_SIZE[0], 3), STAGE, dtype=np.uint8)


def _paint(image: np.ndarray, roi: Roi, bgr: tuple[int, int, int]) -> None:
    image[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = bgr


def _paint_empty(image: np.ndarray, pip: Roi, core: Roi) -> None:
    """An empty pip: stage showing through, with the small white circle."""
    _paint(image, pip, STAGE)
    _paint(image, core, WHITE)


def _paint_icon(image: np.ndarray, pip: Roi, icon: tuple[int, int, int]) -> None:
    """A lit pip: an icon disc covering the whole pip, no white circle left."""
    _paint(image, pip, icon)


def _frame(image: np.ndarray) -> Frame:
    return Frame(image=image, captured_at=datetime.now(timezone.utc))


def _hud(image: np.ndarray) -> None:
    """Paint whatever the calibrated HUD gate requires to read `present`."""
    _paint(image, HUD_GATE_ROI, _hsv_bgr(60, 200, 200))


def _in_match(p1_lit: int, p2_lit: int, icon: tuple[int, int, int] | None = None) -> np.ndarray:
    icon = icon or _hsv_bgr(11, 200, 200)
    image = _blank()
    _hud(image)
    for side_pips, side_cores, side_bgs, lit in (
        (P1_PIPS, P1_CORES, P1_BGS, p1_lit),
        (P2_PIPS, P2_CORES, P2_BGS, p2_lit),
    ):
        for i, (pip, core, bg) in enumerate(zip(side_pips, side_cores, side_bgs)):
            _paint(image, bg, STAGE)
            if i < lit:
                _paint_icon(image, pip, icon)
            else:
                _paint_empty(image, pip, core)
    return image


def test_rounds_to_win_is_three() -> None:
    assert ROUNDS_TO_WIN == 3


def test_hud_absent_reads_unknown() -> None:
    """No match HUD on screen: no reading, whatever the pips look like."""
    obs = TokonPipDetector().observe(_frame(_blank()))

    assert obs.screen is Screen.UNKNOWN


def test_fresh_match_reads_in_match_zero_zero() -> None:
    """0-0 is the marker Confirmer's cooldown-release signal, so it must be
    published, not merely implied."""
    obs = TokonPipDetector().observe(_frame(_in_match(0, 0)))

    assert obs.screen is Screen.IN_MATCH
    assert obs.winner is None
    assert obs.details[DETAIL_P1_ROUNDS] == "0"
    assert obs.details[DETAIL_P2_ROUNDS] == "0"


def test_p1_sweeps_three_pips_and_wins() -> None:
    obs = TokonPipDetector().observe(_frame(_in_match(3, 0)))

    assert obs.screen is Screen.MATCH_END
    assert obs.winner is Side.P1
    assert obs.details[DETAIL_P1_ROUNDS] == "3"
    assert obs.confidence > 0.0


def test_p2_wins_a_full_length_match() -> None:
    obs = TokonPipDetector().observe(_frame(_in_match(2, 3)))

    assert obs.screen is Screen.MATCH_END
    assert obs.winner is Side.P2
    assert obs.details[DETAIL_P1_ROUNDS] == "2"
    assert obs.details[DETAIL_P2_ROUNDS] == "3"


def test_a_match_in_progress_names_no_winner() -> None:
    obs = TokonPipDetector().observe(_frame(_in_match(2, 2)))

    assert obs.screen is Screen.IN_MATCH
    assert obs.winner is None


def test_both_sides_reading_three_refuses_to_guess() -> None:
    """Impossible in a real match, so it means the ROIs are misreading.
    Fail safe: no winner."""
    obs = TokonPipDetector().observe(_frame(_in_match(3, 3)))

    assert obs.screen is Screen.IN_MATCH
    assert obs.winner is None


def test_icons_of_any_colour_count_as_lit() -> None:
    """The icon palette is not known to be limited to the two colours the
    reference stills happen to show. A colour-band test would pass the warm
    cases and silently fail the rest -- this is the test that catches it."""
    for hue in (0, 11, 25, 60, 100, 130, 165):
        image = _in_match(3, 0, icon=_hsv_bgr(hue, 200, 200))

        obs = TokonPipDetector().observe(_frame(image))

        assert obs.winner is Side.P1, f"hue {hue} was not read as a lit icon"


def test_an_obscured_pip_is_not_lit() -> None:
    """A super flash leaves neither a white circle nor an icon. `lit` requires
    positive evidence of an icon, so this must read as not lit -- otherwise a
    screen-wide effect fires a false match_end."""
    image = _in_match(0, 0)
    for pip, bg in zip(P1_PIPS, P1_BGS):
        _paint(image, pip, STAGE)   # circle gone, but nothing drawn in its place
        _paint(image, bg, STAGE)

    obs = TokonPipDetector().observe(_frame(image))

    assert obs.screen is Screen.IN_MATCH
    assert obs.winner is None
    assert obs.details[DETAIL_P1_ROUNDS] == "0"


def test_rois_are_exposed_for_the_cli_preview() -> None:
    rois = TokonPipDetector().rois()

    assert "hud_gate" in rois
    for side in ("p1", "p2"):
        for index in (1, 2, 3):
            assert f"{side}_pip_{index}" in rois


def test_it_supports_match_end_only() -> None:
    assert TokonPipDetector().supported_events() == frozenset({EventType.MATCH_END})


def test_it_registers_itself() -> None:
    detector = TokonPipDetector()
    register(detector)

    assert get_detector(Game.TOKON) is detector
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/detectors/test_tokon_pips.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fgc_detector.detectors.tokon'`.

- [ ] **Step 3: Write the detector**

Create `src/fgc_detector/detectors/tokon.py`. **Replace every `Roi(...)` and every threshold below with the values from `docs/superpowers/reports/2026-08-26-tokon-calibration.md`** — the placeholders here are structure, not data:

```python
"""Marvel TOKON round-pip detector.

TOKON shows three round pips per side in a row flanking the central match
clock. An empty pip is a small white circle; a lit one is that circle replaced
by an icon -- and the icon can be one of many colours, so this detector never
tests for a colour. It asks the two questions that survive an unknown palette:
is the white circle gone, and is something icon-shaped there instead?

Both halves are required. Testing only for the white circle's absence would
invert the project's fail-safe direction: a pip hidden behind a super flash
would read as lit, and three of them would fire a false match_end. Requiring
positive icon evidence means an obscured pip reads as *not lit* -- a missed
match end, which the operator recovers from.

A colour band was measured and rejected: it scores perfectly on the reference
stills only because every icon in them is warm. A hue-agnostic saturation test
was also measured and rejected -- the blue-sky stage bleeds through the box
around the small circle and scores empty pips as lit. See
docs/superpowers/reports/2026-08-26-tokon-calibration.md for both, and for how
every constant below was measured.
"""

from __future__ import annotations

from ..types import (
    DETAIL_P1_ROUNDS,
    DETAIL_P2_ROUNDS,
    EventType,
    Frame,
    Game,
    Observation,
    Screen,
    Side,
)
from .registry import register
from .roi import Roi, color_fill_ratio, pale_fill_ratio, region_difference

#: Canonical resolution these ROIs are expressed in. Frames are normalised to
#: this before observe() runs.
CANONICAL_SIZE = (1920, 1080)

ROUNDS_TO_WIN = 3

# --- MEASURED CONSTANTS: transcribed verbatim from the Task 4 calibration
# --- report (docs/superpowers/reports/2026-08-26-tokon-calibration.md).
#: Icon-sized box per pip, centre-adjacent first (pips fill centre-outward).
P1_PIPS = (Roi(0, 0, 1, 1), Roi(0, 0, 1, 1), Roi(0, 0, 1, 1))
P2_PIPS = (Roi(0, 0, 1, 1), Roi(0, 0, 1, 1), Roi(0, 0, 1, 1))
#: Tight box on the white circle at each pip's centre.
P1_CORES = (Roi(0, 0, 1, 1), Roi(0, 0, 1, 1), Roi(0, 0, 1, 1))
P2_CORES = (Roi(0, 0, 1, 1), Roi(0, 0, 1, 1), Roi(0, 0, 1, 1))
#: Background reference just outside each pip, for the icon-present test.
P1_BGS = (Roi(0, 0, 1, 1), Roi(0, 0, 1, 1), Roi(0, 0, 1, 1))
P2_BGS = (Roi(0, 0, 1, 1), Roi(0, 0, 1, 1), Roi(0, 0, 1, 1))

HUD_GATE_ROI = Roi(0, 0, 1, 1)
HUD_HUE = (35, 85)
HUD_SAT_MIN = 60
HUD_VAL_MIN = 90
#: HUD-gate fill at or above which a match HUD is considered on screen.
HUD_PRESENT_MIN = 0.0

PALE_SAT_MAX = 60
PALE_VAL_MIN = 150
#: White-circle fill at or below which the circle is considered gone.
WHITE_ABSENT_MAX = 0.0
#: Pip-vs-background difference at or above which an icon is considered present.
ICON_PRESENT_MIN = 0.0


class TokonPipDetector:
    """Counts lit round pips without reading their colour. Stateless and pure."""

    canonical_size = CANONICAL_SIZE
    game = Game.TOKON

    def rois(self) -> dict[str, Roi]:
        named: dict[str, Roi] = {"hud_gate": HUD_GATE_ROI}
        for side, pips, cores, bgs in (
            ("p1", P1_PIPS, P1_CORES, P1_BGS),
            ("p2", P2_PIPS, P2_CORES, P2_BGS),
        ):
            for index, (pip, core, bg) in enumerate(zip(pips, cores, bgs), start=1):
                named[f"{side}_pip_{index}"] = pip
                named[f"{side}_core_{index}"] = core
                named[f"{side}_bg_{index}"] = bg
        return named

    def supported_events(self) -> frozenset[EventType]:
        return frozenset({EventType.MATCH_END})

    def _icon_scores(self, image, pips, cores, bgs) -> list[float]:
        """Per pip: the icon-present score, or 0.0 where the circle is still there.

        A pip scores only when BOTH halves agree -- the white circle is gone AND
        the pip differs from its own background. A pip that fails either half
        scores 0.0 and is not lit, which is what makes an obscured pip safe.
        """
        scores = []
        for pip, core, bg in zip(pips, cores, bgs):
            white = pale_fill_ratio(image, core, sat_max=PALE_SAT_MAX, val_min=PALE_VAL_MIN)
            icon = region_difference(image, pip, bg)
            scores.append(icon if (white <= WHITE_ABSENT_MAX and icon >= ICON_PRESENT_MIN) else 0.0)
        return scores

    def observe(self, frame: Frame) -> Observation:
        image = frame.image

        hud = color_fill_ratio(
            image,
            HUD_GATE_ROI,
            hue_lo=HUD_HUE[0],
            hue_hi=HUD_HUE[1],
            sat_min=HUD_SAT_MIN,
            val_min=HUD_VAL_MIN,
        )
        if hud < HUD_PRESENT_MIN:
            return Observation(screen=Screen.UNKNOWN, debug={"hud_gate": hud})

        p1_scores = self._icon_scores(image, P1_PIPS, P1_CORES, P1_BGS)
        p2_scores = self._icon_scores(image, P2_PIPS, P2_CORES, P2_BGS)
        p1_lit = sum(1 for score in p1_scores if score > 0.0)
        p2_lit = sum(1 for score in p2_scores if score > 0.0)

        debug = {"hud_gate": hud}
        for side, scores in (("p1", p1_scores), ("p2", p2_scores)):
            for index, score in enumerate(scores, start=1):
                debug[f"{side}_pip_{index}"] = score

        # Published on every IN_MATCH and MATCH_END observation under the shared
        # constants the Confirmer reads: a fresh 0-0 is its cooldown-release
        # signal, so these keys must never drift onto bare string literals.
        details = {DETAIL_P1_ROUNDS: str(p1_lit), DETAIL_P2_ROUNDS: str(p2_lit)}

        p1_won = p1_lit >= ROUNDS_TO_WIN
        p2_won = p2_lit >= ROUNDS_TO_WIN
        if p1_won == p2_won:
            # Neither side is done, or both read as done -- the latter is
            # impossible in a real match and means the ROIs are misreading.
            # Refuse to guess a winner.
            return Observation(screen=Screen.IN_MATCH, details=details, debug=debug)

        winner = Side.P1 if p1_won else Side.P2
        winner_scores = p1_scores if p1_won else p2_scores
        return Observation(
            screen=Screen.MATCH_END,
            winner=winner,
            confidence=min(winner_scores),
            details=details,
            debug=debug,
        )


register(TokonPipDetector())
```

If Task 4 chose `match_template` for the white-circle half instead of `pale_fill_ratio`, swap that
one call and add the committed template crop — the two-part rule and everything else is unchanged.

**If Task 4 decided to build the character-select branch**, add `CHAR_SELECT_ROI` and
`CHAR_SELECT_PRESENT` to the constants block, `"char_select": CHAR_SELECT_ROI` to `rois()`, and this
check as the **first** thing `observe()` does — before the HUD gate, because a frame that could read
as either must resolve to `CHAR_SELECT`: it is the Confirmer's cooldown exit, and mistaking it for
something else wedges the detector until the safety valve.

```python
        char_select = color_fill_ratio(
            image,
            CHAR_SELECT_ROI,
            hue_lo=CHAR_SELECT_HUE[0],
            hue_hi=CHAR_SELECT_HUE[1],
            sat_min=CHAR_SELECT_SAT_MIN,
            val_min=CHAR_SELECT_VAL_MIN,
        )
        if char_select >= CHAR_SELECT_PRESENT:
            return Observation(
                screen=Screen.CHAR_SELECT,
                confidence=char_select,
                debug={"char_select": char_select},
            )
```

and add this test to `tests/detectors/test_tokon_pips.py`:

```python
def test_team_select_reads_char_select() -> None:
    """The Confirmer's cooldown exit. Checked before the HUD gate, so an
    ambiguous frame resolves here rather than reading as a match."""
    image = _blank()
    _hud(image)  # even with a HUD-like reading, team select wins
    _paint(image, CHAR_SELECT_ROI, _hsv_bgr(*CHAR_SELECT_FILL_HSV))

    obs = TokonPipDetector().observe(_frame(image))

    assert obs.screen is Screen.CHAR_SELECT
```

**If Task 4 decided to omit it**, write the reason into the module docstring — the marker
`Confirmer` also releases cooldown on an agreeing fresh 0-0 reading, which this detector publishes
at the start of every match's first round, so `CHAR_SELECT` is not required for TOKON's cooldown to
clear. Do not invent an ROI for a screen you could not measure.

- [ ] **Step 4: Register the module on import**

In `src/fgc_detector/detectors/__init__.py`, add the import in alphabetical order:

```python
from . import avatar  # noqa: F401
from . import sf6  # noqa: F401
from . import tokon  # noqa: F401
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/detectors/test_tokon_pips.py -v`
Expected: PASS — all twelve tests green (thirteen if the character-select branch was built).

If `test_icons_of_any_colour_count_as_lit` or `test_an_obscured_pip_is_not_lit` fails, **do not
loosen a threshold**: the two-part rule or a calibrated constant is wrong. Re-read the report.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS — no regressions.

- [ ] **Step 7: Commit**

```bash
git add src/fgc_detector/detectors/tokon.py src/fgc_detector/detectors/__init__.py tests/detectors/test_tokon_pips.py
git commit -m "feat: add TokonPipDetector reading round pips without reading their colour"
```

---

### Task 6: Corpus regression and confirmer integration

Synthetic frames prove the logic; these prove the constants against real pixels, and prove the detector actually drives the confirmer.

**Files:**
- Create: `tests/detectors/test_tokon_corpus.py`

**Interfaces:**
- Consumes: `TokonPipDetector` (Task 5), `samples/tokon/*.png` (Task 3), `Confirmer` / `ConfirmerConfig` from `fgc_detector.confirmer`.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Write the failing tests**

Create `tests/detectors/test_tokon_corpus.py`:

```python
"""Corpus-driven and confirmer-integration tests for TokonPipDetector.

Part A parametrizes over the real, labelled `samples/tokon/in_match_*.png`
corpus: every filename is ground truth for both players' round-pip counts, and
a frame reads MATCH_END with a winner exactly when one side's count is 3.

Part B covers `between_*.png`. It does not assert every between frame reads
UNKNOWN -- it asserts the safety invariant that actually matters: no between
frame ever reports a nonzero pip count or names a winner. That invariant is
not vacuous; it fails immediately if the detector reads stage colour, a K.O.
banner, or a results screen as a lit pip.

Part C proves the contract (DETAIL_P1_ROUNDS/DETAIL_P2_ROUNDS + MATCH_END +
winner) drives the reused marker Confirmer end to end, using synthetic
Observations -- no images, no real clock.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import pytest

from fgc_detector.confirmer import Confirmer, ConfirmerConfig
from fgc_detector.detectors.tokon import TokonPipDetector
from fgc_detector.types import (
    DETAIL_P1_ROUNDS,
    DETAIL_P2_ROUNDS,
    Frame,
    Game,
    Observation,
    Screen,
    Side,
)

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "samples" / "tokon"

_IN_MATCH_PATTERN = re.compile(r"^in_match_p1-(\d)_p2-(\d)_\d+\.png$")

_IN_MATCH_CASES: list[tuple[str, int, int]] = []
_BETWEEN_CASES: list[str] = []
for path in sorted(CORPUS_DIR.glob("*.png")):
    match = _IN_MATCH_PATTERN.match(path.name)
    if match:
        _IN_MATCH_CASES.append((path.name, int(match.group(1)), int(match.group(2))))
    elif path.name.startswith("between_"):
        _BETWEEN_CASES.append(path.name)


def _frame(name: str) -> Frame:
    image = cv2.imread(str(CORPUS_DIR / name))
    assert image is not None, f"corpus frame {name} did not load"
    return Frame(image=image, captured_at=datetime.now(timezone.utc))


def test_the_corpus_is_not_empty() -> None:
    """Guards against a silently-empty parametrization making Part A vacuous."""
    assert len(_IN_MATCH_CASES) >= 15
    assert len(_BETWEEN_CASES) >= 3


@pytest.mark.parametrize("name,p1,p2", _IN_MATCH_CASES)
def test_corpus_pip_counts_match_ground_truth(name: str, p1: int, p2: int) -> None:
    obs = TokonPipDetector().observe(_frame(name))

    assert obs.details[DETAIL_P1_ROUNDS] == str(p1), obs.debug
    assert obs.details[DETAIL_P2_ROUNDS] == str(p2), obs.debug


@pytest.mark.parametrize("name,p1,p2", _IN_MATCH_CASES)
def test_corpus_winner_matches_ground_truth(name: str, p1: int, p2: int) -> None:
    obs = TokonPipDetector().observe(_frame(name))

    if p1 == 3 and p2 < 3:
        assert obs.screen is Screen.MATCH_END and obs.winner is Side.P1, obs.debug
    elif p2 == 3 and p1 < 3:
        assert obs.screen is Screen.MATCH_END and obs.winner is Side.P2, obs.debug
    else:
        assert obs.screen is Screen.IN_MATCH and obs.winner is None, obs.debug


@pytest.mark.parametrize("name", _BETWEEN_CASES)
def test_between_frames_never_report_pips_or_a_winner(name: str) -> None:
    obs = TokonPipDetector().observe(_frame(name))

    assert obs.winner is None, obs.debug
    assert obs.details.get(DETAIL_P1_ROUNDS, "0") == "0", obs.debug
    assert obs.details.get(DETAIL_P2_ROUNDS, "0") == "0", obs.debug


def _observation(screen: Screen, p1: int, p2: int, winner: Side | None = None) -> Observation:
    return Observation(
        screen=screen,
        winner=winner,
        confidence=0.9,
        details={DETAIL_P1_ROUNDS: str(p1), DETAIL_P2_ROUNDS: str(p2)},
    )


def test_the_detector_contract_fires_exactly_one_event_through_the_confirmer() -> None:
    confirmer = Confirmer(Game.TOKON, ConfirmerConfig(agreement_frames=3))
    confirmer.arm()
    now = datetime.now(timezone.utc)
    events = []

    for step, obs in enumerate(
        [
            _observation(Screen.IN_MATCH, 2, 2),
            _observation(Screen.MATCH_END, 3, 2, Side.P1),
            _observation(Screen.MATCH_END, 3, 2, Side.P1),
            _observation(Screen.MATCH_END, 3, 2, Side.P1),
            _observation(Screen.MATCH_END, 3, 2, Side.P1),
        ]
    ):
        event = confirmer.observe(obs, now + timedelta(seconds=step * 0.2))
        if event is not None:
            events.append(event)

    assert len(events) == 1
    assert events[0].winner is Side.P1
    assert events[0].game is Game.TOKON


def test_a_fresh_zero_zero_reading_releases_cooldown_for_the_next_match() -> None:
    """Players rematch without passing through team select, so 0-0 has to be
    what re-arms the detector -- otherwise match 2 of every session is missed."""
    confirmer = Confirmer(Game.TOKON, ConfirmerConfig(agreement_frames=3))
    confirmer.arm()
    now = datetime.now(timezone.utc)
    step = 0

    def feed(obs: Observation):
        nonlocal step
        step += 1
        return confirmer.observe(obs, now + timedelta(seconds=step * 0.2))

    feed(_observation(Screen.IN_MATCH, 2, 2))
    for _ in range(3):
        feed(_observation(Screen.MATCH_END, 3, 2, Side.P1))
    for _ in range(3):
        feed(_observation(Screen.IN_MATCH, 0, 0))

    events = [feed(_observation(Screen.MATCH_END, 0, 3, Side.P2)) for _ in range(3)]

    assert [event for event in events if event is not None][0].winner is Side.P2
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/detectors/test_tokon_corpus.py -v`
Expected: PASS.

A genuine corpus-label conflict — a frame the calibration report explains, such as a mid-animation
pip caught half-lit — gets a strict `xfail` with the reason written out. **Never weaken an
assertion to make a frame pass.** A wrong reading on a clean frame means the constants are wrong.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/detectors/test_tokon_corpus.py
git commit -m "test: TOKON corpus regression and confirmer integration"
```

---

### Task 7: End-to-end replay validation

The one check that exercises the whole pipeline against real, moving footage.

**Files:**
- Modify: `docs/superpowers/reports/2026-08-26-tokon-calibration.md` (results section)

**Interfaces:**
- Consumes: the registered `TokonPipDetector` (Task 5), the ground-truth list from Task 4.
- Produces: a recorded replay result.

- [ ] **Step 1: Run the replay**

Run: `uv run fgc-detect replay --game tokon --video ~/repos/tokon/TOKON.mp4`

Expected: one `match_end` per match, with the correct winner, at the second each match is decided.
A few frames of confirmation lag is fine.

- [ ] **Step 2: Compare against ground truth**

Check the emitted `(ts, winner)` pairs against the ground-truth list in the calibration report.
Investigate, do not paper over:
- **A missing event** — the deciding frames failed the lit test or the HUD gate.
- **An extra event** — the cooldown released early, or a replay/effect frame read as a match end.
  An event firing during a super rather than at a K.O. is the occlusion signature the fail-safe rule
  exists to prevent; treat it as a calibration defect, not a confirmer tuning problem.

- [ ] **Step 3: Re-run with evidence if anything is off**

Run: `uv run fgc-detect replay --game tokon --video ~/repos/tokon/TOKON.mp4 --evidence-dir evidence/`

Inspect the dumped frame and observation behind each event.

- [ ] **Step 4: Record the result**

Add a "Replay validation" section to the calibration report: the command, the emitted events, the
comparison against ground truth, and any limitation found. If a limitation is real and accepted,
also add it to `docs/TODO.md` in Task 8.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/reports/2026-08-26-tokon-calibration.md
git commit -m "docs: record TOKON end-to-end replay validation"
```

---

### Task 8: Documentation

Makes the pips-are-the-default decision explicit, and records the two follow-ups this work is
deliberately not doing.

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `docs/TODO.md`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing.

- [ ] **Step 1: Update the README**

- Line 5: add Marvel TOKON to the list of validated games.
- The games table (~line 22): add a row — `| Marvel TOKON | the three round pips flanking the match clock | white-circle-replaced-by-icon, colour-agnostic + marker confirmer (3 pips → win) |`
- Line ~25, which currently reads "Digit counting is SF6's answer to the interface; pip-colour counting is Avatar's. Neither is a…": replace the even-handed framing with the decision. **Counting round pips is the default approach for a new game; SF6's games-won-in-set digit counter is the exception its HUD forces.** Say plainly that a new game starts from "where are its round pips, and how do they change when a round is won?".
- The mermaid diagram labels (~lines 47, 51): add TOKON alongside SF6 and Avatar.
- Line ~122: add `tokon` to the `game` config values.
- The spec link block (~line 81): link the TOKON spec and calibration report.

- [ ] **Step 2: Update CLAUDE.md**

- In "How to add a new game" step 3, make the default explicit: start from round pips; reach for a
  dedicated detector when the pips are not brightness-distinct; leave the pip path only when the HUD
  offers nothing pip-shaped.
- Add TOKON as a worked example alongside Avatar, and note what makes it different: the icons are
  **not** side-coded and **not** a fixed palette, so it reads the marker's identity (white circle vs
  icon) rather than its colour.
- Add to "Known hazards & gotchas": an absence test inverts the fail-safe direction — `lit` must
  require positive evidence, or an occluded pip fires a false event.
- Add the two new `roi.py` primitives to the primitive list in step 3: `pale_fill_ratio` (a white
  marker) and `region_difference` (a region versus its own background).

- [ ] **Step 3: Update docs/TODO.md**

Add two entries:

```markdown
## TOKON: verify ROIs against a native 1080p capture (open)

TOKON's ROIs were calibrated from `~/repos/tokon/TOKON.mp4`, which is 1280x714 -- a 720p frame six
rows short, so it is cropped, not merely scaled. `normalize()` accepts it (0.8% inside the aspect
tolerance) and stretches it vertically to 1080, so the committed y-coordinates may sit a few pixels
off against a native 1920x1080 OBS capture, and the pips are only ~22px across. Accepted at
calibration time (user decision, 2026-08-25). To close it: with TOKON on screen, run
`fgc-detect capture --config config.toml --out obs_frames` then
`fgc-detect roi --game tokon --sample obs_frames/frame_00000.png --out roi_check.png`, confirm the
boxes sit on the pips, and adjust the y-offsets if they miss. Fails safe meanwhile: a misaligned ROI
reads "not lit", so the failure mode is a missed event, never a false winner.

## MarkerRoundDetector has no users and three near-copies (open)

`detectors/marker.py` was written as the shared base for pip games and has **zero registered
games**. Avatar and TOKON both hand-rolled their own detectors instead, because the base samples
brightness only: Avatar's pips fill with saturated colour, and TOKON's empty pips are *brighter*
(median value ~168) than its lit ones (~133), so `fill_ratio` reads them backwards. Three copies of
the same count-and-compare-and-refuse-to-guess logic now exist, which is exactly where drift causes
a false winner. Either give `MarkerLayout` a pluggable sampler and migrate the three onto it, or
delete the base and accept per-game modules as policy. Decide when the fourth pip game lands.
```

- [ ] **Step 4: Verify the docs are accurate**

Run: `uv run pytest -q`
Expected: PASS — 321+ passed.

Re-read the README's games table and confirm every claim matches what the code actually does.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md docs/TODO.md
git commit -m "docs: make round pips the documented default; add TOKON"
```

- [ ] **Step 6: Open a draft PR**

```bash
git push -u origin feat/tokon-detector
gh pr create --draft --base main --title "Marvel TOKON detector (colour-agnostic round pips)" --body "Implements docs/superpowers/specs/2026-08-25-tokon-pip-detector.md. See docs/superpowers/reports/2026-08-26-tokon-calibration.md for measured constants and the lit-test bake-off."
```

---

## Definition of done

- `uv run pytest` green, with the TOKON corpus tests actually parametrizing over committed frames.
- `uv run fgc-detect replay --game tokon --video ~/repos/tokon/TOKON.mp4` names the right winners at
  the right times, recorded in the calibration report.
- Every constant in `tokon.py` traces to a measurement in the calibration report.
- The two open items are in `docs/TODO.md`.
- Draft PR open against `main`.
