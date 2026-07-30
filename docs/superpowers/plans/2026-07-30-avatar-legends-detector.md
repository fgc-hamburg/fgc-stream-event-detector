# Avatar Legends Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third game, Avatar Legends, that fires one `match_end` event per match by counting the coloured round pips (P1 red / P2 blue) flanking the clock emblem.

**Architecture:** A dedicated stateless `AvatarPipDetector` reads the pips by colour (new reusable `color_fill_ratio` helper) and reports `IN_MATCH` / `MATCH_END` / `CHAR_SELECT` / `UNKNOWN` on the same `Observation` contract as `MarkerRoundDetector`. It reuses the existing marker `Confirmer` (reach-2-pips → fire) and the `Screen` / round-count-details contract unchanged. `make_confirmer` already routes every non-SF6 game to the marker `Confirmer`, so no factory change is needed.

**Tech Stack:** Python ≥3.12, uv, opencv-python-headless, numpy, pytest.

**Design doc:** [`docs/superpowers/specs/2026-07-30-avatar-legends-detector.md`](../specs/2026-07-30-avatar-legends-detector.md)

## Global Constraints

Every task's requirements implicitly include these:

- **Every closed-set value is an enum.** Avatar is `Game.AVATAR = "avatar"`; never a bare `"avatar"` string outside the enum definition.
- **Detectors are pure and stateless.** `observe()` classifies one frame; no history, clock, or I/O. All temporal logic stays in the `Confirmer`.
- **Shared string-contract constants only.** Round counts are published under `DETAIL_P1_ROUNDS` / `DETAIL_P2_ROUNDS` from `types.py` — never bare literals; producer and consumer must not drift.
- **Never guess ROI coordinates or thresholds — measure them from `~/repos/avatar.mp4`.** (Task 3 is the only source of pip/gate ROIs, hue bands, and thresholds; later tasks transcribe its measured values.)
- **Fail safe.** Any ambiguous reading resolves to "no event" (a missed match end), never a false winner. A frame where neither or both sides read as 2 pips is `IN_MATCH`, not a guess.
- **`rounds_to_win = 2`; one `match_end` per match**, naming the side that reached 2 pips.
- **Tests must be able to fail for the right reason.** No test may require OBS, GPU, network, or a real clock. Detector tests use synthetic frames or the committed corpus PNGs; no test loads `avatar.mp4` at run time.
- **Reuse, don't fork:** the marker `Confirmer`, the `Screen`/details contract, and `Roi` are reused unchanged. `color_fill_ratio` is added to `roi.py` as a general primitive, not Avatar-specific code.

## File Structure

- Modify `src/fgc_detector/types.py` — add `Game.AVATAR`.
- Modify `src/fgc_detector/detectors/roi.py` — add `color_fill_ratio`.
- Create `scripts/build_avatar_corpus.py` — reproducible corpus extraction.
- Create `samples/avatar/*.png` — labelled real-frame corpus.
- Create `src/fgc_detector/detectors/avatar.py` — `AvatarPipDetector` + measured constants + self-registration.
- Modify `src/fgc_detector/detectors/__init__.py` — import `avatar` so it registers.
- Modify `config.example.toml` — add `"avatar"` to the `enabled_games` roster.
- Create `tests/detectors/test_color_fill.py`, `tests/detectors/test_avatar_pips.py`.
- Create `docs/superpowers/reports/2026-07-30-avatar-calibration.md` — calibration + validation log (measured constants, ground-truth timestamps, replay result).

---

### Task 1: `Game.AVATAR` enum value and confirmer routing

**Files:**
- Modify: `src/fgc_detector/types.py` (the `Game` enum, ~line 21-23)
- Test: `tests/test_types.py`, `tests/test_confirmation.py`

**Interfaces:**
- Produces: `Game.AVATAR` (value `"avatar"`), consumed by every later task and by `make_confirmer`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_types.py` add:

```python
def test_avatar_is_a_game() -> None:
    from fgc_detector.types import Game
    assert Game("avatar") is Game.AVATAR
    assert Game.AVATAR.value == "avatar"
```

In `tests/test_confirmation.py` add (import the concrete `Confirmer` as the existing tests there do):

```python
def test_avatar_uses_the_marker_confirmer() -> None:
    from fgc_detector.confirmation import make_confirmer
    from fgc_detector.confirmer import Confirmer, ConfirmerConfig
    from fgc_detector.types import Game
    confirmer = make_confirmer(Game.AVATAR, ConfirmerConfig())
    assert isinstance(confirmer, Confirmer)
    assert confirmer.game is Game.AVATAR
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_types.py::test_avatar_is_a_game tests/test_confirmation.py::test_avatar_uses_the_marker_confirmer -v`
Expected: FAIL — `ValueError: 'avatar' is not a valid Game`.

- [ ] **Step 3: Add the enum value**

In `src/fgc_detector/types.py`, in the `Game` enum:

```python
class Game(StrEnum):
    SF6 = "sf6"
    TEKKEN8 = "tekken8"
    AVATAR = "avatar"
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_types.py tests/test_confirmation.py -v`
Expected: PASS. (`make_confirmer` needs no change — its `else` branch already returns the marker `Confirmer` for every non-SF6 game.)

- [ ] **Step 5: Commit**

```bash
git add src/fgc_detector/types.py tests/test_types.py tests/test_confirmation.py
git commit -m "feat: add Game.AVATAR enum value"
```

---

### Task 2: `color_fill_ratio` colour-aware sampling primitive

**Files:**
- Modify: `src/fgc_detector/detectors/roi.py`
- Test: `tests/detectors/test_color_fill.py` (new)

**Interfaces:**
- Produces: `color_fill_ratio(image, roi, *, hue_lo, hue_hi, sat_min, val_min) -> float`, consumed by `AvatarPipDetector` (Task 4). Hue is OpenCV's 0–179 scale; `hue_lo > hue_hi` means a wrap-around band (for red near 0/179).

- [ ] **Step 1: Write the failing tests**

Create `tests/detectors/test_color_fill.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/detectors/test_color_fill.py -v`
Expected: FAIL — `ImportError: cannot import name 'color_fill_ratio'`.

- [ ] **Step 3: Implement the primitive**

Append to `src/fgc_detector/detectors/roi.py` (after `match_template`):

```python
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
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/detectors/test_color_fill.py -v`
Expected: PASS (all 6).

- [ ] **Step 5: Commit**

```bash
git add src/fgc_detector/detectors/roi.py tests/detectors/test_color_fill.py
git commit -m "feat: add color_fill_ratio HSV sampling primitive"
```

---

### Task 3: Calibrate pip/gate ROIs and build the labelled corpus

This is the de-risking / measurement task (the analogue of the SF6 counter calibration). It produces **all** the measured constants later tasks transcribe, a reproducible corpus, and the ground-truth match-end timestamps. Nothing here is guessed.

**Files:**
- Create: `scripts/build_avatar_corpus.py`
- Create: `samples/avatar/*.png` (committed corpus)
- Create: `docs/superpowers/reports/2026-07-30-avatar-calibration.md`

**Interfaces:**
- Produces (into the calibration report, for Task 4 to transcribe verbatim): `CANONICAL_SIZE`; four pip ROIs `P1_PIP_1`, `P1_PIP_2`, `P2_PIP_1`, `P2_PIP_2`; the HUD-present gate metric + threshold; the char-select gate ROI + threshold; the red and blue hue bands + `SAT_MIN`/`VAL_MIN`; and the per-side `PIP_LIT` threshold. Plus the ground-truth `(match_end_second, winner)` list.

**Starting facts already established (from design investigation — verify, don't trust blindly):**
- `avatar.mp4` is 1280×714, 29.97 fps, ~505 s. Detectors run at canonical 1920×1080 (frames are `normalize`d first, as in `build_sf6_corpus.py`).
- The clock emblem is a dark hexagonal box, top-centre (~x600–680, y20–105 at 1280-wide; ×1.5 for canonical).
- The pips are the **two stacked angular bars immediately flanking the emblem** — left = P1 (fills red), right = P2 (fills blue). Empty pip = dark interior + cyan/gold outline; filled = saturated red/blue.
- The circles *below* the life bars are a per-round resource gauge — **NOT** pips; do not sample them.
- Matches are separated by stage changes / character-select screens; "BEGIN"/"ROUND 2"/"FINISH" cards mark round boundaries. Candidate match-end regions found while sampling: around the orange→gray stage change (~135 s) and the gray→orange change (~290 s); there are more matches through the green stage to ~505 s.

- [ ] **Step 1: Locate the pips and gate precisely.** Extract frames from `~/repos/avatar.mp4` with ffmpeg/opencv. Diff a clean 0-0 start frame against match-point frames (a side showing 2 filled pips) to isolate each pip bar's rectangle, separated from the cyan health-bar edge and the resource-gauge circles. Record the four pip ROIs, the emblem ROI, and a char-select ROI in **canonical 1920×1080** coordinates. Save a few annotated crops into the calibration report.

- [ ] **Step 2: Measure the colour bands and thresholds.** For each pip ROI across empty / red-filled / blue-filled states, dump HSV histograms. Choose: the red band (wrap-around near 0/179), the blue band (tight, must exclude cyan H~90), `SAT_MIN`, `VAL_MIN`, and the `PIP_LIT` fill-ratio threshold — each sitting in a clean gap between empty and filled. **Primary risk: blue pip vs cyan outline.** If the gap is tight, tighten the P2 pip ROIs to the bar interior and/or raise `SAT_MIN`. Record the measured gaps (empty vs lit ratio per state) in the report so the choice is justified, not asserted.

- [ ] **Step 3: Choose the HUD-present gate.** The emblem is dark; measure mean grayscale of the emblem ROI during play vs on char-select / transition / victory screens, and pick a metric + threshold that separates them (e.g. emblem mean-gray below a max during play). If a darkness gate does not separate cleanly, fall back to "at least one player's pip-region structure is present" — document whichever is used and why.

- [ ] **Step 4: Determine ground-truth match ends.** From the labelled states, identify each second where one side first reaches 2 pips (match decided) and the winner. This is the validation baseline (user delegated deriving these).

- [ ] **Step 5: Write `scripts/build_avatar_corpus.py`.** Model it on `scripts/build_sf6_corpus.py`: a `FRAMES` list of `(second, state, p1_pips, p2_pips)` at stable, hand-verified moments, `normalize`d to 1920×1080, written to `samples/avatar/` as `<state>_p1-<n>_p2-<n>_<idx>.png`, with `state="between"` (p1/p2 `None`) for HUD-absent/char-select/transition frames. Cover: `0-0`, `1-0`, `1-1`, `2-0` (P1 win), `0-2` and/or `1-2` (P2 win), at least one stage-swap-mid-match frame, and several `between` frames. Run it to generate `samples/avatar/*.png`.

Header to use:

```python
#!/usr/bin/env python3
"""Extract a labelled Avatar Legends golden corpus from a clean game-capture VOD.

Frames are named `<state>[_p1-<n>_p2-<n>]_<idx>.png` and written to samples/avatar/.
Ground-truth labels come from a hand-verified clip (see the 2026-07-30 calibration
report); regenerate with:  python scripts/build_avatar_corpus.py <clean_avatar.mp4>
"""
```

- [ ] **Step 6: Write the calibration report** `docs/superpowers/reports/2026-07-30-avatar-calibration.md` containing every measured constant (as a copy-paste-ready Python block for Task 4), the measured empty-vs-lit gaps, the gate metric, and the ground-truth `(second, winner)` match-end list.

- [ ] **Step 7: Commit**

```bash
git add scripts/build_avatar_corpus.py samples/avatar docs/superpowers/reports/2026-07-30-avatar-calibration.md
git commit -m "feat: calibrate Avatar pip ROIs and build labelled corpus"
```

**Note for the controller:** this task needs visual judgement and iterative threshold tuning on real footage. If the implementer cannot achieve clean empty-vs-lit gaps (especially blue vs cyan), that is a BLOCKED signal to escalate — not a reason to weaken thresholds until tests pass.

---

### Task 4: `AvatarPipDetector` and registration

**Files:**
- Create: `src/fgc_detector/detectors/avatar.py`
- Modify: `src/fgc_detector/detectors/__init__.py`
- Modify: `config.example.toml`
- Test: `tests/detectors/test_avatar_pips.py`

**Interfaces:**
- Consumes: `color_fill_ratio` (Task 2), the measured constants from the Task 3 report, `Roi`, the `Screen`/`Observation`/`DETAIL_P*_ROUNDS` contract.
- Produces: `AvatarPipDetector` (registered under `Game.AVATAR`), exposing `observe`, `rois`, `supported_events`, `canonical_size`, `game`.

- [ ] **Step 1: Write the failing tests**

Create `tests/detectors/test_avatar_pips.py`. These build synthetic canonical frames by painting the measured pip ROIs with red/blue/dark and the emblem ROI dark, so each screen classification and winner is asserted against a known input. Import the measured ROIs/bands from `avatar.py` (single source of truth):

```python
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from fgc_detector.detectors.avatar import (
    AvatarPipDetector,
    CANONICAL_SIZE,
    EMBLEM_ROI,
    P1_PIP_1,
    P1_PIP_2,
    P2_PIP_1,
    P2_PIP_2,
)
from fgc_detector.detectors.registry import get_detector, register
from fgc_detector.detectors.roi import Roi
from fgc_detector.types import (
    DETAIL_P1_ROUNDS,
    DETAIL_P2_ROUNDS,
    EventType,
    Frame,
    Game,
    Screen,
)

RED = (0, 0, 255)     # BGR
BLUE = (255, 0, 0)    # BGR
DARK = (20, 20, 20)


def _blank() -> np.ndarray:
    # Mid-grey background that is neither a lit pip nor the dark emblem.
    return np.full((CANONICAL_SIZE[1], CANONICAL_SIZE[0], 3), 90, dtype=np.uint8)


def _paint(img: np.ndarray, roi: Roi, bgr: tuple[int, int, int]) -> None:
    img[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = bgr


def _frame(img: np.ndarray) -> Frame:
    return Frame(image=img, captured_at=datetime.now(timezone.utc))


def _in_match(p1_lit: int, p2_lit: int) -> np.ndarray:
    img = _blank()
    _paint(img, EMBLEM_ROI, DARK)  # HUD present
    for i, roi in enumerate((P1_PIP_1, P1_PIP_2)):
        _paint(img, roi, RED if i < p1_lit else DARK)
    for i, roi in enumerate((P2_PIP_1, P2_PIP_2)):
        _paint(img, roi, BLUE if i < p2_lit else DARK)
    return img


def test_no_emblem_reads_unknown() -> None:
    img = _blank()  # emblem region left mid-grey: HUD absent
    obs = AvatarPipDetector().observe(_frame(img))
    assert obs.screen is Screen.UNKNOWN


def test_zero_zero_is_in_match_with_zero_rounds() -> None:
    obs = AvatarPipDetector().observe(_frame(_in_match(0, 0)))
    assert obs.screen is Screen.IN_MATCH
    assert obs.details[DETAIL_P1_ROUNDS] == "0"
    assert obs.details[DETAIL_P2_ROUNDS] == "0"
    assert obs.winner is None


def test_one_each_is_in_match_no_winner() -> None:
    obs = AvatarPipDetector().observe(_frame(_in_match(1, 1)))
    assert obs.screen is Screen.IN_MATCH
    assert obs.details[DETAIL_P1_ROUNDS] == "1"
    assert obs.details[DETAIL_P2_ROUNDS] == "1"
    assert obs.winner is None


def test_two_pips_p1_is_match_end_p1() -> None:
    obs = AvatarPipDetector().observe(_frame(_in_match(2, 1)))
    assert obs.screen is Screen.MATCH_END
    assert obs.winner is Side.P1
    assert obs.details[DETAIL_P1_ROUNDS] == "2"


def test_two_pips_p2_is_match_end_p2() -> None:
    obs = AvatarPipDetector().observe(_frame(_in_match(0, 2)))
    assert obs.screen is Screen.MATCH_END
    assert obs.winner is Side.P2


def test_both_two_refuses_to_guess() -> None:
    obs = AvatarPipDetector().observe(_frame(_in_match(2, 2)))
    assert obs.screen is Screen.IN_MATCH
    assert obs.winner is None


def test_observe_is_pure() -> None:
    frame = _frame(_in_match(2, 0))
    d = AvatarPipDetector()
    assert d.observe(frame).payload == d.observe(frame).payload


def test_rois_within_canonical_bounds() -> None:
    w, h = CANONICAL_SIZE
    for roi in AvatarPipDetector().rois().values():
        assert 0 <= roi.x and 0 <= roi.y
        assert roi.x + roi.w <= w and roi.y + roi.h <= h


def test_supported_events_is_match_end_only() -> None:
    assert AvatarPipDetector().supported_events() == frozenset({EventType.MATCH_END})


def test_registered_for_avatar() -> None:
    d = AvatarPipDetector()
    register(d)  # autouse clean_registry fixture clears the import-time registration
    assert get_detector(Game.AVATAR) is d
```

Add `from fgc_detector.types import Side` to the imports (used by two tests).

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/detectors/test_avatar_pips.py -v`
Expected: FAIL — `ModuleNotFoundError: fgc_detector.detectors.avatar`.

- [ ] **Step 3: Implement the detector.** Create `src/fgc_detector/detectors/avatar.py`. Transcribe the constant block **verbatim from the Task 3 calibration report** (ROIs, hue bands, `SAT_MIN`, `VAL_MIN`, `PIP_LIT`, gate metric/threshold). The logic below is complete and final:

```python
"""Avatar Legends round-pip detector.

Avatar shows each player's round wins as two angular bars flanking the central
clock emblem: P1's two bars (left) fill red, P2's two (right) fill blue. Two
filled bars = that side won the match (best of 3). This reads the four bars by
colour in one frame and reports what it sees; the decision that a match ended
belongs to the marker Confirmer, not this pure detector.

Pips fill with a saturated colour (not brightness) and empty pips carry bright
cyan/gold outlines, so they are read with color_fill_ratio rather than the
brightness-based fill_ratio -- see docs/superpowers/specs/2026-07-30-avatar-
legends-detector.md and the 2026-07-30 calibration report for why these
constants, and how they were measured (never guessed).
"""

from __future__ import annotations

import cv2

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
from .roi import Roi, color_fill_ratio

#: Canonical resolution these ROIs are expressed in. Frames are normalised to
#: this before observe() runs.
CANONICAL_SIZE = (1920, 1080)

# --- MEASURED CONSTANTS: transcribe verbatim from the Task 3 calibration
# --- report (docs/superpowers/reports/2026-07-30-avatar-calibration.md).
EMBLEM_ROI = Roi(...)      # dark clock box, HUD-present anchor
CHAR_SELECT_ROI = Roi(...) # character-select marker
P1_PIP_1 = Roi(...)
P1_PIP_2 = Roi(...)
P2_PIP_1 = Roi(...)
P2_PIP_2 = Roi(...)

RED_HUE = (..., ...)   # (hue_lo, hue_hi), wrap-around near 0/179
BLUE_HUE = (..., ...)  # (hue_lo, hue_hi), tight band excluding cyan ~90
SAT_MIN = ...
VAL_MIN = ...
#: pip fill-ratio at or above which a bar counts as lit.
PIP_LIT = ...

#: emblem mean-grayscale at or below which the HUD is considered present.
EMBLEM_DARK_MAX = ...
#: char-select marker fill-ratio at or above which char select is showing.
CHAR_SELECT_PRESENT = ...

ROUNDS_TO_WIN = 2


class AvatarPipDetector:
    """Counts lit round pips by colour. Stateless and pure."""

    canonical_size = CANONICAL_SIZE
    game = Game.AVATAR

    def rois(self) -> dict[str, Roi]:
        return {
            "emblem": EMBLEM_ROI,
            "char_select": CHAR_SELECT_ROI,
            "p1_pip_1": P1_PIP_1,
            "p1_pip_2": P1_PIP_2,
            "p2_pip_1": P2_PIP_1,
            "p2_pip_2": P2_PIP_2,
        }

    def supported_events(self) -> frozenset[EventType]:
        return frozenset({EventType.MATCH_END})

    def _lit(self, image, rois, hue) -> tuple[int, list[float]]:
        ratios = [
            color_fill_ratio(
                image, roi, hue_lo=hue[0], hue_hi=hue[1],
                sat_min=SAT_MIN, val_min=VAL_MIN,
            )
            for roi in rois
        ]
        return sum(1 for r in ratios if r >= PIP_LIT), ratios

    def _char_select_present(self, image) -> float:
        from .roi import fill_ratio
        return fill_ratio(image, CHAR_SELECT_ROI)

    def _emblem_mean(self, image) -> float:
        patch = EMBLEM_ROI.crop(image)
        if patch.size == 0:
            return 255.0
        return float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).mean())

    def observe(self, frame: Frame) -> Observation:
        image = frame.image

        # Character select first: it is the Confirmer's primary cooldown exit,
        # so a frame that could read either way must resolve to CHAR_SELECT.
        cs = self._char_select_present(image)
        if cs >= CHAR_SELECT_PRESENT:
            return Observation(screen=Screen.CHAR_SELECT, confidence=cs)

        emblem_mean = self._emblem_mean(image)
        if emblem_mean > EMBLEM_DARK_MAX:
            return Observation(screen=Screen.UNKNOWN, debug={"emblem_mean": emblem_mean})

        p1_lit, p1_ratios = self._lit(image, (P1_PIP_1, P1_PIP_2), RED_HUE)
        p2_lit, p2_ratios = self._lit(image, (P2_PIP_1, P2_PIP_2), BLUE_HUE)
        debug = {
            "emblem_mean": emblem_mean,
            "p1_ratios": p1_ratios,
            "p2_ratios": p2_ratios,
        }
        details = {DETAIL_P1_ROUNDS: str(p1_lit), DETAIL_P2_ROUNDS: str(p2_lit)}

        p1_won = p1_lit >= ROUNDS_TO_WIN
        p2_won = p2_lit >= ROUNDS_TO_WIN
        if p1_won == p2_won:
            # Neither done, or both read done (impossible in a real match ->
            # a misread). Refuse to guess a winner.
            return Observation(screen=Screen.IN_MATCH, details=details, debug=debug)

        winner = Side.P1 if p1_won else Side.P2
        winner_ratios = p1_ratios if p1_won else p2_ratios
        return Observation(
            screen=Screen.MATCH_END,
            winner=winner,
            confidence=min(winner_ratios),
            details=details,
            debug=debug,
        )


register(AvatarPipDetector())
```

If Task 3 chose a non-darkness HUD gate, replace `_emblem_mean` + its threshold check with the measured gate; keep the char-select-first / refuse-to-guess structure exactly.

- [ ] **Step 4: Register on import.** In `src/fgc_detector/detectors/__init__.py`:

```python
from . import avatar  # noqa: F401
from . import sf6  # noqa: F401
```

- [ ] **Step 5: Add Avatar to the example roster.** In `config.example.toml`, `enabled_games`:

```toml
enabled_games = ["sf6", "tekken8", "avatar"]
```

- [ ] **Step 6: Run to verify they pass**

Run: `uv run pytest tests/detectors/test_avatar_pips.py -v`
Expected: PASS (all).

- [ ] **Step 7: Commit**

```bash
git add src/fgc_detector/detectors/avatar.py src/fgc_detector/detectors/__init__.py config.example.toml tests/detectors/test_avatar_pips.py
git commit -m "feat: add AvatarPipDetector reading round pips by colour"
```

---

### Task 5: Corpus regression + confirmer integration + replay validation

**Files:**
- Test: `tests/detectors/test_avatar_corpus.py` (new)
- Modify: `docs/superpowers/reports/2026-07-30-avatar-calibration.md` (append replay result)

**Interfaces:**
- Consumes: the committed corpus (`samples/avatar/`), `AvatarPipDetector`, the marker `Confirmer`, `run_offline`/`VideoFrameSource` for the manual replay check.

- [ ] **Step 1: Write the corpus regression test.** Model on `tests/detectors/test_sf6_counter.py`: parametrize over `samples/avatar/*.png`, asserting each `in_match_p1-<a>_p2-<b>_*.png` frame reads `IN_MATCH` (or `MATCH_END` when a side is at 2) with the labelled pip counts, and each `between_*.png` frame reads `UNKNOWN` or `CHAR_SELECT` with no round-count details. Use a strict `xfail` (with a written reason) for any documented corpus-label conflict, exactly as SF6 does — never weaken a general assertion to absorb one frame.

```python
_IN_MATCH_PATTERN = re.compile(r"^in_match_p1-(\d)_p2-(\d)_\d+\.png$")
# ... load CORPUS_DIR = .../samples/avatar, build cases, assert real ground truth ...
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/detectors/test_avatar_corpus.py -v`
Expected: PASS (every corpus frame classified correctly). If a real frame fails, fix the Task 3 constants (re-measure) — do not weaken the test.

- [ ] **Step 3: Write the confirmer-integration test.** Feed a hand-built observation sequence through the marker `Confirmer` (armed) and assert exactly one `MATCH_END` fires with the right winner. This uses synthetic `Observation`s (no images, no clock beyond injected timestamps), proving the detector's contract drives the reused confirmer end-to-end:

```python
def test_pip_sequence_fires_one_match_end_for_p1() -> None:
    from datetime import datetime, timedelta, timezone
    from fgc_detector.confirmer import Confirmer, ConfirmerConfig
    from fgc_detector.types import (
        DETAIL_P1_ROUNDS, DETAIL_P2_ROUNDS, Observation, Screen, Side, Game,
    )
    c = Confirmer(Game.AVATAR, ConfirmerConfig(agreement_frames=3))
    c.arm()
    t = datetime(2026, 7, 30, tzinfo=timezone.utc)
    def obs(screen, p1, p2, winner=None):
        return Observation(
            screen=screen, winner=winner,
            details={DETAIL_P1_ROUNDS: str(p1), DETAIL_P2_ROUNDS: str(p2)},
        )
    events = []
    # in-match 1-1, then P1 reaches 2 for >= agreement_frames frames
    seq = ([obs(Screen.IN_MATCH, 1, 1)] * 3
           + [obs(Screen.MATCH_END, 2, 1, Side.P1)] * 4)
    for i, o in enumerate(seq):
        e = c.observe(o, t + timedelta(seconds=i))
        if e is not None:
            events.append(e)
    assert len(events) == 1
    assert events[0].winner is Side.P1
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest`
Expected: PASS (existing SF6/engine tests plus the new Avatar tests). Confirms Avatar registration did not disturb the rest.

- [ ] **Step 5: Manual replay validation (documented, not a pytest that needs the VOD).**

Run: `uv run fgc-detect replay --game avatar --video ~/repos/avatar.mp4`
Confirm the emitted `match_end` sequence (winners + timestamps) matches the ground-truth list from Task 3 (a few seconds of confirmation lag is fine; no spurious events mid-round or on transitions). Paste the command output and the pass/fail judgement into the calibration report.

- [ ] **Step 6: Commit**

```bash
git add tests/detectors/test_avatar_corpus.py docs/superpowers/reports/2026-07-30-avatar-calibration.md
git commit -m "test: corpus + confirmer regression and replay validation for Avatar"
```

---

## Out of scope (now-live hazard, tracked separately)

Avatar is the first registered game to use the marker `Confirmer` while SF6 uses `SetScoreConfirmer`, which makes the documented runtime `set_game` strategy-swap hazard (`docs/TODO.md`) genuinely triggerable: switching active game between `sf6` and `avatar` at runtime will not rebuild the confirmer, so detection silently stops until restart. Starting under either game works correctly. Fixing the runtime swap is a separate change (rebuild the confirmer in `EventServer._apply` / the run loop when the active game's strategy changes) and is intentionally **not** in this plan. Flag to the user after implementation.
