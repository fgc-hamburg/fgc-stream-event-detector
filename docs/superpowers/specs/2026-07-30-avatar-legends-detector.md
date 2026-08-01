# Avatar Legends Detector — Design

**Date:** 2026-07-30
**Status:** Approved (from live investigation of real footage + user decisions), building now
**Relates to:** the v1 engine and the SF6 counter detector
([`2026-07-22-sf6-counter-detector.md`](2026-07-22-sf6-counter-detector.md)). This adds a third game
and follows the same "each game detects its own way, shared confirmer + contract" pattern.

## Why a dedicated detector (not `MarkerRoundDetector`)

Avatar Legends *is* a round-pip game, but its pips do not fit the existing brightness-based
`MarkerRoundDetector` in two ways, so per that class's own guidance ("a game whose HUD does not fit
this shape should implement the `Detector` protocol directly rather than bending this class") it gets
its own detector:

1. **Pips fill with saturated colour, not brightness.** A lit P1 pip is red, a lit P2 pip is blue,
   and *empty* pips carry bright cyan/gold outlines. `fill_ratio` (fraction of pixels brighter than a
   grey threshold) gets this backwards: pure red greyscales to ~76 and blue to ~29 (both below the
   128 threshold), while the empty cyan outline is bright. Avatar pips must be read **by colour, per
   side**.
2. **The "match HUD present" gate is a dark element, not a bright one.** `MarkerRoundDetector` gates
   `IN_MATCH` on a bright health bar. Avatar gates on the **clock emblem** — a dark hexagonal box
   with white digits at top-centre — which is reliably present during active play and absent on
   character-select, victory, and loading screens.

What is **reused**, unchanged: the `Roi` primitive, the marker `Confirmer` (all temporal logic), the
`Screen` / `DETAIL_P1_ROUNDS` / `DETAIL_P2_ROUNDS` contract, and the registry / CLI / server (already
game-agnostic). One new **reusable** helper — `color_fill_ratio` — is added to `roi.py` for any
future colour-pip game.

## Game model (confirmed with the user)

- Best-of-3 rounds; **2 round pips filled = that side won the match.**
- Pips are the **stacked angular bars immediately flanking the clock emblem** — two per side. Left
  side is **P1 (fills red)**, right side is **P2 (fills blue)**. (The circles *below* the life bars
  are a per-round resource gauge, NOT round wins — they reset every round and must never be sampled.
  The "WINS N" text is irrelevant and deliberately unused, keeping detection language-independent.)
- Empty pip: dark interior with a cyan/gold outline. Filled pip: saturated red (P1) or blue (P2).

## What "match end" means here

**Per-match** (user decision, 2026-07-30): fire exactly one `match_end` event per match, naming the
side that reached 2 filled pips. This is the natural granularity for a pip-to-N game and matches how
the user described "a WIN." Unlike SF6 (per-game increment on the *next* game's HUD), the deciding
pip is visible on the live match HUD, so — unlike SF6 — **the set/match-deciding event is detected**;
there is no missed-final-game limitation here.

## Components

### 1. `Game.AVATAR` enum value

Add `AVATAR = "avatar"` to the `Game` enum in `types.py`. Every closed-set value stays an enum; no
bare game strings anywhere.

### 2. `color_fill_ratio(image, roi, hue_lo, hue_hi, sat_min, val_min)` in `roi.py`

The colour-aware analogue of `fill_ratio`. Converts the ROI to HSV and returns the fraction of pixels
whose hue falls in `[hue_lo, hue_hi]` (OpenCV hue scale 0–179, wrapping supported for red) **and**
whose saturation ≥ `sat_min` and value ≥ `val_min`. Degrades to `0.0` on an out-of-frame ROI, like
every other `roi.py` primitive. This distinguishes a saturated red/blue *fill* from both the dark
empty interior (low S) and the bright cyan outline / cyan health-bar edge (different hue). Reusable,
not Avatar-specific.

### 3. `AvatarPipDetector` (stateless, implements `Detector`)

`canonical_size = (1920, 1080)`; frames are normalised before it runs. Per frame, `observe` returns
an `Observation` mirroring `MarkerRoundDetector`'s contract so the marker `Confirmer` treats it
identically:

- **Gate:** if the clock-emblem ROI is not present (emblem-darkness test fails), return
  `Screen.UNKNOWN` — no match HUD. This is checked so transitions, victory screens, and
  character-select do not read as a match.
- **Character select:** a `char_select` ROI present ⇒ `Screen.CHAR_SELECT` (the Confirmer's primary
  cooldown-release signal). Checked before `IN_MATCH`, same precedence as the marker detector.
- **Pip counting:** `color_fill_ratio` over 2 P1 (red) ROIs and 2 P2 (blue) ROIs; a pip is "lit" at
  or above a per-colour threshold. `p1_lit` / `p2_lit` ∈ {0, 1, 2}.
- **Details:** publish `DETAIL_P1_ROUNDS = str(p1_lit)` and `DETAIL_P2_ROUNDS = str(p2_lit)` on
  every `IN_MATCH` and `MATCH_END` observation — this is the Confirmer's 0-0 cooldown-release path
  (a fresh match resets pips to 0-0). Never bare string literals; use the shared constants.
- **Outcome:** `p1_won = p1_lit >= 2`, `p2_won = p2_lit >= 2`. If exactly one side won ⇒
  `Screen.MATCH_END`, `winner` = that side, `confidence` = min of that side's two pip fill ratios.
  If neither or (impossibly) both ⇒ `Screen.IN_MATCH`, refuse to guess a winner.
- `supported_events()` ⇒ `{MATCH_END}`. `rois()` exposes every sampled rectangle
  (`clock`, `char_select`, `p1_pip_1/2`, `p2_pip_1/2`) for the `roi` CLI preview.

### 4. Confirmer + factory

Reuse the existing marker `Confirmer` (reach-N-pips → `MATCH_END`, N-frame agreement, cooldown held
until CHAR_SELECT or 0-0 pips or the safety valve). `make_confirmer(Game.AVATAR, config)` returns it,
exactly as it does for a marker game. Register `AvatarPipDetector` in the detector registry.

## Calibration (de-risking task — the real work)

Exact pip ROIs, the clock/char-select gate ROIs, the red/blue hue bands, and the "lit" thresholds
are **measured from `avatar.mp4`, never guessed** (per project rule). This is its own task, mirroring
the SF6 counter prototype:

1. Build a reproducible corpus (`scripts/build_avatar_corpus.py`) of labelled frames at known pip
   states: 0-0, 1-0, 1-1, 2-0 (P1 win), 0-2 / 1-2 (P2 win), plus char-select and transition frames.
2. Diff a real 0-0 start frame against real match-end frames to locate the pip bars precisely and
   separate them from the cyan health-bar edge and the resource-gauge circles.
3. Measure HSV distributions in the pip ROIs across states to set hue bands, `sat_min`/`val_min`, and
   the lit threshold with a clean gap (empty vs red vs blue).
4. Derive the ground-truth match-end timestamps from the corpus (user delegated this).

**Primary risk:** cleanly separating a filled *blue* P2 pip (hue ~110–130 in OpenCV) from the empty
cyan outline / cyan health-bar edge (hue ~90). Red (P1) is far from cyan and low-risk. Fallback if
the blue/cyan gap is too tight: tighten the P2 pip ROI to the bar interior only, and/or require a
higher `sat_min`. Documented here so the implementer treats it as the calibration focus.

## Validation target

`fgc-detect replay --game avatar --video ~/repos/avatar.mp4` emits one `match_end` per match with the
correct winner, at the timestamp each match is decided (a few frames of confirmation lag is fine). No
events mid-round, on KOs that only end a round, on the resource-gauge circles, or on transitions. The
exact expected sequence is fixed once the corpus is built and becomes the regression baseline.

## Testing

Hermetic, as everywhere else — no OBS, GPU, network, or real clock:

- `color_fill_ratio`: synthetic HSV patches (red / blue / cyan / dark) assert the fraction and the
  hue/sat/val gating, including red hue-wrap and out-of-frame degradation to 0.0.
- `AvatarPipDetector.observe`: synthetic frames built from coloured rectangles at the pip ROIs assert
  each screen classification and winner (0-0 → IN_MATCH, 2-0 → MATCH_END P1, 0-2 → MATCH_END P2,
  both-won → IN_MATCH/no-guess, emblem-absent → UNKNOWN, char-select → CHAR_SELECT), and that the
  round-count details are published.
- Confirmer reuse: covered by the existing marker `Confirmer` suite; add an Avatar-parametrised case
  only where the game value matters.
- Corpus regression: `AvatarPipDetector` over the labelled real frames reproduces the expected
  per-frame classifications; a replay test asserts the match-end sequence.

Every test can fail for the right reason: a broken threshold, wrong ROI, or wrong winner flips an
assertion. No test requires the real video to be present at run time beyond the committed corpus PNGs.

## Known limitations

- **Blue/cyan proximity** (see Calibration) is the one genuine reading risk; mitigated by ROI
  tightening and the fallback above, and — like everything in this system — it fails safe: an
  ambiguous pip reads as "not lit," yielding a missed event, never a false winner.
- **Stage/round stage-swaps:** Avatar can change stage between rounds of the same match. Because pips
  are read from the fixed HUD overlay (not the stage), this does not affect detection, but the corpus
  should include at least one such frame to prove it.
