# Avatar Legends Detector — Calibration Report (Task 3)

Source footage: `~/repos/avatar.mp4` (1280x714, 29.97 fps, ~505.8s). All coordinates
below are in **canonical 1920x1080** pixels (frames are `normalize()`d to this
before any ROI is sampled, matching `scripts/build_sf6_corpus.py`'s convention).

## 1. What the pips actually are (verified, corrects an assumption in the plan)

The plan's starting description ("two stacked angular bars immediately flanking
the clock emblem") is correct in spirit but the bars are much smaller and sit
**inside the same dark hexagonal frame as the round-timer digits**, immediately
to the left (P1) and right (P2) of the two-digit countdown number — not in the
larger "wing"-shaped bar further out (that wing bar is a separate hyper/chi
resource meter that continuously drains and refills during a round and is
**not** related to round score — confirmed by watching it fluctuate freely
mid-round while the round score pips stayed constant).

Each pip is a small vertical rectangle (~12x18px canonical) with a tan/gold
border. Empty = dark interior (with a fair amount of ambient stage-colour
bleed-through, see §3). Lit = solid saturated red (P1) / blue (P2) fill,
noticeably higher brightness (V) than any empty state observed.

## 2. Measured ROIs (canonical 1920x1080)

```python
CANONICAL_SIZE = (1920, 1080)

EMBLEM_ROI = Roi(930, 45, 60, 20)       # narrow band inside the dark hexagon,
                                         # above the countdown digits, clear of
                                         # both pip columns
P1_PIP_1 = Roi(892, 85, 12, 18)         # P1 (left), top bar
P1_PIP_2 = Roi(892, 115, 12, 18)        # P1 (left), bottom bar
P2_PIP_1 = Roi(1014, 83, 12, 18)        # P2 (right), top bar
P2_PIP_2 = Roi(1014, 113, 12, 18)       # P2 (right), bottom bar
```

`CHAR_SELECT_ROI` could **not** be measured: no character-select screen ever
appears in the ~505s of footage (matches go results-menu -> title card ->
next match directly). See §6 (BLOCKED note, scoped).

ROIs were located by: extracting frames at 1s/3s cadence across all ~505s,
zooming 5-10x with a pixel grid overlaid, and reading exact bracket edges
directly off the pixel grid (see method in-session; not re-derived from
guesswork). Verified against 3 separate stages (fire-nation orange, rocky
gray, mystic green) to confirm the ROI is stage-independent.

## 3. Colour bands and thresholds

```python
RED_HUE = (170, 20)     # wrap-around band near 0/179 (P1 lit red)
BLUE_HUE = (95, 140)    # tight band excluding cyan (~90) (P2 lit blue)
SAT_MIN = 60
VAL_MIN = 150
PIP_LIT = 0.4            # color_fill_ratio threshold; lit samples cluster
                          # 0.68-1.00, empty samples cluster 0.00-0.06
```

**Primary measured risk was not blue-vs-cyan — it was red-vs-stage-bleed.**
The pip's "empty" interior is not neutral; on the orange fire-nation stage an
empty pip reads a warm, moderately-saturated hue (H~10-20, S~60-125) that is
hue-similar to a genuinely lit P1 pip. The real separator turned out to be
**brightness (V)**: every measured *lit* pip (red or blue) reads V~175-192;
every measured *empty* pip (across all 3 stages) reads V~55-129. `VAL_MIN=150`
sits cleanly in that gap and does the real discriminating work; `SAT_MIN=60`
and the hue bands are still required to tell P1's red from P2's blue and to
reject stray bright-but-neutral pixels (anti-aliased tan borders, etc).

### Measured empty-vs-lit fill-ratio gap (from HSV means during calibration)

| state | ROI | H | S | V |
|---|---|---|---|---|
| P1 lit (red), fire stage | P1_PIP_1 | 8-15 | 92-126 | 175-192 |
| P1 empty, fire stage (bleed) | P1_PIP_1/2 | 11-19 | 61-125 | 68-128 |
| P1 empty, rocky stage | P1_PIP_1/2 | unstable (low sat) | 20-60 | 67-113 |
| P2 lit (blue) | P2_PIP_1/2 | 107-115 | 64-77 | 175-186 |
| P2 empty, any stage | P2_PIP_1/2 | unstable (low sat) | 33-109 | 59-97 |

### Self-validation over the committed corpus (`samples/avatar/*.png`)

Ran `color_fill_ratio` with the constants above over all 100 pip
reads (4 pips x 25 `in_match_*` corpus frames):

```
lit samples: 27   empty samples: 73
MIN lit fill ratio:   0.676  (in_match_p1-0_p2-2_0002.png, P2_PIP_2)
MAX empty fill ratio: 0.060  (in_match_p1-0_p2-0_0004.png, P1_PIP_2)
MARGIN (lit_min - empty_max): 0.616
```

Every empty pip in the corpus reads below `PIP_LIT=0.4`; every lit pip reads
at/above it, with a 0.616 margin — clean separation, no borderline cases.

## 4. HUD-present gate

Measured mean grayscale of `EMBLEM_ROI` (a text/pip-free strip inside the
hexagon):

- In-match (23 corpus frames, 3 stages): mean gray **52-66** (max 65.6).
- Sustained between-screens (results menu, story dialogue): mean gray
  **104-112** — cleanly separated, no overlap with in-match.
- Brief transitional between-screens (black KO wipe, "BALANCE MUST BE
  RESTORED" title card): mean gray **22** (wipe) or **68** (title card) — the
  wipe is clearly dark, and the title card sits *inside* the in-match range.

**`EMBLEM_DARK_MAX = 80`** cleanly separates in-match play from the
*sustained* between-screens (results/dialogue), which is what actually
matters for not firing false readings during a lingering menu. The brief
title-card and KO-wipe frames (each lasting ~1-2s) may transiently read as
"HUD present" under this threshold — this is a measured, documented gap, not
a guess-covered one. It is harmless: no pip graphic is drawn during those
frames, so `color_fill_ratio` reads ~0 for all four pips regardless, giving
`IN_MATCH 0-0` (never a false `MATCH_END`), consistent with the fail-safe
constraint. A brightness-only gate could not do better here without
over-fitting to this specific title-card's incidental background luminance;
Task 4 should treat this as an accepted limitation rather than re-guess a
threshold to "fix" it.

```python
EMBLEM_DARK_MAX = 80.0   # emblem mean-gray <= this => HUD present
```

## 5. Ground truth (match-end second, winner)

Three complete matches were traced end-to-end in the footage (player
`renatomrcosta` on Katara = P1/left/red; `Than0ss` on Zuko = P2/right/blue):

| match | stage | rounds | winner | 2nd-pip-lit second | notes |
|---|---|---|---|---|---|
| 1 | fire-nation (orange) | went to Round 3 (1-1 split, decided in the 3rd) | **P1** | not directly observable (see below) | results screen at t=138s reads "renatomrcosta WON 1 / Than0ss LOST" |
| 2 | rocky (gray) | 2-0 sweep | **P2** | **t≈228.7s** | `P2_PIP_1`/`P2_PIP_2` both cross `PIP_LIT` at t=228.66s; FINISH banner ~229-233s |
| 3 | fire-nation (orange) | 2-0 sweep | **P1** | **t≈323.0s** | `P1_PIP_1`/`P1_PIP_2` both cross `PIP_LIT` at t=322.99s; FINISH banner ~326-333s |

A 4th and 5th match continue past ~335s on a 4th (green/mystic) stage but were
not traced to completion — not needed once 3 clean, cross-validated matches
(covering both winners, 3 different stages, and both a sweep and a
split-then-decided match) were confirmed.

### Known limitation: match 1's decisive round never visibly shows "2 lit"

For match 1 (decided in round 3 after a 1-1 split), the pip UI was traced
frame-by-frame (0.1s steps) from t=129s through the "FINISH" freeze and the
wipe to the results screen at t=138s. It **stays at 1-1 the entire time** —
the winning side's second pip never renders on screen before the game cuts to
the post-match menu. This is a genuine, measured game-UI behavior, not a
detector bug: matches 2 and 3 (sweeps) *do* show the clean 2-lit state before
FINISH, so the pip mechanic and this calibration's constants are correct; it
is specifically the "decided on the last possible round" case where the UI's
final-state animation appears to be skipped or hidden behind the transition.

Consequence for Task 4/5: the marker `Confirmer`, which requires several
frames of agreement before firing, will correctly **not** fire a `match_end`
for a match that ends this way — it will simply miss it (no event emitted).
Per this plan's fail-safe constraint ("any ambiguous reading resolves to no
event, never a false winner"), this is the correct, safe behavior, but it
should be flagged to the user as a real coverage gap: **matches decided in
their final possible round may not produce a `match_end` event**, only
matches decided with a round to spare (i.e. 2-0, or presumably 3-1/3-2 in a
best-of-5) reliably will. This is worth a follow-up investigation with more
footage before Avatar detection is trusted in production, but is out of scope
for this calibration task.

## 6. Corpus

`scripts/build_avatar_corpus.py` (modeled on `scripts/build_sf6_corpus.py`)
extracts 32 frames into `samples/avatar/`:

```
between: 7
in_match_p1-0_p2-0: 10   (0-0, two stages: rocky + fire-nation)
in_match_p1-0_p2-1: 3    (0-1, match 2 round 2)
in_match_p1-1_p2-1: 5    (1-1, match 1 round 3)
in_match_p1-2_p2-0: 4    (2-0 P1 win, match 3 finish)
in_match_p1-0_p2-2: 3    (0-2 P2 win, match 2 finish)
```

`between` frames cover: story-mode dialogue (t=0,2s), the black KO wipe
(t=85s, t=235s, t=335s), the "BALANCE MUST BE RESTORED" title card (t=141s),
and the post-match results/menu screen (t=139s... actually the wipe;
corpus file `between_0001/0002` = dialogue, `0003/0004` = black wipes,
`0005` = title card, `0006/0007` = more black wipes).

No `CHAR_SELECT` frame is included because no such screen exists in the
source footage (§2). `CHAR_SELECT_ROI`/`CHAR_SELECT_PRESENT` are **not**
calibrated by this task; Task 4 should either omit that branch for Avatar or
treat all non-dark, non-pip-populated frames as `UNKNOWN` until real
char-select footage is available.

## 7. Status: DONE (with one flagged, scoped limitation)

All required constants were measured with clean margins:

- `PIP_LIT = 0.4`, `SAT_MIN = 60`, `VAL_MIN = 150`, `RED_HUE = (170, 20)`,
  `BLUE_HUE = (95, 140)` — corpus self-validation margin **0.616** (lit min
  0.676 vs empty max 0.060).
- `EMBLEM_DARK_MAX = 80.0` — cleanly separates in-match from sustained
  between-screens; brief title-card/wipe frames may transiently misread as
  "present" (documented, harmless per fail-safe design).
- `CHAR_SELECT_ROI` — **not measurable**, no such screen in the footage
  (flagged, not guessed).
- Ground truth: 3 traced matches, both winners represented, 2 clean 2-pip
  sweeps (t≈228.7s P2 win, t≈323.0s P1 win) plus 1 documented case where the
  UI never shows the deciding pip (t=138s, P1 win by result-screen only) —
  flagged as a coverage gap for Task 5's replay validation to expect.
