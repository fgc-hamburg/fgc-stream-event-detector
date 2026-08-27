# Marvel TOKON Detector — Design

**Date:** 2026-08-25
**Status:** Approved (from measurement of real footage + user decisions), building now
**Relates to:** the v1 engine, the SF6 counter detector
([`2026-07-22-sf6-counter-detector.md`](2026-07-22-sf6-counter-detector.md)), and the Avatar Legends
pip detector ([`2026-07-30-avatar-legends-detector.md`](2026-07-30-avatar-legends-detector.md)).
This is the fourth game and the second pip-counting game; it follows Avatar's shape closely.

## Round pips are the default; SF6 is the exception

A decision to record once, because it changes how the next game gets read (user, 2026-08-25):
**counting round-win pips is this project's default detection approach.** SF6's games-won-in-set
digit counter is the *exception*, forced by SF6's HUD, not a pattern to imitate. A new game starts
from "where are its round pips, and how do they change when a round is won?" and only leaves that
path when the HUD genuinely offers nothing pip-shaped.

The README and `CLAUDE.md` are updated to say this plainly (see [Documentation](#documentation)).

## Game model (measured from footage, confirmed by the user)

Marvel TOKON shows **three round pips per side**, in a row flanking the central match clock, above
each player's health bar. They fill from the centre-adjacent pip outward as rounds are won.

- **First to 3 rounds.** All three pips lit on one side = that side won the match. Confirmed by the
  user, 2026-08-25, and consistent with all four reference stills.
- An **empty** pip is a **small white circle**.
- A **lit** pip is that white circle **replaced by an icon** — a larger disc carrying a star and a
  letter. **The icon can be one of many colours** (user, 2026-08-25). This fact comes from the user,
  not from measurement: the only icon colours *observed* anywhere in the supplied media are orange-red
  "P" (hue ≈ 11) and yellow "V" (hue ≈ 25). Those two must **not** be mistaken for the palette — a
  scan of the full video turned up no third colour, which is evidence about this clip's coverage, not
  about the game. Calibration is expected to widen the observed set, and the committed test must not
  depend on it.
- Icons are **not** side-coded — unlike Avatar's red P1 / blue P2, the same icons appear on both
  sides, so P1 and P2 are separated by **position only**.
- The letters are presumably a win-type flourish (perfect vs. ordinary); detection ignores them
  entirely, which keeps it language-independent.

The detection question is therefore **"is the small white circle still there, or has an icon taken
its place?"** — a question about the marker's *identity*, not its colour.

Reference media (all supplied by the user, under `~/repos/tokon/`): `TOKON.mp4` (1280×714, 29.97fps,
~683s, containing both a P1 and a P2 match win), and four stills — `vlcsnap-00001.png` (0-0),
`vlcsnap-00002.png` (P1 2, P2 1), `vlcsnap-00003-p1-win.png` (P1 3, P2 2), `vlcsnap-00004-p2-win.png`
(P1 2, P2 3).

## Why a dedicated detector, not `MarkerRoundDetector`

Same reason as Avatar, and worth restating because the numbers are stark. `MarkerRoundDetector` uses
brightness `fill_ratio`, and TOKON's pips are **inverted** under that test: measured across the four
stills, empty pips have a median value of ~168 while lit pips sit at ~133, because an empty pip is a
small white circle and a lit one is a mid-tone icon disc. Brightness sampling would read every
empty pip as *more* filled than a lit one. Per that class's own guidance — "a game whose HUD does not
fit this shape should implement the `Detector` protocol directly rather than bending this class" —
TOKON gets its own module, modeled on `avatar.py` (user decision, 2026-08-25).

What is **reused unchanged**: `Roi`, the marker `Confirmer` (all temporal logic), the `Screen` /
`DETAIL_P1_ROUNDS` / `DETAIL_P2_ROUNDS` contract, and the registry / CLI / server. Which `roi.py`
sampling primitive TOKON uses is decided by the bake-off in
[What discriminates lit from empty](#what-discriminates-lit-from-empty) — `match_template` already
exists; the other candidates need a new *general* primitive added to `roi.py`, not a TOKON-specific
helper. Notably `color_fill_ratio`, the primitive Avatar contributed, is **not** reusable here.

`MarkerRoundDetector` therefore still has **zero registered games** and three hand-rolled pip
detectors around it. That is a real design smell but explicitly **out of scope here** — recorded in
`docs/TODO.md` instead.

## What "match end" means here

One `match_end` event per match, naming the side that reached three lit pips. As with Avatar and
unlike SF6, the deciding pip lights on the **live match HUD**, so the match-deciding event *is*
detected — there is no missed-final-game limitation.

## What discriminates lit from empty

This is the crux of the detector and the part most likely to be got wrong, so the evidence is
recorded here in full. All figures below come from probes on normalized 1920×1080 frames at
**eyeballed** pip ROIs — they are feasibility evidence for choosing an approach, **not** the
calibration. Committed constants come from the calibration task and may differ.

### Two approaches already ruled out, with data

**A warm hue band is out**, on the user's account of the game rather than on measurement. It looks
perfect on the four stills (lit 0.68–0.80, empty 0.000), but only because every icon in those stills
is warm. Since the icon palette is not limited to warm colours, a warm band would silently score
every cool-coloured icon as empty — a **missed** match end. The clip contains no cool icon to
demonstrate this with, which is exactly why the rule is written down rather than discovered later.

**A hue-agnostic saturation test is also out.** Dropping the hue gate and asking only "is this pip
box saturated?" produces measured **false positives**: at t=28s, t=126s and t=128s the scan reports
0.6+ on pips that are all visibly **empty**. The stage is a blue sky, and it bleeds through the box
around the small white circle. Background saturation defeats the test.

### The near-white core probe

Testing the reframing directly — look for the white circle, not the colour — over **54 hand-labelled
pips** (14 lit, 40 empty) drawn from the four stills plus the blue-sky frames:

| test | lit range | empty range | margin |
|---|---|---|---|
| centre saturation, r=11 | 99 – 126 | 7 – 105 | −6.5 ✗ |
| centre saturation, r=6 | 55 – 110 | 6 – 105 | −50 ✗ |
| **near-white fraction, r=6** | 0.000 – 0.035 | 0.056 – 0.868 | **+0.021** ✓ |
| near-white fraction, r=4 | 0.000 – 0.000 | 0.000 – 0.984 | 0.000 ✗ |

Only the near-white core test separates, only at one radius, by about three pixels' worth of margin.
Real, but far too thin to commit to blind.

### Fail-safe constraint: "lit" must rest on positive evidence

Testing for the *empty* marker inverts this project's fail-safe direction, and that inversion is not
acceptable on its own. Today, "I cannot see it clearly ⇒ not lit ⇒ no event" yields a missed match
end, which the operator recovers from. If lit were defined as the *absence* of the white circle, a
pip obscured by a super flash, hit spark, or screen-wide effect would read as **lit** — and three of
them would fire a false `match_end`, which corrupts the scoreboard.

**Whatever test is chosen, `lit` must require positive evidence that an icon is present, never merely
the failure to find the white circle.** In practice this means a two-sided decision: the white circle
is absent **and** something icon-shaped is present. A pip that satisfies neither reads as *not lit*.

### The calibration bake-off (user decision, 2026-08-25)

Rather than commit to a primitive now, calibration **measures these candidates against the labelled
corpus and commits the one with the widest margin**:

1. **Near-white core absence** — fraction of low-saturation, high-value pixels in a tight centre ROI.
   The only candidate that separated in the probe. Needs a new `roi.py` primitive (a "pale fill"
   test: saturation *below* a max, value above a min), since `color_fill_ratio` cannot express an
   upper saturation bound.
2. **Template match on the empty-circle glyph**, via the existing `match_template`. `TM_CCOEFF_NORMED`
   is contrast- and brightness-normalized, so it tolerates a tinted background far better than any
   absolute threshold. A high score means empty.
3. **Annulus-versus-surround contrast** — compare the ring a lit icon covers but the small circle does
   not, against the background immediately outside the pip. Background-independent *by construction*,
   which is precisely what defeated candidate approaches above. Most new code.
4. **Combinations**, in particular (1 or 2) for the empty side paired with (3) as the positive
   icon-present evidence the fail-safe constraint demands.

**A thin margin is a blocker to report, not a threshold to loosen.** If no candidate or combination
separates every labelled state with room to spare, that is a finding to bring back, not something to
tune around.

## Components

### 1. `Game.TOKON` enum value

Add `TOKON = "tokon"` to the `Game` enum in `types.py`. Every closed-set value stays an enum.

### 2. `TokonPipDetector` in `src/fgc_detector/detectors/tokon.py`

Stateless, pure, implements `Detector`. `canonical_size = (1920, 1080)`; frames are normalized
before it runs. `observe()` returns an `Observation` mirroring the marker contract so the marker
`Confirmer` treats it identically to Avatar:

- **HUD gate:** if the HUD-present ROI fails its test, return `Screen.UNKNOWN`. This is what keeps
  team select, K.O. banners, results screens, and title cards from reading as a match. See
  [The HUD-present gate](#the-hud-present-gate) — this is the one genuinely unmeasured piece.
- **Character select:** if a `char_select` ROI is present ⇒ `Screen.CHAR_SELECT`, checked **before**
  `IN_MATCH`, same precedence as `MarkerRoundDetector` (it is the Confirmer's cooldown exit, so a
  frame that could read as either must resolve to `CHAR_SELECT`). Conditional — see
  [Character select](#character-select).
- **Pip counting:** the chosen colour-agnostic lit test, applied identically to three P1 ROIs and
  three P2 ROIs — the same test both sides, since icons are not side-coded. `p1_lit`, `p2_lit` ∈
  {0,1,2,3}. A pip counts as lit only on **positive evidence of an icon**, per the fail-safe
  constraint above; anything else counts as not lit. Pip *order* is irrelevant — the count is
  positional, so the centre-outward fill order is an observation about the game, not a rule the
  detector encodes.
- **Details:** publish `DETAIL_P1_ROUNDS = str(p1_lit)` and `DETAIL_P2_ROUNDS = str(p2_lit)` on
  every `IN_MATCH` **and** `MATCH_END` observation, using the shared constants from `types.py`,
  never bare literals. This is the Confirmer's 0-0 cooldown-release path and cannot be omitted.
- **Outcome:** `p1_won = p1_lit >= 3`, `p2_won = p2_lit >= 3`. Exactly one side won ⇒
  `Screen.MATCH_END`, `winner` = that side, `confidence` = min of that side's three pip scores.
  Neither, or (impossibly) both ⇒ `Screen.IN_MATCH`, refusing to guess a winner.
- `supported_events()` ⇒ `{MATCH_END}`. `rois()` exposes every sampled rectangle for the `roi` CLI
  preview.
- `register(TokonPipDetector())` at the module bottom; `from . import tokon` in
  `detectors/__init__.py`.

### 3. Confirmer + factory

Nothing to build. `make_confirmer` already returns the marker `Confirmer` for every non-SF6 game, and
TOKON wants exactly that: reach-N-pips ⇒ `MATCH_END`, N-frame agreement, cooldown held until
`CHAR_SELECT`, a fresh 0-0 reading, or the safety valve.

### 4. Roster

Add `"tokon"` to `enabled_games` in `config.example.toml` and update the assertion in
`tests/test_config.py::test_example_config_loads_cleanly_with_documented_defaults`.

## The HUD-present gate

Avatar gates on a dark clock emblem. TOKON has no dark box — its clock is pale outlined numerals over
the stage. Candidates, to be decided **by measurement** in calibration:

1. **The green segmented sub-bars** beneath each health bar. Present on both sides in all four
   stills, including the losing side's, whose *main* health bar goes dark at match end — which rules
   the main health bar out as a gate on its own.
2. **The clock-digit band** — a brightness or variance test over the numeral area.

Requirements the chosen gate must meet: reads *present* on every in-match frame including the
match-deciding one, and reads *absent* on team select, the K.O. banner, results/menu screens, and
title cards. Whichever separates with the widest margin wins.

**If no candidate separates cleanly, that is a blocker to report, not a threshold to loosen.**

## Character select

TOKON does have a team-select ("ASSEMBLE") screen — visible at t≈0 in `TOKON.mp4`. Calibration
decides whether it earns a branch:

- If it appears reliably **between matches**, add the `CHAR_SELECT` branch described above and
  calibrate its ROI.
- If it appears only at the start of the session (one occurrence in 683s), **omit it**, exactly as
  Avatar does, and rely on the Confirmer's fresh-0-0 cooldown release. Write the reason down in the
  module docstring and the calibration report.

Either outcome is safe: the 0-0 release covers cooldown on its own, since TOKON pips reset to 0-0 at
the start of every match's first round.

## Calibration (the real work — its own task)

Every ROI and threshold is **measured from `TOKON.mp4`, never guessed**.

1. `scripts/build_tokon_corpus.py`, modeled on `build_avatar_corpus.py`: extract labelled frames,
   `normalize()` them to 1920×1080, and write them to `samples/tokon/` as
   `in_match_p1-<n>_p2-<n>_<idx>.png` and `between_<idx>.png`. Commit the corpus.
2. Cover these states: 0-0, at least two mixed scores, a 3-x P1 win, an x-3 P2 win, and between-match
   frames (team select, K.O. banner, results, title card). The corpus must additionally include, as
   named adversarial cases:
   - **The blue-sky stage** (t≈28s, 126s, 128s) with all pips empty — the frames that already defeat
     a hue-agnostic saturation test. Any candidate that scores these as lit is disqualified.
   - **At least two visually different stages** — the stills span a bright city street and a
     pink-blossom park — to prove the pip ROIs and the lit test generalise across backgrounds.
   - **As many distinct icon colours as the footage contains**, since the palette is unknown and a
     test that only handles warm icons must fail visibly here rather than in production.
   - **Effect-heavy frames** (super flashes, hit sparks, screen-wide effects) over pips of a known
     state — the occlusion case the fail-safe constraint exists for.
3. Keep the corpus to roughly **25–35 frames**. Avatar's is ~40 × ~2MB; this repo is accumulating
   binary weight and TOKON does not need more coverage than Avatar.
4. Locate pip ROIs by diffing known-state frames. No eyeballing — the coordinates used in this spec's
   probes were eyeballed and exist only to justify the approach.
5. **Run the bake-off**: score every candidate from
   [What discriminates lit from empty](#what-discriminates-lit-from-empty) over every labelled pip in
   the corpus, publish the per-candidate margin table in the calibration report, and commit the
   widest-margin candidate that also satisfies the positive-evidence constraint.
5. Write `docs/superpowers/reports/2026-08-25-tokon-calibration.md` with a copy-paste constants
   block, the measured margins, and the ground-truth `(second, winner)` list.

### Resolution hazard (accepted, with an open item)

`TOKON.mp4` is **1280×714** — a 720p frame six rows short, so it is cropped, not merely scaled.
`normalize()` accepts it (0.8% inside the aspect tolerance) and stretches it vertically to 1080. The
operator's real OBS capture is **native 1920×1080** (per `config.toml`), so ROIs measured from the
VOD may sit a few pixels off vertically against live capture — and the pips are only ~22px across.

User decision, 2026-08-25: **calibrate on the VOD and verify against 1080p later.** The spec carries
this as an explicit open item, recorded in `docs/TODO.md`:

> Run `fgc-detect capture --config config.toml --out obs_frames` with TOKON on screen, then
> `fgc-detect roi --game tokon --sample obs_frames/frame_00000.png --out roi_check.png`, and confirm
> the boxes sit on the pips. Adjust the ROI y-offsets if they miss.

This fails safe: a misaligned ROI reads "not lit", producing a missed event, never a false winner.

## Validation target

`uv run fgc-detect replay --game tokon --video ~/repos/tokon/TOKON.mp4` emits one `match_end` per
match with the correct winner, at the second each match is decided (a few frames of confirmation lag
is fine). The clip contains both a P1 and a P2 win, so both directions are exercised. No events
mid-round, on round-ending K.O.s, or on transitions. The exact sequence is fixed once the corpus is
built and becomes the regression baseline, recorded in the calibration report.

## Testing

Hermetic throughout — no OBS, GPU, network, or real clock; no test loads the `.mp4`.

- **`tests/detectors/test_tokon_pips.py`** — synthetic frames painting the measured ROIs assert every
  classification: 0-0 ⇒ `IN_MATCH`, 3-0 ⇒ `MATCH_END` P1, 0-3 ⇒ `MATCH_END` P2, 2-2 ⇒ `IN_MATCH`,
  both-at-3 ⇒ `IN_MATCH` with **no winner named**, HUD-gate-absent ⇒ `UNKNOWN`, and (if built)
  char-select ⇒ `CHAR_SELECT`. Also assert the round-count details are published on both `IN_MATCH`
  and `MATCH_END`.
- **Colour-agnosticism, asserted not assumed** — paint lit pips in a spread of hues (warm, cool,
  and desaturated) and assert every one counts as lit. This is the test that would have caught the
  rejected warm-band approach, so it must exist and must be able to fail.
- **Occlusion safety** — paint a pip neither white-circle nor icon (a flat wash, as a super flash
  would leave) and assert it counts as **not lit**, and that a frame with all six pips so obscured
  yields no winner. This pins the fail-safe direction against the absence-test inversion.
- **`tests/detectors/test_tokon_corpus.py`** — the detector over the committed real frames reproduces
  the ground-truth per-frame classification. A genuine corpus-label conflict gets a strict `xfail`
  with a written reason, never a weakened assertion.
- **Confirmer integration** — a synthetic observation sequence fires exactly one correct event; a
  second sequence asserts a fresh 0-0 reading releases cooldown so match 2 of a session is detected.
- **Roster** — `tests/test_types.py` covers the new enum value; `tests/test_config.py` covers the
  example-config roster.

Every one of these can fail for the right reason: a wrong ROI, a wrong threshold, or a wrong winner
flips an assertion.

## Documentation

- **README** — add TOKON to the supported games, and state that round pips are the default detection
  approach with SF6's digit counter as the documented exception.
- **`CLAUDE.md`** — today it calls the marker *`Confirmer`* the default (line ~85) but presents the
  *detection* strategies neutrally in "How to add a new game" step 3. Make the detection default
  explicit there — start from pips, leave that path only when the HUD offers nothing pip-shaped —
  and add TOKON as a worked example alongside Avatar, noting it is the case where both sides share
  one pip colour.
- **`docs/TODO.md`** — two new items: (a) the live-1080p ROI verification above; (b)
  `MarkerRoundDetector` is unused and brightness-only while three hand-rolled pip detectors exist —
  either give it a pluggable sampler and migrate them, or delete it.

## Out of scope

- The runtime `set_game` strategy-swap hazard. Avatar already introduced the second confirmer
  strategy; TOKON is a third marker-strategy game and changes nothing about it.
- Migrating Avatar or SF6 onto any shared base.
- Any event type other than `match_end`.

## Known limitations

- **Icons are not side-coded**, so sides are separated purely by ROI position. A capture that
  is horizontally offset or differently cropped could in principle read one side's pips as the
  other's. Mitigated by the `roi` preview check before trusting live detection, and bounded by the
  fail-safe rule: a misread that lights both sides to 3 resolves to `IN_MATCH`, not a false winner.
- **The icon palette is unknown.** The lit test is validated against whatever colours appear in ~683s
  of footage; TOKON may have icons this clip never shows. This is the reason the committed test must
  be colour-agnostic rather than tuned to observed hues, and the reason a colour-band approach was
  rejected despite scoring perfectly on the reference stills.
- **Occlusion is the false-positive path.** A pip hidden behind a super flash or screen-wide effect
  is the one situation that could read as lit without an icon being there. The positive-evidence
  constraint and the effect-heavy corpus frames exist to bound it, but this is the failure mode to
  watch in the replay validation — a `match_end` firing during a super, rather than at a K.O., is the
  signature.
- **`rounds_to_win = 3` is a fixed constant.** If TOKON offers a first-to-2 or first-to-5 mode, this
  detector reads it wrong. No such mode is visible in the supplied footage; revisit if one appears.
