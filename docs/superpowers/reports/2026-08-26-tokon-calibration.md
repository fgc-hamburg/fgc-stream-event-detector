# Marvel TOKON pip detector — calibration report

**Date:** 2026-08-26
**Footage:** `~/repos/tokon/TOKON.mp4` — 1280×714, 29.967 fps, 20465 frames (~683 s), five
complete matches. Plus four hand-picked stills (`vlcsnap-00001..4`, same 1280×714 capture)
supplied with the spec.
**Spec:** `docs/superpowers/specs/2026-08-25-tokon-pip-detector.md`
**Plan:** `docs/superpowers/plans/2026-08-26-tokon-pip-detector.md`

---

## 1. What the HUD looks like

Six round-win markers sit along the top of the screen, three either side of the central match
clock, and both sides fill **centre-outward**. First to three rounds wins.

| state | appearance |
|---|---|
| empty | a small (~10 px) near-white circle drawn over the live stage — the slot is ~90 % see-through background |
| lit | an opaque ~27 px disc carrying a character icon: a coloured field (orange, yellow, …), a dark star, and a white `P`/`V` badge at bottom-right |

Icon colour is **not** side-coded — it varies by character — so the two sides are separated by
position only, and "lit" cannot be a hue test.

### Why `MarkerRoundDetector` is unusable here

Brightness `fill_ratio` reads TOKON's markers **inverted**: an empty pip (white circle) has a
higher median V than a lit one (mid-tone icon disc). Measured medians: empty V ≈ 168, lit V ≈ 133.
A brightness threshold would count exactly the wrong pips.

---

## 2. Measured geometry (canonical 1920×1080)

Pip-slot centres were located two independent ways that agree to ±1 px:

* **Empty-circle centroid** — accumulated the near-white mask (`sat ≤ 70`, `val ≥ 170`) over 94
  0-0 frames drawn from five different stages. Peaks at x = 747, 785, 823 | 1097, 1135, 1173,
  each 8–10 px wide, all at y = 44…53.
* **Lit-minus-empty difference** — averaged ~10 frames either side of each pip lighting within the
  same match (so the stage is identical) and took the bounding box of the difference. Disc extent
  x = c ± 14, y = 35…61; the `P`/`V` badge extends the box to y = 67 at bottom-right.

Pitch is a uniform **38 px** and the six slots are **symmetric about x = 960** (960 − 747 = 213,
1173 − 960 = 213; and so on for the other two pairs), which is the cross-check that the two
methods converged on the real layout rather than on a shared bias.

```python
P1_PIP_CENTRES = (747, 785, 823)      # outermost first
P2_PIP_CENTRES = (1097, 1135, 1173)   # innermost first

_CORE_Y, _CORE_SIZE = 45, 6      # inside the empty circle, clear of its edge
_ICON_Y, _ICON_H = 36, 15        # upper half of the lit disc, above its badge
_ICON_W = 24
_BG_Y, _BG_H = 18, 12            # stage directly above the slot; no HUD there
```

The background box sits *above* each slot in the same column, so it tracks whatever stage is
behind that particular pip.

---

## 3. The fail-safe constraint that shaped the design

The obvious test — "the white circle is gone ⇒ lit" — is **inverted and fails open**. Anything
that covers an *empty* slot reads as lit: a character sprite, a super flash, or the HUD simply
vanishing for a KO cinematic. Three such slots would name a false winner, and a false winner
corrupts the scoreboard where a missed one does not.

So a slot is only called **lit** on *positive* evidence of a disc:

```
empty     ⟺ pale_fill_ratio(core) ≥ EMPTY_CORE_PALE_MIN
lit       ⟺ pale_fill_ratio(core) ≤ LIT_CORE_PALE_MAX
             AND region_difference(icon, background) ≥ LIT_ICON_DIFF_MIN
otherwise   ambiguous
```

**One ambiguous slot makes the whole frame `UNKNOWN`.** No counts are published at all.

---

## 4. The bake-off

Candidate lit-tests were scored over the 33 hand-verified scored corpus frames (198 slots) plus
the 20 HUD-absent frames. "Margin" is `min(score over lit slots) − max(score over empty slots)`
on clean frames.

| candidate | margin | verdict |
|---|---|---|
| **`pale_fill_ratio(core 6×6, sat ≤ 90, val ≥ 150)`** | **+0.972** (lit max 0.000, empty min 0.972) | decides *empty* perfectly, but is the inverted test — cannot decide *lit* alone |
| **`region_difference(icon 24×15, background 24×12)`** | **+0.026** (lit min 0.134, empty max 0.109) | the positive evidence; narrow but never crossed |
| `region_difference(icon, gap between pips)` | −0.092 | rejected |
| `region_difference(icon, core)` (inverted) | −0.091 | rejected |
| `region_difference(left-edge strip, background)` | −0.010 | rejected |
| `region_difference(top-ring 20×6, background)` | +0.006 | rejected — too thin |
| `pale_fill_ratio(badge 12×12)` | −0.514 | rejected — stage behind an empty pip is often pale too |
| absolute hue/saturation on the icon band | — | **rejected on measurement**: blue-sky stages bleed through empty pips and produce false positives (spec §4) |

The last row is why `region_difference` is used at all: comparing a slot against *its own local
background* is what makes the reading independent of an arbitrary stage.

### Chosen constants

```python
CORE_SAT_MAX = 90
CORE_VAL_MIN = 150
EMPTY_CORE_PALE_MIN = 0.50    # measured empty ≥ 0.972, lit = 0.000
LIT_CORE_PALE_MAX = 0.10
LIT_ICON_DIFF_MIN = 0.128     # measured clean lit ≥ 0.134, clean empty ≤ 0.109
ROUNDS_TO_WIN = 3
```

`LIT_ICON_DIFF_MIN` sits at ~76 % of the gap rather than in the middle: over-reading a lit pip is
the dangerous direction, so the threshold is biased towards missing one.

### No separate HUD-present gate

The spec listed candidate HUD gates (the clock-digit band, the green segmented sub-bar). Measured
over the whole clip at 5 Hz, the clock-band candidate (`dark ≥ 0.02 AND pale ≥ 0.30`) added
nothing the six-slot resolve rule does not already do, and it *rejected* legitimate washed-out
in-match frames. It was dropped. The gate is the reading itself: a frame is only read when all six
slots resolve to a definite state.

Consequence, identical to Avatar's: a pale cut-scene where all six slots happen to look like empty
circles reads `IN_MATCH 0-0`. That is harmless — no pips are drawn, so both sides read 0 — and it
is exactly the marker `Confirmer`'s 0-0 cooldown-release signal.

### Character select

Not handled. ~683 s of footage was traced end-to-end and no character/team-select screen ever
appears (matches go KO cinematic → results → next match). Its ROI cannot be measured, and guessing
is forbidden. Safe because the marker `Confirmer` also releases cooldown on a fresh agreeing 0-0,
which this detector publishes on every readable in-match frame.

---

## 5. Ground truth

The clip contains five matches. Every entry below was read off a rendered HUD strip **by eye**;
the scan only shortlisted the timestamps.

| # | match window | progression (P1–P2) | winner | HUD shows 3 pips |
|---|---|---|---|---|
| A | 23–111 s | 0-0 → 1-0 → 2-0 | **P1** 3-0 | 111.1–111.9 s |
| B | 124–288 s | 0-0 → 1-0 → 1-1 → 2-1 → 2-2 | **P2** 2-3 | 288.1–288.9 s |
| C | 303–470 s | 0-0 → 0-1 → 0-2 → 1-2 → 2-2 | **P1** 3-2 | 470.1–471.5 s |
| D | 476–582 s | 0-0 → 0-1 → 0-2 | **P2** 0-3 | 582.6–583.8 s |
| E | 588–667 s | 0-0 → 1-0 → 2-0 | **P1** 3-0 | 667.3–668.1 s |

**The winning pip is only visible for ~0.8–1.4 s.** The HUD cuts away the instant the final round
ends and comes back, with the third pip lit, only after the KO cinematic. At the default 5 Hz poll
that is 5–9 samples against `agreement_frames = 3` — it works with margin to spare, but it is the
tightest constraint in this detector and anything that lowers `poll_hz` below ~4 Hz will start
dropping match ends.

---

## 6. Corpus and adversarial cases

`samples/tokon/` (57 frames), built by `scripts/build_tokon_corpus.py`:

| label | n | what it is |
|---|---|---|
| `in_match_p1-N_p2-M` | 33 | clean HUD; every score from 0-0 to each of the four match-ending scores, over six stages |
| `occluded_p1-N_p2-M` | 3 | HUD present but degraded — super-flash wash-out (45.6 s), near-transparent transition (313.0 s), a move effect over a lit pip (405.0 s) |
| `sprite_p1-N_p2-M` | 4 | a sprite or the pip-slide animation fully covers an **empty** slot |
| `between` | 20 | no HUD: KO cinematics, cut-scenes, results screens, title cards |

Results: **33/33 clean frames and 3/3 occluded frames read exactly right**, and no `between` frame
ever reports a nonzero pip count or a match end.

### The four `sprite_*` frames — a real, documented limitation

| t | truth | read | what covers the slot |
|---|---|---|---|
| 101.51 s | 2-0 | 3-0 | a dark maroon character sprite over P1's empty outer slot |
| 107.52 s | 2-0 | 3-0 | the pip-slide animation ghosts an icon into the empty outer slot |
| 461.91 s | 2-2 | 3-2 | a blue flash over P1's empty outer slot |
| 466.51 s | 2-2 | 3-2 | the pip-slide animation, again on P1's outer slot |

These are exactly the fail-open hazard of §3, and no per-frame test can separate them: the slot
genuinely looks like a disc. They are kept in the corpus with truthful labels and a **strict
`xfail`**, so the behaviour is pinned rather than papered over.

Why they are safe in the pipeline: scanning the whole clip at 5 Hz, **every** spurious
three-pip reading is exactly **one sample long** (n = 1 at 101.51, 107.52, 461.91, 466.51), against
the `Confirmer`'s three-sample agreement requirement — a 3× margin. Every genuine match end is
5–9 samples long. `test_a_single_stray_three_pip_reading_never_fires` pins the temporal half of
that argument.

---

## 7. End-to-end validation

```
uv run fgc-detect replay --game tokon --video ~/repos/tokon/TOKON.mp4
```

```json
{"type": "match_end", "game": "tokon", "winner": "p1", "confidence": 0.2078, "ts": "...T00:01:51.522409Z"}
{"type": "match_end", "game": "tokon", "winner": "p2", "confidence": 0.1365, "ts": "...T00:04:48.516680Z"}
{"type": "match_end", "game": "tokon", "winner": "p1", "confidence": 0.1947, "ts": "...T00:07:50.516446Z"}
{"type": "match_end", "game": "tokon", "winner": "p2", "confidence": 0.1478, "ts": "...T00:09:43.039954Z"}
{"type": "match_end", "game": "tokon", "winner": "p1", "confidence": 0.1718, "ts": "...T00:11:07.732914Z"}
```

| fired at | ground truth | winner | ✓ |
|---|---|---|---|
| 111.52 s | A, 111.1–111.9 s | p1 | ✓ |
| 288.52 s | B, 288.1–288.9 s | p2 | ✓ |
| 470.52 s | C, 470.1–471.5 s | p1 | ✓ |
| 583.04 s | D, 582.6–583.8 s | p2 | ✓ |
| 667.73 s | E, 667.3–668.1 s | p1 | ✓ |

**Five matches, five events, correct winner every time, zero false positives, zero misses.**

`confidence` here is the *minimum* `region_difference` across the winning side's three slots — a
raw margin, not a probability, so the 0.13–0.21 range is expected and healthy (the empty-slot
ceiling is 0.109).

---

## 8. Known limitations

1. **Calibrated at 1280×714, not at native 1080p.** The source clip is a 1280×714 capture that
   `normalize()` upscales (aspect 1.7927 vs 1.7778 — inside the 2 % tolerance, so ~0.8 % of
   vertical squash). The ROIs are 6–24 px boxes, so a genuine 1920×1080 capture must be checked
   with `fgc-detect capture` + `fgc-detect roi` before the detector is trusted live. Tracked in
   `docs/TODO.md`.
2. **The three-pip window is short** (§5): below ~4 Hz polling, match ends will be missed.
3. **Sprite occlusion of an empty outer slot over-reads by one pip** (§6). Single-frame in all
   four measured cases; the confirmer absorbs it. A sustained occlusion lasting ≥ 3 polls at the
   exact moment a side is on 2 rounds would produce a false event — no per-frame test can prevent
   that, and the pipeline's answer is that the operator can correct a wrong call.
4. **One capture, one character pair** (Ms. Marvel vs Green Goblin), two observed icon colours
   (orange and yellow). The design does not depend on icon colour, but a second capture with other
   characters would be worth adding to the corpus.
