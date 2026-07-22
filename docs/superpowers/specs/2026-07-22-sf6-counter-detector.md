# SF6 Counter Detector — Design Addendum

**Date:** 2026-07-22
**Status:** Approved (from live investigation of real footage + user decisions), building now
**Supersedes:** the "SF6 layout" approach in the v1 plan (Task 17), which assumed round-win pips.

## Why this differs from the marker approach

The v1 plan assumed every game exposes countable round-win pips and reused `MarkerRoundDetector`.
Investigation of real SF6 footage (`~/repos/sf6.mp4`, clean game capture) showed SF6 has no
easily-sampled round pips. Its clean, reliable, winner-naming signal is the **games-won-in-set
counter** — the digit box beside each player's name, which increments by one for the winner at
each game end.

**Each game detects its own way** (user direction). Digit-counting is SF6's answer to the
`Detector` protocol, not a global rule. `MarkerRoundDetector` remains available for games that
suit it. This addendum adds a second, parallel detection strategy; it does not replace the first.

## What "match end" means here

Per-game end (user decision): fire once per game (best-of-3 rounds), naming the game winner, so
the dashboard increments games-won as a set progresses.

## Components

### 1. `Sf6CounterDetector` (stateless, implements `Detector`)

Reads the two counter boxes and reports what it sees in a single frame. It never decides a game
ended — it cannot, without history.

- Two ROIs, **not symmetric**: the boxes are mirror-image parallelograms, P1's digit left-of-
  centre (~x646–718), P2's right-of-centre (~x1316–1388), both y ~10–60 at 1920×1080.
- Digit recognition is **glyph-normalised**, so P1-derived references also read P2's mirrored box:
  grayscale → Otsu-binarise (digit darker than the light box) → crop to the numeral's bounding
  box → resize to a canonical mask → `match_template` (`TM_CCOEFF_NORMED`) against reference masks
  for 0–3. This is the consumer that justifies `match_template` in `roi.py`.
- Per frame, `observe` returns an `Observation`:
  - `screen = IN_MATCH` when both boxes read a confident digit (box present, brightness gate);
    else `UNKNOWN` (between games, transitions).
  - `details` carries the two counts under **shared constants** `DETAIL_P1_GAMES` / `DETAIL_P2_GAMES`
    exported from `types.py` — never string literals (the producer/consumer contract rule).
  - `winner = None` always. The detector reports state, not outcomes.
  - `confidence` = the weaker of the two digit-match scores.

### 2. Set-score increment confirmation (stateful)

The temporal decision — "a counter went up, so that side won a game" — lives in a stateful
confirmer, mirroring how the marker path keeps all temporal logic out of the pure detector.

It exposes the **same interface the pipeline already depends on** (`observe(observation, now) ->
Event | None`, `arm` / `disarm` / `set_game`, `state`, `armed`, `game`), so it is a drop-in the
pipeline and server treat identically to the existing `Confirmer`. A small factory selects the
strategy per game.

Behaviour:
- **Arming**: emits nothing while disarmed (same as the marker path).
- **N-frame agreement** on the `(p1_games, p2_games)` pair before a reading is trusted — kills
  single-frame digit misreads.
- **Baseline**: holds the last confirmed set score. First confident reading sets the baseline and
  fires nothing.
- **Fire**: when a confirmed reading is exactly one higher on a single side than the baseline
  (`(b1+1, b2)` or `(b1, b2+1)`), fire `match_end(winner=that side)` and adopt the new baseline.
- **Reset**: a confirmed reading `S` that differs from baseline `B` and is elementwise not greater
  (`S[0] <= B[0] and S[1] <= B[1]`) — covering both a decrease and `(0, 0)` after a non-zero
  baseline — is a new set: re-baseline to `S`, fire nothing. Logged at info.
- **Implausible higher jumps**: any other non-single-increment transition (some component greater
  than the baseline but not a clean `+1` on exactly one side — e.g. both sides up, a jump of +2,
  non-adjacent) **holds the baseline unchanged** and fires nothing; wait for a coherent reading.
  Logged at info. This is a deliberate trade-off: holding the baseline is robust against the
  transient misreads that dominate residual risk under N-frame agreement, and its accepted
  limitation is that if the detector is blind through 2+ whole games, the tracked score diverges
  from the real one and detection stalls until the next arm — a visible, per-set-recoverable
  failure, which is preferred over silently reporting a wrong winner.
- **Staleness**: a partial agreement streak older than the staleness window is discarded, as in
  the marker path.

## Validation target

`fgc-detect replay --game sf6 --video ~/repos/sf6.mp4` must emit **exactly four** `match_end`
events with winners **P1, P2, P1, P1**, at roughly 75s, 131s, 244s, 290s — the true game ends.
No events during rounds, KOs, or between-game transitions.

## Known limitation

The clip contains only P2 ∈ {0, 1}; P2=2/3 reading is via cross-box references and unvalidated
against real P2=2/3 frames. Tracked in `docs/TODO.md`.

Calibration facts (ROIs, thresholds, corpus timestamps): see the `sf6-counter-calibration` memory.
