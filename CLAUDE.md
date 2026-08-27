# CLAUDE.md — working in this repo

Guidance for Claude (and humans) working on `fgc-stream-event-detector`. Read this before adding a
game or changing detection. The [README](README.md) covers what the project is and the WebSocket
protocol; this file is the *how-to-work-on-it*.

**One-line summary:** a standalone Python CV process watches an OBS game-capture source and emits
`match_end` (winner named) over a WebSocket. Each game reads its own HUD; a shared confirmer turns
per-frame observations into events. It announces facts — the dashboard decides policy.

---

## Run & test (do this first, every time)

Always run from the repo root with `uv` (Python ≥ 3.12):

```bash
uv sync                      # create .venv + install deps (first time / after dep changes)
uv run pytest                # full hermetic suite — must be green before and after any change
uv run pytest tests/detectors/test_avatar_pips.py -v   # one file, verbose
```

The CLI entry point is `fgc-detect` (installed by `uv sync`). Four subcommands:

| Command | What it does | Needs OBS? |
|---|---|---|
| `replay --game <g> --video <f>` | run a recorded VOD through the pipeline, print `match_end` events | no |
| `roi --game <g> --sample <png> --out <png>` | draw the detector's ROI boxes over a frame | no |
| `capture --config <toml> --out <dir> [--limit N]` | dump normalized frames from the live OBS source | yes |
| `run --config <toml>` | live: OBS → detection → WebSocket server + config UI | yes |

### Test against an mp4 (the fast loop — use this constantly)

```bash
uv run fgc-detect replay --game avatar --video ~/repos/avatar.mp4
# each fired match_end is printed as JSON; ts encodes the position in the video.
uv run fgc-detect replay --game avatar --video ~/repos/avatar.mp4 --evidence-dir evidence/
# ^ also dumps the frame + observation behind every event, for inspection.
```

Ground-truth expectations for the sample clips live in each detector's calibration report under
`docs/superpowers/reports/`. The hermetic equivalent (no video file needed) is the corpus test,
e.g. `tests/detectors/test_avatar_corpus.py`, which runs the detector over committed labelled PNGs.

### Test against live OBS

1. In OBS: enable **Tools → WebSocket Server Settings** (note port/password); have a **Game Capture**
   source (note its exact name — the detector reads *that source*, never program output, so overlays
   and cams don't contaminate the ROIs).
2. `cp config.example.toml config.toml`, set `game`, `obs.source_name`, `obs.port`, `obs.password`.
3. **Verify ROI alignment before trusting anything** — ROIs are calibrated at canonical 1920×1080;
   a differently-cropped/letterboxed capture will misalign them:
   ```bash
   uv run fgc-detect capture --config config.toml --out obs_frames --limit 30
   uv run fgc-detect roi --game <g> --sample obs_frames/frame_00000.png --out roi_check.png
   # open roi_check.png: the boxes MUST sit on the HUD elements. If not, recalibrate (see below).
   ```
4. `uv run fgc-detect run --config config.toml` — serves events on `ws://127.0.0.1:6600`, config UI
   on `http://127.0.0.1:6601`.
5. The confirmer starts **disarmed**; send `{"cmd":"arm"}` over the WebSocket (the dashboard does
   this) or nothing fires. `status` frames report `obs_connected` / `armed` / `state` so you can
   confirm OBS is seen before a match ends.

`run` against a real OBS instance is the one path not exercised by the test suite (the obs-websocket
call is verified against the installed library; ROI alignment is the thing to check by hand).

---

## Architecture (hold this in your head)

```
Frame (image + captured_at)
  → Detector.observe(frame) -> Observation        # PURE, per-game, no memory
  → Confirmer.observe(obs, now) -> Event | None    # STATEFUL, shared, all temporal logic
  → EventServer.broadcast(event)                   # the only outside contact
```

- **Detectors are pure and stateless.** `observe()` classifies exactly one frame — no history, no
  clock, no I/O. This is what makes adding a game cheap and testable. Registry: `get_detector(game)`;
  detectors self-register on import (`register(...)` at module bottom + an import line in
  `detectors/__init__.py`).
- **Confirmers hold all temporal logic** (N-frame agreement, arming, cooldown). Two exist, both
  exposing the same interface (`observe/arm/disarm/set_game/state/armed/game`):
  - `Confirmer` (`confirmer.py`) — the marker/pip path: fires when a side reaches its round count;
    used by Avatar and TOKON (and any `MarkerRoundDetector` game). This is the **default**.
  - `SetScoreConfirmer` (`set_score_confirmer.py`) — SF6's games-won counter: fires on a single-side
    +1 increment.
  - `make_confirmer(game, config)` in `confirmation.py` picks which. It is a small explicit map:
    SF6 → counter, everything else → marker.
- **The detector⇄confirmer contract** (do not drift from it):
  - `Observation(screen, winner=None, confidence=…, details={}, debug={})`, frozen. `Frame(image,
    captured_at)`.
  - `Screen` ∈ `{UNKNOWN, CHAR_SELECT, IN_MATCH, MATCH_END}`.
  - Publish per-side counts in `details` under the **shared string constants** from `types.py`
    (`DETAIL_P1_ROUNDS`/`DETAIL_P2_ROUNDS` for pip/marker games, `DETAIL_P1_GAMES`/`DETAIL_P2_GAMES`
    for SF6) — **never bare string literals**. The marker confirmer releases cooldown when it sees a
    fresh `IN_MATCH 0-0`, so publish 0-0 on a fresh match.
  - On `MATCH_END`, set `winner` and a `confidence`. If the reading is ambiguous (both/neither side
    looks won), return `IN_MATCH` and **refuse to guess a winner**.

---

## How to add a new game

Do these in order. Each detector is "read some pixels, report what you see"; the confirmer and
server are game-agnostic and you rarely touch them.

1. **Get real footage from the user first.** Ask for a VOD (ideally with both P1 and P2 wins) or
   labelled PNGs. **Never invent ROI coordinates or thresholds** — every constant is measured.
2. **Add the enum value** in `src/fgc_detector/types.py`: `Game.X = "x"`. (Every closed-set value is
   an enum; no bare game strings anywhere else.)
3. **Choose a detection strategy. Counting round-win pips is the default; SF6 is the exception.**
   Nearly every fighting game draws a fixed row of round markers per side, so start by assuming
   that shape: measure the marker ROIs, contribute a per-frame "is this marker lit?" test, and let
   the shared marker `Confirmer` (reach-N → win) do all the temporal work. Only reach for a
   different shape when the HUD genuinely has no pip row — SF6 is the one such game so far (it
   shows a games-won-in-set digit, which is why it has its own digit reader *and* its own
   `SetScoreConfirmer`). In pip order of preference:
   - Reuse `MarkerRoundDetector` (`detectors/marker.py`) if the pips are **brightness**-distinct:
     contribute a `MarkerLayout` (data — ROIs, `rounds_to_win`, thresholds), no new logic.
   - Write a **dedicated pip detector** (like `avatar.py`'s colour pips or `tokon.py`'s character
     icons) if the pips are distinguished some other way. It still reports pip counts to the same
     marker `Confirmer` — only the "is it lit?" test is new.
   - Write a **non-pip detector** (like `sf6.py`'s counter) only if there is no pip row at all;
     this also means picking/adding a confirmer strategy in step 5.

   Either way, implement the `Detector` protocol: attributes `game`, `canonical_size =
   (1920, 1080)`; methods `observe(frame) -> Observation`, `rois() -> dict[str, Roi]`,
   `supported_events() -> frozenset[EventType]`. Keep it pure. Reuse `roi.py` primitives
   (`fill_ratio` = brightness, `color_fill_ratio` = hue/sat/val, `match_template` = glyph,
   `pale_fill_ratio` = bright-and-colourless, `region_difference` = one region vs another, which is
   how you read a marker whose own colour varies) or add a new *general* primitive there.
4. **Calibrate from the footage** (the real work — its own task):
   - Extract frames, `normalize()` to 1920×1080, and **measure** ROIs by diffing known states; set
     thresholds to sit in a **clean gap** between states (dump HSV/greyscale numbers, don't eyeball).
   - Write `scripts/build_<game>_corpus.py` (model on `build_avatar_corpus.py`) that writes labelled
     PNGs to `samples/<game>/`. Commit the corpus.
   - **Self-validate**: run your detector/primitive over the corpus and prove every state separates
     with margin. If it doesn't separate cleanly, that's a real blocker — do not weaken thresholds to
     pass a test.
   - Write a calibration report `docs/superpowers/reports/YYYY-MM-DD-<game>-calibration.md` with the
     measured constants (copy-paste block), the margins, and the ground-truth `(second, winner)` list.
5. **Wire the confirmer** in `confirmation.py::make_confirmer` — for a pip game there is nothing to
   do (non-SF6 → marker `Confirmer`, by default). Only add a branch if the game genuinely has no
   pip row and needs a counter-style strategy.
6. **Register**: `register(<Detector>())` at the bottom of the module, and add `from . import <game>`
   to `detectors/__init__.py`.
7. **Roster**: add `"x"` to `enabled_games` in `config.example.toml`, and update the assertion in
   `tests/test_config.py::test_example_config_loads_cleanly_with_documented_defaults`.
8. **Tests** (hermetic, falsifiable):
   - detector unit tests on synthetic frames (paint the measured ROIs) — assert each screen/winner;
   - corpus regression over `samples/<game>/*.png` (model on `test_tokon_corpus.py` /
     `test_avatar_corpus.py` / `test_sf6_counter.py`) — assert real ground truth; use a strict
     `xfail` (with a written reason) for a genuine corpus-label conflict rather than weakening an
     assertion;
   - a confirmer-integration test: feed a synthetic observation sequence and assert exactly one
     correct event fires.
9. **Validate end-to-end**: `uv run fgc-detect replay --game x --video <clip>` produces the right
   winners at the right times. Record the result and any limitation in the calibration report.

---

## Invariants — do not break these

- **Never guess ROIs/thresholds.** Measure them from footage the user provides. If you don't have
  footage, ask; don't estimate.
- **Detectors stay pure.** No history, clock, or I/O in `observe()`. All temporal reasoning lives in
  the confirmer, so it's tested once against synthetic sequences.
- **Fail safe.** An ambiguous reading resolves to *no event* (or `IN_MATCH`), never a false winner. A
  missed match end is recoverable by the operator; a false one corrupts the scoreboard.
- **Every closed set is an enum** (`Game`, `Side`, `EventType`, `Screen`, `ConfirmerState`,
  `Command`). Cross-boundary strings use the `DETAIL_*` constants — never bare literals.
- **Sample the named game-capture source, never program output** — overlays would sit on the ROIs.
- **Canonical resolution is 1920×1080.** ROIs are expressed there; frames are `normalize()`d before
  `observe()`.
- **Tests must be able to fail, and must be hermetic.** No test may require OBS, GPU, network, or a
  real clock. Corpus PNGs are fine; a raw `.mp4` is never loaded by pytest. *A test that cannot fail
  is worse than no test.*
- **The JSON boundary is `events.py` only** — the one place enums become strings.

---

## Known hazards & gotchas

- **Runtime `set_game` strategy swap** (`docs/TODO.md`): `make_confirmer` runs once at startup;
  switching active game between a counter-strategy game (SF6) and a marker-strategy game (Avatar) at
  runtime does not rebuild the confirmer, so detection silently stops until restart. Starting under
  either game is fine. Fix when it bites: rebuild the confirmer in `EventServer._apply` when the
  active game's strategy changes.
- **OBS capture resolution ≠ calibration resolution** → misaligned ROIs. Always run the `capture` +
  `roi` check (above) against the real capture before trusting live detection.
- **Character-select is optional.** Avatar and TOKON both omit it (no such screen in their
  footage); the marker confirmer's 0-0 fresh-match path covers cooldown release. Only add a
  `CHAR_SELECT` branch if you have footage to calibrate its ROI.
- **A "the empty marker is gone" test fails open.** If you read a pip by the *absence* of its empty
  state, anything that covers an empty slot (a sprite, a flash, the HUD vanishing) reads as lit —
  and enough of those name a false winner. Require **positive evidence of the lit marker** as well,
  and resolve anything that is neither to `UNKNOWN`. See `tokon.py` and its calibration report.
- **Measure the live capture rate against a source that is actually decoding.** A screenshot of an
  idle or paused OBS source is an order of magnitude cheaper than one of a source mid-playback, so
  a rate measured on a static frame is worthless. The historical trap: requesting **PNG** cost
  1128ms/frame at 1280x720 against JPEG q=80's 104ms, because lossless compression of a noisy game
  frame is expensive *inside OBS*. That was the difference between ~1Hz and ~9Hz — and below ~2Hz
  no brief match-end marker can ever accumulate `agreement_frames`. `ObsFrameSource` now requests
  JPEG and logs the rate it achieves; `capture` still requests PNG because ROIs are *measured* from
  those frames. If you change the capture format, re-validate it by re-encoding every committed
  corpus frame and re-running its detector.
- **The winning marker may only be on screen for ~1 second.** TOKON hides the HUD for the KO
  cinematic and brings it back with the final pip lit for 0.8–1.4 s. Measure that window during
  calibration: it sets the floor on `obs.poll_hz` for that game.
- **Dark-screen HUD gates** can misread all-dark transition frames as in-match (see Avatar's TODO
  note) — harmless (reads 0-0, fires nothing) but worth a variance/digit-presence check if it bites.

---

## Where things live / workflow

- Design specs: `docs/superpowers/specs/`. Implementation plans: `docs/superpowers/plans/`.
  Calibration reports: `docs/superpowers/reports/`. Deferred work: `docs/TODO.md`.
- Labelled corpora: `samples/<game>/`, rebuilt by `scripts/build_<game>_corpus.py`.
- This project is built with **spec → plan → subagent-driven implementation** (the superpowers
  workflow). Make bespoke, focused commits. When adding a game, one stacked PR per game is the norm;
  base a new feature branch on the branch it extends and open a **draft** PR against this repo
  (`origin` = `fgc-hamburg/fgc-stream-event-detector`). Never push to `master` or force-push.
- Commit only when asked; branch off `master` (or the parent feature branch) rather than committing
  to it directly.
