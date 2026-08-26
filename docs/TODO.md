# Deferred work

Tracked items intentionally postponed. Each is real, just not now.

## Avatar: harden the HUD-present gate against dark transition frames (deferred 2026-07-30)

`AvatarPipDetector` gates "match HUD present" on the clock emblem being **dark**
(`EMBLEM_ROI` mean-gray <= `EMBLEM_DARK_MAX`). An all-dark transition frame (KO
wipe, title card) also passes this gate, so such frames read `IN_MATCH 0-0`
instead of `UNKNOWN`. This is harmless in the validated capture (replay fired 4
well-spaced events, no duplicates) and the 0-0 reading is in fact the Confirmer's
cooldown-release signal — but `confirmer.py::_observe_cooldown` documents that its
safety design assumes a detector reports `UNKNOWN` (not `IN_MATCH 0-0`) on
non-gameplay screens; a differently-timed wipe or a lingering win screen could in
principle trigger the premature-cooldown-release / duplicate-`match_end` path that
docstring warns about. To close it: distinguish a real emblem (dark box **with
bright digits** → high grayscale variance/contrast in `EMBLEM_ROI`) from a
uniformly-dark wipe (low variance) — e.g. require both `mean <= EMBLEM_DARK_MAX`
and `std >= <measured threshold>`. Needs the variance threshold measured from
footage across in-match vs wipe frames (never guessed); the committed corpus
(`samples/avatar/` in_match vs between frames) already contains both cases to
measure from. Also revisit once real character-select footage exists (that branch
was omitted for lack of any char-select screen in the calibration clip).

## TOKON: verify the ROIs against a native 1920x1080 capture (open, 2026-08-26)

`TokonPipDetector` was calibrated entirely from `~/repos/tokon/TOKON.mp4`, a **1280x714** capture
that `normalize()` upscales to 1920x1080 (aspect 1.7927 vs 1.7778 — inside the 2% tolerance, so
~0.8% of vertical squash is baked into every measured coordinate). The ROIs are 6-24px boxes, so a
small systematic offset matters. Before trusting live detection:

```bash
uv run fgc-detect capture --config config.toml --out obs_frames --limit 30
uv run fgc-detect roi --game tokon --sample obs_frames/frame_00000.png --out roi_check.png
```

The 18 boxes must sit on the six pip slots (core inside the white circle, icon over the disc,
background on bare stage above it). If they are off, re-measure against the 1080p frames and add
them to `samples/tokon/`. See section 8 of
`docs/superpowers/reports/2026-08-26-tokon-calibration.md`.

## MarkerRoundDetector has no users (open, 2026-08-26)

`detectors/marker.py` (`MarkerLayout` + `MarkerRoundDetector`, brightness `fill_ratio`) is
registered for **zero** games, while `avatar.py`, `sf6.py` and `tokon.py` each hand-roll their own
pip/marker loop. Now that pip counting is the documented default (CLAUDE.md, "How to add a new
game" step 3), the generic path should either grow to cover the real cases — parametrising the
per-slot "is it lit?" test instead of hard-coding brightness — or be deleted so it stops looking
like the route to take. Decide when a fourth pip game lands.

## Tekken 8 detector (deferred 2026-07-22)

v1 shipped with SF6 only. Tekken 8 is deferred at the user's request.

- Tekken 8 detects its own way — **do not assume it reuses SF6's digit-counter approach.**
  Each game implements the `Detector` protocol however suits its HUD. Options to weigh when
  picked up: round-win pips (fits `MarkerRoundDetector`), a games-in-set counter (fits the SF6
  digit approach), or something else.
- Needs sample media from the real capture setup (a VOD, ideally with both P1 and P2 wins).
  Ask the user; never invent ROI coordinates.
- Confirm the set format (Tekken sets are commonly first-to-3) from the samples.

## SF6: validate P2 counts of 2 and 3 (open)

The clip used to build the initial SF6 detector (`~/repos/sf6.mp4`) only contains P2 ∈ {0, 1} —
P2 never wins a 2nd or 3rd game in it. The detector reads P2=2/3 via cross-box (P1-derived)
references, which is **unvalidated against real P2=2/3 frames**. Close this with a clip where P2
wins at least two games, then add those frames to the corpus.

## SF6: detect the set-deciding game via the RESULT screen (open)

The counter/increment method catches games 1..N-1 of a set but never the set-clinching game N: its
incremented score never appears on an in-match HUD (there is no next game), only on SF6's post-set
RESULT screen ("Player 1 ... WON 3 - 1 LOST ... Player 2"). For the initial approach the operator
supplies the final game (user decision, 2026-07-22). To close it: add a second read path to the SF6
detector that recognises the RESULT screen and reads its two numbers (left = P1 games, right = P2),
feeding them to the confirmer so the final increment fires. Needs its own ROIs/calibration; only one
result frame exists in the current clip (~291-292.5s of `~/repos/sf6.mp4`).

## Quality follow-ups from review (deferred to a final pass)

- `save_config` / `load_config` hand-mirror their field lists in two places; a field added to one
  and forgotten in the other silently resets on every save (already happened once). Derive one
  from the other.
- `config_event` does registry lookups on every broadcast for a value that never changes at runtime.
- Generalize confirmer selection: the pipeline/CLI pick a confirmation strategy per game via a
  factory. Revisit if a third strategy appears.
  - Concrete hazard: runtime `set_game` does not re-select the confirmer strategy;
    `make_confirmer` runs once at startup. Switching between games with different strategies
    (e.g. SF6 counter ↔ a future marker game) at runtime silently stops detection until restart.
    Fix when the second strategy game lands: rebuild the confirmer in the run loop / server when
    the active game's strategy changes.
