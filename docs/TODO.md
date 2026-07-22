# Deferred work

Tracked items intentionally postponed. Each is real, just not now.

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
- `match_template` / `mean_color` in `detectors/roi.py`: confirm they now have a consumer (the SF6
  digit reader should use `match_template`); if `mean_color` stays unused, delete it (YAGNI).
- `config_event` does registry lookups on every broadcast for a value that never changes at runtime.
- Generalize confirmer selection: the pipeline/CLI pick a confirmation strategy per game via a
  factory. Revisit if a third strategy appears.
