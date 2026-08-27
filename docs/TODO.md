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

## TOKON: verify the ROIs against a native 1920x1080 capture (RESOLVED, 2026-08-27)

Done. Verified against `~/repos/tokon/tokon-3.mp4` (native 1920x1080, bt709 SDR, recorded through
OBS). **The ROIs were correct**: the empty pip's pale disc measures centre (747.5, 48.5) against the
assumed (747, 48), and the 38px pitch matches exactly — the 1280x714 calibration source introduced
no measurable offset.

The native capture did expose a different defect in the same detector: the lit-pip test was blind on
full-resolution footage. Fixed and re-measured in
`docs/superpowers/reports/2026-08-27-tokon-recalibration.md`, and `samples/tokon/` now carries 27
native frames (`scripts/build_tokon_native_corpus.py`) so the corpus spans two unrelated captures.

## ObsFrameSource achieves less than the configured poll_hz (RESOLVED, 2026-08-27)

Fixed. Two compounding causes, both found only after measuring against a source that was actually
decoding — an earlier 6.16 Hz measurement was taken against a static black frame and was worthless.
The real live figure was **0.95 Hz** at `poll_hz = 10.0`, which made every TOKON match end
undetectable (3 agreeing frames need 2.10s; the longest win window is 1.20s).

1. Screenshots were requested as **PNG**. Lossless compression of a noisy game frame is expensive
   inside OBS and dominates while OBS is also decoding: 1128ms vs JPEG q=80's 104ms at 1280x720.
   Now JPEG q=80, validated against every committed corpus. `capture` still uses PNG — ROIs are
   measured from those frames.
2. `frames()` slept a full `1/poll_hz` **after** each capture, making the period
   `capture_latency + 1/poll_hz`. Now sleeps only the unused remainder of the interval.

Result: 0.95 Hz -> 8.64 Hz. The source now logs its achieved rate after 20 captures and warns below
half the configured `poll_hz`, so a shortfall is visible instead of silent.

## replay ignores config.toml (open, 2026-08-27)

`_cmd_replay` builds `make_confirmer(game, ConfirmerConfig())` (`cli.py:80`), so `replay` always runs
the *default* `agreement_frames=3` regardless of what `config.toml` says. An operator whose live
config sets a different value gets a green replay and a silent live failure — which is exactly how
the 2026-08-27 TOKON bug presented. Either take `--config`, or print the confirmer settings replay
is actually using.

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
