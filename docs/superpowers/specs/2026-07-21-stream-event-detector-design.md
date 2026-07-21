# FGC Stream Event Detector — Design

**Date:** 2026-07-21
**Status:** Approved, ready for implementation planning
**Scope:** A standalone detector process that watches an OBS game-capture source and emits stream
events. v1 emits a single event — match end, naming the winner — consumed by the FGC Scoreboard
control dashboard.

---

## Problem

Operators currently type every score by hand. When a match ends on stream, the scoreboard only
updates when a human notices and clicks. We want the stream to detect match end on its own and
announce who won, leaving the decision of what to *do* about it to the dashboard.

## Non-goals

- Deciding what happens on detection. The detector announces facts; the dashboard owns policy
  (increment score, prompt the operator, report to start.gg).
- Round-level or set-level detection. v1 fires once per **game/match end** only. Set logic
  (FT2/FT3) stays in the dashboard, which already knows the format from start.gg.
- Player identity. The detector reports `p1`/`p2` by screen side, never by name.
- Games beyond SF6 and Tekken 8. The interface is built so a third game is additive, but no
  third game is implemented.

---

## Why computer vision, and why not something else

Research into alternatives found no usable non-video signal for the target games:

| Source | Verdict |
|---|---|
| SF6 Buckler's Boot Camp API | Login-gated, online-ranked only, batch not live. Unusable for local play. |
| SF6 memory reading (REFramework) | Lua scripting disabled in online matches; DLL injection on a tournament station is a policy problem. |
| Tekken 8 memory reading | Readers exist (TekkenOverlay lineage) but offsets break every patch. Fragile. |
| GGST replay API | Online replays only, not real-time, not local sets. |
| 2XKO | Riot + Vanguard anti-cheat. Memory access permanently off the table. |

Games that expose real-time data (Melee via Slippi) have solved this problem; the games run on
console at most events do not. CV is the only realistic path.

Prior art worth knowing: **SmartCV-SF6** (skpeter) does this for SF6 and emits over a WebSocket
on port 6565, consumed by **S.M.A.R.T.** which reports to start.gg. It covers SF6, GGST and
UNI2 — not Tekken 8, not 2XKO. It requires English game language and no UI mods, because it
OCRs text. It remains a viable fallback backend for SF6 (see *Alternatives considered*).

**Round-marker sampling is preferred over banner OCR.** Round-win markers are position-fixed and
language-independent; a filled-vs-empty threshold on a fixed box needs no OCR pass and imposes no
game-language requirement.

---

## Architecture

A standalone Python process, separate repository, no dependency on the scoreboard repo.

```
obs-websocket ──frames──▶ Detector.observe() ──Observations──▶ Confirmer ──Events──▶ WebSocket server
 (game source)             (per-game, pure)      (shared, stateful)                    (dashboard)
```

Four seams. Each is independently testable; the dashboard touches only the WebSocket.

Keeping this out of the scoreboard repo preserves that repo's stdlib-only property (`server.py`
runs on system Python with no dependencies), which is worth protecting.

### Frame sources

One interface, three implementations:

- **`ObsFrameSource`** — obs-websocket v5 `GetSourceScreenshot`, targeting the **game capture
  source by name**. Polls at a configurable rate, default 5fps, requesting a downscaled image.
- **`FolderFrameSource`** — a directory of PNGs. What tests and ROI tuning run against.
- **`VideoFrameSource`** — a recorded VOD, for replaying a full stream session offline.

`ObsFrameSource` must sample the **game capture source, never the program output**. Program output
composites commentator cams, sponsor overlays and transition stingers over the ROIs. This is the
primary reason for choosing obs-websocket over the alternatives:

- OBS Virtual Camera: ~300ms latency, lag drift over long sessions, a long-standing OpenCV
  DirectShow bug returning blank frames on Windows, and it only exposes composited output.
- NDI: technically excellent (low latency, low CPU) but adds a dependency and network config,
  and again gives program output rather than a named source.
- OBS Python scripting API: cannot practically read source textures.

Known obs-websocket gotcha: in Studio Mode, `GetSourceScreenshot` can return the preview rather
than program when both show the same scene. Irrelevant here since we target a source directly,
but worth noting if that ever changes.

Frames are normalized to the detector's `canonical_size` before ROI sampling. On aspect-ratio
mismatch the frame is rejected rather than sampled — see *Failure handling*.

### Detector interface

**Detectors are stateless and pure.** They classify a single frame. They never decide that a match
ended; all temporal reasoning lives in the Confirmer.

```python
class Screen(Enum):
    UNKNOWN = auto()
    CHAR_SELECT = auto()
    IN_MATCH = auto()
    MATCH_END = auto()

class Side(Enum):
    P1 = auto()
    P2 = auto()

@dataclass(frozen=True)
class Observation:
    screen: Screen
    winner: Side | None            # meaningful only when screen is MATCH_END
    details: Mapping[str, str]     # game-specific facts seen on this screen
    confidence: float              # 0.0–1.0
    debug: dict                    # raw per-ROI scores, for logging and tuning

class Detector(Protocol):
    game: str                          # "sf6" / "tekken8" — matches the dashboard's game key
    canonical_size: tuple[int, int]    # frames normalized to this before ROI sampling
    def observe(self, frame: Frame) -> Observation: ...
```

This split is the load-bearing decision. Adding a third game becomes "read some pixels, report
what you see" — no re-implementing debounce, arming, or cooldown. And a test is one assertion
against a PNG, with no OBS and no clock.

`details` exists to keep future observation types additive. A character-select detector fills
`{"p1_character": "ryu", "p2_character": "chun-li"}` and the Confirmer's agreement rule already
generalizes over it. **v1 populates `details` as an empty mapping and emits only `match_end`.**
Emitting a second event type later requires no interface change.

### Confirmer

A shared state machine consuming `Observation`s and producing events.

```
IDLE ──sees IN_MATCH──▶ LIVE ──N consecutive MATCH_END, agreeing payload──▶ FIRE
  ▲                                                                          │
  └─── CHAR_SELECT seen, or T seconds without MATCH_END ─── COOLDOWN ◀───────┘
```

- **N-frame agreement** (default N=3) suppresses single-frame noise. The agreement compares the
  full confirmable payload (`winner` + `details`), not just the winner, so the rule extends to
  future event types unchanged.
- **Must pass through LIVE.** A MATCH_END observation cannot fire from IDLE.
- **COOLDOWN** is the replay/attract-mode defense. SF6 and Tekken both show gameplay-looking
  footage with real HUDs after a set. Without cooldown, the post-match replay of the same KO
  reports a phantom second game. Exit requires a clean screen transition.
- **Arming.** The Confirmer emits nothing while disarmed. The dashboard arms when a set is loaded
  from the start.gg queue and disarms after reporting. This eliminates training-mode and casual
  play between sets as a false-positive class outright.

Constraining *when* detection is allowed to fire is more effective than improving the CV. Both
mature tools in this space do the same thing: S.M.A.R.T. requires a set to be pre-assigned to a
station before it will report anything.

### Wire protocol

Detector → dashboard:

```json
{"type":"match_end","game":"sf6","winner":"p1","confidence":0.94,"ts":"2026-07-21T10:40:00Z"}
{"type":"status","armed":true,"state":"live","game":"sf6","obs_connected":true}
```

Dashboard → detector:

```json
{"cmd":"arm"}
{"cmd":"disarm"}
{"cmd":"set_game","game":"tekken8"}
```

`status` is emitted on every state change and on connect, so a freshly-connected dashboard knows
where things stand without polling. The detector has no concept of a bracket, a set, or a score.

**Every value in a closed set is an enum in Python, never a bare string.** Event types, game keys,
sides, inbound commands and Confirmer states are all `StrEnum` members; `Screen` is a plain `Enum`
because it never crosses the wire. Strings exist only at the JSON boundary, where serialization
reads `.value` and deserialization parses back into the enum, rejecting unknown members explicitly
rather than propagating an unrecognized string inward.

This matters more than it looks. A typo in a bare `"match_end"` literal produces an event the
dashboard silently ignores, which is indistinguishable on stream from the detector simply not
firing — the same silent-and-confident failure mode called out in *Failure handling*. A closed
enum turns that into an error at the point of the mistake.

### Configuration

A single TOML config file covering: OBS host/port/password, the game-capture source name, poll
rate, active game, WebSocket and UI listen ports, agreement threshold N, cooldown seconds, the
enabled-game roster, the enabled-event set, and per-game ROI definitions and thresholds.

The detector also serves a small static configuration page on its own port. The operator uses it
to see live status, arm and disarm, choose the active game, edit the game roster, and toggle which
event types are delivered. Changes apply immediately and are written back to the config file, so
the file stays the source of truth and remains hand-editable.

Two properties are deliberate. **The page has no HTTP API** — it is served as a single static file
and does everything over the same WebSocket protocol the dashboard uses, so there is one protocol
to maintain and the page can do nothing the dashboard cannot. And **the detector serves it itself**
rather than embedding it in the scoreboard dashboard, so the detector stays usable standalone and
the scoreboard's stdlib-only server needs no knowledge of detector internals.

Only one game is on screen at a time, so the enabled-game roster is the list the operator picks the
active game from, not a set of detectors running concurrently. Event filtering covers detection
events only; `status` and `config` messages are always delivered, because letting an operator
disable them would leave the dashboard blind with no way to recover.

---

## Calibration tooling

Included in v1, not deferred. Without it, per-game work is guesswork.

- `detector capture --out samples/sf6/` — dumps frames while you play, building the corpus.
- `detector roi --game sf6` — renders ROI boxes over a sample frame to show what is being sampled.
- `detector replay --video vod.mp4 --game sf6` — prints the event timeline for a recorded VOD,
  diffable against what actually happened.

`replay` is the highest-leverage tool: it tunes a detector against last week's VOD at many times
realtime, instead of standing in front of a console.

**ROI coordinates are not specified in this document.** They cannot be derived from documentation;
they require real frames from the actual capture setup at the actual resolution. Each per-game
implementation step begins by requesting sample media from the user for that game, then follows
*collect samples → define ROIs → tune thresholds → assert against corpus*.

---

## Failure handling

The most dangerous failure mode is silent: a game patch restyles the HUD, fixed ROIs stop matching
what they used to, and the detector fails **confidently** with no error.

- **Every fire logs the full `debug` dict and writes the triggering frame to disk.** Confidence
  drift becomes visible before it causes a wrong call on stream.
- **OBS disconnect** → exponential backoff reconnect, `status` event with `obs_connected: false`,
  never fire while disconnected.
- **Missing or black source** → `Screen.UNKNOWN`, never fire.
- **Resolution mismatch** → normalize to `canonical_size`. On aspect-ratio mismatch, return
  `UNKNOWN` rather than sampling misaligned pixels.
- **Unparseable frame** → `UNKNOWN`. The detector degrades to silence, never to guessing.

---

## Testing

- **Golden corpus.** Labeled sample PNGs per game, asserted against `observe()`. No OBS, no
  network, no clock.
- **Confirmer tests.** Synthetic `Observation` sequences, no images at all. Must include an
  explicit regression test for the replay-after-KO scenario, and one for firing while disarmed.
- **Frame source tests.** `FolderFrameSource` and `VideoFrameSource` against fixtures;
  `ObsFrameSource` against a stubbed websocket.

CI requires neither OBS nor a GPU.

---

## Stack

Python 3.12, `opencv-python-headless`, `numpy`, `websockets`, `obsws-python`, dependencies managed
with `uv`. Lives in its own repository, `fgc-stream-event-detector`.

---

## Alternatives considered

**Advanced Scene Switcher macros + webhook.** ASS is actively maintained and has OpenCV pattern
matching, Tesseract OCR, and HTTP request actions built in — a zero-code v1 is genuinely possible.
Rejected as a destination because the logic would live in OBS GUI config: not version-controlled,
not testable, and expressing "N consecutive agreeing frames, only while armed" in macro form gets
unwieldy. Still useful as a quick check that a given ROI is detectable at all.

**Wrapping SmartCV-SF6 for SF6.** Would save real work on one of the two games. Rejected as the
primary plan because it means carrying two detection architectures, and SmartCV is a small project
with one release tag, Discord-only support, an English-language requirement, and a ~3GB PyTorch
GPU build. Retained as a fallback: because the detector boundary is a narrow event contract,
SmartCV could later back the SF6 detector without the dashboard noticing.

**ML/object detection (YOLO, CNN).** Rejected for the core task. Object detection answers "where
is it", but with a fixed camera, fixed resolution and fixed HUD the location is already known —
ROI plus threshold is strictly better engineering. Published scoreboard-reading YOLO work reports
around 70% accuracy, well below what this needs. The one place a small classifier would genuinely
beat hand-tuned thresholds is screen-state classification over a fixed crop; noted as a possible
future refinement if threshold tuning proves brittle.

---

## Future extensions (explicitly not in v1)

- **Character-select detection**, emitting `characters_locked` with character names. The interface
  already supports it via `details` and the generalized agreement rule. The cost is not
  architectural but data: it needs a labeled sample per character per side, and it is a classifier
  rather than a threshold.
- **Round-level events**, emitting on each round marker change rather than only at match end.
- **2XKO**, deliberately sequenced last: it is still in playtest and Riot restyles UI between
  patches, which is exactly the silent-failure case above.
