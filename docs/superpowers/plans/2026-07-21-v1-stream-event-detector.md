# FGC Stream Event Detector v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone process that watches an OBS game-capture source and emits a `match_end` event naming the winner over a WebSocket, for Street Fighter 6 and Tekken 8.

**Architecture:** Frames flow from a `FrameSource` into a stateless per-game `Detector` that classifies a single frame into an `Observation`. Observations flow into a shared stateful `Confirmer` that owns all temporal logic — N-frame agreement, arming, and replay-mode cooldown — and emits `Event`s. A WebSocket server broadcasts events and accepts arm/disarm/set-game commands. The four seams are independently testable; nothing but the WebSocket is visible to the dashboard.

**Tech Stack:** Python 3.12, `opencv-python-headless`, `numpy`, `websockets`, `obsws-python`, `pytest`, dependencies managed with `uv`.

## Global Constraints

- **Python 3.12 minimum.** `StrEnum` (3.11+) and PEP 604 unions are used throughout.
- **Every value in a closed set is an enum, never a bare string.** Event types, game keys, sides, inbound commands, and Confirmer states are `StrEnum`. `Screen` is a plain `Enum` because it never crosses the wire. Strings exist only at the JSON boundary.
- **Deserialization rejects unknown enum members explicitly.** Never propagate an unrecognized string inward.
- **The detector never fires while disarmed, while OBS is disconnected, or on an unparseable frame.** It degrades to silence, never to guessing.
- **Detectors are stateless and pure.** A `Detector.observe()` implementation may not hold state between frames, read a clock, or perform I/O.
- **No test may require OBS, a GPU, a network, or a real clock.** All time is injected.
- **Every fire logs its full `debug` mapping and writes the triggering frame to disk.** A HUD restyle after a game patch breaks fixed ROIs silently; logged confidence is the only way to see the drift coming.
- Package root is `src/fgc_detector/`. Tests mirror it under `tests/`.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, pytest config |
| `src/fgc_detector/types.py` | All enums; `Frame`, `Observation` dataclasses |
| `src/fgc_detector/events.py` | `MatchEndEvent`, `StatusEvent`, command parsing — the JSON boundary |
| `src/fgc_detector/frames/normalize.py` | Canonical resize + aspect-ratio rejection |
| `src/fgc_detector/frames/source.py` | `FrameSource` protocol |
| `src/fgc_detector/frames/offline.py` | `FolderFrameSource`, `VideoFrameSource` |
| `src/fgc_detector/frames/obs.py` | `ObsFrameSource` (obs-websocket) |
| `src/fgc_detector/detectors/roi.py` | `Roi`, `fill_ratio`, `match_template` primitives |
| `src/fgc_detector/detectors/registry.py` | `Detector` protocol, `get_detector(Game)` |
| `src/fgc_detector/detectors/sf6.py` | SF6 detector |
| `src/fgc_detector/detectors/tekken8.py` | Tekken 8 detector |
| `src/fgc_detector/confirmer.py` | The state machine |
| `src/fgc_detector/server.py` | WebSocket server: broadcast + inbound commands |
| `src/fgc_detector/config.py` | TOML config loading |
| `src/fgc_detector/observability.py` | Fire logging + frame dumping |
| `src/fgc_detector/cli.py` | `run`, `capture`, `roi`, `replay` subcommands |

Tasks 1–12 are game-agnostic and fully testable with synthetic data. Tasks 13–14 need real sample media and are deliberately last.

---

### Task 1: Scaffolding and core types

**Files:**
- Create: `pyproject.toml`, `src/fgc_detector/__init__.py`, `src/fgc_detector/types.py`
- Test: `tests/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Game`, `Side`, `EventType`, `Command`, `ConfirmerState` (all `StrEnum`); `Screen` (plain `Enum`); `Frame`, `Observation` frozen dataclasses; `Observation.payload` property.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "fgc-stream-event-detector"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.0",
    "opencv-python-headless>=4.10",
    "websockets>=13.0",
    "obsws-python>=1.7",
]

[project.scripts]
fgc-detect = "fgc_detector.cli:main"

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/fgc_detector"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_types.py`:

```python
import pytest

from fgc_detector.types import (
    Command,
    ConfirmerState,
    EventType,
    Game,
    Observation,
    Screen,
    Side,
)


def test_wire_enums_have_stable_string_values():
    assert Game.SF6.value == "sf6"
    assert Game.TEKKEN8.value == "tekken8"
    assert Side.P1.value == "p1"
    assert Side.P2.value == "p2"
    assert EventType.MATCH_END.value == "match_end"
    assert EventType.STATUS.value == "status"
    assert Command.ARM.value == "arm"
    assert Command.SET_GAME.value == "set_game"
    assert ConfirmerState.IDLE.value == "idle"


def test_wire_enums_parse_from_their_value():
    assert Game("sf6") is Game.SF6
    assert Side("p2") is Side.P2


def test_unknown_enum_value_raises():
    with pytest.raises(ValueError):
        Game("smash")


def test_screen_is_not_a_string_enum():
    # Screen never crosses the wire, so it must not be silently comparable to a string.
    assert Screen.IN_MATCH != "in_match"


def test_observation_defaults_are_empty_and_shared():
    a = Observation(screen=Screen.UNKNOWN)
    b = Observation(screen=Screen.UNKNOWN)
    assert a.details == {}
    assert a == b


def test_payload_ignores_confidence_and_debug():
    # Two observations that agree on the facts must compare equal for N-frame
    # agreement, even when their confidence scores differ frame to frame.
    a = Observation(Screen.MATCH_END, winner=Side.P1, confidence=0.91, debug={"x": 1.0})
    b = Observation(Screen.MATCH_END, winner=Side.P1, confidence=0.97, debug={"x": 2.0})
    assert a.payload == b.payload
    assert a != b


def test_payload_differs_on_winner():
    a = Observation(Screen.MATCH_END, winner=Side.P1)
    b = Observation(Screen.MATCH_END, winner=Side.P2)
    assert a.payload != b.payload
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fgc_detector.types'`

- [ ] **Step 4: Write the implementation**

Create `src/fgc_detector/__init__.py` (empty file) and `src/fgc_detector/types.py`:

```python
"""Core value types.

Everything in a closed set is an enum. Values that cross the WebSocket are
StrEnum so serialization is a `.value` lookup and deserialization is a
constructor call that raises on anything unrecognized. Screen is deliberately
NOT a StrEnum: it is internal to detection and must never be confused with a
wire value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, StrEnum, auto
from types import MappingProxyType
from typing import Mapping

import numpy as np


class Game(StrEnum):
    SF6 = "sf6"
    TEKKEN8 = "tekken8"


class Side(StrEnum):
    P1 = "p1"
    P2 = "p2"


class EventType(StrEnum):
    MATCH_END = "match_end"
    STATUS = "status"


class Command(StrEnum):
    ARM = "arm"
    DISARM = "disarm"
    SET_GAME = "set_game"


class ConfirmerState(StrEnum):
    IDLE = "idle"
    LIVE = "live"
    COOLDOWN = "cooldown"


class Screen(Enum):
    """What the detector believes is on screen right now."""

    UNKNOWN = auto()
    CHAR_SELECT = auto()
    IN_MATCH = auto()
    MATCH_END = auto()


_EMPTY_STR_MAP: Mapping[str, str] = MappingProxyType({})
_EMPTY_NUM_MAP: Mapping[str, float] = MappingProxyType({})


@dataclass(frozen=True)
class Frame:
    """A single captured image, already normalized to a detector's canonical size."""

    image: np.ndarray  # BGR, uint8
    captured_at: datetime


@dataclass(frozen=True)
class Observation:
    """A detector's read of exactly one frame. Carries no history."""

    screen: Screen
    winner: Side | None = None
    details: Mapping[str, str] = _EMPTY_STR_MAP
    confidence: float = 0.0
    debug: Mapping[str, float] = _EMPTY_NUM_MAP

    @property
    def payload(self) -> tuple[Screen, Side | None, tuple[tuple[str, str], ...]]:
        """The facts that N-frame agreement compares.

        Deliberately excludes confidence and debug, which jitter frame to frame
        and would otherwise prevent any two observations from ever agreeing.
        Includes `details` so future event types (character lock) inherit the
        agreement rule with no change to the Confirmer.
        """
        return (self.screen, self.winner, tuple(sorted(self.details.items())))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_types.py -v`
Expected: PASS, 7 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/fgc_detector/__init__.py src/fgc_detector/types.py tests/test_types.py
git commit -m "feat: core enums and observation types"
```

---

### Task 2: Event serialization and command parsing

**Files:**
- Create: `src/fgc_detector/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Consumes: `Game`, `Side`, `EventType`, `Command`, `ConfirmerState` from `types.py`.
- Produces: `MatchEndEvent(game, winner, confidence, ts)`, `StatusEvent(game, armed, state, obs_connected, ts)`, both with `.to_dict() -> dict` and `.to_json() -> str`; `ArmCommand`, `DisarmCommand`, `SetGameCommand(game)`; `parse_command(raw: str)`; `CommandError`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_events.py`:

```python
import json
from datetime import datetime, timezone

import pytest

from fgc_detector.events import (
    ArmCommand,
    CommandError,
    DisarmCommand,
    MatchEndEvent,
    SetGameCommand,
    StatusEvent,
    parse_command,
)
from fgc_detector.types import ConfirmerState, Game, Side

TS = datetime(2026, 7, 21, 10, 40, 0, tzinfo=timezone.utc)


def test_match_end_event_serializes_enum_values_not_repr():
    event = MatchEndEvent(game=Game.SF6, winner=Side.P1, confidence=0.9412, ts=TS)
    assert event.to_dict() == {
        "type": "match_end",
        "game": "sf6",
        "winner": "p1",
        "confidence": 0.9412,
        "ts": "2026-07-21T10:40:00Z",
    }


def test_match_end_event_json_round_trips_to_expected_keys():
    payload = json.loads(MatchEndEvent(Game.TEKKEN8, Side.P2, 0.8, TS).to_json())
    assert payload["type"] == "match_end"
    assert payload["game"] == "tekken8"
    assert payload["winner"] == "p2"


def test_status_event_serializes():
    event = StatusEvent(
        game=Game.SF6, armed=True, state=ConfirmerState.LIVE, obs_connected=True, ts=TS
    )
    assert event.to_dict() == {
        "type": "status",
        "game": "sf6",
        "armed": True,
        "state": "live",
        "obs_connected": True,
        "ts": "2026-07-21T10:40:00Z",
    }


def test_naive_timestamps_are_rejected():
    with pytest.raises(ValueError):
        MatchEndEvent(Game.SF6, Side.P1, 0.9, datetime(2026, 7, 21, 10, 40, 0)).to_dict()


def test_parse_arm_and_disarm():
    assert parse_command('{"cmd":"arm"}') == ArmCommand()
    assert parse_command('{"cmd":"disarm"}') == DisarmCommand()


def test_parse_set_game():
    assert parse_command('{"cmd":"set_game","game":"tekken8"}') == SetGameCommand(Game.TEKKEN8)


def test_unknown_command_is_rejected():
    with pytest.raises(CommandError, match="unknown command"):
        parse_command('{"cmd":"self_destruct"}')


def test_unknown_game_is_rejected():
    with pytest.raises(CommandError, match="unknown game"):
        parse_command('{"cmd":"set_game","game":"smash"}')


def test_set_game_without_game_is_rejected():
    with pytest.raises(CommandError, match="requires a 'game'"):
        parse_command('{"cmd":"set_game"}')


def test_malformed_json_is_rejected():
    with pytest.raises(CommandError, match="not valid JSON"):
        parse_command("{not json")


def test_non_object_json_is_rejected():
    with pytest.raises(CommandError, match="JSON object"):
        parse_command('["arm"]')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fgc_detector.events'`

- [ ] **Step 3: Write the implementation**

Create `src/fgc_detector/events.py`:

```python
"""The JSON boundary.

This is the only module allowed to turn an enum into a string or a string into
an enum. Everything inward of here is typed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar

from .types import Command, ConfirmerState, EventType, Game, Side


class CommandError(ValueError):
    """An inbound message could not be understood. Never fatal — reply and continue."""


def _iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        raise ValueError(f"timestamp must be timezone-aware, got naive {ts!r}")
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class MatchEndEvent:
    game: Game
    winner: Side
    confidence: float
    ts: datetime

    TYPE: ClassVar[EventType] = EventType.MATCH_END

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.TYPE.value,
            "game": self.game.value,
            "winner": self.winner.value,
            "confidence": round(self.confidence, 4),
            "ts": _iso(self.ts),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass(frozen=True)
class StatusEvent:
    game: Game
    armed: bool
    state: ConfirmerState
    obs_connected: bool
    ts: datetime

    TYPE: ClassVar[EventType] = EventType.STATUS

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.TYPE.value,
            "game": self.game.value,
            "armed": self.armed,
            "state": self.state.value,
            "obs_connected": self.obs_connected,
            "ts": _iso(self.ts),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


Event = MatchEndEvent | StatusEvent


@dataclass(frozen=True)
class ArmCommand:
    pass


@dataclass(frozen=True)
class DisarmCommand:
    pass


@dataclass(frozen=True)
class SetGameCommand:
    game: Game


ParsedCommand = ArmCommand | DisarmCommand | SetGameCommand


def parse_command(raw: str) -> ParsedCommand:
    """Parse an inbound dashboard message, rejecting anything unrecognized.

    Raises CommandError rather than returning a sentinel: an unknown command is
    a bug in the caller, and swallowing it silently is exactly the failure mode
    enums exist to prevent.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CommandError(f"not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise CommandError(f"expected a JSON object, got {type(payload).__name__}")

    raw_cmd = payload.get("cmd")
    try:
        command = Command(raw_cmd)
    except ValueError as exc:
        raise CommandError(f"unknown command: {raw_cmd!r}") from exc

    match command:
        case Command.ARM:
            return ArmCommand()
        case Command.DISARM:
            return DisarmCommand()
        case Command.SET_GAME:
            raw_game = payload.get("game")
            if raw_game is None:
                raise CommandError("set_game requires a 'game' field")
            try:
                return SetGameCommand(Game(raw_game))
            except ValueError as exc:
                raise CommandError(f"unknown game: {raw_game!r}") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_events.py -v`
Expected: PASS, 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/fgc_detector/events.py tests/test_events.py
git commit -m "feat: event serialization and command parsing at the JSON boundary"
```

---

### Task 3: Frame normalization

**Files:**
- Create: `src/fgc_detector/frames/__init__.py`, `src/fgc_detector/frames/normalize.py`
- Test: `tests/frames/test_normalize.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `normalize(image: np.ndarray, canonical: tuple[int, int], aspect_tolerance: float = 0.02) -> np.ndarray | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/frames/__init__.py` (empty) and `tests/frames/test_normalize.py`:

```python
import numpy as np

from fgc_detector.frames.normalize import normalize

CANONICAL = (1920, 1080)


def _image(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), 128, dtype=np.uint8)


def test_already_canonical_is_returned_unchanged():
    image = _image(1920, 1080)
    result = normalize(image, CANONICAL)
    assert result is image


def test_smaller_16_9_is_upscaled_to_canonical():
    result = normalize(_image(1280, 720), CANONICAL)
    assert result is not None
    assert result.shape == (1080, 1920, 3)


def test_larger_16_9_is_downscaled_to_canonical():
    result = normalize(_image(3840, 2160), CANONICAL)
    assert result is not None
    assert result.shape == (1080, 1920, 3)


def test_wrong_aspect_ratio_is_rejected():
    # 4:3 capture — sampling fixed 16:9 ROIs against it would read misaligned
    # pixels and report confident nonsense, so refuse instead.
    assert normalize(_image(1024, 768), CANONICAL) is None


def test_pillarboxed_ultrawide_is_rejected():
    assert normalize(_image(2560, 1080), CANONICAL) is None


def test_empty_image_is_rejected():
    assert normalize(np.zeros((0, 0, 3), dtype=np.uint8), CANONICAL) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/frames/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fgc_detector.frames'`

- [ ] **Step 3: Write the implementation**

Create `src/fgc_detector/frames/__init__.py` (empty file) and `src/fgc_detector/frames/normalize.py`:

```python
"""Resolution normalization.

Detectors sample fixed pixel rectangles. Those rectangles are only meaningful at
one aspect ratio, so a frame whose aspect does not match is rejected outright
rather than squashed to fit — a squashed frame produces confident garbage, which
is worse than no reading at all.
"""

from __future__ import annotations

import cv2
import numpy as np


def normalize(
    image: np.ndarray,
    canonical: tuple[int, int],
    aspect_tolerance: float = 0.02,
) -> np.ndarray | None:
    """Scale `image` to `canonical` (width, height), or return None if it can't be.

    Returns None for empty images and for any aspect ratio differing from the
    canonical one by more than `aspect_tolerance` (relative).
    """
    if image.size == 0 or image.ndim != 3:
        return None

    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return None

    target_width, target_height = canonical
    if (width, height) == (target_width, target_height):
        return image

    target_aspect = target_width / target_height
    actual_aspect = width / height
    if abs(actual_aspect - target_aspect) / target_aspect > aspect_tolerance:
        return None

    # INTER_AREA for downscale preserves the flat colour regions that fill-ratio
    # sampling depends on; INTER_LINEAR is the right choice going up.
    interpolation = cv2.INTER_AREA if width > target_width else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/frames/test_normalize.py -v`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/fgc_detector/frames/ tests/frames/
git commit -m "feat: frame normalization with aspect-ratio rejection"
```

---

### Task 4: Offline frame sources

**Files:**
- Create: `src/fgc_detector/frames/source.py`, `src/fgc_detector/frames/offline.py`
- Test: `tests/frames/test_offline.py`

**Interfaces:**
- Consumes: `Frame` from `types.py`; `normalize` from `frames/normalize.py`.
- Produces: `FrameSource` protocol with `frames() -> Iterator[Frame]` and `close() -> None`; `FolderFrameSource(path, canonical)`; `VideoFrameSource(path, canonical, sample_every=1)`.

- [ ] **Step 1: Write the failing test**

Create `tests/frames/test_offline.py`:

```python
from datetime import datetime, timezone

import cv2
import numpy as np
import pytest

from fgc_detector.frames.offline import FolderFrameSource, VideoFrameSource

CANONICAL = (1920, 1080)


@pytest.fixture
def png_dir(tmp_path):
    for index, value in enumerate([10, 20, 30]):
        image = np.full((720, 1280, 3), value, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"frame_{index:04d}.png"), image)
    return tmp_path


def test_folder_source_yields_frames_in_filename_order(png_dir):
    source = FolderFrameSource(png_dir, CANONICAL)
    frames = list(source.frames())
    assert len(frames) == 3
    assert [int(frame.image[0, 0, 0]) for frame in frames] == [10, 20, 30]


def test_folder_source_normalizes_to_canonical(png_dir):
    frames = list(FolderFrameSource(png_dir, CANONICAL).frames())
    assert all(frame.image.shape == (1080, 1920, 3) for frame in frames)


def test_folder_source_frames_are_timezone_aware(png_dir):
    frame = next(iter(FolderFrameSource(png_dir, CANONICAL).frames()))
    assert frame.captured_at.tzinfo is not None


def test_folder_source_skips_unreadable_and_wrong_aspect_files(tmp_path):
    cv2.imwrite(str(tmp_path / "a_good.png"), np.full((720, 1280, 3), 5, dtype=np.uint8))
    cv2.imwrite(str(tmp_path / "b_bad_aspect.png"), np.full((768, 1024, 3), 5, dtype=np.uint8))
    (tmp_path / "c_not_an_image.png").write_text("garbage")
    frames = list(FolderFrameSource(tmp_path, CANONICAL).frames())
    assert len(frames) == 1


def test_folder_source_on_empty_directory_yields_nothing(tmp_path):
    assert list(FolderFrameSource(tmp_path, CANONICAL).frames()) == []


def test_video_source_yields_every_nth_frame(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720))
    for index in range(10):
        writer.write(np.full((720, 1280, 3), index * 10, dtype=np.uint8))
    writer.release()

    frames = list(VideoFrameSource(path, CANONICAL, sample_every=5).frames())
    assert len(frames) == 2


def test_video_source_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(VideoFrameSource(tmp_path / "nope.mp4", CANONICAL).frames())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/frames/test_offline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fgc_detector.frames.offline'`

- [ ] **Step 3: Write the `FrameSource` protocol**

Create `src/fgc_detector/frames/source.py`:

```python
"""The frame-source seam.

Three implementations exist: a folder of PNGs and a video file (both offline,
used by tests and by ROI tuning) and OBS (live). Everything downstream is
written against this protocol so tuning a detector never requires OBS running.
"""

from __future__ import annotations

from typing import Iterator, Protocol

from ..types import Frame


class FrameSource(Protocol):
    def frames(self) -> Iterator[Frame]:
        """Yield frames until exhausted or closed."""
        ...

    def close(self) -> None:
        """Release any underlying resource. Safe to call more than once."""
        ...
```

- [ ] **Step 4: Write the offline implementations**

Create `src/fgc_detector/frames/offline.py`:

```python
"""Offline frame sources: a folder of stills, and a recorded video.

These are what make per-game detector work tractable — a detector can be tuned
against last week's VOD at many times realtime instead of in front of a console.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import cv2

from ..types import Frame
from .normalize import normalize

log = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class FolderFrameSource:
    """Yields every readable image in a directory, sorted by filename."""

    def __init__(self, path: Path, canonical: tuple[int, int]) -> None:
        self._path = Path(path)
        self._canonical = canonical

    def frames(self) -> Iterator[Frame]:
        paths = sorted(
            entry
            for entry in self._path.iterdir()
            if entry.is_file() and entry.suffix.lower() in _IMAGE_SUFFIXES
        )
        for entry in paths:
            image = cv2.imread(str(entry))
            if image is None:
                log.warning("skipping unreadable image: %s", entry)
                continue
            normalized = normalize(image, self._canonical)
            if normalized is None:
                log.warning("skipping wrong-aspect image: %s %s", entry, image.shape)
                continue
            yield Frame(image=normalized, captured_at=datetime.now(timezone.utc))

    def close(self) -> None:
        return None


class VideoFrameSource:
    """Yields every `sample_every`-th frame of a video file."""

    def __init__(
        self, path: Path, canonical: tuple[int, int], sample_every: int = 1
    ) -> None:
        if sample_every < 1:
            raise ValueError(f"sample_every must be >= 1, got {sample_every}")
        self._path = Path(path)
        self._canonical = canonical
        self._sample_every = sample_every
        self._capture: cv2.VideoCapture | None = None

    def frames(self) -> Iterator[Frame]:
        if not self._path.exists():
            raise FileNotFoundError(self._path)
        self._capture = cv2.VideoCapture(str(self._path))
        if not self._capture.isOpened():
            raise FileNotFoundError(f"could not open video: {self._path}")

        index = 0
        try:
            while True:
                ok, image = self._capture.read()
                if not ok:
                    return
                if index % self._sample_every == 0:
                    normalized = normalize(image, self._canonical)
                    if normalized is not None:
                        yield Frame(
                            image=normalized, captured_at=datetime.now(timezone.utc)
                        )
                index += 1
        finally:
            self.close()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/frames/test_offline.py -v`
Expected: PASS, 7 passed

- [ ] **Step 6: Commit**

```bash
git add src/fgc_detector/frames/source.py src/fgc_detector/frames/offline.py tests/frames/test_offline.py
git commit -m "feat: folder and video frame sources"
```

---

### Task 5: OBS frame source

**Files:**
- Create: `src/fgc_detector/frames/obs.py`
- Test: `tests/frames/test_obs.py`

**Interfaces:**
- Consumes: `Frame`, `normalize`, `FrameSource`.
- Produces: `ObsFrameSource(client_factory, source_name, canonical, poll_hz=5.0, sleeper=time.sleep)` with `frames()`, `close()`, and a `connected: bool` property.

**Background:** obs-websocket v5's `GetSourceScreenshot` returns a data URI of the form `data:image/png;base64,<payload>`. It targets a **named source**, not the program output — this is the whole reason for choosing it, because program output composites commentator cams and sponsor overlays over the ROIs. Requesting a downscaled image keeps the cost off OBS's graphics thread; polling above ~10Hz is wasteful for match-end detection.

- [ ] **Step 1: Write the failing test**

Create `tests/frames/test_obs.py`:

```python
import base64

import cv2
import numpy as np
import pytest

from fgc_detector.frames.obs import ObsFrameSource

CANONICAL = (1920, 1080)


def _data_uri(width: int = 1280, height: int = 720, value: int = 42) -> str:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode()


class FakeResponse:
    def __init__(self, image_data: str) -> None:
        self.image_data = image_data


class FakeClient:
    """Stands in for obsws_python.ReqClient."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.disconnected = False

    def get_source_screenshot(self, name, img_format, width, height, quality):
        self.calls.append((name, img_format, width, height))
        if not self._responses:
            raise StopIteration
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return FakeResponse(response)

    def disconnect(self):
        self.disconnected = True


def _source(responses, **kwargs):
    client = FakeClient(responses)
    source = ObsFrameSource(
        client_factory=lambda: client,
        source_name="Game Capture",
        canonical=CANONICAL,
        sleeper=lambda _seconds: None,
        **kwargs,
    )
    return source, client


def test_decodes_screenshot_into_normalized_frame():
    source, _ = _source([_data_uri()])
    frame = next(source.frames())
    assert frame.image.shape == (1080, 1920, 3)
    assert frame.captured_at.tzinfo is not None


def test_requests_the_configured_source_by_name():
    source, client = _source([_data_uri()])
    next(source.frames())
    assert client.calls[0][0] == "Game Capture"


def test_marks_connected_after_a_successful_capture():
    source, _ = _source([_data_uri()])
    next(source.frames())
    assert source.connected is True


def test_survives_a_transient_capture_error_and_recovers():
    source, _ = _source([ConnectionError("boom"), _data_uri()])
    frames = source.frames()
    frame = next(frames)
    assert frame.image.shape == (1080, 1920, 3)


def test_marks_disconnected_while_erroring():
    source, _ = _source([ConnectionError("boom"), _data_uri()])
    frames = source.frames()
    # Drive one failing attempt without consuming a frame.
    source._attempt_once()
    assert source.connected is False


def test_wrong_aspect_screenshot_yields_no_frame():
    source, _ = _source([_data_uri(width=1024, height=768), _data_uri()])
    frame = next(source.frames())
    assert frame.image.shape == (1080, 1920, 3)


def test_close_disconnects_the_client():
    source, client = _source([_data_uri()])
    next(source.frames())
    source.close()
    assert client.disconnected is True


def test_invalid_poll_rate_rejected():
    with pytest.raises(ValueError):
        ObsFrameSource(
            client_factory=lambda: FakeClient([]),
            source_name="x",
            canonical=CANONICAL,
            poll_hz=0,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/frames/test_obs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fgc_detector.frames.obs'`

- [ ] **Step 3: Write the implementation**

Create `src/fgc_detector/frames/obs.py`:

```python
"""Live frames from OBS via obs-websocket v5.

Targets a named source (the game capture) rather than program output, so
overlays, commentator cams and transition stingers can never contaminate the
ROIs a detector samples.
"""

from __future__ import annotations

import base64
import logging
import time
from datetime import datetime, timezone
from typing import Callable, Iterator

import cv2
import numpy as np

from ..types import Frame
from .normalize import normalize

log = logging.getLogger(__name__)

_MAX_BACKOFF_SECONDS = 10.0
_DATA_URI_PREFIX = "base64,"


class ObsFrameSource:
    def __init__(
        self,
        client_factory: Callable[[], object],
        source_name: str,
        canonical: tuple[int, int],
        poll_hz: float = 5.0,
        request_size: tuple[int, int] = (1280, 720),
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if poll_hz <= 0:
            raise ValueError(f"poll_hz must be > 0, got {poll_hz}")
        self._client_factory = client_factory
        self._source_name = source_name
        self._canonical = canonical
        self._interval = 1.0 / poll_hz
        self._request_size = request_size
        self._sleep = sleeper
        self._client: object | None = None
        self._connected = False
        self._backoff = 0.5

    @property
    def connected(self) -> bool:
        return self._connected

    def _ensure_client(self) -> object:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def _attempt_once(self) -> Frame | None:
        """One capture attempt. Returns None on any failure, never raises."""
        try:
            client = self._ensure_client()
            response = client.get_source_screenshot(
                self._source_name, "png", *self._request_size, -1
            )
            image = self._decode(response.image_data)
        except Exception as exc:  # obsws raises a wide variety; none are fatal here
            log.warning("OBS capture failed: %s", exc)
            self._connected = False
            self._client = None
            return None

        self._connected = True
        self._backoff = 0.5
        if image is None:
            return None
        normalized = normalize(image, self._canonical)
        if normalized is None:
            log.warning("OBS returned a wrong-aspect image: %s", image.shape)
            return None
        return Frame(image=normalized, captured_at=datetime.now(timezone.utc))

    @staticmethod
    def _decode(image_data: str) -> np.ndarray | None:
        _, _, payload = image_data.partition(_DATA_URI_PREFIX)
        if not payload:
            log.warning("unexpected screenshot payload format")
            return None
        raw = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)

    def frames(self) -> Iterator[Frame]:
        while True:
            frame = self._attempt_once()
            if frame is not None:
                yield frame
                self._sleep(self._interval)
                continue
            if self._connected:
                # Connected but this frame was unusable — keep the normal cadence.
                self._sleep(self._interval)
            else:
                self._sleep(self._backoff)
                self._backoff = min(self._backoff * 2, _MAX_BACKOFF_SECONDS)

    def close(self) -> None:
        client = self._client
        self._client = None
        self._connected = False
        if client is not None:
            try:
                client.disconnect()
            except Exception as exc:
                log.debug("error disconnecting OBS client: %s", exc)


def default_client_factory(host: str, port: int, password: str) -> Callable[[], object]:
    """Build a factory that opens a fresh obsws ReqClient on demand."""

    def factory() -> object:
        import obsws_python

        return obsws_python.ReqClient(host=host, port=port, password=password)

    return factory
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/frames/test_obs.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/fgc_detector/frames/obs.py tests/frames/test_obs.py
git commit -m "feat: OBS frame source with reconnect backoff"
```

---

### Task 6: ROI sampling primitives

**Files:**
- Create: `src/fgc_detector/detectors/__init__.py`, `src/fgc_detector/detectors/roi.py`
- Test: `tests/detectors/test_roi.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Roi(x, y, w, h)` with `.crop(image)`; `fill_ratio(image, roi, threshold=128)`; `mean_color(image, roi)`; `match_template(image, roi, template)`.

**Why this task exists before any game detector:** these primitives are testable against synthetic arrays with known values, so all the arithmetic is proven correct before anyone squints at a real screenshot. When an SF6 ROI later misbehaves, the bug is in the coordinates, not the math.

- [ ] **Step 1: Write the failing test**

Create `tests/detectors/__init__.py` (empty) and `tests/detectors/test_roi.py`:

```python
import numpy as np
import pytest

from fgc_detector.detectors.roi import Roi, fill_ratio, match_template, mean_color


def _black(width: int = 100, height: int = 100) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_crop_returns_the_requested_rectangle():
    image = _black()
    image[10:20, 30:50] = 255
    cropped = Roi(30, 10, 20, 10).crop(image)
    assert cropped.shape == (10, 20, 3)
    assert cropped.min() == 255


def test_roi_rejects_non_positive_size():
    with pytest.raises(ValueError):
        Roi(0, 0, 0, 10)


def test_fill_ratio_all_dark_is_zero():
    assert fill_ratio(_black(), Roi(0, 0, 10, 10)) == 0.0


def test_fill_ratio_all_bright_is_one():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert fill_ratio(image, Roi(0, 0, 10, 10)) == 1.0


def test_fill_ratio_half_bright_is_half():
    image = _black()
    image[0:5, 0:10] = 255
    assert fill_ratio(image, Roi(0, 0, 10, 10)) == pytest.approx(0.5)


def test_fill_ratio_respects_threshold():
    image = np.full((100, 100, 3), 100, dtype=np.uint8)
    assert fill_ratio(image, Roi(0, 0, 10, 10), threshold=50) == 1.0
    assert fill_ratio(image, Roi(0, 0, 10, 10), threshold=150) == 0.0


def test_roi_outside_image_bounds_returns_zero_not_a_crash():
    # A resolution change can push an ROI off the frame. Degrade to "saw
    # nothing" rather than raising in the middle of a live match.
    assert fill_ratio(_black(50, 50), Roi(40, 40, 100, 100)) == 0.0


def test_mean_color_is_bgr_order():
    image = _black()
    image[0:10, 0:10] = (255, 0, 0)  # pure blue in BGR
    blue, green, red = mean_color(image, Roi(0, 0, 10, 10))
    assert (blue, green, red) == pytest.approx((255.0, 0.0, 0.0))


def test_match_template_identical_region_scores_one():
    image = _black()
    image[10:30, 10:30] = 200
    template = image[10:30, 10:30].copy()
    assert match_template(image, Roi(10, 10, 20, 20), template) == pytest.approx(1.0, abs=1e-3)


def test_match_template_mismatched_region_scores_low():
    image = _black()
    template = np.full((20, 20, 3), 255, dtype=np.uint8)
    template[0:10] = 0  # give the template variance so correlation is defined
    assert match_template(image, Roi(10, 10, 20, 20), template) < 0.5


def test_match_template_size_mismatch_raises():
    with pytest.raises(ValueError):
        match_template(_black(), Roi(0, 0, 20, 20), np.zeros((5, 5, 3), dtype=np.uint8))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/detectors/test_roi.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fgc_detector.detectors'`

- [ ] **Step 3: Write the implementation**

Create `src/fgc_detector/detectors/__init__.py` (empty file) and `src/fgc_detector/detectors/roi.py`:

```python
"""Fixed-region sampling primitives.

Round-win markers are position-fixed and language-independent, which is why
fill-ratio sampling is preferred over OCR: no game-language requirement, and a
threshold is far cheaper than a text recognizer.

Every function degrades to a neutral reading rather than raising when an ROI
falls outside the frame. A crash mid-match is worse than a missed detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Roi:
    """A rectangle in canonical-resolution pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"ROI must have positive size, got {self.w}x{self.h}")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"ROI origin must be non-negative, got ({self.x}, {self.y})")

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Return the ROI's pixels, or an empty array if it falls outside `image`."""
        height, width = image.shape[:2]
        if self.x + self.w > width or self.y + self.h > height:
            return np.empty((0, 0, 3), dtype=image.dtype)
        return image[self.y : self.y + self.h, self.x : self.x + self.w]


def fill_ratio(image: np.ndarray, roi: Roi, threshold: int = 128) -> float:
    """Fraction of the ROI's pixels brighter than `threshold` after grayscaling.

    This is the workhorse for round-win markers: an unfilled marker is dark, a
    filled one is bright, and the ratio between them is a wide, stable gap.
    """
    patch = roi.crop(image)
    if patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(np.count_nonzero(gray > threshold) / gray.size)


def mean_color(image: np.ndarray, roi: Roi) -> tuple[float, float, float]:
    """Mean BGR of the ROI, or (0, 0, 0) if it falls outside the frame."""
    patch = roi.crop(image)
    if patch.size == 0:
        return (0.0, 0.0, 0.0)
    blue, green, red = patch.reshape(-1, 3).mean(axis=0)
    return (float(blue), float(green), float(red))


def match_template(image: np.ndarray, roi: Roi, template: np.ndarray) -> float:
    """Normalized correlation of the ROI against `template`, in 0.0–1.0.

    The template must be exactly the ROI's size; this is a fixed-position match,
    not a search, because the HUD element's location is already known.
    """
    if template.shape[:2] != (roi.h, roi.w):
        raise ValueError(
            f"template is {template.shape[:2]}, ROI is {(roi.h, roi.w)}; they must match"
        )
    patch = roi.crop(image)
    if patch.size == 0:
        return 0.0
    score = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
    return float(max(0.0, score[0][0]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/detectors/test_roi.py -v`
Expected: PASS, 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/fgc_detector/detectors/ tests/detectors/
git commit -m "feat: ROI sampling primitives"
```

---

### Task 7: Detector protocol and registry

**Files:**
- Create: `src/fgc_detector/detectors/registry.py`
- Test: `tests/detectors/test_registry.py`

**Interfaces:**
- Consumes: `Game`, `Screen`, `Observation`, `Frame` from `types.py`; `Roi` from `detectors/roi.py`.
- Produces: `Detector` protocol (`game: Game`, `canonical_size: tuple[int, int]`, `observe(frame) -> Observation`, `rois() -> dict[str, Roi]`); `register(detector)`; `get_detector(game) -> Detector`; `UnknownGameError`; `NullDetector` for tests.

- [ ] **Step 1: Write the failing test**

Create `tests/detectors/test_registry.py`:

```python
import pytest

from fgc_detector.detectors.registry import (
    NullDetector,
    UnknownGameError,
    get_detector,
    register,
)
from fgc_detector.types import Game, Screen


def test_null_detector_always_reports_unknown():
    detector = NullDetector(Game.SF6)
    observation = detector.observe(frame=None)
    assert observation.screen is Screen.UNKNOWN
    assert observation.winner is None


def test_register_then_get_returns_the_same_instance():
    detector = NullDetector(Game.TEKKEN8)
    register(detector)
    assert get_detector(Game.TEKKEN8) is detector


def test_get_unregistered_game_raises_with_a_useful_message():
    with pytest.raises(UnknownGameError, match="no detector registered"):
        get_detector(Game.SF6)


def test_registering_twice_for_one_game_raises():
    register(NullDetector(Game.SF6))
    with pytest.raises(ValueError, match="already registered"):
        register(NullDetector(Game.SF6))
```

Note for the implementer: these tests mutate module-level registry state, so add this fixture at the top of the file to isolate them:

```python
@pytest.fixture(autouse=True)
def _clean_registry():
    from fgc_detector.detectors import registry

    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/detectors/test_registry.py -v`
Expected: FAIL with `ImportError: cannot import name 'NullDetector'`

- [ ] **Step 3: Write the implementation**

Create `src/fgc_detector/detectors/registry.py`:

```python
"""The detector seam.

Detectors are stateless and pure: `observe()` classifies exactly one frame and
may not keep history, read a clock, or do I/O. All temporal reasoning lives in
the Confirmer. This is what makes adding a third game cheap — a new detector is
"read some pixels, report what you see" and inherits debounce, arming and
cooldown for free.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Frame, Game, Observation, Screen
from .roi import Roi


class UnknownGameError(LookupError):
    pass


@runtime_checkable
class Detector(Protocol):
    game: Game
    canonical_size: tuple[int, int]

    def observe(self, frame: Frame) -> Observation:
        """Classify a single frame. Pure: same frame in, same observation out."""
        ...

    def rois(self) -> dict[str, Roi]:
        """The detector's named sampling rectangles, for the `roi` CLI preview."""
        ...


_REGISTRY: dict[Game, Detector] = {}


def register(detector: Detector) -> None:
    if detector.game in _REGISTRY:
        raise ValueError(f"a detector for {detector.game.value} is already registered")
    _REGISTRY[detector.game] = detector


def get_detector(game: Game) -> Detector:
    try:
        return _REGISTRY[game]
    except KeyError as exc:
        raise UnknownGameError(f"no detector registered for {game.value}") from exc


class NullDetector:
    """Reports UNKNOWN for every frame. Used by tests and as a safe default."""

    canonical_size = (1920, 1080)

    def __init__(self, game: Game) -> None:
        self.game = game

    def observe(self, frame: Frame) -> Observation:
        return Observation(screen=Screen.UNKNOWN)

    def rois(self) -> dict[str, Roi]:
        return {}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/detectors/test_registry.py -v`
Expected: PASS, 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/fgc_detector/detectors/registry.py tests/detectors/test_registry.py
git commit -m "feat: detector protocol and registry"
```

---

### Task 8: The Confirmer state machine

**Files:**
- Create: `src/fgc_detector/confirmer.py`
- Test: `tests/test_confirmer.py`

**Interfaces:**
- Consumes: `Observation`, `Screen`, `Side`, `Game`, `ConfirmerState`; `MatchEndEvent`.
- Produces: `ConfirmerConfig(agreement_frames=3, cooldown_max_seconds=180.0)`; `Confirmer(game, config)` with `.state`, `.armed`, `.arm()`, `.disarm()`, `.set_game(game)`, `.observe(observation, now) -> MatchEndEvent | None`.

**This is the most important task in the plan.** Every false positive the system will ever produce is prevented here or not at all.

State machine:

```
IDLE ──sees IN_MATCH──▶ LIVE ──N consecutive MATCH_END, agreeing payload──▶ FIRE
  ▲                                                                          │
  └── CHAR_SELECT, disarm, or cooldown_max_seconds safety valve ─ COOLDOWN ◀─┘
```

**Cooldown exits only on CHAR_SELECT** (or a disarm/arm cycle, or a long safety-valve timeout). It deliberately does *not* exit on "some seconds without MATCH_END", because the post-match replay in SF6 and Tekken shows real gameplay with a real HUD: a time-based exit would return to IDLE during the replay, see the replayed IN_MATCH, go LIVE, and then fire again on the replayed KO. Requiring a definitive between-games screen is the only exit that survives that sequence. The safety valve exists so a missed CHAR_SELECT detection cannot wedge the detector forever; arming is the primary reset in practice, since the dashboard disarms after reporting.

- [ ] **Step 1: Write the failing test**

Create `tests/test_confirmer.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest

from fgc_detector.confirmer import Confirmer, ConfirmerConfig
from fgc_detector.types import ConfirmerState, Game, Observation, Screen, Side

START = datetime(2026, 7, 21, 20, 0, 0, tzinfo=timezone.utc)


def _in_match() -> Observation:
    return Observation(screen=Screen.IN_MATCH)


def _match_end(winner: Side, confidence: float = 0.9) -> Observation:
    return Observation(screen=Screen.MATCH_END, winner=winner, confidence=confidence)


def _char_select() -> Observation:
    return Observation(screen=Screen.CHAR_SELECT)


class Driver:
    """Feeds observations to a Confirmer on a deterministic clock."""

    def __init__(self, confirmer: Confirmer, step: float = 0.2) -> None:
        self.confirmer = confirmer
        self.now = START
        self.step = timedelta(seconds=step)
        self.events = []

    def feed(self, observation: Observation, times: int = 1):
        for _ in range(times):
            event = self.confirmer.observe(observation, self.now)
            if event is not None:
                self.events.append(event)
            self.now += self.step
        return self

    def advance(self, seconds: float):
        self.now += timedelta(seconds=seconds)
        return self


@pytest.fixture
def driver():
    confirmer = Confirmer(Game.SF6, ConfirmerConfig(agreement_frames=3))
    confirmer.arm()
    return Driver(confirmer)


def test_starts_idle_and_disarmed():
    confirmer = Confirmer(Game.SF6, ConfirmerConfig())
    assert confirmer.state is ConfirmerState.IDLE
    assert confirmer.armed is False


def test_fires_after_n_agreeing_match_end_frames(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    assert len(driver.events) == 1
    assert driver.events[0].winner is Side.P1
    assert driver.events[0].game is Game.SF6


def test_does_not_fire_before_n_frames(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 2)
    assert driver.events == []


def test_does_not_fire_when_frames_disagree_on_winner(driver):
    driver.feed(_in_match(), 5)
    driver.feed(_match_end(Side.P1), 2).feed(_match_end(Side.P2), 2)
    assert driver.events == []


def test_disagreement_restarts_the_streak_rather_than_resetting_to_zero(driver):
    driver.feed(_in_match(), 5)
    driver.feed(_match_end(Side.P1), 2).feed(_match_end(Side.P2), 3)
    assert len(driver.events) == 1
    assert driver.events[0].winner is Side.P2


def test_confidence_jitter_does_not_break_agreement(driver):
    driver.feed(_in_match(), 5)
    for confidence in (0.81, 0.93, 0.88):
        driver.feed(_match_end(Side.P1, confidence))
    assert len(driver.events) == 1


def test_reported_confidence_is_the_minimum_of_the_streak(driver):
    driver.feed(_in_match(), 5)
    for confidence in (0.9, 0.7, 0.95):
        driver.feed(_match_end(Side.P1, confidence))
    assert driver.events[0].confidence == pytest.approx(0.7)


def test_cannot_fire_from_idle_without_seeing_a_live_match(driver):
    driver.feed(_match_end(Side.P1), 10)
    assert driver.events == []


def test_in_match_frames_interrupt_a_partial_streak(driver):
    driver.feed(_in_match(), 5)
    driver.feed(_match_end(Side.P1), 2).feed(_in_match(), 1).feed(_match_end(Side.P1), 2)
    assert driver.events == []


def test_fires_only_once_for_a_sustained_match_end_screen(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 60)
    assert len(driver.events) == 1


def test_post_match_replay_does_not_fire_a_phantom_second_event(driver):
    """The single most important regression test in the suite.

    After a set, SF6 and Tekken show a replay: real gameplay, real HUD, and a
    real KO at the end of it. A detector without cooldown reports that replayed
    KO as a second game.
    """
    driver.feed(_in_match(), 10).feed(_match_end(Side.P1), 3)
    assert len(driver.events) == 1

    driver.advance(10)  # replay begins
    driver.feed(_in_match(), 30).feed(_match_end(Side.P1), 10)
    assert len(driver.events) == 1, "replayed KO must not fire a second event"


def test_char_select_ends_cooldown_and_the_next_match_can_fire(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    driver.feed(_char_select(), 5)
    assert driver.confirmer.state is ConfirmerState.IDLE
    driver.feed(_in_match(), 5).feed(_match_end(Side.P2), 3)
    assert len(driver.events) == 2
    assert driver.events[1].winner is Side.P2


def test_safety_valve_releases_cooldown_if_char_select_is_never_seen(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    assert driver.confirmer.state is ConfirmerState.COOLDOWN
    driver.advance(181).feed(Observation(Screen.UNKNOWN))
    assert driver.confirmer.state is ConfirmerState.IDLE


def test_disarmed_confirmer_never_fires():
    confirmer = Confirmer(Game.SF6, ConfirmerConfig(agreement_frames=3))
    driver = Driver(confirmer)  # never armed
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 10)
    assert driver.events == []


def test_disarming_mid_streak_discards_it(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 2)
    driver.confirmer.disarm()
    driver.confirmer.arm()
    driver.feed(_match_end(Side.P1), 2)
    assert driver.events == []


def test_arming_resets_state_to_idle(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    assert driver.confirmer.state is ConfirmerState.COOLDOWN
    driver.confirmer.arm()
    assert driver.confirmer.state is ConfirmerState.IDLE


def test_set_game_changes_the_reported_game_and_resets(driver):
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 2)
    driver.confirmer.set_game(Game.TEKKEN8)
    assert driver.confirmer.state is ConfirmerState.IDLE
    driver.feed(_in_match(), 5).feed(_match_end(Side.P1), 3)
    assert driver.events[0].game is Game.TEKKEN8


def test_unknown_screens_do_not_disturb_a_live_match(driver):
    driver.feed(_in_match(), 5).feed(Observation(Screen.UNKNOWN), 3)
    assert driver.confirmer.state is ConfirmerState.LIVE
    driver.feed(_match_end(Side.P1), 3)
    assert len(driver.events) == 1


def test_match_end_without_a_winner_is_ignored(driver):
    driver.feed(_in_match(), 5).feed(Observation(Screen.MATCH_END, winner=None), 10)
    assert driver.events == []


def test_agreement_frames_must_be_positive():
    with pytest.raises(ValueError):
        ConfirmerConfig(agreement_frames=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_confirmer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fgc_detector.confirmer'`

- [ ] **Step 3: Write the implementation**

Create `src/fgc_detector/confirmer.py`:

```python
"""The state machine that turns per-frame observations into events.

All temporal logic lives here and nowhere else, so per-game detectors stay pure
and every false-positive defence is tested in one place against synthetic
observation sequences — no images, no OBS, no real clock.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .events import MatchEndEvent
from .types import ConfirmerState, Game, Observation, Screen, Side

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfirmerConfig:
    agreement_frames: int = 3
    cooldown_max_seconds: float = 180.0

    def __post_init__(self) -> None:
        if self.agreement_frames < 1:
            raise ValueError(
                f"agreement_frames must be >= 1, got {self.agreement_frames}"
            )
        if self.cooldown_max_seconds <= 0:
            raise ValueError(
                f"cooldown_max_seconds must be > 0, got {self.cooldown_max_seconds}"
            )


class Confirmer:
    def __init__(self, game: Game, config: ConfirmerConfig) -> None:
        self._game = game
        self._config = config
        self._armed = False
        self._state = ConfirmerState.IDLE
        self._streak: list[Observation] = []
        self._cooldown_started: datetime | None = None

    @property
    def state(self) -> ConfirmerState:
        return self._state

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def game(self) -> Game:
        return self._game

    def arm(self) -> None:
        """Arm and reset. The dashboard calls this when a set is loaded."""
        self._armed = True
        self._reset()

    def disarm(self) -> None:
        self._armed = False
        self._reset()

    def set_game(self, game: Game) -> None:
        self._game = game
        self._reset()

    def _reset(self) -> None:
        self._state = ConfirmerState.IDLE
        self._streak.clear()
        self._cooldown_started = None

    def observe(self, observation: Observation, now: datetime) -> MatchEndEvent | None:
        """Feed one observation. Returns an event only when one is confirmed."""
        if not self._armed:
            return None

        match self._state:
            case ConfirmerState.IDLE:
                return self._observe_idle(observation)
            case ConfirmerState.LIVE:
                return self._observe_live(observation, now)
            case ConfirmerState.COOLDOWN:
                return self._observe_cooldown(observation, now)

    def _observe_idle(self, observation: Observation) -> None:
        if observation.screen is Screen.IN_MATCH:
            self._state = ConfirmerState.LIVE
            self._streak.clear()
        return None

    def _observe_live(
        self, observation: Observation, now: datetime
    ) -> MatchEndEvent | None:
        if observation.screen is Screen.CHAR_SELECT:
            # The match was abandoned or we misread; start over cleanly.
            self._reset()
            return None

        if observation.screen is Screen.IN_MATCH:
            self._streak.clear()
            return None

        if observation.screen is not Screen.MATCH_END or observation.winner is None:
            # UNKNOWN frames are common (transitions, flashes) and must not
            # break a run of agreeing MATCH_END frames, but they don't extend
            # one either.
            return None

        if self._streak and self._streak[-1].payload != observation.payload:
            self._streak.clear()
        self._streak.append(observation)

        if len(self._streak) < self._config.agreement_frames:
            return None

        winner = observation.winner
        confidence = min(item.confidence for item in self._streak)
        self._state = ConfirmerState.COOLDOWN
        self._cooldown_started = now
        self._streak.clear()
        log.info(
            "confirmed match_end game=%s winner=%s confidence=%.4f",
            self._game.value,
            winner.value,
            confidence,
        )
        return MatchEndEvent(
            game=self._game, winner=winner, confidence=confidence, ts=now
        )

    def _observe_cooldown(self, observation: Observation, now: datetime) -> None:
        """Hold until a definitive between-games screen.

        Deliberately does not exit on elapsed time alone: the post-match replay
        shows real gameplay and a real KO, so a time-based exit would re-arm
        mid-replay and fire on the replayed KO. CHAR_SELECT is the only signal
        that unambiguously means "the previous game is over". The safety valve
        below prevents a missed CHAR_SELECT from wedging the detector forever.
        """
        if observation.screen is Screen.CHAR_SELECT:
            self._reset()
            return None

        if self._cooldown_started is not None:
            elapsed = now - self._cooldown_started
            if elapsed > timedelta(seconds=self._config.cooldown_max_seconds):
                log.warning(
                    "cooldown safety valve released after %.0fs without CHAR_SELECT; "
                    "the character-select ROI may be miscalibrated",
                    elapsed.total_seconds(),
                )
                self._reset()
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_confirmer.py -v`
Expected: PASS, 20 passed

- [ ] **Step 5: Commit**

```bash
git add src/fgc_detector/confirmer.py tests/test_confirmer.py
git commit -m "feat: confirmer state machine with replay-mode cooldown"
```

---

### Task 9: WebSocket server

**Files:**
- Create: `src/fgc_detector/server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `MatchEndEvent`, `StatusEvent`, `parse_command`, `CommandError`, `ArmCommand`, `DisarmCommand`, `SetGameCommand`; `Confirmer`.
- Produces: `EventServer(confirmer, host, port, obs_connected_getter)` with `async serve()`, `async broadcast(event)`, `async handle_client(websocket)`, and `status_event(now) -> StatusEvent`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_server.py`:

```python
import json
from datetime import datetime, timezone

import pytest

from fgc_detector.confirmer import Confirmer, ConfirmerConfig
from fgc_detector.server import EventServer
from fgc_detector.types import ConfirmerState, Game, Side

TS = datetime(2026, 7, 21, 20, 0, 0, tzinfo=timezone.utc)


class FakeSocket:
    """Minimal stand-in for a websockets server connection."""

    def __init__(self, inbound=()):
        self.sent: list[str] = []
        self._inbound = list(inbound)

    async def send(self, message: str) -> None:
        self.sent.append(message)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._inbound:
            raise StopAsyncIteration
        return self._inbound.pop(0)

    def payloads(self) -> list[dict]:
        return [json.loads(message) for message in self.sent]


@pytest.fixture
def server():
    confirmer = Confirmer(Game.SF6, ConfirmerConfig())
    return EventServer(
        confirmer=confirmer, host="127.0.0.1", port=0, obs_connected_getter=lambda: True
    )


@pytest.mark.asyncio
async def test_new_client_immediately_receives_status(server):
    socket = FakeSocket()
    await server.handle_client(socket)
    first = socket.payloads()[0]
    assert first["type"] == "status"
    assert first["game"] == "sf6"
    assert first["armed"] is False


@pytest.mark.asyncio
async def test_arm_command_arms_the_confirmer_and_echoes_status(server):
    socket = FakeSocket(['{"cmd":"arm"}'])
    await server.handle_client(socket)
    assert server.confirmer.armed is True
    assert socket.payloads()[-1]["armed"] is True


@pytest.mark.asyncio
async def test_disarm_command_disarms(server):
    server.confirmer.arm()
    socket = FakeSocket(['{"cmd":"disarm"}'])
    await server.handle_client(socket)
    assert server.confirmer.armed is False


@pytest.mark.asyncio
async def test_set_game_command_switches_game(server):
    socket = FakeSocket(['{"cmd":"set_game","game":"tekken8"}'])
    await server.handle_client(socket)
    assert server.confirmer.game is Game.TEKKEN8
    assert socket.payloads()[-1]["game"] == "tekken8"


@pytest.mark.asyncio
async def test_bad_command_returns_an_error_and_keeps_the_connection(server):
    socket = FakeSocket(['{"cmd":"nonsense"}', '{"cmd":"arm"}'])
    await server.handle_client(socket)
    payloads = socket.payloads()
    assert any(item.get("error") for item in payloads)
    assert server.confirmer.armed is True, "connection must survive a bad command"


@pytest.mark.asyncio
async def test_broadcast_reaches_every_connected_client(server):
    first, second = FakeSocket(), FakeSocket()
    server._clients.update({first, second})
    await server.broadcast(MatchEndEventFactory())
    assert json.loads(first.sent[-1])["type"] == "match_end"
    assert json.loads(second.sent[-1])["type"] == "match_end"


@pytest.mark.asyncio
async def test_broadcast_drops_a_client_that_errors(server):
    class Broken(FakeSocket):
        async def send(self, message):
            raise ConnectionResetError

    broken, healthy = Broken(), FakeSocket()
    server._clients.update({broken, healthy})
    await server.broadcast(MatchEndEventFactory())
    assert broken not in server._clients
    assert healthy in server._clients


def MatchEndEventFactory():
    from fgc_detector.events import MatchEndEvent

    return MatchEndEvent(game=Game.SF6, winner=Side.P1, confidence=0.9, ts=TS)


def test_status_event_reflects_confirmer_state(server):
    server.confirmer.arm()
    status = server.status_event(TS)
    assert status.armed is True
    assert status.state is ConfirmerState.IDLE
    assert status.obs_connected is True
```

Add `pytest-asyncio` to the dev dependency group and configure it:

```toml
[dependency-groups]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fgc_detector.server'`

- [ ] **Step 3: Write the implementation**

Create `src/fgc_detector/server.py`:

```python
"""WebSocket server: the detector's only contact with the outside world.

Emits events, accepts arm/disarm/set_game commands, and pushes a status event on
connect and on every state change so a freshly-connected dashboard never has to
poll. The server knows nothing about brackets, sets, or scores.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable

import websockets

from .confirmer import Confirmer
from .events import (
    ArmCommand,
    CommandError,
    DisarmCommand,
    Event,
    SetGameCommand,
    StatusEvent,
    parse_command,
)

log = logging.getLogger(__name__)


class EventServer:
    def __init__(
        self,
        confirmer: Confirmer,
        host: str,
        port: int,
        obs_connected_getter: Callable[[], bool],
    ) -> None:
        self.confirmer = confirmer
        self._host = host
        self._port = port
        self._obs_connected = obs_connected_getter
        self._clients: set[Any] = set()

    def status_event(self, now: datetime) -> StatusEvent:
        return StatusEvent(
            game=self.confirmer.game,
            armed=self.confirmer.armed,
            state=self.confirmer.state,
            obs_connected=self._obs_connected(),
            ts=now,
        )

    async def broadcast(self, event: Event) -> None:
        message = event.to_json()
        for client in list(self._clients):
            try:
                await client.send(message)
            except Exception as exc:
                log.info("dropping client after send failure: %s", exc)
                self._clients.discard(client)

    async def _send_status(self, socket: Any) -> None:
        await socket.send(self.status_event(datetime.now(timezone.utc)).to_json())

    async def handle_client(self, socket: Any) -> None:
        self._clients.add(socket)
        try:
            await self._send_status(socket)
            async for raw in socket:
                await self._handle_message(socket, raw)
        finally:
            self._clients.discard(socket)

    async def _handle_message(self, socket: Any, raw: str) -> None:
        try:
            command = parse_command(raw)
        except CommandError as exc:
            log.warning("rejected inbound message: %s", exc)
            await socket.send(json.dumps({"error": str(exc)}))
            return

        match command:
            case ArmCommand():
                self.confirmer.arm()
            case DisarmCommand():
                self.confirmer.disarm()
            case SetGameCommand(game=game):
                self.confirmer.set_game(game)

        await self._send_status(socket)

    async def serve(self) -> None:
        log.info("event server listening on ws://%s:%s", self._host, self._port)
        async with websockets.serve(self.handle_client, self._host, self._port):
            await asyncio.Future()  # run forever
```

Note: add `import asyncio` to the imports — `serve()` uses it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/fgc_detector/server.py tests/test_server.py
git commit -m "feat: websocket event server with arm/disarm commands"
```

---

### Task 10: Configuration

**Files:**
- Create: `src/fgc_detector/config.py`, `config.example.toml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `Game`; `ConfirmerConfig`.
- Produces: `AppConfig` with `.obs`, `.server`, `.game`, `.confirmer` sections; `load_config(path) -> AppConfig`; `ConfigError`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
import pytest

from fgc_detector.config import ConfigError, load_config
from fgc_detector.types import Game

VALID = """
game = "sf6"

[obs]
host = "localhost"
port = 4455
password = "secret"
source_name = "Game Capture"
poll_hz = 5.0

[server]
host = "127.0.0.1"
port = 6600

[confirmer]
agreement_frames = 3
cooldown_max_seconds = 180.0
"""


def _write(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_loads_a_valid_config(tmp_path):
    config = load_config(_write(tmp_path, VALID))
    assert config.game is Game.SF6
    assert config.obs.source_name == "Game Capture"
    assert config.obs.poll_hz == 5.0
    assert config.server.port == 6600
    assert config.confirmer.agreement_frames == 3


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.toml")


def test_unknown_game_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="unknown game"):
        load_config(_write(tmp_path, VALID.replace('"sf6"', '"smash"')))


def test_missing_source_name_raises_config_error(tmp_path):
    text = VALID.replace('source_name = "Game Capture"\n', "")
    with pytest.raises(ConfigError, match="source_name"):
        load_config(_write(tmp_path, text))


def test_defaults_are_applied_for_optional_fields(tmp_path):
    minimal = """
game = "tekken8"

[obs]
source_name = "Capture"
"""
    config = load_config(_write(tmp_path, minimal))
    assert config.obs.host == "localhost"
    assert config.obs.port == 4455
    assert config.server.port == 6600
    assert config.confirmer.agreement_frames == 3


def test_malformed_toml_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="could not be parsed"):
        load_config(_write(tmp_path, "this is [not toml"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fgc_detector.config'`

- [ ] **Step 3: Write the implementation**

Create `src/fgc_detector/config.py`:

```python
"""TOML configuration loading, with every failure reported as ConfigError."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .confirmer import ConfirmerConfig
from .types import Game


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ObsConfig:
    source_name: str
    host: str = "localhost"
    port: int = 4455
    password: str = ""
    poll_hz: float = 5.0


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 6600


@dataclass(frozen=True)
class AppConfig:
    game: Game
    obs: ObsConfig
    server: ServerConfig
    confirmer: ConfirmerConfig


def load_config(path: Path) -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    try:
        raw: dict[str, Any] = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config could not be parsed: {exc}") from exc

    try:
        game = Game(raw.get("game"))
    except ValueError as exc:
        raise ConfigError(f"unknown game: {raw.get('game')!r}") from exc

    obs_section = raw.get("obs", {})
    source_name = obs_section.get("source_name")
    if not source_name:
        raise ConfigError("obs.source_name is required")

    try:
        obs = ObsConfig(
            source_name=source_name,
            host=obs_section.get("host", "localhost"),
            port=int(obs_section.get("port", 4455)),
            password=obs_section.get("password", ""),
            poll_hz=float(obs_section.get("poll_hz", 5.0)),
        )
        server_section = raw.get("server", {})
        server = ServerConfig(
            host=server_section.get("host", "127.0.0.1"),
            port=int(server_section.get("port", 6600)),
        )
        confirmer_section = raw.get("confirmer", {})
        confirmer = ConfirmerConfig(
            agreement_frames=int(confirmer_section.get("agreement_frames", 3)),
            cooldown_max_seconds=float(
                confirmer_section.get("cooldown_max_seconds", 180.0)
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid config value: {exc}") from exc

    return AppConfig(game=game, obs=obs, server=server, confirmer=confirmer)
```

- [ ] **Step 4: Create `config.example.toml`**

```toml
# Which game the detector is currently watching.
# The dashboard can change this at runtime with {"cmd":"set_game","game":"..."}.
game = "sf6"

[obs]
# The name of the GAME CAPTURE source in OBS — not a scene, and not the program
# output. Sampling program output would let overlays and commentator cams sit on
# top of the regions the detector reads.
source_name = "Game Capture"
host = "localhost"
port = 4455
password = ""
# Match-end detection does not need a high rate. Above ~10Hz you are paying
# OBS's graphics thread for nothing.
poll_hz = 5.0

[server]
host = "127.0.0.1"
port = 6600

[confirmer]
# Consecutive agreeing frames required before an event fires.
agreement_frames = 3
# Safety valve: release cooldown after this long without seeing character
# select, in case that ROI is miscalibrated. Arming is the primary reset.
cooldown_max_seconds = 180.0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS, 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/fgc_detector/config.py config.example.toml tests/test_config.py
git commit -m "feat: TOML configuration loading"
```

---

### Task 11: Observability — fire logging and frame dumps

**Files:**
- Create: `src/fgc_detector/observability.py`
- Test: `tests/test_observability.py`

**Interfaces:**
- Consumes: `Frame`, `Observation`, `MatchEndEvent`.
- Produces: `FireRecorder(output_dir)` with `record(event, frame, observation) -> Path`.

**Why this task is not optional:** a game patch that restyles the HUD breaks fixed ROIs with no error at all. The detector keeps running and keeps reporting, just wrongly. A logged confidence score plus the actual triggering frame is the only way to notice drift before it costs someone a match on stream.

- [ ] **Step 1: Write the failing test**

Create `tests/test_observability.py`:

```python
import json
from datetime import datetime, timezone

import cv2
import numpy as np

from fgc_detector.events import MatchEndEvent
from fgc_detector.observability import FireRecorder
from fgc_detector.types import Frame, Game, Observation, Screen, Side

TS = datetime(2026, 7, 21, 20, 15, 30, tzinfo=timezone.utc)


def _frame() -> Frame:
    return Frame(image=np.full((1080, 1920, 3), 77, dtype=np.uint8), captured_at=TS)


def _event() -> MatchEndEvent:
    return MatchEndEvent(game=Game.SF6, winner=Side.P1, confidence=0.87, ts=TS)


def _observation() -> Observation:
    return Observation(
        screen=Screen.MATCH_END,
        winner=Side.P1,
        confidence=0.87,
        debug={"p1_marker_2": 0.94, "p2_marker_2": 0.03},
    )


def test_writes_both_a_png_and_a_json_sidecar(tmp_path):
    recorder = FireRecorder(tmp_path)
    png_path = recorder.record(_event(), _frame(), _observation())
    assert png_path.exists()
    assert png_path.with_suffix(".json").exists()


def test_png_contains_the_triggering_frame(tmp_path):
    png_path = FireRecorder(tmp_path).record(_event(), _frame(), _observation())
    image = cv2.imread(str(png_path))
    assert image.shape == (1080, 1920, 3)
    assert int(image[0, 0, 0]) == 77


def test_sidecar_records_the_full_debug_mapping(tmp_path):
    png_path = FireRecorder(tmp_path).record(_event(), _frame(), _observation())
    sidecar = json.loads(png_path.with_suffix(".json").read_text())
    assert sidecar["event"]["winner"] == "p1"
    assert sidecar["debug"] == {"p1_marker_2": 0.94, "p2_marker_2": 0.03}
    assert sidecar["screen"] == "MATCH_END"


def test_filename_is_sortable_and_identifies_the_call(tmp_path):
    png_path = FireRecorder(tmp_path).record(_event(), _frame(), _observation())
    assert png_path.name.startswith("2026-07-21T20-15-30")
    assert "sf6" in png_path.name
    assert "p1" in png_path.name


def test_creates_the_output_directory_if_absent(tmp_path):
    target = tmp_path / "nested" / "fires"
    FireRecorder(target).record(_event(), _frame(), _observation())
    assert target.is_dir()


def test_two_fires_in_the_same_second_do_not_collide(tmp_path):
    recorder = FireRecorder(tmp_path)
    first = recorder.record(_event(), _frame(), _observation())
    second = recorder.record(_event(), _frame(), _observation())
    assert first != second
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_observability.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fgc_detector.observability'`

- [ ] **Step 3: Write the implementation**

Create `src/fgc_detector/observability.py`:

```python
"""Recording every fire, so silent detector drift becomes visible.

A HUD restyle after a game patch breaks fixed ROIs without raising anything.
The detector will keep firing, confidently and wrongly. The triggering frame
plus the raw per-ROI scores are the evidence needed to spot that happening.
"""

from __future__ import annotations

import json
import logging
from itertools import count
from pathlib import Path

import cv2

from .events import MatchEndEvent
from .types import Frame, Observation

log = logging.getLogger(__name__)


class FireRecorder:
    def __init__(self, output_dir: Path) -> None:
        self._dir = Path(output_dir)
        self._counter = count()

    def record(
        self, event: MatchEndEvent, frame: Frame, observation: Observation
    ) -> Path:
        """Write the triggering frame and its scores. Returns the PNG's path."""
        self._dir.mkdir(parents=True, exist_ok=True)

        stamp = event.ts.strftime("%Y-%m-%dT%H-%M-%S")
        name = f"{stamp}_{event.game.value}_{event.winner.value}_{next(self._counter):03d}"
        png_path = self._dir / f"{name}.png"

        cv2.imwrite(str(png_path), frame.image)
        sidecar = {
            "event": event.to_dict(),
            "screen": observation.screen.name,
            "confidence": observation.confidence,
            "details": dict(observation.details),
            "debug": dict(observation.debug),
        }
        png_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
        log.info("recorded fire evidence: %s", png_path)
        return png_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_observability.py -v`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/fgc_detector/observability.py tests/test_observability.py
git commit -m "feat: record triggering frame and scores on every fire"
```

---

### Task 12: CLI and the pipeline that wires everything together

**Files:**
- Create: `src/fgc_detector/pipeline.py`, `src/fgc_detector/cli.py`
- Test: `tests/test_pipeline.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 1–11.
- Produces: `run_offline(source, detector, confirmer, recorder=None) -> list[MatchEndEvent]`; `main(argv) -> int` with subcommands `run`, `capture`, `roi`, `replay`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline.py`:

```python
from datetime import datetime, timedelta, timezone

import numpy as np

from fgc_detector.confirmer import Confirmer, ConfirmerConfig
from fgc_detector.pipeline import run_offline
from fgc_detector.types import Frame, Game, Observation, Screen, Side

START = datetime(2026, 7, 21, 20, 0, 0, tzinfo=timezone.utc)


class ScriptedDetector:
    """Returns a pre-written observation per frame, ignoring pixels."""

    game = Game.SF6
    canonical_size = (1920, 1080)

    def __init__(self, script):
        self._script = list(script)
        self._index = 0

    def observe(self, frame):
        observation = self._script[self._index]
        self._index += 1
        return observation


class ScriptedSource:
    def __init__(self, count):
        self._count = count

    def frames(self):
        for index in range(self._count):
            yield Frame(
                image=np.zeros((1080, 1920, 3), dtype=np.uint8),
                captured_at=START + timedelta(seconds=index * 0.2),
            )

    def close(self):
        pass


def test_pipeline_emits_one_event_for_a_clean_match():
    script = [Observation(Screen.IN_MATCH)] * 5 + [
        Observation(Screen.MATCH_END, winner=Side.P1, confidence=0.9)
    ] * 3
    confirmer = Confirmer(Game.SF6, ConfirmerConfig(agreement_frames=3))
    confirmer.arm()
    events = run_offline(ScriptedSource(len(script)), ScriptedDetector(script), confirmer)
    assert len(events) == 1
    assert events[0].winner is Side.P1


def test_pipeline_emits_nothing_when_disarmed():
    script = [Observation(Screen.IN_MATCH)] * 5 + [
        Observation(Screen.MATCH_END, winner=Side.P1, confidence=0.9)
    ] * 3
    confirmer = Confirmer(Game.SF6, ConfirmerConfig(agreement_frames=3))
    events = run_offline(ScriptedSource(len(script)), ScriptedDetector(script), confirmer)
    assert events == []


def test_pipeline_uses_the_frame_timestamp_not_wall_clock():
    script = [Observation(Screen.IN_MATCH)] * 2 + [
        Observation(Screen.MATCH_END, winner=Side.P2, confidence=0.8)
    ] * 3
    confirmer = Confirmer(Game.SF6, ConfirmerConfig(agreement_frames=3))
    confirmer.arm()
    events = run_offline(ScriptedSource(len(script)), ScriptedDetector(script), confirmer)
    assert events[0].ts == START + timedelta(seconds=0.8)


def test_pipeline_records_evidence_when_a_recorder_is_supplied(tmp_path):
    from fgc_detector.observability import FireRecorder

    script = [Observation(Screen.IN_MATCH)] * 2 + [
        Observation(Screen.MATCH_END, winner=Side.P1, confidence=0.9)
    ] * 3
    confirmer = Confirmer(Game.SF6, ConfirmerConfig(agreement_frames=3))
    confirmer.arm()
    run_offline(
        ScriptedSource(len(script)),
        ScriptedDetector(script),
        confirmer,
        recorder=FireRecorder(tmp_path),
    )
    assert len(list(tmp_path.glob("*.png"))) == 1
```

Create `tests/test_cli.py`:

```python
import pytest

from fgc_detector.cli import main


def test_no_subcommand_prints_usage_and_fails():
    with pytest.raises(SystemExit):
        main([])


def test_unknown_subcommand_fails():
    with pytest.raises(SystemExit):
        main(["frobnicate"])


def test_replay_on_a_missing_video_returns_nonzero(tmp_path, capsys):
    code = main(["replay", "--video", str(tmp_path / "absent.mp4"), "--game", "sf6"])
    assert code != 0


def test_roi_on_a_missing_sample_returns_nonzero(tmp_path):
    code = main(["roi", "--game", "sf6", "--sample", str(tmp_path / "absent.png")])
    assert code != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fgc_detector.pipeline'`

- [ ] **Step 3: Write the pipeline**

Create `src/fgc_detector/pipeline.py`:

```python
"""Wiring: frames in, events out.

Time comes from the frame's own timestamp rather than the wall clock, so a VOD
replayed at many times realtime produces exactly the events it would have
produced live.
"""

from __future__ import annotations

import logging
from typing import Iterable

from .confirmer import Confirmer
from .detectors.registry import Detector
from .events import MatchEndEvent
from .frames.source import FrameSource
from .observability import FireRecorder

log = logging.getLogger(__name__)


def run_offline(
    source: FrameSource,
    detector: Detector,
    confirmer: Confirmer,
    recorder: FireRecorder | None = None,
) -> list[MatchEndEvent]:
    """Drive the pipeline to exhaustion, returning every confirmed event."""
    events: list[MatchEndEvent] = []
    try:
        for frame in source.frames():
            observation = detector.observe(frame)
            event = confirmer.observe(observation, frame.captured_at)
            if event is None:
                continue
            events.append(event)
            if recorder is not None:
                recorder.record(event, frame, observation)
    finally:
        source.close()
    return events
```

- [ ] **Step 4: Write the CLI**

Create `src/fgc_detector/cli.py`:

```python
"""Command-line entry points.

`run`      — live: OBS frames, websocket server, the real thing
`capture`  — dump frames to disk to build a labelled sample corpus
`roi`      — draw a detector's ROIs over a sample so you can see what it reads
`replay`   — run a recorded VOD through the pipeline and print the event timeline
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import cv2

from .config import ConfigError, load_config
from .confirmer import Confirmer, ConfirmerConfig
from .detectors.registry import UnknownGameError, get_detector
from .frames.obs import ObsFrameSource, default_client_factory
from .frames.offline import VideoFrameSource
from .observability import FireRecorder
from .pipeline import run_offline
from .server import EventServer
from .types import Game

log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fgc-detect")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run live against OBS")
    run_parser.add_argument("--config", type=Path, default=Path("config.toml"))

    capture_parser = subparsers.add_parser("capture", help="dump frames to disk")
    capture_parser.add_argument("--config", type=Path, default=Path("config.toml"))
    capture_parser.add_argument("--out", type=Path, required=True)
    capture_parser.add_argument("--limit", type=int, default=0)

    roi_parser = subparsers.add_parser("roi", help="visualize a detector's ROIs")
    roi_parser.add_argument("--game", required=True)
    roi_parser.add_argument("--sample", type=Path, required=True)
    roi_parser.add_argument("--out", type=Path, default=Path("roi_preview.png"))

    replay_parser = subparsers.add_parser("replay", help="run a VOD through the pipeline")
    replay_parser.add_argument("--game", required=True)
    replay_parser.add_argument("--video", type=Path, required=True)
    replay_parser.add_argument("--sample-every", type=int, default=6)
    replay_parser.add_argument("--evidence-dir", type=Path, default=None)

    return parser


def _parse_game(value: str) -> Game:
    try:
        return Game(value)
    except ValueError:
        raise SystemExit(
            f"unknown game {value!r}; expected one of: "
            + ", ".join(item.value for item in Game)
        )


def _cmd_replay(args: argparse.Namespace) -> int:
    game = _parse_game(args.game)
    try:
        detector = get_detector(game)
    except UnknownGameError as exc:
        print(exc, file=sys.stderr)
        return 2

    confirmer = Confirmer(game, ConfirmerConfig())
    confirmer.arm()

    source = VideoFrameSource(args.video, detector.canonical_size, args.sample_every)
    recorder = FireRecorder(args.evidence_dir) if args.evidence_dir else None
    try:
        events = run_offline(source, detector, confirmer, recorder)
    except FileNotFoundError as exc:
        print(f"could not open video: {exc}", file=sys.stderr)
        return 2

    if not events:
        print("no events detected")
    for event in events:
        print(event.to_json())
    return 0


def _cmd_roi(args: argparse.Namespace) -> int:
    game = _parse_game(args.game)
    try:
        detector = get_detector(game)
    except UnknownGameError as exc:
        print(exc, file=sys.stderr)
        return 2

    image = cv2.imread(str(args.sample))
    if image is None:
        print(f"could not read sample image: {args.sample}", file=sys.stderr)
        return 2

    preview = image.copy()
    for name, roi in detector.rois().items():
        cv2.rectangle(
            preview, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 255, 0), 2
        )
        cv2.putText(
            preview, name, (roi.x, max(0, roi.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
    cv2.imwrite(str(args.out), preview)
    print(f"wrote {args.out}")
    return 0


def _cmd_capture(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    detector = get_detector(config.game)
    source = ObsFrameSource(
        client_factory=default_client_factory(
            config.obs.host, config.obs.port, config.obs.password
        ),
        source_name=config.obs.source_name,
        canonical=detector.canonical_size,
        poll_hz=config.obs.poll_hz,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        for frame in source.frames():
            path = args.out / f"frame_{written:05d}.png"
            cv2.imwrite(str(path), frame.image)
            written += 1
            if written % 10 == 0:
                print(f"captured {written} frames", file=sys.stderr)
            if args.limit and written >= args.limit:
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
    print(f"wrote {written} frames to {args.out}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    detector = get_detector(config.game)
    confirmer = Confirmer(config.game, config.confirmer)
    source = ObsFrameSource(
        client_factory=default_client_factory(
            config.obs.host, config.obs.port, config.obs.password
        ),
        source_name=config.obs.source_name,
        canonical=detector.canonical_size,
        poll_hz=config.obs.poll_hz,
    )
    server = EventServer(
        confirmer=confirmer,
        host=config.server.host,
        port=config.server.port,
        obs_connected_getter=lambda: source.connected,
    )
    recorder = FireRecorder(Path("evidence"))

    async def pump() -> None:
        loop = asyncio.get_running_loop()
        frames = source.frames()
        last_signature = None
        while True:
            frame = await loop.run_in_executor(None, next, frames)
            active = get_detector(confirmer.game)
            observation = active.observe(frame)
            event = confirmer.observe(observation, frame.captured_at)
            if event is not None:
                recorder.record(event, frame, observation)
                await server.broadcast(event)

            # Status on every state change, so the dashboard can distinguish
            # "idle" from "holding in cooldown until character select".
            signature = (confirmer.state, confirmer.armed, source.connected)
            if signature != last_signature:
                last_signature = signature
                await server.broadcast(server.status_event(frame.captured_at))

    async def main_async() -> None:
        await asyncio.gather(server.serve(), pump())

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        return 0
    finally:
        source.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "run": _cmd_run,
        "capture": _cmd_capture,
        "roi": _cmd_roi,
        "replay": _cmd_replay,
    }
    try:
        return handlers[args.command](args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

Note the `pump()` loop above broadcasts a `status` event whenever the Confirmer's state or OBS connectivity changes. The spec requires status on *every* state change, not only on connect and after commands — without this, a dashboard cannot tell IDLE from COOLDOWN, and COOLDOWN is precisely when the detector is deliberately refusing to fire. An operator seeing nothing happen deserves to know the difference between "not detecting" and "holding until character select".

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py tests/test_cli.py -v`
Expected: PASS, 8 passed

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -v`
Expected: PASS, all tests green

- [ ] **Step 7: Commit**

```bash
git add src/fgc_detector/pipeline.py src/fgc_detector/cli.py src/fgc_detector/detectors/registry.py tests/test_pipeline.py tests/test_cli.py
git commit -m "feat: pipeline and CLI with run/capture/roi/replay"
```

---

### Task 13: Street Fighter 6 detector

**Files:**
- Create: `src/fgc_detector/detectors/sf6.py`, `tests/detectors/test_sf6.py`
- Create: `samples/sf6/` (committed corpus)

**Interfaces:**
- Consumes: `Roi`, `fill_ratio`, `match_template`; `Detector` protocol; `register`.
- Produces: `Sf6Detector` registered for `Game.SF6`, with `rois() -> dict[str, Roi]`.

- [ ] **Step 1: Ask the user for sample media — do not skip or fake this**

**STOP and request samples before writing any code in this task.** ROI coordinates cannot be derived from documentation; they depend on the actual capture resolution and HUD scale. Ask for:

1. A VOD (or a few minutes of one) of real SF6 matches from the actual stream setup, ideally including a full set with the post-match replay.
2. If a VOD is awkward, stills instead: 5+ frames mid-match, 5+ at match end for **each** side winning, 5+ on character select, and 5+ of the post-match replay.

Then label them into `samples/sf6/` using this naming convention, which the test reads directly:

```
samples/sf6/in_match_0001.png
samples/sf6/match_end_p1_0001.png
samples/sf6/match_end_p2_0001.png
samples/sf6/char_select_0001.png
samples/sf6/replay_0001.png        # labelled in_match; must NOT read as match_end
```

- [ ] **Step 2: Locate the ROIs**

Open a `match_end_p1` sample and find the round-win markers beside each health bar. Record pixel coordinates at 1920×1080. Verify with:

```bash
uv run fgc-detect roi --game sf6 --sample samples/sf6/match_end_p1_0001.png --out /tmp/sf6_roi.png
```

(Requires a first-pass `Sf6Detector` with guessed coordinates; iterate until the boxes sit on the markers.)

- [ ] **Step 3: Write the corpus test**

Create `tests/detectors/test_sf6.py`:

```python
from pathlib import Path

import cv2
import pytest

from fgc_detector.detectors.sf6 import Sf6Detector
from fgc_detector.frames.normalize import normalize
from fgc_detector.types import Frame, Screen, Side

SAMPLES = Path(__file__).parent.parent.parent / "samples" / "sf6"


def _load(path: Path) -> Frame:
    from datetime import datetime, timezone

    image = cv2.imread(str(path))
    assert image is not None, f"could not read {path}"
    normalized = normalize(image, Sf6Detector().canonical_size)
    assert normalized is not None, f"{path} has an unexpected aspect ratio"
    return Frame(image=normalized, captured_at=datetime.now(timezone.utc))


def _samples(prefix: str) -> list[Path]:
    paths = sorted(SAMPLES.glob(f"{prefix}_*.png"))
    assert paths, f"no {prefix} samples found in {SAMPLES}"
    return paths


@pytest.mark.parametrize("path", _samples("in_match"), ids=lambda p: p.name)
def test_in_match_samples_classify_as_in_match(path):
    assert Sf6Detector().observe(_load(path)).screen is Screen.IN_MATCH


@pytest.mark.parametrize("path", _samples("match_end_p1"), ids=lambda p: p.name)
def test_p1_wins_samples_report_p1(path):
    observation = Sf6Detector().observe(_load(path))
    assert observation.screen is Screen.MATCH_END
    assert observation.winner is Side.P1


@pytest.mark.parametrize("path", _samples("match_end_p2"), ids=lambda p: p.name)
def test_p2_wins_samples_report_p2(path):
    observation = Sf6Detector().observe(_load(path))
    assert observation.screen is Screen.MATCH_END
    assert observation.winner is Side.P2


@pytest.mark.parametrize("path", _samples("char_select"), ids=lambda p: p.name)
def test_char_select_samples_classify_as_char_select(path):
    # The Confirmer's cooldown exits only on CHAR_SELECT, so a miss here wedges
    # the detector until the safety valve fires.
    assert Sf6Detector().observe(_load(path)).screen is Screen.CHAR_SELECT


@pytest.mark.parametrize("path", _samples("replay"), ids=lambda p: p.name)
def test_replay_samples_never_report_match_end(path):
    assert Sf6Detector().observe(_load(path)).screen is not Screen.MATCH_END


def test_detector_is_pure():
    detector = Sf6Detector()
    frame = _load(_samples("match_end_p1")[0])
    assert detector.observe(frame) == detector.observe(frame)


def test_rois_are_named_and_within_canonical_bounds():
    detector = Sf6Detector()
    width, height = detector.canonical_size
    assert detector.rois()
    for name, roi in detector.rois().items():
        assert roi.x + roi.w <= width, name
        assert roi.y + roi.h <= height, name
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/detectors/test_sf6.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fgc_detector.detectors.sf6'`

- [ ] **Step 5: Write the detector**

Create `src/fgc_detector/detectors/sf6.py`. Fill the ROI coordinates from Step 2 — the values below are **placeholders that will not work until measured**:

```python
"""Street Fighter 6 detector.

Reads the round-win markers beside each health bar. Markers are position-fixed
and language-independent, so unlike an OCR approach this imposes no requirement
on the game's display language.
"""

from __future__ import annotations

from ..types import Frame, Game, Observation, Screen, Side
from .registry import register
from .roi import Roi, fill_ratio

# MEASURE THESE from samples/sf6/ before trusting anything. See Task 13 Step 2.
_ROIS: dict[str, Roi] = {
    "p1_round_1": Roi(x=0, y=0, w=1, h=1),
    "p1_round_2": Roi(x=0, y=0, w=1, h=1),
    "p2_round_1": Roi(x=0, y=0, w=1, h=1),
    "p2_round_2": Roi(x=0, y=0, w=1, h=1),
    "health_bar_p1": Roi(x=0, y=0, w=1, h=1),
    "char_select_marker": Roi(x=0, y=0, w=1, h=1),
}

_MARKER_FILLED = 0.60
_HEALTH_BAR_PRESENT = 0.30
_CHAR_SELECT_PRESENT = 0.50
_ROUNDS_TO_WIN = 2


class Sf6Detector:
    game = Game.SF6
    canonical_size = (1920, 1080)

    def rois(self) -> dict[str, Roi]:
        return dict(_ROIS)

    def observe(self, frame: Frame) -> Observation:
        image = frame.image
        scores = {name: fill_ratio(image, roi) for name, roi in _ROIS.items()}

        if scores["char_select_marker"] >= _CHAR_SELECT_PRESENT:
            return Observation(
                screen=Screen.CHAR_SELECT,
                confidence=scores["char_select_marker"],
                debug=scores,
            )

        if scores["health_bar_p1"] < _HEALTH_BAR_PRESENT:
            return Observation(screen=Screen.UNKNOWN, debug=scores)

        p1_rounds = sum(
            1
            for index in range(1, _ROUNDS_TO_WIN + 1)
            if scores[f"p1_round_{index}"] >= _MARKER_FILLED
        )
        p2_rounds = sum(
            1
            for index in range(1, _ROUNDS_TO_WIN + 1)
            if scores[f"p2_round_{index}"] >= _MARKER_FILLED
        )

        if p1_rounds >= _ROUNDS_TO_WIN and p2_rounds < _ROUNDS_TO_WIN:
            winner = Side.P1
        elif p2_rounds >= _ROUNDS_TO_WIN and p1_rounds < _ROUNDS_TO_WIN:
            winner = Side.P2
        else:
            return Observation(screen=Screen.IN_MATCH, debug=scores)

        marker_scores = [
            scores[f"{winner.value}_round_{index}"]
            for index in range(1, _ROUNDS_TO_WIN + 1)
        ]
        return Observation(
            screen=Screen.MATCH_END,
            winner=winner,
            confidence=min(marker_scores),
            debug=scores,
        )


register(Sf6Detector())
```

Import the module in `src/fgc_detector/detectors/__init__.py` so registration happens:

```python
from . import sf6  # noqa: F401  (registers the SF6 detector)
```

- [ ] **Step 6: Tune until the corpus passes**

Run: `uv run pytest tests/detectors/test_sf6.py -v`

Iterate on coordinates and the three thresholds until every test passes. If a threshold has to sit inside a narrow band to work, the ROI is probably in the wrong place — widen the gap by moving the box rather than tightening the number.

- [ ] **Step 7: Verify against a full VOD end to end**

```bash
uv run fgc-detect replay --game sf6 --video <the-vod>.mp4 --evidence-dir /tmp/sf6_evidence
```

Expected: exactly one `match_end` per game actually played, and none during post-match replays. Compare the printed timeline against the VOD by hand.

- [ ] **Step 8: Commit**

```bash
git add src/fgc_detector/detectors/sf6.py src/fgc_detector/detectors/__init__.py samples/sf6/ tests/detectors/test_sf6.py
git commit -m "feat: SF6 detector with sample corpus"
```

---

### Task 14: Tekken 8 detector

**Files:**
- Create: `src/fgc_detector/detectors/tekken8.py`, `tests/detectors/test_tekken8.py`
- Create: `samples/tekken8/` (committed corpus)

**Interfaces:**
- Consumes: `Roi`, `fill_ratio`; `register`.
- Produces: `Tekken8Detector` registered for `Game.TEKKEN8`, with `rois() -> dict[str, Roi]`.

- [ ] **Step 1: Ask the user for Tekken 8 sample media — do not skip or fake this**

**STOP and request samples.** Same requirements and naming convention as Task 13 Step 1, into `samples/tekken8/`:

```
samples/tekken8/in_match_0001.png
samples/tekken8/match_end_p1_0001.png
samples/tekken8/match_end_p2_0001.png
samples/tekken8/char_select_0001.png
samples/tekken8/replay_0001.png
```

Tekken 8 note: sets are commonly first-to-3 rather than first-to-2, so there are more round markers under each health bar. Confirm the round count in the samples rather than assuming — `_ROUNDS_TO_WIN` differs from SF6.

- [ ] **Step 2: Locate the ROIs**

Same procedure as Task 13 Step 2, using:

```bash
uv run fgc-detect roi --game tekken8 --sample samples/tekken8/match_end_p1_0001.png --out /tmp/t8_roi.png
```

- [ ] **Step 3: Write the corpus test**

Create `tests/detectors/test_tekken8.py` — the same suite as `test_sf6.py`, with `Sf6Detector` replaced by `Tekken8Detector` and `SAMPLES` pointing at `samples/tekken8`. Repeated in full rather than shared, because the two detectors will diverge as their thresholds are tuned independently:

```python
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pytest

from fgc_detector.detectors.tekken8 import Tekken8Detector
from fgc_detector.frames.normalize import normalize
from fgc_detector.types import Frame, Screen, Side

SAMPLES = Path(__file__).parent.parent.parent / "samples" / "tekken8"


def _load(path: Path) -> Frame:
    image = cv2.imread(str(path))
    assert image is not None, f"could not read {path}"
    normalized = normalize(image, Tekken8Detector().canonical_size)
    assert normalized is not None, f"{path} has an unexpected aspect ratio"
    return Frame(image=normalized, captured_at=datetime.now(timezone.utc))


def _samples(prefix: str) -> list[Path]:
    paths = sorted(SAMPLES.glob(f"{prefix}_*.png"))
    assert paths, f"no {prefix} samples found in {SAMPLES}"
    return paths


@pytest.mark.parametrize("path", _samples("in_match"), ids=lambda p: p.name)
def test_in_match_samples_classify_as_in_match(path):
    assert Tekken8Detector().observe(_load(path)).screen is Screen.IN_MATCH


@pytest.mark.parametrize("path", _samples("match_end_p1"), ids=lambda p: p.name)
def test_p1_wins_samples_report_p1(path):
    observation = Tekken8Detector().observe(_load(path))
    assert observation.screen is Screen.MATCH_END
    assert observation.winner is Side.P1


@pytest.mark.parametrize("path", _samples("match_end_p2"), ids=lambda p: p.name)
def test_p2_wins_samples_report_p2(path):
    observation = Tekken8Detector().observe(_load(path))
    assert observation.screen is Screen.MATCH_END
    assert observation.winner is Side.P2


@pytest.mark.parametrize("path", _samples("char_select"), ids=lambda p: p.name)
def test_char_select_samples_classify_as_char_select(path):
    assert Tekken8Detector().observe(_load(path)).screen is Screen.CHAR_SELECT


@pytest.mark.parametrize("path", _samples("replay"), ids=lambda p: p.name)
def test_replay_samples_never_report_match_end(path):
    assert Tekken8Detector().observe(_load(path)).screen is not Screen.MATCH_END


def test_detector_is_pure():
    detector = Tekken8Detector()
    frame = _load(_samples("match_end_p1")[0])
    assert detector.observe(frame) == detector.observe(frame)


def test_rois_are_named_and_within_canonical_bounds():
    detector = Tekken8Detector()
    width, height = detector.canonical_size
    assert detector.rois()
    for name, roi in detector.rois().items():
        assert roi.x + roi.w <= width, name
        assert roi.y + roi.h <= height, name
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/detectors/test_tekken8.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fgc_detector.detectors.tekken8'`

- [ ] **Step 5: Write the detector**

Create `src/fgc_detector/detectors/tekken8.py`, structured exactly as `sf6.py` but with Tekken's own ROIs, thresholds, and `_ROUNDS_TO_WIN` as confirmed in Step 1:

```python
"""Tekken 8 detector.

Reads the round-win markers under each health bar. Same strategy as SF6, with
its own coordinates and round count — sets are typically first-to-3.
"""

from __future__ import annotations

from ..types import Frame, Game, Observation, Screen, Side
from .registry import register
from .roi import Roi, fill_ratio

# MEASURE THESE from samples/tekken8/ before trusting anything.
_ROIS: dict[str, Roi] = {
    "p1_round_1": Roi(x=0, y=0, w=1, h=1),
    "p1_round_2": Roi(x=0, y=0, w=1, h=1),
    "p1_round_3": Roi(x=0, y=0, w=1, h=1),
    "p2_round_1": Roi(x=0, y=0, w=1, h=1),
    "p2_round_2": Roi(x=0, y=0, w=1, h=1),
    "p2_round_3": Roi(x=0, y=0, w=1, h=1),
    "health_bar_p1": Roi(x=0, y=0, w=1, h=1),
    "char_select_marker": Roi(x=0, y=0, w=1, h=1),
}

_MARKER_FILLED = 0.60
_HEALTH_BAR_PRESENT = 0.30
_CHAR_SELECT_PRESENT = 0.50
_ROUNDS_TO_WIN = 3


class Tekken8Detector:
    game = Game.TEKKEN8
    canonical_size = (1920, 1080)

    def rois(self) -> dict[str, Roi]:
        return dict(_ROIS)

    def observe(self, frame: Frame) -> Observation:
        image = frame.image
        scores = {name: fill_ratio(image, roi) for name, roi in _ROIS.items()}

        if scores["char_select_marker"] >= _CHAR_SELECT_PRESENT:
            return Observation(
                screen=Screen.CHAR_SELECT,
                confidence=scores["char_select_marker"],
                debug=scores,
            )

        if scores["health_bar_p1"] < _HEALTH_BAR_PRESENT:
            return Observation(screen=Screen.UNKNOWN, debug=scores)

        p1_rounds = sum(
            1
            for index in range(1, _ROUNDS_TO_WIN + 1)
            if scores[f"p1_round_{index}"] >= _MARKER_FILLED
        )
        p2_rounds = sum(
            1
            for index in range(1, _ROUNDS_TO_WIN + 1)
            if scores[f"p2_round_{index}"] >= _MARKER_FILLED
        )

        if p1_rounds >= _ROUNDS_TO_WIN and p2_rounds < _ROUNDS_TO_WIN:
            winner = Side.P1
        elif p2_rounds >= _ROUNDS_TO_WIN and p1_rounds < _ROUNDS_TO_WIN:
            winner = Side.P2
        else:
            return Observation(screen=Screen.IN_MATCH, debug=scores)

        marker_scores = [
            scores[f"{winner.value}_round_{index}"]
            for index in range(1, _ROUNDS_TO_WIN + 1)
        ]
        return Observation(
            screen=Screen.MATCH_END,
            winner=winner,
            confidence=min(marker_scores),
            debug=scores,
        )


register(Tekken8Detector())
```

Add to `src/fgc_detector/detectors/__init__.py`:

```python
from . import sf6  # noqa: F401  (registers the SF6 detector)
from . import tekken8  # noqa: F401  (registers the Tekken 8 detector)
```

- [ ] **Step 6: Tune until the corpus passes**

Run: `uv run pytest tests/detectors/test_tekken8.py -v`

- [ ] **Step 7: Verify against a full VOD end to end**

```bash
uv run fgc-detect replay --game tekken8 --video <the-vod>.mp4 --evidence-dir /tmp/t8_evidence
```

- [ ] **Step 8: Run the full suite and commit**

```bash
uv run pytest -v
git add src/fgc_detector/detectors/tekken8.py src/fgc_detector/detectors/__init__.py samples/tekken8/ tests/detectors/test_tekken8.py
git commit -m "feat: Tekken 8 detector with sample corpus"
```

---

## Definition of done for v1

- [ ] `uv run pytest` passes with no OBS, GPU, network, or real clock involved.
- [ ] `fgc-detect run --config config.toml` connects to OBS, serves on `ws://127.0.0.1:6600`, and pushes a `status` event to a client on connect.
- [ ] Sending `{"cmd":"arm"}` flips `armed` to true in the next `status` event.
- [ ] After a fire, a `status` event reports `state: "cooldown"`, and a later one reports `state: "idle"` once character select is seen.
- [ ] A real SF6 game played on the stream setup produces exactly one `match_end` with the correct winner, and the post-match replay produces none.
- [ ] The same holds for Tekken 8.
- [ ] Every fire leaves a PNG and a JSON sidecar in `evidence/`.
