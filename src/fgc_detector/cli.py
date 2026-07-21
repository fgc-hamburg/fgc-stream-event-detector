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
    except UnknownGameError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
