# Contributing

Thanks for looking at fgc-stream-event-detector. This is a small, focused project — please keep
contributions scoped and read this file before opening a PR.

## Dev setup

```bash
uv sync                  # installs deps + the fgc-detect entry point
uv run pytest            # full suite — no OBS, GPU, network, or real clock required
```

See the [README](README.md) for how to run the detector against OBS or a recorded video, and for
the architecture overview (detector vs. confirmer vs. server).

## Code style & architecture

- The **detector** is pure and per-game: one frame in, one `Observation` out, no memory. Don't
  sneak temporal/stateful logic in here.
- The **confirmer** is the only place that holds state across frames (arm/disarm, N-frame
  agreement, `set_game`).
- The **server** is the only thing that talks to the outside world (WebSocket, HTTP UI).
- New games implement the `Detector` protocol; they don't need to reuse SF6's digit-counting
  approach — match whatever the game's UI actually exposes.
- Keep pull requests focused on one change. Don't bundle unrelated refactors with a feature or
  fix — open a separate PR for those.

## Commits & PRs

- Write commit messages that explain *why*, not just *what* (the diff already shows what changed).
- Keep commits reasonably atomic; squash noisy WIP commits before opening a PR.
- In the PR description, state what changed and how you tested it (e.g. `uv run pytest`, a replay
  run against a sample video, a live OBS session).
- Make sure `uv run pytest` passes before requesting review.

## Reporting issues

When filing a bug, include:

- What you expected vs. what happened.
- The game/config involved (`config.toml`, which `Detector` — e.g. `sf6`).
- Logs or the relevant frame(s)/evidence dump if the detector misfired (see `evidence/`), with any
  identifying stream content redacted.
- Steps to reproduce, or the recorded video/replay command if you have one.

For feature requests, briefly describe the use case — this project deliberately keeps scope narrow
(see [`docs/TODO.md`](docs/TODO.md) for what's already planned/deferred).
