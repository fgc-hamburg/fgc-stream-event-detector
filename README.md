# FGC Stream Event Detector

Watches an OBS game-capture source and emits stream events over a WebSocket.

v1 emits one event — **match end, naming the winner** — for **Street Fighter 6** and **Tekken 8**.
It is consumed by the [FGC Scoreboard](https://github.com/renatomrcosta/fgc-scoreboard) control
dashboard, which decides what to do with it.

```
obs-websocket ──frames──▶ Detector.observe() ──Observations──▶ Confirmer ──Events──▶ WebSocket
 (game source)             (per-game, pure)      (shared, stateful)                  (dashboard)
```

The detector announces facts. It has no concept of a bracket, a set, or a score — policy lives in
the dashboard.

```json
{"type":"match_end","game":"sf6","winner":"p1","confidence":0.94,"ts":"2026-07-21T10:40:00Z"}
```

## Status

Design complete, implementation not started. See
[`docs/superpowers/specs/2026-07-21-stream-event-detector-design.md`](docs/superpowers/specs/2026-07-21-stream-event-detector-design.md)
for the full design, including why computer vision is the only viable approach for these titles
and which alternatives were rejected.
