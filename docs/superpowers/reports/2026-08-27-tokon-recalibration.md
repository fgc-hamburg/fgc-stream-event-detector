# TOKON recalibration — the lit-pip test, 2026-08-27

Supersedes the lit-pip section of the [2026-08-26 calibration
report](2026-08-26-tokon-calibration.md). Pip geometry, the empty-pip test and
the confirmer wiring are unchanged and still measured as described there.

## 1. Symptom

Running the TOKON detector live against OBS produced **no `match_end` events at
all**. Pausing the video at a match end sometimes did fire one. A second,
natively-recorded clip fired nothing at all, paused or playing.

## 2. Root cause

`LIT_ICON_DIFF_MIN = 0.128` — a threshold on `region_difference(icon,
background)`, the mean colour of the icon band versus the mean of the stage
directly above it.

On native 1920×1080 footage **every genuinely lit pip falls below it**:

| win window (tokon-3.mp4) | lit-slot mean difference |
|---|---|
| 145.4–146.5 s (0-3) | 0.091 – 0.133 |
| 674.9–675.4 s (2-3) | 0.080 – 0.133 |
| 913.7–914.4 s (1-3) | 0.075 – 0.103 |

A slot that is neither confidently empty nor confidently lit is `_AMBIGUOUS`,
and one ambiguous slot makes the whole frame `UNKNOWN`. So the detector
returned `UNKNOWN` on **54 %** of frames sampled through OBS and rejected
**39 %** of genuine lit pips. Nothing could ever fire.

This also explains the paused-vs-playing difference on the older clip, where
readings straddled the threshold: a paused frame that happens to clear it
repeats identically and accumulates the required agreeing frames, while during
playback the readings alternate and never produce a consecutive run.

### Why it was mis-set

The 2026-08-26 calibration measured lit slots against **empty** slots and found
a gap of +0.026 (lit ≥ 0.134, empty ≤ 0.109). But empty slots never reach this
test — the pale-core branch takes them first. The population that actually
reaches it is slots whose white circle is *gone*: lit pips, sprites, flashes,
absent HUD. Measured against that population on the original corpus, with
per-slot ground truth:

| feature | genuine lit min | spurious max | gap |
|---|---|---|---|
| `region_difference` (shipped) | 0.133 | 0.224 | **−0.091** |

The shipped threshold never separated cleanly on its own corpus either. The
symptom was masked because the `sprite_*` frames were parked as `xfail`, and
because the worst passing replay confidence, 0.1365, cleared 0.128 by 6.6 % —
a margin that read as tight but survivable, and did not survive a second
recording.

## 3. Ruled out, with measurements

| Suspect | Measurement | Verdict |
|---|---|---|
| Sample rate too low | 6.16 Hz achieved end-to-end; simulating the real `Confirmer` over the VOD at that cadence with measured jitter fires 5/5 in 200/200 randomized runs | not the cause |
| OBS capture latency | 50 ms median at 1280×720 | not the cause |
| ROI misalignment | empty pale-disc centre measures (747.5, 48.5) vs assumed (747, 48); pitch 38 px exact | not the cause |
| `request_size=(1280,720)` blurring the icon | 0.084 vs 0.085 on identical pixels — region means survive resampling | not the cause |

## 4. Bake-off

No single scalar separates a lit disc from a character sprite covering an empty
slot. A sprite is opaque and strongly coloured, so it satisfies any
*background-relative* test on its own. Measured per-slot on the original
corpus (80 genuine lit, 35 spurious):

| candidate | lit min | spurious max | gap |
|---|---|---|---|
| `region_difference` (mean vs mean) | 0.133 | 0.224 | −0.091 |
| `region_deviation_fraction` (covered pixels) | 0.719 | 0.981 | −0.261 |
| badge pale fraction | 0.347 | 0.661 | −0.314 |
| annulus uniformity | 0.066 | 0.230 | −0.164 |
| annulus vs core | 0.099 | 0.195 | −0.096 |

The discriminator that works is the **white P/V badge**: it is HUD chrome, not
character art, so it is identical whichever character won, and a sprite has no
reason to reproduce it. It carries most of the power; the other two reject the
residue. As a conjunction:

    badge >= 0.30  AND  covered_fraction >= 0.35  AND  mean_difference >= 0.05

- **0 of 80** genuine lit slots rejected (original corpus)
- **0 of 11** genuine lit slots rejected (native capture)
- **2 of 35** spurious slots accepted — both P1's outermost pip in
  `sprite_*_0002`, already documented `xfail` frames

Measured minima over genuinely lit slots, original corpus / native capture:
badge 0.347 / 0.570, covered fraction 0.719 / 0.41, mean difference 0.133 /
0.075. Each threshold sits below the lower of the two.

## 5. Constants (copy-paste)

```python
#: The white P/V badge on the lower-right of a lit disc.
_BADGE_DX, _BADGE_Y, _BADGE_SIZE = 5, 52, 11

LIT_BADGE_PALE_MIN = 0.30
LIT_ICON_DEV_FRAC_MIN = 0.35
LIT_ICON_DIFF_MIN = 0.05     # was 0.128
```

New general primitive in `roi.py`: `region_deviation_fraction(image, a, b,
threshold=30)` — the fraction of `a`'s pixels differing from `b`'s mean by more
than `threshold`. Where `region_difference` compares region means (diluting a
small icon against a large background), this counts the pixels the icon covers.

## 6. Ground truth — `~/repos/tokon/tokon-3.mp4`

1920×1080, H.264, bt709 SDR, 60 fps, 917.7 s. Recorded through OBS from an HDR
webm source, so it is the production colour pipeline; OpenCV reads it directly.
Operator-supplied match ends: **2:25, 5:51, 11:15, 15:14**.

| operator time | detected | winner | note |
|---|---|---|---|
| 2:25 | 00:02:25.6 | p2 | |
| 5:51 | — | — | **not observable**, see below |
| 11:15 | 00:11:15.1 | p2 | |
| 15:14 | 00:15:13.9 | p2 | |

The 5:51 match end cannot be detected from pips. A 60 fps scan shows the HUD
reading 2-1 with strong evidence until 348.72 s, then cutting straight to the
results screen. The winning third pip is **never drawn**. This is a property of
the footage, not a detector defect.

Regression on the original VOD (`TOKON.mp4`): the same five events with
identical confidences — 0.2078 p1, 0.1365 p2, 0.1947 p1, 0.1478 p2, 0.1718 p1.

## 7. Live-rate requirement, and the capture cost that broke it

With the detector fixed, live detection still fired nothing unless the video
was paused. That was a second, independent defect.

**Measure the capture rate against a source that is actually decoding.** An
earlier measurement of 6.16 Hz was taken while the OBS source showed a static
black frame and was worthless. Against a playing 1080p60 source the real
figure was **0.95 Hz** — three agreeing frames need 2.10 s of continuous
window, and the longest window is 1.20 s, so every match end was missed.
Pausing "worked" only because a frozen frame accumulates agreement for free.

Two compounding causes, both now fixed:

1. **PNG screenshots.** Lossless compression of a noisy game frame is
   expensive *inside OBS* and dominates while OBS is also decoding. Measured
   on a live source at 1280×720: **PNG 1128 ms vs JPEG q=80 104 ms**, an 11×
   difference. `ObsFrameSource` now requests JPEG q=80. `fgc-detect capture`
   still requests PNG, because ROIs are *measured* from those frames.
2. **Pacing.** `frames()` slept a full `1/poll_hz` *after* each capture, so the
   achieved period was `capture_latency + 1/poll_hz` and `poll_hz` was a rate
   the source could never reach. It now sleeps only the unused remainder.

Together: **0.95 Hz → 8.64 Hz** at `poll_hz = 10.0`, measured end-to-end
through `ObsFrameSource` against the playing source.

JPEG safety was checked hermetically by re-encoding every committed corpus
frame at q=90/80/70 and re-running its detector. At q=80, **no frame** in the
TOKON, Avatar or SF6 corpora changed its reported screen, winner or counts.
The only movement at other qualities was `UNKNOWN → IN_MATCH 0-0` on
HUD-absent TOKON frames — the benign direction the corpus test already allows,
and never a spurious `MATCH_END`.

Contiguous `MATCH_END` duration per window under the new lit test:

| window | duration | samples @ 0.95 Hz (before) | samples @ 8.64 Hz (after) |
|---|---|---|---|
| 2:25 | 1.20 s | 2.1 ✗ | 11.4 |
| 11:15 | **0.58 s** | 1.6 ✗ | 6.0 |
| 15:14 | 0.78 s | 1.7 ✗ | 7.7 |

`confirmer.agreement_frames = 3` now has 2–3.8× margin on every window. The
source logs the rate it actually achieves after 20 captures, and warns when
that falls below half the configured `poll_hz`, so this cannot go unnoticed
again.

## 8. Corpus

`scripts/build_tokon_native_corpus.py` adds 27 labelled native frames to
`samples/tokon/` (indices 100+), including the three observable wins, the
2-1 states either side of the sprite frame, one `sprite_p1-2_p2-1` adversarial
frame, and ten `between` frames covering super flashes, KO cinematics and
results screens. The corpus now pins the detector against two unrelated
captures, which is what would have caught this defect at calibration time.

## 9. Known limitations

- A sprite covering P1's outermost pip in `sprite_*_0002` still over-reads.
  Unchanged from the previous report; each occurrence is a single sample
  against the confirmer's 3-frame agreement.
- A pale results screen reads all six slots as empty, i.e. `IN_MATCH 0-0`.
  Harmless for firing, but it releases the confirmer's cooldown early.
- A match end whose winning pip is never drawn (5:51 above) cannot be detected.
