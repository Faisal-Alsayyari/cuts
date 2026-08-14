# cuts

Efficient semantic indexing and temporal navigation of long-form video.

Point it at a multi-hour screen recording and get back a chapter list:

```
0:00 writing parser
7:32 compiling
9:11 debugging segfault
21:54 reading docs
29:13 implementing fix
38:51 tests finally pass
```

The primary target is screen captures of coding sessions, where the hard part
is not finding visual cuts — there usually aren't any — but finding the points
where *what the person is doing* changes.

## Install

Requires Python 3.12 (3.13 lacks wheels for `ctranslate2` / `rapidocr`).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Model weights (DINOv2, the OCR ONNX models, Whisper) download on first use.
Chapter titles need Claude API credentials — set `ANTHROPIC_API_KEY`, or run
`ant auth login`. Without credentials the pipeline still runs and falls back to
crude keyword-derived titles.

## Run

```bash
python -m cuts chapters session.mp4
python -m cuts chapters session.mp4 --chapters 12 --ocr-stride 3 -v
python -m cuts label session.events.json          # re-title without re-analyzing
```

Useful flags: `--interval` (sampling rate), `--chapters` (target count),
`--min-chapter`, `--ocr-stride` (main speed lever), `--no-ocr` / `--no-dino` /
`--no-asr` (isolate a channel), `--backend`, `--no-labels`.

Each run writes `<video>.events.json` — the full index, including per-event
OCR text, transcript, sample indices, and boundary scores.

## How it works

```
sample 1 fps  ->  per-frame signals  ->  fused similarity curve
              ->  local minima  ->  merge to M events  ->  label
```

**Stage 1 (segmentation)** implements
[arXiv 2603.00983](https://arxiv.org/abs/2603.00983), *"Event-Anchored Frame
Selection for Effective Long-Video Understanding"* (Chen, Luo, Zeng, Lin, Xie,
Chao, Ji, Zheng). Frames are sampled uniformly, each is scored by its weighted
similarity to its temporal neighbours (paper eq. 1, window `l=3`, linearly
decaying weights), boundaries are cut at local minima of that curve, and
adjacent partitions are merged by mean-feature similarity until the target
event count `M` is reached.

**Our extension: the similarity curve fuses multiple channels.** The paper uses
DINOv2 alone. DINOv2 is self-supervised on natural images, and screen
recordings are a hard distribution shift — two completely different code
screens are both "monospace text on a dark background". Measured on a synthetic
coding-session fixture with six known activity changes:

| Channel | Boundary recall | False positives |
|---------|----------------:|----------------:|
| DINOv2 only (paper-faithful) | 3/5 | 1 |
| OCR text only | 5/5 | 0 |
| Fused 50/50 (default) | 5/5 | 0 |

Text carries this domain. Fusion costs nothing and keeps the visual channel for
footage where it does help. Set `weight_dino=1.0, weight_ocr=0.0` to recover
the paper's exact behaviour. Channels are z-scored before fusion because their
native scales are incomparable (DINOv2 adjacent cosine sits around 0.99 on
screen capture; TF-IDF cosine is far lower and much more spread out).

**Labeling** sends every event's evidence to Claude in a single call, so titles
stay distinct and consistently phrased across the video. Labeling each event
separately reliably produces near-duplicate titles for adjacent events.

**Stages 2-3 of the paper** (query-driven anchor selection + adaptive MMR) are
implemented in [cuts/select.py](cuts/select.py) but are *not* used by chapter
generation, which is query-free. They are the engine for the future "find the
moment matching this description" workflow.

## Layout

| Module | Role |
|--------|------|
| [cuts/media.py](cuts/media.py) | VFR-safe decoding, uniform sampling, frame lookup |
| [cuts/signals/dino.py](cuts/signals/dino.py) | DINOv2 embeddings (visual channel) |
| [cuts/signals/ocr.py](cuts/signals/ocr.py) | On-screen text (text channel) |
| [cuts/signals/asr.py](cuts/signals/asr.py) | Whisper transcript (speech channel) |
| [cuts/signals/text_features.py](cuts/signals/text_features.py) | TF-IDF vectors for text channels |
| [cuts/segmentation.py](cuts/segmentation.py) | EFS Stage 1 |
| [cuts/labeling.py](cuts/labeling.py) | Chapter titles (Claude, heuristic fallback) |
| [cuts/select.py](cuts/select.py) | EFS Stages 2-3, query-driven (unused today) |
| [cuts/pipeline.py](cuts/pipeline.py) | Orchestrator |
| [cuts/cli.py](cuts/cli.py) | CLI |

Every module has a `python -m cuts.<module> <video>` debug entry point that
runs and reports on that stage alone.

All tuning constants live in [cuts/config.py](cuts/config.py) as one
`CutsConfig` dataclass. No module body hardcodes a threshold.

## Performance and limits

Measured on an M3 / 8 GB, 143s fixture at 1 fps: **27s per video-minute** with
OCR on every sample, **10.6s per video-minute** at `--ocr-stride 3`, with
identical boundaries in both cases.

OCR dominates; DINOv2 on MPS is a small fraction. Two known scaling limits:

- **Decode is linear in total video length**, not in sample count, because PyAV
  can only reliably seek to keyframes. A multi-hour input is decode-bound.
  Keyframe-seek sampling is the fix if that becomes the bottleneck.
- **Extrapolating to 30 hours** gives roughly 5 hours at `--ocr-stride 3`. That
  is untested at that scale; raise `--interval` and `--ocr-stride` together for
  very long inputs.

Memory is bounded regardless of length — frames are reduced to features and
discarded inside a single streaming pass, never accumulated.

## What this is not

- No training. Inference only.
- No shot-boundary detection — it was removed; it does not describe what
  changes in a screen recording.
- No web UI yet.
- No agent/skill layer yet.

## Status

Segmentation is validated against a synthetic fixture with known ground truth
(6/6 boundaries exact). It has **not** been validated on real devlog footage,
and the Claude labeling path has not been run end-to-end (no credentials were
available on the dev machine). Both are the next things to verify.
