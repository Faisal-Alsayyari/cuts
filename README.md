# cuts

Frame-accurate shot boundary and UI state change event detection for raw
devlog screen recordings. Inference-only; no training, no UI, no audio.

## Architecture

Two-stage coarse-to-fine, then a third sub-shot pass:

1. **Stage 1 — Coarse candidates** (`cuts/detectors/`)
   - Arm A: PySceneDetect (`AdaptiveDetector` + `ContentDetector`).
   - Arm B: TransNetV2 (PyTorch port).
   - Union, merged within ±4 frames by `cuts/detectors/ensemble.py`.
2. **Stage 2 — Local refinement** (`cuts/refinement.py`)
   - Decode ±W frames around each candidate with PyAV (every frame, no NONKEY).
   - Per-frame discontinuity = weighted (HSV histogram Bhattacharyya) + (1 − SSIM).
   - argmax → hard cut frame; elevated span ≥ N frames → gradual transition.
   - Motion-vs-state-change post-filter drops cursor/animation false positives.
3. **Stage 3 — UI state change events** (`cuts/event_detector.py`)
   - Inside each shot, sample every N frames; SSIM-delta scan.
   - Each elevated window is re-refined with Stage 2 to recover frame-accurate
     `(start_frame, end_frame)`.
   - Optional CLIP labeling (off by default).

Frame indexing is ordinal (`enumerate(container.decode(stream))`), matching
TransNetV2's internal frame ordering. Time conversions use the per-frame PTS
table built by [cuts/frame_extractor.py](cuts/frame_extractor.py) — never
`frame_idx / fps`, which is wrong for VFR.

## Install

```powershell
pip install -r requirements.txt
```

## Run

```powershell
# Full pipeline (System D) on a video
python -m cuts.pipeline path\to\video.mp4 D

# Other benchmark systems
python -m cuts.pipeline path\to\video.mp4 A   # ContentDetector only
python -m cuts.pipeline path\to\video.mp4 B   # AdaptiveDetector only
python -m cuts.pipeline path\to\video.mp4 C   # TransNetV2 only
python -m cuts.pipeline path\to\video.mp4 E   # AutoShot only (requires checkpoint, see below)
python -m cuts.pipeline path\to\video.mp4 F   # OmniShotCut only (requires checkpoint, see below)
```

### Systems E/F checkpoints

Systems E (AutoShot) and F (OmniShotCut) load third-party repos and
pre-trained weights that are **not** committed to this repository (see
[.gitignore](.gitignore)) — download them once per machine:

```powershell
# AutoShot
git clone https://github.com/wentaozhu/AutoShot
# then grab the .pth checkpoint linked from that repo's README

# OmniShotCut
git clone https://github.com/UVA-Computer-Vision-Lab/OmniShotCut
mkdir OmniShotCut\checkpoints
# download OmniShotCut_ckpt.pth from https://huggingface.co/uva-cv-lab/OmniShotCut into checkpoints/
```

Then point `AutoShotConfig`/`OmniShotCutConfig` (`repo_path`, `checkpoint_path`)
at the cloned directory and `.pth` file — see the docstrings in
[cuts/detectors/autoshot_detector.py](cuts/detectors/autoshot_detector.py) and
[cuts/detectors/omnishotcut_detector.py](cuts/detectors/omnishotcut_detector.py).

## Module debug entry points

Every module is independently runnable:

| Module | Command |
|--------|---------|
| `cuts.config` | `python -m cuts.config` |
| `cuts.frame_extractor` | `python -m cuts.frame_extractor <video>` |
| `cuts.detectors.pyscenedetect_detector` | `python -m cuts.detectors.pyscenedetect_detector <video>` |
| `cuts.detectors.transnetv2_detector` | `python -m cuts.detectors.transnetv2_detector <video>` |
| `cuts.detectors.ensemble` | `python -m cuts.detectors.ensemble` |
| `cuts.refinement` | `python -m cuts.refinement <video> <frame_idx> ...` |
| `cuts.event_detector` | `python -m cuts.event_detector <video>` |
| `cuts.benchmark.evaluator` | `python -m cuts.benchmark.evaluator` |
| `cuts.benchmark.visualizer` | `python -m cuts.benchmark.visualizer <video> <out.png>` |
| `cuts.pipeline` | `python -m cuts.pipeline <video> [A\|B\|C\|D\|E\|F]` |

## Configuration

All thresholds, window sizes, and the device selection live in
[cuts/config.py](cuts/config.py) as a single `CutsConfig` dataclass with
sub-dataclasses per stage. No module body hardcodes a tuning constant —
modify or replace `CutsConfig` to sweep.

## Benchmark

Annotation CSV columns: `video_id, clip_path, type, label, start_frame, end_frame`.

- `type` ∈ `hard_cut`, `gradual_transition`, `ui_event`
- Hard cuts: matched within ±2 frames.
- Gradual / UI: matched at interval IoU ≥ 0.5.

Run an example evaluation:

```python
from cuts.benchmark.evaluator import evaluate, to_markdown_table
from cuts.benchmark.schema import load_annotations
from cuts.config import CutsConfig
from cuts.pipeline import run_system

cfg = CutsConfig()
gts = load_annotations("benchmark/annotations.csv")
results = []
for system in ("A", "B", "C", "D"):
    res = run_system(system, "video.mp4", cfg)
    results.append(evaluate(
        system=system,
        predictions=list(res.boundaries) + list(res.events),
        annotations=[a for a in gts if a.video_id == "video"],
        config=cfg.benchmark,
        video_duration_sec=res.duration_sec,
        runtime_sec=res.runtime_sec,
    ))
print(to_markdown_table(results, "hard_cut"))
print(to_markdown_table(results, "overall"))
```

## What this is not

- No training. Inference only.
- No web UI; only matplotlib timeline plots in [cuts/benchmark/visualizer.py](cuts/benchmark/visualizer.py).
- No audio.
- No LLM/VLM beyond optional CLIP for event labeling.
- No streaming / distributed infrastructure.

## Generated artifacts (not committed)

These directories/files are regenerated by running the tools and are excluded
via [.gitignore](.gitignore) rather than tracked in git:

- `checkpoints/`, `OmniShotCut/checkpoints/`, any `*.pth` \u2014 downloaded model weights.
- `retrieval_index/` \u2014 output of `python -m cuts.retrieval.cli index ...`.
- `refine_experiment/` \u2014 debug thumbnails from `cuts.benchmark.refine_experiment`.
- `.venv/` \u2014 local virtual environment.
