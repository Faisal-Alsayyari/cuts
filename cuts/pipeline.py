"""Pipeline orchestrator.

Glue layer that runs:

    Stage 1 (detectors)  ->  ensemble merge
    Stage 2 (refinement) ->  frame-accurate hard / gradual boundaries
    Stage 3 (event detector) -> sub-shot UI events

`run_full_pipeline(video_path, config)` returns a `PipelineResult` containing
shots, refined boundaries, and UI events. `run_system(system_name, ...)` runs
one of the four benchmark configurations:

    A: PySceneDetect ContentDetector only
    B: PySceneDetect AdaptiveDetector only
    C: TransNetV2 only
    D: Hybrid ensemble + Stage 2 refinement (full pipeline)

Each system path returns predictions in the same shape so the evaluator can
compare them apples-to-apples.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Tuple

from cuts.config import CutsConfig
from cuts.detectors.ensemble import BoundaryCandidate, merge_candidates
from cuts.detectors.pyscenedetect_detector import (
    detect_with_adaptive,
    detect_with_content,
)
from cuts.event_detector import UIEvent, detect_events_in_shots
from cuts.frame_extractor import build_frame_index
from cuts.refinement import RefinedBoundary, refine_candidates


@dataclass
class PipelineResult:
    """Bundle of everything the pipeline produces for a single video."""

    video_path: str
    n_frames: int                                     # decoded frame count (ground truth)
    duration_sec: float                               # total wall-clock duration
    boundaries: List[RefinedBoundary] = field(default_factory=list)
    shots: List[Tuple[int, int]] = field(default_factory=list)  # (start, end) inclusive
    events: List[UIEvent] = field(default_factory=list)
    runtime_sec: float = 0.0
    _refine_sec: float = 0.0   # wall time for Stage 2 refinement (C and D)


# ---------------------------------------------------------------------------
# Full hybrid pipeline (System D)
# ---------------------------------------------------------------------------

def run_full_pipeline(video_path: str, config: CutsConfig) -> PipelineResult:
    """Run Stages 1 + 2 + 3 and return everything."""
    t_start = time.time()
    v = config.verbose

    def _t() -> str:
        return f"+{time.time() - t_start:.2f}s"

    # Build the frame index up-front: gives us ground-truth frame count and
    # the wall-clock duration (used for runtime-per-minute reporting).
    if v: print(f"[D] frame_index: building...")
    frame_index = build_frame_index(video_path, cache_dir=config.cache_dir)
    n_frames = len(frame_index)
    duration_sec = frame_index[-1][1] if frame_index else 0.0
    if v: print(f"[D] frame_index: {n_frames} frames ({duration_sec:.2f}s) {_t()}")

    # -------- Stage 1: candidates from both detector arms --------------------
    cands: List[BoundaryCandidate] = []
    if v: print(f"[D] Stage 1a: PySceneDetect AdaptiveDetector...")
    adaptive_cands = detect_with_adaptive(video_path, config.pyscenedetect)
    if v: print(f"[D] Stage 1a: {len(adaptive_cands)} candidates {_t()}")
    cands.extend(adaptive_cands)

    if v: print(f"[D] Stage 1b: PySceneDetect ContentDetector...")
    content_cands = detect_with_content(video_path, config.pyscenedetect)
    if v: print(f"[D] Stage 1b: {len(content_cands)} candidates {_t()}")
    cands.extend(content_cands)

    # TransNetV2 is optional at runtime: it has a heavy import cost and isn't
    # always installed. Failure here degrades to PySceneDetect-only candidates.
    if v: print(f"[D] Stage 1c: TransNetV2...")
    try:
        from cuts.detectors.transnetv2_detector import detect as detect_transnet
        tn_cands = detect_transnet(video_path, config.transnetv2)
        if v: print(f"[D] Stage 1c: {len(tn_cands)} candidates {_t()}")
        cands.extend(tn_cands)
    except Exception as exc:  # pragma: no cover - environment-dependent
        print(f"[pipeline] TransNetV2 unavailable, skipping ({exc!r})")

    merged = merge_candidates(cands, config.ensemble)
    if v: print(f"[D] Stage 1 ensemble merge: {len(cands)} raw -> {len(merged)} merged {_t()}")

    # -------- Stage 2: frame-accurate refinement -----------------------------
    if v: print(f"[D] Stage 2: refining {len(merged)} candidates...")
    t_refine = time.time()
    boundaries = refine_candidates(video_path, merged, config.refinement)
    refine_sec = time.time() - t_refine
    if v: print(f"[D] Stage 2: {len(boundaries)} boundaries {_t()}")

    # Derive shots from the refined boundary list. A shot is the interval from
    # one boundary to the next; the first shot starts at frame 0, the last
    # ends at the last frame of the video.
    shots = _boundaries_to_shots(boundaries, n_frames)
    if v: print(f"[D] shots derived: {len(shots)} shots {_t()}")

    # -------- Stage 3: sub-shot UI events ------------------------------------
    if v: print(f"[D] Stage 3: event detection on {len(shots)} shots...")
    events = detect_events_in_shots(video_path, shots, config)
    if v: print(f"[D] Stage 3: {len(events)} events {_t()}")

    runtime = time.time() - t_start
    return PipelineResult(
        video_path=video_path,
        n_frames=n_frames,
        duration_sec=duration_sec,
        boundaries=boundaries,
        shots=shots,
        events=events,
        runtime_sec=runtime,
        _refine_sec=refine_sec,
    )


# ---------------------------------------------------------------------------
# Single-system runners (for the benchmark comparison table)
# ---------------------------------------------------------------------------

def run_system(system: str, video_path: str, config: CutsConfig) -> PipelineResult:
    """Run one of A/B/C/D/E/F and return predictions in PipelineResult shape.

    A, B, C, E, F return UNREFINED detector boundaries — that is what the
    comparison table is meant to show.  D runs the full hybrid pipeline.

    Systems:
        A — PySceneDetect ContentDetector only
        B — PySceneDetect AdaptiveDetector only
        C — TransNetV2 only
        D — Hybrid ensemble + Stage 2 refinement (full production pipeline)
        E — AutoShot only (NAS-optimised 3-D ConvNet + Transformer, CVPR 2023)
        F — OmniShotCut only (Shot-Query Transformer, arXiv 2025)
    """
    t_start = time.time()
    frame_index = build_frame_index(video_path, cache_dir=config.cache_dir)
    n_frames = len(frame_index)
    duration_sec = frame_index[-1][1] if frame_index else 0.0

    if system == "A":
        cands = detect_with_content(video_path, config.pyscenedetect)
    elif system == "B":
        cands = detect_with_adaptive(video_path, config.pyscenedetect)
    elif system == "C":
        from cuts.detectors.transnetv2_detector import detect as detect_transnet
        cands = detect_transnet(video_path, config.transnetv2)
    elif system == "D":
        # Full pipeline: short-circuit to run_full_pipeline so we don't
        # duplicate the orchestration logic.
        result = run_full_pipeline(video_path, config)
        return result
    elif system == "E":
        from cuts.detectors.autoshot_detector import detect as detect_autoshot
        cands = detect_autoshot(video_path, config.autoshot)
    elif system == "F":
        from cuts.detectors.omnishotcut_detector import detect as detect_omnishotcut
        cands = detect_omnishotcut(video_path, config.omnishotcut)
    else:
        raise ValueError(f"Unknown system: {system!r}")

    # Wrap raw candidates as RefinedBoundary objects so downstream evaluator /
    # visualizer code is uniform across systems. We don't run Stage 2 here —
    # that's the point of A/B: show the raw detector output.
    boundaries: List[RefinedBoundary] = []
    for c in cands:
        boundaries.append(RefinedBoundary(
            frame_idx=c.frame_idx,
            end_frame_idx=c.end_frame_idx,
            pts=frame_index[c.frame_idx][0] if 0 <= c.frame_idx < n_frames else 0,
            time_sec=frame_index[c.frame_idx][1] if 0 <= c.frame_idx < n_frames else 0.0,
            type="gradual_transition" if c.is_gradual else "hard_cut",
            signal_peak=0.0,
            signal_width=0,
            sources=c.sources,
        ))

    runtime = time.time() - t_start
    return PipelineResult(
        video_path=video_path,
        n_frames=n_frames,
        duration_sec=duration_sec,
        boundaries=boundaries,
        shots=_boundaries_to_shots(boundaries, n_frames),
        events=[],  # A/B do not run event detection
        runtime_sec=runtime,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _boundaries_to_shots(
    boundaries: List[RefinedBoundary], n_frames: int
) -> List[Tuple[int, int]]:
    """Convert a list of boundaries into (shot_start, shot_end) intervals.

    A boundary's `frame_idx` is the FIRST frame of the new shot. So the
    previous shot runs [prev_start, boundary.frame_idx - 1] and the new shot
    starts at boundary.frame_idx. For gradual transitions, we assign the
    transition span to the END of the outgoing shot conservatively.
    """
    if n_frames <= 0:
        return []
    sorted_b = sorted(boundaries, key=lambda b: b.frame_idx)

    shots: List[Tuple[int, int]] = []
    cursor = 0  # start frame of the current shot
    for b in sorted_b:
        # End the current shot at the frame just before this boundary.
        end = max(cursor, b.frame_idx - 1)
        if end >= cursor:
            shots.append((cursor, end))
        # Next shot starts at the boundary's first frame.
        cursor = b.frame_idx
    # Tail: last shot to end-of-video.
    if cursor < n_frames:
        shots.append((cursor, n_frames - 1))
    return shots


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(result: PipelineResult) -> None:
    """Console summary of a PipelineResult."""
    print(f"video: {result.video_path}")
    print(f"  frames: {result.n_frames} ({result.duration_sec:.2f}s)")
    print(f"  runtime: {result.runtime_sec:.2f}s "
          f"({result.runtime_sec / max(result.duration_sec / 60.0, 1e-6):.2f}s/min)")

    n_hard = sum(1 for b in result.boundaries if b.type == "hard_cut")
    n_grad = sum(1 for b in result.boundaries if b.type == "gradual_transition")
    refine_note = f"  (refined in {result._refine_sec:.2f}s)" if result._refine_sec > 0 else ""
    print(f"  boundaries: {len(result.boundaries)}  ({n_hard} hard cuts, {n_grad} gradual){refine_note}")
    sorted_b = sorted(result.boundaries, key=lambda b: b.frame_idx)
    for b in sorted_b:
        if b.coarse_frame_idx is not None and b.coarse_frame_idx != b.frame_idx:
            delta = b.frame_idx - b.coarse_frame_idx
            coarse_str = f"  coarse={b.coarse_frame_idx} \u2192 refined={b.frame_idx} (\u0394{delta:+d})"
        elif b.coarse_frame_idx is not None:
            coarse_str = f"  coarse={b.coarse_frame_idx} \u2192 refined={b.frame_idx} (exact)"
        else:
            coarse_str = f"  frame {b.frame_idx}"
        if b.type == "hard_cut":
            print(f"    [{b.type}]{coarse_str}  t={b.time_sec:.3f}s  "
                  f"peak={b.signal_peak:.3f}  src={','.join(b.sources)}")
        else:
            print(f"    [{b.type}]  frames {b.frame_idx}–{b.end_frame_idx}  "
                  f"t={b.time_sec:.3f}s  peak={b.signal_peak:.3f}  src={','.join(b.sources)}")

    print(f"  shots: {len(result.shots)}")
    for i, (s, e) in enumerate(result.shots):
        dur = e - s + 1
        print(f"    shot {i}  frames {s}–{e}  ({dur} frames)")

    print(f"  ui events: {len(result.events)}")
    sorted_e = sorted(result.events, key=lambda ev: ev.frame_idx)
    for ev in sorted_e:
        if ev.frame_idx == ev.end_frame_idx:
            span = f"frame {ev.frame_idx}"
        else:
            span = f"frames {ev.frame_idx}–{ev.end_frame_idx}"
        label_str = f"  label={ev.label}" if ev.label else ""
        print(f"    [event]  {span}  t={ev.time_sec:.3f}s  "
              f"shot={ev.shot_index}  peak={ev.signal_peak:.3f}{label_str}")


if __name__ == "__main__":
    # CLI: `python -m cuts.pipeline <video_path> [system]`. system defaults to D.
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m cuts.pipeline <video_path> [A|B|C|D|E|F]")
        sys.exit(1)
    path = sys.argv[1]
    system = sys.argv[2] if len(sys.argv) > 2 else "D"
    cfg = CutsConfig()
    if "--debug" in sys.argv[3:] or "debug" in sys.argv[3:]:
        cfg.verbose = True
    res = run_system(system, path, cfg)
    print(f"=== System {system} ===")
    _print_summary(res)
