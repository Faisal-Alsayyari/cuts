"""Refinement experiment: confidence gating + asymmetric window comparison.

Tests three window shapes × two gating modes against manually-labeled GT cuts.
TransNetV2 runs once; all refinement passes reuse the same candidates + probs.

Usage:
    python -m cuts.benchmark.refine_experiment <video_path> <gt_json>
        [--output-dir <dir>]
        [--conf-threshold τ]    # gate: skip if confidence >= τ
        [--conf-pct X]          # gate: refine only bottom X% by confidence

The script always runs 6 variants:
    window A [-6,+6]  × {no gate, gated}
    window B [-6,+2]  × {no gate, gated}
    window C [-8, 0]  × {no gate, gated}

Per-boundary stats are printed as an aligned table.
Composite thumbnails are saved to <output_dir>/thumbnails/.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from cuts.config import CutsConfig, RefinementConfig
from cuts.detectors.transnetv2_detector import detect
from cuts.detectors.ensemble import BoundaryCandidate
from cuts.frame_extractor import build_frame_index, collect_windows_single_pass
from cuts.refinement import RefinedBoundary, refine_candidates


# ---------------------------------------------------------------------------
# Window configs under test
# ---------------------------------------------------------------------------

WINDOW_CONFIGS: List[Tuple[str, int, int]] = [
    ("sym  [-6,+6]",  6, 6),
    ("left [-6,+2]",  6, 2),
    ("left [-8, 0]",  8, 0),
]

# ---------------------------------------------------------------------------
# Thumbnail helpers
# ---------------------------------------------------------------------------

_THUMB_W = 256
_THUMB_H = 144
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _thumb(image: np.ndarray, label: str, border_color: Optional[Tuple] = None) -> np.ndarray:
    t = cv2.resize(image, (_THUMB_W, _THUMB_H), interpolation=cv2.INTER_AREA)
    cv2.putText(t, label, (4, _THUMB_H - 6), _FONT, 0.45,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(t, label, (4, _THUMB_H - 6), _FONT, 0.45,
                (0, 0, 0), 1, cv2.LINE_AA)
    if border_color:
        cv2.rectangle(t, (0, 0), (_THUMB_W - 1, _THUMB_H - 1), border_color, 3)
    return t


def _save_composite(
    output_dir: Path,
    boundary_idx: int,
    coarse_frame: int,
    confidence: float,
    gt_frame: Optional[int],
    variant_results: List[Tuple[str, Optional[int], bool]],  # (label, refined_frame, gated)
    all_needed_frames: Dict[int, np.ndarray],                # frame_idx -> BGR image
) -> None:
    """Save a composite PNG: one row = coarse | GT | variant1 | variant2 | ..."""
    rows = []

    def _get(f: Optional[int], label: str, color: Optional[Tuple]) -> np.ndarray:
        if f is not None and f in all_needed_frames:
            return _thumb(all_needed_frames[f], label, color)
        placeholder = np.zeros((_THUMB_H, _THUMB_W, 3), dtype=np.uint8)
        cv2.putText(placeholder, label, (4, _THUMB_H // 2), _FONT, 0.4,
                    (120, 120, 120), 1, cv2.LINE_AA)
        return placeholder

    panels = [_get(coarse_frame, f"coarse={coarse_frame}", (0, 0, 200))]
    if gt_frame is not None:
        panels.append(_get(gt_frame, f"GT={gt_frame}", (0, 200, 0)))
    for v_label, v_frame, gated in variant_results:
        short = v_label.split()[0]
        gate_tag = " [gated]" if gated else ""
        panels.append(_get(v_frame, f"{short}={v_frame}{gate_tag}", (200, 150, 0) if gated else None))

    strip = np.hstack(panels)
    # Header
    hdr = np.zeros((26, strip.shape[1], 3), dtype=np.uint8)
    title = f"Boundary #{boundary_idx}  coarse={coarse_frame}  conf={confidence:.3f}"
    cv2.putText(hdr, title, (4, 18), _FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    img = np.vstack([hdr, strip])

    path = output_dir / f"boundary_{boundary_idx:03d}_coarse_{coarse_frame}.png"
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# Experiment core
# ---------------------------------------------------------------------------

def run_experiment(
    video_path: str,
    gt_path: str,
    output_dir: str,
    conf_threshold: Optional[float],
    conf_pct: Optional[float],
) -> None:
    out = Path(output_dir)
    thumb_dir = out / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth.
    gt_data = json.loads(Path(gt_path).read_text())
    gt_cuts: List[int] = gt_data.get("cuts", [])
    gt_set = set(gt_cuts)

    # Build frame index for PTS lookup.
    frame_index = build_frame_index(video_path)
    n_frames = len(frame_index)

    # Run TransNetV2 once.
    print(f"Running TransNetV2 on: {video_path}")
    t0 = time.perf_counter()
    cfg = CutsConfig()
    candidates: List[BoundaryCandidate] = detect(video_path, cfg.transnetv2)
    tn_sec = time.perf_counter() - t0
    print(f"TransNetV2: {len(candidates)} candidates in {tn_sec:.2f}s\n")

    if not candidates:
        print("No candidates. Exiting.")
        return

    # Base refinement config (motion filter off for hard-cut boundary detection).
    base_cfg = dataclasses.replace(cfg.refinement, motion_filter=False)

    # Build all 6 variants: 3 windows × 2 gating modes.
    Variant = Tuple[str, RefinementConfig]
    variants: List[Variant] = []
    for w_label, wl, wr in WINDOW_CONFIGS:
        for gated in (False, True):
            gate_label = "gated" if gated else "full"
            label = f"{w_label} {gate_label}"
            vcfg = dataclasses.replace(
                base_cfg,
                window_left=wl,
                window_right=wr,
                confidence_threshold=conf_threshold if gated else None,
                confidence_top_pct=conf_pct if gated else None,
            )
            variants.append((label, vcfg))

    # Run each variant and collect results.
    variant_boundaries: List[List[RefinedBoundary]] = []
    variant_times: List[float] = []
    for v_label, vcfg in variants:
        t0 = time.perf_counter()
        bounds = refine_candidates(video_path, candidates, vcfg)
        elapsed = time.perf_counter() - t0
        variant_boundaries.append(bounds)
        variant_times.append(elapsed)
        print(f"  {v_label:30s}  {len(bounds):2d} boundaries  {elapsed:.2f}s")

    print()

    # Collect all frame indices needed for thumbnails.
    needed: set = set()
    for b in candidates:
        needed.add(b.frame_idx)
    for gt_f in gt_cuts:
        needed.add(gt_f)
    for vbs in variant_boundaries:
        for b in vbs:
            needed.add(b.frame_idx)

    # Decode all needed frames in one pass (each as a single-frame window).
    needed_sorted = sorted(needed)
    needed_windows = [(f, f) for f in needed_sorted]
    raw_frame_lists = collect_windows_single_pass(video_path, needed_windows)
    frame_images: Dict[int, np.ndarray] = {}
    for f_idx, flist in zip(needed_sorted, raw_frame_lists):
        if flist:
            frame_images[f_idx] = flist[0].image

    # Per-candidate stats table.
    print(_table_header())
    rows = []
    for i, cand in enumerate(candidates):
        # Find closest GT cut.
        gt_nearest = min(gt_cuts, key=lambda g: abs(g - cand.frame_idx)) if gt_cuts else None
        delta_coarse = (cand.frame_idx - gt_nearest) if gt_nearest is not None else None

        # Collect refined frame per variant.
        variant_cells: List[Tuple[str, Optional[int], float, bool]] = []
        for v_label, vbs in zip([v[0] for v in variants], variant_boundaries):
            # Find the boundary in vbs that originated from cand.frame_idx.
            match = next((b for b in vbs if b.coarse_frame_idx == cand.frame_idx), None)
            if match is None:
                # Motion-filtered out.
                variant_cells.append((v_label, None, 0.0, False))
                continue
            gated = (match.signal_time_sec == 0.0 and match.coarse_frame_idx == match.frame_idx
                     and match.signal_peak == 0.0)
            variant_cells.append((v_label, match.frame_idx, match.signal_time_sec, gated))

        rows.append((i, cand, gt_nearest, delta_coarse, variant_cells))
        print(_table_row(i, cand, gt_nearest, delta_coarse, variant_cells))

        # Save composite thumbnail.
        v_for_thumb = [(v[0], v[1], v[3]) for v in variant_cells]
        _save_composite(thumb_dir, i, cand.frame_idx, cand.confidence,
                        gt_nearest, v_for_thumb, frame_images)

    # Summary.
    print()
    print("=== Variant summary ===")
    print(f"  {'Variant':<30}  {'refined':>7}  {'exact GT':>8}  {'±1 GT':>5}  {'time':>6}")
    for (v_label, _), vbs, elapsed in zip(variants, variant_boundaries, variant_times):
        n_exact = sum(1 for b in vbs if b.frame_idx in gt_set)
        n_near1 = sum(1 for b in vbs
                      if any(abs(b.frame_idx - g) <= 1 for g in gt_cuts))
        print(f"  {v_label:<30}  {len(vbs):>7}  {n_exact:>8}  {n_near1:>5}  {elapsed:>5.2f}s")

    print(f"\nThumbnails saved to: {thumb_dir}")


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _table_header() -> str:
    v_hdrs = "  ".join(f"{'A-full':>8} {'A-gate':>8} {'B-full':>8} {'B-gate':>8} "
                        f"{'C-full':>8} {'C-gate':>8}".split())
    return (f"{'#':>3}  {'coarse':>6}  {'conf':>5}  {'GT':>6}  {'Δcrs':>5}  "
            f"{'A-full':>8}  {'A-gate':>8}  {'B-full':>8}  {'B-gate':>8}  "
            f"{'C-full':>8}  {'C-gate':>8}")


def _table_row(
    idx: int,
    cand: BoundaryCandidate,
    gt: Optional[int],
    delta_coarse: Optional[int],
    variants: List[Tuple[str, Optional[int], float, bool]],
) -> str:
    def _fmt(frame: Optional[int], gated: bool) -> str:
        if frame is None:
            return "  [drop]"
        tag = "*" if gated else " "
        return f"{tag}{frame:>7}"

    v_cols = "  ".join(_fmt(v[1], v[3]) for v in variants)
    return (f"{idx:>3}  {cand.frame_idx:>6}  {cand.confidence:>5.3f}  "
            f"{str(gt) if gt is not None else 'n/a':>6}  "
            f"{('+' if delta_coarse and delta_coarse > 0 else '') + str(delta_coarse) if delta_coarse is not None else 'n/a':>5}  "
            f"{v_cols}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refinement experiment: confidence gating + asymmetric windows.")
    parser.add_argument("video_path", help="Path to the video file.")
    parser.add_argument("gt_json", help="Ground truth JSON from label_cuts.py.")
    parser.add_argument("--output-dir", default="refine_experiment",
                        help="Output directory for stats and thumbnails.")
    parser.add_argument("--conf-threshold", type=float, default=None,
                        help="Confidence threshold τ: skip refinement if conf >= τ.")
    parser.add_argument("--conf-pct", type=float, default=None,
                        help="Refine only bottom X%% of candidates by confidence.")
    args = parser.parse_args()

    run_experiment(
        video_path=args.video_path,
        gt_path=args.gt_json,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        conf_pct=args.conf_pct,
    )
