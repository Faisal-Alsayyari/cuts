"""Evaluator: precision/recall/F1, frame-error stats, and runtime per minute.

Matching rules:

* HARD CUT: a predicted boundary is a TP if a GT hard cut exists within
  `hard_cut_tolerance_frames` (default ±2). Each GT may match at most one
  prediction (greedy nearest-first).
* GRADUAL TRANSITION and UI EVENT: a predicted interval is a TP if there is a
  GT interval of the same type with IoU >= `interval_iou_threshold`
  (default 0.5). Greedy highest-IoU-first matching; each GT used at most once.

Reported metrics (per type and overall):
* precision, recall, F1
* median absolute frame error for true positives (using start frame)
* % of TPs within 1 frame of GT
* % of TPs within 3 frames of GT
* runtime in seconds per minute of video for the system that produced
  the predictions (passed in by the caller)

The output is a list of dicts that the caller can dump directly to CSV /
markdown via `to_markdown_table` / `to_csv`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from cuts.benchmark.schema import Annotation
from cuts.config import BenchmarkConfig


# A "prediction" is any object with `.frame_idx`, `.end_frame_idx`, `.type`.
# We accept duck-typed objects so RefinedBoundary and UIEvent both work.
@dataclass
class _PredView:
    frame_idx: int
    end_frame_idx: int
    type: str  # "hard_cut" | "gradual_transition" | "ui_event"


def _coerce_predictions(predictions: Iterable[object]) -> List[_PredView]:
    """Adapt heterogeneous prediction objects to a uniform view.

    UIEvent objects don't have `.type` set; we infer "ui_event" for those.
    RefinedBoundary objects have `.type` of "hard_cut"/"gradual_transition".
    """
    out: List[_PredView] = []
    for p in predictions:
        ptype = getattr(p, "type", None)
        if ptype is None:
            # Treat anything without an explicit type as a UI event (Stage 3).
            ptype = "ui_event"
        out.append(_PredView(
            frame_idx=int(getattr(p, "frame_idx")),
            end_frame_idx=int(getattr(p, "end_frame_idx", getattr(p, "frame_idx"))),
            type=ptype,
        ))
    return out


@dataclass
class EvalResult:
    """Per-type metrics + an aggregate overall row."""

    system: str
    runtime_sec_per_minute: float
    # Per-type and "overall" rows; each row has {precision, recall, f1, ...}.
    by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def evaluate(
    system: str,
    predictions: Iterable[object],
    annotations: List[Annotation],
    config: BenchmarkConfig,
    video_duration_sec: float,
    runtime_sec: float,
) -> EvalResult:
    """Evaluate one system's predictions against GT for a single video."""
    preds = _coerce_predictions(predictions)

    # Bucket by type so each type's matching rule is applied independently.
    # GT type buckets:
    gt_by_type: Dict[str, List[Annotation]] = {"hard_cut": [], "gradual_transition": [], "ui_event": []}
    for a in annotations:
        gt_by_type[a.type].append(a)

    pred_by_type: Dict[str, List[_PredView]] = {"hard_cut": [], "gradual_transition": [], "ui_event": []}
    for p in preds:
        if p.type in pred_by_type:
            pred_by_type[p.type].append(p)
        # Predictions with unexpected types are ignored (logged silently).

    by_type: Dict[str, Dict[str, float]] = {}
    # ---- HARD CUTS: tolerance-window matching --------------------------------
    by_type["hard_cut"] = _eval_hard_cuts(
        pred_by_type["hard_cut"], gt_by_type["hard_cut"], config.hard_cut_tolerance_frames
    )
    # ---- GRADUAL TRANSITIONS: interval IoU matching --------------------------
    by_type["gradual_transition"] = _eval_intervals(
        pred_by_type["gradual_transition"],
        gt_by_type["gradual_transition"],
        config.interval_iou_threshold,
    )
    # ---- UI EVENTS: interval IoU matching ------------------------------------
    by_type["ui_event"] = _eval_intervals(
        pred_by_type["ui_event"],
        gt_by_type["ui_event"],
        config.interval_iou_threshold,
    )
    # ---- OVERALL row: micro-averaged across all types -----------------------
    by_type["overall"] = _aggregate_overall(by_type)

    # Runtime per minute of video. Avoid divide-by-zero on tiny clips.
    minutes = max(video_duration_sec / 60.0, 1e-6)
    sec_per_minute = runtime_sec / minutes

    return EvalResult(
        system=system,
        runtime_sec_per_minute=sec_per_minute,
        by_type=by_type,
    )


# ---------------------------------------------------------------------------
# Type-specific matchers
# ---------------------------------------------------------------------------

def _eval_hard_cuts(
    preds: List[_PredView], gts: List[Annotation], tolerance: int
) -> Dict[str, float]:
    """Greedy nearest-frame matching within ±tolerance for hard cuts."""
    # Sort both by frame index for predictable greedy matching.
    preds_sorted = sorted(preds, key=lambda p: p.frame_idx)
    gts_sorted = sorted(gts, key=lambda g: g.start_frame)

    matched_gt_indices: set = set()
    tp_errors: List[int] = []  # absolute frame errors for matched pairs

    # For each prediction (in order) find the nearest unmatched GT within tol.
    for p in preds_sorted:
        best_gt_idx = -1
        best_err = tolerance + 1  # any match must beat this
        for gi, g in enumerate(gts_sorted):
            if gi in matched_gt_indices:
                continue
            err = abs(p.frame_idx - g.start_frame)
            if err <= tolerance and err < best_err:
                best_err = err
                best_gt_idx = gi
        if best_gt_idx >= 0:
            matched_gt_indices.add(best_gt_idx)
            tp_errors.append(best_err)

    return _metrics(
        tp=len(tp_errors),
        fp=len(preds_sorted) - len(tp_errors),
        fn=len(gts_sorted) - len(matched_gt_indices),
        tp_errors=tp_errors,
    )


def _eval_intervals(
    preds: List[_PredView], gts: List[Annotation], iou_threshold: float
) -> Dict[str, float]:
    """Greedy highest-IoU matching for interval predictions (gradual / UI)."""
    # Build a (pred_idx, gt_idx, iou, start_err) table for all eligible pairs,
    # then sort by IoU desc and assign greedily.
    pairs: List[Tuple[int, int, float, int]] = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            iou = _interval_iou((p.frame_idx, p.end_frame_idx), (g.start_frame, g.end_frame))
            if iou >= iou_threshold:
                start_err = abs(p.frame_idx - g.start_frame)
                pairs.append((pi, gi, iou, start_err))
    pairs.sort(key=lambda x: x[2], reverse=True)

    matched_pred: set = set()
    matched_gt: set = set()
    tp_errors: List[int] = []
    for pi, gi, _iou, err in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        tp_errors.append(err)

    return _metrics(
        tp=len(tp_errors),
        fp=len(preds) - len(matched_pred),
        fn=len(gts) - len(matched_gt),
        tp_errors=tp_errors,
    )


def _interval_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """IoU on integer frame intervals, both inclusive."""
    a_start, a_end = a
    b_start, b_end = b
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    if overlap <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start) + 1
    return overlap / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _metrics(tp: int, fp: int, fn: int, tp_errors: List[int]) -> Dict[str, float]:
    """Standard P/R/F1 + frame-error breakdown for a single type bucket."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    if tp_errors:
        median_err = float(np.median(tp_errors))
        within_1 = float(np.mean([e <= 1 for e in tp_errors]))
        within_3 = float(np.mean([e <= 3 for e in tp_errors]))
    else:
        median_err = float("nan")
        within_1 = 0.0
        within_3 = 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_frame_error": median_err,
        "pct_within_1_frame": within_1,
        "pct_within_3_frames": within_3,
    }


def _aggregate_overall(by_type: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Micro-averaged overall metrics across all per-type buckets."""
    tp = sum(by_type[t]["tp"] for t in by_type)
    fp = sum(by_type[t]["fp"] for t in by_type)
    fn = sum(by_type[t]["fn"] for t in by_type)
    # Pool TP errors: weight each type's median by its TP count via concat.
    # We don't have raw errors here, so we approximate by averaging medians
    # weighted by TP count (good enough for the report card).
    medians = []
    for t, m in by_type.items():
        if not np.isnan(m["median_frame_error"]) and m["tp"] > 0:
            medians.extend([m["median_frame_error"]] * int(m["tp"]))
    median_err = float(np.median(medians)) if medians else float("nan")

    # Pool the within-K stats with the same TP weighting.
    def _weighted(key: str) -> float:
        total = sum(m["tp"] for m in by_type.values())
        if total <= 0:
            return 0.0
        return sum(m[key] * m["tp"] for m in by_type.values()) / total

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_frame_error": median_err,
        "pct_within_1_frame": _weighted("pct_within_1_frame"),
        "pct_within_3_frames": _weighted("pct_within_3_frames"),
    }


# ---------------------------------------------------------------------------
# Comparison-table formatters (markdown + CSV)
# ---------------------------------------------------------------------------

def to_markdown_table(results: List[EvalResult], type_filter: Optional[str] = None) -> str:
    """Render a comparison Markdown table across systems for one type (default: overall)."""
    t = type_filter or "overall"
    header = (
        f"| System | P | R | F1 | Median Err | % ≤1f | % ≤3f | sec/min |\n"
        f"|--------|---|---|----|-----------:|------:|------:|--------:|"
    )
    rows = []
    for r in results:
        m = r.by_type.get(t, {})
        rows.append(
            f"| {r.system} "
            f"| {m.get('precision', 0):.3f} "
            f"| {m.get('recall', 0):.3f} "
            f"| {m.get('f1', 0):.3f} "
            f"| {m.get('median_frame_error', float('nan')):.2f} "
            f"| {m.get('pct_within_1_frame', 0):.2%} "
            f"| {m.get('pct_within_3_frames', 0):.2%} "
            f"| {r.runtime_sec_per_minute:.2f} |"
        )
    return f"### Type: `{t}`\n\n" + header + "\n" + "\n".join(rows) + "\n"


def to_csv(results: List[EvalResult]) -> str:
    """Render every (system, type) row as CSV text."""
    cols = [
        "system", "type", "tp", "fp", "fn",
        "precision", "recall", "f1",
        "median_frame_error", "pct_within_1_frame", "pct_within_3_frames",
        "runtime_sec_per_minute",
    ]
    lines = [",".join(cols)]
    for r in results:
        for t, m in r.by_type.items():
            lines.append(",".join([
                r.system, t,
                f"{int(m['tp'])}", f"{int(m['fp'])}", f"{int(m['fn'])}",
                f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}",
                f"{m['median_frame_error']:.4f}",
                f"{m['pct_within_1_frame']:.4f}", f"{m['pct_within_3_frames']:.4f}",
                f"{r.runtime_sec_per_minute:.4f}",
            ]))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    # Smoke test on synthetic data: confirm metrics behave as expected.
    gts = [
        Annotation("v", "x", "hard_cut", "hard_cut", 100, 100),
        Annotation("v", "x", "hard_cut", "hard_cut", 200, 200),
        Annotation("v", "x", "gradual_transition", "dissolve", 300, 320),
    ]
    preds = [
        _PredView(101, 101, "hard_cut"),       # within tol -> TP
        _PredView(199, 199, "hard_cut"),       # within tol -> TP
        _PredView(305, 322, "gradual_transition"),  # IoU should pass -> TP
        _PredView(900, 900, "hard_cut"),       # FP
    ]
    res = evaluate("smoke", preds, gts, BenchmarkConfig(), 60.0, 12.0)
    print(to_markdown_table([res], "hard_cut"))
    print(to_markdown_table([res], "overall"))
