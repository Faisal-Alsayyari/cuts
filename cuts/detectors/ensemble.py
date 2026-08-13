"""Ensemble fusion for Stage 1 candidate boundary generation.

Both detector arms (PySceneDetect, TransNetV2) emit lists of candidate frame
indices. They will *mostly* agree but disagree by 1–4 frames on the exact frame
of any given cut, and each will catch some boundaries the other misses. We:

1. Collect every candidate as a `BoundaryCandidate` tagged with its source.
2. Sort by frame index.
3. Merge candidates that fall within `merge_tolerance_frames` of each other,
   preserving every source label that contributed (so downstream code can
   reason about agreement).
4. Emit a sorted, deduplicated list.

The merged frame index is the *median* of the cluster — robust to a single
detector being off by a few frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

import numpy as np

from cuts.config import EnsembleConfig


@dataclass
class BoundaryCandidate:
    """One Stage-1 candidate boundary, before refinement."""

    frame_idx: int                     # ordinal index in decoding order
    sources: List[str] = field(default_factory=list)  # e.g. ["adaptive", "transnetv2"]
    # Optional: detector-reported boundary type (hard vs gradual). The ensemble
    # preserves this info if either detector flagged a gradual transition.
    is_gradual: bool = False
    # For gradual transitions, the detector-estimated end frame. For hard cuts
    # this equals frame_idx.
    end_frame_idx: int = -1
    # Detector confidence in [0, 1]. For TransNetV2 this is the peak per-frame
    # probability in a ±2 frame spike window around the boundary. 0.0 = unknown.
    confidence: float = 0.0
    # Detector confidence in [0, 1]. For TransNetV2 this is the peak per-frame
    # probability in a ±2 frame spike window around the boundary. 0.0 = unknown.
    confidence: float = 0.0

    def __post_init__(self) -> None:
        # Normalize: if end_frame_idx wasn't set, mirror frame_idx (hard-cut
        # convention). Downstream code treats start==end as a hard cut.
        if self.end_frame_idx < 0:
            self.end_frame_idx = self.frame_idx


def merge_candidates(
    candidates: Iterable[BoundaryCandidate],
    config: EnsembleConfig,
) -> List[BoundaryCandidate]:
    """Cluster nearby candidates into a single deduplicated, sorted list.

    Two candidates whose frame indices are within `merge_tolerance_frames` of
    each other are merged. Clusters can chain (A within tol of B, B within tol
    of C => all three merge), which is intentional — it prevents an arbitrary
    cut-off when three detectors place a boundary at frames 100/103/106 with
    tol=4.
    """
    sorted_cands = sorted(candidates, key=lambda c: c.frame_idx)
    if not sorted_cands:
        return []

    merged: List[BoundaryCandidate] = []
    # Start the first cluster with the first candidate.
    cluster: List[BoundaryCandidate] = [sorted_cands[0]]

    for cand in sorted_cands[1:]:
        # Chain merging: compare against the most recent member of the cluster.
        if cand.frame_idx - cluster[-1].frame_idx <= config.merge_tolerance_frames:
            cluster.append(cand)
        else:
            merged.append(_collapse_cluster(cluster))
            cluster = [cand]
    merged.append(_collapse_cluster(cluster))
    return merged


def _collapse_cluster(cluster: List[BoundaryCandidate]) -> BoundaryCandidate:
    """Reduce a list of nearby candidates to a single `BoundaryCandidate`.

    * frame_idx: median of the cluster (robust to outliers among 2–3 detectors).
    * end_frame_idx: max of cluster end_frame_idx values (covers the longest
      gradual span any detector reported).
    * is_gradual: True iff *any* member was gradual.
    * sources: union of all source labels, preserving order of first appearance.
    """
    frame_idx = int(np.median([c.frame_idx for c in cluster]))
    end_frame_idx = max(c.end_frame_idx for c in cluster)
    is_gradual = any(c.is_gradual for c in cluster)

    # Preserve order-of-appearance for sources, dedup on the way.
    seen: set = set()
    sources: List[str] = []
    for c in cluster:
        for s in c.sources:
            if s not in seen:
                seen.add(s)
                sources.append(s)

    return BoundaryCandidate(
        frame_idx=frame_idx,
        end_frame_idx=max(end_frame_idx, frame_idx),
        is_gradual=is_gradual,
        sources=sources,
    )


if __name__ == "__main__":
    # Smoke test: feed in synthetic candidates from two "detectors" and confirm
    # they merge correctly.
    cfg = EnsembleConfig(merge_tolerance_frames=4)
    fake = [
        BoundaryCandidate(frame_idx=100, sources=["adaptive"]),
        BoundaryCandidate(frame_idx=102, sources=["transnetv2"]),
        BoundaryCandidate(frame_idx=250, sources=["transnetv2"], is_gradual=True, end_frame_idx=260),
        BoundaryCandidate(frame_idx=251, sources=["adaptive"]),
        BoundaryCandidate(frame_idx=900, sources=["content"]),
    ]
    out = merge_candidates(fake, cfg)
    for c in out:
        print(c)
