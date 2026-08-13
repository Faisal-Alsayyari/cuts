"""Stage 3 — Sub-shot UI state change event detection.

Coarse-to-fine event pipeline (independent of cut detection):

  E1 — cheap full-video diff signal
       mean-abs pixel diff on thumbnail (O(N), ~ms total)
  E2 — candidate selection
       keep top-K pairs by diff magnitude (global cap)
  E3 — local SSIM refinement
       run existing _refine_candidate on ±W window (full-res)

Event detection is intentionally decoupled from the cut pipeline.  If Stage 1
finds zero cuts, Stage 3 still runs its own coarse scan over every shot.  SSIM
is used ONLY in E3 (local windows), never as a full-video scan.  Using SSIM as
a coarse signal is O(H×W) per pair and takes 1–3 s/pair at 1080p — that is the
bug this architecture is designed to prevent.

CLIP labeling is OPTIONAL and OFF by default.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from cuts.config import CutsConfig, EventDetectorConfig
from cuts.detectors.ensemble import BoundaryCandidate
from cuts.frame_extractor import DecodedFrame, collect_windows_single_pass, collect_strided_shots
from cuts.refinement import RefinedBoundary, _refine_candidate


@dataclass
class UIEvent:
    """One sub-shot UI state change event."""

    frame_idx: int           # start frame
    end_frame_idx: int       # end frame (== frame_idx for instantaneous)
    pts: int
    time_sec: float
    shot_index: int          # which shot this event lives in
    signal_peak: float       # max SSIM-delta observed
    label: Optional[str] = None  # only set when CLIP labeling is enabled

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "end_frame_idx": self.end_frame_idx,
            "pts": self.pts,
            "time_sec": self.time_sec,
            "shot_index": self.shot_index,
            "signal_peak": self.signal_peak,
            "label": self.label,
        }


def detect_events_in_shots(
    video_path: str,
    shots: List[Tuple[int, int]],
    config: CutsConfig,
) -> List[UIEvent]:
    """Find UI events in every shot in TWO sequential decode passes (not one per shot).

    Pass 1: `collect_strided_shots` — one decode pass collecting every stride-th
            frame from all valid shots simultaneously.
    Pass 2: `collect_windows_single_pass` — one decode pass collecting the ±W
            refinement window for every SSIM-delta candidate across all shots.

    This replaces the old per-shot decode pattern which was O(S × video_length)
    for the scan and O(S × C × video_length) for per-shot refine_candidates.
    """
    ev_cfg: EventDetectorConfig = config.events
    v = config.verbose
    t0 = time.perf_counter()

    def _t(since: float) -> str:
        return f"{time.perf_counter() - since:.2f}s"

    # Filter shots long enough to host a meaningful sub-shot event.
    valid_shots: List[Tuple[int, int, int]] = [  # (shot_idx, start, end)
        (shot_idx, s, e)
        for shot_idx, (s, e) in enumerate(shots)
        if e - s + 1 >= ev_cfg.min_shot_length_frames
    ]
    if v:
        total_valid_frames = sum(e - s + 1 for _, s, e in valid_shots)
        print(f"[S3] {len(valid_shots)}/{len(shots)} shots qualify "
              f"(≥{ev_cfg.min_shot_length_frames} frames), "
              f"{total_valid_frames} frames to scan")
    if not valid_shots:
        return []

    # ----- E1: cheap coarse diff scan (one decode pass, all shots) ----------
    # Uses mean-abs pixel diff on thumbnails — O(N) and ~milliseconds total.
    # SSIM is NOT used here; it belongs only in E3 local refinement.
    t_e1 = time.perf_counter()
    if v: print(f"[S3/E1] collecting strided frames (stride={ev_cfg.sample_stride}, thumb={ev_cfg.scan_thumb_long_edge}px)...")
    shot_intervals = [(s, e) for _, s, e in valid_shots]
    all_shot_frames = collect_strided_shots(video_path, shot_intervals, ev_cfg.sample_stride)
    n_strided = sum(len(f) for f in all_shot_frames)
    if v: print(f"[S3/E1] decode done: {n_strided} strided frames in {_t(t_e1)}")

    all_candidates: List[BoundaryCandidate] = []
    cand_shot_ids: List[int] = []   # parallel to all_candidates
    cand_deltas:   List[float] = [] # parallel to all_candidates; used for top-K sort

    thumb = ev_cfg.scan_thumb_long_edge  # 0 = no resize

    for (shot_idx, _s, _e), frames in zip(valid_shots, all_shot_frames):
        if len(frames) < 2:
            continue

        # Convert to grayscale thumbnails for the coarse diff.
        gray_samples: List[Tuple[int, np.ndarray]] = []
        for f in frames:
            gray = cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)
            if thumb > 0:
                h, w = gray.shape
                long_edge = max(h, w)
                if long_edge > thumb:
                    scale = thumb / long_edge
                    gray = cv2.resize(
                        gray,
                        (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
            gray_samples.append((f.frame_idx, gray))

        for i in range(len(gray_samples) - 1):
            a = gray_samples[i][1].astype(np.float32)
            b = gray_samples[i + 1][1].astype(np.float32)
            # Normalised mean absolute difference: 0 = identical, 1 = max change.
            delta = float(np.mean(np.abs(a - b)) / 255.0)
            if delta < ev_cfg.coarse_diff_threshold:
                continue
            # The change sits between these two sample frames; use the midpoint
            # as the E3 refinement centre.
            mid = (gray_samples[i][0] + gray_samples[i + 1][0]) // 2
            all_candidates.append(BoundaryCandidate(frame_idx=mid, sources=["event_scan"]))
            cand_shot_ids.append(shot_idx)
            cand_deltas.append(delta)

    if not all_candidates:
        if v: print(f"[S3/E1] diff scan done in {_t(t_e1)}: 0 candidates above threshold={ev_cfg.coarse_diff_threshold}")
        return []
    if v: print(f"[S3/E1] diff scan done in {_t(t_e1)}: {len(all_candidates)} candidates above threshold={ev_cfg.coarse_diff_threshold}")

    # ----- E2: top-K candidate pruning (global, by diff magnitude) -----------
    # Caps the number of expensive E3 refinements regardless of video length.
    k = ev_cfg.top_k_candidates
    if len(all_candidates) > k:
        # Sort descending by delta; keep the k strongest candidates.
        ranked = sorted(
            range(len(all_candidates)), key=lambda i: cand_deltas[i], reverse=True
        )[:k]
        # Restore frame order (ascending frame_idx) so Pass 2 windows are sorted.
        ranked.sort(key=lambda i: all_candidates[i].frame_idx)
        n_before = len(all_candidates)
        all_candidates = [all_candidates[i] for i in ranked]
        cand_shot_ids  = [cand_shot_ids[i]  for i in ranked]
        cand_deltas    = [cand_deltas[i]    for i in ranked]
        if v: print(f"[S3/E2] pruned to top-{k} candidates (was {n_before})")
    else:
        if v: print(f"[S3/E2] {len(all_candidates)} candidates (≤ top_k={k}, no pruning needed)")

    # ----- E3: local SSIM refinement on top-K candidates (one decode pass) -----
    t_e3 = time.perf_counter()
    if v: print(f"[S3/E3] collecting ±{config.refinement.half_window} windows for {len(all_candidates)} candidates...")
    W = config.refinement.half_window
    windows = [(max(0, c.frame_idx - W), c.frame_idx + W) for c in all_candidates]
    all_window_frames = collect_windows_single_pass(video_path, windows)
    if v: print(f"[S3/E3] window decode done in {_t(t_e3)}; running SSIM refinement...")

    events: List[UIEvent] = []
    for cand, frames, shot_idx in zip(all_candidates, all_window_frames, cand_shot_ids):
        r = _refine_candidate(cand, config.refinement, frames)
        if r is None:
            continue
        events.append(UIEvent(
            frame_idx=r.frame_idx,
            end_frame_idx=r.end_frame_idx,
            pts=r.pts,
            time_sec=r.time_sec,
            shot_index=shot_idx,
            signal_peak=r.signal_peak,
        ))
    if v: print(f"[S3/E3] refinement done in {_t(t_e3)}: {len(events)} events from {len(all_candidates)} candidates")
    if v: print(f"[S3] total Stage 3 time: {_t(t0)}")

    # Optional CLIP labeling (off by default).
    if ev_cfg.use_clip_labeling and events:
        labels = _label_events_with_clip(video_path, events, config.device)
        for e, label in zip(events, labels):
            e.label = label

    return events


# ---------------------------------------------------------------------------
# Optional CLIP labeling
# ---------------------------------------------------------------------------

# Lazy module-level CLIP cache (loading is heavy).
_CLIP = None


def _label_events_with_clip(
    video_path: str, events: List, device: str
) -> List[Optional[str]]:
    """Return a list of free-text labels (one per event), or None on failure.

    This is a very thin wrapper: it picks the start frame of each event,
    encodes them in one batch, and currently returns the frame index as a
    placeholder string. A real labeling implementation would compare image
    features against a fixed text-prompt vocabulary using cosine similarity;
    we leave that hook open and unfilled because it is explicitly low priority.
    """
    try:
        global _CLIP
        if _CLIP is None:
            import open_clip  # lazy
            import torch  # lazy

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            model = model.to(device).eval()
            _CLIP = (model, preprocess, torch)
        model, preprocess, torch = _CLIP

        # Fetch all event start frames in one decode pass (avoids O(E × length) scans).
        from PIL import Image  # lazy

        clip_windows = [(ev.frame_idx, ev.frame_idx) for ev in events]
        all_clip_frames = collect_windows_single_pass(video_path, clip_windows)
        images = []
        for clip_frames in all_clip_frames:
            if not clip_frames:
                continue
            img_bgr = clip_frames[0].image
            images.append(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
        if not images:
            return [None] * len(events)

        batch = torch.stack([preprocess(img) for img in images]).to(device)
        with torch.no_grad():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        # Placeholder: report feature norm as a stand-in label so downstream
        # consumers see *something* useful when this path is enabled.
        return [f"clip_feat_norm={float(features[i].norm()):.3f}" for i in range(len(images))]
    except Exception as exc:  # pragma: no cover - optional path
        # Never let labeling failures break localization.
        return [f"clip_error: {exc!r}"] * len(events)


if __name__ == "__main__":
    # Standalone debug: assume a single-shot video covering [0, N-1] and scan it.
    import sys

    from cuts.frame_extractor import build_frame_index

    if len(sys.argv) < 2:
        print("usage: python -m cuts.event_detector <video_path>")
        sys.exit(1)
    cfg = CutsConfig()
    path = sys.argv[1]
    n_frames = len(build_frame_index(path))
    shots = [(0, n_frames - 1)]
    events = detect_events_in_shots(path, shots, cfg)
    print(f"{len(events)} UI events detected (treating whole video as one shot)")
    for e in events[:20]:
        print(e.to_dict())
