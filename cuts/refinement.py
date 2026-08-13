"""Stage 2 — Local frame-level refinement.

For each Stage-1 candidate frame `f_c`, decode the window [f_c-W, f_c+W] using
PyAV (every frame, no NONKEY skipping) and compute a per-frame discontinuity
signal between consecutive frames. Then:

* If the signal has a clean spike at one frame, output a HARD CUT at the
  argmax frame inside the window.
* If the signal is elevated over more than `min_gradual_frames` consecutive
  frames, output a GRADUAL TRANSITION whose [start, end] spans the elevated
  region.

We deliberately do NOT use optical flow: UI state changes are structural
discontinuities (modal appears, view switches), not motion events. Optical
flow gives noisy, expensive, and misleading signal here.

The discontinuity signal is a weighted sum of:
  * HSV histogram Bhattacharyya distance (best for abrupt UI state changes)
  * Mean absolute pixel difference on grayscale thumbnail (fast; spikes cleanly
    at hard cuts and structural UI changes).

Post-filter (motion vs state-change disambiguation): a candidate whose
elevated-signal span is shorter than `motion_min_persistent_frames` AND whose
SSIM returns to near-baseline within `motion_recovery_frames` is dropped — it
is almost certainly a cursor flick or one-frame animation, not a state change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import cv2
import numpy as np

from cuts.config import RefinementConfig
from cuts.detectors.ensemble import BoundaryCandidate
from cuts.frame_extractor import DecodedFrame, collect_windows_single_pass


@dataclass
class RefinedBoundary:
    """Output of Stage 2: one frame-accurate boundary."""

    frame_idx: int            # for hard_cut: the cut frame; for gradual: span start
    end_frame_idx: int        # for hard_cut: == frame_idx; for gradual: span end
    pts: int                  # PTS of frame_idx, in stream time_base units
    time_sec: float           # wall-clock time of frame_idx
    type: str                 # "hard_cut" or "gradual_transition"
    signal_peak: float        # max combined signal value across the window
    signal_width: int         # number of frames where signal was above threshold
    sources: List[str]        # which detectors flagged this candidate
    # Frame index emitted by the coarse detector before refinement. Set by
    # refine_candidates(); None when boundaries were not refined (A/B/C raw).
    coarse_frame_idx: Optional[int] = None
    # Wall time (seconds) spent computing the signal for this candidate only
    # (excludes batch decode time). Useful for per-candidate profiling.
    signal_time_sec: float = 0.0
    # Wall time (seconds) spent computing the signal for this candidate only
    # (excludes batch decode time). Useful for per-candidate profiling.
    signal_time_sec: float = 0.0
    # Raw per-pair signal across the window — useful for visual debugging,
    # discarded by default to keep memory low. Set in `_refine_candidate`.
    signal_curve: Optional[List[float]] = None

    def to_dict(self) -> dict:
        """Serialize to the spec-mandated output format."""
        return {
            "frame_idx": self.frame_idx,
            "end_frame_idx": self.end_frame_idx,
            "pts": self.pts,
            "time_sec": self.time_sec,
            "type": self.type,
            "signal_peak": self.signal_peak,
        }


# ---------------------------------------------------------------------------
# Per-pair discontinuity primitives
# ---------------------------------------------------------------------------

def hist_delta(frame_a: np.ndarray, frame_b: np.ndarray, bins: int, thumb: int = 0) -> float:
    """HSV 3D-histogram Bhattacharyya distance between two BGR frames.

    Best signal for ABRUPT UI state changes (modal appears, view switches)
    where the global color distribution shifts. Output is in [0, 1].
    If `thumb > 0`, both frames are downscaled to that long-edge size first —
    the histogram captures global color distribution, so spatial resolution
    doesn't affect the value but dramatically reduces calcHist cost.
    """
    if thumb > 0:
        h, w = frame_a.shape[:2]
        long_edge = max(h, w)
        if long_edge > thumb:
            scale = thumb / long_edge
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            frame_a = cv2.resize(frame_a, new_size, interpolation=cv2.INTER_AREA)
            frame_b = cv2.resize(frame_b, new_size, interpolation=cv2.INTER_AREA)
    hsv_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2HSV)
    # 3D histogram over (H, S, V). Hue range is 0–180 in OpenCV's 8-bit HSV.
    h_a = cv2.calcHist([hsv_a], [0, 1, 2], None, [bins, bins, bins],
                       [0, 180, 0, 256, 0, 256])
    h_b = cv2.calcHist([hsv_b], [0, 1, 2], None, [bins, bins, bins],
                       [0, 180, 0, 256, 0, 256])
    # Normalize so Bhattacharyya is comparable across resolutions.
    cv2.normalize(h_a, h_a)
    cv2.normalize(h_b, h_b)
    return float(cv2.compareHist(h_a, h_b, cv2.HISTCMP_BHATTACHARYYA))


def mad_delta(frame_a: np.ndarray, frame_b: np.ndarray, thumb: int = 0) -> float:
    """Mean absolute pixel difference on grayscale, normalised to [0, 1].

    Spikes cleanly at hard cuts and structural UI state changes. Orders of
    magnitude faster than SSIM because it is O(pixels) with no convolutions.
    If `thumb > 0`, both frames are downscaled to that long-edge size before
    comparison — the spike location is unaffected by resolution.
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if thumb > 0:
        h, w = gray_a.shape
        long_edge = max(h, w)
        if long_edge > thumb:
            scale = thumb / long_edge
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            gray_a = cv2.resize(gray_a, new_size, interpolation=cv2.INTER_AREA)
            gray_b = cv2.resize(gray_b, new_size, interpolation=cv2.INTER_AREA)
    return float(np.mean(np.abs(gray_a - gray_b)) / 255.0)


def luminance_delta(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Mean-luminance absolute delta. Cheap; useful for fade/black-frame logs."""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    return float(abs(gray_a.mean() - gray_b.mean()) / 255.0)


# ---------------------------------------------------------------------------
# Combined signal across a list of frames
# ---------------------------------------------------------------------------

def compute_signal(
    frames: List[DecodedFrame], config: RefinementConfig
) -> np.ndarray:
    """Return per-pair combined discontinuity signal of length len(frames)-1.

    signal[i] is the discontinuity between `frames[i]` and `frames[i+1]`.
    Both components (histogram and MAD) are in [0, 1]; no normalisation needed.
    """
    n = len(frames)
    signal = np.zeros(max(0, n - 1), dtype=np.float32)
    thumb = config.refine_thumb_long_edge
    for i in range(n - 1):
        h = hist_delta(frames[i].image, frames[i + 1].image, config.hist_bins, thumb)
        m = mad_delta(frames[i].image, frames[i + 1].image, thumb)
        signal[i] = config.weight_hist * h + config.weight_mad * m
    return signal


# ---------------------------------------------------------------------------
# Refinement entry point
# ---------------------------------------------------------------------------

def refine_candidates(
    video_path: str,
    candidates: Iterable[BoundaryCandidate],
    config: RefinementConfig,
) -> List[RefinedBoundary]:
    """Refine every candidate; drop motion-only false positives.


    Confidence gating: if `config.confidence_threshold` or
    `config.confidence_top_pct` is set, candidates whose detector confidence
    exceeds the gate are passed through unchanged (coarse frame, no signal
    computation). Only the gated-in candidates are decoded and refined.

    Asymmetric windows: `config.window_left` / `config.window_right` override
    `config.half_window` for the before/after extent of each search window.
    """
    import time as _time
    cand_list = list(candidates)
    if not cand_list:
        return []

    # --- Resolve window extents -----------------------------------------
    wl = config.window_left if config.window_left is not None else config.half_window
    wr = config.window_right if config.window_right is not None else config.half_window

    # --- Confidence gating ----------------------------------------------
    # Determine which candidates to actually refine vs pass through.
    to_refine_idx: List[int] = []
    pass_through_idx: List[int] = []

    if config.confidence_top_pct is not None:
        # Bottom X% by confidence get refined; top (100-X)% pass through.
        pct = max(0.0, min(100.0, config.confidence_top_pct))
        confs = [c.confidence for c in cand_list]
        if confs:
            threshold = float(np.percentile(confs, pct))
            for i, c in enumerate(cand_list):
                if c.confidence <= threshold:
                    to_refine_idx.append(i)
                else:
                    pass_through_idx.append(i)
        else:
            to_refine_idx = list(range(len(cand_list)))
    elif config.confidence_threshold is not None:
        tau = config.confidence_threshold
        for i, c in enumerate(cand_list):
            if c.confidence < tau:
                to_refine_idx.append(i)
            else:
                pass_through_idx.append(i)
    else:
        to_refine_idx = list(range(len(cand_list)))

    # --- Decode windows for candidates that need refinement -------------
    to_refine_cands = [cand_list[i] for i in to_refine_idx]
    windows = [(max(0, c.frame_idx - wl), c.frame_idx + wr) for c in to_refine_cands]
    all_window_frames = collect_windows_single_pass(video_path, windows) if windows else []

    # Build result list preserving original candidate order.
    results: List[Optional[RefinedBoundary]] = [None] * len(cand_list)

    # Refine gated-in candidates.
    for i, (idx, cand, frames) in enumerate(zip(to_refine_idx, to_refine_cands, all_window_frames)):
        t0 = _time.perf_counter()
        result = _refine_candidate(cand, config, frames)
        t1 = _time.perf_counter()
        if result is None:
            continue
        result.coarse_frame_idx = cand.frame_idx
        result.signal_time_sec = t1 - t0
        results[idx] = result

    # Pass through high-confidence candidates unchanged.
    for idx in pass_through_idx:
        cand = cand_list[idx]
        # We don't have decoded frames, so construct a minimal RefinedBoundary
        # from what the coarse detector told us (no PTS/time_sec available here
        # without a frame index lookup — use 0.0 as sentinel; pipeline.py has
        # the frame_index to enrich if needed).
        results[idx] = RefinedBoundary(
            frame_idx=cand.frame_idx,
            end_frame_idx=cand.end_frame_idx,
            pts=0,
            time_sec=0.0,
            type="gradual_transition" if cand.is_gradual else "hard_cut",
            signal_peak=0.0,
            signal_width=0,
            sources=cand.sources,
            coarse_frame_idx=cand.frame_idx,
            signal_time_sec=0.0,
        )

    return [r for r in results if r is not None]
    # --- Confidence gating ----------------------------------------------
    # Determine which candidates to actually refine vs pass through.
    to_refine_idx: List[int] = []
    pass_through_idx: List[int] = []

    if config.confidence_top_pct is not None:
        # Bottom X% by confidence get refined; top (100-X)% pass through.
        pct = max(0.0, min(100.0, config.confidence_top_pct))
        confs = [c.confidence for c in cand_list]
        if confs:
            threshold = float(np.percentile(confs, pct))
            for i, c in enumerate(cand_list):
                if c.confidence <= threshold:
                    to_refine_idx.append(i)
                else:
                    pass_through_idx.append(i)
        else:
            to_refine_idx = list(range(len(cand_list)))
    elif config.confidence_threshold is not None:
        tau = config.confidence_threshold
        for i, c in enumerate(cand_list):
            if c.confidence < tau:
                to_refine_idx.append(i)
            else:
                pass_through_idx.append(i)
    else:
        to_refine_idx = list(range(len(cand_list)))

    # --- Decode windows for candidates that need refinement -------------
    to_refine_cands = [cand_list[i] for i in to_refine_idx]
    windows = [(max(0, c.frame_idx - wl), c.frame_idx + wr) for c in to_refine_cands]
    all_window_frames = collect_windows_single_pass(video_path, windows) if windows else []

    # Build result list preserving original candidate order.
    results: List[Optional[RefinedBoundary]] = [None] * len(cand_list)

    # Refine gated-in candidates.
    for i, (idx, cand, frames) in enumerate(zip(to_refine_idx, to_refine_cands, all_window_frames)):
        t0 = _time.perf_counter()
        result = _refine_candidate(cand, config, frames)
        t1 = _time.perf_counter()
        if result is None:
            continue
        result.coarse_frame_idx = cand.frame_idx
        result.signal_time_sec = t1 - t0
        results[idx] = result

    # Pass through high-confidence candidates unchanged.
    for idx in pass_through_idx:
        cand = cand_list[idx]
        # We don't have decoded frames, so construct a minimal RefinedBoundary
        # from what the coarse detector told us (no PTS/time_sec available here
        # without a frame index lookup — use 0.0 as sentinel; pipeline.py has
        # the frame_index to enrich if needed).
        results[idx] = RefinedBoundary(
            frame_idx=cand.frame_idx,
            end_frame_idx=cand.end_frame_idx,
            pts=0,
            time_sec=0.0,
            type="gradual_transition" if cand.is_gradual else "hard_cut",
            signal_peak=0.0,
            signal_width=0,
            sources=cand.sources,
            coarse_frame_idx=cand.frame_idx,
            signal_time_sec=0.0,
        )

    return [r for r in results if r is not None]


def _refine_candidate(
    cand: BoundaryCandidate,
    config: RefinementConfig,
    frames: List[DecodedFrame],
) -> Optional[RefinedBoundary]:
    """Refine a single candidate to frame-accurate (start, end) + type.

    Accepts pre-decoded frames from `collect_windows_single_pass`; the caller
    is responsible for supplying the ±W window around `cand.frame_idx`.
    """
    window = frames
    if len(window) < 2:
        # Edge of video — cannot compute a signal. Pass through as hard cut.
        if not window:
            return None
        f = window[0]
        return RefinedBoundary(
            frame_idx=f.frame_idx,
            end_frame_idx=f.frame_idx,
            pts=f.pts,
            time_sec=f.time_sec,
            type="hard_cut",
            signal_peak=0.0,
            signal_width=0,
            sources=cand.sources,
        )

    signal = compute_signal(window, config)

    # Find the range of frame *pairs* whose signal exceeds the threshold.
    above = signal > config.transition_threshold

    # If nothing exceeded threshold, fall back to argmax — Stage 1 said there's
    # something here, so emit the strongest pair as a hard cut. Stage 2's job
    # is to refine *position*, not to second-guess Stage 1's recall.
    if not above.any():
        peak_pair = int(np.argmax(signal))
        peak_frame = window[peak_pair + 1].frame_idx  # the "after" frame of the pair
        f = window[peak_pair + 1]
        return RefinedBoundary(
            frame_idx=peak_frame,
            end_frame_idx=peak_frame,
            pts=f.pts,
            time_sec=f.time_sec,
            type="hard_cut",
            signal_peak=float(signal[peak_pair]),
            signal_width=0,
            sources=cand.sources,
        )

    # Locate the longest contiguous elevated span (handles cases where the
    # window contains both a real cut and adjacent noise).
    span_start_pair, span_end_pair = _longest_true_run(above)
    span_width = span_end_pair - span_start_pair + 1

    # POST-FILTER for cursor flicks / one-frame animations:
    #   * very short elevated span (1–2 frames)
    #   * AND the signal has dropped back to baseline within recovery window
    # If both true, this is motion, not a state change — drop the candidate.
    # Disabled when motion_filter=False (e.g. hard-cut boundary refinement).
    if (
        config.motion_filter
        and span_width <= config.motion_min_persistent_frames
        and _recovers_quickly(signal, span_end_pair, config)
    ):
        return None

    # Decide hard vs gradual based on span width.
    if span_width >= config.min_gradual_frames:
        # GRADUAL: report the [start, end] *frame* indices of the span.
        # signal[i] is between window[i] and window[i+1], so the span of
        # *frames* affected is window[span_start_pair] .. window[span_end_pair+1].
        start_frame = window[span_start_pair]
        end_frame = window[min(span_end_pair + 1, len(window) - 1)]
        peak_pair = int(span_start_pair + np.argmax(signal[span_start_pair:span_end_pair + 1]))
        return RefinedBoundary(
            frame_idx=start_frame.frame_idx,
            end_frame_idx=end_frame.frame_idx,
            pts=start_frame.pts,
            time_sec=start_frame.time_sec,
            type="gradual_transition",
            signal_peak=float(signal[peak_pair]),
            signal_width=span_width,
            sources=cand.sources,
        )

    # HARD CUT: refine to the argmax pair within the elevated span.
    peak_pair = int(span_start_pair + np.argmax(signal[span_start_pair:span_end_pair + 1]))
    cut_frame = window[peak_pair + 1]  # the first frame of the new shot
    return RefinedBoundary(
        frame_idx=cut_frame.frame_idx,
        end_frame_idx=cut_frame.frame_idx,
        pts=cut_frame.pts,
        time_sec=cut_frame.time_sec,
        type="hard_cut",
        signal_peak=float(signal[peak_pair]),
        signal_width=span_width,
        sources=cand.sources,
    )


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """Return (start, end) inclusive of the longest True run in a 1D bool array."""
    best_start, best_end, best_len = 0, 0, 0
    cur_start = -1
    for i, v in enumerate(mask):
        if v and cur_start < 0:
            cur_start = i
        elif not v and cur_start >= 0:
            length = i - cur_start
            if length > best_len:
                best_len = length
                best_start, best_end = cur_start, i - 1
            cur_start = -1
    if cur_start >= 0:
        length = len(mask) - cur_start
        if length > best_len:
            best_start, best_end = cur_start, len(mask) - 1
    return best_start, best_end


def _recovers_quickly(
    signal: np.ndarray, span_end_pair: int, config: RefinementConfig
) -> bool:
    """True if the signal returns near baseline within `motion_recovery_frames`.

    "Near baseline" = below half the transition threshold. Used to decide
    whether a brief elevation looks like motion (recovers) vs a real state
    change (signal stays elevated because the new state looks different).
    """
    baseline = config.transition_threshold * 0.5
    look_to = min(len(signal), span_end_pair + 1 + config.motion_recovery_frames)
    tail = signal[span_end_pair + 1:look_to]
    if tail.size == 0:
        return False
    return bool(np.all(tail < baseline))


if __name__ == "__main__":
    # Standalone debug: refine a few hand-specified candidate frames.
    import sys

    from cuts.config import CutsConfig

    if len(sys.argv) < 3:
        print("usage: python -m cuts.refinement <video_path> <frame_idx> [<frame_idx> ...]")
        sys.exit(1)
    cfg = CutsConfig().refinement
    path = sys.argv[1]
    cands = [BoundaryCandidate(frame_idx=int(x), sources=["manual"]) for x in sys.argv[2:]]
    out = refine_candidates(path, cands, cfg)
    for r in out:
        print(r.to_dict(), "width=", r.signal_width, "sources=", r.sources)
