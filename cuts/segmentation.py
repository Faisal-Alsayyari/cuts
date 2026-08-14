"""EFS Stage 1 — event boundaries from a fused temporal similarity curve.

Implements Stage 1 of "Event-Anchored Frame Selection for Effective Long-Video
Understanding" (Chen et al., arXiv 2603.00983), generalized from one signal
channel to several.

The paper's algorithm:

  1. Sample candidate frames uniformly (1 fps) and embed them with DINOv2.
  2. Score each frame by its weighted similarity to its temporal neighbours
     (paper eq. 1):

         s_i = sum_{j in N(i)} w_{|i-j|} * cos(f_i, f_j)
               ------------------------------------------
                        sum_{j in N(i)} w_{|i-j|}

     with a window of half-width l (default 3) and weights that fall off
     linearly with distance.
  3. Cut at local minima of that curve — the points of maximal visual change.
  4. If the resulting partition count exceeds the target M, repeatedly merge
     the most-similar adjacent pair (cosine between their mean features) until
     exactly M events remain.

Our generalization: steps 2 and 4 run per *channel* (DINOv2 embeddings, OCR
TF-IDF vectors, transcript TF-IDF vectors) and the curves are fused. The
paper's exact behaviour is recovered by setting `weight_dino=1.0` with the
other weights at zero.

Why fuse at all: DINOv2 is self-supervised on natural images, and on screen
recordings two completely different code screens are both "monospace text on a
dark background". Measured on a synthetic coding-session fixture, every true
boundary did land among the deepest DINO dips — but some true boundaries
ranked *below* false dips caused by nothing more than content scrolling. The
text channel separates those cases because the words on screen change
completely at a real activity change and barely at all during a scroll.

Fusion happens in z-score space. The channels have incomparable native scales
(DINOv2 adjacent cosine sits around 0.99 on screen capture; TF-IDF cosine is
far lower and much more spread out), so a raw weighted average would let
whichever channel has the larger variance silently dominate. Standardizing
each curve first makes the configured weights mean what they say. Boundary
detection only needs relative minima, so losing the absolute scale costs
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import SegmentationConfig
from .schema import Event
from .signals.text_features import text_features


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class SegmentationInput:
    """Everything segmentation needs about the sampled frames.

    All sequences are parallel and ordered by time. `dino` may be None when
    the visual channel is disabled; `ocr_texts` / `transcript_texts` may be
    None when those signals were not collected.
    """

    frame_indices: List[int]
    times: List[float]
    dino: Optional[np.ndarray] = None
    ocr_texts: Optional[List[str]] = None
    transcript_texts: Optional[List[str]] = None

    def __len__(self) -> int:
        return len(self.frame_indices)


# ---------------------------------------------------------------------------
# Paper eq. 1 — weighted-neighbourhood similarity curve
# ---------------------------------------------------------------------------

def similarity_curve(features: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Weighted temporal similarity score per sample (paper eq. 1).

    `features` must be an ``(N, D)`` L2-normalized matrix, so cosine similarity
    is a dot product. Returns a length-``N`` array where a LOW value means the
    sample disagrees with its neighbourhood — i.e. a candidate boundary.

    Weights fall off linearly: a neighbour at distance d contributes
    ``1 - d / (l + 1)``, so d=1 dominates and d=l contributes least. Samples
    near the ends of the video simply have fewer neighbours; averaging by the
    realized weight sum keeps their scores comparable rather than
    artificially depressed.
    """
    n = len(features)
    if n == 0 or features.shape[1] == 0:
        return np.zeros(n, dtype=np.float32)
    if n == 1:
        return np.ones(1, dtype=np.float32)

    numer = np.zeros(n, dtype=np.float64)
    denom = np.zeros(n, dtype=np.float64)

    l = max(1, int(window_size))
    for d in range(1, l + 1):
        if d >= n:
            break
        w = 1.0 - d / (l + 1.0)
        if w <= 0:
            continue
        # sims[k] = cos(features[k], features[k + d])
        sims = np.sum(features[:-d] * features[d:], axis=1).astype(np.float64)
        # Each pair informs both of its endpoints.
        numer[: n - d] += w * sims
        denom[: n - d] += w
        numer[d:] += w * sims
        denom[d:] += w

    out = np.ones(n, dtype=np.float32)
    valid = denom > 0
    out[valid] = (numer[valid] / denom[valid]).astype(np.float32)
    return out


def _zscore(curve: np.ndarray) -> np.ndarray:
    """Standardize a curve so channels become comparable before fusion."""
    if curve.size == 0:
        return curve
    mu = float(curve.mean())
    sd = float(curve.std())
    if sd < 1e-8:
        # A perfectly flat channel carries no boundary information; return
        # zeros so it contributes nothing rather than amplifying float noise.
        return np.zeros_like(curve)
    return ((curve - mu) / sd).astype(np.float32)


# ---------------------------------------------------------------------------
# Channel assembly + fusion
# ---------------------------------------------------------------------------

def build_channels(
    inp: SegmentationInput, config: SegmentationConfig
) -> Dict[str, np.ndarray]:
    """Build the per-channel L2-normalized feature matrices that are available.

    A channel is included only when it has a non-zero configured weight AND
    actually carries content. Text channels that produced no tokens at all
    (no audio, or OCR turned up nothing) are dropped rather than contributing
    an all-zero matrix, which would otherwise read as "everything changed".
    """
    channels: Dict[str, np.ndarray] = {}

    if config.weight_dino > 0 and inp.dino is not None and len(inp.dino):
        channels["dino"] = inp.dino

    if config.weight_ocr > 0 and inp.ocr_texts:
        feats = text_features(inp.ocr_texts)
        if feats.shape[1] > 0 and np.any(feats):
            channels["ocr"] = feats

    if config.weight_asr > 0 and inp.transcript_texts:
        feats = text_features(inp.transcript_texts)
        if feats.shape[1] > 0 and np.any(feats):
            channels["asr"] = feats

    return channels


def fuse_curves(
    curves: Dict[str, np.ndarray], config: SegmentationConfig
) -> np.ndarray:
    """Weighted fusion of per-channel curves, in z-score space.

    Weights are renormalized over the channels actually present, so disabling
    a channel reweights the survivors instead of shrinking the fused signal.
    """
    if not curves:
        return np.zeros(0, dtype=np.float32)

    weights = {
        "dino": config.weight_dino,
        "ocr": config.weight_ocr,
        "asr": config.weight_asr,
    }
    active = {k: max(0.0, weights.get(k, 0.0)) for k in curves}
    total = sum(active.values())
    if total <= 0:
        # Configured weights were all zero for the available channels; fall
        # back to an equal blend rather than returning a dead signal.
        active = {k: 1.0 for k in curves}
        total = float(len(curves))

    fused: Optional[np.ndarray] = None
    for name, curve in curves.items():
        z = _zscore(curve) * (active[name] / total)
        fused = z if fused is None else fused + z
    return fused.astype(np.float32)


# ---------------------------------------------------------------------------
# Boundary detection
# ---------------------------------------------------------------------------

def find_local_minima(curve: np.ndarray) -> List[int]:
    """Indices of local minima in `curve`.

    A point qualifies when it is no greater than both neighbours. Plateaus are
    reported once, at their centre, so a flat-bottomed dip does not emit a
    cluster of adjacent boundaries.
    """
    n = len(curve)
    if n < 3:
        return []

    minima: List[int] = []
    i = 1
    while i < n - 1:
        if curve[i] <= curve[i - 1] and curve[i] <= curve[i + 1]:
            # Walk to the end of any plateau at this value.
            j = i
            while j + 1 < n - 1 and curve[j + 1] == curve[i]:
                j += 1
            if curve[j] <= curve[j + 1]:
                minima.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    return minima


# ---------------------------------------------------------------------------
# Merge-to-target (paper step 4)
# ---------------------------------------------------------------------------

def _partition_means(
    channels: Dict[str, np.ndarray], bounds: List[int], n: int
) -> Dict[str, np.ndarray]:
    """Mean feature vector per partition, per channel, L2-renormalized.

    `bounds` holds the start index of each partition after the first, so
    partitions are ``[0, b0), [b0, b1), ..., [b_{k-1}, n)``.
    """
    starts = [0] + list(bounds)
    ends = list(bounds) + [n]
    out: Dict[str, np.ndarray] = {}
    for name, feats in channels.items():
        means = np.zeros((len(starts), feats.shape[1]), dtype=np.float32)
        for p, (s, e) in enumerate(zip(starts, ends)):
            if e > s:
                means[p] = feats[s:e].mean(axis=0)
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        np.divide(means, norms, out=means, where=norms > 0)
        out[name] = means
    return out


def _adjacent_partition_similarity(
    means: Dict[str, np.ndarray], config: SegmentationConfig
) -> np.ndarray:
    """Fused similarity between each consecutive pair of partitions.

    Unlike curve fusion this uses raw cosines rather than z-scores: with only
    a handful of partitions there is not enough population to standardize
    against, and all channels produce cosines on a comparable [0, 1] range for
    normalized non-negative-ish features. Higher = more similar = merge first.
    """
    weights = {
        "dino": config.weight_dino,
        "ocr": config.weight_ocr,
        "asr": config.weight_asr,
    }
    active = {k: max(0.0, weights.get(k, 0.0)) for k in means}
    total = sum(active.values()) or float(len(means))
    if sum(active.values()) <= 0:
        active = {k: 1.0 for k in means}

    fused: Optional[np.ndarray] = None
    for name, m in means.items():
        if len(m) < 2:
            return np.zeros(0, dtype=np.float32)
        sims = np.sum(m[:-1] * m[1:], axis=1).astype(np.float32)
        contrib = sims * (active[name] / total)
        fused = contrib if fused is None else fused + contrib
    return fused if fused is not None else np.zeros(0, dtype=np.float32)


def merge_to_target(
    bounds: List[int],
    channels: Dict[str, np.ndarray],
    n: int,
    target: int,
    config: SegmentationConfig,
) -> List[int]:
    """Iteratively merge the most-similar adjacent partitions down to `target`.

    This is the paper's step 4. Merging by *partition mean* similarity rather
    than by boundary depth is what makes the result stable: a boundary can look
    sharp locally while separating two stretches that are, taken as wholes,
    the same activity.
    """
    bounds = sorted(set(bounds))
    while len(bounds) + 1 > target and bounds:
        means = _partition_means(channels, bounds, n)
        sims = _adjacent_partition_similarity(means, config)
        if sims.size == 0:
            break
        # sims[i] compares partition i and i+1, which are separated by
        # bounds[i]; dropping that boundary merges them.
        bounds.pop(int(np.argmax(sims)))
    return bounds


def _enforce_min_duration(
    bounds: List[int], times: Sequence[float], min_sec: float, total_end: float
) -> List[int]:
    """Drop boundaries that would create an event shorter than `min_sec`.

    Applied after merging so the target count is respected first. Walks left to
    right and removes any boundary too close to the previous one, which
    prevents three-second "chapters" without disturbing well-spaced ones.
    """
    if min_sec <= 0 or not bounds:
        return bounds
    kept: List[int] = []
    last_start_t = times[0] if times else 0.0
    for b in sorted(bounds):
        t = times[b] if b < len(times) else total_end
        if t - last_start_t >= min_sec:
            kept.append(b)
            last_start_t = t
    # The final event must also clear the bar; if not, drop its opening bound.
    if kept:
        last_t = times[kept[-1]] if kept[-1] < len(times) else total_end
        if total_end - last_t < min_sec:
            kept.pop()
    return kept


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def resolve_target_events(
    config: SegmentationConfig, duration_sec: float
) -> int:
    """Decide how many events to aim for.

    An explicit `target_events` wins. Otherwise scale with duration so a
    3-minute clip and a 3-hour session both get a sensible chapter count,
    clamped to [min_events, max_events].
    """
    if config.target_events is not None:
        return max(1, int(config.target_events))
    per = max(0.1, config.minutes_per_event)
    derived = int(round((duration_sec / 60.0) / per))
    return int(np.clip(derived, config.min_events, config.max_events))


def segment(
    inp: SegmentationInput,
    config: SegmentationConfig,
    video_id: str = "video",
    duration_sec: float = 0.0,
    verbose: bool = False,
) -> List[Event]:
    """Run EFS Stage 1 and return the resulting events.

    Returns a single whole-video event when there is not enough signal to
    partition on (very short input, or every channel unavailable) — callers
    always get a valid, complete cover of the timeline.
    """
    n = len(inp)
    if n == 0:
        return []

    end_time = duration_sec or (inp.times[-1] if inp.times else 0.0)

    channels = build_channels(inp, config)
    if verbose:
        print(f"  channels: {list(channels.keys()) or ['(none)']}")
    if not channels or n < 3:
        return _events_from_bounds([], inp, video_id, end_time, None)

    curves = {
        name: similarity_curve(feats, config.window_size)
        for name, feats in channels.items()
    }
    fused = fuse_curves(curves, config)

    minima = find_local_minima(fused)
    target = resolve_target_events(config, end_time)
    if verbose:
        print(f"  local minima: {len(minima)}  ->  target events: {target}")

    bounds = merge_to_target(minima, channels, n, target, config)
    bounds = _enforce_min_duration(
        bounds, inp.times, config.min_event_sec, end_time
    )
    if verbose:
        print(f"  final boundaries: {len(bounds)} -> {len(bounds) + 1} events")

    return _events_from_bounds(bounds, inp, video_id, end_time, fused)


def _events_from_bounds(
    bounds: List[int],
    inp: SegmentationInput,
    video_id: str,
    end_time: float,
    fused: Optional[np.ndarray],
) -> List[Event]:
    """Materialize Event records from sample-space boundary indices."""
    n = len(inp)
    starts = [0] + list(bounds)
    ends = list(bounds) + [n]

    events: List[Event] = []
    for k, (s, e) in enumerate(zip(starts, ends)):
        if e <= s:
            continue
        sample_idx = list(range(s, e))
        start_time = inp.times[s]
        # An event runs up to the start of the next one, so the timeline is a
        # gapless cover; the final event runs to the true end of the video.
        end_t = inp.times[e] if e < n else end_time
        events.append(Event(
            video_id=video_id,
            event_id=f"{k:04d}",
            start_frame=inp.frame_indices[s],
            end_frame=inp.frame_indices[min(e, n - 1)],
            start_time=float(start_time),
            end_time=float(max(end_t, start_time)),
            sample_indices=[inp.frame_indices[i] for i in sample_idx],
            metadata={
                "boundary_score": (
                    float(fused[s]) if fused is not None and s < len(fused) else None
                ),
                "n_samples": len(sample_idx),
            },
        ))
    return events


if __name__ == "__main__":
    # Standalone debug: segment a video using the visual channel only, which
    # needs no OCR/ASR setup, and print the resulting chapter skeleton.
    import sys
    import time

    from .config import CutsConfig
    from .media import get_duration_sec, iter_sampled_frames
    from .schema import format_timestamp
    from .signals.dino import embed_stream

    if len(sys.argv) < 2:
        print("usage: python -m cuts.segmentation <video_path> [interval_sec]")
        sys.exit(1)
    path = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    cfg = CutsConfig()
    cfg.segmentation.weight_ocr = 0.0  # visual-only for this debug entry point
    cfg.verbose = True

    t0 = time.time()
    metas: List[Tuple[int, float]] = []
    frames = []
    for sf in iter_sampled_frames(path, interval_sec=interval):
        metas.append((sf.frame_idx, sf.time_sec))
        frames.append(sf)
    emb = embed_stream(iter(frames), cfg.dino, device=cfg.device,
                       batch_size=cfg.sampling.batch_size)
    del frames

    inp = SegmentationInput(
        frame_indices=[m[0] for m in metas],
        times=[m[1] for m in metas],
        dino=emb,
    )
    evs = segment(inp, cfg.segmentation, duration_sec=get_duration_sec(path),
                  verbose=True)
    print(f"\n{len(evs)} events in {time.time() - t0:.1f}s:")
    for e in evs:
        print(f"  {format_timestamp(e.start_time):>7}  "
              f"({e.duration_sec:5.1f}s, {e.metadata['n_samples']} samples)")
