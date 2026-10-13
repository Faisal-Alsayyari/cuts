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

Our generalization: steps 2 and 4 run per *channel* (DINOv2 embeddings, VLM
description TF-IDF vectors, transcript TF-IDF vectors) and the curves are
fused. The paper's exact behaviour is recovered by setting `weight_dino=1.0`
with the other weights at zero.

Why fuse at all: DINOv2 is self-supervised on natural images, and on screen
recordings two completely different code screens are both "monospace text on a
dark background". Measured on a synthetic coding-session fixture, every true
boundary did land among the deepest DINO dips — but some true boundaries
ranked *below* false dips caused by nothing more than content scrolling. The
text channel separates those cases because the words on screen change
completely at a real activity change and barely at all during a scroll.

Here the "sampled frames" are Pass 1 observations (`cuts.observe`): the
DINOv2 vector and the description are both recorded per observation, so this
segmenter reads the same cache as `cuts.coalesce` and needs no decode of its
own. It returns boundary *positions*; `cuts.chapters` turns them into events.

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
from typing import Dict, List, Optional, Sequence

import numpy as np

from .chapters import enforce_min_duration
from .config import SegmentationConfig
from .observe import OK, UNDECODABLE, Observation
from .signals.text_features import text_features


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class SegmentationInput:
    """Everything segmentation needs about the samples.

    All sequences are parallel and ordered by time. `dino` may be None when
    the visual channel is disabled; `vlm_texts` / `transcript_texts` may be
    None when those signals were not collected.
    """

    times: List[float]
    dino: Optional[np.ndarray] = None
    vlm_texts: Optional[List[str]] = None
    transcript_texts: Optional[List[str]] = None

    def __len__(self) -> int:
        return len(self.times)


@dataclass
class Segmentation:
    """What `segment` decided: boundary positions after the first event."""

    bounds: List[int]
    fused: Optional[np.ndarray] = None
    channels: List[str] = field(default_factory=list)

    @property
    def starts(self) -> List[int]:
        return [0] + list(self.bounds)

    def boundary_score(self, start: int) -> Optional[float]:
        if self.fused is None or start >= len(self.fused):
            return None
        return float(self.fused[start])


def carry_forward(
    texts: Sequence[Optional[str]],
    reset_at: Optional[Sequence[int]] = None,
) -> List[str]:
    """Fill gaps with the most recent description.

    A sample with no description (withheld, or its call failed) is not a
    change on screen: filling it with "" would read as "everything vanished"
    and manufacture a boundary. `reset_at` lists positions where that
    assumption breaks — the file joins, since screen content does not
    persist across a gap of hours between two recordings.
    """
    resets = set(reset_at or ())
    out: List[str] = []
    last = ""
    for i, t in enumerate(texts):
        if i in resets:
            last = ""
        if t:
            last = t
        out.append(last)
    return out


def usable_observations(
    observations: Sequence[Observation], use_dino: bool
) -> List[Observation]:
    """The observations the curve can place: decodable, and embedded if needed."""
    return [o for o in observations
            if o.status != UNDECODABLE
            and (not use_dino or o.embedding is not None)]


def input_from_observations(
    observations: Sequence[Observation],
    use_dino: bool,
    joins: Sequence[int] = (),
) -> SegmentationInput:
    """Build the curve's input from (usable) observations."""
    dino = None
    if use_dino and observations and all(o.embedding is not None for o in observations):
        dino = np.asarray([o.embedding for o in observations], dtype=np.float32)
    texts = carry_forward(
        [o.description if o.status == OK else None for o in observations],
        reset_at=joins,
    )
    return SegmentationInput(
        times=[o.t for o in observations],
        dino=dino,
        vlm_texts=texts,
        transcript_texts=[o.transcript_text for o in observations],
    )


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
    (no audio, or every VLM description empty) are dropped rather than
    contributing an all-zero matrix, which would otherwise read as
    "everything changed".
    """
    channels: Dict[str, np.ndarray] = {}

    if config.weight_dino > 0 and inp.dino is not None and len(inp.dino):
        channels["dino"] = inp.dino

    if config.weight_vlm > 0 and inp.vlm_texts:
        feats = text_features(inp.vlm_texts)
        if feats.shape[1] > 0 and np.any(feats):
            channels["vlm"] = feats

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
        "vlm": config.weight_vlm,
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

    A minimum marks *where* the curve dips, not which side of the cut each
    sample belongs to — see `snap_to_cuts` for that.
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


def snap_to_cuts(
    minima: Sequence[int],
    channels: Dict[str, np.ndarray],
    config: SegmentationConfig,
) -> List[int]:
    """Turn each dip into the position where the new event starts.

    The neighbourhood score is symmetric: a clean change between samples
    k-1 and k depresses *both* of them by the same amount, so which one is
    the minimum is decided by codec noise, and half the time the chapter
    would start one sample early. The cut itself is the least similar
    adjacent pair, which is not ambiguous — so each minimum m is moved to
    m if the pair (m-1, m) is the sharper change, else to m+1. Adjacent
    similarity is fused across channels the same way the curve is.
    """
    n = next(iter(channels.values())).shape[0] if channels else 0
    if n < 2:
        return list(minima)
    adjacent = {
        name: np.sum(feats[:-1] * feats[1:], axis=1).astype(np.float32)
        for name, feats in channels.items()
    }
    # adj[k] = fused similarity between samples k and k+1
    adj = fuse_curves(adjacent, config)
    out = set()
    for m in minima:
        before = adj[m - 1] if m >= 1 else np.inf
        after = adj[m] if m < n - 1 else np.inf
        start = m if before <= after else m + 1
        if 0 < start < n:
            out.add(start)
    return sorted(out)


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
        "vlm": config.weight_vlm,
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
    locked: Optional[Sequence[int]] = None,
) -> List[int]:
    """Iteratively merge the most-similar adjacent partitions down to `target`.

    This is the paper's step 4. Merging by *partition mean* similarity rather
    than by boundary depth is what makes the result stable: a boundary can look
    sharp locally while separating two stretches that are, taken as wholes,
    the same activity.

    `locked` boundaries are never merged away. They exist for structural cuts
    the signal cannot be expected to argue for — chiefly the joins between
    source files, which are real discontinuities in the recording no matter
    how similar the pixels on either side happen to look. When every remaining
    boundary is locked the merge stops early, so the target count is a goal
    rather than a guarantee.
    """
    locked_set = set(locked or ())
    bounds = sorted(set(bounds) | locked_set)
    while len(bounds) + 1 > target and bounds:
        means = _partition_means(channels, bounds, n)
        sims = _adjacent_partition_similarity(means, config)
        if sims.size == 0:
            break
        # sims[i] compares partition i and i+1, which are separated by
        # bounds[i]; dropping that boundary merges them.
        candidates = [i for i, b in enumerate(bounds) if b not in locked_set]
        if not candidates:
            break
        bounds.pop(max(candidates, key=lambda i: float(sims[i])))
    return bounds


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
    duration_sec: float = 0.0,
    verbose: bool = False,
    forced_bounds: Optional[Sequence[int]] = None,
) -> Segmentation:
    """Run EFS Stage 1 and return where events begin.

    With too little to partition on (very short input, or no channel with
    content) only the forced boundaries are returned — callers always get a
    valid, complete cover of the timeline.

    `forced_bounds` are sample positions that must open a new event regardless
    of what the signal says: the source-file joins.
    """
    n = len(inp)
    end_time = duration_sec or (inp.times[-1] if inp.times else 0.0)

    # A forced bound at 0 is redundant (the first event starts there anyway)
    # and one at n would open an empty trailing event.
    locked = sorted({int(b) for b in (forced_bounds or ()) if 0 < b < n})

    channels = build_channels(inp, config)
    if verbose:
        print(f"  channels: {list(channels.keys()) or ['(none)']}")
    if not channels or n < 3:
        return Segmentation(bounds=locked, channels=list(channels))

    curves = {
        name: similarity_curve(feats, config.window_size)
        for name, feats in channels.items()
    }
    fused = fuse_curves(curves, config)

    minima = snap_to_cuts(find_local_minima(fused), channels, config)
    target = resolve_target_events(config, end_time)
    # The locked boundaries alone already imply this many events; asking for
    # fewer is unsatisfiable, so raise the target instead of merging forever.
    target = max(target, len(locked) + 1)
    if verbose:
        print(f"  local minima: {len(minima)}  ->  target events: {target}"
              + (f"  ({len(locked)} locked)" if locked else ""))

    bounds = merge_to_target(minima, channels, n, target, config, locked=locked)
    bounds = enforce_min_duration(
        bounds, inp.times, config.min_event_sec, end_time, locked=locked
    )
    if verbose:
        print(f"  final boundaries: {len(bounds)} -> {len(bounds) + 1} events")
    return Segmentation(bounds=list(bounds), fused=fused, channels=list(channels))


if __name__ == "__main__":
    # Standalone debug: segment a cached observations file with the curve
    # (text and, when cached, visual channels) and print the skeleton.
    import sys

    from .chapters import join_positions
    from .config import CutsConfig
    from .observe import read_observations
    from .schema import format_timestamp

    if len(sys.argv) < 2:
        print("usage: python -m cuts.segmentation <observations.jsonl>")
        sys.exit(1)
    cfg = CutsConfig()
    obs, _meta = read_observations(sys.argv[1])
    use_dino = any(o.embedding is not None for o in obs)
    obs = usable_observations(obs, use_dino)
    joins = join_positions(obs)
    seg = segment(input_from_observations(obs, use_dino, joins),
                  cfg.segmentation, verbose=True, forced_bounds=joins)
    for s0 in seg.starts:
        print(f"  {format_timestamp(obs[s0].t):>7}  score={seg.boundary_score(s0)}")
