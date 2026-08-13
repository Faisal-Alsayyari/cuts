"""Representative frame sampling for retrieval segments.

For each SegmentRecord we pick a small set of frame indices that should
stand in for the whole segment during OCR / embedding / display. The choice
is deliberately sparse:

* one frame near the start (inset from boundary to skip fades),
* one frame at the midpoint,
* one frame near the end,
* plus one frame every ``sample_period_sec`` for long segments,
* hard-capped at ``max_frames_per_segment``.

Decoding is done in a single PyAV pass: we collect a set of target frame
indices across all segments, open the video once, and emit BGR frames at
each target. Frames are then written as JPEGs under
``<index_dir>/frames/<segment_id>_<k>.jpg`` and the segment records are
updated in place with paths, frame indices, and frame times.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

import av
import cv2
import numpy as np

from ..config import RetrievalConfig
from .schema import SegmentRecord


def _choose_frames_for_segment(
    seg: SegmentRecord,
    frame_index: List[Tuple[int, float]],
    config: RetrievalConfig,
) -> List[int]:
    """Return the frame indices to sample for one segment, sorted."""
    sf, ef = seg.start_frame, seg.end_frame
    if ef <= sf:
        return [sf]

    start_t = seg.start_time
    end_t = seg.end_time
    dur = max(0.0, end_t - start_t)

    # Start inset: move `sample_edge_inset_sec` in from each edge, clamped so
    # we never cross the midpoint when the segment is very short.
    inset = min(config.sample_edge_inset_sec, dur / 4.0) if dur > 0 else 0.0

    targets_t: List[float] = []
    targets_t.append(start_t + inset)
    targets_t.append((start_t + end_t) / 2.0)
    targets_t.append(end_t - inset)

    # Periodic sampling only when the segment has room for >= 2 periods.
    period = config.sample_period_sec
    if period > 0 and dur >= period * 2:
        t = start_t + period
        while t < end_t - inset * 0.5:
            targets_t.append(t)
            t += period

    # Convert times to nearest frame indices (search inside this segment's
    # frame span; frame_index is monotonic so a bisect-style scan works).
    chosen: Set[int] = set()
    span = frame_index[sf:ef + 1]
    span_times = [t for _, t in span]
    for tt in targets_t:
        # argmin |span_times - tt|
        idx_in_span = int(min(range(len(span_times)),
                              key=lambda i: abs(span_times[i] - tt)))
        chosen.add(sf + idx_in_span)

    frames = sorted(chosen)

    # Cap: keep the first, the last, and as many evenly spaced in between as
    # fit under the cap.
    cap = config.max_frames_per_segment
    if len(frames) > cap:
        # Keep endpoints and sample the rest uniformly.
        keep_idx = [0, len(frames) - 1]
        interior = cap - 2
        if interior > 0:
            step = (len(frames) - 1) / (interior + 1)
            for k in range(1, interior + 1):
                keep_idx.append(int(round(step * k)))
        keep_idx = sorted(set(keep_idx))[:cap]
        frames = [frames[i] for i in keep_idx]

    return frames


def _resize_long_edge(img: np.ndarray, long_edge: int) -> np.ndarray:
    if long_edge <= 0:
        return img
    h, w = img.shape[:2]
    if max(h, w) <= long_edge:
        return img
    if w >= h:
        new_w = long_edge
        new_h = int(round(h * (long_edge / w)))
    else:
        new_h = long_edge
        new_w = int(round(w * (long_edge / h)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def sample_frames(
    video_path: str,
    segments: List[SegmentRecord],
    index_dir: str,
    config: RetrievalConfig,
    frame_index: Optional[List[Tuple[int, float]]] = None,
    verbose: bool = False,
) -> None:
    """Decode representative frames for every segment in a single pass.

    Mutates ``segments`` in place: fills ``representative_frames``,
    ``metadata["frame_indices"]`` and ``metadata["frame_times"]``. Writes JPEGs
    under ``<index_dir>/frames/``.

    ``frame_index`` must be provided (the caller owns the table; we don't
    rebuild it here because it is used by the segmenter just before).
    """
    if not segments:
        return
    if frame_index is None:
        raise ValueError("sample_frames requires a prebuilt frame_index")

    frames_dir = os.path.join(index_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Build: target_frame_idx -> list of (segment_index, k-within-segment)
    targets: Dict[int, List[Tuple[int, int]]] = {}
    segment_frame_lists: List[List[int]] = []
    for si, seg in enumerate(segments):
        chosen = _choose_frames_for_segment(seg, frame_index, config)
        segment_frame_lists.append(chosen)
        for k, f in enumerate(chosen):
            targets.setdefault(f, []).append((si, k))
        # Pre-allocate per-segment slots so order is stable.
        seg.representative_frames = [""] * len(chosen)
        seg.metadata["frame_indices"] = list(chosen)
        seg.metadata["frame_times"] = [frame_index[f][1] for f in chosen]

    if not targets:
        return

    global_end = max(targets.keys())

    # Single decode pass: whenever the decoder emits a frame whose index is a
    # target, write the resized JPEG to disk and fill the path into every
    # segment slot that wanted it.
    container = av.open(video_path)
    written = 0
    try:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i > global_end:
                break
            if i not in targets:
                continue
            img = frame.to_ndarray(format="bgr24")
            img = _resize_long_edge(img, config.rep_frame_long_edge)
            for (si, k) in targets[i]:
                seg = segments[si]
                fname = f"{seg.segment_id}_{k:02d}.jpg"
                rel_path = os.path.join("frames", fname).replace(os.sep, "/")
                abs_path = os.path.join(frames_dir, fname)
                cv2.imwrite(
                    abs_path,
                    img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), config.rep_frame_jpeg_quality],
                )
                seg.representative_frames[k] = rel_path
                written += 1
    finally:
        container.close()

    if verbose:
        print(f"  sampled {written} representative frames "
              f"across {len(segments)} segments")


def load_representative_image(index_dir: str, rel_path: str) -> np.ndarray:
    """Convenience loader used by OCR / CLIP stages."""
    abs_path = os.path.join(index_dir, rel_path)
    img = cv2.imread(abs_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"representative frame missing: {abs_path}")
    return img
