"""Frame-accurate decoding and uniform temporal sampling via PyAV.

This is the foundation of the pipeline. Every module that reasons about frame
indices, PTS, or wall-clock times must do so via this module — never via
`frame_idx / fps`, which is wrong for VFR (variable frame rate) content.
Screen recorders are a common source of VFR output, so this matters here.

What this module guarantees:

* `build_frame_index(path)` materializes `frame_idx -> (pts, time_sec)` in a
  single decoding pass. This is the ground-truth lookup table for converting
  between frame indices and wall-clock times.
* `iter_sampled_frames(path, interval_sec)` streams one frame every
  `interval_sec` seconds. This is the entry point for the whole semantic
  pipeline (the EFS paper's "uniformly sampled at 1 fps" candidate set).
* `decode_frames_at(path, indices)` pulls a specific set of frames in one pass,
  for representative thumbnails after segmentation has run.

Memory note: `iter_sampled_frames` is a generator that yields full-resolution
frames one at a time and never accumulates. Consumers are expected to batch a
bounded number of them (see `SamplingConfig.batch_size`) and discard pixel data
as soon as per-frame signals are extracted. Holding every sampled frame of even
a 45-minute capture would exceed available memory on an 8 GB machine.

Performance note: decoding is sequential. PyAV can only reliably seek to
keyframes, so we decode every frame and pay `to_ndarray()` (the expensive step)
only on sampled ones. This is correct and fast enough for hour-scale inputs,
but it is linear in *total* video length rather than in sample count — a
multi-hour capture is decode-bound. Keyframe-seek sampling is the known
optimization if that becomes the bottleneck.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import av  # PyAV — ground truth for decoding order, PTS, and time_base
import numpy as np


@dataclass
class SampledFrame:
    """One decoded video frame with everything downstream code needs."""

    frame_idx: int        # ordinal index in decoding order, 0-based
    pts: int              # presentation timestamp in stream time_base units
    time_sec: float       # wall-clock time = pts * time_base, in seconds
    image: np.ndarray     # HxWx3 BGR uint8 (matches OpenCV conventions)


# ---------------------------------------------------------------------------
# Lookup table: frame_idx -> (pts, time_sec)
# ---------------------------------------------------------------------------

def build_frame_index(
    video_path: str, cache_dir: Optional[str] = None
) -> List[Tuple[int, float]]:
    """Single-pass scan to build the frame_idx -> (pts, time_sec) table.

    This is intentionally a *decode-light* pass: we still decode (PyAV does not
    expose per-frame PTS without decoding for many codecs reliably), but we
    discard the pixel data immediately, which is where the cost is.
    """
    cache_path = _cache_path_for(video_path, cache_dir)
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    container = av.open(video_path)
    table: List[Tuple[int, float]] = []
    try:
        stream = container.streams.video[0]
        time_base = stream.time_base
        for frame in container.decode(stream):
            pts = frame.pts if frame.pts is not None else (frame.dts or 0)
            table.append((int(pts), float(pts * time_base)))
    finally:
        container.close()

    if cache_path is not None:
        # Best-effort cache write; failure here is non-fatal.
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(table, f)
        except OSError:
            pass

    return table


def _cache_path_for(video_path: str, cache_dir: Optional[str]) -> Optional[str]:
    """Return the cache file path for `video_path`, or None if caching is off.

    Keyed on absolute path + mtime + size so re-encoding the same path
    invalidates the cache automatically.
    """
    if cache_dir is None:
        return None
    abs_path = os.path.abspath(video_path)
    try:
        st = os.stat(abs_path)
    except OSError:
        return None
    key = f"{abs_path.replace(os.sep, '_')}__{int(st.st_mtime)}__{st.st_size}.pkl"
    return os.path.join(cache_dir, "frame_index", key)


# ---------------------------------------------------------------------------
# Stream metadata helpers
# ---------------------------------------------------------------------------

def get_fps(video_path: str) -> float:
    """Return the average frame rate as a float.

    For VFR content this is only an *average* — never use it to compute
    per-frame timestamps; use the frame index table for that.
    """
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate
        return float(rate) if rate else 0.0
    finally:
        container.close()


def get_duration_sec(video_path: str) -> float:
    """Return container-reported duration in seconds (0.0 when unavailable).

    Cheap (no decoding). Use `build_frame_index()[-1][1]` when exactness
    matters; this is for progress estimation and sizing decisions.
    """
    container = av.open(video_path)
    try:
        if container.duration is not None:
            return float(container.duration / av.time_base)
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        return 0.0
    finally:
        container.close()


def has_audio(video_path: str) -> bool:
    """Return True when the container has at least one audio stream.

    Screen recordings frequently have no narration track, so every
    audio-dependent stage must check this and degrade gracefully.
    """
    try:
        container = av.open(video_path)
    except Exception:
        return False
    try:
        return len(container.streams.audio) > 0
    finally:
        container.close()


# ---------------------------------------------------------------------------
# Uniform temporal sampling — the entry point for the semantic pipeline
# ---------------------------------------------------------------------------

def iter_sampled_frames(
    video_path: str,
    interval_sec: float = 1.0,
    max_frames: Optional[int] = None,
) -> Iterator[SampledFrame]:
    """Yield one frame every `interval_sec` seconds of wall-clock time.

    Sampling is driven by each frame's true PTS-derived timestamp rather than
    by a frame-count stride, so the spacing stays correct under VFR. The first
    frame of the video is always emitted, then a frame is emitted whenever the
    elapsed time since the last emission reaches `interval_sec`.

    `to_ndarray()` is called only for emitted frames — that conversion, not the
    decode itself, dominates cost, so a 1 fps sample of 30 fps footage does
    roughly 1/30th of the pixel work of a full decode.

    Yields frames at native resolution; the caller is responsible for resizing
    and for not retaining more than a bounded batch at once.
    """
    if interval_sec <= 0:
        raise ValueError(f"interval_sec must be positive, got {interval_sec}")

    container = av.open(video_path)
    emitted = 0
    try:
        stream = container.streams.video[0]
        # Let PyAV use several threads for decode; this is a large win on the
        # sequential pass and costs nothing in memory terms.
        stream.thread_type = "AUTO"
        time_base = stream.time_base
        next_emit_at = 0.0

        for i, frame in enumerate(container.decode(stream)):
            pts = frame.pts if frame.pts is not None else (frame.dts or 0)
            t = float(pts * time_base)
            if t + 1e-9 < next_emit_at:
                continue
            yield SampledFrame(
                frame_idx=i,
                pts=int(pts),
                time_sec=t,
                # bgr24 matches OpenCV's channel order so downstream cv2 calls
                # don't need an extra cvtColor.
                image=frame.to_ndarray(format="bgr24"),
            )
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                break
            # Advance from the actual emitted time rather than accumulating the
            # nominal interval, so a long gap (dropped frames, VFR stall) does
            # not cause a burst of catch-up emissions afterwards.
            next_emit_at = t + interval_sec
    finally:
        container.close()


def decode_frames_at(
    video_path: str, indices: Iterable[int]
) -> Dict[int, np.ndarray]:
    """Decode a specific set of frame indices in ONE sequential pass.

    The naive alternative — reopening the video per index — is O(N x length).
    This opens once, materializes only the requested frames, and stops as soon
    as the last one is reached.

    Used after segmentation to pull representative thumbnails for labeling.
    """
    wanted = sorted(set(int(i) for i in indices))
    if not wanted:
        return {}
    wanted_set = set(wanted)
    global_end = wanted[-1]

    out: Dict[int, np.ndarray] = {}
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i > global_end:
                break
            if i in wanted_set:
                out[i] = frame.to_ndarray(format="bgr24")
    finally:
        container.close()
    return out


# ---------------------------------------------------------------------------
# Image helpers shared by every signal extractor
# ---------------------------------------------------------------------------

def resize_long_edge(img: np.ndarray, long_edge: int) -> np.ndarray:
    """Downscale so the longer side is `long_edge` px. No-op when already smaller.

    `long_edge <= 0` disables resizing entirely.
    """
    if long_edge <= 0:
        return img
    import cv2

    h, w = img.shape[:2]
    if max(h, w) <= long_edge:
        return img
    if w >= h:
        new_w, new_h = long_edge, max(1, int(round(h * (long_edge / w))))
    else:
        new_h, new_w = long_edge, max(1, int(round(w * (long_edge / h))))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    # Standalone debug: print metadata and sample a few frames.
    import sys
    import time

    if len(sys.argv) < 2:
        print("usage: python -m cuts.media <video_path> [interval_sec]")
        sys.exit(1)
    path = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    print(f"video:    {path}")
    print(f"fps:      {get_fps(path):.4f}")
    print(f"duration: {get_duration_sec(path):.2f}s")
    print(f"audio:    {has_audio(path)}")

    t0 = time.time()
    n = 0
    last_t = 0.0
    for sf in iter_sampled_frames(path, interval_sec=interval):
        if n < 5:
            print(f"  sample {n}: frame={sf.frame_idx} t={sf.time_sec:.3f}s "
                  f"shape={sf.image.shape}")
        last_t = sf.time_sec
        n += 1
    print(f"sampled {n} frames (last t={last_t:.2f}s) in {time.time() - t0:.2f}s")
