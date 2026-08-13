"""Frame-accurate decoding via PyAV.

This is the foundation of the entire pipeline. Every module that reasons about
frame indices, PTS, or wall-clock times must do so via this module — never via
`frame_idx / fps`, which is wrong for VFR (variable frame rate) content.

What this module guarantees:

* `iter_frames(path)` yields every decoded frame in decoding order, with the
  ordinal index `i` (0, 1, 2, ...) exposed alongside `pts` and `time_sec`.
* `build_frame_index(path)` materializes a list `frame_idx -> (pts, time_sec)`
  in a single decoding pass. This is the ground-truth lookup table the rest of
  the pipeline uses to convert between frame indices and times.
* `extract_window(path, center, W)` decodes a [center-W, center+W] window with
  every frame (no NONKEY skipping), suitable for Stage 2 refinement.
* `decode_strided(path, stride)` yields every Nth decoded frame, for Stage 3.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import av  # PyAV — ground truth for decoding order, PTS, and time_base
import numpy as np


@dataclass
class DecodedFrame:
    """One decoded video frame with everything downstream code needs."""

    frame_idx: int        # ordinal index in decoding order, 0-based
    pts: int              # presentation timestamp in stream time_base units
    time_sec: float       # wall-clock time = pts * time_base, in seconds
    image: np.ndarray     # HxWx3 BGR uint8 (matches OpenCV conventions)


# ---------------------------------------------------------------------------
# Core iteration
# ---------------------------------------------------------------------------

def iter_frames(video_path: str) -> Iterator[DecodedFrame]:
    """Yield every frame of `video_path` in decoding order.

    The frame index `i` is purely ordinal — it counts decoded frames as they
    come out of the decoder. This is the same convention TransNetV2 uses
    internally, so frame indices line up across the pipeline.
    """
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        # `time_base` is a Fraction; PTS is an integer in those units.
        time_base = stream.time_base
        for i, frame in enumerate(container.decode(stream)):
            # Some containers can emit frames with pts=None at boundaries.
            # Fall back to dts when that happens; both are in time_base units.
            pts = frame.pts if frame.pts is not None else (frame.dts or 0)
            yield DecodedFrame(
                frame_idx=i,
                pts=int(pts),
                time_sec=float(pts * time_base),
                # bgr24 matches OpenCV's color order so downstream cv2 calls
                # don't need an extra cvtColor for histogram / SSIM ops.
                image=frame.to_ndarray(format="bgr24"),
            )
    finally:
        container.close()


# ---------------------------------------------------------------------------
# Lookup table: frame_idx -> (pts, time_sec)
# ---------------------------------------------------------------------------

def build_frame_index(
    video_path: str, cache_dir: Optional[str] = None
) -> List[Tuple[int, float]]:
    """Single-pass scan to build the frame_idx -> (pts, time_sec) table.

    This is intentionally a *decode-light* pass: we still decode (PyAV does not
    expose per-frame PTS without decoding for many codecs reliably), but we
    discard the pixel data immediately. For a 5-minute 1080p clip this is on
    the order of seconds. The result is the ground-truth frame count and
    timing map for the rest of the pipeline.
    """
    cache_path = _cache_path_for(video_path, cache_dir)
    if cache_path is not None and os.path.exists(cache_path):
        # Cache hit — skip the decode pass entirely.
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

    We key on absolute path + mtime + size so that re-encoding the same path
    invalidates the cache automatically.
    """
    if cache_dir is None:
        return None
    abs_path = os.path.abspath(video_path)
    try:
        st = os.stat(abs_path)
    except OSError:
        return None
    # Hash-free key: simple string with mtime/size makes debugging easier.
    key = f"{abs_path.replace(os.sep, '_')}__{int(st.st_mtime)}__{st.st_size}.pkl"
    return os.path.join(cache_dir, "frame_index", key)


# ---------------------------------------------------------------------------
# Stream metadata helpers
# ---------------------------------------------------------------------------

def get_fps(video_path: str) -> float:
    """Return the average frame rate of `video_path` as a float.

    Uses `stream.average_rate` rather than `guessed_rate`. For VFR content this
    is still only an *average* — never use it to compute per-frame timestamps;
    use the frame index table for that.
    """
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        if stream.average_rate is None:
            # Fall back to base_rate when average_rate is missing (rare).
            rate = stream.base_rate
        else:
            rate = stream.average_rate
        return float(rate)
    finally:
        container.close()


def get_frame_count(video_path: str) -> int:
    """Return the container-reported frame count.

    Note: this is unreliable for some encoders (it can be wrong by ±1 or, for
    streamed/partially written files, very wrong). Use `len(build_frame_index)`
    when you need ground truth.
    """
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        return int(stream.frames) if stream.frames else 0
    finally:
        container.close()


# ---------------------------------------------------------------------------
# Window decoding for Stage 2 refinement
# ---------------------------------------------------------------------------

def extract_window(
    video_path: str, center_frame: int, half_window: int
) -> List[DecodedFrame]:
    """Decode the [center-W, center+W] window with EVERY frame (not just keys).

    Implementation note: PyAV `container.seek()` is fast but lands on the
    nearest preceding keyframe and is unreliable for non-keyframe targets. For
    the small windows we use (±8 by default) decoding from the preceding
    keyframe forward is correct and acceptably fast. Critically we do NOT set
    `skip_frame = "NONKEY"` — refinement requires every frame.
    """
    start = max(0, center_frame - half_window)
    end = center_frame + half_window  # inclusive
    container = av.open(video_path)
    out: List[DecodedFrame] = []
    try:
        stream = container.streams.video[0]
        time_base = stream.time_base
        # Decode from the start; for short clips this is a few seconds even
        # without seeking. For longer videos prefer batching candidates by
        # locality (see refinement.py for the batched variant).
        for i, frame in enumerate(container.decode(stream)):
            if i < start:
                continue
            if i > end:
                break
            pts = frame.pts if frame.pts is not None else (frame.dts or 0)
            out.append(DecodedFrame(
                frame_idx=i,
                pts=int(pts),
                time_sec=float(pts * time_base),
                image=frame.to_ndarray(format="bgr24"),
            ))
    finally:
        container.close()
    return out


def decode_strided(
    video_path: str,
    stride: int,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> Iterator[DecodedFrame]:
    """Yield every `stride`-th decoded frame in [start_frame, end_frame].

    Used by Stage 3 (event detector) to scan inside a shot cheaply. We still
    decode every frame (because we cannot reliably skip non-keyframes) but we
    only emit one in `stride`.
    """
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        time_base = stream.time_base
        for i, frame in enumerate(container.decode(stream)):
            if i < start_frame:
                continue
            if end_frame is not None and i > end_frame:
                break
            # Anchor the modulus to start_frame so the first emitted frame is
            # always start_frame itself (predictable comparison pairing).
            if (i - start_frame) % stride != 0:
                continue
            pts = frame.pts if frame.pts is not None else (frame.dts or 0)
            yield DecodedFrame(
                frame_idx=i,
                pts=int(pts),
                time_sec=float(pts * time_base),
                image=frame.to_ndarray(format="bgr24"),
            )
    finally:
        container.close()


# ---------------------------------------------------------------------------
# Batch single-pass helpers — eliminate O(N × video_length) decode loops
# ---------------------------------------------------------------------------

def collect_windows_single_pass(
    video_path: str,
    windows: List[Tuple[int, int]],
) -> List[List[DecodedFrame]]:
    """Collect multiple frame windows in ONE sequential decode pass.

    The naive alternative — calling `extract_window` once per candidate — opens
    and scans the video from frame 0 for EVERY candidate, giving O(N × length)
    work. This function opens the video once and activates/deactivates per-window
    collection buffers as the decoder advances.

    `windows`: (start_frame, end_frame) INCLUSIVE pairs, in any order.
    Returns a list in the SAME ORDER as `windows`; each entry is the list of
    DecodedFrames that fall in that window.

    Used by Stage 2 (refinement) to batch all candidate windows.
    """
    if not windows:
        return []

    # Sort window indices by start_frame so we can activate them in stream order.
    order = sorted(range(len(windows)), key=lambda i: windows[i][0])
    sorted_wins = [windows[i] for i in order]
    # Furthest frame we ever need; stop decoding after this.
    global_end = max(w[1] for w in windows)

    # Pre-allocate one result buffer per window, indexed by ORIGINAL index.
    results: List[List[DecodedFrame]] = [[] for _ in windows]

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        time_base = stream.time_base
        next_win_ptr = 0       # next index into sorted_wins to activate
        active: List[int] = [] # original indices of currently open windows

        for i, frame in enumerate(container.decode(stream)):
            # Stop decoding once we've passed all windows — saves decoding
            # the tail of the video entirely.
            if i > global_end:
                break

            # Activate any windows whose start_frame <= current frame index.
            while (next_win_ptr < len(sorted_wins)
                   and sorted_wins[next_win_ptr][0] <= i):
                active.append(order[next_win_ptr])
                next_win_ptr += 1

            if not active:
                # We are before the first window — skip pixel decode entirely.
                continue

            # Decode pixels once; all active windows that cover frame i share
            # the same DecodedFrame object (images are never mutated downstream).
            pts = frame.pts if frame.pts is not None else (frame.dts or 0)
            decoded = DecodedFrame(
                frame_idx=i,
                pts=int(pts),
                time_sec=float(pts * time_base),
                image=frame.to_ndarray(format="bgr24"),
            )

            # Distribute to active windows; drop any whose end_frame == i
            # (just received their last frame).
            next_active: List[int] = []
            for orig_idx in active:
                if i <= windows[orig_idx][1]:
                    results[orig_idx].append(decoded)
                    if i < windows[orig_idx][1]:
                        # Window still has more frames coming.
                        next_active.append(orig_idx)
                    # i == end_frame: window complete, drop from active.
            active = next_active

    finally:
        container.close()

    return results


def collect_strided_shots(
    video_path: str,
    shots: List[Tuple[int, int]],
    stride: int,
) -> List[List[DecodedFrame]]:
    """Collect every stride-th frame from each shot in ONE sequential decode pass.

    The naive alternative — calling `decode_strided` once per shot — opens and
    scans from frame 0 for each shot, giving O(S × length) work. This function
    opens the video once and emits stride-sampled frames for every shot
    simultaneously.

    `shots`: (start_frame, end_frame) INCLUSIVE pairs, in any order.
    Returns a list in the SAME ORDER as `shots`; each entry contains the
    stride-sampled DecodedFrames for that shot.

    Used by Stage 3 (event detector) to scan all shots in one pass.
    """
    if not shots:
        return []

    order = sorted(range(len(shots)), key=lambda i: shots[i][0])
    sorted_shots = [shots[i] for i in order]
    global_end = max(s[1] for s in shots)

    results: List[List[DecodedFrame]] = [[] for _ in shots]

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        time_base = stream.time_base
        next_shot_ptr = 0
        active: List[int] = []  # original indices of active shots

        for i, frame in enumerate(container.decode(stream)):
            if i > global_end:
                break

            # Activate shots whose start_frame <= current frame.
            while (next_shot_ptr < len(sorted_shots)
                   and sorted_shots[next_shot_ptr][0] <= i):
                active.append(order[next_shot_ptr])
                next_shot_ptr += 1

            if not active:
                continue

            # Determine which active shots want a pixel-decoded frame at index i
            # (stride-aligned from each shot's own start_frame) and which ended.
            next_active: List[int] = []
            wanted_by: List[int] = []
            for orig_idx in active:
                s_start, s_end = shots[orig_idx]
                if i > s_end:
                    continue  # shot ended; drop from active
                next_active.append(orig_idx)
                # Anchor stride to each shot's start so the first sample is
                # always start_frame itself — consistent with decode_strided.
                if (i - s_start) % stride == 0:
                    wanted_by.append(orig_idx)
            active = next_active

            if not wanted_by:
                # No shot wants a pixel-decoded frame at this index; skip
                # the expensive to_ndarray() call.
                continue

            pts = frame.pts if frame.pts is not None else (frame.dts or 0)
            decoded = DecodedFrame(
                frame_idx=i,
                pts=int(pts),
                time_sec=float(pts * time_base),
                image=frame.to_ndarray(format="bgr24"),
            )
            for orig_idx in wanted_by:
                results[orig_idx].append(decoded)

    finally:
        container.close()

    return results


if __name__ == "__main__":
    # Standalone debugging entry point: print metadata + first few frames.
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m cuts.frame_extractor <video_path>")
        sys.exit(1)
    path = sys.argv[1]
    print(f"video: {path}")
    print(f"fps (average_rate): {get_fps(path):.4f}")
    print(f"container frame count: {get_frame_count(path)}")
    table = build_frame_index(path)
    print(f"true frame count (decoded): {len(table)}")
    if table:
        print(f"first frame: pts={table[0][0]}, time={table[0][1]:.4f}s")
        print(f"last  frame: pts={table[-1][0]}, time={table[-1][1]:.4f}s")
