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
* `Timeline` / `iter_timeline_frames` present *several* files as one continuous
  virtual timeline, which is how a session split across days is analyzed.

Why a virtual timeline rather than concatenating the files with ffmpeg: a
stream-copy concat only works when every input shares a codec, resolution and
pixel format, which recordings made on different days routinely do not. The
fallback is a full re-encode — hours of CPU and tens of gigabytes of scratch
disk for eight hours of capture, before any analysis has started. Worse, the
merged file forgets which recording a moment came from, and "file 2 at
14:22" is exactly what you need to actually go find the footage. Reading the
files in sequence costs nothing extra and keeps that mapping.

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


# ---------------------------------------------------------------------------
# Multi-file timelines
# ---------------------------------------------------------------------------

@dataclass
class Source:
    """One input file's placement on the global timeline.

    `time_offset` and `frame_offset` convert this file's local coordinates into
    global ones: ``global = local + offset``. Both are finalized by
    `iter_timeline_frames`, since the true extent of a file is only known once
    it has been read.

    `frame_span` is one past the last frame index this source contributed, so
    global frame ranges are ``[frame_offset, frame_offset + frame_span)`` and
    never overlap between sources.
    """

    path: str
    index: int
    duration_sec: float = 0.0
    time_offset: float = 0.0
    frame_offset: int = 0
    frame_span: int = 0

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def end_time(self) -> float:
        return self.time_offset + self.duration_sec

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "index": self.index,
            "duration_sec": self.duration_sec,
            "time_offset": self.time_offset,
            "frame_offset": self.frame_offset,
            "frame_span": self.frame_span,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "Source":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})  # type: ignore[arg-type]


@dataclass
class Timeline:
    """An ordered set of files presented as one continuous recording.

    A single file is just a timeline of length one, so the pipeline has exactly
    one code path rather than a special case for the common input.
    """

    sources: List[Source]

    @classmethod
    def build(cls, paths: Iterable[str]) -> "Timeline":
        """Probe each file's duration and lay them out end to end, in order.

        Order is taken as given — the caller is responsible for passing files
        in chronological order, since filesystem order is not reliably
        chronological and mtime is destroyed by copying.
        """
        sources: List[Source] = []
        cursor = 0.0
        for i, p in enumerate(paths):
            dur = get_duration_sec(p)
            sources.append(Source(
                path=os.path.abspath(p), index=i,
                duration_sec=dur, time_offset=cursor,
            ))
            cursor += dur
        if not sources:
            raise ValueError("a timeline needs at least one source file")
        return cls(sources=sources)

    def __len__(self) -> int:
        return len(self.sources)

    @property
    def is_multi(self) -> bool:
        return len(self.sources) > 1

    @property
    def duration_sec(self) -> float:
        return sum(s.duration_sec for s in self.sources)

    @property
    def first_path(self) -> str:
        return self.sources[0].path

    def source_for_frame(self, global_frame: int) -> Optional[Source]:
        for s in self.sources:
            if s.frame_offset <= global_frame < s.frame_offset + s.frame_span:
                return s
        return None

    def local_frame(self, global_frame: int) -> Optional[Tuple[Source, int]]:
        """Map a global frame index back to (source, local frame index)."""
        s = self.source_for_frame(global_frame)
        return None if s is None else (s, global_frame - s.frame_offset)

    def source_for_time(self, global_time: float) -> Source:
        """Map a global timestamp to the source that contains it.

        Clamps rather than failing: a timestamp exactly at the end of the last
        file is a legitimate event end time.
        """
        for s in self.sources:
            if s.time_offset <= global_time < s.end_time:
                return s
        return self.sources[-1] if global_time >= 0 else self.sources[0]

    def local_time(self, global_time: float) -> Tuple[Source, float]:
        """Map a global timestamp to (source, seconds into that file)."""
        s = self.source_for_time(global_time)
        return s, max(0.0, global_time - s.time_offset)

    def to_dicts(self) -> List[Dict[str, object]]:
        return [s.to_dict() for s in self.sources]

    @classmethod
    def from_dicts(cls, ds: Iterable[Dict[str, object]]) -> "Timeline":
        return cls(sources=[Source.from_dict(d) for d in ds])


def iter_timeline_frames(
    timeline: Timeline,
    interval_sec: float = 1.0,
) -> Iterator[Tuple[SampledFrame, Source]]:
    """Stream every source in order as one continuous sampled timeline.

    Yields ``(frame, source)`` where the frame's `frame_idx` and `time_sec`
    have already been shifted into global coordinates, so downstream code that
    only understands "one long video" needs no changes at all.

    Offsets are assigned *during* iteration rather than up front, from each
    file's last observed sample time. Container-reported duration is used as a
    floor, not as the truth: some screen recorders under-report it, and taking
    it at face value would make later files overlap earlier ones and produce
    non-monotonic timestamps.
    """
    frame_cursor = 0
    time_cursor = 0.0

    for src in timeline.sources:
        src.frame_offset = frame_cursor
        src.time_offset = time_cursor

        last_local_frame = -1
        last_local_time = 0.0

        for sf in iter_sampled_frames(src.path, interval_sec=interval_sec):
            last_local_frame = sf.frame_idx
            last_local_time = sf.time_sec
            yield (
                SampledFrame(
                    frame_idx=src.frame_offset + sf.frame_idx,
                    pts=sf.pts,
                    time_sec=src.time_offset + sf.time_sec,
                    image=sf.image,
                ),
                src,
            )

        src.frame_span = last_local_frame + 1
        # The file extends at least to its last sampled frame; keep whichever
        # of that and the container's claim is larger.
        src.duration_sec = max(src.duration_sec, last_local_time)
        frame_cursor = src.frame_offset + src.frame_span
        time_cursor = src.time_offset + src.duration_sec


def probe_video(video_path: str) -> Dict[str, object]:
    """Cheap metadata read: duration, fps, resolution, codec, audio presence.

    No decoding, so this stays instant even on a multi-hour file. Used by the
    `probe` command to tell you what a run is going to cost before you start
    one.
    """
    info: Dict[str, object] = {
        "path": video_path,
        "duration_sec": 0.0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "codec": "?",
        "has_audio": False,
        "error": None,
    }
    try:
        container = av.open(video_path)
    except Exception as exc:
        info["error"] = str(exc)
        return info
    try:
        if container.duration is not None:
            info["duration_sec"] = float(container.duration / av.time_base)
        info["has_audio"] = len(container.streams.audio) > 0
        if container.streams.video:
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.base_rate
            info["fps"] = float(rate) if rate else 0.0
            info["width"] = int(stream.codec_context.width or 0)
            info["height"] = int(stream.codec_context.height or 0)
            info["codec"] = stream.codec_context.name or "?"
            if not info["duration_sec"] and stream.duration and stream.time_base:
                info["duration_sec"] = float(stream.duration * stream.time_base)
    finally:
        container.close()
    return info


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
    # Standalone debug: print metadata and walk the timeline.
    import sys
    import time

    if len(sys.argv) < 2:
        print("usage: python -m cuts.media <video_path> [more_videos...]")
        sys.exit(1)
    paths = sys.argv[1:]

    for p in paths:
        meta = probe_video(p)
        print(f"{os.path.basename(p)}: {meta['duration_sec']:.1f}s "
              f"{meta['width']}x{meta['height']} @{meta['fps']:.2f}fps "
              f"{meta['codec']} audio={meta['has_audio']}")

    tl = Timeline.build(paths)
    print(f"\ntimeline: {len(tl)} source(s), {tl.duration_sec:.1f}s total")

    t0 = time.time()
    n = 0
    last_t = 0.0
    for sf, src in iter_timeline_frames(tl, interval_sec=1.0):
        if n < 3:
            print(f"  sample {n}: global_frame={sf.frame_idx} "
                  f"t={sf.time_sec:.3f}s from {src.name} shape={sf.image.shape}")
        last_t = sf.time_sec
        n += 1
    print(f"sampled {n} frames (last t={last_t:.2f}s) in {time.time() - t0:.2f}s")
    for s in tl.sources:
        print(f"  {s.name}: t+{s.time_offset:.1f}s frames "
              f"[{s.frame_offset}, {s.frame_offset + s.frame_span})")
