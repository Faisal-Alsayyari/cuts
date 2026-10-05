"""Small, GUI-side media reads: file metadata and single thumbnails.

These use PyAV directly rather than going through the backend, because they
are presentation concerns — "how long is this file", "what does 12:30 look
like" — and asking a subprocess to answer them would cost a Python start-up
per thumbnail. Nothing here is analysis, and nothing here imports `cuts`.

PyAV is optional at this layer: without it, the add-files dialog shows no
durations and the timeline shows no filmstrip, and everything else works.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import av  # type: ignore
except ImportError:  # pragma: no cover - exercised only without PyAV
    av = None


def available() -> bool:
    return av is not None


@dataclass
class FileInfo:
    path: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = False
    error: str = ""

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}" if self.width else ""


def probe(path: str) -> FileInfo:
    """Container metadata only — no decoding, so instant on a multi-hour file."""
    info = FileInfo(path=path)
    if av is None:
        info.error = "PyAV not installed"
        return info
    try:
        container = av.open(path)
    except Exception as exc:  # noqa: BLE001 - surface anything as a message
        info.error = str(exc) or type(exc).__name__
        return info
    try:
        if container.duration is not None:
            info.duration = float(container.duration / av.time_base)
        info.has_audio = len(container.streams.audio) > 0
        if container.streams.video:
            st = container.streams.video[0]
            rate = st.average_rate or st.base_rate
            info.fps = float(rate) if rate else 0.0
            info.width = int(st.codec_context.width or 0)
            info.height = int(st.codec_context.height or 0)
            if not info.duration and st.duration and st.time_base:
                info.duration = float(st.duration * st.time_base)
        else:
            info.error = "no video stream"
    finally:
        container.close()
    return info


def grab_frame(path: str, t: float, width: int, max_frames: int = 900):
    """The frame at `t` seconds into `path`, scaled to `width`, as RGB ndarray.

    Seeks to the keyframe before `t` and decodes forward to it. Returns None
    when the file cannot be opened or nothing decodes.
    """
    if av is None or not os.path.exists(path):
        return None
    try:
        container = av.open(path)
    except Exception:  # noqa: BLE001
        return None
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = stream.time_base
        try:
            container.seek(int(max(0.0, t) / tb), stream=stream,
                           backward=True, any_frame=False)
        except Exception:  # noqa: BLE001
            pass
        last = None
        for i, frame in enumerate(container.decode(stream)):
            last = frame
            pts = frame.pts if frame.pts is not None else (frame.dts or 0)
            if float(pts * tb) + 1e-6 >= t or i >= max_frames:
                break
        if last is None:
            return None
        w = max(16, int(width))
        h = max(2, int(round(last.height * w / max(1, last.width))) // 2 * 2)
        import numpy as np

        # Rows come back padded to the decoder's alignment; QImage wants them
        # packed.
        return np.ascontiguousarray(
            last.reformat(width=w, height=h, format="rgb24").to_ndarray())
    except Exception:  # noqa: BLE001
        return None
    finally:
        container.close()
