"""On-disk schema for the retrieval milestone.

A `SegmentRecord` is the atomic unit of the searchable index. Each record is
a timestamped span of video plus every signal we extracted about it: OCR text
from representative frames, ASR transcript overlapping its time window, and
paths to the representative frame thumbnails on disk.

Frame indices AND time-seconds are both stored because:
  - frame indices are the canonical identifier (VFR-safe, matches TransNetV2),
  - time-seconds are what the CTO demo actually shows and what we index
    transcript overlaps against.

JSON layout (one file per video) is a top-level `{"video_id", "video_path",
"segments": [SegmentRecord, ...]}`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SegmentRecord:
    """One indexable segment of video.

    Attributes
    ----------
    video_id:
        Stable identifier for the source video (typically the file stem).
        Used to namespace on-disk artifacts and join results across queries.
    segment_id:
        Opaque identifier unique within the video. Currently a zero-padded
        index (``"000042"``) but callers must not depend on the format.
    start_frame, end_frame:
        Inclusive ordinal frame indices (same convention as
        `DecodedFrame.frame_idx`).
    start_time, end_time:
        Seconds from the start of the video, computed from
        ``pts * time_base`` — NEVER from ``frame_idx / fps``.
    source:
        Why this segment exists. One of:
          * ``"shot"``             — one TransNetV2 shot small enough to keep whole
          * ``"shot_split"``       — sub-segment of a long shot split on time
          * ``"shot_merged"``      — several short shots merged into one
        Other values are reserved for future OCR-change / transcript-based
        segmenters.
    representative_frames:
        Paths (POSIX-style, relative to the index directory) of the JPEGs
        written by the sampler. Order matches ``metadata["frame_indices"]``.
    ocr_text:
        Concatenated, deduplicated OCR text across representative frames for
        this segment. Empty string when OCR has not been run or produced
        nothing.
    transcript_text:
        Concatenated ASR transcript aligned to this segment's time window.
        Empty string when ASR is disabled or no audio stream was present.
    metadata:
        Free-form bag of non-indexed attributes. Stable keys:
          * ``shot_id``: index of the originating TransNetV2 shot
          * ``frame_indices``: list of sampled frame indices, 1:1 with
            ``representative_frames``
          * ``frame_times``: list of time-seconds, 1:1 with the above
          * ``confidence``: reserved, currently always ``None``
          * ``ocr_stale``: ``True`` when OCR was reused from an adjacent
            segment via perceptual-hash dedup (rather than recomputed)
    """

    video_id: str
    segment_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    source: str
    representative_frames: List[str] = field(default_factory=list)
    ocr_text: str = ""
    ocr_text_raw: str = ""
    transcript_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- convenience --------------------------------------------------------

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def combined_text(self) -> str:
        """Text used for keyword / embedding indexing.

        OCR and transcript are concatenated with a double-space separator so
        tokenizers see them as distinct fields without inventing a new token.
        """
        if self.ocr_text and self.transcript_text:
            return f"{self.ocr_text}  {self.transcript_text}"
        return self.ocr_text or self.transcript_text

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SegmentRecord":
        # Be tolerant of extra keys written by future versions.
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Top-level index file (segments.json) IO
# ---------------------------------------------------------------------------

def write_segments(
    path: str,
    video_id: str,
    video_path: str,
    segments: List[SegmentRecord],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically write the per-video segments JSON file."""
    import os
    import tempfile

    payload: Dict[str, Any] = {
        "video_id": video_id,
        "video_path": video_path,
        "segments": [s.to_dict() for s in segments],
    }
    if extra:
        payload["extra"] = extra

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Write to a sibling temp file then rename so partial writes never clobber
    # a previous good index.
    fd, tmp = tempfile.mkstemp(prefix=".segments_", suffix=".json",
                               dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_segments(path: str) -> Dict[str, Any]:
    """Load a segments.json; returns the full dict, not just the list."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["segments"] = [SegmentRecord.from_dict(s) for s in payload["segments"]]
    return payload
