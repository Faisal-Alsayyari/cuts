"""On-disk schema for semantic video indexes.

An `Event` is the atomic unit produced by segmentation: a contiguous span of
video that is internally homogeneous in what the user was doing. A `Chapter`
is an `Event` that has been given a human-readable title.

Frame indices AND time-seconds are both stored because:
  - frame indices are the canonical identifier (VFR-safe, decode-order),
  - time-seconds are what a chapter list actually displays and what transcript
    overlap is computed against.

JSON layout (one file per video):

    {
      "video_id":   "session-04",
      "video_path": "/abs/path/session-04.mp4",
      "duration_sec": 2712.4,
      "events": [Event, ...]
    }
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Event:
    """One semantically homogeneous span of video.

    Attributes
    ----------
    event_id:
        Zero-padded index, unique within the video. Callers must not depend
        on the format.
    start_frame, end_frame:
        Inclusive ordinal frame indices (decode order, matching
        `media.SampledFrame.frame_idx`).
    start_time, end_time:
        Seconds from video start, computed from ``pts * time_base`` — never
        from ``frame_idx / fps``.
    title:
        Human-readable chapter label ("debugging segfault"). Empty until
        `cuts.labeling` has run.
    summary:
        Optional longer description, when the labeler produced one.
    ocr_text:
        Deduplicated on-screen text observed across this event's samples.
    transcript_text:
        Speech transcribed within this event's time window. Empty when the
        source had no audio or ASR was disabled.
    sample_indices:
        Frame indices of the sampled frames that fall inside this event.
    representative_frames:
        Frame indices chosen to stand in for the event (for thumbnails and
        for vision-based labeling).
    metadata:
        Free-form bag. Stable keys:
          * ``boundary_score``: fused similarity at the opening boundary
            (lower = sharper transition into this event)
          * ``merged_from``: number of initial partitions merged into this one
    """

    video_id: str
    event_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    title: str = ""
    summary: str = ""
    ocr_text: str = ""
    transcript_text: str = ""
    sample_indices: List[int] = field(default_factory=list)
    representative_frames: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def combined_text(self) -> str:
        """Text used for keyword / embedding indexing.

        OCR and transcript are joined with a double space so tokenizers see
        them as distinct fields without inventing a separator token.
        """
        if self.ocr_text and self.transcript_text:
            return f"{self.ocr_text}  {self.transcript_text}"
        return self.ocr_text or self.transcript_text

    def timestamp(self) -> str:
        """Start time as ``M:SS`` (or ``H:MM:SS`` past an hour)."""
        return format_timestamp(self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        # Tolerate extra keys written by future versions.
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``M:SS``, or ``H:MM:SS`` once past an hour.

    This is the format chapter lists are conventionally written in.
    """
    total = int(max(0.0, seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Index file IO
# ---------------------------------------------------------------------------

def write_events(
    path: str,
    video_id: str,
    video_path: str,
    events: List[Event],
    duration_sec: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically write the per-video events JSON file.

    Writes to a sibling temp file then renames, so a partial write never
    clobbers a previously good index.
    """
    payload: Dict[str, Any] = {
        "video_id": video_id,
        "video_path": video_path,
        "duration_sec": duration_sec,
        "events": [e.to_dict() for e in events],
    }
    if extra:
        payload["extra"] = extra

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".events_", suffix=".json",
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


def read_events(path: str) -> Dict[str, Any]:
    """Load an events JSON; returns the full payload, not just the list."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["events"] = [Event.from_dict(e) for e in payload.get("events", [])]
    return payload


def to_chapter_list(events: List[Event]) -> str:
    """Render events as a plain chapter list.

        0:00 writing parser
        7:32 compiling

    This is the product's primary output format — it pastes directly into a
    YouTube description or an editor's marker track.
    """
    lines = []
    for e in events:
        title = e.title or "(unlabeled)"
        lines.append(f"{format_timestamp(e.start_time)} {title}")
    return "\n".join(lines)
