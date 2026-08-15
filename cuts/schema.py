"""On-disk schema for semantic video indexes.

An `Event` is the atomic unit produced by segmentation: a contiguous span of
video that is internally homogeneous in what the user was doing. A `Chapter`
is an `Event` that has been given a human-readable title.

Frame indices AND time-seconds are both stored because:
  - frame indices are the canonical identifier (VFR-safe, decode-order),
  - time-seconds are what a chapter list actually displays and what transcript
    overlap is computed against.

A recording may be split across several files (one per day, say). Those are
analyzed as a single continuous timeline, so `start_time` is always global —
seconds from the start of the *first* file. Each event additionally records
which file it came from and its offset within that file, because that is what
you need in order to actually open the footage in an editor.

JSON layout (one file per timeline, not per video file):

    {
      "video_id":   "session-04",
      "video_path": "/abs/path/session-04.mp4",   # first source
      "duration_sec": 2712.4,
      "sources": [ {path, index, duration_sec, time_offset,
                    frame_offset, frame_span}, ... ],
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
    source_path, source_index, source_start_time:
        Which input file this event opens in, and how far into that file it
        starts. For a single-file input these are just the video itself and
        `source_start_time == start_time`. For a multi-file timeline they are
        how you find the footage: "day-2.mp4 at 14:22" rather than a global
        offset you would have to work backwards from.
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
    source_path: str = ""
    source_index: int = 0
    source_start_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def source_name(self) -> str:
        """Basename of the file this event opens in ("" if unknown)."""
        return os.path.basename(self.source_path) if self.source_path else ""

    def source_timestamp(self) -> str:
        """Start time *within its own source file*, as ``M:SS``/``H:MM:SS``."""
        return format_timestamp(self.source_start_time)

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
    sources: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Atomically write the events JSON file for one timeline.

    Writes to a sibling temp file then renames, so a partial write never
    clobbers a previously good index.

    `sources` is the timeline layout. It is what lets `cuts label` re-open the
    right file for each event later without re-running the analysis pass.
    """
    payload: Dict[str, Any] = {
        "video_id": video_id,
        "video_path": video_path,
        "duration_sec": duration_sec,
        "sources": sources or [{
            "path": video_path, "index": 0, "duration_sec": duration_sec,
            "time_offset": 0.0, "frame_offset": 0, "frame_span": 0,
        }],
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
    """Load an events JSON; returns the full payload, not just the list.

    Indexes written before multi-file support have no "sources" key; one is
    synthesized from `video_path` so callers never have to special-case it.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["events"] = [Event.from_dict(e) for e in payload.get("events", [])]
    if not payload.get("sources"):
        payload["sources"] = [{
            "path": payload.get("video_path", ""),
            "index": 0,
            "duration_sec": payload.get("duration_sec", 0.0),
            "time_offset": 0.0,
            "frame_offset": 0,
            "frame_span": 0,
        }]
    return payload


def to_chapter_list(events: List[Event], show_source: bool = False) -> str:
    """Render events as a plain chapter list.

        0:00 writing parser
        7:32 compiling

    This is the product's primary output format — it pastes directly into a
    YouTube description or an editor's marker track.

    With `show_source`, each line also carries the file and in-file offset:

        1:04:11 debugging segfault      [day-2.mp4 @ 4:11]

    which is what you need when the timeline spans several recordings and the
    global timestamp does not correspond to anything you can scrub to.
    """
    lines = []
    width = max((len(e.title or "(unlabeled)") for e in events), default=0)
    for e in events:
        title = e.title or "(unlabeled)"
        line = f"{format_timestamp(e.start_time)} {title}"
        if show_source and e.source_path:
            line = (f"{format_timestamp(e.start_time)} {title:<{width}}"
                    f"   [{e.source_name} @ {e.source_timestamp()}]")
        lines.append(line)
    return "\n".join(lines)
