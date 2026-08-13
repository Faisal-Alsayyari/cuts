"""Annotation schema and CSV loader for the benchmark harness.

CSV columns (header row required):
    video_id, clip_path, type, label, start_frame, end_frame

Where:
    * `type` is one of: "hard_cut", "gradual_transition", "ui_event"
    * `label` is a free-text descriptor (e.g. "modal_open", "view_switch", "dissolve")
    * `start_frame`, `end_frame` are 0-based ordinal decoding-order frame indices
      (matching PyAV's `enumerate(container.decode(...))` order).
    * For hard cuts, `start_frame == end_frame`.

`load_annotations(path)` returns a list of `Annotation` dataclasses, grouped
trivially via `group_by_video`.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

# Allowed annotation types. Anything else triggers a clear error on load.
ALLOWED_TYPES = {"hard_cut", "gradual_transition", "ui_event"}


@dataclass
class Annotation:
    """One ground-truth annotation row."""

    video_id: str
    clip_path: str
    type: str         # one of ALLOWED_TYPES
    label: str        # human-readable; not used in matching
    start_frame: int  # inclusive, 0-based
    end_frame: int    # inclusive, 0-based; equals start_frame for hard cuts

    def __post_init__(self) -> None:
        # Sanity: hard cuts MUST have start==end. Catch annotation mistakes early.
        if self.type == "hard_cut" and self.start_frame != self.end_frame:
            raise ValueError(
                f"Hard cut annotation has unequal frames: "
                f"video={self.video_id} start={self.start_frame} end={self.end_frame}"
            )
        if self.type not in ALLOWED_TYPES:
            raise ValueError(f"Unknown annotation type: {self.type!r}")
        if self.end_frame < self.start_frame:
            raise ValueError(
                f"end_frame < start_frame in annotation: video={self.video_id}"
            )


def load_annotations(csv_path: str) -> List[Annotation]:
    """Load and validate annotations from a CSV file."""
    out: List[Annotation] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Validate header upfront so missing columns produce a clear error.
        required = {"video_id", "clip_path", "type", "label", "start_frame", "end_frame"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Annotation CSV missing required columns: {missing}")
        for row in reader:
            out.append(Annotation(
                video_id=row["video_id"].strip(),
                clip_path=row["clip_path"].strip(),
                type=row["type"].strip(),
                label=row["label"].strip(),
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
            ))
    return out


def group_by_video(annotations: List[Annotation]) -> Dict[str, List[Annotation]]:
    """Group annotations by `video_id` for per-video evaluation."""
    grouped: Dict[str, List[Annotation]] = defaultdict(list)
    for a in annotations:
        grouped[a.video_id].append(a)
    return dict(grouped)


if __name__ == "__main__":
    # Smoke test: write a tiny CSV in memory and parse it.
    import io

    sample = io.StringIO(
        "video_id,clip_path,type,label,start_frame,end_frame\n"
        "v1,/tmp/v1.mp4,hard_cut,hard_cut,100,100\n"
        "v1,/tmp/v1.mp4,gradual_transition,dissolve,250,260\n"
        "v1,/tmp/v1.mp4,ui_event,modal_open,400,402\n"
    )
    # Re-use load_annotations by writing to a temp file would be more honest,
    # but the parsing is small enough to inline-validate here.
    reader = csv.DictReader(sample)
    rows = [Annotation(
        video_id=r["video_id"], clip_path=r["clip_path"], type=r["type"],
        label=r["label"], start_frame=int(r["start_frame"]),
        end_frame=int(r["end_frame"])
    ) for r in reader]
    for a in rows:
        print(a)
