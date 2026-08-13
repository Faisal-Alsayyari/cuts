"""Hybrid shot + time-cap segmenter.

TransNetV2 gives us good structural boundaries (shots). For devlog / screen
recording footage a single shot can be minutes long, which makes it useless
as a retrievable unit. Conversely, very short shots (transitions, flashes)
are not meaningful events.

This module turns a list of shots into a list of `SegmentRecord`s with:
  * long shots split into equal sub-segments of <= ``max_segment_sec``,
  * runs of short shots (< ``min_segment_sec``) merged into the next segment.

The segmenter is pure (no video decoding) and trivially unit-testable.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from ..config import RetrievalConfig
from .schema import SegmentRecord


def build_segments(
    video_id: str,
    shots: List[Tuple[int, int]],
    frame_index: List[Tuple[int, float]],
    config: RetrievalConfig,
) -> List[SegmentRecord]:
    """Turn TransNetV2 shots into indexable SegmentRecords.

    Parameters
    ----------
    video_id:
        Stable id (typically the file stem).
    shots:
        List of ``(start_frame, end_frame)`` INCLUSIVE pairs, already sorted
        in time. This is the exact shape produced by ``pipeline._boundaries_to_shots``
        and by the TransNetV2 detector path.
    frame_index:
        The ``frame_idx -> (pts, time_sec)`` table from
        ``frame_extractor.build_frame_index``. Used to compute wall-clock
        times without ever touching ``frame_idx / fps``.
    config:
        The retrieval sub-config; see ``RetrievalConfig`` for the knobs.

    Returns
    -------
    A list of SegmentRecord with ``source`` ∈ {"shot", "shot_split",
    "shot_merged"} and empty OCR/ASR/representative_frames fields (those are
    populated by later stages).
    """
    if not shots:
        return []
    if not frame_index:
        raise ValueError("frame_index is empty; did build_frame_index run?")

    n_frames = len(frame_index)

    def t(frame: int) -> float:
        # Clamp to the table; TransNetV2 occasionally emits an end_frame
        # equal to n_frames (one past the last decoded frame).
        f = max(0, min(frame, n_frames - 1))
        return frame_index[f][1]

    # ------------------------------------------------------------------
    # Pass 1: merge very short shots forward into the next shot.
    # ------------------------------------------------------------------
    # We operate on (start_frame, end_frame, source, shot_ids) tuples.
    merged: List[dict] = []
    pending_ids: List[int] = []
    pending_start: int = -1

    for shot_id, (s, e) in enumerate(shots):
        dur = t(e) - t(s)
        if pending_ids:
            # Extend the accumulator regardless; we'll decide to close on duration.
            total_dur = t(e) - t(pending_start)
            pending_ids.append(shot_id)
            if total_dur >= config.min_segment_sec:
                merged.append({
                    "start": pending_start,
                    "end": e,
                    "shot_ids": list(pending_ids),
                    "source": "shot_merged" if len(pending_ids) > 1 else "shot",
                })
                pending_ids = []
                pending_start = -1
            continue

        if dur < config.min_segment_sec:
            # Start (or continue) a merge run.
            pending_start = s
            pending_ids = [shot_id]
        else:
            merged.append({
                "start": s,
                "end": e,
                "shot_ids": [shot_id],
                "source": "shot",
            })

    # Trailing short-shot tail: attach to the previous segment if one exists,
    # otherwise keep as its own (short) segment so we never drop content.
    if pending_ids:
        s = pending_start
        e = shots[pending_ids[-1]][1]
        if merged:
            prev = merged[-1]
            prev["end"] = e
            prev["shot_ids"].extend(pending_ids)
            prev["source"] = "shot_merged"
        else:
            merged.append({
                "start": s,
                "end": e,
                "shot_ids": list(pending_ids),
                "source": "shot_merged" if len(pending_ids) > 1 else "shot",
            })

    # ------------------------------------------------------------------
    # Pass 2: split any segment longer than max_segment_sec into equal parts.
    # ------------------------------------------------------------------
    final: List[SegmentRecord] = []
    seg_counter = 0

    for seg in merged:
        s, e = seg["start"], seg["end"]
        dur = t(e) - t(s)
        max_dur = config.max_segment_sec

        if dur <= max_dur or e <= s:
            final.append(_make_record(
                video_id, seg_counter, s, e, seg["source"], seg["shot_ids"],
                frame_index,
            ))
            seg_counter += 1
            continue

        # Split into ceil(dur / max_dur) roughly-equal sub-segments by FRAME
        # index (good enough; frames per second is ~constant within a shot).
        n_splits = int(math.ceil(dur / max_dur))
        # Use frame positions so sub-segments are contiguous and cover [s, e].
        total_frames = e - s + 1
        per = total_frames // n_splits
        remainder = total_frames - per * n_splits

        cursor = s
        for k in range(n_splits):
            size = per + (1 if k < remainder else 0)
            sub_s = cursor
            sub_e = cursor + size - 1
            if k == n_splits - 1:
                sub_e = e  # absorb any rounding
            final.append(_make_record(
                video_id, seg_counter, sub_s, sub_e,
                "shot_split", seg["shot_ids"],
                frame_index,
            ))
            seg_counter += 1
            cursor = sub_e + 1

    return final


def _make_record(
    video_id: str,
    idx: int,
    start_frame: int,
    end_frame: int,
    source: str,
    shot_ids: List[int],
    frame_index: List[Tuple[int, float]],
) -> SegmentRecord:
    n = len(frame_index)
    sf = max(0, min(start_frame, n - 1))
    ef = max(0, min(end_frame, n - 1))
    return SegmentRecord(
        video_id=video_id,
        segment_id=f"{idx:06d}",
        start_frame=sf,
        end_frame=ef,
        start_time=frame_index[sf][1],
        end_time=frame_index[ef][1],
        source=source,
        metadata={
            "shot_id": shot_ids[0] if len(shot_ids) == 1 else shot_ids,
            "shot_ids": shot_ids,
            "confidence": None,
            "frame_indices": [],
            "frame_times": [],
        },
    )
