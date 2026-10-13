"""Turning boundary decisions into events — shared by both segmenters.

Both segmenters answer the same question in the same shape: *at which
observations does a new chapter begin?* `cuts.coalesce` gets the answer from a
model, `cuts.segmentation` from a similarity curve. Everything after that —
enforcing the minimum length without breaking file joins, materializing
gapless events with their evidence, and recording where each one sits in its
source file — is identical, and lives here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .media import Timeline
from .observe import WITHHELD, Observation
from .schema import Event


def join_positions(observations: Sequence[Observation]) -> List[int]:
    """Positions where a new source file begins (excluding the first file).

    A new file means the recording stopped and restarted — a real
    discontinuity however similar the pixels look across it — so these are
    passed to both segmenters as boundaries that must survive.
    """
    out: List[int] = []
    for i in range(1, len(observations)):
        if observations[i].source_index != observations[i - 1].source_index:
            out.append(i)
    return out


def enforce_min_duration(
    bounds: Sequence[int],
    times: Sequence[float],
    min_sec: float,
    total_end: float,
    locked: Optional[Sequence[int]] = None,
) -> List[int]:
    """Drop boundaries that would create an event shorter than `min_sec`.

    `bounds` are the start positions of every event *after the first*. Walks
    left to right and removes any boundary too close to the previously kept
    one, then drops the last boundary if the final event is too short.

    `locked` boundaries always survive. A file that ends two seconds after
    its last activity change still ends there; suppressing that join to
    satisfy a minimum length would silently stitch two recordings into one
    chapter.
    """
    locked_set = set(locked or ())
    bounds = sorted(bounds)
    if min_sec <= 0 or not bounds:
        return bounds
    kept: List[int] = []
    last_start_t = times[0] if times else 0.0
    for b in bounds:
        t = times[b] if b < len(times) else total_end
        if b in locked_set or t - last_start_t >= min_sec:
            kept.append(b)
            last_start_t = t
    if kept and kept[-1] not in locked_set:
        last_t = times[kept[-1]] if kept[-1] < len(times) else total_end
        if total_end - last_t < min_sec:
            kept.pop()
    return kept


def dedup(texts) -> List[str]:
    """Distinct non-empty strings in first-seen order (case-insensitive).

    A static screen repeats the same description verbatim across samples,
    so the raw concatenation would be one sentence copy-pasted a dozen times.
    """
    seen, out = set(), []
    for t in texts:
        t = (t or "").strip()
        key = t.lower()
        if t and key not in seen:
            seen.add(key)
            out.append(t)
    return out


def pick_representative_times(
    members: Sequence[Observation], n: int = 3
) -> List[float]:
    """Up to `n` timestamps spread across the chapter, avoiding its endpoints.

    The first and last samples sit next to a boundary, where the screen is
    most likely mid-transition. Withheld frames are skipped — there is no
    point choosing one we already declined to upload.
    """
    usable = [o for o in members if o.status != WITHHELD]
    if not usable or n <= 0:
        return []
    if len(usable) <= n:
        return [o.t for o in usable]
    step = len(usable) / (n + 1)
    return [usable[min(len(usable) - 1, int(step * (k + 1)))].t for k in range(n)]


def events_from_starts(
    starts: Sequence[int],
    observations: Sequence[Observation],
    video_id: str,
    duration_sec: float,
    info: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Event]:
    """Materialize gapless events from the positions where chapters begin.

    `starts` must begin with 0 and be strictly increasing. Chapter *k* runs
    from its first observation's time to the next chapter's first
    observation, the last one to `duration_sec`. `info[k]`, when given,
    supplies `title`, `reason` and extra `metadata` for chapter *k*.
    """
    n = len(observations)
    ends = list(starts[1:]) + [n]
    events: List[Event] = []
    for k, (s, e) in enumerate(zip(starts, ends)):
        members = observations[s:e]
        meta_in = (info[k] if info and k < len(info) else {}) or {}
        start_time = members[0].t
        end_time = observations[e].t if e < n else duration_sec
        events.append(Event(
            video_id=video_id,
            event_id=f"{k:04d}",
            start_time=float(start_time),
            end_time=float(max(end_time, start_time)),
            title=str(meta_in.get("title", "") or "").strip(),
            reason=str(meta_in.get("reason", "") or "").strip(),
            vlm_text="\n".join(dedup(o.description for o in members)),
            transcript_text=" ".join(dedup(o.transcript_text for o in members)),
            sample_times=[o.t for o in members],
            representative_times=pick_representative_times(members, 3),
            metadata={
                **(meta_in.get("metadata") or {}),
                "n_samples": len(members),
                "n_withheld": sum(1 for o in members if o.status == WITHHELD),
            },
        ))
    return events


def attach_sources(events: List[Event], timeline: Timeline) -> None:
    """Record which file each event opens in, and where inside it.

    Run last: refinement can move an event's start, which is what this is
    computed from. A global timestamp is not something you can scrub to when
    the recording lives in five separate files.
    """
    for e in events:
        src, local = timeline.local_time(e.start_time)
        e.source_path = src.path
        e.source_index = src.index
        e.source_start_time = local
