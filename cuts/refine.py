"""Single-bisection boundary refinement.

Coarse sampling means every detected boundary carries roughly half a sampling
interval of timestamp uncertainty: the true transition happened somewhere
between the last sample before it and the first sample after, and the coarse
pass never looked in between. At the default 30s interval that is up to 15s
of slop on every chapter start.

This pass closes that gap by exactly one level, not recursively: for each
detected (non-forced) boundary, decode the single frame at the midpoint
between the two straddling samples, describe it with the same VLM used for
Pass 1, and ask a small classification call — given the descriptions on both
sides plus the new midpoint description — which half of the gap the
transition actually falls in. The boundary is moved to the midpoint only when
the transition is confidently placed in the first half; otherwise the coarse
timestamp already sits as close to the transition as one extra sample can get
it, so it is left alone.

File joins are never refined: they are exact discontinuities with no gap to
search inside.

The pass runs in three phases, so the API stages are concurrent across
boundaries instead of serialized per boundary:

  1. gather  — decode + OCR/redact + encode each midpoint frame (local)
  2. describe — every midpoint, one frame per call, concurrently. One per call
     because the description prompt tells the model its frames are
     consecutive, and midpoints of different boundaries are not.
  3. decide  — one concurrent pass of per-boundary classification calls
"""

from __future__ import annotations

import dataclasses
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Set, Tuple

from .config import CutsConfig
from .media import Timeline, decode_frame_near, resize_long_edge
from .observe import Observation
from .schema import Event


def refine_boundaries(
    events: List[Event],
    observations: Sequence[Observation],
    timeline: Timeline,
    config: CutsConfig,
    locked: Optional[Set[int]] = None,
    verbose: bool = False,
) -> int:
    """Move each refinable boundary toward its midpoint, in place.

    `observations` are the ones the events were built from; `locked` holds
    positions in them that must not move (file joins). Mutates `start_time` /
    `end_time` on adjacent events and returns how many boundaries moved.
    """
    if len(events) < 2:
        return 0

    from .signals.vlm import decide_refinement, describe_batch, encode_frame
    scrub = None
    if config.ocr.enabled:
        from .redact import scrub_frame as scrub

    locked = locked or set()
    times = [o.t for o in observations]
    # Boundaries are located by time, which is what both sides agree on.
    pos_of_time = {round(t, 3): i for i, t in enumerate(times)}

    # ---- Phase 1: gather midpoint frames (local, sequential) --------------
    candidates: List[Tuple[int, int, float]] = []   # (event, position, t_mid)
    encoded: List[str] = []
    ocr_texts: List[str] = []
    for i in range(1, len(events)):
        b = pos_of_time.get(round(events[i].start_time, 3))
        if b is None or b == 0 or b in locked:
            continue
        t_mid = (times[b - 1] + times[b]) / 2.0
        src, local_t = timeline.local_time(t_mid)
        if not os.path.exists(src.path):
            continue
        try:
            frame = decode_frame_near(src.path, local_t)
        except Exception as exc:
            if verbose:
                print(f"  [refine] t={t_mid:.0f}s: decode failed ({exc!r})")
            continue
        if frame is None:
            continue

        # A midpoint frame never went through Pass 1's scrub, so it gets its
        # own before it is encoded for upload.
        ocr_text, safe = "", True
        if scrub is not None:
            ocr_text, safe, hits = scrub(
                frame.image, config.ocr, config.redact, verbose=verbose
            )
            if hits and verbose:
                print(f"  [redact] refine t={t_mid:.0f}s: "
                      f"{', '.join(sorted(set(hits)))}")
        if not safe:
            # Keep the coarse boundary rather than upload the frame.
            continue

        candidates.append((i, b, t_mid))
        ocr_texts.append(ocr_text)
        encoded.append(encode_frame(
            resize_long_edge(frame.image, config.vlm.frame_long_edge)
        ))

    if not candidates:
        if verbose:
            print("  [refine] no refinable boundaries")
        return 0

    # ---- Phase 2: describe every midpoint (one frame per call) ------------
    mid_descriptions = describe_batch(
        encoded, dataclasses.replace(config.vlm, frames_per_call=1),
        ocr_texts=ocr_texts, verbose=verbose,
    )

    # ---- Phase 3: classify each gap (concurrent) --------------------------
    model = config.refine.model or config.vlm.model

    def _decide(arg) -> str:
        (_i, b, _t_mid), desc_mid = arg
        if not desc_mid:
            return "unclear"
        return decide_refinement(
            observations[b - 1].description, desc_mid,
            observations[b].description,
            model, config.refine.max_tokens, verbose=verbose,
        )

    with ThreadPoolExecutor(max_workers=min(len(candidates), 16)) as pool:
        transitions = list(pool.map(_decide, zip(candidates, mid_descriptions)))

    moved = 0
    for (i, _b, t_mid), transition in zip(candidates, transitions):
        if transition == "first_half":
            events[i - 1].end_time = events[i].start_time = t_mid
            moved += 1

    if verbose:
        print(f"  [refine] moved {moved}/{len(candidates)} boundaries toward "
              f"their midpoint")
    return moved
