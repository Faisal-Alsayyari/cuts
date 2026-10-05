"""Single-bisection boundary refinement.

Coarse sampling means every detected boundary carries roughly half a sampling
interval of timestamp uncertainty: the true transition happened somewhere
between the last sample before it and the first sample after, and the coarse
pass never looked in between. At the default 30s interval that is up to 15s
of slop on every chapter start.

This pass closes that gap by exactly one level, not recursively: for each
detected (non-forced) boundary, decode the single frame at the midpoint
between the two straddling samples, describe it with the same VLM used for
the main text channel, and ask a small classification call — given the
descriptions on both sides plus the new midpoint description — which half of
the gap the transition actually falls in. The boundary is snapped to the
midpoint only when the transition is confidently placed in the first half;
otherwise the original coarse timestamp already sits as close to the
transition as one extra sample can get it, so it is left alone.

Only detected boundaries are refined. File joins (`forced_bounds` in
`cuts.segmentation`) are already exact discontinuities with no gap to search
inside.

Random single-frame decode (`media.decode_frame_near`) is what makes this
possible without breaking the pipeline's single streaming pass: this runs
after segmentation, on a handful of boundaries, not on the whole video.

The pass runs in three phases rather than one loop, so the two API stages are
concurrent across boundaries instead of serialized per boundary:

  1. gather  — decode + OCR + encode each midpoint frame (local, sequential)
  2. describe — one batched, concurrent pass over every midpoint frame
  3. decide  — one concurrent pass of per-boundary classification calls

Boundaries are independent of each other, so nothing here needs to be
sequential; only phase 1 is, because it is local CPU/IO on a handful of
frames rather than network latency.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Set, Tuple

from .config import CutsConfig
from .media import Timeline, decode_frame_near
from .schema import Event
from .segmentation import SegmentationInput


def refine_boundaries(
    events: List[Event],
    inp: SegmentationInput,
    vlm_texts: Sequence[str],
    timeline: Timeline,
    config: CutsConfig,
    locked: Optional[Set[int]] = None,
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Snap each detected boundary's timestamp toward its midpoint, in place.

    Mutates `start_time`/`end_time` on adjacent events; sample_indices and
    frame ranges are left as segmentation produced them, since the midpoint
    frame is a one-off decode outside the uniform sample grid.
    """
    if not config.refine.enabled or len(events) < 2 or not vlm_texts:
        return

    from .signals.vlm import decide_refinement, describe_batch, encode_frame
    from .media import resize_long_edge
    want_ocr_grounding = config.ocr.enabled
    if want_ocr_grounding:
        from .redact import scrub_frame

    locked = locked or set()
    # Boundaries are located by *time*, not by frame index. Under seek-based
    # sampling (`cuts.observe`) a sample has no decode ordinal at all, and even
    # on the sequential path the time is the thing both sides already agree on.
    pos_of_time = {round(t, 3): i for i, t in enumerate(inp.times)}

    # ---- Phase 1: gather midpoint frames (local, sequential) --------------
    # (event index, sample position of the boundary, midpoint time)
    candidates: List[Tuple[int, int, float]] = []
    encoded: List[str] = []
    ocr_texts: List[str] = []

    for i in range(1, len(events)):
        cur = events[i]
        b = pos_of_time.get(round(cur.start_time, 3))
        if b is None or b == 0 or b in locked or b >= len(vlm_texts):
            continue

        t_mid = (inp.times[b - 1] + inp.times[b]) / 2.0
        src, local_t = timeline.local_time(t_mid)
        frame = decode_frame_near(src.path, local_t)
        if frame is None:
            continue

        # Blacks out any credential in frame.image before it is encoded. A
        # midpoint frame is a one-off decode that never went through the main
        # loop's scrub, so it needs its own.
        ocr_text, safe = "", True
        if want_ocr_grounding:
            ocr_text, safe, hits = scrub_frame(
                frame.image, config.ocr, config.redact, verbose=verbose
            )
            if hits and verbose:
                print(f"  [redact] refine t={t_mid:.0f}s: "
                      f"{', '.join(sorted(set(hits)))}")
        if not safe:
            # Skip this boundary rather than upload the frame. The boundary
            # simply keeps its coarse timestamp, which is what happens anyway
            # whenever the classification comes back "unclear".
            if verbose:
                print(f"  [redact] refine t={t_mid:.0f}s: boundary skipped "
                      f"(secret could not be localized)")
            continue

        candidates.append((i, b, t_mid))
        ocr_texts.append(ocr_text)
        encoded.append(encode_frame(
            resize_long_edge(frame.image, config.vlm.frame_long_edge)
        ))

    if not candidates:
        if verbose:
            print("  [refine] no refinable boundaries (all forced or at edges)")
        return

    # ---- Phase 2: describe every midpoint (batched, concurrent) -----------
    mid_descriptions = describe_batch(
        encoded, config.vlm, ocr_texts=ocr_texts,
        api_key=api_key, verbose=verbose,
    )

    # ---- Phase 3: classify each gap (concurrent) --------------------------
    def _decide(arg) -> str:
        (_i, b, _t_mid), desc_mid = arg
        return decide_refinement(
            vlm_texts[b - 1], desc_mid, vlm_texts[b],
            config.refine, config.vlm.model,
            api_key=api_key, verbose=verbose,
        )

    with ThreadPoolExecutor(max_workers=max(1, config.vlm.concurrency)) as pool:
        transitions = list(pool.map(_decide, zip(candidates, mid_descriptions)))

    moved = 0
    for (i, _b, t_mid), transition in zip(candidates, transitions):
        if transition == "first_half":
            events[i - 1].end_time = events[i].start_time = t_mid
            moved += 1

    if verbose:
        print(f"  [refine] moved {moved}/{len(candidates)} boundaries toward "
              f"their midpoint")
