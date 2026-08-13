"""Arm B — TransNetV2 (PyTorch port).

TransNetV2 is a deep network trained on shot boundary detection. The PyTorch
port (`transnetv2-pytorch`) exposes:

* `predict_video(path)` -> (video_frames, single_frame_predictions, all_frame_predictions)
    - `single_frame_predictions`: 1D float array, one value per frame in
      decoding order, representing per-frame transition probability. Values
      near 1.0 indicate a boundary.
    - `all_frame_predictions`: per-frame "many-hot" auxiliary output (less
      useful for our purposes).
* `predictions_to_scenes(single_frame_predictions, threshold=0.5)` -> ndarray
  of shape (N, 2) with [start_frame, end_frame] columns.

For our screen-recording domain we typically lower the threshold to 0.3–0.4
(see `TransNetV2Config.scene_threshold`) and additionally inspect the raw
per-frame probability curve to flag gradual transitions: a wide bump (probs
elevated above `gradual_threshold` for more than `min_gradual_frames` frames)
is a dissolve, not a hard cut.

VFR note: the frame indices returned here are ordinal decoding-order indices
(0, 1, 2, ...). They line up with PyAV's `enumerate(container.decode(stream))`
indexing, which is what the rest of the pipeline assumes.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from cuts.config import TransNetV2Config
from cuts.detectors.ensemble import BoundaryCandidate


# Lazy-loaded module-level singleton — TransNetV2 init is heavy and we don't
# want to pay for it on import. `_get_model()` builds it on first use.
_MODEL = None


def _get_model():
    """Lazily construct the TransNetV2 model (PyTorch port)."""
    global _MODEL
    if _MODEL is None:
        # Imported lazily so that environments without the package can still
        # import the rest of cuts (e.g. for benchmark-only workflows).
        from transnetv2_pytorch import TransNetV2  # type: ignore

        _MODEL = TransNetV2()
    return _MODEL


def detect(
    video_path: str, config: TransNetV2Config
) -> List[BoundaryCandidate]:
    """Run TransNetV2 on `video_path` and return boundary candidates.

    A "boundary" here is the START frame of each scene (after the first).
    Hard cuts are scenes whose [start, end] span is 1 frame between adjacent
    scenes; gradual transitions show up as a wide elevated bump in the raw
    probability curve, which we detect separately and tag with `is_gradual`.
    """
    model = _get_model()

    # The PyTorch port mirrors the TF API: predict_video accepts a path string.
    _video_frames, single_frame_predictions, _all_frame_predictions = model.predict_video(video_path)
    probs = np.asarray(single_frame_predictions).reshape(-1)

    # Use the configurable threshold rather than the library default of 0.5.
    scenes = model.predictions_to_scenes(probs, threshold=config.scene_threshold)

    # Detect gradual transition spans from the raw probability curve. These are
    # contiguous runs of frames where probs > gradual_threshold and the run is
    # at least min_gradual_frames long.
    gradual_spans = _find_elevated_spans(
        probs,
        threshold=config.gradual_threshold,
        min_length=config.min_gradual_frames,
    )

    return _scenes_to_candidates(scenes, gradual_spans, probs)


def _scenes_to_candidates(
    scenes: np.ndarray, gradual_spans: List[Tuple[int, int]], probs: np.ndarray
) -> List[BoundaryCandidate]:
    """Convert (N,2) scene array + gradual spans into BoundaryCandidate list."""
    candidates: List[BoundaryCandidate] = []

    # The boundary between scene[i] and scene[i+1] sits between scene[i].end
    # and scene[i+1].start. We emit scene[i+1].start as the boundary frame.
    # Scene 0's start is t=0 (not a cut).
    n = len(probs)
    if len(scenes) > 1:
        for s_idx in range(1, len(scenes)):
            start_frame = int(scenes[s_idx][0])

            # Confidence = peak TransNetV2 probability in a ±2 frame spike
            # window around the boundary frame. The spike often peaks 1 frame
            # before `start_frame`, so a small window is more robust than
            # sampling exactly at start_frame.
            lo = max(0, start_frame - 2)
            hi = min(n, start_frame + 3)
            conf = float(probs[lo:hi].max()) if hi > lo else 0.0

            # Check whether this boundary falls inside (or adjacent to) a
            # gradual span. If so, mark it gradual and use the span endpoints.
            gradual_match = _matching_span(start_frame, gradual_spans)
            if gradual_match is not None:
                gs, ge = gradual_match
                candidates.append(BoundaryCandidate(
                    frame_idx=gs,
                    end_frame_idx=ge,
                    is_gradual=True,
                    sources=["transnetv2"],
                    confidence=conf,
                ))
            else:
                candidates.append(BoundaryCandidate(
                    frame_idx=start_frame,
                    sources=["transnetv2"],
                    confidence=conf,
                ))
    return candidates


def _find_elevated_spans(
    probs: np.ndarray, threshold: float, min_length: int
) -> List[Tuple[int, int]]:
    """Return [(start_frame, end_frame), ...] where probs > threshold contiguously.

    Used to identify dissolves: TransNetV2's per-frame probability curve
    "humps" smoothly across a dissolve rather than spiking at one frame.
    """
    above = probs > threshold
    spans: List[Tuple[int, int]] = []
    in_span = False
    span_start = 0
    for i, hot in enumerate(above):
        if hot and not in_span:
            in_span = True
            span_start = i
        elif not hot and in_span:
            in_span = False
            if i - span_start >= min_length:
                # End is inclusive (last hot frame).
                spans.append((span_start, i - 1))
    if in_span and len(above) - span_start >= min_length:
        spans.append((span_start, len(above) - 1))
    return spans


def _matching_span(frame: int, spans: List[Tuple[int, int]]) -> Tuple[int, int] | None:
    """Return the span containing `frame` (or adjacent within 1 frame), else None."""
    for gs, ge in spans:
        if gs - 1 <= frame <= ge + 1:
            return gs, ge
    return None


if __name__ == "__main__":
    # Standalone debug: run on a video and print first 20 candidates.
    import sys

    from cuts.config import CutsConfig

    if len(sys.argv) < 2:
        print("usage: python -m cuts.detectors.transnetv2_detector <video_path>")
        sys.exit(1)
    cfg = CutsConfig().transnetv2
    cands = detect(sys.argv[1], cfg)
    print(f"TransNetV2: {len(cands)} candidates")
    for c in cands[:20]:
        kind = "gradual" if c.is_gradual else "hard"
        print(f"  {kind:7s} frame {c.frame_idx} -> {c.end_frame_idx}")
