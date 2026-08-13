"""Arm — OmniShotCut (Shot-Query Transformer for holistic SBD).

OmniShotCut (Kiteretsu77 / UVA-Computer-Vision-Lab, arXiv 2025) is a
Shot-Query Video Transformer that detects shot boundaries in diverse content
(anime, vlogs, games, screen recordings, sports, etc.) and classifies both
intra-shot content types and inter-shot transition types.

Unlike TransNetV2 and AutoShot, which produce a per-frame probability curve,
OmniShotCut directly predicts shot intervals [start_frame, end_frame] and
labels the transition entering each shot:

    0  new_start        — first shot; no boundary (skipped)
    1  hard_cut         — instantaneous cut → is_gradual=False
    2  transition_source— shot fading out → is_gradual=True
    3  transition       — transition frames (dissolve, fade, wipe) → is_gradual=True
    4  sudden_jump      — fast-motion jump cut → is_gradual=False

The model therefore produces boundaries with transition-type semantics
"for free", without needing a secondary gradual-transition detection pass.

Setup (two steps required before first use):
    1. Clone the OmniShotCut repository:
           git clone https://github.com/UVA-Computer-Vision-Lab/OmniShotCut
    2. Download the checkpoint:
           mkdir OmniShotCut/checkpoints
           wget -P OmniShotCut/checkpoints \\
               https://huggingface.co/uva-cv-lab/OmniShotCut/resolve/main/OmniShotCut_ckpt.pth
    3. Point `OmniShotCutConfig.repo_path` at the cloned directory and
       `OmniShotCutConfig.checkpoint_path` at the .pth file.

The model is loaded lazily and cached as a module-level singleton.

VFR note: OmniShotCut decodes frames via ffmpeg internally (in
`single_video_inference`), producing presentation-order frame indices that
align with the ordinal indices used throughout the rest of the pipeline.
"""

from __future__ import annotations

import sys
from typing import List

from cuts.config import OmniShotCutConfig
from cuts.detectors.ensemble import BoundaryCandidate


# Inter-label IDs (from OmniShotCut's config/label_correspondence.py) that
# correspond to gradual transitions in our taxonomy.
_GRADUAL_INTER_IDS = frozenset({2, 3})  # transition_source, transition

# Module-level singleton — lazy-loaded on first call to `detect()`.
_MODEL = None
_MODEL_CACHE_KEY: tuple | None = None  # (repo_path, checkpoint_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(
    video_path: str, config: OmniShotCutConfig
) -> List[BoundaryCandidate]:
    """Run OmniShotCut on *video_path* and return boundary candidates.

    OmniShotCut returns shot intervals directly; each interval boundary after
    the first is emitted as a BoundaryCandidate.  The inter-shot label is used
    to set `is_gradual` without any secondary signal analysis.
    """
    model, model_args = _get_model(config)

    # single_video_inference is imported lazily here to avoid a hard
    # dependency at module import time (same pattern as transnetv2_detector).
    from test_code.inference import single_video_inference  # type: ignore

    pred_ranges, _pred_intra, pred_inter, _frames_np, _fps = (
        single_video_inference(
            video_path,
            model,
            model_args,
            num_context_frames=config.num_context_frames,
        )
    )

    return _shots_to_candidates(pred_ranges, pred_inter)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _get_model(config: OmniShotCutConfig):
    """Lazily load the OmniShotCut model from the cloned repo."""
    global _MODEL, _MODEL_CACHE_KEY

    cache_key = (config.repo_path, config.checkpoint_path)
    if _MODEL is not None and _MODEL_CACHE_KEY == cache_key:
        return _MODEL

    if not config.repo_path:
        raise RuntimeError(
            "OmniShotCutConfig.repo_path must point to the cloned "
            "https://github.com/UVA-Computer-Vision-Lab/OmniShotCut directory. "
            "See the module docstring for setup instructions."
        )
    if not config.checkpoint_path:
        raise RuntimeError(
            "OmniShotCutConfig.checkpoint_path must point to the "
            "OmniShotCut_ckpt.pth checkpoint file."
        )

    if config.repo_path not in sys.path:
        sys.path.insert(0, config.repo_path)

    import torch  # type: ignore
    import argparse  # type: ignore

    # PyTorch 2.6+ requires safe_globals for argparse.Namespace in checkpoint
    torch.serialization.add_safe_globals([argparse.Namespace])

    from test_code.inference import load_model  # type: ignore

    model, model_args = load_model(config.checkpoint_path)

    _MODEL = (model, model_args)
    _MODEL_CACHE_KEY = cache_key
    return _MODEL


# ---------------------------------------------------------------------------
# Shot intervals → BoundaryCandidate conversion
# ---------------------------------------------------------------------------

def _shots_to_candidates(
    pred_ranges: List[List[int]],
    pred_inter_labels: List[int],
) -> List[BoundaryCandidate]:
    """Convert OmniShotCut shot intervals to a BoundaryCandidate list.

    `pred_ranges[i]` is [start_frame, end_frame] (inclusive, 0-based) for
    shot i.  `pred_inter_labels[i]` is the inter-shot label describing how
    shot i was reached from shot i-1.

    Shot 0's label is always `new_start` (0), which is skipped — its start
    frame is the beginning of the video, not a cut.
    """
    candidates: List[BoundaryCandidate] = []
    for i in range(1, len(pred_ranges)):
        start_frame = int(pred_ranges[i][0])
        inter_id = int(pred_inter_labels[i]) if i < len(pred_inter_labels) else 1
        is_gradual = inter_id in _GRADUAL_INTER_IDS
        candidates.append(BoundaryCandidate(
            frame_idx=start_frame,
            sources=["omnishotcut"],
            is_gradual=is_gradual,
        ))
    return candidates
