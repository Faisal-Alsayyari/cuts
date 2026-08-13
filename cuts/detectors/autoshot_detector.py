"""Arm — AutoShot (NAS-optimised 3-D ConvNet + Transformer for SBD).

AutoShot (Zhu et al., CVPR NAS Workshop 2023) uses neural architecture search
to find a 3-D convolutional + Transformer encoder that outperforms TransNetV2
on ClipShots, BBC, and RAI by ~1 % F1. Like TransNetV2, it emits a scalar
per-frame transition probability, so the two models share the same downstream
BoundaryCandidate conversion logic.

Setup (two steps required before first use):
    1. Clone the AutoShot repository:
           git clone https://github.com/wentaozhu/AutoShot
    2. Download the pre-trained checkpoint from the link in the README
       (Baidu Drive or Google Drive; the .pth file, NOT the .pickle file).
    3. Point `AutoShotConfig.repo_path` at the cloned directory and
       `AutoShotConfig.checkpoint_path` at the .pth file.

The model is loaded lazily and cached as a module-level singleton so that
repeated calls on the same video do not pay the cold-start cost.

VFR note: AutoShot decodes frames via ffmpeg at 48×27 px.  ffmpeg produces
frames in presentation order, which aligns with PyAV's decoding-order index
used everywhere else in the pipeline.
"""

from __future__ import annotations

import sys
from typing import List, Tuple

import numpy as np

from cuts.config import AutoShotConfig
from cuts.detectors.ensemble import BoundaryCandidate


# Module-level singleton — lazy-loaded on first call to `detect()`.
_MODEL = None
_MODEL_CACHE_KEY: tuple | None = None  # (repo_path, checkpoint_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(video_path: str, config: AutoShotConfig) -> List[BoundaryCandidate]:
    """Run AutoShot on *video_path* and return boundary candidates.

    A boundary is emitted for the start frame of every scene after the first.
    Frames whose raw probability exceeds `config.gradual_threshold` for a run
    of >= `config.min_gradual_frames` are classified as gradual transitions.
    """
    model, device = _get_model(config)

    frames = _decode_frames(video_path)            # (T, 27, 48, 3) uint8
    probs = _run_inference(model, device, frames)   # (T,) float32 in [0, 1]

    scenes = _probs_to_scenes(probs, threshold=config.scene_threshold)
    gradual_spans = _find_elevated_spans(
        probs,
        threshold=config.gradual_threshold,
        min_length=config.min_gradual_frames,
    )
    return _scenes_to_candidates(scenes, gradual_spans, probs)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _get_model(config: AutoShotConfig):
    """Lazily construct the AutoShot model from the cloned repo."""
    global _MODEL, _MODEL_CACHE_KEY

    cache_key = (config.repo_path, config.checkpoint_path)
    if _MODEL is not None and _MODEL_CACHE_KEY == cache_key:
        return _MODEL

    if not config.repo_path:
        raise RuntimeError(
            "AutoShotConfig.repo_path must point to the cloned "
            "https://github.com/wentaozhu/AutoShot directory. "
            "See the module docstring for setup instructions."
        )
    if not config.checkpoint_path:
        raise RuntimeError(
            "AutoShotConfig.checkpoint_path must point to the AutoShot "
            ".pth checkpoint file downloaded from the project's drive link."
        )

    # Make the AutoShot repo importable without polluting the front of sys.path
    # permanently — insert only if not already present.
    if config.repo_path not in sys.path:
        sys.path.insert(0, config.repo_path)

    import torch  # type: ignore
    # The model class lives in a file named after the NAS-found architecture.
    from supernet_flattransf_3_8_8_8_13_12_0_16_60 import TransNetV2Supernet  # type: ignore

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = TransNetV2Supernet().eval()
    if device == "cuda":
        model = model.cuda()

    # Partial load: only weights whose names exist in the current model are
    # applied.  This matches the pattern used in the AutoShot evaluation script
    # and survives minor version mismatches.
    state = torch.load(
        config.checkpoint_path, map_location=device, weights_only=True
    )
    model_dict = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_dict}
    model_dict.update(filtered)
    model.load_state_dict(model_dict)

    _MODEL = (model, device)
    _MODEL_CACHE_KEY = cache_key
    return _MODEL


# ---------------------------------------------------------------------------
# Frame decoding
# ---------------------------------------------------------------------------

def _decode_frames(
    video_path: str, width: int = 48, height: int = 27
) -> np.ndarray:
    """Decode *video_path* to a (T, H, W, 3) uint8 array at low resolution.

    AutoShot was trained on 48×27 frames (same resolution as TransNetV2).
    Using ffmpeg's built-in scaler avoids loading PIL/torchvision.
    """
    import ffmpeg  # type: ignore

    stream, _ = (
        ffmpeg.input(video_path)
        .output(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s=f"{width}x{height}",
        )
        .run(capture_stdout=True, capture_stderr=True)
    )
    return np.frombuffer(stream, np.uint8).reshape([-1, height, width, 3])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(model, device: str, frames: np.ndarray) -> np.ndarray:
    """Run batched AutoShot inference and return per-frame probabilities (T,).

    The model processes 100-frame windows with a 50-frame stride.  Each window
    has 25 context frames on each side (not returned); the central 50 frames
    form the output for that window.  The video is padded at both ends so that
    every real frame appears in at least one window's valid region.
    """
    import torch  # type: ignore

    n = len(frames)

    # Compute end-padding so total length is divisible by 50 frames.
    pad_end = 50 - (n % 50)
    if pad_end == 50:
        pad_end = 0

    padded = np.concatenate(
        [frames[:1]] * 25 + [frames] + [frames[-1:]] * (pad_end + 25),
        axis=0,
    )

    predictions: List[np.ndarray] = []
    for i in range(0, len(padded) - 50, 50):
        batch = padded[i : i + 100]  # (100, H, W, C)
        # AutoShot expects (1, C, T, H, W).
        tensor = (
            torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...])
            .float()
            .to(device)
        )
        with torch.inference_mode():
            out = model(tensor)
            if isinstance(out, tuple):
                out = out[0]
            # out[0] shape: (T=100, 1).  Sigmoid gives transition probability.
            chunk = torch.sigmoid(out[0]).detach().cpu().numpy()
        # Only the central 50 frames are valid (context excluded).
        predictions.append(chunk[25:75].reshape(-1))  # (50,)

    return np.concatenate(predictions)[:n]


# ---------------------------------------------------------------------------
# Probability → BoundaryCandidate conversion
# (mirrors the same helpers in transnetv2_detector.py)
# ---------------------------------------------------------------------------

def _probs_to_scenes(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold probabilities and convert to (N, 2) [start, end] scene array.

    A new scene begins when the binary signal transitions from 1 → 0, which
    mirrors the `predictions_to_scenes` function in AutoShot's own utils.py.
    """
    binary = (probs > threshold).astype(np.int32)
    scenes: List[List[int]] = []
    t_prev = 0
    start = 0
    for i, t in enumerate(binary):
        if t_prev == 1 and t == 0:
            scenes.append([start, max(0, i - 1)])
            start = i
        t_prev = t
    scenes.append([start, len(binary) - 1])
    return np.array(scenes, dtype=np.int32)


def _find_elevated_spans(
    probs: np.ndarray, threshold: float, min_length: int
) -> List[Tuple[int, int]]:
    """Return (start, end) spans where probs > threshold for >= min_length frames."""
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
                spans.append((span_start, i - 1))
    if in_span and len(above) - span_start >= min_length:
        spans.append((span_start, len(above) - 1))
    return spans


def _matching_span(
    frame: int, spans: List[Tuple[int, int]]
) -> Tuple[int, int] | None:
    """Return the span that contains (or is adjacent to) *frame*, else None."""
    for gs, ge in spans:
        if gs - 1 <= frame <= ge + 1:
            return gs, ge
    return None


def _scenes_to_candidates(
    scenes: np.ndarray,
    gradual_spans: List[Tuple[int, int]],
    probs: np.ndarray,
) -> List[BoundaryCandidate]:
    """Convert (N, 2) scene array + gradual spans to BoundaryCandidate list."""
    n = len(probs)
    candidates: List[BoundaryCandidate] = []
    if len(scenes) > 1:
        for s_idx in range(1, len(scenes)):
            start_frame = int(scenes[s_idx][0])
            lo = max(0, start_frame - 2)
            hi = min(n, start_frame + 3)
            conf = float(probs[lo:hi].max()) if hi > lo else 0.0
            gradual_match = _matching_span(start_frame, gradual_spans)
            if gradual_match is not None:
                gs, ge = gradual_match
                candidates.append(BoundaryCandidate(
                    frame_idx=gs,
                    end_frame_idx=ge,
                    is_gradual=True,
                    sources=["autoshot"],
                    confidence=conf,
                ))
            else:
                candidates.append(BoundaryCandidate(
                    frame_idx=start_frame,
                    sources=["autoshot"],
                    confidence=conf,
                ))
    return candidates
