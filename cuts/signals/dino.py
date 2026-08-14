"""Visual channel — DINOv2 frame embeddings.

This is the signal the EFS paper (arXiv 2603.00983) builds its temporal
similarity curve from. DINOv2 is self-supervised on natural images and
produces features where visually similar frames land close together under
cosine similarity, without any task-specific training.

Domain caveat, deliberately recorded here because it drives the fusion design
in `cuts.segmentation`: DINOv2 was trained on natural images, and screen
recordings are a substantial distribution shift. Two screens showing entirely
different source files are both "monospace text on a dark background" and can
sit very close in DINO space, even though the semantic content changed
completely. That is why the pipeline fuses this channel with OCR text rather
than relying on it alone. Measure both on your own footage before assuming
either dominates.

Embeddings are L2-normalized on the way out, so cosine similarity between two
frames is a plain dot product.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..config import DinoConfig
from ..media import SampledFrame


# Module-level singleton — model init is heavy (weights download on first use)
# and we do not want to pay it per batch.
_MODEL = None
_MODEL_KEY: Optional[tuple] = None


def _get_model(config: DinoConfig, device: str):
    """Lazily load the DINOv2 model + image processor, cached per (model, device)."""
    global _MODEL, _MODEL_KEY

    key = (config.model_name, device)
    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL

    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(config.model_name)
    model = AutoModel.from_pretrained(config.model_name)
    model.eval().to(device)
    # Inference only — no gradients anywhere in this pipeline.
    for p in model.parameters():
        p.requires_grad_(False)

    _MODEL = (model, processor, torch)
    _MODEL_KEY = key
    return _MODEL


def embed_frames(
    frames: Sequence[SampledFrame],
    config: DinoConfig,
    device: str = "cpu",
    batch_size: int = 16,
    verbose: bool = False,
) -> np.ndarray:
    """Embed a sequence of sampled frames.

    Returns an ``(N, D)`` float32 array of L2-normalized embeddings, row i
    corresponding to ``frames[i]``. Returns shape ``(0, 0)`` for empty input.

    The whole sequence is embedded in one call, so the caller must keep the
    frame list to a size that fits in memory. For long videos prefer
    `embed_stream`, which consumes an iterator and never holds more than
    `batch_size` frames at once.
    """
    if not frames:
        return np.zeros((0, 0), dtype=np.float32)
    return embed_stream(
        iter(frames), config, device=device, batch_size=batch_size, verbose=verbose
    )


def embed_stream(
    frames,
    config: DinoConfig,
    device: str = "cpu",
    batch_size: int = 16,
    verbose: bool = False,
) -> np.ndarray:
    """Embed frames from an iterator, holding at most `batch_size` at a time.

    This is the memory-safe path for long videos: the caller can hand us the
    output of `media.iter_sampled_frames` directly and peak pixel memory stays
    bounded regardless of video length.
    """
    import cv2

    model, processor, torch = _get_model(config, device)

    out_chunks: List[np.ndarray] = []
    batch: List[np.ndarray] = []
    n_done = 0

    def _flush() -> None:
        nonlocal batch, n_done
        if not batch:
            return
        # The HF processor handles resize/crop/normalize and expects RGB.
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            if config.use_cls_token:
                # CLS token: index 0 of the final hidden state. Preferred over
                # pooler_output because it is present on every DINOv2 variant.
                feats = outputs.last_hidden_state[:, 0]
            else:
                # Mean over patch tokens (excludes CLS at index 0).
                feats = outputs.last_hidden_state[:, 1:].mean(dim=1)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        out_chunks.append(feats.float().cpu().numpy().astype(np.float32))
        n_done += len(batch)
        if verbose:
            print(f"    embedded {n_done} frames")
        batch = []

    for sf in frames:
        # cv2 gives BGR; DINOv2's processor expects RGB.
        batch.append(cv2.cvtColor(sf.image, cv2.COLOR_BGR2RGB))
        if len(batch) >= batch_size:
            _flush()
    _flush()

    if not out_chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(out_chunks, axis=0)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Full pairwise cosine similarity. Assumes rows are already L2-normalized.

    Only used for small candidate sets (e.g. MMR redundancy in `cuts.select`);
    segmentation never needs the full N x N matrix.
    """
    if embeddings.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return embeddings @ embeddings.T


def adjacent_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity between each consecutive pair of embeddings.

    Returns a length ``N-1`` array where element i is ``cos(emb[i], emb[i+1])``.
    This is the raw material for the EFS temporal similarity curve.
    """
    if len(embeddings) < 2:
        return np.zeros(0, dtype=np.float32)
    return np.sum(embeddings[:-1] * embeddings[1:], axis=1).astype(np.float32)


if __name__ == "__main__":
    # Standalone debug: embed a video's sampled frames and report timing +
    # how discriminative the embeddings actually are on this footage.
    import sys
    import time

    from ..config import CutsConfig
    from ..media import iter_sampled_frames

    if len(sys.argv) < 2:
        print("usage: python -m cuts.signals.dino <video_path> [interval_sec]")
        sys.exit(1)
    cfg = CutsConfig()
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    print(f"model:  {cfg.dino.model_name}")
    print(f"device: {cfg.device}")
    t0 = time.time()
    emb = embed_stream(
        iter_sampled_frames(sys.argv[1], interval_sec=interval),
        cfg.dino,
        device=cfg.device,
        batch_size=cfg.sampling.batch_size,
        verbose=True,
    )
    print(f"embeddings: {emb.shape} in {time.time() - t0:.2f}s")
    if len(emb) >= 2:
        adj = adjacent_similarity(emb)
        print(f"adjacent cosine: min={adj.min():.4f} mean={adj.mean():.4f} "
              f"max={adj.max():.4f}")
        print("  (a narrow range here means DINO is NOT discriminating this "
              "footage well — expect OCR to carry the signal instead)")
