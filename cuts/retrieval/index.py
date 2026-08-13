"""Index construction for retrieval.

Given a list of SegmentRecords with OCR + transcript text already attached,
builds:

  * a BM25 index over the per-segment combined text (required),
  * sentence-transformers text embeddings (``config.enable_text_embeddings``),
  * open-clip image embeddings of each segment's first representative frame
    (``config.enable_image_embeddings``).

All artifacts are written under ``<index_dir>/`` with the layout:

    segments.json           (written by the caller via schema.write_segments)
    bm25.pkl                pickled {"corpus": tokens, "bm25": BM25Okapi}
    text_emb.npy            (N_seg, 384) float32, L2-normalized
    image_emb.npy           (N_seg, 512) float32, L2-normalized
    meta.json               model names, counts, config hash

Embeddings are stored as simple ``.npy`` matrices because N_seg will be in
the hundreds to low thousands — brute-force matmul at query time is fine and
avoids a FAISS dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from dataclasses import asdict
from typing import List, Optional

import cv2
import numpy as np

from ..config import RetrievalConfig
from .schema import SegmentRecord


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """Simple alnum/underscore tokenizer used for both indexing and querying.

    Lowercasing is essential because OCR output is often SHOUTY UI TEXT.
    """
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def _build_bm25(segments: List[SegmentRecord]):
    from rank_bm25 import BM25Okapi  # type: ignore

    corpus = [tokenize(seg.combined_text) for seg in segments]
    # BM25Okapi handles empty docs (score = 0); do not filter them.
    bm25 = BM25Okapi(corpus if any(corpus) else [["__placeholder__"]])
    return corpus, bm25


# ---------------------------------------------------------------------------
# Text embeddings (sentence-transformers)
# ---------------------------------------------------------------------------

def _build_text_embeddings(
    segments: List[SegmentRecord],
    config: RetrievalConfig,
    device: str,
    verbose: bool = False,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(config.text_embed_model, device=device)
    texts = [s.combined_text if s.combined_text else " " for s in segments]
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=verbose,
        normalize_embeddings=True,  # cosine -> dot product
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# Image embeddings (open-clip)
# ---------------------------------------------------------------------------

def _build_image_embeddings(
    segments: List[SegmentRecord],
    index_dir: str,
    config: RetrievalConfig,
    device: str,
    verbose: bool = False,
) -> np.ndarray:
    import open_clip  # type: ignore
    import torch  # type: ignore
    from PIL import Image  # type: ignore

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.clip_model, pretrained=config.clip_pretrained
    )
    model.eval().to(device)

    dim = model.visual.output_dim if hasattr(model.visual, "output_dim") else 512
    out = np.zeros((len(segments), dim), dtype=np.float32)

    with torch.no_grad():
        batch_imgs = []
        batch_idx = []

        def flush():
            if not batch_imgs:
                return
            tensor = torch.stack(batch_imgs).to(device)
            feats = model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            for k, si in enumerate(batch_idx):
                out[si] = feats[k].cpu().numpy().astype(np.float32)
            batch_imgs.clear()
            batch_idx.clear()

        for si, seg in enumerate(segments):
            if not seg.representative_frames:
                continue
            rel = seg.representative_frames[len(seg.representative_frames) // 2]
            abs_path = os.path.join(index_dir, rel)
            img_bgr = cv2.imread(abs_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            batch_imgs.append(preprocess(pil))
            batch_idx.append(si)
            if len(batch_imgs) >= 16:
                flush()
        flush()

    if verbose:
        filled = int((out != 0).any(axis=1).sum())
        print(f"  CLIP image embeddings: {filled}/{len(segments)} segments")
    return out


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def build_index(
    segments: List[SegmentRecord],
    index_dir: str,
    config: RetrievalConfig,
    device: str = "cpu",
    verbose: bool = False,
) -> None:
    """Build and persist all index artifacts for ``segments``.

    Writes:
      * ``bm25.pkl``        — always
      * ``text_emb.npy``    — when ``config.enable_text_embeddings``
      * ``image_emb.npy``   — when ``config.enable_image_embeddings``
      * ``meta.json``       — model/version info + segment count
    """
    os.makedirs(index_dir, exist_ok=True)

    # BM25 is cheap and always built.
    corpus, bm25 = _build_bm25(segments)
    with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
        pickle.dump({"corpus": corpus, "bm25": bm25}, f)

    meta = {
        "n_segments": len(segments),
        "has_text_emb": False,
        "has_image_emb": False,
        "text_embed_model": None,
        "clip_model": None,
        "config_hash": _config_hash(config),
    }

    if config.enable_text_embeddings:
        if verbose:
            print("  building text embeddings...")
        emb = _build_text_embeddings(segments, config, device, verbose=verbose)
        np.save(os.path.join(index_dir, "text_emb.npy"), emb)
        meta["has_text_emb"] = True
        meta["text_embed_model"] = config.text_embed_model

    if config.enable_image_embeddings:
        if verbose:
            print("  building CLIP image embeddings...")
        emb = _build_image_embeddings(segments, index_dir, config, device,
                                       verbose=verbose)
        np.save(os.path.join(index_dir, "image_emb.npy"), emb)
        meta["has_image_emb"] = True
        meta["clip_model"] = f"{config.clip_model}/{config.clip_pretrained}"

    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _config_hash(config: RetrievalConfig) -> str:
    """Short stable hash of the retrieval config, for debug / cache invalidation."""
    h = hashlib.sha1(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
    return h[:12]
