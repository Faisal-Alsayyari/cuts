"""EFS Stages 2-3 — query-driven keyframe selection.

This completes the implementation of arXiv 2603.00983 (Chen et al.). Stage 1
lives in `cuts.segmentation` and is query-free, which is why chapter
generation uses it. Stages 2 and 3, implemented here, **require a text query**:

  Stage 2 — Anchor selection. Within each event, pick the single frame most
            relevant to the query:  k_j = argmax_{i in G_j} s_i^itm

  Stage 3 — Adaptive MMR. Refine globally across all anchors, trading query
            relevance against visual redundancy:

              argmax_{I_i in C} [ lambda * sim(I_i, Q)
                                  - (1 - lambda) * max_{I_j in K} sim(I_i, I_j) ]

            with thresholds theta_strict = clip(mu - alpha*sigma, 0, 1) and
            theta_loose = clip(mu + alpha*sigma, 0, 1), relaxed from strict
            toward loose until k frames have been selected.

Nothing in the chapter pipeline calls this. It is here because it is the
natural engine for the "find the moment matching this description" workflow —
given a query, it returns the k frames across a multi-hour recording that best
cover it, which is exactly what you would hand to a vision model to answer a
question about the footage.

Deviation from the paper, recorded honestly: the paper scores query-frame
relevance with BLIP2-ITM. This uses open-clip, which is far lighter and needs
no extra checkpoint, but is a weaker relevance model. Swap `_clip_scores` for
a BLIP2-ITM call to reproduce the paper's numbers exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .config import SelectConfig


@dataclass
class SelectedFrame:
    """One keyframe chosen for a query."""

    frame_idx: int
    time_sec: float
    event_index: int      # which event it anchors
    relevance: float      # query-relevance score in [0, 1]
    is_anchor: bool       # True when chosen as its event's Stage-2 anchor


# ---------------------------------------------------------------------------
# Query-frame relevance (paper: BLIP2-ITM; here: open-clip)
# ---------------------------------------------------------------------------

_CLIP = None
_CLIP_KEY: Optional[tuple] = None


def _get_clip(config: SelectConfig, device: str):
    global _CLIP, _CLIP_KEY
    key = (config.clip_model, config.clip_pretrained, device)
    if _CLIP is not None and _CLIP_KEY == key:
        return _CLIP

    import open_clip  # type: ignore
    import torch  # type: ignore

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.clip_model, pretrained=config.clip_pretrained
    )
    tokenizer = open_clip.get_tokenizer(config.clip_model)
    model.eval().to(device)
    _CLIP = (model, preprocess, tokenizer, torch)
    _CLIP_KEY = key
    return _CLIP


def clip_relevance(
    images: Sequence[np.ndarray],
    query: str,
    config: SelectConfig,
    device: str = "cpu",
    batch_size: int = 16,
) -> np.ndarray:
    """Cosine similarity between each BGR frame and the query text.

    Returned scores are rescaled from cosine's [-1, 1] into [0, 1], because the
    paper's adaptive thresholds clip to [0, 1] and assume that range.
    """
    if not images:
        return np.zeros(0, dtype=np.float32)

    import cv2
    from PIL import Image  # type: ignore

    model, preprocess, tokenizer, torch = _get_clip(config, device)

    with torch.no_grad():
        text_feat = model.encode_text(tokenizer([query]).to(device))
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        scores: List[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            tensor = torch.stack([
                preprocess(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
                for im in chunk
            ]).to(device)
            feats = model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scores.append((feats @ text_feat.T).squeeze(-1).float().cpu().numpy())

    sims = np.concatenate(scores).astype(np.float32)
    return ((sims + 1.0) / 2.0).clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Stage 2 — anchors
# ---------------------------------------------------------------------------

def select_anchors(
    event_spans: Sequence[Sequence[int]],
    relevance: np.ndarray,
) -> List[int]:
    """One anchor per event: the most query-relevant sample in that event.

    `event_spans[j]` holds the sample positions belonging to event j.
    Returns positions into the same sample space, one per non-empty event.
    """
    anchors: List[int] = []
    for span in event_spans:
        if len(span) == 0:
            continue
        best = max(span, key=lambda p: float(relevance[p]))
        anchors.append(int(best))
    return anchors


# ---------------------------------------------------------------------------
# Stage 3 — adaptive MMR
# ---------------------------------------------------------------------------

def adaptive_mmr(
    embeddings: np.ndarray,
    relevance: np.ndarray,
    anchors: Sequence[int],
    config: SelectConfig,
) -> List[int]:
    """Refine anchors into a final set of `config.top_k` frames.

    Standard MMR maximizes relevance minus redundancy against what is already
    picked. The paper's adaptation is the *candidate gate*: a candidate must be
    dissimilar enough from the initial anchors to be admitted, and that
    dissimilarity threshold is relaxed from strict toward loose until k frames
    have been selected. This keeps event coverage from collapsing when the
    query happens to match one visually uniform stretch of the video.
    """
    n = len(embeddings)
    k = min(config.top_k, n)
    if n == 0 or k == 0:
        return []
    if n <= k:
        return list(range(n))

    anchors = list(anchors)
    if not anchors:
        anchors = [int(np.argmax(relevance))]

    # mu / sigma of each candidate's peak similarity to the anchor set.
    anchor_mat = embeddings[anchors]                       # (A, D)
    max_sim_to_anchors = (embeddings @ anchor_mat.T).max(axis=1)  # (N,)
    mu = float(max_sim_to_anchors.mean())
    sigma = float(max_sim_to_anchors.std())
    a = config.alpha
    theta_strict = float(np.clip(mu - a * sigma, 0.0, 1.0))
    theta_loose = float(np.clip(mu + a * sigma, 0.0, 1.0))

    lam = config.mmr_lambda
    selected: List[int] = []
    # Anchors seed the selection, most relevant first.
    for idx in sorted(anchors, key=lambda i: -float(relevance[i])):
        if len(selected) >= k:
            break
        if idx not in selected:
            selected.append(int(idx))

    # Relax the gate in even steps from strict to loose until we reach k.
    n_steps = 5
    thresholds = (
        [theta_strict + (theta_loose - theta_strict) * s / (n_steps - 1)
         for s in range(n_steps)]
        if theta_loose > theta_strict else [theta_strict]
    )

    for theta in thresholds:
        while len(selected) < k:
            sel_mat = embeddings[selected]
            redundancy = (embeddings @ sel_mat.T).max(axis=1)
            mmr = lam * relevance - (1.0 - lam) * redundancy
            # Gate: admit only candidates below the current similarity ceiling.
            eligible = np.where(
                (max_sim_to_anchors <= theta)
                & ~np.isin(np.arange(n), selected)
            )[0]
            if eligible.size == 0:
                break
            selected.append(int(eligible[int(np.argmax(mmr[eligible]))]))
        if len(selected) >= k:
            break

    # Final top-up if even the loose gate could not fill k.
    if len(selected) < k:
        remaining = [i for i in np.argsort(-relevance) if i not in selected]
        selected.extend(int(i) for i in remaining[: k - len(selected)])

    return sorted(selected)


def select_frames(
    query: str,
    embeddings: np.ndarray,
    relevance: np.ndarray,
    event_spans: Sequence[Sequence[int]],
    frame_indices: Sequence[int],
    times: Sequence[float],
    config: SelectConfig,
) -> List[SelectedFrame]:
    """Full Stages 2+3: anchors, then adaptive-MMR refinement to `top_k`."""
    anchors = select_anchors(event_spans, relevance)
    chosen = adaptive_mmr(embeddings, relevance, anchors, config)

    event_of = {}
    for j, span in enumerate(event_spans):
        for p in span:
            event_of[p] = j

    anchor_set = set(anchors)
    return [
        SelectedFrame(
            frame_idx=frame_indices[p],
            time_sec=times[p],
            event_index=event_of.get(p, -1),
            relevance=float(relevance[p]),
            is_anchor=p in anchor_set,
        )
        for p in chosen
    ]
