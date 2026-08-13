"""Query-time retrieval over a built index.

Loads the artifacts that ``index.build_index`` produced and answers queries
with ranked SegmentRecords. Ranking is Reciprocal Rank Fusion across
whichever of BM25 / text embeddings / CLIP image embeddings is available.

    score(doc) = sum over sources of  1 / (rrf_k + rank_s(doc))

RRF is parameter-light, robust to score-scale mismatches, and empirically
competitive with anything more elaborate for small indices.

Each result carries:
  * the SegmentRecord (so start/end time + representative frames travel),
  * a score,
  * a list of sources that contributed (``["bm25", "embedding", "clip"]``),
  * a matched text snippet (±40 chars around a BM25 hit),
  * a human-readable explanation string.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import RetrievalConfig
from .index import tokenize
from .schema import SegmentRecord, read_segments


@dataclass
class SearchResult:
    """One ranked retrieval result."""

    segment: SegmentRecord
    score: float
    sources: List[str] = field(default_factory=list)
    matched_snippet: str = ""
    explanation: str = ""
    # Per-source ranks (1-indexed; missing when source didn't contribute).
    ranks: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment": self.segment.to_dict(),
            "score": self.score,
            "sources": self.sources,
            "matched_snippet": self.matched_snippet,
            "explanation": self.explanation,
            "ranks": self.ranks,
            "start_time": self.segment.start_time,
            "end_time": self.segment.end_time,
            "start_frame": self.segment.start_frame,
            "end_frame": self.segment.end_frame,
        }


class Searcher:
    """Query engine over a single indexed video directory."""

    def __init__(self, index_dir: str, config: Optional[RetrievalConfig] = None):
        self.index_dir = index_dir
        self.config = config or RetrievalConfig()

        payload = read_segments(os.path.join(index_dir, "segments.json"))
        self.video_id: str = payload["video_id"]
        self.video_path: str = payload["video_path"]
        self.segments: List[SegmentRecord] = payload["segments"]

        meta_path = os.path.join(index_dir, "meta.json")
        self.meta: Dict[str, Any] = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

        # BM25.
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            bm = pickle.load(f)
        self._bm25 = bm["bm25"]
        self._corpus = bm["corpus"]

        # Text embeddings.
        self._text_emb: Optional[np.ndarray] = None
        text_path = os.path.join(index_dir, "text_emb.npy")
        if os.path.exists(text_path):
            self._text_emb = np.load(text_path)

        # Image embeddings.
        self._image_emb: Optional[np.ndarray] = None
        image_path = os.path.join(index_dir, "image_emb.npy")
        if os.path.exists(image_path):
            self._image_emb = np.load(image_path)

        # Lazily-initialized encoders.
        self._text_encoder = None
        self._clip = None

    # ------------------------------------------------------------------
    # Per-source ranking
    # ------------------------------------------------------------------

    def _bm25_scores(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        if not tokens:
            return np.zeros(len(self.segments), dtype=np.float32)
        return np.asarray(self._bm25.get_scores(tokens), dtype=np.float32)

    def _text_emb_scores(self, query: str) -> Optional[np.ndarray]:
        if self._text_emb is None:
            return None
        if self._text_encoder is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._text_encoder = SentenceTransformer(
                self.meta.get("text_embed_model") or self.config.text_embed_model,
            )
        q = self._text_encoder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)
        # Segments with empty text have zero-norm rows; matmul still safe.
        return self._text_emb @ q

    def _clip_scores(self, query: str) -> Optional[np.ndarray]:
        if self._image_emb is None:
            return None
        if self._clip is None:
            import open_clip  # type: ignore
            import torch  # type: ignore

            model_name = self.config.clip_model
            pretrained = self.config.clip_pretrained
            if self.meta.get("clip_model"):
                parts = self.meta["clip_model"].split("/", 1)
                if len(parts) == 2:
                    model_name, pretrained = parts
            model, _, _preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            model.eval()
            self._clip = (model, tokenizer, torch)

        model, tokenizer, torch = self._clip
        with torch.no_grad():
            toks = tokenizer([query])
            feats = model.encode_text(toks)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        q = feats[0].cpu().numpy().astype(np.float32)
        return self._image_emb @ q

    # ------------------------------------------------------------------
    # Fusion + result formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _ranks_from_scores(scores: np.ndarray) -> np.ndarray:
        """Return 1-indexed ranks (rank 1 = highest score)."""
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        return ranks + 1  # 1-indexed

    def query(
        self,
        text: str,
        top_k: Optional[int] = None,
        sources: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Run a hybrid query. ``sources`` restricts which rankings contribute.

        Valid sources: "bm25", "embedding", "clip". Default: all available.
        """
        if top_k is None:
            top_k = self.config.default_top_k
        n = len(self.segments)
        if n == 0:
            return []

        per_source_scores: Dict[str, np.ndarray] = {}
        per_source_ranks: Dict[str, np.ndarray] = {}

        wanted = set(sources) if sources else {"bm25", "embedding", "clip"}

        if "bm25" in wanted:
            s = self._bm25_scores(text)
            if s.any():
                per_source_scores["bm25"] = s
                per_source_ranks["bm25"] = self._ranks_from_scores(s)

        if "embedding" in wanted:
            s = self._text_emb_scores(text)
            if s is not None:
                per_source_scores["embedding"] = s
                per_source_ranks["embedding"] = self._ranks_from_scores(s)

        if "clip" in wanted:
            s = self._clip_scores(text)
            if s is not None:
                per_source_scores["clip"] = s
                per_source_ranks["clip"] = self._ranks_from_scores(s)

        if not per_source_ranks:
            return []

        # RRF fusion.
        k = self.config.rrf_k
        fused = np.zeros(n, dtype=np.float64)
        for ranks in per_source_ranks.values():
            fused += 1.0 / (k + ranks)

        # Only keep docs that at least one source ranked meaningfully (i.e. had
        # a non-zero score). A doc absent from every per_source_scores map
        # still gets an RRF contribution from each source because ranks cover
        # all N; to avoid scoring empty docs, filter by bm25 hits OR positive
        # embedding score when embeddings are enabled.
        keep_mask = np.zeros(n, dtype=bool)
        if "bm25" in per_source_scores:
            keep_mask |= per_source_scores["bm25"] > 0
        if "embedding" in per_source_scores:
            # Cosine threshold is weak; anything > 0.15 is at least topical.
            keep_mask |= per_source_scores["embedding"] > 0.15
        if "clip" in per_source_scores:
            keep_mask |= per_source_scores["clip"] > 0.15

        if not keep_mask.any():
            return []

        fused_masked = np.where(keep_mask, fused, -np.inf)
        top_idx = np.argsort(-fused_masked)[:top_k]

        results: List[SearchResult] = []
        for si in top_idx:
            if not np.isfinite(fused_masked[si]):
                break
            seg = self.segments[int(si)]
            contributors = []
            ranks: Dict[str, int] = {}
            for name, scores in per_source_scores.items():
                if scores[si] > 0:
                    contributors.append(name)
                    ranks[name] = int(per_source_ranks[name][si])

            snippet = _build_snippet(
                seg, text, self.config.snippet_half_chars,
                had_bm25_hit="bm25" in contributors,
            )
            explanation = _build_explanation(contributors, ranks,
                                             per_source_scores, int(si))

            results.append(SearchResult(
                segment=seg,
                score=float(fused[int(si)]),
                sources=contributors,
                matched_snippet=snippet,
                explanation=explanation,
                ranks=ranks,
            ))
        return results


# ---------------------------------------------------------------------------
# Snippet + explanation helpers
# ---------------------------------------------------------------------------

def _build_snippet(
    seg: SegmentRecord, query: str, half: int, had_bm25_hit: bool
) -> str:
    text = seg.combined_text
    if not text:
        return ""
    if had_bm25_hit:
        ql = [t for t in tokenize(query)]
        low = text.lower()
        for tok in ql:
            pos = low.find(tok)
            if pos >= 0:
                start = max(0, pos - half)
                end = min(len(text), pos + len(tok) + half)
                clip = text[start:end].replace("\n", " ")
                prefix = "…" if start > 0 else ""
                suffix = "…" if end < len(text) else ""
                return f"{prefix}{clip}{suffix}"
    # Fallback: first 160 chars on a single line.
    flat = text.replace("\n", " ")
    return flat[:160] + ("…" if len(flat) > 160 else "")


def _build_explanation(
    contributors: List[str],
    ranks: Dict[str, int],
    scores: Dict[str, np.ndarray],
    si: int,
) -> str:
    parts = []
    for name in contributors:
        if name == "bm25":
            parts.append(f"BM25 rank #{ranks[name]} (score {scores[name][si]:.2f})")
        elif name == "embedding":
            parts.append(
                f"text-embedding rank #{ranks[name]} (cos {scores[name][si]:.2f})"
            )
        elif name == "clip":
            parts.append(
                f"CLIP rank #{ranks[name]} (cos {scores[name][si]:.2f})"
            )
    return "; ".join(parts) if parts else "no contributing signals"
