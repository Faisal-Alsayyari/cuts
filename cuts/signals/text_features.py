"""Turn per-sample text into L2-normalized vectors for similarity comparison.

Both the VLM and ASR channels produce a string per sampled frame. To feed them
into the same temporal-similarity machinery as DINOv2 embeddings, each string
becomes a vector whose cosine similarity with its neighbours is meaningful.

We use TF-IDF rather than raw term frequency, and the IDF term is doing real
work in this domain: a screen recording has persistent chrome — menu bars,
status lines, the editor's own filename tabs — that a description can end up
mentioning in nearly every frame, and a scene-setting phrase a VLM tends to
repeat verbatim ("a code editor is open") is no different in effect. Under
plain TF that constant text dominates the similarity and flattens the curve.
IDF drives those ubiquitous tokens toward zero weight so similarity is decided
by the text that actually changes.

Vocabulary is capped (`max_vocab`) so a multi-hour video cannot blow up memory
with a vector per sample over a hundred-thousand-token vocabulary.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence

import numpy as np


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """Alnum/underscore tokenizer, lowercased.

    Lowercasing matters because case is not a meaningful signal here — a VLM
    describing the same file name or term should not be treated as saying
    something different because of capitalization, and it keeps behaviour
    consistent for anything still feeding this from raw OCR text.
    """
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def text_features(
    texts: Sequence[str],
    max_vocab: int = 20000,
    min_df: int = 1,
) -> np.ndarray:
    """Build an ``(N, V)`` L2-normalized TF-IDF matrix over `texts`.

    Rows for empty strings are all-zero, which yields cosine similarity 0
    against everything. Callers must decide what a zero row means for them —
    `cuts.segmentation` treats stretches with no text as "no opinion" and lets
    the other channels decide, rather than reading them as a hard change.

    Returns shape ``(N, 0)`` when no usable tokens exist anywhere.
    """
    n = len(texts)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    tokenized = [tokenize(t) for t in texts]

    # Document frequency across samples, to build the vocabulary and the IDF.
    df: Counter = Counter()
    for toks in tokenized:
        df.update(set(toks))
    if not df:
        return np.zeros((n, 0), dtype=np.float32)

    # Keep the most frequent terms, subject to min_df. Sorting by DF descending
    # keeps the terms that actually recur; hapax OCR noise falls off the end.
    vocab_terms = [t for t, c in df.most_common(max_vocab) if c >= min_df]
    if not vocab_terms:
        return np.zeros((n, 0), dtype=np.float32)
    vocab = {t: i for i, t in enumerate(vocab_terms)}

    # Smoothed IDF: log(1 + N / (1 + df)). Terms present in every sample get a
    # weight near log(1) = 0, which is exactly the chrome-suppression we want.
    idf = np.zeros(len(vocab), dtype=np.float32)
    for term, col in vocab.items():
        idf[col] = math.log(1.0 + n / (1.0 + df[term]))

    mat = np.zeros((n, len(vocab)), dtype=np.float32)
    for row, toks in enumerate(tokenized):
        if not toks:
            continue
        counts = Counter(toks)
        for term, tf in counts.items():
            col = vocab.get(term)
            if col is None:
                continue
            # Sublinear TF damps runaway repetition (e.g. a wall of identical
            # log lines) so one repeated token cannot dominate the vector.
            mat[row, col] = (1.0 + math.log(tf)) * idf[col]

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    np.divide(mat, norms, out=mat, where=norms > 0)
    return mat
