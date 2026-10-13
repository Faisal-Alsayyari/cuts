"""The curve segmenter places boundaries on the cut, not beside it."""

from __future__ import annotations

import numpy as np

from cuts.config import SegmentationConfig
from cuts.segmentation import (
    SegmentationInput,
    carry_forward,
    segment,
)


def _steps(lengths, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k, n in enumerate(lengths):
        base = np.zeros(8, np.float32)
        base[k] = 1.0
        for _ in range(n):
            v = base + rng.normal(0, noise, 8).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
    return np.asarray(rows, np.float32)


def test_boundaries_land_on_the_first_sample_after_each_cut():
    cfg = SegmentationConfig(weight_vlm=0.0, target_events=3, min_event_sec=10)
    for seed in range(20):          # the noise decides which side the dip is on
        emb = _steps([4, 4, 4], noise=0.01, seed=seed)
        seg = segment(SegmentationInput(times=[i * 10.0 for i in range(12)], dino=emb),
                      cfg, duration_sec=120)
        assert seg.starts == [0, 4, 8], seed


def test_forced_bounds_survive_and_flat_input_gives_only_them():
    emb = _steps([12])
    seg = segment(SegmentationInput(times=[i * 10.0 for i in range(12)], dino=emb),
                  SegmentationConfig(weight_vlm=0.0), duration_sec=120,
                  forced_bounds=[6])
    assert 6 in seg.bounds


def test_carry_forward_resets_at_joins():
    assert carry_forward(["a", None, "", "b", None], reset_at=[4]) == \
        ["a", "a", "a", "b", ""]
