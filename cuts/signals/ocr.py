"""Text channel — on-screen text read from sampled frames.

For screen recordings this is usually the *strongest* semantic signal. A
self-supervised visual model sees "monospace text on a dark background" both
before and after the user switches from editing a parser to reading a stack
trace; the words on screen, meanwhile, change completely. Empirically, adding
this channel is what recovers boundaries the visual channel ranks too weakly
to survive merging.

Design notes carried over from the previous implementation because they
measurably mattered:

  * OCR runs on full-resolution decoded frames, never on re-encoded JPEG
    thumbnails — compression artefacts materially hurt small on-screen text.
  * Optional integer upscaling (2-3x, bicubic) before OCR. Strongly
    recommended for 1080p capture where editor/terminal glyphs are small
    relative to the canvas.
  * Optional fractional ROI crops, so persistent chrome (taskbars, window
    decorations) can be excluded before OCR rather than filtered afterwards.

Unlike the previous version this module is *streaming*: it exposes a
per-image call so the pipeline can OCR each frame as it is decoded and then
discard the pixels. Holding every sampled frame of a 45-minute 1080p capture
would need roughly 16 GB.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from ..config import OCRConfig


# Lazy singleton — engine init is slow on first call (loads ONNX weights).
_ENGINE = None


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
        _ENGINE = RapidOCR()
    return _ENGINE


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def _upscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Upscale by an integer factor with bicubic interpolation. 1 = no-op."""
    if factor <= 1:
        return img
    import cv2
    h, w = img.shape[:2]
    return cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)


def _apply_roi(
    img: np.ndarray, roi: Tuple[float, float, float, float]
) -> np.ndarray:
    """Crop a fractional (x1, y1, x2, y2) box. Degenerate boxes return the full image."""
    h, w = img.shape[:2]
    x1 = max(0, int(roi[0] * w))
    y1 = max(0, int(roi[1] * h))
    x2 = min(w, int(roi[2] * w))
    y2 = min(h, int(roi[3] * h))
    if x2 <= x1 or y2 <= y1:
        return img
    return img[y1:y2, x1:x2]


def _prepare_patches(img: np.ndarray, config: OCRConfig) -> List[np.ndarray]:
    """Return the upscaled patches to OCR for one frame.

    Upscaling is applied *after* cropping so small ROIs get the full benefit.
    """
    patches = (
        [_apply_roi(img, box) for box in config.roi] if config.roi else [img]
    )
    return [_upscale(p, config.upscale_factor) for p in patches]


# ---------------------------------------------------------------------------
# Per-image OCR
# ---------------------------------------------------------------------------

def ocr_lines(
    img: np.ndarray, config: OCRConfig
) -> List[Tuple[str, float]]:
    """OCR one BGR frame; return (text, confidence) pairs above min_confidence."""
    engine = _get_engine()
    out: List[Tuple[str, float]] = []
    for patch in _prepare_patches(img, config):
        result, _ = engine(patch)
        if not result:
            continue
        for row in result:
            if len(row) < 3:
                continue
            text, score = row[1], row[2]
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
            text = (text or "").strip()
            if text and score >= config.min_confidence:
                out.append((text, score))
    return out


def ocr_image(img: np.ndarray, config: OCRConfig) -> str:
    """OCR one BGR frame and return its text as newline-joined lines.

    Duplicate lines within a single frame are collapsed (OCR often reports the
    same string from overlapping ROI crops), preserving first-seen order.
    """
    seen = set()
    kept: List[str] = []
    for text, _score in ocr_lines(img, config):
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(text)
    return "\n".join(kept)


def save_debug_png(
    img: np.ndarray, config: OCRConfig, out_dir: str, name: str
) -> None:
    """Write the exact patches fed to OCR, for visual inspection of failures."""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    for ri, patch in enumerate(_prepare_patches(img, config)):
        cv2.imwrite(os.path.join(out_dir, f"{name}_roi{ri}.png"), patch)


# ---------------------------------------------------------------------------
# Stride handling
# ---------------------------------------------------------------------------

def should_ocr(sample_position: int, config: OCRConfig) -> bool:
    """Whether the sample at this position should be OCR'd, given the stride.

    OCR is by far the slowest per-frame signal, so `stride > 1` lets callers
    trade text resolution for wall time.
    """
    stride = max(1, config.stride)
    return sample_position % stride == 0


def carry_forward(texts: List[Optional[str]]) -> List[str]:
    """Fill skipped positions with the most recent OCR result.

    With `stride > 1` only some samples are read. Screen content persists
    between reads, so the correct fill is the last observed text rather than
    an empty string — an empty string would read as "all text vanished" and
    manufacture a boundary at every skipped frame.
    """
    out: List[str] = []
    last = ""
    for t in texts:
        if t is not None:
            last = t
        out.append(last)
    return out


if __name__ == "__main__":
    # Standalone debug: OCR a handful of sampled frames and show what was read.
    import sys
    import time

    from ..config import CutsConfig
    from ..media import iter_sampled_frames

    if len(sys.argv) < 2:
        print("usage: python -m cuts.signals.ocr <video_path> [interval_sec]")
        sys.exit(1)
    cfg = CutsConfig()
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

    t0 = time.time()
    for i, sf in enumerate(iter_sampled_frames(sys.argv[1], interval_sec=interval)):
        t1 = time.time()
        text = ocr_image(sf.image, cfg.ocr)
        lines = text.split("\n") if text else []
        print(f"\n[t={sf.time_sec:6.1f}s] {len(lines)} lines "
              f"({time.time() - t1:.2f}s)")
        for ln in lines[:6]:
            print(f"    {ln!r}")
        if i >= 5:
            break
    print(f"\ntotal {time.time() - t0:.1f}s")
