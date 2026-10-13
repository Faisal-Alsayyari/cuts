"""Local OCR — grounding text for the VLM description call.

Not an independent channel: `cuts.signals.vlm` is what drives segmentation
and evidence text. This module exists solely to give that vision call exact
strings (file names, error messages) instead of asking it to read them off a
downscaled JPEG, which is where it goes wrong on small monospace text.

It only runs on frames that are about to be uploaded for description — one
call in, one string out.

It has a second job: the engine returns a bounding box alongside every line it
reads, and `cuts.redact` uses those boxes to black out credentials in the
pixels before a frame is uploaded. `ocr_lines` exists for that caller;
`ocr_image` is the same thing with the boxes discarded.

Design notes carried over from the OCR-as-channel implementation because they
measurably mattered:

  * OCR runs on full-resolution decoded frames, never on the resized/JPEG
    version sent to the API — compression and downscaling artefacts
    materially hurt small on-screen text.
  * Optional integer upscaling (2-3x, bicubic) before OCR. Strongly
    recommended for 1080p+ capture where editor/terminal glyphs are small
    relative to the canvas.
  * Optional fractional ROI crops, so persistent chrome (taskbars, window
    decorations) can be excluded before OCR rather than filtered afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..config import OCRConfig


@dataclass
class OCRLine:
    """One line of text the engine read, with where it sits in the frame.

    `box` is an axis-aligned (x1, y1, x2, y2) in **original frame
    coordinates** — the engine's own quad is in patch space, after cropping
    and upscaling, and is mapped back by `ocr_lines`.
    """

    text: str
    box: Tuple[int, int, int, int]
    score: float

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
) -> Tuple[np.ndarray, int, int]:
    """Crop a fractional (x1, y1, x2, y2) box, with the crop's frame offset.

    Degenerate boxes return the full image — and must then report offset
    (0, 0) rather than the box's nominal origin, or every box read from that
    patch would be shifted by a crop that never happened.
    """
    h, w = img.shape[:2]
    x1 = max(0, int(roi[0] * w))
    y1 = max(0, int(roi[1] * h))
    x2 = min(w, int(roi[2] * w))
    y2 = min(h, int(roi[3] * h))
    if x2 <= x1 or y2 <= y1:
        return img, 0, 0
    return img[y1:y2, x1:x2], x1, y1


def _prepare_patches(
    img: np.ndarray, config: OCRConfig
) -> List[Tuple[np.ndarray, int, int]]:
    """Return the upscaled patches to OCR, each with its frame-space offset.

    Upscaling is applied *after* cropping so small ROIs get the full benefit.
    The offset is the crop's origin in the original frame; combined with
    `upscale_factor` it is what maps a box back out of patch space.
    """
    crops: List[Tuple[np.ndarray, int, int]] = (
        [_apply_roi(img, box) for box in config.roi]
        if config.roi else [(img, 0, 0)]
    )
    return [
        (_upscale(p, config.upscale_factor), ox, oy) for p, ox, oy in crops
    ]


# ---------------------------------------------------------------------------
# Per-image OCR
# ---------------------------------------------------------------------------

def ocr_lines(
    img: np.ndarray, config: OCRConfig, verbose: bool = False
) -> Optional[List[OCRLine]]:
    """OCR one BGR frame into lines, with boxes in original frame coordinates.

    Returns `None` when the engine itself failed, and `[]` when it ran fine
    and the frame genuinely has no legible text. That distinction is
    load-bearing for `cuts.redact`: "OCR broke" must never be mistaken for
    "there is nothing here to redact", because the first means we cannot
    vouch for the frame and the second means we can.

    Lines are returned unfiltered by duplication — dedup is a text-assembly
    concern (see `join_lines`), and a redactor needs every box, including the
    duplicate one an overlapping ROI produced.
    """
    try:
        engine = _get_engine()
        scale = max(1, config.upscale_factor)
        out: List[OCRLine] = []
        for patch, off_x, off_y in _prepare_patches(img, config):
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
                if not text or score < config.min_confidence:
                    continue
                # The engine's quad is in patch space: cropped, then upscaled.
                # Undo both, in that order, to land back in frame coordinates.
                pts = np.asarray(row[0], dtype=float).reshape(-1, 2)
                xs, ys = pts[:, 0] / scale + off_x, pts[:, 1] / scale + off_y
                out.append(OCRLine(
                    text=text,
                    box=(int(xs.min()), int(ys.min()),
                         int(np.ceil(xs.max())), int(np.ceil(ys.max()))),
                    score=score,
                ))
        return out
    except Exception as exc:
        if verbose:
            print(f"  [ocr] grounding failed: {exc!r}")
        return None


def join_lines(lines: List[OCRLine], config: OCRConfig) -> str:
    """Render OCR lines into the grounding string handed to the VLM call.

    Duplicate lines within a single frame are collapsed (OCR often reports
    the same string from overlapping ROI crops), preserving first-seen order.
    The joined result is capped at `config.max_chars` — this is a grounding
    hint for a one-sentence description, not a transcript, so a dense screen
    does not need to be passed through in full.
    """
    seen = set()
    kept: List[str] = []
    for line in lines:
        key = line.text.lower()
        if key and key not in seen:
            seen.add(key)
            kept.append(line.text)
    text = "\n".join(kept)
    if config.max_chars > 0 and len(text) > config.max_chars:
        text = text[: config.max_chars]
    return text


def ocr_image(img: np.ndarray, config: OCRConfig, verbose: bool = False) -> str:
    """OCR one BGR frame and return its text as newline-joined lines.

    Returns "" on any engine failure rather than raising: this is a
    best-effort hint for the VLM call, not a required input, and one bad
    frame should not abort a run that would otherwise complete fine.

    Callers that upload the frame should use `cuts.redact.scrub_frame`
    instead — it needs the boxes this discards.
    """
    lines = ocr_lines(img, config, verbose=verbose)
    return join_lines(lines, config) if lines else ""


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
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

    t0 = time.time()
    for i, sf in enumerate(iter_sampled_frames(sys.argv[1], interval_sec=interval)):
        t1 = time.time()
        text = ocr_image(sf.image, cfg.ocr, verbose=True)
        lines = text.split("\n") if text else []
        print(f"\n[t={sf.time_sec:6.1f}s] {len(lines)} lines "
              f"({time.time() - t1:.2f}s)")
        for ln in lines[:6]:
            print(f"    {ln!r}")
        if i >= 5:
            break
    print(f"\ntotal {time.time() - t0:.1f}s")
