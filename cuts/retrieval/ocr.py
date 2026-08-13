"""OCR over representative frames — v2.

Key improvements over v1:
  * Decodes full-resolution frames directly from the video in a single PyAV
    pass, not from the saved representative JPEGs. Eliminates JPEG compression
    artefacts which materially hurt OCR on small on-screen text.
  * Optional upscaling (2–3×, INTER_CUBIC) applied before OCR. Strongly
    recommended for 1080p screen recordings where terminal/editor text is
    small relative to the canvas.
  * Configurable ROI crop regions (fractional (x1,y1,x2,y2) boxes) so
    taskbars, window chrome, or irrelevant screen areas are stripped before
    OCR. Each ROI is upscaled independently and OCR'd in one call.
  * Returns per-line confidence scores; stored in segment.metadata["ocr_lines"]
    as {text, max_confidence, avg_confidence, frame_count, kept}.
  * Cross-frame frequency filtering: a line must appear in >= N sampled frames
    to survive into ocr_text (the cleaned, indexed text). Lines below the
    threshold are still preserved in ocr_text_raw for debugging. High-confidence
    lines bypass the filter regardless of frame count.
  * Stores both ocr_text (cleaned, for BM25/embedding indexing) and
    ocr_text_raw (all lines above min_conf, for debugging).
  * Optional PNG saving of the exact images fed to OCR under
    <index_dir>/ocr_debug/ — one file per frame per ROI.
  * config.ocr_debug enables verbose per-segment, per-frame line breakdown.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import av
import cv2
import numpy as np

from ..config import RetrievalConfig
from .schema import SegmentRecord


# Lazy singletons — OCR engine init is slow on first call (downloads ONNX weights).
_RAPIDOCR = None
_IMAGEHASH = None


def _get_ocr():
    global _RAPIDOCR
    if _RAPIDOCR is None:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
        _RAPIDOCR = RapidOCR()
    return _RAPIDOCR


def _get_imagehash():
    global _IMAGEHASH
    if _IMAGEHASH is None:
        import imagehash  # type: ignore
        from PIL import Image  # type: ignore
        _IMAGEHASH = (imagehash, Image)
    return _IMAGEHASH


def _phash(img_bgr: np.ndarray):
    imagehash, Image = _get_imagehash()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


# ---------------------------------------------------------------------------
# Image pre-processing helpers
# ---------------------------------------------------------------------------

def _upscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Upscale by integer factor with bicubic interpolation. 1 = no-op."""
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w * factor, h * factor),
                      interpolation=cv2.INTER_CUBIC)


def _apply_roi(img: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
    """Crop a fractional (x1, y1, x2, y2) box from a BGR image.

    Returns the full image on degenerate boxes (x2<=x1 or y2<=y1).
    """
    h, w = img.shape[:2]
    x1 = max(0, int(roi[0] * w))
    y1 = max(0, int(roi[1] * h))
    x2 = min(w, int(roi[2] * w))
    y2 = min(h, int(roi[3] * h))
    if x2 <= x1 or y2 <= y1:
        return img
    return img[y1:y2, x1:x2]


def _prepare_ocr_inputs(
    img_bgr: np.ndarray,
    config: RetrievalConfig,
) -> List[np.ndarray]:
    """Return the list of upscaled image patches to OCR for one decoded frame.

    When ROIs are configured: one upscaled patch per ROI.
    When no ROIs: one upscaled full-frame patch.
    The upscale step is applied after cropping so small crops benefit fully.
    """
    rois = config.ocr_roi
    patches = [_apply_roi(img_bgr, box) for box in rois] if rois else [img_bgr]
    factor = config.ocr_upscale_factor
    return [_upscale(p, factor) for p in patches]


def _save_debug_pngs(
    patches: List[np.ndarray],
    debug_dir: str,
    seg_id: str,
    frame_k: int,
) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    for ri, patch in enumerate(patches):
        fname = f"{seg_id}_{frame_k:02d}_roi{ri}.png"
        cv2.imwrite(os.path.join(debug_dir, fname), patch)


# ---------------------------------------------------------------------------
# Per-image OCR
# ---------------------------------------------------------------------------

def _ocr_image(img_bgr: np.ndarray, min_conf: float) -> List[Tuple[str, float]]:
    """Run OCR on one image; return (text, confidence) pairs above min_conf."""
    engine = _get_ocr()
    result, _ = engine(img_bgr)
    out: List[Tuple[str, float]] = []
    if not result:
        return out
    for row in result:
        if len(row) < 3:
            continue
        text, score = row[1], row[2]
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        text = (text or "").strip()
        if text and score >= min_conf:
            out.append((text, score))
    return out


# ---------------------------------------------------------------------------
# Cross-frame aggregation
# ---------------------------------------------------------------------------

def _merge_frame_results(
    frame_results: List[List[Tuple[str, float]]],
    config: RetrievalConfig,
) -> Tuple[str, str, List[dict]]:
    """Aggregate per-frame OCR results into cleaned and raw text for one segment.

    Parameters
    ----------
    frame_results:
        One list per sampled frame, each containing (text, confidence) pairs.
    config:
        Uses ``ocr_token_min_frames`` and ``ocr_high_conf_threshold``.

    Returns
    -------
    ocr_text : str
        Cleaned: only lines that pass the cross-frame frequency filter.
        This is what gets indexed for BM25 / embeddings.
    ocr_text_raw : str
        All unique lines above min_conf from any frame, sorted by frequency
        descending. For debugging / display.
    ocr_lines : list[dict]
        Per-line metadata: text, max_confidence, avg_confidence,
        frame_count, kept.
    """
    n_frames = len(frame_results)
    # OrderedDict preserves first-seen order; key = lowercased line.
    line_info: "OrderedDict[str, dict]" = OrderedDict()

    for frame_lines in frame_results:
        seen_in_frame: Set[str] = set()
        for text, score in frame_lines:
            key = text.strip().lower()
            if not key or key in seen_in_frame:
                continue
            seen_in_frame.add(key)
            if key not in line_info:
                line_info[key] = {"text": text.strip(), "scores": [], "frame_count": 0}
            line_info[key]["scores"].append(score)
            line_info[key]["frame_count"] += 1

    if not line_info:
        return "", "", []

    # Sort by frequency desc, then max confidence desc for stable ranking.
    sorted_info = sorted(
        line_info.values(),
        key=lambda x: (-x["frame_count"], -max(x["scores"])),
    )

    # How many frames must a line appear in to be "kept" in ocr_text?
    min_frames = max(1, min(config.ocr_token_min_frames, n_frames))
    high_conf = config.ocr_high_conf_threshold

    raw_lines: List[str] = []
    cleaned_lines: List[str] = []
    ocr_lines_meta: List[dict] = []

    for info in sorted_info:
        max_score = max(info["scores"])
        avg_score = sum(info["scores"]) / len(info["scores"])
        passes = (info["frame_count"] >= min_frames or max_score >= high_conf)
        raw_lines.append(info["text"])
        if passes:
            cleaned_lines.append(info["text"])
        ocr_lines_meta.append({
            "text": info["text"],
            "max_confidence": round(max_score, 4),
            "avg_confidence": round(avg_score, 4),
            "frame_count": info["frame_count"],
            "kept": passes,
        })

    return "\n".join(cleaned_lines), "\n".join(raw_lines), ocr_lines_meta


# ---------------------------------------------------------------------------
# Full-resolution frame collection (single decode pass)
# ---------------------------------------------------------------------------

def _collect_ocr_frames(
    video_path: str,
    segments: List[SegmentRecord],
) -> Dict[int, np.ndarray]:
    """Decode full-resolution BGR frames needed by all segments in one pass.

    Collects the union of all frame indices referenced in segment metadata,
    decodes them at native resolution (no resize), and returns a mapping
    frame_idx -> BGR ndarray. This is much better for OCR quality than reading
    the saved representative JPEGs which were downscaled to 720p.
    """
    needed: Set[int] = set()
    for seg in segments:
        for fidx in seg.metadata.get("frame_indices", []):
            needed.add(int(fidx))
    if not needed:
        return {}

    global_end = max(needed)
    frames: Dict[int, np.ndarray] = {}
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i > global_end:
                break
            if i in needed:
                frames[i] = frame.to_ndarray(format="bgr24")
    finally:
        container.close()
    return frames


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_ocr(
    video_path: str,
    segments: List[SegmentRecord],
    index_dir: str,
    config: RetrievalConfig,
    verbose: bool = False,
) -> None:
    """Run OCR on every segment's representative frames, in place.

    Decodes full-resolution frames from the video in a single PyAV pass,
    applies optional upscaling and ROI cropping, then runs OCR per patch.
    Mutates each segment:
      - ``ocr_text``              — cleaned text (cross-frame filtered), for indexing
      - ``ocr_text_raw``          — all unique lines above min_conf, for debugging
      - ``metadata["ocr_lines"]`` — per-line {text, confidences, frame_count, kept}
      - ``metadata["ocr_stale"]`` — True when text was reused via pHash dedup
    """
    if not segments:
        return

    debug = config.ocr_debug
    save_debug = config.ocr_save_debug_png or debug
    debug_dir = os.path.join(index_dir, "ocr_debug") if save_debug else None

    if verbose:
        print(f"  OCR: decoding full-resolution frames from video ...")
    all_frames = _collect_ocr_frames(video_path, segments)

    last_phash = None
    last_text = ""
    last_text_raw = ""
    total_frames = 0
    ocr_calls = 0
    fresh_segments = 0

    for seg in segments:
        seg.metadata.setdefault("ocr_stale", False)
        frame_indices: List[int] = [
            int(f) for f in seg.metadata.get("frame_indices", [])
        ]

        if not frame_indices:
            seg.ocr_text = ""
            seg.ocr_text_raw = ""
            seg.metadata["ocr_lines"] = []
            continue

        imgs = [all_frames[fi] for fi in frame_indices if fi in all_frames]
        if not imgs:
            seg.ocr_text = ""
            seg.ocr_text_raw = ""
            seg.metadata["ocr_lines"] = []
            continue

        total_frames += len(imgs)

        # pHash dedup: compare first frame of this segment to last frame of
        # previous segment. If perceptually identical, inherit the text.
        this_first_phash = _phash(imgs[0])
        dedup = False
        if last_phash is not None:
            dist = this_first_phash - last_phash
            if dist <= config.ocr_phash_dedup_threshold and last_text:
                dedup = True

        if dedup:
            seg.ocr_text = last_text
            seg.ocr_text_raw = last_text_raw
            seg.metadata["ocr_stale"] = True
            seg.metadata["ocr_lines"] = []
            if debug:
                print(f"  [{seg.segment_id}] pHash dedup (dist={dist}) "
                      f"— reusing {len(last_text)} chars from previous segment")
        else:
            fresh_segments += 1
            if debug:
                print(f"  [{seg.segment_id}] OCR on {len(imgs)} frame(s):")

            frame_results: List[List[Tuple[str, float]]] = []
            for k, (fi, img) in enumerate(zip(frame_indices, imgs)):
                patches = _prepare_ocr_inputs(img, config)

                if debug_dir is not None:
                    _save_debug_pngs(patches, debug_dir, seg.segment_id, k)

                frame_lines: List[Tuple[str, float]] = []
                for patch in patches:
                    frame_lines.extend(_ocr_image(patch, config.ocr_min_confidence))
                    ocr_calls += 1
                frame_results.append(frame_lines)

                if debug:
                    h, w = img.shape[:2]
                    print(f"    frame {fi} ({w}×{h} → "
                          f"{w * config.ocr_upscale_factor}×"
                          f"{h * config.ocr_upscale_factor}): "
                          f"{len(frame_lines)} line(s)")
                    for text, score in frame_lines:
                        print(f"      [{score:.3f}] {text!r}")

            cleaned, raw, ocr_lines_meta = _merge_frame_results(
                frame_results, config
            )
            seg.ocr_text = cleaned
            seg.ocr_text_raw = raw
            seg.metadata["ocr_lines"] = ocr_lines_meta

            if debug:
                kept = sum(1 for ln in ocr_lines_meta if ln["kept"])
                dropped = len(ocr_lines_meta) - kept
                print(f"    → kept {kept} lines in ocr_text, "
                      f"filtered {dropped} (cross-frame rule)")

        last_phash = _phash(imgs[-1])
        last_text = seg.ocr_text
        last_text_raw = seg.ocr_text_raw

    if verbose:
        stale = sum(1 for s in segments if s.metadata.get("ocr_stale"))
        print(f"  OCR: {ocr_calls} engine calls, {fresh_segments} fresh segments, "
              f"{stale} reused via pHash, {total_frames} frames decoded")
