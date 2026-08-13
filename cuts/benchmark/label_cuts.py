"""Ground truth labeling tool for cut refinement experiments.

Runs TransNetV2 on a video, saves a contact-sheet image for each candidate
boundary (showing ±2 frames), lets you type corrections, and writes a GT JSON.

Usage:
    python -m cuts.benchmark.label_cuts <video_path> [--output-dir <dir>]

Output:
    <output_dir>/<video_stem>.gt.json     — ground truth cut frames
    <output_dir>/thumbnails/<video_stem>/ — contact sheet PNGs per boundary
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

from cuts.config import CutsConfig
from cuts.detectors.transnetv2_detector import detect
from cuts.frame_extractor import build_frame_index, collect_windows_single_pass
from cuts.detectors.ensemble import BoundaryCandidate


# ---------------------------------------------------------------------------
# Contact-sheet generation
# ---------------------------------------------------------------------------

_THUMB_W = 320
_THUMB_H = 180
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICKNESS = 1


def _make_contact_sheet(
    frames: list,          # List[DecodedFrame] from collect_windows_single_pass
    center_frame: int,     # coarse boundary frame
    confidence: float,
    boundary_idx: int,
) -> np.ndarray:
    """Build a horizontal strip of thumbnails annotated with frame numbers.

    Shows up to 5 frames centred on the coarse boundary: [-2, -1, [0], +1, +2].
    The coarse frame is highlighted with a red border.
    """
    thumbs = []
    for f in frames:
        img = cv2.resize(f.image, (_THUMB_W, _THUMB_H), interpolation=cv2.INTER_AREA)
        # Annotate with frame number.
        label = str(f.frame_idx)
        cv2.putText(img, label, (6, _THUMB_H - 8), _FONT, _FONT_SCALE,
                    (255, 255, 255), _FONT_THICKNESS + 1, cv2.LINE_AA)
        cv2.putText(img, label, (6, _THUMB_H - 8), _FONT, _FONT_SCALE,
                    (0, 0, 0), _FONT_THICKNESS, cv2.LINE_AA)
        # Red border on the coarse boundary frame.
        if f.frame_idx == center_frame:
            cv2.rectangle(img, (0, 0), (_THUMB_W - 1, _THUMB_H - 1), (0, 0, 220), 3)
        thumbs.append(img)

    if not thumbs:
        return np.zeros((_THUMB_H, _THUMB_W, 3), dtype=np.uint8)

    sheet = np.hstack(thumbs)
    # Header bar.
    header_h = 28
    header = np.zeros((header_h, sheet.shape[1], 3), dtype=np.uint8)
    title = (f"Boundary #{boundary_idx}  coarse=frame {center_frame}"
             f"  conf={confidence:.3f}  (red border = coarse frame)")
    cv2.putText(header, title, (6, 20), _FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([header, sheet])


# ---------------------------------------------------------------------------
# Main labeling flow
# ---------------------------------------------------------------------------

def label_video(video_path: str, output_dir: str) -> None:
    stem = Path(video_path).stem
    thumb_dir = Path(output_dir) / "thumbnails" / stem
    thumb_dir.mkdir(parents=True, exist_ok=True)
    gt_path = Path(output_dir) / f"{stem}.gt.json"

    print(f"Running TransNetV2 on: {video_path}")
    cfg = CutsConfig()
    candidates: List[BoundaryCandidate] = detect(video_path, cfg.transnetv2)
    print(f"Found {len(candidates)} candidates.\n")

    if not candidates:
        print("No boundaries detected. Nothing to label.")
        _write_gt(gt_path, video_path, [])
        return

    # Decode ±2 frame windows around each candidate in one pass.
    windows = [(max(0, c.frame_idx - 2), c.frame_idx + 2) for c in candidates]
    all_frames = collect_windows_single_pass(video_path, windows)

    gt_frames: List[int] = []
    for i, (cand, frames) in enumerate(zip(candidates, all_frames)):
        # Save contact sheet.
        sheet = _make_contact_sheet(frames, cand.frame_idx, cand.confidence, i)
        sheet_path = thumb_dir / f"boundary_{i:03d}_coarse_{cand.frame_idx}.png"
        cv2.imwrite(str(sheet_path), sheet)

        # Print context and ask for correction.
        kind = "gradual" if cand.is_gradual else "hard"
        print(f"  [{i:3d}] {kind:7s}  coarse=frame {cand.frame_idx}"
              f"  conf={cand.confidence:.3f}  thumbnail: {sheet_path.name}")
        raw = input(f"         GT frame (Enter = accept {cand.frame_idx}, 's' = skip): ").strip()
        if raw.lower() == "s":
            print(f"         Skipped.")
            continue
        if raw == "":
            gt_frames.append(cand.frame_idx)
            print(f"         Accepted: {cand.frame_idx}")
        else:
            try:
                gt_frames.append(int(raw))
                print(f"         Corrected: {int(raw)}")
            except ValueError:
                print(f"         Invalid input '{raw}', accepting coarse frame.")
                gt_frames.append(cand.frame_idx)

    _write_gt(gt_path, video_path, gt_frames)
    print(f"\nGround truth saved to: {gt_path}")
    print(f"Thumbnails saved to:   {thumb_dir}")


def _write_gt(path: Path, video_path: str, cuts: List[int]) -> None:
    data = {"video_path": str(video_path), "cuts": sorted(cuts)}
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactively label TransNetV2 cut candidates to build GT JSON.")
    parser.add_argument("video_path", help="Path to the video file.")
    parser.add_argument("--output-dir", default="gt_labels",
                        help="Directory for GT JSON and thumbnails (default: gt_labels/).")
    args = parser.parse_args()
    label_video(args.video_path, args.output_dir)
