"""Visualizer: timeline plot with GT vs predicted boundaries + thumbnails.

Goal is FAST visual debugging, not publication graphics. We use matplotlib and
pull thumbnails directly from the video via PyAV.

Output: a single PNG per video showing:
  * X axis = frame index
  * GT boundaries drawn as green vertical bars (intervals shaded for gradual/UI)
  * Predicted boundaries drawn as red vertical bars (intervals shaded too)
  * Thumbnails of the predicted cut frames stacked along the top
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from cuts.benchmark.schema import Annotation
from cuts.frame_extractor import extract_window


def visualize(
    video_path: str,
    predictions: Iterable[object],
    annotations: List[Annotation],
    output_png: str,
    n_frames_total: int,
    max_thumbnails: int = 12,
    title: Optional[str] = None,
) -> None:
    """Render a debug timeline + thumbnails to `output_png`."""
    preds = list(predictions)

    # Two-axis figure: top axis = thumbnails strip, bottom axis = timeline.
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.05)
    ax_thumbs = fig.add_subplot(gs[0])
    ax_timeline = fig.add_subplot(gs[1])

    # ---- Timeline -----------------------------------------------------------
    ax_timeline.set_xlim(0, max(1, n_frames_total - 1))
    ax_timeline.set_ylim(0, 1)
    ax_timeline.set_yticks([])
    ax_timeline.set_xlabel("frame index")

    # GT first (green). Use shaded rect for non-instantaneous, line for hard cuts.
    for a in annotations:
        if a.start_frame == a.end_frame:
            ax_timeline.axvline(a.start_frame, color="green", linewidth=1.2, alpha=0.85)
        else:
            ax_timeline.axvspan(a.start_frame, a.end_frame, color="green", alpha=0.25)

    # Predictions (red).
    for p in preds:
        s = int(getattr(p, "frame_idx"))
        e = int(getattr(p, "end_frame_idx", s))
        if s == e:
            ax_timeline.axvline(s, color="red", linewidth=1.0, alpha=0.85)
        else:
            ax_timeline.axvspan(s, e, color="red", alpha=0.25)

    # Manual legend (avoid auto-collision with shaded spans).
    ax_timeline.plot([], [], color="green", label="GT")
    ax_timeline.plot([], [], color="red", label="pred")
    ax_timeline.legend(loc="upper right", fontsize=8)

    # ---- Thumbnail strip ---------------------------------------------------
    # Pick up to `max_thumbnails` predicted cut frames evenly spaced through preds.
    if preds:
        idxs = np.linspace(0, len(preds) - 1, num=min(max_thumbnails, len(preds))).astype(int)
        chosen = [preds[i] for i in idxs]
    else:
        chosen = []

    ax_thumbs.set_xlim(0, max(1, n_frames_total - 1))
    ax_thumbs.set_ylim(0, 1)
    ax_thumbs.axis("off")

    # We need 1 frame per thumbnail. Cheapest: extract_window with W=0.
    for p in chosen:
        f_idx = int(getattr(p, "frame_idx"))
        try:
            window = extract_window(video_path, f_idx, half_window=0)
        except Exception:
            continue
        if not window:
            continue
        img_bgr = window[0].image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Downsize the thumbnail to keep the figure light.
        h, w = img_rgb.shape[:2]
        target_h = 80
        target_w = max(1, int(w * (target_h / h)))
        thumb = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Place the thumbnail centered on its frame index, using axes-coords
        # offsets for height (the X axis is in frame coordinates).
        thumb_width_frames = max(1, n_frames_total // (max_thumbnails * 2))
        ax_thumbs.imshow(
            thumb,
            extent=(f_idx - thumb_width_frames / 2, f_idx + thumb_width_frames / 2, 0.0, 1.0),
            aspect="auto",
        )
        # Tick label at the bottom of each thumbnail.
        ax_thumbs.text(f_idx, -0.05, str(f_idx), ha="center", va="top", fontsize=7)

    if title:
        fig.suptitle(title, fontsize=10)

    fig.savefig(output_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Standalone debug: synthetic-only demo (won't load real video).
    import sys

    if len(sys.argv) < 3:
        print("usage: python -m cuts.benchmark.visualizer <video_path> <output_png>")
        sys.exit(1)
    from cuts.frame_extractor import build_frame_index
    path = sys.argv[1]
    out = sys.argv[2]
    n = len(build_frame_index(path))
    visualize(path, predictions=[], annotations=[], output_png=out, n_frames_total=n,
              title="empty timeline (sanity)")
    print(f"wrote {out}")
