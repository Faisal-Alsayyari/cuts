"""Command-line interface.

    python -m cuts chapters <video> [options]     # video -> chapter list
    python -m cuts label <events.json> [backend]  # (re)label an existing index

`chapters` is the primary command and the product's whole surface today: point
it at a recording, get back a timestamped chapter list you can paste into a
description or an editor's marker track.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import CutsConfig
from .schema import format_timestamp, to_chapter_list, write_events


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--interval", type=float, default=None,
                   help="seconds between sampled frames (default 1.0; raise "
                        "for very long inputs)")
    p.add_argument("--chapters", type=int, default=None,
                   help="target number of chapters (default: scale with duration)")
    p.add_argument("--min-chapter", type=float, default=None,
                   help="minimum chapter length in seconds (default 20)")
    p.add_argument("--no-ocr", action="store_true",
                   help="disable the OCR text channel (visual signal only)")
    p.add_argument("--ocr-stride", type=int, default=None,
                   help="OCR every Nth sample instead of every one (default 1). "
                        "OCR dominates runtime, so this is the main speed lever "
                        "for long videos; skipped samples inherit the last read.")
    p.add_argument("--no-dino", action="store_true",
                   help="disable the DINOv2 visual channel (text signal only)")
    p.add_argument("--no-asr", action="store_true",
                   help="skip transcription even when the video has audio")
    p.add_argument("--backend", choices=["auto", "claude", "heuristic"],
                   default="auto",
                   help="chapter-title backend (default auto: Claude when "
                        "credentials are present, else heuristic)")
    p.add_argument("--no-labels", action="store_true",
                   help="segment only; skip titling entirely")
    p.add_argument("-o", "--out", default=None,
                   help="write the full index JSON here "
                        "(default: <video_stem>.events.json beside the video)")
    p.add_argument("-v", "--verbose", action="store_true")


def _config_from_args(args) -> CutsConfig:
    cfg = CutsConfig()
    cfg.verbose = args.verbose
    if args.interval is not None:
        cfg.sampling.sample_interval_sec = args.interval
    if args.chapters is not None:
        cfg.segmentation.target_events = args.chapters
    if args.min_chapter is not None:
        cfg.segmentation.min_event_sec = args.min_chapter
    if args.no_ocr:
        cfg.ocr.enabled = False
        cfg.segmentation.weight_ocr = 0.0
    if args.ocr_stride is not None:
        cfg.ocr.stride = max(1, args.ocr_stride)
    if args.no_dino:
        cfg.segmentation.weight_dino = 0.0
    if args.no_asr:
        cfg.asr.enabled = False
    if args.no_labels:
        cfg.labeling.enabled = False
    return cfg


def cmd_chapters(args) -> int:
    video = str(Path(args.video).resolve())
    if not os.path.exists(video):
        print(f"error: video not found: {video}", file=sys.stderr)
        return 2

    from .pipeline import run

    cfg = _config_from_args(args)
    result = run(video, cfg, label_backend=args.backend)

    if not result.events:
        print("error: no events produced (video too short, or no usable signal)",
              file=sys.stderr)
        return 1

    out_path = args.out or str(Path(video).with_suffix(".events.json"))
    write_events(out_path, result.video_id, video, result.events,
                 duration_sec=result.duration_sec,
                 extra={
                     "n_samples": result.n_samples,
                     "channels": result.channels_used,
                     "label_backend": result.label_backend,
                     "stage_times": result.stage_times,
                 })

    print()
    print(to_chapter_list(result.events))
    print()
    per_min = result.runtime_sec / max(result.duration_sec / 60.0, 1e-6)
    print(f"{len(result.events)} chapters from {result.n_samples} samples "
          f"across {format_timestamp(result.duration_sec)}")
    print(f"signals: {', '.join(result.channels_used) or 'none'}   "
          f"titles: {result.label_backend}")
    print(f"{result.runtime_sec:.1f}s total ({per_min:.1f}s per video-minute)")
    print(f"index: {out_path}")
    return 0


def cmd_label(args) -> int:
    from .labeling import label_events
    from .schema import read_events

    path = str(Path(args.events_json).resolve())
    if not os.path.exists(path):
        print(f"error: not found: {path}", file=sys.stderr)
        return 2

    cfg = CutsConfig()
    cfg.verbose = args.verbose
    payload = read_events(path)
    events = payload["events"]
    used = label_events(events, payload["video_path"], cfg.labeling,
                        backend=args.backend, verbose=args.verbose)
    write_events(path, payload["video_id"], payload["video_path"], events,
                 duration_sec=payload.get("duration_sec", 0.0))
    print()
    print(to_chapter_list(events))
    print(f"\ntitles: {used}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cuts",
        description="Semantic indexing and temporal navigation of long video.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ch = sub.add_parser("chapters", help="turn a video into a chapter list")
    p_ch.add_argument("video")
    _add_common(p_ch)
    p_ch.set_defaults(func=cmd_chapters)

    p_lb = sub.add_parser("label", help="(re)label an existing events.json")
    p_lb.add_argument("events_json")
    p_lb.add_argument("--backend", choices=["auto", "claude", "heuristic"],
                      default="auto")
    p_lb.add_argument("-v", "--verbose", action="store_true")
    p_lb.set_defaults(func=cmd_label)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
