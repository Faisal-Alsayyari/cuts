"""Command-line interface.

    python -m cuts chapters <video...> [options]  # video(s) -> chapter list
    python -m cuts probe <video...>               # what a run would cost
    python -m cuts label <events.json> [backend]  # (re)label an existing index

`chapters` is the primary command and the product's whole surface today: point
it at a recording, get back a timestamped chapter list you can paste into a
description or an editor's marker track.

Several files given to `chapters` are treated as one continuous recording, in
the order written on the command line — the case where a single session was
captured across multiple days. Shell globs sort lexicographically, which is
chronological only if the filenames were numbered that way, so check the
order `probe` echoes back before trusting it.
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
    p.add_argument("--no-join-boundaries", action="store_true",
                   help="with multiple inputs, do NOT force a chapter break "
                        "at each file join (default: force one, since a new "
                        "file is a real break in the recording)")
    p.add_argument("-o", "--out", default=None,
                   help="write the full index JSON here "
                        "(default: <first_video_stem>.events.json beside it)")
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


def _resolve_videos(paths) -> list:
    """Absolute paths, in the order given, erroring on the first missing one."""
    out = []
    for p in paths:
        full = str(Path(p).resolve())
        if not os.path.exists(full):
            print(f"error: video not found: {full}", file=sys.stderr)
            sys.exit(2)
        out.append(full)
    return out


def _default_out_path(videos) -> str:
    """Where the index goes when -o was not given.

    Named after the first file, with a count suffix when there are several, so
    indexing the same first file with different companions does not silently
    overwrite an earlier run.
    """
    first = Path(videos[0])
    if len(videos) == 1:
        return str(first.with_suffix(".events.json"))
    return str(first.with_name(f"{first.stem}+{len(videos) - 1}.events.json"))


def cmd_probe(args) -> int:
    """Report what the inputs are and roughly what a run will cost."""
    from .media import probe_video

    videos = _resolve_videos(args.video)
    total = 0.0
    print(f"{len(videos)} file(s), in the order they will be read:\n")
    for i, path in enumerate(videos):
        meta = probe_video(path)
        if meta["error"]:
            print(f"  {i}: {Path(path).name}  ERROR: {meta['error']}")
            continue
        dur = float(meta["duration_sec"])
        print(f"  {i}: {Path(path).name}")
        print(f"       starts at {format_timestamp(total)}  "
              f"length {format_timestamp(dur)}")
        print(f"       {meta['width']}x{meta['height']} @ {meta['fps']:.2f}fps "
              f"{meta['codec']}   audio: "
              f"{'yes' if meta['has_audio'] else 'no'}")
        total += dur

    mins = total / 60.0
    print(f"\ntimeline: {format_timestamp(total)} ({mins:.1f} min)")
    # Measured on an M3: ~27 s per video-minute with OCR on every sample,
    # ~11 s/min at stride 3. Only ever benchmarked on a short clip, so this is
    # an order-of-magnitude estimate, not a promise.
    print(f"estimated runtime: ~{mins * 27 / 60:.0f} min at default settings, "
          f"~{mins * 11 / 60:.0f} min with --ocr-stride 3")
    print("(estimates extrapolated from a short clip; treat as rough)")
    return 0


def cmd_chapters(args) -> int:
    videos = _resolve_videos(args.video)

    from .pipeline import run

    cfg = _config_from_args(args)
    result = run(videos, cfg, label_backend=args.backend,
                 join_boundaries=not args.no_join_boundaries)

    if not result.events:
        print("error: no events produced (video too short, or no usable signal)",
              file=sys.stderr)
        return 1

    out_path = args.out or _default_out_path(videos)
    sources = result.timeline.to_dicts() if result.timeline else None
    write_events(out_path, result.video_id, videos[0], result.events,
                 duration_sec=result.duration_sec,
                 sources=sources,
                 extra={
                     "n_samples": result.n_samples,
                     "channels": result.channels_used,
                     "label_backend": result.label_backend,
                     "stage_times": result.stage_times,
                 })

    multi = result.is_multi
    print()
    print(to_chapter_list(result.events, show_source=multi))
    print()
    per_min = result.runtime_sec / max(result.duration_sec / 60.0, 1e-6)
    across = f"across {format_timestamp(result.duration_sec)}"
    if multi:
        across += f" of {len(result.timeline)} files"
    print(f"{len(result.events)} chapters from {result.n_samples} samples "
          f"{across}")
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

    from .media import Timeline

    cfg = CutsConfig()
    cfg.verbose = args.verbose
    payload = read_events(path)
    events = payload["events"]
    timeline = Timeline.from_dicts(payload["sources"])
    used = label_events(events, timeline, cfg.labeling,
                        backend=args.backend, verbose=args.verbose)
    write_events(path, payload["video_id"], payload["video_path"], events,
                 duration_sec=payload.get("duration_sec", 0.0),
                 sources=payload["sources"])
    print()
    print(to_chapter_list(events, show_source=timeline.is_multi))
    print(f"\ntitles: {used}")
    return 0


def _load_dotenv() -> None:
    """Load ANTHROPIC_API_KEY (and friends) from a .env file if present.

    Looked for in the current directory and walked upward, same as most .env
    tooling. A real environment variable always wins over the file, so this
    is safe to run unconditionally.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)


def main(argv=None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(
        prog="python -m cuts",
        description="Semantic indexing and temporal navigation of long video.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ch = sub.add_parser("chapters", help="turn video(s) into a chapter list")
    p_ch.add_argument("video", nargs="+",
                      help="one or more video files, in chronological order; "
                           "several are read as one continuous recording")
    _add_common(p_ch)
    p_ch.set_defaults(func=cmd_chapters)

    p_pr = sub.add_parser("probe",
                          help="show input metadata and estimated runtime")
    p_pr.add_argument("video", nargs="+")
    p_pr.set_defaults(func=cmd_probe)

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
