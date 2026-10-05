"""Command-line interface.

    python -m cuts chapters <video...> [options]  # video(s) -> chapter list
    python -m cuts probe <video...>               # what a run would cost
    python -m cuts observe <video...>             # Pass 1 -> observations.jsonl
    python -m cuts coalesce <observations.jsonl>  # Pass 2 -> events.json
    python -m cuts label <events.json> [backend]  # (re)label an existing index

`chapters` is the primary command and the product's whole surface today: point
it at a recording, get back a timestamped chapter list you can paste into a
description or an editor's marker track. It runs both passes.

`observe` and `coalesce` are those two passes on their own, and splitting them
is what makes the system workable in practice. Observing is the expensive half
and does not depend on how the footage is later carved up, so it is cached;
coalescing is one call over that cache. Iterating on chapter quality — a
different `--intent`, a different model — then costs one call instead of a
full re-analysis.

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
import time
from pathlib import Path
from typing import List

from .config import CutsConfig
from .schema import format_timestamp, to_chapter_list, write_events


def _add_observe_args(p: argparse.ArgumentParser) -> None:
    """Pass 1 knobs — shared by `chapters` and the standalone `observe`."""
    p.add_argument("--interval", type=float, default=None,
                   help="seconds between sampled frames (default 30; lower "
                        "for fast-changing footage, at the cost of more "
                        "VLM calls)")
    p.add_argument("--shard-sec", type=float, default=None,
                   help="seconds of timeline per independent shard "
                        "(default 300)")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel shard workers (default: CPU count)")
    p.add_argument("--executor", choices=["thread", "process", "serial"],
                   default=None,
                   help="how shards are run (default thread; process is the "
                        "rehearsal for containers, serial is for debugging)")
    p.add_argument("--no-seek", action="store_true",
                   help="decode sequentially instead of seeking to each "
                        "sample. Much slower on long input; only needed for "
                        "footage where seeking is unreliable")
    p.add_argument("--no-cache", action="store_true",
                   help="do not read or write <video>.observations.jsonl")
    p.add_argument("--no-vlm", action="store_true",
                   help="disable the VLM text channel (visual signal only; "
                        "also skips boundary refinement, which depends on it)")
    p.add_argument("--vlm-model", default=None,
                   help="model for per-frame screen descriptions "
                        "(default claude-haiku-4-5)")
    p.add_argument("--no-redact", action="store_true",
                   help="do NOT black out credentials (API keys, tokens, "
                        "passwords) found in frames before they are uploaded. "
                        "Only for footage you know is clean, or when a false "
                        "positive is hiding something you need to see")
    p.add_argument("--no-asr", action="store_true",
                   help="skip transcription even when the video has audio")
    p.add_argument("-v", "--verbose", action="store_true")


def _add_coalesce_args(p: argparse.ArgumentParser) -> None:
    """Pass 2 knobs — shared by `chapters` and the standalone `coalesce`."""
    p.add_argument("--intent", default=None,
                   help="what this index is FOR (\"a tutorial walkthrough\", "
                        "\"find where I broke the build\"). Coalescing judges "
                        "what deserves its own chapter against this — without "
                        "it, \"boring\" can only mean \"visually similar\"")
    p.add_argument("--chapters", type=int, default=None,
                   help="target number of chapters. Only the curve segmenter "
                        "honours this; the LLM segmenter decides the count "
                        "from the content")
    p.add_argument("--min-chapter", type=float, default=None,
                   help="minimum chapter length in seconds (default 20)")
    p.add_argument("--include-ocr", action="store_true",
                   help="send per-frame OCR text to the coalescing call. "
                        "Roughly 8x the prompt size; worth it when titles "
                        "need exact on-screen strings")
    p.add_argument("--coalesce-model", default=None,
                   help="model for the coalescing call (default claude-opus-5)")


def _add_common(p: argparse.ArgumentParser) -> None:
    _add_observe_args(p)
    _add_coalesce_args(p)
    p.add_argument("--segmenter", choices=["llm", "curve"], default=None,
                   help="how chapter boundaries are decided. 'llm' (default) "
                        "reads every frame description and coalesces by "
                        "judgement; 'curve' uses the DINOv2+TF-IDF similarity "
                        "curve and a count budget — deterministic, and the "
                        "baseline to compare against")
    p.add_argument("--vlm-stride", type=int, default=None,
                   help="describe every Nth sample instead of every one "
                        "(default 1); skipped samples inherit the last "
                        "description. Curve segmenter only")
    p.add_argument("--no-refine", action="store_true",
                   help="skip the boundary-refinement pass (one extra VLM "
                        "look at each detected boundary's midpoint)")
    p.add_argument("--no-dino", action="store_true",
                   help="disable the DINOv2 visual channel. Curve segmenter "
                        "only — the LLM segmenter never runs DINOv2")
    p.add_argument("--backend", choices=["auto", "claude", "heuristic"],
                   default="auto",
                   help="chapter-title backend (default auto: Claude when "
                        "credentials are present, else heuristic). Curve "
                        "segmenter only; the LLM segmenter titles as it cuts")
    p.add_argument("--no-labels", action="store_true",
                   help="segment only; skip titling entirely")
    p.add_argument("--no-join-boundaries", action="store_true",
                   help="with multiple inputs, do NOT force a chapter break "
                        "at each file join (default: force one, since a new "
                        "file is a real break in the recording)")
    p.add_argument("-o", "--out", default=None,
                   help="write the full index JSON here "
                        "(default: <first_video_stem>.events.json beside it)")


def _apply_observe_args(cfg: CutsConfig, args) -> None:
    """Pass 1 flags. Shared so `observe` and `chapters` cannot drift apart."""
    cfg.verbose = getattr(args, "verbose", False)
    if getattr(args, "interval", None) is not None:
        cfg.sampling.sample_interval_sec = args.interval
    if getattr(args, "shard_sec", None) is not None:
        cfg.observe.shard_sec = args.shard_sec
    if getattr(args, "workers", None) is not None:
        cfg.observe.workers = max(0, args.workers)
    if getattr(args, "executor", None):
        cfg.observe.executor = args.executor
    if getattr(args, "no_seek", False):
        cfg.observe.seek = False
    if getattr(args, "no_cache", False):
        cfg.observe.cache = False
    if getattr(args, "no_vlm", False):
        cfg.vlm.enabled = False
        cfg.segmentation.weight_vlm = 0.0
    if getattr(args, "vlm_model", None):
        cfg.vlm.model = args.vlm_model
    if getattr(args, "no_redact", False):
        cfg.redact.enabled = False
    if getattr(args, "no_asr", False):
        cfg.asr.enabled = False


def _apply_coalesce_args(cfg: CutsConfig, args) -> None:
    """Pass 2 flags. Shared between `coalesce` and `chapters`."""
    if getattr(args, "intent", None):
        cfg.coalesce.intent = args.intent
    if getattr(args, "include_ocr", False):
        cfg.coalesce.include_ocr = True
    if getattr(args, "coalesce_model", None):
        cfg.coalesce.model = args.coalesce_model
    if getattr(args, "chapters", None) is not None:
        cfg.segmentation.target_events = args.chapters
    if getattr(args, "min_chapter", None) is not None:
        cfg.segmentation.min_event_sec = args.min_chapter


def _config_from_args(args) -> CutsConfig:
    cfg = CutsConfig()
    _apply_observe_args(cfg, args)
    _apply_coalesce_args(cfg, args)
    if getattr(args, "segmenter", None):
        cfg.segmenter = args.segmenter
    if args.vlm_stride is not None:
        cfg.vlm.stride = max(1, args.vlm_stride)
    if args.no_refine:
        cfg.refine.enabled = False
    if args.no_dino:
        cfg.segmentation.weight_dino = 0.0
    if args.no_labels:
        cfg.labeling.enabled = False
        cfg.coalesce.enabled = False
    if cfg.segmenter == "llm" and args.chapters is not None:
        print("[cli] --chapters is ignored by the LLM segmenter, which "
              "decides the count from the content; use --segmenter curve "
              "to force a target")
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

    from .config import CutsConfig
    cfg = CutsConfig()
    n_samples = int(total / cfg.sampling.sample_interval_sec)
    # Decode is linear in total video length *regardless* of --interval:
    # PyAV can only reliably seek to keyframes, so every frame gets decoded
    # even when only one in thirty is kept (see cuts/media.py). ~4s/video-min
    # for decode+DINO and ~1.5s/call for a VLM description are both rough
    # guesses extrapolated from a short clip, not measurements at this scale.
    decode_est_min = mins * 4 / 60
    vlm_est_min = n_samples * 1.5 / max(1, cfg.vlm.concurrency) / 60
    print(f"~{n_samples} samples at the default "
          f"{cfg.sampling.sample_interval_sec:.0f}s interval")
    print(f"estimated runtime: ~{decode_est_min + vlm_est_min:.0f} min "
          f"(~{decode_est_min:.0f} min decode+DINO, roughly flat regardless "
          f"of --interval, + ~{vlm_est_min:.0f} min of VLM calls at "
          f"concurrency={cfg.vlm.concurrency})")
    print("(rough order-of-magnitude estimate from a short clip and guessed "
          "per-call latency, not a measurement at this scale)")
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
                     "segmenter": result.segmenter,
                     "intent": cfg.coalesce.intent,
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


def cmd_observe(args) -> int:
    """Pass 1 alone: video(s) -> observations.jsonl.

    Worth having as its own command for two reasons. It is the expensive half
    of a run, so materializing it lets Pass 2 be re-run against a cache for
    the price of one call. And it is the unit that eventually moves into a
    container — being able to invoke it directly is what makes that testable.
    """
    videos = _resolve_videos(args.video)

    from .media import Timeline
    from .observe import observe, observations_path

    cfg = CutsConfig()
    _apply_observe_args(cfg, args)

    timeline = Timeline.build(videos)
    out_path = args.out or observations_path(videos[0])
    t0 = time.time()
    # Goes through `observe` rather than `observe_timeline` so this honours the
    # same cache `chapters` does — running it twice, or running it and then
    # running `chapters`, must not pay for Pass 1 twice.
    observations = observe(timeline, cfg, verbose=args.verbose,
                           cache_path=out_path)
    if not observations:
        print("error: no observations produced", file=sys.stderr)
        return 1

    elapsed = time.time() - t0
    described = sum(1 for o in observations if o.description)
    withheld = sum(1 for o in observations if o.withheld)
    print(f"\n{len(observations)} observations across "
          f"{format_timestamp(timeline.duration_sec)} "
          f"({described} described"
          + (f", {withheld} withheld" if withheld else "") + ")")
    print(f"{elapsed:.1f}s total "
          f"({elapsed / max(timeline.duration_sec / 60.0, 1e-6):.1f}s per "
          f"video-minute)")
    print(f"observations: {out_path}")
    return 0


def cmd_coalesce(args) -> int:
    """Pass 2 alone: observations.jsonl -> events.json.

    This is the iteration loop. Coalescing is one call over cached text, so
    trying a different intent, model or prompt costs cents and seconds rather
    than a re-analysis of the footage.
    """
    path = str(Path(args.observations).resolve())
    if not os.path.exists(path):
        print(f"error: not found: {path}", file=sys.stderr)
        return 2

    from .coalesce import coalesce
    from .media import Timeline
    from .observe import read_observations

    cfg = CutsConfig()
    cfg.verbose = args.verbose
    _apply_coalesce_args(cfg, args)

    observations, meta = read_observations(path)
    if not observations:
        print(f"error: no observations in {path}", file=sys.stderr)
        return 1

    files = [f[0] for f in meta.get("files", []) if os.path.exists(f[0])]
    timeline = Timeline.build(files) if files else None
    duration = timeline.duration_sec if timeline else observations[-1].t

    locked: List[int] = []
    seen = set()
    for i, o in enumerate(observations):
        if o.source_index not in seen:
            seen.add(o.source_index)
            locked.append(i)

    events, backend = coalesce(
        observations, cfg.coalesce, cfg.segmentation,
        video_id=Path(path).stem.replace(".observations", ""),
        duration_sec=duration, timeline=timeline, locked=locked[1:],
        verbose=args.verbose,
    )
    if not events:
        print("error: coalescing produced no chapters", file=sys.stderr)
        return 1

    if timeline is not None:
        from .pipeline import _attach_sources
        _attach_sources(events, timeline)

    out_path = args.out or str(Path(path).with_name(
        Path(path).stem.replace(".observations", "") + ".events.json"))
    write_events(out_path, Path(out_path).stem, files[0] if files else "",
                 events, duration_sec=duration,
                 sources=timeline.to_dicts() if timeline else None,
                 extra={"segmenter": "llm", "intent": cfg.coalesce.intent,
                        "n_observations": len(observations)})

    print()
    print(to_chapter_list(events, show_source=bool(timeline and timeline.is_multi)))
    print(f"\n{len(events)} chapters from {len(observations)} observations "
          f"({backend})")
    for e in events:
        print(f"  {e.timestamp():>7} [{e.metadata.get('kind', '?'):>7}] "
              f"{e.duration_sec / 60:5.1f}min  {e.summary}")
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
    if args.no_redact:
        cfg.redact.enabled = False
    payload = read_events(path)
    events = payload["events"]
    timeline = Timeline.from_dicts(payload["sources"])
    # Labeling re-decodes representative frames and uploads them, so this
    # standalone path needs the same redaction the full pipeline applies.
    used = label_events(events, timeline, cfg.labeling,
                        backend=args.backend, verbose=args.verbose,
                        ocr_config=cfg.ocr, redact_config=cfg.redact)
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

    p_ob = sub.add_parser(
        "observe",
        help="Pass 1 only: video(s) -> observations.jsonl")
    p_ob.add_argument("video", nargs="+")
    _add_observe_args(p_ob)
    p_ob.add_argument("-o", "--out", default=None,
                      help="write observations here (default: "
                           "<first_video_stem>.observations.jsonl)")
    p_ob.set_defaults(func=cmd_observe)

    p_co = sub.add_parser(
        "coalesce",
        help="Pass 2 only: observations.jsonl -> events.json")
    p_co.add_argument("observations")
    _add_coalesce_args(p_co)
    p_co.add_argument("-o", "--out", default=None,
                      help="write the index JSON here")
    p_co.add_argument("-v", "--verbose", action="store_true")
    p_co.set_defaults(func=cmd_coalesce)

    p_lb = sub.add_parser("label", help="(re)label an existing events.json")
    p_lb.add_argument("events_json")
    p_lb.add_argument("--backend", choices=["auto", "claude", "heuristic"],
                      default="auto")
    p_lb.add_argument("--no-redact", action="store_true",
                      help="do NOT black out credentials in the frames sent "
                           "with the labeling call")
    p_lb.add_argument("-v", "--verbose", action="store_true")
    p_lb.set_defaults(func=cmd_label)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
