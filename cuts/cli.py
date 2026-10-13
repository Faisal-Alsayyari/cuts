"""Command-line interface.

    python -m cuts chapters <video...> [options]  # video(s) -> chapter list
    python -m cuts probe <video...>               # what a run would cost
    python -m cuts observe <video...>             # Pass 1 -> observations.jsonl
    python -m cuts coalesce <observations.jsonl>  # Pass 2 -> events.json
    python -m cuts label <events.json> [backend]  # (re)title an existing index

`chapters` is the primary command: point it at a recording, get back a
timestamped chapter list you can paste into a description or an editor's
marker track. It runs both passes.

`observe` and `coalesce` are those two passes on their own. Observing is the
expensive half and does not depend on how the footage is later carved up, so
it is cached (and resumable); `coalesce` runs everything after Pass 1 against
that cache — the same code `chapters` runs — so iterating on chapter quality
with a different `--intent` or model costs one call instead of a re-analysis.

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
from .schema import format_timestamp, read_events, to_chapter_list, write_events


def _add_observe_args(p: argparse.ArgumentParser) -> None:
    """Pass 1 knobs — shared by `chapters` and the standalone `observe`."""
    p.add_argument("--interval", type=float, default=None,
                   help="seconds between sampled frames (default 30; lower "
                        "for fast-changing footage, at the cost of more "
                        "description calls)")
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
                   help="do not read or write the observations cache")
    p.add_argument("--no-vlm", action="store_true",
                   help="do not describe frames (curve segmenter only: "
                        "visual channel alone, no refinement)")
    p.add_argument("--vlm-model", default=None,
                   help="model for per-frame screen descriptions "
                        "(default claude-haiku-4-5)")
    p.add_argument("--no-redact", action="store_true",
                   help="do NOT black out credentials (API keys, tokens, "
                        "passwords) found in frames or speech before they are "
                        "uploaded. Only for footage you know is clean")
    p.add_argument("--no-asr", action="store_true",
                   help="skip transcription even when the video has audio")
    p.add_argument("-v", "--verbose", action="store_true")


def _add_chapter_args(p: argparse.ArgumentParser) -> None:
    """Pass 2 knobs — shared by `chapters` and `coalesce`."""
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
    p.add_argument("--no-refine", action="store_true",
                   help="skip the boundary-refinement pass (one extra look "
                        "at each detected boundary's midpoint)")
    p.add_argument("--backend", choices=["auto", "claude", "heuristic"],
                   default="auto",
                   help="where titles come from when the curve segmenter drew "
                        "the chapters (including the fallback when coalescing "
                        "fails). Default auto: Claude when credentials are "
                        "present, else keyword heuristics")
    p.add_argument("--no-labels", action="store_true",
                   help="skip the separate titling call for curve chapters. "
                        "LLM chapters are titled by the call that draws them")
    p.add_argument("--no-join-boundaries", action="store_true",
                   help="with multiple inputs, do NOT force a chapter break "
                        "at each file join")
    p.add_argument("-o", "--out", default=None,
                   help="write the index JSON here")


def _add_chapters_only(p: argparse.ArgumentParser) -> None:
    p.add_argument("--segmenter", choices=["llm", "curve"], default=None,
                   help="how chapter boundaries are decided. 'llm' (default) "
                        "reads every frame description and coalesces by "
                        "judgement; 'curve' uses the DINOv2+TF-IDF similarity "
                        "curve and a count budget — deterministic, the "
                        "baseline to compare against, and the only mode that "
                        "works without API credentials")
    p.add_argument("--no-dino", action="store_true",
                   help="disable the DINOv2 visual channel (curve segmenter)")


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


def _apply_chapter_args(cfg: CutsConfig, args) -> None:
    """Pass 2 flags. Shared between `chapters` and `coalesce`."""
    cfg.verbose = cfg.verbose or getattr(args, "verbose", False)
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
    if getattr(args, "no_refine", False):
        cfg.refine.enabled = False
    if getattr(args, "no_labels", False):
        cfg.labeling.enabled = False
    if getattr(args, "segmenter", None):
        cfg.segmenter = args.segmenter
    if getattr(args, "no_dino", False):
        cfg.segmentation.weight_dino = 0.0
    if cfg.segmenter == "llm" and getattr(args, "chapters", None) is not None:
        print("[cli] --chapters is ignored by the LLM segmenter, which "
              "decides the count from the content; use --segmenter curve "
              "to force a target")


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
    overwrite an earlier run. The observation cache is named the same way.
    """
    first = Path(videos[0])
    if len(videos) == 1:
        return str(first.with_suffix(".events.json"))
    return str(first.with_name(f"{first.stem}+{len(videos) - 1}.events.json"))


def _finish(result, cfg: CutsConfig, out_path: str) -> int:
    """Write the index and print the chapter list. Shared by chapters/coalesce."""
    write_events(out_path, result.video_id, result.timeline.first_path,
                 result.events, duration_sec=result.duration_sec,
                 sources=result.timeline.to_dicts(),
                 extra=result.extra(cfg.coalesce.intent))

    print()
    print(to_chapter_list(result.events, show_source=result.is_multi))
    print()
    per_min = result.runtime_sec / max(result.duration_sec / 60.0, 1e-6)
    across = f"across {format_timestamp(result.duration_sec)}"
    if result.is_multi:
        across += f" of {len(result.timeline)} files"
    print(f"{len(result.events)} chapters from {result.n_samples} samples {across}")
    print(f"segmenter: {result.segmenter}   titles: {result.titles}   "
          f"signals: {', '.join(result.channels) or 'none'}")
    print(f"{result.runtime_sec:.1f}s total ({per_min:.1f}s per video-minute)")
    print(f"index: {out_path}")
    return 0


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

    print(f"\ntimeline: {format_timestamp(total)} ({total / 60.0:.1f} min)")

    cfg = CutsConfig()
    n_samples = int(total / cfg.sampling.sample_interval_sec) + 1
    n_calls = -(-n_samples // cfg.vlm.frames_per_call)
    # Rough per-unit costs, not measurements at scale: a seek + decode is
    # ~0.1s and OCR (2x upscale, all cores) ~1s per sample, done
    # ~workers-in-parallel but CPU-bound; a description call ~3s, overlapped
    # up to LLMConfig.max_concurrency.
    local_min = n_samples * 1.1 / 60
    api_min = n_calls * 3.0 / max(1, cfg.llm.max_concurrency) / 60
    print(f"~{n_samples} samples at the default "
          f"{cfg.sampling.sample_interval_sec:.0f}s interval, "
          f"~{n_calls} description calls")
    print(f"estimated Pass 1: ~{max(local_min, api_min):.0f}-"
          f"{local_min + api_min:.0f} min (decode+OCR ~{local_min:.0f} min, "
          f"calls ~{api_min:.0f} min at concurrency {cfg.llm.max_concurrency}; "
          f"they overlap)")
    print("(rough order-of-magnitude estimate, not a measurement; a cached "
          "re-run skips Pass 1 entirely)")
    return 0


def cmd_chapters(args) -> int:
    videos = _resolve_videos(args.video)

    from .pipeline import PipelineError, run

    cfg = CutsConfig()
    _apply_observe_args(cfg, args)
    _apply_chapter_args(cfg, args)
    try:
        result = run(videos, cfg, title_backend=args.backend,
                     join_boundaries=not args.no_join_boundaries)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return _finish(result, cfg, args.out or _default_out_path(videos))


def cmd_observe(args) -> int:
    """Pass 1 alone: video(s) -> observations.jsonl.

    Goes through the same `observe` (and cache) as `chapters`, so running it
    and then `chapters` with the same sampling settings pays for Pass 1 once.
    """
    videos = _resolve_videos(args.video)

    from . import llm
    from .media import Timeline
    from .observe import Work, observe, observations_path
    from .pipeline import _NO_CREDENTIALS

    cfg = CutsConfig()
    _apply_observe_args(cfg, args)
    llm.configure(cfg.llm.max_concurrency)
    if cfg.vlm.enabled and not llm.has_credentials():
        print(f"error: {_NO_CREDENTIALS}", file=sys.stderr)
        return 2

    timeline = Timeline.build(videos)
    out_path = args.out or observations_path(videos)
    observations = observe(
        timeline, cfg,
        Work(describe=cfg.vlm.enabled, embed=False, transcribe=cfg.asr.enabled),
        verbose=args.verbose, cache_path=out_path,
    )
    if not observations:
        print("error: no observations produced", file=sys.stderr)
        return 1
    print(f"observations: {out_path}")
    return 0


def cmd_coalesce(args) -> int:
    """Everything after Pass 1, from a cache: observations.jsonl -> events.json.

    The same code `chapters` runs after observing — coalescing, the curve
    fallback, refinement (when the videos are still on disk) — so a
    re-chapter differs from a full run only in not re-reading frames.
    """
    path = str(Path(args.observations).resolve())
    if not os.path.exists(path):
        print(f"error: not found: {path}", file=sys.stderr)
        return 2

    from . import llm
    from .media import Timeline
    from .observe import confirm_durations, read_observations, timeline_from_meta
    from .pipeline import PipelineError, chapters_from_observations

    cfg = CutsConfig()
    cfg.segmenter = "llm"
    _apply_chapter_args(cfg, args)
    llm.configure(cfg.llm.max_concurrency)

    observations, meta = read_observations(path)
    if not observations:
        print(f"error: no observations in {path}", file=sys.stderr)
        return 1
    timeline = timeline_from_meta(meta, os.path.dirname(path))
    if timeline is None:
        # A version-1 cache recorded file paths but not the timeline layout.
        files = [f[0] for f in meta.get("files", []) if os.path.exists(f[0])]
        if not files:
            print("error: this cache predates timeline layouts and its videos "
                  "are gone; re-run `chapters` on the videos", file=sys.stderr)
            return 1
        timeline = Timeline.build(files)
        confirm_durations(timeline)

    try:
        result = chapters_from_observations(
            observations, timeline, cfg, title_backend=args.backend,
            join_boundaries=not args.no_join_boundaries,
        )
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    suffix = ".observations.jsonl"
    default_out = (path[: -len(suffix)] + ".events.json" if path.endswith(suffix)
                   else str(Path(path).with_suffix(".events.json")))
    rc = _finish(result, cfg, args.out or default_out)
    for e in result.events:
        print(f"  {e.timestamp():>7} [{e.metadata.get('kind', '-'):>7}] "
              f"{e.duration_sec / 60:5.1f}min  {e.reason}")
    return rc


def cmd_label(args) -> int:
    from .labeling import label_events
    from .media import Timeline

    path = str(Path(args.events_json).resolve())
    if not os.path.exists(path):
        print(f"error: not found: {path}", file=sys.stderr)
        return 2

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
    extra = dict(payload.get("extra") or {})
    extra["titles"] = used
    write_events(path, payload["video_id"], payload["video_path"], events,
                 duration_sec=payload.get("duration_sec", 0.0),
                 sources=payload["sources"], extra=extra)
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
    _add_observe_args(p_ch)
    _add_chapter_args(p_ch)
    _add_chapters_only(p_ch)
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
                      help="write observations here (default: beside the "
                           "first video, where `chapters` looks for them)")
    p_ob.set_defaults(func=cmd_observe)

    p_co = sub.add_parser(
        "coalesce",
        help="everything after Pass 1, from a cache: observations.jsonl -> "
             "events.json")
    p_co.add_argument("observations")
    _add_chapter_args(p_co)
    p_co.add_argument("-v", "--verbose", action="store_true")
    p_co.set_defaults(func=cmd_coalesce)

    p_lb = sub.add_parser("label", help="(re)title an existing events.json")
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
