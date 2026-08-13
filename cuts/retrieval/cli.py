"""CLI for the retrieval milestone.

Usage:

    python -m cuts.retrieval.cli index <video> [--out <dir>] [--asr] [--clip]
    python -m cuts.retrieval.cli search <index-dir> "<query>" [-k 5] \
                                        [--sources bm25,embedding,clip]

The ``index`` command runs:
    TransNetV2 -> segmenter -> sampler -> OCR -> (optional ASR) -> index
and writes everything under ``<out>/<video_stem>/``. Default ``<out>`` is
``./retrieval_index``.

The ``search`` command loads an index directory and prints ranked results
with timestamps, thumbnail paths, and evidence snippets.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

from ..config import CutsConfig
from ..frame_extractor import build_frame_index
from .schema import SegmentRecord, write_segments
from .segmenter import build_segments
from .sampler import sample_frames


def _derive_shots(video_path: str, config: CutsConfig, n_frames: int) -> List[Tuple[int, int]]:
    """Run TransNetV2 and convert its boundaries to (start, end) shots.

    We deliberately rely on TransNetV2 alone here — the milestone plan says
    cut detection is solved and we should not pay PySceneDetect + refinement
    costs at indexing time.
    """
    from ..detectors.transnetv2_detector import detect as tnv2_detect
    from ..detectors.ensemble import BoundaryCandidate  # noqa: F401
    from ..pipeline import _boundaries_to_shots
    from ..refinement import RefinedBoundary

    cands = tnv2_detect(video_path, config.transnetv2)

    # Wrap BoundaryCandidate as RefinedBoundary the same way pipeline.run_system(C) does.
    boundaries: List[RefinedBoundary] = []
    for c in cands:
        boundaries.append(RefinedBoundary(
            frame_idx=c.frame_idx,
            end_frame_idx=c.end_frame_idx if c.end_frame_idx >= 0 else c.frame_idx,
            pts=0,
            time_sec=0.0,
            type="gradual_transition" if c.is_gradual else "hard_cut",
            signal_peak=c.confidence,
            signal_width=0,
            sources=list(c.sources),
            coarse_frame_idx=c.frame_idx,
            signal_time_sec=0.0,
            signal_curve=[],
        ))
    return _boundaries_to_shots(boundaries, n_frames)


def cmd_index(args: argparse.Namespace) -> int:
    video_path = str(Path(args.video).resolve())
    if not os.path.exists(video_path):
        print(f"error: video not found: {video_path}", file=sys.stderr)
        return 2

    config = CutsConfig()
    config.verbose = args.verbose
    if args.asr:
        config.retrieval.enable_asr = True
    if args.clip:
        config.retrieval.enable_image_embeddings = True
    if args.no_text_emb:
        config.retrieval.enable_text_embeddings = False

    video_id = Path(video_path).stem
    out_root = Path(args.out).resolve()
    index_dir = out_root / video_id
    index_dir.mkdir(parents=True, exist_ok=True)

    def stage(label: str) -> float:
        if args.verbose:
            print(f"[{label}] ...")
        return time.time()

    t0 = time.time()

    t = stage("frame_index")
    frame_index = build_frame_index(video_path, cache_dir=config.cache_dir)
    n_frames = len(frame_index)
    duration_sec = frame_index[-1][1] if frame_index else 0.0
    if args.verbose:
        print(f"  {n_frames} frames, {duration_sec:.2f}s "
              f"(+{time.time() - t:.2f}s)")

    t = stage("shots (TransNetV2)")
    shots = _derive_shots(video_path, config, n_frames)
    if args.verbose:
        print(f"  {len(shots)} shots (+{time.time() - t:.2f}s)")

    t = stage("segmenter")
    segments: List[SegmentRecord] = build_segments(
        video_id, shots, frame_index, config.retrieval
    )
    if args.verbose:
        print(f"  {len(segments)} segments (+{time.time() - t:.2f}s)")

    t = stage("sampler")
    sample_frames(video_path, segments, str(index_dir),
                  config.retrieval, frame_index=frame_index,
                  verbose=args.verbose)
    if args.verbose:
        print(f"  (+{time.time() - t:.2f}s)")

    t = stage("OCR")
    from .ocr import run_ocr
    if getattr(args, "ocr_debug", False):
        config.retrieval.ocr_debug = True
        config.retrieval.ocr_save_debug_png = True
    run_ocr(video_path, segments, str(index_dir), config.retrieval, verbose=args.verbose)
    if args.verbose:
        print(f"  (+{time.time() - t:.2f}s)")

    if config.retrieval.enable_asr:
        t = stage("ASR")
        from .asr import transcribe, attach_transcript
        asr_segments = transcribe(video_path, config.retrieval,
                                  device=config.device, verbose=args.verbose)
        attach_transcript(segments, asr_segments)
        if args.verbose:
            print(f"  (+{time.time() - t:.2f}s)")

    t = stage("segments.json")
    write_segments(
        str(index_dir / "segments.json"),
        video_id=video_id,
        video_path=video_path,
        segments=segments,
        extra={"duration_sec": duration_sec, "n_frames": n_frames,
               "n_shots": len(shots)},
    )
    if args.verbose:
        print(f"  (+{time.time() - t:.2f}s)")

    t = stage("index (BM25 + embeddings)")
    from .index import build_index
    build_index(segments, str(index_dir), config.retrieval,
                device=config.device, verbose=args.verbose)
    if args.verbose:
        print(f"  (+{time.time() - t:.2f}s)")

    total = time.time() - t0
    per_min = total / max(duration_sec / 60.0, 1e-6)
    print(f"indexed {video_id}: {len(segments)} segments, "
          f"{total:.1f}s total ({per_min:.1f}s / video-minute)")
    print(f"index dir: {index_dir}")
    return 0


def _fmt_time(sec: float) -> str:
    """Format seconds as M:SS (e.g. 4:33)."""
    total = int(sec)
    return f"{total // 60}:{total % 60:02d}"


def _highlight(text: str, query_tokens: list, use_ansi: bool = True) -> str:
    """Wrap each query token match in ANSI bold, or [brackets] when not a TTY."""
    import re
    result = text
    # Longest tokens first to avoid partial-match clobbering.
    for tok in sorted(set(query_tokens), key=len, reverse=True):
        if not tok:
            continue
        if use_ansi:
            repl = lambda m: f"\033[1m{m.group(0)}\033[0m"
        else:
            repl = lambda m: f"[{m.group(0)}]"
        result = re.sub(re.escape(tok), repl, result, flags=re.IGNORECASE)
    return result


def cmd_search(args: argparse.Namespace) -> int:
    from .search import Searcher
    from .index import tokenize

    index_dir = str(Path(args.index_dir).resolve())
    if not os.path.exists(os.path.join(index_dir, "segments.json")):
        print(f"error: no segments.json in {index_dir}", file=sys.stderr)
        return 2

    # Support both quoted single-arg and bare multi-word args.
    query = " ".join(args.query) if isinstance(args.query, list) else args.query
    verbose = getattr(args, "verbose", False)

    config = CutsConfig().retrieval
    searcher = Searcher(index_dir, config=config)
    sources = None
    if args.sources:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    results = searcher.query(query, top_k=args.top_k, sources=sources)

    W = 56
    DIV = "\u2500" * W
    use_ansi = sys.stdout.isatty()
    query_tokens = tokenize(query)

    if not results:
        print(f'\nNo results for: "{query}"')
        return 0

    print(f'\nTop results for: "{query}"\n{DIV}')

    for i, r in enumerate(results, start=1):
        seg = r.segment
        ts = _fmt_time(seg.start_time)

        # Best text line: prefer BM25-matched snippet, then first OCR line.
        snippet = (r.matched_snippet
                   or (seg.ocr_text.split("\n")[0] if seg.ocr_text else "")
                   or (seg.ocr_text_raw.split("\n")[0] if seg.ocr_text_raw else ""))
        snippet_hi = _highlight(snippet.strip(), query_tokens, use_ansi) if snippet else "(no text)"

        print(f"\n  #{i}  {ts}")
        print(f"      \u2192 {snippet_hi}")

        if seg.representative_frames:
            print(f"\n      thumbnail: {seg.representative_frames[0]}")

        if verbose:
            print(f"\n      score={r.score:.4f}  sources: {', '.join(r.sources) or '-'}")
            print(f"      why: {r.explanation}")
            if seg.ocr_text:
                preview = seg.ocr_text[:200].replace("\n", " / ")
                print(f"      ocr: {preview}")

        if i < len(results):
            print(f"\n  {DIV}")

    print(f"\n{DIV}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cuts.retrieval.cli",
        description="Build and query a retrieval index over a video.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="build a retrieval index for a video")
    p_idx.add_argument("video", help="path to the video file")
    p_idx.add_argument("--out", default="retrieval_index",
                       help="output root directory (default: retrieval_index)")
    p_idx.add_argument("--asr", action="store_true",
                       help="enable ASR (faster-whisper)")
    p_idx.add_argument("--clip", action="store_true",
                       help="enable CLIP image embeddings")
    p_idx.add_argument("--no-text-emb", action="store_true",
                       help="disable text embeddings (BM25-only index)")
    p_idx.add_argument("--verbose", action="store_true")
    p_idx.add_argument("--ocr-debug", action="store_true",
                       help="verbose per-frame OCR breakdown + save debug PNGs to ocr_debug/")
    p_idx.set_defaults(func=cmd_index)

    p_s = sub.add_parser("search", help="query a retrieval index")
    p_s.add_argument("index_dir", help="path to a built index directory")
    p_s.add_argument("query", nargs="+",
                     help='text query; multi-word works with or without quotes')
    p_s.add_argument("-k", "--top-k", type=int, default=5)
    p_s.add_argument("--sources", default=None,
                     help="comma-separated subset of {bm25,embedding,clip}")
    p_s.add_argument("--verbose", action="store_true",
                     help="show scores, sources, and OCR text for each result")
    p_s.set_defaults(func=cmd_search)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
