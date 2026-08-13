"""Small manual benchmark for the retrieval milestone.

Input: a YAML file with a list of queries, each with 1+ known-good time
ranges, plus the path to a prebuilt index directory.

    index_dir: path/to/retrieval_index/<video_id>
    queries:
      - text: "CUDA error"
        truth:
          - [124.5, 180.0]
      - text: "opening Chrome"
        truth:
          - [303.0, 340.0]
          - [512.0, 540.0]

Metrics (rough, on purpose):
  * top-1 hit:   any truth range overlaps the #1 result's [start, end].
  * top-5 hit:   any truth range overlaps any result in the top-5.
  * time error: midpoint distance between the top-1 result and the closest
                truth range. Only reported when top-1 hit.
  * per-source: the same three metrics computed for each of
                bm25-only, embedding-only, clip-only, hybrid (all).

Output: prints a markdown table to stdout and optionally writes it to --out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Tuple

from ..config import CutsConfig
from ..retrieval.search import Searcher


@dataclass
class QueryCase:
    text: str
    truth: List[Tuple[float, float]]


def _load_queries(path: str) -> List[QueryCase]:
    import yaml  # type: ignore
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    out = []
    for q in data.get("queries", []):
        truth = [tuple(r) for r in q.get("truth", [])]
        out.append(QueryCase(text=q["text"], truth=truth))
    return out


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _mid(a: Tuple[float, float]) -> float:
    return 0.5 * (a[0] + a[1])


def _nearest_truth_mid_err(
    pred: Tuple[float, float], truths: List[Tuple[float, float]]
) -> float:
    return min(abs(_mid(pred) - _mid(t)) for t in truths)


def _eval_one(
    searcher: Searcher,
    case: QueryCase,
    top_k: int,
    sources: List[str] = None,
) -> Dict[str, Any]:
    results = searcher.query(case.text, top_k=top_k, sources=sources)
    if not results:
        return {
            "top1": False, "topk": False, "time_err": None,
            "top1_result": None, "results": [],
        }

    top1 = results[0]
    top1_pred = (top1.segment.start_time, top1.segment.end_time)
    top1_hit = any(_overlap(top1_pred, t) > 0 for t in case.truth)

    topk_hit = False
    for r in results:
        span = (r.segment.start_time, r.segment.end_time)
        if any(_overlap(span, t) > 0 for t in case.truth):
            topk_hit = True
            break

    time_err = (_nearest_truth_mid_err(top1_pred, case.truth)
                if top1_hit and case.truth else None)

    return {
        "top1": top1_hit,
        "topk": topk_hit,
        "time_err": time_err,
        "top1_result": {
            "start": top1.segment.start_time,
            "end": top1.segment.end_time,
            "sources": top1.sources,
            "snippet": top1.matched_snippet,
            "score": top1.score,
        },
        "results": [r.to_dict() for r in results],
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    top1 = sum(1 for r in rows if r["top1"])
    topk = sum(1 for r in rows if r["topk"])
    errs = [r["time_err"] for r in rows if r["time_err"] is not None]
    return {
        "n": n,
        "top1_rate": top1 / n if n else 0.0,
        "topk_rate": topk / n if n else 0.0,
        "median_time_err_sec": median(errs) if errs else None,
    }


def _markdown_table(
    per_query: List[Tuple[str, Dict[str, Dict[str, Any]]]],
    source_labels: List[str],
) -> str:
    """Render per-query outcomes across ranking variants."""
    lines = []
    header = "| query | " + " | ".join(source_labels) + " |"
    sep = "| --- |" + " --- |" * len(source_labels)
    lines.append(header)
    lines.append(sep)
    for text, per_source in per_query:
        cells = []
        for label in source_labels:
            row = per_source[label]
            mark = "✓" if row["top1"] else ("~" if row["topk"] else "✗")
            err = f" ({row['time_err']:.1f}s)" if row["time_err"] is not None else ""
            cells.append(f"{mark}{err}")
        lines.append(f"| {text} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _summary_table(agg: Dict[str, Dict[str, Any]], source_labels: List[str]) -> str:
    lines = ["| variant | top-1 | top-5 | median time err |",
             "| --- | --- | --- | --- |"]
    for label in source_labels:
        a = agg[label]
        err = f"{a['median_time_err_sec']:.1f}s" if a["median_time_err_sec"] is not None else "-"
        lines.append(f"| {label} | {a['top1_rate']:.0%} | {a['topk_rate']:.0%} | {err} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cuts.benchmark.retrieval_eval",
        description="Evaluate a retrieval index against hand-labeled queries.",
    )
    parser.add_argument("--index", required=True, help="index directory")
    parser.add_argument("--queries", required=True,
                        help="YAML file with queries + truth time ranges")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--out", default=None,
                        help="optional path to write markdown report")
    parser.add_argument("--json", default=None,
                        help="optional path to write raw JSON results")
    args = parser.parse_args(argv)

    index_dir = str(Path(args.index).resolve())
    if not os.path.exists(os.path.join(index_dir, "segments.json")):
        print(f"error: no segments.json in {index_dir}", file=sys.stderr)
        return 2

    cases = _load_queries(args.queries)
    if not cases:
        print("error: queries file has no queries", file=sys.stderr)
        return 2

    config = CutsConfig().retrieval
    searcher = Searcher(index_dir, config=config)

    # Pick which source variants we can evaluate given what's in the index.
    variants: List[Tuple[str, List[str]]] = [("bm25", ["bm25"])]
    if os.path.exists(os.path.join(index_dir, "text_emb.npy")):
        variants.append(("embedding", ["embedding"]))
    if os.path.exists(os.path.join(index_dir, "image_emb.npy")):
        variants.append(("clip", ["clip"]))
    # Hybrid = everything available.
    all_sources = [s for _lbl, srcs in variants for s in srcs]
    variants.append(("hybrid", all_sources))

    labels = [lbl for lbl, _ in variants]

    # Run.
    per_query: List[Tuple[str, Dict[str, Dict[str, Any]]]] = []
    raw_rows: Dict[str, List[Dict[str, Any]]] = {lbl: [] for lbl in labels}

    for case in cases:
        per_source: Dict[str, Dict[str, Any]] = {}
        for lbl, srcs in variants:
            row = _eval_one(searcher, case, args.top_k, sources=srcs)
            per_source[lbl] = row
            raw_rows[lbl].append(row)
        per_query.append((case.text, per_source))

    agg = {lbl: _aggregate(raw_rows[lbl]) for lbl in labels}

    md = []
    md.append(f"# Retrieval eval — {searcher.video_id}")
    md.append("")
    md.append(f"Index: `{index_dir}`")
    md.append(f"Queries: `{args.queries}` ({len(cases)} queries, top-{args.top_k})")
    md.append("")
    md.append("## Per-query outcomes (✓ top-1, ~ top-5 only, ✗ miss)")
    md.append("")
    md.append(_markdown_table(per_query, labels))
    md.append("")
    md.append("## Aggregate")
    md.append("")
    md.append(_summary_table(agg, labels))
    md_out = "\n".join(md)

    print(md_out)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md_out + "\n")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({
                "video_id": searcher.video_id,
                "index_dir": index_dir,
                "aggregate": agg,
                "per_query": [
                    {"query": q, "by_variant": ps}
                    for q, ps in per_query
                ],
            }, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
