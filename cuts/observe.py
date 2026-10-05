"""Pass 1 — independent, timestamped observations of the footage.

One `Observation` is one sampled frame: when it happened, what OCR read off
it, and one sentence describing what was on screen. Nothing more. It is
deliberately *not* an `Event` (`cuts.schema`) — an event has extent and
meaning, and deciding that is Pass 2's job (`cuts.coalesce`). Pass 1 only
looks.

**Every observation is produced independently.** The timeline is cut into
fixed-duration shards and each is handed to `observe_shard`, a pure function:
no shared state, no globals, no dependence on any other shard's result. That
is the whole design, and it buys three things:

  * *Decode that no longer grows with video length.* The old sampling loop
    could already fire description calls concurrently — that was never the
    bottleneck. The bottleneck was decode, which was sequential and linear in
    total length, so no amount of API concurrency could make a three-hour
    capture finish in less than a full decode. Seek-based sampling
    (`media.iter_frames_at_times`) costs the same at the end of a file as at
    the start: ~5x faster on real capture, and flat with depth, which is what
    lets a shard start in the middle of a file at all.
  * *A cache worth having.* Observations are the expensive part of a run and
    they do not depend on how the footage is later carved up, so they are
    written to `<stem>.observations.jsonl` and reused. Pass 2 can then be
    re-run any number of times — different intent, different prompt, different
    model — against one paid observation pass. In practice this is what makes
    coalescing iterable at all.
  * *A path to containers that needs no rewrite.* `observe_shard` already
    takes everything it needs by value and returns plain data. Moving it into
    a container is a change of executor, not of function.

**What sharding does not buy, measured rather than assumed:** much local
parallelism. RapidOCR's ONNX session already uses every core, so OCR is
saturating the machine before any shard splitting happens — on 8 cores, four
worker *processes* came out slower than running serially (44.1s vs 34.4s),
because they oversubscribe a runtime that was already parallel. Threads
recover a modest 1.4x by overlapping decode and API latency with OCR, and that
is the honest ceiling on one box. The single-machine speedup in this rewrite
comes from seeking, not from sharding. Sharding pays off when workers are
*separate machines*, and in the meantime it pays for itself through the cache
and by making `observe_shard` the primitive the zoom tool will reuse.

The one thing sharding gives up: `iter_timeline_frames` discovers each file's
true extent while streaming through it, and nothing here streams. So durations
are confirmed up front with `media.true_duration_sec`, and frame *ordinals*
are simply not assigned — a seeked frame has no meaningful decode index, which
is why observations are keyed on time.

Descriptions are still batched in small groups within a shard. That does not
compromise independence — a shard is self-contained either way — and it keeps
the neighbouring-frame context that `signals.vlm` uses to recognize a
partially visible window. Total description calls in flight is
`workers x chunks-per-shard`, which the shard size bounds naturally.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import CutsConfig
from .media import (
    Timeline,
    iter_frames_at_times,
    iter_sampled_frames,
    resize_long_edge,
    true_duration_sec,
)

# Bumped when a change makes previously written observation files unusable.
CACHE_VERSION = 1


@dataclass
class Observation:
    """One sampled frame, described. The atomic unit of Pass 1.

    `t` is global timeline seconds and is the identifier — there is no frame
    ordinal, because the frame was reached by seeking. `source_time` is the
    same instant expressed inside its own file, which is what you need to
    actually open the footage.

    `withheld` marks a frame that was never uploaded because `cuts.redact`
    found a credential it could not paint over. That is recorded rather than
    dropped: a gap in the timeline and a frame we deliberately declined to
    look at are different facts, and Pass 2 should be able to tell them apart.
    """

    t: float
    source_index: int = 0
    source_time: float = 0.0
    description: str = ""
    ocr_text: str = ""
    transcript_text: str = ""
    # Names of the credential patterns that matched, never the matched text.
    redacted: List[str] = field(default_factory=list)
    withheld: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Observation":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Shard:
    """A contiguous slice of one source file, and the samples wanted from it.

    `sample_times` are **local** to the file; add `time_offset` for global.
    They are carried explicitly rather than derived from an interval so that
    the same function serves a different sampling plan without modification —
    which is exactly what the future `zoom(t0, t1, n)` tool is.

    A shard never spans a file join. One worker touches one file, and the
    join-is-a-boundary semantics the pipeline relies on stay intact.
    """

    source_index: int
    path: str
    time_offset: float
    sample_times: List[float]

    @property
    def start_time(self) -> float:
        return self.time_offset + (self.sample_times[0] if self.sample_times else 0.0)

    @property
    def end_time(self) -> float:
        return self.time_offset + (self.sample_times[-1] if self.sample_times else 0.0)

    def __len__(self) -> int:
        return len(self.sample_times)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_shards(
    timeline: Timeline, interval_sec: float, shard_sec: float
) -> List[Shard]:
    """Cut the timeline into independent units of work.

    Sample times are laid out per *file* from its own zero, so the grid is
    identical to what a sequential pass over that file would have produced.
    Shards then partition that grid; no sample is planned twice and none is
    dropped between shards.
    """
    if interval_sec <= 0:
        raise ValueError(f"interval_sec must be positive, got {interval_sec}")
    per_shard = max(1, int(round(max(shard_sec, interval_sec) / interval_sec)))

    shards: List[Shard] = []
    for src in timeline.sources:
        n = max(1, int(src.duration_sec / interval_sec) + 1)
        local_times = [i * interval_sec for i in range(n)
                       if i * interval_sec <= src.duration_sec + 1e-6]
        for i in range(0, len(local_times), per_shard):
            shards.append(Shard(
                source_index=src.index,
                path=src.path,
                time_offset=src.time_offset,
                sample_times=local_times[i:i + per_shard],
            ))
    return shards


def confirm_durations(timeline: Timeline, verbose: bool = False) -> None:
    """Re-probe each source's true extent and re-lay the timeline on it.

    Sharding cannot discover a file's real length by streaming through it the
    way `iter_timeline_frames` does, and a container that under-reports would
    otherwise mean the tail of a file is never sampled at all.
    """
    cursor = 0.0
    for src in timeline.sources:
        true_dur = true_duration_sec(src.path, hint=src.duration_sec)
        if verbose and abs(true_dur - src.duration_sec) > 1.0:
            print(f"  [observe] {src.name}: container claims "
                  f"{src.duration_sec:.0f}s, actually {true_dur:.0f}s")
        src.duration_sec = true_dur
        src.time_offset = cursor
        cursor += true_dur


# ---------------------------------------------------------------------------
# The unit of work
# ---------------------------------------------------------------------------

def _decode_shard_sequential(
    shard: Shard, tolerance_sec: float
) -> List[Tuple[float, Optional[object]]]:
    """Sequential-decode fallback for footage where seeking is unreliable.

    Correct but linear in position: this decodes the file from its start, so a
    shard late in a long capture pays for everything in front of it. Only
    worth it when `iter_frames_at_times` cannot land reliably.
    """
    wanted = list(shard.sample_times)
    if not wanted:
        return []
    out: List[Tuple[float, Optional[object]]] = []
    cursor = 0
    for sf in iter_sampled_frames(shard.path, interval_sec=1.0):
        if cursor >= len(wanted):
            break
        if sf.time_sec + 1e-9 >= wanted[cursor]:
            out.append((wanted[cursor],
                        sf if abs(sf.time_sec - wanted[cursor]) <= tolerance_sec
                        else None))
            cursor += 1
    out.extend((t, None) for t in wanted[cursor:])
    return out


def observe_shard(
    shard: Shard,
    config: CutsConfig,
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> List[Observation]:
    """Observe one shard. **Pure**: same inputs, same outputs, no shared state.

    This is the containerizable unit. It takes everything it needs by value
    and returns plain data, so relocating it behind a process boundary, a
    queue, or an HTTP endpoint changes the executor and nothing else.

    Per sample: decode -> OCR -> redact -> encode -> describe. The redaction
    step is not optional bookkeeping — this function uploads frames, so it is
    an upload path in exactly the sense `cuts.redact` cares about, and
    `scrub_frame` blacks out credentials in the decoded frame *before* the
    resize whose output gets sent.
    """
    from .signals.vlm import describe_batch, encode_frame

    want_ocr = config.ocr.enabled
    scrub = None
    if want_ocr:
        from .redact import scrub_frame as scrub

    if config.observe.seek:
        frames = list(iter_frames_at_times(
            shard.path, shard.sample_times,
            tolerance_sec=config.observe.seek_tolerance_sec,
        ))
    else:
        frames = _decode_shard_sequential(
            shard, config.observe.seek_tolerance_sec
        )

    observations: List[Observation] = []
    pending_enc: List[str] = []
    pending_ocr: List[str] = []
    pending_pos: List[int] = []

    for local_t, sf in frames:
        obs = Observation(
            t=shard.time_offset + local_t,
            source_index=shard.source_index,
            source_time=local_t,
        )
        if sf is None:
            # No frame landed here. Recorded as withheld so the timeline keeps
            # its shape rather than silently losing a sample.
            obs.withheld = True
            observations.append(obs)
            continue

        safe = True
        if scrub is not None:
            # Mutates sf.image in place, so the redaction survives the resize
            # and JPEG encode below.
            obs.ocr_text, safe, hits = scrub(
                sf.image, config.ocr, config.redact, verbose=False
            )
            obs.redacted = list(hits)
        if not safe:
            obs.withheld = True
            observations.append(obs)
            continue

        pending_pos.append(len(observations))
        pending_enc.append(encode_frame(
            resize_long_edge(sf.image, config.vlm.frame_long_edge)
        ))
        pending_ocr.append(obs.ocr_text)
        observations.append(obs)

    if pending_enc and config.vlm.enabled:
        descriptions = describe_batch(
            pending_enc, config.vlm, ocr_texts=pending_ocr,
            api_key=api_key, verbose=verbose,
        )
        for pos, desc in zip(pending_pos, descriptions):
            observations[pos].description = desc

    return observations


# ---------------------------------------------------------------------------
# Running every shard
# ---------------------------------------------------------------------------

def _run_shard(args) -> List[Observation]:
    """Top-level trampoline so `ProcessPoolExecutor` can pickle the call."""
    shard, config, api_key, verbose = args
    return observe_shard(shard, config, api_key=api_key, verbose=verbose)


def observe_timeline(
    timeline: Timeline,
    config: CutsConfig,
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> List[Observation]:
    """Observe the whole timeline, shard by shard, in parallel.

    Returns observations sorted by time. Shard results are independent, so the
    only thing the order of completion affects is how quickly this returns.
    """
    confirm_durations(timeline, verbose=verbose)
    shards = plan_shards(
        timeline, config.sampling.sample_interval_sec, config.observe.shard_sec
    )
    n_samples = sum(len(s) for s in shards)

    if config.redact.enabled and not config.ocr.enabled:
        print("[observe] WARNING: OCR is disabled, so credential redaction "
              "cannot run; frames will be uploaded unredacted")

    workers = config.observe.workers or (os.cpu_count() or 4)
    workers = max(1, min(workers, len(shards)))
    if verbose:
        print(f"  [observe] {len(shards)} shard(s), {n_samples} samples, "
              f"{workers} {config.observe.executor} worker(s), "
              f"seek={config.observe.seek}")

    payloads = [(s, config, api_key, verbose) for s in shards]
    results: List[List[Observation]]

    if config.observe.executor == "serial" or workers == 1:
        results = [_run_shard(p) for p in payloads]
    else:
        pool_cls = (ThreadPoolExecutor
                    if config.observe.executor == "thread"
                    else ProcessPoolExecutor)
        with pool_cls(max_workers=workers) as pool:
            results = list(pool.map(_run_shard, payloads))

    observations = [o for chunk in results for o in chunk]
    observations.sort(key=lambda o: o.t)

    hits = [name for o in observations for name in o.redacted]
    withheld = sum(1 for o in observations if o.withheld)
    if hits:
        # Printed unconditionally: that the footage contains a live credential
        # is the most useful thing this pass learns.
        n_frames = sum(1 for o in observations if o.redacted)
        print(f"[redact] blacked out credentials in {n_frames} frame(s) "
              f"({', '.join(sorted(set(hits)))})")
    if withheld and verbose:
        print(f"  [observe] {withheld} sample(s) withheld or undecodable")
    return observations


# ---------------------------------------------------------------------------
# On-disk cache
# ---------------------------------------------------------------------------

def observations_path(video_path: str, out_dir: Optional[str] = None) -> str:
    """Where a timeline's observations live: `<first_video_stem>.observations.jsonl`."""
    p = Path(video_path)
    name = f"{p.stem}.observations.jsonl"
    return str(Path(out_dir) / name) if out_dir else str(p.with_name(name))


def _fingerprint(timeline: Timeline, config: CutsConfig) -> Dict[str, Any]:
    """What must match for cached observations to still be valid.

    Keyed on the inputs that change what was *looked at* — the files
    themselves, the sampling grid, and whether redaction was on. Deliberately
    not on the coalescing settings, which is the entire point: Pass 2 can be
    re-run with a different model, intent or prompt against the same cache.
    """
    files = []
    for s in timeline.sources:
        try:
            st = os.stat(s.path)
            files.append([s.path, int(st.st_mtime), st.st_size])
        except OSError:
            files.append([s.path, 0, 0])
    return {
        "version": CACHE_VERSION,
        "files": files,
        "interval_sec": config.sampling.sample_interval_sec,
        "redact": config.redact.enabled,
        "ocr": config.ocr.enabled,
        "vlm_model": config.vlm.model if config.vlm.enabled else None,
    }


def write_observations(
    path: str,
    observations: Sequence[Observation],
    meta: Dict[str, Any],
) -> None:
    """Write observations as JSONL with a leading `_meta` header line.

    JSONL rather than one JSON array so a long run can be streamed, tailed
    while it works, and truncated without losing everything before the tear.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for o in observations:
            f.write(json.dumps(o.to_dict(), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_observations(path: str) -> Tuple[List[Observation], Dict[str, Any]]:
    """Read an observations file. Returns `(observations, meta)`."""
    observations: List[Observation] = []
    meta: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "_meta" in d:
                meta = d["_meta"]
            else:
                observations.append(Observation.from_dict(d))
    observations.sort(key=lambda o: o.t)
    return observations, meta


def load_cached(
    timeline: Timeline, config: CutsConfig, path: Optional[str] = None
) -> Optional[List[Observation]]:
    """Return cached observations when they match the current inputs, else None."""
    if not config.observe.cache:
        return None
    path = path or observations_path(timeline.first_path)
    if not os.path.exists(path):
        return None
    try:
        observations, meta = read_observations(path)
    except (OSError, json.JSONDecodeError):
        return None
    if not observations or meta != _fingerprint(timeline, config):
        return None
    return observations


def observe(
    timeline: Timeline,
    config: CutsConfig,
    api_key: Optional[str] = None,
    verbose: bool = False,
    cache_path: Optional[str] = None,
) -> List[Observation]:
    """Pass 1 end to end: reuse the cache when it is valid, else do the work.

    The cache check runs before `confirm_durations`, so a cache hit costs one
    file stat per source and no decoding at all.
    """
    cache_path = cache_path or observations_path(timeline.first_path)
    cached = load_cached(timeline, config, cache_path)
    if cached is not None:
        print(f"[observe] reusing {len(cached)} cached observation(s) "
              f"from {os.path.basename(cache_path)}")
        return cached

    observations = observe_timeline(
        timeline, config, api_key=api_key, verbose=verbose
    )
    if config.observe.cache and observations:
        try:
            write_observations(cache_path, observations,
                               _fingerprint(timeline, config))
            if verbose:
                print(f"  [observe] cached to {cache_path}")
        except OSError as exc:
            print(f"[observe] could not write observation cache: {exc}")
    return observations


if __name__ == "__main__":
    # Standalone debug: observe the first few minutes of a video and print
    # what Pass 1 actually produces, without touching Pass 2.
    import sys
    import time

    from .schema import format_timestamp

    if len(sys.argv) < 2:
        print("usage: python -m cuts.observe <video_path> [minutes]")
        sys.exit(1)
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

    cfg = CutsConfig()
    cfg.verbose = True
    cfg.observe.cache = False
    tl = Timeline.build([sys.argv[1]])
    confirm_durations(tl, verbose=True)
    for s in tl.sources:
        s.duration_sec = min(s.duration_sec, minutes * 60.0)

    t0 = time.time()
    obs = observe_timeline(tl, cfg, verbose=True)
    print(f"\n{len(obs)} observations in {time.time() - t0:.1f}s "
          f"({(time.time() - t0) / max(1, len(obs)):.2f}s each)\n")
    for o in obs:
        flag = " [withheld]" if o.withheld else ""
        print(f"  {format_timestamp(o.t):>7}{flag}  {o.description or '(none)'}")
