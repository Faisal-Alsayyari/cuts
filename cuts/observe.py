"""Pass 1 — independent, timestamped observations of the footage.

One `Observation` is one sampled frame: when it happened, what OCR read off
it, one sentence describing what was on screen, and (for the curve segmenter)
a DINOv2 embedding. Nothing more. It is deliberately *not* an `Event`
(`cuts.schema`) — an event has extent and meaning, and deciding that is Pass
2's job. Pass 1 only looks. Both segmenters read the same observations.

**Every observation is produced independently.** The timeline's sample grid
is cut into shards and each is handed to `observe_shard`, a function of its
arguments only: it takes an explicit list of sample times, so the same
function serves a fresh run, a resumed run, a retry of the samples whose
description failed, and adding embeddings to a cache that was built without
them. Seek-based sampling (`media.iter_frames_at_times`) costs the same at the
end of a file as at the start, which is what lets a shard start anywhere.

**The cache is incremental.** Observations are written to
`<stem>.observations.jsonl` shard by shard as they finish, so an interrupted
run loses at most the shards in flight. On the next run, every grid sample is
checked against the cache and only the missing work is done:

  * a sample with no observation at all;
  * a sample whose description failed (`status == "failed"`) — a transient
    API error is retried, never cached as an empty description;
  * a sample without an embedding, when the curve segmenter needs one;
  * every description, if `--vlm-model` changed; every embedding, if the
    DINOv2 model changed.

What invalidates the whole cache is only what changes *what was looked at*:
the files (by name and size, so moving them does not), the sampling interval,
and the OCR/redaction settings. Coalescing settings are deliberately not part
of it — Pass 2 can be re-run with a different model or intent for the price
of one call.

**What sharding does not buy, measured rather than assumed:** much local
parallelism. RapidOCR's ONNX session already uses every core, so OCR is
saturating the machine before any shard splitting happens — on 8 cores, four
worker *processes* came out slower than running serially (44.1s vs 34.4s).
Threads recover a modest 1.4x by overlapping decode and API latency with OCR.
The single-machine speedup comes from seeking; sharding pays off through the
cache, resumption, and (later) workers on separate machines.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .config import CutsConfig
from .media import (
    Timeline,
    has_audio,
    iter_frames_at_times,
    iter_sampled_frames,
    resize_long_edge,
    true_duration_sec,
)

# Bumped when a change makes previously written observation files unusable.
CACHE_VERSION = 2

# Observation.status values.
OK = "ok"                    # frame decoded; described if description was requested
FAILED = "failed"            # description requested but not obtained; retried next run
WITHHELD = "withheld"        # a credential could not be painted over; never uploaded
UNDECODABLE = "undecodable"  # no frame landed within tolerance of the sample time

Key = Tuple[int, float]


@dataclass
class Observation:
    """One sampled frame, observed. The atomic unit of Pass 1.

    `t` is global timeline seconds — the sample's grid time, not the decoded
    frame's pts (which lands within `ObserveConfig.seek_tolerance_sec`, in
    practice within 0.05s). `source_time` is the same instant inside its own
    file, which is what you need to actually open the footage; together with
    `source_index` it is the observation's identity in the cache.
    """

    t: float
    source_index: int = 0
    source_time: float = 0.0
    status: str = OK
    description: str = ""
    ocr_text: str = ""
    transcript_text: str = ""
    # Names of the credential patterns that matched, never the matched text.
    redacted: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None

    @property
    def key(self) -> Key:
        return (self.source_index, round(self.source_time, 3))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["embedding"] is None:
            del d["embedding"]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Observation":
        d = dict(d)
        if "status" not in d:
            # Version-1 files had a single `withheld` flag.
            d["status"] = WITHHELD if d.get("withheld") else OK
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Work:
    """What Pass 1 should produce, decided by the caller."""

    describe: bool = True     # upload frames for a description (needs credentials)
    embed: bool = False       # DINOv2 embeddings (curve segmenter)
    transcribe: bool = True   # speech transcript, when a source has audio


@dataclass
class Shard:
    """A slice of one source file's sample grid, and what to do with it.

    `sample_times` are **local** to the file; add `time_offset` for global.
    They are explicit rather than derived from an interval, which is what
    lets resumption and retries hand over an arbitrary subset of the grid.
    A shard never spans a file join.
    """

    source_index: int
    path: str
    time_offset: float
    sample_times: List[float]
    describe: bool = True
    embed: bool = False

    def __len__(self) -> int:
        return len(self.sample_times)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def confirm_durations(timeline: Timeline, verbose: bool = False) -> None:
    """Re-probe each source's true extent and re-lay the timeline on it.

    Runs on every Pass 1, cache hit or not: global observation times, chapter
    file offsets, refinement frames and transcript alignment all depend on
    these offsets, so they must not differ between a fresh and a cached run.
    It costs one seek to the end of each file.
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


def sample_grid(timeline: Timeline, interval_sec: float) -> Dict[int, List[float]]:
    """Local sample times per source: 0, interval, 2·interval, ... < duration.

    Strictly before the end: a sample *at* the reported duration usually
    lies past the last frame and would only ever come back undecodable.
    """
    if interval_sec <= 0:
        raise ValueError(f"interval_sec must be positive, got {interval_sec}")
    grid: Dict[int, List[float]] = {}
    for src in timeline.sources:
        n = int(src.duration_sec / interval_sec) + 1
        times = [i * interval_sec for i in range(n)
                 if i * interval_sec < src.duration_sec - 1e-6]
        grid[src.index] = times or [0.0]
    return grid


def plan_shards(
    timeline: Timeline,
    wanted: Dict[Tuple[int, bool, bool], List[float]],
    per_shard: int,
) -> List[Shard]:
    """Chunk wanted sample times into shards of at most `per_shard`.

    `wanted` maps ``(source_index, describe, embed)`` to local times needing
    that work, so every shard does one kind of job on one file.
    """
    shards: List[Shard] = []
    for (si, describe, embed), times in sorted(wanted.items()):
        src = timeline.sources[si]
        times = sorted(times)
        for i in range(0, len(times), max(1, per_shard)):
            shards.append(Shard(
                source_index=si, path=src.path, time_offset=src.time_offset,
                sample_times=times[i:i + per_shard],
                describe=describe, embed=embed,
            ))
    return shards


# ---------------------------------------------------------------------------
# The unit of work
# ---------------------------------------------------------------------------

def _decode_sequential(
    path: str, sample_times: Sequence[float], tolerance_sec: float
) -> Iterator[Tuple[float, Optional[object]]]:
    """Sequential-decode fallback for footage where seeking is unreliable.

    Correct but linear in position: this decodes the file from its start, so
    a shard late in a long capture pays for everything in front of it.
    """
    wanted = sorted(sample_times)
    cursor = 0
    for sf in iter_sampled_frames(path, interval_sec=1.0):
        if cursor >= len(wanted):
            return
        if sf.time_sec + 1e-9 >= wanted[cursor]:
            ok = abs(sf.time_sec - wanted[cursor]) <= tolerance_sec
            yield wanted[cursor], (sf if ok else None)
            cursor += 1
    for t in wanted[cursor:]:
        yield t, None


def observe_shard(
    shard: Shard, config: CutsConfig, verbose: bool = False
) -> List[Observation]:
    """Observe one shard. Depends only on its arguments and the files.

    Per sample: decode -> (if describing) OCR + redact -> encode, and (if
    embedding) downscale for DINOv2. Then one batched description pass and
    one embedding pass over the shard.

    Redaction is not optional bookkeeping: this function uploads frames, and
    `scrub_frame` blacks out credentials in the decoded frame *before* the
    resize whose output gets sent. OCR runs only when frames will be
    uploaded — it exists to ground the description and to find what to
    redact, and it is the dominant local cost.
    """
    scrub = None
    if shard.describe and config.ocr.enabled:
        from .redact import scrub_frame as scrub
    if shard.describe:
        from .signals.vlm import describe_batch, encode_frame

    if config.observe.seek:
        frames: Iterable = iter_frames_at_times(
            shard.path, shard.sample_times,
            tolerance_sec=config.observe.seek_tolerance_sec,
        )
    else:
        frames = _decode_sequential(
            shard.path, shard.sample_times, config.observe.seek_tolerance_sec
        )

    observations: List[Observation] = []
    pending_pos: List[int] = []
    pending_enc: List[str] = []
    pending_ocr: List[str] = []
    embed_pos: List[int] = []
    embed_imgs: List[Any] = []

    for local_t, sf in frames:
        obs = Observation(
            t=shard.time_offset + local_t,
            source_index=shard.source_index,
            source_time=local_t,
        )
        if sf is None:
            obs.status = UNDECODABLE
            observations.append(obs)
            continue

        if shard.describe:
            safe = True
            if scrub is not None:
                # Mutates sf.image in place, so the redaction survives the
                # resize and JPEG encode below (and the embedding).
                obs.ocr_text, safe, hits = scrub(
                    sf.image, config.ocr, config.redact, verbose=False
                )
                obs.redacted = list(hits)
            if safe:
                pending_pos.append(len(observations))
                pending_enc.append(encode_frame(
                    resize_long_edge(sf.image, config.vlm.frame_long_edge)
                ))
                pending_ocr.append(obs.ocr_text)
            else:
                obs.status = WITHHELD

        if shard.embed:
            # Local only, so a withheld frame is still embedded.
            embed_pos.append(len(observations))
            embed_imgs.append(
                resize_long_edge(sf.image, config.sampling.frame_long_edge)
            )
        observations.append(obs)

    if embed_imgs:
        from .signals.dino import embed_images

        vectors = embed_images(
            embed_imgs, config.dino,
            device=config.resolved_device(config.dino.device),
            batch_size=config.sampling.batch_size,
        )
        for pos, vec in zip(embed_pos, vectors):
            observations[pos].embedding = [round(float(x), 5) for x in vec]

    if pending_enc:
        descriptions = describe_batch(
            pending_enc, config.vlm, ocr_texts=pending_ocr, verbose=verbose,
        )
        for pos, desc in zip(pending_pos, descriptions):
            if desc:
                observations[pos].description = desc
            else:
                observations[pos].status = FAILED

    return observations


def _run_shard(args) -> List[Observation]:
    """Top-level trampoline so `ProcessPoolExecutor` can pickle the call."""
    shard, config, verbose = args
    return observe_shard(shard, config, verbose=verbose)


def _run_shards(
    shards: List[Shard], config: CutsConfig, verbose: bool
) -> Iterator[Tuple[Shard, Optional[List[Observation]], Optional[BaseException]]]:
    """Yield ``(shard, observations, None)`` or ``(shard, None, exc)`` as each finishes.

    A shard that raises does not take the run down with it: its samples are
    recorded as failed by the caller and retried on the next run.
    """
    workers = config.observe.workers or (os.cpu_count() or 4)
    workers = max(1, min(workers, len(shards)))
    if config.observe.executor == "serial" or workers == 1:
        for s in shards:
            try:
                yield s, _run_shard((s, config, verbose)), None
            except Exception as exc:
                yield s, None, exc
        return

    pool_cls = (ThreadPoolExecutor if config.observe.executor == "thread"
                else ProcessPoolExecutor)
    with pool_cls(max_workers=workers) as pool:
        futures = {pool.submit(_run_shard, (s, config, verbose)): s for s in shards}
        for fut in as_completed(futures):
            try:
                yield futures[fut], fut.result(), None
            except Exception as exc:
                yield futures[fut], None, exc


def _merge(target: Observation, new: Observation, shard: Shard) -> None:
    """Fold a shard's result for one sample into the observation on record."""
    if new.status == UNDECODABLE:
        if not target.description:
            target.status = UNDECODABLE
        return
    if shard.describe:
        target.status = new.status
        target.description = new.description
        target.ocr_text = new.ocr_text
        target.redacted = list(new.redacted)
    if shard.embed:
        target.embedding = new.embedding


# ---------------------------------------------------------------------------
# Speech
# ---------------------------------------------------------------------------

def _attach_transcripts(
    timeline: Timeline,
    observations: List[Observation],
    config: CutsConfig,
    meta: Dict[str, Any],
    verbose: bool,
) -> bool:
    """Align the speech transcript onto observations. Returns True if new.

    Raw segments are cached in the meta header in *file-local* time, keyed by
    source, and re-aligned on every run — alignment is cheap and depends on
    the timeline offsets, transcription is not cheap and does not. Segment
    text goes through `redact.scrub_text` before it is stored or sent: a key
    read aloud reaches the coalescing prompt just as surely as one on screen.
    """
    from .signals.asr import align_to_samples

    cached = meta.get("asr") or {}
    fresh = cached.get("model") != config.asr.model
    if fresh:
        from .redact import scrub_text
        from .signals.asr import transcribe

        segments: List[List[Any]] = []
        hits: List[str] = []
        audible = [s for s in timeline.sources if has_audio(s.path)]
        if verbose:
            print(f"  [asr] transcribing {len(audible)} source(s) "
                  f"with {config.asr.model}..." if audible
                  else "  [asr] skipped (no audio stream in any source)")
        for s in audible:
            for start, end, text in transcribe(
                s.path, config.asr, device=config.device, verbose=verbose
            ):
                text, found = scrub_text(text, config.redact)
                hits.extend(found)
                segments.append([s.index, start, end, text])
        if hits:
            print(f"[redact] transcript: replaced spoken credentials "
                  f"({', '.join(sorted(set(hits)))})")
        meta["asr"] = cached = {"model": config.asr.model, "segments": segments}

    offsets = {s.index: s.time_offset for s in timeline.sources}
    global_segs = [(offsets.get(si, 0.0) + a, offsets.get(si, 0.0) + b, text)
                   for si, a, b, text in cached.get("segments", [])]
    for o, text in zip(observations,
                       align_to_samples(global_segs, [o.t for o in observations])):
        o.transcript_text = text
    return fresh


# ---------------------------------------------------------------------------
# On-disk cache
# ---------------------------------------------------------------------------

def observations_path(video_paths: Sequence[str], out_dir: Optional[str] = None) -> str:
    """`<first_stem>.observations.jsonl`, or `<first_stem>+N...` for N extra files.

    Named like the index (`cli._default_out_path`), so one file analyzed
    alone and the same file as the first of three do not share a cache.
    """
    first = Path(video_paths[0])
    stem = first.stem if len(video_paths) == 1 else f"{first.stem}+{len(video_paths) - 1}"
    name = f"{stem}.observations.jsonl"
    return str(Path(out_dir) / name) if out_dir else str(first.with_name(name))


def _fingerprint(timeline: Timeline, config: CutsConfig) -> Dict[str, Any]:
    """What must match for cached observations to be usable at all.

    Keyed on what changes *what was looked at*. Files are identified by name
    and size rather than absolute path and mtime, so moving or copying the
    footage keeps its cache.
    """
    files = []
    for s in timeline.sources:
        try:
            size = os.path.getsize(s.path)
        except OSError:
            size = 0
        files.append([os.path.basename(s.path), size])
    return {
        "version": CACHE_VERSION,
        "files": files,
        "interval_sec": config.sampling.sample_interval_sec,
        "redact": config.redact.enabled,
        "ocr": config.ocr.enabled,
    }


def read_observations(path: str) -> Tuple[List[Observation], Dict[str, Any]]:
    """Read an observations file. Returns `(observations, meta)`.

    Later lines for the same sample replace earlier ones (the file is
    appended to as shards finish), and an unparseable line — the torn tail
    of an interrupted run — is skipped.
    """
    by_key: Dict[Key, Observation] = {}
    meta: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in d:
                meta = d["_meta"]
                continue
            try:
                o = Observation.from_dict(d)
            except (TypeError, ValueError):
                continue
            by_key[o.key] = o
    return sorted(by_key.values(), key=lambda o: o.t), meta


def write_observations(
    path: str, observations: Iterable[Observation], meta: Dict[str, Any]
) -> None:
    """Write observations as JSONL behind a `_meta` header line, atomically."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for o in observations:
            f.write(json.dumps(o.to_dict(), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _append(path: str, observations: Iterable[Observation]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for o in observations:
            f.write(json.dumps(o.to_dict(), ensure_ascii=False) + "\n")


def timeline_from_meta(meta: Dict[str, Any], search_dir: str = "") -> Optional[Timeline]:
    """Rebuild the confirmed timeline a cache was made on.

    Paths that no longer exist are looked for by name in `search_dir`
    (normally the cache's own directory), the common case after a folder of
    recordings is moved.
    """
    sources = meta.get("sources")
    if not sources:
        return None
    tl = Timeline.from_dicts(sources)
    for s in tl.sources:
        if not os.path.exists(s.path) and search_dir:
            cand = os.path.join(search_dir, os.path.basename(s.path))
            if os.path.exists(cand):
                s.path = cand
    return tl


# ---------------------------------------------------------------------------
# Pass 1 end to end
# ---------------------------------------------------------------------------

def observe(
    timeline: Timeline,
    config: CutsConfig,
    work: Work,
    verbose: bool = False,
    cache_path: Optional[str] = None,
) -> List[Observation]:
    """Observe the whole timeline, reusing whatever the cache already holds.

    Returns one observation per grid sample, sorted by time.
    """
    confirm_durations(timeline, verbose=verbose)
    cache_path = cache_path or observations_path([s.path for s in timeline.sources])
    use_cache = config.observe.cache

    if work.describe and config.redact.enabled and not config.ocr.enabled:
        print("[observe] WARNING: OCR is disabled, so credential redaction "
              "cannot run; frames will be uploaded unredacted")

    core = _fingerprint(timeline, config)
    cached: Dict[Key, Observation] = {}
    cached_meta: Dict[str, Any] = {}
    if use_cache and os.path.exists(cache_path):
        try:
            obs, cached_meta = read_observations(cache_path)
        except OSError:
            obs, cached_meta = [], {}
        if all(cached_meta.get(k) == v for k, v in core.items()):
            cached = {o.key: o for o in obs}
        else:
            cached_meta = {}

    meta: Dict[str, Any] = {**core, "sources": timeline.to_dicts(),
                            "vlm_model": cached_meta.get("vlm_model"),
                            "embed_model": cached_meta.get("embed_model"),
                            "asr": cached_meta.get("asr")}

    # A model change invalidates that field everywhere, and nothing else.
    if work.describe and meta["vlm_model"] not in (None, config.vlm.model):
        for o in cached.values():
            if o.status in (OK, FAILED):
                o.description, o.status = "", FAILED
    if work.embed and meta["embed_model"] not in (None, config.dino.model_name):
        for o in cached.values():
            o.embedding = None
    if work.describe:
        meta["vlm_model"] = config.vlm.model
    if work.embed:
        meta["embed_model"] = config.dino.model_name

    # Every grid sample gets an observation; collect what still needs doing.
    records: Dict[Key, Observation] = {}
    wanted: Dict[Tuple[int, bool, bool], List[float]] = {}
    grid = sample_grid(timeline, config.sampling.sample_interval_sec)
    for src in timeline.sources:
        for lt in grid[src.index]:
            key = (src.index, round(lt, 3))
            o = cached.get(key)
            if o is None:
                o = Observation(t=src.time_offset + lt, source_index=src.index,
                                source_time=lt,
                                status=FAILED if work.describe else OK)
                need_desc, need_embed = work.describe, work.embed
            else:
                o.t = src.time_offset + lt
                need_desc = (work.describe and o.status in (OK, FAILED)
                             and not o.description)
                need_embed = (work.embed and o.embedding is None
                              and o.status != UNDECODABLE)
            records[key] = o
            if need_desc or need_embed:
                wanted.setdefault((src.index, need_desc, need_embed), []).append(lt)

    if cached:
        reused = len(records) - sum(len(v) for v in wanted.values())
        print(f"[observe] reusing {reused} cached observation(s) "
              f"from {os.path.basename(cache_path)}")

    interval = config.sampling.sample_interval_sec
    per_shard = max(1, int(round(max(config.observe.shard_sec, interval) / interval)))
    shards = plan_shards(timeline, wanted, per_shard)

    if shards:
        n_samples = sum(len(s) for s in shards)
        if verbose:
            print(f"  [observe] {len(shards)} shard(s), {n_samples} samples, "
                  f"{config.observe.executor} executor, seek={config.observe.seek}")
        if use_cache:
            # Snapshot what is known now, then append shard by shard.
            write_observations(cache_path, sorted(records.values(), key=lambda o: o.t), meta)
        done = 0
        for shard, result, exc in _run_shards(shards, config, verbose):
            done += len(shard)
            if exc is not None:
                print(f"[observe] shard {os.path.basename(shard.path)} "
                      f"@{shard.sample_times[0]:.0f}s failed ({exc!r}); its "
                      f"{len(shard)} sample(s) will be retried next run")
                touched = []
                for lt in shard.sample_times:
                    o = records[(shard.source_index, round(lt, 3))]
                    if shard.describe and not o.description:
                        o.status = FAILED
                    touched.append(o)
            else:
                touched = []
                for new in result:
                    o = records[new.key]
                    _merge(o, new, shard)
                    touched.append(o)
            if use_cache:
                _append(cache_path, touched)
            if verbose:
                print(f"  [observe] {done}/{n_samples} samples")

    observations = sorted(records.values(), key=lambda o: o.t)

    transcripts_changed = False
    if config.asr.enabled:
        transcripts_changed = _attach_transcripts(
            timeline, observations, config, meta, verbose
        )
    else:
        for o in observations:
            o.transcript_text = ""

    if use_cache and (shards or transcripts_changed or meta != cached_meta):
        try:
            write_observations(cache_path, observations, meta)
            if verbose:
                print(f"  [observe] cached to {cache_path}")
        except OSError as exc:
            print(f"[observe] could not write observation cache: {exc}")

    _report(observations, work)
    return observations


def _report(observations: List[Observation], work: Work) -> None:
    """Always printed: what Pass 1 actually has, including what it lacks."""
    counts = {s: 0 for s in (OK, FAILED, WITHHELD, UNDECODABLE)}
    for o in observations:
        counts[o.status] = counts.get(o.status, 0) + 1
    described = sum(1 for o in observations if o.description)
    if work.describe:
        print(f"[observe] {len(observations)} samples: {described} described, "
              f"{counts[FAILED]} failed, {counts[WITHHELD]} withheld, "
              f"{counts[UNDECODABLE]} undecodable")
        if counts[FAILED]:
            print(f"[observe] WARNING: {counts[FAILED]} description(s) failed; "
                  f"re-running retries only those")
    else:
        print(f"[observe] {len(observations)} samples "
              f"({counts[UNDECODABLE]} undecodable)")
    redacted = [o for o in observations if o.redacted]
    if redacted:
        names = sorted({n for o in redacted for n in o.redacted})
        print(f"[redact] blacked out credentials in {len(redacted)} frame(s) "
              f"({', '.join(names)})")


if __name__ == "__main__":
    # Standalone debug: observe the first few minutes of a video, uncached,
    # and print what Pass 1 actually produces.
    import sys
    import time

    from . import llm
    from .schema import format_timestamp

    if len(sys.argv) < 2:
        print("usage: python -m cuts.observe <video_path> [minutes]")
        sys.exit(1)
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

    cfg = CutsConfig()
    tl = Timeline.build([sys.argv[1]])
    confirm_durations(tl, verbose=True)
    src = tl.sources[0]
    times = [t for t in sample_grid(tl, cfg.sampling.sample_interval_sec)[0]
             if t < minutes * 60.0]

    t0 = time.time()
    obs = observe_shard(
        Shard(0, src.path, 0.0, times, describe=llm.has_credentials()),
        cfg, verbose=True,
    )
    print(f"\n{len(obs)} observations in {time.time() - t0:.1f}s\n")
    for o in obs:
        print(f"  {format_timestamp(o.t):>7} [{o.status:>11}]  {o.description or '-'}")
