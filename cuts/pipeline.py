"""End-to-end orchestrator: video in, labeled chapters out.

    sample (30s)  ->  per-frame signals  ->  segment  ->  refine  ->  label

The one structural decision worth understanding here is that **signal
extraction happens inside a single streaming decode pass**. Frames are
decoded, immediately reduced to a DINOv2 embedding and (batched, concurrently)
a VLM description — grounded by a synchronous local OCR pass on the same
frame, see `signals.ocr` — and then discarded. Nothing accumulates except the
per-sample features and short description strings, which are tiny.

The alternative — decode all sampled frames, then embed, then describe — is
the obvious way to write it and does not survive contact with real input: a
45-minute 1080p capture sampled even at 30s is still hundreds of frames at
several MB each. The streaming form holds one batch (16 frames) regardless of
video length.

The VLM channel is network-latency-bound rather than compute-bound, so unlike
DINO's batching (which exists to bound memory) its batching also exists to
run description calls concurrently — see `signals.vlm.describe_batch`.

ASR is the exception: faster-whisper wants a whole audio track, so it runs as
its own pass, per source file, and its output is aligned onto sample times
afterwards.

Input may be one file or several. Several files are read in order as a single
continuous timeline (see `cuts.media.Timeline`) — the case where one session
was recorded across multiple days. Everything downstream of sampling sees one
long video; only the joins are special, and they are handled by forcing a
chapter boundary at each one.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .config import CutsConfig, SegmentationConfig
from .media import (
    Timeline,
    has_audio,
    iter_timeline_frames,
    resize_long_edge,
)
from .schema import Event
from .segmentation import SegmentationInput, segment


@dataclass
class PipelineResult:
    """Everything one run produces."""

    video_id: str
    video_path: str          # first source, for single-file compatibility
    duration_sec: float
    n_samples: int
    events: List[Event] = field(default_factory=list)
    channels_used: List[str] = field(default_factory=list)
    label_backend: str = "none"
    runtime_sec: float = 0.0
    stage_times: dict = field(default_factory=dict)
    timeline: Optional[Timeline] = None
    # Which boundary strategy produced these events: "llm" or "curve".
    segmenter: str = "curve"
    # Frames that had a credential blacked out, and frames withheld from
    # upload entirely because one could not be painted over. See cuts.redact.
    redacted_frames: int = 0
    withheld_frames: int = 0

    @property
    def is_multi(self) -> bool:
        return self.timeline is not None and self.timeline.is_multi


def _timeline_id(timeline: Timeline) -> str:
    """Name for the whole timeline: the first file's stem, plus a count."""
    stem = Path(timeline.sources[0].path).stem
    extra = len(timeline.sources) - 1
    return stem if extra == 0 else f"{stem}+{extra}"


def run(
    video_paths: Union[str, Sequence[str]],
    config: CutsConfig,
    label_backend: str = "auto",
    api_key: Optional[str] = None,
    join_boundaries: bool = True,
) -> PipelineResult:
    """Run the full pipeline over one video or an ordered list of them.

    Several files are treated as one continuous recording, in the order given.
    With `join_boundaries` (the default) each file's start also forces a
    chapter boundary: a new file means the camera stopped and started, which
    is a real discontinuity even when the screen looks identical across it.
    """
    if isinstance(video_paths, str):
        video_paths = [video_paths]
    timeline = Timeline.build(video_paths)

    t_start = time.time()
    v = config.verbose
    stage_times: dict = {}
    video_id = _timeline_id(timeline)

    def _mark(name: str, t0: float) -> None:
        stage_times[name] = time.time() - t0
        if v:
            print(f"  [{name}] {stage_times[name]:.1f}s")

    if config.segmenter == "llm":
        return _run_llm(timeline, config, video_id, api_key, join_boundaries,
                        t_start, stage_times, _mark, v)

    if v:
        if timeline.is_multi:
            print(f"timeline: {len(timeline)} files, "
                  f"{timeline.duration_sec / 60:.1f} min total")
            for s in timeline.sources:
                print(f"  {s.index}: {s.name} ({s.duration_sec / 60:.1f} min)")
        else:
            print(f"video: {timeline.first_path}")
            print(f"  duration {timeline.duration_sec:.0f}s")
        print(f"  device={config.device}")

    # ---- Streaming sample + per-frame signals -----------------------------
    t0 = time.time()
    want_dino = config.segmentation.weight_dino > 0
    want_vlm = config.vlm.enabled and config.segmentation.weight_vlm > 0
    have_creds = bool(api_key or os.environ.get("ANTHROPIC_API_KEY"))
    if want_vlm and not have_creds:
        print("[pipeline] no ANTHROPIC_API_KEY found; skipping the VLM text "
              "channel (segmenting on the visual channel alone)")
        want_vlm = False

    # OCR is not a channel here — it only ever grounds the VLM description
    # call with exact strings a downscaled image can't reliably carry (see
    # cuts/signals/ocr.py). It runs synchronously in this loop (it's
    # local-compute-bound, unlike the VLM call) on the same frames the VLM
    # stride already selected, before either gets resized for its own use.
    want_ocr_grounding = want_vlm and config.ocr.enabled
    # Redaction rides on that same OCR pass (see cuts/redact.py), so turning
    # OCR off turns redaction off with it — which is worth saying out loud,
    # because it silently starts uploading raw frames.
    if want_vlm and config.redact.enabled and not config.ocr.enabled:
        print("[pipeline] WARNING: OCR is disabled, so credential redaction "
              "cannot run; frames will be uploaded unredacted")

    frame_indices: List[int] = []
    times: List[float] = []
    vlm_raw: List[Optional[str]] = []
    sample_source: List[int] = []      # source index per sample
    source_starts: List[int] = []      # sample positions where a file begins
    emb_chunks: List[np.ndarray] = []
    batch: List = []
    vlm_pending_enc: List[str] = []
    vlm_pending_ocr: List[str] = []
    vlm_pending_positions: List[int] = []
    # (positions, Future) per submitted chunk, oldest first.
    vlm_inflight: List[Tuple[List[int], Future]] = []
    vlm_executor: Optional[ThreadPoolExecutor] = None

    if want_dino:
        from .signals.dino import embed_stream
    if want_vlm:
        from .signals.vlm import describe_frames, encode_frame
        vlm_executor = ThreadPoolExecutor(
            max_workers=max(1, config.vlm.concurrency)
        )
    if want_ocr_grounding:
        from .redact import scrub_frame

    # Redaction tallies, reported after the pass. Names of matched patterns
    # only — never the matched text, which is the thing being redacted.
    redact_hits: List[str] = []
    redact_frames = 0
    redact_skipped = 0

    def _flush_batch() -> None:
        if not batch:
            return
        emb_chunks.append(embed_stream(
            iter(batch), config.dino,
            device=config.resolved_device(config.dino.device),
            batch_size=config.sampling.batch_size,
        ))
        batch.clear()

    def _resolve_oldest_vlm() -> None:
        """Wait for the oldest in-flight chunk and write its descriptions."""
        positions, future = vlm_inflight.pop(0)
        for p, d in zip(positions, future.result()):
            vlm_raw[p] = d

    def _submit_vlm_chunk() -> None:
        """Hand the pending frames to the executor WITHOUT waiting on them.

        This is what lets decoding overlap the network: decode is CPU-bound,
        description calls are latency-bound, and blocking here would make the
        two alternate instead of running at once.

        Backpressure keeps that from turning into unbounded memory — a fast
        decode can otherwise queue every encoded frame in the video before
        the first call returns.
        """
        if not vlm_pending_enc:
            return
        vlm_inflight.append((
            list(vlm_pending_positions),
            vlm_executor.submit(
                describe_frames,
                list(vlm_pending_enc), config.vlm,
                ocr_texts=list(vlm_pending_ocr),
                api_key=api_key, verbose=v,
            ),
        ))
        vlm_pending_enc.clear()
        vlm_pending_ocr.clear()
        vlm_pending_positions.clear()
        while len(vlm_inflight) > max(1, config.vlm.max_inflight_chunks):
            _resolve_oldest_vlm()

    if v:
        print(f"  [sample] every {config.sampling.sample_interval_sec}s "
              f"(dino={want_dino}, vlm={want_vlm}, "
              f"ocr_grounding={want_ocr_grounding})")

    current_source = -1
    local_pos = 0

    for pos, (sf, src) in enumerate(iter_timeline_frames(
        timeline, interval_sec=config.sampling.sample_interval_sec
    )):
        if src.index != current_source:
            current_source = src.index
            source_starts.append(pos)
            # Close the pending chunk at a file join so one call never mixes
            # frames from two recordings: the model is told its frames are
            # consecutive, and across a join they are hours apart.
            if want_vlm:
                _submit_vlm_chunk()
            # VLM stride restarts per file, so the first frame of every
            # source is always described. Inheriting the previous file's
            # description across a join would blur the one boundary we are
            # most certain about.
            local_pos = 0
            if v and timeline.is_multi:
                print(f"    -> {src.name} (global t={sf.time_sec:.0f}s)")

        frame_indices.append(sf.frame_idx)
        times.append(sf.time_sec)
        sample_source.append(src.index)

        if want_vlm:
            vlm_raw.append(None)
            if local_pos % max(1, config.vlm.stride) == 0:
                # OCR needs the frame at full resolution, before either this
                # or the DINO branch below resizes it. That ordering *is*
                # load-bearing now: scrub_frame blacks out credentials in
                # sf.image in place, so it has to run before the resize whose
                # output gets uploaded — and running it here means the DINO
                # branch below inherits the redacted frame too.
                ocr_text, safe = "", True
                if want_ocr_grounding:
                    ocr_text, safe, hits = scrub_frame(
                        sf.image, config.ocr, config.redact, verbose=v
                    )
                    if hits:
                        redact_hits.extend(hits)
                        redact_frames += 1
                        if v:
                            print(f"    [redact] t={sf.time_sec:.0f}s: "
                                  f"{', '.join(sorted(set(hits)))}")
                if not safe:
                    # A credential we could not paint over, or an OCR failure
                    # that leaves the frame unvouched-for. Leaving vlm_raw[pos]
                    # as None is not a gap: carry_forward fills it with the
                    # previous description, so segmentation sees continuity
                    # rather than an invented change.
                    redact_skipped += 1
                    if v:
                        print(f"    [redact] t={sf.time_sec:.0f}s: frame "
                              f"withheld (secret could not be localized)")
                else:
                    # Encode here rather than on an API thread: an encoded
                    # frame is ~15x smaller than the array, and anything
                    # submitted is held until its call returns.
                    vlm_pending_enc.append(encode_frame(
                        resize_long_edge(sf.image, config.vlm.frame_long_edge)
                    ))
                    vlm_pending_ocr.append(ocr_text)
                    vlm_pending_positions.append(pos)
                    if len(vlm_pending_enc) >= max(1, config.vlm.frames_per_call):
                        _submit_vlm_chunk()
        local_pos += 1

        if want_dino:
            sf.image = resize_long_edge(sf.image, config.sampling.frame_long_edge)
            batch.append(sf)
            if len(batch) >= config.sampling.batch_size:
                _flush_batch()

        if v and pos and pos % 200 == 0:
            print(f"    ...{pos} samples ({sf.time_sec:.0f}s) "
                  f"[{time.time() - t0:.0f}s elapsed]")

    if want_dino:
        _flush_batch()
    if want_vlm:
        # Decoding is done; submit the tail and wait out whatever is still in
        # flight. Everything up to here has been overlapping the network, so
        # this is only the last few calls rather than the whole channel.
        _submit_vlm_chunk()
        if v and vlm_inflight:
            print(f"  [vlm] waiting on {len(vlm_inflight)} in-flight "
                  f"description call(s)...")
        while vlm_inflight:
            _resolve_oldest_vlm()
        vlm_executor.shutdown()
    # Printed unconditionally, not just under -v: that the footage contains a
    # live credential is the most useful thing this pass learns, and the user
    # needs it to know what to go rotate.
    if redact_hits or redact_skipped:
        kinds = ", ".join(sorted(set(redact_hits))) or "unlocalized"
        print(f"[redact] blacked out credentials in {redact_frames} frame(s) "
              f"({kinds})"
              + (f"; withheld {redact_skipped} frame(s) entirely"
                 if redact_skipped else ""))
    _mark("signals", t0)

    duration_sec = timeline.duration_sec
    n_samples = len(frame_indices)
    if v:
        print(f"  {n_samples} samples collected")
    if n_samples == 0:
        return PipelineResult(
            video_id=video_id, video_path=timeline.first_path,
            duration_sec=duration_sec, n_samples=0,
            runtime_sec=time.time() - t_start, stage_times=stage_times,
            timeline=timeline,
        )

    # ---- ASR (own pass per source; needs whole audio tracks) --------------
    # Runs after sampling because that is when the timeline offsets are final:
    # a container's claimed duration can be short, and transcript times have to
    # be shifted by each file's *actual* placement or they land in the wrong
    # chapter.
    asr_segments: List[Tuple[float, float, str]] = []
    if config.asr.enabled:
        audible = [s for s in timeline.sources if has_audio(s.path)]
        if audible:
            t0 = time.time()
            if v:
                print(f"  [asr] transcribing {len(audible)} source(s) "
                      f"with {config.asr.model}...")
            from .signals.asr import transcribe
            for s in audible:
                segs = transcribe(
                    s.path, config.asr, device=config.device, verbose=v
                )
                asr_segments.extend(
                    (start + s.time_offset, end + s.time_offset, text)
                    for start, end, text in segs
                )
            _mark("asr", t0)
        elif v:
            print("  [asr] skipped (no audio stream in any source)")
    elif v:
        print("  [asr] disabled")

    dino = (
        np.concatenate(emb_chunks) if emb_chunks
        else np.zeros((0, 0), dtype=np.float32)
    )

    vlm_texts: Optional[List[str]] = None
    if want_vlm:
        from .signals.vlm import carry_forward
        vlm_texts = carry_forward(vlm_raw, reset_at=source_starts)

    transcript_texts: Optional[List[str]] = None
    if asr_segments:
        from .signals.asr import align_to_samples
        transcript_texts = align_to_samples(asr_segments, times)

    # ---- Segmentation -----------------------------------------------------
    t0 = time.time()
    inp = SegmentationInput(
        frame_indices=frame_indices,
        times=times,
        dino=dino if len(dino) else None,
        vlm_texts=vlm_texts,
        transcript_texts=transcript_texts,
    )
    events = segment(
        inp, config.segmentation, video_id=video_id,
        duration_sec=duration_sec, verbose=v,
        forced_bounds=source_starts if join_boundaries else None,
    )
    _mark("segmentation", t0)

    # ---- Boundary refinement (single bisection on detected boundaries) ----
    if config.refine.enabled and want_vlm and vlm_texts and len(events) > 1:
        t0 = time.time()
        from .refine import refine_boundaries
        refine_boundaries(
            events, inp, vlm_texts, timeline, config,
            locked=set(source_starts), api_key=api_key, verbose=v,
        )
        _mark("refine", t0)

    # Attach each event's evidence, so labeling and search have it. Sources
    # must be attached after refinement: it can move an event's start_time,
    # which is what source-file placement is computed from.
    _attach_sources(events, timeline)
    _attach_evidence(events, inp, frame_indices)

    from .segmentation import build_channels
    channels_used = list(build_channels(inp, config.segmentation).keys())

    # ---- Labeling ---------------------------------------------------------
    backend = "none"
    if config.labeling.enabled and events:
        t0 = time.time()
        from .labeling import label_events
        backend = label_events(
            events, timeline, config.labeling,
            backend=label_backend, api_key=api_key, verbose=v,
            ocr_config=config.ocr, redact_config=config.redact,
        )
        _mark("labeling", t0)

    return PipelineResult(
        video_id=video_id,
        video_path=timeline.first_path,
        duration_sec=duration_sec,
        n_samples=n_samples,
        events=events,
        channels_used=channels_used,
        label_backend=backend,
        runtime_sec=time.time() - t_start,
        stage_times=stage_times,
        timeline=timeline,
        redacted_frames=redact_frames,
        withheld_frames=redact_skipped,
    )


def _run_llm(
    timeline: Timeline,
    config: CutsConfig,
    video_id: str,
    api_key: Optional[str],
    join_boundaries: bool,
    t_start: float,
    stage_times: dict,
    _mark,
    v: bool,
) -> PipelineResult:
    """The two-pass path: independent observations, then LLM coalescing.

    Structurally simpler than the curve path it replaces — there is no visual
    embedding, no similarity curve, no count budget, and no separate titling
    call, because the model that draws a boundary also names it.
    """
    from .coalesce import coalesce
    from .observe import observe

    if v:
        print(f"video: {timeline.first_path}" if not timeline.is_multi
              else f"timeline: {len(timeline)} files")
        print(f"  duration {timeline.duration_sec:.0f}s  device={config.device}")

    # ---- Pass 1: observe (sharded, parallel, cached) ----------------------
    t0 = time.time()
    observations = observe(timeline, config, api_key=api_key, verbose=v)
    _mark("observe", t0)

    duration_sec = timeline.duration_sec
    if not observations:
        return PipelineResult(
            video_id=video_id, video_path=timeline.first_path,
            duration_sec=duration_sec, n_samples=0,
            runtime_sec=time.time() - t_start, stage_times=stage_times,
            timeline=timeline, segmenter="llm",
        )

    times = [o.t for o in observations]

    # ---- ASR (own pass per source; needs whole audio tracks) --------------
    if config.asr.enabled:
        audible = [s for s in timeline.sources if has_audio(s.path)]
        if audible:
            t0 = time.time()
            if v:
                print(f"  [asr] transcribing {len(audible)} source(s)...")
            from .signals.asr import align_to_samples, transcribe
            segs: List[Tuple[float, float, str]] = []
            for s in audible:
                for start, end, text in transcribe(
                    s.path, config.asr, device=config.device, verbose=v
                ):
                    segs.append((start + s.time_offset, end + s.time_offset, text))
            for o, text in zip(observations, align_to_samples(segs, times)):
                o.transcript_text = text
            _mark("asr", t0)
        elif v:
            print("  [asr] skipped (no audio stream in any source)")

    # A new capture session is a real discontinuity however similar the pixels
    # look across it, so its first observation must open a chapter.
    source_starts: List[int] = []
    seen: set = set()
    for i, o in enumerate(observations):
        if o.source_index not in seen:
            seen.add(o.source_index)
            source_starts.append(i)
    locked = source_starts[1:] if join_boundaries else []

    # ---- Pass 2: coalesce -------------------------------------------------
    t0 = time.time()
    events, backend = coalesce(
        observations, config.coalesce, config.segmentation,
        video_id=video_id, duration_sec=duration_sec, timeline=timeline,
        locked=locked, api_key=api_key, verbose=v,
    ) if config.coalesce.enabled else ([], "failed")
    _mark("coalesce", t0)

    if backend == "failed":
        events, backend = _fallback_curve(
            observations, timeline, config, video_id, duration_sec,
            locked, api_key, v,
        )

    # ---- Boundary refinement ---------------------------------------------
    # Coalescing picks boundaries from the sample grid, so they carry the same
    # half-interval of slop the grid always had; this is what removes it.
    if config.refine.enabled and config.vlm.enabled and len(events) > 1:
        t0 = time.time()
        from .refine import refine_boundaries
        inp = SegmentationInput(
            frame_indices=list(range(len(observations))),
            times=times,
            vlm_texts=[o.description for o in observations],
        )
        refine_boundaries(
            events, inp, [o.description for o in observations], timeline,
            config, locked=set(source_starts), api_key=api_key, verbose=v,
        )
        _mark("refine", t0)

    _attach_sources(events, timeline)

    channels = ["vlm"] if config.vlm.enabled else []
    if any(o.transcript_text for o in observations):
        channels.append("asr")

    return PipelineResult(
        video_id=video_id,
        video_path=timeline.first_path,
        duration_sec=duration_sec,
        n_samples=len(observations),
        events=events,
        channels_used=channels,
        label_backend=backend,
        runtime_sec=time.time() - t_start,
        stage_times=stage_times,
        timeline=timeline,
        redacted_frames=sum(1 for o in observations if o.redacted),
        withheld_frames=sum(1 for o in observations if o.withheld),
        segmenter="llm",
    )


def _fallback_curve(
    observations,
    timeline: Timeline,
    config: CutsConfig,
    video_id: str,
    duration_sec: float,
    locked: List[int],
    api_key: Optional[str],
    v: bool,
):
    """Segment the observations with the curve when coalescing failed.

    Deliberately not a re-run of the full `curve` pipeline: that would decode
    the whole video again for a DINOv2 channel, after Pass 1 has already been
    paid for. This reuses the observations as a text channel and runs the same
    deterministic machinery over them, so the cost of the fallback is one
    labeling call rather than a second analysis.
    """
    print("[pipeline] coalescing failed; falling back to curve segmentation "
          "over the observations already collected")
    from .labeling import label_events

    seg = SegmentationConfig(**vars(config.segmentation))
    seg.weight_dino = 0.0
    inp = SegmentationInput(
        frame_indices=list(range(len(observations))),
        times=[o.t for o in observations],
        vlm_texts=[o.description for o in observations],
        transcript_texts=[o.transcript_text for o in observations],
    )
    events = segment(inp, seg, video_id=video_id, duration_sec=duration_sec,
                     verbose=v, forced_bounds=locked)
    _attach_observation_evidence(events, observations)
    used = label_events(
        events, timeline, config.labeling, backend="auto",
        api_key=api_key, verbose=v,
        ocr_config=config.ocr, redact_config=config.redact,
    )
    return events, f"curve-fallback/{used}"


def _attach_observation_evidence(events: List[Event], observations) -> None:
    """Fill events produced from observations with their evidence and times.

    Frame *indices* are deliberately not set: these observations were reached
    by seeking and have no decode ordinal, so timestamps are the only honest
    identifier (see `cuts.schema`).
    """
    from .coalesce import _dedup, _pick_representative_times

    # `segment` was handed `frame_indices=range(n)`, so what it wrote into
    # `sample_indices` are positions in `observations`, not frame ordinals.
    n = len(observations)
    for e in events:
        members = [observations[i] for i in e.sample_indices if 0 <= i < n]
        if not members:
            continue
        e.sample_times = [o.t for o in members]
        e.sample_indices = []
        e.start_frame = e.end_frame = -1
        e.vlm_text = "\n".join(_dedup(o.description for o in members))
        e.transcript_text = " ".join(_dedup(o.transcript_text for o in members))
        e.representative_times = _pick_representative_times(members, 3)
        e.representative_frames = []


def _attach_sources(events: List[Event], timeline: Timeline) -> None:
    """Record which file each event opens in, and where inside it.

    A global timestamp is not something you can scrub to when the recording
    lives in five separate files, so every event carries the file and in-file
    offset alongside it.
    """
    for e in events:
        src, local = timeline.local_time(e.start_time)
        e.source_path = src.path
        e.source_index = src.index
        e.source_start_time = local


def _attach_evidence(
    events: List[Event],
    inp: SegmentationInput,
    frame_indices: List[int],
) -> None:
    """Fill each event's vlm_text / transcript_text from its samples.

    Text is deduplicated across the event's samples: a `stride > 1` carry
    forward, or simply a static screen across consecutive samples, repeats
    the same description over and over, so the raw concatenation would
    otherwise be mostly one description copy-pasted a dozen times.
    """
    pos_of = {f: i for i, f in enumerate(frame_indices)}
    for e in events:
        positions = [pos_of[f] for f in e.sample_indices if f in pos_of]
        if not positions:
            continue

        if inp.vlm_texts:
            seen = set()
            descriptions: List[str] = []
            for p in positions:
                d = (inp.vlm_texts[p] or "").strip()
                key = d.lower()
                if key and key not in seen:
                    seen.add(key)
                    descriptions.append(d)
            e.vlm_text = "\n".join(descriptions)

        if inp.transcript_texts:
            seen_t = set()
            chunks: List[str] = []
            for p in positions:
                t = (inp.transcript_texts[p] or "").strip()
                if t and t not in seen_t:
                    seen_t.add(t)
                    chunks.append(t)
            e.transcript_text = " ".join(chunks)

        from .labeling import _pick_representative
        e.representative_frames = _pick_representative(e, 3)
