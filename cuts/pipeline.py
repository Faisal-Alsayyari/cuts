"""End-to-end orchestrator: video in, labeled chapters out.

    sample (1 fps)  ->  per-frame signals  ->  segment  ->  label  ->  chapters

The one structural decision worth understanding here is that **signal
extraction happens inside a single streaming decode pass**. Frames are decoded,
immediately reduced to a DINOv2 embedding and an OCR string, and then
discarded. Nothing accumulates except the per-sample features, which are tiny.

The alternative — decode all sampled frames, then embed, then OCR — is the
obvious way to write it and does not survive contact with real input: a
45-minute 1080p capture sampled at 1 fps is ~2,700 frames at ~6 MB each, about
16 GB of pixel data. The streaming form holds one batch (16 frames, ~100 MB)
regardless of video length.

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

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .config import CutsConfig
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
    want_ocr = config.ocr.enabled and config.segmentation.weight_ocr > 0

    frame_indices: List[int] = []
    times: List[float] = []
    ocr_raw: List[Optional[str]] = []
    sample_source: List[int] = []      # source index per sample
    source_starts: List[int] = []      # sample positions where a file begins
    emb_chunks: List[np.ndarray] = []
    batch: List = []

    if want_dino:
        from .signals.dino import embed_stream
    if want_ocr:
        from .signals.ocr import ocr_image, should_ocr

    def _flush_batch() -> None:
        if not batch:
            return
        emb_chunks.append(embed_stream(
            iter(batch), config.dino,
            device=config.resolved_device(config.dino.device),
            batch_size=config.sampling.batch_size,
        ))
        batch.clear()

    if v:
        print(f"  [sample] every {config.sampling.sample_interval_sec}s "
              f"(dino={want_dino}, ocr={want_ocr})")

    current_source = -1
    local_pos = 0

    for pos, (sf, src) in enumerate(iter_timeline_frames(
        timeline, interval_sec=config.sampling.sample_interval_sec
    )):
        if src.index != current_source:
            current_source = src.index
            source_starts.append(pos)
            # OCR stride restarts per file, so the first frame of every source
            # is always read. Inheriting the previous file's text across a join
            # would blur the one boundary we are most certain about.
            local_pos = 0
            if v and timeline.is_multi:
                print(f"    -> {src.name} (global t={sf.time_sec:.0f}s)")

        frame_indices.append(sf.frame_idx)
        times.append(sf.time_sec)
        sample_source.append(src.index)

        # OCR wants full resolution; the embedder does not, so downscale only
        # for the batch we hand to DINOv2.
        if want_ocr:
            ocr_raw.append(
                ocr_image(sf.image, config.ocr)
                if should_ocr(local_pos, config.ocr) else None
            )
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

    ocr_texts: Optional[List[str]] = None
    if want_ocr:
        from .signals.ocr import carry_forward
        ocr_texts = carry_forward(ocr_raw, reset_at=source_starts)

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
        ocr_texts=ocr_texts,
        transcript_texts=transcript_texts,
    )
    events = segment(
        inp, config.segmentation, video_id=video_id,
        duration_sec=duration_sec, verbose=v,
        forced_bounds=source_starts if join_boundaries else None,
    )
    _mark("segmentation", t0)

    # Attach each event's evidence, so labeling and search have it.
    _attach_evidence(events, inp, frame_indices)
    _attach_sources(events, timeline)

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
    )


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
    """Fill each event's ocr_text / transcript_text from its samples.

    Text is deduplicated across the event's samples: consecutive frames of a
    screen recording repeat almost all of their text, so the raw concatenation
    is mostly the same lines over and over.
    """
    pos_of = {f: i for i, f in enumerate(frame_indices)}
    for e in events:
        positions = [pos_of[f] for f in e.sample_indices if f in pos_of]
        if not positions:
            continue

        if inp.ocr_texts:
            seen = set()
            lines: List[str] = []
            for p in positions:
                for line in (inp.ocr_texts[p] or "").split("\n"):
                    key = line.strip().lower()
                    if key and key not in seen:
                        seen.add(key)
                        lines.append(line.strip())
            e.ocr_text = "\n".join(lines)

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
