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

ASR is the exception: faster-whisper wants the whole audio track, so it runs
as its own pass before sampling, and its output is aligned onto sample times
afterwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import CutsConfig
from .media import (
    get_duration_sec,
    has_audio,
    iter_sampled_frames,
    resize_long_edge,
)
from .schema import Event
from .segmentation import SegmentationInput, segment


@dataclass
class PipelineResult:
    """Everything one run produces."""

    video_id: str
    video_path: str
    duration_sec: float
    n_samples: int
    events: List[Event] = field(default_factory=list)
    channels_used: List[str] = field(default_factory=list)
    label_backend: str = "none"
    runtime_sec: float = 0.0
    stage_times: dict = field(default_factory=dict)


def run(
    video_path: str,
    config: CutsConfig,
    label_backend: str = "auto",
    api_key: Optional[str] = None,
) -> PipelineResult:
    """Run the full pipeline on one video."""
    t_start = time.time()
    v = config.verbose
    stage_times: dict = {}
    video_id = Path(video_path).stem

    def _mark(name: str, t0: float) -> None:
        stage_times[name] = time.time() - t0
        if v:
            print(f"  [{name}] {stage_times[name]:.1f}s")

    duration_sec = get_duration_sec(video_path)
    if v:
        print(f"video: {video_path}")
        print(f"  duration {duration_sec:.0f}s, device={config.device}")

    # ---- ASR (own pass; needs the whole audio track) ----------------------
    asr_segments: List[Tuple[float, float, str]] = []
    if config.asr.enabled and has_audio(video_path):
        t0 = time.time()
        if v:
            print(f"  [asr] transcribing with {config.asr.model}...")
        from .signals.asr import transcribe
        asr_segments = transcribe(
            video_path, config.asr, device=config.device, verbose=v
        )
        _mark("asr", t0)
    elif v:
        print("  [asr] skipped (no audio stream or disabled)")

    # ---- Streaming sample + per-frame signals -----------------------------
    t0 = time.time()
    want_dino = config.segmentation.weight_dino > 0
    want_ocr = config.ocr.enabled and config.segmentation.weight_ocr > 0

    frame_indices: List[int] = []
    times: List[float] = []
    ocr_raw: List[Optional[str]] = []
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

    for pos, sf in enumerate(iter_sampled_frames(
        video_path, interval_sec=config.sampling.sample_interval_sec
    )):
        frame_indices.append(sf.frame_idx)
        times.append(sf.time_sec)

        # OCR wants full resolution; the embedder does not, so downscale only
        # for the batch we hand to DINOv2.
        if want_ocr:
            ocr_raw.append(
                ocr_image(sf.image, config.ocr)
                if should_ocr(pos, config.ocr) else None
            )

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

    n_samples = len(frame_indices)
    if v:
        print(f"  {n_samples} samples collected")
    if n_samples == 0:
        return PipelineResult(
            video_id=video_id, video_path=video_path,
            duration_sec=duration_sec, n_samples=0,
            runtime_sec=time.time() - t_start, stage_times=stage_times,
        )

    dino = (
        np.concatenate(emb_chunks) if emb_chunks
        else np.zeros((0, 0), dtype=np.float32)
    )

    ocr_texts: Optional[List[str]] = None
    if want_ocr:
        from .signals.ocr import carry_forward
        ocr_texts = carry_forward(ocr_raw)

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
    )
    _mark("segmentation", t0)

    # Attach each event's evidence, so labeling and search have it.
    _attach_evidence(events, inp, frame_indices)

    from .segmentation import build_channels
    channels_used = list(build_channels(inp, config.segmentation).keys())

    # ---- Labeling ---------------------------------------------------------
    backend = "none"
    if config.labeling.enabled and events:
        t0 = time.time()
        from .labeling import label_events
        backend = label_events(
            events, video_path, config.labeling,
            backend=label_backend, api_key=api_key, verbose=v,
        )
        _mark("labeling", t0)

    return PipelineResult(
        video_id=video_id,
        video_path=video_path,
        duration_sec=duration_sec,
        n_samples=n_samples,
        events=events,
        channels_used=channels_used,
        label_backend=backend,
        runtime_sec=time.time() - t_start,
        stage_times=stage_times,
    )


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
