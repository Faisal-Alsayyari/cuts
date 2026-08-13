"""Optional ASR (audio transcription) over the full video.

Wraps `faster-whisper`. We transcribe the whole audio track once, producing
(start_sec, end_sec, text) segments, then align those to our video
SegmentRecords by time-overlap.

All heavy deps are imported lazily so the rest of the retrieval package works
on machines without `faster-whisper` installed.

Pipeline:
  1. ``has_audio(video_path)`` — quick PyAV probe; skip if False.
  2. ``transcribe(video_path, config)`` — returns list of
     (start_sec, end_sec, text) tuples from faster-whisper.
  3. ``attach_transcript(segments, asr_segments)`` — for each SegmentRecord,
     concatenate all ASR segments whose time window overlaps the segment.
"""

from __future__ import annotations

from typing import List, Tuple

import av

from ..config import RetrievalConfig
from .schema import SegmentRecord


# Lazy ASR model singleton keyed by (model_name, device).
_ASR_CACHE = {}


def has_audio(video_path: str) -> bool:
    """Return True when the container has at least one audio stream."""
    try:
        container = av.open(video_path)
    except av.AVError:
        return False
    try:
        return len(container.streams.audio) > 0
    finally:
        container.close()


def _get_asr(model_name: str, device: str):
    key = (model_name, device)
    if key in _ASR_CACHE:
        return _ASR_CACHE[key]
    from faster_whisper import WhisperModel  # type: ignore

    # compute_type: float16 on GPU is much faster; int8 is fine on CPU.
    compute_type = "float16" if device == "cuda" else "int8"
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception:
        # Fall back to CPU/int8 if GPU init failed (e.g. cuDNN missing).
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
    _ASR_CACHE[key] = model
    return model


def transcribe(
    video_path: str,
    config: RetrievalConfig,
    device: str = "cpu",
    verbose: bool = False,
) -> List[Tuple[float, float, str]]:
    """Run faster-whisper on the full audio track of ``video_path``.

    Returns a list of (start_sec, end_sec, text) tuples. Returns [] when
    ASR is disabled, the video has no audio, or faster-whisper is not
    installed.
    """
    if not config.enable_asr:
        return []
    if not has_audio(video_path):
        if verbose:
            print("  ASR: no audio stream, skipping")
        return []
    try:
        model = _get_asr(config.asr_model, device)
    except ImportError:
        if verbose:
            print("  ASR: faster-whisper not installed, skipping")
        return []

    segments, _info = model.transcribe(
        video_path,
        beam_size=config.asr_beam_size,
        vad_filter=True,  # skip silence so timings stay tight
    )
    out: List[Tuple[float, float, str]] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        out.append((float(seg.start), float(seg.end), text))
    if verbose:
        print(f"  ASR: {len(out)} transcript segments")
    return out


def attach_transcript(
    segments: List[SegmentRecord],
    asr_segments: List[Tuple[float, float, str]],
) -> None:
    """For each SegmentRecord, concatenate overlapping ASR text.

    Overlap is time-interval intersection > 0. One ASR segment may attach to
    multiple video segments (a sentence crossing a shot boundary).
    """
    if not segments or not asr_segments:
        return

    # Sort ASR by start for cheap two-pointer walk.
    asr_sorted = sorted(asr_segments, key=lambda x: x[0])

    for seg in segments:
        s, e = seg.start_time, seg.end_time
        pieces: List[str] = []
        for (a_s, a_e, text) in asr_sorted:
            if a_e < s:
                continue
            if a_s > e:
                break
            # Overlap exists.
            pieces.append(text)
        if pieces:
            seg.transcript_text = " ".join(pieces)
