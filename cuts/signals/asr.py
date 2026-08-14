"""Speech channel — transcript from the audio track.

Wraps `faster-whisper`. The whole audio track is transcribed in one pass,
producing (start_sec, end_sec, text) segments, which are then aligned onto the
video's sample times so speech becomes a per-sample signal like OCR text.

Screen recordings frequently have no narration at all, so every entry point
here degrades quietly: no audio stream, no faster-whisper installed, or ASR
disabled all return empty rather than raising. Callers treat an empty
transcript as "this channel has no opinion", not as an error.

All heavy imports are lazy so the rest of the package works on machines
without faster-whisper.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..config import ASRConfig
from ..media import has_audio


# Lazy model singleton keyed by (model_name, device).
_ASR_CACHE: dict = {}


def _get_model(model_name: str, device: str):
    key = (model_name, device)
    if key in _ASR_CACHE:
        return _ASR_CACHE[key]
    from faster_whisper import WhisperModel  # type: ignore

    # float16 is much faster on CUDA; int8 is the right choice on CPU. MPS is
    # not a supported ctranslate2 backend, so Apple Silicon runs on CPU here.
    if device == "cuda":
        compute_type = "float16"
    else:
        device, compute_type = "cpu", "int8"
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception:
        # GPU init can fail on a machine that reports CUDA but lacks cuDNN.
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
    _ASR_CACHE[key] = model
    return model


def transcribe(
    video_path: str,
    config: ASRConfig,
    device: str = "cpu",
    verbose: bool = False,
) -> List[Tuple[float, float, str]]:
    """Transcribe the full audio track.

    Returns (start_sec, end_sec, text) tuples, or [] when ASR is disabled,
    the video has no audio, or faster-whisper is unavailable.
    """
    if not config.enabled:
        return []
    if not has_audio(video_path):
        if verbose:
            print("  ASR: no audio stream, skipping")
        return []
    try:
        model = _get_model(config.model, device)
    except ImportError:
        if verbose:
            print("  ASR: faster-whisper not installed, skipping")
        return []

    segments, _info = model.transcribe(
        video_path,
        beam_size=config.beam_size,
        vad_filter=config.vad_filter,
    )
    out: List[Tuple[float, float, str]] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            out.append((float(seg.start), float(seg.end), text))
    if verbose:
        print(f"  ASR: {len(out)} transcript segments")
    return out


def align_to_samples(
    asr_segments: Sequence[Tuple[float, float, str]],
    sample_times: Sequence[float],
    window_sec: float = 0.0,
) -> List[str]:
    """Map transcript segments onto sample times.

    Returns one string per sample time: the text of every transcript segment
    whose span covers that instant. `window_sec` widens each sample into an
    interval, which is useful when samples are sparse relative to speech —
    with 1 fps sampling and typical Whisper segment lengths the default
    point-in-time match is already dense enough.

    Both sequences are walked once in time order rather than compared
    pairwise; a long video can have thousands of each.
    """
    n = len(sample_times)
    out: List[str] = [""] * n
    if not asr_segments or n == 0:
        return out

    ordered = sorted(asr_segments, key=lambda x: x[0])
    cursor = 0
    active: List[Tuple[float, float, str]] = []

    for i, t in enumerate(sample_times):
        lo, hi = t - window_sec, t + window_sec
        # Admit every segment that has started by the end of this window.
        while cursor < len(ordered) and ordered[cursor][0] <= hi:
            active.append(ordered[cursor])
            cursor += 1
        # Retire segments that ended before this window began.
        active = [s for s in active if s[1] >= lo]
        if active:
            out[i] = " ".join(s[2] for s in active)
    return out


if __name__ == "__main__":
    # Standalone debug: transcribe a video and show the first few segments.
    import sys
    import time

    from ..config import CutsConfig

    if len(sys.argv) < 2:
        print("usage: python -m cuts.signals.asr <video_path>")
        sys.exit(1)
    cfg = CutsConfig()
    t0 = time.time()
    segs = transcribe(sys.argv[1], cfg.asr, device=cfg.device, verbose=True)
    print(f"{len(segs)} segments in {time.time() - t0:.1f}s")
    for s, e, txt in segs[:10]:
        print(f"  [{s:7.2f}-{e:7.2f}] {txt}")
