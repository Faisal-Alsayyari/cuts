"""Single configuration dataclass for the entire `cuts` pipeline.

`cuts` performs semantic indexing and temporal navigation of long-form video.
The primary target is screen recordings (coding sessions), where the goal is
to turn a multi-hour raw capture into a small set of labeled, timestamped
chapters.

All thresholds, window sizes, device strings, and model toggles live here so
that no module body hardcodes a tuning constant. Every module receives a
`CutsConfig` instance (or a sub-field of one) at call time, which makes
hyperparameter sweeps trivial: instantiate a different config and re-run.

Section names map onto the pipeline stages:

    sampling     -> cuts.media           (decode 1 frame per N seconds)
    dino / ocr / asr -> cuts.signals.*   (per-sample feature extraction)
    segmentation -> cuts.segmentation    (EFS Stage 1: events from signals)
    labeling     -> cuts.labeling        (Claude-written chapter titles)
    select       -> cuts.select          (EFS Stages 2-3: query-driven frames)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


def _default_device() -> str:
    """Pick the best available torch device. Centralized so every module agrees.

    Order is CUDA -> MPS -> CPU. MPS matters here: the primary dev machine is
    Apple Silicon, where a CUDA-only check silently degrades every model to
    CPU and makes DINOv2 roughly an order of magnitude slower.
    """
    try:
        import torch
    except ImportError:  # torch is optional for config-only imports
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class SamplingConfig:
    """How densely we walk the video before any model runs.

    The EFS paper samples candidate frames uniformly at 1 fps. That is also a
    sensible default for screen recordings: a coding session changes state on
    the order of seconds, not frames.
    """

    # Seconds between sampled frames. 1.0 == the paper's 1 fps.
    # Raise this for very long inputs; a 30-hour capture at 1 fps is 108k
    # frames, where 2-4s sampling costs little recall and a lot less compute.
    sample_interval_sec: float = 1.0
    # Long-edge (pixels) each sampled frame is resized to before it is handed
    # to any signal extractor. DINOv2 crops to 224 anyway, and OCR gets the
    # full-resolution frame separately, so this only bounds peak memory.
    frame_long_edge: int = 448
    # Decoded frames are never all held at once; this bounds the working set
    # handed to the embedder per batch. Tuned down for 8 GB unified memory.
    batch_size: int = 16


@dataclass
class DinoConfig:
    """Visual embedding backbone for the EFS temporal similarity curve."""

    # HuggingFace model id. ViT-S/14 (21M params) is the 8 GB-friendly choice;
    # facebook/dinov2-base (86M) is meaningfully stronger if memory allows.
    model_name: str = "facebook/dinov2-small"
    # Use the pooled CLS embedding rather than patch tokens. The EFS similarity
    # curve only needs one vector per frame.
    use_cls_token: bool = True
    # Inference device. Empty string = inherit CutsConfig.device.
    device: str = ""


@dataclass
class OCRConfig:
    """Text extracted from sampled frames.

    For screen recordings this is often a *stronger* semantic signal than the
    visual embedding: two different files of code look nearly identical to a
    self-supervised natural-image model, but their text differs completely.
    """

    enabled: bool = True
    # Minimum OCR detection confidence to keep a line. rapidocr returns scores
    # in [0, 1]; 0.5 cuts most garbage without losing small UI text.
    min_confidence: float = 0.5
    # Upscale factor applied before OCR. 2-3x is strongly recommended for
    # 1080p screen recordings where editor/terminal text is small. 1 = off.
    upscale_factor: int = 2
    # Optional fractional (x1, y1, x2, y2) crop boxes in [0, 1] space. OCR runs
    # on each crop separately and results are merged. None = full frame.
    # Example: [(0.0, 0.0, 1.0, 0.90)] to exclude a bottom 10% taskbar.
    roi: Optional[List[Tuple[float, float, float, float]]] = None
    # OCR is by far the slowest per-frame signal. Run it on every Nth sampled
    # frame and interpolate between them. 1 = OCR every sampled frame.
    stride: int = 1
    # Write the exact images fed to OCR as PNGs for visual inspection.
    save_debug_png: bool = False


@dataclass
class ASRConfig:
    """Optional speech transcript, used both as a signal and as label context."""

    # Auto = transcribe when the container has an audio stream, skip otherwise.
    # Screen recordings frequently have no narration, so this must degrade
    # gracefully rather than error.
    enabled: bool = True
    # faster-whisper model name. "base" for CPU/MPS, "small" with a GPU.
    model: str = "base"
    # Beam size for decoding. 1 = greedy (fastest).
    beam_size: int = 1
    # Skip silence so segment timings stay tight.
    vad_filter: bool = True


@dataclass
class SegmentationConfig:
    """EFS Stage 1 — event boundaries from a fused temporal similarity curve.

    Implements arXiv 2603.00983 Stage 1 (Chen et al., "Event-Anchored Frame
    Selection for Effective Long-Video Understanding"), generalized so the
    similarity curve can fuse multiple signal channels rather than DINOv2
    alone. The paper's exact behaviour is recovered with
    `weight_dino=1.0, weight_ocr=0.0, weight_asr=0.0`.
    """

    # --- Similarity curve (paper eq. 1) ---------------------------------
    # Sliding-window half-width l. The paper tests {3, 5, 7} and finds 3 best.
    # Each frame's score averages cos(f_i, f_j) over neighbours j, weighted
    # linearly by 1 - |i-j|/(l+1) so nearer neighbours dominate.
    window_size: int = 3

    # --- Channel fusion (our extension for the screen-recording domain) --
    # Weights are normalized over whatever channels are actually available, so
    # dropping OCR or ASR reweights the rest rather than shrinking the signal.
    weight_dino: float = 0.5
    weight_ocr: float = 0.5
    weight_asr: float = 0.0

    # --- Boundary count -------------------------------------------------
    # Target number of events M. The paper sweeps 1-16 and finds 10-14 best
    # for ~17-minute videos. None = derive from duration via minutes_per_event.
    target_events: Optional[int] = None
    # Used when target_events is None: aim for one event per this many minutes.
    minutes_per_event: float = 4.0
    # Hard floor/ceiling on the derived event count.
    min_events: int = 3
    max_events: int = 24
    # An event shorter than this is merged into its neighbour regardless of the
    # target count. Prevents 3-second "chapters" in the output.
    min_event_sec: float = 20.0


@dataclass
class LabelingConfig:
    """Chapter titles written by Claude from per-event evidence."""

    enabled: bool = True
    # Model id.
    model: str = "claude-opus-5"
    # Max tokens per labeling response. Titles themselves are a few words, but
    # this budget also covers thinking, which is on by default on Opus 5 —
    # sizing it to the visible output alone truncates the response.
    max_tokens: int = 8000
    # Reasoning depth. Labeling is a judgement call over short evidence, not a
    # hard reasoning problem, so medium beats the default `high` on latency
    # and cost without hurting title quality.
    effort: str = "medium"
    # Ask the API to fall back to another model if safety classifiers decline
    # the request. Screen recordings rarely trigger this, but a refusal would
    # otherwise silently cost the whole labeling pass.
    use_fallbacks: bool = True
    # How many representative frames per event are sent to the model.
    # 0 = text-only labeling (OCR + transcript), which is much cheaper and
    # often sufficient for screen recordings, where OCR already captures the
    # semantically decisive content.
    frames_per_event: int = 2
    # Long-edge (pixels) of frames sent to the API. Keeps payloads small.
    frame_long_edge: int = 768
    # Characters of OCR text per event included in the prompt.
    max_ocr_chars: int = 1500
    # Characters of transcript per event included in the prompt.
    max_transcript_chars: int = 1500
    # Label all events in a single call so the model can keep titles
    # non-redundant and consistent across the video. False = one call each.
    batch_all_events: bool = True


@dataclass
class SelectConfig:
    """EFS Stages 2-3 — query-driven keyframe selection.

    This is the part of arXiv 2603.00983 that requires a text query. It is NOT
    used by chapter generation (which is query-free); it exists for retrieval
    and for the future "find the moment matching this description" workflow.
    """

    # Number of keyframes to return (paper tests k in {8, 16, 32, 64}).
    top_k: int = 16
    # MMR tradeoff: lambda * query_relevance - (1 - lambda) * redundancy.
    mmr_lambda: float = 0.5
    # Adaptive-threshold relaxation factor alpha. The paper finds 0.5 best on
    # VideoMME and 0.1 on LongVideoBench/MLVU.
    alpha: float = 0.5
    # Scoring backend for query-frame relevance. "clip" uses open-clip and is
    # the lightweight stand-in for the paper's BLIP2-ITM.
    scorer: str = "clip"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"


@dataclass
class SearchConfig:
    """Text search over indexed events (BM25 + optional embeddings, RRF)."""

    # sentence-transformers model for event text embeddings.
    text_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    enable_text_embeddings: bool = True
    # Reciprocal Rank Fusion constant. 60 is the canonical default.
    rrf_k: int = 60
    default_top_k: int = 5
    snippet_half_chars: int = 40


@dataclass
class CutsConfig:
    """Top-level config aggregating every sub-config plus runtime knobs."""

    device: str = field(default_factory=_default_device)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    dino: DinoConfig = field(default_factory=DinoConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    # Optional cache directory for the frame_idx -> (pts, time_sec) lookup
    # tables and computed embeddings. None disables caching.
    cache_dir: Optional[str] = None

    # When True, each pipeline stage prints its name, input size, and elapsed
    # time to stdout.
    verbose: bool = False

    def resolved_device(self, override: str = "") -> str:
        """Return `override` when set, else the top-level device."""
        return override or self.device


if __name__ == "__main__":
    # Quick smoke test: print the default config and confirm device selection.
    cfg = CutsConfig()
    print("Default CutsConfig:")
    print(f"  device                       = {cfg.device}")
    print(f"  sampling.sample_interval_sec = {cfg.sampling.sample_interval_sec}")
    print(f"  dino.model_name              = {cfg.dino.model_name}")
    print(f"  segmentation.window_size     = {cfg.segmentation.window_size}")
    print(f"  segmentation weights (d/o/a) = "
          f"{cfg.segmentation.weight_dino}/{cfg.segmentation.weight_ocr}/"
          f"{cfg.segmentation.weight_asr}")
    print(f"  labeling.model               = {cfg.labeling.model}")
