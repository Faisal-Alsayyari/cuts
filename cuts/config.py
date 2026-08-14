"""Single configuration dataclass for the entire `cuts` pipeline.

All thresholds, window sizes, device strings, and model toggles live here so that
no module body hardcodes a tuning constant. Every module receives a `CutsConfig`
instance (or a sub-field of one) at call time. This makes hyperparameter sweeps
on the benchmark trivial: instantiate a different config and re-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch  # only used here to pick the default device — cheap import


def _default_device() -> str:
    """Pick CUDA when available, else CPU. Centralized so every module agrees."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PySceneDetectConfig:
    """Tuning knobs for Arm A (PySceneDetect)."""

    # AdaptiveDetector adapts threshold to local content; lower values => more
    # sensitive. Screen recordings are higher-contrast than film, so the film
    # default (3.0) over-suppresses real boundaries. Start at 2.0 and tune.
    adaptive_threshold: float = 2.0
    # ContentDetector is the legacy detector; we still run it for recall.
    content_threshold: float = 27.0
    # Don't print PySceneDetect's progress bar inside batch jobs.
    show_progress: bool = False


@dataclass
class TransNetV2Config:
    """Tuning knobs for Arm B (TransNetV2)."""

    # Probability threshold for `predictions_to_scenes`. Default 0.5 is tuned for
    # broadcast video. Screen recordings often need 0.3–0.4 for subtle cuts.
    scene_threshold: float = 0.4
    # Threshold above which a frame is considered "in" a gradual transition,
    # used to estimate the span of dissolves from `single_frame_predictions`.
    gradual_threshold: float = 0.3
    # Minimum number of consecutive elevated frames to call something gradual.
    min_gradual_frames: int = 3


@dataclass
class EnsembleConfig:
    """Tuning knobs for merging detector outputs (Stage 1 fusion)."""

    # Two boundary candidates within this many frames of each other are merged
    # into a single candidate. ±4 covers typical detector disagreement on the
    # exact frame of a cut without collapsing real adjacent cuts.
    merge_tolerance_frames: int = 4


@dataclass
class RefinementConfig:
    """Tuning knobs for Stage 2 local frame-level refinement."""

    # Half-window around each candidate to decode and analyze. Total decoded
    # frames per candidate is 2*W + 1.
    half_window: int = 8
    # Weights for the combined per-frame discontinuity signal.
    # Signal = w_hist * histogram_delta + w_mad * mean_abs_diff.
    weight_hist: float = 0.5
    weight_mad: float = 0.5
    # Thumbnail size (long edge, pixels) for the MAD signal inside refinement
    # windows. Coarse signal on a 256px thumbnail is more than sufficient to
    # locate a spike at the correct frame pair. Set 0 to disable resizing.
    refine_thumb_long_edge: int = 256
    # Histogram bins per channel for HSV histograms. 32^3 is a good speed/recall
    # tradeoff. Lower => faster, less discriminative.
    hist_bins: int = 32
    # gradual transitions (span of elevated signal > min_gradual_frames).
    # Calibrated for MAD+hist combined signal: MAD gives 0.10–0.35 on a hard
    # cut (vs SSIM's 0.70–0.90), so 0.15 is the right breakpoint.
    transition_threshold: float = 0.15
    min_gradual_frames: int = 3
    # Asymmetric search window: number of frames to decode BEFORE (window_left)
    # and AFTER (window_right) the coarse candidate. Both default to half_window
    # when None. Examples:
    #   symmetric:       window_left=6, window_right=6  (same as half_window=6)
    #   mildly left:     window_left=6, window_right=2
    #   very left:       window_left=8, window_right=0  (window ends AT coarse frame)
    window_left: Optional[int] = None
    window_right: Optional[int] = None
    # Confidence gating: skip refinement (pass through raw candidate unchanged)
    # for cuts whose detector confidence >= this threshold. None = always refine.
    # Only meaningful when candidates carry a non-zero confidence (TransNetV2).
    confidence_threshold: Optional[float] = None
    # Alternatively, refine only the bottom X% of candidates by confidence.
    # Range (0, 100]. None = always refine. Takes precedence over
    # confidence_threshold when both are set.
    confidence_top_pct: Optional[float] = None
    # Post-filter: discard candidates whose elevated span is shorter than this
    # AND whose signal returns to baseline within `motion_recovery_frames`. These
    # are typically cursor-flick / animation false positives, not state changes.
    # Disable (False) when refining hard-cut boundaries from a coarse detector:
    # a genuine cut always recovers quickly because the new shot is stable.
    motion_filter: bool = True
    motion_recovery_frames: int = 3
    motion_min_persistent_frames: int = 2


@dataclass
class EventDetectorConfig:
    """Tuning knobs for Stage 3 sub-shot UI state change events."""

    # Sample every N frames inside a shot when scanning for UI events.
    # Smaller => slower but catches shorter events. 3 is a good default for
    # 30/60 fps screen recordings.
    sample_stride: int = 3

    # ---- Stage E1: cheap coarse diff signal ---------------------------------
    # Mean-abs pixel difference (0–1 normalised) threshold for a strided-sample
    # pair to become a candidate. Replace the old ssim_delta_threshold for E1.
    # 0.05 = 5% average pixel change per thumbnail pixel.
    coarse_diff_threshold: float = 0.05
    # Resize each sampled frame to this long-edge size (pixels) before the E1
    # diff. UI state changes affect large regions so 256 px preserves the
    # signal; full-res diff is unnecessary and memory-hungry here.
    # Set 0 to disable resizing (useful for regression testing).
    scan_thumb_long_edge: int = 256

    # ---- Stage E2: candidate pruning ----------------------------------------
    # Keep only the top-K candidates by diff magnitude before running E3 SSIM
    # refinement. Caps E3 cost regardless of video length or content dynamics.
    top_k_candidates: int = 50

    # ---- Legacy / E3 --------------------------------------------------------
    # Kept for backward compatibility. No longer used by the coarse E1 scan;
    # the operative threshold for candidate selection is coarse_diff_threshold.
    ssim_delta_threshold: float = 0.15
    # Minimum shot length (in frames) worth scanning. Tiny shots cannot host
    # meaningful sub-shot events and just add noise.
    min_shot_length_frames: int = 30
    # Whether to use CLIP to label the event type. Off by default (localization
    # only is the prototype goal). When True, runs a CLIP forward pass per event.
    use_clip_labeling: bool = False


@dataclass
class BenchmarkConfig:
    """Tuning knobs for the benchmark evaluator."""

    # Tolerance window (in frames) for matching a predicted hard cut to a GT
    # hard cut. ±2 frames is the standard in the shot-detection literature.
    hard_cut_tolerance_frames: int = 2
    # Minimum interval IoU to count a gradual / UI event prediction as TP.
    interval_iou_threshold: float = 0.5


@dataclass
class RetrievalConfig:
    """Tuning knobs for the event-retrieval milestone (cuts/retrieval/*).

    Indexing pipeline: TransNetV2 shots -> hybrid segmenter -> representative
    frame sampling -> OCR + optional ASR -> BM25 + optional text/image
    embeddings -> on-disk index. Query time: RRF hybrid over available signals.
    """

    # ---- Segmenter ---------------------------------------------------------
    # Any TransNetV2 shot longer than this is split into equal sub-segments.
    max_segment_sec: float = 20.0
    # Any shot shorter than this is merged into the next shot. Avoids 0.2s
    # transition fragments becoming their own searchable unit.
    min_segment_sec: float = 2.0

    # ---- Representative frame sampling ------------------------------------
    # Seconds inset from segment start/end for the first/last representative
    # frames. Keeps us off of fade/blank frames at shot boundaries.
    sample_edge_inset_sec: float = 0.5
    # Periodic sample spacing inside long segments (also applied even if the
    # segment is at or under the max cap, only activates for >= this*2 seconds).
    sample_period_sec: float = 5.0
    # Hard cap on representative frames per segment regardless of length.
    max_frames_per_segment: int = 6
    # Thumbnail long-edge (pixels) when writing representative frames to disk.
    # 720 keeps OCR happy and stays small. Set 0 to keep native resolution.
    rep_frame_long_edge: int = 720
    # JPEG quality for representative frames on disk.
    rep_frame_jpeg_quality: int = 85

    # ---- OCR ---------------------------------------------------------------
    # Minimum OCR detection confidence to keep a line. rapidocr returns scores
    # in [0, 1]; 0.5 cuts most garbage without losing small UI text.
    ocr_min_confidence: float = 0.5
    # Perceptual-hash Hamming distance threshold for cross-segment OCR dedup.
    # If the last rep-frame of segment k-1 and the first rep-frame of segment k
    # are within this distance, reuse segment k-1's OCR and mark ocr_stale.
    ocr_phash_dedup_threshold: int = 6
    # Upscale factor applied to frames (and ROI crops) before OCR.
    # 2–3x is strongly recommended for 1080p screen recordings. 1 = off.
    ocr_upscale_factor: int = 2
    # When True, write the exact images fed to OCR as PNGs under
    # <index_dir>/ocr_debug/ for visual inspection.
    ocr_save_debug_png: bool = False
    # Optional list of fractional (x1, y1, x2, y2) crop boxes in [0, 1] space.
    # OCR runs on each crop separately; results are merged. None = full frame.
    # Example: [(0.0, 0.0, 1.0, 0.90)] to exclude a bottom 10% taskbar.
    ocr_roi: Optional[List[Tuple[float, float, float, float]]] = None
    # Minimum frames a cleaned line must appear in (ocr_text). Set to 1 to
    # disable cross-frame filtering. Auto-capped at n_frames for the segment.
    ocr_token_min_frames: int = 2
    # High-confidence bypass: lines with max score >= this threshold are kept
    # in ocr_text even if they appeared in fewer than ocr_token_min_frames.
    ocr_high_conf_threshold: float = 0.85
    # Verbose per-frame OCR breakdown; also enables ocr_save_debug_png.
    ocr_debug: bool = False

    # ---- ASR ---------------------------------------------------------------
    # Whether to attempt ASR at all. Lazy-imports faster-whisper; if False, no
    # transcript text is indexed even when audio is present.
    enable_asr: bool = False
    # faster-whisper model name. "base" for CPU, "small" when a GPU is around.
    asr_model: str = "base"
    # Beam size for faster-whisper decoding. 1 = greedy (fastest).
    asr_beam_size: int = 1

    # ---- Index / embeddings -----------------------------------------------
    # sentence-transformers model for segment text embeddings. MiniLM is 384-d,
    # CPU-friendly, and strong for short paraphrase queries.
    text_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    enable_text_embeddings: bool = True
    # open-clip image+text model for visual queries. Optional — off by default.
    enable_image_embeddings: bool = False
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"

    # ---- Search -----------------------------------------------------------
    # Reciprocal Rank Fusion constant. 60 is the canonical default.
    rrf_k: int = 60
    # Max results returned per query by default.
    default_top_k: int = 5
    # Snippet window (characters on each side of a BM25 token hit).
    snippet_half_chars: int = 40


@dataclass
class CutsConfig:
    """Top-level config aggregating every sub-config plus runtime knobs."""

    device: str = field(default_factory=_default_device)
    # Sub-configs are dataclasses so they can be replaced wholesale in sweeps.
    pyscenedetect: PySceneDetectConfig = field(default_factory=PySceneDetectConfig)
    transnetv2: TransNetV2Config = field(default_factory=TransNetV2Config)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    events: EventDetectorConfig = field(default_factory=EventDetectorConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    # Optional cache directory for the frame_idx -> (pts, time_sec) lookup
    # tables. None disables caching (recomputed every run).
    cache_dir: Optional[str] = None

    # When True, each pipeline stage prints its name, input size, and elapsed
    # time to stdout. Enable via `python -m cuts.pipeline <video> D --debug`.
    verbose: bool = False


if __name__ == "__main__":
    # Quick smoke test: print the default config and confirm device selection.
    cfg = CutsConfig()
    print("Default CutsConfig:")
    print(f"  device = {cfg.device}")
    print(f"  pyscenedetect.adaptive_threshold = {cfg.pyscenedetect.adaptive_threshold}")
    print(f"  transnetv2.scene_threshold = {cfg.transnetv2.scene_threshold}")
    print(f"  refinement.half_window = {cfg.refinement.half_window}")
    print(f"  events.sample_stride = {cfg.events.sample_stride}")
