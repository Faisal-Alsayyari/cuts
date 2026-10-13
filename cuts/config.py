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

    sampling     -> cuts.observe         (the sample grid; DINOv2 frame size)
    observe      -> cuts.observe         (Pass 1: sharded, resumable, cached)
    dino / vlm / ocr / asr -> cuts.signals.*  (per-sample feature extraction;
                                                ocr grounds vlm, not a channel)
    redact       -> cuts.redact          (black out credentials pre-upload)
    llm          -> cuts.llm             (shared client + global request cap)
    coalesce     -> cuts.coalesce        (Pass 2: chapters from observations)
    segmentation -> cuts.segmentation    (EFS Stage 1: the `curve` alternative
                                          to coalesce, and its fallback)
    refine       -> cuts.refine          (single-bisection boundary sharpening)
    labeling     -> cuts.labeling        (chapter titles for curve chapters;
                                          coalesce titles its own)

`CutsConfig.segmenter` picks between the two boundary strategies — see its
comment for what the choice costs.
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

    The EFS paper samples candidate frames uniformly at 1 fps; that made sense
    when the per-sample cost was a cheap local embedding. Now that the text
    channel is a Claude vision call (see VLMConfig), sample count drives API
    cost and wall-clock directly, and a coding session's activity does not
    meaningfully change sample-to-sample at 1 fps anyway. 30s is a coarser
    default that still catches activity changes reliably in practice; the
    boundary-refinement pass (cuts.refine) recovers precision within a gap
    without paying for dense sampling everywhere.
    """

    # Seconds between sampled frames.
    sample_interval_sec: float = 30.0
    # Long-edge (pixels) each sampled frame is resized to before it is handed
    # to DINOv2 (curve segmenter only). The VLM channel does its own resize
    # (VLMConfig.frame_long_edge) since it wants a different, larger, target.
    frame_long_edge: int = 448
    # Frames per DINOv2 forward pass. Tuned down for 8 GB unified memory.
    batch_size: int = 16


@dataclass
class ObserveConfig:
    """Pass 1 — how sampled frames are gathered, in independent shards.

    The timeline is cut into fixed-duration shards and each is processed by a
    pure function (`cuts.observe.observe_shard`) with no shared state. Because
    seek cost does not grow with depth into the file (see the performance note
    in `cuts.media`), a shard covering the last ten minutes of a three-hour
    capture costs the same as one covering the first ten. Wall time is
    therefore set by the slowest single shard rather than by total length —
    the property that makes this containerizable later without touching the
    function itself.
    """

    # Seconds of timeline per shard. At the default 30s sampling this is ~10
    # samples per shard: large enough to amortize worker startup and the OCR
    # engine's ONNX load, small enough to parallelize usefully.
    shard_sec: float = 300.0
    # Parallel workers. 0 = os.cpu_count().
    workers: int = 0
    # "thread" | "process" | "serial".
    #
    # Threads, on measurement, and the reason is worth knowing: RapidOCR's
    # ONNX session already parallelizes across every core, so OCR — the
    # dominant local cost — is saturating the machine before any sharding
    # happens. Adding processes on top oversubscribes it. Measured on a 9-min
    # capture, 9 shards, 8 cores: serial 34.4s, 8 threads 24.3s, 4 processes
    # 44.1s. Processes are *slower than serial*.
    #
    # Threads win the remaining 1.4x by overlapping decode and API latency
    # with OCR, and they are safe here: thread, process and serial runs were
    # verified to produce byte-identical OCR text.
    #
    # "process" stays available because it is the honest local rehearsal for
    # the container case, where each worker is a separate machine and the
    # oversubscription does not exist. "serial" is for debugging and for
    # proving shards really are independent.
    executor: str = "thread"
    # Seek to each sample instead of decoding sequentially. False falls back to
    # a sequential decode within each shard — correct but linear, for footage
    # where seeking proves unreliable.
    seek: bool = True
    # How far from the requested time a decoded frame may land before it is
    # discarded as a miss. Measured error on real 1080p screen capture is under
    # 0.05s, so this is a guard against pathological containers, not a
    # working tolerance.
    seek_tolerance_sec: float = 2.0
    # Cache observations to <stem>.observations.jsonl and reuse them when the
    # inputs match. This is what makes Pass 2 cheap to iterate on: coalescing
    # can be re-run any number of times against one paid observation pass.
    # The cache is written shard by shard, so an interrupted run resumes, and
    # samples whose description failed are retried on the next run.
    cache: bool = True


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
    """Local OCR, used only as grounding text for the VLM description call.

    This is not an independent channel or a segmentation signal (that role
    belongs to VLMConfig below) — it exists solely to hand the vision model
    exact strings it would otherwise have to guess at from a downscaled
    image. It runs on the same frames the VLM already samples, at full
    resolution, before any resize, so it does not inherit the API image's
    downscale-driven legibility loss.
    """

    enabled: bool = True
    # Minimum OCR detection confidence to keep a line. rapidocr returns scores
    # in [0, 1]; 0.5 cuts most garbage without losing small UI text.
    min_confidence: float = 0.5
    # Upscale factor applied before OCR. 2-3x is strongly recommended for
    # 1080p+ screen recordings where editor/terminal text is small. 1 = off.
    upscale_factor: int = 2
    # Optional fractional (x1, y1, x2, y2) crop boxes in [0, 1] space. OCR runs
    # on each crop separately and results are merged. None = full frame.
    roi: Optional[List[Tuple[float, float, float, float]]] = None
    # Cap on the grounding text handed to the VLM call, in characters. This is
    # a hint, not the deliverable, so a dense screen (a long stack trace, a
    # wall of log lines) does not need to be passed through in full to give
    # the model what it needs.
    max_chars: int = 1000


@dataclass
class RedactConfig:
    """Black out credentials in a frame before it is uploaded.

    A screen recording of real work will sooner or later contain a `.env`
    file, a terminal echoing a token, or a dashboard showing a live key — and
    every sampled frame is sent to an external API. This pass is what stands
    between the two: the OCR text extracted for grounding (see OCRConfig) is
    matched against known credential shapes, and any line that matches is
    filled solid black in the frame *and* replaced in the grounding text.

    It only catches what OCR actually read. Text too small, too low-contrast,
    or in a font the engine mangles is not seen here and is not redacted, so
    this reduces exposure rather than guaranteeing its absence. It is not a
    substitute for keeping credentials off screen while recording.

    Disabling this while leaving `OCRConfig.enabled` on is a deliberate
    escape hatch for footage known to be clean; disabling *OCR* also disables
    redaction, since there is nothing left to match on, and the pipeline warns
    when that combination is set.
    """

    enabled: bool = True
    # Pixels of margin added around a matched line's box before filling. OCR
    # boxes hug the glyphs, and a tight fill can leave a readable ghost of the
    # ascenders and descenders once JPEG compression smears the edges.
    pad_px: int = 4
    # What a redacted line is replaced with in the grounding text. Kept
    # visible rather than deleted so the vision model can describe "a config
    # file with a redacted secret" instead of inventing what was there.
    placeholder: str = "[REDACTED]"
    # When a credential is found in the frame's text but cannot be tied to a
    # specific box, there is nothing to draw over — so the frame is skipped
    # entirely rather than uploaded with the secret still in the pixels.
    skip_frame_on_unlocalized: bool = True
    # Extra regexes appended to the built-in set in cuts.redact. Useful for
    # site-specific credential shapes (an internal token prefix, say).
    extra_patterns: List[str] = field(default_factory=list)


@dataclass
class VLMConfig:
    """Per-frame screen descriptions, written by a Claude vision call.

    This is the text channel: a small vision model writes a short factual
    description of what's on screen, grounded with exact strings from local
    OCR (see OCRConfig) so it doesn't have to read small text off a
    downscaled image itself. For screen recordings this is usually a
    *stronger* semantic signal than the visual embedding alone — two
    different files of code look nearly identical to a self-supervised
    natural-image model, but what this channel says about them differs
    completely — and unlike a raw OCR dump it is naturally concise: evidence
    that reads like an event log entry, not a transcript to skim past.

    The tradeoff this makes deliberately: local compute for network calls.
    Requires Anthropic credentials. There is no fallback to a plain-OCR
    channel: without credentials the default `llm` segmenter refuses to
    start, and `--segmenter curve` runs on the visual channel alone.

    Concurrency is not configured here: every API call goes through
    `cuts.llm`, which caps requests in flight globally (LLMConfig).
    """

    enabled: bool = True
    # claude-haiku-4-5 is fast and cheap enough that per-frame description at
    # hundreds of samples costs a few dollars, not a redesign of the sampling
    # rate. Bump to a stronger model if descriptions of dense/small text need
    # to be more reliable — though with OCR grounding, model quality is a
    # smaller lever than it was for reading text directly off pixels.
    model: str = "claude-haiku-4-5"
    # Frames batched into a single API call. Several frames per call is
    # cheaper (the system prompt is sent once instead of once per frame) and
    # lower-latency (fewer round trips), and it lets the model see a frame's
    # neighbours — which it uses to identify what it is looking at, without
    # ever referring to them in the output. See `_DESCRIBE_SYSTEM` in
    # cuts/signals/vlm.py for why descriptions must stay standalone.
    frames_per_call: int = 6
    # Per-frame output budget. The request's actual max_tokens is derived from
    # this and the batch size, so raising `frames_per_call` cannot silently
    # truncate the tail of a response. Short by design: one sentence per
    # frame, not a transcript.
    max_tokens_per_frame: int = 60
    # Long-edge (pixels) sent to the API. Keeps payload size (and therefore
    # cost) small. Deliberately not raised for legibility — that's what OCR
    # grounding is for; this only needs to carry visual/layout context.
    frame_long_edge: int = 768


@dataclass
class LLMConfig:
    """Shared settings for every Claude API call (see cuts.llm)."""

    # Requests in flight at once, across every shard, thread and stage of a
    # run. Description calls are network-latency-bound, so this is the speed
    # lever for Pass 1; it is also what keeps a many-core machine from
    # tripping rate limits by firing one batch per shard at once.
    max_concurrency: int = 8


@dataclass
class RefineConfig:
    """Single-bisection boundary sharpening (see cuts.refine).

    Coarse sampling means a detected boundary is only known to within one
    sampling interval. This pass closes that gap by exactly one level — no
    recursion — for each detected boundary: decode the frame at the midpoint,
    describe it, and ask a small classification call which half of the gap
    the transition actually falls in.
    """

    enabled: bool = True
    # "" = reuse VLMConfig.model.
    model: str = ""
    max_tokens: int = 100


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
    `weight_dino=1.0, weight_vlm=0.0, weight_asr=0.0`.
    """

    # --- Similarity curve (paper eq. 1) ---------------------------------
    # Sliding-window half-width l. The paper tests {3, 5, 7} and finds 3 best.
    # Each frame's score averages cos(f_i, f_j) over neighbours j, weighted
    # linearly by 1 - |i-j|/(l+1) so nearer neighbours dominate.
    window_size: int = 3

    # --- Channel fusion (our extension for the screen-recording domain) --
    # Weights are normalized over whatever channels are actually available, so
    # dropping the VLM channel or ASR reweights the rest rather than shrinking
    # the signal.
    weight_dino: float = 0.5
    weight_vlm: float = 0.5
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
class CoalesceConfig:
    """Pass 2 — chapter boundaries decided by a model reading every observation.

    The alternative to `SegmentationConfig`, and the reason this exists: that
    one spends a *count budget*, merging the most-similar adjacent partitions
    until M remain. It cannot tell that twenty minutes of irrelevant debugging
    is one thing, and it produces roughly uniform chapters because a budget
    forces them. A model reading the descriptions makes a judgement instead,
    so a dead stretch can collapse to one chapter and a pivotal minute can
    split into several.

    What that costs, stated plainly: boundaries are no longer deterministic,
    and the chapter count and minimum duration are no longer guaranteed by
    construction. They are enforced afterwards by validation
    (`cuts.coalesce._validate_and_repair`), and `--segmenter curve` remains
    available as a deterministic baseline to compare against.
    """

    enabled: bool = True
    # This is one call per run over the whole timeline, and it is the call that
    # decides the product's entire output — worth the strong model.
    model: str = "claude-opus-5"
    # Covers thinking as well as the visible cut points; sized for a long
    # timeline's worth of chapters.
    max_tokens: int = 16000
    effort: str = "medium"
    # Free-text statement of what the index is FOR ("a tutorial walkthrough",
    # "find where I broke the build"). "Boring" is not a property of footage —
    # it is a property of footage relative to a purpose, and without this the
    # model can only fall back on re-deriving visual sameness.
    intent: str = ""
    # Include per-observation OCR text in the prompt. Off by default on size:
    # OCR runs ~250 tokens per observation against ~33 for a description, so it
    # is roughly 8x the whole rest of the prompt and dominates cost on long
    # timelines. Worth turning on when titles need exact on-screen strings.
    include_ocr: bool = False
    # Past this many observations the single-call form gets unwieldy and
    # windowing would be needed; the run warns rather than silently degrading.
    max_observations: int = 1500


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
    # 0 = text-only labeling (VLM descriptions + transcript), which is much
    # cheaper and often sufficient for screen recordings, where the VLM
    # channel already captures the semantically decisive content.
    frames_per_event: int = 2
    # Long-edge (pixels) of frames sent to the API. Keeps payloads small.
    frame_long_edge: int = 768
    # Characters of VLM description text per event included in the prompt.
    max_vlm_chars: int = 1500
    # Characters of transcript per event included in the prompt.
    max_transcript_chars: int = 1500


@dataclass
class CutsConfig:
    """Top-level config aggregating every sub-config plus runtime knobs."""

    device: str = field(default_factory=_default_device)

    # Which pass decides chapter boundaries:
    #   "llm"   — Pass 1 observations + Pass 2 coalescing (cuts.observe /
    #             cuts.coalesce). Judgement-based, non-deterministic, and the
    #             only path that can collapse a long uninteresting stretch.
    #   "curve" — the DINOv2 + TF-IDF similarity curve and count-budget merge
    #             (cuts.segmentation), over the same Pass 1 observations.
    #             Deterministic for a given observation cache; the baseline to
    #             compare against, the fallback when coalescing fails, and the
    #             only mode that works without API credentials.
    segmenter: str = "llm"

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    observe: ObserveConfig = field(default_factory=ObserveConfig)
    dino: DinoConfig = field(default_factory=DinoConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    redact: RedactConfig = field(default_factory=RedactConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    coalesce: CoalesceConfig = field(default_factory=CoalesceConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)

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
    print(f"  segmentation weights (d/v/a) = "
          f"{cfg.segmentation.weight_dino}/{cfg.segmentation.weight_vlm}/"
          f"{cfg.segmentation.weight_asr}")
    print(f"  labeling.model               = {cfg.labeling.model}")
