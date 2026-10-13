"""End-to-end orchestrator: video in, titled chapters out.

There is one pipeline, whichever segmenter is chosen:

    Pass 1  observe     sample grid -> seek -> OCR + redact -> describe
                        (+ DINOv2 embedding for the curve) -> transcript
                        -> resumable cache                  => observations
    Pass 2  segment     coalesce (one model call: cut points + titles)
                        or the similarity curve (also coalesce's fallback)
            titles      from coalescing, else one labeling call
            refine      one bisection per detected boundary
            sources     which file, and where in it, each chapter opens

`chapters_from_observations` is everything after Pass 1. `run` calls it on
fresh observations; `cuts coalesce` calls it on a cache, which is why
re-chaptering from cached notes gets the same transcripts, fallback and
refinement as a full run.

Input may be one file or several. Several files are read in order as a single
continuous timeline (see `cuts.media.Timeline`); every file join is forced to
be a chapter boundary unless `join_boundaries=False`.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from . import llm
from .chapters import attach_sources, events_from_starts, join_positions
from .config import CutsConfig
from .media import Timeline
from .observe import FAILED, UNDECODABLE, WITHHELD, Observation, Work, observe
from .schema import Event

# PipelineResult.segmenter when coalescing failed and the curve took over.
FALLBACK = "llm->curve"


class PipelineError(RuntimeError):
    """A run that cannot produce a meaningful index. The message says why."""


@dataclass
class PipelineResult:
    """Everything one run produces."""

    video_id: str
    timeline: Timeline
    duration_sec: float
    observations: List[Observation]
    events: List[Event]
    # Which boundary strategy produced the events: "llm", "curve", or
    # FALLBACK when coalescing failed and the curve took over.
    segmenter: str
    # Where the titles came from: "llm" (the coalescing call), "claude" or
    # "heuristic" (cuts.labeling), or "none".
    titles: str
    channels: List[str] = field(default_factory=list)
    stage_times: Dict[str, float] = field(default_factory=dict)
    runtime_sec: float = 0.0

    @property
    def n_samples(self) -> int:
        return len(self.observations)

    @property
    def is_multi(self) -> bool:
        return self.timeline.is_multi

    def sample_counts(self) -> Dict[str, int]:
        return {
            "described": sum(1 for o in self.observations if o.description),
            "failed": sum(1 for o in self.observations if o.status == FAILED),
            "withheld": sum(1 for o in self.observations if o.status == WITHHELD),
            "undecodable": sum(1 for o in self.observations
                               if o.status == UNDECODABLE),
            "redacted": sum(1 for o in self.observations if o.redacted),
        }

    def extra(self, intent: str = "") -> Dict[str, Any]:
        """Run facts for the index's `extra` block."""
        return {
            "n_samples": self.n_samples,
            "samples": self.sample_counts(),
            "channels": self.channels,
            "segmenter": self.segmenter,
            "titles": self.titles,
            "intent": intent,
            "stage_times": self.stage_times,
        }


def timeline_id(timeline: Timeline) -> str:
    """Name for the whole timeline: the first file's stem, plus a count."""
    stem = Path(timeline.sources[0].path).stem
    extra = len(timeline.sources) - 1
    return stem if extra == 0 else f"{stem}+{extra}"


_NO_CREDENTIALS = (
    "no Anthropic credentials found (set ANTHROPIC_API_KEY, put it in a .env "
    "file, or run `ant auth login`). The default llm segmenter needs them to "
    "describe frames; --segmenter curve works without them, on the visual "
    "channel only with keyword titles."
)


def preflight(config: CutsConfig) -> bool:
    """Refuse configurations that cannot work, before anything is decoded.

    Returns whether API credentials are available.
    """
    creds = llm.has_credentials()
    if config.segmenter not in ("llm", "curve"):
        raise PipelineError(f"unknown segmenter {config.segmenter!r}")
    if config.segmenter == "llm":
        if not creds:
            raise PipelineError(_NO_CREDENTIALS)
        if not config.vlm.enabled:
            raise PipelineError("--no-vlm leaves the llm segmenter nothing to "
                                "read; use it with --segmenter curve")
    if _wants_embeddings(config) and not (
        importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")
    ):
        raise PipelineError("the curve segmenter's visual channel needs torch and "
                            "transformers (pip install 'cuts[curve]'); "
                            "or pass --no-dino")
    if config.segmenter == "curve" and config.vlm.enabled and not creds:
        print("[pipeline] no Anthropic credentials found; the curve segmenter "
              "will use the visual channel only, with keyword titles")
    return creds


def _wants_embeddings(config: CutsConfig) -> bool:
    return config.segmenter == "curve" and config.segmentation.weight_dino > 0


def run(
    video_paths: Union[str, Sequence[str]],
    config: CutsConfig,
    title_backend: str = "auto",
    join_boundaries: bool = True,
    cache_path: Optional[str] = None,
) -> PipelineResult:
    """Run the full pipeline over one video or an ordered list of them.

    Raises `PipelineError` for runs that cannot produce a meaningful index —
    most importantly the default segmenter without credentials, which used to
    run to completion and produce nothing useful.
    """
    if isinstance(video_paths, str):
        video_paths = [video_paths]
    t_start = time.time()
    llm.configure(config.llm.max_concurrency)
    creds = preflight(config)

    timeline = Timeline.build(video_paths)
    v = config.verbose
    if v:
        print(f"timeline: {len(timeline)} file(s), "
              f"{timeline.duration_sec / 60:.1f} min (header)  "
              f"segmenter={config.segmenter}  device={config.device}")

    stage_times: Dict[str, float] = {}
    t0 = time.time()
    observations = observe(
        timeline, config,
        Work(describe=config.vlm.enabled and creds,
             embed=_wants_embeddings(config),
             transcribe=config.asr.enabled),
        verbose=v, cache_path=cache_path,
    )
    _mark(stage_times, "observe", t0, v)

    return chapters_from_observations(
        observations, timeline, config, title_backend=title_backend,
        join_boundaries=join_boundaries, stage_times=stage_times,
        t_start=t_start, creds=creds,
    )


def chapters_from_observations(
    observations: List[Observation],
    timeline: Timeline,
    config: CutsConfig,
    title_backend: str = "auto",
    join_boundaries: bool = True,
    stage_times: Optional[Dict[str, float]] = None,
    t_start: Optional[float] = None,
    creds: Optional[bool] = None,
) -> PipelineResult:
    """Pass 2 and everything after it, from observations already in hand."""
    t_start = t_start or time.time()
    stage_times = stage_times if stage_times is not None else {}
    creds = llm.has_credentials() if creds is None else creds
    v = config.verbose
    duration = timeline.duration_sec
    video_id = timeline_id(timeline)

    if not observations:
        raise PipelineError("no samples to analyze (is the video empty?)")
    if config.segmenter == "llm":
        if not creds:
            raise PipelineError(_NO_CREDENTIALS)
        if not any(o.description for o in observations):
            raise PipelineError(
                "no frame was described (every description call failed); "
                "re-run to retry them — the cache keeps everything else")

    segmenter = config.segmenter
    events: Optional[List[Event]] = None
    used = observations      # the observations the events are built from

    # ---- Segment ------------------------------------------------------------
    t0 = time.time()
    if segmenter == "llm":
        from .coalesce import coalesce

        joins = join_positions(observations) if join_boundaries else []
        chapters = coalesce(
            observations, config.coalesce, config.segmentation.min_event_sec,
            duration_sec=duration, timeline=timeline, locked=joins, verbose=v,
        ) if config.coalesce.enabled else None
        if chapters is not None:
            for c in chapters:
                c["metadata"]["segmenter"] = "llm"
            events = events_from_starts(
                [c["start"] for c in chapters], observations, video_id,
                duration, info=chapters,
            )
            channels = ["vlm"] + (
                ["asr"] if any(o.transcript_text for o in observations) else [])
        else:
            print("[pipeline] coalescing failed; falling back to curve "
                  "segmentation over the observations already collected")
            segmenter = FALLBACK

    if events is None:
        from .segmentation import input_from_observations, segment, usable_observations

        use_dino = (config.segmentation.weight_dino > 0
                    and any(o.embedding is not None for o in observations))
        used = usable_observations(observations, use_dino)
        if not used:
            raise PipelineError("no decodable samples to segment")
        joins = join_positions(used) if join_boundaries else []
        seg = segment(
            input_from_observations(used, use_dino, joins),
            config.segmentation, duration_sec=duration, verbose=v,
            forced_bounds=joins,
        )
        info = [{"metadata": {"segmenter": "curve",
                              "boundary_score": seg.boundary_score(s)}}
                for s in seg.starts]
        events = events_from_starts(seg.starts, used, video_id, duration, info=info)
        channels = seg.channels
    _mark(stage_times, "segment", t0, v)

    # ---- Titles ---------------------------------------------------------------
    if segmenter == "llm":
        titles = "llm"
    elif config.labeling.enabled:
        from .labeling import label_events

        t0 = time.time()
        titles = label_events(
            events, timeline, config.labeling, backend=title_backend,
            verbose=v, ocr_config=config.ocr, redact_config=config.redact,
        )
        _mark(stage_times, "labeling", t0, v)
    else:
        titles = "none"

    # ---- Refine ---------------------------------------------------------------
    if config.refine.enabled and config.vlm.enabled and creds and len(events) > 1:
        from .refine import refine_boundaries

        t0 = time.time()
        # Joins are never refined, forced or not: the gap across one spans two
        # recordings, so its "midpoint" is not a moment in either.
        refine_boundaries(
            events, used, timeline, config,
            locked=set(join_positions(used)), verbose=v,
        )
        _mark(stage_times, "refine", t0, v)

    attach_sources(events, timeline)
    return PipelineResult(
        video_id=video_id, timeline=timeline, duration_sec=duration,
        observations=observations, events=events, segmenter=segmenter,
        titles=titles, channels=channels, stage_times=stage_times,
        runtime_sec=time.time() - t_start,
    )


def _mark(stage_times: Dict[str, float], name: str, t0: float, verbose: bool) -> None:
    stage_times[name] = time.time() - t0
    if verbose:
        print(f"  [{name}] {stage_times[name]:.1f}s")
