"""Turn segmented events into named chapters.

Segmentation says *where* the boundaries are; this module says *what each span
is*. Given an event's evidence — VLM screen descriptions from its frames, any
transcript in its time window, and optionally a couple of representative
frames — it writes the short activity label a chapter list needs ("debugging
segfault").

Two backends:

  * ``claude``    — the Claude API. Best quality; needs credentials.
  * ``heuristic`` — dominant-term extraction over the VLM evidence text. No
    extra dependencies, no network, much cruder labels. It exists so the
    pipeline produces a complete result without credentials, and so
    segmentation can be evaluated on its own.

All events are labeled in a *single* Claude call by default. That is not just a
cost optimization: seeing the whole video at once is what lets the model keep
titles distinct and consistently phrased. Labeling events one at a time
reliably produces near-duplicate titles for adjacent events, because each call
has no idea what the others said.
"""

from __future__ import annotations

import base64
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .config import LabelingConfig, OCRConfig, RedactConfig
from .media import (
    Timeline,
    decode_frames_at,
    iter_frames_at_times,
    resize_long_edge,
)
from .schema import Event, format_timestamp
from .signals.text_features import tokenize


# The response schema. Structured outputs guarantee we get parseable JSON back
# rather than a prose answer we have to scrape a list out of.
_CHAPTER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "The event_id this title belongs to.",
                    },
                    "title": {
                        "type": "string",
                        "description": (
                            "Short lowercase activity label, 2-5 words, "
                            "e.g. 'debugging segfault'."
                        ),
                    },
                },
                "required": ["event_id", "title"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["chapters"],
    "additionalProperties": False,
}


_SYSTEM = """\
You write chapter titles for long screen recordings, usually programming \
sessions. You are given consecutive segments of one recording; for each, you \
see short descriptions of what was on screen at several points during the \
segment (written by a vision model watching the recording), any spoken \
transcript, and sometimes a frame.

Write one title per segment describing WHAT THE PERSON WAS DOING, not what was \
on screen. "debugging segfault" and "reading docs" are good; "terminal window" \
and "code editor" are not — they describe the UI, which is the same all video.

Rules:
- 2-5 words, lowercase, no trailing punctuation.
- Prefer concrete specifics from the evidence ("writing parser", "fixing \
null deref") over generic ones ("coding", "working").
- You see every segment at once: keep titles distinct from each other. If two \
segments really are the same activity, differentiate by what changed.
- The descriptions are a vision model's summary, not a literal transcript —
weigh them for what they say about the activity, not as exact quotes.
- If a segment's evidence is genuinely too thin to tell, use "unclear".
- Return exactly one entry per segment, matching event_id."""


# ---------------------------------------------------------------------------
# Evidence assembly
# ---------------------------------------------------------------------------

def _pick_representative(event: Event, n: int) -> List[int]:
    """Choose up to `n` sample frame indices spread across the event.

    Endpoints are avoided: the first and last samples of an event sit next to
    a boundary, where the screen is most likely mid-transition.
    """
    samples = event.sample_indices
    if not samples or n <= 0:
        return []
    if len(samples) <= n:
        return list(samples)
    # Evenly spaced interior positions.
    step = len(samples) / (n + 1)
    return [samples[min(len(samples) - 1, int(step * (k + 1)))] for k in range(n)]


def _pick_times(event: Event, n: int) -> List[float]:
    """`_pick_representative` on timestamps: the `llm` path's frame identifier.

    `cuts.coalesce` already chose representatives when it built the event, so
    the usual case here is just honouring them; this re-spreads only when the
    caller wants a different count than was stored.
    """
    times = event.representative_times or event.sample_times
    if not times or n <= 0:
        return []
    if len(times) <= n:
        return list(times)
    step = len(times) / (n + 1)
    return [times[min(len(times) - 1, int(step * (k + 1)))] for k in range(n)]


def _event_evidence(
    event: Event, config: LabelingConfig, show_source: bool = False
) -> str:
    """Render one event's text evidence into the prompt."""
    header = (
        f"[{event.event_id}] {format_timestamp(event.start_time)}"
        f"-{format_timestamp(event.end_time)} ({event.duration_sec:.0f}s)"
    )
    if show_source and event.source_name:
        header += f"  session: {event.source_name} @ {event.source_timestamp()}"
    parts = [header]
    vlm = (event.vlm_text or "").strip()
    if vlm:
        parts.append(f"on-screen descriptions:\n{vlm[: config.max_vlm_chars]}")
    tr = (event.transcript_text or "").strip()
    if tr:
        parts.append(f"transcript:\n{tr[: config.max_transcript_chars]}")
    if not vlm and not tr:
        parts.append("(no text evidence)")
    return "\n".join(parts)


def _decode_representatives(
    timeline: Timeline,
    events: Sequence[Event],
    config: LabelingConfig,
    verbose: bool = False,
) -> Dict[str, List[Any]]:
    """Decode each event's representative frames, keyed by event_id.

    Events carry their representatives either as timestamps or as frame
    indices, and which one depends on how they were produced:

      * `representative_times` — the `llm` path. Frames were reached by
        seeking, so they never had an ordinal. Decoded the same way, by
        seeking, which also means only the wanted frames are touched.
      * `representative_frames` — the `curve` path, which decodes sequentially
        and therefore knows each frame's index. Decoded in one pass per file,
        since the naive alternative is O(N x length).

    Times win when an event has both. Frames are grouped by the file that owns
    them either way, so each source is opened once rather than once per frame.
    """
    by_time: Dict[str, List[Tuple[str, float]]] = {}
    by_index: Dict[str, List[Tuple[str, int]]] = {}
    times_per_source: Dict[str, List[float]] = {}
    idx_per_source: Dict[str, List[int]] = {}

    for e in events:
        if e.representative_times:
            picks: List[Tuple[str, float]] = []
            for g in _pick_times(e, config.frames_per_event):
                src, local = timeline.local_time(g)
                picks.append((src.path, local))
                times_per_source.setdefault(src.path, []).append(local)
            by_time[e.event_id] = picks
        elif e.representative_frames or e.sample_indices:
            picks_i: List[Tuple[str, int]] = []
            for g in _pick_representative(e, config.frames_per_event):
                found = timeline.local_frame(g)
                # Single-source indexes written before frame_span was recorded,
                # and any index outside every known span, fall back to file 0.
                src, local = found if found else (timeline.sources[0], g)
                picks_i.append((src.path, local))
                idx_per_source.setdefault(src.path, []).append(local)
            by_index[e.event_id] = picks_i

    n_wanted = (sum(len(v) for v in by_time.values())
                + sum(len(v) for v in by_index.values()))
    if not n_wanted:
        return {}
    if verbose:
        n_src = len(set(times_per_source) | set(idx_per_source))
        print(f"  decoding {n_wanted} frames for labeling "
              f"across {n_src} file(s)...")

    decoded_t: Dict[str, Dict[float, Any]] = {
        path: {t: (sf.image if sf else None)
               for t, sf in iter_frames_at_times(path, ts)}
        for path, ts in times_per_source.items()
    }
    decoded_i: Dict[str, Dict[int, Any]] = {
        path: decode_frames_at(path, idxs)
        for path, idxs in idx_per_source.items()
    }

    out: Dict[str, List[Any]] = {}
    for eid, picks in by_time.items():
        out[eid] = [decoded_t.get(p, {}).get(t) for p, t in picks]
    for eid, picks_i in by_index.items():
        out[eid] = [decoded_i.get(p, {}).get(i) for p, i in picks_i]
    return out


def _encode_frames(
    timeline: Timeline,
    events: Sequence[Event],
    config: LabelingConfig,
    verbose: bool = False,
    ocr_config: Optional[OCRConfig] = None,
    redact_config: Optional[RedactConfig] = None,
) -> Dict[str, List[str]]:
    """Decode and base64-JPEG-encode representative frames, keyed by event_id.

    Decoding is delegated to `_decode_representatives`, which resolves an
    event's representatives by timestamp or by frame index depending on how it
    was produced.

    These frames are decoded fresh here rather than kept from the sampling
    pass, so they have never been through `cuts.redact` — this is an upload
    path in its own right, and it re-runs the scrub. Doing it here rather than
    reusing a flag from the pipeline is what keeps a standalone
    `python -m cuts.labeling events.json` run safe too.
    """
    if config.frames_per_event <= 0:
        return {}
    import cv2

    scrub = None
    if redact_config is not None and redact_config.enabled:
        if ocr_config is None or not ocr_config.enabled:
            print("[labeling] WARNING: OCR is disabled, so credential "
                  "redaction cannot run; frames will be uploaded unredacted")
        else:
            from .redact import scrub_frame as scrub

    wanted = _decode_representatives(timeline, events, config, verbose=verbose)
    if not wanted:
        return {}

    out: Dict[str, List[str]] = {}
    hits_seen: List[str] = []
    withheld = 0
    for eid, imgs in wanted.items():
        encoded: List[str] = []
        for img in imgs:
            if img is None:
                continue
            if scrub is not None:
                # Mutates img in place, so the redaction survives the resize
                # and JPEG encode below.
                _text, safe, hits = scrub(
                    img, ocr_config, redact_config, verbose=verbose
                )
                hits_seen.extend(hits)
                if not safe:
                    # Drop the frame. An event left with none falls through to
                    # text-only labeling, which is already a supported mode.
                    withheld += 1
                    continue
            img = resize_long_edge(img, config.frame_long_edge)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                encoded.append(base64.standard_b64encode(buf.tobytes()).decode())
        if encoded:
            out[eid] = encoded
    if hits_seen or withheld:
        kinds = ", ".join(sorted(set(hits_seen))) or "unlocalized"
        print(f"[redact] labeling: blacked out credentials ({kinds})"
              + (f"; withheld {withheld} frame(s)" if withheld else ""))
    return out


def _build_content(
    events: Sequence[Event],
    frames: Dict[str, List[str]],
    config: LabelingConfig,
    n_sources: int = 1,
) -> List[Dict[str, Any]]:
    """Build the user-turn content blocks: evidence text interleaved with frames."""
    preamble = (
        f"This recording has {len(events)} segments, in order. "
        f"Write a title for each."
    )
    if n_sources > 1:
        # Without this the model tends to force narrative continuity across a
        # join — "continuing the refactor" — when the two sides are unrelated
        # work done days apart.
        preamble += (
            f" The recording is split across {n_sources} separate capture "
            f"sessions, marked below; work does not necessarily continue "
            f"across a session change."
        )
    blocks: List[Dict[str, Any]] = [{"type": "text", "text": preamble}]
    for e in events:
        blocks.append({
            "type": "text",
            "text": _event_evidence(e, config, show_source=n_sources > 1),
        })
        for b64 in frames.get(e.event_id, []):
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            })
    return blocks


# ---------------------------------------------------------------------------
# Claude backend
# ---------------------------------------------------------------------------

def _call_claude(
    content: List[Dict[str, Any]],
    config: LabelingConfig,
    api_key: Optional[str] = None,
) -> List[Dict[str, str]]:
    """One structured-output call returning [{event_id, title}, ...].

    Raises RuntimeError on refusal or truncation so the caller can decide
    whether to degrade to heuristic labels rather than silently emitting
    partial results.
    """
    import anthropic

    # A bare constructor also resolves an `ant auth login` profile, so do not
    # require an explicit key.
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    request: Dict[str, Any] = dict(
        model=config.model,
        max_tokens=config.max_tokens,
        system=_SYSTEM,
        messages=[{"role": "user", "content": content}],
        output_config={
            "effort": config.effort,
            "format": {"type": "json_schema", "schema": _CHAPTER_SCHEMA},
        },
    )

    if config.use_fallbacks:
        try:
            response = client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                **request,
            )
        except anthropic.BadRequestError:
            # The fallbacks beta is not available to this account/SDK version;
            # the labeling call itself is still perfectly valid without it.
            response = client.messages.create(**request)
    else:
        response = client.messages.create(**request)

    if response.stop_reason == "refusal":
        detail = getattr(response, "stop_details", None)
        category = getattr(detail, "category", None) if detail else None
        raise RuntimeError(f"labeling refused by safety classifiers ({category})")
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            "labeling response hit max_tokens; raise LabelingConfig.max_tokens"
        )

    text = next((b.text for b in response.content if b.type == "text"), "")
    if not text:
        raise RuntimeError("labeling response contained no text block")
    return json.loads(text).get("chapters", [])


# ---------------------------------------------------------------------------
# Heuristic backend
# ---------------------------------------------------------------------------

# Tokens that appear in nearly every frame of a screen recording and say
# nothing about the activity. Kept deliberately small — the real filtering is
# the cross-event document-frequency check below.
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "you", "not", "are",
    "was", "have", "has", "but", "all", "can", "will", "def", "self", "return",
    "import", "true", "false", "none", "null", "file", "edit", "view", "help",
}


def _heuristic_labels(events: Sequence[Event]) -> List[Dict[str, str]]:
    """Label each event by its most distinctive terms in the evidence text.

    Distinctive means high frequency *within* the event and low frequency
    across the rest of the video — the same intuition as the TF-IDF weighting
    used for segmentation. Words that recur in every event's description
    (persistent UI chrome, when the evidence came from OCR; generic framing
    language, when it came from a VLM) are therefore never chosen.
    """
    per_event: List[Counter] = []
    doc_freq: Counter = Counter()
    for e in events:
        toks = [
            t for t in tokenize(e.combined_text)
            if len(t) > 2 and t not in _STOPWORDS and not t.isdigit()
        ]
        counts = Counter(toks)
        per_event.append(counts)
        doc_freq.update(set(counts))

    n = max(1, len(events))
    out: List[Dict[str, str]] = []
    for e, counts in zip(events, per_event):
        if not counts:
            out.append({"event_id": e.event_id, "title": "unclear"})
            continue
        # Score = in-event frequency, damped by how many events also show it.
        scored = sorted(
            counts.items(),
            key=lambda kv: kv[1] * (n / (1 + doc_freq[kv[0]])),
            reverse=True,
        )
        out.append({
            "event_id": e.event_id,
            "title": " ".join(t for t, _ in scored[:3]),
        })
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def label_events(
    events: List[Event],
    source: Union[str, Timeline],
    config: LabelingConfig,
    backend: str = "claude",
    api_key: Optional[str] = None,
    verbose: bool = False,
    ocr_config: Optional[OCRConfig] = None,
    redact_config: Optional[RedactConfig] = None,
) -> str:
    """Set `title` on each event in place. Returns the backend actually used.

    `source` is the `Timeline` the events were segmented from, or a single
    video path (which is wrapped into a one-file timeline).

    `ocr_config`/`redact_config` drive credential redaction on the frames sent
    with the labeling call (see `cuts.redact`). They default to the standard
    config rather than to "off", so a caller that forgets to pass them — the
    standalone `__main__` path, say — still gets redaction.

    `backend` is "claude", "heuristic", or "auto". Under "auto" — and under
    "claude" when the call fails — labeling degrades to heuristic rather than
    aborting: a chapter list with crude titles is far more useful than no
    output at all after a full segmentation pass.
    """
    if not events or not config.enabled:
        return "none"

    timeline = Timeline.build([source]) if isinstance(source, str) else source

    have_creds = bool(api_key or os.environ.get("ANTHROPIC_API_KEY"))
    if backend == "auto":
        backend = "claude" if have_creds else "heuristic"

    # Fail safe, not open: a caller that passes neither gets redaction on with
    # stock settings rather than silently uploading raw frames.
    if ocr_config is None:
        ocr_config = OCRConfig()
    if redact_config is None:
        redact_config = RedactConfig()

    if backend == "claude":
        try:
            frames = _encode_frames(
                timeline, events, config, verbose=verbose,
                ocr_config=ocr_config, redact_config=redact_config,
            )
            content = _build_content(events, frames, config,
                                     n_sources=len(timeline))
            if verbose:
                n_img = sum(len(v) for v in frames.values())
                print(f"  labeling {len(events)} events via {config.model} "
                      f"({n_img} frames)...")
            chapters = _call_claude(content, config, api_key=api_key)
            used = "claude"
        except Exception as exc:
            print(f"[labeling] Claude backend failed ({exc!r}); "
                  f"falling back to heuristic labels")
            chapters = _heuristic_labels(events)
            used = "heuristic"
    else:
        chapters = _heuristic_labels(events)
        used = "heuristic"

    by_id = {c.get("event_id"): c.get("title", "") for c in chapters}
    for e in events:
        e.title = (by_id.get(e.event_id) or "").strip()
    return used


if __name__ == "__main__":
    # Standalone debug: label an existing events.json in place.
    import sys

    from .config import CutsConfig
    from .schema import read_events, to_chapter_list, write_events

    if len(sys.argv) < 2:
        print("usage: python -m cuts.labeling <events.json> [claude|heuristic|auto]")
        sys.exit(1)
    path = sys.argv[1]
    backend = sys.argv[2] if len(sys.argv) > 2 else "auto"

    cfg = CutsConfig()
    payload = read_events(path)
    evs = payload["events"]
    tl = Timeline.from_dicts(payload["sources"])
    used = label_events(evs, tl, cfg.labeling, backend=backend, verbose=True)
    write_events(path, payload["video_id"], payload["video_path"], evs,
                 duration_sec=payload.get("duration_sec", 0.0),
                 sources=payload["sources"])
    print(f"\nbackend: {used}\n")
    print(to_chapter_list(evs))
