"""Turn segmented events into named chapters.

Segmentation says *where* the boundaries are; this module says *what each span
is*. Given an event's evidence — OCR text from its frames, any transcript in
its time window, and optionally a couple of representative frames — it writes
the short activity label a chapter list needs ("debugging segfault").

Two backends:

  * ``claude``    — the Claude API. Best quality; needs credentials.
  * ``heuristic`` — dominant-OCR-term extraction. No dependencies, no network,
    much cruder labels. It exists so the pipeline produces a complete result
    without credentials, and so segmentation can be evaluated on its own.

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
from typing import Any, Dict, List, Optional, Sequence

from .config import LabelingConfig
from .media import decode_frames_at, resize_long_edge
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
see the text that was on screen (via OCR, so expect noise and broken words), \
any spoken transcript, and sometimes a frame.

Write one title per segment describing WHAT THE PERSON WAS DOING, not what was \
on screen. "debugging segfault" and "reading docs" are good; "terminal window" \
and "code editor" are not — they describe the UI, which is the same all video.

Rules:
- 2-5 words, lowercase, no trailing punctuation.
- Prefer concrete specifics from the evidence ("writing parser", "fixing \
null deref") over generic ones ("coding", "working").
- You see every segment at once: keep titles distinct from each other. If two \
segments really are the same activity, differentiate by what changed.
- OCR noise is expected. Infer the activity; do not quote garbled text.
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


def _event_evidence(event: Event, config: LabelingConfig) -> str:
    """Render one event's text evidence into the prompt."""
    parts = [
        f"[{event.event_id}] {format_timestamp(event.start_time)}"
        f"-{format_timestamp(event.end_time)} ({event.duration_sec:.0f}s)"
    ]
    ocr = (event.ocr_text or "").strip()
    if ocr:
        parts.append(f"on-screen text:\n{ocr[: config.max_ocr_chars]}")
    tr = (event.transcript_text or "").strip()
    if tr:
        parts.append(f"transcript:\n{tr[: config.max_transcript_chars]}")
    if not ocr and not tr:
        parts.append("(no text evidence)")
    return "\n".join(parts)


def _encode_frames(
    video_path: str,
    events: Sequence[Event],
    config: LabelingConfig,
    verbose: bool = False,
) -> Dict[str, List[str]]:
    """Decode and base64-JPEG-encode representative frames, keyed by event_id.

    All frames across all events come from one decode pass.
    """
    if config.frames_per_event <= 0:
        return {}
    import cv2

    wanted: Dict[str, List[int]] = {
        e.event_id: _pick_representative(e, config.frames_per_event) for e in events
    }
    all_idx = [i for idxs in wanted.values() for i in idxs]
    if not all_idx:
        return {}

    if verbose:
        print(f"  decoding {len(all_idx)} frames for labeling...")
    decoded = decode_frames_at(video_path, all_idx)

    out: Dict[str, List[str]] = {}
    for eid, idxs in wanted.items():
        encoded: List[str] = []
        for i in idxs:
            img = decoded.get(i)
            if img is None:
                continue
            img = resize_long_edge(img, config.frame_long_edge)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                encoded.append(base64.standard_b64encode(buf.tobytes()).decode())
        if encoded:
            out[eid] = encoded
    return out


def _build_content(
    events: Sequence[Event],
    frames: Dict[str, List[str]],
    config: LabelingConfig,
) -> List[Dict[str, Any]]:
    """Build the user-turn content blocks: evidence text interleaved with frames."""
    blocks: List[Dict[str, Any]] = [{
        "type": "text",
        "text": (
            f"This recording has {len(events)} segments, in order. "
            f"Write a title for each."
        ),
    }]
    for e in events:
        blocks.append({"type": "text", "text": _event_evidence(e, config)})
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
    """Label each event by its most distinctive OCR terms.

    Distinctive means high frequency *within* the event and low frequency
    across the rest of the video — the same intuition as the TF-IDF weighting
    used for segmentation. Persistent UI chrome appears everywhere and is
    therefore never chosen.
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
    video_path: str,
    config: LabelingConfig,
    backend: str = "claude",
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """Set `title` on each event in place. Returns the backend actually used.

    `backend` is "claude", "heuristic", or "auto". Under "auto" — and under
    "claude" when the call fails — labeling degrades to heuristic rather than
    aborting: a chapter list with crude titles is far more useful than no
    output at all after a full segmentation pass.
    """
    if not events or not config.enabled:
        return "none"

    have_creds = bool(api_key or os.environ.get("ANTHROPIC_API_KEY"))
    if backend == "auto":
        backend = "claude" if have_creds else "heuristic"

    if backend == "claude":
        try:
            frames = _encode_frames(video_path, events, config, verbose=verbose)
            content = _build_content(events, frames, config)
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
    used = label_events(evs, payload["video_path"], cfg.labeling,
                        backend=backend, verbose=True)
    write_events(path, payload["video_id"], payload["video_path"], evs,
                 duration_sec=payload.get("duration_sec", 0.0))
    print(f"\nbackend: {used}\n")
    print(to_chapter_list(evs))
