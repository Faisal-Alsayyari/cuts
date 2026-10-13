"""Text channel — per-frame screen descriptions from a Claude vision call.

A small vision model writes one short factual sentence per sampled frame,
grounded by exact strings from local OCR (`cuts.signals.ocr`). Those
descriptions are the evidence Pass 2 reads (`cuts.coalesce`), the text channel
of the curve segmenter (`cuts.segmentation`, via TF-IDF), and what is stored
per chapter in the index.

**Several frames go in one call** (`VLMConfig.frames_per_call`). That is
cheaper — the system prompt is sent once per call rather than once per frame —
and it lets the model see each frame's neighbours, which it uses to work out
*what it is looking at*. It must never refer to them in the output: every
description has to stand alone, because the curve segmenter compares
consecutive descriptions by TF-IDF cosine, and "same as the previous frame"
shares almost no tokens with a description that names the file, which reads
as a large change and manufactures a boundary that isn't there. Repeating a
neighbour's wording verbatim when nothing changed is the *correct* output.

Because the prompt tells the model its frames are consecutive, callers whose
frames are *not* neighbours (boundary refinement's midpoints) must send them
one per call.

A failed call yields `None` for its frames, never "", so "the model was not
reached" stays distinguishable from "the model said nothing" — the observation
cache records the former as `failed` and retries it on the next run.

`decide_refinement` is a second, distinct kind of call used by `cuts.refine`:
given descriptions of the frames on either side of a detected boundary plus
one new frame from the midpoint, it decides which half of the gap the true
transition falls in.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .. import llm
from ..config import VLMConfig

_DESCRIBE_SYSTEM = """\
You are scouting footage from a screen-recorded coding session to help index \
it for later navigation. You are given several consecutive frames from one \
recording, in chronological order, each numbered and each accompanied (when \
available) by OCR text extracted from that same frame — the OCR may be noisy, \
incomplete, or out of reading order.

For EACH numbered frame, write exactly ONE sentence, at most 25 words: the \
active application/window, the single most distinguishing specific detail (a \
file name, function or variable name, error message, terminal command, or \
page title), and what the person appears to be doing.

CRITICAL — each description must STAND ALONE. Use the neighbouring frames to \
work out what you are looking at (for instance, to recognize a partially \
visible or ambiguous window as the same editor that is clearly visible in an \
adjacent frame). But never refer to another frame in your output. Do not \
write "same as previous", "still", "continues", "unchanged", or "now". Always \
re-state the application, file, and activity explicitly — even when that means \
repeating the previous frame's description word for word. These descriptions \
are compared against each other automatically to detect when the activity \
changes; a description that points at another frame instead of describing its \
own breaks that comparison and invents a change that did not happen. When \
nothing changed between two frames, near-identical descriptions are exactly \
the right answer.

Treat the OCR text as the source of truth for exact names, strings, and \
error messages — do not guess small text from the image yourself. Use the \
image for what OCR cannot tell you: which application is active, general \
layout, and anything with no legible on-screen text (diagrams, previews, \
webcam overlays). Do not quote the OCR text verbatim; synthesize it into the \
sentence.

Prioritize the one detail that would distinguish a frame from a \
similar-looking one over generic phrasing ("a code editor is open"). Do not \
describe window chrome, OS taskbars, or webcam overlays unless they are the \
only content in the frame.

Return exactly one entry per frame, matching the frame numbers given."""

_DESCRIBE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "descriptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "frame": {
                        "type": "integer",
                        "description": "The frame number this describes.",
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "One standalone sentence, at most 25 words, "
                            "describing what is on screen in this frame."
                        ),
                    },
                },
                "required": ["frame", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["descriptions"],
    "additionalProperties": False,
}

_REFINE_SYSTEM = """\
You are scouting footage of a screen-recorded coding session to pin down \
exactly where an activity change happens. You are given three short frame \
descriptions from one sampling gap, in order: the last frame before a \
detected change (A), a frame from the midpoint of the gap (M), and the \
first frame after the change (B).

Decide whether the actual transition from A's activity to B's activity \
happens in the first half of the gap (between A and M) or the second half \
(between M and B). If the descriptions don't give you enough to tell \
confidently, answer "unclear" and the current boundary will be kept as is."""

_REFINE_SCHEMA = {
    "type": "object",
    "properties": {
        "transition": {
            "type": "string",
            "enum": ["first_half", "second_half", "unclear"],
        }
    },
    "required": ["transition"],
    "additionalProperties": False,
}


def encode_frame(img: np.ndarray, quality: int = 85) -> str:
    """Encode one BGR frame as a base64 JPEG string, ready to send.

    Encoding happens in the decode loop rather than on an API thread: an
    encoded frame is roughly 15x smaller than the numpy array it came from,
    so pending uploads stay small.

    The caller is responsible for resizing first (`VLMConfig.frame_long_edge`).
    """
    import cv2

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.standard_b64encode(buf.tobytes()).decode()


def _trim_words(text: str, max_words: int = 30) -> str:
    """Hard backstop on description length, at a word boundary.

    The prompt asks for one ~25-word sentence, but prompts are never fully
    reliable — this caps what actually gets stored/dedup'd/sent to labeling
    regardless of whether the model complied.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


# ---------------------------------------------------------------------------
# Per-frame description
# ---------------------------------------------------------------------------

def describe_frames(
    encoded: Sequence[str],
    config: VLMConfig,
    ocr_texts: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> List[Optional[str]]:
    """ONE Claude vision call describing several consecutive frames.

    `encoded` holds base64 JPEG strings from `encode_frame`, in chronological
    order. `ocr_texts`, when given, must be the same length — "" for a frame
    with no grounding text.

    Always returns exactly `len(encoded)` entries, in order, so a missing
    entry can never shift the rest onto the wrong moment. An entry is `None`
    when that frame got no description: the call failed outright, or the
    model omitted that frame or returned an empty sentence for it.
    """
    n = len(encoded)
    if n == 0:
        return []
    texts = list(ocr_texts) if ocr_texts is not None else [""] * n

    content: List[Dict[str, Any]] = [{
        "type": "text",
        "text": (
            f"Here are {n} consecutive frames from one recording, in order. "
            f"Describe each one."
        ),
    }]
    for i, (b64, ocr) in enumerate(zip(encoded, texts)):
        header = f"--- Frame {i} ---"
        if ocr:
            header += f"\nOCR text extracted from this frame:\n{ocr}"
        content.append({"type": "text", "text": header})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        })

    try:
        data = llm.structured_call(
            model=config.model,
            system=_DESCRIBE_SYSTEM,
            content=content,
            schema=_DESCRIBE_SCHEMA,
            # Derived from the batch size so a larger frames_per_call cannot
            # silently truncate the tail of the response; the slack covers
            # the JSON scaffolding around the descriptions themselves.
            max_tokens=config.max_tokens_per_frame * n + 200,
        )
    except llm.LLMError as exc:
        print(f"  [vlm] describe failed for {n} frame(s): {exc}")
        return [None] * n

    by_frame: Dict[int, str] = {}
    for entry in data.get("descriptions", []) or []:
        try:
            idx = int(entry["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        desc = _trim_words(str(entry.get("description", "")).strip())
        if 0 <= idx < n and desc:
            by_frame[idx] = desc
    if verbose and len(by_frame) != n:
        print(f"  [vlm] got {len(by_frame)}/{n} descriptions in one response")
    return [by_frame.get(i) for i in range(n)]


def describe_batch(
    encoded: Sequence[str],
    config: VLMConfig,
    ocr_texts: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> List[Optional[str]]:
    """Describe many frames, chunked into calls of `frames_per_call`.

    Chunks run concurrently; how many requests are actually in flight is
    bounded globally by `cuts.llm`, not here. Results come back flattened in
    input order, `None` where a frame got no description.
    """
    n = len(encoded)
    if n == 0:
        return []
    texts = list(ocr_texts) if ocr_texts is not None else [""] * n
    size = max(1, config.frames_per_call)
    chunks = [
        (encoded[i:i + size], texts[i:i + size]) for i in range(0, n, size)
    ]
    if len(chunks) == 1:
        return describe_frames(chunks[0][0], config, chunks[0][1], verbose)
    with ThreadPoolExecutor(max_workers=min(len(chunks), 16)) as pool:
        results = list(pool.map(
            lambda ch: describe_frames(ch[0], config, ch[1], verbose),
            chunks,
        ))
    return [d for chunk in results for d in chunk]


# ---------------------------------------------------------------------------
# Boundary-refinement decision
# ---------------------------------------------------------------------------

def decide_refinement(
    desc_before: str,
    desc_mid: str,
    desc_after: str,
    model: str,
    max_tokens: int,
    verbose: bool = False,
) -> str:
    """Classify which half of a sampling gap a transition falls in.

    Returns "first_half", "second_half", or "unclear". Anything that goes
    wrong is "unclear", which the caller treats as "keep the boundary".
    """
    prompt = (
        f"A: {desc_before or '(no evidence)'}\n"
        f"M: {desc_mid or '(no evidence)'}\n"
        f"B: {desc_after or '(no evidence)'}"
    )
    try:
        data = llm.structured_call(
            model=model, system=_REFINE_SYSTEM, content=prompt,
            schema=_REFINE_SCHEMA, max_tokens=max_tokens,
        )
    except llm.LLMError as exc:
        if verbose:
            print(f"  [refine] decision failed: {exc}")
        return "unclear"
    transition = data.get("transition")
    return transition if transition in ("first_half", "second_half") else "unclear"


if __name__ == "__main__":
    # Standalone debug: describe the first handful of sampled frames the same
    # way the pipeline does — OCR-grounded, batched, concurrent.
    import sys
    import time

    from ..config import CutsConfig
    from ..media import iter_sampled_frames, resize_long_edge
    from .ocr import ocr_image

    if not llm.has_credentials():
        print("no Anthropic credentials found")
        sys.exit(2)

    if len(sys.argv) < 2:
        print("usage: python -m cuts.signals.vlm <video_path> [interval_sec]")
        sys.exit(1)
    cfg = CutsConfig()
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    n_frames = 12

    t0 = time.time()
    stamps, encoded, ocr_texts = [], [], []
    for i, sf in enumerate(iter_sampled_frames(sys.argv[1], interval_sec=interval)):
        if i >= n_frames:
            break
        stamps.append(sf.time_sec)
        ocr_texts.append(ocr_image(sf.image, cfg.ocr, verbose=True))
        encoded.append(encode_frame(
            resize_long_edge(sf.image, cfg.vlm.frame_long_edge)
        ))
    t_local = time.time() - t0
    print(f"\n{len(encoded)} frames decoded + OCR'd in {t_local:.1f}s; "
          f"describing in chunks of {cfg.vlm.frames_per_call}...")

    t1 = time.time()
    descs = describe_batch(encoded, cfg.vlm, ocr_texts=ocr_texts, verbose=True)
    for t, ocr, d in zip(stamps, ocr_texts, descs):
        words = len(d.split()) if d else 0
        print(f"\n[t={t:6.1f}s] ({len(ocr)} chars ocr, {words} words)")
        print(f"    {d!r}")
    print(f"\ndescribe {time.time() - t1:.1f}s / total {time.time() - t0:.1f}s")
