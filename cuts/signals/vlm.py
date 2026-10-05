"""Text channel — per-frame screen descriptions from a Claude vision call.

Replaces local OCR as the text channel. Instead of transcribing every visible
string (noisy, and a wall of text that is not actually the deliverable),
a small vision model writes a short factual description of what's on screen.
That description does double duty: fed through the same TF-IDF machinery as
the old OCR text, it drives the fused similarity curve in `cuts.segmentation`
exactly as before, and it is also what gets stored as event evidence — so the
index holds a description, not a transcript.

There is no "is this frame different enough to bother describing" decision
step: that call would cost the same API round-trip as just describing the
frame, so the segmentation math (which already exists for this) decides
sameness from the descriptions rather than an LLM deciding it up front.

**Several frames go in one call** (`VLMConfig.frames_per_call`). That is
cheaper — the system prompt is sent once per call rather than once per frame —
and it lets the model see each frame's neighbours, which it uses to work out
*what it is looking at*. It must never refer to them in the output: every
description has to stand alone, because `cuts.segmentation` compares
consecutive descriptions by TF-IDF cosine, and "same as the previous frame"
shares almost no tokens with a description that names the file, which reads
as a large change and manufactures a boundary that isn't there. Repeating a
neighbour's wording verbatim when nothing changed is the *correct* output:
cosine ≈ 1.0, meaning no change.

Description calls are network-latency-bound, not compute-bound, so
`describe_batch` fires the chunks concurrently rather than one at a time, and
`cuts.pipeline` submits them without waiting so decoding overlaps the network.

`decide_refinement` is a second, distinct kind of call used by `cuts.refine`:
given descriptions of the frames on either side of a detected boundary plus
one new frame from the midpoint, it decides which half of the gap the true
transition falls in.
"""

from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import RefineConfig, VLMConfig

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


def _client(api_key: Optional[str] = None):
    import anthropic

    # A bare constructor also resolves an `ant auth login` profile, so do not
    # require an explicit key.
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def encode_frame(img: np.ndarray, quality: int = 85) -> str:
    """Encode one BGR frame as a base64 JPEG string, ready to send.

    Public because the pipeline calls it in its own decode loop rather than
    letting the API threads do it: an encoded frame is roughly 15x smaller
    than the numpy array it came from, and with calls submitted without
    waiting, whatever is in flight is held in memory until its call returns.

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
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> List[str]:
    """ONE Claude vision call describing several consecutive frames.

    `encoded` holds base64 JPEG strings from `encode_frame`, in chronological
    order. `ocr_texts`, when given, must be the same length — "" for a frame
    with no grounding text.

    Always returns exactly `len(encoded)` descriptions, in order. A frame the
    model omits (or a call that fails outright) yields "" for that frame
    rather than raising or shifting the rest: an empty description is a zero
    TF-IDF row that contributes no opinion to the similarity curve, which is
    a far better failure mode than a misaligned sequence, where every
    subsequent description would be attributed to the wrong moment.
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
        client = _client(api_key)
        response = client.messages.create(
            model=config.model,
            # Derived from the batch size so a larger frames_per_call cannot
            # silently truncate the tail of the response; the slack covers
            # the JSON scaffolding around the descriptions themselves.
            max_tokens=config.max_tokens_per_frame * n + 200,
            system=_DESCRIBE_SYSTEM,
            messages=[{"role": "user", "content": content}],
            output_config={
                "format": {"type": "json_schema", "schema": _DESCRIBE_SCHEMA},
            },
        )
    except Exception as exc:
        if verbose:
            print(f"  [vlm] describe failed for {n} frame(s): {exc!r}")
        return [""] * n

    text = next((b.text for b in response.content if b.type == "text"), "")
    if not text:
        return [""] * n
    try:
        entries = json.loads(text).get("descriptions", [])
    except json.JSONDecodeError:
        if verbose:
            print(f"  [vlm] unparseable description response for {n} frame(s)")
        return [""] * n

    by_frame: Dict[int, str] = {}
    for entry in entries:
        try:
            idx = int(entry["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < n:
            by_frame[idx] = _trim_words(str(entry.get("description", "")).strip())
    if verbose and len(by_frame) != n:
        print(f"  [vlm] got {len(by_frame)}/{n} descriptions in one response")
    return [by_frame.get(i, "") for i in range(n)]


def describe_image(
    img: np.ndarray,
    config: VLMConfig,
    ocr_text: str = "",
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """Describe a single frame. Thin wrapper over `describe_frames`.

    Used for one-off frames outside the main sampling pass — chiefly the
    midpoint frame in `cuts.refine`.
    """
    out = describe_frames(
        [encode_frame(img)], config, ocr_texts=[ocr_text],
        api_key=api_key, verbose=verbose,
    )
    return out[0] if out else ""


def describe_batch(
    encoded: Sequence[str],
    config: VLMConfig,
    ocr_texts: Optional[Sequence[str]] = None,
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> List[str]:
    """Describe many frames, chunked into calls and run concurrently.

    Splits `encoded` into groups of `config.frames_per_call`, sends each group
    as one call, and runs the calls concurrently — description calls are
    network-latency-bound, so serial round trips would dominate wall time.
    Results are flattened back into input order.

    `cuts.pipeline` does not use this on its main path (it submits chunks
    itself so decoding can proceed without waiting); this is for callers that
    have all their frames up front, such as `cuts.refine`.
    """
    n = len(encoded)
    if n == 0:
        return []
    texts = list(ocr_texts) if ocr_texts is not None else [""] * n
    size = max(1, config.frames_per_call)
    chunks = [
        (encoded[i:i + size], texts[i:i + size]) for i in range(0, n, size)
    ]
    with ThreadPoolExecutor(max_workers=max(1, config.concurrency)) as pool:
        results = list(pool.map(
            lambda ch: describe_frames(
                ch[0], config, ocr_texts=ch[1],
                api_key=api_key, verbose=verbose,
            ),
            chunks,
        ))
    return [d for chunk in results for d in chunk]


# ---------------------------------------------------------------------------
# Stride handling / carry-forward (unchanged semantics from local OCR)
# ---------------------------------------------------------------------------

def carry_forward(
    texts: List[Optional[str]],
    reset_at: Optional[Sequence[int]] = None,
) -> List[str]:
    """Fill skipped positions with the most recent description.

    With `stride > 1` only some samples are described. Screen content
    persists between reads, so the correct fill is the last observed
    description rather than an empty string, which would read as "everything
    on screen vanished" and manufacture a boundary at every skipped frame.

    `reset_at` lists positions where that assumption breaks and the carry
    must be dropped. The pipeline passes the source-file joins: screen
    content does not persist across a gap of hours between two recordings.
    """
    resets = set(reset_at or ())
    out: List[str] = []
    last = ""
    for i, t in enumerate(texts):
        if i in resets:
            last = ""
        if t is not None:
            last = t
        out.append(last)
    return out


# ---------------------------------------------------------------------------
# Boundary-refinement decision
# ---------------------------------------------------------------------------

def decide_refinement(
    desc_before: str,
    desc_mid: str,
    desc_after: str,
    config: RefineConfig,
    vlm_model: str,
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """Classify which half of a sampling gap a transition falls in.

    Returns "first_half", "second_half", or "unclear" (treated by the caller
    the same as "second_half": keep the original boundary rather than move
    it on a low-confidence guess).
    """
    prompt = (
        f"A: {desc_before or '(no evidence)'}\n"
        f"M: {desc_mid or '(no evidence)'}\n"
        f"B: {desc_after or '(no evidence)'}"
    )
    try:
        client = _client(api_key)
        response = client.messages.create(
            model=config.model or vlm_model,
            max_tokens=config.max_tokens,
            system=_REFINE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "format": {"type": "json_schema", "schema": _REFINE_SCHEMA},
            },
        )
    except Exception as exc:
        if verbose:
            print(f"  [refine] decision failed: {exc!r}")
        return "unclear"

    text = next((b.text for b in response.content if b.type == "text"), "")
    if not text:
        return "unclear"
    try:
        return json.loads(text).get("transition", "unclear")
    except json.JSONDecodeError:
        return "unclear"


if __name__ == "__main__":
    # Standalone debug: describe the first handful of sampled frames the same
    # way the pipeline does — OCR-grounded, batched, concurrent.
    import sys
    import time

    from ..config import CutsConfig
    from ..media import iter_sampled_frames, resize_long_edge
    from .ocr import ocr_image

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
        print(f"\n[t={t:6.1f}s] ({len(ocr)} chars ocr, {len(d.split())} words)")
        print(f"    {d!r}")
    print(f"\ndescribe {time.time() - t1:.1f}s / total {time.time() - t0:.1f}s")
