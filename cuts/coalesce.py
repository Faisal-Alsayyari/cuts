"""Pass 2 — chapters decided by a model reading every observation.

Pass 1 (`cuts.observe`) produces one description per sampled frame and no
opinion about where anything begins or ends. This module reads all of them at
once and decides which runs of observations are one thing.

**Why this replaces the similarity curve.** `cuts.segmentation` finds boundary
candidates at local minima of a fused DINOv2 + TF-IDF similarity curve, then
deletes the weakest until a *count budget* is met. Three consequences follow
from that being a budget rather than a judgement: chapters come out roughly
uniform because the budget forces them to, twenty minutes of one uninteresting
detour gets chopped into pieces because the budget has pieces to spend, and
only *adjacent* partitions can ever merge. None of those are tuning problems;
they are what optimizing a count instead of understanding the content means.
A model reading the descriptions can collapse a long dead stretch into a
single chapter and split a pivotal minute into several, because it is deciding
what the footage is about.

**The model returns cut points, not spans.** It emits observation indices where
a new chapter begins; chapter *k* runs from cut *k* to cut *k+1*. This is
deliberate and does most of the reliability work:

  * A gap is not expressible. Contiguity and complete coverage are structural
    rather than something to validate and repair after the fact.
  * Overlaps are not expressible either.
  * Timestamps are never invented. Every boundary lands on a real sample time,
    because the model chooses among samples rather than typing seconds.

What is still checked (`_validate_and_repair`): ordering, range, the mandatory
cut at zero, minimum duration, and the file joins that must stay boundaries no
matter what the model thought.

**What this costs relative to the curve.** Boundaries stop being
deterministic, and chapter count and minimum length stop being guaranteed by
construction — they are enforced afterwards instead. `--segmenter curve` stays
available as the deterministic baseline to compare against, and is also where
the pipeline falls back when this call fails.

Titles come from this same call. Under the curve path titling is a second,
separate pass (`cuts.labeling`) that has to re-derive what each span was
about; here the model that drew a boundary is the one that names it, which is
both cheaper and better grounded.

This module only decides *where chapters begin* and what they are called;
`cuts.chapters` turns that into events, exactly as it does for the curve.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from . import llm
from .chapters import enforce_min_duration
from .config import CoalesceConfig
from .media import Timeline
from .observe import FAILED, UNDECODABLE, WITHHELD, Observation
from .schema import format_timestamp

# `kind` is not used for chapter generation today. It is recorded because the
# eventual footage-cutting pass needs a priority signal, and the model that
# read the evidence is the cheapest place to get one.
_KINDS = ["pivotal", "active", "routine", "idle"]

_SYSTEM = """\
You are indexing a long screen recording, usually a programming session, by \
deciding where its chapters begin.

You are given numbered observations in chronological order. Each is one \
sampled frame: a timestamp and a one-sentence description of what was on \
screen at that instant. Consecutive observations are usually a fixed interval \
apart, so a long run of similar descriptions means a long stretch of the same \
activity.

Return the observation indices where a NEW CHAPTER BEGINS. Chapter k runs from \
your kth index up to (not including) your k+1th, so the chapters tile the \
whole recording with no gaps and no overlaps. The first index MUST be 0.

How to decide where a chapter begins:
- A chapter is a stretch of ONE activity. Start a new one when what the person \
is DOING changes — not when the window, file, or visual layout changes. \
Scrolling through a file, switching between an editor and a terminal that are \
part of the same task, or the same error reappearing are all the SAME chapter.
- Chapters do NOT have to be similar lengths, and usually should not be. A \
twenty-minute stretch of the same fruitless debugging is ONE chapter, however \
many observations it spans. A dense few minutes where something is actually \
figured out may deserve several.
- Do not manufacture boundaries to make chapters evenly sized, and do not \
merge genuinely different activities just because they are adjacent and short.
- Prefer fewer, more meaningful chapters over many shallow ones. Someone \
scanning this list wants to find a moment, not read a transcript.

For each chapter also give:
- title: 2-5 words, lowercase, no trailing punctuation, describing WHAT THE \
PERSON WAS DOING ("debugging a segfault", "reading vercel docs"), never what \
the UI looked like ("code editor open", "terminal window"). Keep every title \
distinct from the others; you can see them all at once. Use "unclear" only \
when the evidence genuinely will not support a guess.
- kind: "pivotal" when something is decided, fixed, or discovered; "active" \
for ordinary productive work; "routine" for repetitive or mechanical work \
including long unproductive debugging; "idle" when little or nothing is \
happening.
- why: one short phrase citing the evidence that set this boundary. This is \
read by a human checking your judgement, so be specific ("obs 24-61 are all \
the same failing deploy log") rather than restating the title."""

_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "integer",
                        "description": (
                            "Index of the first observation in this chapter. "
                            "The first chapter's value must be 0."
                        ),
                    },
                    "title": {
                        "type": "string",
                        "description": (
                            "2-5 words, lowercase, describing the activity."
                        ),
                    },
                    "kind": {"type": "string", "enum": _KINDS},
                    "why": {
                        "type": "string",
                        "description": (
                            "Short evidence-citing justification for this "
                            "boundary."
                        ),
                    },
                },
                "required": ["start", "title", "kind", "why"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["chapters"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

# How an observation with no description is shown to the model. These are
# different facts and it should be able to tell them apart: a frame we chose
# not to upload is not a frame that could not be read.
_MISSING = {
    WITHHELD: "(frame withheld: a credential was on screen)",
    UNDECODABLE: "(no frame at this time)",
    FAILED: "(description unavailable)",
}


def render_observations(
    observations: Sequence[Observation],
    config: CoalesceConfig,
    timeline: Optional[Timeline] = None,
) -> str:
    """Render observations as the numbered list the model indexes into.

    Compact on purpose: this is the bulk of the prompt, and every character
    here is multiplied by the sample count. One line per observation, with
    transcript and OCR only when they exist and are wanted.
    """
    multi = timeline is not None and timeline.is_multi
    lines: List[str] = []
    last_source = -1
    for i, o in enumerate(observations):
        if multi and o.source_index != last_source:
            last_source = o.source_index
            name = timeline.sources[o.source_index].name
            lines.append(f"--- new capture session: {name} ---")
        text = o.description or _MISSING.get(o.status, "(no description)")
        lines.append(f"[{i}] {format_timestamp(o.t)}  {text}")
        if o.transcript_text:
            lines.append(f"      said: {o.transcript_text[:300]}")
        if config.include_ocr and o.ocr_text:
            flat = " / ".join(o.ocr_text.split("\n"))[:400]
            lines.append(f"      screen text: {flat}")
    return "\n".join(lines)


def _build_prompt(
    observations: Sequence[Observation],
    config: CoalesceConfig,
    duration_sec: float,
    timeline: Optional[Timeline] = None,
) -> str:
    parts: List[str] = []
    if config.intent.strip():
        # Stated first and framed as the deciding criterion: "boring" is not a
        # property of footage, it is a property of footage relative to a
        # purpose, and without this the model can only re-derive sameness.
        parts.append(
            f"What this index is for: {config.intent.strip()}\n"
            f"Judge what deserves its own chapter against that purpose. "
            f"Work that does not serve it should be coalesced, however long "
            f"it ran."
        )
    header = (
        f"{len(observations)} observations spanning "
        f"{format_timestamp(duration_sec)} of recording."
    )
    if timeline is not None and timeline.is_multi:
        header += (
            f" The recording is split across {len(timeline)} separate capture "
            f"sessions, marked below; work does not necessarily continue "
            f"across a session change, and each one must start a new chapter."
        )
    parts.append(header)
    parts.append(render_observations(observations, config, timeline))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

def _call_claude(prompt: str, config: CoalesceConfig) -> List[Dict[str, Any]]:
    """One structured-output call returning the raw chapter dicts.

    Raises `llm.LLMError` rather than returning something partial: the caller
    falls back to the curve segmenter, which is a far better outcome than half
    a timeline.
    """
    data = llm.structured_call(
        model=config.model, system=_SYSTEM, content=prompt, schema=_SCHEMA,
        max_tokens=config.max_tokens, effort=config.effort,
    )
    chapters = data.get("chapters") or []
    if not chapters:
        raise llm.LLMError("coalescing returned no chapters")
    return chapters


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_and_repair(
    chapters: List[Dict[str, Any]],
    observations: Sequence[Observation],
    min_event_sec: float,
    duration_sec: float,
    locked: Optional[Sequence[int]] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Force the model's cut points into a valid partition, keeping its titles.

    Returns one dict per chapter, in order, each with `start` (an observation
    position), `title`, `reason` and `metadata.kind`.

    Cut points already make gaps and overlaps unexpressible, so this handles
    what remains: bad indices, bad ordering, a missing opening cut, file joins
    the model dropped, and chapters below the minimum length.

    Repair rather than reject. A boundary in slightly the wrong place is worth
    far more than no chapter list, and the failures here are the mechanical
    kind — the judgement about what goes with what is still the model's.
    """
    n = len(observations)
    locked_set = {int(b) for b in (locked or ()) if 0 < b < n}

    # Keep the first title offered for any given index; drop what cannot be
    # read. Every later step uses the parsed integer, never the raw value.
    by_start: Dict[int, Dict[str, Any]] = {}
    for c in chapters:
        try:
            idx = int(c["start"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < n and idx not in by_start:
            by_start[idx] = {**c, "start": idx}

    # A chapter must open at 0, and every file join must open one.
    if 0 not in by_start:
        by_start[0] = {"start": 0, "title": "", "kind": "active",
                       "why": "inserted: recording starts here"}
    for b in locked_set:
        if b not in by_start:
            by_start[b] = {"start": b, "title": "", "kind": "active",
                           "why": "inserted: new capture session"}

    starts = sorted(by_start)

    # Minimum duration, walked left to right exactly as the curve path does.
    times = [o.t for o in observations]
    kept = enforce_min_duration(
        starts[1:], times, min_event_sec, duration_sec,
        locked=sorted(locked_set),
    )
    final = [0] + list(kept)

    if verbose and len(final) != len(starts):
        print(f"  [coalesce] validation dropped "
              f"{len(starts) - len(final)} boundary(ies) under "
              f"{min_event_sec:.0f}s")

    out: List[Dict[str, Any]] = []
    for st in final:
        c = by_start[st]
        kind = str(c.get("kind", "")).strip()
        out.append({
            "start": st,
            "title": str(c.get("title", "") or "").strip(),
            "reason": str(c.get("why", "") or "").strip(),
            "metadata": {"kind": kind if kind in _KINDS else "active"},
        })
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def coalesce(
    observations: Sequence[Observation],
    config: CoalesceConfig,
    min_event_sec: float,
    duration_sec: float = 0.0,
    timeline: Optional[Timeline] = None,
    locked: Optional[Sequence[int]] = None,
    verbose: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """Decide where chapters begin and what they are called.

    Returns validated chapter dicts (see `_validate_and_repair`), ready for
    `cuts.chapters.events_from_starts`, or None when the call did not produce
    a usable partition — the caller then falls back to the curve segmenter.

    `locked` are observation positions that must open a chapter regardless
    of what the model decided; the pipeline passes the source-file joins.
    """
    n = len(observations)
    if n == 0:
        return None
    end_time = duration_sec or observations[-1].t
    if n == 1:
        return [{"start": 0, "title": "", "reason": "",
                 "metadata": {"kind": "active"}}]

    if n > config.max_observations:
        print(f"[coalesce] WARNING: {n} observations exceeds "
              f"max_observations={config.max_observations}; the prompt will be "
              f"large and quality may suffer. Raise --interval to sample less "
              f"densely, or split the input.")

    prompt = _build_prompt(observations, config, end_time, timeline)
    if verbose:
        print(f"  [coalesce] {n} observations -> {config.model} "
              f"(~{len(prompt) // 4} prompt tokens, "
              f"ocr={'on' if config.include_ocr else 'off'})")

    try:
        raw = _call_claude(prompt, config)
    except llm.LLMError as exc:
        print(f"[coalesce] failed ({exc})")
        return None

    chapters = _validate_and_repair(
        raw, observations, min_event_sec, end_time, locked=locked, verbose=verbose
    )
    if verbose:
        print(f"  [coalesce] {len(raw)} proposed -> {len(chapters)} chapters")
    return chapters


if __name__ == "__main__":
    # Standalone debug: coalesce a cached observations file and print the
    # decision, without touching the index. `cuts coalesce` is the real
    # command; this is the quick look.
    import sys

    from .chapters import join_positions
    from .config import CutsConfig
    from .observe import read_observations

    if len(sys.argv) < 2:
        print("usage: python -m cuts.coalesce <observations.jsonl> [intent]")
        sys.exit(1)

    cfg = CutsConfig()
    if len(sys.argv) > 2:
        cfg.coalesce.intent = sys.argv[2]
    obs, _meta = read_observations(sys.argv[1])
    print(f"{len(obs)} observations, {format_timestamp(obs[-1].t)} of footage")
    result = coalesce(obs, cfg.coalesce, cfg.segmentation.min_event_sec,
                      locked=join_positions(obs), verbose=True)
    for c in result or []:
        print(f"  {format_timestamp(obs[c['start']].t):>7} "
              f"[{c['metadata']['kind']:>7}] {c['title']}  — {c['reason']}")
