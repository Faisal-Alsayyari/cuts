# 08 — LLM integration (and the state of "agentic" functionality)

> **Pre-refactor.** This document describes the code at `c39a118`. The pipeline was since unified on branch `refactor/unify-pipeline`; see [12-unified-pipeline.md](12-unified-pipeline.md) for what changed.

## 1. Inventory of Claude API calls

Every call is a single, non-streaming `messages.create` with a JSON-schema
structured output (`output_config.format`). There are no tools, no
multi-turn conversations, no streaming, no prompt caching, and no Batch API
use.

| # | Call | Where | Model (default) | Inputs | Output schema | Per run |
|---|------|-------|-----------------|--------|---------------|---------|
| 1 | Describe frames | `signals/vlm.py:184` `describe_frames` | `claude-haiku-4-5` (`--vlm-model`) | ≤6 JPEGs (768 px) + per-frame OCR text | `{descriptions:[{frame:int, description:str}]}` | ⌈samples/6⌉ (+ midpoints) |
| 2 | Coalesce chapters | `coalesce.py:211` `_call_claude` | `claude-opus-5` (`--coalesce-model`), `effort=medium`, `max_tokens=16000` | all observations as text (+ intent, transcript, optional OCR) | `{chapters:[{start:int, title, kind:enum, why}]}` | 1 (LLM path) |
| 3 | Refinement decision | `signals/vlm.py:363` `decide_refinement` | `refine.model or vlm.model` → Haiku 4.5, `max_tokens=100` | 3 descriptions (A/M/B) as text | `{transition: first_half\|second_half\|unclear}` | 1 per refinable boundary |
| 4 | Label chapters | `labeling.py:339` `_call_claude` | `claude-opus-5`, `effort=medium`, `max_tokens=8000`, beta `server-side-fallback-2026-07-01` + `fallbacks="default"` | per-event descriptions/transcript + 2 JPEGs per event | `{chapters:[{event_id, title}]}` | 1 (curve path, fallback, `cuts label`) |

The parameter shapes (`output_config.format` with `json_schema`,
`output_config.effort`, the `server-side-fallback-2026-07-01` beta with
`fallbacks="default"`) and the model IDs `claude-opus-5` and
`claude-haiku-4-5` were checked against the current API reference and are
valid. Claude Haiku 4.5 does not accept `effort`, and calls 1 and 3 do not
send it. Newer model generations (Opus 5.5, Haiku 5.5) exist; the defaults
are a generation behind but this is a choice, not an error.

### Client construction and credentials

- Calls 1 and 3 build a client per call via `_client(api_key)`
  (`vlm.py:141`); calls 2 and 4 build one per call inline. No client is
  shared or reused across threads.
- `anthropic.Anthropic()` with no key resolves credentials from the
  environment (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, an `ant auth
  login` profile, …). The pipeline's own up-front credential checks
  (`pipeline.py:135`, `labeling.py:483`) only look at `ANTHROPIC_API_KEY`,
  so a profile-only setup is treated as "no credentials" by those checks
  while the SDK itself could authenticate.
- `api_key` is plumbed through every function but **no CLI command passes
  one**; credentials always come from the environment (`.env` is loaded by
  `cli._load_dotenv`).
- The SDK's defaults apply: 2 automatic retries for connection errors,
  408/409/429/5xx, and a 10-minute timeout. There is no application-level
  retry, rate limiting, or backoff.

## 2. Failure semantics per call

| Call | Exception | Refusal | `max_tokens` | Bad JSON | Effect on the run |
|------|-----------|---------|--------------|----------|-------------------|
| Describe | → `""` per frame | not checked (text likely empty → `""`) | not checked; truncated JSON → `""` for all frames in the call | → `""` | Run continues; empty descriptions are **cached** |
| Coalesce | raise → `"failed"` | raise → `"failed"` | raise → `"failed"` | raise → `"failed"` | Fallback to curve segmentation + labeling |
| Refine decide | → `"unclear"` | → `"unclear"` | → `"unclear"` if no text | → `"unclear"` | Boundary not moved |
| Label | → heuristic titles | raise → heuristic | raise → heuristic | raise → heuristic | Crude titles |

The asymmetry matters: the *cheapest and most numerous* call (describe)
fails silently and its failures persist across runs via the cache, while the
expensive calls fail loudly and degrade gracefully.

## 3. Prompt design highlights

- **Describe** (`_DESCRIBE_SYSTEM`, `vlm.py:48`): one ≤25-word sentence per
  frame naming application, most distinguishing detail and activity;
  descriptions must stand alone (no "same as previous"); OCR is the source
  of truth for exact strings; ignore window chrome/taskbars/webcams.
  The stand-alone rule exists for the curve path's TF-IDF comparison. On the
  LLM path the coalescing model reads all descriptions at once, so the rule
  is less essential there but still applied.
- **Coalesce** (`coalesce.py:60`): cut on activity change not layout
  change, uneven lengths encouraged, prefer fewer chapters, titles are
  activities not UI, `kind` priority signal, `why` must cite observation
  numbers. Intent is placed first in the user message.
- **Refine** (`vlm.py:116`): A/M/B descriptions, answer which half of the
  gap holds the transition or "unclear".
- **Label** (`labeling.py:75`): titles describe what the person was doing,
  2–5 lowercase words, distinct across segments, "unclear" when evidence is
  thin.

## 4. Data sent off-machine

| Upload | Pixels | Text | Redaction applied |
|--------|--------|------|-------------------|
| Describe (Pass 1 / curve) | 768 px JPEG q85 per sample | OCR grounding (redacted) | Yes, via `scrub_frame` before resize (curve path: only on frames to be described, which are the only ones uploaded) |
| Refine midpoints | 768 px JPEG | OCR grounding | Yes (`refine.py:103`) |
| Refine decision | none | 3 descriptions | n/a |
| Coalesce | none | descriptions, transcripts (≤300 chars each), OCR text with `--include-ocr` (≤400 chars each) | OCR text is already redacted; **transcripts are never scanned for credentials** |
| Label | 768 px JPEG q80, 2 per event | descriptions, transcripts | Frames yes; text not re-scanned |

Spoken credentials (or a credential read aloud) reach the coalescing and
labeling prompts unfiltered.

## 5. Cost drivers (structural, not measured)

- Describe calls scale with `duration / interval`; each carries up to six
  images plus OCR text. This is the dominant API cost.
- Coalesce is one call whose input grows linearly with observation count
  (~33 tokens per description per the README; ~250 more per observation
  with `--include-ocr`), with a hard warning at 1500 observations.
- No prompt caching is used. The describe system prompt is identical across
  hundreds of calls and would be a natural cache candidate, though it may
  be below the model's minimum cacheable prefix length.

## 6. Agentic functionality: what exists

**There is no agent loop, tool use, or skill system in the codebase.** All
LLM interaction is the four fixed, single-shot calls above, orchestrated by
ordinary Python control flow. There is no `.claude/` directory, no skills
folder, no tool definitions, and nothing that lets a model request more
frames or re-query observations.

What exists is groundwork that the README describes as the "hinge" for a
future agent layer:

| Groundwork | Where | Status |
|------------|-------|--------|
| `observe_shard` takes an explicit list of sample times, so a `zoom(t0, t1, n)` tool would be the same function with a denser list | `observe.py:113-139`, `:223` | Not wired to anything |
| `iter_frames_at_times(resume_window_sec=10)` keeps decoding instead of re-seeking for dense requests | `media.py:578-583`, `:606` | Only exercised by sparse sampling today |
| "`get_ocr(t0, t1)` is a read against the cached observations" (README) | — | No such function exists |
| `select.py` query-driven frame selection ("find the moment matching this description") | `select.py` | Implemented, unreachable, depends on undeclared `open_clip` |
| `kind` per chapter as a priority signal for a future cutting pass | `coalesce.py:55-58` | Produced and displayed in the GUI; nothing consumes it |
| GUI isolates itself from backend internals because "the backend is still moving (… an agent layer)" | `cuts_gui/__init__.py` | Design intent only |

Anyone building the agent layer should note: observations carry no image
embeddings, the cache has no random-access index (it is a sorted JSONL
file loaded whole), and Pass 1 cannot currently be invoked on an arbitrary
time list from the CLI.
