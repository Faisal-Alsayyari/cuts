# 11 — Findings: bugs, gaps, dead code, inconsistencies, bottlenecks, debt

Everything here was found by reading the code at `6c551be`. Items marked
**(verified)** were also confirmed by a read-only check in the local
environment; everything else is established from the source alone and was
not executed. Severity reflects impact on the default `chapters` run.

## Summary

| # | Finding | Severity | Area |
|---|---------|----------|------|
| 1.1 | Failed description calls are cached as empty descriptions and reused | High | Pass 1 / cache |
| 1.2 | Default path without credentials yields one "unclear" chapter per file; README and GUI promise visual-only segmentation | High | Pipeline / docs |
| 1.3 | `except av.AVError` — attribute does not exist in installed PyAV 17 **(verified)** | Medium | Media |
| 1.4 | Pass 1 is all-or-nothing: one shard exception discards all work | Medium | Pass 1 |
| 1.5 | Cache hits skip duration confirmation; multi-file offsets can disagree with cached times | Medium | Pass 1 / timeline |
| 1.6 | Refinement batches unrelated midpoint frames as "consecutive" | Medium | Refinement |
| 1.7 | `--no-labels` on the LLM path forces the "coalescing failed" fallback | Low | CLI |
| 1.8 | `cuts coalesce` rebuilds the timeline from headers, can index past `sources`, writes a different `video_id` | Low | CLI |
| 1.9 | `cuts label` drops the index's `extra` block | Low | CLI |
| 1.10 | GUI Run panel stops updating after 5 000 log lines | Low | GUI |
| 1.11 | Final grid sample can fall past the last frame and become a spurious `withheld` | Low | Pass 1 |

---

## 1. Correctness bugs

### 1.1 Transient API failures are cached permanently

- `describe_frames` returns `[""] * n` on any exception, empty response,
  or bad JSON (`cuts/signals/vlm.py:244-257`), logging only under `-v`.
- `observe()` writes whatever came back to the cache
  (`cuts/observe.py:486-489`), and the fingerprint
  (`cuts/observe.py:384-406`) does not record whether descriptions
  succeeded.
- **Scenario:** a run during a rate-limit storm, an outage, or with no
  credentials produces observations with empty descriptions. Every later
  `chapters`/`coalesce` run — including the GUI's Re-chapter — silently
  reuses them. Coalescing sees `(no description)` lines, and its output
  quality collapses with no visible error. Only `--no-cache` (or deleting
  the file) recovers.

### 1.2 Default path without credentials

- `_run_llm` (`cuts/pipeline.py:454`) never checks for credentials, unlike
  the curve path (`cuts/pipeline.py:134-139`).
- **Scenario:** no `ANTHROPIC_API_KEY`, no profile. Every describe call
  fails → empty, cached (1.1). Coalesce fails → `_fallback_curve` sets
  `weight_dino = 0`, so the only channel is TF-IDF over empty strings,
  which `build_channels` drops; with no audio there are no channels and
  `segment()` returns one event per file (`cuts/segmentation.py:417-418`).
  Heuristic labels on empty text → `"unclear"`.
- The README ("without it the pipeline still produces chapters (segmenting
  on the visual channel alone …)") and the GUI dialog
  (`cuts_gui/dialogs.py:197-203`) describe the curve path's behaviour, not
  the default's.

### 1.3 `av.AVError` no longer exists in the installed PyAV

- **(verified)** In `.venv`, PyAV 17.0.1: `'AVError' in dir(av)` is
  `False`; `FFmpegError` exists.
- Used at `cuts/media.py:208`, `:539`, `:608`. When `container.seek`
  raises, Python evaluates `av.AVError` to match the `except` clause, which
  raises `AttributeError` instead.
- Impact: `true_duration_sec` is wrapped in an outer `except Exception` and
  is safe. `decode_frame_near` (refinement) and `iter_frames_at_times`
  (Pass 1, labeling) propagate the `AttributeError`; in Pass 1 that aborts
  the run (1.4). Dependencies are unpinned, so any environment with
  PyAV ≥ 14 behaves this way.

### 1.4 Pass 1 is all-or-nothing

- `observe_timeline` uses `pool.map` (`cuts/observe.py:354-355`); the first
  exception from any shard propagates. Exceptions not caught inside
  `observe_shard` include `av.open` failure, `encode_frame`'s
  `RuntimeError`, and 1.3.
- The cache is written only after all shards finish (`observe.py:486`), so
  a crash, a cancel from the GUI, or Ctrl-C at 95 % loses everything. With
  Pass 1 being "the expensive half" (8 minutes for 87 minutes of footage per
  the README) this is the largest robustness gap.

### 1.5 Cache hits skip duration confirmation, so offsets can disagree with cached observations

- Cache miss: `confirm_durations` re-lays source offsets from true file ends
  *before* observations are created (`cuts/observe.py:328`); observation
  `t = confirmed_offset + local_t`.
- Cache hit: `observe()` returns before `confirm_durations`
  (`cuts/observe.py:476-481`); the `Timeline` keeps header durations.
- **Scenario:** two files, where file 1's container under-reports its length
  by 40 s. On the first run, file 2's observations are stamped with the
  confirmed offset. On a cached rerun, `timeline.local_time()` uses the
  40 s-shorter header offset, so `_attach_sources` records the wrong
  in-file offset (or wrong file) for events near the join, refinement
  decodes the wrong midpoint frame, and ASR transcripts are shifted 40 s
  relative to observations. Single-file runs only see a slightly different
  `duration_sec`.
- `cuts coalesce` has the same issue (it always uses `Timeline.build`).

### 1.6 Refinement batches unrelated frames as consecutive

- `refine_boundaries` sends every midpoint frame through `describe_batch`
  (`cuts/refine.py:131`), which groups them six per call.
- `_DESCRIBE_SYSTEM` (`cuts/signals/vlm.py:48-58`) tells the model the
  frames are "consecutive frames from one recording" and to use neighbours
  to identify ambiguous windows. Midpoints from boundaries minutes or hours
  apart (or from different files) are not neighbours; the model is invited to
  borrow context across them, which can bias exactly the description the
  A/M/B decision depends on.

### 1.7 `--no-labels` on the LLM path

- `_config_from_args` sets both `labeling.enabled = False` and
  `coalesce.enabled = False` (`cuts/cli.py:185-187`).
- `_run_llm` treats disabled coalescing as failure
  (`cuts/pipeline.py:527-538`): it prints "coalescing failed; falling back
  …", runs curve segmentation over descriptions, and `label_events` returns
  `"none"`. The user gets curve boundaries rather than LLM boundaries without
  titles, plus a misleading failure message. The GUI would also surface it
  as a warning.

### 1.8 `cuts coalesce` edge cases

- Rebuilds the timeline with `Timeline.build` (header durations; see 1.5)
  from only the files that still exist (`cuts/cli.py:377-378`). If a middle
  file of a multi-file cache is missing, `timeline.sources` is shorter than
  the observations' `source_index` range, and
  `render_observations` indexes `timeline.sources[o.source_index]`
  (`cuts/coalesce.py:163`) → `IndexError`.
- `write_events(out_path, Path(out_path).stem, …)` (`cuts/cli.py:404`)
  stores `video_id = "<stem>.events"`, while `chapters` stores `"<stem>"`.
  The GUI displays `video_id` as the index title, so it changes after a
  Re-chapter.

### 1.9 `cuts label` drops metadata

`cmd_label` rewrites the index without `extra` (`cuts/cli.py:444-446`),
losing `segmenter`, `intent`, `stage_times` and sample counts. The GUI's
facts panel then shows nothing for that session.

### 1.10 GUI Run panel freezes on long logs

`JobRunner._consume` trims `RunState.log` to the last 5 000 lines
(`cuts_gui/jobs.py:217-218`). `RunPanel.show_state` appends
`st.log[self._lines_shown:]` and only resets when
`_lines_shown > len(st.log)` (`cuts_gui/panels.py:868-878`). After the
first trim both values are 5 000, so no further lines are shown for the
rest of the run. Likely with `-v` (which the GUI always passes) on long or
failing runs, e.g. one `[vlm] describe failed` line per call.

### 1.11 Trailing withheld observation (likely)

`plan_shards` includes `t = duration` when `duration` is a multiple of the
interval (within 1e-6), and `true_duration_sec` returns
`max(header, last_frame_pts)`; the header is commonly a little past the last
frame's pts. A request at that time finds no frame `>= target` and yields
`None`, producing a final `withheld` observation that renders as
`(frame withheld)` in the coalescing prompt. Not reproduced; follows from
`cuts/observe.py:162-164` and `cuts/media.py:214`.

---

## 2. Incomplete features

| Feature | Evidence | State |
|---------|----------|-------|
| Agent layer (`zoom(t0,t1,n)`, `get_ocr(t0,t1)`) | README "What this is not"; `Shard.sample_times` design; `resume_window_sec` | No tools, no loop, no `get_ocr`. See [08 §6](08-llm-integration.md#6-agentic-functionality-what-exists). |
| Query-driven retrieval (EFS stages 2–3) | `cuts/select.py` | Implemented but unreachable; undeclared `open_clip` dependency; no adapter for LLM-path data. |
| Text search (BM25 + embeddings + RRF) | `SearchConfig` (`config.py:449`), `retrieval_index/` artefacts, `sentence_transformers` in venv | Code removed; config and artefacts remain. |
| Footage cutting using `kind` | `coalesce.py:55-58` | `kind` produced and coloured in the GUI; nothing consumes it. |
| Windowed coalescing for >1500 observations | `CoalesceConfig.max_observations` | Only a warning; no windowing. |
| Per-event labeling mode | `LabelingConfig.batch_all_events` | Field never read. |
| Frame-index cache | `build_frame_index`, `CutsConfig.cache_dir` | No caller; `cache_dir` never set. |
| Titles for validator-inserted chapters | `_validate_and_repair` inserts `title: ""` | Shown as `(unlabeled)`/`(untitled)`. Also the `n == 1` case. |
| Incremental / resumable Pass 1 | JSONL docstring mentions streaming and tailing (`observe.py:415-417`) | File is written in one go at the end. |
| Transcript caching | `Observation.transcript_text` | Never persisted; recomputed every run; absent from `cuts coalesce`. |
| `--vlm-stride` / `max_inflight_chunks` on the default path | config | Curve path only. |
| Tests | README "Status" describes validation runs | No tests in the repository. |

## 3. Unused code

| Symbol | Location |
|--------|----------|
| `cuts/select.py` (whole module) | — |
| `SearchConfig`, `CutsConfig.search`, `CutsConfig.cache_dir` | `config.py:449`, `:489`, `:493` |
| `LabelingConfig.batch_all_events` | `config.py:422` |
| `media.build_frame_index`, `_cache_path_for`, `get_fps`, `Timeline.source_for_frame` (only via `local_frame`) | `media.py:89`, `:126`, `:147` |
| `vlm.describe_image` (docstring says refine uses it; refine uses `describe_batch`) | `signals/vlm.py:272` |
| `dino.embed_frames`, `dino.cosine_similarity_matrix`, `dino.adjacent_similarity` (debug only) | `signals/dino.py:60`, `:140`, `:151` |
| `text_features.adjacent_similarity` | `signals/text_features.py:102` |
| `align_to_samples(window_sec=…)` parameter (always 0) | `signals/asr.py:89` |
| `Event.metadata["merged_from"]` (documented, never written) | `schema.py:101` |

## 4. Architectural inconsistencies

1. **Two pipelines, one function.** `pipeline.run` contains the whole curve
   path inline (~330 lines) and dispatches the LLM path to `_run_llm`. They
   share ASR logic by copy (`pipeline.py:354-374` vs `:496-513`), and the
   module docstring describes only the curve path as "the" pipeline.
2. **`label_backend` is overloaded.** On the LLM path it holds the
   segmenter outcome (`"llm"`, `"curve-fallback/heuristic"`,
   `pipeline.py:570`); the CLI prints it as "titles:". `--backend` is
   silently ignored on the LLM path and in the fallback (`pipeline.py:614`).
3. **`Event.summary` holds the boundary justification.** Coalescing stores
   `why` in `summary` (`coalesce.py:352`); the GUI renders it as "Why it
   starts here". The schema documents `summary` as "optional longer
   description".
4. **`withheld` means two things.** Redaction-withheld and undecodable are
   indistinguishable in the cache, the prompt, and the GUI's "flagged"
   filter.
5. **Inconsistent credential detection.** The curve path and labeling check
   only `ANTHROPIC_API_KEY`; the SDK accepts several sources; the LLM path
   checks nothing; the GUI checks env + `.env`.
6. **Concurrency knobs apply to one path each.** `vlm.concurrency` bounds
   per-shard pools on the LLM path but not total concurrency;
   `max_inflight_chunks` and `stride` only affect the curve path.
7. **Observation time is the requested time, not the frame time.**
   `observe.py:264-268` stamps `t` from the grid; the actual frame may be up
   to `seek_tolerance_sec` (2 s) later.
8. **Private cross-module imports** (`coalesce → segmentation._enforce_min_duration`,
   `pipeline → coalesce._dedup/_pick_representative_times`,
   `pipeline → labeling._pick_representative`, `cli → pipeline._attach_sources`)
   couple modules through underscore APIs.
9. **Cache identity differs between backend and GUI.** Backend: absolute
   path + mtime + size; GUI `_notes_match`: basename + size. After moving
   videos the GUI offers Re-chapter (fine, `coalesce` doesn't check), but a
   full Analyze will not reuse the cache.
10. **Redaction covers pixels and OCR text, not speech.** Transcripts go to
    the coalescing and labeling prompts unscanned.
11. **GUI filmstrip shows unredacted frames** that the notes panel takes
    care to pixelate (`timeline.py:547-575` vs `panels.py:728-745`).
    Local display only.
12. **Documentation drift** (code is authoritative):
    - README tells users to `cp .env.example .env`; `.env.example` was
      deleted in `6c551be`.
    - README "No web UI yet" and its module table do not mention `cuts_gui`
      at all.
    - `cuts/__init__.py` still describes "frame-accurate shot boundary and UI
      state change event detection".
    - `pipeline.py` module docstring describes the curve path only.
    - `signals/vlm.py` docstrings say the pipeline does not use
      `describe_batch` on its main path; Pass 1 does.
    - `cli.cmd_probe` estimate assumes linear decode + DINO (curve path).
    - `Event` docstring: `sample_times` "canonical", but empty on the curve
      path; `metadata` keys listed are stale.
    - README "Performance" bullet about calls "submitted without waiting"
      and `max_inflight_chunks` describes the curve path, not Pass 1.

## 5. Potential bottlenecks

| Bottleneck | Where | Why it matters |
|------------|-------|----------------|
| OCR at 2× upscale on every sample | `signals/ocr.py:96-111` | A 1080p frame becomes 3840×2160 before detection; ONNX uses all cores, which is why sharding gives only ~1.4× locally. Dominant local cost. |
| No overlap of decode/OCR with API calls inside a shard | `observe.py:263-302` | Each shard decodes+OCRs all its samples, *then* makes its calls. Overlap only comes from other shards running. |
| Uncoordinated API concurrency, new client per call | `observe.py:354`, `vlm.py:141`, `:317` | Up to `workers × chunks` concurrent requests with no shared limiter; connection pools not reused; 429s become silent empty descriptions after 2 retries (1.1). |
| Single coalescing call | `coalesce.py:211` | Prompt grows linearly; past ~1500 observations only a warning; non-streaming with 16k max tokens. |
| ASR re-run on every invocation, sequential per file | `pipeline.py:496-513` | Whole-file Whisper on each `chapters` run even when Pass 1 is cached. |
| Curve path sequential full decode | `media.py:241` | Linear in total length; the README measures ~78 s vs ~14 s seek-based on 87 min. |
| `merge_to_target` recomputes all partition means each iteration | `segmentation.py:316-326` | O(minima × N × D) with a dense up-to-20k-column TF-IDF matrix. |
| Dense TF-IDF matrices | `text_features.py:84` | `N × V` float32; large for long inputs. |
| Labeling decode on curve path | `media.py:642` via `labeling.py:216` | Sequential decode up to the last representative frame per file. |
| `--no-seek` fallback | `observe.py:197-220` | Each shard decodes from the file start → roughly quadratic total cost. |
| GUI thumbnails | `cuts_gui/media.py:76` | Opens and seeks the container per thumbnail; two workers. |

## 6. Technical debt and environment

- **No automated tests**, while the code contains intricate invariants
  (cut-point validation, redaction box mapping, seek tolerance, VFR timing)
  that the README says were verified manually.
- **Unpinned dependencies** (`requirements.txt`, `pyproject.toml`); the
  PyAV 17 break (1.3) is a direct consequence.
- **Local `.venv` does not match the documented environment** **(verified)**:
  Python 3.14 (README requires 3.12), and `anthropic`, `python-dotenv`,
  `PySide6` are not installed.
- **`PySide6` undeclared**, yet `cuts_gui` is included by the
  `include = ["cuts*"]` package glob; no `cuts-gui` entry point.
- **Large untracked artefacts** in the working tree (`OmniShotCut/`,
  `checkpoints/`, `retrieval_index/` ≈ 620 MB together), plus stale
  `cuts.egg-info/` and `dummy.json`, all from removed features.
- **`print`-based logging** everywhere, which the GUI parses with regexes
  for progress. Any wording change degrades GUI progress display (the GUI
  documents this risk itself).
- **Mutable config passed by reference** and mutated by the CLI; only the
  fallback path copies before mutating.
- **Lazy-import style** keeps start-up light but hides import errors until
  mid-run (e.g. a missing `anthropic` package is swallowed inside
  `describe_frames`'s `try`).

## 7. Suggested priorities (for whoever picks this up)

These are observations about where effort would pay off most, not changes
made as part of this documentation.

1. Stop caching failed descriptions (record success in the cache, or skip
   writing observations whose description call failed) and check
   credentials up front on the LLM path (1.1, 1.2).
2. Replace `av.AVError` with `av.FFmpegError` (or `Exception`) and pin
   dependency versions (1.3).
3. Persist Pass 1 incrementally per shard and make shard failures local
   (1.4).
4. Run `confirm_durations` (or persist confirmed offsets in the cache meta)
   on cache hits too (1.5).
5. Describe refinement midpoints individually or with a prompt that does
   not claim adjacency (1.6).
6. Add a minimal test suite around `_validate_and_repair`,
   `redact_lines`, `plan_shards`, `iter_frames_at_times`, and
   `segment()` — the README already lists the cases.
