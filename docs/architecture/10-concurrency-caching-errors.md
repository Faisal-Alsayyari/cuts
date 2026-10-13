# 10 — Concurrency, caching, and error handling (cross-cutting)

> **Pre-refactor.** This document describes the code at `c39a118`. The pipeline was since unified on branch `refactor/unify-pipeline`; see [12-unified-pipeline.md](12-unified-pipeline.md) for what changed.

## 1. Every thread and process the system creates

| Component | Mechanism | Size | Lifetime | Shared state |
|-----------|-----------|------|----------|--------------|
| Pass 1 shard pool | `ThreadPoolExecutor` (default) / `ProcessPoolExecutor` / serial | `min(workers or cpu_count, #shards)` | One `observe_timeline` call | RapidOCR engine global (threads), `print` |
| Per-shard describe pool | `ThreadPoolExecutor` in `describe_batch` | `vlm.concurrency` (6) | One `describe_batch` call | none |
| Curve-path VLM executor | `ThreadPoolExecutor` | `vlm.concurrency` (6), backlog ≤ `max_inflight_chunks` (12) | The decode loop | `vlm_raw` list, written by the main thread only when resolving futures |
| Refinement describe pool | `describe_batch` | 6 | One call | none |
| Refinement decide pool | `ThreadPoolExecutor` | 6 | One call | none |
| PyAV decode threads | `stream.thread_type = "AUTO"` | FFmpeg-managed | Per container | none |
| ONNX Runtime (RapidOCR) | intra-op threads | all cores | Process | the engine |
| torch (DINOv2) | intra-op threads / GPU | — | Process | model global |
| GUI subprocess | `QProcess` | 1 at a time | One job | files on disk |
| GUI thumbnails | `QThreadPool` | 2 | App | pending stack under a `threading.Lock`, results via a queued Qt signal |

### Nesting on the LLM path

`workers` shard threads each create their own describe pool, so the number
of OS threads is `workers × (1 + 6)` at peak and the number of concurrent
HTTP requests is bounded only by how many describe chunks each shard has
(≈2 at defaults). Nothing coordinates request rate across shards; the SDK's
two retries are the only defence against 429s, and a request that still
fails becomes an empty description.

### Thread-safety notes

- `ocr._get_engine` (`signals/ocr.py:56`) lazily initialises a module
  global without a lock; two shard threads can both construct an engine on
  first use (wasteful, not incorrect). All threads then call the same
  `RapidOCR` instance concurrently. RapidOCR calls into ONNX Runtime
  sessions, whose `run` is documented as thread-safe; the Python wrapper's
  own thread safety is not established by this codebase.
- `dino._get_model` and `asr._get_model` have the same unlocked-singleton
  pattern but are only called from the main thread.
- `redact.scrub_frame` mutates the frame passed to it; each frame belongs to
  exactly one shard/loop iteration, so there is no sharing.
- Logging is `print` from many threads; lines can interleave but each
  `print` call is a single write in practice.

### Process executor specifics

`_run_shard` is a top-level function so it pickles. Each worker process
re-imports `cuts`, creates its own OCR engine (one ONNX load per process),
and receives a pickled `CutsConfig`. Measured slower than threads on a
single machine (see `ObserveConfig.executor` comment, `config.py:101-118`).

## 2. Every cache

| Cache | Key | Location | Invalidation | Notes |
|-------|-----|----------|--------------|-------|
| Observation cache | fingerprint: version, `[abs path, mtime, size]` per file, interval, redact, ocr, vlm model | `<first_stem>.observations.jsonl` | Exact meta mismatch; `--no-cache`; bump `CACHE_VERSION` | Written once at the end. Does not record description success, prompt version, OCR settings, or patterns. Shared path for different file sets that start with the same file. |
| RapidOCR engine | process | memory | never | |
| DINOv2 model | `(model_name, device)` | memory | on key change | |
| Whisper model | `(model_name, device)` | memory | never | |
| CLIP model (`select.py`) | `(model, pretrained, device)` | memory | — | unreachable |
| Frame-index pickle | path+mtime+size | `<cache_dir>/frame_index/` | — | unreachable (`build_frame_index` has no caller, `cache_dir` is never set) |
| HF / Whisper weights | library-managed | user cache dirs | library-managed | downloaded on first use |
| GUI thumbnails | `path|t(2dp)|width` | memory, LRU 800 | eviction; failures remembered for the session | |
| GUI library | session id | `library.json` | — | |

## 3. Error-handling map

The general policy, stated in many docstrings, is "degrade rather than
abort": a chapter list with worse evidence is preferred to no output. The
table shows where that holds and where it does not.

| Failure | Where caught | Outcome |
|---------|--------------|---------|
| Video file missing | `cli._resolve_videos` | exit 2 before any work |
| `av.open` fails during Pass 1 | **not caught** in `observe_shard` | exception propagates through `pool.map`; whole run aborts; no cache written |
| Seek raises inside `iter_frames_at_times` / `decode_frame_near` | `except av.AVError` — attribute missing in PyAV 17 | `AttributeError` propagates (see findings §1.3) |
| Seek raises inside `true_duration_sec` | outer `except Exception` | returns header duration (safe) |
| Frame not within tolerance / past EOF | `iter_frames_at_times` yields `None` | observation `withheld=True` |
| OCR engine exception | `ocr_lines` → `None` | redaction on: frame withheld; off: no grounding text, frame still uploaded |
| Credential not localisable | `redact_lines` → `safe=False` | frame withheld (Pass 1, curve) / boundary not refined / labeling frame dropped |
| Invalid user regex | `_patterns` | printed and skipped |
| JPEG encode fails | `encode_frame` raises `RuntimeError` | not caught → aborts the shard/run |
| Describe call fails (any reason) | `describe_frames` | `""` descriptions, logged only with `-v`, **cached** |
| Coalesce call fails/refuses/truncates | `coalesce` | `"failed"` → curve fallback + labeling |
| Refinement decision fails | `decide_refinement` | `"unclear"` → no move |
| Labeling call fails | `label_events` | heuristic titles |
| Fallbacks beta rejected | `labeling._call_claude` catches `BadRequestError` | retried without the beta (also catches any other 400, e.g. an invalid model, which then fails again and falls to heuristic) |
| faster-whisper not installed | `asr.transcribe` | no transcript |
| Whisper GPU init fails | `asr._get_model` | CPU fallback |
| Whisper transcription raises | **not caught** | run aborts (on the LLM path, after Pass 1 has already been cached) |
| Cache unreadable / corrupt | `load_cached` catches `OSError`, `JSONDecodeError` | treated as a miss (other exceptions, e.g. a `TypeError` from an unexpected field type, propagate) |
| Cache write fails | `observe` catches `OSError` | message printed, run continues |
| events.json write fails | `write_events` cleans its temp file and re-raises | CLI crashes with a traceback |
| Empty result | `cmd_chapters` | exit 1, nothing written |
| GUI: process cannot start | `JobRunner._failed_to_start` | FAILED with a "check the interpreter" message |
| GUI: index unreadable | `MainWindow._show_session` | banner + bare (chapterless) index |

## 4. Atomicity of writes

| File | Method |
|------|--------|
| `events.json` | `mkstemp` in same dir → `json.dump` → `os.replace`; temp removed on error |
| `observations.jsonl` | write `<path>.tmp` → `os.replace`; a crash mid-write leaves a stray `.tmp` (not cleaned) |
| `library.json` | `mkstemp` → `os.replace`; temp removed on `OSError` |
| frame-index pickle | best-effort direct write (unreachable code) |
