# cuts

Efficient semantic indexing and temporal navigation of long-form video.

Point it at a multi-hour screen recording and get back a chapter list:

```
0:00 writing parser
7:32 compiling
9:11 debugging segfault
21:54 reading docs
29:13 implementing fix
38:51 tests finally pass
```

The primary target is screen captures of coding sessions, where the hard part
is not finding visual cuts — there usually aren't any — but finding the points
where *what the person is doing* changes. Input can be one file or a session
split across several days' worth of them.

## Install

Requires Python 3.12 (3.13 lacks wheels for `ctranslate2`, faster-whisper's
backend).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Model weights (DINOv2, Whisper) download locally on first use. The text
channel and chapter titles both call the Claude API, so `ANTHROPIC_API_KEY`
is required for a full run — without it the pipeline still produces chapters
(segmenting on the visual channel alone, with crude keyword-derived titles),
but the text signal that normally carries this domain is simply absent.

The easiest way to set the key: `cp .env.example .env` and fill it in. The CLI
loads `.env` automatically (see `_load_dotenv` in
[cuts/cli.py](cuts/cli.py)); `.env` is gitignored.

## Run

```bash
python -m cuts probe session.mp4                  # metadata + runtime estimate
python -m cuts chapters session.mp4
python -m cuts chapters session.mp4 --intent "a tutorial walkthrough" -v
python -m cuts label session.events.json          # re-title without re-analyzing
```

The two passes are also separately invocable, and splitting them is how you
actually work on chapter quality:

```bash
python -m cuts observe session.mp4                # Pass 1 -> observations.jsonl
python -m cuts coalesce session.observations.jsonl --intent "..."
python -m cuts coalesce session.observations.jsonl --intent "something else"
```

Observing is the expensive half and is cached; coalescing is one call over
that cache. Trying a different intent or model costs cents and seconds instead
of a full re-analysis of the footage.

Useful flags: `--interval` (sampling rate, default 30s), `--intent` (what the
index is for), `--segmenter llm|curve`, `--min-chapter`, `--include-ocr`,
`--shard-sec` / `--workers` / `--executor` / `--no-seek` (Pass 1 execution),
`--no-cache`, `--vlm-model` / `--no-vlm`, `--no-refine`, `--no-asr`,
`--no-redact`, `--no-labels`. Curve-only: `--chapters`, `--no-dino`,
`--vlm-stride`, `--backend`.

Each run writes `<video>.events.json` — the full index, including per-event
screen descriptions, transcript, sample times, and the model's stated reason
for each boundary. Pass 1 additionally writes `<video>.observations.jsonl`.

## Recordings split across several files

One session captured across several days is several `.mp4`s. Pass them in
chronological order and they are read as one continuous timeline:

```bash
python -m cuts probe day1.mp4 day2.mp4 day3.mp4      # check the order first
python -m cuts chapters day1.mp4 day2.mp4 day3.mp4
```

```
0:00    writing parser            [day1.mp4 @ 0:00]
25:10   compiling                 [day1.mp4 @ 25:10]
45:00   compiling                 [day2.mp4 @ 0:00]
1:00:22 debugging segfault        [day2.mp4 @ 15:22]
```

Timestamps are global, but every chapter also carries the file it opens in and
its offset *inside* that file, because a global timestamp is not something you
can scrub to when the footage lives in five separate recordings.

**The files are never concatenated.** A stream-copy `ffmpeg` concat requires
every input to share a codec, resolution and pixel format, which recordings
made on different days routinely do not; the fallback is a full re-encode —
hours of CPU and tens of gigabytes of scratch disk before any analysis starts.
Reading the files in sequence costs nothing extra and keeps the mapping back to
the source.

**Each file join forces a chapter boundary.** A new file means the recording
stopped and restarted, which is a real discontinuity even when the screen looks
identical across it — and it often does, because you left the editor open. On a
three-file fixture whose second file deliberately opens on the same content the
first one ended with, forcing joins recovers 7/7 ground-truth boundaries;
`--no-join-boundaries` recovers 6/7, losing exactly that invisible join.

Forced boundaries are also exempt from `--min-chapter`: a file that ends two
seconds after its last activity change still ends there.

## How it works

Two passes. The first looks at frames and has no opinion about structure; the
second decides the structure and never looks at a frame it wasn't given.

```
Pass 1  observe    shard the timeline -> seek to each sample -> OCR
        (cached)   -> redact secrets -> describe        =>  observations.jsonl

Pass 2  coalesce   read every observation at once -> cut points + titles
                   -> validate -> refine boundaries    =>  events.json
```

**Pass 1 produces observations, not events.** An observation is one sampled
frame: a timestamp, the OCR text, and one sentence describing what was on
screen. Nothing about where anything begins or ends. Each is produced
independently — the timeline is cut into fixed-duration shards and each shard
is handled by a pure function ([cuts/observe.py](cuts/observe.py)) with no
shared state, so the executor is swappable: threads today, containers later,
same function.

What makes that possible is seeking. The old sampling loop decoded *every*
frame and discarded most of them, so its cost was linear in total video length
and, worse, linear in the video *before* the frames it wanted — a worker could
not start in the middle. `iter_frames_at_times` seeks to each sample instead.
Measured on an 87-minute 1080p capture (8.3s keyframe interval): **~0.08s per
sample whether the samples sit 5%, 50% or 95% through the file**, landing
within 0.03s of the requested time. Sampling the whole file takes ~14s against
~78s of full sequential decode.

Observations are cached to `<video>.observations.jsonl` and reused. They are
the expensive half of a run and they do not depend on how the footage is later
carved up, so Pass 2 can be re-run any number of times — different `--intent`,
different model — for the price of one API call.

**Pass 2 decides chapters by reading, not by budgeting.** All observations go
into a single call ([cuts/coalesce.py](cuts/coalesce.py)) which returns the
observation indices where a new chapter begins, plus a title for each.

Returning *cut points* rather than spans is what makes this reliable: a gap or
an overlap is not expressible, so complete coverage is structural rather than
something to check afterwards, and no timestamp is ever invented — every
boundary lands on a real sample time because the model chooses among samples
instead of typing seconds. What is still validated
(`_validate_and_repair`): ordering, range, the mandatory cut at zero, minimum
duration, and the file joins that must stay boundaries regardless.

This is the part the old segmenter structurally could not do. It found
boundary candidates at local minima of a similarity curve and deleted the
weakest until a **count budget** was met — so the *number* of chapters was
chosen in advance and then spent wherever the curve happened to dip.

Both were run over the same 174 observations from an 87-minute capture, which
holds the evidence constant and isolates the boundary strategy:

| | curve | llm |
|---|---|---|
| chapters | 22 | 15 |
| under 2 min | 13 | 0 |
| longest unbroken stretch | 21.0 min | 15.5 min |

The curve's output is not uniform — it is *clustered*. Thirteen chapters
shorter than two minutes, several bunched into the last quarter of the video,
alongside a 21-minute stretch it never cut at all. That distribution is set by
where the TF-IDF cosine wobbles, not by where the work changes.

The underlying difference is what each one is reacting to. Over one stretch
where the person is working through the same TypeScript error for six minutes
— glancing at Vercel and a chat window in between — the curve cuts three
times, because the screen contents genuinely changed. The LLM keeps it as one
`routine` chapter and says why: *"obs 156-168 repeatedly show 'two errors
found' TS1109"*. **The curve is responding to surface change; the LLM is
responding to activity change.** Which is right depends on what you want, but
only one of them can tell the difference.

Titles come from this same call, so there is no separate labeling pass — the
model that drew a boundary is the one that names it.

### `--intent`: what the index is *for*

```bash
python -m cuts chapters session.mp4 --intent "a tutorial walkthrough"
python -m cuts chapters session.mp4 --intent "find where I broke the build"
```

Coalescing judges what deserves its own chapter against this. It matters more
than it looks: "boring" is not a property of footage, it is a property of
footage relative to a purpose. Without an intent the model can only fall back
on visual and topical sameness, which is most of what the old similarity curve
was already doing.

### What the LLM segmenter gives up

Two guarantees, both previously true by construction:

* **Determinism.** The same input no longer implies the same boundaries. If
  you need to answer "did this change improve segmentation?", compare against
  `--segmenter curve`, which is kept for exactly this.
* **Chapter count and minimum length.** Enforced by validation afterwards
  rather than by the merge loop. `--chapters` is therefore ignored under
  `--segmenter llm` — the whole point is that the count comes from the
  content.

If the coalescing call fails, the run does not lose Pass 1. It falls back to
curve segmentation *over the observations already collected* — no re-decode,
no DINOv2, one labeling call.

### The older path: `--segmenter curve`

Still present, still the deterministic baseline, and still what the numbers
below were measured on.

**Stage 1 (segmentation)** implements
[arXiv 2603.00983](https://arxiv.org/abs/2603.00983), *"Event-Anchored Frame
Selection for Effective Long-Video Understanding"* (Chen, Luo, Zeng, Lin, Xie,
Chao, Ji, Zheng). Frames are sampled uniformly, each is scored by its weighted
similarity to its temporal neighbours (paper eq. 1, window `l=3`, linearly
decaying weights), boundaries are cut at local minima of that curve, and
adjacent partitions are merged by mean-feature similarity until the target
event count `M` is reached.

**Our extension: the similarity curve fuses multiple channels.** The paper uses
DINOv2 alone. DINOv2 is self-supervised on natural images, and screen
recordings are a hard distribution shift — two completely different code
screens are both "monospace text on a dark background". Text carries this
domain: what's on screen changes completely at a real activity change and
barely at all during a scroll, where DINOv2 alone can't tell the difference.
Set `weight_dino=1.0, weight_vlm=0.0` to recover the paper's exact behaviour.
Channels are z-scored before fusion because their native scales are
incomparable (DINOv2 adjacent cosine sits around 0.99 on screen capture; TF-IDF
cosine is far lower and much more spread out).

*A note on the numbers below:* the fusion approach was validated with local
OCR as the text channel (5/5 boundary recall vs. DINO-only's 3/5, on a
synthetic fixture). The text channel has since been replaced with per-frame
VLM descriptions (below) — same role in the pipeline, same fusion math, but
not yet re-measured against that fixture. Rerunning it is the natural next
validation step.

**The text channel: per-frame VLM descriptions, grounded by local OCR.**
Rather than transcribing every visible string, a Claude vision call
([cuts/signals/vlm.py](cuts/signals/vlm.py)) writes one short factual
sentence per sampled frame — active app, the single most distinguishing
detail, what the person appears to be doing. A vision model alone reads small
downscaled screen text unreliably, so local OCR ([cuts/signals/ocr.py](cuts/signals/ocr.py))
runs on the same frame at full resolution first and its text is passed into
the same call as grounding — exact strings for the model to synthesize
around, not a transcript for it to read. OCR here is not an independent
channel or a segmentation signal; it exists only to serve this one call, and
is cheap again now that sampling is coarse (it was removed earlier specifically
because it was expensive at the old 1fps rate). This moves the pipeline's
dominant cost off local compute and onto the API, and the evidence stored per
event is a one-sentence description, not a transcript to skim past. There is
no fallback to a plain-OCR channel: without `ANTHROPIC_API_KEY` this channel
is simply skipped, and segmentation runs on the visual channel alone.

**Descriptions are batched, and each one must stand alone.** Six consecutive
frames go in a single call (`VLMConfig.frames_per_call`), which sends the
system prompt once instead of once per frame and lets the model use a frame's
neighbours to work out what it is looking at. It is explicitly forbidden from
*referring* to them: no "same as previous", no "still" — always re-name the
app, file, and activity, even when that repeats the previous description word
for word. This is not stylistic. Segmentation compares consecutive
descriptions by TF-IDF cosine, so "same as the last frame" shares almost no
tokens with a description that names the file and reads as a large change,
manufacturing a boundary that isn't there. Two near-identical descriptions
are the correct output for two unchanged frames — and batching makes them
*more* likely, since a model describing frames together phrases an unchanged
screen consistently, where independent calls would word it differently each
time and invent dissimilarity.

**Credentials are blacked out before any frame is uploaded.** A screen
recording of real work eventually catches a `.env` file, a terminal echoing a
token, or a dashboard showing a live key — and every sampled frame goes to an
external API. [cuts/redact.py](cuts/redact.py) sits in front of all three
upload paths (sampling, refinement, labeling). It reuses the OCR pass that
already runs for grounding: the engine returns a bounding box with every line
it reads, so a line matching a known credential shape is filled solid black in
the frame *and* replaced with `[REDACTED]` in the grounding text — scrubbing
only the image would ship the key as a string instead of as pixels. The frame
is mutated in place, so the redaction also survives into the local visual
embedding. When a credential is found but cannot be tied to a box (split
across two OCR lines), there is nothing to paint over and the frame is
withheld from upload entirely; the sampling pass carries the previous
description forward, so segmentation sees continuity rather than a gap. Hits
are reported by pattern name (`openai_key`) and never by content — a run that
finds something prints `[redact] blacked out credentials in 3 frame(s)`,
which is also your cue to go rotate it.

**What redaction does not do.** It catches what OCR actually read. Text too
small, too low-contrast, or in a font the engine mangles is not seen here and
is not redacted. Patterns anchor on distinctive prefixes (`sk-`, `AKIA`,
`ghp_`) and stay loose about the body precisely because OCR corrupts
characters — in testing it read `OPENAI_API_KEY` as `OPENAl_APl_KEY` and still
matched — but that only helps where the prefix survived. Assignment patterns
(`DATABASE_PASSWORD=`, `MY_API_KEY=`) cover the `.env` case, at the cost of
occasionally blacking out an uppercase constant like `CACHE_KEY = "profile"`;
over-redaction costs one line, under-redaction costs a key. **This reduces
exposure. It does not guarantee it, and it is not a substitute for keeping
credentials off screen while recording.** `--no-redact` turns it off for
footage you know is clean. Disabling OCR disables redaction with it, and the
pipeline warns when you do.

**Boundary refinement narrows the coarse grid, one level.**
[cuts/refine.py](cuts/refine.py) sharpens each detected boundary after
segmentation: a 30s sampling interval means a boundary's true timestamp is
only known to within ±15s, so refinement decodes the single frame at the
midpoint, describes it, and asks a small classification call which half of
the gap the transition actually falls in. This is one bisection, not a
recursive search — the interval, not the chapter target, is the knob for
gap-closing precision beyond that.

**Labeling** sends every event's evidence to Claude in a single call, so titles
stay distinct and consistently phrased across the video. Labeling each event
separately reliably produces near-duplicate titles for adjacent events.

**Stages 2-3 of the paper** (query-driven anchor selection + adaptive MMR) are
implemented in [cuts/select.py](cuts/select.py) but are *not* used by chapter
generation, which is query-free. They are the engine for the future "find the
moment matching this description" workflow.

## Layout

| Module | Role |
|--------|------|
| [cuts/media.py](cuts/media.py) | VFR-safe decoding, seek-based + sequential sampling, multi-file timelines |
| [cuts/observe.py](cuts/observe.py) | **Pass 1** — sharded independent observations, cache |
| [cuts/coalesce.py](cuts/coalesce.py) | **Pass 2** — cut points and titles, with validation |
| [cuts/signals/vlm.py](cuts/signals/vlm.py) | Per-frame screen descriptions |
| [cuts/signals/ocr.py](cuts/signals/ocr.py) | Local OCR (grounds vlm.py's call; not an independent channel) |
| [cuts/redact.py](cuts/redact.py) | Blacks out credentials in frames before upload |
| [cuts/signals/asr.py](cuts/signals/asr.py) | Whisper transcript (speech channel) |
| [cuts/refine.py](cuts/refine.py) | Single-bisection boundary sharpening |
| [cuts/signals/dino.py](cuts/signals/dino.py) | DINOv2 embeddings — `--segmenter curve` only |
| [cuts/signals/text_features.py](cuts/signals/text_features.py) | TF-IDF vectors — `--segmenter curve` only |
| [cuts/segmentation.py](cuts/segmentation.py) | EFS Stage 1 — `--segmenter curve` only |
| [cuts/labeling.py](cuts/labeling.py) | Chapter titles — curve path and standalone `cuts label` |
| [cuts/select.py](cuts/select.py) | EFS Stages 2-3, query-driven (unused today) |
| [cuts/pipeline.py](cuts/pipeline.py) | Orchestrator |
| [cuts/cli.py](cuts/cli.py) | CLI |

Every module has a `python -m cuts.<module> <video>` debug entry point that
runs and reports on that stage alone.

All tuning constants live in [cuts/config.py](cuts/config.py) as one
`CutsConfig` dataclass. No module body hardcodes a threshold.

## Performance and limits

The cost profile changed with the move from local OCR to VLM descriptions, and
has not yet been carefully re-benchmarked at scale — treat everything in this
section as an order-of-magnitude estimate (`cuts probe` prints one per input),
not a measurement.

- **Decode no longer grows with video length** under `--segmenter llm`.
  Sampling seeks to each wanted frame rather than decode-parsing every frame
  and discarding most, which was previously the floor under how fast any run
  could go. Measured flat at ~0.08s per sample at 5%, 50% and 95% through an
  87-minute capture; whole-file sampling ~14s versus ~78s of sequential
  decode. `--segmenter curve` still decodes sequentially and still has the
  old floor. `--no-seek` forces the old behaviour if a container's seeking
  turns out to be unreliable.
- **Sharding does not buy much *local* parallelism, and that is measured
  rather than assumed.** RapidOCR's ONNX session already uses every core, so
  OCR is saturating the machine before shards are split at all. On 8 cores:
  serial 34.4s, 8 threads 24.3s, 4 *processes* 44.1s — processes come out
  slower than serial because they oversubscribe a runtime that was already
  parallel. Threads (the default) recover ~1.4x by overlapping decode and API
  latency with OCR, and that is the honest ceiling on one machine. The
  single-machine speedup in this design comes from seeking, not from
  sharding; sharding pays off when workers are separate machines, and in the
  meantime it pays for itself through the observation cache.
- **The VLM channel is network-latency-bound**, not compute-bound. Calls are
  submitted *without waiting*, so decoding (CPU) overlaps description calls
  (network) instead of the two alternating; `VLMConfig.concurrency` (default
  6) bounds calls in flight and `max_inflight_chunks` bounds how far decode
  may run ahead, which is what keeps queued frames from growing without limit
  on a long input. The OCR grounding step runs synchronously in the decode
  loop since it's local compute — cheap at today's sample counts, but worth
  watching if `--interval` is lowered a lot.
- **Boundary refinement is parallel too**: it gathers every midpoint frame
  first (local decode + OCR), then describes them in one batched concurrent
  pass and classifies every gap concurrently, rather than looping one
  boundary at a time.
- **Extrapolating to 30 hours** is untested at that scale in the new
  architecture; the numbers this section used to cite were measured under the
  local-OCR pipeline and no longer apply.

- **Pass 2 is one call, and its size is dominated by OCR if you let it.**
  Descriptions run ~33 tokens per observation; OCR runs ~250. That is why
  `--include-ocr` is off by default — it is roughly 8x the rest of the prompt
  and, on a multi-hour timeline, most of the cost. Past
  `CoalesceConfig.max_observations` (1500) the run warns rather than silently
  degrading; the fix is a coarser `--interval`, and windowing the timeline is
  the unbuilt alternative.

Memory is bounded regardless of length — observations hold text, never pixels,
and each shard discards its frames as it goes.

## What this is not

- No training. Inference only.
- No offline / no-credentials mode. The text channel and chapter titles both
  call the Claude API; without `ANTHROPIC_API_KEY` you get visual-only
  segmentation and heuristic titles, not full offline operation.
- No shot-boundary detection — it was removed; it does not describe what
  changes in a screen recording.
- No web UI yet.
- **No agent layer yet, but the hinge for it exists.** `observe_shard` takes
  an explicit list of sample times rather than deriving them from an
  interval, so a `zoom(t0, t1, n)` tool is the same function with a different
  time list — and closely-spaced requests already decode one keyframe group
  instead of one per frame. `get_ocr(t0, t1)` is a read against the cached
  observations. Neither is wired up; Pass 2 is a single call today, not a
  tool loop.
- **No footage cutting.** Each chapter records a `kind`
  (`pivotal | active | routine | idle`) because a cutting pass would need a
  priority signal and the model that read the evidence is the cheapest place
  to get one. Nothing consumes it yet.

## Status

Segmentation is validated against two synthetic fixtures with known ground
truth, both under the previous local-OCR text channel:

- single file, 6 activities — 6/6 boundaries exact, at `--chapters 6
  --min-chapter 15`. At the default `--min-chapter 20` the last boundary is
  suppressed (the final activity is 18s long) and you get 5. The floor is
  doing what it is supposed to; the fixture is just shorter than the settings
  assume.
- three files, 7 activities, one join deliberately invisible to every content
  signal — 7/7 with forced join boundaries, 6/7 without.

The two-pass architecture is verified as follows:

- **Seeking is exact on real footage.** Across two captures (9 min and 87
  min), every sample landed within 0.033s of its requested time, and cost was
  flat with depth into the file. This gated the whole design.
- **Shards are genuinely independent.** Serial, threaded and process
  executors produce byte-identical OCR text over the same plan, and repeated
  runs agree. If this did not hold, "containerizable" would be a claim rather
  than a property.
- **Coalescing survives bad model output.** `_validate_and_repair` is tested
  against unordered cuts, duplicates, out-of-range and negative indices,
  missing/`None`/string `start` values, a missing opening cut, 40 proposals
  below the minimum duration, dropped file joins, and entirely empty input.
  Every case yields a contiguous gapless partition covering all observations.
- **The curve segmenter did not regress.** `--segmenter curve` reproduces
  boundaries byte-identically against a pristine checkout at matched
  intervals, so the deterministic baseline still means something.

Credential redaction is verified locally, without API calls, by intercepting
the encoded bytes each upload path would actually send and re-OCRing them:

- all four upload paths (observation, sampling, refinement, labeling) emit
  **zero** credential matches on a fixture with a planted `.env` scene, where
  the same runs with `--no-redact` leak on 6, 6, 2, and 4 frames
  respectively — and for the observation path, in the grounding text too;
- boxes map back correctly through both the OCR upscale and ROI crops, so
  redaction lands on the secret and not on neighbouring code;
- ordinary code (`token = tokens[i]`, `password: str`, git SHAs, UUIDs,
  `get_secret_name()`) produces no false positives;
- an OCR engine failure returns *unsafe* rather than silently-clean, so a
  frame nothing could vouch for is withheld instead of uploaded.

Both passes run end to end on real footage. On an 87-minute 1080p capture:
Pass 1 produced 174 observations in 8.0 min (5.5s per video-minute, all 174
described); Pass 2 turned them into 15 chapters in one call from a ~6.3k-token
prompt. Chapter lengths ranged from 1.5 to 15.5 minutes, and the long
repetitive stretches were coalesced rather than subdivided — see the table
above for the head-to-head against `--segmenter curve`.

Not yet verified:

- **Coalescing quality, past a handful of runs.** The validation layer is
  well tested; the *judgement* is not. There is no ground-truth fixture for
  "should these twenty minutes be one chapter?", and the synthetic fixtures
  cannot supply one — they were built to have unambiguous boundaries, which
  is the opposite of the case this pass exists to handle. Building a fixture
  with a deliberately long, deliberately boring stretch is the obvious next
  step.
- **Whether `--intent` changes the output as much as it should.** It is
  threaded into the prompt and stated first, but the claim that it materially
  reshapes coalescing is currently reasoning, not measurement. Run the same
  cached observations under two opposed intents and diff.
- **The VLM text channel and boundary refinement against a ground-truth
  fixture.** The boundary-recall numbers above predate both and were measured
  with local OCR as the text channel.
- **Redaction against real footage.** The fixture plants credentials in a
  clean, high-contrast, synthetic render. Real captures are noisier, and
  recall depends entirely on whether OCR reads the credential at all.
- **Multi-file timelines under the LLM segmenter.** Joins are locked as
  boundaries and the code path is shared, but the real multi-day case has
  only been exercised under the curve segmenter.
- **Anything past ~90 minutes.** An 87-minute real capture runs end to end;
  multi-hour inputs are extrapolation, and past ~1500 observations the
  single-call Pass 2 is a warning rather than a supported mode.
