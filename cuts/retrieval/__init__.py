"""Event-retrieval milestone for `cuts`.

Turns a long raw video (typically a devlog / screen recording) into a
timestamped, searchable index on top of TransNetV2 shot boundaries.

Pipeline:
    TransNetV2 shots
      -> segmenter (hybrid shot + time cap)
      -> representative frame sampler
      -> OCR (+ optional ASR)
      -> index (BM25 + optional text/image embeddings)
      -> search (RRF hybrid, returns timestamped SegmentRecords)

The design goals live in the milestone plan:
    - grounded timestamps on every result (pts * time_base, never frame/fps),
    - cheap-first signals (OCR on a few sampled frames, ASR optional),
    - explainable retrieval (BM25 evidence snippets alongside embeddings),
    - no training, no dense per-frame work.
"""

from .schema import SegmentRecord

__all__ = ["SegmentRecord"]
