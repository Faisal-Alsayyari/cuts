"""Arm A — PySceneDetect (AdaptiveDetector + ContentDetector).

We run BOTH detectors and contribute their boundaries as separate sources to
the ensemble:

* AdaptiveDetector adapts its threshold to local content; it is the better of
  the two for screen recordings and handles gradual transitions more
  gracefully. We treat it as the primary signal.
* ContentDetector is more aggressive at fixed thresholds; we include it for
  recall (catches boundaries Adaptive misses) at the cost of false positives,
  which Stage 2 refinement is responsible for filtering.

Both produce SCENE intervals — i.e. `(start_timecode, end_timecode)` per scene.
A "boundary" is the *start* of every scene after the first (the start of scene
0 is the start of the video, not a cut). We convert to ordinal frame indices
via `timecode.get_frames()`.
"""

from __future__ import annotations

from typing import List

from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector, ContentDetector

from cuts.config import PySceneDetectConfig
from cuts.detectors.ensemble import BoundaryCandidate


def detect_with_adaptive(
    video_path: str, config: PySceneDetectConfig
) -> List[BoundaryCandidate]:
    """Run AdaptiveDetector and return boundary candidates."""
    return _run_detector(
        video_path,
        AdaptiveDetector(adaptive_threshold=config.adaptive_threshold),
        source_label="pyscenedetect_adaptive",
        show_progress=config.show_progress,
    )


def detect_with_content(
    video_path: str, config: PySceneDetectConfig
) -> List[BoundaryCandidate]:
    """Run ContentDetector and return boundary candidates."""
    return _run_detector(
        video_path,
        ContentDetector(threshold=config.content_threshold),
        source_label="pyscenedetect_content",
        show_progress=config.show_progress,
    )


def detect_all(
    video_path: str, config: PySceneDetectConfig
) -> List[BoundaryCandidate]:
    """Convenience: run both PySceneDetect detectors and return their union.

    The ensemble layer (`cuts.detectors.ensemble.merge_candidates`) is
    responsible for deduplicating across nearby boundaries — we just emit
    everything here.
    """
    return detect_with_adaptive(video_path, config) + detect_with_content(video_path, config)


def _run_detector(
    video_path: str, detector, source_label: str, show_progress: bool
) -> List[BoundaryCandidate]:
    """Shared plumbing: open video, attach a detector, return boundaries.

    PySceneDetect emits SCENES (intervals); the boundaries we care about are
    the START frames of every scene EXCEPT the first one (which is just the
    beginning of the video and not a cut).
    """
    video = open_video(video_path)
    manager = SceneManager()
    manager.add_detector(detector)
    manager.detect_scenes(video, show_progress=show_progress)
    scene_list = manager.get_scene_list()

    candidates: List[BoundaryCandidate] = []
    # Skip index 0 deliberately — its start is the video's t=0, not a cut.
    for start_tc, _end_tc in scene_list[1:]:
        # `get_frames()` returns the 0-based frame index of this timecode.
        # PySceneDetect's frame indexing matches PyAV's decoding-order index
        # for CFR content; for VFR content there can be off-by-a-few drift,
        # which Stage 2 refinement (operating in a ±W window) absorbs.
        frame_idx = int(start_tc.get_frames())
        candidates.append(BoundaryCandidate(frame_idx=frame_idx, sources=[source_label]))
    return candidates


if __name__ == "__main__":
    # Standalone debug: print boundaries from both detectors on a video.
    import sys

    from cuts.config import CutsConfig

    if len(sys.argv) < 2:
        print("usage: python -m cuts.detectors.pyscenedetect_detector <video_path>")
        sys.exit(1)
    cfg = CutsConfig().pyscenedetect
    path = sys.argv[1]
    a = detect_with_adaptive(path, cfg)
    c = detect_with_content(path, cfg)
    print(f"AdaptiveDetector: {len(a)} boundaries")
    for b in a[:20]:
        print(f"  frame {b.frame_idx}")
    print(f"ContentDetector:  {len(c)} boundaries")
    for b in c[:20]:
        print(f"  frame {b.frame_idx}")
