"""Stage 1 detectors and the ensemble that fuses them.

Each detector returns a list of `BoundaryCandidate` objects (see
`cuts.detectors.ensemble`). The ensemble merges nearby candidates from
different detectors into a single deduplicated, sorted list.
"""

from cuts.detectors.ensemble import BoundaryCandidate, merge_candidates

__all__ = ["BoundaryCandidate", "merge_candidates"]
