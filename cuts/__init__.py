"""cuts — frame-accurate shot boundary and UI state change event detection pipeline.

Public API surface is intentionally minimal; users should compose modules directly
or invoke the orchestrator in `cuts.pipeline`. See README for usage.
"""

from cuts.config import CutsConfig  # re-export the single source-of-truth config

__all__ = ["CutsConfig"]
