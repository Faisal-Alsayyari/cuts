"""cuts — semantic indexing and temporal navigation of long-form video.

Turns long screen recordings into a titled chapter list. Run it with
`python -m cuts chapters <video...>`; `cuts.pipeline.run` is the same thing
as a function. See README for usage.
"""

from cuts.config import CutsConfig  # re-export the single source-of-truth config

__all__ = ["CutsConfig"]
