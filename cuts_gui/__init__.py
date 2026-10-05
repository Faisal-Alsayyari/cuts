"""cuts_gui — a desktop front end for `cuts`.

    python -m cuts_gui

The GUI never imports the `cuts` package. Everything it knows about the
backend — which command runs an analysis, which flags it takes, where the
index lands, what the index looks like on disk, what a progress line means —
lives in one module, `cuts_gui.backend`. Analysis runs as a `python -m cuts`
subprocess and its results are read back from the files it writes.

That boundary is the point of the design. The backend is still moving
(optimizations, an agent layer), and the things it is most likely to change —
pipeline internals, module layout, the Python API — are exactly the things the
GUI does not touch. What the GUI depends on is the CLI and the on-disk index
format, which are the backend's public surface anyway, and a change to either
is a change to `backend.py` alone. Everything else here speaks the GUI's own
vocabulary (`cuts_gui.model`), not the backend's.

Running it out of process has two further benefits that are not about
decoupling: the heavy runtime (torch, ONNX, Whisper) never loads into the UI
process, and cancelling a run is killing a process rather than trying to
interrupt a thread pool mid-API-call.
"""
