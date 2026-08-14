"""Per-frame signal extractors.

Each module here turns sampled frames into one *channel* of evidence about
what is happening on screen at a given time:

    dino.py  -> visual embeddings   (DINOv2; the EFS paper's channel)
    ocr.py   -> on-screen text      (rapidocr; usually dominant for screen capture)
    asr.py   -> spoken transcript   (faster-whisper; optional, audio permitting)

`cuts.segmentation` fuses whichever channels are available into a single
temporal similarity curve, so any one of them can be disabled without the
rest of the pipeline changing shape.
"""
