"""Per-frame signal extractors.

Each module here turns sampled frames into one *channel* of evidence about
what is happening on screen at a given time:

    dino.py  -> visual embeddings   (DINOv2; the EFS paper's channel)
    vlm.py   -> on-screen text      (Claude vision description; usually dominant for screen capture)
    ocr.py   -> (not a channel) local OCR, grounds vlm.py's description call
                with exact strings a downscaled image can't reliably carry
    asr.py   -> spoken transcript   (faster-whisper; optional, audio permitting)

`cuts.segmentation` fuses whichever channels are available into a single
temporal similarity curve, so any one of them can be disabled without the
rest of the pipeline changing shape.
"""
