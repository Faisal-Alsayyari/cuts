"""Per-frame signal extractors.

Each module here turns sampled frames (or the audio track) into one kind of
evidence about what is happening on screen:

    vlm.py   -> one-sentence screen descriptions (Claude vision; the evidence
                Pass 2 reads, and the curve segmenter's text channel)
    ocr.py   -> (not a channel) local OCR that grounds vlm.py's call with exact
                strings, and gives cuts.redact the boxes it paints over
    dino.py  -> visual embeddings (DINOv2; curve segmenter only)
    asr.py   -> spoken transcript (faster-whisper; optional, audio permitting)
    text_features.py -> TF-IDF vectors for the curve segmenter's text channels
"""
