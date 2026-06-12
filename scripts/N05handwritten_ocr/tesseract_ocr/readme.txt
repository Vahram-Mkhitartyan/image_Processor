Tesseract OCR Expert
====================

Purpose
-------
This expert will provide a conventional OCR opinion for an N05 text-unit crop.
It is intentionally disabled until language configuration, preprocessing, and
confidence normalization are implemented.

Interface
---------
    expert.py
    get_expert_manifest(settings=None)
    recognize(crop_path, context=None, settings=None)
