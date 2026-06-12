Legacy Code
===========

Purpose
-------
This folder preserves retired implementations and experiments for reference.
Nothing here is imported by the active pipeline unless explicitly restored.

Current Contents
----------------
handwriting_detector.py:
    Older handwriting detection flow.

N00_file_preparation/:
    Retired broad region detection, structural line detection, row splitting,
    and field splitting utilities from the original preparation flow.

N02_crop_refiner/:
    Previous mixed-color Surgeon implementation with text-unit grouping, border
    profiles, 4x4 quality scoring, and debug helpers.

Important
---------
The active N02 is layer-aware and intentionally being rebuilt from a smaller
foundation. Do not copy the old mixed-color N02 back into runtime wholesale.
