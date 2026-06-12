N02 Crop Refiner
================

Purpose
-------
N02 consumes N01/ScribeMap group metadata and rebuilds OCR-ready text units.
It groups split fragments conservatively, saves refined crops, scores crop quality,
and writes debug metadata for inspection.

Current flow
------------
1. crop_refiner.py orchestrates the node through CropRefiner.refine_document().
2. n02_normalization.py normalizes N01 group records into bbox-based records.
3. n02_text_units.py builds line buckets and conservative same-line text units.
4. n02_border_profiles.py attaches split cursive fragments using border ink profiles.
5. n02_quality.py scores each final crop with a 4x4 ink-density audit.
6. n02_records.py builds final refined group records for N03.
7. n02_debug_preview.py renders labeled boxes for visual debugging.

Files
-----
- crop_refiner.py: public orchestrator and CropRefiner class.
- n02_settings.py: RefinerSettings dataclass and settings loading.
- n02_io.py: JSON, image, crop-saving, and output-path helpers.
- n02_geometry.py: bbox math helpers.
- n02_normalization.py: N01 input normalization.
- n02_text_units.py: line bucketing and text-unit grouping.
- n02_border_profiles.py: Armenian cursive split-fragment attachment rules.
- n02_quality.py: 4x4 crop-quality scoring.
- n02_records.py: output record builders.
- n02_debug_preview.py: debug preview renderer.
- settings.json: node defaults.

Output
------
When run through the main pipeline, N02 writes to:

    temp_processing/<document_id>/n02_crop_refiner/

N02 outputs include:
- temp_processing/<document_id>/n02_crop_refiner/metadata/<document_id>_refined_groups.json
- temp_processing/<document_id>/n02_crop_refiner/refined_crops/accepted/
- temp_processing/<document_id>/n02_crop_refiner/refined_crops/review/
- temp_processing/<document_id>/n02_crop_refiner/refined_crops/rejected/
- temp_processing/<document_id>/n02_crop_refiner/debug/<document_id>_n02_refined_boxes_preview.jpeg

Notes
-----
N02 does not run OCR and does not run the visual classifier. It currently avoids
final crop rejection where possible so downstream classification can still inspect
questionable crops.
