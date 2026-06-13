Node N03: Visual Classification Router
======================================

Purpose
-------
N03 runs Minos v2.0 on N02 classification crops and routes each crop by visual
content. It does not run OCR.

Commands
--------

    .venv/bin/python scripts/main.py visual
    .venv/bin/python scripts/main.py visual_classification
    .venv/bin/python scripts/main.py n03

Main Entry
----------

    classifier.py
    classify_document(refined_groups_path, output_dir, model_path, settings_path)

Model
-----

    models/minos_v2_0_best.keras

Minos returns three sigmoid scores in this order:

    printed_present
    handwriting_present
    noise

Final Visual Classes
--------------------

    mixed
    printed_only
    handwriting_only
    empty_or_noise
    review

Routes
------

    mixed            -> printed_ocr + handwritten_ocr
    printed_only      -> printed_ocr
    handwriting_only  -> handwritten_ocr
    empty_or_noise    -> archive
    review            -> review

Input Crop Selection
--------------------
N03 selects the first existing crop in this order:

    classification_crop_path
    analysis_crop_path
    refined_crop_path
    context_crop_path
    original_crop_path

The binary analysis_mask_crop_path is never used by Minos.

By default N03 does not copy crops into classified class folders. routed_crop_path
references the existing N02 full-text crop. Set save_classified_copies=true only
for an intentional review/export run.

N02 Policy Handling
-------------------
Groups with minos_required=false are intentionally skipped. Current red
correction/markup records use this path. Rejected records are skipped unless
include_rejected is enabled. Missing crop paths are also recorded as skips.

One explicit bypass exists for N02 correction pairing:

    force_handwritten_ocr=true

These known red replacement-text records do not run Minos. N03 emits them into
the normal routes contract as handwriting_only with routing_method set to
n02_red_correction_pairing. Crossed blue originals remain in skipped metadata
as suppressed_replaced_text evidence.

Settings
--------
visual_classification_settings.json controls:

    printed_threshold
    handwriting_threshold
    noise_threshold
    mixed_handwriting_safety_threshold
    include_rejected
    save_classified_copies
    reset_output

handwriting_threshold is the canonical key. The accidental legacy spelling
handwritten_threshold is accepted as an alias and normalized internally.
Threshold overrides are merged with defaults, so partial settings remain valid.

The mixed safety threshold intentionally prefers false mixed over missed mixed
when printed evidence is strong and handwriting evidence is borderline.

Output Folder
-------------

    temp_processing/<document_id>/n03_visual_classification/
        classified/              empty unless copies are explicitly enabled
        metadata/<document_id>_n03_visual_classification_routes.json
        debug/

Metadata contains successful routes plus explicit skipped and failed records.
CUDA warnings on machines without NVIDIA drivers are TensorFlow CPU fallback
messages; they are not crop failures.
