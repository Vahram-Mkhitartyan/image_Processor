Node N04: Printed OCR
=====================

Purpose
-------
N04 consumes N03 routes classified as printed_only or mixed, prepares crops for
Tesseract, and stores raw Armenian printed OCR candidates. It does not decide
final reconstructed truth.

Commands
--------

    .venv/bin/python scripts/main.py printed_ocr
    .venv/bin/python scripts/main.py printed
    .venv/bin/python scripts/main.py n04

Main Entry
----------

    printed_ocr.py
    build_printed_text_map(visual_routes_path, output_dir, settings_path=None)

Current Flow
------------
1. Load N03 route metadata.
2. Select printed_only and mixed routes.
3. Reference the canonical N02 full-text crop without duplicating it.
4. Normalize to dark ink on white, then scale and pad a Tesseract-ready crop.
5. Run primary and fallback Armenian OCR configurations.
6. Store raw candidates, coordinates, skips, failures, and summary metadata.

OCR Engine
----------

    Tesseract 5.5.2
    primary language: hye-calfa-n
    fallback language: hye
    default page segmentation mode: 6

Crop selection priority:

    classification_crop_path
    routed_crop_path
    analysis_crop_path
    refined_crop_path
    original_crop_path
    source_crop_path

analysis_mask_crop_path is intentionally excluded because it is the inverted
white-ink-on-black topology mask for ScribeTrace. N04 starts from the regular
full-text visual crop, converts it to grayscale, enforces dark ink on a white
background, and uses normal Otsu binary thresholding.

Files
-----
printed_ocr.py:
    Public orchestrator.

n04_constants.py:
    Node identity, Tesseract version, languages, and route classes.

n04_routing.py:
    N03 route selection, bbox extraction, and crop selection.

n04_crops.py:
    Crop copying and Tesseract preparation.

n04_ocr_engine.py:
    Tesseract subprocess execution and raw candidate construction.

n04_records.py:
    Printed text-unit and summary records.

n04_io.py:
    JSON and output-folder helpers.

Full-Document Calfa Probe
-------------------------
test_calfa_full_document.py runs the locally installed hye-calfa-n Tesseract
language model against one complete, unprocessed document. It saves plain text
plus a JSON report containing word boxes and confidence statistics.

Run the default test_1 probe:

    .venv/bin/python scripts/N04_printed_ocr/test_calfa_full_document.py

This is a local Tesseract baseline only. It is not equivalent to Calfa's online
document-layout and AI service.

Output Folder
-------------

    temp_processing/<document_id>/n04_printed_ocr/
        tesseract_ready/printed_only/
        tesseract_ready/mixed/
        metadata/<document_id>_printed_text_map.json
        debug/
        full_document_test/

Output candidates use trusted_as_final=false. A later reconstruction node should
validate labels, schema expectations, language rules, and confidence evidence.
