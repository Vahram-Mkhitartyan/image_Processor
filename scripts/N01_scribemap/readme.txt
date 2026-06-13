Node N01: ScribeMap
===================

Purpose
-------
N01 detects and groups text-like connected regions from N00 masks. It is a
geometry node, not an ML classifier, and it does not create the masks it reads.

Main Entry
----------

    scribemap_detector.py
    ScribeMapBWDetector
    run_from_preparation_state(preparation_state, output_dir)

Main Files
----------
scribemap_detector.py:
    Orchestrates component extraction, grouping, per-layer runs, metadata, crops,
    and previews.

scribemap_components.py:
    Connected-component extraction.

scribemap_grouping.py:
    Spatial grouping and rejection of tiny, square, line-like, or oversized
    candidates.

scribemap_crops.py:
    Optional component/group crop saving.

scribemap_preview.py:
    Component, accepted-group, rejected-group, mask, and per-layer previews.

scribemap_io.py:
    Image, folder, and JSON helpers.

Input Contract
--------------
N01 receives the complete N00 state. It uses the prepared image as its coordinate
space and runs ScribeMap independently on these semantic masks:

    blue_ink_mask
    red_ink_mask
    green_ink_mask
    unknown_color_ink_mask
    black_ink_mask

The legacy content_ink_mask run is retained for compatibility/debugging, but the
active pipeline bridge uses flattened groups from the real color layers.

N01 also carries N00 blue_continuity_mask and red_continuity_mask paths in
continuity_mask_paths. These masks are not used to classify or group N01
regions. They are forwarded so N02 can build topology-safe ScribeTrace crops
without changing semantic color ownership.

Layer Meaning
-------------

    blue          probable handwriting
    red           probable correction or markup
    green         probable colored handwriting
    unknown_color ambiguous colored ink
    black         printed text, dark handwriting, or form structure

Core Flow
---------
1. Read N00 prepared image and masks.
2. Detect connected components for each layer.
3. Build candidate groups using ScribeMap geometry.
4. Reject obvious artifacts.
5. Save group metadata and document-level previews.
6. Flatten active layer groups into the neutral N02 bridge contract.

N01 no longer materializes legacy per-group JPEGs during normal pipeline runs.
N02 owns the canonical full-text crop for every active color-layer group.

Output Folder
-------------

    temp_processing/<document_id>/n01_scribemap/
        components/
        groups/
        debug/
        metadata/

Full ScribeMap metadata:

    metadata/scribemap_from_prepared_masks.json

N02 bridge metadata:

    metadata/<document_id>_classified_groups.json

The bridge filename is historical. Its records remain unclassified:

    label: unclassified
    confidence: null
    classification_method: not_run

Current Four-Document Baseline
------------------------------
After the June 2026 color tuning, current total active groups are approximately:

    test_1: 102
    test_2: 97
    test_3: 76
    test_4: 65

These values are diagnostics, not permanent requirements.

When To Edit N01
----------------
Edit N01 when components or boxes are too fragmented, too large, over-merged, or
artifact-heavy. Edit N00 instead when the underlying masks contain the wrong
pixels.
