Node N02: Layer-Aware Crop Preparation
======================================

Purpose
-------
N02 consumes N01 color-layer groups and creates explicit crop views for Minos,
OCR, correction analysis, and ScribeTrace processing.

Despite the historical folder name, the current implementation is not yet the
rebuilt grouping Surgeon. It currently preserves each N01 source group, applies
small bbox padding, assigns a layer policy, and generates crop artifacts.

Main Entry
----------

    crop_refiner.py
    CropRefiner.refine_document(classified_groups_json_path, output_path=None)

Current Flow
------------
1. Load the N01 bridge JSON and full ScribeMap result.
2. Collect configured blue, red, green, unknown-color, and black groups.
3. Detect accepted red groups that credibly cross blue text.
4. Suppress crossed blue groups and promote nearby red replacement writing.
5. Split obvious stacked multi-line groups before crop generation.
6. Preserve layer identity and original source-group ids.
7. Apply a small padded bbox.
8. Create one canonical full-text crop and one topology mask per group.
9. Write refined group metadata for N03 and future nodes.

Crop Artifacts
--------------
full_text:
    The single target-layer visual crop used by Minos, OCR, N05, and the UI.
    analysis_crop_path, classification_crop_path, and refined_crop_path all
    reference this same physical file for backward compatibility.

analysis_mask:
    Binary white-ink-on-black topology crop. Blue and red use N00 continuity
    masks when available, which borrow only geometrically supported crossing
    pixels from the other color. Other layers use their exact semantic masks.
    It is not dilated and is never a Minos input.

The visual crop remains target-layer-only. The continuity repair affects only
analysis_mask_crop_path for topology consumers. Original/context duplicate
views are no longer materialized during normal runs.

Layer Policies
--------------
blue/green:
    Probable handwriting. Minos is used as an audit/router.

blue crossed by red:
    Preserved as deleted-original evidence, but minos_required=false and
    is_final_text_candidate=false. It does not enter Minos or OCR.

black:
    Unknown dark-ink role. Minos is the primary router.

red:
    Correction/markup evidence. Minos is intentionally skipped, the record is
    preserved, and the default destination remains N06 correction resolver.
    A text-shaped red group paired with crossed blue text is instead marked
    force_handwritten_ocr=true and routed directly to N05.

Correction pairing uses actual red pixels from accepted N01 red groups. It
requires a configurable horizontal or vertical span across the blue bbox, then
selects up to two nearby red groups large enough to contain writing. Tiny red
specks and rejected/unmapped red pixels cannot suppress blue text.

Stacked Text Guard
------------------
Some N01 groups can accidentally contain two handwritten words stacked above
each other. N02 now checks suspicious wide/tall groups before crop generation.

The guard first looks for truly separated horizontal ink bands. If the bands
are connected by tails or stray pixels, it can use a conservative projection
valley split: a horizontal trough is accepted only when there is enough real
ink above and below it.

When a split happens, N02 replaces the parent source group with child records:

    <parent_source_group_id>_line01
    <parent_source_group_id>_line02

The metadata records `stacked_text_split.events` so every split is inspectable.
This is intentionally conservative; it should prevent obvious two-row crops
without becoming a general handwriting segmenter.

unknown_color:
    Ambiguous color. Minos performs fallback routing.

Output Folder
-------------

    temp_processing/<document_id>/n02_crop_refiner/
        metadata/<document_id>_refined_groups.json
        crops/<layer>/full_text/
        crops/<layer>/analysis_mask/

Important Record Fields
-----------------------

    text_unit_id
    source_group_id
    source_layer_group_id
    layer
    bbox
    final_bbox
    full_text_crop_path
    classification_crop_path
    analysis_crop_path
    context_crop_path
    analysis_mask_crop_path
    refined_crop_path
    mask_source
    semantic_mask_source
    analysis_mask_policy
    visual_layer_source
    minos_required
    minos_mode
    recommended_next_node
    preserve_as_evidence
    force_handwritten_ocr
    correction_role
    crossing_red_source_group_ids
    replacement_red_source_group_ids
    replaces_blue_source_group_ids
    correction_evidence
    refiner

Settings
--------
The current CropRefiner actively uses:

    input_mode / legacy alias input_group_mode
    layers_to_refine / legacy alias scribemap_2_layers_to_refine
    crop_padding_px
    debug_preview_enabled
    correction_routing_enabled
    correction_min_crossing_ink_pixels
    correction_min_horizontal_span_ratio
    correction_min_vertical_span_ratio
    correction_replacement_max_distance_px
    correction_replacement_min_width_px
    correction_replacement_min_height_px
    correction_replacement_min_area
    correction_replacement_max_aspect_ratio
    correction_max_replacements_per_blue
    stacked_text_split_enabled
    stacked_text_split_layers
    stacked_text_min_height_px
    stacked_text_min_width_px
    stacked_text_min_aspect_ratio
    stacked_text_row_ink_ratio
    stacked_text_min_gap_px
    stacked_text_merge_gap_px
    stacked_text_projection_valley_enabled
    stacked_text_projection_valley_max_ratio
    stacked_text_projection_min_side_ink_ratio
    stacked_text_min_segment_height_px
    stacked_text_segment_padding_px
    stacked_text_max_segments

Artifact Policy
---------------
N02 stores exactly two crop files per refined group:

    one full-text visual crop
    one binary topology mask

Compatibility path fields intentionally alias the full-text file instead of
creating duplicate PNGs.

The current settings.json still contains fields from the previous Surgeon
implementation. Unknown fields are ignored by the current settings coercion.

Next Architectural Step
-----------------------
The new Surgeon will be rebuilt gradually as layer-specific grouping policies:

    blue: handwriting continuation and border-profile evidence
    black: conservative printed-text left/right grouping
    red: correction/markup-specific treatment
    green/unknown: independent conservative policies

Do not restore the old mixed-color grouping logic from legacy by accident. The
previous implementation is preserved under scripts/legacy/N02_crop_refiner/.
