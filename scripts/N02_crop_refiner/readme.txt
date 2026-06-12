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
3. Preserve layer identity and original source-group ids.
4. Apply a small padded bbox.
5. Create five crop views per group.
6. Write refined group metadata for N03 and future nodes.

Crop Views
----------
original:
    All available semantic visual layers within the bbox. Debug use only.

analysis:
    Target layer isolated on white. Backward-compatible refined_crop_path.

classification:
    Explicit copy of the target-layer-only visual crop. Preferred Minos input.

context:
    Layer-dependent contextual composition. For example, blue includes black
    context and red includes red, blue, and black.

analysis_mask:
    Exact binary white-ink-on-black crop from the target N00 layer mask.
    ScribeTrace uses it for component, skeleton, and graph analysis. It is not
    dilated and is never a Minos input.

Layer Policies
--------------
blue/green:
    Probable handwriting. Minos is used as an audit/router.

black:
    Unknown dark-ink role. Minos is the primary router.

red:
    Correction/markup evidence. Minos is intentionally skipped, the record is
    preserved, and the recommended future destination is N06 correction resolver.

unknown_color:
    Ambiguous color. Minos performs fallback routing.

Output Folder
-------------

    temp_processing/<document_id>/n02_crop_refiner/
        metadata/<document_id>_refined_groups.json
        crops/<layer>/original/
        crops/<layer>/analysis/
        crops/<layer>/classification/
        crops/<layer>/context/
        crops/<layer>/analysis_mask/

Important Record Fields
-----------------------

    text_unit_id
    source_group_id
    source_layer_group_id
    layer
    bbox
    final_bbox
    classification_crop_path
    analysis_crop_path
    context_crop_path
    analysis_mask_crop_path
    refined_crop_path
    mask_source
    visual_layer_source
    minos_required
    minos_mode
    recommended_next_node
    preserve_as_evidence
    refiner

Settings
--------
The current CropRefiner actively uses:

    input_mode / legacy alias input_group_mode
    layers_to_refine / legacy alias scribemap_2_layers_to_refine
    crop_padding_px
    debug_preview_enabled

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
