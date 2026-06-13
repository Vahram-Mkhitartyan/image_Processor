Node N05: Handwriting Mixture Of Experts
========================================

Purpose
-------
N05 consumes N03 routes classified as handwriting_only or mixed, plus explicit
force_handwritten_ocr red replacement routes from N02 correction pairing. It
prepares a coordinate-aware handwriting map and orchestrates four independent
recognition experts. The experts will eventually produce competing evidence for
a learned consensus model.

Before any expert runs, N05 also builds deterministic character-unit
segmentation hypotheses. The current implementation uses ScribeTrace geometry:
image-space projection valleys propose boundaries and the thinned skeleton
validates blank gaps or narrow left-to-right cursive bridges. It does not
perform recognition or choose a final sequence.

Current Status
--------------
The orchestration and input-map contract are active in main.py and the full
pipeline. Expert execution remains disabled in settings.json while the
recognizers are built, so normal N05 pipeline records remain placeholders.

ScribeTrace has progressed beyond its placeholder contract and can run
standalone. It now converts exact N02 masks through components, Zhang-Suen
thinning, graph topology, ordered paths, geometric landmarks, numeric features,
visible ink-hole evidence, traversal-direction statistics, and symbolic path
sequences. It does not yet perform Armenian pattern matching or return
recognized text.

Its current encoder emits 104 numeric features. The feature_names field is the
persisted schema for vector positions. Training and inference must store and
compare that exact ordered schema instead of assuming dictionary insertion
order or relying only on the schema version label.

Commands
--------

    .venv/bin/python scripts/main.py n05
    .venv/bin/python scripts/main.py handwritten
    .venv/bin/python scripts/main.py handwritten_ocr

The older command names remain stable aliases. They invoke the N05 expert
orchestrator, not one monolithic handwriting OCR engine.

Main Entry
----------

    expert_orchestrator.py
    build_handwriting_expert_map(visual_routes_path, output_dir, settings_path=None)

Expert Structure
----------------

    tesseract_ocr/
        Conventional OCR expert for OCR-ready text units.

    scribetrace/
        Active raster-to-vector geometry engine. Armenian pattern matching and
        character recognition are the next stages.

    character_detector/
        Character-level Armenian detector being remade from the current glyph
        OCR work. This folder owns numeric_label_map.json and scan_matenadata.py.

    word_level_ocr/
        Whole-word recognition expert.

Each expert exposes:

    get_expert_manifest(settings=None)
    recognize(crop_path, context=None, settings=None)

Disabled or unfinished expert interfaces return attempted=false rather than
inventing OCR output.

Current Flow
------------
1. Load N03 visual-route metadata.
2. Select handwriting_only, mixed, and forced red-replacement routes.
3. Reference the canonical N02 full-text crop without copying it.
4. Preserve coordinates, crop lineage, masks, and routing evidence.
5. Always create a whole-unit character hypothesis.
6. Measure mask geometry, borders, connected components, and vertical
   projection valleys.
7. Attach small floating upper marks to the nearest plausible lower/main
   component so they are preserved with that character candidate.
8. Thin the mask with Zhang-Suen and validate valley cuts against skeleton
   crossings.
9. Sort accepted boundaries left-to-right and materialize one contiguous
   character-sequence hypothesis plus a debug overlay.
10. Build the four-expert registry from settings.json.
11. Keep existing experts on the original whole-unit crop for now.
12. Save every proposal inside the handwriting text map.

The universal proposer no longer writes segment images. Each recognition expert
will own its eventual segmentation strategy. copy_selected_crops can be enabled
for an intentional export, but defaults to false.

Character Unit Proposer
-----------------------

    character_unit_proposer.py
    propose_character_units(handwritten_text_unit, folders)

Each proposal contains:

    h0_whole
        Mandatory baseline preserving the complete text unit.

    diagnostics
        Width, height, aspect ratio, ink pixels, connected components, border
        contact, and the full vertical projection profile.

    recovery_needed / recovery_reasons
        Diagnostic flags for border contact, unusually wide units, and many
        disconnected components. Recovery flags do not reject a crop.

    trace_supported_character_sequence hypothesis
        One ordered sequence containing every accepted character crop. A blank
        gap must separate meaningful left/right vector groups. A joined-letter
        cut must identify one exact non-loop TracePath edge; virtually removing
        it must create exactly two substantial, side-dominant vector subgraphs
        away from junctions.

The debug overlay labels accepted boundaries as:

    G
        Existing disconnected vector groups separated by blank space.

    P12:7
        Connector TracePath 12 split after ordered path point 7.

The proposer never runs OCR or Random Forest inference, never modifies source
artifacts, and never retries recursively. It saves new mask and visual crops
under character_unit_proposer/segments/. The complete h0_whole input remains
the mandatory fallback. Segment hypotheses are evidence only; expert scoring
and final sequence selection belong to a later version.

Crop selection priority:

    routed_crop_path
    refined_crop_path
    source_crop_path

Settings
--------
settings.json controls output reset behavior and whether each expert is enabled.
All experts currently remain disabled in the integrated pipeline. ScribeTrace
may still be exercised directly while its evidence pipeline is developed.

MatenaData
----------
Character-level training assets live under:

    character_detector/numeric_label_map.json
    character_detector/scan_matenadata.py

Training and evaluation programs remain in Cyber_Lin_Kuei_Assembly.

Output Folder
-------------

    temp_processing/<document_id>/n05_handwritten_ocr/
        crops/handwriting_only/
        crops/mixed/
        crops/fallback_from_printed_only/
        character_unit_proposer/segments/
        character_unit_proposer/debug/
        metadata/<document_id>_handwritten_text_map.json
        debug/skeletons/
        debug/overlays/
        scribetrace/metadata/<stable_unit_id>_scribetrace.json
        scribetrace/debug/

The runtime output folder keeps its historical name for pipeline compatibility.
Standalone and integrated ScribeTrace debug files are owned by N05:

    n05_handwritten_ocr/scribetrace/debug/

ScribeTrace remains disabled in integrated settings until its recognition layer
is ready. Its standalone geometric/vector engine and tests remain active.
