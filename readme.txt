Armenian OCR Pipeline
=====================

Purpose
-------
This project is a staged OCR system for scanned Armenian documents containing
printed text, handwriting, corrections, stamps, form lines, and scanner noise.

The active runtime separates ink by color, detects regions, creates layer-aware
crop views, routes crops with Minos, and builds printed and handwriting-expert
maps.

Active Pipeline
---------------

    N00 file preparation
    -> N01 ScribeMap region detection
    -> N02 layer-aware crop preparation
    -> N03 Minos visual classification
    -> N04 printed OCR
    -> N05 handwriting mixture of experts

N04 and N05 both consume N03 routes. Mixed crops are sent to both branches.
N04 currently runs printed Tesseract. N05 builds the coordinate-aware map and
orchestrates Tesseract, ScribeTrace, character-detector, and word-level experts.
Full recognition and learned expert consensus are still pending. ScribeTrace
already converts binary Armenian ink through topology-preserving thinning,
graph construction, ordered path extraction, geometric landmarks, numeric
feature vectors, and symbolic path sequences.

Main Commands
-------------
Run commands from the project root:

    .venv/bin/python main.py status
    .venv/bin/python main.py pipeline

The root launcher forwards to the full controller in scripts/main.py. Direct
controller commands remain available:

    .venv/bin/python scripts/main.py status
    .venv/bin/python scripts/main.py counts
    .venv/bin/python scripts/main.py lines
    .venv/bin/python scripts/main.py doctor
    .venv/bin/python scripts/main.py setup
    .venv/bin/python scripts/main.py prep
    .venv/bin/python scripts/main.py scribemap
    .venv/bin/python scripts/main.py refine
    .venv/bin/python scripts/main.py visual
    .venv/bin/python scripts/main.py printed_ocr
    .venv/bin/python scripts/main.py handwritten_ocr
    .venv/bin/python scripts/main.py pipeline
    .venv/bin/python scripts/main.py clean
    .venv/bin/python scripts/main.py train

Shell Completion
----------------
Enable command completion in the current Bash terminal:

    source scripts/main_completion.bash

Then commands can be completed with Tab:

    ocr ref<Tab>
    python main.py ref<Tab>

Run this to print the activation instructions again:

    .venv/bin/python main.py completion

Aliases
-------

    visual_classification, n03  -> visual
    printed, n04                -> printed_ocr
    handwritten, n05            -> handwritten_ocr
    batch                        -> scribemap

Phase Behavior
--------------
prep:
    Runs N00 only.

scribemap:
    Recreates the per-document temp folder, then runs N00 and N01.

refine:
    Runs N02 from existing N01 metadata. Run scribemap first.

visual:
    Runs N03 from existing N02 metadata using Minos v2.0.

printed_ocr:
    Runs N04 from existing N03 routes using Tesseract.

handwritten_ocr:
    Runs the N05 mixture-of-experts orchestrator from existing N03 routes.

pipeline:
    Runs scribemap, refine, visual, printed OCR, and N05 expert orchestration.

clean:
    Removes generated runtime outputs and caches. It preserves models,
    classifier_dataset_presence, scripts, handwritten_text, and environments.

doctor:
    Runs read-only checks for project structure, Python dependencies, syntax,
    JSON validity, phase contracts, crop references, and stale output paths.

Project Folders
---------------
handwritten_text/
    Input documents.

temp_processing/<document_id>/
    Per-document node outputs.

final_results/
    Per-document phase summary JSON files.

failed_results/
    Reserved failure output folder.

models/
    Minos and other trained models. Preserved by clean.

classifier_dataset_presence/
    Minos training dataset with mixed, printed_only, handwriting_only, and
    empty_or_noise classes. Preserved by clean.

scripts/Cyber_Lin_Kuei_Assembly/
    Model training and evaluation tools.

scripts/pipeline_control/Los_Pollos_Hermanos/
    Internal batch orchestration package.

Per-Document Output Layout
--------------------------

    temp_processing/<document_id>/
        input_document.<ext>
        n00_file_preparation/
        n01_scribemap/
        n02_crop_refiner/
        n03_visual_classification/
        n04_printed_ocr/
        n05_handwritten_ocr/
            scribetrace/debug/

Important Metadata Contracts
----------------------------
N01 bridge metadata:

    temp_processing/<document_id>/n01_scribemap/metadata/
        <document_id>_classified_groups.json

The historical filename says classified, but these records are neutral ScribeMap
groups. No ML classification happens in N01.

N02 metadata:

    temp_processing/<document_id>/n02_crop_refiner/metadata/
        <document_id>_refined_groups.json

N03 routes:

    temp_processing/<document_id>/n03_visual_classification/metadata/
        <document_id>_n03_visual_classification_routes.json

N04 printed map:

    temp_processing/<document_id>/n04_printed_ocr/metadata/
        <document_id>_printed_text_map.json

N05 handwriting expert map:

    temp_processing/<document_id>/n05_handwritten_ocr/metadata/
        <document_id>_handwritten_text_map.json

Node Responsibilities
---------------------
N00 owns image preparation and mask creation.
N01 owns geometry-based component detection and grouping per color layer.
N02 owns layer policy and crop-view generation. The new Surgeon grouping logic
is not implemented yet.
N03 owns Minos visual classification and OCR route recommendations.
N04 owns raw printed OCR candidates.
N05 owns the handwriting-map contract and mixture-of-experts orchestration.
ScribeTrace owns binary-mask topology, ordered paths, landmarks, and
deterministic vector/sequence evidence; it does not yet return Armenian text.

Current Color Contract
----------------------
N00 produces exclusive masks for:

    blue
    red
    green
    unknown_color
    black

A final pixel belongs to at most one semantic layer. Weak color pixels are only
recovered near strong same-color ink, preventing warm paper from becoming red.

Layer policy currently treats:

    blue/green   as probable handwriting
    black        as requiring Minos routing
    red          as correction/markup evidence for future N06
    unknown      as requiring fallback Minos routing

Development Notes
-----------------
- Keep masks in N00 and grouping in N01.
- Keep N02 crop preparation separate from Minos decisions.
- Consume metadata contracts rather than inferring meaning from folder names.
- Raw OCR candidates are evidence, not final reconstructed truth.
- Old experiments and the previous N02 implementation live under scripts/legacy/.
