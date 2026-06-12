Pipeline Control
================

Purpose
-------
This folder contains runtime orchestration, not an OCR node. It discovers input
documents, dispatches phases, maintains final result summaries, and keeps phase
entrypoints stable for main.py.

Public Entry
------------

    scripts/pipeline_control/batch_processor.py

This small wrapper calls the internal package entrypoint.

Internal Package
----------------

    scripts/pipeline_control/Los_Pollos_Hermanos/

Supported Phases
----------------

    prep
    scribemap
    refine
    visual / visual_classification / n03
    printed_ocr / printed / n04
    handwritten_ocr / handwritten / n05
    pipeline

Full pipeline means:

    N00 -> N01 -> N02 -> N03 -> N04 -> N05

Normally use main.py:

    .venv/bin/python scripts/main.py pipeline

Direct controller use is also possible:

    .venv/bin/python scripts/pipeline_control/batch_processor.py --phase refine

Phase Requirements
------------------
prep:
    Reads handwritten_text directly.

scribemap:
    Rebuilds the document temp folder and runs N00 + N01.

refine:
    Requires N01 classified_groups compatibility metadata.

visual:
    Requires N02 refined_groups metadata and the Minos model.

printed_ocr / handwritten_ocr:
    Require N03 route metadata. The handwritten_ocr command is a compatibility
    alias for the N05 mixture-of-experts orchestrator.

Minos runs only in N03. The removed v1.4 classifier is not part of this runner.
