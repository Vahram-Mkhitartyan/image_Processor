Los Pollos Hermanos
===================

Purpose
-------
Internal batch pipeline controller. The name stays. The public entrypoint remains:

    scripts/pipeline_control/batch_processor.py

Files
-----
batch_processor.py:
    CLI argument parsing, input-document loop, and batch totals.

dispatcher.py:
    Runs one requested phase for one document. The pipeline branch executes N01
    through N05 in order; the ScribeMap phase includes N00 preparation.

paths.py:
    Shared node paths, model path, input/output roots, and supported extensions.

    Active Minos model: models/minos_v2_0_best.keras

document_io.py:
    Creates runtime roots, discovers input documents, copies temp inputs, and
    reads/writes final result summaries.

phase_prep.py:
    N00-only phase.

phase_scribemap.py:
    N00 + N01 phase. Flattens real blue/red/green/unknown/black layer groups into
    the historical classified_groups bridge contract.

phase_refine.py:
    N02 crop-view generation.

phase_visual.py:
    N03 Minos classification and route generation.

phase_printed_ocr.py:
    N04 Tesseract printed OCR map.

phase_n05_experts.py:
    Loads the N05 expert orchestrator and builds the handwriting expert map.

Stable Command
--------------

    .venv/bin/python scripts/main.py pipeline

Keeping phase modules split prevents the batch processor from becoming a
900-line chicken monster again.

Project Doctor
--------------
Run the read-only diagnostic suite with:

    .venv/bin/python scripts/main.py doctor

It checks core paths, the selected Python environment, required imports, active
Python syntax, settings JSON, N05 expert package interfaces, retired artifacts,
per-document N00/N01/N02 contracts, crop references, and stale paths in final
result files. Warnings do not fail the command; structural, syntax, dependency,
and contract failures do.
