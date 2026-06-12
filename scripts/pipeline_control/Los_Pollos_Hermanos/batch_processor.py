"""CLI entrypoint for the batch pipeline controller.

This module lives in the intentionally quiet internal folder. The public wrapper
remains `scripts/pipeline_control/batch_processor.py`.
"""

import argparse

from .dispatcher import run_phase_for_document
from .document_io import ensure_batch_folders, get_document_paths
from .paths import FAILED_RESULTS_DIR, FINAL_RESULTS_DIR, INPUT_DOCUMENTS_DIR, TEMP_PROCESSING_DIR


def main():
    """Run a selected phase for all documents in handwritten_text."""
    parser = argparse.ArgumentParser(description="Batch runner by phase.")

    parser.add_argument(
        "--phase",
        choices=[
            "prep",
            "scribemap",
            "refine",
            "visual",
            "visual_classification",
            "n03",
            "printed_ocr",
            "printed",
            "n04",
            "handwritten_ocr",
            "handwritten",
            "n05",
            "pipeline"
        ],
        default="pipeline",
        help="Phase to run for all files in handwritten_text."
    )

    args = parser.parse_args()

    ensure_batch_folders()

    document_paths = get_document_paths(INPUT_DOCUMENTS_DIR)

    if not document_paths:
        print("No supported documents found in the input folder.")
    else:
        print("Documents found:", len(document_paths))

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for document_path in document_paths:
        result = run_phase_for_document(
            document_path=document_path,
            phase=args.phase
        )

        if result["status"] == "processed":
            processed_count += 1

        elif result["status"] == "skipped":
            skipped_count += 1

        else:
            failed_count += 1

    print("Batch complete.")
    print("Processed:", processed_count)
    print("Skipped:", skipped_count)
    print("Failed:", failed_count)
    print("Input folder:", INPUT_DOCUMENTS_DIR)
    print("Temp folder:", TEMP_PROCESSING_DIR)
    print("Final results folder:", FINAL_RESULTS_DIR)
    print("Failed results folder:", FAILED_RESULTS_DIR)


if __name__ == "__main__":
    main()
