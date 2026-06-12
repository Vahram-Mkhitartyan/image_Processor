"""N04 printed-OCR phase runner."""

import importlib.util
import os

from .document_io import get_document_id, get_result_path, load_existing_result, save_json
from .paths import PRINTED_OCR_PATH, PRINTED_OCR_SETTINGS_PATH, TEMP_PROCESSING_DIR

PRINTED_OCR = None


def get_printed_ocr_module():
    """
    Load the N04 printed-OCR module only when the phase runs.

    Returns:
        Imported N04 printed OCR module.
    """
    global PRINTED_OCR

    if PRINTED_OCR is None:
        if not os.path.exists(PRINTED_OCR_PATH):
            raise FileNotFoundError(
                f"N04 printed OCR script not found: {PRINTED_OCR_PATH}"
            )

        spec = importlib.util.spec_from_file_location(
            "printed_ocr_module",
            PRINTED_OCR_PATH
        )

        module = importlib.util.module_from_spec(spec)

        assert spec.loader is not None

        spec.loader.exec_module(module)

        PRINTED_OCR = module

    return PRINTED_OCR


def run_printed_ocr_phase(document_path):
    """
    Run the N04 printed-OCR phase for one document.

    Args:
        document_path: Path to one input document.

    Returns:
        Status dictionary for the printed-OCR phase.
    """
    document_id = get_document_id(document_path)

    document_output_dir = f"{TEMP_PROCESSING_DIR}/{document_id}"

    visual_routes_path = (
        f"{document_output_dir}/n03_visual_classification/metadata/"
        f"{document_id}_n03_visual_classification_routes.json"
    )

    if not os.path.exists(visual_routes_path):
        result_path = get_result_path(document_id)
        result_payload = load_existing_result(document_id)

        result_payload["status"] = "printed_ocr_skipped"
        result_payload.setdefault("phases", {})

        result_payload["phases"]["printed_ocr"] = {
            "status": "skipped",
            "reason": "missing_visual_classification_routes",
            "missing_path": visual_routes_path,
            "next_step": "Run visual classification first."
        }

        save_json(result_payload, result_path)

        print("Skipped N04 printed OCR:", document_id)
        print("Reason: missing N03 visual classification routes.")
        print("Expected:", visual_routes_path)
        print("Next step: run visual classification first, then printed OCR.")
        print("Result saved:", result_path)
        print("-------------------------")

        return {
            "document_id": document_id,
            "status": "skipped",
            "result_path": result_path,
            "reason": "missing_visual_classification_routes"
        }

    printed_ocr_module = get_printed_ocr_module()

    printed_ocr_output_dir = f"{document_output_dir}/n04_printed_ocr"

    printed_ocr_result = printed_ocr_module.build_printed_text_map(
        visual_routes_path=visual_routes_path,
        output_dir=printed_ocr_output_dir,
        settings_path=PRINTED_OCR_SETTINGS_PATH
    )

    result_path = get_result_path(document_id)
    result_payload = load_existing_result(document_id)

    result_payload["status"] = "printed_ocr_completed"

    result_payload["printed_ocr_path"] = printed_ocr_result.get("metadata_path")
    result_payload["printed_ocr_dir"] = printed_ocr_output_dir

    result_payload.setdefault("phases", {})

    result_payload["phases"]["printed_ocr"] = {
        "status": "completed",
        "metadata_path": printed_ocr_result.get("metadata_path"),
        "output_dir": printed_ocr_output_dir,
        "summary": printed_ocr_result.get("summary", {})
    }

    save_json(result_payload, result_path)

    print("N04 printed OCR completed:", document_id)
    print("N04 metadata:", printed_ocr_result.get("metadata_path"))
    print("N04 output:", printed_ocr_output_dir)
    print("Result saved:", result_path)
    print("-------------------------")

    return {
        "document_id": document_id,
        "status": "processed",
        "result_path": result_path
    }
