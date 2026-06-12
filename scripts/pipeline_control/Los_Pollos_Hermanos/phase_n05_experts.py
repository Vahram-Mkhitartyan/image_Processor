"""N05 handwriting mixture-of-experts phase runner."""

import importlib.util
import os

from .document_io import get_document_id, get_result_path, load_existing_result, save_json
from .paths import N05_ORCHESTRATOR_PATH, N05_SETTINGS_PATH, TEMP_PROCESSING_DIR

N05_ORCHESTRATOR = None


def get_n05_orchestrator_module():
    """
    Load the N05 mixture-of-experts orchestrator only when the phase runs.

    Returns:
        Imported N05 expert orchestrator module.
    """
    global N05_ORCHESTRATOR

    if N05_ORCHESTRATOR is None:
        if not os.path.exists(N05_ORCHESTRATOR_PATH):
            raise FileNotFoundError(
                f"N05 expert orchestrator not found: {N05_ORCHESTRATOR_PATH}"
            )

        spec = importlib.util.spec_from_file_location(
            "n05_expert_orchestrator",
            N05_ORCHESTRATOR_PATH
        )

        module = importlib.util.module_from_spec(spec)

        assert spec.loader is not None

        spec.loader.exec_module(module)

        N05_ORCHESTRATOR = module

    return N05_ORCHESTRATOR


def run_n05_expert_phase(document_path):
    """
    Run the N05 handwriting-expert orchestration phase for one document.

    Args:
        document_path: Path to one input document.

    Returns:
        Status dictionary for the N05 expert-orchestration phase.
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

        result_payload["status"] = "handwritten_ocr_skipped"
        result_payload.setdefault("phases", {})

        result_payload["phases"]["handwritten_ocr"] = {
            "status": "skipped",
            "reason": "missing_visual_classification_routes",
            "missing_path": visual_routes_path,
            "next_step": "Run visual classification first."
        }

        save_json(result_payload, result_path)

        print("Skipped N05 expert orchestration:", document_id)
        print("Reason: missing N03 visual classification routes.")
        print("Expected:", visual_routes_path)
        print("Next step: run visual classification first, then N05.")
        print("Result saved:", result_path)
        print("-------------------------")

        return {
            "document_id": document_id,
            "status": "skipped",
            "result_path": result_path,
            "reason": "missing_visual_classification_routes"
        }

    n05_orchestrator = get_n05_orchestrator_module()

    handwritten_ocr_output_dir = f"{document_output_dir}/n05_handwritten_ocr"

    handwritten_ocr_result = n05_orchestrator.build_handwriting_expert_map(
        visual_routes_path=visual_routes_path,
        output_dir=handwritten_ocr_output_dir,
        settings_path=N05_SETTINGS_PATH
    )

    result_path = get_result_path(document_id)
    result_payload = load_existing_result(document_id)

    result_payload["status"] = "handwritten_ocr_completed"

    result_payload["handwritten_ocr_path"] = handwritten_ocr_result.get("metadata_path")
    result_payload["handwritten_ocr_dir"] = handwritten_ocr_output_dir

    result_payload.setdefault("phases", {})

    result_payload["phases"]["handwritten_ocr"] = {
        "status": "completed",
        "metadata_path": handwritten_ocr_result.get("metadata_path"),
        "output_dir": handwritten_ocr_output_dir,
        "summary": handwritten_ocr_result.get("summary", {})
    }

    save_json(result_payload, result_path)

    print("N05 expert orchestration completed:", document_id)
    print("N05 metadata:", handwritten_ocr_result.get("metadata_path"))
    print("N05 output:", handwritten_ocr_output_dir)
    print("Result saved:", result_path)
    print("-------------------------")

    return {
        "document_id": document_id,
        "status": "processed",
        "result_path": result_path
    }
