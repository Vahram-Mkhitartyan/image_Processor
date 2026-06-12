"""N03 visual-classification phase runner."""

import importlib.util
import os

from .document_io import get_document_id, get_result_path, load_existing_result, save_json
from .paths import (
    MINOS_MODEL_PATH,
    TEMP_PROCESSING_DIR,
    VISUAL_CLASSIFICATION_PATH,
    VISUAL_CLASSIFICATION_SETTINGS_PATH,
)

VISUAL_CLASSIFIER = None


def get_visual_classifier_module():
    """
    Load the N03 visual-classification module only when the phase runs.

    This keeps batch startup light and avoids TensorFlow loading unless
    visual classification is actually requested.

    Returns:
        Imported N03 classifier module.
    """
    global VISUAL_CLASSIFIER

    if VISUAL_CLASSIFIER is None:
        if not os.path.exists(VISUAL_CLASSIFICATION_PATH):
            raise FileNotFoundError(
                f"N03 classifier script not found: {VISUAL_CLASSIFICATION_PATH}"
            )

        spec = importlib.util.spec_from_file_location(
            "visual_classification_router_module",
            VISUAL_CLASSIFICATION_PATH
        )

        module = importlib.util.module_from_spec(spec)

        assert spec.loader is not None

        spec.loader.exec_module(module)

        VISUAL_CLASSIFIER = module

    return VISUAL_CLASSIFIER


def run_visual_classification_phase(document_path):
    """
    Run the N03 visual-classification routing phase for one document.

    Args:
        document_path: Path to one input document.

    Returns:
        Status dictionary for the visual-classification phase.
    """
    document_id = get_document_id(document_path)

    document_output_dir = f"{TEMP_PROCESSING_DIR}/{document_id}"

    refined_groups_path = (
        f"{document_output_dir}/n02_crop_refiner/metadata/"
        f"{document_id}_refined_groups.json"
    )

    if not os.path.exists(refined_groups_path):
        result_path = get_result_path(document_id)
        result_payload = load_existing_result(document_id)

        result_payload["status"] = "visual_classification_skipped"
        result_payload.setdefault("phases", {})

        result_payload["phases"]["visual_classification"] = {
            "status": "skipped",
            "reason": "missing_refined_groups",
            "missing_path": refined_groups_path,
            "next_step": "Run refine phase first."
        }

        save_json(result_payload, result_path)

        print("Skipped N03 visual classification:", document_id)
        print("Reason: missing refined groups.")
        print("Expected:", refined_groups_path)
        print("Next step: run refine first, then visual classification.")
        print("Result saved:", result_path)
        print("-------------------------")

        return {
            "document_id": document_id,
            "status": "skipped",
            "result_path": result_path,
            "reason": "missing_refined_groups"
        }

    classifier_module = get_visual_classifier_module()

    visual_output_dir = f"{document_output_dir}/n03_visual_classification"

    visual_result = classifier_module.classify_document(
        refined_groups_path=refined_groups_path,
        output_dir=visual_output_dir,
        model_path=MINOS_MODEL_PATH,
        settings_path=VISUAL_CLASSIFICATION_SETTINGS_PATH
    )

    result_path = get_result_path(document_id)
    result_payload = load_existing_result(document_id)

    result_payload["status"] = "visual_classified"

    result_payload["visual_classification_path"] = visual_result.get("metadata_path")
    result_payload["visual_classification_dir"] = visual_output_dir

    result_payload.setdefault("phases", {})

    result_payload["phases"]["visual_classification"] = {
        "status": "completed",
        "metadata_path": visual_result.get("metadata_path"),
        "output_dir": visual_output_dir,
        "summary": visual_result.get("summary", {})
    }

    save_json(result_payload, result_path)

    print("N03 visual classified:", document_id)
    print("N03 metadata:", visual_result.get("metadata_path"))
    print("N03 output:", visual_output_dir)
    print("Result saved:", result_path)
    print("-------------------------")

    return {
        "document_id": document_id,
        "status": "processed",
        "result_path": result_path
    }
