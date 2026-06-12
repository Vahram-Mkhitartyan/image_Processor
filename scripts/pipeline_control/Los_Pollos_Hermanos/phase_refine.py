"""N02 crop-refinement phase runner."""

import importlib.util
import os

from .document_io import get_document_id, get_result_path, load_existing_result, save_json
from .paths import CROP_REFINER_PATH, TEMP_PROCESSING_DIR

REFINER = None
CROP_REFINER_CLASS = None


def get_crop_refiner_class():
    """Dynamically load the N02 CropRefiner class."""
    global CROP_REFINER_CLASS

    if CROP_REFINER_CLASS is None:
        spec = importlib.util.spec_from_file_location(
            "crop_refiner_module",
            CROP_REFINER_PATH
        )
        module = importlib.util.module_from_spec(spec)

        assert spec.loader is not None

        spec.loader.exec_module(module)
        CROP_REFINER_CLASS = module.CropRefiner

    return CROP_REFINER_CLASS


def run_refine_phase(document_path):
    """
    Run the crop-refinement phase for one document.

    Args:
        document_path: Path to one input document.

    Returns:
        Status dictionary for the refine phase.
    """
    document_id = get_document_id(document_path)

    document_output_dir = f"{TEMP_PROCESSING_DIR}/{document_id}"

    classified_groups_path = (
        f"{document_output_dir}/n01_scribemap/metadata/"
        f"{document_id}_classified_groups.json"
    )

    if not os.path.exists(classified_groups_path):
        result_path = get_result_path(document_id)
        result_payload = load_existing_result(document_id)

        result_payload["status"] = "refine_skipped"
        result_payload.setdefault("phases", {})

        result_payload["phases"]["refine"] = {
            "status": "skipped",
            "reason": "missing_classified_groups",
            "missing_path": classified_groups_path,
            "next_step": "Run scribemap phase first."
        }

        save_json(result_payload, result_path)

        print("Skipped refine:", document_id)
        print("Reason: missing classified groups.")
        print("Expected:", classified_groups_path)
        print("Next step: run scribemap first, then refine.")
        print("Result saved:", result_path)
        print("-------------------------")

        return {
            "document_id": document_id,
            "status": "skipped",
            "result_path": result_path,
            "reason": "missing_classified_groups"
        }

    global REFINER

    if REFINER is None:
        REFINER = get_crop_refiner_class()()

    refined_groups_path = (
        f"{document_output_dir}/n02_crop_refiner/metadata/"
        f"{document_id}_refined_groups.json"
    )

    refine_result = REFINER.refine_document(
        classified_groups_json_path=classified_groups_path,
        output_path=refined_groups_path
    )

    out_path = get_result_path(document_id)
    result_payload = load_existing_result(document_id)

    result_payload["status"] = "refined"
    result_payload["classified_groups_path"] = classified_groups_path
    result_payload["refined_groups_path"] = refined_groups_path
    result_payload["crop_refiner_output_dir"] = f"{document_output_dir}/n02_crop_refiner"
    result_payload["refined_crops_dir"] = refine_result.get("refined_crops_dir")

    result_payload.setdefault("phases", {})

    result_payload["phases"]["refine"] = {
        "status": "completed",
        "group_count": refine_result.get("group_count", 0),
        "refined_groups_path": refined_groups_path,
        "output_dir": f"{document_output_dir}/n02_crop_refiner",
        "refined_crops_dir": refine_result.get("refined_crops_dir")
    }

    save_json(result_payload, out_path)

    print("Refined:", document_id)
    print("Refined groups file:", refined_groups_path)
    print("Refined crops dir:", refine_result.get("refined_crops_dir"))
    print("Result saved:", out_path)
    print("-------------------------")

    return {
        "document_id": document_id,
        "status": "processed",
        "result_path": out_path
    }
