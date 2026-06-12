"""N00 file-preparation phase runner."""

from . import paths

from N00_file_preparation.file_preparation import prepare_file

from .document_io import (
    ensure_temp_input,
    get_document_id,
    get_result_path,
    load_existing_result,
    save_json,
)
from .paths import TEMP_PROCESSING_DIR


def run_prep_phase(document_path):
    """
    Run only the file-preparation phase for one document.

    Args:
        document_path: Path to one input document.

    Returns:
        Status dictionary for the prep phase.
    """
    document_id = get_document_id(document_path)

    document_output_dir = f"{TEMP_PROCESSING_DIR}/{document_id}"
    prep_output_dir = f"{document_output_dir}/n00_file_preparation"

    temp_document_path = ensure_temp_input(
        document_path=document_path,
        document_output_dir=document_output_dir
    )

    preparation_steps = [
        "load_image",
        "rotate_major",
        "convert_to_grayscale",
        "denoise_image",
        "improve_contrast",
        "threshold_image",
        "deskew_image",
        "crop_white_margins",
        "create_scribemap_masks",
        "create_color_layer_masks",
        "save_outputs",
    ]

    state = prepare_file(
        input_path=temp_document_path,
        output_dir=prep_output_dir,
        steps=preparation_steps,
        settings={
            "manual_major_rotation": 0
        }
    )

    out_path = get_result_path(document_id)
    result_payload = load_existing_result(document_id)

    result_payload["status"] = "prepared"
    result_payload["input_path"] = temp_document_path
    result_payload["prep_output_dir"] = prep_output_dir

    result_payload.setdefault("phases", {})

    result_payload["phases"]["prep"] = {
        "status": "completed",
        "artifacts": state.get("artifacts", {}),
        "metadata": state.get("metadata", {})
    }

    save_json(result_payload, out_path)

    print("Prepared:", document_id)
    print("Prep output:", prep_output_dir)
    print("Result saved:", out_path)
    print("-------------------------")

    return {
        "document_id": document_id,
        "status": "processed",
        "result_path": out_path
    }
