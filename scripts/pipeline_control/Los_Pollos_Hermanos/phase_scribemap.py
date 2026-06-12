"""N00+N01 preparation and ScribeMap batch phase."""

import os
import shutil

from . import paths

from N00_file_preparation.file_preparation import prepare_file
from N01_scribemap.scribemap_detector import ScribeMapBWDetector

from .document_io import get_document_id, get_result_path, load_existing_result, save_json
from .paths import FAILED_RESULTS_DIR, TEMP_PROCESSING_DIR

REAL_SCRIBEMAP_LAYERS = ["blue", "red", "green", "unknown_color", "black"]


def role_guess_for_layer(layer_name):
    """Return a simple semantic hint for a ScribeMap color layer."""
    if layer_name == "blue":
        return "probable_handwriting"

    if layer_name == "red":
        return "probable_markup_or_correction"

    if layer_name == "black":
        return "probable_printed_or_form_structure"

    if layer_name == "unknown_color":
        return "unknown_colored_ink"

    if layer_name == "green":
        return "colored_ink"

    return "unknown_layer"


def collect_layer_groups(scribemap_result):
    """Flatten real ScribeMap layer groups into the pipeline group contract."""
    layer_results = scribemap_result.get("layer_results", {})
    collected = []

    for layer_name in REAL_SCRIBEMAP_LAYERS:
        layer_payload = layer_results.get(layer_name, {})

        for index, group in enumerate(layer_payload.get("groups", []), start=1):
            record = dict(group)
            record["layer"] = layer_name
            record["group_uid"] = (
                record.get("group_uid")
                or f"{layer_name}_{record.get('group_id', index):04d}"
            )
            record["role_guess"] = role_guess_for_layer(layer_name)
            collected.append(record)

    return collected


def build_unclassified_group_record(document_id, group, fallback_layer=None):
    """
    Convert a ScribeMap group into the neutral group contract used by N02.

    Args:
        document_id: Stable identifier derived from the document filename.
        group: ScribeMap group dictionary.
        fallback_layer: Optional layer name when the group came from layer output.

    Returns:
        Group dictionary with no ML/model classification attached.
    """
    layer = group.get("layer", fallback_layer)
    group_id = group.get("group_uid", group.get("group_id"))

    return {
        "document_id": document_id,
        "group_id": group_id,
        "original_group_id": group.get("group_id"),
        "group_uid": group.get("group_uid"),
        "layer": layer,
        "source_type": "scribemap_layer_group" if layer else "scribemap_content_group",
        "role_guess": group.get("role_guess", role_guess_for_layer(layer) if layer else "legacy_content_group"),

        "source_crop_path": group.get("crop_path"),
        "classified_crop_path": None,

        "bbox": {
            "x1": group["x1"],
            "y1": group["y1"],
            "x2": group["x2"],
            "y2": group["y2"]
        },

        "crop_bbox": group.get("crop_bbox"),
        "component_count": group["component_count"],
        "density": group["density"],
        "aspect_ratio": group["aspect_ratio"],
        "group_flags": group["group_flags"],

        "classification": {
            "label": "unclassified",
            "confidence": None,
            "handwriting_score": None,
            "contains_handwriting": None,
            "class_scores": {},
            "model_version": None,
            "classification_method": "not_run"
        }
    }


def process_single_document(document_path):
    """
    Run preparation and ScribeMap for one document without ML classification.

    Args:
        document_path: Path to one input document.

    Returns:
        Status dictionary for the processed document.
    """
    document_id = get_document_id(document_path)

    try:
        document_output_dir = f"{TEMP_PROCESSING_DIR}/{document_id}"

        if os.path.exists(document_output_dir):
            shutil.rmtree(document_output_dir)

        os.makedirs(document_output_dir, exist_ok=True)

        file_extension = os.path.splitext(document_path)[1].lower()
        temp_document_path = f"{document_output_dir}/input_document{file_extension}"

        shutil.copy2(document_path, temp_document_path)

        file_preparation_output_dir = f"{document_output_dir}/n00_file_preparation"
        scribemap_output_dir = f"{document_output_dir}/n01_scribemap"

        metadata_dir = f"{scribemap_output_dir}/metadata"
        os.makedirs(metadata_dir, exist_ok=True)

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

        preparation_state = prepare_file(
            input_path=temp_document_path,
            output_dir=file_preparation_output_dir,
            steps=preparation_steps,
            settings={
                "manual_major_rotation": 0
            }
        )

        scribemap = ScribeMapBWDetector(
            settings={
                "save_group_crops": True,
            }
        )

        scribemap_result = scribemap.run_from_preparation_state(
            preparation_state=preparation_state,
            output_dir=scribemap_output_dir
        )

        layer_groups = collect_layer_groups(scribemap_result)
        groups_for_pipeline = layer_groups or scribemap_result.get("groups", [])

        unclassified_groups = [
            build_unclassified_group_record(
                document_id=document_id,
                group=group,
                fallback_layer=group.get("layer"),
            )
            for group in groups_for_pipeline
        ]
        layer_group_counts = {
            layer_name: scribemap_result.get("layer_results", {}).get(layer_name, {}).get("group_count", 0)
            for layer_name in REAL_SCRIBEMAP_LAYERS
        }

        detector_result = {
            "document_id": document_id,
            "input_bw_image_path": temp_document_path,

            "scribemap_group_count": len(unclassified_groups),
            "legacy_content_group_count": scribemap_result.get("group_count", 0),
            "layer_group_counts": layer_group_counts,
            "layer_group_count": sum(layer_group_counts.values()),
            "active_group_source": "scribemap_color_layers" if layer_groups else "scribemap_content_ink_mask",

            "classified_group_count": len(unclassified_groups),
            "unclassified_group_count": len(unclassified_groups),

            "handwriting_count": 0,
            "printed_count": 0,
            "empty_or_noise_count": 0,
            "review_count": 0,

            "classified_groups": unclassified_groups,

            "scribemap_result_path": scribemap_result["metadata_path"],
            "scribemap_artifacts": scribemap_result["artifacts"],
            "file_preparation_output_dir": file_preparation_output_dir,
            "scribemap_output_dir": scribemap_output_dir,

            "classification_status": "not_run",
            "classification_method": "none"
        }

        summary_path = f"{metadata_dir}/{document_id}_classified_groups.json"

        save_json(detector_result, summary_path)

        detector_result["summary_path"] = summary_path

        result_path = get_result_path(document_id)
        result_payload = load_existing_result(document_id)

        result_payload.update(detector_result)
        result_payload["status"] = "scribemap_completed"
        result_payload.setdefault("phases", {})

        result_payload["phases"]["scribemap"] = {
            "status": "completed",
            "scribemap_group_count": detector_result.get("scribemap_group_count"),
            "legacy_content_group_count": detector_result.get("legacy_content_group_count"),
            "layer_group_counts": detector_result.get("layer_group_counts"),
            "active_group_source": detector_result.get("active_group_source"),
            "classified_group_count": detector_result.get("classified_group_count"),
            "classification_status": "not_run",
            "summary_path": detector_result.get("summary_path"),
            "file_preparation_output_dir": file_preparation_output_dir,
            "scribemap_output_dir": scribemap_output_dir,
        }

        save_json(result_payload, result_path)

        print("Prepared temp folder for:", document_id)
        print("Original path:", document_path)
        print("Temp path:", temp_document_path)
        print("N00 output:", file_preparation_output_dir)
        print("N01 output:", scribemap_output_dir)
        print("ScribeMap groups:", detector_result.get("scribemap_group_count"))
        print("ScribeMap source:", detector_result.get("active_group_source"))
        print("Classification: not run")
        print("Result saved:", result_path)
        print("-------------------------")

        return {
            "document_id": document_id,
            "status": "processed",
            "result_path": result_path
        }

    except Exception as error:
        failed_result = {
            "document_id": document_id,
            "source_path": document_path,
            "status": "failed",
            "error": str(error)
        }

        failed_path = f"{FAILED_RESULTS_DIR}/{document_id}_failed.json"

        save_json(failed_result, failed_path)

        print("Failed document:", document_id)
        print("Error:", str(error))
        print("Failure saved:", failed_path)
        print("-------------------------")

        return {
            "document_id": document_id,
            "status": "failed",
            "failed_path": failed_path,
            "error": str(error)
        }


def run_scribemap_phase(document_path):
    """
    Run the ScribeMap phase for one document.

    Args:
        document_path: Path to one input document.

    Returns:
        Status dictionary for the ScribeMap phase.
    """
    return process_single_document(document_path)
