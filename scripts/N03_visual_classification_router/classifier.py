import os
import json
import shutil

import cv2
import numpy as np
import tensorflow as tf


# CONSTANTS---------------------------------------------------

# Minos was trained on square 128x128 grayscale crops.
IMAGE_SIZE = 128

BASE_DIR = "/home/vahram/Desktop/image_Processor"

# Locked model currently used by N03.
DEFAULT_MODEL_PATH = f"{BASE_DIR}/models/minos_v2_0_best.keras"

MODEL_NAME = "Minos"
MODEL_VERSION = "2.0"

NODE_NAME = "N03_visual_classification_router"
NODE_VERSION = "0.1.0"


DEFAULT_THRESHOLDS = {
    "printed_threshold": 0.45,
    "handwriting_threshold": 0.45,
    "noise_threshold": 0.65,

    # Safety net: if printed is strong and handwriting has a weak-but-real
    # signal, route as mixed instead of risking printed-only.
    "mixed_handwriting_safety_threshold": 0.25
}


# SUPPORTED IMAGE FORMATS-------------------------------------

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# IO HELPERS--------------------------------------------------

def load_json(input_path):
    """
    Load JSON data from disk.

    Args:
        input_path: Path to the JSON file.

    Returns:
        Parsed JSON payload.
    """
    with open(input_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, output_path):
    """
    Save JSON data to disk.

    Args:
        data: Serializable payload.
        output_path: Path where JSON should be written.

    Returns:
        Path to the saved file.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    return output_path


def ensure_dir(path):
    """Create a directory if it does not exist yet."""
    os.makedirs(path, exist_ok=True)


def check_file_exists(path, label="file"):
    """
    Fail early with a readable label if a required file is missing.

    Args:
        path: Required file path.
        label: Human-readable file role for error messages.

    Returns:
        None.
    """
    if not path:
        raise FileNotFoundError(f"Missing {label}: path is empty")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_settings(settings_path=None):
    """
    Load optional N03 settings.

    Args:
        settings_path: Optional path to settings JSON.

    Returns:
        Settings dictionary, or empty dict when no settings file is present.
    """
    if settings_path is None:
        return {}

    if not os.path.exists(settings_path):
        return {}

    return load_json(settings_path)


def normalize_thresholds(configured_thresholds=None):
    """Merge threshold overrides with defaults and normalize legacy names.

    Args:
        configured_thresholds: Optional threshold dictionary from settings.

    Returns:
        Complete threshold dictionary using ``handwriting_threshold``.
    """
    configured = dict(configured_thresholds or {})

    if (
        "handwritten_threshold" in configured
        and "handwriting_threshold" not in configured
    ):
        configured["handwriting_threshold"] = configured["handwritten_threshold"]

    configured.pop("handwritten_threshold", None)

    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(configured)

    return thresholds


# OUTPUT FOLDER HELPERS--------------------------------------

def create_output_folders(output_dir):
    """
    Create the debug-friendly N03 output folder structure.

    N03 output structure:

    output_dir/
        classified/
            mixed/
            printed_only/
            handwriting_only/
            empty_or_noise/
            review/
        metadata/
        debug/

    The classified folders are for human inspection.
    The metadata folder is the real machine-readable output.
    The debug folder holds optional visual-inspection artifacts.
    """
    folders = {
        "root": output_dir,

        "classified": f"{output_dir}/classified",

        "mixed": f"{output_dir}/classified/mixed",
        "printed_only": f"{output_dir}/classified/printed_only",
        "handwriting_only": f"{output_dir}/classified/handwriting_only",
        "empty_or_noise": f"{output_dir}/classified/empty_or_noise",
        "review": f"{output_dir}/classified/review",

        "metadata": f"{output_dir}/metadata",
        "debug": f"{output_dir}/debug",
    }

    for folder in folders.values():
        ensure_dir(folder)

    return folders


def reset_output_dir(output_dir):
    """
    Delete previous N03 output for this document and recreate it.

    This keeps reruns clean.

    This prevents stale copied crops from previous classifications.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)


# IMAGE PREPROCESSING-----------------------------------------

def resize_with_padding(image, target_size=IMAGE_SIZE):
    """
    Resize an image into a padded square canvas.

    Args:
        image: Grayscale crop image.
        target_size: Output square side length.

    Returns:
        Square grayscale image with white padding.
    """
    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError("Invalid image size for padding")

    # Scale the longest side to the model input size.
    scale = target_size / max(height, width)

    new_width = max(int(width * scale), 1)
    new_height = max(int(height * scale), 1)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    # White padding matches the prepared document background.
    canvas = 255 * np.ones((target_size, target_size), dtype=np.uint8)

    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width,
    ] = resized

    return canvas


def prepare_crop_for_minos(crop_path):
    """
    Load and prepare one crop for Minos inference.

    Steps:
    1. Load crop as grayscale.
    2. Resize with padding to 128x128.
    3. Normalize pixel values to [0, 1].
    4. Add channel dimension.
    5. Add batch dimension.

    Final shape:
        (1, 128, 128, 1)
    """
    image = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not load crop image: {crop_path}")

    padded = resize_with_padding(
        image,
        target_size=IMAGE_SIZE,
    )

    normalized = padded.astype("float32") / 255.0

    # Shape becomes: (128, 128, 1)
    normalized = np.expand_dims(normalized, axis=-1)

    # Shape becomes: (1, 128, 128, 1)
    batch = np.expand_dims(normalized, axis=0)

    return batch


def load_minos_model(model_path):
    """
    Load the locked Minos v2.0 model.

    Minos is the ML model.
    N03_visual_classification_router is the pipeline node.

    The model outputs 3 sigmoid scores:
    - printed_present
    - handwriting_present
    - noise
    """
    check_file_exists(model_path, label="Minos model")

    return tf.keras.models.load_model(model_path)


# MINOS CLASSIFICATION HELPERS-------------------------------

def derive_visual_class(scores, thresholds):
    """
    Convert Minos raw output scores into one final visual class.

    Minos output order:
        scores[0] = printed_present
        scores[1] = handwriting_present
        scores[2] = noise

    Routing philosophy:
    - Missing mixed is dangerous.
    - False mixed is acceptable.
    - If printed is strong and handwriting is borderline, choose mixed.
    """
    printed_score = float(scores[0])
    handwriting_score = float(scores[1])
    noise_score = float(scores[2])

    printed_threshold = thresholds["printed_threshold"]
    handwriting_threshold = thresholds["handwriting_threshold"]
    noise_threshold = thresholds["noise_threshold"]

    # Extra safety threshold for mixed crops with weak handwriting signal.
    mixed_handwriting_safety_threshold = thresholds.get(
        "mixed_handwriting_safety_threshold",
        0.25,
    )

    printed_present = printed_score >= printed_threshold
    handwriting_present = handwriting_score >= handwriting_threshold
    noise_present = noise_score >= noise_threshold

    # Pure noise only counts as noise if text signals are not also present.
    if noise_present and not printed_present and not handwriting_present:
        return "empty_or_noise"

    # Strong normal mixed case.
    if printed_present and handwriting_present:
        return "mixed"

    # Safety rule: avoid sending possible mixed crops to printed-only.
    if printed_present and handwriting_score >= mixed_handwriting_safety_threshold:
        return "mixed"

    if printed_present:
        return "printed_only"

    if handwriting_present:
        return "handwriting_only"

    return "review"


def get_recommended_route(visual_class):
    """
    Convert final visual class into downstream route recommendation.

    N03 does not run OCR.
    N03 only recommends which downstream OCR node should receive the crop.
    """
    if visual_class == "mixed":
        return ["printed_ocr", "handwritten_ocr"]

    if visual_class == "printed_only":
        return ["printed_ocr"]

    if visual_class == "handwriting_only":
        return ["handwritten_ocr"]

    if visual_class == "empty_or_noise":
        return ["archive"]

    return ["review"]


def build_score_dict(scores):
    """
    Convert raw Minos scores into a readable dictionary for metadata.
    """
    return {
        "printed_present": round(float(scores[0]), 6),
        "handwriting_present": round(float(scores[1]), 6),
        "noise": round(float(scores[2]), 6),
    }


def classify_crop(model, crop_path, thresholds):
    """
    Classify one N02 classification crop with Minos.

    Input:
        crop_path: path to one N02 classification crop

    Output:
        classification dictionary containing:
        - visual_class
        - recommended_route
        - scores
        - thresholds used
    """
    batch = prepare_crop_for_minos(crop_path)

    prediction = model.predict(batch, verbose=0)[0]

    visual_class = derive_visual_class(
        scores=prediction,
        thresholds=thresholds,
    )

    return {
        "visual_class": visual_class,
        "recommended_route": get_recommended_route(visual_class),
        "scores": build_score_dict(prediction),
        "thresholds": dict(thresholds),
    }


# N02 REFINED GROUP HELPERS----------------------------------

def get_refiner_status(group):
    """
    Safely get the N02 refiner status for a group.

    Expected statuses:
    - accepted
    - review
    - rejected

    If status is missing, we treat it as review.
    """
    refiner = group.get("refiner", {})

    return refiner.get("status", "review")


def get_refiner_next_node(group):
    """
    Safely get the N02 suggested next node.

    N02 may already suggest whether a crop should continue,
    but N03 makes the visual routing decision.
    """
    refiner = group.get("refiner", {})

    return refiner.get("next_node")


def get_minos_input_crop_path(group):
    """
    Return the best crop path for Minos visual classification.

    Color Update rule:
    Minos should classify the layer-isolated visual crop, not the full/context
    crop. Context/original crops may contain multiple layers and can create
    fake mixed predictions.

    Never use analysis_mask_crop_path here. That crop is binary math input for
    future ScribeTrace, not visual classifier input.
    """
    candidates = [
        ("classification_crop_path", group.get("classification_crop_path")),
        ("analysis_crop_path", group.get("analysis_crop_path")),
        ("refined_crop_path", group.get("refined_crop_path")),

        # Fallbacks only. These can include extra layers.
        ("context_crop_path", group.get("context_crop_path")),
        ("original_crop_path", group.get("original_crop_path")),
    ]

    for source_name, crop_path in candidates:
        if crop_path and os.path.exists(crop_path):
            return crop_path, source_name

    return None, None


def should_process_refined_group(group, settings):
    """
    Decide whether a refined N02 group should be classified by Minos.

    Main Color Update rules:
    - If N02 says minos_required is false, skip.
    - If N02 rejected the group and include_rejected is false, skip.
    - If no valid visual crop exists, skip.
    """
    if group.get("minos_required") is False:
        return False

    status = get_refiner_status(group)

    include_rejected = settings.get("include_rejected", False)

    if status == "rejected" and not include_rejected:
        return False

    minos_input_crop_path, _ = get_minos_input_crop_path(group)

    if minos_input_crop_path is None:
        return False

    return True


def get_skip_reason_for_group(group, settings):
    """
    Return a human-readable reason why a group is skipped by Minos.
    """
    if group.get("minos_required") is False:
        return "minos_not_required_by_n02_policy"

    status = get_refiner_status(group)
    include_rejected = settings.get("include_rejected", False)

    if status == "rejected" and not include_rejected:
        return "rejected_by_refiner"

    minos_input_crop_path, _ = get_minos_input_crop_path(group)

    if minos_input_crop_path is None:
        return "no_valid_minos_input_crop"

    return "unknown_skip_reason"


# ROUTE RECORD HELPERS---------------------------------------

def copy_crop_to_visual_class_folder(crop_path, folders, visual_class):
    """
    Copy the crop Minos actually used into its N03 visual class folder.

    This is mainly for human/debug inspection.

    Example:
        visual_class = "mixed"

    Then crop is copied into:
        output_dir/classified/mixed/
    """
    if visual_class not in folders:
        visual_class = "review"

    file_name = os.path.basename(crop_path)

    target_path = f"{folders[visual_class]}/{file_name}"

    shutil.copy2(crop_path, target_path)

    return target_path


def build_route_record(group, classification, routed_crop_path):
    """
    Build one machine-readable N03 route record.

    Color Update contract:
    - preserve N02 group identity
    - preserve all N02 crop views
    - record exactly which crop Minos used
    - use classification_crop_path as preferred Minos input
    - never use analysis_mask_crop_path as Minos input
    """
    minos_input_crop_path, minos_input_crop_source = get_minos_input_crop_path(group)

    return {
        # Identity.
        "document_id": group.get("document_id"),
        "text_unit_id": group.get("text_unit_id"),
        "group_id": group.get("group_id"),
        "source_group_id": group.get("source_group_id"),
        "source_layer_group_id": group.get("source_layer_group_id"),
        "layer": group.get("layer"),

        # Original / legacy crop references.
        "source_crop_path": group.get("source_crop_path"),
        "refined_crop_path": group.get("refined_crop_path"),
        "routed_crop_path": routed_crop_path,

        # Color Update crop references.
        "original_crop_path": group.get("original_crop_path"),
        "analysis_crop_path": group.get("analysis_crop_path"),
        "classification_crop_path": group.get("classification_crop_path"),
        "classification_crop_source": group.get("classification_crop_source"),
        "classification_crop_policy": group.get("classification_crop_policy"),
        "classification_layer": group.get("classification_layer"),
        "context_crop_path": group.get("context_crop_path"),
        "analysis_mask_crop_path": group.get("analysis_mask_crop_path"),

        # Actual Minos input audit.
        "minos_input_crop_path": minos_input_crop_path,
        "minos_input_crop_source": minos_input_crop_source,

        # N00/N02 source tracking.
        "mask_source": group.get("mask_source"),
        "visual_layer_source": group.get("visual_layer_source"),

        # Spatial metadata.
        "bbox": group.get("bbox"),
        "crop_bbox": group.get("crop_bbox"),
        "final_bbox": group.get("final_bbox"),

        # N02 policy / routing metadata.
        "layer_hypothesis": group.get("layer_hypothesis"),
        "role_guess": group.get("role_guess"),
        "recommended_next_node": group.get("recommended_next_node"),
        "minos_required": group.get("minos_required", True),
        "minos_mode": group.get("minos_mode"),
        "is_final_text_candidate": group.get("is_final_text_candidate", True),
        "preserve_as_evidence": group.get("preserve_as_evidence", False),
        "n02_policy": group.get("policy"),

        # N02 status metadata.
        "refiner_status": get_refiner_status(group),
        "refiner_next_node": get_refiner_next_node(group),
        "refiner": group.get("refiner"),

        # N03 result.
        "visual_classification": {
            "node": NODE_NAME,
            "node_version": NODE_VERSION,
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "visual_class": classification["visual_class"],
            "recommended_route": classification["recommended_route"],
            "scores": classification["scores"],
            "thresholds": classification["thresholds"],
        },
    }


def build_skipped_record(group, settings):
    """
    Build a route record for a group skipped by Minos.

    Example:
        Red correction evidence should usually appear here:
        minos_skipped = true
        skip_reason = minos_not_required_by_n02_policy
        route = N06_correction_resolver
    """
    minos_input_crop_path, minos_input_crop_source = get_minos_input_crop_path(group)

    skip_reason = get_skip_reason_for_group(group, settings)

    return {
        "document_id": group.get("document_id"),
        "text_unit_id": group.get("text_unit_id"),
        "group_id": group.get("group_id"),
        "source_group_id": group.get("source_group_id"),
        "source_layer_group_id": group.get("source_layer_group_id"),
        "layer": group.get("layer"),

        "bbox": group.get("bbox"),
        "crop_bbox": group.get("crop_bbox"),
        "final_bbox": group.get("final_bbox"),

        "layer_hypothesis": group.get("layer_hypothesis"),
        "role_guess": group.get("role_guess"),
        "recommended_next_node": group.get("recommended_next_node"),

        "minos_required": group.get("minos_required", True),
        "minos_mode": group.get("minos_mode"),
        "minos_skipped": True,
        "skip_reason": skip_reason,

        "is_final_text_candidate": group.get("is_final_text_candidate", True),
        "preserve_as_evidence": group.get("preserve_as_evidence", False),

        "route": group.get("recommended_next_node", "review"),

        "original_crop_path": group.get("original_crop_path"),
        "analysis_crop_path": group.get("analysis_crop_path"),
        "classification_crop_path": group.get("classification_crop_path"),
        "classification_crop_source": group.get("classification_crop_source"),
        "classification_crop_policy": group.get("classification_crop_policy"),
        "classification_layer": group.get("classification_layer"),
        "context_crop_path": group.get("context_crop_path"),
        "analysis_mask_crop_path": group.get("analysis_mask_crop_path"),
        "refined_crop_path": group.get("refined_crop_path"),

        "mask_source": group.get("mask_source"),
        "visual_layer_source": group.get("visual_layer_source"),

        "minos_input_crop_path": minos_input_crop_path,
        "minos_input_crop_source": minos_input_crop_source,

        "n02_policy": group.get("policy"),
        "refiner_status": get_refiner_status(group),
        "refiner_next_node": get_refiner_next_node(group),
        "refiner": group.get("refiner"),
    }


def build_failed_record(group, error):
    """
    Build a record for a crop N03 tried to process but failed.

    Failure is different from skipped:
    - skipped = intentionally not processed
    - failed = should have processed, but something crashed
    """
    minos_input_crop_path, minos_input_crop_source = get_minos_input_crop_path(group)

    return {
        "document_id": group.get("document_id"),
        "text_unit_id": group.get("text_unit_id"),
        "group_id": group.get("group_id"),
        "source_group_id": group.get("source_group_id"),
        "source_layer_group_id": group.get("source_layer_group_id"),
        "layer": group.get("layer"),

        "bbox": group.get("bbox"),
        "crop_bbox": group.get("crop_bbox"),
        "final_bbox": group.get("final_bbox"),

        "layer_hypothesis": group.get("layer_hypothesis"),
        "role_guess": group.get("role_guess"),
        "recommended_next_node": group.get("recommended_next_node"),

        "minos_required": group.get("minos_required", True),
        "minos_mode": group.get("minos_mode"),
        "minos_failed": True,

        "is_final_text_candidate": group.get("is_final_text_candidate", True),
        "preserve_as_evidence": group.get("preserve_as_evidence", False),

        "source_crop_path": group.get("source_crop_path"),
        "refined_crop_path": group.get("refined_crop_path"),
        "original_crop_path": group.get("original_crop_path"),
        "analysis_crop_path": group.get("analysis_crop_path"),
        "classification_crop_path": group.get("classification_crop_path"),
        "classification_crop_source": group.get("classification_crop_source"),
        "classification_crop_policy": group.get("classification_crop_policy"),
        "classification_layer": group.get("classification_layer"),
        "context_crop_path": group.get("context_crop_path"),
        "analysis_mask_crop_path": group.get("analysis_mask_crop_path"),

        "mask_source": group.get("mask_source"),
        "visual_layer_source": group.get("visual_layer_source"),

        "minos_input_crop_path": minos_input_crop_path,
        "minos_input_crop_source": minos_input_crop_source,

        "n02_policy": group.get("policy"),
        "refiner_status": get_refiner_status(group),
        "refiner_next_node": get_refiner_next_node(group),
        "refiner": group.get("refiner"),

        "error": str(error),
    }


# SUMMARY HELPERS--------------------------------------------

def summarize_routes(route_records, skipped_records, failed_records):
    """
    Build a compact summary of N03 routing results.

    This summary is useful for:
    - terminal output
    - final result JSON
    - quick debugging
    - deciding whether routing looks reasonable
    """
    summary = {
        "processed_count": len(route_records),
        "skipped_count": len(skipped_records),
        "failed_count": len(failed_records),

        "mixed_count": 0,
        "printed_only_count": 0,
        "handwriting_only_count": 0,
        "empty_or_noise_count": 0,
        "review_count": 0,

        "skipped_by_policy_count": 0,
        "skipped_rejected_count": 0,
        "skipped_missing_crop_count": 0,
    }

    for record in route_records:
        visual_class = record["visual_classification"]["visual_class"]

        if visual_class == "mixed":
            summary["mixed_count"] += 1

        elif visual_class == "printed_only":
            summary["printed_only_count"] += 1

        elif visual_class == "handwriting_only":
            summary["handwriting_only_count"] += 1

        elif visual_class == "empty_or_noise":
            summary["empty_or_noise_count"] += 1

        else:
            summary["review_count"] += 1

    for record in skipped_records:
        skip_reason = record.get("skip_reason")

        if skip_reason == "minos_not_required_by_n02_policy":
            summary["skipped_by_policy_count"] += 1

        elif skip_reason == "rejected_by_refiner":
            summary["skipped_rejected_count"] += 1

        elif skip_reason == "no_valid_minos_input_crop":
            summary["skipped_missing_crop_count"] += 1

    return summary


def print_summary(document_id, summary, metadata_path):
    """
    Print a clean terminal summary after N03 finishes.

    Keep this short because the detailed information is already
    stored in the metadata JSON.
    """
    print("-------------------------")
    print("N03 visual classification completed.")
    print("Document:", document_id)
    print("Processed:", summary["processed_count"])
    print("Skipped:", summary["skipped_count"])
    print("Failed:", summary["failed_count"])
    print("Mixed:", summary["mixed_count"])
    print("Printed only:", summary["printed_only_count"])
    print("Handwriting only:", summary["handwriting_only_count"])
    print("Empty/noise:", summary["empty_or_noise_count"])
    print("Review:", summary["review_count"])
    print("Skipped by N02 policy:", summary["skipped_by_policy_count"])
    print("Skipped rejected:", summary["skipped_rejected_count"])
    print("Skipped missing crop:", summary["skipped_missing_crop_count"])
    print("Metadata:", metadata_path)
    print("-------------------------")


# MAIN DOCUMENT CLASSIFIER-----------------------------------

def classify_document(
    refined_groups_path,
    output_dir,
    model_path=DEFAULT_MODEL_PATH,
    settings_path=None,
):
    """
    Run N03 visual classification routing for one document.

    Input:
        refined_groups_path:
            Path to N02 refined groups JSON.

        output_dir:
            Folder where N03 writes classified crops and metadata.

        model_path:
            Path to the locked Minos v2.0 model.

        settings_path:
            Optional N03 settings JSON.

    Output:
        result dictionary containing:
        - node identity
        - model identity
        - document id
        - route records
        - skipped records
        - failed records
        - summary
        - metadata path

    Important:
        N03 does not run OCR.
        N03 only recommends downstream OCR routes.
    """
    # Validate required inputs early.
    check_file_exists(
        refined_groups_path,
        label="N02 refined groups JSON",
    )

    check_file_exists(
        model_path,
        label="Minos model",
    )

    # Load optional settings.
    settings = load_settings(settings_path)

    # Settings can override thresholds.
    thresholds = normalize_thresholds(
        settings.get("thresholds"),
    )

    # Keep a normalized include_rejected value inside settings because helper
    # functions read from settings directly.
    settings["include_rejected"] = settings.get(
        "include_rejected",
        False,
    )

    # By default, reruns clear previous N03 output.
    reset_output = settings.get(
        "reset_output",
        True,
    )

    if reset_output:
        reset_output_dir(output_dir)

    folders = create_output_folders(output_dir)

    # Load N02 refined output.
    refined_payload = load_json(refined_groups_path)

    document_id = refined_payload.get(
        "document_id",
        "unknown_document",
    )

    refined_groups = refined_payload.get(
        "refined_groups",
        [],
    )

    # Load Minos once per document, not once per crop.
    model = load_minos_model(model_path)

    route_records = []
    skipped_records = []
    failed_records = []

    for group in refined_groups:
        # Respect N02 policy/status before running Minos.
        if not should_process_refined_group(group, settings):
            skipped_records.append(
                build_skipped_record(
                    group=group,
                    settings=settings,
                )
            )
            continue

        minos_input_crop_path, minos_input_crop_source = get_minos_input_crop_path(group)

        try:
            check_file_exists(
                minos_input_crop_path,
                label="Minos input crop",
            )

            # Run Minos on the Color Update classification crop.
            classification = classify_crop(
                model=model,
                crop_path=minos_input_crop_path,
                thresholds=thresholds,
            )

            # Copy the exact crop Minos used to the visual class folder.
            routed_crop_path = copy_crop_to_visual_class_folder(
                crop_path=minos_input_crop_path,
                folders=folders,
                visual_class=classification["visual_class"],
            )

            # Build machine-readable route record.
            route_record = build_route_record(
                group=group,
                classification=classification,
                routed_crop_path=routed_crop_path,
            )

            route_records.append(route_record)

        except Exception as error:
            failed_records.append(
                build_failed_record(
                    group=group,
                    error=error,
                )
            )

    summary = summarize_routes(
        route_records=route_records,
        skipped_records=skipped_records,
        failed_records=failed_records,
    )

    # Build final N03 result payload.
    result = {
        "node": NODE_NAME,
        "node_version": NODE_VERSION,

        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_path": model_path,

        "document_id": document_id,

        "refined_groups_path": refined_groups_path,
        "output_dir": output_dir,
        "classified_dir": folders["classified"],
        "metadata_dir": folders["metadata"],

        "thresholds": thresholds,
        "include_rejected": settings["include_rejected"],

        "summary": summary,

        # Main N03 output contract for N04/N05/N06.
        "routes": route_records,

        # Debug/accountability records.
        "skipped": skipped_records,
        "failed": failed_records,
    }

    metadata_path = (
        f"{folders['metadata']}/"
        f"{document_id}_n03_visual_classification_routes.json"
    )

    save_json(
        data=result,
        output_path=metadata_path,
    )

    result["metadata_path"] = metadata_path

    print_summary(
        document_id=document_id,
        summary=summary,
        metadata_path=metadata_path,
    )

    return result
