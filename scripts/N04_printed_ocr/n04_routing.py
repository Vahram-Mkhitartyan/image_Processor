"""N03 route loading, filtering, and bbox helpers for N04."""

from n04_constants import PRINTED_VISUAL_CLASSES
from n04_io import check_file_exists, load_json

def load_n03_visual_routes(visual_routes_path):
    """
    Load N03 visual classification routes JSON.

    N04 consumes N03 output as its input contract.

    Expected N03 fields:
    - document_id
    - routes
    - summary
    """
    check_file_exists(
        visual_routes_path,
        label="N03 visual routes JSON"
    )

    payload = load_json(visual_routes_path)

    if "routes" not in payload:
        raise KeyError("N03 routes JSON has no 'routes' key.")

    return payload


def get_visual_class(route_record):
    """
    Extract the visual class from one N03 route record.

    Expected values:
    - mixed
    - printed_only
    - handwriting_only
    - empty_or_noise
    - review
    """
    visual_info = route_record.get("visual_classification", {})

    return visual_info.get("visual_class")


def should_send_to_printed_ocr(route_record):
    """
    Decide whether this N03 route should enter N04.

    N04 printed OCR only needs:
    - printed_only
    - mixed

    It ignores:
    - handwriting_only
    - empty_or_noise
    - review
    """
    visual_class = get_visual_class(route_record)

    return visual_class in PRINTED_VISUAL_CLASSES


def select_printed_candidates(route_records):
    """
    Select all N03 routes that should be included in printed text mapping.

    Input:
        N03 route records

    Output:
        list of route records where visual_class is printed_only or mixed
    """
    selected = []

    for route_record in route_records:
        if should_send_to_printed_ocr(route_record):
            selected.append(route_record)

    return selected    


def build_document_bbox(route_record):
    """
    Build a clean document-level bounding box.

    Color Update rule:
    Prefer final_bbox because N02 may expand/sanitize the original bbox.
    Fall back to bbox only if final_bbox is missing.
    """
    bbox = route_record.get("final_bbox") or route_record.get("bbox")

    if bbox is None:
        return None

    x1 = int(bbox.get("x1", 0))
    y1 = int(bbox.get("y1", 0))
    x2 = int(bbox.get("x2", 0))
    y2 = int(bbox.get("y2", 0))

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": max(x2 - x1, 0),
        "height": max(y2 - y1, 0)
    }

def build_crop_bbox(route_record):
    """
    Build the crop-level bounding box if available.

    crop_bbox describes the crop-local coordinate space produced earlier
    in the pipeline.

    If it is missing, return None.
    """
    crop_bbox = route_record.get("crop_bbox")

    if crop_bbox is None:
        return None

    x1 = int(crop_bbox.get("x1", 0))
    y1 = int(crop_bbox.get("y1", 0))
    x2 = int(crop_bbox.get("x2", 0))
    y2 = int(crop_bbox.get("y2", 0))

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": max(x2 - x1, 0),
        "height": max(y2 - y1, 0)
    }


def get_best_crop_path_for_printed_ocr(route_record):
    """
    Pick the crop path N04 should use for printed OCR.

    Color Update priority:
    1. classification_crop_path
       - N02 target-layer-only crop on white background.
       - Best direct input for OCR.

    2. routed_crop_path
       - N03 copy of the same crop inside classified folders.
       - Good debug fallback.

    3. analysis_crop_path
       - Usually same visual content as classification crop.

    4. refined_crop_path
       - Backward compatibility.

    5. original/source crop fallbacks.
       - These may contain extra layers and should not be preferred.
    """
    crop_keys = [
        "classification_crop_path",
        "routed_crop_path",
        "analysis_crop_path",
        "refined_crop_path",
        "original_crop_path",
        "source_crop_path",
    ]

    for key in crop_keys:
        crop_path = route_record.get(key)

        if crop_path:
            return crop_path

    return None