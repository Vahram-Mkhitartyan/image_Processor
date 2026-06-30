"""Crop copying and Tesseract-preparation helpers for N04."""

import os

import cv2

from n04_constants import PRINTED_VISUAL_CLASSES
from n04_io import check_file_exists
from n04_routing import get_best_crop_path_for_printed_ocr, get_visual_class


def _safe_bbox(bbox):
    """Return an integer bbox dict, or None if the bbox is unusable."""

    if not isinstance(bbox, dict):
        return None
    try:
        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _clamp_bbox(bbox, width, height, padding=0):
    """Clamp a document bbox into an image shape."""

    return {
        "x1": max(0, int(bbox["x1"]) - padding),
        "y1": max(0, int(bbox["y1"]) - padding),
        "x2": min(int(width), int(bbox["x2"]) + padding),
        "y2": min(int(height), int(bbox["y2"]) + padding),
    }


def _bbox_intersects(a, b):
    """Return True when two bboxes overlap."""

    return (
        int(a["x1"]) < int(b["x2"])
        and int(a["x2"]) > int(b["x1"])
        and int(a["y1"]) < int(b["y2"])
        and int(a["y2"]) > int(b["y1"])
    )


def _bbox_union(boxes):
    """Union a non-empty bbox list."""

    return {
        "x1": min(int(box["x1"]) for box in boxes),
        "y1": min(int(box["y1"]) for box in boxes),
        "x2": max(int(box["x2"]) for box in boxes),
        "y2": max(int(box["y2"]) for box in boxes),
    }


def _document_dir_from_n04_folders(folders):
    """Resolve temp_processing/<document_id> from the N04 output folder."""

    root = os.path.abspath(folders["root"])
    if os.path.basename(root) == "n04_printed_ocr":
        return os.path.dirname(root)
    return os.path.dirname(root)


def _full_mask_paths_for_black_printed_crop(folders):
    """Return full-document masks used by the black printed OCR bridge."""

    document_dir = _document_dir_from_n04_folders(folders)
    mask_dir = os.path.join(document_dir, "n00_file_preparation", "masks")
    return {
        "bbox_mask": os.path.join(mask_dir, "13_black_ink_mask.png"),
        "ocr_source": os.path.join(mask_dir, "19_printed_ocr_tesseract_mask.png"),
        "ocr_fallback": os.path.join(mask_dir, "18_printed_ocr_ink_mask.png"),
    }


def _binary_ink_mask(mask_image):
    """Threshold a grayscale mask into white foreground on black background."""

    if mask_image is None:
        return None
    if mask_image.ndim != 2:
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        mask_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    # Most pipeline masks are already white ink on black. If a mask comes in as
    # black ink on white, invert so connected components still mean "ink".
    border = cv2.hconcat(
        [
            binary[0:1, :],
            binary[-1:, :],
            binary[:, 0:1].reshape(1, -1),
            binary[:, -1:].reshape(1, -1),
        ]
    )
    if float(border.mean()) > 127.5:
        binary = cv2.bitwise_not(binary)
    return binary


def _derive_black_ocr_bbox(route_bbox, black_mask, settings):
    """Expand a route bbox by following nearby black-mask components."""

    height, width = black_mask.shape[:2]
    bbox_width = max(route_bbox["x2"] - route_bbox["x1"], 1)
    bbox_height = max(route_bbox["y2"] - route_bbox["y1"], 1)

    left_context = min(
        int(settings.get("black_bbox_max_left_context_px", 180)),
        max(
            int(settings.get("black_bbox_left_context_px", 48)),
            int(bbox_width * float(settings.get("black_bbox_left_context_ratio", 0.65))),
        ),
    )
    right_context = min(
        int(settings.get("black_bbox_max_right_context_px", 90)),
        max(
            int(settings.get("black_bbox_right_context_px", 24)),
            int(bbox_width * float(settings.get("black_bbox_right_context_ratio", 0.25))),
        ),
    )
    vertical_context = max(
        int(settings.get("black_bbox_vertical_context_px", 6)),
        int(bbox_height * float(settings.get("black_bbox_vertical_context_ratio", 0.2))),
    )

    search_box = _clamp_bbox(
        {
            "x1": route_bbox["x1"] - left_context,
            "y1": route_bbox["y1"] - vertical_context,
            "x2": route_bbox["x2"] + right_context,
            "y2": route_bbox["y2"] + vertical_context,
        },
        width=width,
        height=height,
    )
    row_band = _clamp_bbox(
        {
            "x1": 0,
            "y1": route_bbox["y1"] - vertical_context,
            "x2": width,
            "y2": route_bbox["y2"] + vertical_context,
        },
        width=width,
        height=height,
    )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        black_mask,
        connectivity=8,
    )
    del labels

    min_area = int(settings.get("black_bbox_min_component_area", 2))
    line_width_ratio = float(settings.get("black_bbox_line_width_ratio", 0.82))
    line_aspect_ratio = float(settings.get("black_bbox_line_aspect_ratio", 36.0))
    boxes = [route_bbox]

    for component_index in range(1, component_count):
        x, y, w, h, area = stats[component_index]
        if int(area) < min_area:
            continue
        component_box = {
            "x1": int(x),
            "y1": int(y),
            "x2": int(x + w),
            "y2": int(y + h),
        }
        if not _bbox_intersects(component_box, search_box):
            continue
        if not _bbox_intersects(component_box, row_band):
            continue

        # Printed forms contain long ruling lines. They are black ink too, but
        # they are not text; exclude them before expanding the OCR crop.
        aspect = float(w) / max(float(h), 1.0)
        if (float(w) / max(float(width), 1.0)) >= line_width_ratio:
            continue
        if aspect >= line_aspect_ratio and int(h) <= max(3, bbox_height // 6):
            continue

        boxes.append(component_box)

    derived = _clamp_bbox(
        _bbox_union(boxes),
        width=width,
        height=height,
        padding=int(settings.get("black_bbox_output_padding_px", 3)),
    )
    return derived


def _write_crop(source_image, bbox, output_path):
    """Write a bbox crop and return its absolute path."""

    crop = source_image[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]
    if crop.size == 0:
        return None
    cv2.imwrite(output_path, crop)
    return os.path.abspath(output_path)


def build_black_mask_reflected_ocr_crop(route_record, folders, settings):
    """Build a black-mask bbox crop reflected onto the OCR-friendly mask.

    The black mask is good at deciding *where* printed text lives. The
    printed-OCR mask is better for Tesseract pixels. This bridge keeps both
    truths instead of forcing one mask to do both jobs.
    """

    if not (
        route_record.get("layer") == "black"
        and route_record.get("mask_source") == "black_ink_mask"
    ):
        return None

    route_bbox = _safe_bbox(route_record.get("final_bbox") or route_record.get("bbox"))
    if route_bbox is None:
        return None

    paths = _full_mask_paths_for_black_printed_crop(folders)
    if not os.path.exists(paths["bbox_mask"]):
        return None

    raw_black_mask = cv2.imread(paths["bbox_mask"], cv2.IMREAD_GRAYSCALE)
    black_mask = _binary_ink_mask(raw_black_mask)
    if black_mask is None:
        return None

    ocr_source_path = paths["ocr_source"]
    source_kind = "printed_ocr_tesseract_mask"
    if not os.path.exists(ocr_source_path):
        ocr_source_path = paths["ocr_fallback"]
        source_kind = "printed_ocr_ink_mask"
    if not os.path.exists(ocr_source_path):
        return None

    ocr_source = cv2.imread(ocr_source_path, cv2.IMREAD_GRAYSCALE)
    if ocr_source is None:
        return None

    derived_bbox = _derive_black_ocr_bbox(route_bbox, black_mask, settings or {})
    target_folder = (
        folders["printed_only"]
        if get_visual_class(route_record) == "printed_only"
        else folders["mixed"]
    )
    group_id = str(
        route_record.get("group_id")
        or route_record.get("text_unit_id")
        or route_record.get("source_group_id")
        or (
            f"black_{route_bbox['x1']}_{route_bbox['y1']}"
            f"_{route_bbox['x2']}_{route_bbox['y2']}"
        )
    )
    file_name = f"{group_id}_black_reflected_ocr_crop.png"
    output_path = os.path.join(target_folder, file_name)
    crop_path = _write_crop(ocr_source, derived_bbox, output_path)
    if crop_path is None:
        return None

    route_record["original_document_bbox"] = route_bbox
    route_record["black_mask_derived_bbox"] = derived_bbox
    route_record["black_mask_derived_bbox_reason"] = "black_mask_components_reflected_to_printed_ocr_mask"
    route_record["black_mask_reflected_ocr_crop_path"] = crop_path
    route_record["black_mask_reflected_ocr_source"] = os.path.abspath(ocr_source_path)
    route_record["black_mask_reflected_ocr_source_kind"] = source_kind
    route_record["n04_crop_bbox_source"] = "black_mask_derived_bbox"
    return crop_path


def normalize_dark_ink_on_white(image):
    """Normalize grayscale evidence to dark foreground on a white background.

    Args:
        image: Grayscale crop that may use either binary polarity.

    Returns:
        Grayscale crop whose border/background is white.
    """
    if image.ndim != 2:
        raise ValueError("Tesseract polarity normalization expects grayscale.")

    border_pixels = cv2.hconcat(
        [
            image[0:1, :],
            image[-1:, :],
            image[:, 0:1].reshape(1, -1),
            image[:, -1:].reshape(1, -1),
        ]
    )

    if float(border_pixels.mean()) < 127.5:
        return cv2.bitwise_not(image)

    return image


def copy_candidate_crop_to_n04(route_record, folders, settings=None):
    """
    Resolve the canonical N02 full-text crop selected for N04.

    N04 selected classes:
    - printed_only
    - mixed

    Example output:
        n04_printed_ocr/crops/printed_only/...
        n04_printed_ocr/crops/mixed/...

    Returns:
        Existing crop path, or None if no usable crop path exists.
    """
    visual_class = get_visual_class(route_record)

    if visual_class not in PRINTED_VISUAL_CLASSES:
        return None

    reflected_crop = build_black_mask_reflected_ocr_crop(
        route_record=route_record,
        folders=folders,
        settings=settings or {},
    )
    if reflected_crop:
        return reflected_crop

    crop_path = get_best_crop_path_for_printed_ocr(route_record)

    if crop_path is None:
        return None

    check_file_exists(
        crop_path,
        label="selected printed OCR crop"
    )

    return os.path.abspath(crop_path)


def prepare_crop_for_tesseract(image, scale=3, border=20):
    """
    Prepare a small printed crop for Tesseract.

    Steps:
    1. Enlarge using nearest-neighbor pixel replication.
    2. Add white border so Tesseract has breathing room.
    3. Binarize with Otsu thresholding.

    This is useful because many N04 crops are tiny, and raw Tesseract
    performs poorly on small Armenian printed text.
    """
    normalized = normalize_dark_ink_on_white(image)

    upscaled = cv2.resize(
        normalized,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST
    )

    bordered = cv2.copyMakeBorder(
        upscaled,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=255
    )

    _, thresholded = cv2.threshold(
        bordered,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # The explicit white border is also a final polarity assertion. If a future
    # preprocessing change reverses the image, restore Tesseract's preferred
    # black-text-on-white convention here.
    border_sample = thresholded[0, :]
    if float(border_sample.mean()) < 127.5:
        thresholded = cv2.bitwise_not(thresholded)

    return thresholded


def save_tesseract_ready_crop(crop_path, output_path, scale=3, border=20):
    """
    Load a crop, prepare it for Tesseract, and save the prepared version.

    Input:
        crop_path:
            Original selected N04 crop.

        output_path:
            Path where the prepared crop should be saved.

    Output:
        output_path
    """
    image = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not load crop for Tesseract: {crop_path}")

    prepared = prepare_crop_for_tesseract(
        image=image,
        scale=scale,
        border=border
    )

    cv2.imwrite(output_path, prepared)

    return output_path


def prepare_n04_crop_for_tesseract(route_record, copied_crop_path, folders):
    """
    Create a Tesseract-ready version of the N04 copied crop.

    The original copied crop stays in:
        crops/printed_only/
        crops/mixed/

    The OCR-prepared crop goes into:
        tesseract_ready/printed_only/
        tesseract_ready/mixed/

    Returns:
        Path to the prepared crop.
    """
    visual_class = get_visual_class(route_record)

    if visual_class == "printed_only":
        target_folder = folders["tesseract_ready_printed_only"]

    elif visual_class == "mixed":
        target_folder = folders["tesseract_ready_mixed"]

    else:
        return None

    file_name = os.path.basename(copied_crop_path)

    name, _ = os.path.splitext(file_name)

    prepared_path = f"{target_folder}/{name}_tesseract_ready.png"

    save_tesseract_ready_crop(
        crop_path=copied_crop_path,
        output_path=prepared_path,
        scale=3,
        border=20
    )

    return prepared_path
