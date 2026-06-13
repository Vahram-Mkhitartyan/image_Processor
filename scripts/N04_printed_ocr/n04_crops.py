"""Crop copying and Tesseract-preparation helpers for N04."""

import os

import cv2

from n04_constants import PRINTED_VISUAL_CLASSES
from n04_io import check_file_exists
from n04_routing import get_best_crop_path_for_printed_ocr, get_visual_class


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


def copy_candidate_crop_to_n04(route_record, folders):
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
