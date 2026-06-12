"""JSON, image, crop, and output-path helpers for N02."""

import json
import os
import shutil

import cv2

def load_json(json_path):
    """Load a JSON file from disk.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Parsed JSON data.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, json_path):
    """Save data as pretty JSON.

    Args:
        data: JSON-serializable object to save.
        json_path: Output path.

    Returns:
        The output path.
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    return json_path


def load_gray_image(image_path):
    """Load an image as grayscale.

    Args:
        image_path: Path to an image file.

    Returns:
        Grayscale OpenCV image array.

    Raises:
        ValueError: If OpenCV cannot read the image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not load grayscale image: {image_path}")

    return image


def ensure_dir(path):
    """Create a directory if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The same directory path.
    """
    os.makedirs(path, exist_ok=True)
    return path


def crop_image(gray_image, bbox):
    """Extract a crop from a grayscale image using a bbox.

    Args:
        gray_image: Grayscale OpenCV image array.
        bbox: Dictionary with x1, y1, x2, y2.

    Returns:
        Cropped image array.
    """
    x1 = int(bbox["x1"])
    y1 = int(bbox["y1"])
    x2 = int(bbox["x2"])
    y2 = int(bbox["y2"])

    return gray_image[y1:y2, x1:x2]


def save_crop(gray_image, bbox, crops_dir, text_unit_id, status, layer=None):
    """Save one refined crop into a status-specific folder.

    Args:
        gray_image: Grayscale source image array.
        bbox: Final crop bbox.
        crops_dir: Root refined-crops directory.
        text_unit_id: Numeric text-unit identifier.
        status: One of accepted, review, or rejected.
        layer: Optional ScribeMap layer name for readable filenames.

    Returns:
        Path to the saved crop image.

    Raises:
        ValueError: If OpenCV fails to write the crop.
    """
    status_dir = ensure_dir(os.path.join(crops_dir, status))
    crop = crop_image(gray_image, bbox)
    layer_prefix = f"{layer}_" if layer else ""

    output_name = (
        f"{layer_prefix}text_unit_{int(text_unit_id):04d}"
        f"_x{int(bbox['x1']):04d}_x{int(bbox['x2']):04d}"
        f"_y{int(bbox['y1']):04d}_y{int(bbox['y2']):04d}.jpeg"
    )
    output_path = os.path.join(status_dir, output_name)

    success = cv2.imwrite(output_path, crop)

    if not success:
        raise ValueError(f"Could not save crop: {output_path}")

    return output_path


def resolve_refinement_image_path(payload):
    """Find the image whose coordinates match the N01/ScribeMap bboxes.

    Args:
        payload: Classified groups payload from the pipeline.

    Returns:
        Image path to crop from.
    """
    scribemap_result_path = payload.get("scribemap_result_path")

    if scribemap_result_path and os.path.exists(scribemap_result_path):
        scribemap_result = load_json(scribemap_result_path)
        prepared_path = scribemap_result.get("prepared_bw_image_path")

        if prepared_path and os.path.exists(prepared_path):
            return prepared_path

    artifacts = payload.get("scribemap_artifacts", {})
    bw_input_path = artifacts.get("bw_input")

    if bw_input_path and os.path.exists(bw_input_path):
        return bw_input_path

    return payload["input_bw_image_path"]


def prepare_refined_crops_dir(output_path):
    """Create a clean refined-crops directory beside the metadata folder.

    Args:
        output_path: Final refined metadata JSON path.

    Returns:
        Path to the refined crops root directory.
    """
    metadata_dir = os.path.dirname(output_path)
    document_output_dir = os.path.dirname(metadata_dir)
    refined_crops_dir = os.path.join(document_output_dir, "refined_crops")

    if os.path.exists(refined_crops_dir):
        shutil.rmtree(refined_crops_dir)

    ensure_dir(refined_crops_dir)
    ensure_dir(os.path.join(refined_crops_dir, "accepted"))
    ensure_dir(os.path.join(refined_crops_dir, "review"))
    ensure_dir(os.path.join(refined_crops_dir, "rejected"))

    return refined_crops_dir


def build_debug_preview_path(output_path, document_id):
    """Build the standard N02 debug preview path.

    Args:
        output_path: Final refined metadata JSON path.
        document_id: Document id.

    Returns:
        Debug preview image path.
    """
    metadata_dir = os.path.dirname(output_path)
    document_output_dir = os.path.dirname(metadata_dir)
    debug_dir = ensure_dir(os.path.join(document_output_dir, "debug"))

    return os.path.join(debug_dir, f"{document_id}_n02_refined_boxes_preview.jpeg")
