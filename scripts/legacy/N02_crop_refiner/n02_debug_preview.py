"""Debug preview rendering for N02 refined boxes."""

import os

import cv2
import numpy as np

from n02_io import ensure_dir

def status_color(status):
    """Map a refined status to an OpenCV BGR color.

    Args:
        status: accepted, review, rejected, or unknown.

    Returns:
        OpenCV BGR color tuple.
    """
    if status == "accepted":
        return (0, 200, 0)

    if status == "review":
        return (0, 165, 255)

    if status == "rejected":
        return (0, 0, 255)

    return (255, 0, 255)


def layer_color(layer):
    """Map a ScribeMap layer to a visible debug-preview color.

    Args:
        layer: ScribeMap layer name.

    Returns:
        OpenCV BGR color tuple.
    """
    if layer == "blue":
        return (255, 0, 0)

    if layer == "red":
        return (0, 0, 255)

    if layer == "green":
        return (0, 180, 0)

    if layer == "black":
        # Cyan makes black-layer boxes visible over dark ink.
        return (255, 255, 0)

    if layer == "unknown_color":
        return (0, 255, 255)

    return (255, 0, 255)


def layer_label(layer):
    """Return a compact, unambiguous layer label.

    Args:
        layer: ScribeMap layer name.

    Returns:
        Short uppercase label for previews.
    """
    labels = {
        "blue": "BLUE",
        "red": "RED",
        "green": "GREEN",
        "black": "BLACK",
        "unknown_color": "UNK",
        "legacy": "LEG",
    }

    return labels.get(layer, str(layer or "LEG").upper())


def layer_slug(layer):
    """Return a filesystem-friendly layer name.

    Args:
        layer: ScribeMap layer name.

    Returns:
        Stable layer slug for files and JSON keys.
    """
    if layer == "unknown_color":
        return "other_color"

    return str(layer or "legacy")


def groups_by_layer(refined_groups):
    """Group refined records by ScribeMap layer.

    Args:
        refined_groups: List of final refined group dictionaries.

    Returns:
        Dictionary keyed by filesystem-friendly layer slug.
    """
    grouped = {}

    for group in refined_groups:
        slug = layer_slug(group.get("layer", "legacy"))
        grouped.setdefault(slug, []).append(group)

    return grouped


def draw_labeled_box(image, bbox, label, color, thickness=2):
    """Draw one labeled bbox on a debug image.

    Args:
        image: OpenCV color image to draw on.
        bbox: Bbox dictionary.
        label: Text label.
        color: OpenCV BGR color tuple.
        thickness: Rectangle line thickness.

    Returns:
        None. The image is modified in place.
    """
    x1 = int(bbox["x1"])
    y1 = int(bbox["y1"])
    x2 = int(bbox["x2"])
    y2 = int(bbox["y2"])

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        image,
        str(label),
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_mask_box(mask, bbox):
    """Draw one filled rectangle onto a binary debug mask.

    Args:
        mask: Single-channel debug mask image.
        bbox: Bbox dictionary.

    Returns:
        None. The mask is modified in place.
    """
    x1 = int(bbox["x1"])
    y1 = int(bbox["y1"])
    x2 = int(bbox["x2"])
    y2 = int(bbox["y2"])

    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)


def save_debug_image(image, output_path):
    """Save one debug image with explicit error handling.

    Args:
        image: OpenCV image array.
        output_path: Path where the image should be saved.

    Returns:
        Output path.
    """
    ensure_dir(os.path.dirname(output_path))

    if not cv2.imwrite(output_path, image):
        raise ValueError(f"Could not save debug image: {output_path}")

    return output_path


def render_debug_preview(image_path, refined_groups, output_path):
    """Render final N02 boxes back onto the source image.

    Args:
        image_path: Source/prepared image path.
        refined_groups: List of final refined group dictionaries.
        output_path: Path where the preview image should be saved.

    Returns:
        Output preview path.

    Raises:
        ValueError: If the image cannot be read or written.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not load debug preview image: {image_path}")

    for group in refined_groups:
        refiner = group.get("refiner", {})
        bbox = refiner.get("final_bbox") or group.get("final_bbox")

        if bbox is None:
            continue

        status = refiner.get("status", "review")
        score = int(refiner.get("final_score", group.get("quality_score", 0)))
        text_unit_id = group.get("text_unit_id", "?")
        source_count = group.get("source_group_count", 1)
        layer = str(group.get("layer", "legacy"))
        label = f"{layer_label(layer)}-TU{text_unit_id}x{source_count}:{status[:1]}:{score}"

        draw_labeled_box(
            image=image,
            bbox=bbox,
            label=label,
            color=layer_color(layer),
            thickness=3 if layer == "black" else 2,
        )

    ensure_dir(os.path.dirname(output_path))

    if not cv2.imwrite(output_path, image):
        raise ValueError(f"Could not save debug preview: {output_path}")

    return output_path


def render_layer_debug_outputs(image_path, refined_groups, output_dir, document_id):
    """Render separate debug masks and previews for each ScribeMap layer.

    Args:
        image_path: Source/prepared image path.
        refined_groups: List of final refined group dictionaries.
        output_dir: N02 debug output directory.
        document_id: Current document id.

    Returns:
        Dictionary keyed by layer slug with preview/mask paths and counts.
    """
    base_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if base_image is None:
        raise ValueError(f"Could not load layer debug image: {image_path}")

    image_height, image_width = base_image.shape[:2]
    layer_outputs = {}

    for slug, layer_groups in sorted(groups_by_layer(refined_groups).items()):
        preview = base_image.copy()
        mask = np.zeros((image_height, image_width), dtype="uint8")

        for group in layer_groups:
            refiner = group.get("refiner", {})
            bbox = refiner.get("final_bbox") or group.get("final_bbox")

            if bbox is None:
                continue

            status = refiner.get("status", "review")
            score = int(refiner.get("final_score", group.get("quality_score", 0)))
            text_unit_id = group.get("text_unit_id", "?")
            source_count = group.get("source_group_count", 1)
            layer = group.get("layer", slug)
            label = f"{layer_label(layer)}-TU{text_unit_id}x{source_count}:{status[:1]}:{score}"

            draw_labeled_box(
                image=preview,
                bbox=bbox,
                label=label,
                color=layer_color(layer),
                thickness=3 if layer == "black" else 2,
            )
            draw_mask_box(mask, bbox)

        preview_path = os.path.join(
            output_dir,
            "layer_previews",
            f"{document_id}_n02_{slug}_boxes_preview.jpeg",
        )
        mask_path = os.path.join(
            output_dir,
            "layer_masks",
            f"{document_id}_n02_{slug}_boxes_mask.png",
        )

        layer_outputs[slug] = {
            "layer": slug,
            "group_count": len(layer_groups),
            "preview_path": save_debug_image(preview, preview_path),
            "mask_path": save_debug_image(mask, mask_path),
        }

    return layer_outputs
