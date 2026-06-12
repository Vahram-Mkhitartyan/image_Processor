"""Bounding-box geometry helpers for N02."""

def normalize_bbox(raw_bbox):
    """Convert bbox-like input into a clean integer bbox.

    Args:
        raw_bbox: Dictionary containing x1, y1, x2, y2.

    Returns:
        Normalized bbox dictionary with integer coordinates and ordered corners.
    """
    x1 = int(round(raw_bbox["x1"]))
    y1 = int(round(raw_bbox["y1"]))
    x2 = int(round(raw_bbox["x2"]))
    y2 = int(round(raw_bbox["y2"]))

    return {
        "x1": min(x1, x2),
        "y1": min(y1, y2),
        "x2": max(x1, x2),
        "y2": max(y1, y2),
    }


def bbox_width(bbox):
    """Calculate bbox width.

    Args:
        bbox: Dictionary with x1, y1, x2, y2.

    Returns:
        Width in pixels, with a minimum of 1.
    """
    return max(int(bbox["x2"]) - int(bbox["x1"]), 1)


def bbox_height(bbox):
    """Calculate bbox height.

    Args:
        bbox: Dictionary with x1, y1, x2, y2.

    Returns:
        Height in pixels, with a minimum of 1.
    """
    return max(int(bbox["y2"]) - int(bbox["y1"]), 1)


def bbox_area(bbox):
    """Calculate bbox area.

    Args:
        bbox: Dictionary with x1, y1, x2, y2.

    Returns:
        Area in pixels.
    """
    return bbox_width(bbox) * bbox_height(bbox)


def bbox_center_x(bbox):
    """Calculate horizontal bbox center.

    Args:
        bbox: Dictionary with x1, y1, x2, y2.

    Returns:
        Center x-coordinate as a float.
    """
    return (int(bbox["x1"]) + int(bbox["x2"])) / 2.0


def bbox_center_y(bbox):
    """Calculate vertical bbox center.

    Args:
        bbox: Dictionary with x1, y1, x2, y2.

    Returns:
        Center y-coordinate as a float.
    """
    return (int(bbox["y1"]) + int(bbox["y2"])) / 2.0


def bbox_aspect_ratio(bbox):
    """Calculate bbox width/height ratio.

    Args:
        bbox: Dictionary with x1, y1, x2, y2.

    Returns:
        Aspect ratio as a float.
    """
    return bbox_width(bbox) / max(bbox_height(bbox), 1)


def merge_bboxes(bboxes):
    """Build one bbox that contains every input bbox.

    Args:
        bboxes: Iterable of bbox dictionaries.

    Returns:
        Merged bbox dictionary.

    Raises:
        ValueError: If no boxes are provided.
    """
    boxes = list(bboxes)

    if not boxes:
        raise ValueError("Cannot merge an empty bbox list.")

    return {
        "x1": min(int(bbox["x1"]) for bbox in boxes),
        "y1": min(int(bbox["y1"]) for bbox in boxes),
        "x2": max(int(bbox["x2"]) for bbox in boxes),
        "y2": max(int(bbox["y2"]) for bbox in boxes),
    }


def vertical_overlap_ratio(bbox_a, bbox_b):
    """Measure vertical overlap normalized by the smaller bbox height.

    Args:
        bbox_a: First bbox.
        bbox_b: Second bbox.

    Returns:
        Ratio from 0.0 upward, usually 0.0 to 1.0.
    """
    overlap_top = max(int(bbox_a["y1"]), int(bbox_b["y1"]))
    overlap_bottom = min(int(bbox_a["y2"]), int(bbox_b["y2"]))
    overlap = max(0, overlap_bottom - overlap_top)
    smaller_height = max(min(bbox_height(bbox_a), bbox_height(bbox_b)), 1)

    return overlap / smaller_height


def horizontal_overlap_ratio(bbox_a, bbox_b):
    """Measure horizontal overlap normalized by the smaller bbox width.

    Args:
        bbox_a: First bbox.
        bbox_b: Second bbox.

    Returns:
        Ratio from 0.0 upward, usually 0.0 to 1.0.
    """
    overlap_left = max(int(bbox_a["x1"]), int(bbox_b["x1"]))
    overlap_right = min(int(bbox_a["x2"]), int(bbox_b["x2"]))
    overlap = max(0, overlap_right - overlap_left)
    smaller_width = max(min(bbox_width(bbox_a), bbox_width(bbox_b)), 1)

    return overlap / smaller_width


def horizontal_gap(left_bbox, right_bbox):
    """Measure horizontal distance from left_bbox to right_bbox.

    Args:
        left_bbox: Expected left-side bbox.
        right_bbox: Expected right-side bbox.

    Returns:
        Positive gap for separated boxes, 0 for touching boxes, negative value
        when boxes overlap horizontally.
    """
    return int(right_bbox["x1"]) - int(left_bbox["x2"])


def height_ratio(bbox_a, bbox_b):
    """Measure relative height difference between two boxes.

    Args:
        bbox_a: First bbox.
        bbox_b: Second bbox.

    Returns:
        Larger height divided by smaller height.
    """
    smaller = max(min(bbox_height(bbox_a), bbox_height(bbox_b)), 1)
    larger = max(bbox_height(bbox_a), bbox_height(bbox_b))

    return larger / smaller


def clamp_bbox_to_image(bbox, image_shape):
    """Clamp bbox coordinates to image bounds.

    Args:
        bbox: Dictionary with x1, y1, x2, y2.
        image_shape: OpenCV image shape tuple.

    Returns:
        Bbox clipped to the image with at least 1px width and height.
    """
    image_height, image_width = image_shape[:2]

    x1 = max(0, min(int(bbox["x1"]), image_width - 1))
    y1 = max(0, min(int(bbox["y1"]), image_height - 1))
    x2 = max(x1 + 1, min(int(bbox["x2"]), image_width))
    y2 = max(y1 + 1, min(int(bbox["y2"]), image_height))

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def pad_bbox(bbox, padding_px, image_shape):
    """Expand a bbox by a small padding amount and clamp it to the image.

    Args:
        bbox: Dictionary with x1, y1, x2, y2.
        padding_px: Number of pixels to add on each side.
        image_shape: OpenCV image shape tuple.

    Returns:
        Padded and clamped bbox.
    """
    padded = {
        "x1": int(bbox["x1"]) - int(padding_px),
        "y1": int(bbox["y1"]) - int(padding_px),
        "x2": int(bbox["x2"]) + int(padding_px),
        "y2": int(bbox["y2"]) + int(padding_px),
    }

    return clamp_bbox_to_image(padded, image_shape)
