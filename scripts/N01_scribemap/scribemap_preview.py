import cv2


def normalize_to_grayscale(image):
    # Accept either single-channel or BGR input and normalize to 1 channel.
    """Normalize an image to one grayscale channel.
    
    Args:
        image: Input image array.
    
    Returns:
        Grayscale image array.
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def draw_components_preview(bw_image, components):
    # Green boxes for raw connected components.
    """Draw component boxes for visual debugging.
    
    Args:
        bw_image: Black/white or grayscale source image array.
        components: List of detected micro-component dictionaries.
    
    Returns:
        Preview image with component boxes.
    """
    gray = normalize_to_grayscale(bw_image)
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for component in components:
        cv2.rectangle(
            preview,
            (component["x1"], component["y1"]),
            (component["x2"], component["y2"]),
            (0, 255, 0),
            1
        )

    return preview


def draw_groups_preview(bw_image, groups):
    # Red boxes + IDs for grouped regions.
    """Draw accepted group boxes for visual debugging.
    
    Args:
        bw_image: Black/white or grayscale source image array.
        groups: List of accepted group dictionaries.
    
    Returns:
        Preview image with group boxes.
    """
    gray = normalize_to_grayscale(bw_image)
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for group in groups:
        cv2.rectangle(
            preview,
            (group["x1"], group["y1"]),
            (group["x2"], group["y2"]),
            (0, 0, 255),
            2
        )
        cv2.putText(
            preview,
            str(group["group_id"]),
            (group["x1"], max(group["y1"] - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1
        )

    return preview


def draw_rejected_groups_preview(bw_image, rejected_groups):
    # Magenta boxes + short reason tags for rejected groups.
    """Draw rejected group boxes and reason labels.
    
    Args:
        bw_image: Black/white or grayscale source image array.
        rejected_groups: List of rejected group dictionaries.
    
    Returns:
        Preview image with rejected group boxes.
    """
    gray = normalize_to_grayscale(bw_image)
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for group in rejected_groups:
        x1, y1, x2, y2 = group["x1"], group["y1"], group["x2"], group["y2"]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            preview,
            group.get("reject_reason", "rejected")[:8],
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 0, 255),
            1
        )

    return preview


def draw_line_masks_preview(
    bw_image,
    horizontal_line_mask,
    short_horizontal_line_mask,
    combined_horizontal_line_mask,
    vertical_line_mask,
    grouped_vertical_line_mask
):
    # Overlays each structural mask with a distinct color, so mask interactions
    # can be inspected visually when tuning line-removal thresholds.
    """Overlay structural line masks on the page preview.
    
    Args:
        bw_image: Black/white or grayscale source image array.
        horizontal_line_mask: Binary mask containing long horizontal line candidates.
        short_horizontal_line_mask: Binary mask containing short horizontal line candidates.
        combined_horizontal_line_mask: Binary mask containing all horizontal line candidates.
        vertical_line_mask: Binary mask containing vertical line candidates.
        grouped_vertical_line_mask: Binary mask containing selected grouped vertical lines.
    
    Returns:
        Preview image with mask contours overlaid.
    """
    gray = normalize_to_grayscale(bw_image)
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    mask_specs = [
        (horizontal_line_mask, (255, 0, 0), 2),
        (short_horizontal_line_mask, (0, 0, 255), 1),
        (combined_horizontal_line_mask, (0, 255, 0), 1),
        (vertical_line_mask, (0, 255, 255), 2),
        (grouped_vertical_line_mask, (255, 0, 255), 2),
    ]

    for mask, color, thickness in mask_specs:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(preview, (x, y), (x + w, y + h), color, thickness)

    return preview
