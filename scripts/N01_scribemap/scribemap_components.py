import cv2


def detect_micro_components(content_ink_mask, settings):
    """Connected-component extraction over the prepared content ink mask.
    
    Input mask convention:
    - foreground ink pixels are non-zero
    - background is zero
    
    Args:
        content_ink_mask: Binary content-ink mask produced by file preparation.
        settings: Optional configuration dictionary used to override defaults.
    
    Returns:
        Computed result for the caller.
    """
    # OpenCV returns per-label stats and centroids for each connected blob.
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        content_ink_mask,
        connectivity=8
    )

    components = []

    min_area = settings.get("min_component_area", 15)
    max_area = settings.get("max_component_area", 999000)
    min_width = settings.get("min_component_width", 1)
    min_height = settings.get("min_component_height", 1)
    wide_line_aspect_ratio = settings.get("wide_line_aspect_ratio", 20)
    vertical_line_aspect_ratio = settings.get("vertical_line_aspect_ratio", 0.2)

    for label_id in range(1, num_labels):
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        if w < min_width or h < min_height:
            continue

        shape_flags = []

        if area > max_area:
            shape_flags.append("large_component")

        aspect_ratio = w / max(h, 1)

        if aspect_ratio > wide_line_aspect_ratio:
            shape_flags.append("wide_line_like")

        if aspect_ratio < vertical_line_aspect_ratio:
            shape_flags.append("vertical_line_like")

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)
        box_area = max(w * h, 1)
        # Density ~= how filled the component box is.
        # Very low values often indicate thin strokes;
        # high values can indicate dots/blobs.
        density = area / box_area
        center_x, center_y = centroids[label_id]

        components.append({
            "component_id": len(components) + 1,
            "label_id": int(label_id),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": int(w),
            "height": int(h),
            "ink_area": int(area),
            "box_area": int(box_area),
            "density": round(float(density), 4),
            "aspect_ratio": round(float(aspect_ratio), 3),
            "shape_flags": shape_flags,
            "center_x": round(float(center_x), 2),
            "center_y": round(float(center_y), 2)
        })

    components = sorted(
        components,
        key=lambda component: (component["y1"], component["x1"])
    )

    return components
