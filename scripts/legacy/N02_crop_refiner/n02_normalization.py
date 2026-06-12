"""Input normalization helpers for N02 source groups."""

from n02_geometry import (
    bbox_area,
    bbox_aspect_ratio,
    bbox_center_x,
    bbox_center_y,
    bbox_height,
    bbox_width,
    normalize_bbox,
)

def extract_bbox_from_group(raw_group):
    """Extract bbox coordinates from either supported N01 group shape.

    Args:
        raw_group: Source group dictionary from N01/ScribeMap.

    Returns:
        Normalized bbox dictionary.

    Raises:
        KeyError: If bbox coordinates are missing.
    """
    if "bbox" in raw_group and raw_group["bbox"] is not None:
        return normalize_bbox(raw_group["bbox"])

    return normalize_bbox(raw_group)


def get_source_group_id(raw_group, fallback_id):
    """Read the original group id while keeping a deterministic fallback.

    Args:
        raw_group: Source group dictionary from N01/ScribeMap.
        fallback_id: ID to use when the source record has no group_id.

    Returns:
        Original group id when present, otherwise fallback_id.
    """
    return raw_group.get("group_id", raw_group.get("id", fallback_id))


def normalize_source_group(raw_group, fallback_id):
    """Convert one N01 group into N02's internal normalized shape.

    Args:
        raw_group: Source group dictionary from N01/ScribeMap.
        fallback_id: Deterministic id used if the source has no group id.

    Returns:
        Normalized group dictionary with geometry metrics and source metadata.
    """
    bbox = extract_bbox_from_group(raw_group)
    source_group_id = get_source_group_id(raw_group, fallback_id)

    return {
        "source_group_id": source_group_id,
        "layer": raw_group.get("layer", "legacy"),
        "source_type": raw_group.get("source_type", "legacy_group"),
        "role_guess": raw_group.get("role_guess"),
        "recommended_next_node": raw_group.get("recommended_next_node"),
        "bbox": bbox,
        "width": bbox_width(bbox),
        "height": bbox_height(bbox),
        "area": bbox_area(bbox),
        "center_x": bbox_center_x(bbox),
        "center_y": bbox_center_y(bbox),
        "aspect_ratio": bbox_aspect_ratio(bbox),
        "density": float(raw_group.get("density", 0.0) or 0.0),
        "component_count": int(raw_group.get("component_count", 1) or 1),
        "classification": dict(raw_group.get("classification", {})),
        "source": raw_group,
    }


def normalize_source_groups(raw_groups):
    """Normalize all N01 groups and sort them in reading-ish order.

    Args:
        raw_groups: Iterable of N01/ScribeMap group dictionaries.

    Returns:
        List of normalized group dictionaries sorted by y1, then x1, then id.
    """
    normalized = [
        normalize_source_group(raw_group, fallback_id=index + 1)
        for index, raw_group in enumerate(raw_groups)
    ]

    return sorted(
        normalized,
        key=lambda group: (
            group["bbox"]["y1"],
            group["bbox"]["x1"],
            group["source_group_id"],
        )
    )


def filter_early_artifacts(normalized_groups, settings):
    """Keep all normalized groups for N02 grouping.

    Args:
        normalized_groups: List of normalized N01 group dictionaries.
        settings: RefinerSettings instance. Kept for API symmetry.

    Returns:
        Tuple of all groups and an empty rejection list. N02 is a grouping
        node now; filtering belongs to the downstream classifier.
    """
    return list(normalized_groups), []


def role_guess_for_layer(layer_name):
    """Assign a simple role guess from a ScribeMap 2.0 color layer."""
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

    if layer_name == "colored":
        return "combined_colored_ink_debug"

    return "unknown_layer"


def collect_scribemap_2_layer_groups(scribemap_result, layers_to_refine):
    """Collect selected ScribeMap 2.0 layer groups as N02 source groups.

    Args:
        scribemap_result: Full ScribeMap result JSON.
        layers_to_refine: Iterable of layer names, usually ["blue"] first.

    Returns:
        Flat list of group dictionaries compatible with N02 normalization.
    """
    layer_results = scribemap_result.get("layer_results", {})
    selected_layers = set(layers_to_refine or [])
    collected = []

    for layer_name, layer_payload in layer_results.items():
        if layer_name not in selected_layers:
            continue

        for index, group in enumerate(layer_payload.get("groups", []), start=1):
            record = dict(group)

            source_id = (
                record.get("group_uid")
                or f"{layer_name}_{record.get('group_id', index):04d}"
            )

            record["group_id"] = source_id
            record["layer"] = layer_name
            record["source_type"] = "scribemap_2_layer_group"
            record["role_guess"] = role_guess_for_layer(layer_name)
            record["recommended_next_node"] = (
                "N05_handwritten_ocr"
                if layer_name == "blue"
                else "review"
            )

            collected.append(record)

    return collected
