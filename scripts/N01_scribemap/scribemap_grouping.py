"""
Grouping math for ScribeMap.

This module is intentionally stateless: all behavior is controlled by
the `settings` dict passed into the functions.
"""


def vertical_overlap_ratio(component_a, component_b):
    # Overlap on Y axis, normalized by the smaller component height.
    # 1.0 means full overlap; 0.0 means no vertical overlap.
    """Measure vertical overlap between two components.
    
    Args:
        component_a: First component metadata dictionary.
        component_b: Second component metadata dictionary.
    
    Returns:
        Overlap ratio from 0.0 to 1.0.
    """
    top = max(component_a["y1"], component_b["y1"])
    bottom = min(component_a["y2"], component_b["y2"])
    overlap = max(0, bottom - top)
    min_height = max(min(component_a["height"], component_b["height"]), 1)
    return overlap / min_height


def horizontal_gap(component_a, component_b):
    # Positive: B starts to the right of A (normal reading order).
    # Negative: boxes overlap in X.
    """Measure horizontal distance between two components.
    
    Args:
        component_a: First component metadata dictionary.
        component_b: Second component metadata dictionary.
    
    Returns:
        Horizontal pixel gap between boxes.
    """
    return component_b["x1"] - component_a["x2"]


def center_y_distance(component_a, component_b):
    # Baseline similarity proxy: smaller center distance means same text line.
    """Measure vertical center distance between components.
    
    Args:
        component_a: First component metadata dictionary.
        component_b: Second component metadata dictionary.
    
    Returns:
        Absolute center-y distance in pixels.
    """
    return abs(component_a["center_y"] - component_b["center_y"])


def height_ratio(component_a, component_b):
    # Scale compatibility check between components.
    # Near 1.0 => similar size; larger => potential mismatch/noise.
    """Measure relative height scale between components.
    
    Args:
        component_a: First component metadata dictionary.
        component_b: Second component metadata dictionary.
    
    Returns:
        Ratio of larger height to smaller height.
    """
    smaller = max(min(component_a["height"], component_b["height"]), 1)
    larger = max(component_a["height"], component_b["height"])
    return larger / smaller


def is_wide_line_like(component):
    """Check whether a component was flagged as line-like.
    
    Args:
        component: Component metadata dictionary.
    
    Returns:
        True when the component has a wide-line flag.
    """
    return "wide_line_like" in component.get("shape_flags", [])


def is_small_component_candidate(component):
    """Return True when a component is a tiny satellite candidate."""

    return "small_component_candidate" in component.get("shape_flags", [])


def component_box_distance(component_a, component_b):
    """Return horizontal and vertical bbox gap between two components."""

    horizontal_gap_px = max(
        0,
        max(component_a["x1"], component_b["x1"])
        - min(component_a["x2"], component_b["x2"]),
    )
    vertical_gap_px = max(
        0,
        max(component_a["y1"], component_b["y1"])
        - min(component_a["y2"], component_b["y2"]),
    )
    return horizontal_gap_px, vertical_gap_px


def should_attach_small_component(component_a, component_b, settings):
    """Attach tiny ink fragments only when a nearby real anchor exists.

    Small Armenian marks, eroded tails, and recovered pixels can be meaningful,
    but isolated dust should not become its own crop. This rule lets one tiny
    component attach to a nearby non-tiny anchor while preventing tiny-to-tiny
    chain merges.
    """

    small_a = is_small_component_candidate(component_a)
    small_b = is_small_component_candidate(component_b)
    if small_a == small_b:
        return False

    small = component_a if small_a else component_b
    anchor = component_b if small_a else component_a
    if is_wide_line_like(anchor):
        return False

    max_x_gap = settings.get("satellite_attach_max_x_gap", 18)
    max_y_gap = settings.get("satellite_attach_max_y_gap", 20)
    max_center_distance = settings.get("satellite_attach_max_center_distance", 34)
    max_anchor_height_ratio = settings.get("satellite_attach_max_anchor_height_ratio", 9.0)

    gap_x, gap_y = component_box_distance(small, anchor)
    if gap_x > max_x_gap or gap_y > max_y_gap:
        return False
    if center_y_distance(small, anchor) > max_center_distance:
        return False
    if height_ratio(small, anchor) > max_anchor_height_ratio:
        return False

    return True


def should_connect_components(component_a, component_b, settings):
    """Pairwise edge decision for graph construction.
    
    We combine geometric constraints:
    - horizontal distance
    - baseline consistency
    - vertical overlap
    - relative scale
    - merged box limits (to prevent absurd pair merges)
    
    Args:
        component_a: First component metadata dictionary.
        component_b: Second component metadata dictionary.
        settings: Optional configuration dictionary used to override defaults.
    
    Returns:
        True when the pair should be merged.
    """
    max_horizontal_gap = settings.get("group_max_horizontal_gap", 45)
    y_tolerance = settings.get("group_y_tolerance", 22)
    min_vertical_overlap = settings.get("group_min_vertical_overlap", 0.18)
    max_height_ratio = settings.get("group_max_height_ratio", 3.7)
    ignore_wide_line_like = settings.get("group_ignore_wide_line_like", True)

    if component_b["x1"] < component_a["x1"]:
        # Canonicalize pair order so gap/merge math is consistent.
        component_a, component_b = component_b, component_a

    if settings.get("enable_small_component_attachment", True):
        if should_attach_small_component(component_a, component_b, settings):
            return True

    if is_small_component_candidate(component_a) or is_small_component_candidate(component_b):
        return False

    gap = horizontal_gap(component_a, component_b)
    if gap > max_horizontal_gap:
        return False
    if gap < -max(component_a["width"], component_b["width"]):
        return False
    if center_y_distance(component_a, component_b) > y_tolerance:
        return False
    if vertical_overlap_ratio(component_a, component_b) < min_vertical_overlap:
        return False
    if height_ratio(component_a, component_b) > max_height_ratio:
        return False

    if ignore_wide_line_like:
        if is_wide_line_like(component_a) or is_wide_line_like(component_b):
            return False

    merged_x1 = min(component_a["x1"], component_b["x1"])
    merged_y1 = min(component_a["y1"], component_b["y1"])
    merged_x2 = max(component_a["x2"], component_b["x2"])
    merged_y2 = max(component_a["y2"], component_b["y2"])
    # Pair-level safety limits: even if local rules pass, reject pair
    # that would instantly form an oversized merged box.
    merged_width = merged_x2 - merged_x1
    merged_height = merged_y2 - merged_y1

    if merged_width > settings.get("stage_a_max_pair_merge_width", 370):
        return False
    if merged_height > settings.get("stage_a_max_pair_merge_height", 82):
        return False

    return True


def find_parent(parents, item):
    # Union-find with path compression.
    """Find a union-find root with path compression.
    
    Args:
        parents: Union-find parent mapping.
        item: Union-find item key.
    
    Returns:
        Root item for the union-find set.
    """
    if parents[item] != item:
        parents[item] = find_parent(parents, parents[item])
    return parents[item]


def union_components(parents, ranks, item_a, item_b):
    # Union-by-rank keeps trees shallow for near-constant amortized find().
    """Merge two union-find sets.
    
    Args:
        parents: Union-find parent mapping.
        ranks: Union-find rank mapping.
        item_a: First union-find item key.
        item_b: Second union-find item key.
    
    Returns:
        None.
    """
    root_a = find_parent(parents, item_a)
    root_b = find_parent(parents, item_b)

    if root_a == root_b:
        return

    if ranks[root_a] < ranks[root_b]:
        parents[root_a] = root_b
    elif ranks[root_a] > ranks[root_b]:
        parents[root_b] = root_a
    else:
        parents[root_b] = root_a
        ranks[root_a] += 1


def can_union_without_oversized_group(parents, components, index_a, index_b, settings):
    """Global guard against chain-merging.
    
    Pair checks alone are insufficient because many small valid joins can
    accumulate into one giant cluster. This function simulates the merged set
    and enforces max group dimensions before performing union.
    
    Args:
        parents: Union-find parent mapping.
        components: List of detected micro-component dictionaries.
        index_a: Index of the first component.
        index_b: Index of the second component.
        settings: Optional configuration dictionary used to override defaults.
    
    Returns:
        True when the candidate union is within limits.
    """
    root_a = find_parent(parents, index_a)
    root_b = find_parent(parents, index_b)
    if root_a == root_b:
        return True

    indexes_in_a = []
    indexes_in_b = []

    for index in range(len(components)):
        root = find_parent(parents, index)
        if root == root_a:
            indexes_in_a.append(index)
        if root == root_b:
            indexes_in_b.append(index)

    merged_components = [components[index] for index in (indexes_in_a + indexes_in_b)]

    x1 = min(component["x1"] for component in merged_components)
    y1 = min(component["y1"] for component in merged_components)
    x2 = max(component["x2"] for component in merged_components)
    y2 = max(component["y2"] for component in merged_components)
    merged_width = x2 - x1
    merged_height = y2 - y1

    if merged_width > settings.get("stage_a_max_group_width", 295):
        return False
    if merged_height > settings.get("stage_a_max_group_height", 98):
        return False

    return True


def build_component_groups(components, settings):
    """Build Stage-A groups from micro-components using:
    1) candidate pair scan in reading order
    2) pairwise connect rule
    3) union-find transitive closure
    4) per-group geometry aggregation
    
    Args:
        components: List of detected micro-component dictionaries.
        settings: Optional configuration dictionary used to override defaults.
    
    Returns:
        List of grouped component dictionaries.
    """
    if len(components) == 0:
        return []

    parents = {index: index for index in range(len(components))}
    ranks = {index: 0 for index in range(len(components))}
    sorted_indexes = sorted(range(len(components)), key=lambda i: (components[i]["y1"], components[i]["x1"]))
    comparison_y_window = settings.get("group_comparison_y_window", 64)

    for position, index_a in enumerate(sorted_indexes):
        component_a = components[index_a]

        for index_b in sorted_indexes[position + 1:]:
            component_b = components[index_b]

            if component_b["y1"] - component_a["y2"] > comparison_y_window:
                # Early stop: components farther down page are unlikely
                # to belong to same local text region.
                break

            if should_connect_components(component_a, component_b, settings):
                if can_union_without_oversized_group(parents, components, index_a, index_b, settings):
                    union_components(parents, ranks, index_a, index_b)

    grouped_indexes = {}
    for index in range(len(components)):
        root = find_parent(parents, index)
        grouped_indexes.setdefault(root, []).append(index)

    groups = []
    min_group_components = settings.get("min_group_components", 1)

    for indexes in grouped_indexes.values():
        if len(indexes) < min_group_components:
            continue

        group_components = [components[index] for index in indexes]
        satellite_component_count = sum(
            1 for component in group_components
            if is_small_component_candidate(component)
        )
        x1 = min(component["x1"] for component in group_components)
        y1 = min(component["y1"] for component in group_components)
        x2 = max(component["x2"] for component in group_components)
        y2 = max(component["y2"] for component in group_components)
        width = x2 - x1
        height = y2 - y1
        box_area = max(width * height, 1)
        total_ink_area = sum(component["ink_area"] for component in group_components)
        # Density = fraction of group bounding box occupied by ink pixels.
        # Useful for downstream rejection of filled blobs and structural debris.
        density = total_ink_area / box_area
        # Aspect ratio encodes horizontal-vs-vertical line-likeness.
        aspect_ratio = width / max(height, 1)

        group_flags = []
        if aspect_ratio > settings.get("group_wide_aspect_ratio", 20):
            group_flags.append("wide_group_line_like")
        if aspect_ratio < settings.get("group_vertical_aspect_ratio", 0.2):
            group_flags.append("vertical_group_line_like")
        if len(group_components) == 1:
            group_flags.append("single_component_group")
        if satellite_component_count:
            group_flags.append("has_small_component_satellites")

        groups.append({
            "group_id": len(groups) + 1,
            "component_ids": [component["component_id"] for component in group_components],
            "component_count": len(group_components),
            "satellite_component_count": int(satellite_component_count),
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "width": int(width),
            "height": int(height),
            "box_area": int(box_area),
            "ink_area": int(total_ink_area),
            "density": round(float(density), 4),
            "aspect_ratio": round(float(aspect_ratio), 3),
            "center_x": round(float((x1 + x2) / 2), 2),
            "center_y": round(float((y1 + y2) / 2), 2),
            "group_flags": group_flags
        })

    groups = sorted(groups, key=lambda group: (group["y1"], group["x1"]))
    for index, group in enumerate(groups, start=1):
        group["group_id"] = index

    return groups


def filter_line_like_groups(groups, settings):
    """Final heuristic cleanup.
    
    Removes obvious structural artifacts before ML classification:
    - long thin horizontal groups
    - tall thin vertical groups
    - tiny noise
    - tiny dense blobs
    
    Args:
        groups: List of accepted group dictionaries.
        settings: Optional configuration dictionary used to override defaults.
    
    Returns:
        Tuple of kept groups and rejected groups.
    """
    kept_groups = []
    rejected_groups = []

    min_horizontal_width = settings.get("reject_horizontal_min_width", 18)
    max_horizontal_height = settings.get("reject_horizontal_max_height", 10)
    min_horizontal_aspect = settings.get("reject_horizontal_min_aspect", 5)
    min_vertical_height = settings.get("reject_vertical_min_height", 18)
    max_vertical_width = settings.get("reject_vertical_max_width", 10)
    min_vertical_aspect = settings.get("reject_vertical_min_aspect", 4)
    max_tiny_area = settings.get("reject_tiny_group_max_area", 60)
    max_tiny_width = settings.get("reject_tiny_group_max_width", 10)
    max_tiny_height = settings.get("reject_tiny_group_max_height", 10)
    max_dense_blob_area = settings.get("reject_dense_blob_max_area", 200)
    min_dense_blob_density = settings.get("reject_dense_blob_min_density", 0.80)
    max_square_blob_width = settings.get("reject_square_blob_max_width", 18)
    max_square_blob_height = settings.get("reject_square_blob_max_height", 18)
    max_square_blob_area = settings.get("reject_square_blob_max_area", 260)
    min_square_blob_density = settings.get("reject_square_blob_min_density", 0.50)
    min_square_blob_aspect = settings.get("reject_square_blob_min_aspect", 0.70)
    max_square_blob_aspect = settings.get("reject_square_blob_max_aspect", 1.45)
    max_square_blob_components = settings.get("reject_square_blob_max_components", 2)
    min_hollow_square_width = settings.get("reject_hollow_square_min_width", 24)
    max_hollow_square_width = settings.get("reject_hollow_square_max_width", 40)
    min_hollow_square_height = settings.get("reject_hollow_square_min_height", 20)
    max_hollow_square_height = settings.get("reject_hollow_square_max_height", 36)
    max_hollow_square_area = settings.get("reject_hollow_square_max_area", 1300)
    min_hollow_square_density = settings.get("reject_hollow_square_min_density", 0.08)
    max_hollow_square_density = settings.get("reject_hollow_square_max_density", 0.25)
    min_hollow_square_aspect = settings.get("reject_hollow_square_min_aspect", 0.80)
    max_hollow_square_aspect = settings.get("reject_hollow_square_max_aspect", 1.50)
    max_hollow_square_components = settings.get("reject_hollow_square_max_components", 3)
    max_group_width_limit = settings.get("reject_oversized_group_max_width", 350)
    max_group_height_limit = settings.get("reject_oversized_group_max_height", 122)
    max_group_area_limit = settings.get("reject_oversized_group_max_area", 34000)

    for group in groups:
        width = group["width"]
        height = group["height"]
        area = group["box_area"]
        density = group.get("density", 0)
        horizontal_aspect = width / max(height, 1)
        vertical_aspect = height / max(width, 1)
        component_count = group.get("component_count", 1)
        reject_reason = None

        if (
            width > max_group_width_limit
            or height > max_group_height_limit
            or area > max_group_area_limit
        ):
            reject_reason = "oversized_group"
        elif width >= min_horizontal_width and height <= max_horizontal_height and horizontal_aspect >= min_horizontal_aspect:
            reject_reason = "horizontal_line_like_group"
        elif height >= min_vertical_height and width <= max_vertical_width and vertical_aspect >= min_vertical_aspect:
            reject_reason = "vertical_line_like_group"
        elif area <= max_tiny_area and width <= max_tiny_width and height <= max_tiny_height:
            reject_reason = "tiny_isolated_group"
        elif area <= max_dense_blob_area and density >= min_dense_blob_density:
            reject_reason = "small_dense_blob"
        elif (
            width <= max_square_blob_width
            and height <= max_square_blob_height
            and area <= max_square_blob_area
            and density >= min_square_blob_density
            and min_square_blob_aspect <= horizontal_aspect <= max_square_blob_aspect
            and component_count <= max_square_blob_components
        ):
            reject_reason = "small_dense_square_blob"
        elif (
            min_hollow_square_width <= width <= max_hollow_square_width
            and min_hollow_square_height <= height <= max_hollow_square_height
            and area <= max_hollow_square_area
            and min_hollow_square_density <= density <= max_hollow_square_density
            and min_hollow_square_aspect <= horizontal_aspect <= max_hollow_square_aspect
            and component_count <= max_hollow_square_components
        ):
            reject_reason = "small_hollow_square_blob"

        if reject_reason is not None:
            rejected_group = dict(group)
            rejected_group["reject_reason"] = reject_reason
            rejected_groups.append(rejected_group)
        else:
            kept_groups.append(group)

    for index, group in enumerate(kept_groups, start=1):
        group["group_id"] = index

    return kept_groups, rejected_groups
