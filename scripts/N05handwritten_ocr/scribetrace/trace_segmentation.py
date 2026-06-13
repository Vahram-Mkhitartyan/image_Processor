"""Propose character boundaries from raster valleys and skeleton bottlenecks."""

from dataclasses import dataclass

import cv2
import numpy as np

from .trace_common import edge_key
from .trace_paths import TracePathExtractor
from .trace_skeleton import (
    SkeletonGraph,
    SkeletonPointExtractor,
    TraceSkeletonizer,
)


DEFAULT_SEGMENTATION_SETTINGS = {
    "enabled": True,
    "maximum_hypotheses": 5,
    "minimum_unit_width_px": 24,
    "minimum_unit_aspect_ratio": 1.60,
    "minimum_segment_width_ratio": 0.15,
    "minimum_segment_ink_ratio": 0.12,
    "maximum_projection_ratio": 0.40,
    "maximum_skeleton_crossings": 2,
    "cut_search_radius_px": 2,
    "minimum_cut_spacing_ratio": 0.08,
    "floating_component_max_area_ratio": 0.18,
    "floating_component_max_height_ratio": 0.45,
    "floating_attachment_max_horizontal_gap_px": 12,
    "minimum_component_ink_pixels": 2,
    "minimum_vector_subgraph_points": 2,
    "minimum_vector_subgraph_ratio": 0.15,
    "minimum_side_dominance": 0.75,
    "minimum_junction_distance_px": 2,
}


@dataclass(frozen=True)
class ComponentRecord:
    """Store deterministic connected-component geometry."""

    component_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    ink_pixels: int
    is_floating: bool = False
    attached_to_component_id: int | None = None

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height

    @property
    def center_x(self):
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self):
        return (self.y1 + self.y2) / 2.0

    def to_dict(self):
        """Return JSON-safe component and attachment evidence."""
        return {
            "component_id": self.component_id,
            "bbox": {
                "x1": self.x1,
                "y1": self.y1,
                "x2": self.x2,
                "y2": self.y2,
            },
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "ink_pixels": self.ink_pixels,
            "is_floating": self.is_floating,
            "attached_to_component_id": self.attached_to_component_id,
        }


def normalize_segmentation_settings(settings=None):
    """Merge and validate ScribeTrace character-segmentation settings."""
    normalized = dict(DEFAULT_SEGMENTATION_SETTINGS)
    if settings:
        normalized.update(settings)

    integer_keys = (
        "maximum_hypotheses",
        "minimum_unit_width_px",
        "maximum_skeleton_crossings",
        "cut_search_radius_px",
        "floating_attachment_max_horizontal_gap_px",
        "minimum_component_ink_pixels",
        "minimum_vector_subgraph_points",
        "minimum_junction_distance_px",
    )
    ratio_keys = (
        "minimum_unit_aspect_ratio",
        "minimum_segment_width_ratio",
        "minimum_segment_ink_ratio",
        "maximum_projection_ratio",
        "minimum_cut_spacing_ratio",
        "floating_component_max_area_ratio",
        "floating_component_max_height_ratio",
        "minimum_vector_subgraph_ratio",
        "minimum_side_dominance",
    )

    for key in integer_keys:
        normalized[key] = int(normalized[key])
        if normalized[key] < 0:
            raise ValueError(f"{key} must be non-negative.")
    for key in ratio_keys:
        normalized[key] = float(normalized[key])
        if normalized[key] < 0:
            raise ValueError(f"{key} must be non-negative.")

    if normalized["maximum_hypotheses"] < 1:
        raise ValueError("maximum_hypotheses must be at least 1.")
    if not 0 < normalized["minimum_segment_width_ratio"] < 0.5:
        raise ValueError("minimum_segment_width_ratio must be between 0 and 0.5.")
    if not 0 < normalized["minimum_segment_ink_ratio"] < 0.5:
        raise ValueError("minimum_segment_ink_ratio must be between 0 and 0.5.")
    if not 0 <= normalized["maximum_projection_ratio"] <= 1:
        raise ValueError("maximum_projection_ratio must be between 0 and 1.")
    if not 0 < normalized["minimum_vector_subgraph_ratio"] < 0.5:
        raise ValueError("minimum_vector_subgraph_ratio must be between 0 and 0.5.")
    if not 0.5 <= normalized["minimum_side_dominance"] <= 1:
        raise ValueError("minimum_side_dominance must be between 0.5 and 1.")
    return normalized


def _horizontal_gap(component, candidate):
    """Return the horizontal distance between two component boxes."""
    if component.x2 < candidate.x1:
        return candidate.x1 - component.x2
    if candidate.x2 < component.x1:
        return component.x1 - candidate.x2
    return 0


def _extract_components(mask, settings):
    """Extract components and attach small upper marks to lower bodies."""
    height, width = mask.shape[:2]
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )
    raw = []
    for label in range(1, count):
        ink_pixels = int(stats[label, cv2.CC_STAT_AREA])
        if ink_pixels < settings["minimum_component_ink_pixels"]:
            continue
        x1 = int(stats[label, cv2.CC_STAT_LEFT])
        y1 = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        raw.append(
            ComponentRecord(
                component_id=len(raw),
                x1=x1,
                y1=y1,
                x2=x1 + component_width,
                y2=y1 + component_height,
                ink_pixels=ink_pixels,
            )
        )

    if not raw:
        return []

    largest_area = max(component.area for component in raw)
    main_components = []
    floating_ids = set()
    for component in raw:
        small_area = (
            component.area
            <= largest_area * settings["floating_component_max_area_ratio"]
        )
        short_height = (
            component.height
            <= height * settings["floating_component_max_height_ratio"]
        )
        has_body_below = any(
            candidate.component_id != component.component_id
            and candidate.center_y > component.center_y
            and candidate.ink_pixels > component.ink_pixels
            for candidate in raw
        )
        if small_area and short_height and has_body_below:
            floating_ids.add(component.component_id)
        else:
            main_components.append(component)

    if not main_components:
        main_components = [max(raw, key=lambda item: (item.ink_pixels, -item.component_id))]
        floating_ids.discard(main_components[0].component_id)

    attached = []
    for component in raw:
        if component.component_id not in floating_ids:
            attached.append(component)
            continue

        candidates = [
            candidate
            for candidate in main_components
            if candidate.center_y > component.center_y
            and _horizontal_gap(component, candidate)
            <= settings["floating_attachment_max_horizontal_gap_px"]
        ]
        if not candidates:
            candidates = [
                candidate
                for candidate in main_components
                if candidate.center_y > component.center_y
            ]
        parent = min(
            candidates,
            key=lambda candidate: (
                _horizontal_gap(component, candidate),
                abs(component.center_x - candidate.center_x),
                candidate.y1,
                candidate.x1,
                candidate.component_id,
            ),
            default=None,
        )
        attached.append(
            ComponentRecord(
                component_id=component.component_id,
                x1=component.x1,
                y1=component.y1,
                x2=component.x2,
                y2=component.y2,
                ink_pixels=component.ink_pixels,
                is_floating=True,
                attached_to_component_id=(
                    parent.component_id if parent is not None else None
                ),
            )
        )

    return sorted(attached, key=lambda item: item.component_id)


def _protected_cut_ranges(components):
    """Return x-ranges where a cut would slice a detached attached mark."""
    return [
        (component.x1, component.x2, component.component_id)
        for component in components
        if component.is_floating and component.attached_to_component_id is not None
    ]


def _local_minimum_positions(values, start, end):
    """Return stable local minima, collapsing equal-valued runs."""
    minima = []
    x = start
    while x < end:
        run_start = x
        run_value = int(values[x])
        while x + 1 < end and int(values[x + 1]) == run_value:
            x += 1
        run_end = x
        left_value = int(values[run_start - 1]) if run_start > start else run_value
        right_value = int(values[run_end + 1]) if run_end + 1 < end else run_value
        if run_value <= left_value and run_value <= right_value:
            minima.append((run_start + run_end) // 2)
        x += 1
    return minima


def _best_cut_in_window(
    candidate_x,
    projection,
    skeleton_projection,
    start,
    end,
    radius,
):
    """Move a valley candidate onto the least destructive nearby column."""
    possible = range(
        max(start, candidate_x - radius),
        min(end, candidate_x + radius + 1),
    )
    return min(
        possible,
        key=lambda x: (
            int(skeleton_projection[x]),
            int(projection[x]),
            abs(x - candidate_x),
            x,
        ),
    )


def _graph_components(graph, allowed_coordinates=None, removed_edges=None):
    """Return deterministic graph components after optional virtual cuts."""
    allowed = (
        set(graph.point_lookup)
        if allowed_coordinates is None
        else set(allowed_coordinates)
    )
    removed = set(removed_edges or ())
    unvisited = set(allowed)
    components = []
    while unvisited:
        start = min(unvisited, key=lambda value: (value[1], value[0]))
        unvisited.remove(start)
        stack = [start]
        component = []
        while stack:
            coordinate = stack.pop()
            component.append(coordinate)
            point = graph.point_lookup[coordinate]
            for neighbor in reversed(graph.neighbors_of(point)):
                neighbor_coordinate = neighbor.to_tuple()
                if (
                    neighbor_coordinate in unvisited
                    and neighbor_coordinate in allowed
                    and edge_key(point, neighbor) not in removed
                ):
                    unvisited.remove(neighbor_coordinate)
                    stack.append(neighbor_coordinate)
        components.append(sorted(component, key=lambda value: (value[1], value[0])))
    components.sort(
        key=lambda component: (
            min(value[0] for value in component),
            min(value[1] for value in component),
            -len(component),
        )
    )
    return components


def _edge_path_lookup(trace_paths):
    """Map every traversed skeleton edge to its ordered path location."""
    lookup = {}
    for path in trace_paths:
        for index, (point_a, point_b) in enumerate(
            zip(path.points, path.points[1:])
        ):
            lookup.setdefault(edge_key(point_a, point_b), []).append(
                {
                    "path_id": path.path_id,
                    "edge_index": index,
                    "is_closed": path.is_closed,
                }
            )
    return lookup


def _crossing_edges(graph, cut_x):
    """Return graph edges crossing the vertical boundary before cut_x."""
    return [
        edge
        for edge in graph.all_edges()
        if (
            (edge[0][0] < cut_x <= edge[1][0])
            or (edge[1][0] < cut_x <= edge[0][0])
        )
    ]


def _side_metrics(coordinates, cut_x):
    """Measure how strongly one vector subgraph belongs to each side."""
    total = max(1, len(coordinates))
    left_count = sum(x < cut_x for x, _ in coordinates)
    right_count = total - left_count
    return {
        "point_count": len(coordinates),
        "left_point_count": left_count,
        "right_point_count": right_count,
        "left_dominance": float(left_count / total),
        "right_dominance": float(right_count / total),
        "bbox": {
            "x1": min(x for x, _ in coordinates),
            "y1": min(y for _, y in coordinates),
            "x2": max(x for x, _ in coordinates) + 1,
            "y2": max(y for _, y in coordinates) + 1,
        },
    }


def _path_ids_for_coordinates(trace_paths, coordinates):
    """Return paths with at least one point inside a vector subgraph."""
    coordinate_set = set(coordinates)
    return sorted(
        path.path_id
        for path in trace_paths
        if any(point.to_tuple() in coordinate_set for point in path.points)
    )


def _near_junction(graph, crossing_edges, maximum_distance):
    """Reject connector cuts occurring at or immediately beside a junction."""
    junction_coordinates = [
        point.to_tuple()
        for point in graph.junctions()
    ]
    return any(
        max(abs(x - junction_x), abs(y - junction_y)) <= maximum_distance
        for edge in crossing_edges
        for x, y in edge
        for junction_x, junction_y in junction_coordinates
    )


def _validate_disconnected_groups(graph, trace_paths, cut_x, settings):
    """Validate a blank gap through existing left/right vector components."""
    components = _graph_components(graph)
    left_components = []
    right_components = []
    for component in components:
        metrics = _side_metrics(component, cut_x)
        if metrics["left_dominance"] >= settings["minimum_side_dominance"]:
            left_components.append(component)
        elif metrics["right_dominance"] >= settings["minimum_side_dominance"]:
            right_components.append(component)
        else:
            return None

    left_coordinates = [point for group in left_components for point in group]
    right_coordinates = [point for group in right_components for point in group]
    if (
        len(left_coordinates) < settings["minimum_vector_subgraph_points"]
        or len(right_coordinates) < settings["minimum_vector_subgraph_points"]
    ):
        return None

    return {
        "mode": "disconnected_vector_groups",
        "connector_path_id": None,
        "split_after_point_index": None,
        "crossing_edges": [],
        "left_subgraph": _side_metrics(left_coordinates, cut_x),
        "right_subgraph": _side_metrics(right_coordinates, cut_x),
        "left_path_ids": _path_ids_for_coordinates(
            trace_paths, left_coordinates
        ),
        "right_path_ids": _path_ids_for_coordinates(
            trace_paths, right_coordinates
        ),
    }


def _validate_connector_path(graph, trace_paths, path_lookup, cut_x, settings):
    """Virtually cut one path edge and require two coherent vector subgraphs."""
    crossing_edges = _crossing_edges(graph, cut_x)
    if not crossing_edges or len(crossing_edges) > settings["maximum_skeleton_crossings"]:
        return None
    if _near_junction(
        graph,
        crossing_edges,
        settings["minimum_junction_distance_px"],
    ):
        return None

    path_locations = [
        location
        for edge in crossing_edges
        for location in path_lookup.get(edge, [])
    ]
    path_ids = {location["path_id"] for location in path_locations}
    if len(path_locations) != len(crossing_edges) or len(path_ids) != 1:
        return None
    if any(location["is_closed"] for location in path_locations):
        return None

    connector_path_id = next(iter(path_ids))
    connector_path = next(
        path for path in trace_paths if path.path_id == connector_path_id
    )
    original_component = next(
        (
            component
            for component in _graph_components(graph)
            if crossing_edges[0][0] in component
        ),
        None,
    )
    if original_component is None:
        return None

    split_components = _graph_components(
        graph,
        allowed_coordinates=original_component,
        removed_edges=crossing_edges,
    )
    if len(split_components) != 2:
        return None

    metrics = [_side_metrics(component, cut_x) for component in split_components]
    left_index = max(range(2), key=lambda index: metrics[index]["left_dominance"])
    right_index = 1 - left_index
    left_component = split_components[left_index]
    right_component = split_components[right_index]
    left_metrics = metrics[left_index]
    right_metrics = metrics[right_index]
    minimum_points = max(
        settings["minimum_vector_subgraph_points"],
        int(
            round(
                len(original_component)
                * settings["minimum_vector_subgraph_ratio"]
            )
        ),
    )
    if (
        left_metrics["point_count"] < minimum_points
        or right_metrics["point_count"] < minimum_points
        or left_metrics["left_dominance"] < settings["minimum_side_dominance"]
        or right_metrics["right_dominance"] < settings["minimum_side_dominance"]
    ):
        return None

    split_after_index = min(
        location["edge_index"]
        for location in path_locations
    )
    return {
        "mode": "connector_path_disconnection",
        "connector_path_id": connector_path_id,
        "connector_path_point_count": connector_path.point_count(),
        "connector_path_length": float(connector_path.length()),
        "split_after_point_index": split_after_index,
        "crossing_edges": [
            {
                "from": {"x": edge[0][0], "y": edge[0][1]},
                "to": {"x": edge[1][0], "y": edge[1][1]},
            }
            for edge in crossing_edges
        ],
        "left_connector_range": [0, split_after_index],
        "right_connector_range": [
            split_after_index + 1,
            connector_path.point_count() - 1,
        ],
        "left_subgraph": left_metrics,
        "right_subgraph": right_metrics,
        "left_path_ids": _path_ids_for_coordinates(
            trace_paths, left_component
        ),
        "right_path_ids": _path_ids_for_coordinates(
            trace_paths, right_component
        ),
    }


def propose_trace_validated_cuts(mask, settings=None):
    """
    Rank deterministic two-way cuts using mask and skeleton agreement.

    Args:
        mask: White-on-black uint8 analysis mask.
        settings: Optional character-segmentation setting overrides.

    Returns:
        JSON-safe diagnostics and accepted split candidates.
    """
    settings = normalize_segmentation_settings(settings)
    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    height, width = binary.shape[:2]
    ink = binary > 0
    ink_pixels = int(np.count_nonzero(ink))
    aspect_ratio = float(width / height) if height else 0.0
    projection = np.count_nonzero(ink, axis=0).astype(int)
    components = _extract_components(binary, settings)

    skeleton = TraceSkeletonizer().skeletonize(binary)
    skeleton_points = SkeletonPointExtractor().extract_points(skeleton)
    skeleton_graph = SkeletonGraph(skeleton_points)
    trace_paths = TracePathExtractor().extract_paths(skeleton_graph)
    path_lookup = _edge_path_lookup(trace_paths)
    graph_summary = skeleton_graph.to_dict()
    skeleton_projection = np.count_nonzero(skeleton > 0, axis=0).astype(int)
    diagnostics = {
        "width": int(width),
        "height": int(height),
        "aspect_ratio": aspect_ratio,
        "ink_pixel_count": ink_pixels,
        "connected_component_count": len(components),
        "components": [component.to_dict() for component in components],
        "floating_component_count": sum(
            component.is_floating for component in components
        ),
        "attached_floating_component_count": sum(
            component.attached_to_component_id is not None
            for component in components
            if component.is_floating
        ),
        "vertical_projection_profile": projection.astype(int).tolist(),
        "skeleton_vertical_projection_profile": (
            skeleton_projection.astype(int).tolist()
        ),
        "skeleton_pixel_count": int(np.count_nonzero(skeleton)),
        "skeleton_graph": graph_summary,
        "trace_path_count": len(trace_paths),
        "trace_paths": [path.to_dict() for path in trace_paths],
    }

    looks_multi_letter = (
        settings["enabled"]
        and width >= settings["minimum_unit_width_px"]
        and aspect_ratio >= settings["minimum_unit_aspect_ratio"]
        and ink_pixels > 0
    )
    if not looks_multi_letter:
        return {
            "diagnostics": diagnostics,
            "split_candidates": [],
            "skeleton_mask": skeleton,
        }

    minimum_width = max(
        3,
        int(round(width * settings["minimum_segment_width_ratio"])),
    )
    start = minimum_width
    end = width - minimum_width
    if end <= start:
        return {
            "diagnostics": diagnostics,
            "split_candidates": [],
            "skeleton_mask": skeleton,
        }

    maximum_projection = max(1, int(projection.max(initial=0)))
    minimum_side_ink = max(
        1,
        int(round(ink_pixels * settings["minimum_segment_ink_ratio"])),
    )
    protected_ranges = _protected_cut_ranges(components)
    raw_candidates = _local_minimum_positions(projection, start, end)
    candidates = []
    for raw_x in raw_candidates:
        cut_x = _best_cut_in_window(
            raw_x,
            projection,
            skeleton_projection,
            start,
            end,
            settings["cut_search_radius_px"],
        )
        if any(x1 < cut_x < x2 for x1, x2, _ in protected_ranges):
            continue

        left_ink = int(np.count_nonzero(ink[:, :cut_x]))
        right_ink = int(np.count_nonzero(ink[:, cut_x:]))
        if left_ink < minimum_side_ink or right_ink < minimum_side_ink:
            continue

        projection_ratio = float(projection[cut_x] / maximum_projection)
        skeleton_crossings = int(skeleton_projection[cut_x])
        if projection_ratio > settings["maximum_projection_ratio"]:
            continue
        if skeleton_crossings > settings["maximum_skeleton_crossings"]:
            continue

        if skeleton_crossings == 0:
            vector_split = _validate_disconnected_groups(
                skeleton_graph,
                trace_paths,
                cut_x,
                settings,
            )
        else:
            vector_split = _validate_connector_path(
                skeleton_graph,
                trace_paths,
                path_lookup,
                cut_x,
                settings,
            )
        if vector_split is None:
            continue

        valley_strength = 1.0 - projection_ratio
        left_vector_ratio = (
            vector_split["left_subgraph"]["point_count"]
            / max(1, graph_summary["point_count"])
        )
        right_vector_ratio = (
            vector_split["right_subgraph"]["point_count"]
            / max(1, graph_summary["point_count"])
        )
        topology_strength = min(1.0, left_vector_ratio + right_vector_ratio)
        balance = 1.0 - abs(left_ink - right_ink) / max(1, ink_pixels)
        score = 0.45 * valley_strength + 0.40 * topology_strength + 0.15 * balance
        candidates.append(
            {
                "cut_x": int(cut_x),
                "score": float(score),
                "projection_value": int(projection[cut_x]),
                "projection_ratio": projection_ratio,
                "skeleton_crossings": skeleton_crossings,
                "left_ink_pixels": left_ink,
                "right_ink_pixels": right_ink,
                "left_ink_ratio": float(left_ink / ink_pixels),
                "right_ink_ratio": float(right_ink / ink_pixels),
                "validation": vector_split["mode"],
                "vector_split": vector_split,
            }
        )

    candidates.sort(
        key=lambda item: (
            -item["score"],
            item["skeleton_crossings"],
            item["projection_value"],
            abs(item["cut_x"] - width / 2.0),
            item["cut_x"],
        )
    )
    selected = []
    minimum_spacing = max(
        3,
        int(round(width * settings["minimum_cut_spacing_ratio"])),
    )
    for candidate in candidates:
        if any(
            abs(candidate["cut_x"] - existing["cut_x"]) < minimum_spacing
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= settings["maximum_hypotheses"]:
            break

    return {
        "diagnostics": diagnostics,
        "split_candidates": selected,
        "skeleton_mask": skeleton,
    }


__all__ = [
    "DEFAULT_SEGMENTATION_SETTINGS",
    "normalize_segmentation_settings",
    "propose_trace_validated_cuts",
]
