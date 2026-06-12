"""Deterministic 104-feature geometry encoder for ScribeTrace evidence."""

import math

from .trace_models import BoundingBox, TraceFeatureVector
from .trace_settings import normalize_trace_settings

class TraceFeatureEncoder:
    """
    Convert ScribeTrace geometric evidence into ML-ready features.

    This does not recognize letters.
    It produces compact evidence that can teach a model letter structure.
    """

    TOKEN_BY_LANDMARK_TYPE = {
        "start": "S",
        "end": "E",
        "global_left": "GL",
        "global_right": "GR",
        "global_top": "GT",
        "global_bottom": "GB",
        "local_top_peak": "TP",
        "local_bottom_valley": "BV",
        "local_left_turn": "LT",
        "local_right_turn": "RT",
    }

    def __init__(self, settings=None):
        self.settings = normalize_trace_settings(settings)

    def _count_landmarks_by_type(self, landmarks):
        counts = {}

        for landmark in landmarks:
            counts[landmark.landmark_type] = (
                counts.get(landmark.landmark_type, 0) + 1
            )

        return counts

    def _safe_divide(self, numerator, denominator):
        if denominator == 0:
            return 0.0
        return float(numerator) / float(denominator)

    def _path_length_stats(self, trace_paths):
        lengths = [path.length() for path in trace_paths]

        if not lengths:
            return {
                "total_path_length": 0.0,
                "mean_path_length": 0.0,
                "max_path_length": 0.0,
                "min_path_length": 0.0,
            }

        return {
            "total_path_length": float(sum(lengths)),
            "mean_path_length": float(sum(lengths) / len(lengths)),
            "max_path_length": float(max(lengths)),
            "min_path_length": float(min(lengths)),
        }

    def _component_stats(self, components):
        if not components:
            return {
                "total_ink_pixels": 0.0,
                "mean_component_area": 0.0,
                "max_component_area": 0.0,
                "mean_component_density": 0.0,
            }

        areas = [component.area() for component in components]
        densities = [component.ink_density() for component in components]

        return {
            "total_ink_pixels": float(
                sum(component.point_count() for component in components)
            ),
            "mean_component_area": float(sum(areas) / len(areas)),
            "max_component_area": float(max(areas)),
            "mean_component_density": float(sum(densities) / len(densities)),
        }

    def _empty_spatial_stats(self):
        """Return stable zeros for spatial mass and endpoint features."""
        keys = [
            "ink_bbox_width",
            "ink_bbox_height",
            "ink_bbox_area",
            "ink_bbox_aspect_ratio",
            "ink_bbox_fill_ratio",
            "ink_centroid_x_in_bbox",
            "ink_centroid_y_in_bbox",
            "ink_centroid_dx_from_center",
            "ink_centroid_dy_from_center",
            "ink_left_half_ratio",
            "ink_right_half_ratio",
            "ink_top_half_ratio",
            "ink_bottom_half_ratio",
            "ink_left_right_balance",
            "ink_top_bottom_balance",
            "ink_top_left_quadrant_ratio",
            "ink_top_right_quadrant_ratio",
            "ink_bottom_left_quadrant_ratio",
            "ink_bottom_right_quadrant_ratio",
            "ink_center_band_ratio",
            "ink_left_edge_ratio",
            "ink_right_edge_ratio",
            "ink_top_edge_ratio",
            "ink_bottom_edge_ratio",
            "endpoint_mean_x_in_bbox",
            "endpoint_mean_y_in_bbox",
            "endpoint_horizontal_spread_ratio",
            "endpoint_vertical_spread_ratio",
            "endpoint_left_half_ratio",
            "endpoint_right_half_ratio",
            "endpoint_top_half_ratio",
            "endpoint_bottom_half_ratio",
            "hole_area_sum",
            "max_hole_area",
            "hole_area_to_ink_ratio",
            "hole_area_to_bbox_ratio",
            "hole_mean_x_in_bbox",
            "hole_mean_y_in_bbox",
            "mean_path_bbox_width",
            "mean_path_bbox_height",
            "mean_path_bbox_aspect_ratio",
            "mean_path_bbox_area_ratio",
            "max_path_bbox_area_ratio",
        ]
        return {key: 0.0 for key in keys}

    def _points_bbox(self, points):
        """Build a BoundingBox from points or return None for empty input."""
        points = list(points or [])
        if not points:
            return None
        return BoundingBox.from_points(points)

    def _point_position_in_bbox(self, point_or_xy, box):
        """Return normalized x/y position inside a bounding box."""
        if box is None:
            return 0.0, 0.0

        if hasattr(point_or_xy, "x"):
            x_value = float(point_or_xy.x)
            y_value = float(point_or_xy.y)
        else:
            x_value, y_value = point_or_xy
            x_value = float(x_value)
            y_value = float(y_value)

        width_denominator = max(1.0, float(box.width() - 1))
        height_denominator = max(1.0, float(box.height() - 1))
        return (
            (x_value - float(box.x1)) / width_denominator,
            (y_value - float(box.y1)) / height_denominator,
        )

    def _ratio_inside_region(self, points, predicate):
        """Return share of points satisfying a spatial predicate."""
        points = list(points or [])
        if not points:
            return 0.0
        return float(sum(1 for point in points if predicate(point))) / float(len(points))

    def _spatial_stats(self, components, skeleton_graph, trace_paths, ink_holes):
        """
        Aggregate glyph mass, endpoint, loop, and path-box placement features.

        v0_2 already captures movement behavior. These features help v0_2.1
        separate confusable Armenian glyphs where the same motion appears in
        different spatial positions, or where loops/endpoints sit differently.
        """
        ink_points = []
        for component in components:
            ink_points.extend(component.points)

        if not ink_points:
            return self._empty_spatial_stats()

        box = BoundingBox.from_points(ink_points)
        width = float(max(1, box.width()))
        height = float(max(1, box.height()))
        area = float(max(1, box.area()))
        ink_count = float(len(ink_points))
        center_x = (float(box.x1) + float(box.x2 - 1)) / 2.0
        center_y = (float(box.y1) + float(box.y2 - 1)) / 2.0

        centroid_x = sum(point.x for point in ink_points) / ink_count
        centroid_y = sum(point.y for point in ink_points) / ink_count
        centroid_x_in_bbox, centroid_y_in_bbox = self._point_position_in_bbox(
            (centroid_x, centroid_y),
            box,
        )

        left_half = self._ratio_inside_region(ink_points, lambda point: point.x <= center_x)
        right_half = self._ratio_inside_region(ink_points, lambda point: point.x > center_x)
        top_half = self._ratio_inside_region(ink_points, lambda point: point.y <= center_y)
        bottom_half = self._ratio_inside_region(ink_points, lambda point: point.y > center_y)

        left_edge_limit = float(box.x1) + width * 0.20
        right_edge_limit = float(box.x2) - width * 0.20
        top_edge_limit = float(box.y1) + height * 0.20
        bottom_edge_limit = float(box.y2) - height * 0.20

        endpoint_points = []
        if skeleton_graph is not None:
            try:
                endpoint_points = list(skeleton_graph.endpoints())
            except AttributeError:
                endpoint_points = []

        endpoint_positions = [
            self._point_position_in_bbox(point, box)
            for point in endpoint_points
            if box.contains_point(point)
        ]

        if endpoint_positions:
            endpoint_x_values = [item[0] for item in endpoint_positions]
            endpoint_y_values = [item[1] for item in endpoint_positions]
            endpoint_mean_x = sum(endpoint_x_values) / len(endpoint_x_values)
            endpoint_mean_y = sum(endpoint_y_values) / len(endpoint_y_values)
            endpoint_horizontal_spread = max(endpoint_x_values) - min(endpoint_x_values)
            endpoint_vertical_spread = max(endpoint_y_values) - min(endpoint_y_values)
            endpoint_left_half = sum(1 for x_value, _ in endpoint_positions if x_value <= 0.5) / len(endpoint_positions)
            endpoint_right_half = sum(1 for x_value, _ in endpoint_positions if x_value > 0.5) / len(endpoint_positions)
            endpoint_top_half = sum(1 for _, y_value in endpoint_positions if y_value <= 0.5) / len(endpoint_positions)
            endpoint_bottom_half = sum(1 for _, y_value in endpoint_positions if y_value > 0.5) / len(endpoint_positions)
        else:
            endpoint_mean_x = 0.0
            endpoint_mean_y = 0.0
            endpoint_horizontal_spread = 0.0
            endpoint_vertical_spread = 0.0
            endpoint_left_half = 0.0
            endpoint_right_half = 0.0
            endpoint_top_half = 0.0
            endpoint_bottom_half = 0.0

        hole_areas = [float(hole.area()) for hole in ink_holes]
        hole_area_sum = float(sum(hole_areas))
        max_hole_area = float(max(hole_areas)) if hole_areas else 0.0
        hole_centers = [
            self._point_position_in_bbox(hole.center(), box)
            for hole in ink_holes
        ]
        if hole_centers:
            hole_mean_x = sum(item[0] for item in hole_centers) / len(hole_centers)
            hole_mean_y = sum(item[1] for item in hole_centers) / len(hole_centers)
        else:
            hole_mean_x = 0.0
            hole_mean_y = 0.0

        path_boxes = [path.bounding_box for path in trace_paths]
        if path_boxes:
            path_widths = [float(max(1, item.width())) for item in path_boxes]
            path_heights = [float(max(1, item.height())) for item in path_boxes]
            path_areas = [float(max(1, item.area())) for item in path_boxes]
            path_aspects = [w / h for w, h in zip(path_widths, path_heights)]
            mean_path_bbox_width = sum(path_widths) / len(path_widths)
            mean_path_bbox_height = sum(path_heights) / len(path_heights)
            mean_path_bbox_aspect = sum(path_aspects) / len(path_aspects)
            mean_path_bbox_area_ratio = sum(item / area for item in path_areas) / len(path_areas)
            max_path_bbox_area_ratio = max(item / area for item in path_areas)
        else:
            mean_path_bbox_width = 0.0
            mean_path_bbox_height = 0.0
            mean_path_bbox_aspect = 0.0
            mean_path_bbox_area_ratio = 0.0
            max_path_bbox_area_ratio = 0.0

        return {
            "ink_bbox_width": width,
            "ink_bbox_height": height,
            "ink_bbox_area": area,
            "ink_bbox_aspect_ratio": self._safe_divide(width, height),
            "ink_bbox_fill_ratio": self._safe_divide(ink_count, area),
            "ink_centroid_x_in_bbox": float(centroid_x_in_bbox),
            "ink_centroid_y_in_bbox": float(centroid_y_in_bbox),
            "ink_centroid_dx_from_center": float(centroid_x_in_bbox - 0.5),
            "ink_centroid_dy_from_center": float(centroid_y_in_bbox - 0.5),
            "ink_left_half_ratio": float(left_half),
            "ink_right_half_ratio": float(right_half),
            "ink_top_half_ratio": float(top_half),
            "ink_bottom_half_ratio": float(bottom_half),
            "ink_left_right_balance": float(right_half - left_half),
            "ink_top_bottom_balance": float(bottom_half - top_half),
            "ink_top_left_quadrant_ratio": self._ratio_inside_region(
                ink_points, lambda point: point.x <= center_x and point.y <= center_y
            ),
            "ink_top_right_quadrant_ratio": self._ratio_inside_region(
                ink_points, lambda point: point.x > center_x and point.y <= center_y
            ),
            "ink_bottom_left_quadrant_ratio": self._ratio_inside_region(
                ink_points, lambda point: point.x <= center_x and point.y > center_y
            ),
            "ink_bottom_right_quadrant_ratio": self._ratio_inside_region(
                ink_points, lambda point: point.x > center_x and point.y > center_y
            ),
            "ink_center_band_ratio": self._ratio_inside_region(
                ink_points,
                lambda point: (
                    float(box.x1) + width * 0.33 <= point.x <= float(box.x1) + width * 0.67
                    and float(box.y1) + height * 0.33 <= point.y <= float(box.y1) + height * 0.67
                ),
            ),
            "ink_left_edge_ratio": self._ratio_inside_region(
                ink_points, lambda point: point.x <= left_edge_limit
            ),
            "ink_right_edge_ratio": self._ratio_inside_region(
                ink_points, lambda point: point.x >= right_edge_limit
            ),
            "ink_top_edge_ratio": self._ratio_inside_region(
                ink_points, lambda point: point.y <= top_edge_limit
            ),
            "ink_bottom_edge_ratio": self._ratio_inside_region(
                ink_points, lambda point: point.y >= bottom_edge_limit
            ),
            "endpoint_mean_x_in_bbox": float(endpoint_mean_x),
            "endpoint_mean_y_in_bbox": float(endpoint_mean_y),
            "endpoint_horizontal_spread_ratio": float(endpoint_horizontal_spread),
            "endpoint_vertical_spread_ratio": float(endpoint_vertical_spread),
            "endpoint_left_half_ratio": float(endpoint_left_half),
            "endpoint_right_half_ratio": float(endpoint_right_half),
            "endpoint_top_half_ratio": float(endpoint_top_half),
            "endpoint_bottom_half_ratio": float(endpoint_bottom_half),
            "hole_area_sum": hole_area_sum,
            "max_hole_area": max_hole_area,
            "hole_area_to_ink_ratio": self._safe_divide(hole_area_sum, ink_count),
            "hole_area_to_bbox_ratio": self._safe_divide(hole_area_sum, area),
            "hole_mean_x_in_bbox": float(hole_mean_x),
            "hole_mean_y_in_bbox": float(hole_mean_y),
            "mean_path_bbox_width": float(mean_path_bbox_width),
            "mean_path_bbox_height": float(mean_path_bbox_height),
            "mean_path_bbox_aspect_ratio": float(mean_path_bbox_aspect),
            "mean_path_bbox_area_ratio": float(mean_path_bbox_area_ratio),
            "max_path_bbox_area_ratio": float(max_path_bbox_area_ratio),
        }

    def _direction_stats_for_path(self, path):
        """Return deterministic stroke-direction statistics for one path."""
        points = list(path.points)

        if len(points) < 2:
            return {
                "step_count": 0.0,
                "horizontal_direction_change_count": 0.0,
                "vertical_direction_change_count": 0.0,
                "direction_change_count": 0.0,
                "clockwise_turn_count": 0.0,
                "counterclockwise_turn_count": 0.0,
                "up_step_count": 0.0,
                "down_step_count": 0.0,
                "left_step_count": 0.0,
                "right_step_count": 0.0,
                "diagonal_step_count": 0.0,
                "net_dx": 0.0,
                "net_dy": 0.0,
                "abs_net_dx": 0.0,
                "abs_net_dy": 0.0,
                "straightness": 0.0,
                "direction_entropy": 0.0,
            }

        steps = []
        for point_a, point_b in zip(points, points[1:]):
            dx_raw = point_b.x - point_a.x
            dy_raw = point_b.y - point_a.y
            if dx_raw == 0 and dy_raw == 0:
                continue
            dx = 1 if dx_raw > 0 else -1 if dx_raw < 0 else 0
            dy = 1 if dy_raw > 0 else -1 if dy_raw < 0 else 0
            steps.append((dx, dy))

        if not steps:
            return {
                "step_count": 0.0,
                "horizontal_direction_change_count": 0.0,
                "vertical_direction_change_count": 0.0,
                "direction_change_count": 0.0,
                "clockwise_turn_count": 0.0,
                "counterclockwise_turn_count": 0.0,
                "up_step_count": 0.0,
                "down_step_count": 0.0,
                "left_step_count": 0.0,
                "right_step_count": 0.0,
                "diagonal_step_count": 0.0,
                "net_dx": 0.0,
                "net_dy": 0.0,
                "abs_net_dx": 0.0,
                "abs_net_dy": 0.0,
                "straightness": 0.0,
                "direction_entropy": 0.0,
            }

        horizontal_changes = 0
        vertical_changes = 0
        direction_changes = 0
        clockwise_turns = 0
        counterclockwise_turns = 0

        for previous_step, current_step in zip(steps, steps[1:]):
            previous_dx, previous_dy = previous_step
            current_dx, current_dy = current_step

            if previous_dx != 0 and current_dx != 0 and previous_dx != current_dx:
                horizontal_changes += 1
            if previous_dy != 0 and current_dy != 0 and previous_dy != current_dy:
                vertical_changes += 1
            if previous_step != current_step:
                direction_changes += 1

            cross = previous_dx * current_dy - previous_dy * current_dx
            if cross > 0:
                counterclockwise_turns += 1
            elif cross < 0:
                clockwise_turns += 1

        direction_histogram = {}
        for step in steps:
            direction_histogram[step] = direction_histogram.get(step, 0) + 1

        entropy = 0.0
        for count in direction_histogram.values():
            probability = count / len(steps)
            entropy -= probability * math.log2(probability)

        net_dx = float(points[-1].x - points[0].x)
        net_dy = float(points[-1].y - points[0].y)
        path_length = path.length()
        chord_length = math.hypot(net_dx, net_dy)

        return {
            "step_count": float(len(steps)),
            "horizontal_direction_change_count": float(horizontal_changes),
            "vertical_direction_change_count": float(vertical_changes),
            "direction_change_count": float(direction_changes),
            "clockwise_turn_count": float(clockwise_turns),
            "counterclockwise_turn_count": float(counterclockwise_turns),
            "up_step_count": float(sum(1 for _, dy in steps if dy < 0)),
            "down_step_count": float(sum(1 for _, dy in steps if dy > 0)),
            "left_step_count": float(sum(1 for dx, _ in steps if dx < 0)),
            "right_step_count": float(sum(1 for dx, _ in steps if dx > 0)),
            "diagonal_step_count": float(sum(1 for dx, dy in steps if dx != 0 and dy != 0)),
            "net_dx": net_dx,
            "net_dy": net_dy,
            "abs_net_dx": abs(net_dx),
            "abs_net_dy": abs(net_dy),
            "straightness": self._safe_divide(chord_length, path_length),
            "direction_entropy": float(entropy),
        }

    def _direction_stats(self, trace_paths):
        """Aggregate path-level direction statistics into RF-safe features."""
        if not trace_paths:
            return {
                "step_count": 0.0,
                "horizontal_direction_change_count": 0.0,
                "vertical_direction_change_count": 0.0,
                "direction_change_count": 0.0,
                "clockwise_turn_count": 0.0,
                "counterclockwise_turn_count": 0.0,
                "up_step_count": 0.0,
                "down_step_count": 0.0,
                "left_step_count": 0.0,
                "right_step_count": 0.0,
                "diagonal_step_count": 0.0,
                "net_dx_sum": 0.0,
                "net_dy_sum": 0.0,
                "abs_net_dx_sum": 0.0,
                "abs_net_dy_sum": 0.0,
                "mean_path_straightness": 0.0,
                "mean_direction_entropy": 0.0,
                "max_direction_entropy": 0.0,
            }

        per_path = [self._direction_stats_for_path(path) for path in trace_paths]
        step_count = sum(item["step_count"] for item in per_path)

        return {
            "step_count": float(step_count),
            "horizontal_direction_change_count": float(sum(item["horizontal_direction_change_count"] for item in per_path)),
            "vertical_direction_change_count": float(sum(item["vertical_direction_change_count"] for item in per_path)),
            "direction_change_count": float(sum(item["direction_change_count"] for item in per_path)),
            "clockwise_turn_count": float(sum(item["clockwise_turn_count"] for item in per_path)),
            "counterclockwise_turn_count": float(sum(item["counterclockwise_turn_count"] for item in per_path)),
            "up_step_count": float(sum(item["up_step_count"] for item in per_path)),
            "down_step_count": float(sum(item["down_step_count"] for item in per_path)),
            "left_step_count": float(sum(item["left_step_count"] for item in per_path)),
            "right_step_count": float(sum(item["right_step_count"] for item in per_path)),
            "diagonal_step_count": float(sum(item["diagonal_step_count"] for item in per_path)),
            "net_dx_sum": float(sum(item["net_dx"] for item in per_path)),
            "net_dy_sum": float(sum(item["net_dy"] for item in per_path)),
            "abs_net_dx_sum": float(sum(item["abs_net_dx"] for item in per_path)),
            "abs_net_dy_sum": float(sum(item["abs_net_dy"] for item in per_path)),
            "mean_path_straightness": float(sum(item["straightness"] for item in per_path) / len(per_path)),
            "mean_direction_entropy": float(sum(item["direction_entropy"] for item in per_path) / len(per_path)),
            "max_direction_entropy": float(max(item["direction_entropy"] for item in per_path)),
        }

    def _sequence_for_path(self, path, landmarks):
        path_landmarks = [
            landmark
            for landmark in landmarks
            if landmark.path_id == path.path_id
        ]

        # Path position carries the shape order. Type and ID only resolve
        # multiple landmark labels attached to the same skeleton pixel.
        path_landmarks.sort(
            key=lambda landmark: (
                landmark.index_on_path,
                landmark.landmark_type,
                landmark.landmark_id,
            )
        )

        tokens = []

        for landmark in path_landmarks:
            token = self.TOKEN_BY_LANDMARK_TYPE.get(
                landmark.landmark_type,
                landmark.landmark_type,
            )
            tokens.append(token)

        return tokens

    def build_sequence(self, trace_paths, landmarks):
        """
        Build deterministic symbolic path sequence.

        Example:
        P0:S-TP-BV-TP-E | P1:S-BV-E
        """
        full_sequence = []

        for path in sorted(trace_paths, key=lambda item: item.path_id):
            path_tokens = self._sequence_for_path(path, landmarks)

            # Preserve topology in the token stream so sequence models can
            # distinguish loops and small marks from ordinary open strokes.
            if path.is_closed:
                prefix = f"P{path.path_id}:LOOP"
            elif path.is_short:
                prefix = f"P{path.path_id}:SHORT"
            else:
                prefix = f"P{path.path_id}"

            full_sequence.append(prefix)
            full_sequence.extend(path_tokens)
            full_sequence.append("|")

        if full_sequence and full_sequence[-1] == "|":
            full_sequence.pop()

        return full_sequence

    def encode(
        self,
        components,
        skeleton_graph,
        trace_paths,
        landmarks,
        metrics=None,
        ink_holes=None,
    ):
        """
        Return deterministic numeric vector and symbolic sequence.
        """
        metrics = metrics or {}
        ink_holes = ink_holes or []

        landmark_counts = self._count_landmarks_by_type(landmarks)
        path_stats = self._path_length_stats(trace_paths)
        component_stats = self._component_stats(components)
        direction_stats = self._direction_stats(trace_paths)
        spatial_stats = self._spatial_stats(
            components,
            skeleton_graph,
            trace_paths,
            ink_holes,
        )
        matched_ink_hole_count = float(metrics.get("ink_hole_match_count", 0))
        unmatched_ink_hole_count = float(metrics.get("unmatched_ink_hole_count", 0))

        skeleton_graph_data = (
            skeleton_graph.to_dict()
            if skeleton_graph is not None
            else {}
        )

        feature_map = {
            "component_count": float(len(components)),
            "path_count": float(len(trace_paths)),
            "landmark_count": float(len(landmarks)),

            "endpoint_count": float(
                skeleton_graph_data.get("endpoint_count", 0)
            ),
            "junction_pixel_count": float(
                skeleton_graph_data.get("junction_pixel_count", 0)
            ),
            "junction_cluster_count": float(
                skeleton_graph_data.get("junction_cluster_count", 0)
            ),
            "closed_loop_count": float(
                sum(path.is_closed for path in trace_paths)
            ),
            "short_path_count": float(
                sum(path.is_short for path in trace_paths)
            ),

            "top_peak_count": float(
                landmark_counts.get("local_top_peak", 0)
            ),
            "bottom_valley_count": float(
                landmark_counts.get("local_bottom_valley", 0)
            ),
            "left_turn_count": float(
                landmark_counts.get("local_left_turn", 0)
            ),
            "right_turn_count": float(
                landmark_counts.get("local_right_turn", 0)
            ),

            "global_top_count": float(
                landmark_counts.get("global_top", 0)
            ),
            "global_bottom_count": float(
                landmark_counts.get("global_bottom", 0)
            ),
            "global_left_count": float(
                landmark_counts.get("global_left", 0)
            ),
            "global_right_count": float(
                landmark_counts.get("global_right", 0)
            ),
            "ink_hole_count": float(len(ink_holes)),
            "ink_hole_component_count": float(
                len({hole.component_id for hole in ink_holes})
            ),

            "ink_hole_match_count": matched_ink_hole_count,
            "unmatched_ink_hole_count": unmatched_ink_hole_count,
            "attached_loop_count": float(metrics.get("attached_loop_count", 0)),
            "total_path_length": path_stats["total_path_length"],
            "mean_path_length": path_stats["mean_path_length"],
            "max_path_length": path_stats["max_path_length"],
            "min_path_length": path_stats["min_path_length"],

            "total_ink_pixels": component_stats["total_ink_pixels"],
            "mean_component_area": component_stats["mean_component_area"],
            "max_component_area": component_stats["max_component_area"],
            "mean_component_density": component_stats["mean_component_density"],

            "step_count": direction_stats["step_count"],
            "horizontal_direction_change_count": direction_stats["horizontal_direction_change_count"],
            "vertical_direction_change_count": direction_stats["vertical_direction_change_count"],
            "direction_change_count": direction_stats["direction_change_count"],
            "clockwise_turn_count": direction_stats["clockwise_turn_count"],
            "counterclockwise_turn_count": direction_stats["counterclockwise_turn_count"],
            "up_step_count": direction_stats["up_step_count"],
            "down_step_count": direction_stats["down_step_count"],
            "left_step_count": direction_stats["left_step_count"],
            "right_step_count": direction_stats["right_step_count"],
            "diagonal_step_count": direction_stats["diagonal_step_count"],
            "net_dx_sum": direction_stats["net_dx_sum"],
            "net_dy_sum": direction_stats["net_dy_sum"],
            "abs_net_dx_sum": direction_stats["abs_net_dx_sum"],
            "abs_net_dy_sum": direction_stats["abs_net_dy_sum"],
            "mean_path_straightness": direction_stats["mean_path_straightness"],
            "mean_direction_entropy": direction_stats["mean_direction_entropy"],
            "max_direction_entropy": direction_stats["max_direction_entropy"],
        }

        feature_map.update(spatial_stats)

        feature_map["peaks_per_path"] = self._safe_divide(
            feature_map["top_peak_count"],
            feature_map["path_count"],
        )
        feature_map["valleys_per_path"] = self._safe_divide(
            feature_map["bottom_valley_count"],
            feature_map["path_count"],
        )
        feature_map["junctions_per_path"] = self._safe_divide(
            feature_map["junction_cluster_count"],
            feature_map["path_count"],
        )
        feature_map["length_per_ink_pixel"] = self._safe_divide(
            feature_map["total_path_length"],
            feature_map["total_ink_pixels"],
        )
        feature_map["direction_changes_per_path"] = self._safe_divide(
            feature_map["direction_change_count"],
            feature_map["path_count"],
        )
        feature_map["horizontal_direction_changes_per_path"] = self._safe_divide(
            feature_map["horizontal_direction_change_count"],
            feature_map["path_count"],
        )
        feature_map["vertical_direction_changes_per_path"] = self._safe_divide(
            feature_map["vertical_direction_change_count"],
            feature_map["path_count"],
        )
        feature_map["up_step_ratio"] = self._safe_divide(
            feature_map["up_step_count"],
            feature_map["step_count"],
        )
        feature_map["down_step_ratio"] = self._safe_divide(
            feature_map["down_step_count"],
            feature_map["step_count"],
        )
        feature_map["left_step_ratio"] = self._safe_divide(
            feature_map["left_step_count"],
            feature_map["step_count"],
        )
        feature_map["right_step_ratio"] = self._safe_divide(
            feature_map["right_step_count"],
            feature_map["step_count"],
        )
        feature_map["diagonal_step_ratio"] = self._safe_divide(
            feature_map["diagonal_step_count"],
            feature_map["step_count"],
        )
        feature_map["clockwise_turn_ratio"] = self._safe_divide(
            feature_map["clockwise_turn_count"],
            feature_map["direction_change_count"],
        )
        feature_map["counterclockwise_turn_ratio"] = self._safe_divide(
            feature_map["counterclockwise_turn_count"],
            feature_map["direction_change_count"],
        )

        # Alphabetical ordering makes the numeric vector reproducible across
        # processes and protects future training data from dictionary changes.
        feature_names = sorted(feature_map)
        vector = [feature_map[name] for name in feature_names]

        sequence = self.build_sequence(trace_paths, landmarks)
        sequence_string = " ".join(sequence)

        return TraceFeatureVector(
            vector=vector,
            feature_names=feature_names,
            sequence=sequence,
            sequence_string=sequence_string,
        )


