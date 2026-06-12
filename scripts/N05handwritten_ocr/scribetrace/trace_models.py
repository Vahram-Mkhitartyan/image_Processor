"""Core immutable-style records used throughout ScribeTrace."""

import math

from .trace_common import EXPERT_NAME, NEIGHBOR_OFFSETS, coordinate_key, sanitize_identifier

_coordinate_key = coordinate_key
_sanitize_identifier = sanitize_identifier

class TraceInput:
    """Describe the preferred exact mask and visual fallback for one text unit."""

    def __init__(
        self,
        crop_path="",
        mask_crop_path=None,
        visual_crop_path=None,
        context_crop_path=None,
        output_dir=None,
        document_id=None,
        text_unit_id=None,
        source_group_id=None,
        source_layer_group_id=None,
        layer=None,
        document_bbox=None,
        final_bbox=None,
    ):
        self.crop_path = crop_path
        self.mask_crop_path = mask_crop_path
        self.visual_crop_path = visual_crop_path
        self.context_crop_path = context_crop_path
        self.output_dir = output_dir
        self.document_id = document_id
        self.text_unit_id = text_unit_id
        self.source_group_id = source_group_id
        self.source_layer_group_id = source_layer_group_id
        self.layer = layer
        self.document_bbox = document_bbox
        self.final_bbox = final_bbox

    @classmethod
    def from_context(cls, crop_path, context):
        """Build a trace input from the shared N05 expert context."""
        context = context or {}
        return cls(
            crop_path=crop_path,
            mask_crop_path=context.get("scribetrace_mask_crop_path"),
            visual_crop_path=context.get("scribetrace_visual_crop_path"),
            context_crop_path=context.get("scribetrace_context_crop_path"),
            output_dir=context.get("scribetrace_output_dir"),
            document_id=context.get("document_id"),
            text_unit_id=context.get("text_unit_id"),
            source_group_id=context.get("source_group_id"),
            source_layer_group_id=context.get("source_layer_group_id"),
            layer=context.get("layer"),
            document_bbox=context.get("document_bbox"),
            final_bbox=context.get("final_bbox"),
        )

    def validate(self):
        """Require at least one potential image source."""
        if not any((self.mask_crop_path, self.visual_crop_path, self.crop_path)):
            raise ValueError(
                "TraceInput requires mask_crop_path, visual_crop_path, or crop_path."
            )

    def stable_unit_id(self):
        """Return a stable identifier safe for use in debug filenames."""
        if self.text_unit_id is not None:
            value = self.text_unit_id
        elif self.source_group_id is not None:
            value = self.source_group_id
        else:
            value = "unknown_unit"
        return _sanitize_identifier(value)

    def preferred_trace_path(self):
        """Return the preferred source path without asserting readability."""
        return self.mask_crop_path or self.visual_crop_path or self.crop_path

    def visual_fallback_paths(self):
        """Return distinct visual fallback candidates in priority order."""
        paths = []
        for path in (self.visual_crop_path, self.crop_path):
            if path and path not in paths:
                paths.append(path)
        return paths

    def to_dict(self):
        """Return JSON-ready input metadata."""
        return {
            "crop_path": self.crop_path,
            "mask_crop_path": self.mask_crop_path,
            "visual_crop_path": self.visual_crop_path,
            "context_crop_path": self.context_crop_path,
            "output_dir": self.output_dir,
            "source_layer_group_id": self.source_layer_group_id,
            "document_id": self.document_id,
            "text_unit_id": self.text_unit_id,
            "source_group_id": self.source_group_id,
            "layer": self.layer,
            "document_bbox": self.document_bbox,
            "final_bbox": self.final_bbox,
            "stable_unit_id": self.stable_unit_id(),
            "preferred_trace_path": self.preferred_trace_path(),
        }


class PixelPoint:
    """Represent one integer-coordinate pixel."""

    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def to_tuple(self):
        return (self.x, self.y)

    def to_dict(self):
        return {"x": self.x, "y": self.y}

    def neighbors(self):
        """Return all eight neighboring coordinates."""
        return [
            PixelPoint(self.x + dx, self.y + dy)
            for dx, dy in NEIGHBOR_OFFSETS
        ]

    def is_same(self, other):
        return self.to_tuple() == other.to_tuple()


class SkeletonPoint(PixelPoint):
    """Represent one pixel in a thinned centerline graph."""


class BoundingBox:
    """Store an exclusive-coordinate bounding box."""

    def __init__(self, x1, y1, x2, y2):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)

    @classmethod
    def from_points(cls, points):
        """Build a box enclosing every supplied point."""
        if not points:
            raise ValueError("BoundingBox requires at least one point.")
        return cls(
            min(point.x for point in points),
            min(point.y for point in points),
            max(point.x for point in points) + 1,
            max(point.y for point in points) + 1,
        )

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def area(self):
        return self.width() * self.height()

    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def contains_point(self, point):
        return self.x1 <= point.x < self.x2 and self.y1 <= point.y < self.y2

    def to_dict(self):
        center_x, center_y = self.center()
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width(),
            "height": self.height(),
            "area": self.area(),
            "center": {"x": center_x, "y": center_y},
        }


class InkComponent:
    """Represent one accepted 8-connected island of ink pixels."""

    def __init__(self, component_id, points):
        if not points:
            raise ValueError("InkComponent requires at least one PixelPoint.")
        self.component_id = int(component_id)
        self.points = sorted(list(points), key=_coordinate_key)
        self.bounding_box = BoundingBox.from_points(self.points)
        self.point_lookup = {point.to_tuple() for point in self.points}

    def point_count(self):
        return len(self.points)

    def width(self):
        return self.bounding_box.width()

    def height(self):
        return self.bounding_box.height()

    def area(self):
        return self.bounding_box.area()

    def center(self):
        return self.bounding_box.center()

    def ink_density(self):
        return self.point_count() / self.area() if self.area() else 0.0

    def contains_point(self, point):
        return self.bounding_box.contains_point(point)

    def has_ink_at(self, point):
        return point.to_tuple() in self.point_lookup

    def to_dict(self):
        center_x, center_y = self.center()
        return {
            "component_id": self.component_id,
            "point_count": self.point_count(),
            "width": self.width(),
            "height": self.height(),
            "area": self.area(),
            "ink_density": self.ink_density(),
            "center": {"x": center_x, "y": center_y},
            "bounding_box": self.bounding_box.to_dict(),
        }

    def to_debug_dict(self, include_points=False):
        data = self.to_dict()
        if include_points:
            data["points"] = [point.to_dict() for point in self.points]
        return data

class InkHole:
    """Represent one enclosed background hole inside an ink component."""

    def __init__(self, hole_id, component_id, points):
        if not points:
            raise ValueError("InkHole requires at least one PixelPoint.")

        self.hole_id = int(hole_id)
        self.component_id = int(component_id)
        self.points = sorted(list(points), key=_coordinate_key)
        self.bounding_box = BoundingBox.from_points(self.points)

    def point_count(self):
        return len(self.points)

    def area(self):
        return self.bounding_box.area()

    def center(self):
        return self.bounding_box.center()

    def to_dict(self):
        center_x, center_y = self.center()
        return {
            "hole_id": self.hole_id,
            "component_id": self.component_id,
            "point_count": self.point_count(),
            "area": self.area(),
            "bounding_box": self.bounding_box.to_dict(),
            "center": {"x": center_x, "y": center_y},
        }


class TracePath:
    """Represent one ordered logical segment in the contracted graph."""

    def __init__(
        self,
        path_id,
        points,
        is_closed=False,
        start_node_type=None,
        end_node_type=None,
        start_junction_cluster_id=None,
        end_junction_cluster_id=None,
        stop_reason=None,
        is_short=False,
        merged_from_path_ids=None,
    ):
        if not points:
            raise ValueError("TracePath requires at least one SkeletonPoint.")
        self.path_id = int(path_id)
        self.points = list(points)
        self.is_closed = bool(is_closed)
        self.start_node_type = start_node_type
        self.end_node_type = end_node_type
        self.start_junction_cluster_id = start_junction_cluster_id
        self.end_junction_cluster_id = end_junction_cluster_id
        self.stop_reason = stop_reason
        self.is_short = bool(is_short)
        self.merged_from_path_ids = list(merged_from_path_ids or [])
        self.bounding_box = BoundingBox.from_points(self.points)

    def point_count(self):
        return len(self.points)

    def start_point(self):
        return self.points[0]

    def end_point(self):
        return self.points[-1]

    def length(self):
        """Return geometric pixel length using Euclidean step distances."""
        return sum(
            math.hypot(
                point_b.x - point_a.x,
                point_b.y - point_a.y,
            )
            for point_a, point_b in zip(self.points, self.points[1:])
        )

    def to_dict(self):
        return {
            "path_id": self.path_id,
            "point_count": self.point_count(),
            "length": self.length(),
            "start_point": self.start_point().to_dict(),
            "end_point": self.end_point().to_dict(),
            "bounding_box": self.bounding_box.to_dict(),
            "stop_reason": self.stop_reason,
            "is_closed": self.is_closed,
            "is_short": self.is_short,
            "start_node_type": self.start_node_type,
            "end_node_type": self.end_node_type,
            "start_junction_cluster_id": self.start_junction_cluster_id,
            "end_junction_cluster_id": self.end_junction_cluster_id,
            "merged_from_path_ids": self.merged_from_path_ids,
        }

    def to_debug_dict(self, include_points=False):
        data = self.to_dict()
        if include_points:
            data["points"] = [point.to_dict() for point in self.points]
        return data


class TraceLandmark:
    """
    Represent one geometric breaking point along an ordered TracePath.

    Landmarks are not graph nodes.
    Graph nodes describe topology: endpoints, junctions, loops.
    Landmarks describe shape: peaks, valleys, turns, extrema.
    """

    def __init__(
        self,
        landmark_id,
        path_id,
        point,
        landmark_type,
        index_on_path,
        prominence=0,
    ):
        self.landmark_id = int(landmark_id)
        self.path_id = int(path_id)
        self.point = point
        self.landmark_type = str(landmark_type)
        self.index_on_path = int(index_on_path)
        self.prominence = float(prominence)

    def to_dict(self):
        return {
            "landmark_id": self.landmark_id,
            "path_id": self.path_id,
            "landmark_type": self.landmark_type,
            "index_on_path": self.index_on_path,
            "prominence": self.prominence,
            "point": self.point.to_dict(),
        }


class TraceResult:
    """Store JSON-ready ScribeTrace evidence without pretending it is OCR."""

    def __init__(
        self,
        expert_name=EXPERT_NAME,
        status="not_run",
        trace_input=None,
        settings=None,
        components=None,
        error=None,
        reason=None,
        debug_paths=None,
        metrics=None,
        trace_paths=None,
        landmarks=None,
        feature_vector=None,
        result_json_path=None,
        ink_holes=None,
    ):
        self.expert_name = expert_name
        self.status = status
        self.trace_input = trace_input
        self.settings = settings
        self.components = components or []
        self.error = error
        self.reason = reason
        self.debug_paths = debug_paths or {}
        self.metrics = metrics or {}
        self.trace_paths = trace_paths or []
        self.landmarks = landmarks or []
        self.feature_vector = feature_vector
        self.result_json_path = result_json_path
        self.ink_holes = ink_holes or []

    def component_count(self):
        return len(self.components)

    def ml_features_dict(self):
        if self.feature_vector is None:
            return None

        quality_flags = {
            "status": self.status,
            "has_error": self.error is not None,
            "component_limit_exceeded": self.reason == "component_limit_exceeded",
            "fallback_used": bool(self.metrics.get("fallback_used", False)),
            "has_skeleton": self.metrics.get("skeleton_point_count", 0) > 0,
            "has_paths": len(self.trace_paths) > 0,
            "has_ink_holes": len(self.ink_holes) > 0,
        }

        return {
            "schema_version": "scribetrace_ml_v1",
            "label": None,
            "feature_names": self.feature_vector.feature_names,
            "vector": self.feature_vector.vector,
            "sequence": self.feature_vector.sequence,
            "sequence_string": self.feature_vector.sequence_string,
            "quality_flags": quality_flags,
        }

    def to_dict(self):
        return {
            "expert_name": self.expert_name,
            "status": self.status,
            "reason": self.reason,
            "component_count": self.component_count(),
            "trace_input": self.trace_input.to_dict() if self.trace_input else None,
            "settings": self.settings.to_dict() if self.settings else None,
            "components": [component.to_dict() for component in self.components],
            "error": self.error,
            "debug_paths": self.debug_paths,
            "result_json_path": self.result_json_path,
            "metrics": self.metrics,
            "path_count": len(self.trace_paths),
            "trace_paths": [path.to_dict() for path in self.trace_paths],
            "landmark_count": len(self.landmarks),
            "landmarks": [landmark.to_dict() for landmark in self.landmarks],
            "feature_vector": (
                self.feature_vector.to_dict()
                if self.feature_vector is not None
                else None
            ),
            "ml_features": self.ml_features_dict(),
            "ink_hole_count": len(self.ink_holes),
            "ink_holes": [hole.to_dict() for hole in self.ink_holes],
        }

    def to_debug_dict(self, include_points=False):
        data = self.to_dict()
        data["components"] = [
            component.to_debug_dict(include_points=include_points)
            for component in self.components
        ]
        data["trace_paths"] = [
            path.to_debug_dict(include_points=include_points)
            for path in self.trace_paths
        ]
        return data


class TraceFeatureVector:
    """
    Store ML-ready ScribeTrace features.

    vector:
        Numeric values for classical ML models.

    sequence:
        Ordered symbolic landmark/path tokens for sequence models.

    feature_names:
        Names matching vector positions.
    """

    def __init__(
        self,
        vector=None,
        feature_names=None,
        sequence=None,
        sequence_string="",
    ):
        self.vector = list(vector or [])
        self.feature_names = list(feature_names or [])
        self.sequence = list(sequence or [])
        self.sequence_string = sequence_string

    def to_dict(self):
        return {
            "feature_names": self.feature_names,
            "vector": self.vector,
            "sequence": self.sequence,
            "sequence_string": self.sequence_string,
        }

