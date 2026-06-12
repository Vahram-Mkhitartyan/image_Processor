"""Binary mask loading, connected components, and enclosed-hole analysis."""

import os

import cv2
import numpy as np

from .trace_common import coordinate_key
from .trace_models import InkComponent, InkHole, PixelPoint
from .trace_settings import normalize_trace_settings

_coordinate_key = coordinate_key


def _box_overlap_area(box_a, box_b):
    """Return the intersection area of two exclusive-coordinate boxes."""
    x1 = max(box_a.x1, box_b.x1)
    y1 = max(box_a.y1, box_b.y1)
    x2 = min(box_a.x2, box_b.x2)
    y2 = min(box_a.y2, box_b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def _box_area(box):
    """Return a non-negative exclusive-coordinate bounding-box area."""
    return max(0, box.width()) * max(0, box.height())


def match_ink_holes_to_closed_paths(ink_holes, trace_paths):
    """Match visual ink holes to closed skeleton paths by box overlap."""
    matches = []
    closed_paths = [path for path in trace_paths if path.is_closed]
    for hole in ink_holes:
        best_path = None
        best_score = 0.0
        for path in closed_paths:
            overlap = _box_overlap_area(hole.bounding_box, path.bounding_box)
            denominator = max(
                1,
                min(_box_area(hole.bounding_box), _box_area(path.bounding_box)),
            )
            score = overlap / denominator
            if score > best_score:
                best_score = score
                best_path = path
        if best_path is not None and best_score >= 0.25:
            matches.append(
                {
                    "hole_id": hole.hole_id,
                    "component_id": hole.component_id,
                    "matched_path_id": best_path.path_id,
                    "overlap_score": float(best_score),
                }
            )
    return matches

class TraceMaskAdapter:
    """Resolve and binarize an exact mask or a visual Otsu fallback."""

    def __init__(self, settings=None):
        self.settings = normalize_trace_settings(settings)

    def load_grayscale(self, image_path):
        """Load a readable grayscale image or raise a precise error."""
        if not image_path:
            raise ValueError("Image path is missing.")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image

    def _try_load(self, image_path):
        """Return a grayscale image only when the candidate is readable."""
        if not image_path or not os.path.isfile(image_path):
            return None
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    def binarize(self, grayscale, source_type):
        """Convert input to white ink on black according to source polarity."""
        mode = self.settings.ink_threshold_mode
        is_visual = source_type == "visual_crop_fallback"
        inverse_flag = cv2.THRESH_BINARY_INV if is_visual else cv2.THRESH_BINARY

        if mode == "auto":
            effective_mode = "otsu_inverse" if is_visual else "binary_128"
            if is_visual:
                _, binary = cv2.threshold(
                    grayscale, 0, 255, inverse_flag | cv2.THRESH_OTSU
                )
            else:
                _, binary = cv2.threshold(grayscale, 128, 255, inverse_flag)
        elif mode == "otsu":
            effective_mode = "otsu_inverse" if is_visual else "otsu"
            _, binary = cv2.threshold(
                grayscale, 0, 255, inverse_flag | cv2.THRESH_OTSU
            )
        elif mode == "fixed":
            effective_mode = (
                "fixed_inverse" if is_visual else "fixed"
            )
            _, binary = cv2.threshold(
                grayscale,
                self.settings.fixed_threshold_value,
                255,
                inverse_flag,
            )
        else:
            effective_mode = "binary_inverse_128" if is_visual else "binary_128"
            _, binary = cv2.threshold(grayscale, 128, 255, inverse_flag)

        return binary, effective_mode

    def resolve_trace_mask(self, trace_input):
        """Resolve the exact mask first, then a readable visual fallback."""
        mask = self._try_load(trace_input.mask_crop_path)
        if mask is not None:
            source_path = trace_input.mask_crop_path
            source_type = "analysis_mask"
            fallback_used = False
        else:
            source_path = None
            mask = None
            for candidate in trace_input.visual_fallback_paths():
                candidate_image = self._try_load(candidate)
                if candidate_image is not None:
                    source_path = candidate
                    mask = candidate_image
                    break
            if mask is None:
                raise FileNotFoundError(
                    "No readable ScribeTrace source. "
                    f"analysis_mask={trace_input.mask_crop_path!r}, "
                    f"visual_candidates={trace_input.visual_fallback_paths()!r}"
                )
            source_type = "visual_crop_fallback"
            fallback_used = True

        binary, effective_mode = self.binarize(mask, source_type)
        provenance = {
            "source_path": source_path,
            "source_type": source_type,
            "fallback_used": fallback_used,
            "requested_threshold_mode": self.settings.ink_threshold_mode,
            "threshold_mode": effective_mode,
        }
        return binary, provenance

    def load_trace_mask(self, mask_path):
        """Compatibility helper for loading a white-on-black mask path."""
        grayscale = self.load_grayscale(mask_path)
        binary, _ = self.binarize(grayscale, "analysis_mask")
        return binary


class InkComponentExtractor:
    """Extract components and rebuild a mask containing accepted ink only."""

    def __init__(self, settings=None):
        self.settings = normalize_trace_settings(settings)

    def is_ink_pixel(self, pixel_value):
        return int(np.asarray(pixel_value).flat[0]) > 128

    def collect_ink_points(self, mask):
        """Collect white pixels in deterministic row-major order."""
        coordinates = np.argwhere(np.asarray(mask) > 128)
        return [PixelPoint(x, y) for y, x in coordinates]

    def group_connected_points(self, points):
        """Partition points into deterministic 8-connected components."""
        point_lookup = {point.to_tuple(): point for point in points}
        unvisited = set(point_lookup)
        groups = []

        while unvisited:
            start_coord = min(unvisited, key=lambda value: (value[1], value[0]))
            unvisited.remove(start_coord)
            stack = [point_lookup[start_coord]]
            group = []

            while stack:
                current = stack.pop()
                group.append(current)
                neighbors = sorted(current.neighbors(), key=_coordinate_key, reverse=True)
                for neighbor in neighbors:
                    coordinate = neighbor.to_tuple()
                    if coordinate in unvisited:
                        unvisited.remove(coordinate)
                        stack.append(point_lookup[coordinate])

            groups.append(sorted(group, key=_coordinate_key))

        groups.sort(
            key=lambda group: (
                min(point.y for point in group),
                min(point.x for point in group),
                max(point.y for point in group),
                max(point.x for point in group),
            )
        )
        return groups

    def analyze_mask(self, mask):
        """Return accepted components, rejected counts, and the cleaned mask."""
        points = self.collect_ink_points(mask)
        groups = self.group_connected_points(points)
        accepted_groups = [
            group
            for group in groups
            if len(group) >= self.settings.minimum_ink_pixels
        ]
        components = [
            InkComponent(component_id, group)
            for component_id, group in enumerate(accepted_groups)
        ]
        cleaned_mask = np.zeros(np.asarray(mask).shape[:2], dtype=np.uint8)
        for component in components:
            for point in component.points:
                cleaned_mask[point.y, point.x] = 255

        return {
            "components": components,
            "cleaned_mask": cleaned_mask,
            "raw_component_count": len(groups),
            "accepted_component_count": len(components),
            "rejected_small_component_count": len(groups) - len(components),
            "raw_ink_pixel_count": len(points),
            "accepted_ink_pixel_count": sum(
                component.point_count() for component in components
            ),
        }

    def extract_from_mask(self, mask):
        return self.analyze_mask(mask)["components"]

    def extract_from_mask_path(self, mask_path):
        adapter = TraceMaskAdapter(self.settings)
        return self.extract_from_mask(adapter.load_trace_mask(mask_path))

class InkHoleDetector:
    """
    Detect enclosed background holes inside accepted ink components.

    This catches visual loops/counters even when skeletonization breaks the
    centerline into endpoint/junction paths.
    """

    def __init__(self, settings=None):
        self.settings = normalize_trace_settings(settings)

    def _component_mask(self, component, full_shape):
        mask = np.zeros(full_shape[:2], dtype=np.uint8)

        for point in component.points:
            mask[point.y, point.x] = 255

        return mask

    def _extract_holes_for_component(self, component, full_shape, starting_hole_id):
        component_mask = self._component_mask(component, full_shape)
        box = component.bounding_box

        # Crop with one-pixel padding so flood fill can identify outside background.
        x1 = max(0, box.x1 - 1)
        y1 = max(0, box.y1 - 1)
        x2 = min(component_mask.shape[1], box.x2 + 1)
        y2 = min(component_mask.shape[0], box.y2 + 1)

        crop = component_mask[y1:y2, x1:x2]
        background = (crop == 0).astype(np.uint8) * 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            background,
            8,
        )

        holes = []
        next_hole_id = starting_hole_id

        crop_height, crop_width = crop.shape[:2]

        for label_id in range(1, num_labels):
            left = stats[label_id, cv2.CC_STAT_LEFT]
            top = stats[label_id, cv2.CC_STAT_TOP]
            width = stats[label_id, cv2.CC_STAT_WIDTH]
            height = stats[label_id, cv2.CC_STAT_HEIGHT]
            area = stats[label_id, cv2.CC_STAT_AREA]

            touches_crop_border = (
                left == 0
                or top == 0
                or left + width >= crop_width
                or top + height >= crop_height
            )

            # Background connected to crop border is outside air, not a hole.
            if touches_crop_border:
                continue

            if area < 2:
                continue

            coordinates = np.argwhere(labels == label_id)
            points = [
                PixelPoint(x=int(local_x + x1), y=int(local_y + y1))
                for local_y, local_x in coordinates
            ]

            holes.append(
                InkHole(
                    hole_id=next_hole_id,
                    component_id=component.component_id,
                    points=points,
                )
            )
            next_hole_id += 1

        return holes

    def detect_holes(self, components, full_shape):
        holes = []
        next_hole_id = 0

        for component in components:
            component_holes = self._extract_holes_for_component(
                component=component,
                full_shape=full_shape,
                starting_hole_id=next_hole_id,
            )
            holes.extend(component_holes)
            next_hole_id += len(component_holes)

        return holes
