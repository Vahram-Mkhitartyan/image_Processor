"""ScribeTrace reconstruction controller.

This module keeps reconstruction separate from damage detection.

Responsibilities:
    keep h0_original as a no-repair baseline
    receive a damage verdict and/or allowed defense list
    run only the allowed defense tools
    retrace every candidate mask
    score before/after topology and pixel intervention
    select the best accepted repair or keep original

Compatibility:
    class name TheoreticalReconstructor is preserved so existing imports keep working
    result keeps the fields used by the trainer/exporter:
        selected_hypothesis_id
        selected_feature_source
        selected_feature_vector
        accepted_count
        candidate_count
        total_candidate_count
        attack_mapping
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Any, Iterable

import cv2
import numpy as np

from .trace_common import sanitize_identifier
from .trace_features import TraceFeatureEncoder
from .trace_inference import predict_rf_candidates
from .trace_masks import InkComponentExtractor, InkHoleDetector
from .trace_models import TraceResult
from .trace_paths import TraceLandmarkExtractor, TracePathExtractor
from .trace_settings import normalize_trace_settings
from .trace_skeleton import (
    SkeletonGraph,
    SkeletonPointExtractor,
    TraceSkeletonizer,
)


# ============================================================
# Defaults and routing
# ============================================================

DEFAULT_ATTACK_TO_DEFENSES = {
    "clean": ["no_repair"],
    "unknown": ["no_repair"],
    "light_cut": [
        "horizontal_gap_closing",
        "vertical_gap_closing",
        "endpoint_bridge",
    ],
    "light_blur": [
        "threshold_normalization",
    ],
    "scanner_noise": [
        "component_denoising",
        "median_denoising",
    ],
    "light_erosion": [
        "conservative_stroke_recovery",
        "horizontal_gap_closing",
        "vertical_gap_closing",
        "endpoint_bridge",
    ],
    "ink_overlap": [
        "contamination_opening",
    ],
    "stamp_interference": [
        "linear_artifact_removal",
        "contamination_opening",
    ],
    "bleed_through": [
        "threshold_normalization",
        "component_denoising",
        "median_denoising",
    ],
    "edge_crop_loss": [
        "border_continuation",
    ],
    "threshold_failure": [
        "threshold_normalization",
        "horizontal_gap_closing",
        "vertical_gap_closing",
        "endpoint_bridge",
    ],
    "compression_artifacts": [
        "threshold_normalization",
        "median_denoising",
    ],
}

DEFENSES_THAT_ADD_INK = {
    "horizontal_gap_closing",
    "vertical_gap_closing",
    "endpoint_bridge",
    "conservative_stroke_recovery",
    "border_continuation",
}

DEFENSES_THAT_REMOVE_INK = {
    "component_denoising",
    "median_denoising",
    "contamination_opening",
    "linear_artifact_removal",
}

DEFENSES_THAT_CAN_ADD_AND_REMOVE = {
    "threshold_normalization",
}

SUPPORTED_DEFENSES = (
    "no_repair",
    "horizontal_gap_closing",
    "vertical_gap_closing",
    "endpoint_bridge",
    "conservative_stroke_recovery",
    "component_denoising",
    "median_denoising",
    "contamination_opening",
    "linear_artifact_removal",
    "threshold_normalization",
    "border_continuation",
)


# ============================================================
# Data containers
# ============================================================

@dataclass
class DefenseCandidate:
    """One proposed reconstruction candidate produced by one defense tool."""

    hypothesis_id: str
    defense_name: str
    mask: np.ndarray
    added_mask: np.ndarray
    removed_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Small helpers
# ============================================================

def _setting(settings, name: str, default: Any) -> Any:
    """Read a setting with a safe fallback for older checkpoints."""
    return getattr(settings, name, default)


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize any mask-like array into 0/255 uint8."""
    if mask is None:
        raise ValueError("mask cannot be None")
    array = np.asarray(mask)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    return np.where(array > 0, 255, 0).astype(np.uint8)


def _changed_masks(original: np.ndarray, candidate: np.ndarray):
    """Return added and removed foreground masks."""
    original = _binary_mask(original)
    candidate = _binary_mask(candidate)
    added_mask = cv2.bitwise_and(candidate, cv2.bitwise_not(original))
    removed_mask = cv2.bitwise_and(original, cv2.bitwise_not(candidate))
    return added_mask, removed_mask


def _angle_degrees(vector_a, vector_b):
    """Return the unsigned angle between two non-zero vectors."""
    magnitude_a = math.hypot(*vector_a)
    magnitude_b = math.hypot(*vector_b)
    if magnitude_a == 0 or magnitude_b == 0:
        return 180.0
    cosine = (
        vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
    ) / (magnitude_a * magnitude_b)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _graph_component_lookup(graph):
    """Map every skeleton coordinate to a deterministic component ID."""
    unvisited = set(graph.point_lookup)
    lookup = {}
    component_id = 0
    while unvisited:
        start = min(unvisited, key=lambda value: (value[1], value[0]))
        unvisited.remove(start)
        stack = [start]
        while stack:
            coordinate = stack.pop()
            lookup[coordinate] = component_id
            point = graph.point_lookup[coordinate]
            for neighbor in reversed(graph.neighbors_of(point)):
                candidate = neighbor.to_tuple()
                if candidate in unvisited:
                    unvisited.remove(candidate)
                    stack.append(candidate)
        component_id += 1
    return lookup


def _endpoint_tangent(endpoint, trace_paths, tangent_points):
    """Estimate the outward continuation direction at one path endpoint."""
    coordinate = endpoint.to_tuple()
    candidates = []
    for path in trace_paths:
        points = path.points
        if path.is_closed or len(points) < 2:
            continue
        if points[0].to_tuple() == coordinate:
            index = min(len(points) - 1, tangent_points - 1)
            interior = points[index]
            candidates.append(
                (
                    path.path_id,
                    endpoint.x - interior.x,
                    endpoint.y - interior.y,
                )
            )
        if points[-1].to_tuple() == coordinate:
            index = max(0, len(points) - tangent_points)
            interior = points[index]
            candidates.append(
                (
                    path.path_id,
                    endpoint.x - interior.x,
                    endpoint.y - interior.y,
                )
            )
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])


def _line_coordinates(point_a, point_b):
    """Rasterize a one-pixel candidate bridge in deterministic order."""
    width = abs(point_b.x - point_a.x) + 3
    height = abs(point_b.y - point_a.y) + 3
    x_offset = min(point_a.x, point_b.x) - 1
    y_offset = min(point_a.y, point_b.y) - 1
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.line(
        canvas,
        (point_a.x - x_offset, point_a.y - y_offset),
        (point_b.x - x_offset, point_b.y - y_offset),
        255,
        1,
        cv2.LINE_8,
    )
    return [
        (int(x + x_offset), int(y + y_offset))
        for y, x in np.argwhere(canvas > 0)
    ]


def _estimate_bridge_thickness(mask, point_a, point_b):
    """Estimate conservative stroke width from endpoint distance values."""
    binary = (mask > 0).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    radii = [
        float(distance[point.y, point.x])
        for point in (point_a, point_b)
        if 0 <= point.y < distance.shape[0]
        and 0 <= point.x < distance.shape[1]
    ]
    radius = sum(radii) / len(radii) if radii else 1.0
    return max(1, min(4, int(round(radius * 1.5))))


def _trace_mask(mask, settings):
    """Retrace a hypothetical mask without invoking reconstruction recursively."""
    mask = _binary_mask(mask)
    component_analysis = InkComponentExtractor(settings).analyze_mask(mask)
    components = component_analysis["components"]
    cleaned_mask = component_analysis["cleaned_mask"]
    ink_holes = InkHoleDetector(settings).detect_holes(
        components,
        cleaned_mask.shape,
    )
    skeleton = TraceSkeletonizer(settings).skeletonize(cleaned_mask)
    points = SkeletonPointExtractor().extract_points(skeleton)
    graph = SkeletonGraph(points)
    path_extractor = TracePathExtractor(settings)
    paths = path_extractor.extract_paths(graph)
    landmarks = TraceLandmarkExtractor(settings).extract_landmarks(paths)
    metrics = {
        key: value
        for key, value in component_analysis.items()
        if key not in {"components", "cleaned_mask"}
    }
    metrics.update(path_extractor.metrics)
    metrics.update(
        {
            "skeleton_point_count": len(points),
            "skeleton_graph": graph.to_dict(),
            "ink_hole_count": len(ink_holes),
            "landmark_count": len(landmarks),
        }
    )
    feature_vector = TraceFeatureEncoder(settings).encode(
        components=components,
        skeleton_graph=graph,
        trace_paths=paths,
        landmarks=landmarks,
        metrics=metrics,
        ink_holes=ink_holes,
    )
    result = TraceResult(
        status="completed",
        settings=settings,
        components=components,
        trace_paths=paths,
        landmarks=landmarks,
        feature_vector=feature_vector,
        ink_holes=ink_holes,
        metrics=metrics,
    )
    return {
        "result": result,
        "mask": cleaned_mask,
        "skeleton": skeleton,
        "graph": graph,
        "paths": paths,
    }


def _topology_snapshot(trace_result):
    """Extract comparable topology counts from one trace result."""
    graph = trace_result.metrics.get("skeleton_graph") or {}
    return {
        "component_count": len(trace_result.components),
        "endpoint_count": int(graph.get("endpoint_count", 0)),
        "junction_cluster_count": int(graph.get("junction_cluster_count", 0)),
        "isolated_point_count": int(graph.get("isolated_point_count", 0)),
        "path_count": len(trace_result.trace_paths),
        "short_path_count": int(trace_result.metrics.get("short_path_count", 0)),
        "closed_loop_count": int(trace_result.metrics.get("closed_loop_count", 0)),
        "ink_hole_count": len(trace_result.ink_holes),
        "skeleton_point_count": int(
            trace_result.metrics.get("skeleton_point_count", 0)
        ),
    }


def _topology_delta(original, candidate):
    """Return before-after topology deltas. Positive values mean candidate has more."""
    keys = sorted(set(original) | set(candidate))
    return {
        key: int(candidate.get(key, 0) - original.get(key, 0))
        for key in keys
    }


def _recognition_snapshot(trace_result):
    """Return optional RF evidence without making it mandatory for repair."""
    try:
        candidates = predict_rf_candidates(trace_result, top_k=5)
    except Exception as error:
        return {
            "available": False,
            "top1_confidence": 0.0,
            "top5": [],
            "error": str(error),
        }
    return {
        "available": bool(candidates),
        "top1_confidence": (
            float(candidates[0]["confidence"]) if candidates else 0.0
        ),
        "top5": candidates,
        "error": None,
    }


def _topology_gain(original, candidate, defense_name):
    """Generic topology score for all defense types.

    The score intentionally stays conservative. It rewards simplification of
    fragmentation signals and penalizes invented complexity. For contamination
    removal defenses, losing endpoints/components is not automatically bad.
    """
    component_gain = max(0, original["component_count"] - candidate["component_count"])
    endpoint_gain = max(0, original["endpoint_count"] - candidate["endpoint_count"])
    isolated_gain = max(0, original["isolated_point_count"] - candidate["isolated_point_count"])
    short_path_gain = max(0, original["short_path_count"] - candidate["short_path_count"])

    new_components = max(0, candidate["component_count"] - original["component_count"])
    new_endpoints = max(0, candidate["endpoint_count"] - original["endpoint_count"])
    new_junctions = max(0, candidate["junction_cluster_count"] - original["junction_cluster_count"])
    new_short_paths = max(0, candidate["short_path_count"] - original["short_path_count"])
    lost_holes = max(0, original["ink_hole_count"] - candidate["ink_hole_count"])
    gained_holes = max(0, candidate["ink_hole_count"] - original["ink_hole_count"])

    if defense_name in DEFENSES_THAT_REMOVE_INK:
        gain = (
            0.30 * min(2, component_gain) / 2.0
            + 0.20 * min(2, endpoint_gain) / 2.0
            + 0.20 * min(1, isolated_gain)
            + 0.15 * min(2, short_path_gain) / 2.0
            - 0.30 * min(1, new_junctions)
            - 0.20 * min(1, lost_holes)
        )
    else:
        gain = (
            0.30 * min(2, component_gain) / 2.0
            + 0.30 * min(4, endpoint_gain) / 4.0
            + 0.15 * min(1, isolated_gain)
            + 0.15 * min(2, short_path_gain) / 2.0
            + 0.10 * min(1, gained_holes)
            - 0.25 * min(1, new_components)
            - 0.20 * min(2, new_endpoints) / 2.0
            - 0.30 * min(1, new_junctions)
            - 0.20 * min(1, new_short_paths)
            - 0.15 * min(1, lost_holes)
        )
    return max(-1.0, min(1.0, gain))


# ============================================================
# Reconstruction controller
# ============================================================

class TheoreticalReconstructor:
    """Generate, retrace, verify, and rank defense hypotheses.

    The name is preserved for compatibility with existing ScribeTrace code.
    In the new split, this class is the reconstruction controller/verifier.
    """

    def __init__(self, settings=None):
        self.settings = normalize_trace_settings(settings)

    # --------------------------------------------------------
    # Diagnosis / routing
    # --------------------------------------------------------

    def diagnose(self, mask, trace_result):
        """Describe topology damage signals without modifying the mask."""
        mask = _binary_mask(mask)
        topology = _topology_snapshot(trace_result)
        graph_summary = trace_result.metrics.get("skeleton_graph") or {}
        reasons = []

        if topology["component_count"] > 1:
            reasons.append("disconnected_ink_components")
        if topology["endpoint_count"] >= 4:
            reasons.append("abnormal_endpoint_count")
        if topology["isolated_point_count"] > 0:
            reasons.append("isolated_skeleton_points")
        if topology["short_path_count"] > 0:
            reasons.append("fragmented_short_paths")
        if topology["ink_hole_count"] > topology["closed_loop_count"]:
            reasons.append("fragmented_loop_topology")

        ink = mask > 0
        border_contacts = {
            "left": bool(np.any(ink[:, 0])),
            "right": bool(np.any(ink[:, -1])),
            "top": bool(np.any(ink[0, :])),
            "bottom": bool(np.any(ink[-1, :])),
        }
        if any(border_contacts.values()):
            reasons.append("border_clipping_possible")

        small_component_count = 0
        for component in getattr(trace_result, "components", []):
            area = getattr(component, "area", None)
            if area is not None and area <= _setting(
                self.settings,
                "reconstruction_noise_max_component_area",
                6,
            ):
                small_component_count += 1
        if small_component_count >= 3:
            reasons.append("many_small_components")

        return {
            "damage_suspected": bool(reasons),
            "damage_reasons": reasons,
            "topology": topology,
            "crossing_number_histogram": graph_summary.get(
                "crossing_number_histogram",
                {},
            ),
            "border_contacts": border_contacts,
            "small_component_count": int(small_component_count),
        }

    def _normalize_damage_verdict(self, damage_verdict):
        """Accept None, string, list, or dict verdicts and normalize them."""
        if damage_verdict is None:
            return {
                "label": "unknown",
                "confidence": 0.0,
                "source": "none",
                "labels": [],
            }
        if isinstance(damage_verdict, str):
            return {
                "label": damage_verdict,
                "confidence": 1.0,
                "source": "external",
                "labels": [damage_verdict],
            }
        if isinstance(damage_verdict, (list, tuple)):
            labels = [str(item) for item in damage_verdict if item]
            return {
                "label": labels[0] if labels else "unknown",
                "confidence": 1.0 if labels else 0.0,
                "source": "external_list",
                "labels": labels,
            }
        if isinstance(damage_verdict, dict):
            labels = damage_verdict.get("labels")
            if labels is None:
                label = damage_verdict.get("label") or damage_verdict.get("damage_type")
                labels = [label] if label else []
            labels = [str(item) for item in labels if item]
            label = str(damage_verdict.get("label") or (labels[0] if labels else "unknown"))
            return {
                **damage_verdict,
                "label": label,
                "confidence": float(damage_verdict.get("confidence", 0.0)),
                "source": str(damage_verdict.get("source", "external_dict")),
                "labels": labels,
            }
        return {
            "label": "unknown",
            "confidence": 0.0,
            "source": f"unsupported:{type(damage_verdict).__name__}",
            "labels": [],
        }

    def _resolve_allowed_defenses(self, damage_verdict, allowed_defenses, diagnosis):
        """Resolve the defense list from external routing or conservative fallback."""
        if allowed_defenses is not None:
            cleaned = []
            for defense_name in allowed_defenses:
                defense_name = str(defense_name)
                if defense_name == "no_repair":
                    continue
                if defense_name not in SUPPORTED_DEFENSES:
                    continue
                if defense_name not in cleaned:
                    cleaned.append(defense_name)
            return cleaned

        labels = list(damage_verdict.get("labels") or [])
        if not labels and damage_verdict.get("label"):
            labels = [damage_verdict["label"]]

        defenses = []
        for label in labels:
            for defense_name in DEFAULT_ATTACK_TO_DEFENSES.get(label, []):
                if defense_name == "no_repair":
                    continue
                if defense_name not in defenses:
                    defenses.append(defense_name)

        if defenses:
            return defenses

        # Fallback when no damage model is connected yet.
        reasons = set(diagnosis.get("damage_reasons", []))
        if not reasons:
            return []
        if {
            "disconnected_ink_components",
            "abnormal_endpoint_count",
            "fragmented_short_paths",
            "fragmented_loop_topology",
        } & reasons:
            defenses.extend(
                [
                    "horizontal_gap_closing",
                    "vertical_gap_closing",
                    "endpoint_bridge",
                ]
            )
        if "many_small_components" in reasons or "isolated_skeleton_points" in reasons:
            defenses.extend(["component_denoising", "median_denoising"])
        if "border_clipping_possible" in reasons:
            defenses.append("border_continuation")

        cleaned = []
        for defense_name in defenses:
            if defense_name not in cleaned:
                cleaned.append(defense_name)
        return cleaned

    # --------------------------------------------------------
    # Candidate generation
    # --------------------------------------------------------

    def _make_candidate(self, hypothesis_id, defense_name, original_mask, candidate_mask, metadata=None):
        """Create a DefenseCandidate and compute added/removed masks."""
        original_mask = _binary_mask(original_mask)
        candidate_mask = _binary_mask(candidate_mask)
        added_mask, removed_mask = _changed_masks(original_mask, candidate_mask)
        return DefenseCandidate(
            hypothesis_id=hypothesis_id,
            defense_name=defense_name,
            mask=candidate_mask,
            added_mask=added_mask,
            removed_mask=removed_mask,
            metadata=dict(metadata or {}),
        )

    def _gap_closing_candidate(self, mask, defense_name, index):
        """Generate horizontal or vertical morphological gap closing candidate."""
        mask = _binary_mask(mask)
        gap_px = int(_setting(self.settings, "reconstruction_gap_close_px", 3))
        gap_px = max(1, min(9, gap_px))

        if defense_name == "horizontal_gap_closing":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_px, 1))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap_px))

        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        if np.array_equal(mask, closed):
            return []
        return [
            self._make_candidate(
                f"h{index}_{defense_name}",
                defense_name,
                mask,
                closed,
                {"kernel_size": gap_px},
            )
        ]

    def _conservative_stroke_recovery_candidate(self, mask, index):
        """Generate a conservative stroke recovery candidate by light cross dilation."""
        mask = _binary_mask(mask)
        kernel = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=np.uint8,
        )
        recovered = cv2.dilate(mask, kernel, iterations=1)
        if np.array_equal(mask, recovered):
            return []
        return [
            self._make_candidate(
                f"h{index}_conservative_stroke_recovery",
                "conservative_stroke_recovery",
                mask,
                recovered,
                {"kernel": "3x3_cross", "iterations": 1},
            )
        ]

    def _component_denoising_candidate(self, mask, index):
        """Remove tiny isolated components."""
        mask = _binary_mask(mask)
        max_area = int(_setting(self.settings, "reconstruction_noise_max_component_area", 6))
        max_ratio = float(_setting(self.settings, "reconstruction_noise_max_component_ratio", 0.04))
        total_ink = max(1, int(np.count_nonzero(mask)))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        cleaned = np.zeros_like(mask)
        removed_components = []
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            ratio = area / total_ink
            if area <= max_area or ratio <= max_ratio / 4.0:
                removed_components.append({"label_id": int(label_id), "area": area, "ratio": ratio})
                continue
            cleaned[labels == label_id] = 255
        if np.array_equal(mask, cleaned):
            return []
        return [
            self._make_candidate(
                f"h{index}_component_denoising",
                "component_denoising",
                mask,
                cleaned,
                {
                    "max_component_area": max_area,
                    "max_component_ratio": max_ratio,
                    "removed_components": removed_components[:50],
                    "removed_component_count": len(removed_components),
                },
            )
        ]

    def _median_denoising_candidate(self, mask, index):
        """Median denoising candidate."""
        mask = _binary_mask(mask)
        denoised = cv2.medianBlur(mask, 3)
        denoised = np.where(denoised > 127, 255, 0).astype(np.uint8)
        if np.array_equal(mask, denoised):
            return []
        return [
            self._make_candidate(
                f"h{index}_median_denoising",
                "median_denoising",
                mask,
                denoised,
                {"kernel_size": 3},
            )
        ]

    def _contamination_opening_candidate(self, mask, index):
        """Morphological opening for contamination-like extra ink."""
        mask = _binary_mask(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        if np.array_equal(mask, opened):
            return []
        return [
            self._make_candidate(
                f"h{index}_contamination_opening",
                "contamination_opening",
                mask,
                opened,
                {"kernel": "3x3_cross", "iterations": 1},
            )
        ]

    def _linear_artifact_removal_candidate(self, mask, index):
        """Remove very long horizontal/vertical line artifacts."""
        mask = _binary_mask(mask)
        height, width = mask.shape[:2]
        cleaned = mask.copy()
        artifact_mask = np.zeros_like(mask)

        horizontal_min = max(8, int(width * 0.65))
        vertical_min = max(8, int(height * 0.65))

        # Horizontal runs.
        for y in range(height):
            xs = np.where(mask[y, :] > 0)[0]
            if len(xs) < horizontal_min:
                continue
            # Split into contiguous runs.
            start = xs[0]
            previous = xs[0]
            for x in xs[1:].tolist() + [None]:
                if x is not None and x == previous + 1:
                    previous = x
                    continue
                run_length = previous - start + 1
                if run_length >= horizontal_min:
                    artifact_mask[y, start:previous + 1] = 255
                if x is not None:
                    start = x
                    previous = x

        # Vertical runs.
        for x in range(width):
            ys = np.where(mask[:, x] > 0)[0]
            if len(ys) < vertical_min:
                continue
            start = ys[0]
            previous = ys[0]
            for y in ys[1:].tolist() + [None]:
                if y is not None and y == previous + 1:
                    previous = y
                    continue
                run_length = previous - start + 1
                if run_length >= vertical_min:
                    artifact_mask[start:previous + 1, x] = 255
                if y is not None:
                    start = y
                    previous = y

        cleaned[artifact_mask > 0] = 0
        if np.array_equal(mask, cleaned):
            return []
        return [
            self._make_candidate(
                f"h{index}_linear_artifact_removal",
                "linear_artifact_removal",
                mask,
                cleaned,
                {
                    "horizontal_min_run": horizontal_min,
                    "vertical_min_run": vertical_min,
                    "removed_artifact_pixels": int(np.count_nonzero(artifact_mask)),
                },
            )
        ]

    def _threshold_normalization_candidate(self, mask, index):
        """Stabilize a mask that may have threshold/compression artifacts."""
        mask = _binary_mask(mask)
        median = cv2.medianBlur(mask, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        normalized = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel, iterations=1)
        normalized = np.where(normalized > 127, 255, 0).astype(np.uint8)
        if np.array_equal(mask, normalized):
            return []
        return [
            self._make_candidate(
                f"h{index}_threshold_normalization",
                "threshold_normalization",
                mask,
                normalized,
                {"median_kernel": 3, "closing_kernel": "3x3_cross"},
            )
        ]

    def _border_continuation_candidate(self, mask, diagnosis, index):
        """Extend strokes that touch the crop border by a few pixels inward.

        This is intentionally conservative. It only acts when the current mask
        already touches an image border.
        """
        mask = _binary_mask(mask)
        extension_px = int(_setting(self.settings, "reconstruction_border_extension_px", 3))
        extension_px = max(1, min(8, extension_px))
        height, width = mask.shape[:2]
        extended = mask.copy()
        ink = mask > 0
        contacts = diagnosis.get("border_contacts") or {}

        if contacts.get("left"):
            ys = np.where(ink[:, 0])[0]
            for y in ys:
                x2 = min(width - 1, extension_px)
                cv2.line(extended, (0, int(y)), (x2, int(y)), 255, 1)
        if contacts.get("right"):
            ys = np.where(ink[:, -1])[0]
            for y in ys:
                x1 = max(0, width - 1 - extension_px)
                cv2.line(extended, (width - 1, int(y)), (x1, int(y)), 255, 1)
        if contacts.get("top"):
            xs = np.where(ink[0, :])[0]
            for x in xs:
                y2 = min(height - 1, extension_px)
                cv2.line(extended, (int(x), 0), (int(x), y2), 255, 1)
        if contacts.get("bottom"):
            xs = np.where(ink[-1, :])[0]
            for x in xs:
                y1 = max(0, height - 1 - extension_px)
                cv2.line(extended, (int(x), height - 1), (int(x), y1), 255, 1)

        if np.array_equal(mask, extended):
            return []
        return [
            self._make_candidate(
                f"h{index}_border_continuation",
                "border_continuation",
                mask,
                extended,
                {"extension_px": extension_px, "border_contacts": contacts},
            )
        ]

    def _bridge_candidates(self, graph, trace_paths, mask):
        """Rank endpoint pairs whose continuation tangents face each other."""
        endpoints = graph.endpoints()
        component_lookup = _graph_component_lookup(graph)
        candidates = []
        tangent_points = int(_setting(self.settings, "reconstruction_tangent_points", 3))
        min_sep = float(_setting(self.settings, "reconstruction_min_endpoint_separation_px", 2.0))
        max_length = float(_setting(self.settings, "reconstruction_max_bridge_length_px", 16.0))
        max_angle = float(_setting(self.settings, "reconstruction_max_bridge_angle_degrees", 45.0))
        max_hypotheses = int(_setting(self.settings, "reconstruction_max_hypotheses", 8))

        for first_index, point_a in enumerate(endpoints):
            tangent_a = _endpoint_tangent(point_a, trace_paths, tangent_points)
            if tangent_a is None:
                continue
            for point_b in endpoints[first_index + 1:]:
                distance = math.hypot(point_b.x - point_a.x, point_b.y - point_a.y)
                if not (min_sep <= distance <= max_length):
                    continue
                tangent_b = _endpoint_tangent(point_b, trace_paths, tangent_points)
                if tangent_b is None:
                    continue

                bridge_ab = (point_b.x - point_a.x, point_b.y - point_a.y)
                bridge_ba = (-bridge_ab[0], -bridge_ab[1])
                angle_a = _angle_degrees(tangent_a[1:], bridge_ab)
                angle_b = _angle_degrees(tangent_b[1:], bridge_ba)
                maximum_angle = max(angle_a, angle_b)
                if maximum_angle > max_angle:
                    continue

                line_coordinates = _line_coordinates(point_a, point_b)
                endpoint_clearance = min(5, max(1, int(round(len(line_coordinates) * 0.25))))
                interior = line_coordinates[endpoint_clearance:-endpoint_clearance]
                occupied = sum(mask[y, x] > 0 for x, y in interior)
                occupied_ratio = occupied / max(1, len(interior))
                if occupied_ratio > 0.25:
                    continue

                different_components = (
                    component_lookup.get(point_a.to_tuple())
                    != component_lookup.get(point_b.to_tuple())
                )
                length_score = 1.0 - distance / max(1e-6, max_length)
                angle_score = 1.0 - maximum_angle / max(1.0, max_angle)
                geometry_score = 0.55 * angle_score + 0.35 * length_score + 0.10 * (1.0 - occupied_ratio)
                candidates.append(
                    {
                        "point_a": point_a,
                        "point_b": point_b,
                        "path_id_a": tangent_a[0],
                        "path_id_b": tangent_b[0],
                        "distance": float(distance),
                        "angle_a": float(angle_a),
                        "angle_b": float(angle_b),
                        "geometry_score": float(geometry_score),
                        "different_components": bool(different_components),
                        "occupied_ratio": float(occupied_ratio),
                    }
                )

        candidates.sort(
            key=lambda item: (
                -item["geometry_score"],
                item["distance"],
                item["point_a"].y,
                item["point_a"].x,
                item["point_b"].y,
                item["point_b"].x,
            )
        )
        return candidates[:max_hypotheses]

    def _endpoint_bridge_candidates(self, mask, graph, paths, index_start):
        """Generate endpoint bridge candidates."""
        mask = _binary_mask(mask)
        candidates = []
        for offset, bridge in enumerate(self._bridge_candidates(graph, paths, mask), start=0):
            point_a = bridge["point_a"]
            point_b = bridge["point_b"]
            thickness = _estimate_bridge_thickness(mask, point_a, point_b)
            reconstructed = mask.copy()
            cv2.line(
                reconstructed,
                point_a.to_tuple(),
                point_b.to_tuple(),
                255,
                thickness,
                cv2.LINE_AA,
            )
            reconstructed = np.where(reconstructed > 127, 255, 0).astype(np.uint8)
            if np.array_equal(mask, reconstructed):
                continue
            metadata = {
                "bridge": {
                    "from": point_a.to_dict(),
                    "to": point_b.to_dict(),
                    "distance": bridge["distance"],
                    "thickness": int(thickness),
                    "path_id_a": bridge["path_id_a"],
                    "path_id_b": bridge["path_id_b"],
                    "angle_a": bridge["angle_a"],
                    "angle_b": bridge["angle_b"],
                    "different_components": bridge["different_components"],
                    "occupied_ratio": bridge["occupied_ratio"],
                },
                "geometry_score": bridge["geometry_score"],
            }
            candidates.append(
                self._make_candidate(
                    f"h{index_start + offset}_endpoint_bridge",
                    "endpoint_bridge",
                    mask,
                    reconstructed,
                    metadata,
                )
            )
        return candidates

    def _generate_candidates(self, mask, original_result, diagnosis, allowed_defenses):
        """Run allowed defenses and return candidate masks."""
        mask = _binary_mask(mask)
        points = SkeletonPointExtractor().extract_points(
            TraceSkeletonizer(self.settings).skeletonize(mask)
        )
        graph = SkeletonGraph(points)
        paths = TracePathExtractor(self.settings).extract_paths(graph)

        candidates = []
        next_index = 1
        for defense_name in allowed_defenses:
            if defense_name == "horizontal_gap_closing":
                produced = self._gap_closing_candidate(mask, defense_name, next_index)
            elif defense_name == "vertical_gap_closing":
                produced = self._gap_closing_candidate(mask, defense_name, next_index)
            elif defense_name == "endpoint_bridge":
                produced = self._endpoint_bridge_candidates(mask, graph, paths, next_index)
            elif defense_name == "conservative_stroke_recovery":
                produced = self._conservative_stroke_recovery_candidate(mask, next_index)
            elif defense_name == "component_denoising":
                produced = self._component_denoising_candidate(mask, next_index)
            elif defense_name == "median_denoising":
                produced = self._median_denoising_candidate(mask, next_index)
            elif defense_name == "contamination_opening":
                produced = self._contamination_opening_candidate(mask, next_index)
            elif defense_name == "linear_artifact_removal":
                produced = self._linear_artifact_removal_candidate(mask, next_index)
            elif defense_name == "threshold_normalization":
                produced = self._threshold_normalization_candidate(mask, next_index)
            elif defense_name == "border_continuation":
                produced = self._border_continuation_candidate(mask, diagnosis, next_index)
            else:
                produced = []

            candidates.extend(produced)
            next_index += max(1, len(produced))

        max_total = int(_setting(self.settings, "reconstruction_max_defense_hypotheses", 12))
        if max_total <= 0:
            max_total = int(_setting(self.settings, "reconstruction_max_hypotheses", 8))
        return candidates[:max_total]

    # --------------------------------------------------------
    # Verification / acceptance
    # --------------------------------------------------------

    def _intervention_scores(self, candidate, original_ink):
        added_pixels = int(np.count_nonzero(candidate.added_mask))
        removed_pixels = int(np.count_nonzero(candidate.removed_mask))
        changed_pixels = added_pixels + removed_pixels
        added_ratio = added_pixels / max(1, original_ink)
        removed_ratio = removed_pixels / max(1, original_ink)
        changed_ratio = changed_pixels / max(1, original_ink)
        return {
            "added_pixels": added_pixels,
            "removed_pixels": removed_pixels,
            "changed_pixels": changed_pixels,
            "added_ink_ratio": float(added_ratio),
            "removed_ink_ratio": float(removed_ratio),
            "changed_ink_ratio": float(changed_ratio),
        }

    def _score_candidate(
        self,
        candidate,
        original_topology,
        candidate_topology,
        original_recognition,
        candidate_recognition,
        original_ink,
    ):
        intervention = self._intervention_scores(candidate, original_ink)
        topology_gain = _topology_gain(
            original_topology,
            candidate_topology,
            candidate.defense_name,
        )
        topology_score = max(0.0, min(1.0, 0.5 + topology_gain))

        max_added = float(_setting(self.settings, "reconstruction_max_added_ink_ratio", 0.12))
        max_removed = float(_setting(self.settings, "reconstruction_max_removed_ink_ratio", 0.18))
        max_changed = float(_setting(self.settings, "reconstruction_max_changed_ink_ratio", 0.25))

        added_penalty = intervention["added_ink_ratio"] / max(1e-6, max_added)
        removed_penalty = intervention["removed_ink_ratio"] / max(1e-6, max_removed)
        changed_penalty = intervention["changed_ink_ratio"] / max(1e-6, max_changed)

        if candidate.defense_name in DEFENSES_THAT_ADD_INK:
            intervention_score = max(0.0, 1.0 - added_penalty)
        elif candidate.defense_name in DEFENSES_THAT_REMOVE_INK:
            intervention_score = max(0.0, 1.0 - removed_penalty)
        else:
            intervention_score = max(0.0, 1.0 - changed_penalty)

        geometry_score = float(candidate.metadata.get("geometry_score", intervention_score))
        geometry_score = max(0.0, min(1.0, geometry_score))

        confidence_gain = 0.0
        if candidate_recognition["available"] and original_recognition["available"]:
            confidence_gain = (
                candidate_recognition["top1_confidence"]
                - original_recognition["top1_confidence"]
            )
        confidence_score = max(0.0, min(1.0, 0.5 + confidence_gain))

        topology_weight = float(_setting(self.settings, "reconstruction_topology_weight", 0.55))
        geometry_weight = float(_setting(self.settings, "reconstruction_geometry_weight", 0.30))
        confidence_weight = float(_setting(self.settings, "reconstruction_confidence_weight", 0.15))
        total_weight = max(1e-6, topology_weight + geometry_weight + confidence_weight)

        acceptance_score = (
            topology_weight * topology_score
            + geometry_weight * geometry_score
            + confidence_weight * confidence_score
        ) / total_weight

        return {
            **intervention,
            "topology_gain": float(topology_gain),
            "topology_score": float(topology_score),
            "geometry_score": float(geometry_score),
            "confidence_gain": float(confidence_gain),
            "confidence_score": float(confidence_score),
            "intervention_score": float(intervention_score),
            "acceptance_score": float(acceptance_score),
        }

    def _rejection_reasons(self, candidate, scores, original_topology, candidate_topology):
        reasons = []
        min_topology_gain = float(_setting(self.settings, "reconstruction_min_topology_gain", -0.05))
        min_acceptance_score = float(_setting(self.settings, "reconstruction_min_acceptance_score", 0.55))
        max_added = float(_setting(self.settings, "reconstruction_max_added_ink_ratio", 0.12))
        max_removed = float(_setting(self.settings, "reconstruction_max_removed_ink_ratio", 0.18))
        max_changed = float(_setting(self.settings, "reconstruction_max_changed_ink_ratio", 0.25))

        if scores["topology_gain"] < min_topology_gain:
            reasons.append("insufficient_topology_gain")
        if scores["acceptance_score"] < min_acceptance_score:
            reasons.append("insufficient_acceptance_score")

        if candidate.defense_name in DEFENSES_THAT_ADD_INK:
            if scores["added_ink_ratio"] > max_added:
                reasons.append("added_ink_budget_exceeded")
        elif candidate.defense_name in DEFENSES_THAT_REMOVE_INK:
            if scores["removed_ink_ratio"] > max_removed:
                reasons.append("removed_ink_budget_exceeded")
        else:
            if scores["changed_ink_ratio"] > max_changed:
                reasons.append("changed_ink_budget_exceeded")

        if candidate_topology["junction_cluster_count"] > original_topology["junction_cluster_count"] + 1:
            reasons.append("junction_complexity_increased")
        if candidate_topology["short_path_count"] > original_topology["short_path_count"] + 2:
            reasons.append("fragmentation_increased")

        recognition_required = bool(
            _setting(self.settings, "reconstruction_use_recognition_verification", False)
        )
        if recognition_required and scores["confidence_gain"] < float(
            _setting(self.settings, "reconstruction_min_confidence_gain", 0.0)
        ):
            reasons.append("recognition_confidence_regressed")

        return reasons

    def _save_candidate_debug(self, reconstruction_dir, safe_id, candidate):
        if not _setting(self.settings, "save_debug", False):
            return None, None, None

        mask_path = os.path.join(
            reconstruction_dir,
            f"{safe_id}_{candidate.hypothesis_id}_mask.png",
        )
        added_path = os.path.join(
            reconstruction_dir,
            f"{safe_id}_{candidate.hypothesis_id}_added.png",
        )
        removed_path = os.path.join(
            reconstruction_dir,
            f"{safe_id}_{candidate.hypothesis_id}_removed.png",
        )
        cv2.imwrite(mask_path, candidate.mask)
        cv2.imwrite(added_path, candidate.added_mask)
        cv2.imwrite(removed_path, candidate.removed_mask)
        return mask_path, added_path, removed_path

    def _save_overlay_debug(self, reconstruction_dir, safe_id, original_mask, candidate):
        if not _setting(self.settings, "save_debug", False):
            return None
        overlay_path = os.path.join(
            reconstruction_dir,
            f"{safe_id}_{candidate.hypothesis_id}_overlay.png",
        )
        overlay = cv2.cvtColor(_binary_mask(original_mask), cv2.COLOR_GRAY2BGR)
        # OpenCV BGR: green = added, red = removed.
        overlay[candidate.added_mask > 0] = (70, 230, 70)
        overlay[candidate.removed_mask > 0] = (40, 40, 230)
        cv2.imwrite(overlay_path, overlay)
        return overlay_path

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def run(
        self,
        mask,
        original_result,
        output_dir,
        stable_unit_id,
        damage_verdict=None,
        allowed_defenses=None,
    ):
        """Run reconstruction.

        Args:
            mask: binary damaged mask.
            original_result: original ScribeTrace result for this mask.
            output_dir: debug output root.
            stable_unit_id: stable text unit id for debug filenames.
            damage_verdict: optional dict/string/list from oracle or damage model.
            allowed_defenses: optional explicit defense-name list from router.

        Returns:
            JSON-safe dict with h0_original, candidate hypotheses, and selected candidate.
        """
        mask = _binary_mask(mask)
        enabled = bool(_setting(self.settings, "enable_theoretical_reconstruction", True))
        original_topology = _topology_snapshot(original_result)
        original_recognition = _recognition_snapshot(original_result)
        normalized_damage_verdict = self._normalize_damage_verdict(damage_verdict)

        base_result = {
            "version": "scribetrace_reconstruction_v2",
            "enabled": enabled,
            "cycle": [
                "diagnose",
                "route_defenses",
                "generate_hypotheses",
                "retrace",
                "verify",
                "select",
            ],
            "damage_verdict": normalized_damage_verdict,
            "attack_mapping": DEFAULT_ATTACK_TO_DEFENSES,
            "original_hypothesis": {
                "hypothesis_id": "h0_original",
                "type": "no_repair",
                "accepted": True,
                "selected": False,
                "topology": original_topology,
                "recognition": original_recognition,
            },
            "hypotheses": [],
            "accepted_hypothesis_ids": [],
            "accepted_count": 0,
            "candidate_count": 0,
            "total_candidate_count": 0,
            "selected_hypothesis_id": "h0_original",
            "selected_feature_source": "original",
            "selected_feature_vector": None,
        }

        if not enabled:
            base_result["status"] = "disabled"
            base_result["original_hypothesis"]["selected"] = True
            return base_result

        diagnosis = self.diagnose(mask, original_result)
        base_result["topology_diagnosis"] = diagnosis

        resolved_defenses = self._resolve_allowed_defenses(
            normalized_damage_verdict,
            allowed_defenses,
            diagnosis,
        )
        base_result["allowed_defenses"] = resolved_defenses

        # If external verdict says clean, or no defense is routed, keep h0.
        if not resolved_defenses:
            base_result["status"] = "completed_no_repair_routed"
            base_result["original_hypothesis"]["selected"] = True
            return base_result

        reconstruction_dir = os.path.join(output_dir, "reconstruction")
        os.makedirs(reconstruction_dir, exist_ok=True)
        safe_id = sanitize_identifier(stable_unit_id)
        original_ink = max(1, int(np.count_nonzero(mask)))

        candidates = self._generate_candidates(
            mask,
            original_result,
            diagnosis,
            resolved_defenses,
        )
        base_result["candidate_count"] = len(candidates)
        base_result["total_candidate_count"] = len(candidates)
        base_result["status"] = "completed"

        for candidate in candidates:
            traced = _trace_mask(candidate.mask, self.settings)
            candidate_result = traced["result"]
            candidate_topology = _topology_snapshot(candidate_result)
            candidate_recognition = _recognition_snapshot(candidate_result)
            scores = self._score_candidate(
                candidate,
                original_topology,
                candidate_topology,
                original_recognition,
                candidate_recognition,
                original_ink,
            )
            rejection_reasons = self._rejection_reasons(
                candidate,
                scores,
                original_topology,
                candidate_topology,
            )
            accepted = not rejection_reasons

            mask_path, added_path, removed_path = self._save_candidate_debug(
                reconstruction_dir,
                safe_id,
                candidate,
            )
            overlay_path = self._save_overlay_debug(
                reconstruction_dir,
                safe_id,
                mask,
                candidate,
            )

            hypothesis = {
                "hypothesis_id": candidate.hypothesis_id,
                "type": candidate.defense_name,
                "accepted": accepted,
                "selected": False,
                "rejection_reasons": rejection_reasons,
                **scores,
                "metadata": candidate.metadata,
                "original_topology": original_topology,
                "reconstructed_topology": candidate_topology,
                "topology_delta": _topology_delta(original_topology, candidate_topology),
                "recognition": candidate_recognition,
                "reconstructed_mask_path": mask_path,
                "added_ink_mask_path": added_path,
                "removed_ink_mask_path": removed_path,
                "reconstruction_overlay_path": overlay_path,
                "feature_vector": candidate_result.feature_vector.to_dict()
                if candidate_result.feature_vector is not None
                else None,
            }
            base_result["hypotheses"].append(hypothesis)

        base_result["hypotheses"].sort(
            key=lambda item: (
                not item["accepted"],
                -item["acceptance_score"],
                item["changed_pixels"],
                item["hypothesis_id"],
            )
        )

        max_accepted = int(_setting(self.settings, "reconstruction_max_accepted", 1))
        accepted = [
            item for item in base_result["hypotheses"] if item["accepted"]
        ][:max_accepted]
        accepted_ids = {item["hypothesis_id"] for item in accepted}

        for hypothesis in base_result["hypotheses"]:
            hypothesis["accepted"] = hypothesis["hypothesis_id"] in accepted_ids

        base_result["accepted_hypothesis_ids"] = [
            item["hypothesis_id"] for item in accepted
        ]
        base_result["accepted_count"] = len(accepted)

        if accepted:
            selected = accepted[0]
            selected["selected"] = True
            base_result["selected_hypothesis_id"] = selected["hypothesis_id"]
            base_result["selected_feature_source"] = "reconstructed"
            base_result["selected_feature_vector"] = selected.get("feature_vector")
        else:
            base_result["selected_hypothesis_id"] = "h0_original"
            base_result["selected_feature_source"] = "original"
            base_result["selected_feature_vector"] = None
            base_result["original_hypothesis"]["selected"] = True

        return base_result


# Clearer new name, while keeping backward compatibility.
ReconstructionController = TheoreticalReconstructor

__all__ = ["TheoreticalReconstructor", "ReconstructionController"]