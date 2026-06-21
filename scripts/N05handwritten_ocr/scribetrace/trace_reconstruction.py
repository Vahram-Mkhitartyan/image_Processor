"""Bounded theoretical reconstruction for damaged handwritten topology.

This module is the ScribeTrace reconstruction controller.

It does three jobs:
1. Keep h0_original as the safe fallback.
2. Generate routed repair hypotheses:
   - legacy endpoint_bridge
   - trace_defenses.py candidate generators
3. Retrace, score, accept/reject, and export UI-friendly masks/overlays.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict
from typing import Any

import cv2
import numpy as np

from .trace_common import sanitize_identifier
from .trace_defense_registry import (
    DEFENSE_ENDPOINT_BRIDGE,
    DEFENSE_LINEAR_ARTIFACT_REMOVAL,
    DEFENSE_STAGE_ORDER,
    defenses_for_stage,
    get_defense_spec_dict,
    grouped_defenses_by_stage,
    implemented_in_trace_defenses,
)
from .trace_defenses import DefenseHypothesis, generate_defense_hypotheses
from .trace_debug import TraceDebugWriter
from .trace_features import TraceFeatureEncoder, build_scrilog_observation
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


# Fallback only. condition/condition_router.py is the source of truth.
# This is kept so direct reconstruction debug runs still behave sensibly.
FALLBACK_ATTACK_TO_DEFENSES = {
    "clean": [],
    "light_cut": [
        "horizontal_gap_closing",
        "vertical_gap_closing",
        "endpoint_bridge",
    ],
    "light_blur": ["threshold_normalization"],
    "scanner_noise": [
        "component_denoising",
        "median_denoising",
    ],
    "light_erosion": [
        "conservative_stroke_recovery",
        "endpoint_bridge",
        "horizontal_gap_closing",
        "vertical_gap_closing",
    ],
    "ink_overlap": [
        "linear_artifact_removal",
        "contamination_opening",
        "endpoint_bridge",
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
    "edge_crop_loss": ["border_continuation"],
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


def _setting(settings, name, default):
    return getattr(settings, name, default)


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return uint8 foreground-white binary mask."""
    if mask is None:
        raise ValueError("mask cannot be None")

    arr = np.asarray(mask)

    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    arr = arr.astype(np.uint8)
    unique = np.unique(arr)

    if len(unique) <= 3 and set(int(value) for value in unique).issubset({0, 1, 255}):
        return np.where(arr > 0, 255, 0).astype(np.uint8)

    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    background = float(np.median(border))

    if background > 127:
        return np.where(arr < 215, 255, 0).astype(np.uint8)

    return np.where(arr > 40, 255, 0).astype(np.uint8)


def _visualize_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return black ink on white background for human UI cards."""
    binary = _as_binary_mask(mask)
    return np.where(binary > 0, 0, 255).astype(np.uint8)


def _changed_masks(original: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    original_bin = _as_binary_mask(original)
    candidate_bin = _as_binary_mask(candidate)

    added = np.where((candidate_bin > 0) & (original_bin == 0), 255, 0).astype(np.uint8)
    removed = np.where((candidate_bin == 0) & (original_bin > 0), 255, 0).astype(np.uint8)
    return added, removed


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
    """Map every skeleton coordinate to a deterministic graph component ID."""
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
    """Return a thin connector for endpoint repair.

    Endpoint bridge is a topology repair, not a stroke-restoration brush. Earlier
    versions estimated thickness from local stroke radius, but on small glyph
    crops that turned a missing connector into visible widening. A one-pixel
    bridge is enough for retracing to test whether two skeleton endpoints belong
    together; thicker recovery belongs to the stroke-recovery defenses.
    """
    return 1


def _trace_mask(mask, settings):
    """Retrace a hypothetical mask without invoking reconstruction recursively."""
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
    metrics["scrilog_observation"] = build_scrilog_observation(
        components=components,
        skeleton_graph=graph,
        trace_paths=paths,
        ink_holes=ink_holes,
        mask_shape=cleaned_mask.shape,
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
    return {
        key: candidate.get(key, 0) - original.get(key, 0)
        for key in sorted(set(original) | set(candidate))
    }


def _endpoint_audit_near_mask(mask, focus_mask, settings):
    """Count skeleton endpoints sitting near a local focus/wound mask."""
    focus_bin = _as_binary_mask(focus_mask)

    if int(cv2.countNonZero(focus_bin)) <= 0:
        return {
            "endpoint_count_near_focus": 0,
            "endpoints_near_focus": [],
        }

    focus_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    focus_zone = cv2.dilate(focus_bin, focus_kernel, iterations=1)

    skeleton = TraceSkeletonizer(settings).skeletonize(mask)
    points = SkeletonPointExtractor().extract_points(skeleton)
    graph = SkeletonGraph(points)

    endpoints = [
        point
        for point in graph.endpoints()
        if 0 <= point.y < focus_zone.shape[0]
        and 0 <= point.x < focus_zone.shape[1]
        and focus_zone[point.y, point.x] > 0
    ]

    return {
        "endpoint_count_near_focus": len(endpoints),
        "endpoints_near_focus": [
            point.to_dict()
            for point in sorted(endpoints, key=lambda item: (item.y, item.x))[:24]
        ],
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


def _topology_gain(original, candidate, different_components=False):
    """Score whether a repair reduces damage without inventing complexity."""
    component_gain = max(
        0,
        original["component_count"] - candidate["component_count"],
    )
    endpoint_gain = max(
        0,
        original["endpoint_count"] - candidate["endpoint_count"],
    )
    isolated_gain = max(
        0,
        original["isolated_point_count"] - candidate["isolated_point_count"],
    )
    loop_gain = max(
        0,
        candidate["ink_hole_count"] - original["ink_hole_count"],
    )

    new_junctions = max(
        0,
        candidate["junction_cluster_count"] - original["junction_cluster_count"],
    )
    new_short_paths = max(
        0,
        candidate["short_path_count"] - original["short_path_count"],
    )
    new_components = max(
        0,
        candidate["component_count"] - original["component_count"],
    )

    gain = (
        0.30 * min(1, component_gain)
        + 0.30 * min(2, endpoint_gain) / 2.0
        + 0.12 * min(1, isolated_gain)
        + 0.12 * min(1, loop_gain)
        - 0.20 * min(1, new_junctions)
        - 0.14 * min(1, new_short_paths)
        - 0.18 * min(1, new_components)
    )

    if different_components and component_gain == 0:
        gain -= 0.20

    return max(-1.0, min(1.0, gain))


def _normalize_damage_verdict(damage_verdict):
    """Extract possible damage labels from flexible verdict shapes."""
    if damage_verdict is None:
        return []

    if isinstance(damage_verdict, str):
        return [damage_verdict]

    if isinstance(damage_verdict, (list, tuple, set)):
        return [str(item) for item in damage_verdict if item]

    if isinstance(damage_verdict, dict):
        labels = []
        for key in ("primary_damage", "label", "damage_type", "condition"):
            value = damage_verdict.get(key)
            if value and value not in labels:
                labels.append(str(value))

        for candidate_key in ("top_damage_candidates", "top_k", "candidates"):
            for candidate in damage_verdict.get(candidate_key) or []:
                if isinstance(candidate, dict):
                    value = candidate.get("label") or candidate.get("damage_type")
                else:
                    value = candidate
                if value and value not in labels:
                    labels.append(str(value))
        return labels

    return []


def _unique_names(names):
    output = []
    for name in names or []:
        name = str(name)
        if name and name not in output:
            output.append(name)
    return output


def _hypothesis_selection_priority(hypothesis):
    """Rank smarter/path-aware repairs before broad mask morphology repairs."""
    defense_name = getattr(hypothesis, "defense_name", "")
    metadata = getattr(hypothesis, "metadata", {}) or {}

    defense_priority = {
        DEFENSE_ENDPOINT_BRIDGE: 0,
        "border_continuation": 1,
        "component_denoising": 2,
        "threshold_normalization": 3,
        "median_denoising": 4,
        "horizontal_gap_closing": 5,
        "vertical_gap_closing": 5,
        "conservative_stroke_recovery": 6,
        "contamination_opening": 7,
        "linear_artifact_removal": 8,
    }

    return (
        defense_priority.get(defense_name, 99),
        -float(metadata.get("geometry_score", 0.0)),
        float(metadata.get("changed_ink_ratio", 0.0)),
        str(getattr(hypothesis, "hypothesis_id", "")),
    )


class TheoreticalReconstructor:
    """ScribeTrace reconstruction controller and verifier."""

    def __init__(self, settings=None):
        self.settings = normalize_trace_settings(settings)

    def _empty_result(
        self,
        *,
        status,
        mask,
        original_result,
        output_dir,
        stable_unit_id,
        damage_verdict=None,
        allowed_defenses=None,
        known_damage_recipes=None,
        extra=None,
    ):
        original_topology = _topology_snapshot(original_result)
        debug_paths = self._save_h0_debug(
            mask=mask,
            output_dir=output_dir,
            stable_unit_id=stable_unit_id,
        )
        result = {
            "version": "scribetrace_reconstruction_v3",
            "enabled": bool(self.settings.enable_theoretical_reconstruction),
            "status": status,
            "damage_verdict": damage_verdict,
            "known_damage_recipes": list(known_damage_recipes or []),
            "allowed_defense_types": list(allowed_defenses or []),
            "implemented_defense_types": [],
            "unsupported_defense_types": list(allowed_defenses or []),
            "no_candidate_defense_types": [],
            "stage_defense_plan": {},
            "stage_records": [],
            "original_hypothesis": {
                "hypothesis_id": "h0_original",
                "defense_name": "no_repair",
                "accepted": True,
                "selected": True,
                "topology": original_topology,
                "scrilog_observation": original_result.metrics.get(
                    "scrilog_observation"
                ),
                "mask_path": debug_paths.get("h0_mask_path"),
                "visual_path": debug_paths.get("h0_visual_path"),
                "overlay_path": debug_paths.get("h0_overlay_path"),
            },
            "hypotheses": [],
            "accepted_hypothesis_ids": [],
            "accepted_count": 0,
            "candidate_count": 0,
            "total_candidate_count": 0,
            "selected_hypothesis_id": "h0_original",
            "selected_feature_source": "original",
            "selected_feature_vector": (
                original_result.feature_vector.to_dict()
                if original_result.feature_vector is not None
                else None
            ),
            "selected_scrilog_observation": original_result.metrics.get(
                "scrilog_observation"
            ),
            "selected_mask_path": debug_paths.get("h0_mask_path"),
            "selected_visual_path": debug_paths.get("h0_visual_path"),
            "selected_overlay_path": debug_paths.get("h0_overlay_path"),
            "topology_diagnosis": {
                "original": original_topology,
            },
            "debug_dir": debug_paths.get("debug_dir"),
        }

        if extra:
            result.update(extra)

        return result

    def diagnose(self, original_result):
        """Return lightweight topology signals useful for UI/debug."""
        topology = _topology_snapshot(original_result)
        signals = []

        if topology["component_count"] > 1:
            signals.append("multiple_components")

        if topology["endpoint_count"] >= 4:
            signals.append("many_endpoints")

        if topology["isolated_point_count"] > 0:
            signals.append("isolated_points")

        if topology["short_path_count"] > 0:
            signals.append("short_paths")

        if topology["skeleton_point_count"] == 0:
            signals.append("empty_skeleton")

        return {
            "signals": signals,
            "original": topology,
            "needs_reconstruction": bool(signals),
        }

    def run(
        self,
        mask,
        original_result,
        output_dir,
        stable_unit_id,
        damage_verdict=None,
        allowed_defenses=None,
        known_damage_recipes=None,
    ):
        """Run routed theoretical reconstruction.

        Routing priority:
        1. explicit allowed_defenses from condition router
        2. known_damage_recipes, only for debug/oracle fallback
        3. damage_verdict labels, only for direct debug fallback
        4. no route -> h0_original only
        """
        binary_mask = _as_binary_mask(mask)
        known_damage_recipes = list(known_damage_recipes or [])

        if not self.settings.enable_theoretical_reconstruction:
            return self._empty_result(
                status="disabled",
                mask=binary_mask,
                original_result=original_result,
                output_dir=output_dir,
                stable_unit_id=stable_unit_id,
                damage_verdict=damage_verdict,
                allowed_defenses=allowed_defenses,
                known_damage_recipes=known_damage_recipes,
            )

        routed_defenses, route_source = self._resolve_allowed_defenses(
            explicit_allowed_defenses=allowed_defenses,
            known_damage_recipes=known_damage_recipes,
            damage_verdict=damage_verdict,
        )

        if not routed_defenses:
            return self._empty_result(
                status="completed_no_routed_defenses",
                mask=binary_mask,
                original_result=original_result,
                output_dir=output_dir,
                stable_unit_id=stable_unit_id,
                damage_verdict=damage_verdict,
                allowed_defenses=[],
                known_damage_recipes=known_damage_recipes,
                extra={
                    "route_source": route_source,
                    "topology_diagnosis": self.diagnose(original_result),
                },
            )

        original_topology = _topology_snapshot(original_result)
        original_recognition = (
            _recognition_snapshot(original_result)
            if self.settings.reconstruction_use_recognition_verification
            else {"available": False, "top1_confidence": 0.0, "top5": [], "error": None}
        )

        debug_paths = self._save_h0_debug(
            mask=binary_mask,
            output_dir=output_dir,
            stable_unit_id=stable_unit_id,
        )

        raw_hypotheses, stage_records = self._generate_stage_hypotheses(
            binary_mask=binary_mask,
            original_result=original_result,
            routed_defenses=routed_defenses,
            stable_unit_id=stable_unit_id,
        )

        raw_hypotheses = self._dedupe_hypotheses(raw_hypotheses)
        raw_total_candidate_count = len(raw_hypotheses)
        raw_hypotheses.sort(key=_hypothesis_selection_priority)

        max_hypotheses = int(self.settings.reconstruction_max_hypotheses)
        raw_hypotheses = raw_hypotheses[:max_hypotheses]

        implemented_defenses = _unique_names(
            defense_name
            for stage_record in stage_records
            for defense_name in stage_record["implemented_defense_types"]
        )
        unsupported_defenses = [
            defense_name
            for defense_name in routed_defenses
            if defense_name not in implemented_defenses
        ]
        no_candidate_defenses = [
            defense_name
            for defense_name in implemented_defenses
            if all(
                hypothesis.defense_name != defense_name
                for hypothesis in raw_hypotheses
            )
        ]

        scored_hypotheses = []

        for hypothesis in raw_hypotheses:
            candidate_record = self._evaluate_hypothesis(
                hypothesis=hypothesis,
                original_mask=binary_mask,
                original_result=original_result,
                original_topology=original_topology,
                original_recognition=original_recognition,
                output_dir=output_dir,
                stable_unit_id=stable_unit_id,
            )
            scored_hypotheses.append(candidate_record)

        self._enforce_line_cut_downstream_bridge(scored_hypotheses)

        accepted = [
            candidate_record
            for candidate_record in scored_hypotheses
            if candidate_record["accepted"]
        ]
        accepted.sort(key=lambda item: item["score"], reverse=True)

        selected = accepted[0] if accepted else None
        if selected is not None:
            selected["selected"] = True

        result = {
            "version": "scribetrace_reconstruction_v3",
            "enabled": True,
            "status": "completed",
            "route_source": route_source,
            "damage_verdict": damage_verdict,
            "known_damage_recipes": known_damage_recipes,
            "allowed_defense_types": routed_defenses,
            "implemented_defense_types": implemented_defenses,
            "unsupported_defense_types": unsupported_defenses,
            "no_candidate_defense_types": no_candidate_defenses,
            "stage_defense_plan": grouped_defenses_by_stage(
                routed_defenses,
                include_empty_stages=True,
                include_candidate_stages=False,
            ),
            "stage_records": stage_records,
            "original_hypothesis": {
                "hypothesis_id": "h0_original",
                "defense_name": "no_repair",
                "accepted": True,
                "selected": selected is None,
                "topology": original_topology,
                "scrilog_observation": original_result.metrics.get(
                    "scrilog_observation"
                ),
                "recognition": original_recognition,
                "mask_path": debug_paths.get("h0_mask_path"),
                "visual_path": debug_paths.get("h0_visual_path"),
                "overlay_path": debug_paths.get("h0_overlay_path"),
            },
            "hypotheses": scored_hypotheses,
            "accepted_hypothesis_ids": [
                item["hypothesis_id"] for item in accepted
            ],
            "accepted_count": len(accepted),
            "candidate_count": len(scored_hypotheses),
            "total_candidate_count": raw_total_candidate_count,
            "evaluated_candidate_count": len(raw_hypotheses),
            "selected_hypothesis_id": (
                selected["hypothesis_id"] if selected else "h0_original"
            ),
            "selected_feature_source": (
                "reconstructed" if selected else "original"
            ),
            "selected_feature_vector": (
                selected["feature_vector"]
                if selected
                else (
                    original_result.feature_vector.to_dict()
                    if original_result.feature_vector is not None
                    else None
                )
            ),
            "selected_scrilog_observation": (
                selected.get("scrilog_observation")
                if selected
                else original_result.metrics.get("scrilog_observation")
            ),
            "selected_mask_path": (
                selected["candidate_mask_path"]
                if selected
                else debug_paths.get("h0_mask_path")
            ),
            "selected_visual_path": (
                selected["candidate_visual_path"]
                if selected
                else debug_paths.get("h0_visual_path")
            ),
            "selected_overlay_path": (
                selected["overlay_path"]
                if selected
                else debug_paths.get("h0_overlay_path")
            ),
            "topology_diagnosis": self.diagnose(original_result),
            "debug_dir": debug_paths.get("debug_dir"),
        }

        return result

    def _resolve_allowed_defenses(
        self,
        *,
        explicit_allowed_defenses=None,
        known_damage_recipes=None,
        damage_verdict=None,
    ):
        if explicit_allowed_defenses is not None:
            return _unique_names(explicit_allowed_defenses), "explicit_allowed_defenses"

        defenses = []
        for recipe_name in known_damage_recipes or []:
            defenses.extend(FALLBACK_ATTACK_TO_DEFENSES.get(str(recipe_name), []))
        if defenses:
            return _unique_names(defenses), "known_damage_recipes_fallback"

        for label in _normalize_damage_verdict(damage_verdict):
            defenses.extend(FALLBACK_ATTACK_TO_DEFENSES.get(str(label), []))
        if defenses:
            return _unique_names(defenses), "damage_verdict_fallback"

        return [], "no_route"

    def _rebase_hypothesis_against_original(
        self,
        *,
        hypothesis,
        original_mask,
        parent_state,
    ):
        """Convert a branch-generated hypothesis back to h0-relative UI accounting.

        Important:
        - UI overlays should compare final candidate against h0_original.
        - Scoring should compare the child against its branch parent.
          Example:
              h0 -> line removal -> endpoint bridge

          The endpoint bridge should be judged against the line-removed mask,
          not against h0.
        """
        original_bin = _as_binary_mask(original_mask)
        parent_bin = _as_binary_mask(parent_state["mask"])
        candidate_bin = _as_binary_mask(hypothesis.candidate_mask)

        # h0-relative masks for UI/debug.
        added, removed = _changed_masks(original_bin, candidate_bin)

        added_px = int(cv2.countNonZero(added))
        removed_px = int(cv2.countNonZero(removed))
        changed_px = added_px + removed_px
        original_px = max(1, int(cv2.countNonZero(original_bin)))

        if changed_px == 0:
            return None

        # Branch-local masks for scoring.
        branch_added, branch_removed = _changed_masks(parent_bin, candidate_bin)

        branch_added_px = int(cv2.countNonZero(branch_added))
        branch_removed_px = int(cv2.countNonZero(branch_removed))
        branch_changed_px = branch_added_px + branch_removed_px
        parent_px = max(1, int(cv2.countNonZero(parent_bin)))

        parent_chain = list(parent_state.get("defense_chain", []))
        defense_chain = parent_chain + [hypothesis.defense_name]

        parent_result = parent_state.get("result")
        parent_topology = (
            _topology_snapshot(parent_result)
            if parent_result is not None
            else None
        )

        seed_metadata = dict(parent_state.get("seed_metadata") or {})

        metadata = dict(hypothesis.metadata or {})
        metadata.update(
            {
                "source": "trace_reconstruction",
                "branch_source": "stage_continuation",
                "branch_state_id": parent_state.get("state_id"),
                "branch_parent_hypothesis_id": parent_state.get(
                    "source_hypothesis_id"
                ),
                "branch_parent_stage_index": parent_state.get(
                    "created_stage_index"
                ),
                "defense_chain": defense_chain,

                # This tells scoring/rejection to compare against the branch parent.
                "score_against_branch_parent": True,
                "branch_parent_topology": parent_topology,

                # h0-relative accounting for UI.
                "added_ink_pixels": added_px,
                "removed_ink_pixels": removed_px,
                "changed_ink_pixels": changed_px,
                "added_ink_ratio": added_px / original_px,
                "removed_ink_ratio": removed_px / original_px,
                "changed_ink_ratio": changed_px / original_px,

                # Branch-local accounting for scoring.
                "branch_added_ink_pixels": branch_added_px,
                "branch_removed_ink_pixels": branch_removed_px,
                "branch_changed_ink_pixels": branch_changed_px,
                "branch_added_ink_ratio": branch_added_px / parent_px,
                "branch_removed_ink_ratio": branch_removed_px / parent_px,
                "branch_changed_ink_ratio": branch_changed_px / parent_px,

                # Carry cleanup confidence forward.
                "parent_cleanup_branch_type": parent_state.get("cleanup_branch_type"),
                "parent_cleanup_confidence": float(
                    parent_state.get("cleanup_confidence", 0.0)
                ),
                "parent_line_artifact_confidence": float(
                    seed_metadata.get("line_artifact_confidence", 0.0)
                ),
            }
        )

        return DefenseHypothesis(
            hypothesis_id=hypothesis.hypothesis_id,
            defense_name=hypothesis.defense_name,
            candidate_mask=candidate_bin,
            added_mask=added,
            removed_mask=removed,
            metadata=metadata,
        )


    def _generate_stage_hypotheses(
        self,
        *,
        binary_mask,
        original_result,
        routed_defenses,
        stable_unit_id,
    ):
        """Generate candidate hypotheses in registry stage order.

        Important:
        - h0_original starts as the only branch state.
        - linear_artifact_removal creates cleanup branch states.
        - after a cleanup branch exists, later stages run only on cleanup branches.
        - h0_original remains fallback, but stops generating downstream repairs.
        """
        raw_hypotheses = []
        stage_records = []
        next_index = 1

        stage_map = grouped_defenses_by_stage(
            routed_defenses,
            include_empty_stages=True,
            include_candidate_stages=False,
        )

        original_bin = _as_binary_mask(binary_mask)

        branch_states = [
            {
                "state_id": "h0_original",
                "source_hypothesis_id": None,
                "mask": original_bin,
                "result": original_result,
                "defense_chain": [],
                "available_stage_index": 0,
                "created_stage_index": -1,
                "is_cleanup_branch": False,
                "cleanup_branch_type": None,
            }
        ]

        exclusive_branch_mode = False
        exclusive_branch_reason = None

        max_new_line_states_per_stage = 4
        max_total_branch_states = 8
        min_line_branch_confidence = 0.30

        for stage_index, stage in enumerate(DEFENSE_STAGE_ORDER):
            stage_defenses = list(stage_map.get(stage, []))
            implemented_here = []
            generated_here = []
            new_branch_states = []

            stage_record = {
                "stage": stage,
                "stage_index": stage_index,
                "routed_defense_types": stage_defenses,
                "implemented_defense_types": [],
                "generated_hypothesis_ids": [],
                "generated_count": 0,
                "branch_state_ids_used": [],
                "new_branch_state_ids": [],
                "exclusive_branch_mode": bool(exclusive_branch_mode),
                "exclusive_branch_reason": exclusive_branch_reason,
                "notes": [],
            }

            if not stage_defenses:
                stage_records.append(stage_record)
                continue

            eligible_states = [
                state
                for state in branch_states
                if int(state.get("available_stage_index", 0)) <= stage_index
            ]

            # Critical rule:
            # Once a line-removal cleanup branch exists, later stages must not
            # run on h0_original. h0 remains fallback, but it cannot create
            # endpoint bridges around the contaminating line.
            if exclusive_branch_mode:
                cleanup_states = [
                    state
                    for state in eligible_states
                    if bool(state.get("is_cleanup_branch"))
                ]

                if cleanup_states:
                    eligible_states = cleanup_states

            for state in eligible_states:
                state_id = str(state.get("state_id"))
                state_mask = _as_binary_mask(state["mask"])
                state_result = state["result"]
                state_chain = list(state.get("defense_chain", []))
                is_original_state = state_id == "h0_original"

                if state_id not in stage_record["branch_state_ids_used"]:
                    stage_record["branch_state_ids_used"].append(state_id)

                # ------------------------------------------------------
                # Endpoint bridge.
                #
                # This is implemented inside trace_reconstruction.py.
                # It must be allowed to run on cleanup branch states.
                # ------------------------------------------------------
                if DEFENSE_ENDPOINT_BRIDGE in stage_defenses:
                    bridge_focus_mask = state.get("line_removal_focus_mask")
                    endpoint_hypotheses = self._generate_endpoint_bridge_hypotheses(
                        mask=state_mask,
                        original_result=state_result,
                        start_index=next_index,
                        stable_unit_id=stable_unit_id,
                        focus_mask=bridge_focus_mask,
                        focus_line_angle_degrees=state.get(
                            "line_removal_angle_degrees"
                        ),
                        focus_line_metadata=state.get("seed_metadata"),
                    )

                    # Mark endpoint_bridge as implemented even if it produces
                    # zero candidates. Otherwise it incorrectly appears as
                    # unsupported instead of no_candidate.
                    if DEFENSE_ENDPOINT_BRIDGE not in implemented_here:
                        implemented_here.append(DEFENSE_ENDPOINT_BRIDGE)

                    for hypothesis in endpoint_hypotheses:
                        if is_original_state:
                            hypothesis.metadata["defense_chain"] = [
                                DEFENSE_ENDPOINT_BRIDGE
                            ]
                            rebased = hypothesis
                        else:
                            rebased = self._rebase_hypothesis_against_original(
                                hypothesis=hypothesis,
                                original_mask=original_bin,
                                parent_state=state,
                            )

                        if rebased is None:
                            continue

                        rebased.metadata["stage"] = stage
                        rebased.metadata["stage_index"] = stage_index
                        rebased.metadata["stage_source"] = "trace_reconstruction"
                        rebased.metadata["branch_state_id"] = state_id

                        generated_here.append(rebased)

                    next_index += len(endpoint_hypotheses)

                # ------------------------------------------------------
                # Linear artifact removal.
                #
                # Detector still lives in trace_defenses.py.
                # Reconstruction turns strong line-removal candidates into
                # branch states for later stages.
                # ------------------------------------------------------
                if (
                    DEFENSE_LINEAR_ARTIFACT_REMOVAL in stage_defenses
                    and DEFENSE_LINEAR_ARTIFACT_REMOVAL not in state_chain
                ):
                    line_hypotheses = generate_defense_hypotheses(
                        mask=state_mask,
                        allowed_defenses=[DEFENSE_LINEAR_ARTIFACT_REMOVAL],
                        original_result=state_result,
                        stable_unit_id=stable_unit_id,
                        start_index=next_index,
                    )

                    if line_hypotheses and DEFENSE_LINEAR_ARTIFACT_REMOVAL not in implemented_here:
                        implemented_here.append(DEFENSE_LINEAR_ARTIFACT_REMOVAL)

                    rebased_line_hypotheses = []

                    for hypothesis in line_hypotheses:
                        if is_original_state:
                            hypothesis.metadata["defense_chain"] = [
                                DEFENSE_LINEAR_ARTIFACT_REMOVAL
                            ]
                            rebased = hypothesis
                        else:
                            rebased = self._rebase_hypothesis_against_original(
                                hypothesis=hypothesis,
                                original_mask=original_bin,
                                parent_state=state,
                            )

                        if rebased is None:
                            continue

                        rebased.metadata["stage"] = stage
                        rebased.metadata["stage_index"] = stage_index
                        rebased.metadata["stage_source"] = "trace_reconstruction"
                        rebased.metadata["branch_state_id"] = state_id
                        rebased.metadata["reconstruction_role"] = "cleanup_branch_seed"

                        # This candidate is allowed to be provisional because
                        # later stages are expected to repair the wound.
                        rebased.metadata["creates_branch_state"] = True
                        rebased.metadata["requires_downstream_repair"] = True
                        rebased.metadata["provisional_parent"] = True
                        rebased.metadata["allow_topology_damage_before_repair"] = True

                        rebased_line_hypotheses.append(rebased)
                        generated_here.append(rebased)

                    next_index += len(line_hypotheses)

                    branch_seed_candidates = sorted(
                        rebased_line_hypotheses,
                        key=lambda item: float(
                            item.metadata.get("line_artifact_confidence", 0.0)
                        ),
                        reverse=True,
                    )

                    for seed in branch_seed_candidates[:max_new_line_states_per_stage]:
                        confidence = float(
                            seed.metadata.get("line_artifact_confidence", 0.0)
                        )

                        if confidence < min_line_branch_confidence:
                            seed.metadata["branch_state_rejected_reason"] = (
                                "line_artifact_confidence_too_low"
                            )
                            continue

                        try:
                            seed_trace = _trace_mask(seed.candidate_mask, self.settings)
                            seed_result = seed_trace["result"]
                        except Exception as error:
                            seed.metadata["branch_trace_error"] = str(error)
                            continue

                        seed_chain = list(seed.metadata.get("defense_chain", []))
                        focus_mask = _as_binary_mask(seed.removed_mask)
                        parent_endpoint_audit = _endpoint_audit_near_mask(
                            state_mask,
                            focus_mask,
                            self.settings,
                        )
                        child_endpoint_audit = _endpoint_audit_near_mask(
                            seed.candidate_mask,
                            focus_mask,
                            self.settings,
                        )
                        parent_endpoint_count = int(
                            parent_endpoint_audit["endpoint_count_near_focus"]
                        )
                        child_endpoint_count = int(
                            child_endpoint_audit["endpoint_count_near_focus"]
                        )
                        new_endpoint_count = max(
                            0,
                            child_endpoint_count - parent_endpoint_count,
                        )

                        seed.metadata["line_cut_parent_endpoint_count_near_wound"] = (
                            parent_endpoint_count
                        )
                        seed.metadata["line_cut_child_endpoint_count_near_wound"] = (
                            child_endpoint_count
                        )
                        seed.metadata["line_cut_new_endpoint_count_near_wound"] = (
                            new_endpoint_count
                        )
                        seed.metadata["line_cut_parent_endpoints_near_wound"] = (
                            parent_endpoint_audit["endpoints_near_focus"]
                        )
                        seed.metadata["line_cut_child_endpoints_near_wound"] = (
                            child_endpoint_audit["endpoints_near_focus"]
                        )

                        new_state = {
                            "state_id": seed.hypothesis_id,
                            "source_hypothesis_id": seed.hypothesis_id,
                            "mask": _as_binary_mask(seed.candidate_mask),
                            "result": seed_result,
                            "defense_chain": seed_chain,
                            "available_stage_index": stage_index + 1,
                            "created_stage_index": stage_index,

                            # This is a cleanup branch.
                            "is_cleanup_branch": True,
                            "cleanup_branch_type": DEFENSE_LINEAR_ARTIFACT_REMOVAL,
                            "cleanup_confidence": confidence,

                            # Preserve the parent line-removal evidence for later-stage scoring.
                            "seed_metadata": dict(seed.metadata or {}),
                            "line_removal_angle_degrees": seed.metadata.get(
                                "line_angle_degrees"
                            ),
                            "line_removal_orientation_class": seed.metadata.get(
                                "line_orientation_class"
                            ),
                            "preferred_bridge_angle_from_horizontal": seed.metadata.get(
                                "preferred_bridge_angle_from_horizontal"
                            ),
                            "line_cut_new_endpoint_count": new_endpoint_count,

                            # Endpoint bridge should repair this exact cut, not
                            # hunt for unrelated endpoint pairs elsewhere.
                            "line_removal_focus_mask": focus_mask,
                        }

                        new_branch_states.append(new_state)

                # ------------------------------------------------------
                # Normal trace_defenses.py tools.
                #
                # linear_artifact_removal is excluded because it is handled
                # specially above.
                # ------------------------------------------------------
                trace_defense_names = [
                    defense_name
                    for defense_name in stage_defenses
                    if implemented_in_trace_defenses(defense_name)
                    and defense_name != DEFENSE_LINEAR_ARTIFACT_REMOVAL
                ]

                if trace_defense_names:
                    defense_hypotheses = generate_defense_hypotheses(
                        mask=state_mask,
                        allowed_defenses=trace_defense_names,
                        original_result=state_result,
                        stable_unit_id=stable_unit_id,
                        start_index=next_index,
                    )

                    rebased_defense_hypotheses = []

                    for hypothesis in defense_hypotheses:
                        if is_original_state:
                            hypothesis.metadata["defense_chain"] = [
                                hypothesis.defense_name
                            ]
                            rebased = hypothesis
                        else:
                            rebased = self._rebase_hypothesis_against_original(
                                hypothesis=hypothesis,
                                original_mask=original_bin,
                                parent_state=state,
                            )

                        if rebased is None:
                            continue

                        rebased.metadata["stage"] = stage
                        rebased.metadata["stage_index"] = stage_index
                        rebased.metadata["stage_source"] = "trace_defenses"
                        rebased.metadata["branch_state_id"] = state_id

                        rebased_defense_hypotheses.append(rebased)
                        generated_here.append(rebased)

                    implemented_here.extend(
                        defense_name
                        for defense_name in trace_defense_names
                        if any(
                            hypothesis.defense_name == defense_name
                            for hypothesis in rebased_defense_hypotheses
                        )
                    )

                    next_index += len(defense_hypotheses)

            # Add new cleanup branches only after the stage finishes.
            # That prevents line removal from recursively running again in
            # the same stage.
            if new_branch_states:
                new_branch_states = sorted(
                    new_branch_states,
                    key=lambda state: float(state.get("cleanup_confidence", 0.0)),
                    reverse=True,
                )

                available_slots = max(0, max_total_branch_states - len(branch_states))
                accepted_new_states = new_branch_states[:available_slots]

                branch_states.extend(accepted_new_states)

                stage_record["new_branch_state_ids"] = [
                    state["state_id"] for state in accepted_new_states
                ]

                if any(
                    state.get("cleanup_branch_type") == DEFENSE_LINEAR_ARTIFACT_REMOVAL
                    for state in accepted_new_states
                ):
                    exclusive_branch_mode = True
                    exclusive_branch_reason = DEFENSE_LINEAR_ARTIFACT_REMOVAL
                    stage_record["exclusive_branch_mode_started"] = True
                    stage_record["exclusive_branch_reason"] = exclusive_branch_reason

            for defense_name in stage_defenses:
                if defense_name not in implemented_here:
                    stage_record["notes"].append(
                        f"{defense_name}: no candidate produced or not implemented at this stage"
                    )

            stage_record["implemented_defense_types"] = _unique_names(implemented_here)
            stage_record["generated_hypothesis_ids"] = [
                hypothesis.hypothesis_id
                for hypothesis in generated_here
            ]
            stage_record["generated_count"] = len(generated_here)

            raw_hypotheses.extend(generated_here)
            stage_records.append(stage_record)

        return raw_hypotheses, stage_records

    def _generate_endpoint_bridge_hypotheses(
        self,
        *,
        mask,
        original_result,
        start_index,
        stable_unit_id,
        focus_mask=None,
        focus_line_angle_degrees=None,
        focus_line_metadata=None,
    ):
        binary = _as_binary_mask(mask)
        candidates = self._endpoint_bridge_candidates(
            binary,
            original_result,
            focus_mask=focus_mask,
            focus_line_angle_degrees=focus_line_angle_degrees,
        )
        hypotheses = []

        next_index = int(start_index)
        for candidate in candidates:
            candidate_mask = binary.copy()
            point_a = candidate["point_a"]
            point_b = candidate["point_b"]
            thickness = int(candidate["bridge_thickness"])
            cv2.line(
                candidate_mask,
                (point_a.x, point_a.y),
                (point_b.x, point_b.y),
                255,
                thickness,
                cv2.LINE_8,
            )

            added, removed = _changed_masks(binary, candidate_mask)
            added_px = int(cv2.countNonZero(added))
            if added_px <= 0:
                continue

            original_px = max(1, int(cv2.countNonZero(binary)))
            metadata = {
                "stable_unit_id": stable_unit_id,
                "source": "trace_reconstruction_endpoint_bridge",
                "defense_spec": get_defense_spec_dict("endpoint_bridge"),
                "point_a": point_a.to_dict(),
                "point_b": point_b.to_dict(),
                "path_id_a": candidate["path_id_a"],
                "path_id_b": candidate["path_id_b"],
                "distance": candidate["distance"],
                "angle_a": candidate["angle_a"],
                "angle_b": candidate["angle_b"],
                "best_angle": candidate.get("best_angle"),
                "worst_angle": candidate.get("worst_angle"),
                "bridge_angle_from_horizontal": candidate.get("bridge_angle_from_horizontal"),
                "is_diagonal_bridge": candidate.get("is_diagonal_bridge", False),
                "angle_policy": candidate.get("angle_policy", "unknown"),
                "occupied_ratio": candidate.get("occupied_ratio"),
                "bridge_focus_policy": candidate.get("bridge_focus_policy"),
                "focus_overlap_pixels": candidate.get("focus_overlap_pixels"),
                "focus_overlap_ratio": candidate.get("focus_overlap_ratio"),
                "focus_endpoint_near_count": candidate.get("focus_endpoint_near_count"),
                "removed_line_crossing_pixels": candidate.get(
                    "removed_line_crossing_pixels"
                ),
                "removed_line_crossing_ratio": candidate.get(
                    "removed_line_crossing_ratio"
                ),
                "bridge_crosses_removed_line": candidate.get(
                    "bridge_crosses_removed_line"
                ),
                "focus_line_angle_degrees": candidate.get("focus_line_angle_degrees"),
                "focus_line_orientation_class": (
                    (focus_line_metadata or {}).get("line_orientation_class")
                ),
                "preferred_bridge_angle_from_horizontal": candidate.get(
                    "preferred_bridge_angle_from_horizontal"
                ),
                "bridge_orientation_score": candidate.get("bridge_orientation_score"),
                "bridge_orientation_policy": candidate.get("bridge_orientation_policy"),
                "geometry_score": candidate["geometry_score"],
                "different_paths": candidate["different_paths"],
                "different_components": candidate["different_components"],
                "bridge_thickness": thickness,
                "added_ink_pixels": added_px,
                "removed_ink_pixels": int(cv2.countNonZero(removed)),
                "changed_ink_pixels": added_px + int(cv2.countNonZero(removed)),
                "added_ink_ratio": added_px / original_px,
                "removed_ink_ratio": int(cv2.countNonZero(removed)) / original_px,
                "changed_ink_ratio": (
                    added_px + int(cv2.countNonZero(removed))
                ) / original_px,
            }

            hypotheses.append(
                DefenseHypothesis(
                    hypothesis_id=f"h{next_index}_endpoint_bridge",
                    defense_name="endpoint_bridge",
                    candidate_mask=_as_binary_mask(candidate_mask),
                    added_mask=added,
                    removed_mask=removed,
                    metadata=metadata,
                )
            )
            next_index += 1

        return hypotheses


    def _endpoint_bridge_candidates(
        self,
        mask,
        original_result,
        focus_mask=None,
        focus_line_angle_degrees=None,
    ):
        """Find endpoint bridge candidates.

        Important behavior:
        - Both endpoints must agree with the bridge direction.
        - The angle limit is only slightly relaxed from settings.
        - Near-vertical bridges are still not allowed.
        - If a focus mask is provided, the bridge must touch that wound.
        """
        skeleton = TraceSkeletonizer(self.settings).skeletonize(mask)
        points = SkeletonPointExtractor().extract_points(skeleton)
        graph = SkeletonGraph(points)
        trace_paths = TracePathExtractor(self.settings).extract_paths(graph)
        component_lookup = _graph_component_lookup(graph)

        focus_zone = None
        raw_focus_bin = None
        focus_policy = None
        preferred_bridge_angle = None
        line_abs_angle = None

        if focus_line_angle_degrees is not None:
            try:
                line_abs_angle = min(90.0, abs(float(focus_line_angle_degrees)))
                preferred_bridge_angle = 90.0 - line_abs_angle
            except (TypeError, ValueError):
                line_abs_angle = None
                preferred_bridge_angle = None

        if focus_mask is not None:
            focus_bin = _as_binary_mask(focus_mask)
            if int(cv2.countNonZero(focus_bin)) > 0:
                raw_focus_bin = focus_bin
                # The removed line is the wound. Dilating by a tiny radius lets
                # endpoints sitting beside the cut still count as cut-created.
                focus_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                focus_zone = cv2.dilate(focus_bin, focus_kernel, iterations=1)
                focus_policy = "line_removal_cut_mask"

        endpoint_records = []

        for endpoint in graph.endpoints():
            tangent = _endpoint_tangent(
                endpoint,
                trace_paths,
                self.settings.reconstruction_tangent_points,
            )
            if tangent is None:
                continue

            path_id, tx, ty = tangent
            endpoint_records.append(
                {
                    "point": endpoint,
                    "path_id": path_id,
                    "tangent": (tx, ty),
                    "component_id": component_lookup.get(endpoint.to_tuple()),
                }
            )

        candidates = []

        max_distance = float(self.settings.reconstruction_max_bridge_length_px)
        min_distance = float(self.settings.reconstruction_min_endpoint_separation_px)
        max_angle = float(self.settings.reconstruction_max_bridge_angle_degrees)
        relaxed_angle = min(90.0, max_angle + 8.0)

        for index_a, record_a in enumerate(endpoint_records):
            for record_b in endpoint_records[index_a + 1:]:
                point_a = record_a["point"]
                point_b = record_b["point"]

                # Do not bridge both ends of the same path.
                # That often creates loops or closes shapes incorrectly.
                if record_a["path_id"] == record_b["path_id"]:
                    continue

                dx = point_b.x - point_a.x
                dy = point_b.y - point_a.y
                distance = math.hypot(dx, dy)

                if distance < min_distance or distance > max_distance:
                    continue

                vector_ab = (dx, dy)
                vector_ba = (-dx, -dy)

                angle_a = _angle_degrees(record_a["tangent"], vector_ab)
                angle_b = _angle_degrees(record_b["tangent"], vector_ba)

                bridge_angle_from_horizontal = abs(math.degrees(math.atan2(dy, dx)))

                # Normalize to 0..90 degrees.
                if bridge_angle_from_horizontal > 90:
                    bridge_angle_from_horizontal = 180 - bridge_angle_from_horizontal

                is_near_vertical_bridge = bridge_angle_from_horizontal > 72.0

                # Global endpoint_bridge still blocks near-vertical bridges.
                # A line-removal branch is different: if the removed artifact
                # was horizontal-like, the mathematically correct repair is
                # often a vertical bridge crossing the wound.
                if is_near_vertical_bridge and focus_zone is None:
                    continue

                best_angle = min(angle_a, angle_b)
                worst_angle = max(angle_a, angle_b)

                if angle_a > relaxed_angle or angle_b > relaxed_angle:
                    continue

                line_points = _line_coordinates(point_a, point_b)
                in_bounds = [
                    (x, y)
                    for x, y in line_points
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]
                ]

                if not in_bounds:
                    continue

                focus_overlap_pixels = 0
                focus_overlap_ratio = 0.0
                removed_line_crossing_pixels = 0
                removed_line_crossing_ratio = 0.0
                focus_endpoint_near_count = 0

                if focus_zone is not None:
                    focus_overlap_pixels = sum(
                        1 for x, y in in_bounds if focus_zone[y, x] > 0
                    )
                    focus_overlap_ratio = focus_overlap_pixels / max(1, len(in_bounds))
                    removed_line_crossing_pixels = sum(
                        1 for x, y in in_bounds if raw_focus_bin[y, x] > 0
                    )
                    removed_line_crossing_ratio = (
                        removed_line_crossing_pixels / max(1, len(in_bounds))
                    )
                    focus_endpoint_near_count = int(
                        focus_zone[point_a.y, point_a.x] > 0
                    ) + int(focus_zone[point_b.y, point_b.x] > 0)

                    # After line removal, bridge only the wound created by that
                    # cut. This prevents random endpoint pairs from being joined
                    # just because the cleaned mask now has extra endpoints.
                    if focus_overlap_pixels <= 0:
                        continue
                    if focus_endpoint_near_count <= 0:
                        continue
                    if removed_line_crossing_pixels <= 0:
                        continue

                bridge_orientation_score = 0.5
                bridge_orientation_policy = "unconstrained"

                if preferred_bridge_angle is not None:
                    bridge_orientation_error = abs(
                        bridge_angle_from_horizontal - preferred_bridge_angle
                    )
                    bridge_orientation_score = 1.0 - min(
                        1.0,
                        bridge_orientation_error / 90.0,
                    )
                    bridge_orientation_policy = "perpendicular_to_removed_line"

                    # If the bridge is meant to repair a line-removal wound,
                    # require it to at least roughly oppose the deleted line.
                    # This keeps horizontal artifacts from producing horizontal
                    # bridges elsewhere, and vertical artifacts from producing
                    # vertical bridges along the original artifact direction.
                    if bridge_orientation_score < 0.45:
                        continue

                occupied = sum(1 for x, y in in_bounds if mask[y, x] > 0)
                occupied_ratio = occupied / max(1, len(in_bounds))

                # The line can touch the endpoints, but most of the route should be empty.
                if occupied_ratio > 0.45:
                    continue

                different_paths = record_a["path_id"] != record_b["path_id"]
                different_components = (
                    record_a.get("component_id") != record_b.get("component_id")
                )

                distance_score = 1.0 - min(1.0, distance / max_distance)

                angle_score = 1.0 - min(
                    1.0,
                    worst_angle / max(1.0, relaxed_angle),
                )

                emptiness_score = 1.0 - occupied_ratio

                if focus_zone is not None and preferred_bridge_angle is not None:
                    geometry_score = (
                        0.25 * distance_score
                        + 0.30 * angle_score
                        + 0.20 * emptiness_score
                        + 0.25 * bridge_orientation_score
                    )
                else:
                    geometry_score = (
                        0.35 * distance_score
                        + 0.40 * angle_score
                        + 0.25 * emptiness_score
                    )

                geometry_score = max(0.0, min(1.0, geometry_score))

                candidates.append(
                    {
                        "point_a": point_a,
                        "point_b": point_b,
                        "path_id_a": record_a["path_id"],
                        "path_id_b": record_b["path_id"],
                        "different_paths": different_paths,
                        "different_components": different_components,
                        "distance": float(distance),
                        "angle_a": float(angle_a),
                        "angle_b": float(angle_b),
                        "best_angle": float(best_angle),
                        "worst_angle": float(worst_angle),
                        "bridge_angle_from_horizontal": float(
                            bridge_angle_from_horizontal
                        ),
                        "angle_policy": "endpoint_agreement_relaxed",
                        "angle_limit_degrees": float(relaxed_angle),
                        "occupied_ratio": float(occupied_ratio),
                        "bridge_focus_policy": focus_policy,
                        "focus_overlap_pixels": int(focus_overlap_pixels),
                        "focus_overlap_ratio": float(focus_overlap_ratio),
                        "focus_endpoint_near_count": int(focus_endpoint_near_count),
                        "removed_line_crossing_pixels": int(
                            removed_line_crossing_pixels
                        ),
                        "removed_line_crossing_ratio": float(
                            removed_line_crossing_ratio
                        ),
                        "bridge_crosses_removed_line": bool(
                            removed_line_crossing_pixels > 0
                        ),
                        "focus_line_angle_degrees": (
                            None
                            if focus_line_angle_degrees is None
                            else float(focus_line_angle_degrees)
                        ),
                        "preferred_bridge_angle_from_horizontal": (
                            None
                            if preferred_bridge_angle is None
                            else float(preferred_bridge_angle)
                        ),
                        "bridge_orientation_score": float(bridge_orientation_score),
                        "bridge_orientation_policy": bridge_orientation_policy,
                        "geometry_score": float(geometry_score),
                        "bridge_thickness": _estimate_bridge_thickness(
                            mask,
                            point_a,
                            point_b,
                        ),
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

        return candidates

    def _dedupe_hypotheses(self, hypotheses):
        unique = []
        seen = set()

        for hypothesis in hypotheses:
            key = _as_binary_mask(hypothesis.candidate_mask).tobytes()
            if key in seen:
                continue
            seen.add(key)
            unique.append(hypothesis)

        return unique

    def _evaluate_hypothesis(
        self,
        *,
        hypothesis,
        original_mask,
        original_result,
        original_topology,
        original_recognition,
        output_dir,
        stable_unit_id,
    ):
        candidate_trace = _trace_mask(hypothesis.candidate_mask, self.settings)
        candidate_result = candidate_trace["result"]
        candidate_topology = _topology_snapshot(candidate_result)

        recognition = (
            _recognition_snapshot(candidate_result)
            if self.settings.reconstruction_use_recognition_verification
            else {"available": False, "top1_confidence": 0.0, "top5": [], "error": None}
        )

        scoring_original_topology = original_topology

        if bool(hypothesis.metadata.get("score_against_branch_parent")):
            branch_parent_topology = hypothesis.metadata.get("branch_parent_topology")
            if isinstance(branch_parent_topology, dict):
                scoring_original_topology = branch_parent_topology

        score_info = self._score_hypothesis(
            hypothesis=hypothesis,
            original_topology=scoring_original_topology,
            candidate_topology=candidate_topology,
            original_recognition=original_recognition,
            candidate_recognition=recognition,
        )

        rejection_reasons = self._rejection_reasons(
            hypothesis=hypothesis,
            score_info=score_info,
            original_topology=scoring_original_topology,
            candidate_topology=candidate_topology,
            original_recognition=original_recognition,
            candidate_recognition=recognition,
        )

        accepted = not rejection_reasons

        paths = self._save_candidate_debug(
            hypothesis=hypothesis,
            original_mask=original_mask,
            candidate_trace=candidate_trace,
            output_dir=output_dir,
            stable_unit_id=stable_unit_id,
            selected=False,
        )
        safe_metadata = {
            key: value
            for key, value in dict(hypothesis.metadata).items()
            if not str(key).startswith("_")
        }

        return {
            "hypothesis_id": hypothesis.hypothesis_id,
            "defense_name": hypothesis.defense_name,
            "defense_spec": get_defense_spec_dict(hypothesis.defense_name),
            "stage": hypothesis.metadata.get("stage"),
            "stage_index": hypothesis.metadata.get("stage_index"),
            "stage_source": hypothesis.metadata.get("stage_source"),
            "accepted": accepted,
            "selected": False,
            "score": score_info["score"],
            "score_breakdown": score_info,
            "rejection_reasons": rejection_reasons,
            "topology": candidate_topology,
            "topology_delta": _topology_delta(original_topology, candidate_topology),
            "scoring_topology_delta": _topology_delta(
                scoring_original_topology,
                candidate_topology,
            ),
            "scoring_against_branch_parent": bool(
                hypothesis.metadata.get("score_against_branch_parent")
            ),
            "recognition": recognition,
            "candidate_mask_path": paths["candidate_mask_path"],
            "candidate_visual_path": paths["candidate_visual_path"],
            "added_mask_path": paths["added_mask_path"],
            "removed_mask_path": paths["removed_mask_path"],
            "overlay_path": paths["overlay_path"],
            "retrace_skeleton_path": paths["retrace_skeleton_path"],
            "retrace_graph_path": paths["retrace_graph_path"],
            "retrace_paths_path": paths["retrace_paths_path"],
            "retrace_landmarks_path": paths["retrace_landmarks_path"],
            "feature_vector": (
                candidate_result.feature_vector.to_dict()
                if candidate_result.feature_vector is not None
                else None
            ),
            "scrilog_observation": candidate_result.metrics.get(
                "scrilog_observation"
            ),
            "metadata": safe_metadata,
        }

    def _enforce_line_cut_downstream_bridge(self, scored_hypotheses):
        """Require wound-focused bridges after line cuts that create endpoints.

        A line-removal hypothesis is a cleanup branch seed. If that cut creates
        fresh endpoints near the removed pixels, the final repair should be a
        downstream candidate such as:

            linear_artifact_removal -> endpoint_bridge

        This keeps a cut-only candidate from winning when it wounded the glyph.
        """
        completed_parent_ids = set()

        for record in scored_hypotheses:
            metadata = record.get("metadata") or {}

            if not record.get("accepted"):
                continue

            if record.get("defense_name") != DEFENSE_ENDPOINT_BRIDGE:
                continue

            if metadata.get("bridge_focus_policy") != "line_removal_cut_mask":
                continue

            if int(metadata.get("focus_overlap_pixels") or 0) <= 0:
                continue

            if not bool(metadata.get("bridge_crosses_removed_line")):
                continue

            if int(metadata.get("removed_line_crossing_pixels") or 0) <= 0:
                continue

            if int(metadata.get("focus_endpoint_near_count") or 0) <= 0:
                continue

            parent_id = metadata.get("branch_parent_hypothesis_id")
            if parent_id:
                completed_parent_ids.add(str(parent_id))

        for record in scored_hypotheses:
            metadata = record.get("metadata") or {}

            if record.get("defense_name") != DEFENSE_LINEAR_ARTIFACT_REMOVAL:
                continue

            if not bool(metadata.get("requires_downstream_repair")):
                continue

            new_endpoint_count = int(
                metadata.get("line_cut_new_endpoint_count_near_wound") or 0
            )
            repair_required = new_endpoint_count > 0
            parent_id = str(record.get("hypothesis_id"))
            repair_completed = parent_id in completed_parent_ids

            metadata["downstream_bridge_required"] = bool(repair_required)
            metadata["downstream_bridge_completed"] = bool(repair_completed)
            metadata["completed_by_downstream_bridge"] = bool(repair_completed)

            if not repair_required:
                continue

            record["accepted"] = False
            record["selected"] = False

            rejection_reasons = record.setdefault("rejection_reasons", [])
            if repair_completed:
                reason = "intermediate_line_cut_completed_by_downstream_bridge"
            else:
                reason = "missing_required_downstream_bridge_for_line_cut"

            if reason not in rejection_reasons:
                rejection_reasons.append(reason)

    def _score_hypothesis(
        self,
        *,
        hypothesis,
        original_topology,
        candidate_topology,
        original_recognition,
        candidate_recognition,
    ):
        topology_gain = _topology_gain(
            original_topology,
            candidate_topology,
            different_components=bool(
                hypothesis.metadata.get(
                    "different_components",
                    hypothesis.metadata.get("different_paths"),
                )
            ),
        )

        topology_score = 0.5 + 0.5 * topology_gain
        geometry_score = float(hypothesis.metadata.get("geometry_score", 0.55))

        confidence_gain = (
            candidate_recognition.get("top1_confidence", 0.0)
            - original_recognition.get("top1_confidence", 0.0)
        )
        confidence_score = 0.5 + 0.5 * max(-1.0, min(1.0, confidence_gain))

        if bool(hypothesis.metadata.get("score_against_branch_parent")):
            changed_ratio = float(
                hypothesis.metadata.get(
                    "branch_changed_ink_ratio",
                    hypothesis.metadata.get("changed_ink_ratio", 0.0),
                )
            )
        else:
            changed_ratio = float(hypothesis.metadata.get("changed_ink_ratio", 0.0))

        is_line_removal = hypothesis.defense_name == DEFENSE_LINEAR_ARTIFACT_REMOVAL
        line_artifact_confidence = float(
            hypothesis.metadata.get("line_artifact_confidence", 0.0)
        )

        if is_line_removal:
            # Line removal is different from normal cleanup.
            # It can temporarily damage topology because removing a foreign stroke
            # may create gaps, endpoints, or broken paths before downstream repair.
            #
            # So we reduce the intervention penalty and reward strong line evidence.
            intervention_penalty = min(0.12, changed_ratio * 0.18)
            intervention_policy = "line_artifact_provisional"

            line_confidence_bonus = 0.22 * max(
                0.0,
                min(1.0, line_artifact_confidence),
            )

            # If the detector visually found a strong line, do not let geometry stay neutral.
            # This helps obvious line-removal candidates survive long enough for bridge/gap repair.
            geometry_score = max(
                geometry_score,
                0.68 + 0.22 * max(0.0, min(1.0, line_artifact_confidence)),
            )
        elif bool(hypothesis.metadata.get("score_against_branch_parent")):
            # This is a later-stage repair running on a cleanup branch.
            # Example: line_removal -> endpoint_bridge.
            # It should be judged mostly by branch-local improvement, not h0 damage.
            intervention_penalty = min(0.08, changed_ratio * 0.45)
            intervention_policy = "branch_continuation"

            cleanup_confidence = float(
                hypothesis.metadata.get(
                    "parent_line_artifact_confidence",
                    hypothesis.metadata.get("parent_cleanup_confidence", 0.0),
                )
            )

            line_confidence_bonus = 0.06 * max(0.0, min(1.0, cleanup_confidence))

        elif topology_gain >= 0.45:
            # Severe Aristotel damage may require replacing a lot of pixels.
            # Strong topology improvement means the repair earned that freedom.
            intervention_penalty = min(0.08, changed_ratio * 0.18)
            intervention_policy = "topology_rescue"
            line_confidence_bonus = 0.0

        elif topology_gain >= 0.20:
            intervention_penalty = min(0.14, changed_ratio * 0.35)
            intervention_policy = "topology_assisted"
            line_confidence_bonus = 0.0

        else:
            intervention_penalty = min(0.25, changed_ratio)
            intervention_policy = "strict"
            line_confidence_bonus = 0.0

        score = (
            float(self.settings.reconstruction_topology_weight) * topology_score
            + float(self.settings.reconstruction_geometry_weight) * geometry_score
            + float(self.settings.reconstruction_confidence_weight) * confidence_score
            - intervention_penalty
            + line_confidence_bonus
        )

        return {
            "score": max(0.0, min(1.0, score)),
            "topology_gain": topology_gain,
            "topology_score": topology_score,
            "geometry_score": geometry_score,
            "confidence_gain": confidence_gain,
            "confidence_score": confidence_score,
            "intervention_penalty": intervention_penalty,
            "intervention_policy": intervention_policy,
            "changed_ink_ratio": changed_ratio,

            # New line-removal audit fields.
            "line_artifact_confidence": line_artifact_confidence,
            "line_confidence_bonus": line_confidence_bonus,

            "preferred_acceptance_score": float(
                self.settings.reconstruction_min_acceptance_score
            ),
            "hard_acceptance_floor": max(
                0.0,
                float(self.settings.reconstruction_min_acceptance_score) - 0.12,
            ),
        }

    def _rejection_reasons(
        self,
        *,
        hypothesis,
        score_info,
        original_topology,
        candidate_topology,
        original_recognition,
        candidate_recognition,
    ):
        reasons = []
        metadata = hypothesis.metadata
        spec = get_defense_spec_dict(hypothesis.defense_name)

        is_line_removal = hypothesis.defense_name == DEFENSE_LINEAR_ARTIFACT_REMOVAL
        is_provisional_line_parent = (
            is_line_removal
            and bool(metadata.get("requires_downstream_repair"))
            and bool(metadata.get("allow_topology_damage_before_repair"))
        )

        added_ratio = float(metadata.get("added_ink_ratio", 0.0))
        removed_ratio = float(metadata.get("removed_ink_ratio", 0.0))
        changed_ratio = float(metadata.get("changed_ink_ratio", 0.0))

        soft_score_floor = max(
            0.0,
            float(self.settings.reconstruction_min_acceptance_score) - 0.12,
        )

        # Extreme rewrite protection stays active for everything.
        if added_ratio > 1.25:
            reasons.append("added_ink_ratio_extreme_rewrite")

        if removed_ratio > 0.90:
            reasons.append("removed_ink_ratio_extreme_rewrite")

        if changed_ratio > 1.50:
            reasons.append("changed_ink_ratio_extreme_rewrite")

        if is_provisional_line_parent:
            # Line removal is allowed to temporarily damage topology.
            # It is a parent hypothesis, not a finished repair.
            provisional_floor = 0.30
            line_confidence_floor = 0.30

            if score_info["score"] < provisional_floor:
                reasons.append("line_removal_score_below_provisional_floor")

            if float(metadata.get("line_artifact_confidence", 0.0)) < line_confidence_floor:
                reasons.append("line_artifact_confidence_too_low")

            return reasons

        # Normal topology rejection rules.
        if candidate_topology["component_count"] > original_topology["component_count"] + 2:
            reasons.append("component_count_increased_too_much")

        if (
            candidate_topology["junction_cluster_count"]
            > original_topology["junction_cluster_count"] + 4
        ):
            reasons.append("junction_count_increased_too_much")

        if (
            candidate_topology["short_path_count"]
            > original_topology["short_path_count"] + 6
        ):
            reasons.append("short_path_count_increased_too_much")

        if score_info["topology_gain"] < -0.35:
            reasons.append("topology_regressed_too_much")

        if score_info["score"] < soft_score_floor:
            reasons.append("score_below_acceptance_threshold")

        if (
            self.settings.reconstruction_use_recognition_verification
            and original_recognition.get("available")
            and candidate_recognition.get("available")
        ):
            confidence_gain = score_info["confidence_gain"]
            if confidence_gain < (
                float(self.settings.reconstruction_min_confidence_gain) - 0.05
            ):
                reasons.append("recognition_confidence_gain_too_low")

        return reasons

    def _debug_root(self, output_dir, stable_unit_id):
        stable = sanitize_identifier(stable_unit_id)
        return os.path.join(
            os.path.abspath(output_dir),
            "debug",
            "reconstruction",
            stable,
        )

    def _save_h0_debug(self, *, mask, output_dir, stable_unit_id):
        debug_dir = self._debug_root(output_dir, stable_unit_id)
        os.makedirs(debug_dir, exist_ok=True)

        h0_mask_path = os.path.join(debug_dir, "h0_original_mask.png")
        h0_visual_path = os.path.join(debug_dir, "h0_original_visual.png")
        h0_overlay_path = os.path.join(debug_dir, "h0_original_overlay.png")

        binary = _as_binary_mask(mask)
        cv2.imwrite(h0_mask_path, binary)
        cv2.imwrite(h0_visual_path, _visualize_binary_mask(binary))
        cv2.imwrite(h0_overlay_path, self._make_overlay(binary, binary))

        return {
            "debug_dir": debug_dir,
            "h0_mask_path": h0_mask_path,
            "h0_visual_path": h0_visual_path,
            "h0_overlay_path": h0_overlay_path,
        }

    def _save_candidate_debug(
        self,
        *,
        hypothesis,
        original_mask,
        candidate_trace,
        output_dir,
        stable_unit_id,
        selected=False,
    ):
        debug_dir = self._debug_root(output_dir, stable_unit_id)
        candidates_dir = os.path.join(debug_dir, "candidates")
        os.makedirs(candidates_dir, exist_ok=True)

        prefix = sanitize_identifier(hypothesis.hypothesis_id)

        candidate_mask_path = os.path.join(candidates_dir, f"{prefix}_mask.png")
        candidate_visual_path = os.path.join(candidates_dir, f"{prefix}_visual.png")
        added_mask_path = os.path.join(candidates_dir, f"{prefix}_added.png")
        removed_mask_path = os.path.join(candidates_dir, f"{prefix}_removed.png")
        overlay_path = os.path.join(candidates_dir, f"{prefix}_overlay.png")
        retrace_skeleton_path = os.path.join(
            candidates_dir,
            f"{prefix}_retrace_skeleton.png",
        )
        retrace_graph_path = os.path.join(
            candidates_dir,
            f"{prefix}_retrace_graph.png",
        )
        retrace_paths_path = os.path.join(
            candidates_dir,
            f"{prefix}_retrace_paths.png",
        )
        retrace_landmarks_path = os.path.join(
            candidates_dir,
            f"{prefix}_retrace_landmarks.png",
        )

        candidate_mask = _as_binary_mask(hypothesis.candidate_mask)
        added_mask = _as_binary_mask(hypothesis.added_mask)
        removed_mask = _as_binary_mask(hypothesis.removed_mask)
        reference_mask = _as_binary_mask(
            hypothesis.metadata.get("_debug_reference_mask", original_mask)
        )

        cv2.imwrite(candidate_mask_path, candidate_mask)
        cv2.imwrite(candidate_visual_path, _visualize_binary_mask(candidate_mask))
        cv2.imwrite(added_mask_path, added_mask)
        cv2.imwrite(removed_mask_path, removed_mask)
        cv2.imwrite(
            overlay_path,
            self._make_overlay(reference_mask, candidate_mask),
        )

        saved_retrace_paths = {
            "retrace_skeleton_path": None,
            "retrace_graph_path": None,
            "retrace_paths_path": None,
            "retrace_landmarks_path": None,
        }
        if self.settings.save_debug:
            debug_writer = TraceDebugWriter(self.settings)
            skeleton = candidate_trace["skeleton"]
            graph = candidate_trace["graph"]
            trace_paths = candidate_trace["paths"]
            landmarks = candidate_trace["result"].landmarks
            debug_writer.save_skeleton_debug(skeleton, retrace_skeleton_path)
            debug_writer.save_skeleton_graph_debug(
                skeleton,
                graph,
                retrace_graph_path,
            )
            debug_writer.save_trace_paths_debug(
                skeleton,
                trace_paths,
                retrace_paths_path,
            )
            debug_writer.save_landmarks_debug(
                skeleton,
                trace_paths,
                landmarks,
                retrace_landmarks_path,
            )
            saved_retrace_paths = {
                "retrace_skeleton_path": retrace_skeleton_path,
                "retrace_graph_path": retrace_graph_path,
                "retrace_paths_path": retrace_paths_path,
                "retrace_landmarks_path": retrace_landmarks_path,
            }

        return {
            "candidate_mask_path": candidate_mask_path,
            "candidate_visual_path": candidate_visual_path,
            "added_mask_path": added_mask_path,
            "removed_mask_path": removed_mask_path,
            "overlay_path": overlay_path,
            **saved_retrace_paths,
        }

    def _make_overlay(self, original_mask, candidate_mask):
        """White background, black retained ink, green additions, red removals."""
        original = _as_binary_mask(original_mask)
        candidate = _as_binary_mask(candidate_mask)
        added, removed = _changed_masks(original, candidate)

        overlay = np.full((original.shape[0], original.shape[1], 3), 255, dtype=np.uint8)

        retained = (original > 0) & (candidate > 0)
        candidate_only = (candidate > 0) & (added == 0) & (~retained)

        overlay[retained] = (0, 0, 0)
        overlay[candidate_only] = (0, 0, 0)

        # OpenCV uses BGR. Added = green, removed = red.
        overlay[added > 0] = (0, 180, 0)
        overlay[removed > 0] = (0, 0, 220)

        return overlay


ReconstructionController = TheoreticalReconstructor


__all__ = [
    "TheoreticalReconstructor",
    "ReconstructionController",
]
