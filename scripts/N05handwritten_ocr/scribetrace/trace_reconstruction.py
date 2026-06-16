"""Bounded theoretical reconstruction for damaged handwritten topology."""

import math
import os

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
        if 0 <= point.y < distance.shape[0] and 0 <= point.x < distance.shape[1]
    ]
    radius = sum(radii) / len(radii) if radii else 1.0
    return max(1, min(4, int(round(radius * 1.5))))


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


def _topology_gain(original, candidate, different_components):
    """Score whether the bridge reduces damage without inventing complexity."""
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
        candidate["junction_cluster_count"]
        - original["junction_cluster_count"],
    )
    new_short_paths = max(
        0,
        candidate["short_path_count"] - original["short_path_count"],
    )

    gain = (
        0.35 * min(1, component_gain)
        + 0.30 * min(2, endpoint_gain) / 2.0
        + 0.15 * min(1, isolated_gain)
        + 0.20 * min(1, loop_gain)
        - 0.25 * min(1, new_junctions)
        - 0.15 * min(1, new_short_paths)
    )
    if different_components and component_gain == 0:
        gain -= 0.30
    return max(-1.0, min(1.0, gain))



# First split routing map. The current truth zip only has endpoint-bridge
# reconstruction implemented in this file. Other names are preserved so the
# universal condition router can already speak the final language.
DAMAGE_TO_RECONSTRUCTION_DEFENSES = {
    "clean": [],
    "light_cut": ["horizontal_gap_closing", "vertical_gap_closing", "endpoint_bridge"],
    "light_blur": ["threshold_normalization"],
    "scanner_noise": ["component_denoising", "median_denoising"],
    "light_erosion": [
        "conservative_stroke_recovery",
        "endpoint_bridge",
        "horizontal_gap_closing",
        "vertical_gap_closing",
    ],
    "ink_overlap": ["contamination_opening"],
    "stamp_interference": ["linear_artifact_removal", "contamination_opening"],
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
    "compression_artifacts": ["threshold_normalization", "median_denoising"],
}

IMPLEMENTED_RECONSTRUCTION_DEFENSES = {"endpoint_bridge"}


class TheoreticalReconstructor:
    """Generate, retrace, verify, and rank routed reconstruction hypotheses."""

    def __init__(self, settings=None):
        self.settings = normalize_trace_settings(settings)

    def diagnose(self, mask, trace_result):
        """Describe topology damage signals without modifying the mask."""
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
        return {
            "damage_suspected": bool(reasons),
            "damage_reasons": reasons,
            "topology": topology,
            "crossing_number_histogram": graph_summary.get(
                "crossing_number_histogram",
                {},
            ),
            "border_contacts": border_contacts,
        }

    def _bridge_candidates(self, graph, trace_paths, mask):
        """Rank endpoint pairs whose continuation tangents face each other."""
        endpoints = graph.endpoints()
        component_lookup = _graph_component_lookup(graph)
        candidates = []
        for first_index, point_a in enumerate(endpoints):
            tangent_a = _endpoint_tangent(
                point_a,
                trace_paths,
                self.settings.reconstruction_tangent_points,
            )
            if tangent_a is None:
                continue
            for point_b in endpoints[first_index + 1:]:
                distance = math.hypot(
                    point_b.x - point_a.x,
                    point_b.y - point_a.y,
                )
                if not (
                    self.settings.reconstruction_min_endpoint_separation_px
                    <= distance
                    <= self.settings.reconstruction_max_bridge_length_px
                ):
                    continue
                tangent_b = _endpoint_tangent(
                    point_b,
                    trace_paths,
                    self.settings.reconstruction_tangent_points,
                )
                if tangent_b is None:
                    continue
                bridge_ab = (point_b.x - point_a.x, point_b.y - point_a.y)
                bridge_ba = (-bridge_ab[0], -bridge_ab[1])
                angle_a = _angle_degrees(tangent_a[1:], bridge_ab)
                angle_b = _angle_degrees(tangent_b[1:], bridge_ba)
                maximum_angle = max(angle_a, angle_b)
                if (
                    maximum_angle
                    > self.settings.reconstruction_max_bridge_angle_degrees
                ):
                    continue

                line_coordinates = _line_coordinates(point_a, point_b)
                endpoint_clearance = min(
                    5,
                    max(1, int(round(len(line_coordinates) * 0.25))),
                )
                interior = line_coordinates[
                    endpoint_clearance:-endpoint_clearance
                ]
                occupied = sum(mask[y, x] > 0 for x, y in interior)
                occupied_ratio = occupied / max(1, len(interior))
                if occupied_ratio > 0.25:
                    continue

                different_components = (
                    component_lookup.get(point_a.to_tuple())
                    != component_lookup.get(point_b.to_tuple())
                )
                length_score = 1.0 - (
                    distance
                    / self.settings.reconstruction_max_bridge_length_px
                )
                angle_score = 1.0 - (
                    maximum_angle
                    / max(
                        1.0,
                        self.settings.reconstruction_max_bridge_angle_degrees,
                    )
                )
                geometry_score = (
                    0.55 * angle_score
                    + 0.35 * length_score
                    + 0.10 * (1.0 - occupied_ratio)
                )
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
                        "different_components": different_components,
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
        return candidates[: self.settings.reconstruction_max_hypotheses]

    def _reconstruct_bridge(self, mask, candidate):
        """Draw one minimal bridge onto a copy of the original evidence."""
        reconstructed = np.asarray(mask).copy()
        point_a = candidate["point_a"]
        point_b = candidate["point_b"]
        thickness = _estimate_bridge_thickness(mask, point_a, point_b)
        cv2.line(
            reconstructed,
            point_a.to_tuple(),
            point_b.to_tuple(),
            255,
            thickness,
            cv2.LINE_AA,
        )
        reconstructed = np.where(reconstructed > 127, 255, 0).astype(np.uint8)
        added_mask = cv2.bitwise_and(
            reconstructed,
            cv2.bitwise_not(np.asarray(mask)),
        )
        removed_mask = cv2.bitwise_and(
            np.asarray(mask),
            cv2.bitwise_not(reconstructed),
        )
        return reconstructed, added_mask, removed_mask, thickness

    def _normalize_defense_names(self, values):
        """Return clean, de-duplicated defense names preserving order."""
        names = []
        for value in values or []:
            name = str(value).strip()
            if name and name not in names:
                names.append(name)
        return names

    def _normalize_damage_verdict(self, damage_verdict):
        """Return a JSON-safe damage verdict dictionary."""
        if damage_verdict is None:
            return None
        if hasattr(damage_verdict, "to_dict"):
            return damage_verdict.to_dict()
        if isinstance(damage_verdict, dict):
            return dict(damage_verdict)
        return {"primary_damage": str(damage_verdict), "source": "raw_value"}

    def _normalize_recipe_chain(self, known_damage_recipes):
        """Normalize oracle recipe names from TraceInput or caller."""
        if not known_damage_recipes:
            return []
        if isinstance(known_damage_recipes, str):
            return [known_damage_recipes]
        return [str(item) for item in known_damage_recipes if item]

    def _route_defenses(
        self,
        allowed_defenses=None,
        known_damage_recipes=None,
        damage_verdict=None,
    ):
        """Resolve allowed reconstruction tools without trying everything."""
        explicit = self._normalize_defense_names(allowed_defenses)
        if explicit:
            return explicit, {
                "source": "explicit_allowed_defenses",
                "damage_labels": [],
                "requested_defenses": explicit,
            }

        recipe_chain = self._normalize_recipe_chain(known_damage_recipes)
        if recipe_chain:
            requested = []
            for recipe in recipe_chain:
                requested.extend(
                    DAMAGE_TO_RECONSTRUCTION_DEFENSES.get(recipe, [])
                )
            requested = self._normalize_defense_names(requested)
            return requested, {
                "source": "known_damage_recipes",
                "damage_labels": recipe_chain,
                "requested_defenses": requested,
            }

        verdict = self._normalize_damage_verdict(damage_verdict)
        if verdict:
            label = (
                verdict.get("primary_damage")
                or verdict.get("label")
                or verdict.get("damage")
            )
            if label:
                requested = self._normalize_defense_names(
                    DAMAGE_TO_RECONSTRUCTION_DEFENSES.get(label, [])
                )
                return requested, {
                    "source": "damage_verdict",
                    "damage_labels": [label],
                    "requested_defenses": requested,
                }

        return [], {
            "source": "no_route",
            "damage_labels": [],
            "requested_defenses": [],
        }

    def _base_result(
        self,
        original_result,
        diagnosis,
        original_topology,
        original_recognition,
        damage_verdict,
        allowed_defense_types,
        routing,
        status,
        enabled=True,
    ):
        """Create a stable reconstruction result with h0 selected by default."""
        return {
            "version": "scribetrace_reconstruction_v2",
            "enabled": bool(enabled),
            "status": status,
            "cycle": [
                "condition_route",
                "hypothesize",
                "reconstruct",
                "retrace",
                "verify",
                "accept_or_keep_original",
            ],
            "damage_verdict": self._normalize_damage_verdict(damage_verdict),
            "routing": routing,
            "allowed_defense_types": list(allowed_defense_types or []),
            "implemented_defense_types": sorted(IMPLEMENTED_RECONSTRUCTION_DEFENSES),
            "unsupported_defense_types": [
                defense
                for defense in allowed_defense_types or []
                if defense not in IMPLEMENTED_RECONSTRUCTION_DEFENSES
            ],
            "selected_hypothesis_id": "h0_original",
            "selected_feature_source": "original",
            "selected_feature_vector": None,
            "selected_reconstructed_mask_path": None,
            "original_hypothesis": {
                "hypothesis_id": "h0_original",
                "type": "no_repair",
                "accepted": True,
                "selected": True,
                "topology": original_topology,
                "recognition": original_recognition,
            },
            "diagnosis": diagnosis,
            "topology_diagnosis": diagnosis,
            "hypotheses": [],
            "candidate_count": 0,
            "total_candidate_count": 0,
            "accepted_hypothesis_ids": [],
            "accepted_count": 0,
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
        """Execute routed reconstruction and keep h0_original if unsafe."""
        original_topology = _topology_snapshot(original_result)
        original_recognition = _recognition_snapshot(original_result)
        disabled_diagnosis = {
            "damage_suspected": False,
            "damage_reasons": [],
            "status": "not_evaluated",
            "topology": original_topology,
        }
        if not self.settings.enable_theoretical_reconstruction:
            return self._base_result(
                original_result=original_result,
                diagnosis=disabled_diagnosis,
                original_topology=original_topology,
                original_recognition={
                    "available": False,
                    "reason": "reconstruction_disabled",
                },
                damage_verdict=damage_verdict,
                allowed_defense_types=[],
                routing={"source": "disabled", "requested_defenses": []},
                status="disabled",
                enabled=False,
            )

        diagnosis = self.diagnose(mask, original_result)
        allowed_defense_types, routing = self._route_defenses(
            allowed_defenses=allowed_defenses,
            known_damage_recipes=known_damage_recipes,
            damage_verdict=damage_verdict,
        )
        result = self._base_result(
            original_result=original_result,
            diagnosis=diagnosis,
            original_topology=original_topology,
            original_recognition=original_recognition,
            damage_verdict=damage_verdict,
            allowed_defense_types=allowed_defense_types,
            routing=routing,
            status="initialized",
            enabled=True,
        )

        if not allowed_defense_types:
            result["status"] = "completed_no_routed_defenses"
            return result

        implemented = [
            defense
            for defense in allowed_defense_types
            if defense in IMPLEMENTED_RECONSTRUCTION_DEFENSES
        ]
        if not implemented:
            result["status"] = "completed_no_implemented_defenses"
            return result

        if not diagnosis["damage_suspected"]:
            result["status"] = "completed_no_topology_damage"
            return result

        points = SkeletonPointExtractor().extract_points(
            TraceSkeletonizer(self.settings).skeletonize(mask)
        )
        graph = SkeletonGraph(points)
        paths = TracePathExtractor(self.settings).extract_paths(graph)
        bridge_candidates = []
        if "endpoint_bridge" in implemented:
            bridge_candidates = self._bridge_candidates(graph, paths, mask)

        result["status"] = "completed"
        result["candidate_count"] = len(bridge_candidates)
        result["total_candidate_count"] = len(bridge_candidates)
        reconstruction_dir = os.path.join(output_dir, "reconstruction")
        os.makedirs(reconstruction_dir, exist_ok=True)
        safe_id = sanitize_identifier(stable_unit_id)
        original_ink = max(1, int(np.count_nonzero(mask)))

        for index, candidate in enumerate(bridge_candidates, start=1):
            hypothesis_id = f"h{index}_endpoint_bridge"
            reconstructed, added_mask, removed_mask, thickness = self._reconstruct_bridge(
                mask,
                candidate,
            )
            added_pixels = int(np.count_nonzero(added_mask))
            removed_pixels = int(np.count_nonzero(removed_mask))
            added_ratio = float(added_pixels / original_ink)
            removed_ratio = float(removed_pixels / original_ink)
            changed_ratio = float((added_pixels + removed_pixels) / original_ink)
            traced = _trace_mask(reconstructed, self.settings)
            candidate_result = traced["result"]
            candidate_topology = _topology_snapshot(candidate_result)
            candidate_recognition = _recognition_snapshot(candidate_result)
            topology_gain = _topology_gain(
                original_topology,
                candidate_topology,
                candidate["different_components"],
            )
            confidence_gain = (
                candidate_recognition["top1_confidence"]
                - original_recognition["top1_confidence"]
                if (
                    candidate_recognition["available"]
                    and original_recognition["available"]
                )
                else 0.0
            )
            confidence_score = max(0.0, min(1.0, 0.5 + confidence_gain))
            intervention_score = max(
                0.0,
                1.0
                - changed_ratio
                / max(
                    1e-6,
                    self.settings.reconstruction_max_added_ink_ratio,
                ),
            )
            geometry_score = (
                0.80 * candidate["geometry_score"]
                + 0.20 * intervention_score
            )
            topology_score = max(0.0, min(1.0, 0.5 + topology_gain))
            acceptance_score = (
                self.settings.reconstruction_topology_weight * topology_score
                + self.settings.reconstruction_geometry_weight * geometry_score
                + self.settings.reconstruction_confidence_weight
                * confidence_score
            )
            rejection_reasons = []
            if topology_gain < self.settings.reconstruction_min_topology_gain:
                rejection_reasons.append("insufficient_topology_gain")
            if (
                acceptance_score
                < self.settings.reconstruction_min_acceptance_score
            ):
                rejection_reasons.append("insufficient_acceptance_score")
            if added_ratio > self.settings.reconstruction_max_added_ink_ratio:
                rejection_reasons.append("synthetic_ink_budget_exceeded")
            recognition_verification_required = (
                self.settings.reconstruction_use_recognition_verification
                and original_recognition["available"]
                and candidate_recognition["available"]
            )
            if (
                recognition_verification_required
                and confidence_gain
                < self.settings.reconstruction_min_confidence_gain
            ):
                rejection_reasons.append("recognition_confidence_regressed")
            accepted = not rejection_reasons
            mask_path = os.path.join(
                reconstruction_dir,
                f"{safe_id}_{hypothesis_id}_mask.png",
            )
            added_path = os.path.join(
                reconstruction_dir,
                f"{safe_id}_{hypothesis_id}_added.png",
            )
            removed_path = os.path.join(
                reconstruction_dir,
                f"{safe_id}_{hypothesis_id}_removed.png",
            )
            overlay_path = os.path.join(
                reconstruction_dir,
                f"{safe_id}_{hypothesis_id}_overlay.png",
            )
            if self.settings.save_debug:
                cv2.imwrite(mask_path, reconstructed)
                cv2.imwrite(added_path, added_mask)
                cv2.imwrite(removed_path, removed_mask)
                overlay = cv2.cvtColor(np.asarray(mask), cv2.COLOR_GRAY2BGR)
                overlay[added_mask > 0] = (70, 230, 70)
                overlay[removed_mask > 0] = (70, 70, 230)
                cv2.imwrite(overlay_path, overlay)
            else:
                mask_path = None
                added_path = None
                removed_path = None
                overlay_path = None

            selected_feature_vector = (
                candidate_result.feature_vector.to_dict()
                if candidate_result.feature_vector is not None
                else None
            )
            hypothesis = {
                "hypothesis_id": hypothesis_id,
                "type": "endpoint_bridge",
                "accepted": accepted,
                "selected": False,
                "rejection_reasons": rejection_reasons,
                "acceptance_score": float(acceptance_score),
                "topology_gain": float(topology_gain),
                "confidence_gain": float(confidence_gain),
                "geometry_score": float(geometry_score),
                "added_pixels": added_pixels,
                "removed_pixels": removed_pixels,
                "added_ink_ratio": added_ratio,
                "removed_ink_ratio": removed_ratio,
                "changed_ink_ratio": changed_ratio,
                "bridge": {
                    "from": candidate["point_a"].to_dict(),
                    "to": candidate["point_b"].to_dict(),
                    "distance": candidate["distance"],
                    "thickness": thickness,
                    "path_id_a": candidate["path_id_a"],
                    "path_id_b": candidate["path_id_b"],
                    "angle_a": candidate["angle_a"],
                    "angle_b": candidate["angle_b"],
                    "different_components": candidate[
                        "different_components"
                    ],
                },
                "original_topology": original_topology,
                "reconstructed_topology": candidate_topology,
                "recognition": candidate_recognition,
                "selected_feature_vector": selected_feature_vector,
                "reconstructed_mask_path": mask_path,
                "added_ink_mask_path": added_path,
                "removed_ink_mask_path": removed_path,
                "reconstruction_overlay_path": overlay_path,
            }
            result["hypotheses"].append(hypothesis)

        result["hypotheses"].sort(
            key=lambda item: (
                not item["accepted"],
                -item["acceptance_score"],
                item["added_pixels"],
                item["hypothesis_id"],
            )
        )
        accepted = [
            item
            for item in result["hypotheses"]
            if item["accepted"]
        ][: self.settings.reconstruction_max_accepted]
        accepted_ids = {item["hypothesis_id"] for item in accepted}
        for hypothesis in result["hypotheses"]:
            hypothesis["accepted"] = hypothesis["hypothesis_id"] in accepted_ids
            hypothesis["selected"] = False
        result["accepted_hypothesis_ids"] = [
            item["hypothesis_id"] for item in accepted
        ]
        result["accepted_count"] = len(accepted)
        if accepted:
            selected = accepted[0]
            selected["selected"] = True
            result["selected_hypothesis_id"] = selected["hypothesis_id"]
            result["selected_feature_source"] = "reconstructed"
            result["selected_feature_vector"] = selected.get("selected_feature_vector")
            result["selected_reconstructed_mask_path"] = selected.get(
                "reconstructed_mask_path"
            )
        return result


ReconstructionController = TheoreticalReconstructor

__all__ = ["TheoreticalReconstructor", "ReconstructionController"]