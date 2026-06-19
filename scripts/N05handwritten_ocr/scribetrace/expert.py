"""Public facade and orchestrator for the N05 ScribeTrace expert."""

if __package__ in {None, ""}:
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(__file__)))
    __package__ = "scribetrace"

import json
import os

import cv2
import numpy as np

from .trace_common import (
    EXPERT_NAME,
    NEIGHBOR_OFFSETS,
    SUPPORTED_THRESHOLD_MODES,
    coordinate_key,
    sanitize_identifier,
)
from .trace_debug import TraceDebugWriter
from .trace_features import TraceFeatureEncoder
from .trace_inference import load_rf_model, predict_rf_candidates
from .trace_masks import (
    InkComponentExtractor,
    InkHoleDetector,
    TraceMaskAdapter,
    match_ink_holes_to_closed_paths,
)
from .trace_models import (
    BoundingBox,
    InkComponent,
    InkHole,
    PixelPoint,
    SkeletonPoint,
    TraceFeatureVector,
    TraceInput,
    TraceLandmark,
    TracePath,
    TraceResult,
)
from .trace_paths import PathSignal, TraceLandmarkExtractor, TracePathExtractor
from .trace_settings import TraceSettings, normalize_trace_settings
from .trace_skeleton import SkeletonGraph, SkeletonPointExtractor, TraceSkeletonizer
from .trace_repair import TraceMaskRepairer
from .trace_reconstruction import TheoreticalReconstructor

try:
    from ..condition.condition_inference import predict_condition
    from ..condition.condition_router import route_condition
except ImportError:  # Allows direct local debugging when N05 root is on sys.path.
    from condition.condition_inference import predict_condition
    from condition.condition_router import route_condition

# Compatibility aliases retained for callers that imported private helpers.
_coordinate_key = coordinate_key
_sanitize_identifier = sanitize_identifier


def get_expert_manifest(settings=None):
    """Describe ScribeTrace through the shared N05 expert interface."""
    trace_settings = normalize_trace_settings(settings)
    return {
        "expert_name": EXPERT_NAME,
        "display_name": "ScribeTrace",
        "enabled": trace_settings.enabled,
        "implemented": True,
        "status": "geometric_evidence_ready",
        "unit_level": "text_unit",
        "returns_text": False,
    }


def recognize(crop_path, context=None, settings=None):
    """Run ScribeTrace and return the common N05 expert result shape."""
    trace_input = TraceInput.from_context(crop_path, context)
    trace_result = run_scribetrace(trace_input, settings=settings)
    evidence = trace_result.to_dict()

    try:
        evidence["rf_letter_candidates_for_unit"] = predict_rf_candidates(
            trace_result,
            top_k=5,
        )
    except Exception as error:
        evidence["rf_letter_candidates_for_unit"] = []
        evidence["rf_error"] = str(error)

    return {
        "expert_name": EXPERT_NAME,
        "attempted": trace_result.status in {"completed", "completed_limited"},
        "status": trace_result.status,
        "crop_path": crop_path,
        # The Random Forest is letter-level while this crop can be a word/unit.
        "candidates": [],
        "evidence": evidence,
        "error": trace_result.error,
    }


def _resolve_output_dir(trace_input, source_path):
    """Resolve outputs into the owning document's N05 folder."""
    if trace_input.output_dir:
        return os.path.abspath(trace_input.output_dir)

    normalized_source = os.path.abspath(source_path)
    marker = f"{os.sep}temp_processing{os.sep}"
    if marker in normalized_source:
        project_prefix, document_suffix = normalized_source.split(marker, 1)
        document_id = document_suffix.split(os.sep, 1)[0]
        return os.path.join(
            project_prefix,
            "temp_processing",
            document_id,
            "n05_handwritten_ocr",
            "scribetrace",
        )

    if trace_input.document_id:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        return os.path.join(
            project_root,
            "temp_processing",
            sanitize_identifier(trace_input.document_id),
            "n05_handwritten_ocr",
            "scribetrace",
        )

    return os.path.join(os.path.dirname(normalized_source), "n05_scribetrace")


def _debug_paths_for(output_dir, stable_unit_id):
    """Build all sanitized ScribeTrace debug output paths."""
    debug_dir = os.path.join(output_dir, "debug")
    prefix = sanitize_identifier(stable_unit_id)
    return {
        "component_debug_image": os.path.join(debug_dir, f"{prefix}_components_debug.png"),
        "skeleton_debug_image": os.path.join(debug_dir, f"{prefix}_skeleton_debug.png"),
        "skeleton_graph_debug_image": os.path.join(debug_dir, f"{prefix}_skeleton_graph_debug.png"),
        "trace_paths_debug_image": os.path.join(debug_dir, f"{prefix}_trace_paths_debug.png"),
        "landmarks_debug_image": os.path.join(debug_dir, f"{prefix}_landmarks_debug.png"),
    }


def _result_json_path_for(output_dir, stable_unit_id):
    """Build the sanitized machine-readable result path for one text unit."""
    prefix = sanitize_identifier(stable_unit_id)
    return os.path.join(output_dir, "metadata", f"{prefix}_scribetrace.json")


def save_trace_result_json(trace_result, output_path):
    """Save compact ScribeTrace evidence atomically as readable JSON."""
    output_path = os.path.abspath(output_path)
    temporary_path = f"{output_path}.tmp"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trace_result.result_json_path = output_path

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(trace_result.to_dict(), file, indent=2, ensure_ascii=False)
        file.write("\n")

    os.replace(temporary_path, output_path)
    return output_path


def _finalize_trace_result(trace_result, output_dir, stable_unit_id):
    """Persist an enabled trace result when JSON output is requested."""
    if trace_result.settings and trace_result.settings.save_json:
        save_trace_result_json(
            trace_result,
            _result_json_path_for(output_dir, stable_unit_id),
        )
    return trace_result


def _promote_selected_reconstruction_features(trace_result):
    """Make the selected reconstructed geometry the active ML feature vector."""
    reconstruction = trace_result.reconstruction or {}
    selected_source = reconstruction.get("selected_feature_source", "original")

    if selected_source != "reconstructed":
        reconstruction["active_feature_source"] = "original"
        reconstruction["active_feature_vector_promoted"] = False
        trace_result.metrics["active_feature_source"] = "original"
        return trace_result

    selected = reconstruction.get("selected_feature_vector")
    if not isinstance(selected, dict):
        raise ValueError(
            "Reconstruction selected reconstructed features without a feature vector."
        )

    feature_names = list(selected.get("feature_names") or [])
    vector = list(selected.get("vector") or [])
    if not feature_names or len(feature_names) != len(vector):
        raise ValueError(
            "Selected reconstruction feature names and values are missing or misaligned."
        )

    original_names = (
        trace_result.feature_vector.feature_names
        if trace_result.feature_vector is not None
        else []
    )
    if original_names and feature_names != original_names:
        raise ValueError(
            "Selected reconstruction feature schema differs from the original schema."
        )

    sequence = list(selected.get("sequence") or [])
    trace_result.feature_vector = TraceFeatureVector(
        vector=vector,
        feature_names=feature_names,
        sequence=sequence,
        sequence_string=(
            selected.get("sequence_string")
            or " ".join(str(token) for token in sequence)
        ),
    )
    reconstruction["active_feature_source"] = "reconstructed"
    reconstruction["active_feature_vector_promoted"] = True
    trace_result.metrics["active_feature_source"] = "reconstructed"
    return trace_result


def run_scribetrace(trace_input, settings=None):
    """Produce filtered component, skeleton, graph, path, and ML evidence."""
    if trace_input is None:
        return TraceResult(status="failed", error="TraceInput is missing.")

    try:
        trace_settings = normalize_trace_settings(settings)
    except Exception as error:
        return TraceResult(status="failed", trace_input=trace_input, error=str(error))

    if not trace_settings.enabled:
        return TraceResult(
            status="disabled",
            trace_input=trace_input,
            settings=trace_settings,
        )

    try:
        trace_input.validate()
        binary_mask, provenance = TraceMaskAdapter(trace_settings).resolve_trace_mask(
            trace_input
        )
        if trace_settings.enable_theoretical_reconstruction:
            repaired_mask = np.where(binary_mask > 128, 255, 0).astype(np.uint8)
            repair_metrics = {
                "enabled": False,
                "method": "delegated_to_theoretical_reconstruction",
                "original_ink_pixels": int(cv2.countNonZero(repaired_mask)),
                "repaired_ink_pixels": int(cv2.countNonZero(repaired_mask)),
                "added_pixels": 0,
            }
        else:
            repaired_mask, repair_metrics = TraceMaskRepairer(
                trace_settings
            ).repair(binary_mask)

        component_analysis = InkComponentExtractor(trace_settings).analyze_mask(
            repaired_mask
        )
        components = component_analysis["components"]
        ink_holes = InkHoleDetector(trace_settings).detect_holes(
            components=components,
            full_shape=component_analysis["cleaned_mask"].shape,
        )

        metrics = {
            key: value
            for key, value in component_analysis.items()
            if key not in {"components", "cleaned_mask"}
        }
        metrics.update(provenance)
        metrics["mask_repair"] = repair_metrics
        metrics["ink_hole_count"] = len(ink_holes)
        metrics["ink_hole_component_ids"] = sorted(
            {hole.component_id for hole in ink_holes}
        )

        debug_paths = {
            "component_debug_image": None,
            "skeleton_debug_image": None,
            "skeleton_graph_debug_image": None,
            "trace_paths_debug_image": None,
            "landmarks_debug_image": None,
        }
        output_dir = _resolve_output_dir(trace_input, provenance["source_path"])
        metrics["output_dir"] = output_dir
        planned_debug_paths = _debug_paths_for(
            output_dir,
            trace_input.stable_unit_id(),
        )
        debug_writer = TraceDebugWriter(trace_settings)

        if trace_settings.save_debug:
            debug_writer.save_component_debug(
                provenance["source_path"],
                components,
                planned_debug_paths["component_debug_image"],
            )
            debug_paths["component_debug_image"] = planned_debug_paths[
                "component_debug_image"
            ]

        if len(components) > trace_settings.maximum_component_count_for_full_trace:
            metrics.update(
                {
                    "skeleton_point_count": 0,
                    "skeleton_graph": None,
                    "raw_path_count": 0,
                    "path_count": 0,
                    "closed_loop_count": 0,
                    "short_path_count": 0,
                    "merged_path_count": 0,
                    "merged_source_path_count": 0,
                    "landmark_count": 0,
                    "landmark_type_counts": {},
                }
            )
            return _finalize_trace_result(
                TraceResult(
                    status="completed_limited",
                    reason="component_limit_exceeded",
                    trace_input=trace_input,
                    settings=trace_settings,
                    components=components,
                    trace_paths=[],
                    landmarks=[],
                    feature_vector=None,
                    debug_paths=debug_paths,
                    metrics=metrics,
                    ink_holes=ink_holes,
                ),
                output_dir,
                trace_input.stable_unit_id(),
            )

        skeleton_mask = TraceSkeletonizer(trace_settings).skeletonize(
            component_analysis["cleaned_mask"]
        )
        skeleton_points = SkeletonPointExtractor().extract_points(skeleton_mask)
        skeleton_graph = SkeletonGraph(skeleton_points)
        path_extractor = TracePathExtractor(trace_settings)
        trace_paths = path_extractor.extract_paths(skeleton_graph)
        ink_hole_matches = match_ink_holes_to_closed_paths(ink_holes, trace_paths)
        landmarks = TraceLandmarkExtractor(trace_settings).extract_landmarks(trace_paths)

        landmark_type_counts = {}
        for landmark in landmarks:
            landmark_type_counts[landmark.landmark_type] = (
                landmark_type_counts.get(landmark.landmark_type, 0) + 1
            )

        metrics.update(
            {
                "skeleton_point_count": len(skeleton_points),
                "skeleton_graph": skeleton_graph.to_dict(),
                **path_extractor.metrics,
                "path_lengths": [path.length() for path in trace_paths],
                "landmark_count": len(landmarks),
                "landmark_type_counts": landmark_type_counts,
                "ink_hole_match_count": len(ink_hole_matches),
                "unmatched_ink_hole_count": len(ink_holes) - len(ink_hole_matches),
                "ink_hole_matches": ink_hole_matches,
            }
        )
        feature_vector = TraceFeatureEncoder(trace_settings).encode(
            components=components,
            skeleton_graph=skeleton_graph,
            trace_paths=trace_paths,
            landmarks=landmarks,
            metrics=metrics,
            ink_holes=ink_holes,
        )

        if trace_settings.save_debug:
            debug_writer.save_skeleton_debug(
                skeleton_mask,
                planned_debug_paths["skeleton_debug_image"],
            )
            debug_writer.save_skeleton_graph_debug(
                skeleton_mask,
                skeleton_graph,
                planned_debug_paths["skeleton_graph_debug_image"],
            )
            debug_writer.save_trace_paths_debug(
                skeleton_mask,
                trace_paths,
                planned_debug_paths["trace_paths_debug_image"],
            )
            debug_writer.save_landmarks_debug(
                skeleton_mask,
                trace_paths,
                landmarks,
                planned_debug_paths["landmarks_debug_image"],
            )
            debug_paths.update(planned_debug_paths)

        trace_result = TraceResult(
            status="completed",
            trace_input=trace_input,
            settings=trace_settings,
            components=components,
            trace_paths=trace_paths,
            debug_paths=debug_paths,
            metrics=metrics,
            landmarks=landmarks,
            feature_vector=feature_vector,
            ink_holes=ink_holes,
        )

        condition_verdict = predict_condition(
            trace_result=trace_result,
            known_damage_recipes=getattr(trace_input, "known_damage_recipes", None),
        )
        routing_advice = route_condition(condition_verdict)
        trace_result.condition_verdict = condition_verdict.to_dict()
        trace_result.routing_advice = routing_advice.to_dict()
        metrics["condition"] = trace_result.condition_verdict
        metrics["routing_advice"] = trace_result.routing_advice

        trace_result.reconstruction = TheoreticalReconstructor(
            trace_settings
        ).run(
            mask=component_analysis["cleaned_mask"],
            original_result=trace_result,
            output_dir=output_dir,
            stable_unit_id=trace_input.stable_unit_id(),
            damage_verdict=trace_result.condition_verdict,
            allowed_defenses=routing_advice.scribetrace_allowed_defenses,
            known_damage_recipes=getattr(trace_input, "known_damage_recipes", None),
        )
        metrics["reconstruction"] = {
            "enabled": trace_result.reconstruction.get("enabled", False),
            "status": trace_result.reconstruction.get("status"),
            "candidate_count": trace_result.reconstruction.get(
                "candidate_count",
                0,
            ),
            "accepted_count": trace_result.reconstruction.get(
                "accepted_count",
                0,
            ),
        }
        _promote_selected_reconstruction_features(trace_result)
        return _finalize_trace_result(
            trace_result,
            output_dir,
            trace_input.stable_unit_id(),
        )
    except Exception as error:
        return TraceResult(
            status="failed",
            trace_input=trace_input,
            settings=trace_settings,
            error=str(error),
        )


__all__ = [
    "EXPERT_NAME",
    "NEIGHBOR_OFFSETS",
    "SUPPORTED_THRESHOLD_MODES",
    "BoundingBox",
    "InkComponent",
    "InkComponentExtractor",
    "InkHole",
    "InkHoleDetector",
    "PathSignal",
    "PixelPoint",
    "SkeletonGraph",
    "SkeletonPoint",
    "SkeletonPointExtractor",
    "TraceDebugWriter",
    "TraceFeatureEncoder",
    "TraceFeatureVector",
    "TraceInput",
    "TraceLandmark",
    "TraceLandmarkExtractor",
    "TraceMaskAdapter",
    "TracePath",
    "TracePathExtractor",
    "TraceResult",
    "TraceSettings",
    "TraceSkeletonizer",
    "TheoreticalReconstructor",
    "get_expert_manifest",
    "load_rf_model",
    "match_ink_holes_to_closed_paths",
    "normalize_trace_settings",
    "predict_rf_candidates",
    "recognize",
    "run_scribetrace",
    "save_trace_result_json",
]


if __name__ == "__main__":
    sample_input = TraceInput(
        mask_crop_path=(
            "/home/vahram/Desktop/image_Processor/temp_processing/test_1/"
            "n02_crop_refiner/crops/blue/analysis_mask/"
            "blue_0002_blue_0002_analysis_mask.png"
        ),
        document_id="test_doc",
        text_unit_id="test_unit_001",
        layer="blue",
    )
    sample_result = run_scribetrace(
        sample_input,
        settings={
            "enabled": True,
            "save_debug": True,
            "minimum_ink_pixels": 4,
            "minimum_trace_path_points": 4,
        },
    )
    print("ScribeTrace status:", sample_result.status)
    print("Components:", sample_result.component_count())
    print("Paths:", len(sample_result.trace_paths))
    print("Landmarks:", len(sample_result.landmarks))
    if sample_result.result_json_path:
        print("Result JSON:", sample_result.result_json_path)
    if sample_result.error:
        print("Error:", sample_result.error)
