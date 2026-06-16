"""Feature extraction for universal damage/condition models."""


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_condition_features(trace_result):
    """
    Convert a ScribeTrace result into condition-model features.

    These are not letter-recognition targets. They are condition/topology
    evidence used to identify clean vs damaged vs uncertain crops.
    """
    metrics = trace_result.metrics or {}
    graph = metrics.get("skeleton_graph") or {}
    components = trace_result.components or []
    ink_holes = trace_result.ink_holes or []

    features = {
        "component_count": float(len(components)),
        "path_count": float(len(trace_result.trace_paths or [])),
        "landmark_count": float(len(trace_result.landmarks or [])),
        "ink_hole_count": float(len(ink_holes)),
        "endpoint_count": _safe_float(graph.get("endpoint_count")),
        "junction_cluster_count": _safe_float(graph.get("junction_cluster_count")),
        "isolated_point_count": _safe_float(graph.get("isolated_point_count")),
        "short_path_count": _safe_float(metrics.get("short_path_count")),
        "closed_loop_count": _safe_float(metrics.get("closed_loop_count")),
        "skeleton_point_count": _safe_float(metrics.get("skeleton_point_count")),
        "total_ink_pixels": _safe_float(metrics.get("total_ink_pixels")),
        "ink_bbox_fill_ratio": _safe_float(metrics.get("ink_bbox_fill_ratio")),
        "ink_bbox_aspect_ratio": _safe_float(metrics.get("ink_bbox_aspect_ratio")),
        "fallback_used": 1.0 if metrics.get("fallback_used", False) else 0.0,
        "component_limit_exceeded": 1.0 if trace_result.reason == "component_limit_exceeded" else 0.0,
    }

    # Border contact can indicate edge crop loss / clipping.
    border_contacts = metrics.get("border_contacts") or {}
    for side in ("left", "right", "top", "bottom"):
        features[f"border_contact_{side}"] = 1.0 if border_contacts.get(side, False) else 0.0

    if components:
        areas = [component.area() for component in components]
        point_counts = [component.point_count() for component in components]
        features["max_component_area"] = float(max(areas))
        features["mean_component_area"] = float(sum(areas) / len(areas))
        features["max_component_points"] = float(max(point_counts))
        features["mean_component_points"] = float(sum(point_counts) / len(point_counts))
    else:
        features.update(
            {
                "max_component_area": 0.0,
                "mean_component_area": 0.0,
                "max_component_points": 0.0,
                "mean_component_points": 0.0,
            }
        )

    return features