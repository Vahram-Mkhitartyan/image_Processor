"""Materialize ScribeTrace-validated character crop hypotheses for N05."""

import os
import re

import cv2
import numpy as np

try:
    from .scribetrace.trace_segmentation import propose_trace_validated_cuts
except ImportError:
    from scribetrace.trace_segmentation import propose_trace_validated_cuts


WIDE_ASPECT_RATIO = 1.60
MANY_COMPONENTS_THRESHOLD = 4


def _sanitize_identifier(value):
    """Return a stable identifier safe for generated segment filenames."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown_unit"))
    return cleaned.strip("._") or "unknown_unit"


def _first_existing_path(record, keys):
    """Return the first configured path that currently exists."""
    for key in keys:
        value = record.get(key)
        if value and os.path.isfile(value):
            return os.path.abspath(value)
    return None


def _resolve_unit_id(record):
    """Resolve the most stable available identity for one N05 unit."""
    for key in ("text_unit_id", "group_id", "source_group_id"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return "unknown_unit"


def _resolve_output_dirs(folders):
    """Resolve and create proposer artifact directories."""
    if isinstance(folders, str):
        root = os.path.abspath(folders)
        proposer_dir = os.path.join(root, "character_unit_proposer")
        segments_dir = os.path.join(proposer_dir, "segments")
        debug_dir = os.path.join(proposer_dir, "debug")
    elif isinstance(folders, dict):
        root = os.path.abspath(
            folders.get("root")
            or folders.get("output_dir")
            or folders.get("n05_output_dir")
            or "."
        )
        proposer_dir = os.path.abspath(
            folders.get("character_unit_proposer")
            or os.path.join(root, "character_unit_proposer")
        )
        segments_dir = os.path.abspath(
            folders.get("character_unit_segments")
            or os.path.join(proposer_dir, "segments")
        )
        debug_dir = os.path.abspath(
            folders.get("character_unit_debug")
            or os.path.join(proposer_dir, "debug")
        )
    else:
        raise ValueError("folders must be an N05 folder dictionary or output path.")

    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    return proposer_dir, segments_dir, debug_dir


def _load_visual(path):
    """Load a visual crop when available."""
    return cv2.imread(path, cv2.IMREAD_COLOR) if path else None


def _normalize_analysis_mask(mask):
    """Normalize an analysis mask to white ink on black."""
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def _load_proposal_mask(mask_path, visual):
    """Load the exact analysis mask or derive a visual Otsu fallback."""
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return _normalize_analysis_mask(mask), "analysis_mask"

    if visual is None:
        return None, "unavailable"

    grayscale = cv2.cvtColor(visual, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(
        grayscale,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    return mask, "visual_otsu_fallback"


def _border_diagnostics(mask):
    """Measure border contact without changing the source mask."""
    ink = mask > 0
    height, width = mask.shape[:2]
    return {
        "touches_left_border": bool(np.any(ink[:, 0])) if width else False,
        "touches_right_border": bool(np.any(ink[:, -1])) if width else False,
        "touches_top_border": bool(np.any(ink[0, :])) if height else False,
        "touches_bottom_border": bool(np.any(ink[-1, :])) if height else False,
    }


def _recovery_reasons(diagnostics):
    """Translate geometry into non-rejecting recovery flags."""
    reasons = [
        key
        for key in (
            "touches_left_border",
            "touches_right_border",
            "touches_top_border",
            "touches_bottom_border",
        )
        if diagnostics.get(key)
    ]
    if diagnostics["aspect_ratio"] >= WIDE_ASPECT_RATIO:
        reasons.append("wide_unit_possible_multi_letter")
    if diagnostics["connected_component_count"] >= MANY_COMPONENTS_THRESHOLD:
        reasons.append("many_components")
    return reasons


def _save_image(path, image):
    """Save one generated artifact or raise a precise error."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"Could not save character-unit artifact: {path}")
    return os.path.abspath(path)


def _content_bbox(mask, x1, x2, padding):
    """Return a padded content box inside one proposed horizontal span."""
    segment = mask[:, x1:x2]
    coordinates = np.argwhere(segment > 0)
    if coordinates.size == 0:
        return x1, 0, x2, mask.shape[0]

    local_y1, local_x1 = coordinates.min(axis=0)
    local_y2, local_x2 = coordinates.max(axis=0) + 1
    return (
        max(x1, x1 + int(local_x1) - padding),
        max(0, int(local_y1) - padding),
        min(x2, x1 + int(local_x2) + padding),
        min(mask.shape[0], int(local_y2) + padding),
    )


def _visual_crop(visual, mask_shape, bbox, mask_crop):
    """Crop visual evidence in coordinates corresponding to the mask box."""
    x1, y1, x2, y2 = bbox
    if visual is None:
        rendered = np.full((*mask_crop.shape, 3), 255, dtype=np.uint8)
        rendered[mask_crop > 0] = (0, 0, 0)
        return rendered

    mask_height, mask_width = mask_shape
    visual_height, visual_width = visual.shape[:2]
    visual_x1 = int(round(x1 * visual_width / max(1, mask_width)))
    visual_x2 = int(round(x2 * visual_width / max(1, mask_width)))
    visual_y1 = int(round(y1 * visual_height / max(1, mask_height)))
    visual_y2 = int(round(y2 * visual_height / max(1, mask_height)))
    visual_x1 = max(0, min(visual_x1, visual_width - 1))
    visual_x2 = max(visual_x1 + 1, min(visual_x2, visual_width))
    visual_y1 = max(0, min(visual_y1, visual_height - 1))
    visual_y2 = max(visual_y1 + 1, min(visual_y2, visual_height))
    crop = visual[visual_y1:visual_y2, visual_x1:visual_x2]
    target_height, target_width = mask_crop.shape[:2]
    if crop.shape[:2] != (target_height, target_width):
        crop = cv2.resize(
            crop,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
    return crop


def _save_sequence_segments(
    unit_id,
    candidates,
    mask,
    visual,
    segments_dir,
    crop_padding_px,
    split_overlap_px,
):
    """Materialize one ordered character sequence from validated boundaries."""
    height, width = mask.shape[:2]
    cut_positions = sorted({candidate["cut_x"] for candidate in candidates})
    boundaries = [0, *cut_positions, width]
    hypothesis_id = "h1_vector_supported_sequence"
    segments = []

    for segment_index, (left_boundary, right_boundary) in enumerate(
        zip(boundaries, boundaries[1:])
    ):
        span_x1 = (
            left_boundary
            if segment_index == 0
            else max(0, left_boundary - split_overlap_px)
        )
        span_x2 = (
            right_boundary
            if segment_index == len(boundaries) - 2
            else min(width, right_boundary + split_overlap_px)
        )
        bbox = _content_bbox(mask, span_x1, span_x2, crop_padding_px)
        x1, y1, x2, y2 = bbox
        mask_crop = mask[y1:y2, x1:x2]
        visual_crop = _visual_crop(
            visual,
            mask.shape[:2],
            bbox,
            mask_crop,
        )
        segment_id = f"h1_s{segment_index}"
        prefix = (
            f"{_sanitize_identifier(unit_id)}_"
            f"{hypothesis_id}_{segment_id}"
        )
        mask_path = _save_image(
            os.path.join(segments_dir, f"{prefix}_mask.png"),
            mask_crop,
        )
        visual_path = _save_image(
            os.path.join(segments_dir, f"{prefix}_visual.png"),
            visual_crop,
        )
        segments.append(
            {
                "segment_id": segment_id,
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                },
                "mask_crop_path": mask_path,
                "visual_crop_path": visual_path,
                "role": "character_candidate",
                "left_boundary_x": int(left_boundary),
                "right_boundary_x": int(right_boundary),
            }
        )
    return hypothesis_id, segments, cut_positions


def _save_split_debug(unit_id, mask, skeleton, candidates, debug_dir):
    """Render accepted boundaries over mask and skeleton evidence."""
    canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    canvas[skeleton > 0] = (255, 170, 0)
    for candidate in sorted(candidates, key=lambda item: item["cut_x"]):
        x = candidate["cut_x"]
        vector_split = candidate.get("vector_split", {})
        connector_path_id = vector_split.get("connector_path_id")
        if connector_path_id is None:
            color = (0, 210, 255)
            label = "G"
        else:
            color = (70, 230, 70)
            label = (
                f"P{connector_path_id}:"
                f"{vector_split.get('split_after_point_index', '?')}"
            )
        cv2.line(canvas, (x, 0), (x, mask.shape[0] - 1), color, 1)
        cv2.putText(
            canvas,
            label,
            (max(0, x + 2), 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            color,
            1,
            cv2.LINE_AA,
        )
    path = os.path.join(
        debug_dir,
        f"{_sanitize_identifier(unit_id)}_character_splits.png",
    )
    return _save_image(path, canvas)


def propose_character_units(handwritten_text_unit, folders, settings=None):
    """
    Build whole-unit and ScribeTrace-validated character crop hypotheses.

    Args:
        handwritten_text_unit: One coordinate-aware N05 text-unit record.
        folders: N05 folder dictionary or its output root path.
        settings: Optional proposer settings from N05 settings.json.

    Returns:
        JSON-safe diagnostics, recovery flags, and materialized hypotheses.
    """
    settings = dict(settings or {})
    segmentation_settings = settings.get("segmentation", settings)
    crop_padding_px = max(0, int(settings.get("crop_padding_px", 2)))
    split_overlap_px = max(0, int(settings.get("split_overlap_px", 1)))
    save_debug = bool(settings.get("save_debug", True))
    unit_id = _resolve_unit_id(handwritten_text_unit)
    source_crop_path = _first_existing_path(
        handwritten_text_unit,
        (
            "n05_selected_crop_path",
            "n05_copied_crop_path",
            "classification_crop_path",
            "routed_crop_path",
            "analysis_crop_path",
            "refined_crop_path",
            "original_crop_path",
            "context_crop_path",
            "source_crop_path",
        ),
    )
    source_mask_path = _first_existing_path(
        handwritten_text_unit,
        ("analysis_mask_crop_path", "scribetrace_mask_crop_path"),
    )
    visual = _load_visual(source_crop_path)
    mask, mask_source = _load_proposal_mask(source_mask_path, visual)

    if mask is None:
        return {
            "unit_id": unit_id,
            "status": "failed",
            "source_crop_path": source_crop_path,
            "source_mask_path": source_mask_path,
            "mask_source": mask_source,
            "recovery_needed": False,
            "recovery_reasons": [],
            "diagnostics": None,
            "segmentation_hypotheses": [],
            "split_hints": [],
            "split_artifacts_materialized": False,
            "error": "No readable analysis mask or visual crop.",
        }

    _, segments_dir, debug_dir = _resolve_output_dirs(folders)
    trace_analysis = propose_trace_validated_cuts(
        mask,
        settings=segmentation_settings,
    )
    diagnostics = trace_analysis["diagnostics"]
    diagnostics.update(_border_diagnostics(mask))
    recovery_reasons = _recovery_reasons(diagnostics)
    height, width = mask.shape[:2]
    whole_visual_path = source_crop_path or source_mask_path
    hypotheses = [
        {
            "hypothesis_id": "h0_whole",
            "type": "whole_unit",
            "segments": [
                {
                    "segment_id": "h0_s0",
                    "bbox": {"x1": 0, "y1": 0, "x2": width, "y2": height},
                    "mask_crop_path": source_mask_path,
                    "visual_crop_path": whole_visual_path,
                    "role": "whole_unit",
                }
            ],
            "score_hint": 1.0,
            "reason": "baseline_whole_unit",
        }
    ]

    split_candidates = trace_analysis["split_candidates"]
    if split_candidates:
        hypothesis_id, segments, cut_positions = _save_sequence_segments(
            unit_id=unit_id,
            candidates=split_candidates,
            mask=mask,
            visual=visual,
            segments_dir=segments_dir,
            crop_padding_px=crop_padding_px,
            split_overlap_px=split_overlap_px,
        )
        hypotheses.append(
            {
                "hypothesis_id": hypothesis_id,
                "type": "trace_supported_character_sequence",
                "segments": segments,
                "score_hint": float(
                    sum(candidate["score"] for candidate in split_candidates)
                    / len(split_candidates)
                ),
                "reason": "ordered_vector_supported_boundaries",
                "cut_positions": cut_positions,
                "split_evidence": split_candidates,
            }
        )

    debug_path = None
    if save_debug:
        debug_path = _save_split_debug(
            unit_id,
            mask,
            trace_analysis["skeleton_mask"],
            split_candidates,
            debug_dir,
        )

    return {
        "unit_id": unit_id,
        "status": "completed",
        "source_crop_path": source_crop_path,
        "source_mask_path": source_mask_path,
        "mask_source": mask_source,
        "recovery_needed": bool(recovery_reasons),
        "recovery_reasons": recovery_reasons,
        "diagnostics": diagnostics,
        "segmentation_hypotheses": hypotheses,
        "split_hints": split_candidates,
        "split_artifacts_materialized": bool(split_candidates),
        "split_debug_path": debug_path,
        "error": None,
    }


__all__ = ["propose_character_units"]
