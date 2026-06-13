"""Propose deterministic character-unit segmentations for all N05 experts."""

import os
import re

import cv2
import numpy as np


MAX_SPLIT_HYPOTHESES = 5
MIN_SPLIT_WIDTH_PX = 24
WIDE_ASPECT_RATIO = 1.60
MANY_COMPONENTS_THRESHOLD = 4
MULTI_LETTER_COMPONENT_THRESHOLD = 3
MIN_SEGMENT_WIDTH_RATIO = 0.15
VALLEY_MAX_PROJECTION_RATIO = 0.35
MIN_CUT_SPACING_RATIO = 0.08


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


def _resolve_output_dir(folders):
    """Resolve the character-unit proposer output directory from N05 folders."""
    if isinstance(folders, str):
        n05_output_dir = os.path.abspath(folders)
        proposer_dir = os.path.join(n05_output_dir, "character_unit_proposer")
        segments_dir = os.path.join(proposer_dir, "segments")
    elif isinstance(folders, dict):
        n05_output_dir = os.path.abspath(
            folders.get("root")
            or folders.get("output_dir")
            or folders.get("n05_output_dir")
            or "."
        )
        proposer_dir = os.path.abspath(
            folders.get("character_unit_proposer")
            or os.path.join(n05_output_dir, "character_unit_proposer")
        )
        segments_dir = os.path.abspath(
            folders.get("character_unit_segments")
            or os.path.join(proposer_dir, "segments")
        )
    else:
        raise ValueError("folders must be an N05 folder dictionary or output path.")

    os.makedirs(segments_dir, exist_ok=True)
    return proposer_dir, segments_dir


def _load_visual(path):
    """Load a visual crop when available."""
    if not path:
        return None
    return cv2.imread(path, cv2.IMREAD_COLOR)


def _normalize_analysis_mask(mask):
    """Normalize an analysis mask to white ink on black."""
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Analysis masks are expected white-on-black. Invert obvious opposite
    # polarity inputs so proposal diagnostics remain usable.
    if np.count_nonzero(binary) > binary.size * 0.5:
        binary = cv2.bitwise_not(binary)

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


def _connected_component_count(mask):
    """Count non-background 8-connected components."""
    component_count, _ = cv2.connectedComponents(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )
    return max(0, int(component_count) - 1)


def _build_diagnostics(mask):
    """Measure crop geometry, component structure, borders, and projection."""
    height, width = mask.shape[:2]
    ink = mask > 0
    projection = np.count_nonzero(ink, axis=0).astype(int).tolist()

    return {
        "width": int(width),
        "height": int(height),
        "aspect_ratio": float(width / height) if height else 0.0,
        "ink_pixel_count": int(np.count_nonzero(ink)),
        "connected_component_count": _connected_component_count(mask),
        "touches_left_border": bool(np.any(ink[:, 0])) if width else False,
        "touches_right_border": bool(np.any(ink[:, -1])) if width else False,
        "touches_top_border": bool(np.any(ink[0, :])) if height else False,
        "touches_bottom_border": bool(np.any(ink[-1, :])) if height else False,
        "vertical_projection_profile": projection,
    }


def _recovery_reasons(diagnostics):
    """Translate diagnostics into deterministic recovery flags."""
    reasons = []

    for diagnostic_key, reason in (
        ("touches_left_border", "touches_left_border"),
        ("touches_right_border", "touches_right_border"),
        ("touches_top_border", "touches_top_border"),
        ("touches_bottom_border", "touches_bottom_border"),
    ):
        if diagnostics[diagnostic_key]:
            reasons.append(reason)

    if (
        diagnostics["width"] >= MIN_SPLIT_WIDTH_PX
        and diagnostics["aspect_ratio"] >= WIDE_ASPECT_RATIO
    ):
        reasons.append("wide_unit_possible_multi_letter")

    if diagnostics["connected_component_count"] >= MANY_COMPONENTS_THRESHOLD:
        reasons.append("many_components")

    return reasons


def _should_propose_splits(diagnostics):
    """Return whether unit geometry plausibly represents multiple letters."""
    return (
        diagnostics["width"] >= MIN_SPLIT_WIDTH_PX
        and (
            diagnostics["aspect_ratio"] >= WIDE_ASPECT_RATIO
            or diagnostics["connected_component_count"]
            >= MULTI_LETTER_COMPONENT_THRESHOLD
        )
    )


def _projection_valley_candidates(projection):
    """Rank balanced low-ink projection runs as possible two-way cuts."""
    width = len(projection)
    if width < MIN_SPLIT_WIDTH_PX:
        return []

    maximum = max(projection, default=0)
    if maximum <= 0:
        return []

    minimum_segment_width = max(3, int(round(width * MIN_SEGMENT_WIDTH_RATIO)))
    start = minimum_segment_width
    end = width - minimum_segment_width
    if end <= start:
        return []

    valley_limit = max(1, int(round(maximum * VALLEY_MAX_PROJECTION_RATIO)))
    eligible = [
        x for x in range(start, end) if projection[x] <= valley_limit
    ]

    runs = []
    for x in eligible:
        if not runs or x > runs[-1][-1] + 1:
            runs.append([x])
        else:
            runs[-1].append(x)

    total_ink = float(sum(projection))
    candidates = []
    for run in runs:
        minimum_value = min(projection[x] for x in run)
        minimum_positions = [x for x in run if projection[x] == minimum_value]
        run_center = (run[0] + run[-1]) / 2.0
        cut_x = min(
            minimum_positions,
            key=lambda x: (abs(x - run_center), x),
        )

        left_ink = float(sum(projection[:cut_x]))
        right_ink = float(sum(projection[cut_x:]))
        if left_ink <= 0 or right_ink <= 0:
            continue

        depth_score = 1.0 - (float(minimum_value) / float(maximum))
        balance_score = (
            1.0 - abs(left_ink - right_ink) / total_ink
            if total_ink
            else 0.0
        )
        score = 0.70 * depth_score + 0.30 * balance_score
        candidates.append(
            {
                "cut_x": int(cut_x),
                "projection_value": int(minimum_value),
                "score": float(score),
            }
        )

    candidates.sort(
        key=lambda item: (
            -item["score"],
            item["projection_value"],
            abs(item["cut_x"] - width / 2.0),
            item["cut_x"],
        )
    )

    selected = []
    minimum_spacing = max(3, int(round(width * MIN_CUT_SPACING_RATIO)))
    for candidate in candidates:
        if any(
            abs(candidate["cut_x"] - existing["cut_x"]) < minimum_spacing
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= MAX_SPLIT_HYPOTHESES:
            break

    return selected


def _save_image(path, image):
    """Save one generated segment artifact or raise a precise error."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"Could not save character-unit segment: {path}")
    return os.path.abspath(path)


def _visual_segment(visual, mask, x1, x2):
    """Crop visual evidence in coordinates corresponding to the mask segment."""
    if visual is None:
        white = np.full((mask.shape[0], x2 - x1, 3), 255, dtype=np.uint8)
        ink = mask[:, x1:x2] > 0
        white[ink] = (0, 0, 0)
        return white

    visual_height, visual_width = visual.shape[:2]
    mask_height, mask_width = mask.shape[:2]
    visual_x1 = int(round(x1 * visual_width / max(1, mask_width)))
    visual_x2 = int(round(x2 * visual_width / max(1, mask_width)))
    visual_x1 = max(0, min(visual_x1, visual_width - 1))
    visual_x2 = max(visual_x1 + 1, min(visual_x2, visual_width))

    crop = visual[:, visual_x1:visual_x2]
    target_width = max(1, x2 - x1)
    if crop.shape[:2] != (mask_height, target_width):
        crop = cv2.resize(
            crop,
            (target_width, mask_height),
            interpolation=cv2.INTER_AREA,
        )
    return crop


def _save_split_segments(
    unit_id,
    hypothesis_index,
    cut_x,
    mask,
    visual,
    segments_dir,
):
    """Save both halves and return their JSON-safe segment records."""
    height, width = mask.shape[:2]
    hypothesis_id = f"h{hypothesis_index}_split_x{cut_x:04d}"
    segments = []

    for segment_index, (x1, x2) in enumerate(((0, cut_x), (cut_x, width))):
        segment_id = f"h{hypothesis_index}_s{segment_index}"
        prefix = (
            f"{_sanitize_identifier(unit_id)}_"
            f"{hypothesis_id}_{segment_id}"
        )
        mask_path = _save_image(
            os.path.join(segments_dir, f"{prefix}_mask.png"),
            mask[:, x1:x2],
        )
        visual_path = _save_image(
            os.path.join(segments_dir, f"{prefix}_visual.png"),
            _visual_segment(visual, mask, x1, x2),
        )
        segments.append(
            {
                "segment_id": segment_id,
                "bbox": {
                    "x1": int(x1),
                    "y1": 0,
                    "x2": int(x2),
                    "y2": int(height),
                },
                "mask_crop_path": mask_path,
                "visual_crop_path": visual_path,
                "role": "character_candidate",
            }
        )

    return hypothesis_id, segments


def propose_character_units(handwritten_text_unit, folders):
    """
    Describe one whole text unit and retain non-materialized split hints.

    Args:
        handwritten_text_unit: One coordinate-aware N05 text-unit record.
        folders: N05 folder dictionary or its output root path.

    Returns:
        JSON-safe proposal with diagnostics, recovery flags, and hypotheses.
    """
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
            "error": "No readable analysis mask or visual crop.",
        }

    height, width = mask.shape[:2]
    diagnostics = _build_diagnostics(mask)
    recovery_reasons = _recovery_reasons(diagnostics)
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

    split_hints = []
    if _should_propose_splits(diagnostics):
        split_hints = _projection_valley_candidates(
            diagnostics["vertical_projection_profile"]
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
        "split_hints": split_hints,
        "split_artifacts_materialized": False,
        "error": None,
    }


__all__ = ["propose_character_units"]
