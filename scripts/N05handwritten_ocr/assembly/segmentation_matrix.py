"""Build segmentation-path candidates for N05 assembly."""

from __future__ import annotations

import copy

from .schemas import (
    make_segmentation_path,
    make_segmentation_segment,
    normalize_bbox,
)


def _word_ocr_split_evidence(unit: dict) -> dict:
    """Extract word-OCR split-line hints for segmentation debugging."""

    prediction = (
        unit.get("expert_outputs", {})
        .get("word_level_ocr", {})
        .get("evidence", {})
        .get("prediction", {})
    )
    candidates = prediction.get("split_line_candidates") or []
    normalized = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        normalized.append(
            {
                "x": int(candidate.get("x", 0)),
                "time_step": int(candidate.get("time_step", 0)),
                "probability": float(candidate.get("probability", 0.0)),
            }
        )
    normalized.sort(key=lambda row: (-row["probability"], row["x"], row["time_step"]))
    return {
        "available": bool(normalized),
        "candidate_count": len(normalized),
        "candidates": normalized,
        "predicted_length": prediction.get("predicted_length"),
        "decoded_length": prediction.get("decoded_length"),
        "predicted_bridge_count": prediction.get("predicted_bridge_count"),
        "bridge_confidence": prediction.get("bridge_confidence"),
        "length_confidence": prediction.get("length_confidence"),
        "warning": (
            "Word OCR split lines are soft boundary hints. Assembly may turn "
            "strong hints into candidate paths, but final segmentation is not "
            "decided here."
        ),
    }


def _whole_unit_segment(unit: dict) -> dict:
    """Build the fallback segment that represents the whole text unit."""

    bbox = (
        unit.get("crop_bbox")
        or unit.get("final_bbox")
        or unit.get("document_bbox")
    )
    return make_segmentation_segment(
        segment_id="whole_s0",
        bbox=bbox,
        role="whole_unit",
        source="n05_text_unit",
        mask_crop_path=unit.get("analysis_mask_crop_path")
        or unit.get("scribetrace_mask_crop_path"),
        visual_crop_path=unit.get("n05_selected_crop_path")
        or unit.get("scribetrace_visual_crop_path"),
        source_segment={"text_unit_id": unit.get("text_unit_id")},
    )


def _segment_from_proposal(
    hypothesis_id: str,
    index: int,
    segment: dict,
) -> dict:
    """Normalize one character-unit proposer segment."""

    return make_segmentation_segment(
        segment_id=str(segment.get("segment_id") or f"{hypothesis_id}_s{index}"),
        bbox=normalize_bbox(segment.get("bbox")) or segment.get("bbox"),
        role=str(segment.get("role") or "candidate_segment"),
        source="character_unit_proposer",
        mask_crop_path=segment.get("mask_crop_path"),
        visual_crop_path=segment.get("visual_crop_path"),
        source_segment=segment,
    )


def _path_boundaries(path: dict) -> list[int]:
    """Return internal right-edge boundaries for a segmentation path."""

    segments = path.get("segments") or []
    boundaries = []
    for segment in segments[:-1]:
        bbox = normalize_bbox(segment.get("bbox")) or segment.get("bbox") or {}
        try:
            boundaries.append(int(bbox["x2"]))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(set(boundaries))


def _whole_unit_dimensions(unit: dict, paths: list[dict]) -> tuple[int, int]:
    """Resolve local crop dimensions for generated segmentation paths."""

    proposal = unit.get("character_unit_proposal") or {}
    diagnostics = proposal.get("diagnostics") or {}
    for width_key, height_key in (
        ("width", "height"),
        ("crop_width", "crop_height"),
    ):
        try:
            width = int(diagnostics[width_key])
            height = int(diagnostics[height_key])
            if width > 0 and height > 0:
                return width, height
        except (KeyError, TypeError, ValueError):
            pass

    for path in paths:
        for segment in path.get("segments") or []:
            bbox = normalize_bbox(segment.get("bbox")) or segment.get("bbox") or {}
            try:
                width = int(bbox["x2"])
                height = int(bbox["y2"])
                if width > 0 and height > 0 and segment.get("role") == "whole_unit":
                    return width, height
            except (KeyError, TypeError, ValueError):
                continue

    bbox = (
        unit.get("crop_bbox")
        or unit.get("final_bbox")
        or unit.get("document_bbox")
        or {}
    )
    normalized = normalize_bbox(bbox) or {}
    return int(normalized.get("width", 0)), int(normalized.get("height", 0))


def _select_word_ocr_boundaries(
    split_evidence: dict,
    width: int,
    settings: dict,
) -> list[dict]:
    """Select stable Word OCR boundary hints that can form a path."""

    if width <= 0:
        return []

    candidates = [
        dict(candidate)
        for candidate in split_evidence.get("candidates") or []
        if isinstance(candidate, dict)
    ]
    if not candidates:
        return []

    probability_threshold = float(
        settings.get("word_ocr_boundary_probability_threshold", 0.55)
    )
    min_gap_px = int(settings.get("word_ocr_boundary_min_gap_px", 6))
    edge_margin_px = int(settings.get("word_ocr_boundary_edge_margin_px", 4))
    max_boundaries = int(settings.get("word_ocr_max_boundaries", 12))

    predicted_length = split_evidence.get("predicted_length")
    try:
        predicted_boundary_count = max(0, int(predicted_length) - 1)
    except (TypeError, ValueError):
        predicted_boundary_count = 0
    if predicted_boundary_count:
        max_boundaries = min(max_boundaries, predicted_boundary_count)

    filtered = []
    for candidate in candidates:
        try:
            x = int(candidate.get("x", 0))
            probability = float(candidate.get("probability", 0.0))
        except (TypeError, ValueError):
            continue
        if probability < probability_threshold:
            continue
        if x <= edge_margin_px or x >= width - edge_margin_px:
            continue
        filtered.append(
            {
                "x": x,
                "time_step": int(candidate.get("time_step", 0)),
                "probability": probability,
            }
        )

    filtered.sort(key=lambda item: (-item["probability"], item["x"], item["time_step"]))
    selected = []
    for candidate in filtered:
        if any(abs(candidate["x"] - kept["x"]) < min_gap_px for kept in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max_boundaries:
            break
    return sorted(selected, key=lambda item: item["x"])


def _build_word_ocr_boundary_path(
    unit: dict,
    split_evidence: dict,
    existing_paths: list[dict],
    settings: dict,
) -> dict | None:
    """Turn strong Word OCR split hints into one candidate path."""

    if not bool(settings.get("enable_word_ocr_boundary_paths", True)):
        return None

    width, height = _whole_unit_dimensions(unit, existing_paths)
    boundaries = _select_word_ocr_boundaries(split_evidence, width, settings)
    if not boundaries:
        return None

    min_segment_width_px = int(settings.get("word_ocr_min_segment_width_px", 4))
    boundary_xs = [0, *[item["x"] for item in boundaries], width]
    segments = []
    for index, (x1, x2) in enumerate(zip(boundary_xs, boundary_xs[1:])):
        if x2 - x1 < min_segment_width_px:
            return None
        segments.append(
            make_segmentation_segment(
                segment_id=f"wocr_s{index}",
                bbox={"x1": x1, "y1": 0, "x2": x2, "y2": height},
                role="character_candidate",
                source="word_level_ocr_boundary_head",
                mask_crop_path=unit.get("analysis_mask_crop_path")
                or unit.get("scribetrace_mask_crop_path"),
                visual_crop_path=unit.get("n05_selected_crop_path")
                or unit.get("scribetrace_visual_crop_path"),
                source_segment={
                    "left_boundary_x": x1,
                    "right_boundary_x": x2,
                    "word_ocr_boundary_support": boundaries,
                },
            )
        )

    probabilities = [float(item["probability"]) for item in boundaries]
    average_probability = sum(probabilities) / max(1, len(probabilities))
    predicted_length = split_evidence.get("predicted_length")
    try:
        length_delta = abs(int(predicted_length) - len(segments))
    except (TypeError, ValueError):
        length_delta = 0
    length_score = max(0.0, 1.0 - (length_delta / max(1, len(segments))))
    bridge_count = split_evidence.get("predicted_bridge_count")
    bridge_note = (
        "Predicted bridge count is stored as soft sequence evidence; bridge "
        "coordinates are not available from this model output yet."
    )

    score_hint = 0.72 * average_probability + 0.18 * length_score + 0.10
    return make_segmentation_path(
        path_id="wocr_p0_boundaries",
        path_type="word_ocr_boundary_sequence",
        segments=segments,
        score_hint=min(1.0, score_hint),
        source="word_level_ocr_boundary_head",
        reason="word_ocr_boundary_hints_materialized",
        evidence={
            "boundary_support": boundaries,
            "average_boundary_probability": average_probability,
            "predicted_length": predicted_length,
            "length_delta": length_delta,
            "predicted_bridge_count": bridge_count,
            "bridge_note": bridge_note,
            "source": "word_level_ocr.prediction.split_line_candidates",
        },
    )


def _attach_word_ocr_alignment(path: dict, split_evidence: dict, settings: dict) -> dict:
    """Add soft Word OCR boundary agreement metadata to one path."""

    path = copy.deepcopy(path)
    boundaries = _path_boundaries(path)
    candidates = split_evidence.get("candidates") or []
    tolerance_px = int(settings.get("word_ocr_boundary_match_tolerance_px", 5))
    matches = []
    for boundary in boundaries:
        nearby = [
            candidate
            for candidate in candidates
            if abs(int(candidate.get("x", 0)) - boundary) <= tolerance_px
        ]
        if not nearby:
            continue
        best = max(nearby, key=lambda item: float(item.get("probability", 0.0)))
        matches.append(
            {
                "path_boundary_x": boundary,
                "word_ocr_x": int(best.get("x", 0)),
                "distance_px": abs(int(best.get("x", 0)) - boundary),
                "probability": float(best.get("probability", 0.0)),
            }
        )

    support_count = len(matches)
    average_probability = (
        sum(item["probability"] for item in matches) / support_count
        if support_count
        else 0.0
    )
    expected_boundaries = max(0, int(path.get("segment_count", 1)) - 1)
    support_ratio = support_count / max(1, expected_boundaries)
    alignment_score = 0.60 * support_ratio + 0.40 * average_probability

    path["base_score_hint"] = float(path.get("score_hint", 0.0))
    try:
        predicted_length = int(split_evidence.get("predicted_length") or 0)
    except (TypeError, ValueError):
        predicted_length = 0
    segment_count = int(path.get("segment_count", 0))
    length_delta = abs(predicted_length - segment_count) if predicted_length else 0
    length_agreement = (
        max(0.0, 1.0 - length_delta / max(1, predicted_length))
        if predicted_length
        else 0.0
    )
    path["word_ocr_alignment"] = {
        "available": bool(candidates),
        "path_boundary_count": expected_boundaries,
        "matched_boundary_count": support_count,
        "support_ratio": support_ratio,
        "average_probability": average_probability,
        "alignment_score": alignment_score,
        "predicted_length": predicted_length or None,
        "length_agreement": length_agreement,
        "matches": matches,
    }
    score_hint = min(
        1.0,
        float(path.get("score_hint", 0.0))
        + float(settings.get("word_ocr_alignment_score_boost", 0.15)) * alignment_score,
    )
    # Whole-unit crops are still useful fallbacks, but if Word OCR sees a
    # multi-letter sequence, the assembly surface should prefer real split
    # paths over a single oversized crop.
    if candidates and predicted_length > 1 and segment_count == 1:
        score_hint -= float(settings.get("whole_unit_word_ocr_multi_letter_penalty", 0.35))
    if predicted_length and segment_count > 1:
        score_hint += float(settings.get("word_ocr_length_agreement_boost", 0.08)) * length_agreement
    path["score_hint"] = max(0.0, min(1.0, score_hint))
    path.setdefault("evidence", {})
    path["evidence"]["word_ocr_alignment"] = path["word_ocr_alignment"]
    return path


def build_segmentation_paths(unit: dict, settings: dict | None = None) -> list[dict]:
    """Build candidate segmentation paths for one handwritten text unit.

    Args:
        unit: One N05 handwritten text-unit record.
        settings: Optional assembly settings.

    Returns:
        Deterministically sorted segmentation path records.
    """

    settings = settings or {}
    max_paths = int(settings.get("max_segmentation_paths", 8))
    proposal = unit.get("character_unit_proposal") or {}
    hypotheses = proposal.get("segmentation_hypotheses") or []

    paths: list[dict] = []
    for index, hypothesis in enumerate(hypotheses):
        if not isinstance(hypothesis, dict):
            continue
        hypothesis_id = str(
            hypothesis.get("hypothesis_id")
            or hypothesis.get("id")
            or f"h{index}"
        )
        segments = [
            _segment_from_proposal(hypothesis_id, segment_index, segment)
            for segment_index, segment in enumerate(hypothesis.get("segments") or [])
            if isinstance(segment, dict)
        ]
        if not segments:
            continue
        paths.append(
            make_segmentation_path(
                path_id=hypothesis_id,
                path_type=str(hypothesis.get("type") or "proposal"),
                segments=segments,
                score_hint=float(hypothesis.get("score_hint") or 0.0),
                source="character_unit_proposer",
                reason=str(hypothesis.get("reason") or ""),
                evidence={
                    "recovery_needed": bool(proposal.get("recovery_needed")),
                    "recovery_reasons": proposal.get("recovery_reasons", []),
                    "source_hypothesis": hypothesis,
                },
            )
        )

    if not paths:
        paths.append(
            make_segmentation_path(
                path_id="h0_whole",
                path_type="whole_unit",
                segments=[_whole_unit_segment(unit)],
                score_hint=1.0,
                source="n05_text_unit",
                reason="fallback_whole_unit",
                evidence={
                    "proposal_status": proposal.get("status", "missing"),
                },
            )
        )

    split_evidence = _word_ocr_split_evidence(unit)
    # Existing proposer paths remain first-class citizens. Word OCR only adds
    # alignment evidence or an extra candidate path; it does not overwrite the
    # geometric segment proposals.
    paths = [
        _attach_word_ocr_alignment(path, split_evidence, settings)
        for path in paths
    ]
    word_ocr_path = _build_word_ocr_boundary_path(
        unit,
        split_evidence,
        paths,
        settings,
    )
    if word_ocr_path:
        paths.append(word_ocr_path)

    paths.sort(
        key=lambda path: (
            -float(path.get("score_hint", 0.0)),
            int(path.get("segment_count", 0)),
            str(path.get("path_id", "")),
        )
    )
    return paths[:max_paths]


def build_segmentation_matrix(units: list[dict], settings: dict | None = None) -> list[dict]:
    """Build segmentation candidates for every N05 text unit."""

    return [
        {
            "text_unit_id": unit.get("text_unit_id"),
            "group_id": unit.get("group_id"),
            "selected_path_id": None,
            "path_count": len(paths := build_segmentation_paths(unit, settings)),
            "word_ocr_split_evidence": _word_ocr_split_evidence(unit),
            "paths": paths,
        }
        for unit in units
    ]
