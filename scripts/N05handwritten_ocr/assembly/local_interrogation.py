"""Local segmentation sanity checks for N05 assembly.

This layer is the first small "does this crop shape make sense?" judge. It
does not decide OCR. It only re-ranks segmentation paths and creates safe local
split alternatives when one segment looks suspiciously wide for a single
letter.
"""

from __future__ import annotations

import copy

from .schemas import make_segmentation_segment, normalize_bbox


def _safe_float(value, default: float = 0.0) -> float:
    """Return ``value`` as float without letting bad JSON break assembly."""

    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    """Return ``value`` as int without letting bad JSON break assembly."""

    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _path_bounds(path: dict) -> dict | None:
    """Return the union bbox for all valid segments in a path."""

    boxes = [
        bbox
        for segment in path.get("segments") or []
        if (bbox := normalize_bbox(segment.get("bbox")))
    ]
    if not boxes:
        return None
    return normalize_bbox(
        {
            "x1": min(box["x1"] for box in boxes),
            "y1": min(box["y1"] for box in boxes),
            "x2": max(box["x2"] for box in boxes),
            "y2": max(box["y2"] for box in boxes),
        }
    )


def _expected_count(entry: dict, path: dict) -> int:
    """Resolve how many character-like chunks this path is expected to cover."""

    split_evidence = entry.get("word_ocr_split_evidence") or {}
    predicted_length = _safe_int(split_evidence.get("predicted_length"), 0)
    if predicted_length > 0:
        return predicted_length
    return max(1, _safe_int(path.get("segment_count"), len(path.get("segments") or [])))


def _embedded_geometry(segment: dict) -> dict:
    """Return geometry hints already attached by upstream tools, if present."""

    if isinstance(segment.get("segment_geometry"), dict):
        return segment["segment_geometry"]
    source_segment = segment.get("source_segment") or {}
    if isinstance(source_segment.get("segment_geometry"), dict):
        return source_segment["segment_geometry"]
    return {}


def _profile_geometry(geometry: dict) -> dict:
    """Return Scrististics profile hints nested inside geometry, if present."""

    profile = geometry.get("scrististics_profile")
    return profile if isinstance(profile, dict) else {}


def _judge_segment(
    segment: dict,
    expected_width: float,
    settings: dict,
) -> dict:
    """Return local geometry suspicion for one segment."""

    bbox = normalize_bbox(segment.get("bbox")) or {}
    width = _safe_float(bbox.get("width"), 0.0)
    height = _safe_float(bbox.get("height"), 0.0)
    ratio = width / max(1.0, expected_width)
    too_wide_ratio = _safe_float(
        settings.get("local_interrogation_too_wide_ratio"),
        1.65,
    )
    too_narrow_ratio = _safe_float(
        settings.get("local_interrogation_too_narrow_ratio"),
        0.45,
    )

    geometry = _embedded_geometry(segment)
    profile = _profile_geometry(geometry)
    geometry_too_wide = bool(
        geometry.get("too_wide")
        or geometry.get("likely_joined_letters")
        or profile.get("profile_too_wide")
    )
    geometry_too_narrow = bool(
        geometry.get("too_narrow")
        or geometry.get("likely_fragment")
        or profile.get("profile_too_narrow")
    )
    too_wide = ratio >= too_wide_ratio or geometry_too_wide
    too_narrow = 0.0 < ratio <= too_narrow_ratio or geometry_too_narrow

    suspicion = 0.0
    reasons = []
    if too_wide:
        suspicion += min(1.0, max(0.25, ratio - too_wide_ratio + 0.25))
        reasons.append("segment_too_wide_possible_joined_letters")
    if too_narrow:
        suspicion += min(0.75, max(0.20, too_narrow_ratio - ratio + 0.20))
        reasons.append("segment_too_narrow_possible_fragment")
    if geometry_too_wide:
        suspicion += 0.25
        reasons.append("upstream_geometry_flags_joined_letters")
    if geometry_too_narrow:
        suspicion += 0.15
        reasons.append("upstream_geometry_flags_fragment")

    return {
        "segment_id": segment.get("segment_id"),
        "bbox": bbox,
        "width": width,
        "height": height,
        "expected_width": expected_width,
        "width_to_expected_ratio": ratio,
        "too_wide": too_wide,
        "too_narrow": too_narrow,
        "likely_joined_letters": too_wide,
        "likely_fragment": too_narrow,
        "suspicion": round(suspicion, 6),
        "reasons": reasons,
    }


def _judge_path(entry: dict, path: dict, settings: dict) -> dict:
    """Score one path for local segmentation sanity."""

    bounds = _path_bounds(path) or {}
    expected_count = max(1, _expected_count(entry, path))
    expected_width = _safe_float(bounds.get("width"), 0.0) / expected_count
    segment_judgements = [
        _judge_segment(segment, expected_width, settings)
        for segment in path.get("segments") or []
    ]
    suspicious = [row for row in segment_judgements if row["reasons"]]
    too_wide = [row for row in suspicious if row["too_wide"]]
    too_narrow = [row for row in suspicious if row["too_narrow"]]
    suspicion_total = sum(row["suspicion"] for row in suspicious)
    penalty = _safe_float(
        settings.get("local_interrogation_suspicion_penalty"),
        0.18,
    ) * suspicion_total
    return {
        "enabled": True,
        "expected_segment_count": expected_count,
        "path_width": _safe_float(bounds.get("width"), 0.0),
        "expected_segment_width": expected_width,
        "suspicious_segment_count": len(suspicious),
        "too_wide_segment_count": len(too_wide),
        "too_narrow_segment_count": len(too_narrow),
        "suspicion_total": round(suspicion_total, 6),
        "score_penalty": round(penalty, 6),
        "segments": segment_judgements,
    }


def _split_segment(segment: dict, suffix: str, bbox: dict, source_segment: dict) -> dict:
    """Create one local split segment while preserving source crop paths."""

    return make_segmentation_segment(
        segment_id=f"{segment.get('segment_id', 'segment')}_{suffix}",
        bbox=bbox,
        role="local_interrogation_split_candidate",
        source="assembly_local_interrogation",
        mask_crop_path=segment.get("mask_crop_path"),
        visual_crop_path=segment.get("visual_crop_path"),
        source_segment=source_segment,
    )


def _split_wide_segment_path(
    entry: dict,
    path: dict,
    judgement: dict,
    segment_index: int,
    settings: dict,
) -> dict | None:
    """Build one path alternative by splitting a suspicious segment in two."""

    segments = path.get("segments") or []
    if segment_index < 0 or segment_index >= len(segments):
        return None
    original_segment = segments[segment_index]
    bbox = normalize_bbox(original_segment.get("bbox"))
    if not bbox:
        return None
    min_width = _safe_int(settings.get("local_interrogation_min_split_width_px"), 4)
    midpoint = int(round((bbox["x1"] + bbox["x2"]) / 2.0))
    if midpoint - bbox["x1"] < min_width or bbox["x2"] - midpoint < min_width:
        return None

    source_segment = {
        "local_interrogation_source_path_id": path.get("path_id"),
        "local_interrogation_source_segment_id": original_segment.get("segment_id"),
        "local_interrogation_reason": "split_suspicious_wide_segment",
        "local_interrogation_judgement": judgement,
    }
    left = _split_segment(
        original_segment,
        "li0",
        {"x1": bbox["x1"], "y1": bbox["y1"], "x2": midpoint, "y2": bbox["y2"]},
        source_segment,
    )
    right = _split_segment(
        original_segment,
        "li1",
        {"x1": midpoint, "y1": bbox["y1"], "x2": bbox["x2"], "y2": bbox["y2"]},
        source_segment,
    )
    new_segments = [
        copy.deepcopy(segment)
        for index, segment in enumerate(segments)
        if index != segment_index
    ]
    new_segments[segment_index:segment_index] = [left, right]

    alternative = copy.deepcopy(path)
    alternative["path_id"] = (
        f"{path.get('path_id', 'path')}_li_split_{segment_index}"
    )
    alternative["type"] = "local_interrogation_split"
    alternative["source"] = "assembly_local_interrogation"
    alternative["segments"] = new_segments
    alternative["segment_count"] = len(new_segments)
    alternative["reason"] = "local_interrogation_split_suspicious_wide_segment"
    alternative["status"] = "candidate"
    alternative.setdefault("evidence", {})
    alternative["evidence"]["local_interrogation_split"] = {
        "source_path_id": path.get("path_id"),
        "source_segment_id": original_segment.get("segment_id"),
        "split_x": midpoint,
        "source_judgement": judgement,
    }

    alt_judgement = _judge_path(entry, alternative, settings)
    split_bonus = _safe_float(settings.get("local_interrogation_split_bonus"), 0.08)
    base_score = _safe_float(path.get("base_score_hint"), path.get("score_hint", 0.0))
    alternative["base_score_hint"] = base_score
    alternative["score_hint"] = max(
        0.0,
        min(1.0, base_score + split_bonus - alt_judgement["score_penalty"]),
    )
    alternative["local_interrogation"] = alt_judgement
    alternative["local_interrogation"]["created_from_path_id"] = path.get("path_id")
    alternative["local_interrogation"]["created_by"] = "split_wide_segment"
    return alternative


def _annotate_path(entry: dict, path: dict, settings: dict) -> dict:
    """Attach local interrogation metadata and update the score hint."""

    annotated = copy.deepcopy(path)
    judgement = _judge_path(entry, annotated, settings)
    base_score = _safe_float(
        annotated.get("base_score_hint"),
        annotated.get("score_hint", 0.0),
    )
    annotated["base_score_hint"] = base_score
    annotated["score_hint"] = max(0.0, min(1.0, base_score - judgement["score_penalty"]))
    annotated["local_interrogation"] = judgement
    annotated.setdefault("evidence", {})
    annotated["evidence"]["local_interrogation"] = judgement
    return annotated


def apply_local_interrogation(
    segmentation_matrix: list[dict],
    settings: dict | None = None,
) -> list[dict]:
    """Annotate, re-rank, and extend segmentation paths with local checks."""

    settings = settings or {}
    if not bool(settings.get("enable_local_interrogation", True)):
        return segmentation_matrix

    max_paths = _safe_int(settings.get("max_segmentation_paths"), 8)
    max_alternatives_per_path = _safe_int(
        settings.get("local_interrogation_max_alternatives_per_path"),
        2,
    )
    output = []
    for entry in segmentation_matrix:
        updated_entry = copy.deepcopy(entry)
        original_paths = updated_entry.get("paths") or []
        annotated_paths = [
            _annotate_path(updated_entry, path, settings)
            for path in original_paths
        ]
        alternatives = []
        for path in annotated_paths:
            if len(alternatives) >= max_alternatives_per_path:
                break
            judgements = path.get("local_interrogation", {}).get("segments") or []
            for segment_index, judgement in enumerate(judgements):
                if len(alternatives) >= max_alternatives_per_path:
                    break
                if not judgement.get("too_wide"):
                    continue
                alternative = _split_wide_segment_path(
                    updated_entry,
                    path,
                    judgement,
                    segment_index,
                    settings,
                )
                if alternative:
                    alternatives.append(alternative)

        paths = [*annotated_paths, *alternatives]
        paths.sort(
            key=lambda path: (
                -_safe_float(path.get("score_hint"), 0.0),
                _safe_int(path.get("segment_count"), 0),
                str(path.get("path_id", "")),
            )
        )
        if max_paths > 0:
            paths = paths[:max_paths]
        updated_entry["paths"] = paths
        updated_entry["path_count"] = len(paths)
        updated_entry["local_interrogation"] = {
            "enabled": True,
            "input_path_count": len(original_paths),
            "generated_alternative_count": len(alternatives),
            "selected_path_id_after_rerank": paths[0].get("path_id") if paths else None,
        }
        output.append(updated_entry)
    return output
