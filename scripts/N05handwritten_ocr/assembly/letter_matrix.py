"""Build normalized letter-candidate matrices from available expert evidence."""

from __future__ import annotations

from .schemas import make_evidence_source, make_letter_candidate, make_matrix_cell


def _word_level_prediction(unit: dict) -> dict:
    """Return word-level OCR prediction evidence if present."""

    return (
        unit.get("expert_outputs", {})
        .get("word_level_ocr", {})
        .get("evidence", {})
        .get("prediction", {})
    )


def _word_level_token_candidates(unit: dict) -> list[dict]:
    """Convert greedy word OCR tokens into per-position candidate rows."""

    prediction = _word_level_prediction(unit)
    candidates = []
    for token in prediction.get("tokens") or []:
        if not isinstance(token, dict):
            continue
        char = token.get("char")
        if not char:
            continue
        candidates.append(
            make_letter_candidate(
                char=char,
                score=float(token.get("confidence") or 0.0),
                source="word_level_ocr",
                rank=1,
                confidence=token.get("confidence"),
                evidence={
                    "token_id": token.get("token_id"),
                    "time_start": token.get("time_start"),
                    "time_end": token.get("time_end"),
                    "word_text": prediction.get("text"),
                    "word_confidence": prediction.get("confidence"),
                },
            )
        )
    return candidates


def _scribetrace_unit_candidates(unit: dict) -> list[dict]:
    """Normalize existing whole-unit ScribeTrace RF candidate evidence."""

    evidence = (
        unit.get("expert_outputs", {})
        .get("scribetrace", {})
        .get("evidence", {})
    )
    raw_candidates = (
        evidence.get("rf_letter_candidates_for_unit")
        or unit.get("scribetrace_rf", {}).get("top5_letter_candidates_for_unit")
        or []
    )
    candidates = []
    for index, candidate in enumerate(raw_candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        char = (
            candidate.get("char")
            or candidate.get("label")
            or candidate.get("letter")
            or candidate.get("class_label")
        )
        if not char:
            continue
        confidence = (
            candidate.get("confidence")
            or candidate.get("score")
            or candidate.get("probability")
            or 0.0
        )
        candidates.append(
            make_letter_candidate(
                char=char,
                score=float(confidence or 0.0),
                source="scribetrace_unit_rf",
                rank=int(candidate.get("rank") or index),
                confidence=confidence,
                evidence=candidate,
            )
        )
    return candidates


def _character_detector_candidates(unit: dict) -> list[dict]:
    """Normalize CNN character-detector evidence when it exists."""

    output = unit.get("expert_outputs", {}).get("character_detector", {})
    raw_candidates = output.get("candidates") or output.get("evidence", {}).get("candidates") or []
    candidates = []
    for index, candidate in enumerate(raw_candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        char = candidate.get("char") or candidate.get("label") or candidate.get("letter")
        if not char:
            continue
        confidence = candidate.get("confidence") or candidate.get("score") or 0.0
        candidates.append(
            make_letter_candidate(
                char=char,
                score=float(confidence or 0.0),
                source="character_detector",
                rank=int(candidate.get("rank") or index),
                confidence=confidence,
                evidence=candidate,
            )
        )
    return candidates


def _dedupe_candidates(candidates: list[dict], top_k: int) -> list[dict]:
    """Merge duplicate characters by keeping the strongest source score."""

    by_char: dict[str, dict] = {}
    for candidate in candidates:
        char = candidate.get("char")
        if not char:
            continue
        current = by_char.get(char)
        if current is None or float(candidate.get("score", 0.0)) > float(current.get("score", 0.0)):
            by_char[char] = candidate
    return sorted(
        by_char.values(),
        key=lambda item: (-float(item.get("score", 0.0)), str(item.get("char", ""))),
    )[:top_k]


def build_letter_matrix_for_unit(
    unit: dict,
    segmentation_entry: dict,
    settings: dict | None = None,
) -> dict:
    """Build letter candidate rows for one text unit and its path candidates."""

    settings = settings or {}
    top_k = int(settings.get("top_k_per_position", 5))
    word_candidates = _word_level_token_candidates(unit)
    unit_level_candidates = (
        _scribetrace_unit_candidates(unit)
        + _character_detector_candidates(unit)
    )
    selected_path = (segmentation_entry.get("paths") or [{}])[0]
    segments = selected_path.get("segments") or []
    estimated_positions = max(len(word_candidates), len(segments), 1)

    rows = []
    for position in range(estimated_positions):
        candidates = []
        if position < len(word_candidates):
            candidates.append(word_candidates[position])
        # Whole-unit candidates are informative, but not trusted as final
        # letter-position evidence until segment-level experts are wired.
        if estimated_positions == 1:
            candidates.extend(unit_level_candidates)
        rows.append(
            make_matrix_cell(
                position=position,
                segment=segments[position] if position < len(segments) else None,
                candidates=_dedupe_candidates(candidates, top_k=top_k),
                evidence_sources=[
                    make_evidence_source(
                        "word_level_ocr",
                        status="available" if word_candidates else "missing",
                        payload={"token_count": len(word_candidates)},
                    ),
                    make_evidence_source(
                        "scribetrace_unit_rf",
                        status="available" if unit_level_candidates else "missing",
                        payload={
                            "candidate_count": len(unit_level_candidates),
                            "unit_level_warning": (
                                "Whole-unit RF candidates are not segment-specific yet."
                            ),
                        },
                    ),
                ],
            )
        )

    return {
        "text_unit_id": unit.get("text_unit_id"),
        "group_id": unit.get("group_id"),
        "selected_path_id": selected_path.get("path_id"),
        "estimated_position_count": estimated_positions,
        "rows": rows,
    }


def build_letter_matrix(
    units: list[dict],
    segmentation_matrix: list[dict],
    settings: dict | None = None,
) -> list[dict]:
    """Build all letter matrices aligned with the segmentation matrix."""

    by_unit = {
        str(entry.get("text_unit_id")): entry
        for entry in segmentation_matrix
    }
    matrices = []
    for unit in units:
        key = str(unit.get("text_unit_id"))
        segmentation_entry = by_unit.get(key) or {"paths": []}
        matrices.append(build_letter_matrix_for_unit(unit, segmentation_entry, settings))
    return matrices
