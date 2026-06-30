"""Build normalized letter-candidate matrices from available expert evidence.

This module is deliberately conservative: it does not make the final decision.
It only gathers per-position evidence from Word OCR, selected segmentation
segments, ScribeTrace, ScriLog/ScriStistics, and the character CNN into a
stable row format that the fusion layer can consume later.
"""

from __future__ import annotations

from .schemas import make_evidence_source, make_letter_candidate, make_matrix_cell


SELECTED_PATH_STATUSES = {
    "selected",
    "selected_for_v0_2_segment_artifacts",
    "selected_for_assembly",
}


def _safe_float(value, default: float = 0.0) -> float:
    """Return a float without letting malformed expert payloads break N05."""

    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int | None = None) -> int | None:
    """Return an int when possible, otherwise a stable default."""

    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _candidate_char(candidate: dict) -> str | None:
    """Extract the Armenian label from any known candidate schema."""

    if not isinstance(candidate, dict):
        return None
    return (
        candidate.get("char")
        or candidate.get("label")
        or candidate.get("letter")
        or candidate.get("class_label")
        or candidate.get("text")
    )


def _raw_score(candidate: dict) -> float:
    """Extract a candidate score/confidence/probability as a float."""

    if not isinstance(candidate, dict):
        return 0.0
    return _safe_float(
        candidate.get("confidence")
        if candidate.get("confidence") is not None
        else candidate.get("score")
        if candidate.get("score") is not None
        else candidate.get("probability")
        if candidate.get("probability") is not None
        else candidate.get("likelihood")
        if candidate.get("likelihood") is not None
        else 0.0
    )


def _expert_output(container: dict | None, expert_name: str) -> dict:
    """Return one expert output from a unit or segment container."""

    if not isinstance(container, dict):
        return {}
    output = container.get("expert_outputs", {}).get(expert_name, {})
    return output if isinstance(output, dict) else {}


def _output_status(output: dict) -> str:
    """Convert an expert output object into available/missing/failed status."""

    if not isinstance(output, dict) or not output:
        return "missing"
    return str(output.get("status") or "available")


def _output_candidates(output: dict) -> list[dict]:
    """Return normal candidate lists from common expert-output locations."""

    if not isinstance(output, dict):
        return []
    evidence = output.get("evidence") if isinstance(output.get("evidence"), dict) else {}
    candidates = (
        output.get("candidates")
        or evidence.get("candidates")
        or evidence.get("rf_letter_candidates_for_unit")
        or output.get("rf_letter_candidates_for_unit")
        or []
    )
    return candidates if isinstance(candidates, list) else []


def _word_level_prediction(unit: dict) -> dict:
    """Return word-level OCR prediction evidence if present."""

    return (
        unit.get("expert_outputs", {})
        .get("word_level_ocr", {})
        .get("evidence", {})
        .get("prediction", {})
    )


def _word_level_trust(prediction: dict, settings: dict) -> tuple[float, list[str]]:
    """Return a small trust multiplier for Word OCR token evidence.

    Word OCR is a sequence/context expert. It is useful for rough position hints,
    but it should not dominate segment-level glyph decisions.
    """

    base_weight = _safe_float(settings.get("word_level_token_weight", 0.25), 0.25)
    notes: list[str] = []
    tokens = prediction.get("tokens") or []
    decoded_length = _safe_int(prediction.get("decoded_length"), len(tokens)) or 0
    predicted_length = _safe_int(prediction.get("predicted_length"), None)
    length_confidence = _safe_float(prediction.get("length_confidence"), 0.0)

    multiplier = 1.0
    if predicted_length is not None and decoded_length:
        mismatch = abs(predicted_length - decoded_length)
        if mismatch > 1:
            multiplier *= 0.35
            notes.append(
                "decoded_length_vs_predicted_length_mismatch:"
                f"{decoded_length}_vs_{predicted_length}"
            )
    if length_confidence and length_confidence < 0.50:
        multiplier *= 0.75
        notes.append(f"low_length_confidence:{length_confidence:.3f}")

    return base_weight * multiplier, notes


def _word_level_token_candidates(unit: dict, settings: dict) -> list[dict]:
    """Convert greedy Word OCR tokens into weak per-position candidates."""

    prediction = _word_level_prediction(unit)
    source_weight, trust_notes = _word_level_trust(prediction, settings)
    candidates = []
    for token in prediction.get("tokens") or []:
        if not isinstance(token, dict):
            continue
        char = token.get("char")
        if not char:
            continue
        token_confidence = _safe_float(token.get("confidence"), 0.0)
        token_notes = list(trust_notes)
        if token_confidence < 0.15:
            token_notes.append(f"very_low_token_confidence:{token_confidence:.3f}")
        candidates.append(
            make_letter_candidate(
                char=char,
                score=token_confidence * source_weight,
                source="word_level_ocr_context",
                rank=1,
                confidence=token.get("confidence"),
                evidence={
                    "token_id": token.get("token_id"),
                    "time_start": token.get("time_start"),
                    "time_end": token.get("time_end"),
                    "word_text": prediction.get("text"),
                    "word_confidence": prediction.get("confidence"),
                    "decoded_length": prediction.get("decoded_length"),
                    "predicted_length": prediction.get("predicted_length"),
                    "length_confidence": prediction.get("length_confidence"),
                    "source_weight": source_weight,
                    "notes": token_notes,
                },
            )
        )
    return candidates


def _normalize_candidates(
    raw_candidates: list[dict],
    source: str,
    source_weight: float,
) -> list[dict]:
    """Normalize raw candidates from one expert into matrix candidates."""

    candidates = []
    for index, candidate in enumerate(raw_candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        char = _candidate_char(candidate)
        if not char:
            continue
        confidence = _raw_score(candidate)
        candidates.append(
            make_letter_candidate(
                char=char,
                score=confidence * source_weight,
                source=source,
                rank=_safe_int(candidate.get("rank"), index),
                confidence=confidence,
                evidence={
                    "raw_candidate": candidate,
                    "source_weight": source_weight,
                    "raw_score": confidence,
                },
            )
        )
    return candidates


def _scribetrace_unit_candidates(unit: dict, settings: dict) -> list[dict]:
    """Normalize existing whole-unit ScribeTrace RF candidate evidence."""

    output = _expert_output(unit, "scribetrace")
    evidence = output.get("evidence") if isinstance(output.get("evidence"), dict) else {}
    raw_candidates = (
        evidence.get("rf_letter_candidates_for_unit")
        or unit.get("scribetrace_rf", {}).get("top5_letter_candidates_for_unit")
        or []
    )
    return _normalize_candidates(
        raw_candidates=raw_candidates,
        source="scribetrace_unit_rf_fallback",
        source_weight=_safe_float(settings.get("unit_scribetrace_fallback_weight", 0.65), 0.65),
    )


def _character_detector_unit_candidates(unit: dict, settings: dict) -> list[dict]:
    """Normalize old whole-unit CNN evidence when it exists.

    This is fallback only. CNN should normally be segment-level.
    """

    output = _expert_output(unit, "character_detector")
    return _normalize_candidates(
        raw_candidates=_output_candidates(output),
        source="character_detector_unit_fallback",
        source_weight=_safe_float(settings.get("unit_cnn_fallback_weight", 0.50), 0.50),
    )


def _segment_character_detector_candidates(segment: dict, settings: dict) -> list[dict]:
    """Normalize selected-segment CNN evidence."""

    output = _expert_output(segment, "character_detector")
    return _normalize_candidates(
        raw_candidates=_output_candidates(output),
        source="character_detector_segment",
        source_weight=_safe_float(settings.get("segment_cnn_weight", 1.0), 1.0),
    )


def _segment_scribetrace_candidates(segment: dict, settings: dict) -> list[dict]:
    """Normalize selected-segment ScribeTrace RF evidence."""

    output = _expert_output(segment, "scribetrace")
    return _normalize_candidates(
        raw_candidates=_output_candidates(output),
        source="scribetrace_segment_rf",
        source_weight=_safe_float(settings.get("segment_scribetrace_weight", 1.0), 1.0),
    )


def _candidate_effect_outputs(segment: dict) -> list[dict]:
    """Return ScriLog/ScriStistics-like outputs that may contain effects."""

    outputs = []
    for expert_name in ("scrilog_scrististics", "scrilog", "scrististics"):
        output = _expert_output(segment, expert_name)
        if output:
            outputs.append(output)
    return outputs


def _effect_lists_from_output(output: dict) -> list[dict]:
    """Extract boost/weaken candidate effects from one reasoning output."""

    if not isinstance(output, dict):
        return []
    evidence = output.get("evidence") if isinstance(output.get("evidence"), dict) else {}
    effects: list[dict] = []
    for key in ("candidate_effects", "boosted_candidates"):
        value = output.get(key) or evidence.get(key) or []
        if isinstance(value, list):
            effects.extend(item for item in value if isinstance(item, dict))
    return effects


def _segment_scrististics_effect_candidates(
    segment: dict,
    settings: dict,
    allowed_chars: set[str] | None = None,
) -> list[dict]:
    """Convert positive ScriStistics effects into low-weight support signals.

    ScriStatistics is a geometry prior, not a classifier. It should strengthen
    candidates already proposed by real recognizers, but it must not invent a
    frequent-looking Armenian letter for every weak crop.
    """

    source_weight = _safe_float(settings.get("scrististics_effect_weight", 0.12), 0.12)
    allow_new = bool(settings.get("allow_scrististics_new_candidates", False))
    candidates = []
    for output in _candidate_effect_outputs(segment):
        for index, effect in enumerate(_effect_lists_from_output(output), start=1):
            if str(effect.get("effect") or "").lower() not in {"boost", "support"}:
                continue
            char = _candidate_char(effect)
            if not char:
                continue
            if not allow_new and (not allowed_chars or char not in allowed_chars):
                continue
            strength = _safe_float(effect.get("strength"), 0.0)
            candidates.append(
                make_letter_candidate(
                    char=char,
                    score=strength * source_weight,
                    source="scrististics_segment_soft_effect",
                    rank=_safe_int(effect.get("rank"), index),
                    confidence=strength,
                    evidence={
                        "raw_effect": effect,
                        "source_weight": source_weight,
                        "note": "soft_support_for_existing_candidate_only",
                    },
                )
            )
    return candidates


def _selected_path(segmentation_entry: dict) -> dict:
    """Return the selected segmentation path, falling back to the first path."""

    paths = segmentation_entry.get("paths") or []
    for path in paths:
        if path.get("status") in SELECTED_PATH_STATUSES:
            return path
    return paths[0] if paths else {}


def _dedupe_candidates(candidates: list[dict], top_k: int) -> list[dict]:
    """Merge duplicate characters by summing source evidence.

    Earlier code kept only the strongest source. That threw away useful agreement,
    for example ScribeTrace + CNN both seeing the same letter. This version keeps
    contributor evidence and caps the combined score at 1.0.
    """

    by_char: dict[str, dict] = {}
    for candidate in candidates:
        char = candidate.get("char")
        if not char:
            continue
        score = _safe_float(candidate.get("score"), 0.0)
        current = by_char.setdefault(
            char,
            {
                "char": char,
                "score_sum": 0.0,
                "max_score": 0.0,
                "best_rank": candidate.get("rank"),
                "contributors": [],
                "source_scores": {},
            },
        )
        current["score_sum"] += score
        current["max_score"] = max(_safe_float(current.get("max_score"), 0.0), score)
        rank = candidate.get("rank")
        if rank is not None:
            current_rank = current.get("best_rank")
            if current_rank is None or int(rank) < int(current_rank):
                current["best_rank"] = rank
        source = str(candidate.get("source") or "unknown")
        current["source_scores"][source] = current["source_scores"].get(source, 0.0) + score
        current["contributors"].append(candidate)

    merged = []
    for item in by_char.values():
        combined_score = min(1.0, _safe_float(item.get("score_sum"), 0.0))
        merged.append(
            make_letter_candidate(
                char=item["char"],
                score=combined_score,
                source="fused_candidate_evidence",
                rank=item.get("best_rank"),
                confidence=combined_score,
                evidence={
                    "max_source_score": item.get("max_score"),
                    "score_sum_before_cap": item.get("score_sum"),
                    "source_scores": item.get("source_scores", {}),
                    "contributors": item.get("contributors", []),
                },
            )
        )
    return sorted(
        merged,
        key=lambda item: (
            -_rank_ready_candidate_score(item),
            str(item.get("char", "")),
        ),
    )[:top_k]


def _rank_ready_candidate_score(candidate: dict) -> float:
    """Return a sort score that prefers source agreement over lone confidence."""

    evidence = candidate.get("evidence") if isinstance(candidate.get("evidence"), dict) else {}
    source_scores = evidence.get("source_scores") if isinstance(evidence.get("source_scores"), dict) else {}
    source_count = sum(1 for score in source_scores.values() if _safe_float(score) > 0.0)
    agreement_bonus = min(0.18, max(0, source_count - 1) * 0.06)
    single_source_penalty = 0.20 if source_count <= 1 else 0.0
    return _safe_float(candidate.get("score"), 0.0) + agreement_bonus - single_source_penalty


def _top_raw_char(candidates: list[dict], source_prefix: str) -> str | None:
    """Return the strongest raw candidate from one source family."""

    matching = [
        candidate
        for candidate in candidates
        if str(candidate.get("source") or "").startswith(source_prefix)
    ]
    if not matching:
        return None
    matching.sort(key=lambda item: -_safe_float(item.get("score"), 0.0))
    return matching[0].get("char")


def _row_conflicts(raw_candidates: list[dict]) -> list[str]:
    """Detect simple source disagreements for later fusion/debugging."""

    conflicts = []
    cnn_top = _top_raw_char(raw_candidates, "character_detector_segment")
    trace_top = _top_raw_char(raw_candidates, "scribetrace_segment_rf")
    word_top = _top_raw_char(raw_candidates, "word_level_ocr_context")

    if cnn_top and trace_top and cnn_top != trace_top:
        conflicts.append(f"character_detector_vs_scribetrace:{cnn_top}_vs_{trace_top}")
    if word_top and trace_top and word_top != trace_top:
        conflicts.append(f"word_level_vs_scribetrace:{word_top}_vs_{trace_top}")
    if word_top and cnn_top and word_top != cnn_top:
        conflicts.append(f"word_level_vs_character_detector:{word_top}_vs_{cnn_top}")
    return conflicts


def _segment_evidence_sources(
    segment: dict | None,
    word_candidates: list[dict],
    raw_candidates: list[dict],
    unit_level_candidates: list[dict],
) -> list[dict]:
    """Build evidence-source metadata for one matrix row."""

    segment = segment or {}
    character_output = _expert_output(segment, "character_detector")
    trace_output = _expert_output(segment, "scribetrace")
    effect_outputs = _candidate_effect_outputs(segment)
    return [
        make_evidence_source(
            "word_level_ocr_context",
            status="available" if word_candidates else "missing",
            payload={"token_count": len(word_candidates)},
        ),
        make_evidence_source(
            "character_detector_segment",
            status=_output_status(character_output),
            payload={
                "candidate_count": len(_output_candidates(character_output)),
                "segment_level": bool(character_output),
            },
        ),
        make_evidence_source(
            "scribetrace_segment_rf",
            status=_output_status(trace_output),
            payload={
                "candidate_count": len(_output_candidates(trace_output)),
                "segment_level": bool(trace_output),
            },
        ),
        make_evidence_source(
            "scrististics_segment_soft_effect",
            status="available" if effect_outputs else "missing",
            payload={"effect_output_count": len(effect_outputs)},
            notes=["soft evidence only; should not hard-decide a letter"],
        ),
        make_evidence_source(
            "unit_level_fallback",
            status="available" if unit_level_candidates else "missing",
            payload={
                "candidate_count": len(unit_level_candidates),
                "used_in_row": any(
                    str(candidate.get("source") or "").endswith("fallback")
                    for candidate in raw_candidates
                ),
            },
            notes=["fallback only when segment-level evidence is absent or single-segment"],
        ),
    ]


def _candidates_for_segment(
    unit: dict,
    segment: dict | None,
    position: int,
    word_candidates: list[dict],
    unit_level_candidates: list[dict],
    settings: dict,
) -> list[dict]:
    """Collect raw candidate evidence for one row before dedupe/fusion."""

    candidates = []
    if position < len(word_candidates):
        candidates.append(word_candidates[position])

    segment_candidates = []
    if isinstance(segment, dict):
        segment_candidates.extend(_segment_scribetrace_candidates(segment, settings))
        segment_candidates.extend(_segment_character_detector_candidates(segment, settings))
        allowed_chars = {
            str(candidate.get("char"))
            for candidate in [*candidates, *segment_candidates]
            if candidate.get("char")
        }
        segment_candidates.extend(
            _segment_scrististics_effect_candidates(
                segment,
                settings,
                allowed_chars=allowed_chars,
            )
        )

    candidates.extend(segment_candidates)

    # Only fall back to whole-unit candidates when the selected segment has no
    # segment-level evidence yet. This prevents whole-word predictions from
    # polluting multi-letter rows.
    allow_unit_fallback = bool(settings.get("allow_unit_fallback", True))
    if allow_unit_fallback and not segment_candidates:
        candidates.extend(unit_level_candidates)

    return candidates


def build_letter_matrix_for_unit(
    unit: dict,
    segmentation_entry: dict,
    settings: dict | None = None,
) -> dict:
    """Build letter candidate rows for one text unit and its selected path."""

    settings = settings or {}
    top_k = int(settings.get("top_k_per_position", 5))
    word_candidates = _word_level_token_candidates(unit, settings)
    unit_level_candidates = (
        _scribetrace_unit_candidates(unit, settings)
        + _character_detector_unit_candidates(unit, settings)
    )
    selected_path = _selected_path(segmentation_entry)
    segments = selected_path.get("segments") or []
    estimated_positions = max(len(segments), len(word_candidates), 1)

    rows = []
    for position in range(estimated_positions):
        segment = segments[position] if position < len(segments) else None
        raw_candidates = _candidates_for_segment(
            unit=unit,
            segment=segment,
            position=position,
            word_candidates=word_candidates,
            unit_level_candidates=unit_level_candidates,
            settings=settings,
        )
        merged_candidates = _dedupe_candidates(raw_candidates, top_k=top_k)
        conflicts = _row_conflicts(raw_candidates)
        cell = make_matrix_cell(
            position=position,
            segment=segment,
            candidates=merged_candidates,
            evidence_sources=_segment_evidence_sources(
                segment=segment,
                word_candidates=word_candidates,
                raw_candidates=raw_candidates,
                unit_level_candidates=unit_level_candidates,
            ),
        )
        cell["raw_candidate_count"] = len(raw_candidates)
        cell["conflicts"] = conflicts
        if conflicts:
            cell["status"] = "candidate_conflict"
        rows.append(cell)

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
