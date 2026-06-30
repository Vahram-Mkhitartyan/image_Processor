"""ScribeJudge overlay for N05 decision matrices.

The first version is deliberately transparent. It does not replace the decision
matrix; it annotates it with confusion-history-aware scores and features so we
can train a real meta-model on the same schema later.
"""

from __future__ import annotations

from .confusion_memory import ConfusionMemory, DEFAULT_CONFUSION_SOURCES

SCRIBEJUDGE_VERSION = "scribejudge_overlay_v0_1"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _candidate_sources(position: dict, char: str) -> list[str]:
    for candidate in position.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("char") or "") == str(char):
            return [str(source) for source in candidate.get("sources") or []]
    return []


def _source_scores(position: dict, char: str) -> dict:
    for candidate in position.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("char") or "") == str(char):
            return {
                str(source): _safe_float(score)
                for source, score in (candidate.get("source_scores") or {}).items()
            }
    return {}


def _char_score(position: dict, char: str) -> float:
    for candidate in position.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("char") or "") == str(char):
            return _safe_float(candidate.get("score"), _safe_float(candidate.get("confidence")))
    return 0.0


def _position_record(
    position: dict,
    selected_char: str,
    confusion_memory: ConfusionMemory,
    settings: dict,
) -> dict:
    """Build one position-level judge record."""

    candidates = [
        candidate
        for candidate in position.get("candidates") or []
        if isinstance(candidate, dict) and candidate.get("char")
    ]
    selected_char = str(selected_char or "")
    selected_score = _char_score(position, selected_char)
    selected_sources = _candidate_sources(position, selected_char)
    selected_source_scores = _source_scores(position, selected_char)

    alternatives = []
    for candidate in candidates:
        char = str(candidate.get("char") or "")
        if not char or char == selected_char:
            continue
        risk = confusion_memory.predicted_to_true_risk(selected_char, char)
        candidate_score = _safe_float(candidate.get("score"), _safe_float(candidate.get("confidence")))
        source_count = len(candidate.get("sources") or [])
        alternatives.append(
            {
                "char": char,
                "matrix_rank": _safe_int(candidate.get("rank"), 999),
                "matrix_score": candidate_score,
                "source_count": source_count,
                "sources": candidate.get("sources") or [],
                "confusion_risk_from_selected": risk,
            }
        )

    alternatives.sort(
        key=lambda item: (
            -_safe_float(item.get("confusion_risk_from_selected", {}).get("risk")),
            -_safe_float(item.get("matrix_score")),
            _safe_int(item.get("matrix_rank"), 999),
        )
    )
    best_alternative = alternatives[0] if alternatives else {}
    best_risk = _safe_float(
        best_alternative.get("confusion_risk_from_selected", {}).get("risk")
        if best_alternative else 0.0
    )
    confidence_floor = _safe_float(settings.get("confidence_floor", 0.12), 0.12)
    suspicious_high_confidence = bool(
        selected_score >= _safe_float(settings.get("high_confidence_threshold", 0.55), 0.55)
        and best_risk >= _safe_float(settings.get("confusion_risk_alert_threshold", 0.35), 0.35)
    )
    suspicious_low_confidence = bool(
        selected_score <= confidence_floor
        and len(candidates) > 1
    )
    selected_drop = best_risk * _safe_float(settings.get("selected_confusion_drop", 0.18), 0.18)
    alternative_boost = best_risk * _safe_float(settings.get("alternative_confusion_boost", 0.16), 0.16)
    judged_selected_score = selected_score - selected_drop

    return {
        "index": position.get("index"),
        "selected_char": selected_char,
        "selected_score": selected_score,
        "judged_selected_score": judged_selected_score,
        "selected_sources": selected_sources,
        "selected_source_scores": selected_source_scores,
        "candidate_count": len(candidates),
        "source_count": len(selected_sources),
        "has_conflicts": bool(position.get("conflicts")),
        "suspicious_high_confidence": suspicious_high_confidence,
        "suspicious_low_confidence": suspicious_low_confidence,
        "selected_confusion_drop": selected_drop,
        "best_alternative_boost": alternative_boost,
        "best_confusion_alternative": best_alternative,
        "top_confusion_alternatives": alternatives[:5],
        "features": {
            "selected_score": selected_score,
            "selected_source_count": len(selected_sources),
            "candidate_count": len(candidates),
            "best_confusion_risk": best_risk,
            "has_conflicts": 1.0 if position.get("conflicts") else 0.0,
        },
    }


def _selected_chars_for_row(row: dict) -> list[str]:
    text = str(row.get("selected_text") or row.get("input_text") or "")
    chars = []
    backup_sequence = row.get("letter_backup_sequence") or []
    if backup_sequence:
        for item in backup_sequence:
            if isinstance(item, dict):
                chars.append(str(item.get("l") or ""))
    if chars:
        return chars
    return list(text)


def _token_overlay(row: dict, combined_token: dict, confusion_memory: ConfusionMemory, settings: dict) -> dict:
    positions = [
        position
        for position in combined_token.get("character_candidates") or []
        if isinstance(position, dict)
    ]
    selected_chars = _selected_chars_for_row(row)
    position_overlays = []
    for index, position in enumerate(positions):
        selected_char = selected_chars[index] if index < len(selected_chars) else position.get("char", "")
        position_overlays.append(
            _position_record(
                position=position,
                selected_char=selected_char,
                confusion_memory=confusion_memory,
                settings=settings,
            )
        )

    suspicious_high = sum(1 for item in position_overlays if item.get("suspicious_high_confidence"))
    suspicious_low = sum(1 for item in position_overlays if item.get("suspicious_low_confidence"))
    avg_selected = (
        sum(_safe_float(item.get("selected_score")) for item in position_overlays)
        / max(len(position_overlays), 1)
    )
    avg_judged = (
        sum(_safe_float(item.get("judged_selected_score")) for item in position_overlays)
        / max(len(position_overlays), 1)
    )
    avg_risk = (
        sum(_safe_float(item.get("features", {}).get("best_confusion_risk")) for item in position_overlays)
        / max(len(position_overlays), 1)
    )
    advice = "keep_matrix_choice"
    if suspicious_high:
        advice = "review_confident_confusion_pairs"
    elif suspicious_low:
        advice = "review_low_confidence_positions"

    return {
        "token_id": row.get("token_id"),
        "text_unit_id": row.get("text_unit_id"),
        "selected_text": row.get("selected_text") or "",
        "input_text": row.get("input_text") or "",
        "matrix_score": _safe_float(row.get("selected_score")),
        "judge_score": avg_judged,
        "average_selected_position_score": avg_selected,
        "average_confusion_risk": avg_risk,
        "position_count": len(position_overlays),
        "suspicious_high_confidence_count": suspicious_high,
        "suspicious_low_confidence_count": suspicious_low,
        "advice": advice,
        "positions": position_overlays,
        "trusted_as_final": False,
    }


def build_scribejudge_overlay(
    assembly_map: dict,
    settings: dict | None = None,
    base_dir: str = ".",
) -> dict:
    """Build a confusion-aware overlay for one N05 assembly map."""

    settings = settings or {}
    if settings.get("enabled") is False:
        return {
            "version": SCRIBEJUDGE_VERSION,
            "status": "disabled",
            "trusted_as_final": False,
            "rows": [],
            "summary": {"enabled": False},
        }

    confusion_sources = settings.get("confusion_sources") or DEFAULT_CONFUSION_SOURCES
    confusion_memory = ConfusionMemory.from_sources(confusion_sources, base_dir=base_dir)
    decision_rows = (assembly_map.get("decision_matrix") or {}).get("rows") or []
    combined_tokens = (assembly_map.get("combined_expert_output") or {}).get("word_tokens") or []
    combined_by_id = {str(token.get("token_id") or token.get("text_unit_id")): token for token in combined_tokens}
    rows = []
    for row in decision_rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("token_id") or row.get("text_unit_id"))
        combined_token = combined_by_id.get(key, {})
        rows.append(_token_overlay(row, combined_token, confusion_memory, settings))

    review_count = sum(1 for row in rows if row.get("advice") != "keep_matrix_choice")
    return {
        "version": SCRIBEJUDGE_VERSION,
        "status": "completed",
        "document_id": assembly_map.get("document_id"),
        "trusted_as_final": False,
        "decision_note": (
            "ScribeJudge v0.1 is an advisory layer. It uses confusion history "
            "and visible N05 evidence to flag risky confidence patterns before "
            "a trained meta-model takes over."
        ),
        "confusion_memory": confusion_memory.to_report(),
        "rows": rows,
        "summary": {
            "enabled": True,
            "token_count": len(rows),
            "review_count": review_count,
            "keep_count": len(rows) - review_count,
            "confusion_pair_count": confusion_memory.summary.get("confusion_pair_count", 0),
            "fake_high_confidence_alerts": sum(
                row.get("suspicious_high_confidence_count", 0) for row in rows
            ),
            "low_confidence_alerts": sum(
                row.get("suspicious_low_confidence_count", 0) for row in rows
            ),
        },
    }
