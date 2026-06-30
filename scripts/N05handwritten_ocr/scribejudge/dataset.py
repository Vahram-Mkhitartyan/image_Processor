"""Dataset extraction helpers for future ScribeJudge training.

Rows are JSONL-friendly. The target can be attached later when the synthetic
word generator knows the truth. Until then this still gives us feature dumps for
manual audits and meta-model design.
"""

from __future__ import annotations

import json
from pathlib import Path

from .judge import build_scribejudge_overlay


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _truth_payload(value) -> tuple[str | None, list[str] | None]:
    """Normalize truth records accepted by the dataset builder."""

    if isinstance(value, dict):
        text = value.get("text")
        tokens = value.get("tokens")
        return (
            str(text) if text is not None else None,
            [str(token) for token in tokens] if isinstance(tokens, list) else None,
        )
    if value is None:
        return None, None
    return str(value), None


def _position_truth_records(positions: list[dict], truth_tokens: list[str] | None) -> list[dict]:
    """Attach position-level truth and backup-rank targets."""

    if not truth_tokens:
        return []
    records = []
    for index, position in enumerate(positions):
        truth_char = truth_tokens[index] if index < len(truth_tokens) else ""
        selected_char = str(position.get("selected_char") or "")
        correct_rank = None
        if selected_char == truth_char:
            correct_rank = 1
        else:
            for candidate in position.get("top_confusion_alternatives") or []:
                if str(candidate.get("char") or "") == truth_char:
                    correct_rank = int(candidate.get("matrix_rank") or 999)
                    break
        records.append(
            {
                "index": position.get("index", index),
                "selected_char": selected_char,
                "truth_char": truth_char,
                "selected_correct": bool(selected_char == truth_char),
                "correct_candidate_rank": correct_rank,
                "correct_in_backups": bool(correct_rank is not None and correct_rank > 1),
                "should_promote_backup": bool(correct_rank is not None and correct_rank > 1),
                "truth_missing_from_candidates": bool(correct_rank is None),
            }
        )
    return records


def _token_training_row(token_row: dict, truth=None) -> dict:
    selected_text = str(token_row.get("selected_text") or "")
    truth_text, truth_tokens = _truth_payload(truth)
    position_targets = _position_truth_records(token_row.get("positions") or [], truth_tokens)
    selected_correct_count = sum(1 for item in position_targets if item.get("selected_correct"))
    return {
        "token_id": token_row.get("token_id"),
        "text_unit_id": token_row.get("text_unit_id"),
        "selected_text": selected_text,
        "truth_text": truth_text,
        "is_exact": None if truth_text is None else selected_text == truth_text,
        "truth_tokens": truth_tokens,
        "position_targets": position_targets,
        "char_accuracy": (
            None
            if not position_targets
            else selected_correct_count / max(len(position_targets), 1)
        ),
        "backup_recovery_opportunity_count": sum(
            1 for item in position_targets if item.get("correct_in_backups")
        ),
        "truth_missing_from_candidates_count": sum(
            1 for item in position_targets if item.get("truth_missing_from_candidates")
        ),
        "features": {
            "matrix_score": _safe_float(token_row.get("matrix_score")),
            "judge_score": _safe_float(token_row.get("judge_score")),
            "average_selected_position_score": _safe_float(
                token_row.get("average_selected_position_score")
            ),
            "average_confusion_risk": _safe_float(token_row.get("average_confusion_risk")),
            "position_count": _safe_float(token_row.get("position_count")),
            "suspicious_high_confidence_count": _safe_float(
                token_row.get("suspicious_high_confidence_count")
            ),
            "suspicious_low_confidence_count": _safe_float(
                token_row.get("suspicious_low_confidence_count")
            ),
        },
        "positions": token_row.get("positions") or [],
        "advice": token_row.get("advice"),
    }


def build_scribejudge_rows(
    assembly_map: dict,
    settings: dict | None = None,
    truth_by_token_id: dict[str, str | dict] | None = None,
    base_dir: str = ".",
) -> list[dict]:
    """Return JSONL-ready judge rows from an N05 assembly map."""

    overlay = build_scribejudge_overlay(assembly_map, settings=settings, base_dir=base_dir)
    truth_by_token_id = truth_by_token_id or {}
    rows = []
    for token_row in overlay.get("rows") or []:
        token_id = str(token_row.get("token_id") or token_row.get("text_unit_id") or "")
        rows.append(_token_training_row(token_row, truth=truth_by_token_id.get(token_id)))
    return rows


def write_scribejudge_jsonl(rows: list[dict], output_path: str | Path) -> str:
    """Write ScribeJudge rows as JSONL."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            file.write("\n")
    return str(path)
