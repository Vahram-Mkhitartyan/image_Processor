"""First-pass decision matrix for combined N05 expert evidence.

The matrix is intentionally transparent. Every score is made from visible
terms so bad decisions can be inspected instead of treated like magic.
"""

from __future__ import annotations


DECISION_MATRIX_VERSION = "n05_decision_matrix_v0_1"


def _safe_float(value, default: float = 0.0) -> float:
    """Return a float without trusting loose JSON payloads."""

    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    """Return an int without trusting loose JSON payloads."""

    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _candidate_score(candidate: dict, settings: dict) -> tuple[float, dict]:
    """Score one position-level candidate using visible evidence terms."""

    base_score = _safe_float(candidate.get("score"))
    confidence = _safe_float(candidate.get("confidence"), base_score)
    source_count = _safe_int(candidate.get("source_count"), len(candidate.get("sources") or []))
    source_bonus = min(
        _safe_float(settings.get("max_source_agreement_bonus", 0.18), 0.18),
        max(0, source_count - 1)
        * _safe_float(settings.get("source_agreement_bonus", 0.06), 0.06),
    )
    single_source_penalty = (
        _safe_float(settings.get("single_source_penalty", 0.22), 0.22)
        if source_count <= 1
        else 0.0
    )
    score = (
        base_score * _safe_float(settings.get("score_weight", 0.72), 0.72)
        + confidence * _safe_float(settings.get("confidence_weight", 0.28), 0.28)
        + source_bonus
        - single_source_penalty
    )
    terms = {
        "base_score": base_score,
        "confidence": confidence,
        "source_count": source_count,
        "source_bonus": source_bonus,
        "single_source_penalty": single_source_penalty,
        "candidate_score": score,
    }
    return score, terms


def _position_candidates(position: dict, settings: dict) -> list[dict]:
    """Return scored candidates for one character position."""

    top_k = int(settings.get("top_k_per_position", 5))
    conflict_penalty = (
        _safe_float(settings.get("conflict_penalty", 0.08), 0.08)
        if position.get("conflicts")
        else 0.0
    )
    records = []
    for rank, candidate in enumerate(position.get("candidates") or [], start=1):
        if not isinstance(candidate, dict) or not candidate.get("char"):
            continue
        score, terms = _candidate_score(candidate, settings)
        score -= conflict_penalty
        terms["conflict_penalty"] = conflict_penalty
        terms["final_position_score"] = score
        records.append(
            {
                "index": position.get("index"),
                "char": candidate.get("char"),
                "rank": rank,
                "score": score,
                "source_scores": candidate.get("source_scores") or {},
                "sources": candidate.get("sources") or [],
                "terms": terms,
            }
        )
    if not records and bool(settings.get("allow_empty_position_placeholder", True)):
        records.append(
            {
                "index": position.get("index"),
                "char": "",
                "rank": 1,
                "score": -_safe_float(settings.get("empty_position_penalty", 0.65), 0.65),
                "source_scores": {},
                "sources": [],
                "terms": {
                    "empty_position_penalty": _safe_float(
                        settings.get("empty_position_penalty", 0.65),
                        0.65,
                    )
                },
            }
        )
    records.sort(key=lambda item: (-_safe_float(item.get("score")), item.get("rank", 999)))
    return records[:top_k]


def _beam_search_positions(positions: list[dict], settings: dict) -> list[dict]:
    """Build top word candidates from position-level candidates."""

    beam_width = int(settings.get("beam_width", 32))
    max_sequences = int(settings.get("max_word_candidates", 12))
    beams = [
        {
            "text": "",
            "score_sum": 0.0,
            "positions": [],
            "source_coverage": {},
        }
    ]
    for position in positions:
        next_beams = []
        choices = _position_candidates(position, settings)
        for beam in beams:
            for choice in choices:
                source_coverage = dict(beam.get("source_coverage") or {})
                for source in choice.get("sources") or []:
                    source_coverage[source] = source_coverage.get(source, 0) + 1
                next_beams.append(
                    {
                        "text": beam["text"] + str(choice.get("char") or ""),
                        "score_sum": (
                            _safe_float(beam.get("score_sum"))
                            + _safe_float(choice.get("score"))
                        ),
                        "positions": beam["positions"] + [choice],
                        "source_coverage": source_coverage,
                    }
                )
        beams = sorted(
            next_beams,
            key=lambda item: (
                -_safe_float(item.get("score_sum")),
                len(item.get("text") or ""),
                item.get("text") or "",
            ),
        )[:beam_width]

    output = []
    for rank, beam in enumerate(beams[:max_sequences], start=1):
        position_count = max(1, len(beam.get("positions") or []))
        average_score = _safe_float(beam.get("score_sum")) / position_count
        output.append(
            {
                "rank": rank,
                "text": beam.get("text") or "",
                "score": average_score,
                "score_sum": _safe_float(beam.get("score_sum")),
                "position_count": len(beam.get("positions") or []),
                "source_coverage": dict(sorted((beam.get("source_coverage") or {}).items())),
                "positions": beam.get("positions") or [],
            }
        )
    return output


def _decision_status(best_candidate: dict, token: dict, settings: dict) -> str:
    """Classify the decision as ready, weak, or empty."""

    if not best_candidate or not best_candidate.get("text"):
        return "empty"
    min_ready_score = _safe_float(settings.get("min_ready_score", 0.42), 0.42)
    min_coverage = _safe_float(settings.get("min_populated_ratio", 0.70), 0.70)
    position_count = max(1, _safe_int(token.get("position_count"), 0))
    populated = sum(
        1
        for position in token.get("character_candidates") or []
        if position.get("char")
    )
    if populated / position_count < min_coverage:
        return "weak_low_coverage"
    if _safe_float(best_candidate.get("score")) < min_ready_score:
        return "weak_low_score"
    return "provisional_ready"


def _token_decision(token: dict, settings: dict) -> dict:
    """Build one token-level decision row."""

    positions = [
        position
        for position in token.get("character_candidates") or []
        if isinstance(position, dict)
    ]
    candidates = _beam_search_positions(positions, settings)
    best = candidates[0] if candidates else {}
    return {
        "token_id": token.get("token_id"),
        "text_unit_id": token.get("text_unit_id"),
        "selected_path_id": token.get("selected_path_id"),
        "input_text": token.get("text") or "",
        "selected_text": best.get("text") or "",
        "letter_backup_sequence": token.get("letter_backup_sequence") or [],
        "backup_string": token.get("backup_string") or "",
        "selected_score": _safe_float(best.get("score")),
        "status": _decision_status(best, token, settings),
        "trusted_as_final": False,
        "word_candidates": candidates,
        "position_count": len(positions),
        "positions_with_conflicts": sum(1 for position in positions if position.get("conflicts")),
        "source_coverage": best.get("source_coverage") or {},
    }


def build_decision_matrix(
    combined_expert_output: dict,
    settings: dict | None = None,
) -> dict:
    """Build provisional final candidates from the combined expert output."""

    settings = settings or {}
    rows = [
        _token_decision(token, settings)
        for token in combined_expert_output.get("word_tokens") or []
        if isinstance(token, dict)
    ]
    ready_count = sum(row.get("status") == "provisional_ready" for row in rows)
    return {
        "version": DECISION_MATRIX_VERSION,
        "status": "completed",
        "document_id": combined_expert_output.get("document_id"),
        "trusted_as_final": False,
        "decision_note": (
            "First-pass matrix only. It ranks candidate words from expert "
            "agreement and confidence; N06 linguistic repair and future "
            "correctness-history learning are still allowed to override it."
        ),
        "rows": rows,
        "summary": {
            "token_count": len(rows),
            "provisional_ready_count": ready_count,
            "weak_or_empty_count": len(rows) - ready_count,
            "top_k_per_position": int(settings.get("top_k_per_position", 5)),
            "beam_width": int(settings.get("beam_width", 32)),
            "max_word_candidates": int(settings.get("max_word_candidates", 12)),
        },
    }
