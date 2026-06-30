"""Combined expert output for N05 assembly.

This layer is the handoff contract between N05 and later reasoning nodes. It
does not decide the final OCR result. It packages the strongest current text
hypothesis, backup letters, source agreement, and conflicts into one stable
word-token surface.
"""

from __future__ import annotations


COMBINED_OUTPUT_VERSION = "n05_combined_expert_output_v0_1"


def _safe_float(value, default: float = 0.0) -> float:
    """Convert loose expert scores into a JSON-safe float."""

    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _source_breakdown(candidate: dict) -> dict:
    """Return per-source score metadata for a fused letter candidate."""

    evidence = candidate.get("evidence") if isinstance(candidate.get("evidence"), dict) else {}
    source_scores = evidence.get("source_scores")
    if isinstance(source_scores, dict):
        return {
            str(source): _safe_float(score)
            for source, score in sorted(source_scores.items())
        }
    source = str(candidate.get("source") or "unknown")
    return {source: _safe_float(candidate.get("score"))}


def _candidate_sources(candidate: dict) -> list[str]:
    """Return the expert names that contributed to one candidate."""

    return [
        source
        for source, score in _source_breakdown(candidate).items()
        if _safe_float(score) > 0.0
    ]


def _normalize_candidate(candidate: dict, rank: int) -> dict:
    """Build one combined-output candidate row."""

    char = str(
        candidate.get("char")
        or candidate.get("letter")
        or candidate.get("label")
        or candidate.get("text")
        or ""
    )
    score = _safe_float(candidate.get("score"))
    confidence = _safe_float(candidate.get("confidence"), score)
    sources = _candidate_sources(candidate)
    return {
        "rank": int(candidate.get("rank") or rank),
        "char": char,
        "score": score,
        "confidence": confidence,
        "source_count": len(sources),
        "sources": sources,
        "source_scores": _source_breakdown(candidate),
    }


def _position_record(row: dict, top_k: int) -> dict:
    """Convert one letter-matrix row into an N06-ready character slot."""

    candidates = [
        _normalize_candidate(candidate, rank=index)
        for index, candidate in enumerate(row.get("candidates") or [], start=1)
        if isinstance(candidate, dict)
    ]
    candidates = [
        candidate
        for candidate in candidates
        if candidate.get("char")
    ][:top_k]
    best = candidates[0] if candidates else {}
    segment = row.get("segment") if isinstance(row.get("segment"), dict) else {}
    return {
        "index": int(row.get("position", 0)),
        "char": best.get("char", ""),
        "confidence": _safe_float(best.get("confidence")),
        "score": _safe_float(best.get("score")),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "segment_id": segment.get("segment_id"),
        "bbox": segment.get("bbox"),
        "mask_crop_path": segment.get("mask_crop_path"),
        "visual_crop_path": segment.get("visual_crop_path"),
        "status": row.get("status"),
        "conflicts": row.get("conflicts") or [],
        "evidence_sources": row.get("evidence_sources") or [],
    }


def _unit_confidence(positions: list[dict]) -> float:
    """Return the average best-position confidence for one token."""

    scored = [
        _safe_float(position.get("confidence"))
        for position in positions
        if position.get("char")
    ]
    if not scored:
        return 0.0
    return sum(scored) / len(scored)


def _unit_source_coverage(positions: list[dict]) -> dict:
    """Summarize which experts contributed across a token."""

    counts: dict[str, int] = {}
    for position in positions:
        for candidate in position.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            for source in candidate.get("sources") or []:
                counts[str(source)] = counts.get(str(source), 0) + 1
    return dict(sorted(counts.items()))


def _letter_backup_sequence(positions: list[dict]) -> list[dict]:
    """Build compact letter-plus-backups records for N06 reconstruction."""

    sequence = []
    for position in positions:
        candidates = [
            candidate
            for candidate in position.get("candidates") or []
            if isinstance(candidate, dict) and candidate.get("char")
        ]
        best = candidates[0] if candidates else {}
        backups = []
        seen = {best.get("char")} if best.get("char") else set()
        for candidate in candidates[1:]:
            char = candidate.get("char")
            if not char or char in seen:
                continue
            seen.add(char)
            backups.append(
                {
                    "char": char,
                    "score": _safe_float(candidate.get("score")),
                    "confidence": _safe_float(candidate.get("confidence")),
                    "sources": candidate.get("sources") or [],
                }
            )
        sequence.append(
            {
                "index": position.get("index"),
                "l": best.get("char", ""),
                "b": backups,
            }
        )
    return sequence


def _backup_string(sequence: list[dict]) -> str:
    """Serialize letter-plus-backups in a compact human-readable form."""

    chunks = []
    for record in sequence:
        letters = [f"l:{record.get('l', '')}"]
        for backup in record.get("b") or []:
            letters.append(f"b:{backup.get('char', '')}")
        chunks.append(",".join(letters))
    return "|".join(chunks)


def build_combined_expert_output(
    assembly_map: dict,
    settings: dict | None = None,
) -> dict:
    """Build the unified expert output from the current assembly map."""

    settings = settings or {}
    top_k = int(settings.get("top_k_per_position", 8))
    word_tokens = []
    total_positions = 0
    populated_positions = 0
    conflict_count = 0

    for unit_index, unit_matrix in enumerate(assembly_map.get("letter_matrix") or [], start=1):
        positions = [
            _position_record(row, top_k=top_k)
            for row in unit_matrix.get("rows") or []
            if isinstance(row, dict)
        ]
        text = "".join(position.get("char", "") for position in positions)
        total_positions += len(positions)
        populated_positions += sum(1 for position in positions if position.get("char"))
        conflict_count += sum(1 for position in positions if position.get("conflicts"))
        token_id = str(
            unit_matrix.get("text_unit_id")
            or f"unit_{unit_index:04d}"
        )
        letter_backup_sequence = _letter_backup_sequence(positions)
        word_tokens.append(
            {
                "token_id": token_id,
                "text_unit_id": unit_matrix.get("text_unit_id"),
                "group_id": unit_matrix.get("group_id"),
                "selected_path_id": unit_matrix.get("selected_path_id"),
                "text": text,
                "source": "n05_combined_expert_output",
                "confidence": _unit_confidence(positions),
                "decoded_length": len(text),
                "position_count": len(positions),
                "character_candidates": positions,
                "letter_backup_sequence": letter_backup_sequence,
                "backup_string": _backup_string(letter_backup_sequence),
                "source_coverage": _unit_source_coverage(positions),
                "status": "candidate_ready" if text else "empty_candidate",
            }
        )

    return {
        "version": COMBINED_OUTPUT_VERSION,
        "status": "completed",
        "document_id": assembly_map.get("document_id"),
        "trusted_as_final": False,
        "decision_note": (
            "This is a combined candidate surface, not final OCR. N06 and the "
            "future decision formula should consume the backups and conflicts."
        ),
        "word_tokens": word_tokens,
        "summary": {
            "token_count": len(word_tokens),
            "position_count": total_positions,
            "populated_position_count": populated_positions,
            "empty_position_count": total_positions - populated_positions,
            "positions_with_conflicts": conflict_count,
            "top_k_per_position": top_k,
        },
    }
