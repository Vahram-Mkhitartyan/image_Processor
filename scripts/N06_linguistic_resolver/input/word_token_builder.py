"""Build N06 word-token inputs from simple text or N05-like records."""

from __future__ import annotations

from scripts.N06_linguistic_resolver.schemas import make_word_token


def build_tokens_from_words(words: list[str], source: str = "manual") -> list[dict]:
    """Convert plain strings into N06 word-token records."""

    return [
        make_word_token(
            token_id=f"token_{index:04d}",
            text=word,
            source=source,
        )
        for index, word in enumerate(words, start=1)
    ]


def _extract_character_candidates(unit: dict) -> list[dict]:
    """Extract per-character candidates from known N05-ish shapes."""

    for key in ("character_candidates", "position_candidates", "letter_candidates"):
        if isinstance(unit.get(key), list):
            return unit[key]

    matrix = unit.get("letter_matrix") or unit.get("candidate_matrix") or {}
    rows = matrix.get("rows") if isinstance(matrix, dict) else None
    if not isinstance(rows, list):
        return []

    records = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        candidates = row.get("candidates") or []
        best = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
        records.append(
            {
                "index": int(row.get("position", row_index)),
                "char": best.get("char") or best.get("letter") or best.get("text"),
                "confidence": best.get("confidence") or best.get("score"),
                "candidates": candidates,
            }
        )
    return records


def build_tokens_from_n05_payload(payload: dict) -> list[dict]:
    """Extract candidate word-like strings from an N05 result payload.

    This is intentionally permissive because N05 is still evolving. The builder
    looks for common candidate fields and preserves their original row as
    evidence for later debugging.
    """

    tokens = []
    raw_units = (
        payload.get("handwritten_text_units")
        or payload.get("text_units")
        or payload.get("printed_text_units")
        or []
    )
    for unit_index, unit in enumerate(raw_units, start=1):
        candidates = []
        for key in ("word_level_candidates", "candidates", "reconstructed_candidates"):
            if isinstance(unit.get(key), list):
                candidates.extend(unit[key])
        text = (
            unit.get("selected_text")
            or unit.get("text")
            or unit.get("raw_text")
            or ""
        )
        if not text and candidates:
            first = candidates[0]
            if isinstance(first, dict):
                text = first.get("text") or first.get("word") or ""
        tokens.append(
            make_word_token(
                token_id=str(unit.get("unit_id") or unit.get("text_unit_id") or f"unit_{unit_index:04d}"),
                text=text,
                source="n05_payload",
                candidates=candidates,
                character_candidates=_extract_character_candidates(unit),
                evidence={"source_unit": unit},
            )
        )
    return tokens
