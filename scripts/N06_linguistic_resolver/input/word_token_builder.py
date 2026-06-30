"""Build N06 word-token inputs from simple text or N05-like records."""

from __future__ import annotations

from scripts.N06_linguistic_resolver.schemas import make_word_token


def _compact_label_to_char(value: str) -> str:
    """Normalize compact N05 backup labels into one token-position string."""

    text = str(value or "").strip()
    if text in {"Եվ", "ԵՒ", "եվ", "եւ"}:
        return "և"
    return text


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


def build_token_from_compact_candidate_string(
    compact: str,
    token_id: str = "compact_0001",
    source: str = "compact_candidate_string",
) -> dict:
    """Build one N06 token from `l:ա,b:բ,b:գ|l:ր,b:ռ` backup syntax.

    This mirrors the compact N05 handoff shape:

        l:<selected>,b:<backup1>,b:<backup2>|l:<selected>,b:<backup1>

    The selected `l:` values are joined into the token text. Every position
    preserves the selected letter plus backups as `character_candidates`, which
    lets `ngram_candidate_repair` try bounded substitutions.
    """

    character_candidates = []
    selected_chars = []
    entries = [
        entry.strip()
        for entry in str(compact or "").replace("\n", "|").split("|")
        if entry.strip()
    ]

    for index, entry in enumerate(entries):
        selected = ""
        candidates = []
        for part in entry.split(","):
            if ":" not in part:
                continue
            kind, raw_value = part.split(":", 1)
            kind = kind.strip().lower()
            char = _compact_label_to_char(raw_value)
            if not char:
                continue
            if kind == "l":
                selected = char
                candidates.insert(0, {"char": char, "confidence": 1.0, "role": "selected"})
            elif kind == "b":
                candidates.append({"char": char, "confidence": 0.5, "role": "backup"})

        if not selected and candidates:
            selected = candidates[0]["char"]
        if not selected:
            continue

        selected_chars.append(selected)
        seen = set()
        deduped_candidates = []
        for candidate in candidates:
            char = candidate["char"]
            if char in seen:
                continue
            seen.add(char)
            deduped_candidates.append(candidate)
        character_candidates.append(
            {
                "index": len(selected_chars) - 1,
                "char": selected,
                "confidence": 1.0,
                "candidates": deduped_candidates,
            }
        )

    return make_word_token(
        token_id=token_id,
        text="".join(selected_chars),
        source=source,
        character_candidates=character_candidates,
        evidence={"compact_candidate_string": compact},
    )


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

    combined_output = (
        payload.get("combined_expert_output")
        if isinstance(payload.get("combined_expert_output"), dict)
        else (payload.get("assembly") or {}).get("combined_expert_output")
        if isinstance(payload.get("assembly"), dict)
        else {}
    )
    combined_tokens = (
        combined_output.get("word_tokens")
        if isinstance(combined_output, dict)
        else None
    )
    if isinstance(combined_tokens, list) and combined_tokens:
        tokens = []
        for token_index, token in enumerate(combined_tokens, start=1):
            if not isinstance(token, dict):
                continue
            tokens.append(
                make_word_token(
                    token_id=str(token.get("token_id") or f"combined_{token_index:04d}"),
                    text=token.get("text") or "",
                    source="n05_combined_expert_output",
                    candidates=token.get("word_candidates") or [],
                    character_candidates=token.get("character_candidates") or [],
                    evidence={
                        "source_unit": token,
                        "combined_output_version": combined_output.get("version"),
                    },
                )
            )
        return tokens

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
