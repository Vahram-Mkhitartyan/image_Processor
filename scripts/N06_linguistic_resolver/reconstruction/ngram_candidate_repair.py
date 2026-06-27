"""N-gram guided repair search using N05 backup letters.

The repair search is intentionally bounded: N06 is not allowed to invent random
letters yet. It may only try alternate characters that upstream experts already
placed in the per-position backup list.
"""

from __future__ import annotations

import itertools

from scripts.N06_linguistic_resolver.normalization.armenian_word_normalizer import (
    normalize_armenian_word,
)


def _safe_float(value, default: float = 0.0) -> float:
    """Return a safe float for loose JSON candidate records."""

    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_char(candidate: dict) -> str:
    """Extract a character from any known candidate shape."""

    return str(
        candidate.get("char")
        or candidate.get("letter")
        or candidate.get("label")
        or candidate.get("class_label")
        or candidate.get("text")
        or ""
    )


def _candidate_confidence(candidate: dict) -> float:
    """Extract confidence/score from a character candidate."""

    return _safe_float(
        candidate.get("confidence")
        if candidate.get("confidence") is not None
        else candidate.get("score")
        if candidate.get("score") is not None
        else candidate.get("probability"),
        0.0,
    )


def _normalize_one_char(value: str, normalization_settings: dict) -> str:
    """Normalize an alternate candidate and keep only one-character options."""

    normalized = normalize_armenian_word(value, normalization_settings)
    return normalized if len(normalized) == 1 else ""


def _position_records(
    text: str,
    character_candidates: list[dict],
    settings: dict,
    normalization_settings: dict,
) -> list[dict]:
    """Normalize per-position N05 backup letters for repair search."""

    normalized_text = normalize_armenian_word(text, normalization_settings)
    by_index = {}
    for record in character_candidates or []:
        if not isinstance(record, dict):
            continue
        try:
            index = int(record.get("index", record.get("position")))
        except (TypeError, ValueError):
            continue
        by_index[index] = record

    output = []
    max_alternatives = int(settings.get("max_alternatives_per_position", 5))
    for index, observed in enumerate(normalized_text):
        record = by_index.get(index, {})
        confidence = _safe_float(
            record.get("confidence")
            if record.get("confidence") is not None
            else record.get("score"),
            1.0,
        )
        raw_candidates = record.get("candidates") or record.get("alternatives") or []
        alternatives = []
        seen = {observed}
        for candidate in raw_candidates:
            if isinstance(candidate, str):
                char = _normalize_one_char(candidate, normalization_settings)
                candidate_confidence = 0.0
            elif isinstance(candidate, dict):
                char = _normalize_one_char(_candidate_char(candidate), normalization_settings)
                candidate_confidence = _candidate_confidence(candidate)
            else:
                continue
            if not char or char in seen:
                continue
            seen.add(char)
            alternatives.append(
                {
                    "char": char,
                    "confidence": candidate_confidence,
                    "source_candidate": candidate,
                }
            )
        alternatives.sort(key=lambda row: (-row["confidence"], row["char"]))
        output.append(
            {
                "index": index,
                "observed_char": observed,
                "confidence": confidence,
                "alternatives": alternatives[:max_alternatives],
            }
        )
    output.sort(key=lambda row: (row["confidence"], row["index"]))
    return output


def suggest_ngram_repairs(
    text: str,
    character_candidates: list[dict],
    ngram_model,
    settings: dict | None = None,
    normalization_settings: dict | None = None,
) -> dict:
    """Return N05-backed repair suggestions ranked by n-gram plausibility."""

    settings = settings or {}
    normalization_settings = normalization_settings or {}
    normalized_text = normalize_armenian_word(text, normalization_settings)
    original_score = ngram_model.score_word(text, normalization_settings)
    if not normalized_text or not character_candidates:
        return {
            "enabled": bool(settings.get("enabled", True)),
            "status": "no_character_backups",
            "original_text": text,
            "normalized_text": normalized_text,
            "original_z_score": original_score["z_score"],
            "suggestions": [],
        }

    max_changes = int(settings.get("max_changes", 2))
    max_positions = int(settings.get("max_positions_to_consider", 6))
    max_suggestions = int(settings.get("max_suggestions", 12))
    min_improvement = float(settings.get("min_z_improvement", 0.15))
    positions = [
        row for row in _position_records(
            text,
            character_candidates,
            settings,
            normalization_settings,
        )
        if row["alternatives"]
    ][:max_positions]

    suggestions = []
    base_chars = list(normalized_text)
    for change_count in range(1, max_changes + 1):
        for position_group in itertools.combinations(positions, change_count):
            alternative_lists = [position["alternatives"] for position in position_group]
            for replacements in itertools.product(*alternative_lists):
                chars = base_chars.copy()
                changes = []
                replacement_confidence = 1.0
                for position, replacement in zip(position_group, replacements):
                    chars[position["index"]] = replacement["char"]
                    replacement_confidence *= max(0.001, replacement["confidence"])
                    changes.append(
                        {
                            "index": position["index"],
                            "from": position["observed_char"],
                            "to": replacement["char"],
                            "position_confidence": position["confidence"],
                            "replacement_confidence": replacement["confidence"],
                        }
                    )
                repaired = "".join(chars)
                score = ngram_model.score_word(repaired, normalization_settings)
                improvement = score["z_score"] - original_score["z_score"]
                if improvement < min_improvement:
                    continue
                suggestions.append(
                    {
                        "text": repaired,
                        "change_count": change_count,
                        "z_score": score["z_score"],
                        "z_improvement": improvement,
                        "ngram_confidence": score["confidence"],
                        "replacement_confidence_product": replacement_confidence,
                        "is_gibberish_like": score["is_gibberish_like"],
                        "changes": changes,
                    }
                )

    suggestions.sort(
        key=lambda row: (
            -row["z_score"],
            row["change_count"],
            -row["z_improvement"],
            -row["replacement_confidence_product"],
            row["text"],
        )
    )
    return {
        "enabled": bool(settings.get("enabled", True)),
        "status": "completed",
        "original_text": text,
        "normalized_text": normalized_text,
        "original_z_score": original_score["z_score"],
        "positions_considered": positions,
        "suggestions": suggestions[:max_suggestions],
    }
