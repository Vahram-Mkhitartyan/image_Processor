"""First-pass noun analysis placeholder for N06."""

from __future__ import annotations

from .suffix_rules import detect_known_suffixes


def analyze_noun_surface(word: str) -> dict:
    """Return conservative noun-like hints for a normalized Armenian word."""

    suffix_hints = detect_known_suffixes(word)
    return {
        "analyzer": "noun_analyzer",
        "status": "placeholder",
        "word": word,
        "suffix_hints": suffix_hints,
        "proper_name_like": any(
            hint.get("role") == "surname_suffix_candidate"
            for hint in suffix_hints
        ),
    }
