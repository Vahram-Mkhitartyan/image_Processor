"""Suffix rule placeholders for N06 Armenian morphology.

This module starts deliberately small. The real rule tables will grow from
verified Armenian grammar and document-domain examples instead of guesses.
"""

from __future__ import annotations


COMMON_SURNAME_SUFFIXES = ("յան", "եան")


def detect_known_suffixes(word: str) -> list[dict]:
    """Return obvious suffix hints without claiming a full morphology parse."""

    hints = []
    for suffix in COMMON_SURNAME_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix):
            hints.append(
                {
                    "suffix": suffix,
                    "role": "surname_suffix_candidate",
                    "start": len(word) - len(suffix),
                    "end": len(word),
                    "confidence": 0.35,
                }
            )
    return hints
