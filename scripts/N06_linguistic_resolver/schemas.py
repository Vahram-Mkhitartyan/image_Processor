"""JSON contracts for N06 linguistic resolution."""

from __future__ import annotations


NODE_NAME = "N06_linguistic_resolver"
NODE_VERSION = "0.1.0"


def make_word_token(
    token_id: str,
    text: str,
    source: str = "manual",
    candidates: list[dict] | None = None,
    character_candidates: list[dict] | None = None,
    evidence: dict | None = None,
) -> dict:
    """Build one word token input record for N06."""

    return {
        "token_id": str(token_id),
        "text": str(text or ""),
        "source": source,
        "candidates": candidates or [],
        "character_candidates": character_candidates or [],
        "evidence": evidence or {},
    }


def make_ngram_evidence(
    text: str,
    normalized_text: str,
    status: str,
    score: float,
    confidence: float,
    details: dict,
) -> dict:
    """Build the anti-gibberish letter n-gram evidence block."""

    return {
        "source": "letter_ngram_stats",
        "status": status,
        "text": text,
        "normalized_text": normalized_text,
        "score": float(score),
        "confidence": float(confidence),
        "details": details,
    }


def make_resolved_word(
    token: dict,
    normalized_text: str,
    evidence_sources: list[dict],
    warnings: list[str] | None = None,
) -> dict:
    """Build one N06 resolved-word candidate surface."""

    return {
        "token_id": token.get("token_id"),
        "source": token.get("source"),
        "input_text": token.get("text", ""),
        "normalized_text": normalized_text,
        "evidence_sources": evidence_sources,
        "warnings": warnings or [],
        "trusted_as_final": False,
    }
