"""Morpheme candidate records for future Armenian morphology analysis."""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class MorphemeCandidate:
    """One possible structural explanation for a word part."""

    text: str
    role: str
    start: int
    end: int
    score: float = 0.0
    explanation: str = ""

    def to_dict(self) -> dict:
        """Return a JSON-safe representation."""

        return asdict(self)
