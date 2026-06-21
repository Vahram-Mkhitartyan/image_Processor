"""JSON-safe candidate effects and final ScriLog result records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .constants import SCRILOG_VERSION
from .facts import ScriLogFactBase
from .signature import ScriLogSignature

@dataclass
class ScriLogCandidateEffect:
    """
    Represents ScriLog's effect on a possible letter candidate.

    This is for later N05 fusion.

    Examples:
        boost("ա") because looped structure matched
        block("թ") because required junction evidence is absent
        weaken("գ") because endpoint topology differs

    For v0.1, this may stay empty until we add Armenian class profiles.
    """

    class_label: str
    effect: str
    strength: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_label": self.class_label,
            "effect": self.effect,
            "strength": self.strength,
            "reason": self.reason,
        }


@dataclass
class ScriLogResult:
    """
    Final ScriLog output for one reconstructed glyph/unit.

    This is what N05 will eventually consume.

    It contains:
        - the original structural signature
        - all facts
        - derived families
        - warnings
        - candidate effects
        - explanations
        - rule engine stats
    """

    unit_id: str
    selected_hypothesis_id: str

    signature: ScriLogSignature
    facts: ScriLogFactBase

    derived_families: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    candidate_effects: List[ScriLogCandidateEffect] = field(default_factory=list)
    explanation: List[str] = field(default_factory=list)

    engine_passes: int = 0
    rule_fire_count: int = 0

    def boosted_candidates(self) -> List[ScriLogCandidateEffect]:
        return [
            effect
            for effect in self.candidate_effects
            if effect.effect == "boost"
        ]

    def blocked_candidates(self) -> List[ScriLogCandidateEffect]:
        return [
            effect
            for effect in self.candidate_effects
            if effect.effect == "block"
        ]

    def weakened_candidates(self) -> List[ScriLogCandidateEffect]:
        return [
            effect
            for effect in self.candidate_effects
            if effect.effect == "weaken"
        ]

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @property
    def has_candidate_effects(self) -> bool:
        return len(self.candidate_effects) > 0

    @property
    def is_structurally_unsafe(self) -> bool:
        """
        True if ScriLog found an unsafe reconstruction condition.

        For now, this mainly catches the same dangerous case
        we discussed for selection logic:

            downstream bridge required
            but downstream bridge not completed
        """

        if self.signature.downstream_bridge_required:
            if not self.signature.downstream_bridge_completed:
                return True

        if "bridge_required_but_not_completed" in self.warnings:
            return True

        if "unsafe_reconstruction_candidate" in self.warnings:
            return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON-safe export for reports/debugging/N05.
        """

        return {
            "unit_id": self.unit_id,
            "selected_hypothesis_id": self.selected_hypothesis_id,

            "scrilog_version": SCRILOG_VERSION,

            "signature": self.signature.to_dict(),

            "facts": self.facts.as_strings(),
            "fact_details": self.facts.as_dicts(),

            "derived_families": sorted(set(self.derived_families)),
            "warnings": sorted(set(self.warnings)),

            "candidate_effects": [
                effect.to_dict()
                for effect in self.candidate_effects
            ],

            "boosted_candidates": [
                effect.to_dict()
                for effect in self.boosted_candidates()
            ],

            "blocked_candidates": [
                effect.to_dict()
                for effect in self.blocked_candidates()
            ],

            "weakened_candidates": [
                effect.to_dict()
                for effect in self.weakened_candidates()
            ],

            "explanation": list(dict.fromkeys(self.explanation)),

            "status": {
                "has_warnings": self.has_warnings,
                "has_candidate_effects": self.has_candidate_effects,
                "is_structurally_unsafe": self.is_structurally_unsafe,
            },

            "engine": {
                "passes": self.engine_passes,
                "rule_fire_count": self.rule_fire_count,
                "fact_count": len(self.facts),
            },
        }

    def __repr__(self) -> str:
        return (
            "ScriLogResult("
            f"unit_id={self.unit_id!r}, "
            f"families={len(self.derived_families)}, "
            f"warnings={len(self.warnings)}, "
            f"effects={len(self.candidate_effects)}"
            ")"
        )
