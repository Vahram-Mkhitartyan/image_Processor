"""Dataclasses for universal condition verdicts and MoE routing advice."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DamageCandidate:
    """One possible damage label and confidence."""

    label: str
    confidence: float

    def to_dict(self):
        return asdict(self)


@dataclass
class ConditionVerdict:
    """Universal condition output consumed by MoE experts."""

    condition: str
    repair_needed: bool
    primary_damage: str
    confidence: float
    top_damage_candidates: list[DamageCandidate] = field(default_factory=list)
    severity: float = 0.0
    source: str = "unknown"
    notes: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        data = asdict(self)
        data["top_damage_candidates"] = [
            candidate.to_dict()
            if hasattr(candidate, "to_dict")
            else dict(candidate)
            for candidate in self.top_damage_candidates
        ]
        return data


@dataclass
class ExpertRoutingAdvice:
    """Routing advice for MoE experts after condition detection."""

    scribetrace_allowed_defenses: list[str] = field(default_factory=list)
    expert_weights: dict[str, float] = field(default_factory=dict)
    manual_review: bool = False
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)