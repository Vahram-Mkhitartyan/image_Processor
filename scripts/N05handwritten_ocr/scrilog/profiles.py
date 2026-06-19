"""Armenian class profiles and their symbolic candidate effects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .facts import ScriLogFactBase
from .results import ScriLogCandidateEffect

@dataclass
class ScriLogClassProfile:
    """
    Optional symbolic profile for one candidate class.

    This is NOT active Armenian logic yet.
    It is the mechanism we will use later.

    Example future profile:

        ScriLogClassProfile(
            class_label="թ",
            required_families=["ascender"],
            forbidden_families=["midline_compact"],
            boosted_families=["vertical_dominant"],
            weakened_families=["no_loop"],
        )

    Meaning:
        - If required family is missing -> block candidate.
        - If forbidden family is present -> block candidate.
        - If boosted family is present -> boost candidate.
        - If weakened family is present -> weaken candidate.

    For v0.1:
        profiles can be empty.
    """

    class_label: str

    required_families: List[str] = field(default_factory=list)
    forbidden_families: List[str] = field(default_factory=list)

    boosted_families: List[str] = field(default_factory=list)
    weakened_families: List[str] = field(default_factory=list)

    base_weight: float = 1.0

    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_label": self.class_label,
            "required_families": list(self.required_families),
            "forbidden_families": list(self.forbidden_families),
            "boosted_families": list(self.boosted_families),
            "weakened_families": list(self.weakened_families),
            "base_weight": self.base_weight,
            "notes": list(self.notes),
        }


class ScriLogProfileEvaluator:
    """
    Evaluates class profiles against derived ScriLog facts.

    Input:
        facts:
            family(looped)
            family(right_exit)
            family(no_loop)
            warning(...)

        profiles:
            symbolic descriptions of candidate classes

    Output:
        ScriLogCandidateEffect objects:
            block / boost / weaken

    Important:
        This does not decide final OCR.
        It only produces symbolic evidence for N05 fusion.
    """

    def __init__(
        self,
        profiles: Optional[List[ScriLogClassProfile]] = None,
    ) -> None:
        self.profiles = profiles or []

    def evaluate(
        self,
        facts: ScriLogFactBase,
    ) -> List[ScriLogCandidateEffect]:
        effects: List[ScriLogCandidateEffect] = []

        for profile in self.profiles:
            effects.extend(
                self._evaluate_required_families(
                    profile=profile,
                    facts=facts,
                )
            )

            effects.extend(
                self._evaluate_forbidden_families(
                    profile=profile,
                    facts=facts,
                )
            )

            effects.extend(
                self._evaluate_boosted_families(
                    profile=profile,
                    facts=facts,
                )
            )

            effects.extend(
                self._evaluate_weakened_families(
                    profile=profile,
                    facts=facts,
                )
            )

        return self._deduplicate_effects(effects)

    # --------------------------------------------------------
    # Required families
    # --------------------------------------------------------

    def _evaluate_required_families(
        self,
        profile: ScriLogClassProfile,
        facts: ScriLogFactBase,
    ) -> List[ScriLogCandidateEffect]:
        effects: List[ScriLogCandidateEffect] = []

        for family in profile.required_families:
            if facts.has("family", family):
                continue

            effects.append(
                ScriLogCandidateEffect(
                    class_label=profile.class_label,
                    effect="block",
                    strength=1.0 * profile.base_weight,
                    reason=f"missing_required_family:{family}",
                )
            )

        return effects

    # --------------------------------------------------------
    # Forbidden families
    # --------------------------------------------------------

    def _evaluate_forbidden_families(
        self,
        profile: ScriLogClassProfile,
        facts: ScriLogFactBase,
    ) -> List[ScriLogCandidateEffect]:
        effects: List[ScriLogCandidateEffect] = []

        for family in profile.forbidden_families:
            if not facts.has("family", family):
                continue

            effects.append(
                ScriLogCandidateEffect(
                    class_label=profile.class_label,
                    effect="block",
                    strength=1.0 * profile.base_weight,
                    reason=f"forbidden_family_present:{family}",
                )
            )

        return effects

    # --------------------------------------------------------
    # Boosted families
    # --------------------------------------------------------

    def _evaluate_boosted_families(
        self,
        profile: ScriLogClassProfile,
        facts: ScriLogFactBase,
    ) -> List[ScriLogCandidateEffect]:
        effects: List[ScriLogCandidateEffect] = []

        for family in profile.boosted_families:
            if not facts.has("family", family):
                continue

            effects.append(
                ScriLogCandidateEffect(
                    class_label=profile.class_label,
                    effect="boost",
                    strength=0.25 * profile.base_weight,
                    reason=f"matched_boost_family:{family}",
                )
            )

        return effects

    # --------------------------------------------------------
    # Weakened families
    # --------------------------------------------------------

    def _evaluate_weakened_families(
        self,
        profile: ScriLogClassProfile,
        facts: ScriLogFactBase,
    ) -> List[ScriLogCandidateEffect]:
        effects: List[ScriLogCandidateEffect] = []

        for family in profile.weakened_families:
            if not facts.has("family", family):
                continue

            effects.append(
                ScriLogCandidateEffect(
                    class_label=profile.class_label,
                    effect="weaken",
                    strength=0.25 * profile.base_weight,
                    reason=f"matched_weaken_family:{family}",
                )
            )

        return effects

    # --------------------------------------------------------
    # Deduplication
    # --------------------------------------------------------

    def _deduplicate_effects(
        self,
        effects: List[ScriLogCandidateEffect],
    ) -> List[ScriLogCandidateEffect]:
        """
        Remove exact duplicate effects.

        If the same class/effect/reason appears twice,
        keep only one.

        Later we can make this smarter:
            - combine boosts
            - let block dominate weaken/boost
            - rank candidate effects
        """

        seen = set()
        unique: List[ScriLogCandidateEffect] = []

        for effect in effects:
            key = (
                effect.class_label,
                effect.effect,
                effect.reason,
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(effect)

        return unique
