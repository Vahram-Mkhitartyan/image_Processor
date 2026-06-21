"""ScriLog orchestration from normalized signature through symbolic result."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .constants import DEFAULT_MAX_RULE_PASSES
from .engine import ScriLogEngine, ScriLogRule
from .facts import ScriLogFactBase
from .profiles import ScriLogClassProfile, ScriLogProfileEvaluator
from .results import ScriLogCandidateEffect, ScriLogResult
from .rules import ScriLogRuleFactory
from .signature import ScriLogSignature
from .signature_builder import ScriLogSignatureBuilder

class ScriLogParser:
    """
    Main public ScriLog object.

    This is the object N05/ScribeTrace will call.

    Input:
        ScribeTrace JSON-like payload after reconstruction/vector extraction.

    Process:
        1. Build ScriLogSignature
        2. Seed base facts from signature
        3. Run symbolic rules
        4. Evaluate optional class profiles
        5. Build ScriLogResult

    Usage:
        parser = ScriLogParser()
        result = parser.parse_scribetrace_payload(payload)
    """

    def __init__(
        self,
        rules: Optional[List[ScriLogRule]] = None,
        class_profiles: Optional[List[ScriLogClassProfile]] = None,
        max_rule_passes: int = DEFAULT_MAX_RULE_PASSES,
    ) -> None:
        self.signature_builder = ScriLogSignatureBuilder()

        self.engine = ScriLogEngine(
            rules=rules if rules is not None else ScriLogRuleFactory.build_default_rules(),
            max_passes=max_rule_passes,
        )

        self.profile_evaluator = ScriLogProfileEvaluator(
            profiles=class_profiles,
        )

    def parse_scribetrace_payload(
        self,
        payload: Dict[str, Any],
    ) -> ScriLogResult:
        """
        Parse one ScribeTrace payload into ScriLogResult.
        """

        signature = self.signature_builder.build(payload)

        facts = self._seed_facts_from_signature(signature)

        engine_passes, rule_fire_count = self.engine.run(
            facts=facts,
            signature=signature,
        )

        derived_families = facts.find_arg0("family")
        warnings = facts.find_arg0("warning")

        candidate_effects = self.profile_evaluator.evaluate(facts)

        explanation = self._build_explanation(
            signature=signature,
            facts=facts,
            candidate_effects=candidate_effects,
        )

        return ScriLogResult(
            unit_id=signature.unit_id,
            selected_hypothesis_id=signature.selected_hypothesis_id,
            signature=signature,
            facts=facts,
            derived_families=derived_families,
            warnings=warnings,
            candidate_effects=candidate_effects,
            explanation=explanation,
            engine_passes=engine_passes,
            rule_fire_count=rule_fire_count,
        )

    # --------------------------------------------------------
    # Fact seeding
    # --------------------------------------------------------

    def _seed_facts_from_signature(
        self,
        signature: ScriLogSignature,
    ) -> ScriLogFactBase:
        """
        Convert measured ScriLogSignature fields into base facts.

        These are not derived rule facts yet.
        These are direct facts from ScribeTrace geometry.
        """

        facts = ScriLogFactBase()

        # Identity
        facts.add(
            "unit",
            signature.unit_id,
            origin="signature",
        )

        facts.add(
            "selected_hypothesis",
            signature.selected_hypothesis_id,
            origin="signature",
        )

        # Core topology counts
        facts.add(
            "loop_count",
            signature.loop_count,
            origin="signature",
        )

        facts.add(
            "endpoint_count",
            signature.endpoint_count,
            origin="signature",
        )

        facts.add(
            "junction_count",
            signature.junction_count,
            origin="signature",
        )

        facts.add(
            "path_count",
            signature.path_count,
            origin="signature",
        )

        facts.add(
            "component_count",
            signature.component_count,
            origin="signature",
        )

        # Measured structure flags
        if signature.loop_count > 0:
            facts.add(
                "has_measured",
                "loop",
                origin="signature",
            )

        if signature.endpoint_count > 0:
            facts.add(
                "has_measured",
                "endpoint",
                origin="signature",
            )

        if signature.junction_count > 0:
            facts.add(
                "has_measured",
                "junction",
                origin="signature",
            )

        # Objective border-contact facts
        if signature.has_top_contact:
            facts.add(
                "has_top_contact",
                origin="signature",
            )

        if signature.has_bottom_contact:
            facts.add(
                "has_bottom_contact",
                origin="signature",
            )

        # Zones
        for zone_name, count in signature.endpoint_zones.items():
            facts.add(
                "endpoint_zone",
                zone_name,
                count,
                origin="signature",
            )

        for zone_name, count in signature.junction_zones.items():
            facts.add(
                "junction_zone",
                zone_name,
                count,
                origin="signature",
            )

        for zone_name, count in signature.loop_zones.items():
            facts.add(
                "loop_zone",
                zone_name,
                count,
                origin="signature",
            )

        # Direction ratios
        for direction, ratio in signature.direction_ratios.items():
            facts.add(
                "direction_ratio",
                direction,
                f"{ratio:.4f}",
                origin="signature",
            )

        # Geometry
        facts.add(
            "aspect_ratio",
            f"{signature.aspect_ratio:.4f}",
            origin="signature",
        )

        facts.add(
            "width",
            f"{signature.width:.2f}",
            origin="signature",
        )

        facts.add(
            "height",
            f"{signature.height:.2f}",
            origin="signature",
        )

        facts.add(
            "ink_pixels",
            signature.ink_pixels,
            origin="signature",
        )

        # Reconstruction metadata
        if signature.is_reconstructed:
            facts.add(
                "is_reconstructed",
                origin="signature",
            )

        if signature.line_removal_applied:
            facts.add(
                "line_removal_applied",
                origin="signature",
            )

        if signature.downstream_bridge_required:
            facts.add(
                "downstream_bridge_required",
                origin="signature",
            )

        if signature.downstream_bridge_completed:
            facts.add(
                "downstream_bridge_completed",
                origin="signature",
            )

        facts.add(
            "bridge_state",
            signature.bridge_state,
            origin="signature",
        )

        # Dataset/source kind
        if signature.source_kind != "unknown":
            facts.add(
                "source_kind",
                signature.source_kind,
                origin="signature",
            )

        return facts

    # --------------------------------------------------------
    # Explanation builder
    # --------------------------------------------------------

    def _build_explanation(
        self,
        signature: ScriLogSignature,
        facts: ScriLogFactBase,
        candidate_effects: List[ScriLogCandidateEffect],
    ) -> List[str]:
        """
        Build human-readable explanation lines.

        These are for debugging and later N05 audit reports.
        """

        explanation: List[str] = []

        families = sorted(set(facts.find_arg0("family")))
        warnings = sorted(set(facts.find_arg0("warning")))
        history = sorted(set(facts.find_arg0("history")))
        risks = sorted(set(facts.find_arg0("selection_risk")))

        if families:
            explanation.append(
                "Derived structural families: "
                + ", ".join(families)
                + "."
            )

        if history:
            explanation.append(
                "Reconstruction history: "
                + ", ".join(history)
                + "."
            )

        if warnings:
            explanation.append(
                "Warnings: "
                + ", ".join(warnings)
                + "."
            )

        if risks:
            explanation.append(
                "Selection risks: "
                + ", ".join(risks)
                + "."
            )

        if signature.downstream_bridge_required and not signature.downstream_bridge_completed:
            explanation.append(
                "Bridge repair was required but not completed; this selected geometry is structurally unsafe."
            )

        for effect in candidate_effects:
            explanation.append(
                f"{effect.effect.upper()} {effect.class_label}: {effect.reason}."
            )

        if not explanation:
            explanation.append(
                "No symbolic warnings or candidate effects were produced."
            )

        return explanation
