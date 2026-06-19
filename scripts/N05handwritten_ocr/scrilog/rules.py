"""Default structural rules used to derive symbolic glyph families."""

from __future__ import annotations

from typing import List

from .engine import ScriLogRule, _fact

class ScriLogRuleFactory:
    """
    Creates ScriLog's default structural rules.

    v0.1 rule philosophy:
        - generic structure only
        - no Armenian class mapping yet
        - no candidate blocking yet
        - no image modification
        - no reconstruction decisions

    These rules only derive symbolic families and warnings.

    Example:
        loop_count >= 1
            -> family(looped)

        endpoint_count == 2
            -> family(two_endpoint)

        bridge required but not completed
            -> warning(bridge_required_but_not_completed)
    """

    @staticmethod
    def build_default_rules() -> List[ScriLogRule]:
        rules: List[ScriLogRule] = []

        ScriLogRuleFactory._add_loop_rules(rules)
        ScriLogRuleFactory._add_endpoint_rules(rules)
        ScriLogRuleFactory._add_junction_rules(rules)
        ScriLogRuleFactory._add_extension_rules(rules)
        ScriLogRuleFactory._add_exit_rules(rules)
        ScriLogRuleFactory._add_shape_ratio_rules(rules)
        ScriLogRuleFactory._add_direction_rules(rules)
        ScriLogRuleFactory._add_reconstruction_rules(rules)
        ScriLogRuleFactory._add_combined_family_rules(rules)

        return rules

    # --------------------------------------------------------
    # Loop rules
    # --------------------------------------------------------

    @staticmethod
    def _add_loop_rules(rules: List[ScriLogRule]) -> None:
        rules.append(
            ScriLogRule(
                name="family_looped",
                description="Any glyph with at least one loop/hole is looped.",
                condition=lambda facts, signature: signature.loop_count >= 1,
                action=lambda facts, signature: [
                    _fact("family", "looped"),
                    _fact("has_structure", "loop"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="family_no_loop",
                description="Glyph with zero loops is non-looped.",
                condition=lambda facts, signature: signature.loop_count == 0,
                action=lambda facts, signature: [
                    _fact("family", "no_loop"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="family_multi_loop",
                description="Glyph with two or more loops is structurally complex.",
                condition=lambda facts, signature: signature.loop_count >= 2,
                action=lambda facts, signature: [
                    _fact("family", "multi_loop"),
                    _fact("warning", "multi_loop_complexity"),
                ],
            )
        )

    # --------------------------------------------------------
    # Endpoint rules
    # --------------------------------------------------------

    @staticmethod
    def _add_endpoint_rules(rules: List[ScriLogRule]) -> None:
        rules.append(
            ScriLogRule(
                name="family_zero_endpoint",
                description="Zero endpoints may mean closed glyph or bad skeleton.",
                condition=lambda facts, signature: signature.endpoint_count == 0,
                action=lambda facts, signature: [
                    _fact("family", "closed_or_no_endpoint"),
                    _fact("warning", "zero_endpoints_check_skeleton"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="family_single_endpoint",
                description="Single endpoint is unusual for most handwritten glyph skeletons.",
                condition=lambda facts, signature: signature.endpoint_count == 1,
                action=lambda facts, signature: [
                    _fact("family", "single_endpoint"),
                    _fact("warning", "single_endpoint_unusual"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="family_two_endpoint",
                description="Two endpoints usually indicates one clean path-like glyph.",
                condition=lambda facts, signature: signature.endpoint_count == 2,
                action=lambda facts, signature: [
                    _fact("family", "two_endpoint"),
                    _fact("has_structure", "normal_path_exit_count"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="family_many_endpoint",
                description="Many endpoints may indicate fragmentation/noise.",
                condition=lambda facts, signature: signature.endpoint_count >= 5,
                action=lambda facts, signature: [
                    _fact("family", "many_endpoint"),
                    _fact("warning", "possible_fragmentation_or_noise"),
                ],
            )
        )

    # --------------------------------------------------------
    # Junction rules
    # --------------------------------------------------------

    @staticmethod
    def _add_junction_rules(rules: List[ScriLogRule]) -> None:
        rules.append(
            ScriLogRule(
                name="family_unbranched",
                description="No junctions means unbranched skeleton.",
                condition=lambda facts, signature: signature.junction_count == 0,
                action=lambda facts, signature: [
                    _fact("family", "unbranched"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="family_branched",
                description="One or more junctions means branched skeleton.",
                condition=lambda facts, signature: signature.junction_count >= 1,
                action=lambda facts, signature: [
                    _fact("family", "branched"),
                    _fact("has_structure", "junction"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="warning_overbranched",
                description="Too many junctions may indicate artifact damage or bad reconstruction.",
                condition=lambda facts, signature: signature.junction_count >= 4,
                action=lambda facts, signature: [
                    _fact("warning", "overbranched_check_artifact_or_bad_reconstruction"),
                ],
            )
        )

    # --------------------------------------------------------
    # Ascender / descender rules
    # --------------------------------------------------------

    @staticmethod
    def _add_extension_rules(rules: List[ScriLogRule]) -> None:
        rules.append(
            ScriLogRule(
                name="family_ascender",
                description="Glyph has upper extension.",
                condition=lambda facts, signature: signature.has_ascender,
                action=lambda facts, signature: [
                    _fact("family", "ascender"),
                    _fact("has_structure", "upper_extension"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="family_descender",
                description="Glyph has lower extension.",
                condition=lambda facts, signature: signature.has_descender,
                action=lambda facts, signature: [
                    _fact("family", "descender"),
                    _fact("has_structure", "lower_extension"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="family_midline_compact",
                description="Glyph has no detected upper/lower extension.",
                condition=lambda facts, signature: (
                    not signature.has_ascender
                    and not signature.has_descender
                ),
                action=lambda facts, signature: [
                    _fact("family", "midline_compact"),
                ],
            )
        )

    # --------------------------------------------------------
    # Exit/contact rules
    # --------------------------------------------------------

    @staticmethod
    def _add_exit_rules(rules: List[ScriLogRule]) -> None:
        rules.append(
            ScriLogRule(
                name="family_left_exit",
                description="Glyph has a left-side endpoint/contact.",
                condition=lambda facts, signature: signature.has_left_exit,
                action=lambda facts, signature: [
                    _fact("family", "left_exit"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="family_right_exit",
                description="Glyph has a right-side endpoint/contact.",
                condition=lambda facts, signature: signature.has_right_exit,
                action=lambda facts, signature: [
                    _fact("family", "right_exit"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="family_two_sided_exit",
                description="Glyph has both left and right exits.",
                condition=lambda facts, signature: (
                    signature.has_left_exit
                    and signature.has_right_exit
                ),
                action=lambda facts, signature: [
                    _fact("family", "two_sided_exit"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="family_top_contact",
                description="Glyph touches or exits near the top zone.",
                condition=lambda facts, signature: signature.has_top_contact,
                action=lambda facts, signature: [
                    _fact("family", "top_contact"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="family_bottom_contact",
                description="Glyph touches or exits near the bottom zone.",
                condition=lambda facts, signature: signature.has_bottom_contact,
                action=lambda facts, signature: [
                    _fact("family", "bottom_contact"),
                ],
            )
        )

    # --------------------------------------------------------
    # Shape ratio rules
    # --------------------------------------------------------

    @staticmethod
    def _add_shape_ratio_rules(rules: List[ScriLogRule]) -> None:
        rules.append(
            ScriLogRule(
                name="family_wide",
                description="Glyph aspect ratio is wide.",
                condition=lambda facts, signature: signature.is_wide,
                action=lambda facts, signature: [
                    _fact("family", "wide"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="family_tall",
                description="Glyph aspect ratio is tall.",
                condition=lambda facts, signature: signature.is_tall,
                action=lambda facts, signature: [
                    _fact("family", "tall"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="family_compact",
                description="Glyph aspect ratio is compact/balanced.",
                condition=lambda facts, signature: signature.is_compact,
                action=lambda facts, signature: [
                    _fact("family", "compact"),
                ],
            )
        )

    # --------------------------------------------------------
    # Direction rules
    # --------------------------------------------------------

    @staticmethod
    def _add_direction_rules(rules: List[ScriLogRule]) -> None:
        rules.append(
            ScriLogRule(
                name="family_horizontal_dominant",
                description="Horizontal stroke ratio is dominant.",
                condition=lambda facts, signature: (
                    signature.direction_ratios.get("horizontal", 0.0) >= 0.45
                ),
                action=lambda facts, signature: [
                    _fact("family", "horizontal_dominant"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="family_vertical_dominant",
                description="Vertical stroke ratio is dominant.",
                condition=lambda facts, signature: (
                    signature.direction_ratios.get("vertical", 0.0) >= 0.45
                ),
                action=lambda facts, signature: [
                    _fact("family", "vertical_dominant"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="family_diagonal_rich",
                description="Diagonal stroke ratio is high.",
                condition=lambda facts, signature: (
                    signature.direction_ratios.get("diagonal", 0.0) >= 0.35
                ),
                action=lambda facts, signature: [
                    _fact("family", "diagonal_rich"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="family_curve_rich",
                description="Curved stroke ratio is high.",
                condition=lambda facts, signature: (
                    signature.direction_ratios.get("curve", 0.0) >= 0.35
                ),
                action=lambda facts, signature: [
                    _fact("family", "curve_rich"),
                ],
            )
        )

    # --------------------------------------------------------
    # Reconstruction metadata rules
    # --------------------------------------------------------

    @staticmethod
    def _add_reconstruction_rules(rules: List[ScriLogRule]) -> None:
        rules.append(
            ScriLogRule(
                name="history_reconstructed",
                description="Glyph came from a reconstructed hypothesis.",
                condition=lambda facts, signature: signature.is_reconstructed,
                action=lambda facts, signature: [
                    _fact("history", "reconstructed"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="history_line_removal",
                description="Line removal was applied before this selected geometry.",
                condition=lambda facts, signature: signature.line_removal_applied,
                action=lambda facts, signature: [
                    _fact("history", "line_removal_applied"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="warning_unfinished_bridge",
                description="Bridge was required but not completed.",
                condition=lambda facts, signature: (
                    signature.downstream_bridge_required
                    and not signature.downstream_bridge_completed
                ),
                action=lambda facts, signature: [
                    _fact("warning", "bridge_required_but_not_completed"),
                    _fact("selection_risk", "unsafe_reconstruction_candidate"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="history_completed_bridge",
                description="Bridge was required and completed.",
                condition=lambda facts, signature: (
                    signature.downstream_bridge_required
                    and signature.downstream_bridge_completed
                ),
                action=lambda facts, signature: [
                    _fact("history", "bridge_repair_completed"),
                ],
            )
        )

    # --------------------------------------------------------
    # Combined derived families
    # --------------------------------------------------------

    @staticmethod
    def _add_combined_family_rules(rules: List[ScriLogRule]) -> None:
        rules.append(
            ScriLogRule(
                name="combined_looped_right_exit",
                description="Looped glyph with right exit.",
                condition=lambda facts, signature: (
                    facts.has("family", "looped")
                    and facts.has("family", "right_exit")
                ),
                action=lambda facts, signature: [
                    _fact("family", "looped_right_exit"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="combined_looped_descender",
                description="Looped glyph with descender.",
                condition=lambda facts, signature: (
                    facts.has("family", "looped")
                    and facts.has("family", "descender")
                ),
                action=lambda facts, signature: [
                    _fact("family", "looped_descender"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="combined_simple_midline_two_endpoint",
                description="Compact midline glyph with two endpoints and no branches.",
                condition=lambda facts, signature: (
                    facts.has("family", "midline_compact")
                    and facts.has("family", "two_endpoint")
                    and facts.has("family", "unbranched")
                ),
                action=lambda facts, signature: [
                    _fact("family", "simple_midline_two_endpoint"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="combined_complex_loop_branch_endpoint",
                description="Looped, branched, multi-endpoint glyph.",
                condition=lambda facts, signature: (
                    signature.loop_count >= 1
                    and signature.junction_count >= 1
                    and signature.endpoint_count >= 3
                ),
                action=lambda facts, signature: [
                    _fact("family", "complex_loop_branch_endpoint"),
                ],
            )
        )

        rules.append(
            ScriLogRule(
                name="combined_connected_cursive_like",
                description="Two-sided, horizontal-ish, path-like glyph.",
                condition=lambda facts, signature: (
                    facts.has("family", "two_sided_exit")
                    and (
                        facts.has("family", "horizontal_dominant")
                        or signature.direction_ratios.get("horizontal", 0.0) >= 0.30
                    )
                    and signature.endpoint_count in {2, 3, 4}
                ),
                action=lambda facts, signature: [
                    _fact("family", "connected_cursive_like"),
                ],
            )
        )

