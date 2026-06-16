"""Condition inference stub for the first MoE split.

This file intentionally starts with oracle/rule behavior. A trained damage
classifier can replace the fallback later without changing ScribeTrace.
"""

from .condition_features import extract_condition_features
from .condition_models import DamageCandidate, ConditionVerdict
from .damage_labels import (
    DAMAGE_CLEAN,
    DAMAGE_LIGHT_CUT,
    DAMAGE_UNKNOWN,
    REPAIR_NEEDED_DAMAGE_LABELS,
    UNCERTAIN_DAMAGE_LABELS,
)


def _normalize_recipe_chain(known_damage_recipes):
    if not known_damage_recipes:
        return []
    if isinstance(known_damage_recipes, str):
        return [known_damage_recipes]
    return [str(item) for item in known_damage_recipes if item]


def _oracle_verdict(recipe_chain, features):
    primary = recipe_chain[0] if recipe_chain else DAMAGE_CLEAN
    condition = "clean" if primary == DAMAGE_CLEAN else "damaged"
    repair_needed = primary in REPAIR_NEEDED_DAMAGE_LABELS
    if primary in UNCERTAIN_DAMAGE_LABELS:
        condition = "uncertain"
        repair_needed = primary != DAMAGE_CLEAN
    return ConditionVerdict(
        condition=condition,
        repair_needed=repair_needed,
        primary_damage=primary,
        confidence=1.0,
        top_damage_candidates=[DamageCandidate(label=primary, confidence=1.0)],
        severity=0.0 if primary == DAMAGE_CLEAN else 0.5,
        source="oracle_known_damage_recipes",
        notes=["condition_from_known_damage_recipes"],
        features=features,
    )


def _rule_debug_verdict(features):
    """Temporary fallback until a real damage classifier exists."""
    component_count = features.get("component_count", 0.0)
    endpoint_count = features.get("endpoint_count", 0.0)
    isolated_count = features.get("isolated_point_count", 0.0)
    short_path_count = features.get("short_path_count", 0.0)

    suspicious = []
    if component_count > 1:
        suspicious.append("multiple_components")
    if endpoint_count >= 4:
        suspicious.append("many_endpoints")
    if isolated_count > 0:
        suspicious.append("isolated_points")
    if short_path_count > 0:
        suspicious.append("short_paths")

    if suspicious:
        return ConditionVerdict(
            condition="uncertain",
            repair_needed=True,
            primary_damage=DAMAGE_UNKNOWN,
            confidence=0.35,
            top_damage_candidates=[
                DamageCandidate(label=DAMAGE_UNKNOWN, confidence=0.35),
                DamageCandidate(label=DAMAGE_LIGHT_CUT, confidence=0.20),
            ],
            severity=min(1.0, 0.15 * len(suspicious)),
            source="rule_debug_topology",
            notes=suspicious,
            features=features,
        )

    return ConditionVerdict(
        condition="clean",
        repair_needed=False,
        primary_damage=DAMAGE_CLEAN,
        confidence=0.80,
        top_damage_candidates=[DamageCandidate(label=DAMAGE_CLEAN, confidence=0.80)],
        severity=0.0,
        source="rule_debug_topology",
        notes=["no_strong_damage_signal"],
        features=features,
    )


def predict_condition(trace_result, known_damage_recipes=None):
    """
    Return a universal condition verdict.

    Current modes:
    - known_damage_recipes present: oracle/debug verdict.
    - otherwise: conservative topology-rule verdict.
    """
    features = extract_condition_features(trace_result)
    recipe_chain = _normalize_recipe_chain(known_damage_recipes)
    if recipe_chain:
        return _oracle_verdict(recipe_chain, features)
    return _rule_debug_verdict(features)