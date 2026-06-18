"""Rule-based condition router for the first universal damage split."""

from .condition_models import ConditionVerdict, ExpertRoutingAdvice
from .damage_labels import (
    DAMAGE_BLEED_THROUGH,
    DAMAGE_CLEAN,
    DAMAGE_COMPRESSION_ARTIFACTS,
    DAMAGE_EDGE_CROP_LOSS,
    DAMAGE_INK_OVERLAP,
    DAMAGE_LIGHT_BLUR,
    DAMAGE_LIGHT_CUT,
    DAMAGE_LIGHT_EROSION,
    DAMAGE_SCANNER_NOISE,
    DAMAGE_STAMP_INTERFERENCE,
    DAMAGE_THRESHOLD_FAILURE,
)

# Defense names must match ScribeTrace reconstruction/defense names.
DEFENSE_ENDPOINT_BRIDGE = "endpoint_bridge"
DEFENSE_HORIZONTAL_GAP_CLOSING = "horizontal_gap_closing"
DEFENSE_VERTICAL_GAP_CLOSING = "vertical_gap_closing"
DEFENSE_THRESHOLD_NORMALIZATION = "threshold_normalization"
DEFENSE_COMPONENT_DENOISING = "component_denoising"
DEFENSE_MEDIAN_DENOISING = "median_denoising"
DEFENSE_CONSERVATIVE_STROKE_RECOVERY = "conservative_stroke_recovery"
DEFENSE_CONTAMINATION_OPENING = "contamination_opening"
DEFENSE_LINEAR_ARTIFACT_REMOVAL = "linear_artifact_removal"
DEFENSE_STAMP_EXTERNAL_ARTIFACT_REMOVAL = "stamp_external_artifact_removal"
DEFENSE_BORDER_CONTINUATION = "border_continuation"

SCRIBETRACE_DEFENSES_BY_DAMAGE = {
    DAMAGE_CLEAN: [],
    DAMAGE_LIGHT_CUT: [
        DEFENSE_HORIZONTAL_GAP_CLOSING,
        DEFENSE_VERTICAL_GAP_CLOSING,
        DEFENSE_ENDPOINT_BRIDGE,
    ],
    DAMAGE_LIGHT_BLUR: [
        DEFENSE_THRESHOLD_NORMALIZATION,
    ],
    DAMAGE_SCANNER_NOISE: [
        DEFENSE_COMPONENT_DENOISING,
        DEFENSE_MEDIAN_DENOISING,
    ],
    DAMAGE_LIGHT_EROSION: [
        DEFENSE_CONSERVATIVE_STROKE_RECOVERY,
        DEFENSE_ENDPOINT_BRIDGE,
        DEFENSE_HORIZONTAL_GAP_CLOSING,
        DEFENSE_VERTICAL_GAP_CLOSING,
    ],
    DAMAGE_INK_OVERLAP: [
        DEFENSE_LINEAR_ARTIFACT_REMOVAL,
        DEFENSE_CONTAMINATION_OPENING,
        DEFENSE_ENDPOINT_BRIDGE,
    ],
    DAMAGE_STAMP_INTERFERENCE: [
        DEFENSE_STAMP_EXTERNAL_ARTIFACT_REMOVAL,
        DEFENSE_CONTAMINATION_OPENING,
    ],
    DAMAGE_BLEED_THROUGH: [
        DEFENSE_THRESHOLD_NORMALIZATION,
        DEFENSE_COMPONENT_DENOISING,
        DEFENSE_MEDIAN_DENOISING,
    ],
    DAMAGE_EDGE_CROP_LOSS: [
        DEFENSE_BORDER_CONTINUATION,
    ],
    DAMAGE_THRESHOLD_FAILURE: [
        DEFENSE_THRESHOLD_NORMALIZATION,
        DEFENSE_HORIZONTAL_GAP_CLOSING,
        DEFENSE_VERTICAL_GAP_CLOSING,
        DEFENSE_ENDPOINT_BRIDGE,
    ],
    DAMAGE_COMPRESSION_ARTIFACTS: [
        DEFENSE_THRESHOLD_NORMALIZATION,
        DEFENSE_MEDIAN_DENOISING,
    ],
}

RISKY_DAMAGE_LABELS = {
    DAMAGE_INK_OVERLAP,
    DAMAGE_STAMP_INTERFERENCE,
}


def _default_weights(primary_damage: str):
    """Return conservative first-pass expert weights for MoE fusion later."""
    weights = {
        "scribetrace": 1.0,
        "glyph_cnn": 1.0,
        "shape_features": 0.7,
        "htr_context": 0.5,
    }
    if primary_damage == DAMAGE_CLEAN:
        return weights
    if primary_damage in {DAMAGE_LIGHT_CUT, DAMAGE_LIGHT_EROSION}:
        weights["scribetrace"] = 0.85
        weights["glyph_cnn"] = 0.95
    elif primary_damage in {DAMAGE_LIGHT_BLUR, DAMAGE_COMPRESSION_ARTIFACTS}:
        weights["scribetrace"] = 0.90
        weights["glyph_cnn"] = 0.75
    elif primary_damage in {DAMAGE_SCANNER_NOISE, DAMAGE_BLEED_THROUGH}:
        weights["scribetrace"] = 0.80
        weights["glyph_cnn"] = 0.90
    elif primary_damage in RISKY_DAMAGE_LABELS:
        weights["scribetrace"] = 0.65
        weights["glyph_cnn"] = 0.65
        weights["htr_context"] = 0.85
    elif primary_damage == DAMAGE_EDGE_CROP_LOSS:
        weights["scribetrace"] = 0.70
        weights["glyph_cnn"] = 0.70
        weights["htr_context"] = 0.90
    return weights


def route_condition(verdict: ConditionVerdict | dict | None):
    """Convert a condition verdict into ScribeTrace and MoE routing advice."""
    if verdict is None:
        primary_damage = DAMAGE_CLEAN
        condition = "clean"
        confidence = 1.0
    elif isinstance(verdict, dict):
        primary_damage = verdict.get("primary_damage") or verdict.get("label") or DAMAGE_CLEAN
        condition = verdict.get("condition", "damaged" if primary_damage != DAMAGE_CLEAN else "clean")
        confidence = float(verdict.get("confidence", 0.0))
    else:
        primary_damage = verdict.primary_damage
        condition = verdict.condition
        confidence = float(verdict.confidence)

    allowed = list(SCRIBETRACE_DEFENSES_BY_DAMAGE.get(primary_damage, []))
    manual_review = primary_damage in RISKY_DAMAGE_LABELS or condition == "uncertain"
    reason = f"route_for_{primary_damage}"
    if confidence < 0.45 and primary_damage != DAMAGE_CLEAN:
        manual_review = True
        reason = f"low_confidence_{reason}"

    return ExpertRoutingAdvice(
        scribetrace_allowed_defenses=allowed,
        expert_weights=_default_weights(primary_damage),
        manual_review=manual_review,
        reason=reason,
        metadata={
            "primary_damage": primary_damage,
            "condition": condition,
            "confidence": confidence,
        },
    )
