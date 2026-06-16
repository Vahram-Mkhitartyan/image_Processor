"""ScribeTrace defense registry.

This file does not decide which damage maps to which defense.
That belongs to condition/condition_router.py.

This registry describes ScribeTrace-specific defense tools:
- names
- UI display labels
- additive/subtractive behavior
- safety limits
- whether the tool is implemented in trace_defenses.py or still handled elsewhere
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


DEFENSE_ENDPOINT_BRIDGE = "endpoint_bridge"
DEFENSE_HORIZONTAL_GAP_CLOSING = "horizontal_gap_closing"
DEFENSE_VERTICAL_GAP_CLOSING = "vertical_gap_closing"
DEFENSE_THRESHOLD_NORMALIZATION = "threshold_normalization"
DEFENSE_COMPONENT_DENOISING = "component_denoising"
DEFENSE_MEDIAN_DENOISING = "median_denoising"
DEFENSE_CONSERVATIVE_STROKE_RECOVERY = "conservative_stroke_recovery"
DEFENSE_CONTAMINATION_OPENING = "contamination_opening"
DEFENSE_LINEAR_ARTIFACT_REMOVAL = "linear_artifact_removal"
DEFENSE_BORDER_CONTINUATION = "border_continuation"


@dataclass(frozen=True)
class DefenseSpec:
    """Metadata contract for a ScribeTrace defense."""

    name: str
    display_name: str
    mode: str
    implemented_by: str
    description: str
    max_added_ratio: float = 0.0
    max_removed_ratio: float = 0.0
    max_changed_ratio: float = 0.0
    ui_added_color: str = "green"
    ui_removed_color: str = "red"
    risky: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


DEFENSE_REGISTRY: dict[str, DefenseSpec] = {
    DEFENSE_ENDPOINT_BRIDGE: DefenseSpec(
        name=DEFENSE_ENDPOINT_BRIDGE,
        display_name="Endpoint Bridge",
        mode="additive",
        implemented_by="trace_reconstruction_legacy",
        description="Connects likely broken stroke endpoints with a short bridge.",
        max_added_ratio=0.08,
        max_removed_ratio=0.00,
        max_changed_ratio=0.08,
    ),
    DEFENSE_HORIZONTAL_GAP_CLOSING: DefenseSpec(
        name=DEFENSE_HORIZONTAL_GAP_CLOSING,
        display_name="Horizontal Gap Closing",
        mode="additive",
        implemented_by="trace_defenses",
        description="Closes short horizontal breaks in strokes.",
        max_added_ratio=0.08,
        max_removed_ratio=0.00,
        max_changed_ratio=0.08,
    ),
    DEFENSE_VERTICAL_GAP_CLOSING: DefenseSpec(
        name=DEFENSE_VERTICAL_GAP_CLOSING,
        display_name="Vertical Gap Closing",
        mode="additive",
        implemented_by="trace_defenses",
        description="Closes short vertical breaks in strokes.",
        max_added_ratio=0.08,
        max_removed_ratio=0.00,
        max_changed_ratio=0.08,
    ),
    DEFENSE_THRESHOLD_NORMALIZATION: DefenseSpec(
        name=DEFENSE_THRESHOLD_NORMALIZATION,
        display_name="Threshold Normalization",
        mode="mixed",
        implemented_by="trace_defenses",
        description="Normalizes binary mask after blur, compression, or threshold failure.",
        max_added_ratio=0.10,
        max_removed_ratio=0.10,
        max_changed_ratio=0.14,
    ),
    DEFENSE_COMPONENT_DENOISING: DefenseSpec(
        name=DEFENSE_COMPONENT_DENOISING,
        display_name="Component Denoising",
        mode="subtractive",
        implemented_by="trace_defenses",
        description="Removes tiny detached noise components.",
        max_added_ratio=0.00,
        max_removed_ratio=0.12,
        max_changed_ratio=0.12,
    ),
    DEFENSE_MEDIAN_DENOISING: DefenseSpec(
        name=DEFENSE_MEDIAN_DENOISING,
        display_name="Median Denoising",
        mode="mixed",
        implemented_by="trace_defenses",
        description="Uses median filtering to remove salt-and-pepper noise.",
        max_added_ratio=0.06,
        max_removed_ratio=0.10,
        max_changed_ratio=0.12,
    ),
    DEFENSE_CONSERVATIVE_STROKE_RECOVERY: DefenseSpec(
        name=DEFENSE_CONSERVATIVE_STROKE_RECOVERY,
        display_name="Conservative Stroke Recovery",
        mode="additive",
        implemented_by="trace_defenses",
        description="Adds a small amount of stroke mass after erosion-like damage.",
        max_added_ratio=0.10,
        max_removed_ratio=0.00,
        max_changed_ratio=0.10,
        risky=True,
    ),
    DEFENSE_CONTAMINATION_OPENING: DefenseSpec(
        name=DEFENSE_CONTAMINATION_OPENING,
        display_name="Contamination Opening",
        mode="subtractive",
        implemented_by="trace_defenses",
        description="Attempts to remove thin external contamination without destroying glyph body.",
        max_added_ratio=0.00,
        max_removed_ratio=0.14,
        max_changed_ratio=0.14,
        risky=True,
    ),
    DEFENSE_LINEAR_ARTIFACT_REMOVAL: DefenseSpec(
        name=DEFENSE_LINEAR_ARTIFACT_REMOVAL,
        display_name="Linear Artifact Removal",
        mode="subtractive",
        implemented_by="trace_defenses",
        description="Removes very long straight line artifacts such as stamp/text bars.",
        max_added_ratio=0.00,
        max_removed_ratio=0.12,
        max_changed_ratio=0.12,
        risky=True,
    ),
    DEFENSE_BORDER_CONTINUATION: DefenseSpec(
        name=DEFENSE_BORDER_CONTINUATION,
        display_name="Border Continuation",
        mode="additive",
        implemented_by="trace_defenses",
        description="Extends strokes that appear cut off at crop borders.",
        max_added_ratio=0.08,
        max_removed_ratio=0.00,
        max_changed_ratio=0.08,
        risky=True,
    ),
}


def get_defense_spec(name: str) -> DefenseSpec | None:
    return DEFENSE_REGISTRY.get(name)


def get_defense_spec_dict(name: str) -> dict:
    spec = get_defense_spec(name)
    if spec is None:
        return {
            "name": name,
            "display_name": name,
            "mode": "unknown",
            "implemented_by": "unknown",
            "description": "Unknown defense.",
            "max_added_ratio": 0.0,
            "max_removed_ratio": 0.0,
            "max_changed_ratio": 0.0,
            "ui_added_color": "green",
            "ui_removed_color": "red",
            "risky": True,
        }
    return spec.to_dict()


def implemented_in_trace_defenses(name: str) -> bool:
    spec = get_defense_spec(name)
    return bool(spec and spec.implemented_by == "trace_defenses")


def all_defense_names() -> list[str]:
    return list(DEFENSE_REGISTRY.keys())


def trace_defense_names() -> list[str]:
    return [
        name
        for name, spec in DEFENSE_REGISTRY.items()
        if spec.implemented_by == "trace_defenses"
    ]


__all__ = [
    "DefenseSpec",
    "DEFENSE_ENDPOINT_BRIDGE",
    "DEFENSE_HORIZONTAL_GAP_CLOSING",
    "DEFENSE_VERTICAL_GAP_CLOSING",
    "DEFENSE_THRESHOLD_NORMALIZATION",
    "DEFENSE_COMPONENT_DENOISING",
    "DEFENSE_MEDIAN_DENOISING",
    "DEFENSE_CONSERVATIVE_STROKE_RECOVERY",
    "DEFENSE_CONTAMINATION_OPENING",
    "DEFENSE_LINEAR_ARTIFACT_REMOVAL",
    "DEFENSE_BORDER_CONTINUATION",
    "DEFENSE_REGISTRY",
    "get_defense_spec",
    "get_defense_spec_dict",
    "implemented_in_trace_defenses",
    "all_defense_names",
    "trace_defense_names",
]