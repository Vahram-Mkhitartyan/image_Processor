"""ScribeTrace defense registry.

This file does not decide which damage maps to which defense.
That belongs to condition/condition_router.py.

This registry describes ScribeTrace-specific defense tools:
- names
- UI display labels
- additive/subtractive behavior
- safety limits
- stage ownership
- input/output contracts
- whether the tool is implemented in trace_defenses.py or handled elsewhere

Stage model:
- stage_00_source: before binary mask is finalized
- stage_01_mask: after binary mask exists, before components
- stage_02_component: after connected components, before skeleton
- stage_03_skeleton: after skeleton graph exists
- stage_04_path: after trace paths/endpoints/junctions exist
- stage_05_feature: after feature vector / RF confidence exists
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


DEFENSE_STAGE_00_SOURCE = "stage_00_source"
DEFENSE_STAGE_01_MASK = "stage_01_mask"
DEFENSE_STAGE_02_COMPONENT = "stage_02_component"
DEFENSE_STAGE_03_SKELETON = "stage_03_skeleton"
DEFENSE_STAGE_04_PATH = "stage_04_path"
DEFENSE_STAGE_05_FEATURE = "stage_05_feature"

DEFENSE_STAGE_ORDER = [
    DEFENSE_STAGE_00_SOURCE,
    DEFENSE_STAGE_01_MASK,
    DEFENSE_STAGE_02_COMPONENT,
    DEFENSE_STAGE_03_SKELETON,
    DEFENSE_STAGE_04_PATH,
    DEFENSE_STAGE_05_FEATURE,
]


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


@dataclass(frozen=True)
class DefenseSpec:
    """Metadata contract for a ScribeTrace defense.

    `stage` is the currently implemented execution stage.

    `candidate_stages` lists the conceptual stages where this defense family may
    eventually live. For example, threshold normalization has a source-image form
    later, but the current implementation is a mask-stage fallback.
    """

    name: str
    display_name: str
    mode: str
    implemented_by: str
    description: str

    # Stage-aware reconstruction contract.
    stage: str
    input_type: str
    output_type: str
    requires: tuple[str, ...] = ()
    candidate_stages: tuple[str, ...] = ()
    stage_notes: str = ""

    # Safety / verification contract.
    max_added_ratio: float = 0.0
    max_removed_ratio: float = 0.0
    max_changed_ratio: float = 0.0
    ui_added_color: str = "green"
    ui_removed_color: str = "red"
    risky: bool = False
    risk_level: str = "low"

    def to_dict(self) -> dict:
        return asdict(self)


DEFENSE_REGISTRY: dict[str, DefenseSpec] = {
    DEFENSE_ENDPOINT_BRIDGE: DefenseSpec(
        name=DEFENSE_ENDPOINT_BRIDGE,
        display_name="Endpoint Bridge",
        mode="additive",
        implemented_by="trace_reconstruction",
        description="Connects likely broken stroke endpoints with a short bridge.",
        stage=DEFENSE_STAGE_04_PATH,
        candidate_stages=(DEFENSE_STAGE_04_PATH,),
        input_type="trace_paths",
        output_type="candidate_mask",
        requires=("binary_mask", "trace_paths", "endpoints", "path_tangents"),
        stage_notes=(
            "Path-stage tool. Needs endpoint geometry and tangent direction; "
            "it should not run as a blind mask operation."
        ),
        max_added_ratio=0.08,
        max_removed_ratio=0.00,
        max_changed_ratio=0.08,
        risk_level="medium",
    ),
    DEFENSE_HORIZONTAL_GAP_CLOSING: DefenseSpec(
        name=DEFENSE_HORIZONTAL_GAP_CLOSING,
        display_name="Horizontal Gap Closing",
        mode="additive",
        implemented_by="trace_defenses",
        description="Closes short horizontal breaks in strokes.",
        stage=DEFENSE_STAGE_01_MASK,
        candidate_stages=(DEFENSE_STAGE_01_MASK,),
        input_type="binary_mask",
        output_type="candidate_mask",
        requires=("binary_mask",),
        stage_notes="Mask-stage candidate generator. Cheap, blind, and verifier-dependent.",
        max_added_ratio=0.08,
        max_removed_ratio=0.00,
        max_changed_ratio=0.08,
        risk_level="medium",
    ),
    DEFENSE_VERTICAL_GAP_CLOSING: DefenseSpec(
        name=DEFENSE_VERTICAL_GAP_CLOSING,
        display_name="Vertical Gap Closing",
        mode="additive",
        implemented_by="trace_defenses",
        description="Closes short vertical breaks in strokes.",
        stage=DEFENSE_STAGE_01_MASK,
        candidate_stages=(DEFENSE_STAGE_01_MASK,),
        input_type="binary_mask",
        output_type="candidate_mask",
        requires=("binary_mask",),
        stage_notes="Mask-stage candidate generator. Cheap, blind, and verifier-dependent.",
        max_added_ratio=0.08,
        max_removed_ratio=0.00,
        max_changed_ratio=0.08,
        risk_level="medium",
    ),
    DEFENSE_THRESHOLD_NORMALIZATION: DefenseSpec(
        name=DEFENSE_THRESHOLD_NORMALIZATION,
        display_name="Threshold Normalization",
        mode="mixed",
        implemented_by="trace_defenses",
        description="Normalizes binary mask after blur, compression, or threshold failure.",
        stage=DEFENSE_STAGE_01_MASK,
        candidate_stages=(DEFENSE_STAGE_00_SOURCE, DEFENSE_STAGE_01_MASK),
        input_type="binary_mask",
        output_type="candidate_mask",
        requires=("binary_mask",),
        stage_notes=(
            "Current implementation is mask-stage only. A true threshold repair "
            "belongs at stage_00_source when grayscale/source-image data is available."
        ),
        max_added_ratio=0.10,
        max_removed_ratio=0.10,
        max_changed_ratio=0.14,
        risk_level="medium",
    ),
    DEFENSE_COMPONENT_DENOISING: DefenseSpec(
        name=DEFENSE_COMPONENT_DENOISING,
        display_name="Component Denoising",
        mode="subtractive",
        implemented_by="trace_defenses",
        description="Removes tiny detached noise components.",
        stage=DEFENSE_STAGE_02_COMPONENT,
        candidate_stages=(DEFENSE_STAGE_01_MASK, DEFENSE_STAGE_02_COMPONENT),
        input_type="connected_components",
        output_type="candidate_mask",
        requires=("binary_mask", "components"),
        stage_notes=(
            "Component-stage tool. It can currently derive components internally "
            "from a mask, but the reconstruction orchestrator should treat it as "
            "a component-level defense."
        ),
        max_added_ratio=0.00,
        max_removed_ratio=0.12,
        max_changed_ratio=0.12,
        risk_level="medium",
    ),
    DEFENSE_MEDIAN_DENOISING: DefenseSpec(
        name=DEFENSE_MEDIAN_DENOISING,
        display_name="Median Denoising",
        mode="mixed",
        implemented_by="trace_defenses",
        description="Uses median filtering to remove salt-and-pepper noise.",
        stage=DEFENSE_STAGE_01_MASK,
        candidate_stages=(DEFENSE_STAGE_00_SOURCE, DEFENSE_STAGE_01_MASK),
        input_type="binary_mask",
        output_type="candidate_mask",
        requires=("binary_mask",),
        stage_notes=(
            "Current implementation is binary-mask median cleanup. A stronger "
            "version may belong at stage_00_source on grayscale/source images."
        ),
        max_added_ratio=0.06,
        max_removed_ratio=0.10,
        max_changed_ratio=0.12,
        risk_level="medium",
    ),
    DEFENSE_CONSERVATIVE_STROKE_RECOVERY: DefenseSpec(
        name=DEFENSE_CONSERVATIVE_STROKE_RECOVERY,
        display_name="Conservative Stroke Recovery",
        mode="additive",
        implemented_by="trace_defenses",
        description="Adds a small amount of stroke mass after erosion-like damage.",
        stage=DEFENSE_STAGE_01_MASK,
        candidate_stages=(DEFENSE_STAGE_01_MASK,),
        input_type="binary_mask",
        output_type="candidate_mask",
        requires=("binary_mask",),
        stage_notes="Mask-stage erosion recovery. Must remain tightly budgeted.",
        max_added_ratio=0.10,
        max_removed_ratio=0.00,
        max_changed_ratio=0.10,
        risky=True,
        risk_level="high",
    ),
    DEFENSE_CONTAMINATION_OPENING: DefenseSpec(
        name=DEFENSE_CONTAMINATION_OPENING,
        display_name="Contamination Opening",
        mode="subtractive",
        implemented_by="trace_defenses",
        description="Attempts to remove thin external contamination without destroying glyph body.",
        stage=DEFENSE_STAGE_02_COMPONENT,
        candidate_stages=(DEFENSE_STAGE_02_COMPONENT, DEFENSE_STAGE_04_PATH),
        input_type="connected_components",
        output_type="candidate_mask",
        requires=("binary_mask", "components"),
        stage_notes=(
            "Risky component-stage cleanup. Should not freely delete ink; verifier "
            "must reject if glyph topology gets worse."
        ),
        max_added_ratio=0.00,
        max_removed_ratio=0.14,
        max_changed_ratio=0.14,
        risky=True,
        risk_level="high",
    ),
    DEFENSE_LINEAR_ARTIFACT_REMOVAL: DefenseSpec(
        name=DEFENSE_LINEAR_ARTIFACT_REMOVAL,
        display_name="Linear Artifact Removal",
        mode="subtractive",
        implemented_by="trace_defenses",
        description="Removes very long straight line artifacts such as stamp/text bars.",
        stage=DEFENSE_STAGE_02_COMPONENT,
        candidate_stages=(DEFENSE_STAGE_02_COMPONENT,),
        input_type="connected_components",
        output_type="candidate_mask",
        requires=("binary_mask", "components", "line_artifact_candidate"),
        stage_notes=(
            "Component/artifact-stage tool. Intended for long straight artifacts, "
            "not general handwriting cleanup."
        ),
        max_added_ratio=0.00,
        max_removed_ratio=0.12,
        max_changed_ratio=0.12,
        risky=True,
        risk_level="high",
    ),
    DEFENSE_STAMP_EXTERNAL_ARTIFACT_REMOVAL: DefenseSpec(
        name=DEFENSE_STAMP_EXTERNAL_ARTIFACT_REMOVAL,
        display_name="Stamp External Artifact Removal",
        mode="subtractive",
        implemented_by="trace_defenses",
        description=(
            "Removes external stamp or border artifacts while protecting the "
            "estimated glyph core."
        ),
        stage=DEFENSE_STAGE_02_COMPONENT,
        candidate_stages=(DEFENSE_STAGE_02_COMPONENT,),
        input_type="connected_components",
        output_type="candidate_mask",
        requires=("binary_mask", "components", "external_artifact_candidate"),
        stage_notes=(
            "Component/artifact-stage stamp cleanup. This should run before "
            "generic line removal for stamp_interference because it has a "
            "protected-core safety model."
        ),
        max_added_ratio=0.00,
        max_removed_ratio=0.55,
        max_changed_ratio=0.55,
        risky=True,
        risk_level="high",
    ),
    DEFENSE_BORDER_CONTINUATION: DefenseSpec(
        name=DEFENSE_BORDER_CONTINUATION,
        display_name="Border Continuation",
        mode="additive",
        implemented_by="trace_defenses",
        description="Extends strokes that appear cut off at crop borders.",
        stage=DEFENSE_STAGE_01_MASK,
        candidate_stages=(DEFENSE_STAGE_01_MASK, DEFENSE_STAGE_04_PATH),
        input_type="binary_mask",
        output_type="candidate_mask",
        requires=("binary_mask", "crop_border"),
        stage_notes=(
            "Current implementation is early border-contact continuation. A later "
            "path-stage version should use endpoint direction toward the crop edge."
        ),
        max_added_ratio=0.08,
        max_removed_ratio=0.00,
        max_changed_ratio=0.08,
        risky=True,
        risk_level="high",
    ),
}


def is_valid_defense_stage(stage: str) -> bool:
    return str(stage) in DEFENSE_STAGE_ORDER


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
            "stage": "unknown",
            "input_type": "unknown",
            "output_type": "unknown",
            "requires": (),
            "candidate_stages": (),
            "stage_notes": "",
            "max_added_ratio": 0.0,
            "max_removed_ratio": 0.0,
            "max_changed_ratio": 0.0,
            "ui_added_color": "green",
            "ui_removed_color": "red",
            "risky": True,
            "risk_level": "high",
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


def defenses_for_stage(
    stage: str,
    defense_names: list[str] | tuple[str, ...] | None = None,
    *,
    include_candidate_stages: bool = False,
) -> list[str]:
    """Return defense names that belong to one reconstruction stage.

    By default this uses the currently implemented `stage`.

    Set `include_candidate_stages=True` only for planning/debug views, because
    some candidate stages describe future implementations that are not real yet.
    """
    if not is_valid_defense_stage(stage):
        return []

    allowed = set(defense_names) if defense_names is not None else None
    output = []

    for name, spec in DEFENSE_REGISTRY.items():
        if allowed is not None and name not in allowed:
            continue

        if spec.stage == stage:
            output.append(name)
            continue

        if include_candidate_stages and stage in spec.candidate_stages:
            output.append(name)

    return output


def defense_specs_for_stage(
    stage: str,
    defense_names: list[str] | tuple[str, ...] | None = None,
    *,
    include_candidate_stages: bool = False,
) -> list[DefenseSpec]:
    return [
        DEFENSE_REGISTRY[name]
        for name in defenses_for_stage(
            stage,
            defense_names,
            include_candidate_stages=include_candidate_stages,
        )
        if name in DEFENSE_REGISTRY
    ]


def grouped_defenses_by_stage(
    defense_names: list[str] | tuple[str, ...] | None = None,
    *,
    include_empty_stages: bool = True,
    include_candidate_stages: bool = False,
) -> dict[str, list[str]]:
    grouped = {
        stage: defenses_for_stage(
            stage,
            defense_names,
            include_candidate_stages=include_candidate_stages,
        )
        for stage in DEFENSE_STAGE_ORDER
    }

    if include_empty_stages:
        return grouped

    return {
        stage: names
        for stage, names in grouped.items()
        if names
    }


__all__ = [
    "DefenseSpec",
    "DEFENSE_STAGE_00_SOURCE",
    "DEFENSE_STAGE_01_MASK",
    "DEFENSE_STAGE_02_COMPONENT",
    "DEFENSE_STAGE_03_SKELETON",
    "DEFENSE_STAGE_04_PATH",
    "DEFENSE_STAGE_05_FEATURE",
    "DEFENSE_STAGE_ORDER",
    "DEFENSE_ENDPOINT_BRIDGE",
    "DEFENSE_HORIZONTAL_GAP_CLOSING",
    "DEFENSE_VERTICAL_GAP_CLOSING",
    "DEFENSE_THRESHOLD_NORMALIZATION",
    "DEFENSE_COMPONENT_DENOISING",
    "DEFENSE_MEDIAN_DENOISING",
    "DEFENSE_CONSERVATIVE_STROKE_RECOVERY",
    "DEFENSE_CONTAMINATION_OPENING",
    "DEFENSE_LINEAR_ARTIFACT_REMOVAL",
    "DEFENSE_STAMP_EXTERNAL_ARTIFACT_REMOVAL",
    "DEFENSE_BORDER_CONTINUATION",
    "DEFENSE_REGISTRY",
    "is_valid_defense_stage",
    "get_defense_spec",
    "get_defense_spec_dict",
    "implemented_in_trace_defenses",
    "all_defense_names",
    "trace_defense_names",
    "defenses_for_stage",
    "defense_specs_for_stage",
    "grouped_defenses_by_stage",
]
