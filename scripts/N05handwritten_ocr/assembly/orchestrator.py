"""N05 assembly orchestrator.

The assembly layer consumes text-unit records produced by
``expert_orchestrator.py`` and turns them into decision-ready surfaces:

    text units -> segmentation matrix -> letter matrix -> summary

No final OCR decision is made in v0.1.
"""

from __future__ import annotations

from .combined_output import build_combined_expert_output
from .decision_matrix import build_decision_matrix
from .evidence_fusion import build_assembly_summary
from .letter_matrix import build_letter_matrix
from .local_interrogation import apply_local_interrogation
from .matrix_envelope import build_matrix_envelope
from .schemas import ASSEMBLY_VERSION
from .segment_artifacts import materialize_selected_segments
from .segmentation_matrix import build_segmentation_matrix

try:
    from ..scribejudge import build_scribejudge_overlay
except ImportError:
    from scribejudge import build_scribejudge_overlay


def build_assembly_map(
    document_id: str,
    handwritten_text_units: list[dict],
    settings: dict | None = None,
    output_dir: str | None = None,
) -> dict:
    """Build the first N05 assembly map for one document.

    Args:
        document_id: Stable document identifier.
        handwritten_text_units: N05 unit records with proposer/expert evidence.
        settings: Optional assembly settings.

    Returns:
        JSON-safe assembly map.
    """

    settings = settings or {}
    segmentation_settings = settings.get("segmentation", {})
    letter_settings = settings.get("letter_matrix", {})
    segment_settings = settings.get("segment_artifacts", {})
    combined_settings = settings.get("combined_output", {})
    decision_settings = settings.get("decision_matrix", {})
    scribejudge_settings = settings.get("scribejudge", {})
    segmentation_matrix = build_segmentation_matrix(
        handwritten_text_units,
        settings=segmentation_settings,
    )
    segmentation_matrix = apply_local_interrogation(
        segmentation_matrix,
        settings=segmentation_settings,
    )
    assembly_map = {
        "assembly_version": ASSEMBLY_VERSION,
        "document_id": document_id,
        "status": "assembling",
        "unit_count": len(handwritten_text_units),
        "segmentation_matrix": segmentation_matrix,
        "letter_matrix": [],
        "summary": {},
        "combined_expert_output": {},
        "decision_matrix": {},
        "settings": {
            "segmentation": segmentation_settings,
            "letter_matrix": letter_settings,
            "segment_artifacts": segment_settings,
            "combined_output": combined_settings,
            "decision_matrix": decision_settings,
            "scribejudge": scribejudge_settings,
        },
    }
    if output_dir:
        assembly_map = materialize_selected_segments(
            assembly_map=assembly_map,
            units=handwritten_text_units,
            output_dir=output_dir,
            settings=segment_settings,
        )
    letter_matrix = build_letter_matrix(
        handwritten_text_units,
        assembly_map["segmentation_matrix"],
        settings=letter_settings,
    )
    assembly_map["letter_matrix"] = letter_matrix
    assembly_map.update(
        build_matrix_envelope(
            document_id=document_id,
            unit_count=len(handwritten_text_units),
            segmentation_matrix=assembly_map["segmentation_matrix"],
            letter_matrix=letter_matrix,
        )
    )
    assembly_map["combined_expert_output"] = build_combined_expert_output(
        assembly_map=assembly_map,
        settings=combined_settings,
    )
    assembly_map.setdefault("matrices", {})["combined_output"] = {
        "version": assembly_map["combined_expert_output"].get("version"),
        "status": assembly_map["combined_expert_output"].get("status"),
        "inputs": ["letter_evidence"],
        "outputs": ["n06_word_tokens", "position_candidate_backups"],
        "rows": assembly_map["combined_expert_output"].get("word_tokens", []),
    }
    assembly_map["decision_matrix"] = build_decision_matrix(
        combined_expert_output=assembly_map["combined_expert_output"],
        settings=decision_settings,
    )
    assembly_map.setdefault("matrices", {})["decision"] = {
        "version": assembly_map["decision_matrix"].get("version"),
        "status": assembly_map["decision_matrix"].get("status"),
        "inputs": ["combined_output"],
        "outputs": ["provisional_word_candidates"],
        "rows": assembly_map["decision_matrix"].get("rows", []),
    }
    assembly_map["scribejudge_overlay"] = build_scribejudge_overlay(
        assembly_map=assembly_map,
        settings=scribejudge_settings,
    )
    assembly_map.setdefault("matrices", {})["scribejudge"] = {
        "version": assembly_map["scribejudge_overlay"].get("version"),
        "status": assembly_map["scribejudge_overlay"].get("status"),
        "inputs": ["decision_matrix", "combined_output", "confusion_history"],
        "outputs": ["advisory_risk_overlay", "future_meta_model_features"],
        "rows": assembly_map["scribejudge_overlay"].get("rows", []),
    }
    assembly_map["summary"] = build_assembly_summary(
        assembly_map["segmentation_matrix"],
        letter_matrix,
        matrices=assembly_map.get("matrices", {}),
        expert_job_queue=assembly_map.get("expert_job_queue", []),
    )
    assembly_map["summary"]["combined_output"] = assembly_map[
        "combined_expert_output"
    ].get("summary", {})
    assembly_map["summary"]["decision_matrix"] = assembly_map[
        "decision_matrix"
    ].get("summary", {})
    assembly_map["summary"]["scribejudge_overlay"] = assembly_map[
        "scribejudge_overlay"
    ].get("summary", {})
    assembly_map["status"] = "assembled"
    return assembly_map
