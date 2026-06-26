"""N05 assembly orchestrator.

The assembly layer consumes text-unit records produced by
``expert_orchestrator.py`` and turns them into decision-ready surfaces:

    text units -> segmentation matrix -> letter matrix -> summary

No final OCR decision is made in v0.1.
"""

from __future__ import annotations

from .evidence_fusion import build_assembly_summary
from .letter_matrix import build_letter_matrix
from .local_interrogation import apply_local_interrogation
from .matrix_envelope import build_matrix_envelope
from .schemas import ASSEMBLY_VERSION
from .segment_artifacts import materialize_selected_segments
from .segmentation_matrix import build_segmentation_matrix


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
        "settings": {
            "segmentation": segmentation_settings,
            "letter_matrix": letter_settings,
            "segment_artifacts": segment_settings,
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
    assembly_map["summary"] = build_assembly_summary(
        assembly_map["segmentation_matrix"],
        letter_matrix,
        matrices=assembly_map.get("matrices", {}),
        expert_job_queue=assembly_map.get("expert_job_queue", []),
    )
    assembly_map["status"] = "assembled"
    return assembly_map
