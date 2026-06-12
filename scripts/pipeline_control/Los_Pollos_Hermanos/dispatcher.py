"""Phase dispatch for one document."""

from .phase_prep import run_prep_phase
from .phase_n05_experts import run_n05_expert_phase
from .phase_printed_ocr import run_printed_ocr_phase
from .phase_refine import run_refine_phase
from .phase_scribemap import run_scribemap_phase
from .phase_visual import run_visual_classification_phase


def run_phase_for_document(document_path, phase):
    """
    Dispatch one document to the requested phase.

    Args:
        document_path: Path to one input document.
        phase: Phase name.

    Returns:
        Status dictionary for the requested phase.
    """
    if phase == "prep":
        return run_prep_phase(document_path)

    if phase == "scribemap":
        return run_scribemap_phase(document_path)

    if phase == "refine":
        return run_refine_phase(document_path)

    if phase in ["visual", "visual_classification", "n03"]:
        return run_visual_classification_phase(document_path)

    if phase in ["printed_ocr", "printed", "n04"]:
        return run_printed_ocr_phase(document_path)

    if phase in ["handwritten_ocr", "handwritten", "n05"]:
        return run_n05_expert_phase(document_path)

    if phase == "pipeline":
        # Pipeline == geometry, refinement, visual routing, then OCR expert maps.
        run_scribemap_phase(document_path)
        run_refine_phase(document_path)
        run_visual_classification_phase(document_path)
        run_printed_ocr_phase(document_path)
        return run_n05_expert_phase(document_path)

    raise ValueError(f"Unknown phase: {phase}")
