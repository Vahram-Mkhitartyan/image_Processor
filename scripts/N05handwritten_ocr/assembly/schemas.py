"""JSON contract builders for N05 assembly.

The assembly layer is intentionally plain-dict based. These helpers keep the
schema stable without forcing the rest of N05 to depend on dataclass instances.
"""

from __future__ import annotations

from typing import Any


ASSEMBLY_VERSION = "n05_assembly_v0_1"


def normalize_bbox(value: Any) -> dict | None:
    """Return a JSON-safe bbox with derived dimensions, or ``None``."""

    if not isinstance(value, dict):
        return None
    try:
        x1 = int(value["x1"])
        y1 = int(value["y1"])
        x2 = int(value["x2"])
        y2 = int(value["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": width,
        "height": height,
        "area": width * height,
        "center_x": x1 + width / 2.0,
        "center_y": y1 + height / 2.0,
    }


def make_evidence_source(
    source: str,
    status: str = "available",
    payload: dict | None = None,
    notes: list[str] | None = None,
) -> dict:
    """Build a standard evidence-source envelope."""

    return {
        "source": source,
        "status": status,
        "payload": payload or {},
        "notes": notes or [],
    }


def make_segmentation_segment(
    segment_id: str,
    bbox: dict | None,
    role: str,
    source: str,
    mask_crop_path: str | None = None,
    visual_crop_path: str | None = None,
    source_segment: dict | None = None,
) -> dict:
    """Build one candidate segment inside a segmentation path."""

    return {
        "segment_id": segment_id,
        "bbox": normalize_bbox(bbox) or bbox,
        "role": role,
        "source": source,
        "mask_crop_path": mask_crop_path,
        "visual_crop_path": visual_crop_path,
        "source_segment": source_segment or {},
    }


def make_segmentation_path(
    path_id: str,
    path_type: str,
    segments: list[dict],
    score_hint: float = 0.0,
    source: str = "unknown",
    reason: str = "",
    evidence: dict | None = None,
) -> dict:
    """Build one segmentation-path candidate."""

    return {
        "path_id": path_id,
        "type": path_type,
        "source": source,
        "segment_count": len(segments),
        "segments": segments,
        "score_hint": float(score_hint or 0.0),
        "reason": reason,
        "evidence": evidence or {},
        "status": "candidate",
    }


def make_letter_candidate(
    char: str,
    score: float,
    source: str,
    rank: int | None = None,
    confidence: float | None = None,
    evidence: dict | None = None,
) -> dict:
    """Build one normalized candidate for one matrix cell."""

    return {
        "char": str(char),
        "score": float(score or 0.0),
        "source": source,
        "rank": rank,
        "confidence": confidence,
        "evidence": evidence or {},
    }


def make_matrix_cell(
    position: int,
    segment: dict | None,
    candidates: list[dict],
    evidence_sources: list[dict] | None = None,
) -> dict:
    """Build one row/cell in the letter decision matrix."""

    return {
        "position": int(position),
        "segment": segment,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "evidence_sources": evidence_sources or [],
        "status": "candidate_ready" if candidates else "awaiting_expert_evidence",
    }
