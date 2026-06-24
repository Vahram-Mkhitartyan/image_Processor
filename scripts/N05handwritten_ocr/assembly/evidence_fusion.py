"""Lightweight scoring helpers for N05 assembly v0.1.

This module deliberately avoids final OCR decisions. It only summarizes how
ready the evidence surface is.
"""

from __future__ import annotations


def summarize_segmentation_matrix(segmentation_matrix: list[dict]) -> dict:
    """Return compact segmentation-path readiness metrics."""

    path_counts = [int(entry.get("path_count", 0)) for entry in segmentation_matrix]
    return {
        "text_unit_count": len(segmentation_matrix),
        "total_segmentation_paths": sum(path_counts),
        "max_paths_for_one_unit": max(path_counts, default=0),
        "units_with_multiple_paths": sum(count > 1 for count in path_counts),
    }


def summarize_letter_matrix(letter_matrix: list[dict]) -> dict:
    """Return compact letter-candidate readiness metrics."""

    row_count = 0
    populated_rows = 0
    candidate_count = 0
    for unit_matrix in letter_matrix:
        for row in unit_matrix.get("rows") or []:
            row_count += 1
            count = int(row.get("candidate_count", 0))
            candidate_count += count
            populated_rows += int(count > 0)
    return {
        "matrix_count": len(letter_matrix),
        "row_count": row_count,
        "populated_row_count": populated_rows,
        "empty_row_count": row_count - populated_rows,
        "candidate_count": candidate_count,
    }


def build_assembly_summary(
    segmentation_matrix: list[dict],
    letter_matrix: list[dict],
    matrices: dict | None = None,
    expert_job_queue: list[dict] | None = None,
) -> dict:
    """Build the v0.1 assembly summary block."""

    matrices = matrices or {}
    expert_job_queue = expert_job_queue or []
    return {
        "segmentation": summarize_segmentation_matrix(segmentation_matrix),
        "letters": summarize_letter_matrix(letter_matrix),
        "matrix_envelope": {
            "matrix_count": len(matrices),
            "matrix_statuses": {
                name: matrix.get("status", "unknown")
                for name, matrix in matrices.items()
                if isinstance(matrix, dict)
            },
        },
        "expert_job_queue": {
            "job_count": len(expert_job_queue),
            "queued_count": sum(
                str(job.get("status")) == "queued"
                for job in expert_job_queue
            ),
        },
        "decision_status": "not_finalized",
        "decision_note": (
            "Assembly v0.1 builds candidate surfaces only. Final formula, "
            "beam search, and correctness-history learning are not active yet."
        ),
    }
