"""Future-proof matrix envelope for N05 assembly."""

from __future__ import annotations

from collections import Counter


def _matrix_status(has_rows: bool, has_outputs: bool) -> str:
    """Return a compact matrix readiness status."""

    if has_outputs:
        return "partial"
    if has_rows:
        return "placeholder"
    return "empty"


def _condition_rows_from_segments(segmentation_matrix: list[dict]) -> list[dict]:
    """Collect per-segment condition records into one matrix-like list."""

    rows = []
    for entry in segmentation_matrix:
        for path in entry.get("paths") or []:
            for position, segment in enumerate(path.get("segments") or []):
                condition = segment.get("condition")
                if not condition:
                    continue
                rows.append(
                    {
                        "text_unit_id": entry.get("text_unit_id"),
                        "path_id": path.get("path_id"),
                        "segment_id": segment.get("segment_id"),
                        "position": position,
                        "condition": condition,
                    }
                )
    return rows


def _reconstruction_rows_from_conditions(condition_rows: list[dict]) -> list[dict]:
    """Prepare reconstruction matrix rows without running reconstruction yet."""

    rows = []
    for row in condition_rows:
        verdict = (row.get("condition") or {}).get("verdict") or {}
        routing = (row.get("condition") or {}).get("routing") or {}
        rows.append(
            {
                "text_unit_id": row.get("text_unit_id"),
                "path_id": row.get("path_id"),
                "segment_id": row.get("segment_id"),
                "position": row.get("position"),
                "repair_needed": bool(verdict.get("repair_needed")),
                "primary_damage": verdict.get("primary_damage"),
                "allowed_defenses": routing.get("scribetrace_allowed_defenses", []),
                "status": "queued" if verdict.get("repair_needed") else "not_needed",
            }
        )
    return rows


def build_expert_job_queue(
    segmentation_matrix: list[dict],
    condition_rows: list[dict],
) -> list[dict]:
    """Build planned segment-level expert jobs for future execution."""

    condition_by_segment = {
        (row.get("path_id"), row.get("segment_id")): row
        for row in condition_rows
    }
    jobs = []
    for entry in segmentation_matrix:
        for path in entry.get("paths") or []:
            if path.get("status") != "selected_for_v0_2_segment_artifacts":
                continue
            for position, segment in enumerate(path.get("segments") or []):
                key = (path.get("path_id"), segment.get("segment_id"))
                condition = condition_by_segment.get(key, {}).get("condition", {})
                repair_needed = bool((condition.get("verdict") or {}).get("repair_needed"))
                experts = ["condition"]
                if repair_needed:
                    experts.append("reconstruction")
                experts.extend(["scribetrace", "scrilog", "scrististics", "character_detector"])
                jobs.append(
                    {
                        "job_id": (
                            f"{entry.get('text_unit_id')}_"
                            f"{path.get('path_id')}_"
                            f"{segment.get('segment_id')}"
                        ),
                        "text_unit_id": entry.get("text_unit_id"),
                        "path_id": path.get("path_id"),
                        "segment_id": segment.get("segment_id"),
                        "position": position,
                        "visual_crop_path": segment.get("visual_crop_path"),
                        "mask_crop_path": segment.get("mask_crop_path"),
                        "experts_needed": experts,
                        "status": "queued",
                    }
                )
    return jobs


def build_case_fingerprint(
    document_id: str,
    unit_count: int,
    segmentation_matrix: list[dict],
    letter_matrix: list[dict],
    condition_rows: list[dict],
    expert_job_queue: list[dict],
) -> dict:
    """Build a compact future training fingerprint for formula selection."""

    damage_counts = Counter()
    for row in condition_rows:
        verdict = (row.get("condition") or {}).get("verdict") or {}
        damage_counts[str(verdict.get("primary_damage", "unknown"))] += 1

    word_split_units = sum(
        1
        for entry in segmentation_matrix
        if (entry.get("word_ocr_split_evidence") or {}).get("available")
    )
    populated_rows = sum(
        1
        for unit_matrix in letter_matrix
        for row in unit_matrix.get("rows") or []
        if int(row.get("candidate_count", 0)) > 0
    )
    return {
        "document_id": document_id,
        "unit_count": unit_count,
        "has_word_ocr_split_evidence": word_split_units > 0,
        "units_with_word_ocr_split_evidence": word_split_units,
        "segmentation_path_count": sum(
            int(entry.get("path_count", 0)) for entry in segmentation_matrix
        ),
        "condition_segment_count": len(condition_rows),
        "letter_matrix_populated_rows": populated_rows,
        "expert_job_count": len(expert_job_queue),
        "damage_mix": dict(sorted(damage_counts.items())),
        "expert_coverage": {
            "word_ocr": word_split_units > 0 or populated_rows > 0,
            "condition": bool(condition_rows),
            "segment_jobs": bool(expert_job_queue),
        },
    }


def build_correctness_history_placeholder() -> dict:
    """Reserve the future correctness-history training contract."""

    return {
        "available": False,
        "ground_truth_source": None,
        "final_choice_correct": None,
        "notes": [
            "Reserved for future adaptive formula-selection training.",
        ],
    }


def build_matrix_envelope(
    document_id: str,
    unit_count: int,
    segmentation_matrix: list[dict],
    letter_matrix: list[dict],
) -> dict:
    """Build the future-proof matrix envelope while preserving old fields."""

    condition_rows = _condition_rows_from_segments(segmentation_matrix)
    reconstruction_rows = _reconstruction_rows_from_conditions(condition_rows)
    expert_job_queue = build_expert_job_queue(segmentation_matrix, condition_rows)
    matrices = {
        "segmentation": {
            "version": "segmentation_matrix_v0_2",
            "status": _matrix_status(bool(segmentation_matrix), bool(segmentation_matrix)),
            "inputs": [
                "scribetrain_word_segmenter",
                "word_level_ocr_split_hints_fallback",
                "character_unit_proposer_legacy_disabled",
            ],
            "outputs": ["segmentation_paths", "selected_path_segments"],
            "rows": segmentation_matrix,
        },
        "condition": {
            "version": "condition_matrix_v0_1",
            "status": _matrix_status(bool(segmentation_matrix), bool(condition_rows)),
            "inputs": ["selected_path_segment_crops"],
            "outputs": ["condition_verdict", "routing_advice"],
            "rows": condition_rows,
        },
        "reconstruction": {
            "version": "reconstruction_matrix_placeholder_v0_1",
            "status": _matrix_status(bool(condition_rows), False),
            "inputs": ["condition_matrix"],
            "outputs": ["repair_candidates"],
            "rows": reconstruction_rows,
        },
        "letter_evidence": {
            "version": "letter_evidence_matrix_v0_1",
            "status": _matrix_status(bool(letter_matrix), any(
                int(row.get("candidate_count", 0)) > 0
                for unit_matrix in letter_matrix
                for row in unit_matrix.get("rows") or []
            )),
            "inputs": ["word_level_ocr_tokens", "scribetrace_unit_rf", "character_detector"],
            "outputs": ["letter_candidates_by_position"],
            "rows": letter_matrix,
        },
        "sequence": {
            "version": "sequence_matrix_placeholder_v0_1",
            "status": "placeholder",
            "inputs": ["letter_evidence", "word_level_ocr", "future_postprocessing"],
            "outputs": ["final_sequence_candidates"],
            "rows": [],
        },
    }
    return {
        "matrices": matrices,
        "expert_job_queue": expert_job_queue,
        "case_fingerprint": build_case_fingerprint(
            document_id=document_id,
            unit_count=unit_count,
            segmentation_matrix=segmentation_matrix,
            letter_matrix=letter_matrix,
            condition_rows=condition_rows,
            expert_job_queue=expert_job_queue,
        ),
        "correctness_history": build_correctness_history_placeholder(),
    }
