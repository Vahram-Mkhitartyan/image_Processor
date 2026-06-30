"""Printed text-unit records and summaries for N04."""
from n04_ocr_engine import build_raw_printed_ocr_candidates
from n04_routing import (
    build_crop_bbox,
    build_document_bbox,
    get_best_crop_source_for_printed_ocr,
    get_visual_class,
)


def build_printed_text_unit(
    route_record,
    copied_crop_path,
    tesseract_ready_crop_path=None,
    document_id=None
):
    """
    Build one printed text unit record.

    N04 is not final truth.
    It collects raw printed OCR candidates while preserving:
    - N03 route identity
    - N02 crop paths
    - document coordinates
    - Color Update layer metadata
    - Minos scores
    """
    visual_info = route_record.get("visual_classification", {})
    tesseract_input_source = (
        get_best_crop_source_for_printed_ocr(route_record)
        or "unknown_crop_source"
    )
    analysis_mask_used = tesseract_input_source == "analysis_mask_crop_path"

    return {
        "document_id": route_record.get("document_id") or document_id,

        # Identity from N02/N03.
        "text_unit_id": route_record.get("text_unit_id"),
        "group_id": route_record.get("group_id"),
        "source_group_id": route_record.get("source_group_id"),
        "source_layer_group_id": route_record.get("source_layer_group_id"),
        "layer": route_record.get("layer"),

        # N03 routing.
        "visual_class": get_visual_class(route_record),
        "recommended_route": visual_info.get("recommended_route", []),
        "n03_node": visual_info.get("node"),
        "n03_model": visual_info.get("model"),
        "n03_model_version": visual_info.get("model_version"),

        # Crop contract.
        "source_crop_path": route_record.get("source_crop_path"),
        "refined_crop_path": route_record.get("refined_crop_path"),
        "original_crop_path": route_record.get("original_crop_path"),
        "analysis_crop_path": route_record.get("analysis_crop_path"),
        "classification_crop_path": route_record.get("classification_crop_path"),
        "classification_crop_source": route_record.get("classification_crop_source"),
        "classification_crop_policy": route_record.get("classification_crop_policy"),
        "context_crop_path": route_record.get("context_crop_path"),
        "analysis_mask_crop_path": route_record.get("analysis_mask_crop_path"),
        "routed_crop_path": route_record.get("routed_crop_path"),

        # N04-owned copies.
        "n04_copied_crop_path": copied_crop_path,
        "tesseract_ready_crop_path": tesseract_ready_crop_path,
        "tesseract_input_source": tesseract_input_source,
        "tesseract_input_polarity": "dark_ink_on_white_background",
        "analysis_mask_used_for_tesseract": analysis_mask_used,
        "black_mask_reflected_ocr_crop_path": route_record.get(
            "black_mask_reflected_ocr_crop_path"
        ),
        "black_mask_reflected_ocr_source": route_record.get(
            "black_mask_reflected_ocr_source"
        ),
        "black_mask_reflected_ocr_source_kind": route_record.get(
            "black_mask_reflected_ocr_source_kind"
        ),
        "n04_crop_bbox_source": route_record.get("n04_crop_bbox_source"),

        # Color/layer provenance.
        "mask_source": route_record.get("mask_source"),
        "visual_layer_source": route_record.get("visual_layer_source"),

        # Coordinates.
        "document_bbox": build_document_bbox(route_record),
        "crop_bbox": build_crop_bbox(route_record),
        "final_bbox": route_record.get("final_bbox"),
        "original_document_bbox": route_record.get("original_document_bbox"),
        "black_mask_derived_bbox": route_record.get("black_mask_derived_bbox"),
        "black_mask_derived_bbox_reason": route_record.get(
            "black_mask_derived_bbox_reason"
        ),

        # N03 confidence evidence.
        "n03_scores": visual_info.get("scores", {}),
        "n03_thresholds": visual_info.get("thresholds", {}),

        # N02 policy evidence.
        "layer_hypothesis": route_record.get("layer_hypothesis"),
        "role_guess": route_record.get("role_guess"),
        "minos_mode": route_record.get("minos_mode"),
        "is_final_text_candidate": route_record.get("is_final_text_candidate", True),
        "preserve_as_evidence": route_record.get("preserve_as_evidence", False),

        # N04 raw OCR candidates.
        "printed_ocr": build_raw_printed_ocr_candidates(
            tesseract_ready_crop_path
        ),

        # Important: N04 never declares final truth.
        "trusted_as_final": False,
    }


def _safe_text(value):
    """Normalize OCR text fields for JSON contracts."""

    if value is None:
        return ""
    return str(value).strip()


def _printed_candidate_records(printed_ocr):
    """Convert raw Tesseract outputs into compact context candidates."""

    candidates = []
    if not isinstance(printed_ocr, dict):
        return candidates

    for index, candidate in enumerate(printed_ocr.get("candidates") or [], start=1):
        if not isinstance(candidate, dict):
            continue
        text = _safe_text(candidate.get("text"))
        if not text:
            continue
        candidates.append(
            {
                "rank": index,
                "text": text,
                "engine": candidate.get("engine"),
                "language": candidate.get("language"),
                "psm": candidate.get("psm"),
                "role": candidate.get("role"),
                "status": candidate.get("status"),
                "trusted_as_final": False,
                "lines": candidate.get("lines") or [],
                "error": candidate.get("error"),
            }
        )

    return candidates


def _printed_context_token(unit, index):
    """Build one coordinate-aware printed token for N06/UI context."""

    printed_ocr = unit.get("printed_ocr") or {}
    candidates = _printed_candidate_records(printed_ocr)
    best = candidates[0] if candidates else {}
    return {
        "token_id": str(unit.get("text_unit_id") or f"printed_{index:04d}"),
        "text_unit_id": unit.get("text_unit_id"),
        "source_group_id": unit.get("source_group_id"),
        "layer": unit.get("layer"),
        "visual_class": unit.get("visual_class"),
        "text": best.get("text", ""),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "source": "n04_printed_ocr_context",
        "document_bbox": unit.get("document_bbox"),
        "final_bbox": unit.get("final_bbox"),
        "crop_bbox": unit.get("crop_bbox"),
        "n04_copied_crop_path": unit.get("n04_copied_crop_path"),
        "tesseract_ready_crop_path": unit.get("tesseract_ready_crop_path"),
        "tesseract_input_source": unit.get("tesseract_input_source"),
        "black_mask_reflected_ocr_crop_path": unit.get(
            "black_mask_reflected_ocr_crop_path"
        ),
        "black_mask_reflected_ocr_source_kind": unit.get(
            "black_mask_reflected_ocr_source_kind"
        ),
        "n04_crop_bbox_source": unit.get("n04_crop_bbox_source"),
        "original_document_bbox": unit.get("original_document_bbox"),
        "black_mask_derived_bbox": unit.get("black_mask_derived_bbox"),
        "black_mask_derived_bbox_reason": unit.get(
            "black_mask_derived_bbox_reason"
        ),
        "analysis_mask_used_for_tesseract": bool(
            unit.get("analysis_mask_used_for_tesseract")
        ),
        "mask_source": unit.get("mask_source"),
        "visual_layer_source": unit.get("visual_layer_source"),
        "n03_scores": unit.get("n03_scores") or {},
        "status": (
            "candidate_ready"
            if best.get("text")
            else printed_ocr.get("status", "empty_candidate")
        ),
        "trusted_as_final": False,
    }


def build_printed_context_layer(printed_text_units):
    """Build the N04 coordinate layer that mirrors N05 combined output.

    N05 produces word tokens with document bboxes for handwriting. This layer
    gives N06/UI the same style of surface for printed OCR context, while
    preserving the raw Tesseract candidates and the exact crop source used.
    """

    tokens = [
        _printed_context_token(unit, index)
        for index, unit in enumerate(printed_text_units, start=1)
        if isinstance(unit, dict)
    ]
    black_mask_tokens = [
        token
        for token in tokens
        if token.get("layer") == "black"
        and (
            token.get("analysis_mask_used_for_tesseract") is True
            or token.get("tesseract_input_source")
            == "black_mask_reflected_ocr_crop_path"
        )
    ]
    return {
        "version": "n04_printed_context_layer_v0_1",
        "status": "completed",
        "trusted_as_final": False,
        "coordinate_space": "original_document_image",
        "source": "N04_printed_ocr",
        "word_tokens": tokens,
        "black_mask_word_tokens": black_mask_tokens,
        "summary": {
            "token_count": len(tokens),
            "black_mask_token_count": len(black_mask_tokens),
            "tokens_with_text": sum(1 for token in tokens if token.get("text")),
            "empty_token_count": sum(1 for token in tokens if not token.get("text")),
        },
        "decision_note": (
            "Printed OCR context is raw evidence. N06 should use text, bboxes, "
            "and candidates for reconstruction, not as final truth."
        ),
    }


def summarize_printed_text_map(
    n03_payload,
    selected_routes,
    printed_text_units,
    skipped_records,
    failed_records
):
    """
    Build a compact summary for N04 printed text mapping.

    This summary tells us:
    - how many N03 routes were read
    - how many were selected for printed OCR
    - how many came from printed_only vs mixed
    - how many were skipped
    - how many failed
    - how many OCR attempts happened

    Raw OCR output is preserved as evidence; final text validation belongs to a
    downstream reconstruction stage.
    """
    printed_only_count = 0
    mixed_count = 0

    ocr_attempted_count = 0
    ocr_success_count = 0
    ocr_failed_count = 0
    placeholder_count = 0

    for unit in printed_text_units:
        visual_class = unit.get("visual_class")

        if visual_class == "printed_only":
            printed_only_count += 1

        elif visual_class == "mixed":
            mixed_count += 1

        printed_ocr = unit.get("printed_ocr", {})

        if printed_ocr.get("attempted") is True:
            ocr_attempted_count += 1

            if printed_ocr.get("status") in ["raw_candidate", "raw_candidates", "empty"]:
                ocr_success_count += 1

            elif printed_ocr.get("status") == "failed":
                ocr_failed_count += 1

        else:
            placeholder_count += 1

    return {
        "total_n03_routes_read": len(n03_payload.get("routes", [])),
        "printed_candidates_selected": len(selected_routes),

        "printed_only_count": printed_only_count,
        "mixed_count": mixed_count,

        "printed_text_units_count": len(printed_text_units),

        "skipped_count": len(skipped_records),
        "failed_count": len(failed_records),

        "ocr_attempted_count": ocr_attempted_count,
        "ocr_success_count": ocr_success_count,
        "ocr_failed_count": ocr_failed_count,
        "placeholder_count": placeholder_count
    }


def print_summary(document_id, summary, metadata_path):
    """
    Print a short terminal summary after N04 finishes.

    Detailed data lives inside the printed text map JSON.
    """
    print("-------------------------")
    print("N04 printed text map completed.")
    print("Document:", document_id)
    print("N03 routes read:", summary["total_n03_routes_read"])
    print("Printed candidates selected:", summary["printed_candidates_selected"])
    print("Printed only:", summary["printed_only_count"])
    print("Mixed:", summary["mixed_count"])
    print("Printed text units:", summary["printed_text_units_count"])
    print("Skipped:", summary["skipped_count"])
    print("Failed:", summary["failed_count"])
    print("OCR attempted:", summary["ocr_attempted_count"])
    print("Placeholders:", summary["placeholder_count"])
    print("Metadata:", metadata_path)
    print("-------------------------")
