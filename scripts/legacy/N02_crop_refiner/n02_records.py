"""Output-record builders for N02 refined groups."""

def build_refiner_payload(status, score, bbox, quality_result, next_node):
    """Build the nested refiner payload expected by downstream nodes.

    Args:
        status: Final N02 status.
        score: Numeric crop quality score.
        bbox: Final crop bbox.
        quality_result: Quality evidence dictionary.
        next_node: Suggested next pipeline destination.

    Returns:
        Refiner metadata dictionary.
    """
    return {
        "status": status,
        "status_reason": ",".join(quality_result.get("infractions", [])) or "quality_ok",
        "next_node": next_node,
        "final_bbox": bbox,
        "final_score": score,
        "quality": quality_result,
    }


def build_refined_text_unit_record(text_unit, final_bbox, quality_result, crop_path):
    """Create a final refined-group record for one text unit.

    Args:
        text_unit: Text-unit dictionary from Section 7.
        final_bbox: Final padded/clamped crop bbox.
        quality_result: Crop quality result from Section 8.
        crop_path: Path to the saved crop image.

    Returns:
        Refined group dictionary compatible with N03.
    """
    status = quality_result["status"]
    score = quality_result["score"]
    next_node = "visual_classification" if status in {"accepted", "review"} else "none"

    return {
        "text_unit_id": text_unit["text_unit_id"],
        "line_bucket_id": text_unit["line_bucket_id"],
        "layer": text_unit.get("layer", "legacy"),
        "source_layers": text_unit.get("source_layers", []),
        "source_group_ids": text_unit["source_group_ids"],
        "source_group_count": text_unit["source_group_count"],
        "bbox": dict(text_unit["bbox"]),
        "final_bbox": dict(final_bbox),
        "grouping_reason": text_unit["grouping_reason"],
        "grouping_evidence": text_unit["grouping_evidence"],
        "quality_score": score,
        "quality_evidence": quality_result,
        "refiner": build_refiner_payload(
            status=status,
            score=score,
            bbox=dict(final_bbox),
            quality_result=quality_result,
            next_node=next_node,
        ),
        "refined_crop_path": crop_path,
    }


def build_refinement_summary(refined_groups):
    """Count final statuses for a document.

    Args:
        refined_groups: List of final refined group dictionaries.

    Returns:
        Summary count dictionary.
    """
    summary = {
        "accepted_count": 0,
        "review_count": 0,
        "rejected_count": 0,
        "visual_classification_count": 0,
        "none_count": 0,
    }

    for group in refined_groups:
        refiner = group.get("refiner", {})
        status = refiner.get("status", "review")
        next_node = refiner.get("next_node", "visual_classification")

        if status == "accepted":
            summary["accepted_count"] += 1
        elif status == "rejected":
            summary["rejected_count"] += 1
        else:
            summary["review_count"] += 1

        if next_node == "visual_classification":
            summary["visual_classification_count"] += 1
        elif next_node == "none":
            summary["none_count"] += 1

    return summary
