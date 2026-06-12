"""Border-profile attachment rules for split cursive fragments."""

from n02_geometry import (
    clamp_bbox_to_image,
    horizontal_gap,
    horizontal_overlap_ratio,
    merge_bboxes,
    vertical_overlap_ratio,
)
from n02_io import crop_image
from n02_quality import build_ink_mask
from n02_text_units import build_text_unit_record, merged_text_unit_within_limits

def measure_border_profile(gray_image, bbox, side, settings):
    """Read one crop border as a 0/1 profile.

    Args:
        gray_image: Grayscale source image array.
        bbox: Candidate bbox to scan.
        side: One of left, right, top, or bottom.
        settings: RefinerSettings instance.

    Returns:
        Profile dictionary. For left/right, profile indexes are local y rows.
        For top/bottom, profile indexes are local x columns.
    """
    clamped_bbox = clamp_bbox_to_image(bbox, gray_image.shape)
    crop = crop_image(gray_image, clamped_bbox)

    if crop.size == 0:
        return {
            "side": side,
            "axis": "y" if side in {"left", "right"} else "x",
            "profile": [],
            "touched_indexes": [],
            "band_px": 0,
        }

    ink_mask = build_ink_mask(crop, settings)
    height, width = ink_mask.shape[:2]
    band_px = max(1, int(settings.border_profile_band_px))

    if side == "left":
        band_px = min(band_px, width)
        border_band = ink_mask[:, :band_px]
        profile = [1 if border_band[row, :].max() > 0 else 0 for row in range(height)]
        axis = "y"
    elif side == "right":
        band_px = min(band_px, width)
        border_band = ink_mask[:, width - band_px:width]
        profile = [1 if border_band[row, :].max() > 0 else 0 for row in range(height)]
        axis = "y"
    elif side == "top":
        band_px = min(band_px, height)
        border_band = ink_mask[:band_px, :]
        profile = [1 if border_band[:, col].max() > 0 else 0 for col in range(width)]
        axis = "x"
    elif side == "bottom":
        band_px = min(band_px, height)
        border_band = ink_mask[height - band_px:height, :]
        profile = [1 if border_band[:, col].max() > 0 else 0 for col in range(width)]
        axis = "x"
    else:
        raise ValueError(f"Unsupported border side: {side}")

    touched_indexes = [index for index, value in enumerate(profile) if value == 1]

    return {
        "side": side,
        "axis": axis,
        "profile": profile,
        "touched_indexes": touched_indexes,
        "band_px": band_px,
    }


def profile_offset_for_side(bbox, side):
    """Return document-coordinate offset for a border profile.

    Args:
        bbox: Bbox dictionary.
        side: One of left, right, top, or bottom.

    Returns:
        x1 for top/bottom profiles, y1 for left/right profiles.
    """
    if side in {"top", "bottom"}:
        return int(bbox["x1"])

    return int(bbox["y1"])


def match_border_profiles(first_profile, second_profile, first_offset, second_offset, settings):
    """Compare two 0/1 border profiles in document coordinates.

    Args:
        first_profile: First border profile dictionary.
        second_profile: Second border profile dictionary.
        first_offset: Document-coordinate offset for first profile indexes.
        second_offset: Document-coordinate offset for second profile indexes.
        settings: RefinerSettings instance.

    Returns:
        Match evidence dictionary, or None when too few pixels line up.
    """
    unmatched_second = set(second_profile["touched_indexes"])
    matches = []

    for first_index in first_profile["touched_indexes"]:
        first_global = first_offset + first_index
        best_second_index = None
        best_delta = None

        for second_index in unmatched_second:
            second_global = second_offset + second_index
            delta = abs(first_global - second_global)

            if delta > settings.border_profile_tolerance_px:
                continue

            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_second_index = second_index

        if best_second_index is None:
            continue

        unmatched_second.remove(best_second_index)
        matches.append({
            "first_index": first_index,
            "second_index": best_second_index,
            "first_global": first_global,
            "second_global": second_offset + best_second_index,
            "delta": best_delta,
        })

    if len(matches) < settings.border_profile_min_aligned_pixels:
        return None

    return {
        "matched_pixel_count": len(matches),
        "required_pixel_count": settings.border_profile_min_aligned_pixels,
        "tolerance_px": settings.border_profile_tolerance_px,
        "matches": matches,
        "first_touched_indexes": first_profile["touched_indexes"],
        "second_touched_indexes": second_profile["touched_indexes"],
    }


def build_profile_attachment_evidence(first_unit, second_unit, direction, gap, overlap, match, merged_bbox):
    """Create metadata for an accepted border-profile attachment.

    Args:
        first_unit: First text unit.
        second_unit: Second text unit.
        direction: right or down.
        gap: Gap between the two units on the connection axis.
        overlap: Overlap ratio on the perpendicular axis.
        match: Profile match evidence.
        merged_bbox: Proposed merged bbox.

    Returns:
        Attachment evidence dictionary.
    """
    return {
        "merge_reason": f"border_profile_continuity_{direction}",
        "direction": direction,
        "first_text_unit_id": first_unit["text_unit_id"],
        "second_text_unit_id": second_unit["text_unit_id"],
        "first_source_group_ids": first_unit["source_group_ids"],
        "second_source_group_ids": second_unit["source_group_ids"],
        "gap": gap,
        "overlap_ratio": round(overlap, 6),
        "first_bbox": dict(first_unit["bbox"]),
        "second_bbox": dict(second_unit["bbox"]),
        "candidate_merged_bbox": dict(merged_bbox),
        "border_profile_match": match,
        "merged_bbox": dict(merged_bbox),
    }


def evaluate_right_profile_attachment(left_unit, right_unit, gray_image, settings):
    """Evaluate whether two units connect through right/left border profiles.

    Args:
        left_unit: Candidate unit on the left.
        right_unit: Candidate unit on the right.
        gray_image: Grayscale source image array.
        settings: RefinerSettings instance.

    Returns:
        Tuple of (accepted_evidence, blocked_evidence).
    """
    gap = horizontal_gap(left_unit["bbox"], right_unit["bbox"])
    overlap = vertical_overlap_ratio(left_unit["bbox"], right_unit["bbox"])
    merged_bbox = merge_bboxes([left_unit["bbox"], right_unit["bbox"]])
    proposed_count = left_unit["source_group_count"] + right_unit["source_group_count"]
    blocked = {
        "merge_reason": "border_profile_continuity_right",
        "direction": "right",
        "first_text_unit_id": left_unit["text_unit_id"],
        "second_text_unit_id": right_unit["text_unit_id"],
        "first_source_group_ids": left_unit["source_group_ids"],
        "second_source_group_ids": right_unit["source_group_ids"],
        "gap": gap,
        "overlap_ratio": round(overlap, 6),
        "first_bbox": dict(left_unit["bbox"]),
        "second_bbox": dict(right_unit["bbox"]),
        "candidate_merged_bbox": dict(merged_bbox),
    }

    if gap < 0:
        return None, None

    if gap > settings.border_profile_max_horizontal_gap_px:
        return None, None

    if overlap < settings.border_profile_min_overlap_ratio:
        blocked["blocked_reason"] = "vertical_overlap_too_weak"
        return None, blocked

    if not merged_text_unit_within_limits(merged_bbox, proposed_count, settings):
        blocked["blocked_reason"] = "merged_unit_too_large"
        return None, blocked

    first_profile = measure_border_profile(gray_image, left_unit["bbox"], "right", settings)
    second_profile = measure_border_profile(gray_image, right_unit["bbox"], "left", settings)
    match = match_border_profiles(
        first_profile=first_profile,
        second_profile=second_profile,
        first_offset=profile_offset_for_side(left_unit["bbox"], "right"),
        second_offset=profile_offset_for_side(right_unit["bbox"], "left"),
        settings=settings,
    )

    if match is None:
        blocked["blocked_reason"] = "profiles_do_not_align"
        blocked["first_profile"] = first_profile["profile"]
        blocked["second_profile"] = second_profile["profile"]
        blocked["first_touched_indexes"] = first_profile["touched_indexes"]
        blocked["second_touched_indexes"] = second_profile["touched_indexes"]
        return None, blocked

    return build_profile_attachment_evidence(
        first_unit=left_unit,
        second_unit=right_unit,
        direction="right",
        gap=gap,
        overlap=overlap,
        match=match,
        merged_bbox=merged_bbox,
    ), None


def evaluate_down_profile_attachment(upper_unit, lower_unit, gray_image, settings):
    """Evaluate whether two units connect through bottom/top border profiles.

    Args:
        upper_unit: Candidate unit above.
        lower_unit: Candidate unit below.
        gray_image: Grayscale source image array.
        settings: RefinerSettings instance.

    Returns:
        Tuple of (accepted_evidence, blocked_evidence).
    """
    gap = int(lower_unit["bbox"]["y1"]) - int(upper_unit["bbox"]["y2"])
    overlap = horizontal_overlap_ratio(upper_unit["bbox"], lower_unit["bbox"])
    merged_bbox = merge_bboxes([upper_unit["bbox"], lower_unit["bbox"]])
    proposed_count = upper_unit["source_group_count"] + lower_unit["source_group_count"]
    blocked = {
        "merge_reason": "border_profile_continuity_down",
        "direction": "down",
        "first_text_unit_id": upper_unit["text_unit_id"],
        "second_text_unit_id": lower_unit["text_unit_id"],
        "first_source_group_ids": upper_unit["source_group_ids"],
        "second_source_group_ids": lower_unit["source_group_ids"],
        "gap": gap,
        "overlap_ratio": round(overlap, 6),
        "first_bbox": dict(upper_unit["bbox"]),
        "second_bbox": dict(lower_unit["bbox"]),
        "candidate_merged_bbox": dict(merged_bbox),
    }

    if gap < 0:
        return None, None

    if gap > settings.border_profile_max_vertical_gap_px:
        return None, None

    if overlap < settings.border_profile_min_overlap_ratio:
        blocked["blocked_reason"] = "horizontal_overlap_too_weak"
        return None, blocked

    if not merged_text_unit_within_limits(merged_bbox, proposed_count, settings):
        blocked["blocked_reason"] = "merged_unit_too_large"
        return None, blocked

    first_profile = measure_border_profile(gray_image, upper_unit["bbox"], "bottom", settings)
    second_profile = measure_border_profile(gray_image, lower_unit["bbox"], "top", settings)
    match = match_border_profiles(
        first_profile=first_profile,
        second_profile=second_profile,
        first_offset=profile_offset_for_side(upper_unit["bbox"], "bottom"),
        second_offset=profile_offset_for_side(lower_unit["bbox"], "top"),
        settings=settings,
    )

    if match is None:
        blocked["blocked_reason"] = "profiles_do_not_align"
        blocked["first_profile"] = first_profile["profile"]
        blocked["second_profile"] = second_profile["profile"]
        blocked["first_touched_indexes"] = first_profile["touched_indexes"]
        blocked["second_touched_indexes"] = second_profile["touched_indexes"]
        return None, blocked

    return build_profile_attachment_evidence(
        first_unit=upper_unit,
        second_unit=lower_unit,
        direction="down",
        gap=gap,
        overlap=overlap,
        match=match,
        merged_bbox=merged_bbox,
    ), None


def find_parent(parents, item):
    """Find a union-find parent with path compression.

    Args:
        parents: Parent lookup dictionary.
        item: Item id to resolve.

    Returns:
        Root parent id.
    """
    if parents[item] != item:
        parents[item] = find_parent(parents, parents[item])

    return parents[item]


def union_text_units(parents, left_id, right_id):
    """Union two text-unit ids.

    Args:
        parents: Parent lookup dictionary.
        left_id: First text-unit id.
        right_id: Second text-unit id.

    Returns:
        None.
    """
    left_root = find_parent(parents, left_id)
    right_root = find_parent(parents, right_id)

    if left_root == right_root:
        return

    parents[max(left_root, right_root)] = min(left_root, right_root)


def rebuild_text_units_after_profile_attachment(text_units, attachment_evidence):
    """Rebuild text units after border-profile unions.

    Args:
        text_units: Original text units.
        attachment_evidence: Accepted profile attachment evidence.

    Returns:
        New text-unit list with regenerated ids.
    """
    parents = {unit["text_unit_id"]: unit["text_unit_id"] for unit in text_units}

    for evidence in attachment_evidence:
        union_text_units(
            parents=parents,
            left_id=evidence["first_text_unit_id"],
            right_id=evidence["second_text_unit_id"],
        )

    clusters = {}

    for unit in text_units:
        root = find_parent(parents, unit["text_unit_id"])
        clusters.setdefault(root, []).append(unit)

    ordered_clusters = sorted(
        clusters.values(),
        key=lambda cluster: (
            min(unit["bbox"]["y1"] for unit in cluster),
            min(unit["bbox"]["x1"] for unit in cluster),
        )
    )
    rebuilt = []

    for new_id, cluster in enumerate(ordered_clusters, start=1):
        source_groups = []
        grouping_evidence = []
        line_bucket_ids = [unit["line_bucket_id"] for unit in cluster if unit.get("line_bucket_id") is not None]
        cluster_unit_ids = {unit["text_unit_id"] for unit in cluster}

        for unit in cluster:
            source_groups.extend(unit.get("source_groups", []))
            grouping_evidence.extend(unit.get("grouping_evidence", []))

        grouping_evidence.extend(
            evidence
            for evidence in attachment_evidence
            if (
                evidence["first_text_unit_id"] in cluster_unit_ids
                and evidence["second_text_unit_id"] in cluster_unit_ids
            )
        )

        rebuilt.append(build_text_unit_record(
            text_unit_id=new_id,
            line_bucket_id=min(line_bucket_ids) if line_bucket_ids else None,
            source_groups=source_groups,
            grouping_evidence=grouping_evidence,
        ))

    return rebuilt


def attach_border_profile_fragments(text_units, gray_image, settings):
    """Attach text units whose facing border profiles align.

    Args:
        text_units: Text units after initial bbox grouping.
        gray_image: Grayscale source image array.
        settings: RefinerSettings instance.

    Returns:
        Tuple of rebuilt text units, accepted evidence, and blocked evidence.
    """
    candidate_edges = []
    accepted = []
    blocked = []
    ordered_units = sorted(text_units, key=lambda unit: (unit["bbox"]["y1"], unit["bbox"]["x1"], unit["text_unit_id"]))

    for first_index, first_unit in enumerate(ordered_units):
        for second_unit in ordered_units[first_index + 1:]:
            if first_unit.get("layer", "legacy") != second_unit.get("layer", "legacy"):
                continue

            right_evidence, right_blocked = evaluate_right_profile_attachment(
                left_unit=first_unit,
                right_unit=second_unit,
                gray_image=gray_image,
                settings=settings,
            )

            down_evidence, down_blocked = evaluate_down_profile_attachment(
                upper_unit=first_unit,
                lower_unit=second_unit,
                gray_image=gray_image,
                settings=settings,
            )

            pair_candidates = [
                item
                for item in [right_evidence, down_evidence]
                if item is not None
            ]

            if pair_candidates:
                pair_candidates.sort(
                    key=lambda item: (
                        item["gap"],
                        -item["border_profile_match"]["matched_pixel_count"],
                    )
                )
                candidate_edges.append(pair_candidates[0])
                continue

            for item in [right_blocked, down_blocked]:
                if item is not None:
                    blocked.append(item)

    candidate_edges.sort(
        key=lambda item: (
            item["gap"],
            -item["border_profile_match"]["matched_pixel_count"],
            item["first_text_unit_id"],
            item["second_text_unit_id"],
        )
    )

    parents = {
        unit["text_unit_id"]: unit["text_unit_id"]
        for unit in text_units
    }
    for edge in candidate_edges:
        first_root = find_parent(parents, edge["first_text_unit_id"])
        second_root = find_parent(parents, edge["second_text_unit_id"])

        if first_root == second_root:
            continue

        proposed_units = [
            unit
            for unit in text_units
            if find_parent(parents, unit["text_unit_id"]) in {first_root, second_root}
        ]
        proposed_bbox = merge_bboxes(unit["bbox"] for unit in proposed_units)
        proposed_count = sum(unit["source_group_count"] for unit in proposed_units)

        if not merged_text_unit_within_limits(proposed_bbox, proposed_count, settings):
            rejected_edge = dict(edge)
            rejected_edge["blocked_reason"] = "profile_chain_limit_exceeded"
            rejected_edge["cluster_candidate_bbox"] = proposed_bbox
            rejected_edge["cluster_candidate_source_count"] = proposed_count
            blocked.append(rejected_edge)
            continue

        union_text_units(
            parents=parents,
            left_id=edge["first_text_unit_id"],
            right_id=edge["second_text_unit_id"],
        )
        accepted.append(edge)

    if not accepted:
        return text_units, accepted, blocked

    return rebuild_text_units_after_profile_attachment(text_units, accepted), accepted, blocked
