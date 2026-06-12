"""Document-level line bucketing and conservative text-unit grouping."""

from n02_geometry import (
    bbox_area,
    bbox_aspect_ratio,
    bbox_center_x,
    bbox_center_y,
    bbox_height,
    bbox_width,
    height_ratio,
    horizontal_gap,
    merge_bboxes,
    vertical_overlap_ratio,
)

def build_line_bucket_record(bucket_id, groups):
    """Create a line-bucket metadata record from grouped candidates.

    Args:
        bucket_id: Stable numeric line bucket id.
        groups: Candidate groups assigned to this line bucket.

    Returns:
        Line bucket dictionary sorted left-to-right.
    """
    sorted_groups = sorted(
        groups,
        key=lambda group: (
            group["bbox"]["x1"],
            group["bbox"]["y1"],
            group["source_group_id"],
        )
    )
    bucket_bbox = merge_bboxes(group["bbox"] for group in sorted_groups)

    return {
        "line_bucket_id": bucket_id,
        "bbox": bucket_bbox,
        "center_y": bbox_center_y(bucket_bbox),
        "height": bbox_height(bucket_bbox),
        "source_group_ids": [
            group["source_group_id"]
            for group in sorted_groups
        ],
        "groups": sorted_groups,
    }


def group_fits_line_bucket(group, bucket, settings):
    """Check whether a candidate belongs in an existing line bucket.

    Args:
        group: Normalized candidate group.
        bucket: Existing line bucket record.
        settings: RefinerSettings instance.

    Returns:
        True when the group looks like it belongs to the bucket's row.
    """
    center_y_distance = abs(group["center_y"] - bucket["center_y"])
    overlap_ratio = vertical_overlap_ratio(group["bbox"], bucket["bbox"])

    return (
        center_y_distance <= settings.line_bucket_max_center_y_distance
        or overlap_ratio >= settings.line_bucket_min_vertical_overlap
    )


def rebuild_line_bucket(bucket_id, old_bucket, new_group):
    """Return an updated line bucket after adding one group.

    Args:
        bucket_id: Existing bucket id.
        old_bucket: Current line bucket record.
        new_group: Normalized group to add.

    Returns:
        Updated line bucket record.
    """
    return build_line_bucket_record(
        bucket_id=bucket_id,
        groups=old_bucket["groups"] + [new_group],
    )


def build_line_buckets(candidate_groups, settings):
    """Organize candidate groups into approximate same-line buckets.

    Args:
        candidate_groups: Groups that survived early artifact filtering.
        settings: RefinerSettings instance.

    Returns:
        List of line bucket dictionaries sorted top-to-bottom.
    """
    sorted_groups = sorted(
        candidate_groups,
        key=lambda group: (
            group["bbox"]["y1"],
            group["center_y"],
            group["bbox"]["x1"],
            group["source_group_id"],
        )
    )
    buckets = []

    for group in sorted_groups:
        best_bucket_index = None
        best_distance = None

        for bucket_index, bucket in enumerate(buckets):
            if not group_fits_line_bucket(group, bucket, settings):
                continue

            distance = abs(group["center_y"] - bucket["center_y"])

            if best_distance is None or distance < best_distance:
                best_bucket_index = bucket_index
                best_distance = distance

        if best_bucket_index is None:
            bucket_id = len(buckets) + 1
            buckets.append(
                build_line_bucket_record(
                    bucket_id=bucket_id,
                    groups=[group],
                )
            )
            continue

        bucket_id = buckets[best_bucket_index]["line_bucket_id"]
        buckets[best_bucket_index] = rebuild_line_bucket(
            bucket_id=bucket_id,
            old_bucket=buckets[best_bucket_index],
            new_group=group,
        )

    ordered_buckets = sorted(
        buckets,
        key=lambda bucket: (
            bucket["bbox"]["y1"],
            bucket["bbox"]["x1"],
            bucket["line_bucket_id"],
        )
    )

    for new_id, bucket in enumerate(ordered_buckets, start=1):
        bucket["line_bucket_id"] = new_id

    return ordered_buckets


def merged_text_unit_within_limits(bbox, group_count, settings):
    """Check hard safety limits for a proposed text unit.

    Args:
        bbox: Proposed merged bbox.
        group_count: Number of source groups inside the proposed unit.
        settings: RefinerSettings instance.

    Returns:
        True if the proposed unit remains OCR-sized and safe.
    """
    if bbox_width(bbox) > settings.text_unit_max_merged_width:
        return False

    if bbox_height(bbox) > settings.text_unit_max_merged_height:
        return False

    if bbox_area(bbox) > settings.text_unit_max_merged_area:
        return False

    if group_count > settings.text_unit_max_groups_per_cluster:
        return False

    return True


def can_merge_line_neighbors(left_group, right_group, settings):
    """Decide whether two same-line neighbors should become one text unit.

    Args:
        left_group: Normalized group on the left.
        right_group: Normalized group on the right.
        settings: RefinerSettings instance.

    Returns:
        Tuple of:
            should_merge: Boolean merge decision.
            evidence: Dictionary with measured merge features.
    """
    gap = horizontal_gap(left_group["bbox"], right_group["bbox"])
    overlap = vertical_overlap_ratio(left_group["bbox"], right_group["bbox"])
    center_y_distance = abs(left_group["center_y"] - right_group["center_y"])
    ratio = height_ratio(left_group["bbox"], right_group["bbox"])
    merged_bbox = merge_bboxes([left_group["bbox"], right_group["bbox"]])

    evidence = {
        "direction": "right",
        "left_source_group_id": left_group["source_group_id"],
        "right_source_group_id": right_group["source_group_id"],
        "gap": gap,
        "vertical_overlap_ratio": round(overlap, 6),
        "center_y_distance": round(center_y_distance, 6),
        "height_ratio": round(ratio, 6),
        "left_bbox": dict(left_group["bbox"]),
        "right_bbox": dict(right_group["bbox"]),
        "candidate_merged_bbox": dict(merged_bbox),
    }

    if gap < 0:
        evidence["blocked_reason"] = "horizontal_overlap"
        return False, evidence

    if gap > settings.text_unit_max_horizontal_gap_px:
        evidence["blocked_reason"] = "gap_too_large"
        return False, evidence

    if overlap < settings.text_unit_min_vertical_overlap:
        evidence["blocked_reason"] = "vertical_overlap_too_weak"
        return False, evidence

    if center_y_distance > settings.text_unit_max_center_y_distance:
        evidence["blocked_reason"] = "center_y_distance_too_large"
        return False, evidence

    if ratio > settings.text_unit_max_height_ratio:
        evidence["blocked_reason"] = "height_ratio_too_extreme"
        return False, evidence

    if not merged_text_unit_within_limits(merged_bbox, group_count=2, settings=settings):
        evidence["blocked_reason"] = "merged_bbox_too_large"
        return False, evidence

    evidence["merge_reason"] = "same_line_neighbor_merge"
    return True, evidence


def build_text_unit_record(text_unit_id, line_bucket_id, source_groups, grouping_evidence):
    """Create one OCR-ready text-unit record.

    Args:
        text_unit_id: Stable text-unit id.
        line_bucket_id: ID of the line bucket this unit came from.
        source_groups: Normalized source groups included in this unit.
        grouping_evidence: Evidence records for accepted neighbor merges.

    Returns:
        Text-unit dictionary ready for crop quality scoring.
    """
    ordered_groups = sorted(
        source_groups,
        key=lambda group: (
            group["bbox"]["x1"],
            group["bbox"]["y1"],
            group["source_group_id"],
        )
    )
    bbox = merge_bboxes(group["bbox"] for group in ordered_groups)
    source_group_ids = [
        group["source_group_id"]
        for group in ordered_groups
    ]
    source_layers = sorted({
        group.get("layer", "legacy")
        for group in ordered_groups
    })
    layer = source_layers[0] if len(source_layers) == 1 else "mixed"

    if grouping_evidence:
        reasons = sorted({
            item.get("merge_reason", "grouping_merge")
            for item in grouping_evidence
        })
        grouping_reason = "+".join(reasons)
    else:
        grouping_reason = "single_source_group"

    return {
        "text_unit_id": text_unit_id,
        "line_bucket_id": line_bucket_id,
        "layer": layer,
        "source_layers": source_layers,
        "source_group_ids": source_group_ids,
        "source_group_count": len(source_group_ids),
        "bbox": bbox,
        "width": bbox_width(bbox),
        "height": bbox_height(bbox),
        "area": bbox_area(bbox),
        "center_x": bbox_center_x(bbox),
        "center_y": bbox_center_y(bbox),
        "aspect_ratio": bbox_aspect_ratio(bbox),
        "grouping_reason": grouping_reason,
        "grouping_evidence": grouping_evidence,
        "source_groups": ordered_groups,
    }


def finalize_text_unit_cluster(text_unit_id, line_bucket_id, cluster_groups, cluster_evidence):
    """Finalize one current cluster into a text-unit record.

    Args:
        text_unit_id: Stable text-unit id.
        line_bucket_id: Parent line bucket id.
        cluster_groups: Source groups currently in the cluster.
        cluster_evidence: Accepted merge evidence inside the cluster.

    Returns:
        Text-unit dictionary.
    """
    return build_text_unit_record(
        text_unit_id=text_unit_id,
        line_bucket_id=line_bucket_id,
        source_groups=cluster_groups,
        grouping_evidence=cluster_evidence,
    )


def build_text_units_from_line_bucket(line_bucket, starting_text_unit_id, settings):
    """Merge safe left/right neighbors inside one line bucket.

    Args:
        line_bucket: Line bucket dictionary from Section 6.
        starting_text_unit_id: First id available for generated text units.
        settings: RefinerSettings instance.

    Returns:
        Tuple:
            text_units: Text units produced from this line.
            next_text_unit_id: Next unused text-unit id.
            blocked_merge_evidence: Rejected neighbor-merge evidence.
    """
    groups = line_bucket["groups"]

    if not groups:
        return [], starting_text_unit_id, []

    text_units = []
    blocked_merge_evidence = []
    current_cluster = [groups[0]]
    current_evidence = []
    next_text_unit_id = starting_text_unit_id

    for candidate in groups[1:]:
        cluster_bbox = merge_bboxes(group["bbox"] for group in current_cluster)
        cluster_proxy = {
            "source_group_id": current_cluster[-1]["source_group_id"],
            "bbox": cluster_bbox,
            "width": bbox_width(cluster_bbox),
            "height": bbox_height(cluster_bbox),
            "area": bbox_area(cluster_bbox),
            "center_x": bbox_center_x(cluster_bbox),
            "center_y": bbox_center_y(cluster_bbox),
            "aspect_ratio": bbox_aspect_ratio(cluster_bbox),
        }

        should_merge, evidence = can_merge_line_neighbors(
            left_group=cluster_proxy,
            right_group=candidate,
            settings=settings,
        )

        proposed_bbox = merge_bboxes([cluster_bbox, candidate["bbox"]])
        proposed_count = len(current_cluster) + 1

        if should_merge and not merged_text_unit_within_limits(
            bbox=proposed_bbox,
            group_count=proposed_count,
            settings=settings,
        ):
            should_merge = False
            evidence["blocked_reason"] = "cluster_limit_exceeded"

        if should_merge:
            evidence["merged_bbox"] = dict(proposed_bbox)
            current_cluster.append(candidate)
            current_evidence.append(evidence)
            continue

        blocked_merge_evidence.append(evidence)
        text_units.append(
            finalize_text_unit_cluster(
                text_unit_id=next_text_unit_id,
                line_bucket_id=line_bucket["line_bucket_id"],
                cluster_groups=current_cluster,
                cluster_evidence=current_evidence,
            )
        )
        next_text_unit_id += 1
        current_cluster = [candidate]
        current_evidence = []

    text_units.append(
        finalize_text_unit_cluster(
            text_unit_id=next_text_unit_id,
            line_bucket_id=line_bucket["line_bucket_id"],
            cluster_groups=current_cluster,
            cluster_evidence=current_evidence,
        )
    )
    next_text_unit_id += 1

    return text_units, next_text_unit_id, blocked_merge_evidence


def build_text_units(line_buckets, settings):
    """Build OCR-ready text units from all line buckets.

    Args:
        line_buckets: List of line bucket dictionaries.
        settings: RefinerSettings instance.

    Returns:
        Tuple:
            text_units: List of generated text-unit dictionaries.
            blocked_merge_evidence: List of rejected neighbor-merge records.
    """
    text_units = []
    blocked_merge_evidence = []
    next_text_unit_id = 1

    for line_bucket in line_buckets:
        bucket_units, next_text_unit_id, bucket_blocked = build_text_units_from_line_bucket(
            line_bucket=line_bucket,
            starting_text_unit_id=next_text_unit_id,
            settings=settings,
        )
        text_units.extend(bucket_units)
        blocked_merge_evidence.extend(bucket_blocked)

    return text_units, blocked_merge_evidence
