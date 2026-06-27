"""Split stacked ScribeMap groups when two text rows touch.

Connected components cannot separate two words that physically touch in the
mask. This module adds a conservative horizontal projection guard: if a group is
unusually tall and has a low-ink horizontal valley, split its bbox into top and
bottom row candidates before final artifact filtering.
"""

from __future__ import annotations

import cv2
import numpy as np


def _smooth_projection(projection: np.ndarray, window: int) -> np.ndarray:
    """Return a small moving-average projection for stable valley detection."""

    window = max(1, int(window))
    if window <= 1 or projection.size == 0:
        return projection.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(projection.astype(np.float32), kernel, mode="same")


def _candidate_split_y(crop_mask: np.ndarray, settings: dict) -> tuple[int | None, dict]:
    """Find a strong horizontal low-ink valley inside one group crop."""

    height, width = crop_mask.shape[:2]
    min_part_height = int(settings.get("stacked_group_min_part_height_px", 18))
    if height < min_part_height * 2:
        return None, {"reason": "too_short_for_two_rows"}

    projection = np.count_nonzero(crop_mask > 0, axis=1)
    smoothed = _smooth_projection(
        projection,
        int(settings.get("stacked_group_projection_smoothing_px", 3)),
    )
    search_start = min_part_height
    search_end = height - min_part_height
    if search_end <= search_start:
        return None, {"reason": "no_valid_search_band"}

    search_band = smoothed[search_start:search_end]
    if search_band.size == 0:
        return None, {"reason": "empty_search_band"}

    local_index = int(np.argmin(search_band))
    split_y = search_start + local_index
    valley_value = float(smoothed[split_y])
    max_projection = float(smoothed.max(initial=0.0))
    mean_projection = float(smoothed.mean()) if smoothed.size else 0.0
    valley_ratio_to_max = valley_value / max(1.0, max_projection)
    valley_ratio_to_mean = valley_value / max(1.0, mean_projection)

    top_ink = int(np.count_nonzero(crop_mask[:split_y, :] > 0))
    bottom_ink = int(np.count_nonzero(crop_mask[split_y:, :] > 0))
    min_part_ink = int(settings.get("stacked_group_min_part_ink_px", 20))
    if top_ink < min_part_ink or bottom_ink < min_part_ink:
        return None, {
            "reason": "one_side_has_too_little_ink",
            "top_ink": top_ink,
            "bottom_ink": bottom_ink,
        }

    max_valley_ratio = float(settings.get("stacked_group_max_valley_ratio", 0.22))
    max_mean_ratio = float(settings.get("stacked_group_max_valley_mean_ratio", 0.55))
    if valley_ratio_to_max > max_valley_ratio and valley_ratio_to_mean > max_mean_ratio:
        return None, {
            "reason": "no_strong_horizontal_valley",
            "valley_ratio_to_max": valley_ratio_to_max,
            "valley_ratio_to_mean": valley_ratio_to_mean,
        }

    return split_y, {
        "reason": "horizontal_projection_valley",
        "split_y_local": split_y,
        "valley_ink": valley_value,
        "max_projection": max_projection,
        "mean_projection": mean_projection,
        "valley_ratio_to_max": valley_ratio_to_max,
        "valley_ratio_to_mean": valley_ratio_to_mean,
        "top_ink": top_ink,
        "bottom_ink": bottom_ink,
    }


def _bbox_from_mask(parent_group: dict, mask_slice: np.ndarray, y_offset: int) -> dict | None:
    """Build a tight bbox for the ink inside a split half."""

    points = cv2.findNonZero((mask_slice > 0).astype(np.uint8))
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    x1 = int(parent_group["x1"] + x)
    y1 = int(parent_group["y1"] + y_offset + y)
    x2 = int(x1 + width)
    y2 = int(y1 + height)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _make_split_group(parent_group: dict, bbox: dict, split_index: int, split_meta: dict, mask: np.ndarray) -> dict:
    """Create one split child group with refreshed geometry."""

    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    crop = mask[y1:y2, x1:x2]
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    box_area = max(1, width * height)
    ink_area = int(np.count_nonzero(crop > 0))
    group = dict(parent_group)
    group.update(
        {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "width": int(width),
            "height": int(height),
            "box_area": int(box_area),
            "ink_area": int(ink_area),
            "density": round(float(ink_area / box_area), 4),
            "aspect_ratio": round(float(width / max(1, height)), 3),
            "center_x": round(float((x1 + x2) / 2), 2),
            "center_y": round(float((y1 + y2) / 2), 2),
            "split_parent_group_id": parent_group.get("group_id"),
            "split_index": int(split_index),
            "stacked_split": split_meta,
        }
    )
    flags = list(parent_group.get("group_flags", []))
    if "split_from_stacked_group" not in flags:
        flags.append("split_from_stacked_group")
    group["group_flags"] = flags
    return group


def split_stacked_groups(groups: list[dict], mask: np.ndarray, settings: dict) -> tuple[list[dict], list[dict]]:
    """Split suspicious tall groups into top/bottom row candidates."""

    if not bool(settings.get("enable_stacked_group_split", True)):
        return groups, []

    split_groups = []
    split_events = []
    min_height = int(settings.get("stacked_group_min_height_px", 62))
    min_width = int(settings.get("stacked_group_min_width_px", 20))
    max_aspect = float(settings.get("stacked_group_max_aspect_ratio", 8.0))

    for group in groups:
        width = int(group.get("width", 0))
        height = int(group.get("height", 0))
        aspect = float(group.get("aspect_ratio", width / max(1, height)))
        if height < min_height or width < min_width or aspect > max_aspect:
            split_groups.append(group)
            continue

        x1, y1, x2, y2 = (
            int(group["x1"]),
            int(group["y1"]),
            int(group["x2"]),
            int(group["y2"]),
        )
        crop_mask = mask[y1:y2, x1:x2]
        split_y_local, split_meta = _candidate_split_y(crop_mask, settings)
        if split_y_local is None:
            split_groups.append(group)
            continue

        top_bbox = _bbox_from_mask(group, crop_mask[:split_y_local, :], 0)
        bottom_bbox = _bbox_from_mask(group, crop_mask[split_y_local:, :], split_y_local)
        if not top_bbox or not bottom_bbox:
            split_groups.append(group)
            continue

        split_meta = dict(split_meta)
        split_meta["split_y_absolute"] = int(y1 + split_y_local)
        split_meta["source_group_id"] = group.get("group_id")
        split_groups.extend(
            [
                _make_split_group(group, top_bbox, 0, split_meta, mask),
                _make_split_group(group, bottom_bbox, 1, split_meta, mask),
            ]
        )
        split_events.append(split_meta)

    split_groups = sorted(split_groups, key=lambda row: (row["y1"], row["x1"]))
    for index, group in enumerate(split_groups, start=1):
        group["group_id"] = index
    return split_groups, split_events
