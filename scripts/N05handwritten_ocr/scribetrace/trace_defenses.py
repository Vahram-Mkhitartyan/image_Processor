"""ScribeTrace defense candidate generators.

This file creates visualizable repair hypotheses.

Important:
- It does not decide which defenses are allowed.
- It does not decide final acceptance.
- It only generates candidate masks + added/removed masks.

Reconstruction/verifier will:
- retrace each candidate
- score it
- accept/reject
- save debug UI images
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from .trace_defense_registry import (
    DEFENSE_BORDER_CONTINUATION,
    DEFENSE_COMPONENT_DENOISING,
    DEFENSE_CONSERVATIVE_STROKE_RECOVERY,
    DEFENSE_CONTAMINATION_OPENING,
    DEFENSE_HORIZONTAL_GAP_CLOSING,
    DEFENSE_LINEAR_ARTIFACT_REMOVAL,
    DEFENSE_MEDIAN_DENOISING,
    DEFENSE_STAMP_EXTERNAL_ARTIFACT_REMOVAL,
    DEFENSE_THRESHOLD_NORMALIZATION,
    DEFENSE_VERTICAL_GAP_CLOSING,
    get_defense_spec_dict,
    implemented_in_trace_defenses,
)


@dataclass
class DefenseHypothesis:
    """One candidate repair proposed by a defense tool."""

    hypothesis_id: str
    defense_name: str
    candidate_mask: np.ndarray
    added_mask: np.ndarray
    removed_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_arrays: bool = False) -> dict:
        data = {
            "hypothesis_id": self.hypothesis_id,
            "defense_name": self.defense_name,
            "metadata": dict(self.metadata),
        }
        if include_arrays:
            data["candidate_mask"] = self.candidate_mask
            data["added_mask"] = self.added_mask
            data["removed_mask"] = self.removed_mask
        return data


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return uint8 mask where foreground is 255 and background is 0."""
    if mask is None:
        raise ValueError("mask cannot be None")

    arr = np.asarray(mask)

    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    arr = arr.astype(np.uint8)

    # If already binary-ish, keep it.
    unique = np.unique(arr)
    if len(unique) <= 3 and set(int(v) for v in unique).issubset({0, 1, 255}):
        return np.where(arr > 0, 255, 0).astype(np.uint8)

    # Polarity inference by border median.
    border = np.concatenate(
        [
            arr[0, :],
            arr[-1, :],
            arr[:, 0],
            arr[:, -1],
        ]
    )
    background = float(np.median(border))

    if background > 127:
        # Light background, dark ink.
        return np.where(arr < 215, 255, 0).astype(np.uint8)

    # Dark background, light ink.
    return np.where(arr > 40, 255, 0).astype(np.uint8)


def _changed_masks(original: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    original_bin = _as_binary_mask(original)
    candidate_bin = _as_binary_mask(candidate)

    added = np.where((candidate_bin > 0) & (original_bin == 0), 255, 0).astype(np.uint8)
    removed = np.where((candidate_bin == 0) & (original_bin > 0), 255, 0).astype(np.uint8)
    return added, removed


def _foreground_count(mask: np.ndarray) -> int:
    return int(cv2.countNonZero(_as_binary_mask(mask)))


def _make_hypothesis(
    *,
    index: int,
    defense_name: str,
    original_mask: np.ndarray,
    candidate_mask: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> DefenseHypothesis | None:
    original_bin = _as_binary_mask(original_mask)
    candidate_bin = _as_binary_mask(candidate_mask)

    added, removed = _changed_masks(original_bin, candidate_bin)

    added_px = int(cv2.countNonZero(added))
    removed_px = int(cv2.countNonZero(removed))
    changed_px = added_px + removed_px
    original_px = max(1, int(cv2.countNonZero(original_bin)))

    if changed_px == 0:
        return None

    meta = dict(metadata or {})
    spec = get_defense_spec_dict(defense_name)

    meta.update(
        {
            "defense_spec": spec,
            "added_ink_pixels": added_px,
            "removed_ink_pixels": removed_px,
            "changed_ink_pixels": changed_px,
            "added_ink_ratio": added_px / original_px,
            "removed_ink_ratio": removed_px / original_px,
            "changed_ink_ratio": changed_px / original_px,
        }
    )

    return DefenseHypothesis(
        hypothesis_id=f"h{index}_{defense_name}",
        defense_name=defense_name,
        candidate_mask=candidate_bin,
        added_mask=added,
        removed_mask=removed,
        metadata=meta,
    )


def _unpack_candidate_payload(payload):
    """Allow defense functions to return either mask or (mask, metadata)."""
    if isinstance(payload, tuple) and len(payload) == 2:
        candidate_mask, candidate_metadata = payload
        return candidate_mask, dict(candidate_metadata or {})

    return payload, {}


def _connected_components(mask: np.ndarray):
    binary = _as_binary_mask(mask)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    return count, labels, stats, centroids


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    binary = _as_binary_mask(mask)
    count, labels, stats, _ = _connected_components(binary)

    if count <= 1:
        return binary.copy()

    largest_label = 1
    largest_area = int(stats[1, cv2.CC_STAT_AREA])

    for label in range(2, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > largest_area:
            largest_area = area
            largest_label = label

    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def _glyph_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    binary = _as_binary_mask(mask)
    ys, xs = np.where(binary > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def _defense_stamp_external_artifact_removal(mask: np.ndarray) -> list[Any]:
    """Remove external stamp/border-like line artifacts while protecting glyph core.

    This is intentionally different from linear_artifact_removal.

    linear_artifact_removal:
        - attacks a line crossing the glyph body
        - may temporarily wound topology
        - can require downstream bridge/gap repair

    stamp_external_artifact_removal:
        - attacks long external/border-like artifacts
        - avoids deleting the central glyph body
        - should not be allowed to freely damage topology
    """
    binary = _as_binary_mask(mask)
    h, w = binary.shape[:2]

    total_ink = cv2.countNonZero(binary)
    if total_ink <= 0:
        return []

    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return []

    ink_x1 = int(xs.min())
    ink_x2 = int(xs.max())
    ink_y1 = int(ys.min())
    ink_y2 = int(ys.max())

    ink_w = max(1, ink_x2 - ink_x1 + 1)
    ink_h = max(1, ink_y2 - ink_y1 + 1)

    if ink_w < 8 or ink_h < 8:
        return []

    scored_candidates: list[tuple[float, object]] = []

    def _dedupe_and_rank(
        items: list[tuple[float, object]],
        max_count: int = 12,
    ) -> list[object]:
        items = sorted(items, key=lambda pair: pair[0], reverse=True)

        seen = set()
        output: list[object] = []

        for _, payload in items:
            candidate_mask, _metadata = _unpack_candidate_payload(payload)
            candidate_bin = _as_binary_mask(candidate_mask)

            key = candidate_bin.tobytes()
            if key in seen:
                continue

            seen.add(key)
            output.append(payload)

            if len(output) >= max_count:
                break

        return output

    # Build a conservative protected glyph-core box.
    # Percentiles reduce the influence of long external stamp lines.
    qx1, qx2 = np.percentile(xs, [18, 82])
    qy1, qy2 = np.percentile(ys, [12, 88])

    core_pad = max(2, int(min(ink_w, ink_h) * 0.08))

    core_x1 = max(0, int(qx1) - core_pad)
    core_x2 = min(w - 1, int(qx2) + core_pad)
    core_y1 = max(0, int(qy1) - core_pad)
    core_y2 = min(h - 1, int(qy2) + core_pad)

    protected_core = np.zeros_like(binary)
    protected_core[core_y1 : core_y2 + 1, core_x1 : core_x2 + 1] = 255

    def _artifact_externality(artifact_mask: np.ndarray) -> dict[str, float]:
        artifact_bin = _as_binary_mask(artifact_mask)
        artifact_px = cv2.countNonZero(artifact_bin)

        if artifact_px <= 0:
            return {
                "artifact_pixels": 0.0,
                "protected_overlap_pixels": 0.0,
                "outside_pixels": 0.0,
                "protected_overlap_ratio": 1.0,
                "outside_ratio": 0.0,
            }

        protected_overlap = cv2.bitwise_and(artifact_bin, protected_core)
        protected_px = cv2.countNonZero(protected_overlap)
        outside_px = max(0, artifact_px - protected_px)

        return {
            "artifact_pixels": float(artifact_px),
            "protected_overlap_pixels": float(protected_px),
            "outside_pixels": float(outside_px),
            "protected_overlap_ratio": float(protected_px / max(1, artifact_px)),
            "outside_ratio": float(outside_px / max(1, artifact_px)),
        }

    def _candidate_from_artifact(
        artifact_mask: np.ndarray,
        *,
        mode: str,
        score: float,
        metadata: dict[str, Any],
    ) -> None:
        artifact_bin = _as_binary_mask(artifact_mask)
        artifact_px = cv2.countNonZero(artifact_bin)

        if artifact_px <= 0:
            return

        externality = _artifact_externality(artifact_bin)
        outside_ratio = externality["outside_ratio"]
        protected_overlap_ratio = externality["protected_overlap_ratio"]

        # Main safety rule:
        # A stamp defense must mostly remove stuff outside the protected glyph core.
        if outside_ratio < 0.68:
            return

        if protected_overlap_ratio > 0.32:
            return

        removed_ratio = artifact_px / max(1, total_ink)

        # External stamp removal should not rewrite the whole glyph.
        if removed_ratio > 0.55:
            return

        cleaned = binary.copy()
        cleaned[artifact_bin > 0] = 0

        remaining_ink = cv2.countNonZero(cleaned)
        if remaining_ink <= 0:
            return

        if remaining_ink < int(total_ink * 0.35):
            return

        final_score = max(0.0, min(1.0, score))

        candidate_metadata = {
            "stamp_external_artifact_confidence": float(final_score),
            "geometry_score": float(max(0.70, final_score)),

            "stamp_removal_mode": mode,
            "stamp_removal_policy": "external_artifact_protected_core",

            "protected_core_bbox": {
                "x1": int(core_x1),
                "y1": int(core_y1),
                "x2": int(core_x2),
                "y2": int(core_y2),
            },

            "stamp_artifact_pixels": int(artifact_px),
            "stamp_removed_ratio": float(removed_ratio),
            "stamp_outside_ratio": float(outside_ratio),
            "stamp_protected_overlap_ratio": float(protected_overlap_ratio),

            # This defense should not get the same provisional freedom as line removal.
            "requires_downstream_repair": False,
            "provisional_parent": False,
            "allow_topology_damage_before_repair": False,
        }

        candidate_metadata.update(metadata)

        scored_candidates.append((final_score, (cleaned, candidate_metadata)))

    # ------------------------------------------------------------------
    # Pass 1: component-level external stamp lines.
    # This catches separate external stamp/border fragments.
    # ------------------------------------------------------------------
    count, labels, stats, _centroids = _connected_components(binary)

    if count > 1:
        largest_label = 1
        largest_area = int(stats[1, cv2.CC_STAT_AREA])

        for label in range(2, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area > largest_area:
                largest_area = area
                largest_label = label

        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])

            if area <= 0:
                continue

            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])

            long_span = max(bw, bh)
            short_span = max(1, min(bw, bh))
            aspect_ratio = long_span / short_span

            component_removed_ratio = area / max(1, total_ink)

            if long_span < max(8, int(max(ink_w, ink_h) * 0.25)):
                continue

            if aspect_ratio < 3.2:
                continue

            if short_span > max(7, int(min(ink_w, ink_h) * 0.28)):
                continue

            if component_removed_ratio > 0.45:
                continue

            component_mask = np.where(labels == label, 255, 0).astype(np.uint8)

            externality = _artifact_externality(component_mask)
            outside_ratio = externality["outside_ratio"]
            protected_overlap_ratio = externality["protected_overlap_ratio"]

            if outside_ratio < 0.72:
                continue

            if protected_overlap_ratio > 0.28:
                continue

            length_score = min(1.0, long_span / max(1, max(ink_w, ink_h)))
            aspect_score = min(1.0, aspect_ratio / 8.0)
            outside_score = outside_ratio
            safety_score = 1.0 - min(1.0, component_removed_ratio)

            # Slight penalty if this is the largest component; removing the largest
            # component is dangerous unless it is overwhelmingly external.
            largest_penalty = 0.16 if label == largest_label else 0.0

            score = (
                0.30 * length_score
                + 0.25 * aspect_score
                + 0.30 * outside_score
                + 0.15 * safety_score
                - largest_penalty
            )

            _candidate_from_artifact(
                component_mask,
                mode="external_component",
                score=score,
                metadata={
                    "stamp_component_label": int(label),
                    "stamp_component_bbox": {
                        "x1": int(x),
                        "y1": int(y),
                        "x2": int(x + bw - 1),
                        "y2": int(y + bh - 1),
                    },
                    "stamp_component_area": int(area),
                    "stamp_component_aspect_ratio": float(aspect_ratio),
                    "stamp_component_long_span": int(long_span),
                    "stamp_component_short_span": int(short_span),
                    "stamp_component_removed_ratio": float(component_removed_ratio),
                    "stamp_component_is_largest": bool(label == largest_label),
                },
            )

    # ------------------------------------------------------------------
    # Pass 2: external band scanner.
    # This catches connected or semi-connected external stamp lines.
    # Unlike normal line removal, this is allowed to search near-vertical lines
    # only when they are mostly outside the protected glyph core.
    # ------------------------------------------------------------------
    pad = max(4, int(min(ink_w, ink_h) * 0.15))

    crop_x1 = max(0, ink_x1 - pad)
    crop_y1 = max(0, ink_y1 - pad)
    crop_x2 = min(w - 1, ink_x2 + pad)
    crop_y2 = min(h - 1, ink_y2 + pad)

    crop = binary[crop_y1 : crop_y2 + 1, crop_x1 : crop_x2 + 1]
    ch, cw = crop.shape[:2]

    if ch < 8 or cw < 8:
        return _dedupe_and_rank(scored_candidates)

    def _rotate_image(src: np.ndarray, angle_degrees: float) -> tuple[np.ndarray, np.ndarray]:
        center = (src.shape[1] / 2.0, src.shape[0] / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

        rotated = cv2.warpAffine(
            src,
            matrix,
            (src.shape[1], src.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return rotated, matrix

    def _split_runs(values: np.ndarray, max_gap: int = 3) -> list[np.ndarray]:
        if len(values) == 0:
            return []

        splits = np.where(np.diff(values) > max_gap)[0] + 1
        return np.split(values, splits)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    crop_source = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # Stamp lines can be horizontal, diagonal, or near-vertical.
    # Near-vertical is allowed here only because protected-core overlap is checked later.
    angles = list(range(-85, 86, 5))

    for angle in angles:
        abs_angle = abs(angle)

        rotated, rot_matrix = _rotate_image(crop_source, -angle)
        inv_matrix = cv2.invertAffineTransform(rot_matrix)

        if abs_angle <= 35:
            span_floor = 0.34
            density_floor = 0.09
            outside_floor = 0.68
            protected_overlap_limit = 0.32
        elif abs_angle <= 65:
            span_floor = 0.40
            density_floor = 0.12
            outside_floor = 0.72
            protected_overlap_limit = 0.28
        else:
            # Near-vertical stamp line. Very risky, so demand strong externality.
            span_floor = 0.46
            density_floor = 0.15
            outside_floor = 0.80
            protected_overlap_limit = 0.18

        min_span = max(8, int(max(ink_w, ink_h) * span_floor))

        for band_radius in (0, 1, 2, 3, 4):
            band_height = 2 * band_radius + 1

            if band_radius <= 1:
                width_density_floor = density_floor
                width_outside_floor = outside_floor
            elif band_radius <= 3:
                width_density_floor = max(density_floor, 0.15)
                width_outside_floor = max(outside_floor, 0.74)
            else:
                width_density_floor = max(density_floor, 0.20)
                width_outside_floor = max(outside_floor, 0.78)

            for y in range(band_radius, ch - band_radius):
                band = rotated[y - band_radius : y + band_radius + 1, :]
                cols = np.where(np.any(band > 0, axis=0))[0]

                if len(cols) < min_span:
                    continue

                runs = _split_runs(cols, max_gap=3)

                for run in runs:
                    if len(run) < min_span:
                        continue

                    x_start = int(run[0])
                    x_end = int(run[-1])
                    span = x_end - x_start + 1

                    if span < min_span:
                        continue

                    span_ratio = span / max(1, max(ink_w, ink_h))

                    if span_ratio < span_floor:
                        continue

                    candidate_band = np.zeros_like(rotated)

                    y1_band = max(0, y - band_radius)
                    y2_band = min(ch - 1, y + band_radius)

                    candidate_band[
                        y1_band : y2_band + 1,
                        x_start : x_end + 1,
                    ] = 255

                    corridor_crop = cv2.warpAffine(
                        candidate_band,
                        inv_matrix,
                        (cw, ch),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )

                    corridor_crop = _as_binary_mask(corridor_crop)
                    artifact_crop = cv2.bitwise_and(crop, corridor_crop)
                    artifact_pixels = cv2.countNonZero(artifact_crop)

                    if artifact_pixels <= 0:
                        continue

                    min_artifact_pixels = max(5, int(total_ink * 0.006))
                    if artifact_pixels < min_artifact_pixels:
                        continue

                    corridor_area = max(1, cv2.countNonZero(corridor_crop))
                    corridor_density = artifact_pixels / corridor_area

                    if corridor_density < width_density_floor:
                        continue

                    full_artifact = np.zeros_like(binary)
                    full_artifact[
                        crop_y1 : crop_y2 + 1,
                        crop_x1 : crop_x2 + 1,
                    ] = artifact_crop

                    externality = _artifact_externality(full_artifact)
                    outside_ratio = externality["outside_ratio"]
                    protected_overlap_ratio = externality["protected_overlap_ratio"]

                    if outside_ratio < width_outside_floor:
                        continue

                    if protected_overlap_ratio > protected_overlap_limit:
                        continue

                    removed_ratio = artifact_pixels / max(1, total_ink)

                    if removed_ratio > 0.50:
                        continue

                    length_score = min(1.0, span_ratio)
                    density_score = min(1.0, corridor_density)
                    outside_score = min(1.0, outside_ratio)
                    safety_score = 1.0 - min(1.0, removed_ratio)
                    vertical_penalty = 0.08 if abs_angle > 65 else 0.0
                    width_penalty = min(0.14, band_radius * 0.025)

                    score = (
                        0.25 * length_score
                        + 0.25 * density_score
                        + 0.35 * outside_score
                        + 0.15 * safety_score
                        - vertical_penalty
                        - width_penalty
                    )

                    _candidate_from_artifact(
                        full_artifact,
                        mode="external_band",
                        score=score,
                        metadata={
                            "stamp_band_angle_degrees": float(angle),
                            "stamp_band_width_px": int(band_height),
                            "stamp_band_radius": int(band_radius),
                            "stamp_band_span_ratio": float(span_ratio),
                            "stamp_band_corridor_density": float(corridor_density),
                            "stamp_band_outside_floor": float(width_outside_floor),
                            "stamp_band_protected_overlap_limit": float(
                                protected_overlap_limit
                            ),
                        },
                    )

    return _dedupe_and_rank(scored_candidates)


def _defense_horizontal_gap_closing(mask: np.ndarray) -> list[np.ndarray]:
    binary = _as_binary_mask(mask)
    candidates = []

    geometry_by_width = {
        2: 0.66,
        3: 0.63,
        4: 0.60,
    }

    for width in (2, 3, 4):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        candidates.append(
            (
                closed,
                {
                    "geometry_score": geometry_by_width[width],
                    "horizontal_gap_confidence": geometry_by_width[width],
                    "horizontal_gap_kernel_width": int(width),
                    "horizontal_gap_policy": "slightly_boosted_confidence",
                },
            )
        )

    return candidates


def _defense_vertical_gap_closing(mask: np.ndarray) -> list[np.ndarray]:
    binary = _as_binary_mask(mask)
    candidates = []

    geometry_by_height = {
        2: 0.66,
        3: 0.63,
        4: 0.60,
    }

    for height in (2, 3, 4):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        candidates.append(
            (
                closed,
                {
                    "geometry_score": geometry_by_height[height],
                    "vertical_gap_confidence": geometry_by_height[height],
                    "vertical_gap_kernel_height": int(height),
                    "vertical_gap_policy": "slightly_boosted_confidence",
                },
            )
        )

    return candidates


def _defense_threshold_normalization(mask: np.ndarray) -> list[np.ndarray]:
    binary = _as_binary_mask(mask)
    candidates = []

    # Conservative close-open cleanup.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel, iterations=1)

    candidates.append(cleaned)

    # Slightly more additive normalization for threshold-failure gaps.
    plus = cv2.dilate(binary, close_kernel, iterations=1)
    plus = cv2.erode(plus, close_kernel, iterations=1)
    candidates.append(plus)

    return candidates


def _defense_component_denoising(mask: np.ndarray) -> list[np.ndarray]:
    binary = _as_binary_mask(mask)
    count, labels, stats, _ = _connected_components(binary)

    if count <= 2:
        return []

    total = max(1, int(cv2.countNonZero(binary)))
    candidates = []

    for min_area in (2, 3, 5, 8):
        cleaned = np.zeros_like(binary)

        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                cleaned[labels == label] = 255

        removed = total - int(cv2.countNonZero(cleaned))
        if removed > 0:
            candidates.append(cleaned)

    # Candidate that keeps only largest component.
    largest = _largest_component_mask(binary)
    if cv2.countNonZero(largest) != cv2.countNonZero(binary):
        candidates.append(largest)

    return candidates


def _defense_median_denoising(mask: np.ndarray) -> list[np.ndarray]:
    binary = _as_binary_mask(mask)
    candidates = []

    for kernel_size in (3, 5):
        filtered = cv2.medianBlur(binary, kernel_size)
        candidates.append(_as_binary_mask(filtered))

    return candidates


def _defense_conservative_stroke_recovery(mask: np.ndarray) -> list[np.ndarray]:
    binary = _as_binary_mask(mask)
    candidates = []

    # This is intentionally conservative. It recovers very small erosion damage.
    for kernel_shape in (
        cv2.MORPH_CROSS,
        cv2.MORPH_ELLIPSE,
    ):
        kernel = cv2.getStructuringElement(kernel_shape, (3, 3))
        recovered = cv2.dilate(binary, kernel, iterations=1)

        # Do a light erosion back, so it does not become fat blob repair.
        recovered = cv2.erode(recovered, kernel, iterations=1)

        # Union with original so this is mostly additive/shape recovery.
        recovered = cv2.bitwise_or(binary, recovered)
        candidates.append(recovered)

    return candidates


def _defense_contamination_opening(mask: np.ndarray) -> list[np.ndarray]:
    binary = _as_binary_mask(mask)
    candidates = []

    # Remove thin external contamination. Risky, verifier must judge.
    for kernel_shape in (
        cv2.MORPH_CROSS,
        cv2.MORPH_RECT,
    ):
        kernel = cv2.getStructuringElement(kernel_shape, (2, 2))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        candidates.append(opened)

    return candidates


def _defense_linear_artifact_removal(mask: np.ndarray) -> list[Any]:
    binary = _as_binary_mask(mask)
    h, w = binary.shape[:2]

    total_ink = cv2.countNonZero(binary)
    if total_ink <= 0:
        return []

    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return []

    ink_x1 = int(xs.min())
    ink_x2 = int(xs.max())
    ink_y1 = int(ys.min())
    ink_y2 = int(ys.max())

    ink_w = max(1, ink_x2 - ink_x1 + 1)
    ink_h = max(1, ink_y2 - ink_y1 + 1)

    if ink_w < 8 or ink_h < 8:
        return []

    pad = max(4, int(min(ink_w, ink_h) * 0.15))

    crop_x1 = max(0, ink_x1 - pad)
    crop_y1 = max(0, ink_y1 - pad)
    crop_x2 = min(w - 1, ink_x2 + pad)
    crop_y2 = min(h - 1, ink_y2 + pad)

    crop = binary[crop_y1 : crop_y2 + 1, crop_x1 : crop_x2 + 1]
    ch, cw = crop.shape[:2]

    if ch < 8 or cw < 8:
        return []

    scored_candidates: list[tuple[float, object]] = []

    def _dedupe_and_rank(
        items: list[tuple[float, object]],
        max_count: int = 12,
    ) -> list[object]:
        items = sorted(items, key=lambda pair: pair[0], reverse=True)

        seen = set()
        output: list[object] = []

        for _, payload in items:
            candidate_mask, _metadata = _unpack_candidate_payload(payload)
            candidate_bin = _as_binary_mask(candidate_mask)

            key = candidate_bin.tobytes()
            if key in seen:
                continue

            seen.add(key)
            output.append(payload)

            if len(output) >= max_count:
                break

        return output

    def _rotate_image(src: np.ndarray, angle_degrees: float) -> tuple[np.ndarray, np.ndarray]:
        center = (src.shape[1] / 2.0, src.shape[0] / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

        rotated = cv2.warpAffine(
            src,
            matrix,
            (src.shape[1], src.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return rotated, matrix

    def _invert_affine(matrix: np.ndarray) -> np.ndarray:
        return cv2.invertAffineTransform(matrix)

    def _split_runs(values: np.ndarray, max_gap: int = 3) -> list[np.ndarray]:
        if len(values) == 0:
            return []

        splits = np.where(np.diff(values) > max_gap)[0] + 1
        return np.split(values, splits)

    # Allow horizontal, slanted, diagonal, and near-vertical artifacts.
    # Near-vertical candidates are high risk, so the thresholds below demand
    # stronger evidence before a candidate can survive.
    angles = list(range(-85, 86, 5))

    # Normal close catches thin/broken overlap lines.
    normal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    crop_source_normal = cv2.morphologyEx(
        crop,
        cv2.MORPH_CLOSE,
        normal_kernel,
        iterations=1,
    )

    # Wide close helps thicker careless strokes become detectable as one band.
    wide_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    crop_source_wide = cv2.morphologyEx(
        crop,
        cv2.MORPH_CLOSE,
        wide_kernel,
        iterations=1,
    )

    crop_sources = [
        ("normal", crop_source_normal),
        ("wide", crop_source_wide),
    ]

    for source_name, crop_source in crop_sources:
        for angle in angles:
            abs_angle = abs(angle)

            # Angle-specific risk controls.
            # Low-angle lines are safer.
            # Steep 50-65 degree lines are allowed but stricter.
            if abs_angle <= 35:
                required_span_ratio = 0.38
                required_density = 0.10
                max_removed_ratio = 0.58
            elif abs_angle <= 50:
                required_span_ratio = 0.44
                required_density = 0.13
                max_removed_ratio = 0.50
            elif abs_angle <= 65:
                required_span_ratio = 0.52
                required_density = 0.16
                max_removed_ratio = 0.42
            else:
                required_span_ratio = 0.58
                required_density = 0.20
                max_removed_ratio = 0.34

            # Rotate opposite direction so a slanted artifact becomes horizontal.
            rotated, rot_matrix = _rotate_image(crop_source, -angle)
            inv_matrix = _invert_affine(rot_matrix)

            # Multi-width search:
            # 0 -> 1 px
            # 1 -> 3 px
            # 2 -> 5 px
            # 3 -> 7 px
            # 4 -> 9 px
            # 5 -> 11 px
            for band_radius in (0, 1, 2, 3, 4, 5):
                band_height = 2 * band_radius + 1

                # Wider deletion is riskier, so demand stronger evidence.
                if band_radius <= 1:
                    width_density_floor = required_density
                    width_removed_limit = max_removed_ratio
                    width_span_floor = required_span_ratio
                elif band_radius <= 3:
                    width_density_floor = max(required_density, 0.16)
                    width_removed_limit = min(max_removed_ratio, 0.48)
                    width_span_floor = max(required_span_ratio, 0.44)
                else:
                    width_density_floor = max(required_density, 0.22)
                    width_removed_limit = min(max_removed_ratio, 0.40)
                    width_span_floor = max(required_span_ratio, 0.50)

                # In rotated space the detected artifact is horizontal, but
                # the original reference span depends on line orientation.
                if abs_angle <= 30:
                    line_reference_span = ink_w
                elif abs_angle >= 60:
                    line_reference_span = ink_h
                else:
                    line_reference_span = max(ink_w, ink_h)

                min_span = max(8, int(line_reference_span * width_span_floor))

                for y in range(band_radius, ch - band_radius):
                    band = rotated[y - band_radius : y + band_radius + 1, :]

                    cols = np.where(np.any(band > 0, axis=0))[0]
                    if len(cols) < min_span:
                        continue

                    runs = _split_runs(cols, max_gap=3)

                    for run in runs:
                        if len(run) < min_span:
                            continue

                        x_start = int(run[0])
                        x_end = int(run[-1])
                        span = x_end - x_start + 1

                        if span < min_span:
                            continue

                        span_ratio = span / max(1, line_reference_span)

                        if span_ratio < width_span_floor:
                            continue

                        candidate_band = np.zeros_like(rotated)

                        y1_band = max(0, y - band_radius)
                        y2_band = min(ch - 1, y + band_radius)

                        candidate_band[
                            y1_band : y2_band + 1,
                            x_start : x_end + 1,
                        ] = 255

                        # Map candidate band back to original crop coordinates.
                        corridor_crop = cv2.warpAffine(
                            candidate_band,
                            inv_matrix,
                            (cw, ch),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )

                        corridor_crop = _as_binary_mask(corridor_crop)

                        # Remove one extra pixel around the detected line.
                        # We still intersect with the original ink, so this only
                        # widens deleted foreground and never invents removal.
                        removal_kernel = cv2.getStructuringElement(
                            cv2.MORPH_RECT,
                            (3, 3),
                        )
                        removal_corridor = cv2.dilate(
                            corridor_crop,
                            removal_kernel,
                            iterations=1,
                        )
                        artifact_crop = cv2.bitwise_and(crop, removal_corridor)
                        artifact_pixels = cv2.countNonZero(artifact_crop)

                        if artifact_pixels <= 0:
                            continue

                        min_artifact_pixels = max(5, int(total_ink * 0.006))
                        if artifact_pixels < min_artifact_pixels:
                            continue

                        component_count, _component_labels, component_stats, _ = (
                            _connected_components(artifact_crop)
                        )
                        component_areas = [
                            int(component_stats[label, cv2.CC_STAT_AREA])
                            for label in range(1, component_count)
                        ]

                        if not component_areas:
                            continue

                        largest_component_area = max(component_areas)
                        largest_component_ratio = (
                            largest_component_area / max(1, artifact_pixels)
                        )
                        significant_component_count = sum(
                            1
                            for area in component_areas
                            if area >= max(3, int(artifact_pixels * 0.12))
                        )

                        # A true line artifact should be dominated by one
                        # connected stroke. Aligned but disconnected glyph
                        # fragments are likely real ink, not a removable line.
                        if largest_component_ratio < 0.82:
                            continue

                        if significant_component_count > 1:
                            continue

                        corridor_area = max(1, cv2.countNonZero(corridor_crop))
                        corridor_density = artifact_pixels / corridor_area
                        removed_ratio = artifact_pixels / max(1, total_ink)

                        # Too sparse: probably following curved/wavy glyph body.
                        if corridor_density < width_density_floor:
                            continue

                        # Too destructive: probably eating the actual letter.
                        if removed_ratio > width_removed_limit:
                            continue

                        full_artifact = np.zeros_like(binary)
                        full_artifact[
                            crop_y1 : crop_y2 + 1,
                            crop_x1 : crop_x2 + 1,
                        ] = artifact_crop

                        cleaned = binary.copy()
                        cleaned[full_artifact > 0] = 0

                        remaining_ink = cv2.countNonZero(cleaned)
                        if remaining_ink <= 0:
                            continue

                        if remaining_ink < int(total_ink * 0.30):
                            continue

                        length_score = min(1.0, span_ratio)
                        density_score = min(1.0, corridor_density)
                        safety_score = 1.0 - min(1.0, removed_ratio)
                        angle_score = 1.0 - min(1.0, abs_angle / 75.0)

                        # Wider candidates are allowed, but slightly penalized
                        # so thin/medium candidates win unless wide is clearly better.
                        width_penalty = min(0.18, band_radius * 0.035)

                        score = (
                            0.35 * length_score
                            + 0.30 * density_score
                            + 0.25 * safety_score
                            + 0.10 * angle_score
                            - width_penalty
                        )

                        line_confidence = max(0.0, min(1.0, score))

                        width_policy = (
                            "thin"
                            if band_radius <= 1
                            else "medium"
                            if band_radius <= 3
                            else "wide"
                        )

                        line_endpoints_rotated = np.array(
                            [
                                [[float(x_start), float(y)]],
                                [[float(x_end), float(y)]],
                            ],
                            dtype=np.float32,
                        )
                        line_endpoints_crop = cv2.transform(
                            line_endpoints_rotated,
                            inv_matrix,
                        ).reshape(-1, 2)

                        line_endpoints = []
                        for endpoint_x, endpoint_y in line_endpoints_crop:
                            line_endpoints.append(
                                {
                                    "x": int(
                                        np.clip(
                                            round(float(endpoint_x)) + crop_x1,
                                            0,
                                            w - 1,
                                        )
                                    ),
                                    "y": int(
                                        np.clip(
                                            round(float(endpoint_y)) + crop_y1,
                                            0,
                                            h - 1,
                                        )
                                    ),
                                }
                            )

                        abs_angle_from_horizontal = min(90.0, float(abs_angle))
                        preferred_bridge_angle = 90.0 - abs_angle_from_horizontal

                        if abs_angle_from_horizontal <= 30:
                            orientation_class = "horizontal_like"
                            bridge_preference = "vertical_bridge"
                        elif abs_angle_from_horizontal >= 60:
                            orientation_class = "vertical_like"
                            bridge_preference = "horizontal_bridge"
                        else:
                            orientation_class = "diagonal_like"
                            bridge_preference = "perpendicular_bridge"

                        metadata = {
                            "line_artifact_confidence": float(line_confidence),
                            "geometry_score": float(max(0.72, line_confidence)),

                            # Line removal is a parent repair step.
                            # It may damage topology before endpoint_bridge/gap repair.
                            "requires_downstream_repair": True,
                            "provisional_parent": True,
                            "allow_topology_damage_before_repair": True,

                            "recommended_next_defenses": [
                                "endpoint_bridge",
                                "horizontal_gap_closing",
                                "vertical_gap_closing",
                            ],

                            # Debug / audit fields.
                            "line_removal_policy": "aggressive_same_color_overlap",
                            "line_source": source_name,
                            "line_angle_degrees": float(angle),
                            "line_abs_angle_from_horizontal": float(
                                abs_angle_from_horizontal
                            ),
                            "line_orientation_class": orientation_class,
                            "preferred_bridge_angle_from_horizontal": float(
                                preferred_bridge_angle
                            ),
                            "preferred_bridge_axis": bridge_preference,
                            "line_segment_endpoints": line_endpoints,
                            "line_span_ratio": float(span_ratio),
                            "line_corridor_density": float(corridor_density),
                            "line_removed_ratio": float(removed_ratio),
                            "line_largest_component_ratio": float(
                                largest_component_ratio
                            ),
                            "line_significant_component_count": int(
                                significant_component_count
                            ),

                            "line_width_px": int(band_height),
                            "line_band_radius": int(band_radius),
                            "line_width_policy": width_policy,
                        }

                        scored_candidates.append((score, (cleaned, metadata)))

    return _dedupe_and_rank(scored_candidates)


def _defense_border_continuation(mask: np.ndarray) -> list[np.ndarray]:
    binary = _as_binary_mask(mask)
    h, w = binary.shape[:2]
    candidates = []

    bbox = _glyph_bbox(binary)
    if bbox is None:
        return candidates

    x1, y1, x2, y2 = bbox

    # If ink touches or nearly touches a border, extend it slightly inward/outward.
    near = 2
    extension = 3

    # Left border.
    if x1 <= near:
        candidate = binary.copy()
        ys = np.where(binary[:, max(0, x1): min(w, x1 + 2)] > 0)[0]
        for y in ys:
            y0 = max(0, y - 1)
            y1_local = min(h, y + 2)
            candidate[y0:y1_local, 0:min(w, extension)] = 255
        candidates.append(candidate)

    # Right border.
    if x2 >= w - 1 - near:
        candidate = binary.copy()
        ys = np.where(binary[:, max(0, x2 - 1): min(w, x2 + 1)] > 0)[0]
        for y in ys:
            y0 = max(0, y - 1)
            y1_local = min(h, y + 2)
            candidate[y0:y1_local, max(0, w - extension):w] = 255
        candidates.append(candidate)

    # Top border.
    if y1 <= near:
        candidate = binary.copy()
        xs = np.where(binary[max(0, y1): min(h, y1 + 2), :] > 0)[1]
        for x in xs:
            x0 = max(0, x - 1)
            x1_local = min(w, x + 2)
            candidate[0:min(h, extension), x0:x1_local] = 255
        candidates.append(candidate)

    # Bottom border.
    if y2 >= h - 1 - near:
        candidate = binary.copy()
        xs = np.where(binary[max(0, y2 - 1): min(h, y2 + 1), :] > 0)[1]
        for x in xs:
            x0 = max(0, x - 1)
            x1_local = min(w, x + 2)
            candidate[max(0, h - extension):h, x0:x1_local] = 255
        candidates.append(candidate)

    return candidates


DEFENSE_FUNCTIONS = {
    DEFENSE_HORIZONTAL_GAP_CLOSING: _defense_horizontal_gap_closing,
    DEFENSE_VERTICAL_GAP_CLOSING: _defense_vertical_gap_closing,
    DEFENSE_THRESHOLD_NORMALIZATION: _defense_threshold_normalization,
    DEFENSE_COMPONENT_DENOISING: _defense_component_denoising,
    DEFENSE_MEDIAN_DENOISING: _defense_median_denoising,
    DEFENSE_CONSERVATIVE_STROKE_RECOVERY: _defense_conservative_stroke_recovery,
    DEFENSE_CONTAMINATION_OPENING: _defense_contamination_opening,
    DEFENSE_LINEAR_ARTIFACT_REMOVAL: _defense_linear_artifact_removal,
    DEFENSE_STAMP_EXTERNAL_ARTIFACT_REMOVAL: _defense_stamp_external_artifact_removal,
    DEFENSE_BORDER_CONTINUATION: _defense_border_continuation,
}


def generate_defense_hypotheses(
    mask: np.ndarray,
    allowed_defenses: list[str] | tuple[str, ...] | None,
    original_result=None,
    stable_unit_id: str | None = None,
    start_index: int = 1,
) -> list[DefenseHypothesis]:
    """Generate repair candidates for the allowed defense names.

    Unknown or not-yet-implemented defenses are skipped.
    Endpoint bridge is currently still handled in trace_reconstruction.py.
    """
    if not allowed_defenses:
        return []

    original_bin = _as_binary_mask(mask)
    hypotheses: list[DefenseHypothesis] = []
    next_index = int(start_index)

    seen_candidates: set[bytes] = set()

    for defense_name in allowed_defenses:
        defense_name = str(defense_name)

        if not implemented_in_trace_defenses(defense_name):
            continue

        defense_fn = DEFENSE_FUNCTIONS.get(defense_name)
        if defense_fn is None:
            continue

        raw_candidates = defense_fn(original_bin)
        for payload in raw_candidates:
            candidate_mask, candidate_metadata = _unpack_candidate_payload(payload)

            candidate_bin = _as_binary_mask(candidate_mask)
            key = candidate_bin.tobytes()

            if key in seen_candidates:
                continue

            seen_candidates.add(key)

            metadata = {
                "stable_unit_id": stable_unit_id,
                "source": "trace_defenses",
            }
            metadata.update(candidate_metadata)

            hypothesis = _make_hypothesis(
                index=next_index,
                defense_name=defense_name,
                original_mask=original_bin,
                candidate_mask=candidate_bin,
                metadata=metadata,
            )

            if hypothesis is None:
                continue

            hypotheses.append(hypothesis)
            next_index += 1

    return hypotheses


__all__ = [
    "DefenseHypothesis",
    "generate_defense_hypotheses",
]
