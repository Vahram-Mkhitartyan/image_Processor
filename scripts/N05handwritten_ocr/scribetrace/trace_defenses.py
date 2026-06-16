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


def _defense_horizontal_gap_closing(mask: np.ndarray) -> list[np.ndarray]:
    binary = _as_binary_mask(mask)
    candidates = []

    for width in (2, 3, 4):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        candidates.append(closed)

    return candidates


def _defense_vertical_gap_closing(mask: np.ndarray) -> list[np.ndarray]:
    binary = _as_binary_mask(mask)
    candidates = []

    for height in (2, 3, 4):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        candidates.append(closed)

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


def _defense_linear_artifact_removal(mask: np.ndarray) -> list[np.ndarray]:
    binary = _as_binary_mask(mask)
    h, w = binary.shape[:2]
    candidates = []

    # Conservative line removal: remove long horizontal/vertical runs only.
    # This is intended for stamp/text-bar artifacts, not normal handwriting.
    for orientation in ("horizontal", "vertical"):
        artifact = np.zeros_like(binary)

        if orientation == "horizontal":
            min_run = max(12, int(w * 0.35))
            for y in range(h):
                xs = np.where(binary[y, :] > 0)[0]
                if len(xs) < min_run:
                    continue

                # Find contiguous runs.
                splits = np.where(np.diff(xs) > 1)[0] + 1
                runs = np.split(xs, splits)

                for run in runs:
                    if len(run) >= min_run:
                        artifact[y, run] = 255

        else:
            min_run = max(12, int(h * 0.35))
            for x in range(w):
                ys = np.where(binary[:, x] > 0)[0]
                if len(ys) < min_run:
                    continue

                splits = np.where(np.diff(ys) > 1)[0] + 1
                runs = np.split(ys, splits)

                for run in runs:
                    if len(run) >= min_run:
                        artifact[run, x] = 255

        if cv2.countNonZero(artifact) == 0:
            continue

        # Dilate artifact slightly so the whole straight bar is removed.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        artifact = cv2.dilate(artifact, kernel, iterations=1)
        cleaned = binary.copy()
        cleaned[artifact > 0] = 0

        if cv2.countNonZero(cleaned) > 0:
            candidates.append(cleaned)

    return candidates


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

        for candidate_mask in raw_candidates:
            candidate_bin = _as_binary_mask(candidate_mask)
            key = candidate_bin.tobytes()

            if key in seen_candidates:
                continue

            seen_candidates.add(key)

            hypothesis = _make_hypothesis(
                index=next_index,
                defense_name=defense_name,
                original_mask=original_bin,
                candidate_mask=candidate_bin,
                metadata={
                    "stable_unit_id": stable_unit_id,
                    "source": "trace_defenses",
                },
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