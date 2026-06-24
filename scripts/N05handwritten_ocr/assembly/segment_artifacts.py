"""Materialize selected assembly segments and attach condition evidence."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from ..condition.condition_models import ConditionVerdict, DamageCandidate
    from ..condition.condition_router import route_condition
    from ..condition.damage_labels import (
        DAMAGE_CLEAN,
        DAMAGE_EDGE_CROP_LOSS,
        DAMAGE_LIGHT_CUT,
        DAMAGE_SCANNER_NOISE,
        DAMAGE_UNKNOWN,
    )
except ImportError:
    from condition.condition_models import ConditionVerdict, DamageCandidate  # type: ignore
    from condition.condition_router import route_condition  # type: ignore
    from condition.damage_labels import (  # type: ignore
        DAMAGE_CLEAN,
        DAMAGE_EDGE_CROP_LOSS,
        DAMAGE_LIGHT_CUT,
        DAMAGE_SCANNER_NOISE,
        DAMAGE_UNKNOWN,
    )


def _safe_slug(value: Any) -> str:
    """Return a path-safe identifier fragment."""

    text = str(value if value is not None else "unknown")
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in text)


def _read_image(path: str | Path | None) -> np.ndarray | None:
    """Read an image as grayscale, returning ``None`` on missing paths."""

    if not path:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return image


def _threshold_ink_mask(image: np.ndarray) -> np.ndarray:
    """Return a binary mask where ink is white/255."""

    border = np.concatenate([image[0, :], image[-1, :], image[:, 0], image[:, -1]])
    if float(np.median(border)) > 128.0:
        source = 255 - image
    else:
        source = image
    _, mask = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _clip_bbox_to_image(bbox: dict | None, image: np.ndarray) -> dict | None:
    """Clip a bbox into image bounds, or return ``None`` if it is invalid."""

    if not isinstance(bbox, dict):
        return None
    height, width = image.shape[:2]
    try:
        x1 = max(0, min(width, int(round(float(bbox.get("x1", 0))))))
        y1 = max(0, min(height, int(round(float(bbox.get("y1", 0))))))
        x2 = max(0, min(width, int(round(float(bbox.get("x2", width))))))
        y2 = max(0, min(height, int(round(float(bbox.get("y2", height))))))
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _write_png(path: Path, image: np.ndarray) -> str:
    """Write a PNG image and return its string path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return str(path)


def _copy_or_crop_segment(
    unit: dict,
    segment: dict,
    output_dir: Path,
    path_id: str,
    position: int,
) -> dict:
    """Create visual/mask crops for one selected segment."""

    text_unit_id = _safe_slug(unit.get("text_unit_id") or unit.get("group_id"))
    segment_id = _safe_slug(segment.get("segment_id") or f"s{position}")
    prefix = f"{text_unit_id}_{_safe_slug(path_id)}_{position:02d}_{segment_id}"
    visual_output = output_dir / "visual" / f"{prefix}.png"
    mask_output = output_dir / "mask" / f"{prefix}_mask.png"

    existing_visual = segment.get("visual_crop_path")
    existing_mask = segment.get("mask_crop_path")
    if existing_visual and Path(existing_visual).is_file():
        visual_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(existing_visual, visual_output)
        if existing_mask and Path(existing_mask).is_file():
            mask_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing_mask, mask_output)
        else:
            visual_image = _read_image(visual_output)
            if visual_image is not None:
                _write_png(mask_output, _threshold_ink_mask(visual_image))
        return {
            "status": "copied_existing_segment_crop",
            "visual_crop_path": str(visual_output),
            "mask_crop_path": str(mask_output) if mask_output.is_file() else None,
            "source": "segment_existing_crop_path",
        }

    whole_visual_path = (
        unit.get("n05_selected_crop_path")
        or unit.get("scribetrace_visual_crop_path")
        or unit.get("classification_crop_path")
    )
    whole_image = _read_image(whole_visual_path)
    if whole_image is None:
        return {
            "status": "failed",
            "error": "Unable to read whole-unit visual crop for segment materialization.",
            "visual_crop_path": None,
            "mask_crop_path": None,
        }

    bbox = segment.get("bbox")
    if segment.get("role") == "whole_unit" or not isinstance(bbox, dict):
        crop = whole_image
    else:
        clipped = _clip_bbox_to_image(bbox, whole_image)
        if clipped is None:
            crop = whole_image
        else:
            crop = whole_image[clipped["y1"] : clipped["y2"], clipped["x1"] : clipped["x2"]]
            segment["materialized_bbox"] = clipped

    visual_path = _write_png(visual_output, crop)
    mask_path = _write_png(mask_output, _threshold_ink_mask(crop))
    return {
        "status": "materialized_from_whole_unit",
        "visual_crop_path": visual_path,
        "mask_crop_path": mask_path,
        "source": "whole_unit_crop_bbox",
    }


def _image_condition_verdict(mask_path: str | None) -> ConditionVerdict:
    """Return a conservative image-only condition verdict for a segment."""

    image = _read_image(mask_path)
    if image is None:
        return ConditionVerdict(
            condition="uncertain",
            repair_needed=True,
            primary_damage=DAMAGE_UNKNOWN,
            confidence=0.15,
            top_damage_candidates=[DamageCandidate(DAMAGE_UNKNOWN, 0.15)],
            severity=0.5,
            source="segment_image_condition_fallback",
            notes=["mask_unreadable"],
            features={},
        )

    mask = _threshold_ink_mask(image)
    height, width = mask.shape[:2]
    ink_pixels = int(cv2.countNonZero(mask))
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    component_areas = [
        int(stats[index, cv2.CC_STAT_AREA])
        for index in range(1, components)
        if int(stats[index, cv2.CC_STAT_AREA]) > 0
    ]
    component_count = len(component_areas)
    density = ink_pixels / max(1, height * width)
    border_contacts = {
        "left": bool(np.any(mask[:, 0] > 0)),
        "right": bool(np.any(mask[:, -1] > 0)),
        "top": bool(np.any(mask[0, :] > 0)),
        "bottom": bool(np.any(mask[-1, :] > 0)),
    }
    small_components = sum(area <= 3 for area in component_areas)
    features = {
        "image_width": width,
        "image_height": height,
        "ink_pixel_count": ink_pixels,
        "ink_density": round(density, 6),
        "component_count": component_count,
        "small_component_count": small_components,
        **{f"border_contact_{side}": value for side, value in border_contacts.items()},
    }

    notes: list[str] = []
    if ink_pixels <= 2:
        notes.append("almost_empty_segment")
    if small_components >= 2:
        notes.append("scanner_noise_like_small_components")
    if border_contacts["left"] or border_contacts["right"] or border_contacts["top"] or border_contacts["bottom"]:
        notes.append("ink_touches_segment_border")
    if component_count > 2:
        notes.append("fragmented_components")

    if ink_pixels <= 2:
        primary = DAMAGE_UNKNOWN
        condition = "uncertain"
        repair_needed = True
        confidence = 0.30
    elif small_components >= 2 and density < 0.18:
        primary = DAMAGE_SCANNER_NOISE
        condition = "uncertain"
        repair_needed = False
        confidence = 0.35
    elif "ink_touches_segment_border" in notes:
        primary = DAMAGE_EDGE_CROP_LOSS
        condition = "uncertain"
        repair_needed = True
        confidence = 0.40
    elif component_count > 1:
        primary = DAMAGE_LIGHT_CUT
        condition = "uncertain"
        repair_needed = True
        confidence = 0.38
    else:
        primary = DAMAGE_CLEAN
        condition = "clean"
        repair_needed = False
        confidence = 0.72
        notes.append("image_condition_no_strong_damage_signal")

    return ConditionVerdict(
        condition=condition,
        repair_needed=repair_needed,
        primary_damage=primary,
        confidence=confidence,
        top_damage_candidates=[DamageCandidate(primary, confidence)],
        severity=0.0 if primary == DAMAGE_CLEAN else 0.45,
        source="segment_image_condition_fallback",
        notes=notes,
        features=features,
    )


def materialize_selected_segments(
    assembly_map: dict,
    units: list[dict],
    output_dir: str | Path,
    settings: dict | None = None,
) -> dict:
    """Materialize selected path segments and attach condition evidence."""

    settings = settings or {}
    enabled = bool(settings.get("enabled", True))
    if not enabled:
        assembly_map["segment_artifacts"] = {"enabled": False, "jobs": []}
        return assembly_map

    segment_dir = Path(output_dir) / "segments"
    unit_by_id = {str(unit.get("text_unit_id")): unit for unit in units}
    jobs = []
    for entry in assembly_map.get("segmentation_matrix") or []:
        unit = unit_by_id.get(str(entry.get("text_unit_id")))
        if unit is None:
            continue
        paths = entry.get("paths") or []
        if not paths:
            continue
        selected_path = paths[0]
        entry["selected_path_id"] = selected_path.get("path_id")
        selected_path["status"] = "selected_for_v0_2_segment_artifacts"
        for position, segment in enumerate(selected_path.get("segments") or []):
            artifact = _copy_or_crop_segment(
                unit=unit,
                segment=segment,
                output_dir=segment_dir,
                path_id=str(selected_path.get("path_id") or "path"),
                position=position,
            )
            segment["artifact"] = artifact
            if artifact.get("visual_crop_path"):
                segment["visual_crop_path"] = artifact.get("visual_crop_path")
            if artifact.get("mask_crop_path"):
                segment["mask_crop_path"] = artifact.get("mask_crop_path")
            verdict = _image_condition_verdict(artifact.get("mask_crop_path"))
            routing = route_condition(verdict)
            condition_record = {
                "verdict": verdict.to_dict(),
                "routing": routing.to_dict(),
            }
            segment["condition"] = condition_record
            jobs.append(
                {
                    "job_id": (
                        f"{_safe_slug(entry.get('text_unit_id'))}_"
                        f"{_safe_slug(selected_path.get('path_id'))}_"
                        f"{_safe_slug(segment.get('segment_id'))}"
                    ),
                    "text_unit_id": entry.get("text_unit_id"),
                    "path_id": selected_path.get("path_id"),
                    "segment_id": segment.get("segment_id"),
                    "position": position,
                    "visual_crop_path": segment.get("visual_crop_path"),
                    "mask_crop_path": segment.get("mask_crop_path"),
                    "condition": condition_record,
                    "status": "segment_ready",
                }
            )

    assembly_map["segment_artifacts"] = {
        "enabled": True,
        "segment_dir": str(segment_dir),
        "job_count": len(jobs),
        "jobs": jobs,
    }
    return assembly_map
