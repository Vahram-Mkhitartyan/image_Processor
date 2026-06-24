import json
import os
import shutil
from dataclasses import asdict, dataclass

import cv2
import numpy as np

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SETTINGS_PATH = os.path.join(MODULE_DIR, "settings.json")

@dataclass
class RefinerSettings:
    input_mode: str = "scribemap_2_layers"
    layers_to_refine: tuple = ("blue", "red", "green", "unknown_color", "black")
    crop_padding_px: int = 2
    debug_preview_enabled: bool = True
    correction_routing_enabled: bool = True
    correction_min_crossing_ink_pixels: int = 6
    correction_min_horizontal_span_ratio: float = 0.35
    correction_min_vertical_span_ratio: float = 0.45
    correction_replacement_max_distance_px: int = 55
    correction_replacement_min_width_px: int = 8
    correction_replacement_min_height_px: int = 8
    correction_replacement_min_area: int = 80
    correction_replacement_max_aspect_ratio: float = 12.0
    correction_max_replacements_per_blue: int = 2
    stacked_text_split_enabled: bool = True
    stacked_text_split_layers: tuple = ("blue", "green", "black", "unknown_color")
    stacked_text_min_height_px: int = 68
    stacked_text_min_width_px: int = 70
    stacked_text_min_aspect_ratio: float = 1.35
    stacked_text_row_ink_ratio: float = 0.012
    stacked_text_min_gap_px: int = 4
    stacked_text_merge_gap_px: int = 3
    stacked_text_projection_valley_enabled: bool = True
    stacked_text_projection_valley_max_ratio: float = 0.45
    stacked_text_projection_min_side_ink_ratio: float = 0.22
    stacked_text_min_segment_height_px: int = 12
    stacked_text_segment_padding_px: int = 2
    stacked_text_max_segments: int = 4

    def to_dict(self):
        return asdict(self)

#IO helpers

def load_image(image_path, color=True):
    """Load image from disk."""
    if color:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image


def save_image(image, image_path):
    """Save image to disk."""
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    ok = cv2.imwrite(image_path, image)
    if not ok:
        raise RuntimeError(f"Could not save image: {image_path}")

    return image_path


def clamp_bbox_to_image(bbox, image):
    """Clamp bbox to image bounds."""
    height, width = image.shape[:2]

    x1 = max(0, min(int(bbox["x1"]), width - 1))
    y1 = max(0, min(int(bbox["y1"]), height - 1))
    x2 = max(0, min(int(bbox["x2"]), width))
    y2 = max(0, min(int(bbox["y2"]), height))

    if x2 <= x1:
        x2 = min(width, x1 + 1)

    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def crop_image(image, bbox):
    """Crop image by bbox."""
    safe_bbox = clamp_bbox_to_image(bbox, image)

    return image[
        safe_bbox["y1"]:safe_bbox["y2"],
        safe_bbox["x1"]:safe_bbox["x2"]
    ], safe_bbox

def build_masked_crop(source_image, masks_by_name, bbox, mask_names_to_keep):
    """Build a crop where selected mask pixels stay visible and all else becomes white.

    Args:
        source_image: Color image used as the visual source.
        masks_by_name: Dict of mask_name -> grayscale mask image.
        bbox: Bbox dict.
        mask_names_to_keep: List/tuple of mask names to keep.

    Returns:
        Tuple:
            output_crop
            safe_bbox
    """
    source_crop, safe_bbox = crop_image(source_image, bbox)

    combined_mask = np.zeros(source_crop.shape[:2], dtype=np.uint8)

    for mask_name in mask_names_to_keep:
        mask = masks_by_name.get(mask_name)

        if mask is None:
            continue

        mask_crop, _ = crop_image(mask, safe_bbox)

        if mask_crop.shape[:2] != combined_mask.shape[:2]:
            mask_crop = cv2.resize(
                mask_crop,
                (combined_mask.shape[1], combined_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        combined_mask = cv2.bitwise_or(combined_mask, mask_crop)

    output_crop = np.full_like(source_crop, 255)
    output_crop[combined_mask > 0] = source_crop[combined_mask > 0]

    return output_crop, safe_bbox


def ensure_color_image(image):
    """Return a BGR image for consistent visual crop composition."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def is_visible_ink_pixel(image):
    """Return a mask for non-white pixels in a visual layer artifact."""
    image = ensure_color_image(image)

    return np.any(image < 245, axis=2)


def paste_visual_layer(base_image, layer_image):
    """Paste visible pixels from a layer artifact onto a white/base image."""
    base_image = ensure_color_image(base_image)
    layer_image = ensure_color_image(layer_image)

    if layer_image.shape[:2] != base_image.shape[:2]:
        layer_image = cv2.resize(
            layer_image,
            (base_image.shape[1], base_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    ink_mask = is_visible_ink_pixel(layer_image)
    base_image[ink_mask] = layer_image[ink_mask]

    return base_image


def compose_visual_layers(layer_images_by_name, layer_names, fallback_image=None):
    """Compose selected N00 visual layer artifacts into one saturated crop source."""
    available_layers = [
        layer_images_by_name[name]
        for name in layer_names
        if name in layer_images_by_name and layer_images_by_name[name] is not None
    ]

    if available_layers:
        height, width = available_layers[0].shape[:2]
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)

        for layer_image in available_layers:
            canvas = paste_visual_layer(canvas, layer_image)

        return canvas

    if fallback_image is None:
        raise ValueError("No visual layer images available and no fallback image was provided.")

    return ensure_color_image(fallback_image)


def build_combined_mask_crop(masks_by_name, bbox, mask_names_to_keep, reference_image):
    """Build a raw white-on-black binary mask crop for trace/math use."""
    _, safe_bbox = crop_image(reference_image, bbox)
    height = safe_bbox["y2"] - safe_bbox["y1"]
    width = safe_bbox["x2"] - safe_bbox["x1"]

    combined_mask = np.zeros((height, width), dtype=np.uint8)

    for mask_name in mask_names_to_keep:
        mask = masks_by_name.get(mask_name)

        if mask is None:
            continue

        mask_crop, _ = crop_image(mask, safe_bbox)

        if mask_crop.shape[:2] != combined_mask.shape[:2]:
            mask_crop = cv2.resize(
                mask_crop,
                (combined_mask.shape[1], combined_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        combined_mask = cv2.bitwise_or(combined_mask, mask_crop)

    _, binary_mask = cv2.threshold(combined_mask, 0, 255, cv2.THRESH_BINARY)

    return binary_mask, safe_bbox


def get_visual_layer_source_name(layer):
    """Return the N00 visual artifact key expected for a semantic layer."""
    return {
        "blue": "blue_ink_layer",
        "red": "red_ink_layer",
        "green": "green_ink_layer",
        "black": "black_ink_layer",
        "unknown_color": "unknown_color_ink_layer",
        "colored": "colored_ink_layer",
    }.get(layer)

def get_crop_mask_names_for_record(record):
    """Choose which masks should be kept for analysis/context crop views."""
    layer = record.get("layer")

    if layer == "blue":
        return {
            "analysis": ["blue"],
            "analysis_mask": ["blue_continuity"],
            "context": ["blue", "black"],
        }

    if layer == "green":
        return {
            "analysis": ["green"],
            "analysis_mask": ["green"],
            "context": ["green", "black"],
        }

    if layer == "black":
        return {
            "analysis": ["black"],
            "analysis_mask": ["black"],
            "context": ["black"],
        }

    if layer == "red":
        return {
            "analysis": ["red"],
            "analysis_mask": ["red_continuity"],
            "context": ["red", "blue", "black"],
        }

    if layer == "unknown_color":
        return {
            "analysis": ["unknown_color"],
            "analysis_mask": ["unknown_color"],
            "context": ["unknown_color", "blue", "black"],
        }

    return {
        "analysis": [layer],
        "analysis_mask": [layer],
        "context": [layer],
    }

def save_crop_views_for_record(
    record,
    source_image,
    masks_by_name,
    layer_images_by_name,
    output_dir,
):
    """Save one canonical full-text crop and one topology mask per group."""
    layer = record["layer"]
    text_unit_id = record["text_unit_id"]
    bbox = record["final_bbox"]

    crop_dirs = {
        "full_text": os.path.join(output_dir, "crops", layer, "full_text"),
        "analysis_mask": os.path.join(output_dir, "crops", layer, "analysis_mask"),
    }

    for folder in crop_dirs.values():
        os.makedirs(folder, exist_ok=True)

    base_name = f"{layer}_{text_unit_id:04d}_{record['source_group_id']}"
    mask_names = get_crop_mask_names_for_record(record)
    analysis_mask_names = [
        name
        for name in mask_names["analysis_mask"]
        if name in masks_by_name
    ]
    continuity_repair_used = any(
        name.endswith("_continuity")
        for name in analysis_mask_names
    )

    if not analysis_mask_names:
        analysis_mask_names = mask_names["analysis"]

    # This target-layer-only crop is the single visual artifact used by Minos,
    # OCR, N05, and the UI. Compatibility fields below all reference this file.
    full_text_source = compose_visual_layers(
        layer_images_by_name=layer_images_by_name,
        layer_names=mask_names["analysis"],
        fallback_image=source_image,
    )
    full_text_crop, safe_bbox = crop_image(full_text_source, bbox)
    full_text_path = os.path.join(
        crop_dirs["full_text"],
        f"{base_name}_full_text.png",
    )
    save_image(full_text_crop, full_text_path)

    # Classification keeps exact semantic pixels. The binary topology mask may
    # use a separately generated crossing-repaired mask when N00 provided one.
    analysis_mask_crop, _ = build_combined_mask_crop(
        masks_by_name=masks_by_name,
        bbox=safe_bbox,
        mask_names_to_keep=analysis_mask_names,
        reference_image=source_image,
    )

    analysis_mask_path = os.path.join(
        crop_dirs["analysis_mask"],
        f"{base_name}_analysis_mask.png",
    )
    save_image(analysis_mask_crop, analysis_mask_path)

    visual_layer_source = get_visual_layer_source_name(layer)
    semantic_mask_source = (
        f"{layer}_ink_mask"
        if layer != "unknown_color"
        else "unknown_color_ink_mask"
    )
    mask_source = (
        f"{layer}_continuity_mask"
        if continuity_repair_used
        else semantic_mask_source
    )

    record["full_text_crop_path"] = full_text_path
    record["original_crop_path"] = None
    record["analysis_crop_path"] = full_text_path
    record["classification_crop_path"] = full_text_path
    record["classification_crop_source"] = "full_text_crop_path"
    record["classification_crop_policy"] = "target_layer_only_on_white_background"
    record["classification_layer"] = layer

    record["context_crop_path"] = None
    record["analysis_mask_crop_path"] = analysis_mask_path

    # Backward compatibility for old downstream consumers.
    record["refined_crop_path"] = full_text_path

    record["mask_source"] = mask_source
    record["semantic_mask_source"] = semantic_mask_source
    record["analysis_mask_policy"] = (
        "cross_color_crossing_repaired_target_mask"
        if continuity_repair_used
        else "exact_exclusive_target_layer_mask"
    )
    record["visual_layer_source"] = visual_layer_source
    record["final_bbox"] = safe_bbox

    record["refiner"]["status"] = "review"
    record["refiner"]["crop_views"] = {
        "full_text": full_text_path,
        "analysis_mask": analysis_mask_path,

        "analysis_visual_masks_kept": mask_names["analysis"],
        "analysis_masks_kept": analysis_mask_names,
        "visual_rendering": "n00_exclusive_color_layer_artifacts",
        "classification_rendering": "target_layer_only_on_white_background",
        "classification_crop_source": "full_text_crop_path",
        "analysis_mask_format": "binary_white_ink_on_black_background",
        "analysis_mask_policy": record["analysis_mask_policy"],
    }

    return record

def load_crop_generation_assets(scribemap_result):
    """Load source image, layer masks, and visual layer artifacts for crop generation."""
    artifacts = scribemap_result.get("preparation_artifacts", {})
    layer_mask_paths = scribemap_result.get("layer_mask_paths", {})
    continuity_mask_paths = scribemap_result.get("continuity_mask_paths", {})

    source_image_path = (
        artifacts.get("cropped")
        or scribemap_result.get("prepared_bw_image_path")
        or artifacts.get("colored_ink_layer")
        or artifacts.get("black_ink_layer")
    )

    if not source_image_path:
        raise ValueError("Could not find source image path for crop generation.")

    source_image = load_image(source_image_path, color=True)

    masks_by_name = {}

    for layer_name, mask_path in layer_mask_paths.items():
        if mask_path is None:
            continue

        if os.path.exists(mask_path):
            masks_by_name[layer_name] = load_image(mask_path, color=False)

    for layer_name, mask_path in continuity_mask_paths.items():
        if mask_path is None:
            continue

        if os.path.exists(mask_path):
            masks_by_name[f"{layer_name}_continuity"] = load_image(
                mask_path,
                color=False,
            )

    layer_artifact_keys = {
        "blue": "blue_ink_layer",
        "red": "red_ink_layer",
        "green": "green_ink_layer",
        "unknown_color": "unknown_color_ink_layer",
        "black": "black_ink_layer",
    }

    layer_images_by_name = {}

    for layer_name, artifact_key in layer_artifact_keys.items():
        layer_image_path = artifacts.get(artifact_key)

        if layer_image_path is None:
            continue

        if os.path.exists(layer_image_path):
            layer_images_by_name[layer_name] = load_image(layer_image_path, color=True)

    return source_image, masks_by_name, layer_images_by_name



def coerce_settings(settings=None):
    if settings is None:
        return RefinerSettings()

    if isinstance(settings, RefinerSettings):
        return settings

    if isinstance(settings, dict):
        settings = dict(settings)

        # Backward-compatible aliases from the previous split N02 settings file.
        if "input_group_mode" in settings and "input_mode" not in settings:
            settings["input_mode"] = settings["input_group_mode"]

        if "scribemap_2_layers_to_refine" in settings and "layers_to_refine" not in settings:
            settings["layers_to_refine"] = settings["scribemap_2_layers_to_refine"]

        clean = {}

        for key in RefinerSettings.__dataclass_fields__:
            if key in settings:
                clean[key] = settings[key]

        if "layers_to_refine" in clean and isinstance(clean["layers_to_refine"], list):
            clean["layers_to_refine"] = tuple(clean["layers_to_refine"])

        return RefinerSettings(**clean)

    raise TypeError("settings must be None, dict, or RefinerSettings")



def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, json_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    return json_path


def normalize_bbox(raw):
    """Normalize bbox input into x1, y1, x2, y2 dict."""
    if "bbox" in raw and raw["bbox"] is not None:
        raw = raw["bbox"]

    x1 = int(round(raw["x1"]))
    y1 = int(round(raw["y1"]))
    x2 = int(round(raw["x2"]))
    y2 = int(round(raw["y2"]))

    return {
        "x1": min(x1, x2),
        "y1": min(y1, y2),
        "x2": max(x1, x2),
        "y2": max(y1, y2),
    }

#geometry

def bbox_width(bbox):
    return max(int(bbox["x2"]) - int(bbox["x1"]), 1)


def bbox_height(bbox):
    return max(int(bbox["y2"]) - int(bbox["y1"]), 1)


def bbox_area(bbox):
    return bbox_width(bbox) * bbox_height(bbox)


def bbox_intersection(bbox_a, bbox_b):
    """Return the rectangular intersection of two bboxes, or None."""
    intersection = {
        "x1": max(int(bbox_a["x1"]), int(bbox_b["x1"])),
        "y1": max(int(bbox_a["y1"]), int(bbox_b["y1"])),
        "x2": min(int(bbox_a["x2"]), int(bbox_b["x2"])),
        "y2": min(int(bbox_a["y2"]), int(bbox_b["y2"])),
    }

    if (
        intersection["x2"] <= intersection["x1"]
        or intersection["y2"] <= intersection["y1"]
    ):
        return None

    return intersection


def bbox_distance(bbox_a, bbox_b):
    """Return edge-to-edge Euclidean distance between two bboxes."""
    horizontal_gap = max(
        int(bbox_a["x1"]) - int(bbox_b["x2"]),
        int(bbox_b["x1"]) - int(bbox_a["x2"]),
        0,
    )
    vertical_gap = max(
        int(bbox_a["y1"]) - int(bbox_b["y2"]),
        int(bbox_b["y1"]) - int(bbox_a["y2"]),
        0,
    )
    return float(np.hypot(horizontal_gap, vertical_gap))


def get_red_crossing_evidence(blue_group, red_group, red_mask, settings):
    """Measure whether accepted red ink credibly crosses one blue group.

    Args:
        blue_group: Normalized N01 blue source-group record.
        red_group: Normalized N01 red source-group record.
        red_mask: Full-document exclusive red mask.
        settings: RefinerSettings instance.

    Returns:
        JSON-safe evidence dictionary when the crossing is credible, else None.
    """
    blue_bbox = blue_group["bbox"]
    red_bbox = red_group["bbox"]
    intersection = bbox_intersection(blue_bbox, red_bbox)

    if intersection is None:
        return None

    mask_crop = red_mask[
        intersection["y1"]:intersection["y2"],
        intersection["x1"]:intersection["x2"],
    ]
    ink_y, ink_x = np.nonzero(mask_crop > 0)
    ink_pixels = int(len(ink_x))

    if ink_pixels < settings.correction_min_crossing_ink_pixels:
        return None

    global_x = ink_x + intersection["x1"]
    global_y = ink_y + intersection["y1"]
    horizontal_span = int(global_x.max() - global_x.min() + 1)
    vertical_span = int(global_y.max() - global_y.min() + 1)
    horizontal_ratio = horizontal_span / float(bbox_width(blue_bbox))
    vertical_ratio = vertical_span / float(bbox_height(blue_bbox))

    if (
        horizontal_ratio < settings.correction_min_horizontal_span_ratio
        and vertical_ratio < settings.correction_min_vertical_span_ratio
    ):
        return None

    return {
        "blue_source_group_id": blue_group["source_group_id"],
        "red_source_group_id": red_group["source_group_id"],
        "intersection_bbox": intersection,
        "red_ink_pixels_inside_blue": ink_pixels,
        "horizontal_span_px": horizontal_span,
        "vertical_span_px": vertical_span,
        "horizontal_span_ratio": round(horizontal_ratio, 4),
        "vertical_span_ratio": round(vertical_ratio, 4),
        "reason": "accepted_red_group_crosses_blue_text",
    }


def is_red_replacement_candidate(red_group, settings):
    """Return whether a red group is large enough to contain correction text."""
    width = int(red_group["width"])
    height = int(red_group["height"])
    area = int(red_group["area"])
    aspect_ratio = max(width / float(height), height / float(width))

    return (
        width >= settings.correction_replacement_min_width_px
        and height >= settings.correction_replacement_min_height_px
        and area >= settings.correction_replacement_min_area
        and aspect_ratio <= settings.correction_replacement_max_aspect_ratio
    )


def apply_red_correction_routing(source_groups, red_mask, settings):
    """Pair crossed blue text with nearby red replacement-text candidates.

    Blue text crossed by an accepted red group is preserved as evidence but is
    removed from Minos/OCR routing. Nearby text-shaped red groups are promoted
    directly to N05 without asking Minos to classify their known color role.

    Args:
        source_groups: Mutable normalized N01 source-group records.
        red_mask: Full-document exclusive red mask.
        settings: RefinerSettings instance.

    Returns:
        Summary dictionary describing suppressed blue and promoted red groups.
    """
    summary = {
        "enabled": bool(settings.correction_routing_enabled),
        "suppressed_blue_count": 0,
        "promoted_red_count": 0,
        "unpaired_deleted_blue_count": 0,
        "relationships": [],
    }

    if not settings.correction_routing_enabled or red_mask is None:
        return summary

    blue_groups = [group for group in source_groups if group["layer"] == "blue"]
    red_groups = [group for group in source_groups if group["layer"] == "red"]
    promoted_red_ids = set()

    for blue_group in blue_groups:
        crossing_evidence = [
            evidence
            for red_group in red_groups
            if (
                evidence := get_red_crossing_evidence(
                    blue_group=blue_group,
                    red_group=red_group,
                    red_mask=red_mask,
                    settings=settings,
                )
            ) is not None
        ]

        if not crossing_evidence:
            continue

        crossing_red_ids = {
            evidence["red_source_group_id"]
            for evidence in crossing_evidence
        }
        replacement_candidates = []

        for red_group in red_groups:
            if not is_red_replacement_candidate(red_group, settings):
                continue

            distance = bbox_distance(blue_group["bbox"], red_group["bbox"])
            if distance > settings.correction_replacement_max_distance_px:
                continue

            replacement_candidates.append(
                (
                    0 if red_group["source_group_id"] in crossing_red_ids else 1,
                    round(distance, 4),
                    -int(red_group["area"]),
                    str(red_group["source_group_id"]),
                    red_group,
                )
            )

        replacement_candidates.sort(key=lambda item: item[:4])
        selected_red_groups = [
            item[4]
            for item in replacement_candidates[
                :settings.correction_max_replacements_per_blue
            ]
        ]
        replacement_ids = [
            group["source_group_id"]
            for group in selected_red_groups
        ]

        blue_group.update({
            "role_guess": "red_crossed_deleted_handwriting",
            "recommended_next_node": "suppressed_replaced_text",
            "minos_required": False,
            "minos_mode": "skip_red_crossed_blue",
            "is_final_text_candidate": False,
            "preserve_as_evidence": True,
            "correction_role": "deleted_original",
            "force_handwritten_ocr": False,
            "crossing_red_source_group_ids": sorted(crossing_red_ids),
            "replacement_red_source_group_ids": replacement_ids,
            "correction_evidence": crossing_evidence,
        })

        for red_group in selected_red_groups:
            paired_blue_ids = red_group.setdefault(
                "replaces_blue_source_group_ids",
                [],
            )
            if blue_group["source_group_id"] not in paired_blue_ids:
                paired_blue_ids.append(blue_group["source_group_id"])

            red_group.update({
                "layer_hypothesis": "red_handwritten_correction",
                "role_guess": "replacement_handwriting",
                "recommended_next_node": "N05_handwritten_ocr",
                "minos_required": False,
                "minos_mode": "skip_known_red_replacement",
                "is_final_text_candidate": True,
                "preserve_as_evidence": True,
                "correction_role": "replacement_text",
                "force_handwritten_ocr": True,
            })
            promoted_red_ids.add(red_group["source_group_id"])

        summary["relationships"].append({
            "deleted_blue_source_group_id": blue_group["source_group_id"],
            "crossing_red_source_group_ids": sorted(crossing_red_ids),
            "replacement_red_source_group_ids": replacement_ids,
            "evidence": crossing_evidence,
        })

    summary["suppressed_blue_count"] = len(summary["relationships"])
    summary["promoted_red_count"] = len(promoted_red_ids)
    summary["unpaired_deleted_blue_count"] = sum(
        not relationship["replacement_red_source_group_ids"]
        for relationship in summary["relationships"]
    )
    return summary


def pad_bbox(bbox, padding_px):
    return {
        "x1": int(bbox["x1"]) - int(padding_px),
        "y1": int(bbox["y1"]) - int(padding_px),
        "x2": int(bbox["x2"]) + int(padding_px),
        "y2": int(bbox["y2"]) + int(padding_px),
    }



def get_layer_policy(layer_name):
    """Return layer-aware routing/refinement policy for one ScribeMap layer."""
    policies = {
        "blue": {
            "layer": "blue",
            "layer_hypothesis": "probable_handwriting",
            "role_guess": "probable_handwriting",
            "recommended_next_node": "N05_handwritten_ocr",
            "minos_required": True,
            "minos_mode": "handwriting_audit",
            "is_final_text_candidate": True,
            "preserve_as_evidence": False,
            "route_if_minos_agrees": "N05_handwritten_ocr",
            "route_if_minos_disagrees": "review_or_dual_route",
            "analysis_view": "blue_only",
            "context_view": "blue_with_black_context",
        },

        "green": {
            "layer": "green",
            "layer_hypothesis": "probable_handwriting",
            "role_guess": "probable_handwriting",
            "recommended_next_node": "N05_handwritten_ocr",
            "minos_required": True,
            "minos_mode": "handwriting_audit",
            "is_final_text_candidate": True,
            "preserve_as_evidence": False,
            "route_if_minos_agrees": "N05_handwritten_ocr",
            "route_if_minos_disagrees": "review_or_dual_route",
            "analysis_view": "green_only",
            "context_view": "green_with_black_context",
        },

        "red": {
            "layer": "red",
            "layer_hypothesis": "probable_markup_or_correction",
            "role_guess": "correction_markup",
            "recommended_next_node": "N06_correction_resolver",
            "minos_required": False,
            "minos_mode": "skip_markup_evidence",
            "is_final_text_candidate": False,
            "preserve_as_evidence": True,
            "route_if_writing_like": "correction_text_evidence",
            "route_if_not_writing_like": "correction_shape_evidence",
            "analysis_view": "red_only",
            "context_view": "red_with_blue_black_context",
        },

        "black": {
            "layer": "black",
            "layer_hypothesis": "dark_ink_unknown_role",
            "role_guess": "dark_ink_needs_visual_routing",
            "recommended_next_node": "N03_visual_classification_router",
            "minos_required": True,
            "minos_mode": "primary_router",
            "is_final_text_candidate": True,
            "preserve_as_evidence": False,
            "route_if_printed": "N04_printed_ocr_or_context",
            "route_if_handwriting": "N05_handwritten_ocr",
            "route_if_mixed": "N04_and_N05",
            "route_if_noise": "review",
            "analysis_view": "black_only",
            "context_view": "black_only",
        },

        "unknown_color": {
            "layer": "unknown_color",
            "layer_hypothesis": "ambiguous_colored_ink",
            "role_guess": "unknown_colored_ink",
            "recommended_next_node": "N03_visual_classification_router",
            "minos_required": True,
            "minos_mode": "fallback_router",
            "is_final_text_candidate": True,
            "preserve_as_evidence": False,
            "analysis_view": "unknown_only",
            "context_view": "unknown_with_context",
        },
    }

    return policies.get(layer_name, {
        "layer": layer_name,
        "layer_hypothesis": "unknown_layer",
        "role_guess": "review",
        "recommended_next_node": "review",
        "minos_required": True,
        "minos_mode": "fallback_router",
        "is_final_text_candidate": True,
        "preserve_as_evidence": False,
        "analysis_view": f"{layer_name}_only",
        "context_view": f"{layer_name}_with_context",
    })


def load_scribemap_result_from_payload(payload):
    """Load the full ScribeMap result JSON from a bridge payload."""
    scribemap_result_path = payload.get("scribemap_result_path")

    if not scribemap_result_path:
        raise ValueError(
            "Payload must contain 'scribemap_result_path' for scribemap_2_layers mode."
        )

    if not os.path.exists(scribemap_result_path):
        raise FileNotFoundError(
            f"ScribeMap result not found: {scribemap_result_path}"
        )

    return load_json(scribemap_result_path)


def collect_layer_groups(scribemap_result, layers_to_refine):
    """Collect selected ScribeMap 2.0 layer groups.

    Args:
        scribemap_result: Full ScribeMap result dictionary.
        layers_to_refine: Iterable of layer names, e.g. ("blue", "black").

    Returns:
        Flat list of normalized layer-group records.
    """
    layer_results = scribemap_result.get("layer_results", {})
    collected = []

    for layer_name in layers_to_refine:
        layer_payload = layer_results.get(layer_name)

        if layer_payload is None:
            continue

        policy = get_layer_policy(layer_name)
        groups = layer_payload.get("groups", [])

        for index, group in enumerate(groups, start=1):
            bbox = normalize_bbox(group)

            source_group_id = (
                group.get("group_uid")
                or f"{layer_name}_{group.get('group_id', index):04d}"
            )

            collected.append({
                "source_type": "scribemap_2_layer_group",
                "source_group_id": source_group_id,
                "source_layer_group_id": group.get("group_id", index),
                "layer": layer_name,
                "bbox": bbox,
                "width": bbox_width(bbox),
                "height": bbox_height(bbox),
                "area": bbox_area(bbox),
                "density": group.get("density"),
                "component_count": group.get("component_count"),
                "layer_hypothesis": policy.get("layer_hypothesis"),
                "role_guess": policy.get("role_guess"),
                "recommended_next_node": policy.get("recommended_next_node"),
                "minos_required": policy.get("minos_required", True),
                "minos_mode": policy.get("minos_mode", "fallback_router"),
                "analysis_view": policy.get("analysis_view"),
                "context_view": policy.get("context_view"),
                "policy": policy,
                "source": group,
                "is_final_text_candidate": policy.get("is_final_text_candidate", True),
                "preserve_as_evidence": policy.get("preserve_as_evidence", False),
            })

    return collected


def find_horizontal_text_bands(mask_crop, settings):
    """Find separated horizontal ink bands inside a suspected stacked crop.

    Args:
        mask_crop: Binary layer mask cropped to a source group bbox.
        settings: RefinerSettings instance.

    Returns:
        List of local band dictionaries with y1/y2 and ink counts.
    """
    if mask_crop is None or mask_crop.size == 0:
        return []

    binary = (mask_crop > 0).astype(np.uint8)
    height, width = binary.shape[:2]
    if height <= 0 or width <= 0:
        return []

    projection = binary.sum(axis=1)
    if height >= 3:
        projection = np.convolve(projection, np.ones(3) / 3.0, mode="same")

    row_threshold = max(
        1.0,
        float(width) * float(settings.stacked_text_row_ink_ratio),
    )
    active_rows = projection >= row_threshold

    raw_bands = []
    start = None
    for row_index, is_active in enumerate(active_rows):
        if is_active and start is None:
            start = row_index
        elif not is_active and start is not None:
            raw_bands.append((start, row_index))
            start = None
    if start is not None:
        raw_bands.append((start, height))

    if not raw_bands:
        return []

    merged_bands = []
    for band in raw_bands:
        if not merged_bands:
            merged_bands.append(list(band))
            continue
        gap = band[0] - merged_bands[-1][1]
        if gap <= int(settings.stacked_text_merge_gap_px):
            merged_bands[-1][1] = band[1]
        else:
            merged_bands.append(list(band))

    bands = []
    for y1, y2 in merged_bands:
        band_height = y2 - y1
        ink_pixels = int(binary[y1:y2, :].sum())
        if band_height < int(settings.stacked_text_min_segment_height_px):
            continue
        if ink_pixels <= 0:
            continue
        bands.append({
            "y1": int(y1),
            "y2": int(y2),
            "height": int(band_height),
            "ink_pixels": ink_pixels,
        })

    return bands


def find_projection_valley_split_bands(mask_crop, settings):
    """Find a two-line split when rows are connected but have a clear trough.

    Args:
        mask_crop: Binary layer mask cropped to a source group bbox.
        settings: RefinerSettings instance.

    Returns:
        Two band dictionaries, or an empty list when no safe trough exists.
    """
    if not bool(settings.stacked_text_projection_valley_enabled):
        return []
    if mask_crop is None or mask_crop.size == 0:
        return []

    binary = (mask_crop > 0).astype(np.uint8)
    height, width = binary.shape[:2]
    min_segment_height = int(settings.stacked_text_min_segment_height_px)
    if height < min_segment_height * 2 or width <= 0:
        return []

    projection = binary.sum(axis=1).astype(float)
    smooth_window = 5 if height >= 20 else 3
    projection = np.convolve(
        projection,
        np.ones(smooth_window) / float(smooth_window),
        mode="same",
    )
    max_projection = float(projection.max()) if projection.size else 0.0
    if max_projection <= 0:
        return []

    start = min_segment_height
    end = height - min_segment_height
    if end <= start:
        return []

    total_ink = max(int(binary.sum()), 1)
    min_side_ratio = float(settings.stacked_text_projection_min_side_ink_ratio)
    max_valley_ratio = float(settings.stacked_text_projection_valley_max_ratio)
    valid_valleys = []
    for row_index in range(start, end):
        valley_ratio = float(projection[row_index]) / max_projection
        if valley_ratio > max_valley_ratio:
            continue
        top_ink = int(binary[:row_index, :].sum())
        bottom_ink = int(binary[row_index:, :].sum())
        top_ratio = top_ink / total_ink
        bottom_ratio = bottom_ink / total_ink
        if top_ratio < min_side_ratio or bottom_ratio < min_side_ratio:
            continue
        valid_valleys.append(
            (
                valley_ratio,
                abs(top_ratio - bottom_ratio),
                abs(row_index - height / 2.0),
                row_index,
                top_ink,
                bottom_ink,
            )
        )

    if not valid_valleys:
        return []

    (
        valley_ratio,
        _side_balance,
        _center_distance,
        valley_index,
        top_ink,
        bottom_ink,
    ) = min(valid_valleys)

    return [
        {
            "y1": 0,
            "y2": int(valley_index),
            "height": int(valley_index),
            "ink_pixels": top_ink,
            "split_method": "projection_valley",
            "valley_y": int(valley_index),
            "valley_ratio": round(valley_ratio, 4),
        },
        {
            "y1": int(valley_index),
            "y2": int(height),
            "height": int(height - valley_index),
            "ink_pixels": bottom_ink,
            "split_method": "projection_valley",
            "valley_y": int(valley_index),
            "valley_ratio": round(valley_ratio, 4),
        },
    ]


def should_attempt_stacked_text_split(source_group, settings):
    """Return whether one source group is suspicious enough to inspect."""
    if not bool(settings.stacked_text_split_enabled):
        return False
    if source_group.get("layer") not in set(settings.stacked_text_split_layers):
        return False
    if not source_group.get("is_final_text_candidate", True):
        return False

    width = int(source_group.get("width") or bbox_width(source_group["bbox"]))
    height = int(source_group.get("height") or bbox_height(source_group["bbox"]))
    aspect_ratio = width / max(height, 1)

    return (
        height >= int(settings.stacked_text_min_height_px)
        and width >= int(settings.stacked_text_min_width_px)
        and aspect_ratio >= float(settings.stacked_text_min_aspect_ratio)
    )


def build_stacked_split_children(source_group, bands, mask_crop, settings):
    """Build child source groups from horizontal text bands."""
    bbox = source_group["bbox"]
    binary = (mask_crop > 0).astype(np.uint8)
    parent_id = source_group["source_group_id"]
    padding = int(settings.stacked_text_segment_padding_px)
    children = []

    for index, band in enumerate(bands, start=1):
        local_y1 = max(0, int(band["y1"]) - padding)
        local_y2 = min(binary.shape[0], int(band["y2"]) + padding)
        band_mask = binary[local_y1:local_y2, :]
        coordinates = np.argwhere(band_mask > 0)
        if coordinates.size == 0:
            continue

        _, local_x_values = coordinates[:, 0], coordinates[:, 1]
        local_x1 = max(0, int(local_x_values.min()) - padding)
        local_x2 = min(binary.shape[1], int(local_x_values.max()) + padding + 1)

        child_bbox = {
            "x1": int(bbox["x1"]) + local_x1,
            "y1": int(bbox["y1"]) + local_y1,
            "x2": int(bbox["x1"]) + local_x2,
            "y2": int(bbox["y1"]) + local_y2,
        }
        child = dict(source_group)
        child["source_group_id"] = f"{parent_id}_line{index:02d}"
        child["bbox"] = child_bbox
        child["width"] = bbox_width(child_bbox)
        child["height"] = bbox_height(child_bbox)
        child["area"] = bbox_area(child_bbox)
        child["source_type"] = f"{source_group['source_type']}_stacked_split_child"
        child["parent_stacked_source_group_id"] = parent_id
        child["stacked_split_child_index"] = index
        child["stacked_split_child_count"] = len(bands)
        child["stacked_split_evidence"] = {
            "parent_source_group_id": parent_id,
            "local_band": band,
            "child_bbox": child_bbox,
            "reason": "horizontal_projection_separated_text_bands",
        }
        child["source"] = dict(source_group.get("source", {}))
        child["source"]["parent_stacked_source_group_id"] = parent_id
        children.append(child)

    return children


def split_stacked_source_groups(source_groups, masks_by_name, settings):
    """Split obvious stacked multi-line source groups before crop generation.

    Args:
        source_groups: Normalized ScribeMap layer groups.
        masks_by_name: Layer masks from N01 artifacts.
        settings: RefinerSettings instance.

    Returns:
        Tuple of:
            split source group list
            summary dictionary
    """
    output_groups = []
    events = []

    for source_group in source_groups:
        if not should_attempt_stacked_text_split(source_group, settings):
            output_groups.append(source_group)
            continue

        layer_mask = masks_by_name.get(source_group["layer"])
        if layer_mask is None:
            output_groups.append(source_group)
            continue

        mask_crop, safe_bbox = crop_image(layer_mask, source_group["bbox"])
        bands = find_horizontal_text_bands(mask_crop, settings)
        split_method = "inactive_row_bands"
        if len(bands) < 2:
            valley_bands = find_projection_valley_split_bands(mask_crop, settings)
            if valley_bands:
                bands = valley_bands
                split_method = "projection_valley"
        if len(bands) < 2 or len(bands) > int(settings.stacked_text_max_segments):
            output_groups.append(source_group)
            continue

        gaps = [
            int(next_band["y1"] - current_band["y2"])
            for current_band, next_band in zip(bands, bands[1:])
        ]
        if (
            split_method == "inactive_row_bands"
            and (not gaps or max(gaps) < int(settings.stacked_text_min_gap_px))
        ):
            output_groups.append(source_group)
            continue

        children = build_stacked_split_children(
            source_group=source_group,
            bands=bands,
            mask_crop=mask_crop,
            settings=settings,
        )
        if len(children) < 2:
            output_groups.append(source_group)
            continue

        output_groups.extend(children)
        events.append({
            "parent_source_group_id": source_group["source_group_id"],
            "layer": source_group["layer"],
            "parent_bbox": dict(source_group["bbox"]),
            "safe_bbox": safe_bbox,
            "child_count": len(children),
            "gaps": gaps,
            "bands": bands,
            "split_method": split_method,
            "child_source_group_ids": [
                child["source_group_id"]
                for child in children
            ],
        })

    return output_groups, {
        "enabled": bool(settings.stacked_text_split_enabled),
        "input_group_count": len(source_groups),
        "output_group_count": len(output_groups),
        "split_parent_count": len(events),
        "added_group_count": len(output_groups) - len(source_groups),
        "events": events,
    }


def build_refined_record(text_unit_id, source_group, settings):
    """Build one first-pass refined record from one layer source group."""
    final_bbox = pad_bbox(
        bbox=source_group["bbox"],
        padding_px=settings.crop_padding_px
    )

    return {
        "text_unit_id": text_unit_id,
        "source_type": source_group["source_type"],
        "source_group_id": source_group["source_group_id"],
        "source_layer_group_id": source_group["source_layer_group_id"],

        "layer": source_group["layer"],
        "layer_hypothesis": source_group["layer_hypothesis"],
        "role_guess": source_group["role_guess"],

        "bbox": dict(source_group["bbox"]),
        "final_bbox": dict(final_bbox),

        "width": bbox_width(final_bbox),
        "height": bbox_height(final_bbox),
        "area": bbox_area(final_bbox),

        "density": source_group.get("density"),
        "component_count": source_group.get("component_count"),

        "recommended_next_node": source_group["recommended_next_node"],
        "minos_required": source_group["minos_required"],
        "minos_mode": source_group["minos_mode"],
        "is_final_text_candidate": source_group.get("is_final_text_candidate", True),
        "preserve_as_evidence": source_group.get("preserve_as_evidence", False),
        "force_handwritten_ocr": source_group.get("force_handwritten_ocr", False),
        "correction_role": source_group.get("correction_role"),
        "crossing_red_source_group_ids": source_group.get(
            "crossing_red_source_group_ids",
            [],
        ),
        "replacement_red_source_group_ids": source_group.get(
            "replacement_red_source_group_ids",
            [],
        ),
        "replaces_blue_source_group_ids": source_group.get(
            "replaces_blue_source_group_ids",
            [],
        ),
        "correction_evidence": source_group.get("correction_evidence", []),

        "analysis_view": source_group["analysis_view"],
        "context_view": source_group["context_view"],

        "original_crop_path": None,
        "analysis_crop_path": None,
        "classification_crop_path": None,
        "classification_crop_source": None,
        "classification_crop_policy": None,
        "classification_layer": source_group["layer"],
        "context_crop_path": None,
        "analysis_mask_crop_path": None,
        "refined_crop_path": None,

        "mask_source": None,
        "visual_layer_source": None,

        "overlaps": [],
        "quality": None,

        "refiner": {
            "status": "pending_crop_generation",
            "final_bbox": dict(final_bbox),
            "next_node": source_group["recommended_next_node"],
            "minos_required": source_group["minos_required"],
            "minos_mode": source_group["minos_mode"],
        },

        "source": source_group,
    }


def load_refiner_settings(settings_path=DEFAULT_SETTINGS_PATH):
    """Load N02 settings from JSON.

    Args:
        settings_path: Settings JSON path.

    Returns:
        RefinerSettings instance.
    """
    if settings_path and os.path.exists(settings_path):
        return coerce_settings(load_json(settings_path))

    return coerce_settings(None)


class CropRefiner:
    def __init__(self, settings=None):
        if settings is None:
            self.settings = load_refiner_settings(DEFAULT_SETTINGS_PATH)
        else:
            self.settings = coerce_settings(settings)

    def refine_document(self, classified_groups_json_path, output_path=None):
        payload = load_json(classified_groups_json_path)
        source_image = None
        masks_by_name = {}
        layer_images_by_name = {}
        correction_routing = {
            "enabled": False,
            "suppressed_blue_count": 0,
            "promoted_red_count": 0,
            "unpaired_deleted_blue_count": 0,
            "relationships": [],
        }
        stacked_text_split = {
            "enabled": bool(self.settings.stacked_text_split_enabled),
            "input_group_count": 0,
            "output_group_count": 0,
            "split_parent_count": 0,
            "added_group_count": 0,
            "events": [],
        }

        if self.settings.input_mode == "scribemap_2_layers":
            scribemap_result = load_scribemap_result_from_payload(payload)

            source_groups = collect_layer_groups(
                scribemap_result=scribemap_result,
                layers_to_refine=self.settings.layers_to_refine,
            )
            source_image, masks_by_name, layer_images_by_name = (
                load_crop_generation_assets(scribemap_result)
            )
            correction_routing = apply_red_correction_routing(
                source_groups=source_groups,
                red_mask=masks_by_name.get("red"),
                settings=self.settings,
            )
            source_groups, stacked_text_split = split_stacked_source_groups(
                source_groups=source_groups,
                masks_by_name=masks_by_name,
                settings=self.settings,
            )
        else:
            source_groups = []


        refined_groups = [
            build_refined_record(
                text_unit_id=index + 1,
                source_group=source_group,
                settings=self.settings
            )
            for index, source_group in enumerate(source_groups)
        ]

        if output_path is not None:
            metadata_dir = os.path.dirname(output_path)
            output_dir = os.path.dirname(metadata_dir)
        else:
            output_dir = os.getcwd()

        crops_output_dir = os.path.join(output_dir, "crops")
        if os.path.isdir(crops_output_dir):
            shutil.rmtree(crops_output_dir)

        if self.settings.input_mode == "scribemap_2_layers":
            refined_groups = [
                save_crop_views_for_record(
                    record=record,
                    source_image=source_image,
                    masks_by_name=masks_by_name,
                    layer_images_by_name=layer_images_by_name,
                    output_dir=output_dir,
                )
                for record in refined_groups
            ]

        result = {
            "document_id": payload.get("document_id", "document"),
            "input_mode": self.settings.input_mode,
            "layers_to_refine": list(self.settings.layers_to_refine),
            "settings": self.settings.to_dict(),
            "source_group_count": len(source_groups),
            "source_groups": source_groups,
            "refined_groups": refined_groups,
            "correction_routing": correction_routing,
            "stacked_text_split": stacked_text_split,
            "crops_output_dir": crops_output_dir,
            "refined_crops_dir": crops_output_dir,
            "summary": {
                "source_group_count": len(source_groups),
                "group_count": len(refined_groups),
                "suppressed_blue_count": correction_routing[
                    "suppressed_blue_count"
                ],
                "promoted_red_count": correction_routing[
                    "promoted_red_count"
                ],
                "stacked_split_parent_count": stacked_text_split[
                    "split_parent_count"
                ],
                "stacked_split_added_group_count": stacked_text_split[
                    "added_group_count"
                ],
            }
        }

        if output_path is not None:
            save_json(result, output_path)

        return result


__all__ = [
    "CropRefiner",
    "RefinerSettings",
    "apply_red_correction_routing",
    "coerce_settings",
    "load_refiner_settings",
]
