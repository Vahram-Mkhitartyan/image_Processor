import json
import os
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
            "context": ["blue", "black"],
        }

    if layer == "green":
        return {
            "analysis": ["green"],
            "context": ["green", "black"],
        }

    if layer == "black":
        return {
            "analysis": ["black"],
            "context": ["black"],
        }

    if layer == "red":
        return {
            "analysis": ["red"],
            "context": ["red", "blue", "black"],
        }

    if layer == "unknown_color":
        return {
            "analysis": ["unknown_color"],
            "context": ["unknown_color", "blue", "black"],
        }

    return {
        "analysis": [layer],
        "context": [layer],
    }

def save_crop_views_for_record(
    record,
    source_image,
    masks_by_name,
    layer_images_by_name,
    output_dir,
):
    """Save original, analysis, context, classification, and mask crop views."""
    layer = record["layer"]
    text_unit_id = record["text_unit_id"]
    bbox = record["final_bbox"]

    crop_dirs = {
        "original": os.path.join(output_dir, "crops", layer, "original"),
        "analysis": os.path.join(output_dir, "crops", layer, "analysis"),
        "context": os.path.join(output_dir, "crops", layer, "context"),
        "analysis_mask": os.path.join(output_dir, "crops", layer, "analysis_mask"),
        "classification": os.path.join(output_dir, "crops", layer, "classification"),
    }

    for folder in crop_dirs.values():
        os.makedirs(folder, exist_ok=True)

    base_name = f"{layer}_{text_unit_id:04d}_{record['source_group_id']}"
    mask_names = get_crop_mask_names_for_record(record)

    # Full debug view. This may contain multiple layers.
    # Do not use as the default Minos input.
    all_visual_layers = ["black", "blue", "red", "green", "unknown_color"]

    original_source = compose_visual_layers(
        layer_images_by_name=layer_images_by_name,
        layer_names=all_visual_layers,
        fallback_image=source_image,
    )
    original_crop, safe_bbox = crop_image(original_source, bbox)

    original_path = os.path.join(crop_dirs["original"], f"{base_name}_original.png")
    save_image(original_crop, original_path)

    # Target-layer-only visual view.
    # This is the main classification evidence.
    analysis_source = compose_visual_layers(
        layer_images_by_name=layer_images_by_name,
        layer_names=mask_names["analysis"],
        fallback_image=source_image,
    )
    analysis_crop, _ = crop_image(analysis_source, safe_bbox)

    analysis_path = os.path.join(crop_dirs["analysis"], f"{base_name}_analysis.png")
    save_image(analysis_crop, analysis_path)

    # Save a duplicate under classification/ to make the contract explicit.
    classification_path = os.path.join(
        crop_dirs["classification"],
        f"{base_name}_classification.png",
    )
    save_image(analysis_crop, classification_path)

    # Context view. Useful for correction resolver / debugging, not default Minos input.
    context_source = compose_visual_layers(
        layer_images_by_name=layer_images_by_name,
        layer_names=mask_names["context"],
        fallback_image=source_image,
    )
    context_crop, _ = crop_image(context_source, safe_bbox)

    context_path = os.path.join(crop_dirs["context"], f"{base_name}_context.png")
    save_image(context_crop, context_path)

    # Preserve the exact target-layer pixels; ScribeTrace owns any later
    # topology processing and must not inherit visual context from other layers.
    analysis_mask_crop, _ = build_combined_mask_crop(
        masks_by_name=masks_by_name,
        bbox=safe_bbox,
        mask_names_to_keep=mask_names["analysis"],
        reference_image=source_image,
    )

    analysis_mask_path = os.path.join(
        crop_dirs["analysis_mask"],
        f"{base_name}_analysis_mask.png",
    )
    save_image(analysis_mask_crop, analysis_mask_path)

    visual_layer_source = get_visual_layer_source_name(layer)
    mask_source = f"{layer}_ink_mask" if layer != "unknown_color" else "unknown_color_ink_mask"

    record["original_crop_path"] = original_path
    record["analysis_crop_path"] = analysis_path
    record["classification_crop_path"] = classification_path
    record["classification_crop_source"] = "analysis_crop_path"
    record["classification_crop_policy"] = "target_layer_only_on_white_background"
    record["classification_layer"] = layer

    record["context_crop_path"] = context_path
    record["analysis_mask_crop_path"] = analysis_mask_path

    # Backward compatibility for old downstream consumers.
    record["refined_crop_path"] = analysis_path

    record["mask_source"] = mask_source
    record["visual_layer_source"] = visual_layer_source
    record["final_bbox"] = safe_bbox

    record["refiner"]["status"] = "review"
    record["refiner"]["crop_views"] = {
        "original": original_path,
        "analysis": analysis_path,
        "classification": classification_path,
        "context": context_path,
        "analysis_mask": analysis_mask_path,

        "analysis_masks_kept": mask_names["analysis"],
        "context_masks_kept": mask_names["context"],

        "visual_rendering": "n00_exclusive_color_layer_artifacts",
        "classification_rendering": "target_layer_only_on_white_background",
        "classification_crop_source": "analysis_crop_path",
        "analysis_mask_format": "binary_white_ink_on_black_background",
    }

    return record

def load_crop_generation_assets(scribemap_result):
    """Load source image, layer masks, and visual layer artifacts for crop generation."""
    artifacts = scribemap_result.get("preparation_artifacts", {})
    layer_mask_paths = scribemap_result.get("layer_mask_paths", {})

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

        if self.settings.input_mode == "scribemap_2_layers":
            scribemap_result = load_scribemap_result_from_payload(payload)

            source_groups = collect_layer_groups(
                scribemap_result=scribemap_result,
                layers_to_refine=self.settings.layers_to_refine,
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

        if self.settings.input_mode == "scribemap_2_layers":
            source_image, masks_by_name, layer_images_by_name = load_crop_generation_assets(scribemap_result)

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
            "crops_output_dir": crops_output_dir,
            "refined_crops_dir": crops_output_dir,
            "summary": {
                "source_group_count": len(source_groups),
                "group_count": len(refined_groups)
            }
        }

        if output_path is not None:
            save_json(result, output_path)

        return result


__all__ = ["CropRefiner", "RefinerSettings", "coerce_settings", "load_refiner_settings"]
