"""
Node N02: Crop Refiner
======================

Orchestrator for document-level text-unit grouping and crop-quality scoring.
The heavy helpers live beside this file so the node stays readable while the
public entrypoint remains `CropRefiner.refine_document()`.
"""

import os
import sys

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

DEFAULT_SETTINGS_PATH = os.path.join(MODULE_DIR, "settings.json")

from n02_border_profiles import attach_border_profile_fragments
from n02_debug_preview import groups_by_layer, render_debug_preview, render_layer_debug_outputs
from n02_geometry import pad_bbox
from n02_io import (
    build_debug_preview_path,
    load_gray_image,
    load_json,
    prepare_refined_crops_dir,
    resolve_refinement_image_path,
    save_crop,
    save_json,
)

from n02_normalization import (
    collect_scribemap_2_layer_groups,
    filter_early_artifacts,
    normalize_source_groups,
)

from n02_quality import evaluate_crop_quality
from n02_records import build_refined_text_unit_record, build_refinement_summary
from n02_settings import RefinerSettings, coerce_settings, load_refiner_settings
from n02_text_units import build_line_buckets, build_text_units_from_line_bucket


def build_layer_summary(refined_groups):
    """Build count summaries for every ScribeMap layer.

    Args:
        refined_groups: Final refined group records.

    Returns:
        Dictionary keyed by layer slug.
    """
    summaries = {}

    for layer, groups in groups_by_layer(refined_groups).items():
        layer_summary = build_refinement_summary(groups)
        layer_summary["group_count"] = len(groups)
        summaries[layer] = layer_summary

    return summaries


def build_layer_json_path(output_path, document_id, layer):
    """Build a per-layer refined metadata JSON path.

    Args:
        output_path: Main refined metadata JSON path.
        document_id: Current document id.
        layer: Filesystem-friendly layer slug.

    Returns:
        Per-layer metadata JSON path.
    """
    metadata_dir = os.path.dirname(output_path)
    return os.path.join(metadata_dir, f"{document_id}_refined_groups_{layer}.json")


class CropRefiner:
    """N02 orchestrator for grouping source boxes into OCR-ready crops.

    Args:
        settings: Optional RefinerSettings object or dictionary overrides.

    Returns:
        CropRefiner instance.
    """

    def __init__(self, settings=None):
        """Initialize the crop refiner.

        Args:
            settings: Optional RefinerSettings or dictionary overrides.

        Returns:
            None.
        """
        if settings is None and os.path.exists(DEFAULT_SETTINGS_PATH):
            self.settings = load_refiner_settings(DEFAULT_SETTINGS_PATH)
        else:
            self.settings = coerce_settings(settings)

    def build_layer_separated_text_units(self, candidate_groups):
        """Build line buckets and text units without crossing layer boundaries.

        Args:
            candidate_groups: Normalized candidate groups from N01/ScribeMap.

        Returns:
            Tuple of combined line buckets, text units, and blocked evidence.
        """
        groups_by_layer = {}

        for group in candidate_groups:
            layer = group.get("layer", "legacy")
            groups_by_layer.setdefault(layer, []).append(group)

        line_buckets = []
        text_units = []
        blocked_merge_evidence = []
        next_line_bucket_id = 1
        next_text_unit_id = 1

        for layer in sorted(groups_by_layer.keys()):
            layer_buckets = build_line_buckets(
                candidate_groups=groups_by_layer[layer],
                settings=self.settings,
            )

            for bucket in layer_buckets:
                bucket["line_bucket_id"] = next_line_bucket_id
                bucket["layer"] = layer
                next_line_bucket_id += 1

                bucket_units, next_text_unit_id, bucket_blocked = build_text_units_from_line_bucket(
                    line_bucket=bucket,
                    starting_text_unit_id=next_text_unit_id,
                    settings=self.settings,
                )

                for unit in bucket_units:
                    unit["layer"] = layer
                    unit["source_layers"] = [layer]

                line_buckets.append(bucket)
                text_units.extend(bucket_units)
                blocked_merge_evidence.extend(bucket_blocked)

        text_units.sort(
            key=lambda unit: (
                unit["bbox"]["y1"],
                unit["bbox"]["x1"],
                unit.get("layer", "legacy"),
                unit["text_unit_id"],
            )
        )

        return line_buckets, text_units, blocked_merge_evidence

    def refine_document(self, classified_groups_json_path, output_path=None):
        """Run N02 refinement for one document.

        Args:
            classified_groups_json_path: Path to N01/classified groups JSON.
            output_path: Optional output path for refined metadata JSON.

        Returns:
            Full refinement result dictionary.
        """
        payload = load_json(classified_groups_json_path)
        document_id = payload.get("document_id", "document")

        input_group_mode = getattr(self.settings, "input_group_mode", "legacy_groups")

        if input_group_mode == "scribemap_2_layers":
            scribemap_result_path = payload.get("scribemap_result_path")

            if not scribemap_result_path:
                raise ValueError(
                    "input_group_mode='scribemap_2_layers' requires payload['scribemap_result_path']"
                )

            scribemap_result = load_json(scribemap_result_path)

            raw_groups = collect_scribemap_2_layer_groups(
                scribemap_result=scribemap_result,
                layers_to_refine=getattr(self.settings, "scribemap_2_layers_to_refine", ("blue",)),
            )
        else:
            raw_groups = payload.get("classified_groups", [])

        refinement_image_path = resolve_refinement_image_path(payload)
        gray_image = load_gray_image(refinement_image_path)

        normalized_groups = normalize_source_groups(raw_groups)
        candidate_groups, _ = filter_early_artifacts(
            normalized_groups=normalized_groups,
            settings=self.settings,
        )
        line_buckets, text_units, blocked_merge_evidence = self.build_layer_separated_text_units(
            candidate_groups=candidate_groups,
        )
        text_units, border_profile_attachment_evidence, blocked_border_profile_attachment_evidence = (
            attach_border_profile_fragments(
                text_units=text_units,
                gray_image=gray_image,
                settings=self.settings,
            )
        )

        refined_crops_dir = None

        if output_path is not None:
            refined_crops_dir = prepare_refined_crops_dir(output_path)

        refined_groups = []

        for text_unit in text_units:
            final_bbox = pad_bbox(
                bbox=text_unit["bbox"],
                padding_px=self.settings.crop_padding_px,
                image_shape=gray_image.shape,
            )
            quality_result = evaluate_crop_quality(
                gray_image=gray_image,
                bbox=final_bbox,
                settings=self.settings,
            )
            crop_path = None

            if refined_crops_dir is not None:
                crop_path = save_crop(
                    gray_image=gray_image,
                    bbox=final_bbox,
                    crops_dir=refined_crops_dir,
                    text_unit_id=text_unit["text_unit_id"],
                    status=quality_result["status"],
                    layer=text_unit.get("layer"),
                )

            refined_groups.append(
                build_refined_text_unit_record(
                    text_unit=text_unit,
                    final_bbox=final_bbox,
                    quality_result=quality_result,
                    crop_path=crop_path,
                )
            )

        summary = build_refinement_summary(refined_groups)
        layer_summaries = build_layer_summary(refined_groups)
        debug_preview_path = None
        layer_debug_outputs = {}

        if output_path is not None and self.settings.debug_preview_enabled:
            debug_preview_path = build_debug_preview_path(output_path, document_id)
            render_debug_preview(
                image_path=refinement_image_path,
                refined_groups=refined_groups,
                output_path=debug_preview_path,
            )
            layer_debug_outputs = render_layer_debug_outputs(
                image_path=refinement_image_path,
                refined_groups=refined_groups,
                output_dir=os.path.dirname(debug_preview_path),
                document_id=document_id,
            )

        result = {
            "document_id": document_id,
            "input_bw_image_path": payload.get("input_bw_image_path"),
            "refinement_image_path": refinement_image_path,
            "coordinate_space": "scribemap_prepared_image",
            "settings": self.settings.to_dict(),
            "input_group_mode": input_group_mode,
            "refined_layers": list(getattr(self.settings, "scribemap_2_layers_to_refine", [])),
            "original_group_count": len(raw_groups),
            "normalized_group_count": len(normalized_groups),
            "candidate_group_count": len(candidate_groups),
            "line_bucket_count": len(line_buckets),
            "text_unit_group_count": len(text_units),
            "group_count": len(refined_groups),
            "merged_group_count": max(len(candidate_groups) - len(text_units), 0),
            "refined_crops_dir": refined_crops_dir,
            "debug_preview_path": debug_preview_path,
            "layer_debug_outputs": layer_debug_outputs,
            "summary": summary,
            "layer_summaries": layer_summaries,
            "line_buckets": line_buckets,
            "text_units": text_units,
            "blocked_merge_evidence": blocked_merge_evidence,
            "border_profile_attachment_evidence": border_profile_attachment_evidence,
            "blocked_border_profile_attachment_evidence": blocked_border_profile_attachment_evidence,
            "refined_groups": refined_groups,
        }

        if output_path is not None:
            save_json(result, output_path)

            for layer, layer_groups in groups_by_layer(refined_groups).items():
                layer_result = {
                    "document_id": document_id,
                    "layer": layer,
                    "input_group_mode": input_group_mode,
                    "refinement_image_path": refinement_image_path,
                    "coordinate_space": "scribemap_prepared_image",
                    "group_count": len(layer_groups),
                    "summary": layer_summaries.get(layer, {}),
                    "debug_outputs": layer_debug_outputs.get(layer, {}),
                    "refined_crops_dir": refined_crops_dir,
                    "refined_groups": layer_groups,
                }
                save_json(
                    layer_result,
                    build_layer_json_path(output_path, document_id, layer),
                )

        return result


__all__ = ["CropRefiner", "RefinerSettings", "coerce_settings", "load_refiner_settings"]
