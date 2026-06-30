"""N04 printed OCR orchestrator.

This file keeps the public `build_printed_text_map()` entrypoint while helper
modules handle IO, routing, crop prep, OCR execution, and output records.
"""

import os
import sys

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from n04_constants import NODE_NAME, NODE_VERSION, OCR_ENGINE_NAME, OCR_ENGINE_VERSION
from n04_crops import copy_candidate_crop_to_n04, prepare_n04_crop_for_tesseract
from n04_io import (
    check_file_exists,
    create_output_folders,
    load_settings,
    reset_output_dir,
    save_json,
)
from n04_records import (
    build_printed_context_layer,
    build_printed_text_unit,
    print_summary,
    summarize_printed_text_map,
)
from n04_routing import get_visual_class, load_n03_visual_routes, select_printed_candidates


# MAIN PRINTED TEXT MAP BUILDER-------------------------------

def build_printed_text_map(
    visual_routes_path,
    output_dir,
    settings_path=None
):
    """
    Build the N04 printed text map for one document.

    Args:
        visual_routes_path: Path to N03 visual classification routes JSON.
        output_dir: N04 output folder for this document.
        settings_path: Optional N04 settings JSON.

    Returns:
        Result dictionary containing selected printed text units, summary, and
        metadata path.
    """
    check_file_exists(
        visual_routes_path,
        label="N03 visual routes JSON"
    )

    settings = load_settings(settings_path)

    reset_output = settings.get(
        "reset_output",
        True
    )

    if reset_output:
        reset_output_dir(output_dir)

    folders = create_output_folders(output_dir)

    n03_payload = load_n03_visual_routes(visual_routes_path)

    document_id = n03_payload.get(
        "document_id",
        "unknown_document"
    )

    route_records = n03_payload.get(
        "routes",
        []
    )

    selected_routes = select_printed_candidates(route_records)

    printed_text_units = []
    skipped_records = []
    failed_records = []

    for route_record in selected_routes:
        group_id = route_record.get("group_id")
        visual_class = get_visual_class(route_record)

        try:
            copied_crop_path = copy_candidate_crop_to_n04(
                route_record=route_record,
                folders=folders,
                settings=settings,
            )

            if copied_crop_path is None:
                skipped_records.append({
                    "document_id": document_id,
                    "group_id": group_id,
                    "visual_class": visual_class,
                    "reason": "no_usable_crop_path"
                })
                continue

            tesseract_ready_crop_path = prepare_n04_crop_for_tesseract(
                route_record=route_record,
                copied_crop_path=copied_crop_path,
                folders=folders
            )

            printed_text_unit = build_printed_text_unit(
                route_record=route_record,
                copied_crop_path=copied_crop_path,
                tesseract_ready_crop_path=tesseract_ready_crop_path,
                document_id=document_id
            )

            printed_text_units.append(printed_text_unit)

        except Exception as error:
            failed_records.append({
                "document_id": document_id,
                "group_id": group_id,
                "visual_class": visual_class,
                "error": str(error),
                "route_record": route_record
            })

    summary = summarize_printed_text_map(
        n03_payload=n03_payload,
        selected_routes=selected_routes,
        printed_text_units=printed_text_units,
        skipped_records=skipped_records,
        failed_records=failed_records
    )
    printed_context_layer = build_printed_context_layer(printed_text_units)
    summary["printed_context_token_count"] = printed_context_layer.get(
        "summary",
        {},
    ).get("token_count", 0)
    summary["black_mask_context_token_count"] = printed_context_layer.get(
        "summary",
        {},
    ).get("black_mask_token_count", 0)

    result = {
        "node": NODE_NAME,
        "node_version": NODE_VERSION,

        "document_id": document_id,

        "source_visual_routes_path": visual_routes_path,
        "output_dir": output_dir,
        "crops_dir": folders["crops"],
        "metadata_dir": folders["metadata"],

        "coordinate_space": "original_document_image",

        "ocr_engine": OCR_ENGINE_NAME,
        "ocr_engine_version": OCR_ENGINE_VERSION,

        "summary": summary,

        "printed_text_units": printed_text_units,
        "printed_context_layer": printed_context_layer,
        "skipped": skipped_records,
        "failed": failed_records
    }

    metadata_path = (
        f"{folders['metadata']}/"
        f"{document_id}_printed_text_map.json"
    )

    save_json(
        data=result,
        output_path=metadata_path
    )

    result["metadata_path"] = metadata_path

    print_summary(
        document_id=document_id,
        summary=summary,
        metadata_path=metadata_path
    )

    return result


__all__ = ["build_printed_text_map"]
