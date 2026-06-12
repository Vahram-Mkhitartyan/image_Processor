"""
One-image ScribeTrace RF executor.

Run from project root:
    cd /home/vahram/Desktop/image_Processor
    .venv/bin/python run_scribetrace_rf_one_image.py

This tests integration only. The RF is letter-level, so a full word crop will
still produce one letter-level top-k list for the whole unit.
"""

import json
import os
import shutil
import sys
from pathlib import Path

BASE_DIR = Path("/home/vahram/Desktop/image_Processor")
IMAGE_PATH = Path(
    "/home/vahram/Desktop/image_Processor/temp_processing/test_1/"
    "n02_crop_refiner/crops/blue/analysis_mask/"
    "blue_0005_blue_0005_analysis_mask.png"
)



EXPERT_DIR = BASE_DIR / "scripts" / "N05handwritten_ocr" / "scribetrace"
OUTPUT_DIR = BASE_DIR / "temp_processing" / "test_1" / "n05_handwritten_ocr" / "scribetrace_rf_single_test"

SETTINGS = {
    "enabled": True,
    "save_debug": True,
    "save_json": True,
    "debug_draw_labels": False,
    "ink_threshold_mode": "binary",
    "fixed_threshold_value": 128,
    "minimum_ink_pixels": 4,
    "maximum_component_count_for_full_trace": 50,
    "minimum_trace_path_points": 4,
    "short_path_merge_max_angle_degrees": 35.0,
    "short_path_tangent_points": 3,
    "short_path_merge_min_advantage_degrees": 10.0,
    "local_extrema_min_prominence": 2,
    "local_extrema_min_spacing": 3,
}


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Test image not found: {IMAGE_PATH}")

    if not EXPERT_DIR.exists():
        raise FileNotFoundError(f"ScribeTrace expert folder not found: {EXPERT_DIR}")

    sys.path.insert(0, str(EXPERT_DIR.parent))

    from scribetrace.expert import recognize  # noqa: WPS433

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    context = {
        "document_id": "test_1",
        "text_unit_id": "blue_0017_rf_single_test",
        "source_group_id": "blue_0017",
        "source_layer_group_id": "blue_0017",
        "layer": "blue",
        "scribetrace_mask_crop_path": str(IMAGE_PATH),
        "scribetrace_visual_crop_path": str(IMAGE_PATH),
        "scribetrace_context_crop_path": None,
        "scribetrace_output_dir": str(OUTPUT_DIR),
        "document_bbox": None,
        "final_bbox": None,
    }

    result = recognize(
        crop_path=str(IMAGE_PATH),
        context=context,
        settings=SETTINGS,
    )

    evidence = result.get("evidence", {})
    rf_candidates = evidence.get("rf_letter_candidates_for_unit", [])

    print("\n=== SCRIBETRACE STATUS ===")
    print("status:", result.get("status"))
    print("attempted:", result.get("attempted"))
    print("error:", result.get("error"))

    print("\n=== TRACE COUNTS ===")
    print("components:", evidence.get("component_count"))
    print("paths:", evidence.get("path_count"))
    print("landmarks:", evidence.get("landmark_count"))
    print("ink holes:", evidence.get("ink_hole_count"))

    feature_vector = evidence.get("feature_vector") or {}
    print("\n=== FEATURE VECTOR ===")
    print("feature count:", len(feature_vector.get("feature_names", [])))

    print("\n=== RF LETTER CANDIDATES FOR WHOLE UNIT ===")
    if rf_candidates:
        for candidate in rf_candidates:
            print(
                f"{candidate['rank']}. {candidate['label']} "
                f"confidence={candidate['confidence']:.4f} "
                f"class_id={candidate['class_id']}"
            )
    else:
        print("No RF candidates returned.")
        if "rf_error" in evidence:
            print("rf_error:", evidence["rf_error"])

    output_json = OUTPUT_DIR / "single_image_scribetrace_rf_result.json"
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print("\n=== OUTPUTS ===")
    print("result json:", output_json)
    print("scribetrace json:", evidence.get("result_json_path"))
    print("debug dir:", OUTPUT_DIR / "debug")


if __name__ == "__main__":
    main()
