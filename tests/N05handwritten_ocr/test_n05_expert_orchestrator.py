"""Manual N05 expert-map integration check."""

from scripts.N05handwritten_ocr.expert_orchestrator import (
    build_handwriting_expert_map,
)

BASE_DIR = "/home/vahram/Desktop/image_Processor"


def main():
    """Build the N05 map from an existing N03 test artifact."""
    document_id = "test_1"
    visual_routes_path = (
        f"{BASE_DIR}/temp_processing/{document_id}/"
        f"n03_visual_classification/metadata/"
        f"{document_id}_n03_visual_classification_routes.json"
    )
    output_dir = (
        f"{BASE_DIR}/temp_processing/{document_id}/"
        f"n05_handwritten_ocr"
    )
    build_handwriting_expert_map(
        visual_routes_path=visual_routes_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
