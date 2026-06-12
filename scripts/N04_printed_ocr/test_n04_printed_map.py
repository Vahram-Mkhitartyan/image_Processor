from printed_ocr import build_printed_text_map

BASE_DIR = "/home/vahram/Desktop/image_Processor"

document_id = "test"

visual_routes_path = (
    f"{BASE_DIR}/temp_processing/{document_id}/"
    f"n03_visual_classification/metadata/"
    f"{document_id}_n03_visual_classification_routes.json"
)

output_dir = (
    f"{BASE_DIR}/temp_processing/{document_id}/"
    f"n04_printed_ocr"
)

build_printed_text_map(
    visual_routes_path=visual_routes_path,
    output_dir=output_dir
)