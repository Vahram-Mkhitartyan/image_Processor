"""Shared paths and constants for the pipeline batch controller."""

import os
import sys

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_CONTROL_DIR = os.path.dirname(MODULE_DIR)
SCRIPTS_DIR = os.path.dirname(PIPELINE_CONTROL_DIR)

BASE_DIR = "/home/vahram/Desktop/image_Processor"

FILE_PREPARATION_DIR = f"{SCRIPTS_DIR}/N00_file_preparation"
SCRIBEMAP_DIR = f"{SCRIPTS_DIR}/N01_scribemap"
CROP_REFINER_DIR = f"{SCRIPTS_DIR}/N02_crop_refiner"
VISUAL_CLASSIFICATION_DIR = f"{SCRIPTS_DIR}/N03_visual_classification_router"
PRINTED_OCR_DIR = f"{SCRIPTS_DIR}/N04_printed_ocr"
N05_DIR = f"{SCRIPTS_DIR}/N05handwritten_ocr"

CROP_REFINER_PATH = f"{CROP_REFINER_DIR}/crop_refiner.py"
VISUAL_CLASSIFICATION_PATH = f"{VISUAL_CLASSIFICATION_DIR}/classifier.py"
PRINTED_OCR_PATH = f"{PRINTED_OCR_DIR}/printed_ocr.py"
N05_ORCHESTRATOR_PATH = f"{N05_DIR}/expert_orchestrator.py"

VISUAL_CLASSIFICATION_SETTINGS_PATH = (
    f"{VISUAL_CLASSIFICATION_DIR}/visual_classification_settings.json"
)
PRINTED_OCR_SETTINGS_PATH = f"{PRINTED_OCR_DIR}/settings.json"
N05_SETTINGS_PATH = f"{N05_DIR}/settings.json"
MINOS_MODEL_PATH = f"{BASE_DIR}/models/minos_v2_0_best.keras"

INPUT_DOCUMENTS_DIR = f"{BASE_DIR}/handwritten_text"
TEMP_PROCESSING_DIR = f"{BASE_DIR}/temp_processing"
FINAL_RESULTS_DIR = f"{BASE_DIR}/final_results"
FAILED_RESULTS_DIR = f"{BASE_DIR}/failed_results"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

# Let phase modules import node folders without exposing this package name.
for node_dir in [SCRIPTS_DIR, FILE_PREPARATION_DIR, SCRIBEMAP_DIR, N05_DIR]:
    if node_dir not in sys.path:
        sys.path.insert(0, node_dir)
