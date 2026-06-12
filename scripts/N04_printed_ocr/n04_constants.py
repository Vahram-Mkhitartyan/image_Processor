"""Constants for N04 printed OCR."""

BASE_DIR = "/home/vahram/Desktop/image_Processor"

NODE_NAME = "N04_printed_ocr"
NODE_VERSION = "0.1.0"

OCR_ENGINE_NAME = "tesseract"
OCR_ENGINE_VERSION = "5.5.2"

PRIMARY_OCR_LANGUAGE = "hye-calfa-n"
FALLBACK_OCR_LANGUAGE = "hye"
DEFAULT_OCR_PSM = 6

OCR_CANDIDATE_CONFIGS = [
    {
        "engine": "tesseract",
        "language": PRIMARY_OCR_LANGUAGE,
        "psm": DEFAULT_OCR_PSM,
        "role": "primary"
    },
    {
        "engine": "tesseract",
        "language": FALLBACK_OCR_LANGUAGE,
        "psm": DEFAULT_OCR_PSM,
        "role": "fallback"
    }
]

PRINTED_VISUAL_CLASSES = {
    "printed_only",
    "mixed"
}
