"""Tesseract-based OCR expert contract."""

EXPERT_NAME = "tesseract_ocr"


def get_expert_manifest(settings=None):
    """Describe the Tesseract expert without loading its engine.

    Args:
        settings: Optional expert settings dictionary.

    Returns:
        Expert capability and implementation-status metadata.
    """
    settings = settings or {}
    return {
        "expert_name": EXPERT_NAME,
        "display_name": "Tesseract OCR",
        "enabled": bool(settings.get("enabled", False)),
        "implemented": False,
        "status": "not_implemented",
        "unit_level": "text_unit",
    }


def recognize(crop_path, context=None, settings=None):
    """Return a non-attempted result until the Tesseract adapter is built.

    Args:
        crop_path: Path to the OCR-ready crop.
        context: Optional document and routing evidence.
        settings: Optional expert settings dictionary.

    Returns:
        Standard expert-result dictionary.
    """
    return {
        "expert_name": EXPERT_NAME,
        "attempted": False,
        "status": "not_implemented",
        "crop_path": crop_path,
        "candidates": [],
        "error": None,
    }
