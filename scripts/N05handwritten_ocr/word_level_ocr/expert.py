"""Word-level Armenian OCR expert contract."""

EXPERT_NAME = "word_level_ocr"


def get_expert_manifest(settings=None):
    """Describe the word-level OCR expert without loading future models.

    Args:
        settings: Optional expert settings dictionary.

    Returns:
        Expert capability and implementation-status metadata.
    """
    settings = settings or {}
    return {
        "expert_name": EXPERT_NAME,
        "display_name": "Word-Level OCR",
        "enabled": bool(settings.get("enabled", False)),
        "implemented": False,
        "status": "not_implemented",
        "unit_level": "word",
    }


def recognize(crop_path, context=None, settings=None):
    """Return a non-attempted result until the word model is implemented.

    Args:
        crop_path: Path to the OCR-ready word or text-unit crop.
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
