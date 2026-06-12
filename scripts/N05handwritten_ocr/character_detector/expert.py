"""Character-level Armenian recognition expert contract."""

EXPERT_NAME = "character_detector"


def get_expert_manifest(settings=None):
    """Describe the character detector and its migrated training assets.

    Args:
        settings: Optional expert settings dictionary.

    Returns:
        Expert capability and implementation-status metadata.
    """
    settings = settings or {}
    return {
        "expert_name": EXPERT_NAME,
        "display_name": "Character Detector",
        "enabled": bool(settings.get("enabled", False)),
        "implemented": False,
        "status": "awaiting_current_ocr_remake",
        "unit_level": "character",
        "label_map": "numeric_label_map.json",
    }


def recognize(crop_path, context=None, settings=None):
    """Return a non-attempted result until the current OCR is remade here.

    Args:
        crop_path: Path to the character or text-unit crop.
        context: Optional document and routing evidence.
        settings: Optional expert settings dictionary.

    Returns:
        Standard expert-result dictionary.
    """
    return {
        "expert_name": EXPERT_NAME,
        "attempted": False,
        "status": "awaiting_current_ocr_remake",
        "crop_path": crop_path,
        "candidates": [],
        "error": None,
    }
