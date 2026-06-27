"""Armenian word normalization helpers for N06."""

from __future__ import annotations

import unicodedata


ARMENIAN_UPPER = "ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖ"
ARMENIAN_LOWER = "աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆ"
ARMENIAN_EXTRA = "և"
ARMENIAN_LETTERS = set(ARMENIAN_UPPER + ARMENIAN_LOWER + ARMENIAN_EXTRA)

PUNCTUATION_TO_STRIP = " \t\r\n.,:;!?()[]{}«»\"'`՛՜՝՞։֊-–—_"


def normalize_armenian_word(text: str, settings: dict | None = None) -> str:
    """Normalize one Armenian word candidate without pretending it is final."""

    settings = settings or {}
    value = unicodedata.normalize("NFC", str(text or "")).strip()
    value = value.strip(PUNCTUATION_TO_STRIP)
    if not settings.get("keep_hyphen", False):
        value = value.replace("-", "").replace("֊", "")
    if settings.get("normalize_armenian_ligature", False):
        value = value.replace("և", "եվ")
    if settings.get("lowercase", True):
        value = value.lower()
    return value


def armenian_character_report(text: str) -> dict:
    """Return simple character validity statistics for a normalized token."""

    chars = list(str(text or ""))
    if not chars:
        return {
            "length": 0,
            "armenian_character_count": 0,
            "unknown_character_count": 0,
            "unknown_character_ratio": 0.0,
            "unknown_characters": [],
        }
    unknown = [char for char in chars if char not in ARMENIAN_LETTERS]
    return {
        "length": len(chars),
        "armenian_character_count": len(chars) - len(unknown),
        "unknown_character_count": len(unknown),
        "unknown_character_ratio": len(unknown) / max(1, len(chars)),
        "unknown_characters": sorted(set(unknown)),
    }
