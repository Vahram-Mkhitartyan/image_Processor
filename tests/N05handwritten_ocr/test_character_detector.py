"""Regression tests for the pixel-CNN expert contract."""

import unittest
from pathlib import Path

from scripts.N05handwritten_ocr.character_detector import (
    get_expert_manifest,
    recognize,
)


ROOT = Path(__file__).resolve().parents[2]


class CharacterDetectorTests(unittest.TestCase):
    """Protect model loading, polarity compatibility, and candidate JSON."""

    def test_manifest_reports_implemented_but_disabled_by_default(self):
        manifest = get_expert_manifest()
        self.assertTrue(manifest["implemented"])
        self.assertFalse(manifest["enabled"])
        self.assertEqual("character", manifest["unit_level"])

    def test_disabled_expert_does_not_load_or_attempt_model(self):
        result = recognize("missing.png", settings={"enabled": False})
        self.assertEqual("disabled", result["status"])
        self.assertFalse(result["attempted"])
        self.assertEqual([], result["candidates"])

    def test_real_class_eight_glyph_emits_compatible_top_five(self):
        result = recognize(
            ROOT / "Matenadata" / "8" / "3.png",
            settings={"enabled": True, "device": "cpu", "top_k": 5},
        )
        self.assertEqual("completed", result["status"])
        self.assertTrue(result["attempted"])
        self.assertEqual(5, len(result["candidates"]))
        self.assertEqual(8, result["candidates"][0]["class_id"])
        self.assertEqual("Թ", result["candidates"][0]["label"])
        self.assertEqual(
            "legacy_raw_invert",
            result["evidence"]["preprocessing"]["polarity_mode"],
        )
        self.assertTrue(
            all(
                candidate["provenance"] == "character_detector_cnn"
                for candidate in result["candidates"]
            )
        )

    def test_missing_enabled_input_returns_failed_contract(self):
        result = recognize("missing.png", settings={"enabled": True})
        self.assertEqual("failed", result["status"])
        self.assertTrue(result["attempted"])
        self.assertIn("does not exist", result["error"])


if __name__ == "__main__":
    unittest.main()
