"""Focused tests for red-correction replacement routing."""

import os
import sys
import unittest

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NODE_DIR = os.path.join(PROJECT_ROOT, "scripts", "N02_crop_refiner")
if NODE_DIR not in sys.path:
    sys.path.insert(0, NODE_DIR)

from crop_refiner import RefinerSettings, apply_red_correction_routing


def make_group(layer, group_id, bbox):
    """Build the minimal normalized source-group record required by N02."""
    width = bbox["x2"] - bbox["x1"]
    height = bbox["y2"] - bbox["y1"]
    return {
        "source_group_id": f"{layer}_{group_id:04d}",
        "source_layer_group_id": group_id,
        "layer": layer,
        "bbox": dict(bbox),
        "width": width,
        "height": height,
        "area": width * height,
        "role_guess": "probable_handwriting",
        "recommended_next_node": "N05_handwritten_ocr",
        "minos_required": True,
        "minos_mode": "handwriting_audit",
        "is_final_text_candidate": True,
        "preserve_as_evidence": False,
    }


class CorrectionRoutingTests(unittest.TestCase):
    """Verify that strike marks suppress blue and promote nearby red text."""

    def test_crossed_blue_is_suppressed_and_nearby_red_text_is_promoted(self):
        blue = make_group(
            "blue",
            1,
            {"x1": 10, "y1": 10, "x2": 50, "y2": 30},
        )
        red_strike = make_group(
            "red",
            1,
            {"x1": 15, "y1": 19, "x2": 45, "y2": 21},
        )
        red_replacement = make_group(
            "red",
            2,
            {"x1": 12, "y1": 0, "x2": 42, "y2": 9},
        )
        red_mask = np.zeros((40, 60), dtype=np.uint8)
        red_mask[19:21, 15:45] = 255
        red_mask[1:8, 13:41] = 255

        summary = apply_red_correction_routing(
            source_groups=[blue, red_strike, red_replacement],
            red_mask=red_mask,
            settings=RefinerSettings(),
        )

        self.assertFalse(blue["minos_required"])
        self.assertFalse(blue["is_final_text_candidate"])
        self.assertEqual("deleted_original", blue["correction_role"])
        self.assertTrue(red_replacement["force_handwritten_ocr"])
        self.assertEqual(
            "replacement_text",
            red_replacement["correction_role"],
        )
        self.assertFalse(red_strike.get("force_handwritten_ocr", False))
        self.assertEqual(1, summary["suppressed_blue_count"])
        self.assertEqual(1, summary["promoted_red_count"])

    def test_tiny_red_speck_does_not_suppress_blue(self):
        blue = make_group(
            "blue",
            1,
            {"x1": 10, "y1": 10, "x2": 50, "y2": 30},
        )
        red_speck = make_group(
            "red",
            1,
            {"x1": 25, "y1": 18, "x2": 27, "y2": 20},
        )
        red_mask = np.zeros((40, 60), dtype=np.uint8)
        red_mask[18:20, 25:27] = 255

        summary = apply_red_correction_routing(
            source_groups=[blue, red_speck],
            red_mask=red_mask,
            settings=RefinerSettings(),
        )

        self.assertTrue(blue["minos_required"])
        self.assertTrue(blue["is_final_text_candidate"])
        self.assertEqual(0, summary["suppressed_blue_count"])
        self.assertEqual(0, summary["promoted_red_count"])


if __name__ == "__main__":
    unittest.main()
