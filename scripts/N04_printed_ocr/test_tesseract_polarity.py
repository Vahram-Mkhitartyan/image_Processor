"""Regression tests for N04 Tesseract input polarity and routing."""

import os
import sys
import unittest

import cv2
import numpy as np


NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if NODE_DIR not in sys.path:
    sys.path.insert(0, NODE_DIR)

from n04_crops import prepare_crop_for_tesseract
from n04_routing import get_best_crop_path_for_printed_ocr


class TesseractPolarityTests(unittest.TestCase):
    """Protect dark-text-on-white Tesseract preparation."""

    def test_normal_and_inverted_sources_produce_normal_polarity(self):
        normal = np.full((20, 40), 255, dtype=np.uint8)
        cv2.rectangle(normal, (12, 5), (27, 14), 0, thickness=-1)
        inverted = cv2.bitwise_not(normal)

        prepared_normal = prepare_crop_for_tesseract(
            normal,
            scale=1,
            border=4,
        )
        prepared_inverted = prepare_crop_for_tesseract(
            inverted,
            scale=1,
            border=4,
        )

        self.assertTrue(np.array_equal(prepared_normal, prepared_inverted))
        self.assertEqual(255, int(prepared_normal[0, 0]))
        self.assertEqual(0, int(prepared_normal[14, 24]))

    def test_analysis_mask_is_never_selected_for_printed_ocr(self):
        route = {
            "classification_crop_path": "/tmp/full_text.png",
            "analysis_mask_crop_path": "/tmp/inverted_analysis_mask.png",
        }
        self.assertEqual(
            "/tmp/full_text.png",
            get_best_crop_path_for_printed_ocr(route),
        )

        mask_only_route = {
            "analysis_mask_crop_path": "/tmp/inverted_analysis_mask.png",
        }
        self.assertIsNone(
            get_best_crop_path_for_printed_ocr(mask_only_route)
        )


if __name__ == "__main__":
    unittest.main()
