"""Regression tests for conservative N00 color-edge recovery."""

import os
import sys
import unittest

import numpy as np


NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if NODE_DIR not in sys.path:
    sys.path.insert(0, NODE_DIR)

from file_preparation_scribemap_masks import create_basic_color_ink_masks


class SeededEdgeRecoveryTests(unittest.TestCase):
    """Verify that faded ink grows only from trusted same-color seeds."""

    def test_adjacent_faded_blue_is_recovered_but_isolated_blue_is_not(self):
        image = np.full((7, 12, 3), 255, dtype=np.uint8)

        image[3, 3] = (180, 60, 60)
        image[3, 4] = (230, 200, 200)
        image[3, 9] = (230, 200, 200)

        masks = create_basic_color_ink_masks(
            image,
            {"weak_color_value_max": 210},
        )
        blue = masks["blue_ink_mask"]
        recovered = masks["seeded_blue_edge_mask"]

        self.assertEqual(255, blue[3, 3])
        self.assertEqual(255, blue[3, 4])
        self.assertEqual(255, recovered[3, 4])
        self.assertEqual(0, blue[3, 9])
        self.assertEqual(0, recovered[3, 9])
        self.assertEqual(0, masks["exclusive_overlap_pixels"])

    def test_recovery_can_be_disabled(self):
        image = np.full((7, 8, 3), 255, dtype=np.uint8)
        image[3, 3] = (180, 60, 60)
        image[3, 4] = (230, 200, 200)

        masks = create_basic_color_ink_masks(
            image,
            {
                "weak_color_value_max": 210,
                "seeded_edge_recovery_enabled": False,
            },
        )

        self.assertEqual(255, masks["blue_ink_mask"][3, 3])
        self.assertEqual(0, masks["blue_ink_mask"][3, 4])
        self.assertEqual(0, masks["seeded_blue_edge_mask"][3, 4])


if __name__ == "__main__":
    unittest.main()
