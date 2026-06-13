"""Regression tests for conservative N00 color-edge recovery."""

import os
import sys
import unittest

import numpy as np


NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if NODE_DIR not in sys.path:
    sys.path.insert(0, NODE_DIR)

from file_preparation_scribemap_masks import (
    create_basic_color_ink_masks,
    create_cross_color_continuity_masks,
)
from file_preparation import create_initial_state


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

    def test_pale_edge_requires_seed_and_final_layers_remain_exclusive(self):
        image = np.full((7, 12, 3), 255, dtype=np.uint8)

        image[3, 3] = (180, 60, 60)
        image[3, 4] = (240, 218, 218)
        image[3, 9] = (240, 218, 218)
        image[5, 5] = (60, 60, 180)

        masks = create_basic_color_ink_masks(image, {})

        self.assertEqual(255, masks["blue_ink_mask"][3, 4])
        self.assertEqual(255, masks["seeded_blue_edge_mask"][3, 4])
        self.assertEqual(0, masks["blue_ink_mask"][3, 9])

        layer_count = sum(
            (masks[key] > 0).astype(np.uint8)
            for key in (
                "red_ink_mask",
                "blue_ink_mask",
                "green_ink_mask",
                "unknown_color_ink_mask",
                "black_ink_mask",
            )
        )
        self.assertLessEqual(int(layer_count.max()), 1)
        self.assertEqual(0, masks["exclusive_overlap_pixels"])

    def test_weak_dark_edge_still_requires_direct_seed_contact(self):
        image = np.full((7, 12, 3), 255, dtype=np.uint8)

        image[3, 3] = (180, 60, 60)
        image[3, 4] = (80, 74, 74)
        image[3, 9] = (80, 74, 74)

        masks = create_basic_color_ink_masks(image, {})

        self.assertEqual(255, masks["blue_ink_mask"][3, 4])
        self.assertEqual(255, masks["seeded_blue_edge_mask"][3, 4])
        self.assertEqual(0, masks["blue_ink_mask"][3, 9])
        self.assertEqual(0, masks["exclusive_overlap_pixels"])

    def test_red_crossing_repairs_blue_continuity_without_changing_semantics(self):
        red = np.zeros((11, 11), dtype=np.uint8)
        blue = np.zeros((11, 11), dtype=np.uint8)

        blue[5, 1:10] = 255
        blue[5, 4:7] = 0
        red[2:9, 4:7] = 255

        repaired = create_cross_color_continuity_masks(
            red_mask=red,
            blue_mask=blue,
            settings={"cross_color_bridge_radius_px": 4},
        )

        self.assertTrue(np.all(repaired["blue_continuity_mask"][5, 1:10] > 0))
        self.assertEqual(3, repaired["blue_borrowed_bridge_pixels"])
        self.assertEqual(0, np.count_nonzero(blue[5, 4:7]))
        self.assertTrue(np.all(red[5, 4:7] > 0))

    def test_one_sided_or_parallel_color_pixels_are_not_borrowed(self):
        red = np.zeros((12, 12), dtype=np.uint8)
        blue = np.zeros((12, 12), dtype=np.uint8)

        blue[6, 2:6] = 255
        red[6, 6:9] = 255
        red[4, 2:10] = 255

        repaired = create_cross_color_continuity_masks(
            red_mask=red,
            blue_mask=blue,
            settings={"cross_color_bridge_radius_px": 4},
        )

        self.assertEqual(0, repaired["blue_borrowed_bridge_pixels"])
        self.assertTrue(
            np.array_equal(blue, repaired["blue_continuity_mask"])
        )

    def test_continuity_repair_can_be_disabled(self):
        red = np.zeros((7, 7), dtype=np.uint8)
        blue = np.zeros((7, 7), dtype=np.uint8)
        blue[3, 1:3] = 255
        blue[3, 4:6] = 255
        red[1:6, 3] = 255

        repaired = create_cross_color_continuity_masks(
            red_mask=red,
            blue_mask=blue,
            settings={"cross_color_continuity_enabled": False},
        )

        self.assertTrue(np.array_equal(blue, repaired["blue_continuity_mask"]))
        self.assertTrue(np.array_equal(red, repaired["red_continuity_mask"]))
        self.assertEqual(0, repaired["blue_borrowed_bridge_pixels"])

    def test_pipeline_state_preserves_continuity_settings(self):
        state = create_initial_state(
            input_path="input.png",
            output_dir="output",
            settings={
                "cross_color_continuity_enabled": False,
                "cross_color_bridge_radius_px": 6,
            },
        )

        self.assertFalse(
            state["settings"]["cross_color_continuity_enabled"]
        )
        self.assertEqual(
            6,
            state["settings"]["cross_color_bridge_radius_px"],
        )


if __name__ == "__main__":
    unittest.main()
