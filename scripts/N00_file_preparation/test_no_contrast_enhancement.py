"""Regression tests for the enhancement-free N00 grayscale path."""

import os
import sys
import tempfile
import unittest

import cv2
import numpy as np


NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if NODE_DIR not in sys.path:
    sys.path.insert(0, NODE_DIR)

from file_preparation import prepare_file
from image_preprocessor import ImagePreprocessor


class NoContrastEnhancementTests(unittest.TestCase):
    """Protect direct denoised-to-thresholded processing."""

    def test_threshold_uses_denoised_image_and_removes_retired_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            input_path = os.path.join(temporary_dir, "input.png")
            output_dir = os.path.join(temporary_dir, "output")
            full_images_dir = os.path.join(output_dir, "full_images")
            os.makedirs(full_images_dir)

            image = np.full((80, 120, 3), 245, dtype=np.uint8)
            cv2.putText(
                image,
                "ABC",
                (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (35, 35, 35),
                2,
                cv2.LINE_AA,
            )
            self.assertTrue(cv2.imwrite(input_path, image))

            retired_path = os.path.join(
                full_images_dir,
                "04_enhanced.jpeg",
            )
            self.assertTrue(cv2.imwrite(retired_path, image))

            state = prepare_file(
                input_path=input_path,
                output_dir=output_dir,
                steps=[
                    "load_image",
                    "rotate_major",
                    "convert_to_grayscale",
                    "denoise_image",
                    "threshold_image",
                    "save_outputs",
                ],
            )

            expected = ImagePreprocessor(
                state["settings"]
            ).threshold_image(state["images"]["denoised"])

            self.assertTrue(
                np.array_equal(expected, state["images"]["thresholded"])
            )
            self.assertNotIn("enhanced", state["images"])
            self.assertNotIn("enhanced", state["artifacts"])
            self.assertNotIn("improve_contrast", state["metadata"]["steps_completed"])
            self.assertNotIn("clahe_clip_limit", state["settings"])
            self.assertEqual(14, state["settings"]["threshold_c"])
            self.assertFalse(os.path.exists(retired_path))
            self.assertTrue(
                state["artifacts"]["thresholded"].endswith(
                    "04_thresholded.jpeg"
                )
            )


if __name__ == "__main__":
    unittest.main()
