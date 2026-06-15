"""Focused tests for Aristotel's deterministic, storage-safe modes."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.Cyber_Lin_Kuei_Assembly.Aristotel.recipes import (
    build_default_recipes,
)
from scripts.Cyber_Lin_Kuei_Assembly.Aristotel.runner import (
    AristotelRunner,
    FileCorrupter,
)


class AristotelTests(unittest.TestCase):
    """Verify reproducibility and explicit image-storage behavior."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_root = self.root / "input"
        self.output_root = self.root / "output"
        class_folder = self.input_root / "7"
        class_folder.mkdir(parents=True)

        image = np.full((24, 32), 255, dtype=np.uint8)
        cv2.line(image, (8, 4), (22, 20), 0, 3)
        self.source_path = class_folder / "glyph.png"
        self.assertTrue(cv2.imwrite(str(self.source_path), image))

    def tearDown(self):
        self.temporary_directory.cleanup()

    def build_runner(self):
        """Create a runner with stable default recipes."""
        return AristotelRunner(
            input_root=self.input_root,
            output_root=self.output_root,
            corrupter=FileCorrupter(build_default_recipes(), seed=42),
        )

    def test_same_coordinates_reproduce_identical_sample(self):
        runner = self.build_runner()
        first = list(runner.iter_samples(limit=1, epoch=3, variants=1))
        second = list(runner.iter_samples(limit=1, epoch=3, variants=1))

        self.assertEqual(
            [sample.sample_id for sample in first],
            [sample.sample_id for sample in second],
        )
        for left, right in zip(first, second):
            self.assertTrue(np.array_equal(left.image, right.image))

    def test_epoch_changes_sample_identity(self):
        runner = self.build_runner()
        first = next(runner.iter_samples(limit=1, epoch=0))
        second = next(runner.iter_samples(limit=1, epoch=1))
        self.assertNotEqual(first.sample_id, second.sample_id)
        self.assertNotEqual(first.seed, second.seed)

    def test_stream_mode_writes_nothing(self):
        result = self.build_runner().run(mode="stream", limit=1)
        self.assertEqual(result["produced_count"], 4)
        self.assertEqual(result["saved_image_count"], 0)
        self.assertIsNone(result["output_root"])
        self.assertFalse(self.output_root.exists())

    def test_manifest_regenerates_without_saved_images(self):
        runner = self.build_runner()
        result = runner.run(
            mode="manifest",
            limit=1,
            recipe_names=["light_cut"],
        )
        manifest_path = Path(result["manifest_path"])
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        regenerated = runner.regenerate(record)
        original = next(
            runner.iter_samples(
                limit=1,
                recipe_names=["light_cut"],
            )
        )

        self.assertTrue(np.array_equal(regenerated.image, original.image))
        self.assertEqual(regenerated.sample_id, original.sample_id)
        self.assertEqual(list((self.output_root / "images").rglob("*.png")), [])

    def test_preview_and_export_are_explicit(self):
        preview_runner = self.build_runner()
        preview = preview_runner.run(
            mode="preview",
            limit=1,
            preview_count=2,
        )
        self.assertEqual(preview["saved_image_count"], 2)
        self.assertEqual(
            len(list((self.output_root / "previews").glob("*.png"))),
            2,
        )

        export_root = self.root / "export"
        export_runner = AristotelRunner(
            input_root=self.input_root,
            output_root=export_root,
            corrupter=FileCorrupter(build_default_recipes(), seed=42),
        )
        exported = export_runner.run(
            mode="export",
            limit=1,
            recipe_names=["light_blur"],
        )
        self.assertEqual(exported["saved_image_count"], 1)
        self.assertEqual(len(list((export_root / "images").rglob("*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
