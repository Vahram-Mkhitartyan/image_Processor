"""Regression tests for ScribeTrace v4 export and grouped splitting."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.Cyber_Lin_Kuei_Assembly import scribetrace_random_forest as trainer


class ScribeTraceV4TrainingTests(unittest.TestCase):
    """Protect compressed datasets and source-level split isolation."""

    def test_grouped_split_never_leaks_variants(self):
        labels = []
        source_ids = []
        for class_id in range(3):
            for source_index in range(10):
                source_id = f"{class_id}/glyph_{source_index}.png"
                for _ in range(5):
                    labels.append(class_id)
                    source_ids.append(source_id)

        with (
            patch.object(trainer, "RANDOM_SEED", 42),
            patch.object(trainer, "VALIDATION_RATIO", 0.1),
            patch.object(trainer, "TEST_RATIO", 0.1),
        ):
            indices, sources = trainer.grouped_stratified_split(
                np.asarray(labels, dtype=np.int16),
                source_ids,
            )

        self.assertFalse(sources["train"] & sources["validation"])
        self.assertFalse(sources["train"] & sources["test"])
        self.assertFalse(sources["validation"] & sources["test"])
        self.assertEqual(
            sum(len(value) for value in indices.values()),
            len(labels),
        )
        for split_name, split_indices in indices.items():
            split_source_ids = {source_ids[int(index)] for index in split_indices}
            self.assertEqual(split_source_ids, sources[split_name])

    def test_compressed_jsonl_loads_training_arrays(self):
        feature_names = ["a", "b"]
        rows = [
            {
                "class_id": class_id,
                "image_path": f"/tmp/{class_id}_{index}.png",
                "source_id": f"{class_id}/{index}.png",
                "sample_kind": "clean",
                "feature_names": feature_names,
                "vector": [float(class_id), float(index)],
            }
            for class_id in range(2)
            for index in range(3)
        ]

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "dataset.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps(row) + "\n")

            with patch.object(trainer, "NUM_CLASSES", 2):
                (
                    features,
                    labels,
                    image_paths,
                    source_ids,
                    sample_kinds,
                    loaded_names,
                ) = trainer.load_training_arrays(path)

        self.assertEqual(features.shape, (6, 2))
        self.assertEqual(labels.tolist(), [0, 0, 0, 1, 1, 1])
        self.assertEqual(len(image_paths), 6)
        self.assertEqual(len(source_ids), 6)
        self.assertEqual(set(sample_kinds), {"clean"})
        self.assertEqual(loaded_names, feature_names)


if __name__ == "__main__":
    unittest.main()
