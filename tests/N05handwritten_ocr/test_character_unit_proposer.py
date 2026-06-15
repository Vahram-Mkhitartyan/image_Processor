"""Regression tests for deterministic N05 character-unit proposals."""

import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from scripts.N05handwritten_ocr.character_unit_proposer import (
    propose_character_units,
)
from scripts.N05handwritten_ocr.expert_orchestrator import (
    build_handwriting_expert_map,
)


class CharacterUnitProposerTests(unittest.TestCase):
    """Protect whole-unit, topology-validated split, and recovery behavior."""

    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.root = self.temporary_dir.name
        self.folders = {
            "root": os.path.join(self.root, "n05"),
            "character_unit_proposer": os.path.join(
                self.root,
                "n05",
                "character_unit_proposer",
            ),
            "character_unit_segments": os.path.join(
                self.root,
                "n05",
                "character_unit_proposer",
                "segments",
            ),
        }

    def tearDown(self):
        self.temporary_dir.cleanup()

    def write_unit(self, mask, unit_id="unit_1"):
        """Write matching analysis-mask and visual artifacts."""
        mask_path = os.path.join(self.root, f"{unit_id}_mask.png")
        visual_path = os.path.join(self.root, f"{unit_id}_visual.png")
        visual = np.full((*mask.shape, 3), 255, dtype=np.uint8)
        visual[mask > 0] = (0, 0, 0)
        self.assertTrue(cv2.imwrite(mask_path, mask))
        self.assertTrue(cv2.imwrite(visual_path, visual))
        return {
            "text_unit_id": unit_id,
            "analysis_mask_crop_path": mask_path,
            "n05_copied_crop_path": visual_path,
        }

    def test_narrow_unit_keeps_only_whole_hypothesis(self):
        mask = np.zeros((24, 18), dtype=np.uint8)
        mask[4:20, 6:12] = 255

        proposal = propose_character_units(
            self.write_unit(mask),
            self.folders,
        )

        self.assertEqual("completed", proposal["status"])
        self.assertEqual(1, len(proposal["segmentation_hypotheses"]))
        self.assertEqual(
            "h0_whole",
            proposal["segmentation_hypotheses"][0]["hypothesis_id"],
        )
        self.assertEqual(
            {"x1": 0, "y1": 0, "x2": 18, "y2": 24},
            proposal["segmentation_hypotheses"][0]["segments"][0]["bbox"],
        )

    def test_wide_mask_materializes_blank_gap_split(self):
        mask = np.zeros((24, 64), dtype=np.uint8)
        mask[4:20, 5:25] = 255
        mask[3:21, 39:59] = 255

        proposal = propose_character_units(
            self.write_unit(mask, unit_id="wide"),
            self.folders,
        )

        self.assertIn(
            "wide_unit_possible_multi_letter",
            proposal["recovery_reasons"],
        )
        self.assertGreaterEqual(len(proposal["segmentation_hypotheses"]), 2)
        self.assertGreaterEqual(len(proposal["split_hints"]), 1)
        self.assertLessEqual(len(proposal["split_hints"]), 5)
        first_hint = proposal["split_hints"][0]
        self.assertLess(first_hint["cut_x"], 39)
        self.assertGreater(first_hint["cut_x"], 25)
        self.assertEqual(
            "disconnected_vector_groups",
            first_hint["validation"],
        )
        self.assertTrue(proposal["split_artifacts_materialized"])
        split = proposal["segmentation_hypotheses"][1]
        self.assertEqual("trace_supported_character_sequence", split["type"])
        self.assertEqual(2, len(split["segments"]))
        for segment in split["segments"]:
            self.assertTrue(os.path.isfile(segment["mask_crop_path"]))
            self.assertTrue(os.path.isfile(segment["visual_crop_path"]))

    def test_joint_component_splits_at_one_pixel_skeleton_bridge(self):
        mask = np.zeros((24, 64), dtype=np.uint8)
        mask[4:20, 5:25] = 255
        mask[4:20, 39:59] = 255
        mask[12, 25:39] = 255

        proposal = propose_character_units(
            self.write_unit(mask, unit_id="joint"),
            self.folders,
        )

        self.assertGreaterEqual(len(proposal["segmentation_hypotheses"]), 2)
        candidate = proposal["split_hints"][0]
        self.assertEqual(
            "connector_path_disconnection",
            candidate["validation"],
        )
        self.assertLessEqual(candidate["skeleton_crossings"], 2)
        self.assertGreater(candidate["cut_x"], 25)
        self.assertLess(candidate["cut_x"], 39)
        vector_split = candidate["vector_split"]
        self.assertIsNotNone(vector_split["connector_path_id"])
        self.assertIsNotNone(vector_split["split_after_point_index"])
        self.assertGreater(
            vector_split["left_subgraph"]["point_count"],
            0,
        )
        self.assertGreater(
            vector_split["right_subgraph"]["point_count"],
            0,
        )
        sequence = proposal["segmentation_hypotheses"][1]
        self.assertEqual([candidate["cut_x"]], sequence["cut_positions"])
        self.assertEqual(2, len(sequence["segments"]))

    def test_thick_connection_is_not_treated_as_safe_boundary(self):
        mask = np.zeros((24, 64), dtype=np.uint8)
        mask[4:20, 5:25] = 255
        mask[4:20, 39:59] = 255
        mask[7:17, 25:39] = 255

        proposal = propose_character_units(
            self.write_unit(mask, unit_id="thick_joint"),
            self.folders,
        )

        self.assertEqual(1, len(proposal["segmentation_hypotheses"]))
        self.assertEqual([], proposal["split_hints"])

    def test_multiple_boundaries_create_one_ordered_character_sequence(self):
        mask = np.zeros((24, 90), dtype=np.uint8)
        mask[4:20, 4:24] = 255
        mask[4:20, 35:55] = 255
        mask[4:20, 66:86] = 255

        proposal = propose_character_units(
            self.write_unit(mask, unit_id="three_letters"),
            self.folders,
        )

        self.assertEqual(2, len(proposal["segmentation_hypotheses"]))
        sequence = proposal["segmentation_hypotheses"][1]
        self.assertEqual("trace_supported_character_sequence", sequence["type"])
        self.assertEqual(2, len(sequence["cut_positions"]))
        self.assertEqual(3, len(sequence["segments"]))
        self.assertEqual(
            sorted(sequence["cut_positions"]),
            sequence["cut_positions"],
        )

    def test_upper_mark_attaches_to_lower_candidate_and_survives_crop(self):
        mask = np.zeros((30, 70), dtype=np.uint8)
        mask[8:26, 5:28] = 255
        mask[8:26, 42:65] = 255
        mask[1:4, 50:54] = 255

        proposal = propose_character_units(
            self.write_unit(mask, unit_id="upper_mark"),
            self.folders,
        )

        diagnostics = proposal["diagnostics"]
        floating = [
            component
            for component in diagnostics["components"]
            if component["is_floating"]
        ]
        self.assertEqual(1, len(floating))
        self.assertIsNotNone(floating[0]["attached_to_component_id"])
        split = proposal["segmentation_hypotheses"][1]
        right_segment = split["segments"][1]
        self.assertLessEqual(right_segment["bbox"]["y1"], 1)
        saved_mask = cv2.imread(
            right_segment["mask_crop_path"],
            cv2.IMREAD_GRAYSCALE,
        )
        self.assertGreater(np.count_nonzero(saved_mask[:5]), 0)

    def test_border_contact_sets_recovery_reasons_without_rejecting_unit(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[0:10, 0:8] = 255

        proposal = propose_character_units(
            self.write_unit(mask, unit_id="border"),
            self.folders,
        )

        self.assertTrue(proposal["recovery_needed"])
        self.assertIn("touches_left_border", proposal["recovery_reasons"])
        self.assertIn("touches_top_border", proposal["recovery_reasons"])
        self.assertEqual(
            "h0_whole",
            proposal["segmentation_hypotheses"][0]["hypothesis_id"],
        )

    def test_n05_map_serializes_character_unit_proposal(self):
        mask = np.zeros((24, 64), dtype=np.uint8)
        mask[4:20, 5:25] = 255
        mask[4:20, 39:59] = 255
        unit = self.write_unit(mask, unit_id="map_unit")

        routes_path = os.path.join(self.root, "routes.json")
        settings_path = os.path.join(self.root, "settings.json")
        output_dir = os.path.join(self.root, "n05_map")
        route_payload = {
            "document_id": "doc_1",
            "routes": [
                {
                    "document_id": "doc_1",
                    "text_unit_id": "map_unit",
                    "group_id": 7,
                    "classification_crop_path": unit["n05_copied_crop_path"],
                    "analysis_mask_crop_path": unit[
                        "analysis_mask_crop_path"
                    ],
                    "visual_classification": {
                        "visual_class": "handwriting_only",
                        "recommended_route": "handwritten_ocr",
                        "scores": {},
                        "thresholds": {},
                    },
                }
            ],
        }
        with open(routes_path, "w", encoding="utf-8") as file:
            json.dump(route_payload, file)
        with open(settings_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "reset_output": True,
                    "experts": {"scribetrace": {"enabled": False}},
                },
                file,
            )

        result = build_handwriting_expert_map(
            visual_routes_path=routes_path,
            output_dir=output_dir,
            settings_path=settings_path,
        )

        self.assertEqual(1, len(result["handwritten_text_units"]))
        proposal = result["handwritten_text_units"][0][
            "character_unit_proposal"
        ]
        self.assertEqual("completed", proposal["status"])
        self.assertGreaterEqual(len(proposal["segmentation_hypotheses"]), 2)
        self.assertGreater(len(proposal["split_hints"]), 0)
        self.assertTrue(proposal["split_artifacts_materialized"])
        self.assertTrue(os.path.isfile(result["metadata_path"]))


if __name__ == "__main__":
    unittest.main()
