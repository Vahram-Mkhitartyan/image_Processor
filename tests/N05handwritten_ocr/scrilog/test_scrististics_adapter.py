"""Protect the soft boundary between Scrististics and ScriLog."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.N05handwritten_ocr.scrilog.parser import ScriLogParser
from scripts.N05handwritten_ocr.scrilog.scrististics_adapter import (
    ScrististicsEvidenceAdapter,
)


def distribution(feature, mode, alternate, importance="high"):
    """Build a small empirical distribution for one test feature."""
    return {
        "feature": feature,
        "importance": importance,
        "total": 100,
        "most_common_value": str(mode),
        "values": [
            {"value": str(mode), "count": 90, "percent": 90},
            {"value": str(alternate), "count": 10, "percent": 10},
        ],
    }


class ScrististicsAdapterTests(unittest.TestCase):
    """Ensure empirical evidence stays probabilistic and explainable."""

    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.profile_path = Path(self.temporary_dir.name) / "profile.json"
        self.profile_path.write_text(
            json.dumps(
                {
                    "profile_kind": "empirical_observed_topology",
                    "classes": {
                        "Ա": {
                            "label": "Ա",
                            "raw_class_id": "0",
                            "feature_distributions": {
                                "endpoints": distribution("endpoints", 2, 5),
                                "components": distribution("components", 1, 2),
                            },
                        },
                        "Բ": {
                            "label": "Բ",
                            "raw_class_id": "1",
                            "feature_distributions": {
                                "endpoints": distribution("endpoints", 5, 2),
                                "components": distribution("components", 2, 1),
                            },
                        },
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.payload = {
            "metrics": {
                "scrilog_observation": {
                    "endpoint_count": 2,
                    "component_count": 1,
                }
            }
        }

    def tearDown(self):
        self.temporary_dir.cleanup()

    def test_matching_class_ranks_first_without_statistical_blocks(self):
        adapter = ScrististicsEvidenceAdapter(self.profile_path)
        result = adapter.evaluate(self.payload)

        self.assertEqual("completed", result["status"])
        self.assertEqual("Ա", result["top_candidates"][0]["label"])
        self.assertAlmostEqual(
            1.0,
            sum(row["likelihood"] for row in result["class_scores"]),
            places=5,
        )
        self.assertFalse(result["policy"]["can_block"])
        self.assertFalse(
            any(effect.effect == "block" for effect in result["candidate_effects"])
        )

    def test_parser_keeps_signature_facts_and_attaches_statistical_evidence(self):
        parser = ScriLogParser(
            scrististics_adapter=ScrististicsEvidenceAdapter(self.profile_path)
        )
        result = parser.parse_scribetrace_payload(self.payload).to_dict()

        self.assertEqual(2, result["signature"]["endpoint_count"])
        self.assertEqual("completed", result["statistical_evidence"]["status"])
        self.assertTrue(
            any("endpoint_count(2)" in fact for fact in result["facts"])
        )
        self.assertTrue(
            all(
                effect["provenance"] == "scrististics_empirical_profile"
                for effect in result["candidate_effects"]
            )
        )

    def test_missing_profile_is_nonfatal(self):
        adapter = ScrististicsEvidenceAdapter(
            Path(self.temporary_dir.name) / "missing.json"
        )
        result = adapter.evaluate(self.payload)

        self.assertEqual("unavailable", result["status"])
        self.assertEqual([], result["candidate_effects"])


if __name__ == "__main__":
    unittest.main()
