"""Focused regression tests for deterministic ScribeTrace geometry."""

import json
import math
import os
import tempfile
import unittest

import cv2
import numpy as np

from .expert import (
    SkeletonGraph,
    SkeletonPoint,
    TraceFeatureEncoder,
    TraceInput,
    TraceLandmarkExtractor,
    TraceMaskAdapter,
    TracePath,
    TracePathExtractor,
    TraceSettings,
    run_scribetrace,
)
from .trace_features import TraceFeatureEncoder as ModularTraceFeatureEncoder
from .trace_paths import TracePathExtractor as ModularTracePathExtractor
from .trace_skeleton import SkeletonGraph as ModularSkeletonGraph


def build_graph(coordinates):
    """Build a graph directly from exact skeleton coordinates."""
    return SkeletonGraph(
        [SkeletonPoint(x, y) for x, y in sorted(coordinates, key=lambda p: (p[1], p[0]))]
    )


def extract_paths(coordinates, **settings):
    """Build a graph and extract paths with test-friendly defaults."""
    graph = build_graph(coordinates)
    values = {"minimum_trace_path_points": 1}
    values.update(settings)
    extractor = TracePathExtractor(values)
    return graph, extractor.extract_paths(graph), extractor.metrics


class SkeletonTopologyTests(unittest.TestCase):
    """Verify topology classification on small deterministic skeletons."""

    def test_public_facade_reexports_split_implementation(self):
        self.assertIs(SkeletonGraph, ModularSkeletonGraph)
        self.assertIs(TracePathExtractor, ModularTracePathExtractor)
        self.assertIs(TraceFeatureEncoder, ModularTraceFeatureEncoder)

    def test_simple_strokes_have_two_endpoints_and_no_junctions(self):
        shapes = {
            "horizontal": {(x, 3) for x in range(1, 7)},
            "vertical": {(3, y) for y in range(1, 7)},
            "diagonal": {(index, index) for index in range(1, 7)},
            "l_shape": (
                {(1, y) for y in range(1, 6)}
                | {(x, 5) for x in range(2, 7)}
            ),
        }

        for name, coordinates in shapes.items():
            with self.subTest(name=name):
                graph, paths, _ = extract_paths(coordinates)
                self.assertEqual(2, len(graph.endpoints()))
                self.assertEqual(0, len(graph.junctions()))
                self.assertEqual(1, len(paths))

    def test_t_and_cross_contract_to_one_logical_junction(self):
        shapes = {
            "t": (
                {(x, 2) for x in range(1, 8)}
                | {(4, y) for y in range(3, 8)}
            ),
            "cross": (
                {(x, 4) for x in range(1, 8)}
                | {(4, y) for y in range(1, 8)}
            ),
        }
        expected_branches = {"t": 3, "cross": 4}

        for name, coordinates in shapes.items():
            with self.subTest(name=name):
                graph, paths, _ = extract_paths(coordinates)
                self.assertEqual(1, len(graph.junction_clusters()))
                self.assertEqual(expected_branches[name], len(paths))

    def test_junction_to_junction_segment_is_emitted(self):
        coordinates = (
            {(x, 4) for x in range(1, 10)}
            | {(3, y) for y in range(1, 8)}
            | {(7, y) for y in range(1, 8)}
        )
        graph, paths, _ = extract_paths(coordinates)

        self.assertEqual(2, len(graph.junction_clusters()))
        between_junctions = [
            path
            for path in paths
            if path.start_node_type == "junction"
            and path.end_node_type == "junction"
        ]
        self.assertEqual(1, len(between_junctions))

    def test_closed_square_produces_one_stable_closed_path(self):
        coordinates = (
            {(x, 1) for x in range(1, 7)}
            | {(x, 6) for x in range(1, 7)}
            | {(1, y) for y in range(2, 6)}
            | {(6, y) for y in range(2, 6)}
        )
        graph, paths, _ = extract_paths(coordinates)

        self.assertEqual(0, len(graph.endpoints()))
        self.assertEqual(1, len(paths))
        self.assertTrue(paths[0].is_closed)
        self.assertEqual(
            paths[0].start_point().to_tuple(),
            paths[0].end_point().to_tuple(),
        )
        self.assertEqual((1, 1), paths[0].start_point().to_tuple())

    def test_diagonal_length_is_geometric(self):
        path = TracePath(
            0,
            [
                SkeletonPoint(0, 0),
                SkeletonPoint(1, 1),
                SkeletonPoint(2, 2),
            ],
        )
        self.assertEqual(3, path.point_count())
        self.assertAlmostEqual(2 * math.sqrt(2), path.length())

    def test_repeated_runs_have_identical_ordering(self):
        coordinates = (
            {(x, 4) for x in range(1, 10)}
            | {(3, y) for y in range(1, 8)}
            | {(7, y) for y in range(1, 8)}
        )
        first_graph, first_paths, _ = extract_paths(coordinates)
        second_graph, second_paths, _ = extract_paths(reversed(tuple(coordinates)))

        self.assertEqual(first_graph.to_dict(), second_graph.to_dict())
        self.assertEqual(
            [path.to_dict() for path in first_paths],
            [path.to_dict() for path in second_paths],
        )


class PathMergeTests(unittest.TestCase):
    """Verify that short paths are preserved unless continuation is clear."""

    def test_unambiguous_terminal_spur_merges_into_continuation(self):
        coordinates = (
            {(x, 5) for x in range(4, 11)}
            | {(4, 5), (3, 5)}
            | {(4, y) for y in range(6, 12)}
        )
        _, paths, metrics = extract_paths(
            coordinates,
            minimum_trace_path_points=4,
        )

        merged = [path for path in paths if path.merged_from_path_ids]
        self.assertEqual(1, len(merged))
        self.assertEqual(1, metrics["merged_path_count"])
        self.assertFalse(merged[0].is_short)

    def test_ambiguous_short_branch_is_preserved(self):
        coordinates = (
            {(4, 5), (3, 5)}
            | {(4 + step, 5 - step) for step in range(0, 6)}
            | {(4 + step, 5 + step) for step in range(0, 6)}
        )
        _, paths, metrics = extract_paths(
            coordinates,
            minimum_trace_path_points=4,
            short_path_merge_max_angle_degrees=60,
        )

        self.assertEqual(0, metrics["merged_path_count"])
        self.assertTrue(any(path.is_short for path in paths))

    def test_loop_and_isolated_dot_are_preserved_without_invented_dot_path(self):
        loop = (
            {(x, 1) for x in range(1, 6)}
            | {(x, 5) for x in range(1, 6)}
            | {(1, y) for y in range(2, 5)}
            | {(5, y) for y in range(2, 5)}
        )
        graph, paths, _ = extract_paths(
            loop | {(10, 10)},
            minimum_trace_path_points=20,
        )

        self.assertEqual(1, len(graph.isolated_points()))
        self.assertEqual(1, len(paths))
        self.assertTrue(paths[0].is_closed)
        self.assertTrue(paths[0].is_short)


class LandmarkAndFeatureTests(unittest.TestCase):
    """Protect deterministic shape landmarks and their ML serialization."""

    def build_wave_path(self):
        """Return a path with two top peaks and one bottom valley."""
        return TracePath(
            7,
            [
                SkeletonPoint(0, 3),
                SkeletonPoint(1, 1),
                SkeletonPoint(2, 3),
                SkeletonPoint(3, 1),
                SkeletonPoint(4, 3),
            ],
        )

    def test_coordinate_signals_emit_expected_local_landmarks(self):
        path = self.build_wave_path()
        extractor = TraceLandmarkExtractor(
            {
                "local_extrema_min_prominence": 2,
                "local_extrema_min_spacing": 1,
            }
        )

        landmarks = extractor.extract_landmarks([path])
        local_landmarks = [
            (landmark.landmark_type, landmark.index_on_path)
            for landmark in landmarks
            if landmark.landmark_type.startswith("local_")
        ]

        self.assertEqual(
            [
                ("local_top_peak", 1),
                ("local_bottom_valley", 2),
                ("local_top_peak", 3),
            ],
            local_landmarks,
        )

    def test_feature_schema_and_symbolic_sequence_are_stable(self):
        path = self.build_wave_path()
        graph = build_graph(point.to_tuple() for point in path.points)
        extractor = TraceLandmarkExtractor(
            {
                "local_extrema_min_prominence": 2,
                "local_extrema_min_spacing": 1,
            }
        )
        landmarks = extractor.extract_landmarks([path])
        encoder = TraceFeatureEncoder()

        first = encoder.encode([], graph, [path], landmarks).to_dict()
        second = encoder.encode([], graph, [path], landmarks).to_dict()

        self.assertEqual(first, second)
        self.assertEqual(sorted(first["feature_names"]), first["feature_names"])
        self.assertEqual(len(first["feature_names"]), len(first["vector"]))
        self.assertEqual("P7", first["sequence"][0])
        self.assertIn("TP", first["sequence"])
        self.assertIn("BV", first["sequence"])


class TracePipelineTests(unittest.TestCase):
    """Verify source handling, filtering, limits, and package contracts."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_image(self, name, image):
        path = os.path.join(self.temp_dir.name, name)
        self.assertTrue(cv2.imwrite(path, image))
        return path

    def base_settings(self, **overrides):
        settings = {
            "enabled": True,
            "save_debug": False,
            "minimum_ink_pixels": 2,
            "minimum_trace_path_points": 1,
        }
        settings.update(overrides)
        return settings

    def test_tiny_rejected_component_does_not_reach_skeleton(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5, 2:10] = 255
        mask[15, 15] = 255
        path = self.write_image("mask.png", mask)

        result = run_scribetrace(
            TraceInput(mask_crop_path=path),
            self.base_settings(minimum_ink_pixels=2),
        )

        self.assertEqual("completed", result.status)
        self.assertEqual(2, result.metrics["raw_component_count"])
        self.assertEqual(1, result.metrics["accepted_component_count"])
        self.assertEqual(1, result.metrics["rejected_small_component_count"])
        self.assertEqual(8, result.metrics["skeleton_point_count"])

    def test_full_runs_preserve_component_cluster_and_path_order(self):
        mask = np.zeros((30, 40), dtype=np.uint8)
        mask[5, 3:14] = 255
        mask[15, 20:35] = 255
        mask[11:20, 27] = 255
        path = self.write_image("deterministic.png", mask)
        trace_input = TraceInput(mask_crop_path=path, text_unit_id="stable")

        first = run_scribetrace(trace_input, self.base_settings()).to_dict()
        second = run_scribetrace(trace_input, self.base_settings()).to_dict()

        self.assertEqual(first["components"], second["components"])
        self.assertEqual(
            first["metrics"]["skeleton_graph"],
            second["metrics"]["skeleton_graph"],
        )
        self.assertEqual(first["trace_paths"], second["trace_paths"])

    def test_debug_outputs_are_routed_to_explicit_n05_folder(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[10, 3:17] = 255
        mask_path = self.write_image("debug-mask.png", mask)
        output_dir = os.path.join(self.temp_dir.name, "n05", "scribetrace")

        result = run_scribetrace(
            TraceInput(
                mask_crop_path=mask_path,
                output_dir=output_dir,
                text_unit_id="unit/unsafe",
            ),
            self.base_settings(save_debug=True),
        )

        self.assertEqual("completed", result.status)
        self.assertEqual(output_dir, result.metrics["output_dir"])
        for debug_path in result.debug_paths.values():
            self.assertTrue(
                debug_path.startswith(os.path.join(output_dir, "debug"))
            )
            self.assertTrue(os.path.isfile(debug_path))
            self.assertNotIn("unit/unsafe", debug_path)

    def test_result_is_saved_as_compact_json_with_sanitized_name(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[10, 3:17] = 255
        mask_path = self.write_image("json-mask.png", mask)
        output_dir = os.path.join(self.temp_dir.name, "n05", "scribetrace")

        result = run_scribetrace(
            TraceInput(
                mask_crop_path=mask_path,
                output_dir=output_dir,
                text_unit_id="unit/unsafe",
            ),
            self.base_settings(),
        )

        expected_path = os.path.join(
            output_dir,
            "metadata",
            "unit_unsafe_scribetrace.json",
        )
        self.assertEqual(expected_path, result.result_json_path)
        self.assertTrue(os.path.isfile(expected_path))

        with open(expected_path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        self.assertEqual("completed", payload["status"])
        self.assertEqual(expected_path, payload["result_json_path"])
        self.assertNotIn("points", payload["components"][0])
        self.assertNotIn("points", payload["trace_paths"][0])

    def test_component_limit_returns_limited_evidence(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        for index in range(4):
            x = 2 + index * 6
            mask[5:7, x:x + 2] = 255
        path = self.write_image("many_components.png", mask)

        result = run_scribetrace(
            TraceInput(mask_crop_path=path),
            self.base_settings(
                maximum_component_count_for_full_trace=3,
                minimum_ink_pixels=1,
            ),
        )

        self.assertEqual("completed_limited", result.status)
        self.assertEqual("component_limit_exceeded", result.reason)
        self.assertEqual(4, result.component_count())
        self.assertEqual(0, result.metrics["skeleton_point_count"])
        self.assertEqual([], result.trace_paths)

    def test_missing_mask_falls_back_to_visual_otsu_inverse(self):
        visual = np.full((20, 20), 255, dtype=np.uint8)
        visual[10, 3:17] = 0
        visual_path = self.write_image("visual.png", visual)

        result = run_scribetrace(
            TraceInput(
                mask_crop_path=os.path.join(self.temp_dir.name, "missing.png"),
                visual_crop_path=visual_path,
            ),
            self.base_settings(ink_threshold_mode="otsu"),
        )

        self.assertEqual("completed", result.status)
        self.assertTrue(result.metrics["fallback_used"])
        self.assertEqual("visual_crop_fallback", result.metrics["source_type"])
        self.assertEqual("otsu_inverse", result.metrics["threshold_mode"])
        self.assertEqual(visual_path, result.metrics["source_path"])

    def test_missing_mask_and_visual_crop_fail_clearly(self):
        result = run_scribetrace(
            TraceInput(
                mask_crop_path=os.path.join(self.temp_dir.name, "missing-mask.png"),
                crop_path=os.path.join(self.temp_dir.name, "missing-crop.png"),
            ),
            self.base_settings(),
        )

        self.assertEqual("failed", result.status)
        self.assertIn("No readable ScribeTrace source", result.error)

    def test_threshold_modes_are_honored_and_invalid_mode_is_rejected(self):
        grayscale = np.array([[50, 100, 150, 200]], dtype=np.uint8)
        binary_adapter = TraceMaskAdapter(
            self.base_settings(ink_threshold_mode="binary")
        )
        fixed_adapter = TraceMaskAdapter(
            self.base_settings(
                ink_threshold_mode="fixed",
                fixed_threshold_value=175,
            )
        )

        binary, binary_mode = binary_adapter.binarize(grayscale, "analysis_mask")
        fixed, fixed_mode = fixed_adapter.binarize(grayscale, "analysis_mask")
        self.assertEqual("binary_128", binary_mode)
        self.assertEqual("fixed", fixed_mode)
        self.assertEqual(2, cv2.countNonZero(binary))
        self.assertEqual(1, cv2.countNonZero(fixed))
        with self.assertRaises(ValueError):
            TraceSettings(ink_threshold_mode="mystery")

    def test_completed_limited_counts_as_attempted_evidence(self):
        from .expert import recognize

        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:4, 2:4] = 255
        mask[10:12, 10:12] = 255
        path = self.write_image("limited.png", mask)
        result = recognize(
            path,
            {"scribetrace_mask_crop_path": path},
            self.base_settings(
                minimum_ink_pixels=1,
                maximum_component_count_for_full_trace=1,
            ),
        )
        self.assertTrue(result["attempted"])
        self.assertEqual("completed_limited", result["status"])

    def test_n05_package_interface_imports(self):
        from . import EXPERT_NAME, get_expert_manifest, recognize

        self.assertEqual("scribetrace", EXPERT_NAME)
        self.assertTrue(callable(get_expert_manifest))
        self.assertTrue(callable(recognize))


if __name__ == "__main__":
    unittest.main()
