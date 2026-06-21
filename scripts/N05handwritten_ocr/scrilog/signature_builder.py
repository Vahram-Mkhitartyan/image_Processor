"""Build a normalized ScriLog signature from one ScribeTrace result."""

from __future__ import annotations

from typing import Any, Dict

from .constants import UNKNOWN_ID
from .signature import ScriLogSignature
from .signature_extractors import ScriLogSignatureExtractors
from .utils import _as_bool, _as_int, _first_present, _merge_dicts, _safe_dict

class ScriLogSignatureBuilder(ScriLogSignatureExtractors):
    """
    Builds ScriLogSignature from a ScribeTrace JSON-like payload.

    This builder is intentionally tolerant because ScribeTrace can store
    the same concept in several places depending on the run mode:

        root["metrics"]["skeleton_graph"]["endpoint_count"]
        root["reconstruction"]["original_hypothesis"]["topology"]
        root["reconstruction"]["selected_feature_vector"]
        root["ml_features"]
        selected_hypothesis["feature_vector"]

    Boundary:
        This builder expects ScribeTrace has already done:
            - reconstruction
            - selected hypothesis choice
            - topology/vector extraction

        It does not read pixels.
    """

    def build(self, payload: Dict[str, Any]) -> ScriLogSignature:
        root = _safe_dict(payload)

        selected = self._extract_selected_payload(root)
        topology = self._extract_topology_payload(root, selected)
        vectors = self._extract_vector_payload(root, selected)
        reconstruction = self._extract_reconstruction_payload(root, selected)

        merged = _merge_dicts(
            root,
            selected,
            topology,
            vectors,
            reconstruction,
        )

        unit_id = self._extract_unit_id(root, selected)
        selected_hypothesis_id = self._extract_selected_hypothesis_id(root, selected)

        width, height, aspect_ratio = self._extract_width_height(merged)

        endpoint_zones = self._extract_zone_counts(
            root=root,
            selected=selected,
            topology=topology,
            vectors=vectors,
            preferred_keys=[
                "endpoint_quadrants",
                "endpoint_zones",
                "endpoint_zone_counts",
                "endpoints_by_zone",
                "endpoint_distribution",
            ],
        )

        junction_zones = self._extract_zone_counts(
            root=root,
            selected=selected,
            topology=topology,
            vectors=vectors,
            preferred_keys=[
                "junction_quadrants",
                "junction_zones",
                "junction_zone_counts",
                "junctions_by_zone",
                "junction_distribution",
            ],
        )

        loop_zones = self._extract_zone_counts(
            root=root,
            selected=selected,
            topology=topology,
            vectors=vectors,
            preferred_keys=[
                "loop_zones",
                "hole_zones",
                "loop_zone_counts",
                "hole_zone_counts",
                "holes_by_zone",
                "loop_distribution",
                "hole_distribution",
            ],
        )

        direction_ratios = self._extract_direction_ratios(merged)

        loop_count = _as_int(
            _first_present(
                merged,
                [
                    "loop_count",
                    "loops",
                    "hole_count",
                    "holes",
                    "num_loops",
                    "num_holes",
                    "ink_hole_count",
                    "closed_loop_count",
                ],
                0,
            )
        )

        endpoint_count = _as_int(
            _first_present(
                merged,
                [
                    "endpoint_count",
                    "endpoints",
                    "num_endpoints",
                ],
                sum(endpoint_zones.values()) if endpoint_zones else 0,
            )
        )

        junction_count = _as_int(
            _first_present(
                merged,
                [
                    "junction_count",
                    "junctions",
                    "branch_count",
                    "num_junctions",
                    "junction_cluster_count",
                    "junction_pixel_count",
                ],
                sum(junction_zones.values()) if junction_zones else 0,
            )
        )

        path_count = _as_int(
            _first_present(
                merged,
                [
                    "path_count",
                    "paths",
                    "stroke_path_count",
                    "num_paths",
                    "raw_path_count",
                ],
                0,
            )
        )

        component_count = _as_int(
            _first_present(
                merged,
                [
                    "component_count",
                    "connected_component_count",
                    "num_components",
                    "raw_component_count",
                    "accepted_component_count",
                ],
                0,
            )
        )

        ink_pixels = _as_int(
            _first_present(
                merged,
                [
                    "ink_pixels",
                    "ink_pixel_count",
                    "black_pixels",
                    "foreground_pixels",
                    "total_ink_pixels",
                    "accepted_ink_pixel_count",
                    "raw_ink_pixel_count",
                ],
                0,
            )
        )

        reconstruction_phase = _as_int(
            _first_present(
                merged,
                [
                    "reconstruction_phase",
                    "phase",
                    "selected_phase",
                ],
                0,
            )
        )

        if reconstruction_phase == 0:
            if selected_hypothesis_id != "h0_original" and selected_hypothesis_id != UNKNOWN_ID:
                reconstruction_phase = 1

        line_removal_applied = _as_bool(
            _first_present(
                merged,
                [
                    "line_removal_applied",
                    "line_removed",
                    "has_line_removal",
                ],
                False,
            )
        )

        downstream_bridge_required = _as_bool(
            _first_present(
                merged,
                [
                    "downstream_bridge_required",
                    "bridge_required",
                ],
                False,
            )
        )

        downstream_bridge_completed = _as_bool(
            _first_present(
                merged,
                [
                    "downstream_bridge_completed",
                    "bridge_completed",
                ],
                False,
            )
        )

        has_top_contact, has_bottom_contact = self._infer_border_contacts(merged)

        source_kind = str(
            _first_present(
                merged,
                [
                    "sample_kind",
                    "source_kind",
                    "kind",
                    "source_type",
                ],
                "unknown",
            )
        )

        raw_keys_seen = sorted(
            set(root.keys())
            | set(selected.keys())
            | set(topology.keys())
            | set(vectors.keys())
            | set(reconstruction.keys())
        )

        return ScriLogSignature(
            unit_id=unit_id,
            selected_hypothesis_id=selected_hypothesis_id,

            loop_count=loop_count,
            endpoint_count=endpoint_count,
            junction_count=junction_count,
            path_count=path_count,
            component_count=component_count,

            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            ink_pixels=ink_pixels,

            has_top_contact=has_top_contact,
            has_bottom_contact=has_bottom_contact,

            endpoint_zones=endpoint_zones,
            junction_zones=junction_zones,
            loop_zones=loop_zones,

            direction_ratios=direction_ratios,

            reconstruction_phase=reconstruction_phase,
            line_removal_applied=line_removal_applied,

            downstream_bridge_required=downstream_bridge_required,
            downstream_bridge_completed=downstream_bridge_completed,

            source_kind=source_kind,
            raw_keys_seen=raw_keys_seen,
        )

    # --------------------------------------------------------
    # Payload extraction helpers
