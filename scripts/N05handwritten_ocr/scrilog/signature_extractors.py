"""Schema-tolerant extraction helpers for real ScribeTrace payloads."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .constants import UNKNOWN_ID
from .utils import _as_bool, _as_float, _as_int, _first_present, _merge_dicts, _safe_dict, _safe_list


class ScriLogSignatureExtractors:
    """Reusable payload extraction methods used by the signature builder."""

    def _extract_selected_payload(self, root: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find the selected hypothesis/result object if present.

        Real ScribeTrace may store selection under:

            reconstruction.selected_hypothesis_id

        and the selected object may be:

            reconstruction.original_hypothesis
            reconstruction.hypotheses[i]
        """

        direct_selected = _first_present(
            root,
            [
                "selected_hypothesis",
                "selected_result",
                "selected_trace",
                "selected",
                "hypothesis",
            ],
            None,
        )

        if isinstance(direct_selected, dict):
            return direct_selected

        reconstruction = _safe_dict(root.get("reconstruction"))

        selected_id = _first_present(
            reconstruction,
            [
                "selected_hypothesis_id",
                "hypothesis_id",
                "id",
            ],
            _first_present(
                root,
                [
                    "selected_hypothesis_id",
                    "hypothesis_id",
                    "id",
                ],
                None,
            ),
        )

        original = _safe_dict(reconstruction.get("original_hypothesis"))

        if selected_id is not None:
            selected_id = str(selected_id)

            if original.get("hypothesis_id") == selected_id:
                return original

            for hypothesis in _safe_list(reconstruction.get("hypotheses")):
                hypothesis_dict = _safe_dict(hypothesis)

                if hypothesis_dict.get("hypothesis_id") == selected_id:
                    return hypothesis_dict

        if original.get("selected") is True:
            return original

        return {}

    def _extract_topology_payload(
        self,
        root: Dict[str, Any],
        selected: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Collect topology-like fields from possible locations.
        """

        metrics = _safe_dict(root.get("metrics"))
        skeleton_graph = _safe_dict(metrics.get("skeleton_graph"))

        reconstruction = _safe_dict(root.get("reconstruction"))
        original = _safe_dict(reconstruction.get("original_hypothesis"))
        selected_observation = _safe_dict(
            selected.get("scrilog_observation")
            or reconstruction.get("selected_scrilog_observation")
            or metrics.get("scrilog_observation")
        )

        return _merge_dicts(
            skeleton_graph,
            metrics,
            selected_observation,
            _safe_dict(original.get("topology")),
            _safe_dict(root.get("topology")),
            _safe_dict(selected.get("topology")),
            _safe_dict(root.get("trace_topology")),
            _safe_dict(selected.get("trace_topology")),
        )

    def _feature_vector_to_dict(
        self,
        payload: Any,
    ) -> Dict[str, Any]:
        """
        Convert ScribeTrace feature vector payload into a flat dict.

        ScribeTrace often stores ML features as:

            {
                "feature_names": [...],
                "vector": [...]
            }

        ScriLog wants:

            {
                "endpoint_count": 4.0,
                "ink_bbox_width": 23.0,
                ...
            }
        """

        data = _safe_dict(payload)

        names = data.get("feature_names")
        values = data.get("vector")

        feature_dict: Dict[str, Any] = {}

        if isinstance(names, list) and isinstance(values, list):
            for name, value in zip(names, values):
                feature_dict[str(name)] = value

        # Preserve useful non-vector fields too.
        for key in [
            "sequence",
            "sequence_string",
            "quality_flags",
            "schema_version",
            "label",
        ]:
            if key in data:
                feature_dict[key] = data[key]

        return feature_dict

    def _extract_vector_payload(
        self,
        root: Dict[str, Any],
        selected: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Collect vector/feature-like fields from possible locations.

        Handles both:
            flat feature dicts
        and:
            {"feature_names": [...], "vector": [...]}
        """

        reconstruction = _safe_dict(root.get("reconstruction"))

        root_selected_feature_vector = _safe_dict(root.get("selected_feature_vector"))
        reconstruction_selected_feature_vector = _safe_dict(
            reconstruction.get("selected_feature_vector")
        )

        selected_feature_vector = _safe_dict(selected.get("feature_vector"))

        root_feature_vector = _safe_dict(root.get("feature_vector"))
        selected_raw_feature_vector = _safe_dict(selected.get("feature_vector"))

        root_ml_features = _safe_dict(root.get("ml_features"))
        selected_ml_features = _safe_dict(selected.get("ml_features"))

        return _merge_dicts(
            self._feature_vector_to_dict(root_selected_feature_vector),
            self._feature_vector_to_dict(reconstruction_selected_feature_vector),
            self._feature_vector_to_dict(selected_feature_vector),
            self._feature_vector_to_dict(root_feature_vector),
            self._feature_vector_to_dict(selected_raw_feature_vector),
            self._feature_vector_to_dict(root_ml_features),
            self._feature_vector_to_dict(selected_ml_features),

            _safe_dict(root.get("vectors")),
            _safe_dict(selected.get("vectors")),
            _safe_dict(root.get("features")),
            _safe_dict(selected.get("features")),
        )

    def _extract_reconstruction_payload(
        self,
        root: Dict[str, Any],
        selected: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Collect reconstruction metadata from possible locations.
        """

        reconstruction = _safe_dict(root.get("reconstruction"))

        return _merge_dicts(
            reconstruction,
            _safe_dict(selected.get("reconstruction")),
            _safe_dict(root.get("reconstruction_state")),
            _safe_dict(selected.get("reconstruction_state")),
            _safe_dict(root.get("metadata")),
            _safe_dict(selected.get("metadata")),
        )

    # --------------------------------------------------------
    # Identity helpers
    # --------------------------------------------------------

    def _extract_unit_id(
        self,
        root: Dict[str, Any],
        selected: Dict[str, Any],
    ) -> str:
        trace_input = _safe_dict(root.get("trace_input"))
        metadata = _safe_dict(selected.get("metadata"))

        value = _first_present(
            root,
            [
                "unit_id",
                "sample_id",
                "glyph_id",
                "source_id",
            ],
            _first_present(
                trace_input,
                [
                    "stable_unit_id",
                    "text_unit_id",
                    "document_id",
                ],
                _first_present(
                    metadata,
                    [
                        "stable_unit_id",
                        "text_unit_id",
                    ],
                    _first_present(
                        selected,
                        [
                            "unit_id",
                            "sample_id",
                            "glyph_id",
                            "source_id",
                        ],
                        UNKNOWN_ID,
                    ),
                ),
            ),
        )

        return str(value)

    def _extract_selected_hypothesis_id(
        self,
        root: Dict[str, Any],
        selected: Dict[str, Any],
    ) -> str:
        reconstruction = _safe_dict(root.get("reconstruction"))

        value = _first_present(
            selected,
            [
                "hypothesis_id",
                "id",
                "selected_hypothesis_id",
            ],
            _first_present(
                reconstruction,
                [
                    "selected_hypothesis_id",
                    "hypothesis_id",
                    "id",
                ],
                _first_present(
                    root,
                    [
                        "selected_hypothesis_id",
                        "hypothesis_id",
                        "id",
                    ],
                    UNKNOWN_ID,
                ),
            ),
        )

        return str(value)

    # --------------------------------------------------------
    # Zone helpers
    # --------------------------------------------------------

    def _extract_zone_counts(
        self,
        root: Dict[str, Any],
        selected: Dict[str, Any],
        topology: Dict[str, Any],
        vectors: Dict[str, Any],
        preferred_keys: List[str],
    ) -> Dict[str, int]:
        """
        Extract zone-count dictionaries from several possible locations.

        For now, this mostly supports fake/debug payloads because real
        ScribeTrace exposes endpoint distribution as ratios instead of
        named zones. That is fine: side exits are inferred separately from
        endpoint_*_half_ratio fields.
        """

        candidates = [
            root,
            selected,
            topology,
            vectors,
            _safe_dict(root.get("features")),
            _safe_dict(selected.get("features")),
            _safe_dict(root.get("topology")),
            _safe_dict(selected.get("topology")),
        ]

        for container in candidates:
            for key in preferred_keys:
                raw = container.get(key)

                if not isinstance(raw, dict):
                    continue

                cleaned: Dict[str, int] = {}

                for zone_name, count in raw.items():
                    value = _as_int(count, 0)

                    if value > 0:
                        cleaned[str(zone_name)] = value

                if cleaned:
                    return cleaned

        return {}

    # --------------------------------------------------------
    # Geometry helpers
    # --------------------------------------------------------

    def _extract_width_height(
        self,
        data: Dict[str, Any],
    ) -> Tuple[float, float, float]:
        """
        Extract glyph geometry.

        Real ScribeTrace usually stores this in ML feature names:

            ink_bbox_width
            ink_bbox_height
            ink_bbox_aspect_ratio

        The important rule:
            skip zero values when looking for width/height/aspect.
        """

        def first_positive_float(keys: List[str]) -> float:
            for key in keys:
                value = _as_float(data.get(key), 0.0)

                if value > 0.0:
                    return value

            return 0.0

        width = first_positive_float(
            [
                "ink_bbox_width",
                "bbox_width",
                "component_width",
                "mask_width",
                "width",
                "w",
            ]
        )

        height = first_positive_float(
            [
                "ink_bbox_height",
                "bbox_height",
                "component_height",
                "mask_height",
                "height",
                "h",
            ]
        )

        aspect_ratio = first_positive_float(
            [
                "ink_bbox_aspect_ratio",
                "bbox_aspect_ratio",
                "aspect_ratio",
                "component_aspect_ratio",
            ]
        )

        if aspect_ratio <= 0.0 and width > 0.0 and height > 0.0:
            aspect_ratio = width / height

        return width, height, aspect_ratio

    # --------------------------------------------------------
    # Direction helpers
    # --------------------------------------------------------

    def _extract_direction_ratios(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Extract stroke direction ratios.

        Fake payloads may already contain:
            direction_ratios = {"horizontal": 0.5, ...}

        Real ScribeTrace usually contains:
            left_step_ratio
            right_step_ratio
            up_step_ratio
            down_step_ratio
            diagonal_step_ratio
        """

        existing = data.get("direction_ratios")

        if isinstance(existing, dict):
            cleaned: Dict[str, float] = {}

            for key, value in existing.items():
                number = _as_float(value, 0.0)

                if number > 0.0:
                    cleaned[str(key)] = number

            if cleaned:
                return cleaned

        left_ratio = _as_float(data.get("left_step_ratio"), 0.0)
        right_ratio = _as_float(data.get("right_step_ratio"), 0.0)
        up_ratio = _as_float(data.get("up_step_ratio"), 0.0)
        down_ratio = _as_float(data.get("down_step_ratio"), 0.0)
        diagonal_ratio = _as_float(data.get("diagonal_step_ratio"), 0.0)

        horizontal_raw = left_ratio + right_ratio
        vertical_raw = up_ratio + down_ratio
        diagonal_raw = diagonal_ratio

        total = horizontal_raw + vertical_raw + diagonal_raw

        if total <= 0.0:
            left_count = _as_float(data.get("left_step_count"), 0.0)
            right_count = _as_float(data.get("right_step_count"), 0.0)
            up_count = _as_float(data.get("up_step_count"), 0.0)
            down_count = _as_float(data.get("down_step_count"), 0.0)
            diagonal_count = _as_float(data.get("diagonal_step_count"), 0.0)

            horizontal_raw = left_count + right_count
            vertical_raw = up_count + down_count
            diagonal_raw = diagonal_count

            total = horizontal_raw + vertical_raw + diagonal_raw

        if total <= 0.0:
            return {}

        return {
            "horizontal": horizontal_raw / total,
            "vertical": vertical_raw / total,
            "diagonal": diagonal_raw / total,
            "left": left_ratio,
            "right": right_ratio,
            "up": up_ratio,
            "down": down_ratio,
        }

    # --------------------------------------------------------
    # Structural inference helpers
    # --------------------------------------------------------

    def _infer_border_contacts(self, data: Dict[str, Any]) -> Tuple[bool, bool]:
        """Read objective top/bottom border contacts without semantic guesses."""

        def explicit_bool(keys: List[str]) -> bool:
            for key in keys:
                if key in data:
                    return _as_bool(data.get(key), False)

            return False

        def numeric_positive(keys: List[str]) -> bool:
            for key in keys:
                if _as_float(data.get(key), 0.0) > 0.0:
                    return True

            return False

        border_contacts = _safe_dict(data.get("border_contacts"))
        has_top_contact = (
            explicit_bool(["has_top_contact", "top_contact"])
            or _as_bool(border_contacts.get("top"), False)
            or numeric_positive(["border_contact_top"])
        )

        has_bottom_contact = (
            explicit_bool(["has_bottom_contact", "bottom_contact"])
            or _as_bool(border_contacts.get("bottom"), False)
            or numeric_positive(["border_contact_bottom"])
        )

        return has_top_contact, has_bottom_contact
