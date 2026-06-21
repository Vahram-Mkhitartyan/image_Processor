"""
ScriStatistics / ScriStistics Miner v0.1

Project spelling:
    Folder/module name: scrististics
    Human-readable name: ScriStatistics

Meaning:
    scribe + statistics

ScriStatistics is the statistical profile layer after ScriLog.

Pipeline boundary:

    ScribeTrace
        -> reconstructs / traces glyphs

    ScriLog
        -> converts traced geometry into symbolic facts

    ScriStatistics
        -> learns how often those facts appear per letter/class
        -> tracks expected-vs-traced mismatches
        -> separates common patterns from minor-but-rising errors

ScriStatistics DOES:
    - read labeled annotation JSON
    - group samples by class/letter
    - count structural feature frequencies
    - summarize common observations
    - track minor/common tracing errors when expected-vs-observed data exists

ScriStatistics DOES NOT:
    - modify glyph images
    - choose OCR output
    - replace ScribeTrace
    - replace ScriLog
    - train Random Forest in v0.1

Design:
    Start with transparent statistics first.
    Random Forest tuning comes later, after the profile data is trustworthy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCRISTISTICS_VERSION = "0.1"


# ============================================================
# Feature importance
# ============================================================

FEATURE_IMPORTANCE: Dict[str, str] = {
    # High importance:
    # These usually define the core identity of the glyph.
    "visual_ink_holes": "high",
    "closed_skeleton_loops": "high",
    "endpoints": "high",
    "junction_clusters": "high",
    "components": "high",
    "wide_shape": "high",
    "tall_shape": "high",

    # Medium importance:
    # Useful, but more sensitive to tracing/reconstruction noise.
    "trace_paths": "medium",
    "isolated_points": "medium",
    "short_paths": "medium",
    "touches_left_border": "medium",
    "touches_right_border": "medium",
    "touches_top_border": "medium",
    "touches_bottom_border": "medium",

    # Low importance:
    # Quadrant position can shift due to handwriting style, crop tightness,
    # rotation, or skeleton graph noise.
    "endpoint_top_left": "low",
    "endpoint_top_right": "low",
    "endpoint_bottom_left": "low",
    "endpoint_bottom_right": "low",
    "junction_top_left": "low",
    "junction_top_right": "low",
    "junction_bottom_left": "low",
    "junction_bottom_right": "low",
}


# ============================================================
# Safe helpers
# ============================================================

def _safe_dict(value: Any) -> Dict[str, Any]:
    """
    Return value if it is a dictionary, otherwise return empty dictionary.
    """

    if isinstance(value, dict):
        return value

    return {}


def _safe_list(value: Any) -> List[Any]:
    """
    Normalize unknown input into a list.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [value]


def _as_label(value: Any) -> str:
    """
    Convert class/letter labels into stable strings.
    """

    if value is None:
        return "unknown"

    text = str(value).strip()

    if not text:
        return "unknown"

    return text


def _json_value(value: Any) -> str:
    """
    Convert feature values into stable counter keys.

    Counter keys should not depend on Python object weirdness.
    So values like True, 1, 1.0, and None are normalized.
    """

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))

        return f"{value:.4f}"

    if value is None:
        return "null"

    return str(value)


def _percent(
    count: int,
    total: int,
) -> int:
    """
    Convert count / total into rounded whole-number percent.

    Example:
        82 instead of 82.314814...
    """

    if total <= 0:
        return 0

    return int(round((count / total) * 100))


def _load_json(path: Path) -> Any:
    """
    Read JSON from disk.
    """

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(
    path: Path,
    payload: Dict[str, Any],
) -> None:
    """
    Write JSON to disk.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def _iter_json_files(input_path: Path) -> Iterable[Path]:
    """
    Yield one JSON file or all JSON files inside a directory.
    """

    if input_path.is_file():
        yield input_path
        return

    for path in sorted(input_path.rglob("*.json")):
        if path.is_file():
            yield path


# ============================================================
# Class label mapping
# ============================================================

CLASS_ID_TO_ARMENIAN: Dict[str, str] = {
    "0": "Ա",
    "1": "Բ",
    "2": "Գ",
    "3": "Դ",
    "4": "Ե",
    "5": "Զ",
    "6": "Է",
    "7": "Ը",
    "8": "Թ",
    "9": "Ժ",
    "10": "Ի",
    "11": "Լ",
    "12": "Խ",
    "13": "Ծ",
    "14": "Կ",
    "15": "Հ",
    "16": "Ձ",
    "17": "Ղ",
    "18": "Ճ",
    "19": "Մ",
    "20": "Յ",
    "21": "Ն",
    "22": "Շ",
    "23": "Ո",
    "24": "Ու",
    "25": "Չ",
    "26": "Պ",
    "27": "Ջ",
    "28": "Ռ",
    "29": "Ս",
    "30": "Վ",
    "31": "Տ",
    "32": "Ր",
    "33": "Ց",
    "34": "Փ",
    "35": "Ք",
    "36": "Եվ",
    "37": "Օ",
    "38": "Ֆ",
    "39": "ա",
    "40": "բ",
    "41": "գ",
    "42": "դ",
    "43": "ե",
    "44": "զ",
    "45": "է",
    "46": "ը",
    "47": "թ",
    "48": "ժ",
    "49": "ի",
    "50": "լ",
    "51": "խ",
    "52": "ծ",
    "53": "կ",
    "54": "հ",
    "55": "ձ",
    "56": "ղ",
    "57": "ճ",
    "58": "մ",
    "59": "յ",
    "60": "ն",
    "61": "շ",
    "62": "ո",
    "63": "ու",
    "64": "չ",
    "65": "պ",
    "66": "ջ",
    "67": "ռ",
    "68": "ս",
    "69": "վ",
    "70": "տ",
    "71": "ր",
    "72": "ց",
    "73": "փ",
    "74": "ք",
    "75": "և",
    "76": "օ",
    "77": "ֆ",
}


def normalize_class_label(raw_label: Any) -> str:
    """
    Convert raw class labels into Armenian letter labels.

    Example:
        "8"  -> "Թ"
        8    -> "Թ"
        "39" -> "ա"
        "ա"  -> "ա"

    If the label is not numeric or not in the map, preserve it.
    """

    label = _as_label(raw_label)

    if label in CLASS_ID_TO_ARMENIAN:
        return CLASS_ID_TO_ARMENIAN[label]

    return label


def extract_raw_class_id(record: Dict[str, Any]) -> str:
    """
    Preserve the original numeric class id when available.

    This lets the output contain both:

        raw_class_id: "8"
        label: "Թ"
    """

    for key in [
        "class_label",
        "class_id",
        "label",
        "true_label",
        "class",
        "target",
    ]:
        if key in record and record[key] is not None:
            raw = _as_label(record[key])

            if raw in CLASS_ID_TO_ARMENIAN:
                return raw

    metadata = _safe_dict(record.get("metadata"))

    for key in [
        "class_label",
        "class_id",
        "label",
        "true_label",
        "class",
        "target",
    ]:
        if key in metadata and metadata[key] is not None:
            raw = _as_label(metadata[key])

            if raw in CLASS_ID_TO_ARMENIAN:
                return raw

    return "unknown"


# ============================================================
# Record extraction
# ============================================================

def extract_records(payload: Any) -> List[Dict[str, Any]]:
    """
    Normalize flexible JSON input into a list of records.

    Supported shapes:

        [
            {...},
            {...}
        ]

        {
            "annotations": [...]
        }

        {
            "records": [...]
        }

        {
            "samples": [...]
        }

        single record:
        {
            "class_label": "ա",
            ...
        }
    """

    if isinstance(payload, list):
        return [
            record
            for record in payload
            if isinstance(record, dict)
        ]

    if isinstance(payload, dict):
        for key in [
            "annotations",
            "records",
            "samples",
            "items",
            "glyphs",
        ]:
            value = payload.get(key)

            if isinstance(value, list):
                return [
                    record
                    for record in value
                    if isinstance(record, dict)
                ]

        return [payload]

    return []


def extract_label(record: Dict[str, Any]) -> str:
    """
    Extract the class/letter label from one record.

    Numeric class labels are mapped into Armenian letters.

    Example:
        class_label "8" -> "Թ"
        class_label "39" -> "ա"
    """

    for key in [
        "class_label",
        "letter",
        "label",
        "true_label",
        "class",
        "class_id",
        "target",
    ]:
        if key in record and record[key] is not None:
            return normalize_class_label(record[key])

    metadata = _safe_dict(record.get("metadata"))

    for key in [
        "class_label",
        "letter",
        "label",
        "true_label",
        "class",
        "class_id",
        "target",
    ]:
        if key in metadata and metadata[key] is not None:
            return normalize_class_label(metadata[key])

    return "unknown"


def _first_present_dict(
    record: Dict[str, Any],
    keys: List[str],
) -> Dict[str, Any]:
    """
    Return the first dictionary found under one of the provided keys.
    """

    for key in keys:
        value = record.get(key)

        if isinstance(value, dict):
            return value

    return {}


def extract_observed_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract observed/traced/auto-detected features.

    This is what ScribeTrace/ScriLog observed.

    Supported sources:
        auto
        traced
        observed
        scrilog_auto
        auto_detected
        signature
    """

    observed = _first_present_dict(
        record,
        [
            "traced",
            "auto",
            "observed",
            "scrilog_auto",
            "auto_detected",
            "signature",
        ],
    )

    if observed:
        return observed

    # Fallback:
    # the record itself may already be flat.
    return record


def extract_expected_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract expected/human-verified/manual truth.

    If this exists, ScriStatistics can mine tracing errors:

        expected endpoints = 5
        observed endpoints = 4

    If this does not exist, ScriStatistics only mines observed distributions.
    """

    return _first_present_dict(
        record,
        [
            "expected",
            "human_verified",
            "manual",
            "ground_truth",
            "verified",
            "truth",
        ],
    )


# ============================================================
# Feature flattening
# ============================================================

def _read_first_value(
    containers: List[Dict[str, Any]],
    possible_keys: List[str],
) -> Any:
    """
    Read the first available value from multiple dictionaries.

    This lets ScriStistics accept slightly different JSON shapes.
    """

    for container in containers:
        if not isinstance(container, dict):
            continue

        for key in possible_keys:
            if key in container and container[key] is not None:
                return container[key]

    return None


def flatten_features(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert UI/ScriLog/ScribeTrace-style records into one flat feature map.

    Output example:

        {
            "visual_ink_holes": 0,
            "closed_skeleton_loops": 0,
            "endpoints": 4,
            "junction_clusters": 2,
            "trace_paths": 5,
            "components": 1,
            "endpoint_top_left": 1,
            "endpoint_top_right": 1
        }
    """

    payload = _safe_dict(payload)

    flat: Dict[str, Any] = {}

    core = _safe_dict(payload.get("core_topology"))
    endpoint_quadrants = _safe_dict(payload.get("endpoint_quadrants"))
    junction_quadrants = _safe_dict(payload.get("junction_quadrants"))
    contacts = _safe_dict(payload.get("objective_contacts"))
    shape = _safe_dict(payload.get("shape_family"))

    # --------------------------------------------------------
    # Core topology
    # --------------------------------------------------------

    core_map: Dict[str, List[str]] = {
        "visual_ink_holes": [
            "visual_ink_holes",
            "ink_hole_count",
            "ink_holes",
        ],
        "closed_skeleton_loops": [
            "closed_skeleton_loops",
            "closed_loop_count",
            "loop_count",
            "loops",
        ],
        "endpoints": [
            "endpoints",
            "endpoint_count",
        ],
        "junction_clusters": [
            "junction_clusters",
            "junction_cluster_count",
            "junction_count",
            "junctions",
        ],
        "trace_paths": [
            "trace_paths",
            "path_count",
            "paths",
        ],
        "components": [
            "components",
            "component_count",
        ],
        "isolated_points": [
            "isolated_points",
            "isolated_point_count",
        ],
        "short_paths": [
            "short_paths",
            "short_path_count",
        ],
    }

    for output_key, possible_keys in core_map.items():
        value = _read_first_value(
            containers=[
                core,
                payload,
            ],
            possible_keys=possible_keys,
        )

        if value is not None:
            flat[output_key] = value

    # --------------------------------------------------------
    # Endpoint quadrants
    # --------------------------------------------------------

    endpoint_map: Dict[str, List[str]] = {
        "endpoint_top_left": [
            "top_left",
            "endpoint_top_left",
        ],
        "endpoint_top_right": [
            "top_right",
            "endpoint_top_right",
        ],
        "endpoint_bottom_left": [
            "bottom_left",
            "endpoint_bottom_left",
        ],
        "endpoint_bottom_right": [
            "bottom_right",
            "endpoint_bottom_right",
        ],
    }

    for output_key, possible_keys in endpoint_map.items():
        value = _read_first_value(
            containers=[
                endpoint_quadrants,
                payload,
            ],
            possible_keys=possible_keys,
        )

        if value is not None:
            flat[output_key] = value

    # --------------------------------------------------------
    # Junction quadrants
    # --------------------------------------------------------

    junction_map: Dict[str, List[str]] = {
        "junction_top_left": [
            "top_left",
            "junction_top_left",
        ],
        "junction_top_right": [
            "top_right",
            "junction_top_right",
        ],
        "junction_bottom_left": [
            "bottom_left",
            "junction_bottom_left",
        ],
        "junction_bottom_right": [
            "bottom_right",
            "junction_bottom_right",
        ],
    }

    for output_key, possible_keys in junction_map.items():
        value = _read_first_value(
            containers=[
                junction_quadrants,
                payload,
            ],
            possible_keys=possible_keys,
        )

        if value is not None:
            flat[output_key] = value

    # --------------------------------------------------------
    # Objective contacts
    # --------------------------------------------------------

    contact_map: Dict[str, List[str]] = {
        "touches_left_border": [
            "touches_left_border",
            "left_border_contact",
            "border_contact_left",
        ],
        "touches_right_border": [
            "touches_right_border",
            "right_border_contact",
            "border_contact_right",
        ],
        "touches_top_border": [
            "touches_top_border",
            "top_border_contact",
            "border_contact_top",
        ],
        "touches_bottom_border": [
            "touches_bottom_border",
            "bottom_border_contact",
            "border_contact_bottom",
        ],
    }

    for output_key, possible_keys in contact_map.items():
        value = _read_first_value(
            containers=[
                contacts,
                payload,
            ],
            possible_keys=possible_keys,
        )

        if value is not None:
            flat[output_key] = value

    # --------------------------------------------------------
    # Shape family
    # --------------------------------------------------------

    shape_map: Dict[str, List[str]] = {
        "wide_shape": [
            "wide_shape",
            "wide_family",
            "is_wide",
        ],
        "tall_shape": [
            "tall_shape",
            "tall_family",
            "is_tall",
        ],
    }

    for output_key, possible_keys in shape_map.items():
        value = _read_first_value(
            containers=[
                shape,
                payload,
            ],
            possible_keys=possible_keys,
        )

        if value is not None:
            flat[output_key] = value

    # --------------------------------------------------------
    # ScriLog-style signature support
    # --------------------------------------------------------

    signature = _safe_dict(payload.get("signature"))

    if signature:
        signature_map: Dict[str, str] = {
            "closed_skeleton_loops": "loop_count",
            "endpoints": "endpoint_count",
            "junction_clusters": "junction_count",
            "trace_paths": "path_count",
            "components": "component_count",
        }

        for output_key, signature_key in signature_map.items():
            if output_key not in flat and signature_key in signature:
                flat[output_key] = signature[signature_key]

    # --------------------------------------------------------
    # ScriLog derived family support
    # --------------------------------------------------------

    derived_families = payload.get("derived_families")

    if isinstance(derived_families, list):
        if "wide_shape" not in flat:
            flat["wide_shape"] = "wide" in derived_families

        if "tall_shape" not in flat:
            flat["tall_shape"] = "tall" in derived_families

    return flat


# ============================================================
# Distribution objects
# ============================================================

@dataclass
class FeatureDistribution:
    """
    Stores how often one observed feature value appears.

    Example:
        feature_name = "endpoints"

        counts:
            "4" -> 720
            "3" -> 288
            "5" -> 42

    Later this becomes:

        endpoints = 4 in 80% of cases
        endpoints = 3 in 32% of cases
    """

    feature_name: str
    importance: str
    counts: Counter = field(default_factory=Counter)

    def add(self, value: Any) -> None:
        """
        Add one observed value.
        """

        normalized_value = _json_value(value)
        self.counts[normalized_value] += 1

    def total(self) -> int:
        """
        Total number of recorded values for this feature.
        """

        return sum(self.counts.values())

    def most_common_value(self) -> Optional[str]:
        """
        Return the most common value for this feature.
        """

        if not self.counts:
            return None

        return self.counts.most_common(1)[0][0]

    def to_rows(self) -> List[Dict[str, Any]]:
        """
        Convert distribution into sorted rows.
        """

        total = self.total()
        rows: List[Dict[str, Any]] = []

        for value, count in self.counts.most_common():
            rows.append(
                {
                    "value": value,
                    "count": count,
                    "percent": _percent(
                        count=count,
                        total=total,
                    ),
                }
            )

        return rows

    def to_dict(self) -> Dict[str, Any]:
        """
        Export full feature distribution.
        """

        return {
            "feature": self.feature_name,
            "importance": self.importance,
            "total": self.total(),
            "most_common_value": self.most_common_value(),
            "values": self.to_rows(),
        }


@dataclass
class ErrorDistribution:
    """
    Stores expected-vs-observed mismatches for one feature.

    Example:
        feature_name = "endpoints"

        expected 5, observed 4 -> 27 times
        expected 4, observed 3 -> 11 times

    This only works when records contain both:

        expected / human_verified
        observed / traced / auto
    """

    feature_name: str
    importance: str
    counts: Counter = field(default_factory=Counter)

    def add(
        self,
        expected: Any,
        observed: Any,
    ) -> None:
        """
        Add one mismatch.

        If expected and observed are the same, do not store it as an error.
        """

        expected_value = _json_value(expected)
        observed_value = _json_value(observed)

        if expected_value == observed_value:
            return

        error_key = json.dumps(
            {
                "expected": expected_value,
                "observed": observed_value,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

        self.counts[error_key] += 1

    def total_errors(self) -> int:
        """
        Total number of mismatches for this feature.
        """

        return sum(self.counts.values())

    def to_rows(
        self,
        class_sample_count: int,
    ) -> List[Dict[str, Any]]:
        """
        Convert errors into sorted rows.

        percent_of_class means:

            this mismatch happened in X% of all samples for this letter
        """

        rows: List[Dict[str, Any]] = []

        for raw_key, count in self.counts.most_common():
            pair = json.loads(raw_key)

            rows.append(
                {
                    "expected": pair["expected"],
                    "observed": pair["observed"],
                    "count": count,
                    "percent_of_class": _percent(
                        count=count,
                        total=class_sample_count,
                    ),
                }
            )

        return rows

    def to_dict(
        self,
        class_sample_count: int,
    ) -> Dict[str, Any]:
        """
        Export full error distribution.
        """

        return {
            "feature": self.feature_name,
            "importance": self.importance,
            "error_count": self.total_errors(),
            "errors": self.to_rows(
                class_sample_count=class_sample_count,
            ),
        }


# ============================================================
# Class profile object
# ============================================================

IMPORTANCE_RANK: Dict[str, int] = {
    "high": 0,
    "medium": 1,
    "low": 2,
    "unknown": 3,
}


@dataclass
class ClassStats:
    """
    Stores the statistical profile for one Armenian letter/class.

    Example:
        raw_class_id = "8"
        label = "Թ"

    It stores:

        1. Observed feature distributions
            endpoints = 4 in 80%
            junction_clusters = 2 in 75%

        2. Expected-vs-observed mismatch distributions
            expected endpoints 5, observed 4 in 3%
    """

    label: str
    raw_class_id: str = "unknown"
    sample_count: int = 0
    feature_distributions: Dict[str, FeatureDistribution] = field(default_factory=dict)
    error_distributions: Dict[str, ErrorDistribution] = field(default_factory=dict)

    def add_observed(
        self,
        observed_features: Dict[str, Any],
    ) -> None:
        """
        Add one traced/observed sample into this class profile.
        """

        self.sample_count += 1

        for feature_name, value in observed_features.items():
            importance = FEATURE_IMPORTANCE.get(
                feature_name,
                "unknown",
            )

            if feature_name not in self.feature_distributions:
                self.feature_distributions[feature_name] = FeatureDistribution(
                    feature_name=feature_name,
                    importance=importance,
                )

            self.feature_distributions[feature_name].add(value)

    def add_expected_vs_observed(
        self,
        expected_features: Dict[str, Any],
        observed_features: Dict[str, Any],
    ) -> None:
        """
        Add expected-vs-observed mismatches.

        This only records mismatches for features that exist in both maps.
        """

        for feature_name, expected_value in expected_features.items():
            if feature_name not in observed_features:
                continue

            observed_value = observed_features[feature_name]

            importance = FEATURE_IMPORTANCE.get(
                feature_name,
                "unknown",
            )

            if feature_name not in self.error_distributions:
                self.error_distributions[feature_name] = ErrorDistribution(
                    feature_name=feature_name,
                    importance=importance,
                )

            self.error_distributions[feature_name].add(
                expected=expected_value,
                observed=observed_value,
            )

    def common_observations(
        self,
        threshold_percent: int,
    ) -> List[Dict[str, Any]]:
        """
        Return observations that happen often enough to be considered common.

        Example:
            endpoints = 4 in 80%
        """

        rows: List[Dict[str, Any]] = []

        for distribution in self.feature_distributions.values():
            total = distribution.total()

            if total <= 0:
                continue

            for value, count in distribution.counts.most_common():
                percent = _percent(
                    count=count,
                    total=total,
                )

                if percent >= threshold_percent:
                    rows.append(
                        {
                            "feature": distribution.feature_name,
                            "value": value,
                            "count": count,
                            "percent": percent,
                            "importance": distribution.importance,
                        }
                    )

        return self._sort_observation_rows(rows)

    def secondary_observations(
        self,
        low_percent: int,
        high_percent: int,
    ) -> List[Dict[str, Any]]:
        """
        Return observations that are not dominant, but still common enough
        to matter.

        Example:
            endpoints = 3 in 32%
        """

        rows: List[Dict[str, Any]] = []

        for distribution in self.feature_distributions.values():
            total = distribution.total()

            if total <= 0:
                continue

            for value, count in distribution.counts.most_common():
                percent = _percent(
                    count=count,
                    total=total,
                )

                if low_percent <= percent < high_percent:
                    rows.append(
                        {
                            "feature": distribution.feature_name,
                            "value": value,
                            "count": count,
                            "percent": percent,
                            "importance": distribution.importance,
                        }
                    )

        return self._sort_observation_rows(rows)

    def error_rows(
        self,
        min_percent: int,
        max_percent: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return expected-vs-observed errors in a percent range.

        Used for:
            common_errors
            minor_errors
        """

        rows: List[Dict[str, Any]] = []

        for distribution in self.error_distributions.values():
            error_rows = distribution.to_rows(
                class_sample_count=self.sample_count,
            )

            for error in error_rows:
                percent = int(error.get("percent_of_class", 0))

                if percent < min_percent:
                    continue

                if max_percent is not None and percent >= max_percent:
                    continue

                rows.append(
                    {
                        "feature": distribution.feature_name,
                        "importance": distribution.importance,
                        **error,
                    }
                )

        return self._sort_error_rows(rows)

    def _sort_observation_rows(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Sort observations by importance first, then by percent.
        """

        return sorted(
            rows,
            key=lambda row: (
                IMPORTANCE_RANK.get(
                    str(row.get("importance", "unknown")),
                    3,
                ),
                -int(row.get("percent", 0)),
                str(row.get("feature", "")),
                str(row.get("value", "")),
            ),
        )

    def _sort_error_rows(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Sort errors by importance first, then by percent.
        """

        return sorted(
            rows,
            key=lambda row: (
                IMPORTANCE_RANK.get(
                    str(row.get("importance", "unknown")),
                    3,
                ),
                -int(row.get("percent_of_class", 0)),
                str(row.get("feature", "")),
                str(row.get("expected", "")),
                str(row.get("observed", "")),
            ),
        )

    def to_dict(
        self,
        common_threshold: int,
        secondary_threshold: int,
        minor_error_threshold: int,
        common_error_threshold: int,
    ) -> Dict[str, Any]:
        """
        Export this class profile.
        """

        return {
            "label": self.label,
            "raw_class_id": self.raw_class_id,
            "sample_count": self.sample_count,

            "common_observations": self.common_observations(
                threshold_percent=common_threshold,
            ),

            "secondary_observations": self.secondary_observations(
                low_percent=secondary_threshold,
                high_percent=common_threshold,
            ),

            "feature_distributions": {
                feature_name: distribution.to_dict()
                for feature_name, distribution in sorted(
                    self.feature_distributions.items()
                )
            },

            "common_errors": self.error_rows(
                min_percent=common_error_threshold,
            ),

            "minor_errors": self.error_rows(
                min_percent=minor_error_threshold,
                max_percent=common_error_threshold,
            ),

            "error_distributions": {
                feature_name: distribution.to_dict(
                    class_sample_count=self.sample_count,
                )
                for feature_name, distribution in sorted(
                    self.error_distributions.items()
                )
            },
        }


# ============================================================
# Main miner
# ============================================================
def _class_sort_key(
    item: Tuple[str, ClassStats],
) -> Tuple[int, str]:
    """
    Sort class profiles by numeric raw_class_id when available.
    """

    label, class_stats = item
    raw_class_id = class_stats.raw_class_id

    if raw_class_id.isdigit():
        return int(raw_class_id), label

    return 999, label


class ScriStisticsMiner:
    """
    Main profile miner.

    It reads normalized records and builds one ClassStats profile
    per Armenian letter.

    Example:
        raw class label "8"
            -> label "Թ"
            -> profile stored under "Թ"
            -> raw_class_id preserved as "8"
    """

    def __init__(
        self,
        common_threshold: int = 70,
        secondary_threshold: int = 30,
        minor_error_threshold: int = 3,
        common_error_threshold: int = 10,
    ) -> None:
        self.common_threshold = common_threshold
        self.secondary_threshold = secondary_threshold
        self.minor_error_threshold = minor_error_threshold
        self.common_error_threshold = common_error_threshold

        self.classes: Dict[str, ClassStats] = {}

        self.skipped_records: int = 0
        self.unknown_label_records: int = 0

    def add_record(
        self,
        record: Dict[str, Any],
    ) -> None:
        """
        Add one annotation/ScriLog record.

        This does three things:

            1. Extract class label
            2. Flatten observed features
            3. Optionally compare expected vs observed
        """

        label = extract_label(record)
        raw_class_id = extract_raw_class_id(record)

        if label == "unknown":
            self.unknown_label_records += 1

        observed_payload = extract_observed_payload(record)
        expected_payload = extract_expected_payload(record)

        observed_features = flatten_features(observed_payload)
        expected_features = flatten_features(expected_payload)

        if not observed_features:
            self.skipped_records += 1
            return

        if label not in self.classes:
            self.classes[label] = ClassStats(
                label=label,
                raw_class_id=raw_class_id,
            )

        class_stats = self.classes[label]

        # If first records had unknown raw id but later one has real id,
        # upgrade the stored profile metadata.
        if (
            class_stats.raw_class_id == "unknown"
            and raw_class_id != "unknown"
        ):
            class_stats.raw_class_id = raw_class_id

        class_stats.add_observed(
            observed_features=observed_features,
        )

        if expected_features:
            class_stats.add_expected_vs_observed(
                expected_features=expected_features,
                observed_features=observed_features,
            )

    def add_records(
        self,
        records: List[Dict[str, Any]],
    ) -> None:
        """
        Add many records.
        """

        for record in records:
            self.add_record(record)

    def total_sample_count(self) -> int:
        """
        Total accepted samples across all class profiles.
        """

        return sum(
            class_stats.sample_count
            for class_stats in self.classes.values()
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Export the full mined profile database.
        """

        return {
            "scrististics_version": SCRISTISTICS_VERSION,

            "summary": {
                "class_count": len(self.classes),
                "sample_count": self.total_sample_count(),
                "skipped_records": self.skipped_records,
                "unknown_label_records": self.unknown_label_records,
                "common_threshold_percent": self.common_threshold,
                "secondary_threshold_percent": self.secondary_threshold,
                "minor_error_threshold_percent": self.minor_error_threshold,
                "common_error_threshold_percent": self.common_error_threshold,
            },

            "feature_importance": dict(FEATURE_IMPORTANCE),

            "class_id_to_armenian": dict(CLASS_ID_TO_ARMENIAN),

            "classes": {
                label: class_stats.to_dict(
                    common_threshold=self.common_threshold,
                    secondary_threshold=self.secondary_threshold,
                    minor_error_threshold=self.minor_error_threshold,
                    common_error_threshold=self.common_error_threshold,
                )
                for label, class_stats in sorted(
                    self.classes.items(),
                    key=_class_sort_key,
                )
            },
        }
    

# ============================================================
# Minor error watchlist / trend detection
# ============================================================

def _collect_profile_errors(
    class_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Collect both minor and common errors from one class profile.

    The profile export has:

        minor_errors: [...]
        common_errors: [...]

    For trend detection, both matter.
    """

    minor_errors = _safe_list(
        class_payload.get("minor_errors")
    )

    common_errors = _safe_list(
        class_payload.get("common_errors")
    )

    errors: List[Dict[str, Any]] = []

    for error in minor_errors + common_errors:
        if isinstance(error, dict):
            errors.append(error)

    return errors


def _error_identity(
    error: Dict[str, Any],
) -> Tuple[str, str, str]:
    """
    Build stable identity for one error type.

    Example:
        feature=endpoints
        expected=5
        observed=4

    This lets us compare the same error across versions.
    """

    return (
        str(error.get("feature", "unknown")),
        str(error.get("expected", "unknown")),
        str(error.get("observed", "unknown")),
    )


def build_minor_error_watchlist(
    current_profile: Dict[str, Any],
    previous_profile: Optional[Dict[str, Any]] = None,
    watch_percent: int = 10,
    trend_delta_percent: int = 5,
) -> List[Dict[str, Any]]:
    """
    Build a watchlist of errors that deserve attention.

    Main use case:

        v0.1:
            endpoints expected 5 observed 4 = 3%

        v0.2:
            endpoints expected 5 observed 4 = 10%

        Result:
            add to watchlist

    Rules:
        1. If current error percent >= watch_percent, flag it.
        2. If current percent increased by trend_delta_percent or more, flag it.
    """

    current_classes = _safe_dict(
        current_profile.get("classes")
    )

    previous_classes = {}

    if previous_profile is not None:
        previous_classes = _safe_dict(
            previous_profile.get("classes")
        )

    watchlist: List[Dict[str, Any]] = []

    for label, current_class_payload in current_classes.items():
        current_class_payload = _safe_dict(current_class_payload)

        current_errors = _collect_profile_errors(
            class_payload=current_class_payload,
        )

        previous_class_payload = _safe_dict(
            previous_classes.get(label)
        )

        previous_errors = _collect_profile_errors(
            class_payload=previous_class_payload,
        )

        previous_error_index: Dict[Tuple[str, str, str], int] = {}

        for previous_error in previous_errors:
            key = _error_identity(previous_error)

            previous_error_index[key] = int(
                previous_error.get("percent_of_class", 0)
            )

        for current_error in current_errors:
            key = _error_identity(current_error)

            current_percent = int(
                current_error.get("percent_of_class", 0)
            )

            previous_percent = previous_error_index.get(
                key,
                0,
            )

            delta_percent = current_percent - previous_percent

            should_watch = (
                current_percent >= watch_percent
                or delta_percent >= trend_delta_percent
            )

            if not should_watch:
                continue

            if current_percent >= watch_percent:
                reason = "reached_watch_threshold"
            else:
                reason = "rising_minor_error"

            watchlist.append(
                {
                    "label": label,
                    "raw_class_id": current_class_payload.get(
                        "raw_class_id",
                        "unknown",
                    ),
                    "feature": key[0],
                    "expected": key[1],
                    "observed": key[2],
                    "importance": current_error.get(
                        "importance",
                        "unknown",
                    ),
                    "previous_percent": previous_percent,
                    "current_percent": current_percent,
                    "delta_percent": delta_percent,
                    "count": current_error.get("count", 0),
                    "reason": reason,
                }
            )

    return sorted(
        watchlist,
        key=lambda row: (
            IMPORTANCE_RANK.get(
                str(row.get("importance", "unknown")),
                3,
            ),
            -int(row.get("current_percent", 0)),
            -int(row.get("delta_percent", 0)),
            str(row.get("label", "")),
            str(row.get("feature", "")),
        ),
    )


# ============================================================
# Runner
# ============================================================

def run_miner(
    input_path: Path,
    output_path: Path,
    previous_path: Optional[Path] = None,
    common_threshold: int = 70,
    secondary_threshold: int = 30,
    minor_error_threshold: int = 3,
    common_error_threshold: int = 10,
    watch_percent: int = 10,
    trend_delta_percent: int = 5,
) -> Dict[str, Any]:
    """
    Run ScriStistics mining from a JSON file or directory.

    Input:
        - one JSON file
        - or a directory containing JSON files

    Output:
        - one profile JSON file
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input path does not exist: {input_path}"
        )

    miner = ScriStisticsMiner(
        common_threshold=common_threshold,
        secondary_threshold=secondary_threshold,
        minor_error_threshold=minor_error_threshold,
        common_error_threshold=common_error_threshold,
    )

    loaded_files = 0

    for json_path in _iter_json_files(input_path):
        payload = _load_json(json_path)
        records = extract_records(payload)

        miner.add_records(records)
        loaded_files += 1

    result = miner.to_dict()

    previous_profile: Optional[Dict[str, Any]] = None

    if previous_path is not None and previous_path.exists():
        previous_profile = _load_json(previous_path)

    result["minor_error_watchlist"] = build_minor_error_watchlist(
        current_profile=result,
        previous_profile=previous_profile,
        watch_percent=watch_percent,
        trend_delta_percent=trend_delta_percent,
    )

    result["summary"]["loaded_files"] = loaded_files
    result["summary"]["watch_percent"] = watch_percent
    result["summary"]["trend_delta_percent"] = trend_delta_percent

    _write_json(
        path=output_path,
        payload=result,
    )

    return result


# ============================================================
# CLI
# ============================================================

def build_cli_parser() -> argparse.ArgumentParser:
    """
    Build command line interface.
    """

    parser = argparse.ArgumentParser(
        description="Mine ScriStistics profiles from ScriLog annotation JSON."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input annotation JSON file or directory.",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output ScriStistics profile JSON path.",
    )

    parser.add_argument(
        "--previous",
        default="",
        help="Optional previous profile JSON for trend comparison.",
    )

    parser.add_argument(
        "--common-threshold",
        type=int,
        default=70,
        help="Percent threshold for common observations.",
    )

    parser.add_argument(
        "--secondary-threshold",
        type=int,
        default=30,
        help="Percent threshold for secondary observations.",
    )

    parser.add_argument(
        "--minor-error-threshold",
        type=int,
        default=3,
        help="Percent threshold for minor errors.",
    )

    parser.add_argument(
        "--common-error-threshold",
        type=int,
        default=10,
        help="Percent threshold for common errors.",
    )

    parser.add_argument(
        "--watch-percent",
        type=int,
        default=10,
        help="Error percent that automatically enters the watchlist.",
    )

    parser.add_argument(
        "--trend-delta-percent",
        type=int,
        default=5,
        help="Increase percent needed to flag a rising minor error.",
    )

    return parser


def main() -> None:
    """
    CLI entry point.
    """

    parser = build_cli_parser()
    args = parser.parse_args()

    previous_path: Optional[Path] = None

    if args.previous:
        previous_path = Path(args.previous)

    result = run_miner(
        input_path=Path(args.input),
        output_path=Path(args.out),
        previous_path=previous_path,
        common_threshold=args.common_threshold,
        secondary_threshold=args.secondary_threshold,
        minor_error_threshold=args.minor_error_threshold,
        common_error_threshold=args.common_error_threshold,
        watch_percent=args.watch_percent,
        trend_delta_percent=args.trend_delta_percent,
    )

    status = {
        "status": "completed",
        "output_path": args.out,
        "loaded_files": result["summary"]["loaded_files"],
        "class_count": result["summary"]["class_count"],
        "sample_count": result["summary"]["sample_count"],
        "skipped_records": result["summary"]["skipped_records"],
        "unknown_label_records": result["summary"]["unknown_label_records"],
        "watchlist_count": len(result.get("minor_error_watchlist", [])),
    }

    print(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()