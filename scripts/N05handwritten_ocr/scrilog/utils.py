"""Defensive conversion and dictionary helpers for ScribeTrace payloads."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

def _as_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to int.

    ScriLog will read JSON coming from ScribeTrace.
    Some fields may be missing, None, strings, or floats.
    This helper prevents one bad field from killing the parser.
    """
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default
    


def _as_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.

    Also rejects NaN and infinity because those are dangerous
    for JSON reports and rule comparisons.
    """
    try:
        if value is None:
            return default

        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return default

        return value

    except Exception:
        return default
    

def _as_bool(value: Any, default: bool = False) -> bool:
    """
    Safely convert common JSON-ish values to bool.
    """
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):
        return value.strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

    return default


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

def _safe_dict(value: Any) -> Dict[str, Any]:
    """
    Return value if it is a dict, otherwise return empty dict.
    """
    if isinstance(value, dict):
        return value

    return {}

def _first_present(
    mapping: Dict[str, Any],
    keys: Iterable[str],
    default: Any = None,
) -> Any:
    """
    Return the first existing non-None key from a dictionary.
    """
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]

    return default

def _deep_get(
    mapping: Dict[str, Any],
    path: Iterable[str],
    default: Any = None,
) -> Any:
    """
    Safely read nested dictionaries.

    Example:
        _deep_get(payload, ["selected_hypothesis", "topology", "loop_count"])
    """
    current: Any = mapping

    for part in path:
        if not isinstance(current, dict):
            return default

        if part not in current:
            return default

        current = current[part]

    return current


def _merge_dicts(*items: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge dictionaries left to right.

    Later dictionaries override earlier ones.
    Used when ScribeTrace may store fields in several possible places.
    """
    merged: Dict[str, Any] = {}

    for item in items:
        if isinstance(item, dict):
            merged.update(item)

    return merged
