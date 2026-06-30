"""Confusion-history memory for ScribeJudge.

This module turns existing confusion reports into a small queryable memory.
It is intentionally model-agnostic: Random Forest, CNN, ScriLog, or future
experts can all contribute records in the same shape.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


DEFAULT_CONFUSION_SOURCES = [
    {
        "source": "scribetrace_random_forest_v4_0",
        "kind": "confusion_pairs",
        "path": "reports/scribetrace_random_forest_v4_0/confusion_pairs.json",
        "weight": 1.0,
    },
    {
        "source": "glyph_classifier_v0_2_aristotel",
        "kind": "confusion_matrix",
        "path": "reports/glyph_classifier_v0_2_aristotel/confusion_matrix.csv",
        "label_map_path": "models/glyph_classifier_v0_2_aristotel/numeric_label_map.json",
        "weight": 0.7,
    },
]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_label_map(path: Path | None) -> dict[int, str]:
    if path is None or not path.is_file():
        return {}
    payload = _load_json(path)
    return {int(key): str(value) for key, value in payload.items()}


def _add_memory_record(memory: dict, predicted: str, true: str, record: dict) -> None:
    if not predicted or not true or predicted == true:
        return
    key = (str(predicted), str(true))
    slot = memory.setdefault(
        key,
        {
            "predicted": str(predicted),
            "true": str(true),
            "weighted_count": 0.0,
            "max_share_of_true_class": 0.0,
            "sources": [],
        },
    )
    slot["weighted_count"] += _safe_float(record.get("weighted_count"))
    slot["max_share_of_true_class"] = max(
        _safe_float(slot.get("max_share_of_true_class")),
        _safe_float(record.get("share_of_true_class")),
    )
    slot["sources"].append(record)


def _load_confusion_pairs(source: dict, base_dir: Path, memory: dict) -> int:
    path = (base_dir / source.get("path", "")).resolve()
    payload = _load_json(path)
    pairs = payload.get("pairs") if isinstance(payload.get("pairs"), list) else []
    weight = _safe_float(source.get("weight"), 1.0)
    loaded = 0
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        predicted = str(pair.get("predicted_label") or "")
        true = str(pair.get("true_label") or "")
        count = _safe_float(pair.get("count"))
        share = _safe_float(pair.get("share_of_true_class"))
        _add_memory_record(
            memory,
            predicted=predicted,
            true=true,
            record={
                "source": source.get("source"),
                "kind": "confusion_pairs",
                "count": count,
                "weighted_count": count * weight,
                "share_of_true_class": share,
                "path": str(path),
            },
        )
        loaded += 1
    return loaded


def _load_confusion_matrix(source: dict, base_dir: Path, memory: dict) -> int:
    matrix_path = (base_dir / source.get("path", "")).resolve()
    label_map_path = source.get("label_map_path")
    labels = _load_label_map((base_dir / label_map_path).resolve() if label_map_path else None)
    if not matrix_path.is_file() or not labels:
        return 0
    weight = _safe_float(source.get("weight"), 1.0)
    rows = []
    with matrix_path.open("r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            rows.append([_safe_int(cell) for cell in row])
    loaded = 0
    for true_index, row in enumerate(rows):
        true = labels.get(true_index)
        if not true:
            continue
        true_support = sum(row)
        if true_support <= 0:
            continue
        for predicted_index, count in enumerate(row):
            if predicted_index == true_index or count <= 0:
                continue
            predicted = labels.get(predicted_index)
            if not predicted:
                continue
            share = count / max(true_support, 1)
            _add_memory_record(
                memory,
                predicted=predicted,
                true=true,
                record={
                    "source": source.get("source"),
                    "kind": "confusion_matrix",
                    "count": count,
                    "weighted_count": count * weight,
                    "share_of_true_class": share,
                    "path": str(matrix_path),
                },
            )
            loaded += 1
    return loaded


class ConfusionMemory:
    """Small query layer over model confusion history."""

    def __init__(self, records: dict[tuple[str, str], dict] | None = None, summary: dict | None = None):
        self.records = records or {}
        self.summary = summary or {}

    @classmethod
    def from_sources(cls, sources: list[dict] | None = None, base_dir: str | Path = ".") -> "ConfusionMemory":
        base = Path(base_dir).resolve()
        sources = sources or DEFAULT_CONFUSION_SOURCES
        memory: dict[tuple[str, str], dict] = {}
        loaded_by_source = []
        for source in sources:
            if not isinstance(source, dict) or source.get("enabled") is False:
                continue
            kind = source.get("kind")
            if kind == "confusion_pairs":
                loaded = _load_confusion_pairs(source, base, memory)
            elif kind == "confusion_matrix":
                loaded = _load_confusion_matrix(source, base, memory)
            else:
                loaded = 0
            loaded_by_source.append({"source": source.get("source"), "kind": kind, "records": loaded})
        return cls(
            records=memory,
            summary={
                "source_count": len(loaded_by_source),
                "confusion_pair_count": len(memory),
                "loaded_by_source": loaded_by_source,
            },
        )

    def predicted_to_true_risk(self, predicted: str, truth_candidate: str) -> dict:
        """Return history risk that predicted is actually truth_candidate."""

        record = self.records.get((str(predicted), str(truth_candidate)))
        if not record:
            return {
                "predicted": str(predicted),
                "true": str(truth_candidate),
                "risk": 0.0,
                "weighted_count": 0.0,
                "max_share_of_true_class": 0.0,
                "sources": [],
            }
        weighted_count = _safe_float(record.get("weighted_count"))
        share = _safe_float(record.get("max_share_of_true_class"))
        # Count gives memory strength; share catches rare but brutal pairs.
        risk = min(1.0, (weighted_count / 20.0) * 0.55 + share * 0.45)
        return {
            "predicted": str(predicted),
            "true": str(truth_candidate),
            "risk": risk,
            "weighted_count": weighted_count,
            "max_share_of_true_class": share,
            "sources": record.get("sources") or [],
        }

    def top_alternatives_for_prediction(self, predicted: str, limit: int = 8) -> list[dict]:
        """Return likely true alternatives when a model predicts `predicted`."""

        rows = [
            self.predicted_to_true_risk(predicted, true)
            for pred, true in self.records
            if pred == str(predicted)
        ]
        rows.sort(key=lambda item: (-_safe_float(item.get("risk")), str(item.get("true"))))
        return rows[:limit]

    def to_report(self) -> dict:
        return {
            "version": "scribejudge_confusion_memory_v0_1",
            "summary": self.summary,
        }
