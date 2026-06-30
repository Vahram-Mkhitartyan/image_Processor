"""Baseline trainer for ScribeJudge rows.

This is intentionally small: it trains transparent RandomForest baselines over
ScribeJudge features once enough labeled synthetic rows exist. The goal is not
final magic yet; it is a measurable before/after referee loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


TOKEN_FEATURES = [
    "matrix_score",
    "judge_score",
    "average_selected_position_score",
    "average_confusion_risk",
    "position_count",
    "suspicious_high_confidence_count",
    "suspicious_low_confidence_count",
    "backup_recovery_opportunity_count",
    "truth_missing_from_candidates_count",
]

POSITION_FEATURES = [
    "selected_score",
    "judged_selected_score",
    "source_count",
    "candidate_count",
    "selected_confusion_drop",
    "best_alternative_boost",
    "best_confusion_risk",
    "has_conflicts",
    "suspicious_high_confidence",
    "suspicious_low_confidence",
]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def token_vector(row: dict) -> list[float]:
    features = row.get("features") or {}
    enriched = dict(features)
    enriched["backup_recovery_opportunity_count"] = row.get("backup_recovery_opportunity_count")
    enriched["truth_missing_from_candidates_count"] = row.get("truth_missing_from_candidates_count")
    return [_safe_float(enriched.get(name)) for name in TOKEN_FEATURES]


def position_vector(position: dict) -> list[float]:
    features = position.get("features") or {}
    best_risk = _safe_float(features.get("best_confusion_risk"))
    enriched = {
        "selected_score": position.get("selected_score"),
        "judged_selected_score": position.get("judged_selected_score"),
        "source_count": position.get("source_count"),
        "candidate_count": position.get("candidate_count"),
        "selected_confusion_drop": position.get("selected_confusion_drop"),
        "best_alternative_boost": position.get("best_alternative_boost"),
        "best_confusion_risk": best_risk,
        "has_conflicts": 1.0 if position.get("has_conflicts") else 0.0,
        "suspicious_high_confidence": 1.0 if position.get("suspicious_high_confidence") else 0.0,
        "suspicious_low_confidence": 1.0 if position.get("suspicious_low_confidence") else 0.0,
    }
    return [_safe_float(enriched.get(name)) for name in POSITION_FEATURES]


def build_token_dataset(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = []
    y = []
    for row in rows:
        if row.get("is_exact") is None:
            continue
        x.append(token_vector(row))
        y.append(1 if row.get("is_exact") is True else 0)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64)


def build_position_dataset(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = []
    y = []
    for row in rows:
        targets = row.get("position_targets") or []
        positions = row.get("positions") or []
        for index, position in enumerate(positions):
            if index >= len(targets):
                continue
            target = targets[index]
            if target.get("selected_correct") is None:
                continue
            x.append(position_vector(position))
            y.append(1 if target.get("selected_correct") is True else 0)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64)


def train_classifier(x: np.ndarray, y: np.ndarray, seed: int, min_rows: int) -> tuple[dict, RandomForestClassifier | None]:
    if len(y) < min_rows:
        return {"status": "not_enough_rows", "row_count": int(len(y)), "min_rows": min_rows}, None
    classes = sorted(set(int(v) for v in y.tolist()))
    if len(classes) < 2:
        return {"status": "one_class_only", "row_count": int(len(y)), "classes": classes}, None

    stratify = y if min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=stratify,
    )
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=2,
    )
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    return {
        "status": "trained",
        "row_count": int(len(y)),
        "train_count": int(len(y_train)),
        "test_count": int(len(y_test)),
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "classification_report": classification_report(
            y_test,
            test_pred,
            output_dict=True,
            zero_division=0,
        ),
    }, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline ScribeJudge models.")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--report", default="reports/scribejudge_v0_1/training_report.json")
    parser.add_argument("--model-dir", default="models/scribejudge_v0_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-token-rows", type=int, default=40)
    parser.add_argument("--min-position-rows", type=int, default=120)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_jsonl)
    rows = list(iter_rows(dataset_path))
    token_x, token_y = build_token_dataset(rows)
    position_x, position_y = build_position_dataset(rows)

    token_report, token_model = train_classifier(
        token_x,
        token_y,
        seed=args.seed,
        min_rows=args.min_token_rows,
    )
    position_report, position_model = train_classifier(
        position_x,
        position_y,
        seed=args.seed,
        min_rows=args.min_position_rows,
    )

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    token_model_path = None
    position_model_path = None
    if token_model is not None:
        token_model_path = model_dir / "scribejudge_token_exact_rf.joblib"
        joblib.dump({"model": token_model, "features": TOKEN_FEATURES}, token_model_path)
    if position_model is not None:
        position_model_path = model_dir / "scribejudge_position_selected_correct_rf.joblib"
        joblib.dump({"model": position_model, "features": POSITION_FEATURES}, position_model_path)

    labeled = [row for row in rows if row.get("is_exact") is not None]
    positives = sum(1 for row in labeled if row.get("is_exact") is True)
    report = {
        "model_name": "scribejudge_v0_1_baseline",
        "status": "completed",
        "dataset_jsonl": str(dataset_path),
        "row_count": len(rows),
        "labeled_row_count": len(labeled),
        "exact_positive_count": positives,
        "exact_negative_count": len(labeled) - positives,
        "token_model": token_report,
        "position_model": position_report,
        "token_model_path": str(token_model_path) if token_model_path else None,
        "position_model_path": str(position_model_path) if position_model_path else None,
        "token_feature_contract": TOKEN_FEATURES,
        "position_feature_contract": POSITION_FEATURES,
        "next_step": (
            "Scale synthetic generation, then evaluate whether position model "
            "can safely promote backups without increasing bad corrections."
        ),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("ScribeJudge baseline report:", report_path)
    print("Rows:", len(rows), "token labels:", len(token_y), "position labels:", len(position_y))
    print("Token model:", token_report.get("status"))
    print("Position model:", position_report.get("status"))


if __name__ == "__main__":
    main()
