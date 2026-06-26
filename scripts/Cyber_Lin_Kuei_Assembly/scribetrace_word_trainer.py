"""Train a raw ScribeTrace word-level splitter/recognizer baseline.

This experiment reuses the synthetic word renderer from word_level_ocr_trainer,
but replaces pixel-CRNN input with raw ScribeTrace geometry features.

v0.1 goal:
    rendered synthetic word -> raw ScribeTrace vector -> simple ML heads

The model intentionally does not use theoretical reconstruction, ANTAR, ScriLog,
or Scrististics. Those stay downstream after this module proposes cleaner
letter spans.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Iterable

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples.*",
    category=UserWarning,
)

LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(LOCAL_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

try:
    from .word_level_ocr_trainer import (
        PROJECT_ROOT,
        build_tail_profiles,
        build_token_maps,
        collect_glyph_paths,
        load_json,
        load_label_map,
        load_word_samples,
        render_synthetic_word,
        resolve_path,
        save_json,
    )
except ImportError:
    from word_level_ocr_trainer import (  # type: ignore
        PROJECT_ROOT,
        build_tail_profiles,
        build_token_maps,
        collect_glyph_paths,
        load_json,
        load_label_map,
        load_word_samples,
        render_synthetic_word,
        resolve_path,
        save_json,
    )

try:
    from scripts.N05handwritten_ocr.scribetrace.expert import run_scribetrace
    from scripts.N05handwritten_ocr.scribetrace.trace_models import TraceInput
except ImportError:
    from N05handwritten_ocr.scribetrace.expert import run_scribetrace  # type: ignore
    from N05handwritten_ocr.scribetrace.trace_models import TraceInput  # type: ignore


DEFAULT_SETTINGS_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "Cyber_Lin_Kuei_Assembly"
    / "scribetrace_word_settings.json"
)


def _open_text(path: Path, mode: str):
    """Open plain or gzip-compressed JSONL using text mode."""

    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def _base_trace_settings(settings: dict) -> dict:
    """Return raw ScribeTrace settings for word-level feature export."""

    trace = dict(settings.get("scribetrace", {}))
    trace.setdefault("enabled", True)
    trace.setdefault("save_debug", False)
    trace.setdefault("save_json", False)
    trace.setdefault("ink_threshold_mode", "binary")
    trace.setdefault("fixed_threshold_value", 128)
    trace.setdefault("minimum_ink_pixels", 4)
    trace.setdefault("maximum_component_count_for_full_trace", 120)
    trace.setdefault("enable_theoretical_reconstruction", False)
    return trace


def _write_temp_word_image(image: np.ndarray, index: int) -> Path:
    """Write one rendered word to /tmp so ScribeTrace can use its path contract."""

    temp_dir = Path(tempfile.gettempdir()) / "scribetrace_word_export"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"synthetic_word_{index:08d}.png"
    cv2.imwrite(str(path), image)
    return path


def _boundary_bins(split_x_positions: Iterable[int], width: int, bin_count: int) -> list[int]:
    """Convert split x positions into a fixed multi-label boundary vector."""

    bins = [0 for _ in range(bin_count)]
    if width <= 0:
        return bins
    for position in split_x_positions:
        index = int(round((float(position) / max(1.0, float(width))) * (bin_count - 1)))
        index = max(0, min(bin_count - 1, index))
        bins[index] = 1
    return bins


def _pad_tokens(token_ids: Iterable[int], max_length: int) -> list[int]:
    """Pad token IDs with zero so every row has the same sequence target length."""

    tokens = [int(token) for token in token_ids][:max_length]
    return tokens + [0 for _ in range(max_length - len(tokens))]


def _trace_rendered_word(
    image: np.ndarray,
    index: int,
    trace_settings: dict,
    output_dir: Path,
):
    """Run raw ScribeTrace on one rendered synthetic word."""

    image_path = _write_temp_word_image(image, index)
    trace_input = TraceInput(
        crop_path=str(image_path),
        visual_crop_path=str(image_path),
        output_dir=str(output_dir),
        document_id="synthetic_word_training",
        text_unit_id=f"word_{index:08d}",
        layer="synthetic",
    )
    return run_scribetrace(trace_input, settings=trace_settings)


def _json_safe_float_list(values: Iterable[float]) -> list[float]:
    """Convert feature values to finite JSON-safe floats."""

    safe = []
    for value in values:
        number = float(value)
        if not np.isfinite(number):
            number = 0.0
        safe.append(number)
    return safe


def export_dataset(settings: dict, limit: int | None = None) -> dict:
    """Export synthetic word ScribeTrace vectors to JSONL.

    Args:
        settings: Experiment settings.
        limit: Optional sample cap for smoke/debug runs.

    Returns:
        JSON-safe export report.
    """

    start = time.time()
    seed = int(settings.get("random_seed", 42))
    rng = random.Random(seed)
    dataset_settings = settings["dataset"]
    output_settings = settings["output"]
    max_sequence_length = int(dataset_settings.get("max_sequence_length", 18))
    boundary_bin_count = int(dataset_settings.get("boundary_bin_count", 32))
    sample_count = int(limit or dataset_settings.get("samples", 2000))

    label_map = load_label_map(dataset_settings["label_map_path"])
    char_to_token, token_to_char = build_token_maps(label_map)
    tail_profiles = build_tail_profiles(token_to_char)
    words = load_word_samples(settings, char_to_token)
    glyph_paths = collect_glyph_paths(dataset_settings["matenadata_dir"], label_map)
    trace_settings = _base_trace_settings(settings)
    export_path = resolve_path(output_settings["dataset_jsonl"])
    trace_output_dir = resolve_path(output_settings.get("trace_temp_dir", "temp_processing/scribetrace_word_trace"))
    export_path.parent.mkdir(parents=True, exist_ok=True)
    trace_output_dir.mkdir(parents=True, exist_ok=True)

    accepted = 0
    skipped = 0
    feature_names = None
    status_counts: dict[str, int] = {}

    with _open_text(export_path, "wt") as file:
        for index in range(sample_count):
            sample = rng.choice(words)
            rendered = render_synthetic_word(
                sample=sample,
                glyph_paths=glyph_paths,
                tail_profiles=tail_profiles,
                rendering=settings["rendering"],
                rng=rng,
            )
            trace_result = _trace_rendered_word(
                rendered.image,
                index,
                trace_settings,
                trace_output_dir,
            )
            status = str(trace_result.status)
            status_counts[status] = status_counts.get(status, 0) + 1
            if status != "completed" or trace_result.feature_vector is None:
                skipped += 1
                continue

            vector = _json_safe_float_list(trace_result.feature_vector.vector)
            names = list(trace_result.feature_vector.feature_names)
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                skipped += 1
                continue

            row = {
                "sample_id": f"synthetic_word_{index:08d}",
                "text": sample.text,
                "token_ids": list(sample.token_ids),
                "padded_token_ids": _pad_tokens(sample.token_ids, max_sequence_length),
                "length": len(sample.token_ids),
                "bridge_count": int(rendered.bridge_count),
                "transition_count": int(rendered.transition_count),
                "image_width": int(rendered.image.shape[1]),
                "image_height": int(rendered.image.shape[0]),
                "split_x_positions": list(rendered.split_x_positions),
                "boundary_bins": _boundary_bins(
                    rendered.split_x_positions,
                    rendered.image.shape[1],
                    boundary_bin_count,
                ),
                "feature_names": names,
                "features": vector,
                "sequence_string": trace_result.feature_vector.sequence_string,
                "trace_metrics": {
                    "component_count": trace_result.metrics.get("component_count"),
                    "path_count": trace_result.metrics.get("path_count"),
                    "skeleton_point_count": trace_result.metrics.get("skeleton_point_count"),
                    "landmark_count": trace_result.metrics.get("landmark_count"),
                },
            }
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            accepted += 1

    schema = {
        "feature_names": feature_names or [],
        "max_sequence_length": max_sequence_length,
        "boundary_bin_count": boundary_bin_count,
        "token_to_char": {str(key): value for key, value in token_to_char.items()},
        "target_contract": {
            "length": "token count",
            "padded_token_ids": "zero-padded token IDs",
            "boundary_bins": "fixed-width normalized split position bins",
        },
    }
    schema_path = resolve_path(output_settings["schema_path"])
    save_json(schema, schema_path)
    report = {
        "status": "completed",
        "dataset_jsonl": str(export_path),
        "schema_path": str(schema_path),
        "requested_samples": sample_count,
        "accepted_samples": accepted,
        "skipped_samples": skipped,
        "trace_status_counts": status_counts,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    report_path = resolve_path(output_settings["export_report_path"])
    save_json(report, report_path)
    print(f"exported: {accepted}/{sample_count}")
    print(f"dataset: {export_path}")
    print(f"schema:  {schema_path}")
    return report


def _load_dataset(path: Path) -> list[dict]:
    """Load exported JSONL rows into memory for the baseline trainer."""

    rows = []
    with _open_text(path, "rt") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _feature_matrix(rows: list[dict]) -> np.ndarray:
    """Return the numeric ScribeTrace feature matrix."""

    return np.asarray([row["features"] for row in rows], dtype=np.float32)


def _target_matrix(rows: list[dict], key: str) -> np.ndarray:
    """Return a stacked integer target matrix."""

    return np.asarray([row[key] for row in rows], dtype=np.int64)


def _decode_tokens(tokens: Iterable[int], token_to_char: dict[str, str]) -> str:
    """Convert padded token IDs back to text."""

    chars = []
    for token in tokens:
        token = int(token)
        if token <= 0:
            continue
        chars.append(token_to_char.get(str(token), ""))
    return "".join(chars)


def _boundary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute simple multi-label boundary precision/recall/F1."""

    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    exact = float(np.all(y_true == y_pred, axis=1).mean())
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_boundary_vector_accuracy": exact,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def train_baseline(settings: dict) -> dict:
    """Train the first ScribeTrace-word RandomForest baseline."""

    start = time.time()
    output = settings["output"]
    dataset_path = resolve_path(output["dataset_jsonl"])
    schema_path = resolve_path(output["schema_path"])
    model_dir = resolve_path(output["model_dir"])
    report_dir = resolve_path(output["report_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_dataset(dataset_path)
    schema = load_json(schema_path)
    seed = int(settings.get("random_seed", 42))
    train_rows, test_rows = train_test_split(
        rows,
        test_size=float(settings.get("training", {}).get("test_ratio", 0.2)),
        random_state=seed,
    )
    train_rows, val_rows = train_test_split(
        train_rows,
        test_size=float(settings.get("training", {}).get("validation_ratio", 0.125)),
        random_state=seed,
    )

    x_train = _feature_matrix(train_rows)
    x_val = _feature_matrix(val_rows)
    x_test = _feature_matrix(test_rows)
    y_length_train = np.asarray([row["length"] for row in train_rows], dtype=np.int64)
    y_length_val = np.asarray([row["length"] for row in val_rows], dtype=np.int64)
    y_length_test = np.asarray([row["length"] for row in test_rows], dtype=np.int64)
    y_seq_train = _target_matrix(train_rows, "padded_token_ids")
    y_seq_val = _target_matrix(val_rows, "padded_token_ids")
    y_seq_test = _target_matrix(test_rows, "padded_token_ids")
    y_boundary_train = _target_matrix(train_rows, "boundary_bins")
    y_boundary_val = _target_matrix(val_rows, "boundary_bins")
    y_boundary_test = _target_matrix(test_rows, "boundary_bins")

    forest_settings = dict(settings.get("training", {}).get("random_forest", {}))
    base_params = {
        "n_estimators": int(forest_settings.get("n_estimators", 240)),
        "max_depth": forest_settings.get("max_depth", 28),
        "min_samples_leaf": int(forest_settings.get("min_samples_leaf", 1)),
        "max_features": forest_settings.get("max_features", "sqrt"),
        "n_jobs": int(forest_settings.get("n_jobs", 2)),
        "random_state": seed,
        "class_weight": forest_settings.get("class_weight", None),
    }

    length_model = RandomForestClassifier(**base_params)
    sequence_model = MultiOutputClassifier(RandomForestClassifier(**base_params))
    boundary_model = MultiOutputClassifier(RandomForestClassifier(**base_params))

    length_model.fit(x_train, y_length_train)
    sequence_model.fit(x_train, y_seq_train)
    boundary_model.fit(x_train, y_boundary_train)

    def evaluate_split(name: str, rows_split, x, y_length, y_seq, y_boundary) -> dict:
        length_pred = length_model.predict(x)
        seq_pred = sequence_model.predict(x)
        boundary_pred = boundary_model.predict(x)
        token_to_char = schema["token_to_char"]
        truth_texts = [row["text"] for row in rows_split]
        pred_texts = [_decode_tokens(row, token_to_char) for row in seq_pred]
        exact_word = sum(a == b for a, b in zip(truth_texts, pred_texts)) / max(1, len(truth_texts))
        token_exact = float(np.all(seq_pred == y_seq, axis=1).mean())
        return {
            "name": name,
            "count": len(rows_split),
            "length_accuracy": float(accuracy_score(y_length, length_pred)),
            "token_sequence_exact_accuracy": token_exact,
            "word_exact_accuracy": exact_word,
            "boundary": _boundary_metrics(y_boundary, boundary_pred),
            "examples": [
                {
                    "truth": truth_texts[index],
                    "prediction": pred_texts[index],
                    "true_length": int(y_length[index]),
                    "predicted_length": int(length_pred[index]),
                }
                for index in range(min(20, len(rows_split)))
            ],
        }

    val_metrics = evaluate_split("validation", val_rows, x_val, y_length_val, y_seq_val, y_boundary_val)
    test_metrics = evaluate_split("test", test_rows, x_test, y_length_test, y_seq_test, y_boundary_test)

    model_bundle = {
        "model_name": settings.get("model_name", "scribetrace_word_v0_1"),
        "feature_names": schema.get("feature_names", []),
        "token_to_char": schema.get("token_to_char", {}),
        "max_sequence_length": schema.get("max_sequence_length"),
        "boundary_bin_count": schema.get("boundary_bin_count"),
        "length_model": length_model,
        "sequence_model": sequence_model,
        "boundary_model": boundary_model,
    }
    model_path = model_dir / f"{model_bundle['model_name']}.joblib"
    joblib.dump(model_bundle, model_path)

    report = {
        "model_name": model_bundle["model_name"],
        "dataset_jsonl": str(dataset_path),
        "row_count": len(rows),
        "split": {
            "train": len(train_rows),
            "validation": len(val_rows),
            "test": len(test_rows),
        },
        "validation": val_metrics,
        "test": test_metrics,
        "model_path": str(model_path),
        "elapsed_seconds": round(time.time() - start, 3),
        "notes": [
            "v0.1 baseline uses raw ScribeTrace word vectors only.",
            "Recognition is intentionally simple; boundary F1 is the key early metric.",
            "No reconstruction, ANTAR, ScriLog, or Scrististics are used here.",
        ],
    }
    report_path = save_json(report, report_dir / "training_report.json")
    print(f"model:  {model_path}")
    print(f"report: {report_path}")
    print(
        "test:",
        f"length={test_metrics['length_accuracy']:.4f}",
        f"word={test_metrics['word_exact_accuracy']:.4f}",
        f"boundary_f1={test_metrics['boundary']['f1']:.4f}",
    )
    return report


def main() -> None:
    """CLI entrypoint for ScribeTrace-word export and training."""

    parser = argparse.ArgumentParser(description="Raw ScribeTrace word experiment.")
    parser.add_argument("--settings", default=str(DEFAULT_SETTINGS_PATH))
    parser.add_argument(
        "--mode",
        choices=["export", "train", "export-train"],
        default="export",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    settings = load_json(args.settings)
    if args.mode in {"export", "export-train"}:
        export_dataset(settings, limit=args.limit or None)
    if args.mode in {"train", "export-train"}:
        train_baseline(settings)


if __name__ == "__main__":
    print("the experiment shall start")
    main()
