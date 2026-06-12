"""Random Forest loading and top-k letter inference for ScribeTrace."""

import json
from pathlib import Path

import joblib
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[2]
MODEL_NAME = "scribetrace_random_forest_v0_2_1"
MODEL_DIR = PROJECT_ROOT / "models" / MODEL_NAME
MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}.joblib"
SCHEMA_PATH = MODEL_DIR / "scribetrace_feature_schema.json"
LABEL_MAP_PATH = MODULE_DIR.parent / "character_detector" / "numeric_label_map.json"

_RF_MODEL = None
_RF_SCHEMA = None
_RF_LABEL_MAP = None


def load_rf_model():
    """Load and cache the trained model, exact schema, and Armenian labels."""
    global _RF_MODEL
    global _RF_SCHEMA
    global _RF_LABEL_MAP

    if _RF_MODEL is not None:
        return

    _RF_MODEL = joblib.load(MODEL_PATH)
    with SCHEMA_PATH.open("r", encoding="utf-8") as file:
        _RF_SCHEMA = json.load(file)
    with LABEL_MAP_PATH.open("r", encoding="utf-8") as file:
        _RF_LABEL_MAP = json.load(file)


def predict_rf_candidates(trace_result, top_k=5):
    """Return top-k letter candidates aligned to the persisted feature schema."""
    if trace_result.status != "completed" or trace_result.feature_vector is None:
        return []

    load_rf_model()

    runtime_feature_map = dict(
        zip(
            trace_result.feature_vector.feature_names,
            trace_result.feature_vector.vector,
        )
    )
    schema_feature_names = _RF_SCHEMA["feature_names"]
    missing_features = [
        name for name in schema_feature_names if name not in runtime_feature_map
    ]
    if missing_features:
        raise ValueError(
            "ScribeTrace RF schema mismatch. Missing runtime features: "
            + ", ".join(missing_features[:20])
        )

    vector = np.array(
        [[float(runtime_feature_map[name]) for name in schema_feature_names]],
        dtype=np.float32,
    )
    probabilities = _RF_MODEL.predict_proba(vector)[0]
    probability_indexes = np.argsort(probabilities)[::-1][:top_k]

    candidates = []
    for rank, probability_index in enumerate(probability_indexes, start=1):
        class_id = int(_RF_MODEL.classes_[int(probability_index)])
        label = str(_RF_LABEL_MAP[str(class_id)])
        candidates.append(
            {
                "rank": rank,
                "class_id": class_id,
                "label": label,
                "text": label,
                "confidence": float(probabilities[int(probability_index)]),
                "source": MODEL_NAME,
            }
        )
    return candidates
