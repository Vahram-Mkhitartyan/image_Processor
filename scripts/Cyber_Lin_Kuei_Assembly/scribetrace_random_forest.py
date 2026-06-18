"""
ScribeTrace Random Forest baseline for Armenian glyph classification.

This model does NOT learn from raw pixels.
It learns from deterministic ScribeTrace geometry vectors:

    image -> ScribeTrace -> ml_features.vector -> RandomForest -> class_id

Save this file in:
    scripts/Cyber_Lin_Kuei_Assembly/scribetrace_random_forest.py

Expected dataset:
    Matenadata/
      0/
      1/
      ...
      77/

Expected label map:
    scripts/N05handwritten_ocr/character_detector/numeric_label_map.json
"""

from pathlib import Path
import argparse
import gc
import gzip
import json
import random
import shutil
import tempfile
from collections import Counter, defaultdict

import joblib
import cv2
import numpy as np
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)


# ============================================================
# Config
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SETTINGS_PATH = SCRIPT_DIR / "scribetrace_random_forest_settings.json"

DATASET_DIR = None
LABEL_MAP_PATH = None
SCRIBETRACE_MODULE_ROOT = None
SCRIBETRACE_EXPERT_PATH = None
MODEL_NAME = None
OUTPUT_DIR = None
REPORT_DIR = None
DATASET_EXPORT_DIR = None
NUM_CLASSES = None
RANDOM_SEED = None
DEFAULT_LIMIT_PER_CLASS = None
VALIDATION_RATIO = None
TEST_RATIO = None
SCRIBETRACE_SETTINGS = None
RANDOM_FOREST_PARAMS = None
ACTIVE_SETTINGS_PATH = None
NORMALIZATION_SETTINGS = None
AUGMENTATION_SETTINGS = None

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def resolve_project_path(value):
    """Resolve a configured path relative to the project root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_runtime_settings(settings):
    """Reject incomplete or unsafe trainer settings before expensive work."""
    required = {
        "model_name",
        "dataset_dir",
        "label_map_path",
        "scribetrace_module_root",
        "num_classes",
        "random_seed",
        "split",
        "dataset_normalization",
        "scribetrace",
        "random_forest",
    }
    missing = sorted(required - set(settings))
    if missing:
        raise ValueError(f"Missing Random Forest settings: {missing}")

    if int(settings["num_classes"]) < 2:
        raise ValueError("num_classes must be at least 2.")

    default_limit = settings.get("default_limit_per_class")
    if default_limit is not None and int(default_limit) < 1:
        raise ValueError("default_limit_per_class must be null or at least 1.")

    split = settings["split"]
    validation_ratio = float(split.get("validation_ratio", 0))
    test_ratio = float(split.get("test_ratio", 0))
    if validation_ratio <= 0 or test_ratio <= 0:
        raise ValueError("Validation and test ratios must both be positive.")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("Validation and test ratios must sum to less than 1.")

    trace_settings = settings["scribetrace"]
    if not trace_settings.get("enabled", False):
        raise ValueError("Trainer ScribeTrace settings must set enabled=true.")
    if trace_settings.get("ink_threshold_mode") not in {
        "auto",
        "binary",
        "fixed",
        "otsu",
    }:
        raise ValueError("Unsupported trainer ScribeTrace threshold mode.")

    aristotel_settings = settings.get("aristotel", {})
    if aristotel_settings.get("enabled", False):
        if int(aristotel_settings.get("variants_per_source", 1)) < 1:
            raise ValueError(
                "aristotel.variants_per_source must be at least 1."
            )
        if int(aristotel_settings.get("degradation_epochs", 1)) < 1:
            raise ValueError(
                "aristotel.degradation_epochs must be at least 1."
            )
        recipes = aristotel_settings.get("recipes")
        if recipes is not None and (
            not isinstance(recipes, list)
            or not all(isinstance(name, str) and name for name in recipes)
        ):
            raise ValueError("aristotel.recipes must be a list of names.")

    forest_settings = settings["random_forest"]
    if int(forest_settings.get("n_estimators", 0)) < 1:
        raise ValueError("random_forest.n_estimators must be at least 1.")
    max_depth = forest_settings.get("max_depth")
    if max_depth is not None and int(max_depth) < 1:
        raise ValueError("random_forest.max_depth must be null or at least 1.")
    max_leaf_nodes = forest_settings.get("max_leaf_nodes")
    if max_leaf_nodes is not None and int(max_leaf_nodes) < 2:
        raise ValueError(
            "random_forest.max_leaf_nodes must be null or at least 2."
        )
    if int(forest_settings.get("min_samples_split", 2)) < 2:
        raise ValueError("random_forest.min_samples_split must be at least 2.")
    if int(forest_settings.get("min_samples_leaf", 1)) < 1:
        raise ValueError("random_forest.min_samples_leaf must be at least 1.")
    if int(forest_settings.get("n_jobs", 1)) == 0:
        raise ValueError("random_forest.n_jobs cannot be 0.")


def configure_runtime(settings_path=DEFAULT_SETTINGS_PATH):
    """Load trainer routing and behavior settings into stable module globals."""
    global DATASET_DIR
    global LABEL_MAP_PATH
    global SCRIBETRACE_MODULE_ROOT
    global SCRIBETRACE_EXPERT_PATH
    global MODEL_NAME
    global OUTPUT_DIR
    global REPORT_DIR
    global DATASET_EXPORT_DIR
    global NUM_CLASSES
    global RANDOM_SEED
    global DEFAULT_LIMIT_PER_CLASS
    global VALIDATION_RATIO
    global TEST_RATIO
    global SCRIBETRACE_SETTINGS
    global RANDOM_FOREST_PARAMS
    global ACTIVE_SETTINGS_PATH
    global NORMALIZATION_SETTINGS
    global AUGMENTATION_SETTINGS

    settings_path = Path(settings_path).expanduser().resolve()
    settings = load_json(settings_path)
    validate_runtime_settings(settings)

    MODEL_NAME = str(settings["model_name"])
    DATASET_DIR = resolve_project_path(settings["dataset_dir"])
    LABEL_MAP_PATH = resolve_project_path(settings["label_map_path"])
    SCRIBETRACE_MODULE_ROOT = resolve_project_path(
        settings["scribetrace_module_root"]
    )
    SCRIBETRACE_EXPERT_PATH = (
        SCRIBETRACE_MODULE_ROOT
        / "N05handwritten_ocr"
        / "scribetrace"
        / "expert.py"
    )
    OUTPUT_DIR = resolve_project_path(
        settings.get("output_dir", f"models/{MODEL_NAME}")
    )
    REPORT_DIR = resolve_project_path(
        settings.get("report_dir", f"reports/{MODEL_NAME}")
    )
    DATASET_EXPORT_DIR = resolve_project_path(
        settings.get("dataset_export_dir", f"datasets/{MODEL_NAME}")
    )
    NUM_CLASSES = int(settings["num_classes"])
    RANDOM_SEED = int(settings["random_seed"])
    DEFAULT_LIMIT_PER_CLASS = settings.get("default_limit_per_class")
    if DEFAULT_LIMIT_PER_CLASS is not None:
        DEFAULT_LIMIT_PER_CLASS = int(DEFAULT_LIMIT_PER_CLASS)
    VALIDATION_RATIO = float(settings["split"]["validation_ratio"])
    TEST_RATIO = float(settings["split"]["test_ratio"])
    NORMALIZATION_SETTINGS = dict(settings["dataset_normalization"])
    if int(NORMALIZATION_SETTINGS.get("size", 0)) < 16:
        raise ValueError("dataset_normalization.size must be at least 16.")
    if NORMALIZATION_SETTINGS.get("polarity") not in {
        "auto",
        "dark_on_light",
        "light_on_dark",
    }:
        raise ValueError("Unsupported dataset normalization polarity.")
    SCRIBETRACE_SETTINGS = dict(settings["scribetrace"])
    RANDOM_FOREST_PARAMS = dict(settings["random_forest"])
    AUGMENTATION_SETTINGS = dict(settings.get("aristotel", {}))
    ACTIVE_SETTINGS_PATH = settings_path

    return settings


# ============================================================
# Import ScribeTrace
# ============================================================

def import_scribetrace():
    """
    Import TraceInput and run_scribetrace from expert.py.

    This keeps the Random Forest trainer outside the N05 expert folder,
    while still using the real ScribeTrace pipeline.
    """
    import sys

    if not SCRIBETRACE_EXPERT_PATH.is_file():
        raise FileNotFoundError(
            f"ScribeTrace expert not found: {SCRIBETRACE_EXPERT_PATH}"
        )

    module_root = str(SCRIBETRACE_MODULE_ROOT)
    if module_root not in sys.path:
        sys.path.insert(0, module_root)

    try:
        from N05handwritten_ocr.scribetrace.expert import TraceInput, run_scribetrace
    except Exception as error:
        raise ImportError(
            "Failed to import ScribeTrace expert.\n"
            f"Expected expert.py at: {SCRIBETRACE_EXPERT_PATH}\n"
            f"Package root: {SCRIBETRACE_MODULE_ROOT}\n"
            "Fix scribetrace_module_root in the trainer settings JSON."
        ) from error

    return TraceInput, run_scribetrace


def normalize_image_for_scribetrace(
    image_path: Path,
    output_path: Path,
    size=96,
    polarity="auto",
) -> Path:
    """
    Normalize glyph image before ScribeTrace.

    Keeps aspect ratio, pads to square, and saves a clean binary mask.
    Output convention: white ink on black background.
    """
    image = Image.open(image_path).convert("L")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image_path}")

    scale = min(size / width, size / height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    image = image.resize((new_width, new_height), Image.BILINEAR)

    source_array = np.array(image)
    border = np.concatenate(
        [
            source_array[0, :],
            source_array[-1, :],
            source_array[:, 0],
            source_array[:, -1],
        ]
    )
    background_is_dark = float(np.median(border)) < 128
    if polarity == "dark_on_light":
        background_is_dark = False
    elif polarity == "light_on_dark":
        background_is_dark = True

    canvas_value = 0 if background_is_dark else 255
    canvas = Image.new("L", (size, size), color=canvas_value)

    x_offset = (size - new_width) // 2
    y_offset = (size - new_height) // 2

    canvas.paste(image, (x_offset, y_offset))

    arr = np.array(canvas)

    threshold_flag = (
        cv2.THRESH_BINARY
        if background_is_dark
        else cv2.THRESH_BINARY_INV
    )
    _, ink = cv2.threshold(
        arr,
        0,
        255,
        threshold_flag | cv2.THRESH_OTSU,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ink).save(output_path)

    return output_path


def normalize_array_for_scribetrace(image, size=96, polarity="auto"):
    """Normalize an in-memory glyph to white ink on a black square."""
    source_array = np.asarray(image, dtype=np.uint8)
    if source_array.ndim == 3:
        source_array = cv2.cvtColor(source_array, cv2.COLOR_BGR2GRAY)
    height, width = source_array.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("Invalid in-memory glyph dimensions.")

    border = np.concatenate(
        [
            source_array[0, :],
            source_array[-1, :],
            source_array[:, 0],
            source_array[:, -1],
        ]
    )
    background_is_dark = float(np.median(border)) < 128
    if polarity == "dark_on_light":
        background_is_dark = False
    elif polarity == "light_on_dark":
        background_is_dark = True

    scale = min(size / width, size / height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    resized = cv2.resize(
        source_array,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas_value = 0 if background_is_dark else 255
    canvas = np.full((size, size), canvas_value, dtype=np.uint8)
    x_offset = (size - new_width) // 2
    y_offset = (size - new_height) // 2
    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width,
    ] = resized
    threshold_flag = (
        cv2.THRESH_BINARY
        if background_is_dark
        else cv2.THRESH_BINARY_INV
    )
    _, ink = cv2.threshold(
        canvas,
        0,
        255,
        threshold_flag | cv2.THRESH_OTSU,
    )
    return ink

# ============================================================
# Utilities
# ============================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_EXPORT_DIR.mkdir(parents=True, exist_ok=True)


configure_runtime(DEFAULT_SETTINGS_PATH)


def collect_samples(dataset_dir: Path):
    """
    Expected dataset structure:

    Matenadata/
      0/
      1/
      ...
      77/

    Only folders 0-77 are treated as classes.
    """
    samples = []
    class_counts = Counter()

    for class_id in range(NUM_CLASSES):
        class_dir = dataset_dir / str(class_id)

        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        image_paths = []

        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path)

        image_paths = sorted(image_paths)

        for path in image_paths:
            samples.append((str(path), class_id))
            class_counts[str(class_id)] += 1

    return samples, class_counts


def make_split_report(total_count, train_count, validation_count, test_count):
    """Build serializable train/validation/test size statistics."""
    return {
        "total": total_count,
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
        "train_ratio": round(train_count / total_count, 4) if total_count else 0,
        "validation_ratio": (
            round(validation_count / total_count, 4) if total_count else 0
        ),
        "test_ratio": round(test_count / total_count, 4) if total_count else 0,
    }


def write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def open_jsonl_text(path, mode):
    """Open plain or gzip-compressed JSONL as UTF-8 text."""
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def inspect_training_jsonl(path: Path):
    """
    Count usable JSONL rows and read the feature schema without retaining rows.

    Returns:
        tuple[int, list[str]]: Row count and ordered feature names.
    """
    row_count = 0
    feature_names = None

    with open_jsonl_text(path, "rt") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            row_count += 1
            if feature_names is None:
                try:
                    first_row = json.loads(line)
                    feature_names = list(first_row["feature_names"])
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError(
                        f"Invalid first dataset row at line {line_number}."
                    ) from error

    if row_count == 0 or not feature_names:
        raise ValueError("Dataset JSONL is empty or has no feature schema.")

    return row_count, feature_names


def load_training_arrays(path: Path):
    """
    Load JSONL into compact arrays instead of retaining every row dictionary.

    The file is scanned once for allocation size, then parsed into one float32
    matrix, one small integer label vector, and a path list used by reports.

    Returns:
        Feature matrix, labels, paths, source IDs, sample kinds, and schema.
    """
    row_count, feature_names = inspect_training_jsonl(path)
    feature_count = len(feature_names)
    features = np.empty((row_count, feature_count), dtype=np.float32)
    labels = np.empty(row_count, dtype=np.int16)
    image_paths = [None] * row_count
    source_ids = [None] * row_count
    sample_kinds = [None] * row_count
    row_index = 0

    with open_jsonl_text(path, "rt") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
                row_feature_names = row["feature_names"]
                vector = row["vector"]
                class_id = int(row["class_id"])
                image_path = str(row["image_path"])
                source_id = str(row.get("source_id", image_path))
                sample_kind = str(row.get("sample_kind", "legacy"))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid dataset row at line {line_number}."
                ) from error

            if row_feature_names != feature_names:
                raise ValueError(f"Feature schema mismatch: {image_path}")
            if len(vector) != feature_count:
                raise ValueError(
                    f"Feature vector length mismatch at {image_path}: "
                    f"expected {feature_count}, found {len(vector)}."
                )
            if not 0 <= class_id < NUM_CLASSES:
                raise ValueError(
                    f"Class id {class_id} is outside 0-{NUM_CLASSES - 1}: "
                    f"{image_path}"
                )

            features[row_index] = vector
            labels[row_index] = class_id
            image_paths[row_index] = image_path
            source_ids[row_index] = source_id
            sample_kinds[row_index] = sample_kind
            row_index += 1

    matrix_megabytes = features.nbytes / (1024 * 1024)
    print(
        f"Loaded {row_count} rows x {feature_count} features "
        f"into a {matrix_megabytes:.1f} MB float32 matrix."
    )

    return (
        features,
        labels,
        image_paths,
        source_ids,
        sample_kinds,
        feature_names,
    )


def grouped_stratified_split(labels, source_ids):
    """Split unique source glyphs per class, then expand to all variants."""
    source_class = {}
    source_rows = defaultdict(list)
    for row_index, (class_id, source_id) in enumerate(
        zip(labels.tolist(), source_ids)
    ):
        previous_class = source_class.setdefault(source_id, int(class_id))
        if previous_class != int(class_id):
            raise ValueError(
                f"Source group {source_id!r} contains multiple class labels."
            )
        source_rows[source_id].append(row_index)

    rng = np.random.default_rng(RANDOM_SEED)
    split_sources = {"train": set(), "validation": set(), "test": set()}
    for class_id in sorted(set(source_class.values())):
        class_sources = sorted(
            source_id
            for source_id, value in source_class.items()
            if value == class_id
        )
        if len(class_sources) < 3:
            raise ValueError(
                f"Class {class_id} needs at least 3 unique source glyphs "
                "for grouped train/validation/test splitting."
            )
        shuffled = np.asarray(class_sources, dtype=object)
        rng.shuffle(shuffled)
        test_count = max(1, int(round(len(shuffled) * TEST_RATIO)))
        validation_count = max(
            1,
            int(round(len(shuffled) * VALIDATION_RATIO)),
        )
        if test_count + validation_count >= len(shuffled):
            test_count = 1
            validation_count = 1
        split_sources["test"].update(shuffled[:test_count].tolist())
        split_sources["validation"].update(
            shuffled[test_count:test_count + validation_count].tolist()
        )
        split_sources["train"].update(
            shuffled[test_count + validation_count:].tolist()
        )

    split_indices = {}
    for split_name, sources in split_sources.items():
        indices = []
        for source_id in sorted(sources):
            indices.extend(source_rows[source_id])
        split_indices[split_name] = np.asarray(
            sorted(indices),
            dtype=np.int32,
        )

    overlaps = (
        split_sources["train"] & split_sources["validation"]
        | split_sources["train"] & split_sources["test"]
        | split_sources["validation"] & split_sources["test"]
    )
    if overlaps:
        raise AssertionError("Source leakage detected between dataset splits.")
    return split_indices, split_sources


def build_class_names(label_map: dict):
    """
    Convert numeric label map into class-name list ordered by class id.

    label_map format is expected like:
        {
          "0": "ա",
          "1": "բ",
          ...
        }
    """
    return [label_map[str(i)] for i in range(NUM_CLASSES)]


# ============================================================
# ScribeTrace Dataset Export
# ============================================================

def trace_one_image(
    image_path: str,
    class_id: int,
    label: str,
    TraceInput,
    run_scribetrace,
):
    """
    Run ScribeTrace on one image and return one ML-ready row.

    The row is designed for Random Forest training.
    """
    image_path = Path(image_path)

    # Matenadata stores white glyph ink on black, matching the exact-mask
    # ScribeTrace contract. Do not route these files through visual inversion.
    normalized_path = (
        DATASET_EXPORT_DIR
        / "normalized_masks"
        / str(class_id)
        / f"{image_path.stem}_scribetrace_mask.png"
    )

    normalized_path = normalize_image_for_scribetrace(
        image_path=image_path,
        output_path=normalized_path,
        size=int(NORMALIZATION_SETTINGS["size"]),
        polarity=NORMALIZATION_SETTINGS["polarity"],
    )

    trace_input = TraceInput(
        mask_crop_path=str(normalized_path),
        document_id="matenadata",
        text_unit_id=f"class_{class_id}_{image_path.stem}",
        layer="dataset",
    )

    result = run_scribetrace(
        trace_input,
        settings=SCRIBETRACE_SETTINGS,
    )
    evidence = result.to_dict()

    ml_features = evidence.get("ml_features")

    if result.status not in {"completed", "completed_limited"}:
        return {
            "ok": False,
            "reason": "scribetrace_failed",
            "status": result.status,
            "error": result.error,
            "image_path": str(image_path),
            "class_id": int(class_id),
            "label": label,
        }

    if not ml_features or not ml_features.get("vector"):
        return {
            "ok": False,
            "reason": "missing_ml_features",
            "status": result.status,
            "error": result.error,
            "image_path": str(image_path),
            "class_id": int(class_id),
            "label": label,
        }

    return {
        "ok": True,
        "class_id": int(class_id),
        "label": label,
        "image_path": str(image_path),
        "status": result.status,
        "feature_names": ml_features["feature_names"],
        "vector": ml_features["vector"],
        "sequence_string": ml_features.get("sequence_string", ""),
        "quality_flags": ml_features.get("quality_flags", {}),
        "metrics": {
            "component_count": evidence.get("component_count"),
            "path_count": evidence.get("path_count"),
            "ink_hole_count": evidence.get("ink_hole_count"),
            "closed_loop_count": evidence.get("metrics", {}).get("closed_loop_count"),
            "attached_loop_count": evidence.get("metrics", {}).get("attached_loop_count"),
            "ink_hole_match_count": evidence.get("metrics", {}).get("ink_hole_match_count"),
            "unmatched_ink_hole_count": evidence.get("metrics", {}).get("unmatched_ink_hole_count"),
            "endpoint_count": evidence.get("metrics", {})
            .get("skeleton_graph", {})
            .get("endpoint_count"),
            "junction_cluster_count": evidence.get("metrics", {})
            .get("skeleton_graph", {})
            .get("junction_cluster_count"),
        },
    }


def trace_one_mask(
    mask_path,
    source_image_path,
    class_id,
    label,
    sample_metadata,
    TraceInput,
    run_scribetrace,
):
    """Extract one feature row from a temporary normalized mask."""
    trace_input = TraceInput(
        mask_crop_path=str(mask_path),
        document_id="matenadata_v4",
        text_unit_id=str(sample_metadata["sample_id"]),
        layer="aristotel_training",
    )
    result = run_scribetrace(trace_input, settings=SCRIBETRACE_SETTINGS)
    evidence = result.to_dict()
    ml_features = evidence.get("ml_features")
    base = {
        "class_id": int(class_id),
        "label": label,
        "image_path": str(source_image_path),
        **sample_metadata,
    }
    if result.status not in {"completed", "completed_limited"}:
        return {
            **base,
            "ok": False,
            "reason": "scribetrace_failed",
            "status": result.status,
            "error": result.error,
        }
    if not ml_features or not ml_features.get("vector"):
        return {
            **base,
            "ok": False,
            "reason": "missing_ml_features",
            "status": result.status,
            "error": result.error,
        }
    return {
        **base,
        "ok": True,
        "status": result.status,
        "feature_names": ml_features["feature_names"],
        "vector": ml_features["vector"],
        "sequence_string": ml_features.get("sequence_string", ""),
        "quality_flags": ml_features.get("quality_flags", {}),
        "metrics": {
            "component_count": evidence.get("component_count"),
            "path_count": evidence.get("path_count"),
            "ink_hole_count": evidence.get("ink_hole_count"),
            "closed_loop_count": evidence.get("metrics", {}).get(
                "closed_loop_count"
            ),
            "endpoint_count": evidence.get("metrics", {})
            .get("skeleton_graph", {})
            .get("endpoint_count"),
            "junction_cluster_count": evidence.get("metrics", {})
            .get("skeleton_graph", {})
            .get("junction_cluster_count"),
        },
    }


def import_aristotel():
    """Import the deterministic teacher without coupling runtime N05 to it."""
    import sys

    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from scripts.Cyber_Lin_Kuei_Assembly.Aristotel.recipes import (
        build_default_recipes,
    )
    from scripts.Cyber_Lin_Kuei_Assembly.Aristotel.runner import FileCorrupter
    from scripts.Cyber_Lin_Kuei_Assembly.Aristotel.teacher_models import (
        TeacherInput,
    )

    return TeacherInput, FileCorrupter, build_default_recipes


def export_scribetrace_v4_dataset(limit_per_class=None):
    """Stream clean and Aristotel-degraded glyph geometry into JSONL."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    label_map = load_json(LABEL_MAP_PATH)
    TraceInput, run_scribetrace = import_scribetrace()
    TeacherInput, FileCorrupter, build_default_recipes = import_aristotel()

    samples, class_counts = collect_samples(DATASET_DIR)
    selected = []
    selected_counts = Counter()
    for image_path, class_id in samples:
        if (
            limit_per_class is not None
            and selected_counts[str(class_id)] >= limit_per_class
        ):
            continue
        selected.append((Path(image_path), class_id))
        selected_counts[str(class_id)] += 1

    include_clean = bool(AUGMENTATION_SETTINGS.get("include_clean", True))
    variants = int(AUGMENTATION_SETTINGS.get("variants_per_source", 1))
    epochs = int(AUGMENTATION_SETTINGS.get("degradation_epochs", 1))
    recipe_names = AUGMENTATION_SETTINGS.get("recipes")
    if variants < 1 or epochs < 1:
        raise ValueError("Aristotel variants and epochs must both be positive.")

    corrupter = FileCorrupter(
        build_default_recipes(),
        seed=int(AUGMENTATION_SETTINGS.get("seed", RANDOM_SEED)),
    )
    compressed = bool(AUGMENTATION_SETTINGS.get("compress_jsonl", True))
    extension = ".jsonl.gz" if compressed else ".jsonl"
    dataset_name = (
        f"scribetrace_v4_full{extension}"
        if limit_per_class is None
        else f"scribetrace_v4_limit_{limit_per_class}{extension}"
    )
    dataset_path = DATASET_EXPORT_DIR / dataset_name
    failed_path = DATASET_EXPORT_DIR / dataset_name.replace(
        extension, f"_failed{extension}"
    )
    summary_path = DATASET_EXPORT_DIR / dataset_name.replace(
        extension, "_summary.json"
    )
    recovery_path = DATASET_EXPORT_DIR / dataset_name.replace(
        extension, f"_recovery{extension}"
    )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    failed_count = 0
    feature_schema = None
    sample_kind_counts = Counter()
    recipe_counts = Counter()

    print("=" * 70)
    print(f"Exporting ScribeTrace v4 dataset from {len(selected)} source glyphs")
    print("Damaged images are temporary and are not retained.")
    print("=" * 70)

    with tempfile.TemporaryDirectory(prefix="scribetrace_v4_") as temp_dir:
        temporary_mask = Path(temp_dir) / "active_mask.png"
        with (
            open_jsonl_text(dataset_path, "wt") as dataset_file,
            open_jsonl_text(failed_path, "wt") as failed_file,
            open_jsonl_text(recovery_path, "wt") as recovery_file,
        ):
            for source_index, (image_path, class_id) in enumerate(
                selected,
                start=1,
            ):
                source_id = image_path.relative_to(DATASET_DIR).as_posix()
                teacher_input = TeacherInput(
                    image_path=image_path,
                    label=str(class_id),
                    source_class=str(class_id),
                    source_folder=image_path.parent,
                    metadata={"source_id": source_id},
                )
                generated = []
                if include_clean:
                    clean_image = corrupter.load_image(image_path)
                    generated.append(
                        (
                            clean_image,
                            {
                                "sample_id": f"clean:{source_id}",
                                "source_id": source_id,
                                "sample_kind": "clean",
                                "damage_recipe": None,
                                "damage_seed": None,
                                "damage_epoch": None,
                                "damage_variant": None,
                                "damage_severity": 0.0,
                                "damage_operations": [],
                                "damage_changed_pixels": 0,
                                "damage_changed_ratio": 0.0,
                                "recovery_expected": False,
                                "recovery_label": "clean",
                            },
                        )
                    )
                for epoch in range(epochs):
                    for variant in range(variants):
                        damaged_samples = corrupter.corrupt(
                            teacher_input,
                            epoch=epoch,
                            variant=variant,
                            recipe_names=recipe_names,
                        )
                        for damaged in damaged_samples:
                            generated.append(
                                (
                                    damaged.image,
                                    {
                                        "sample_id": damaged.sample_id,
                                        "source_id": source_id,
                                        "sample_kind": "degraded",
                                        "damage_recipe": damaged.damage_recipe,
                                        "damage_seed": damaged.seed,
                                        "damage_epoch": damaged.epoch,
                                        "damage_variant": damaged.variant,
                                        "damage_severity": damaged.severity,
                                        "damage_operations": damaged.operations,
                                        "damage_changed_pixels":
                                            damaged.changed_pixel_count,
                                        "damage_changed_ratio":
                                            damaged.changed_pixel_ratio,
                                        "recovery_expected": (
                                            damaged.trust_label
                                            == "repair_needed"
                                            and damaged.changed_pixel_count > 0
                                        ),
                                        "recovery_label": damaged.trust_label,
                                        "recipe_signature":
                                            damaged.recipe_signature,
                                    },
                                )
                            )

                for image, metadata in generated:
                    normalized = normalize_array_for_scribetrace(
                        image,
                        size=int(NORMALIZATION_SETTINGS["size"]),
                        polarity=NORMALIZATION_SETTINGS["polarity"],
                    )
                    if not cv2.imwrite(str(temporary_mask), normalized):
                        raise RuntimeError("Could not write temporary trace mask.")
                    row = trace_one_mask(
                        temporary_mask,
                        image_path,
                        class_id,
                        label_map[str(class_id)],
                        metadata,
                        TraceInput,
                        run_scribetrace,
                    )
                    if not row["ok"]:
                        failed_file.write(
                            json.dumps(row, ensure_ascii=False) + "\n"
                        )
                        failed_count += 1
                        continue
                    if feature_schema is None:
                        feature_schema = row["feature_names"]
                    elif feature_schema != row["feature_names"]:
                        raise ValueError(
                            f"Feature schema changed at {metadata['sample_id']}."
                        )
                    dataset_file.write(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )
                    recovery_file.write(
                        json.dumps(
                            {
                                "sample_id": row["sample_id"],
                                "source_id": row["source_id"],
                                "class_id": row["class_id"],
                                "sample_kind": row["sample_kind"],
                                "damage_recipe": row["damage_recipe"],
                                "damage_severity": row["damage_severity"],
                                "damage_changed_pixels": row.get(
                                    "damage_changed_pixels"
                                ),
                                "damage_changed_ratio": row.get(
                                    "damage_changed_ratio"
                                ),
                                "recovery_expected": row["recovery_expected"],
                                "recovery_label": row["recovery_label"],
                                "quality_flags": row["quality_flags"],
                                "metrics": row["metrics"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    exported_count += 1
                    sample_kind_counts[row["sample_kind"]] += 1
                    recipe_counts[str(row["damage_recipe"])] += 1

                if source_index % 250 == 0:
                    print(
                        f"Sources {source_index}/{len(selected)} | "
                        f"rows={exported_count} failed={failed_count}"
                    )

    summary = {
        "dataset_version": "scribetrace_v4",
        "dataset_path": str(dataset_path),
        "failed_path": str(failed_path),
        "recovery_dataset_path": str(recovery_path),
        "source_grouping_field": "source_id",
        "source_count": len(selected),
        "exported_count": exported_count,
        "failed_count": failed_count,
        "include_clean": include_clean,
        "variants_per_source": variants,
        "degradation_epochs": epochs,
        "sample_kind_counts": dict(sample_kind_counts),
        "recipe_counts": dict(recipe_counts),
        "selected_class_counts": dict(selected_counts),
        "raw_class_counts": dict(class_counts),
        "feature_schema": feature_schema,
        "aristotel_settings": AUGMENTATION_SETTINGS,
        "scribetrace_settings": SCRIBETRACE_SETTINGS,
        "retained_generated_images": 0,
        "compressed_jsonl": compressed,
    }
    save_json(summary, summary_path)
    print(f"Dataset: {dataset_path}")
    print(f"Rows: {exported_count}; failed: {failed_count}")
    return dataset_path


def export_scribetrace_dataset(limit_per_class=None):
    """
    Walk Matenadata, run ScribeTrace on every selected image,
    and export JSONL rows for Random Forest.
    """
    print("=" * 70)
    print("Exporting ScribeTrace Random Forest dataset")
    print("=" * 70)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}")

    label_map = load_json(LABEL_MAP_PATH)

    if len(label_map) != NUM_CLASSES:
        raise ValueError(
            f"Label map has {len(label_map)} entries, expected {NUM_CLASSES}"
        )

    TraceInput, run_scribetrace = import_scribetrace()

    samples, class_counts = collect_samples(DATASET_DIR)

    print(f"Collected raw samples: {len(samples)}")
    print(f"Classes found: {len(class_counts)}")

    selected_samples = []
    selected_counts = Counter()

    for image_path, class_id in samples:
        if limit_per_class is not None and selected_counts[str(class_id)] >= limit_per_class:
            continue

        selected_samples.append((image_path, class_id))
        selected_counts[str(class_id)] += 1

    print(f"Selected samples: {len(selected_samples)}")
    print(f"Limit per class: {limit_per_class}")

    rows = []
    failed_rows = []
    feature_schema = None

    for index, (image_path, class_id) in enumerate(selected_samples, start=1):
        label = label_map[str(class_id)]

        row = trace_one_image(
            image_path=image_path,
            class_id=class_id,
            label=label,
            TraceInput=TraceInput,
            run_scribetrace=run_scribetrace,
        )

        if not row["ok"]:
            failed_rows.append(row)
            continue

        feature_names = row["feature_names"]

        if feature_schema is None:
            feature_schema = feature_names
        elif feature_schema != feature_names:
            raise ValueError(
                "Feature schema mismatch.\n"
                f"Image: {image_path}\n"
                f"Expected: {feature_schema}\n"
                f"Got:      {feature_names}"
            )

        rows.append(row)

        if index % 250 == 0:
            print(f"Processed {index}/{len(selected_samples)} | exported={len(rows)} failed={len(failed_rows)}")

    if not rows:
        raise ValueError("No usable ScribeTrace rows were exported.")

    if limit_per_class is None:
        dataset_name = "scribetrace_rf_full.jsonl"
    else:
        dataset_name = f"scribetrace_rf_limit_{limit_per_class}.jsonl"

    dataset_path = DATASET_EXPORT_DIR / dataset_name
    failed_path = DATASET_EXPORT_DIR / dataset_name.replace(".jsonl", "_failed.jsonl")
    summary_path = DATASET_EXPORT_DIR / dataset_name.replace(".jsonl", "_summary.json")

    write_jsonl(rows, dataset_path)
    write_jsonl(failed_rows, failed_path)

    exported_counts = Counter(str(row["class_id"]) for row in rows)

    summary = {
        "dataset_path": str(dataset_path),
        "failed_path": str(failed_path),
        "settings_path": str(ACTIVE_SETTINGS_PATH),
        "dataset_dir": str(DATASET_DIR),
        "label_map_path": str(LABEL_MAP_PATH),
        "dataset_normalization": NORMALIZATION_SETTINGS,
        "scribetrace_settings": SCRIBETRACE_SETTINGS,
        "num_classes": NUM_CLASSES,
        "limit_per_class": limit_per_class,
        "raw_sample_count": len(samples),
        "selected_sample_count": len(selected_samples),
        "exported_count": len(rows),
        "failed_count": len(failed_rows),
        "raw_class_counts": dict(class_counts),
        "selected_class_counts": dict(selected_counts),
        "exported_class_counts": dict(exported_counts),
        "feature_schema": feature_schema,
    }

    save_json(summary, summary_path)

    print()
    print("Export complete.")
    print(f"Exported rows: {len(rows)}")
    print(f"Failed rows:   {len(failed_rows)}")
    print(f"Dataset:       {dataset_path}")
    print(f"Summary:       {summary_path}")

    return dataset_path


# ============================================================
# Random Forest Training / Eval
# ============================================================

def get_top_k_predictions(probabilities: np.ndarray, k: int = 5):
    top_indexes = np.argsort(probabilities, axis=1)[:, ::-1][:, :k]
    top_probs = np.take_along_axis(probabilities, top_indexes, axis=1)

    return top_indexes, top_probs


def evaluate_sample_kinds(y_true, y_pred, probabilities, classes, kinds):
    """Report top-1/top-5 separately for clean and degraded rows."""
    report = {}
    kinds = np.asarray(kinds, dtype=object)
    for kind in sorted(set(kinds.tolist())):
        mask = kinds == kind
        kind_true = y_true[mask]
        kind_pred = y_pred[mask]
        kind_probabilities = probabilities[mask]
        report[kind] = {
            "count": int(np.count_nonzero(mask)),
            "top1": float(accuracy_score(kind_true, kind_pred)),
            "top5": float(
                top_k_accuracy_score(
                    kind_true,
                    kind_probabilities,
                    k=min(5, len(classes)),
                    labels=classes,
                )
            ),
        }
    return report


def save_confusion_outputs(y_true, y_pred, label_map):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    np.savetxt(
        REPORT_DIR / "confusion_matrix.csv",
        cm,
        fmt="%d",
        delimiter=",",
    )

    class_names = build_class_names(label_map)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    (REPORT_DIR / "classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )


def save_confusion_pair_report(y_true, y_pred, label_map, max_pairs=80):
    """Save the strongest off-diagonal class confusions for v0_2.1 repair."""
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    pairs = []
    for true_class in range(NUM_CLASSES):
        true_total = int(cm[true_class].sum())
        if true_total <= 0:
            continue

        for predicted_class in range(NUM_CLASSES):
            if predicted_class == true_class:
                continue

            count = int(cm[true_class, predicted_class])
            if count <= 0:
                continue

            pairs.append(
                {
                    "true_class_id": int(true_class),
                    "true_label": label_map[str(true_class)],
                    "predicted_class_id": int(predicted_class),
                    "predicted_label": label_map[str(predicted_class)],
                    "count": count,
                    "true_class_support": true_total,
                    "share_of_true_class": count / true_total,
                }
            )

    pairs.sort(
        key=lambda item: (
            item["count"],
            item["share_of_true_class"],
        ),
        reverse=True,
    )

    save_json(
        {
            "note": "Largest off-diagonal Random Forest confusions on the test split.",
            "max_pairs": max_pairs,
            "pairs_saved": min(len(pairs), max_pairs),
            "pairs": pairs[:max_pairs],
        },
        REPORT_DIR / "confusion_pairs.json",
    )


def save_feature_health_report(X_train, X_val, X_test, feature_names):
    """Save simple schema health stats so dead/constant features are visible."""
    X_all = np.vstack([X_train, X_val, X_test])
    report = []

    for index, feature_name in enumerate(feature_names):
        column = X_all[:, index]
        finite_mask = np.isfinite(column)
        finite_values = column[finite_mask]

        if len(finite_values) == 0:
            report.append(
                {
                    "feature": feature_name,
                    "finite_count": 0,
                    "nan_or_inf_count": int(len(column)),
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                    "zero_ratio": None,
                    "unique_count_sample": 0,
                    "is_constant": True,
                }
            )
            continue

        unique_count = int(len(np.unique(finite_values[: min(len(finite_values), 5000)])))
        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))

        report.append(
            {
                "feature": feature_name,
                "finite_count": int(len(finite_values)),
                "nan_or_inf_count": int(len(column) - len(finite_values)),
                "min": min_value,
                "max": max_value,
                "mean": float(np.mean(finite_values)),
                "std": float(np.std(finite_values)),
                "zero_ratio": float(np.mean(finite_values == 0)),
                "unique_count_sample": unique_count,
                "is_constant": bool(min_value == max_value),
            }
        )

    save_json(report, REPORT_DIR / "feature_health.json")


def save_topk_examples(
    test_image_paths,
    y_true,
    probabilities,
    model_classes,
    label_map,
    k=5,
    max_examples=300,
):
    top_indexes, top_probs = get_top_k_predictions(probabilities, k=k)

    examples = []

    limit = min(len(y_true), max_examples)

    for i in range(limit):
        true_class = int(y_true[i])

        candidates = []

        for probability_index, prob in zip(top_indexes[i], top_probs[i]):
            class_id = int(model_classes[int(probability_index)])

            candidates.append(
                {
                    "class_id": class_id,
                    "label": label_map[str(class_id)],
                    "probability": float(prob),
                }
            )

        examples.append(
            {
                "sample_index": i,
                "image_path": test_image_paths[i],
                "true_class_id": true_class,
                "true_label": label_map[str(true_class)],
                "top_candidates": candidates,
            }
        )

    save_json(
        {
            "k": k,
            "note": "Limited sample of top-k predictions from the Random Forest test split.",
            "examples_saved": len(examples),
            "examples": examples,
        },
        REPORT_DIR / "topk_examples.json",
    )


def save_feature_importance(model, feature_names):
    importances = [
        {
            "feature": feature_name,
            "importance": float(importance),
        }
        for feature_name, importance in zip(feature_names, model.feature_importances_)
    ]

    importances.sort(key=lambda item: item["importance"], reverse=True)

    save_json(importances, REPORT_DIR / "feature_importance.json")


def train_random_forest(dataset_jsonl: Path):
    print("=" * 70)
    print(f"Training Random Forest: {MODEL_NAME}")
    print("=" * 70)

    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"Dataset JSONL not found: {dataset_jsonl}")

    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}")

    label_map = load_json(LABEL_MAP_PATH)

    (
        X,
        y,
        image_paths,
        source_ids,
        sample_kinds,
        feature_names,
    ) = load_training_arrays(dataset_jsonl)
    dataset_row_count = len(y)

    labels_present = sorted(set(y.tolist()))

    if len(labels_present) != NUM_CLASSES:
        print(
            f"Warning: dataset contains {len(labels_present)} classes, "
            f"expected {NUM_CLASSES}. This is okay for tiny tests, not full training."
        )

    try:
        split_indices, split_sources = grouped_stratified_split(
            y,
            source_ids,
        )
        train_indices = split_indices["train"]
        validation_indices = split_indices["validation"]
        test_indices = split_indices["test"]
    except ValueError as error:
        raise ValueError(
            "Source-grouped train/validation/test split failed. Increase "
            "--limit-per-class or adjust split ratios in the settings JSON."
        ) from error

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[validation_indices]
    y_val = y[validation_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    test_image_paths = [image_paths[int(index)] for index in test_indices]
    validation_sample_kinds = [
        sample_kinds[int(index)] for index in validation_indices
    ]
    test_sample_kinds = [
        sample_kinds[int(index)] for index in test_indices
    ]

    split_report = make_split_report(
        dataset_row_count,
        len(train_indices),
        len(validation_indices),
        len(test_indices),
    )
    split_report["strategy"] = "stratified_by_class_grouped_by_source_id"
    split_report["source_counts"] = {
        name: len(values)
        for name, values in split_sources.items()
    }
    split_report["sample_kind_counts"] = {
        name: dict(
            Counter(sample_kinds[int(index)] for index in indices)
        )
        for name, indices in (
            ("train", train_indices),
            ("validation", validation_indices),
            ("test", test_indices),
        )
    }

    save_json(split_report, REPORT_DIR / "split_report.json")

    shutil.copyfile(
        LABEL_MAP_PATH,
        OUTPUT_DIR / "numeric_label_map.json",
    )

    print("Split:")
    print(f"  Train:      {len(train_indices)}")
    print(f"  Validation: {len(validation_indices)}")
    print(f"  Test:       {len(test_indices)}")

    forest_params = dict(RANDOM_FOREST_PARAMS)
    forest_params["random_state"] = RANDOM_SEED
    print("Random Forest parameters:")
    print(json.dumps(forest_params, indent=2))

    # Release the full dataset and split bookkeeping before tree construction.
    del X
    del y
    del image_paths
    del source_ids
    del sample_kinds
    del train_indices
    del validation_indices
    del test_indices
    gc.collect()

    model = RandomForestClassifier(**forest_params)

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_proba = model.predict_proba(X_val)
    test_proba = model.predict_proba(X_test)

    val_top1 = accuracy_score(y_val, val_pred)
    test_top1 = accuracy_score(y_test, test_pred)

    # For RandomForestClassifier, predict_proba columns follow model.classes_.
    # top_k_accuracy_score needs labels matching probability columns.
    val_top5 = top_k_accuracy_score(
        y_val,
        val_proba,
        k=min(5, len(model.classes_)),
        labels=model.classes_,
    )

    test_top5 = top_k_accuracy_score(
        y_test,
        test_proba,
        k=min(5, len(model.classes_)),
        labels=model.classes_,
    )
    validation_kind_metrics = evaluate_sample_kinds(
        y_val,
        val_pred,
        val_proba,
        model.classes_,
        validation_sample_kinds,
    )
    test_kind_metrics = evaluate_sample_kinds(
        y_test,
        test_pred,
        test_proba,
        model.classes_,
        test_sample_kinds,
    )
    save_json(
        {
            "validation": validation_kind_metrics,
            "test": test_kind_metrics,
        },
        REPORT_DIR / "sample_kind_metrics.json",
    )

    save_confusion_outputs(y_test, test_pred, label_map)
    save_confusion_pair_report(y_test, test_pred, label_map)
    save_feature_health_report(X_train, X_val, X_test, feature_names)
    save_topk_examples(
        test_image_paths,
        y_test,
        test_proba,
        model.classes_,
        label_map,
        k=5,
        max_examples=300,
    )
    save_feature_importance(model, feature_names)

    model_path = OUTPUT_DIR / f"{MODEL_NAME}.joblib"
    schema_path = OUTPUT_DIR / "scribetrace_feature_schema.json"
    report_path = REPORT_DIR / "training_report.json"

    joblib.dump(model, model_path)

    save_json(
        {
            "schema_version": "scribetrace_ml_v1",
            "feature_names": feature_names,
            "num_features": len(feature_names),
            "num_classes": NUM_CLASSES,
            "classes_seen": [int(value) for value in model.classes_.tolist()],
            "label_map_path": str(LABEL_MAP_PATH),
        },
        schema_path,
    )

    final_report = {
        "model_name": MODEL_NAME,
        "model_type": "RandomForestClassifier",
        "dataset_jsonl": str(dataset_jsonl),
        "dataset_rows": dataset_row_count,
        "settings_path": str(ACTIVE_SETTINGS_PATH),
        "num_classes": NUM_CLASSES,
        "random_seed": RANDOM_SEED,
        "dataset_normalization": NORMALIZATION_SETTINGS,
        "scribetrace_settings": SCRIBETRACE_SETTINGS,
        "split": split_report,
        "random_forest_params": forest_params,
        "validation_top1": float(val_top1),
        "validation_top5": float(val_top5),
        "test_top1": float(test_top1),
        "test_top5": float(test_top5),
        "validation_sample_kind_metrics": validation_kind_metrics,
        "test_sample_kind_metrics": test_kind_metrics,
        "model_path": str(model_path),
        "schema_path": str(schema_path),
        "confusion_matrix_path": str(REPORT_DIR / "confusion_matrix.csv"),
        "classification_report_path": str(REPORT_DIR / "classification_report.txt"),
        "topk_examples_path": str(REPORT_DIR / "topk_examples.json"),
        "feature_importance_path": str(REPORT_DIR / "feature_importance.json"),
        "confusion_pairs_path": str(REPORT_DIR / "confusion_pairs.json"),
        "feature_health_path": str(REPORT_DIR / "feature_health.json"),
        "sample_kind_metrics_path": str(
            REPORT_DIR / "sample_kind_metrics.json"
        ),
        "notes": [
            "Random Forest baseline trained on ScribeTrace geometry vectors, not pixels.",
            "This should be compared against glyph_classifier.py CNN baseline.",
            "Top-k output is important because N05 needs candidate letters, not only one winner.",
            "Feature schema must be frozen after full Matenadata export.",
        ],
    }

    save_json(final_report, report_path)

    print()
    print("Training complete.")
    print(f"Validation top1: {val_top1:.4f}")
    print(f"Validation top5: {val_top5:.4f}")
    print(f"Test top1:       {test_top1:.4f}")
    print(f"Test top5:       {test_top5:.4f}")
    print()
    print(f"Model:           {model_path}")
    print(f"Schema:          {schema_path}")
    print(f"Report:          {report_path}")
    print(f"Feature import.: {REPORT_DIR / 'feature_importance.json'}")

    return model_path


# ============================================================
# Prediction helper
# ============================================================

def predict_from_scribetrace_result(result_dict: dict, model_path: Path, k: int = 5):
    """
    Predict top-k glyph candidates from a ScribeTrace result dictionary.

    This is the function N05 can later reuse inside the expert pipeline.
    """
    model = joblib.load(model_path)

    ml_features = result_dict.get("ml_features")

    if not ml_features:
        raise ValueError("Missing ml_features in ScribeTrace result.")

    vector = np.array([ml_features["vector"]], dtype=np.float32)

    probabilities = model.predict_proba(vector)[0]

    top_indexes = np.argsort(probabilities)[::-1][:k]

    label_map = load_json(LABEL_MAP_PATH)

    candidates = []

    for class_index in top_indexes:
        class_id = int(model.classes_[class_index])

        candidates.append(
            {
                "class_id": class_id,
                "label": label_map[str(class_id)],
                "probability": float(probabilities[class_index]),
            }
        )

    return candidates


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--settings",
        type=str,
        default=str(DEFAULT_SETTINGS_PATH),
        help="Random Forest trainer settings JSON.",
    )

    parser.add_argument(
        "--mode",
        choices=["export", "train", "export-train"],
        default="export-train",
        help="export = create JSONL, train = train from JSONL, export-train = both",
    )

    parser.add_argument(
        "--dataset-jsonl",
        type=str,
        default=None,
        help="Existing ScribeTrace JSONL dataset for training mode.",
    )

    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help=(
            "Limit images per class during export. Omit to use the settings "
            "default; use -1 for the full dataset."
        ),
    )

    args = parser.parse_args()

    configure_runtime(args.settings)
    seed_everything(RANDOM_SEED)
    ensure_dirs()

    limit_per_class = (
        DEFAULT_LIMIT_PER_CLASS
        if args.limit_per_class is None
        else args.limit_per_class
    )

    if limit_per_class is not None and limit_per_class < 0:
        limit_per_class = None

    if args.mode == "export":
        if AUGMENTATION_SETTINGS.get("enabled", False):
            export_scribetrace_v4_dataset(
                limit_per_class=limit_per_class
            )
        else:
            export_scribetrace_dataset(limit_per_class=limit_per_class)

    elif args.mode == "train":
        if args.dataset_jsonl is None:
            raise ValueError("--dataset-jsonl is required when --mode train")

        train_random_forest(Path(args.dataset_jsonl))

    elif args.mode == "export-train":
        if AUGMENTATION_SETTINGS.get("enabled", False):
            dataset_path = export_scribetrace_v4_dataset(
                limit_per_class=limit_per_class
            )
        else:
            dataset_path = export_scribetrace_dataset(
                limit_per_class=limit_per_class
            )
        train_random_forest(dataset_path)


if __name__ == "__main__":
    main()
