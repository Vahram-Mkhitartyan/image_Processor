"""Runtime bridge from the trained ScribeTrain model into N05 segmentation.

ScribeTrain lives in the Cyber Lin Kuei training arena, but N05 needs a tiny
inference adapter that can turn one word crop into candidate letter spans. This
module deliberately keeps that bridge small: load model, extract topology
features, decode path candidates, and return assembly-compatible segments.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - handled as a runtime status below.
    torch = None
    F = None

try:
    from .assembly.schemas import make_segmentation_path, make_segmentation_segment
except ImportError:
    from assembly.schemas import make_segmentation_path, make_segmentation_segment  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for import_root in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/scribetrain_monster_v0_1/scribetrain_monster_v0_1.pt"
DEFAULT_SETTINGS_PATH = (
    PROJECT_ROOT
    / "scripts/Cyber_Lin_Kuei_Assembly/scribetrain/scribetrain_monster_v0_1_settings.json"
)


def _load_json(path: str | Path | None) -> dict:
    if not path:
        return {}
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if not candidate.is_file():
        return {}
    with candidate.open("r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_path(path: str | Path | None, default: Path) -> Path:
    candidate = Path(path) if path else default
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def _choose_device(requested: str | None):
    if torch is None:
        return None
    value = str(requested or "auto").lower()
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _read_scribetrain_image(path: str | Path | None):
    """Read a word image as black ink on white background for ScribeTrain."""

    if not path:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    border = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    edge = border[[0, -1], :].ravel().tolist() + border[:, [0, -1]].ravel().tolist()
    # Trainer expects dark ink. If the crop is a white mask on a dark
    # background, invert it before extracting topology bins.
    if edge and float(sorted(edge)[len(edge) // 2]) < 128.0:
        image = 255 - image
    return image


def _bin_to_x(bin_index: int, bin_count: int, width: int) -> int:
    """Map a ScribeTrain boundary bin to a crop-local x coordinate."""

    center = (float(bin_index) + 0.5) / max(1.0, float(bin_count))
    return int(round(center * float(width)))


def _segments_from_cut_bins(
    path_id: str,
    cut_bins: list[int],
    bin_count: int,
    width: int,
    height: int,
    min_segment_width_px: int,
) -> list[dict]:
    """Convert ScribeTrain cut bins into assembly segment bboxes."""

    cuts = []
    for bin_index in sorted({int(value) for value in cut_bins}):
        x = _bin_to_x(bin_index, bin_count, width)
        if x <= 0 or x >= width:
            continue
        if cuts and abs(x - cuts[-1]) < min_segment_width_px:
            continue
        cuts.append(x)

    xs = [0, *cuts, width]
    segments = []
    for index, (x1, x2) in enumerate(zip(xs, xs[1:])):
        if x2 - x1 < min_segment_width_px:
            return []
        segments.append(
            make_segmentation_segment(
                segment_id=f"{path_id}_s{index}",
                bbox={"x1": x1, "y1": 0, "x2": x2, "y2": height},
                role="character_candidate",
                source="scribetrain_word_segmenter",
                # Leave crop paths empty. The assembly materializer must crop
                # from the whole unit using the bbox instead of copying the
                # original word image for every segment.
                mask_crop_path=None,
                visual_crop_path=None,
                source_segment={
                    "left_boundary_x": x1,
                    "right_boundary_x": x2,
                    "source_path_id": path_id,
                },
            )
        )
    return segments


def _path_score_hint(score: float) -> float:
    """Clamp ScribeTrain's path score into the assembly score-hint range."""

    if not math.isfinite(float(score)):
        return 0.0
    return max(0.0, min(1.0, float(score)))


def propose_scribetrain_word_segments(
    handwritten_text_unit: dict,
    settings: dict | None = None,
) -> dict:
    """Run ScribeTrain on one N05 word crop and return segmentation paths."""

    settings = settings or {}
    if not bool(settings.get("enabled", True)):
        return {
            "schema_version": "n05_scribetrain_word_segmentation_v1",
            "attempted": False,
            "status": "disabled",
            "paths": [],
        }
    if torch is None or F is None:
        return {
            "schema_version": "n05_scribetrain_word_segmentation_v1",
            "attempted": True,
            "status": "unavailable",
            "error": "PyTorch is not importable in this environment.",
            "paths": [],
        }

    try:
        from scripts.Cyber_Lin_Kuei_Assembly.scribetrain.scribetrain_monster_v0_1 import (
            FEATURE_NAMES,
            ScribeTrainWordTraceModel,
            build_segmentation_paths as build_scribetrain_paths,
            extract_sequence_features,
        )
    except Exception as error:  # pragma: no cover - import diagnostics.
        return {
            "schema_version": "n05_scribetrain_word_segmentation_v1",
            "attempted": True,
            "status": "import_failed",
            "error": str(error),
            "paths": [],
        }

    model_path = _resolve_path(settings.get("model_path"), DEFAULT_MODEL_PATH)
    if not model_path.is_file():
        return {
            "schema_version": "n05_scribetrain_word_segmentation_v1",
            "attempted": True,
            "status": "model_missing",
            "model_path": str(model_path),
            "paths": [],
        }

    image_path = (
        handwritten_text_unit.get("analysis_mask_crop_path")
        or handwritten_text_unit.get("scribetrace_mask_crop_path")
        or handwritten_text_unit.get("n05_selected_crop_path")
        or handwritten_text_unit.get("scribetrace_visual_crop_path")
    )
    image = _read_scribetrain_image(image_path)
    if image is None:
        return {
            "schema_version": "n05_scribetrain_word_segmentation_v1",
            "attempted": True,
            "status": "image_missing",
            "image_path": image_path,
            "paths": [],
        }

    settings_path = _resolve_path(settings.get("settings_path"), DEFAULT_SETTINGS_PATH)
    train_settings_root = _load_json(settings_path)
    dataset_settings = train_settings_root.get("dataset", {})
    training_settings = train_settings_root.get("training", {})
    feature_settings = train_settings_root.get("feature_extraction", {})

    bin_count = int(settings.get("boundary_bin_count") or dataset_settings.get("boundary_bin_count", 64))
    max_length_class = int(dataset_settings.get("max_sequence_length", 18))
    hidden_size = int(settings.get("hidden_size") or training_settings.get("hidden_size", 96))
    num_layers = int(settings.get("num_layers") or training_settings.get("num_layers", 2))
    dropout = float(settings.get("dropout") or training_settings.get("dropout", 0.15))
    device = _choose_device(settings.get("device") or training_settings.get("device", "auto"))

    try:
        checkpoint = torch.load(str(model_path), map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model = ScribeTrainWordTraceModel(
            feature_count=len(FEATURE_NAMES),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            max_length_class=max_length_class,
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        features_np = extract_sequence_features(
            image,
            bin_count=bin_count,
            feature_settings=feature_settings,
        )
        features = torch.from_numpy(features_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(features)
            boundary = torch.sigmoid(outputs["boundary_logits"])[0].detach().cpu()
            bridge = torch.sigmoid(outputs["bridge_logits"])[0].detach().cpu()
            safety = torch.sigmoid(outputs["cut_safety_logits"])[0].detach().cpu()
            length_probs = F.softmax(outputs["length_logits"], dim=1)[0].detach().cpu()
    except Exception as error:
        return {
            "schema_version": "n05_scribetrain_word_segmentation_v1",
            "attempted": True,
            "status": "failed",
            "model_path": str(model_path),
            "image_path": str(image_path),
            "error": str(error),
            "paths": [],
        }

    predicted_length = int(torch.argmax(length_probs).item())
    expected_split_count = max(0, predicted_length - 1)
    length_confidence = float(length_probs[predicted_length].item())
    raw_paths = build_scribetrain_paths(
        boundary,
        bridge,
        safety,
        threshold=float(settings.get("peak_threshold", training_settings.get("peak_threshold", 0.35))),
        min_distance=int(settings.get("peak_min_distance_bins", training_settings.get("peak_min_distance_bins", 2))),
        top_k=int(settings.get("top_k_paths", train_settings_root.get("output", {}).get("path_top_k", 5))),
        expected_split_count=expected_split_count,
        length_confidence=length_confidence,
        length_weight=float(settings.get("length_path_weight", training_settings.get("length_path_weight", 0.25))),
        length_penalty_strength=float(
            settings.get("length_penalty_strength", training_settings.get("length_penalty_strength", 2.0))
        ),
        image=image,
        snap_radius_bins=int(settings.get("snap_radius_bins", training_settings.get("snap_radius_bins", 2))),
        recall_threshold=settings.get("recall_threshold", training_settings.get("recall_threshold", 0.12)),
        recall_candidate_limit=int(settings.get("recall_candidate_limit", training_settings.get("recall_candidate_limit", 32))),
        sequence_features=features[0].detach().cpu(),
        segment_geometry_weight=float(
            settings.get("segment_geometry_weight", training_settings.get("segment_geometry_weight", 0.15))
        ),
    )

    height, width = image.shape[:2]
    min_width = int(settings.get("min_segment_width_px", 3))
    assembly_paths = []
    for index, raw_path in enumerate(raw_paths):
        path_id = f"st_{raw_path.get('path_id') or index}"
        segments = _segments_from_cut_bins(
            path_id=path_id,
            cut_bins=list(raw_path.get("cut_bins") or []),
            bin_count=bin_count,
            width=width,
            height=height,
            min_segment_width_px=min_width,
        )
        if not segments:
            continue
        assembly_paths.append(
            make_segmentation_path(
                path_id=path_id,
                path_type="scribetrain_word_trace_sequence",
                segments=segments,
                score_hint=_path_score_hint(float(raw_path.get("score", 0.0))),
                source="scribetrain_word_segmenter",
                reason=str(raw_path.get("reason") or "scribetrain_path_candidate"),
                evidence={
                    "model_name": "scribetrain_monster_v0_1",
                    "model_path": str(model_path),
                    "image_path": str(image_path),
                    "bin_count": bin_count,
                    "predicted_length": predicted_length,
                    "expected_split_count": expected_split_count,
                    "length_confidence": length_confidence,
                    "raw_path": raw_path,
                },
            )
        )

    assembly_paths.sort(
        key=lambda path: (
            -float(path.get("score_hint", 0.0)),
            abs(int(path.get("segment_count", 0)) - max(1, predicted_length)),
            str(path.get("path_id", "")),
        )
    )
    return {
        "schema_version": "n05_scribetrain_word_segmentation_v1",
        "attempted": True,
        "status": "completed" if assembly_paths else "no_paths",
        "model_name": "scribetrain_monster_v0_1",
        "model_path": str(model_path),
        "settings_path": str(settings_path),
        "image_path": str(image_path),
        "feature_names": list(FEATURE_NAMES),
        "bin_count": bin_count,
        "predicted_length": predicted_length,
        "expected_split_count": expected_split_count,
        "length_confidence": length_confidence,
        "boundary_top_bins": [
            {"bin": int(index), "probability": float(boundary[index].item())}
            for index in torch.argsort(boundary, descending=True)[:12].tolist()
        ],
        "path_count": len(assembly_paths),
        "paths": assembly_paths,
    }


__all__ = ["propose_scribetrain_word_segments"]
