"""Cached pixel-CNN inference and JSON-safe evidence generation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch

from .model import GlyphClassifier


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[2]
DEFAULT_MODEL_NAME = "glyph_classifier_v0_1"
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / DEFAULT_MODEL_NAME
    / f"{DEFAULT_MODEL_NAME}_best.pt"
)
DEFAULT_LABEL_MAP_PATH = MODULE_DIR / "numeric_label_map.json"

_MODEL_CACHE: dict[tuple[str, str], tuple[Any, dict, dict]] = {}


def _resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return device


def _load_label_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) != 78:
        raise ValueError(f"CNN label map must contain 78 classes: {path}")
    return {str(key): str(value) for key, value in payload.items()}


def load_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    label_map_path: str | Path = DEFAULT_LABEL_MAP_PATH,
    device_name: str = "auto",
):
    """Load and cache the CNN checkpoint, metadata, and label map."""
    resolved_model = Path(model_path).expanduser().resolve()
    resolved_labels = Path(label_map_path).expanduser().resolve()
    if not resolved_model.is_file():
        raise FileNotFoundError(f"Missing CNN checkpoint: {resolved_model}")
    if not resolved_labels.is_file():
        raise FileNotFoundError(f"Missing CNN label map: {resolved_labels}")
    device = _resolve_device(device_name)
    cache_key = (str(resolved_model), str(device))
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return (*cached, device)

    checkpoint = torch.load(
        resolved_model,
        map_location=device,
        weights_only=False,
    )
    num_classes = int(checkpoint.get("num_classes", 78))
    model = GlyphClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    label_map = _load_label_map(resolved_labels)
    if len(label_map) != num_classes:
        raise ValueError(
            f"CNN checkpoint has {num_classes} classes but label map has "
            f"{len(label_map)}."
        )
    metadata = {
        "model_name": str(checkpoint.get("model_name", DEFAULT_MODEL_NAME)),
        "model_path": str(resolved_model),
        "label_map_path": str(resolved_labels),
        "num_classes": num_classes,
        "image_size": int(checkpoint.get("image_size", 64)),
        "epoch": checkpoint.get("epoch"),
        "validation_top1": checkpoint.get("val_top1"),
        "validation_top5": checkpoint.get("val_top5"),
        "framework": "PyTorch",
        "input_polarity_mode": str(
            checkpoint.get("input_polarity_mode", "legacy_raw_invert")
        ),
    }
    _MODEL_CACHE[cache_key] = (model, metadata, label_map)
    return model, metadata, label_map, device


def _detect_background_polarity(image: Image.Image) -> str:
    """Infer whether the outer image background is predominantly dark."""
    array = np.asarray(image.convert("L"), dtype=np.uint8)
    if array.size == 0:
        raise ValueError("CNN input image is empty.")
    border = np.concatenate(
        (array[0, :], array[-1, :], array[:, 0], array[:, -1])
    )
    return "white_ink_on_black" if float(np.median(border)) < 127.5 else "black_ink_on_white"


def prepare_tensor(
    image_path: str | Path,
    image_size: int = 64,
    polarity_mode: str = "legacy_raw_invert",
):
    """Apply the checkpoint's declared resize, padding, and polarity contract."""
    source = Path(image_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"CNN input image does not exist: {source}")
    with Image.open(source) as opened:
        image = opened.convert("L")
    source_width, source_height = image.size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("CNN input image has invalid dimensions.")

    polarity = _detect_background_polarity(image)
    supported_modes = {
        "legacy_raw_invert",
        "normalize_black_ink_on_white",
        "white_ink_on_black__ink_1_background_0",
    }
    if polarity_mode not in supported_modes:
        raise ValueError(
            f"Unsupported CNN input_polarity_mode {polarity_mode!r}; "
            f"expected one of {sorted(supported_modes)}."
        )
    if polarity_mode == "normalize_black_ink_on_white" and polarity == "white_ink_on_black":
        image = ImageOps.invert(image)
    elif (
        polarity_mode == "white_ink_on_black__ink_1_background_0"
        and polarity == "black_ink_on_white"
    ):
        image = ImageOps.invert(image)

    scale = min(image_size / source_width, image_size / source_height)
    resized_width = max(1, int(source_width * scale))
    resized_height = max(1, int(source_height * scale))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = image.resize((resized_width, resized_height), resampling)
    canvas_color = (
        0
        if polarity_mode == "white_ink_on_black__ink_1_background_0"
        else 255
    )
    canvas = Image.new("L", (image_size, image_size), color=canvas_color)
    offset = (
        (image_size - resized_width) // 2,
        (image_size - resized_height) // 2,
    )
    canvas.paste(image, offset)
    if polarity_mode == "white_ink_on_black__ink_1_background_0":
        array = (np.asarray(canvas, dtype=np.uint8) >= 128).astype(np.float32)
        tensor_transform = "threshold_white_ink_as_one"
    else:
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        array = 1.0 - array
        tensor_transform = "one_minus_grayscale"
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    preprocessing = {
        "source_path": str(source),
        "source_size": {"width": source_width, "height": source_height},
        "detected_polarity": polarity,
        "polarity_mode": polarity_mode,
        "polarity_normalized_to": (
            "white_ink_on_black"
            if polarity_mode == "white_ink_on_black__ink_1_background_0"
            else
            "black_ink_on_white"
            if polarity_mode == "normalize_black_ink_on_white"
            else "preserved_source_polarity"
        ),
        "resize_mode": (
            "aspect_preserving_bilinear_with_black_padding"
            if polarity_mode == "white_ink_on_black__ink_1_background_0"
            else "aspect_preserving_bilinear_with_white_padding"
        ),
        "model_size": {"width": image_size, "height": image_size},
        "resized_size": {"width": resized_width, "height": resized_height},
        "paste_offset": {"x": offset[0], "y": offset[1]},
        "tensor_transform": tensor_transform,
    }
    return tensor, preprocessing


def predict(crop_path: str | Path, settings: dict | None = None) -> dict:
    """Return top-k CNN candidates and calibrated diagnostic evidence."""
    settings = settings or {}
    model, metadata, label_map, device = load_model(
        model_path=settings.get("model_path", DEFAULT_MODEL_PATH),
        label_map_path=settings.get("label_map_path", DEFAULT_LABEL_MAP_PATH),
        device_name=str(settings.get("device", "auto")),
    )
    top_k = max(1, min(int(settings.get("top_k", 5)), metadata["num_classes"]))
    tensor, preprocessing = prepare_tensor(
        crop_path,
        image_size=metadata["image_size"],
        polarity_mode=str(
            settings.get(
                "input_polarity_mode",
                metadata["input_polarity_mode"],
            )
        ),
    )
    with torch.inference_mode():
        logits = model(tensor.to(device))
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()
    top_probabilities, top_indexes = torch.topk(probabilities, k=top_k)
    candidates = []
    for rank, (class_tensor, probability_tensor) in enumerate(
        zip(top_indexes, top_probabilities),
        start=1,
    ):
        class_id = int(class_tensor.item())
        label = label_map[str(class_id)]
        candidates.append(
            {
                "rank": rank,
                "class_id": class_id,
                "label": label,
                "text": label,
                "confidence": float(probability_tensor.item()),
                "source": metadata["model_name"],
                "evidence_kind": "pixel_cnn",
                "provenance": "character_detector_cnn",
            }
        )

    probability_array = probabilities.numpy()
    entropy = -float(
        np.sum(probability_array * np.log(np.clip(probability_array, 1e-12, 1.0)))
    )
    normalized_entropy = entropy / math.log(max(2, metadata["num_classes"]))
    top1 = candidates[0]["confidence"]
    top2 = candidates[1]["confidence"] if len(candidates) > 1 else 0.0
    return {
        "schema_version": "n05_candidate_evidence_v1",
        "expert_name": "character_detector",
        "evidence_family": "pixel_cnn",
        "model": {**metadata, "device": str(device)},
        "preprocessing": preprocessing,
        "top_k": top_k,
        "top1_confidence": top1,
        "top1_margin": top1 - top2,
        "normalized_entropy": normalized_entropy,
        "candidates": candidates,
    }
