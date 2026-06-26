"""Train a ScribeTrace-style word sequence splitter.

This is the PyTorch successor lane to ``scribetrace_word_trainer.py``.
The RandomForest baseline sees one whole-word feature vector, which is useful
for proving the idea but weak for split localization. This trainer keeps the
same synthetic word renderer and converts each word into a left-to-right
sequence of topology features, then predicts boundary bins and word length.

v0.1 intentionally trains a splitter-first model:
    synthetic word image -> skeleton/topology x-bins -> boundary + length heads

It does not run theoretical reconstruction, ANTAR, ScriLog, or Scrististics.
Those remain downstream tools once this model proposes plausible letter spans.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as error:  # pragma: no cover - runtime environment guard.
    raise SystemExit(
        "PyTorch is required for scribetrain_monster_v0_1.py. "
        "Run it inside the project .venv."
    ) from error

LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(LOCAL_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

try:
    from scripts.Cyber_Lin_Kuei_Assembly.word_level_ocr_trainer import (
        PROJECT_ROOT,
        WordSample,
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
        WordSample,
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
    from scripts.N05handwritten_ocr.scribetrace.trace_skeleton import (
        SkeletonGraph,
        SkeletonPointExtractor,
        TraceSkeletonizer,
    )
except ImportError:
    from N05handwritten_ocr.scribetrace.trace_skeleton import (  # type: ignore
        SkeletonGraph,
        SkeletonPointExtractor,
        TraceSkeletonizer,
    )


DEFAULT_SETTINGS_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "Cyber_Lin_Kuei_Assembly"
    / "scribetrain/scribetrain_monster_v0_1_settings.json"
)
FEATURE_NAMES = (
    "ink_density",
    "skeleton_density",
    "endpoint_density",
    "junction_density",
    "ink_top_y",
    "ink_bottom_y",
    "ink_center_y",
    "ink_span_y",
    "vertical_transition_density",
    "projection_delta",
)
SCRISTISTICS_PROFILE_PATH = (
    PROJECT_ROOT / "datasets" / "scrististics" / "empirical_profiles_limit_100.json"
)
_SCRISTISTICS_GEOMETRY_ENVELOPE: dict | None = None


@dataclass(frozen=True)
class SequenceSample:
    """One generated training row returned by the dataset.

    Args:
        features: Float tensor shaped ``[bin_count, feature_count]``.
        boundaries: Soft boundary heatmap target shaped ``[bin_count]``.
        hard_boundaries: One-hot true split bins shaped ``[bin_count]``.
        bridge_targets: Soft per-bin target for likely joined/unsafe boundaries.
        cut_safety_targets: Soft per-bin target for likely safe cuts.
        length: Integer token count target.
        text: Human-readable synthetic word text for debugging.
        image: Original rendered word image for debug previews.
    """

    features: torch.Tensor
    boundaries: torch.Tensor
    hard_boundaries: torch.Tensor
    bridge_targets: torch.Tensor
    cut_safety_targets: torch.Tensor
    length: torch.Tensor
    text: str
    image: np.ndarray


def _choose_device(settings: dict) -> torch.device:
    """Resolve the requested training device.

    Args:
        settings: Root settings dictionary.

    Returns:
        A torch device. ``auto`` prefers CUDA, then Apple MPS, then CPU.
    """

    requested = str(settings.get("training", {}).get("device", "auto")).lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ink_mask_from_rendered_word(image: np.ndarray, threshold_value: int = 128) -> np.ndarray:
    """Convert black-on-white rendered text to white-on-black ink evidence.

    Args:
        image: Rendered uint8 word image.
        threshold_value: Pixel values below this are treated as ink.

    Returns:
        Binary uint8 mask where ink pixels are 255.
    """

    gray = np.asarray(image)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    return np.where(gray < int(threshold_value), 255, 0).astype(np.uint8)


def _scrististics_geometry_envelope() -> dict:
    """Load a global Armenian-letter geometry envelope from ScriStatistics."""

    global _SCRISTISTICS_GEOMETRY_ENVELOPE
    if _SCRISTISTICS_GEOMETRY_ENVELOPE is not None:
        return _SCRISTISTICS_GEOMETRY_ENVELOPE

    fallback = {
        "available": False,
        "profile_path": str(SCRISTISTICS_PROFILE_PATH),
        "reference_image_height": 64.0,
    }
    if not SCRISTISTICS_PROFILE_PATH.is_file():
        _SCRISTISTICS_GEOMETRY_ENVELOPE = fallback
        return fallback

    try:
        profile = json.loads(SCRISTISTICS_PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        _SCRISTISTICS_GEOMETRY_ENVELOPE = fallback
        return fallback

    classes = profile.get("classes", {})
    width_p90 = []
    width_p10 = []
    height_p90 = []
    height_p10 = []
    aspect_p90 = []
    aspect_p10 = []
    for class_profile in classes.values():
        geometry = class_profile.get("geometry_profile", {}) if isinstance(class_profile, dict) else {}
        for target, feature, key in (
            (width_p10, "ink_bbox_width", "p10"),
            (width_p90, "ink_bbox_width", "p90"),
            (height_p10, "ink_bbox_height", "p10"),
            (height_p90, "ink_bbox_height", "p90"),
            (aspect_p10, "ink_aspect_ratio", "p10"),
            (aspect_p90, "ink_aspect_ratio", "p90"),
        ):
            value = geometry.get(feature, {}).get(key)
            if isinstance(value, (int, float)) and value > 0:
                target.append(float(value))

    if not width_p90 or not height_p90 or not aspect_p90:
        _SCRISTISTICS_GEOMETRY_ENVELOPE = fallback
        return fallback

    envelope = {
        "available": True,
        "profile_path": str(SCRISTISTICS_PROFILE_PATH),
        "reference_image_height": 64.0,
        "letter_count": len(classes),
        # These are deliberately broad all-letter envelopes. They reject only
        # segments that look implausible for almost any Armenian glyph.
        "width_min_p10": min(width_p10),
        "width_max_p90": max(width_p90),
        "height_min_p10": min(height_p10),
        "height_max_p90": max(height_p90),
        "aspect_min_p10": min(aspect_p10),
        "aspect_max_p90": max(aspect_p90),
    }
    _SCRISTISTICS_GEOMETRY_ENVELOPE = envelope
    return envelope


def _ink_bbox_for_bin_span(
    image: np.ndarray | None,
    start_bin: int,
    end_bin: int,
    bin_count: int,
) -> dict:
    """Measure an actual ink bbox inside one candidate segment span."""

    if image is None:
        return {"available": False}
    mask = _ink_mask_from_rendered_word(image)
    height, width = mask.shape
    x1 = int(round(max(0, start_bin) * width / max(1, bin_count)))
    x2 = int(round(max(start_bin + 1, end_bin) * width / max(1, bin_count)))
    x1 = max(0, min(width - 1, x1))
    x2 = max(x1 + 1, min(width, x2))
    crop = mask[:, x1:x2]
    coords = cv2.findNonZero(crop)
    if coords is None:
        return {
            "available": True,
            "image_height": height,
            "image_width": width,
            "span_x1": x1,
            "span_x2": x2,
            "ink_bbox_width": 0,
            "ink_bbox_height": 0,
            "ink_aspect_ratio": 0.0,
        }
    x, y, bbox_width, bbox_height = cv2.boundingRect(coords)
    return {
        "available": True,
        "image_height": height,
        "image_width": width,
        "span_x1": x1,
        "span_x2": x2,
        "ink_bbox_x": int(x1 + x),
        "ink_bbox_y": int(y),
        "ink_bbox_width": int(bbox_width),
        "ink_bbox_height": int(bbox_height),
        "ink_aspect_ratio": float(bbox_width / max(1, bbox_height)),
    }


def _bin_ranges(width: int, bin_count: int) -> list[tuple[int, int]]:
    """Split an image width into deterministic x-ranges.

    Args:
        width: Image width in pixels. 
        bin_count: Number of model time steps.

    Returns:
        A list of exclusive ``(x1, x2)`` ranges, one per bin.
    """

    width = max(1, int(width))
    ranges = []
    for index in range(bin_count):
        x1 = int(round(index * width / bin_count))
        x2 = int(round((index + 1) * width / bin_count))
        x1 = max(0, min(width - 1, x1))
        x2 = max(x1 + 1, min(width, x2))
        ranges.append((x1, x2))
    return ranges


def _point_count_by_x(points: Iterable, width: int) -> np.ndarray:
    """Count graph points at every x-coordinate.

    Args:
        points: SkeletonPoint-like objects with ``x`` attributes.
        width: Image width.

    Returns:
        Integer array shaped ``[width]``.
    """

    counts = np.zeros((width,), dtype=np.float32)
    for point in points:
        if 0 <= point.x < width:
            counts[point.x] += 1.0
    return counts


def boundary_heatmap_bins(
    split_x_positions: Iterable[int],
    width: int,
    bin_count: int,
    radius: int = 2,
) -> tuple[list[float], list[int]]:
    """Convert split x-positions into soft and hard boundary targets.

    Args:
        split_x_positions: True split x-coordinates from the renderer.
        width: Rendered image width.
        bin_count: Number of x-bins/time steps.
        radius: Number of neighboring bins receiving softer target values.

    Returns:
        ``(soft_heatmap, hard_bins)`` where the soft target peaks at 1.0 and
        decays linearly around each true split, while hard bins preserve the
        exact split count for metrics.
    """

    soft = np.zeros((bin_count,), dtype=np.float32)
    hard = np.zeros((bin_count,), dtype=np.int64)
    width = max(1, int(width))
    radius = max(0, int(radius))
    for position in split_x_positions:
        center = int(round((float(position) / float(width)) * (bin_count - 1)))
        center = max(0, min(bin_count - 1, center))
        hard[center] = 1
        for offset in range(-radius, radius + 1):
            index = center + offset
            if not 0 <= index < bin_count:
                continue
            value = 1.0 if radius == 0 else 1.0 - (abs(offset) / float(radius + 1))
            soft[index] = max(float(soft[index]), value)
    return soft.tolist(), hard.astype(int).tolist()


def bridge_and_cut_safety_targets(
    image: np.ndarray,
    hard_boundaries: Iterable[int],
    bin_count: int,
    radius: int = 2,
    ink_threshold: float = 0.08,
) -> tuple[list[float], list[float]]:
    """Build bridge-risk and cut-safety labels around split candidates.

    Args:
        image: Rendered word image.
        hard_boundaries: One-hot exact boundary bins.
        bin_count: Number of x-bins/time steps.
        radius: Neighborhood around each exact boundary.
        ink_threshold: Mean ink ratio above which a boundary looks bridged.

    Returns:
        ``(bridge_risk, cut_safety)`` soft targets. A boundary with visible ink
        crossing its local x-band is treated as risky; clean valleys are safer.
    """

    mask = _ink_mask_from_rendered_word(image)
    height, width = mask.shape
    binary = (mask > 0).astype(np.uint8)
    ranges = _bin_ranges(width, bin_count)
    hard = list(int(value) for value in hard_boundaries)
    bridge = np.zeros((bin_count,), dtype=np.float32)
    cut_safety = np.zeros((bin_count,), dtype=np.float32)
    radius = max(0, int(radius))

    for center, is_boundary in enumerate(hard):
        if not is_boundary:
            continue
        x1, x2 = ranges[center]
        local_ink_ratio = float(binary[:, x1:x2].sum()) / max(1.0, float(height * (x2 - x1)))
        risk = 1.0 if local_ink_ratio >= ink_threshold else max(0.0, local_ink_ratio / max(1e-6, ink_threshold))
        safe = 1.0 - min(1.0, risk)
        for offset in range(-radius, radius + 1):
            index = center + offset
            if not 0 <= index < bin_count:
                continue
            weight = 1.0 if radius == 0 else 1.0 - (abs(offset) / float(radius + 1))
            bridge[index] = max(float(bridge[index]), risk * weight)
            cut_safety[index] = max(float(cut_safety[index]), safe * weight)

    return bridge.tolist(), cut_safety.tolist()


def split_word_pools(
    words: list[WordSample],
    seed: int,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> tuple[list[WordSample], list[WordSample], list[WordSample]]:
    """Split word vocabulary so train/validation/test do not share words.

    Args:
        words: Tokenized Armenian word candidates.
        seed: Deterministic shuffle seed.
        validation_ratio: Fraction of unique words reserved for validation.
        test_ratio: Fraction of unique words reserved for final testing.

    Returns:
        Train, validation, and test word pools.
    """

    unique_by_text = {sample.text: sample for sample in words}
    unique_words = list(unique_by_text.values())
    rng = random.Random(seed)
    rng.shuffle(unique_words)
    total = len(unique_words)
    test_count = max(1, int(round(total * test_ratio)))
    validation_count = max(1, int(round(total * validation_ratio)))
    train_count = max(1, total - validation_count - test_count)
    train_words = unique_words[:train_count]
    validation_words = unique_words[train_count : train_count + validation_count]
    test_words = unique_words[train_count + validation_count :]
    if not validation_words:
        validation_words = train_words[-1:]
    if not test_words:
        test_words = validation_words[-1:]
    return train_words, validation_words, test_words


def extract_sequence_features(
    image: np.ndarray,
    bin_count: int,
    feature_settings: dict | None = None,
) -> np.ndarray:
    """Extract ScribeTrace-inspired local features from a word image.

    Args:
        image: Black-on-white synthetic word image.
        bin_count: Number of horizontal bins/time steps.
        feature_settings: Optional threshold and topology settings.

    Returns:
        Float32 matrix shaped ``[bin_count, len(FEATURE_NAMES)]``.
    """

    feature_settings = feature_settings or {}
    mask = _ink_mask_from_rendered_word(
        image,
        threshold_value=int(feature_settings.get("threshold_value", 128)),
    )
    height, width = mask.shape
    binary = (mask > 0).astype(np.uint8)
    skeleton = TraceSkeletonizer({}).skeletonize(mask)
    skeleton_binary = (skeleton > 0).astype(np.uint8)
    points = SkeletonPointExtractor().extract_points(skeleton)
    graph = SkeletonGraph(points)

    endpoint_by_x = _point_count_by_x(graph.endpoints(), width)
    junction_by_x = _point_count_by_x(graph.junction_cluster_centers(), width)
    projection = binary.sum(axis=0).astype(np.float32) / max(1.0, float(height))
    projection_delta = np.abs(np.gradient(projection)) if width > 1 else np.zeros_like(projection)
    ranges = _bin_ranges(width, bin_count)
    rows: list[list[float]] = []

    for x1, x2 in ranges:
        ink_region = binary[:, x1:x2]
        skeleton_region = skeleton_binary[:, x1:x2]
        projection_delta_region = projection_delta[x1:x2]
        ink_pixels = float(ink_region.sum())
        area = float(max(1, ink_region.size))
        vertical_transitions = float(np.abs(np.diff(ink_region, axis=0)).sum())

        ys = np.argwhere(ink_region > 0)[:, 0] if ink_pixels > 0 else np.asarray([])
        if ys.size:
            top_y = float(ys.min()) / max(1.0, float(height - 1))
            bottom_y = float(ys.max()) / max(1.0, float(height - 1))
            center_y = float(ys.mean()) / max(1.0, float(height - 1))
            span_y = float(ys.max() - ys.min() + 1) / max(1.0, float(height))
        else:
            top_y = bottom_y = center_y = span_y = 0.0

        rows.append(
            [
                ink_pixels / area,
                float(skeleton_region.sum()) / area,
                float(endpoint_by_x[x1:x2].sum()) / max(1.0, float(x2 - x1)),
                float(junction_by_x[x1:x2].sum()) / max(1.0, float(x2 - x1)),
                top_y,
                bottom_y,
                center_y,
                span_y,
                vertical_transitions / area,
                float(projection_delta_region.mean()) if projection_delta_region.size else 0.0,
            ]
        )

    return np.asarray(rows, dtype=np.float32)


class ScribeTrainSyntheticWordDataset(Dataset):
    """Generate deterministic synthetic word sequence samples on demand.

    Args:
        settings: Root settings dictionary.
        split: Human-readable split name, used only for seed separation.
        sample_count: Number of synthetic samples exposed by this dataset.
        seed_offset: Offset added to the global seed so splits do not overlap.
    """

    def __init__(
        self,
        settings: dict,
        split: str,
        sample_count: int,
        seed_offset: int,
        words: list[WordSample] | None = None,
        glyph_paths: dict[int, list[Path]] | None = None,
        tail_profiles: dict[int, str] | None = None,
    ):
        self.settings = settings
        self.split = split
        self.sample_count = int(sample_count)
        self.seed = int(settings.get("random_seed", 42)) + int(seed_offset)
        self.dataset_settings = settings["dataset"]
        self.rendering = settings["rendering"]
        self.feature_settings = settings.get("feature_extraction", {})
        self.boundary_bin_count = int(self.dataset_settings.get("boundary_bin_count", 64))
        self.boundary_heatmap_radius = int(self.dataset_settings.get("boundary_heatmap_radius", 2))
        self.label_map = load_label_map(self.dataset_settings["label_map_path"])
        _, token_to_char = build_token_maps(self.label_map)
        self.tail_profiles = tail_profiles or build_tail_profiles(token_to_char)
        self.words = words or load_word_samples(settings, build_token_maps(self.label_map)[0])
        self.glyph_paths = glyph_paths or collect_glyph_paths(
            self.dataset_settings["matenadata_dir"],
            self.label_map,
        )

    def __len__(self) -> int:
        """Return the configured synthetic sample count."""

        return self.sample_count

    def __getitem__(self, index: int) -> SequenceSample:
        """Render and featurize one deterministic synthetic word.

        Args:
            index: Dataset index.

        Returns:
            A SequenceSample with topology features and boundary targets.
        """

        rng = random.Random(self.seed + int(index))
        sample = rng.choice(self.words)
        rendered = render_synthetic_word(
            sample=sample,
            glyph_paths=self.glyph_paths,
            tail_profiles=self.tail_profiles,
            rendering=self.rendering,
            rng=rng,
        )
        features = extract_sequence_features(
            rendered.image,
            self.boundary_bin_count,
            self.feature_settings,
        )
        boundaries, hard_boundaries = boundary_heatmap_bins(
            rendered.split_x_positions,
            rendered.image.shape[1],
            self.boundary_bin_count,
            radius=self.boundary_heatmap_radius,
        )
        bridge_targets, cut_safety_targets = bridge_and_cut_safety_targets(
            image=rendered.image,
            hard_boundaries=hard_boundaries,
            bin_count=self.boundary_bin_count,
            radius=self.boundary_heatmap_radius,
            ink_threshold=float(self.dataset_settings.get("bridge_ink_threshold", 0.08)),
        )
        return SequenceSample(
            features=torch.tensor(features, dtype=torch.float32),
            boundaries=torch.tensor(boundaries, dtype=torch.float32),
            hard_boundaries=torch.tensor(hard_boundaries, dtype=torch.float32),
            bridge_targets=torch.tensor(bridge_targets, dtype=torch.float32),
            cut_safety_targets=torch.tensor(cut_safety_targets, dtype=torch.float32),
            length=torch.tensor(len(sample.token_ids), dtype=torch.long),
            text=sample.text,
            image=rendered.image,
        )


def collate_sequence_samples(samples: list[SequenceSample]) -> dict:
    """Stack SequenceSample records into one mini-batch.

    Args:
        samples: Dataset records.

    Returns:
        Batch dictionary used by the training loop.
    """

    return {
        "features": torch.stack([sample.features for sample in samples]),
        "boundaries": torch.stack([sample.boundaries for sample in samples]),
        "hard_boundaries": torch.stack([sample.hard_boundaries for sample in samples]),
        "bridge_targets": torch.stack([sample.bridge_targets for sample in samples]),
        "cut_safety_targets": torch.stack([sample.cut_safety_targets for sample in samples]),
        "lengths": torch.stack([sample.length for sample in samples]),
        "texts": [sample.text for sample in samples],
        "images": [sample.image for sample in samples],
    }


class ScribeTrainWordTraceModel(nn.Module):
    """Small local-feature sequence model for word boundary prediction."""

    def __init__(
        self,
        feature_count: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        max_length_class: int,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(feature_count, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        encoded_size = hidden_size * 2
        self.boundary_head = nn.Sequential(
            nn.Linear(encoded_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.bridge_head = nn.Sequential(
            nn.Linear(encoded_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.cut_safety_head = nn.Sequential(
            nn.Linear(encoded_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.length_head = nn.Sequential(
            nn.Linear(encoded_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max_length_class + 1),
        )

    def forward(self, features: torch.Tensor) -> dict:
        """Run one model pass.

        Args:
            features: Tensor shaped ``[batch, bins, feature_count]``.

        Returns:
            Dict with boundary logits ``[batch, bins]`` and length logits.
        """

        projected = self.input_projection(features)
        encoded, _ = self.encoder(projected)
        boundary_logits = self.boundary_head(encoded).squeeze(-1)
        bridge_logits = self.bridge_head(encoded).squeeze(-1)
        cut_safety_logits = self.cut_safety_head(encoded).squeeze(-1)
        mean_pool = encoded.mean(dim=1)
        max_pool = encoded.max(dim=1).values
        length_logits = self.length_head(torch.cat([mean_pool, max_pool], dim=1))
        return {
            "boundary_logits": boundary_logits,
            "bridge_logits": bridge_logits,
            "cut_safety_logits": cut_safety_logits,
            "length_logits": length_logits,
        }


def _boundary_metrics(boundary_logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute thresholded boundary and split-count metrics.

    Args:
        boundary_logits: Raw model boundary logits.
        targets: Binary multi-label boundary targets.

    Returns:
        JSON-safe metric dictionary.
    """

    predictions = (torch.sigmoid(boundary_logits) >= 0.5).to(torch.int64)
    truth = targets.to(torch.int64)
    tp = int(((predictions == 1) & (truth == 1)).sum().item())
    fp = int(((predictions == 1) & (truth == 0)).sum().item())
    fn = int(((predictions == 0) & (truth == 1)).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    exact = float(torch.all(predictions == truth, dim=1).float().mean().item())
    predicted_split_counts = predictions.sum(dim=1)
    true_split_counts = truth.sum(dim=1)
    split_count_delta = torch.abs(predicted_split_counts - true_split_counts)
    split_count_exact = int((split_count_delta == 0).sum().item())
    split_count_within_one = int((split_count_delta <= 1).sum().item())
    sample_count = int(truth.shape[0])
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "boundary_exact_accuracy": exact,
        "split_count_exact_accuracy": split_count_exact / max(1, sample_count),
        "split_count_within_1_accuracy": split_count_within_one / max(1, sample_count),
        "split_count_exact": split_count_exact,
        "split_count_within_1": split_count_within_one,
        "split_count_sample_count": sample_count,
        "predicted_split_count_mean": float(predicted_split_counts.float().mean().item()),
        "true_split_count_mean": float(true_split_counts.float().mean().item()),
        "boundary_true_positive": tp,
        "boundary_false_positive": fp,
        "boundary_false_negative": fn,
    }


def decode_boundary_peaks(
    probabilities: torch.Tensor,
    threshold: float = 0.35,
    min_distance: int = 2,
) -> torch.Tensor:
    """Decode boundary heatmaps into sparse peak candidates.

    Args:
        probabilities: Sigmoid probability tensor shaped ``[batch, bins]``.
        threshold: Minimum probability for a peak to be considered.
        min_distance: Minimum bin distance between accepted peaks.

    Returns:
        Binary tensor shaped like ``probabilities`` with one active bin per
        accepted local peak.
    """

    probabilities = probabilities.detach().cpu()
    decoded = torch.zeros_like(probabilities, dtype=torch.int64)
    for row_index in range(probabilities.shape[0]):
        row = probabilities[row_index]
        candidates = []
        for bin_index in range(row.numel()):
            value = float(row[bin_index].item())
            if value < threshold:
                continue
            left = float(row[bin_index - 1].item()) if bin_index > 0 else -1.0
            right = float(row[bin_index + 1].item()) if bin_index + 1 < row.numel() else -1.0
            if value >= left and value >= right:
                candidates.append((value, bin_index))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        accepted: list[int] = []
        for _, bin_index in candidates:
            if all(abs(bin_index - existing) >= min_distance for existing in accepted):
                accepted.append(bin_index)
        for bin_index in accepted:
            decoded[row_index, bin_index] = 1
    return decoded


def _peak_distance_metrics(
    boundary_logits: torch.Tensor,
    hard_targets: torch.Tensor,
    threshold: float,
    min_distance: int,
) -> dict:
    """Evaluate decoded boundary peaks against exact split bins.

    Args:
        boundary_logits: Raw model boundary logits.
        hard_targets: One-hot true split bins.
        threshold: Peak probability threshold.
        min_distance: Non-maximum suppression distance in bins.

    Returns:
        JSON-safe decoded split metrics.
    """

    probabilities = torch.sigmoid(boundary_logits.detach().cpu())
    predictions = decode_boundary_peaks(probabilities, threshold, min_distance)
    truth = hard_targets.detach().cpu().to(torch.int64)
    predicted_counts = predictions.sum(dim=1)
    true_counts = truth.sum(dim=1)
    count_delta = torch.abs(predicted_counts - true_counts)
    count_exact = int((count_delta == 0).sum().item())
    count_pm1 = int((count_delta <= 1).sum().item())
    sample_count = int(truth.shape[0])

    matched = 0
    total_true = int(true_counts.sum().item())
    total_predicted = int(predicted_counts.sum().item())
    distance_sum = 0.0
    distance_count = 0
    for row_index in range(truth.shape[0]):
        true_indices = _positive_bin_indices(truth[row_index])
        predicted_indices = _positive_bin_indices(predictions[row_index])
        used_predictions: set[int] = set()
        for true_index in true_indices:
            available_predictions = [
                prediction
                for prediction in predicted_indices
                if prediction not in used_predictions
            ]
            if not available_predictions:
                continue
            best_prediction = min(
                available_predictions,
                key=lambda value: (abs(value - true_index), value),
            )
            distance = abs(best_prediction - true_index)
            distance_sum += float(distance)
            distance_count += 1
            if distance <= min_distance:
                matched += 1
                used_predictions.add(best_prediction)

    precision = matched / max(1, total_predicted)
    recall = matched / max(1, total_true)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    return {
        "peak_threshold": float(threshold),
        "peak_min_distance_bins": int(min_distance),
        "peak_precision": precision,
        "peak_recall": recall,
        "peak_f1": f1,
        "peak_split_count_exact_accuracy": count_exact / max(1, sample_count),
        "peak_split_count_within_1_accuracy": count_pm1 / max(1, sample_count),
        "peak_split_count_exact": count_exact,
        "peak_split_count_within_1": count_pm1,
        "peak_predicted_split_count_mean": float(predicted_counts.float().mean().item()),
        "peak_true_split_count_mean": float(true_counts.float().mean().item()),
        "peak_mean_nearest_true_distance_bins": (
            distance_sum / max(1, distance_count)
        ),
    }


def _segment_geometry_for_path(
    cut_bins: list[int],
    sequence_features: torch.Tensor | np.ndarray | None,
    bin_count: int,
    image: np.ndarray | None = None,
) -> dict:
    """Judge whether path segments look like plausible letter units.

    This is the first Scrististics-style geometric head. It does not need
    retraining: it uses the existing per-bin ScribeTrain features and asks
    whether each proposed span is suspiciously wide, narrow, empty, or complex.
    """

    if sequence_features is None:
        return {
            "available": False,
            "segment_count": len(cut_bins) + 1,
            "path_geometry_penalty": 0.0,
            "path_geometry_score": 1.0,
            "segments": [],
        }

    features = sequence_features.detach().cpu().numpy() if isinstance(sequence_features, torch.Tensor) else np.asarray(sequence_features)
    if features.ndim != 2 or features.shape[0] <= 0:
        return {
            "available": False,
            "segment_count": len(cut_bins) + 1,
            "path_geometry_penalty": 0.0,
            "path_geometry_score": 1.0,
            "segments": [],
        }

    safe_bin_count = int(features.shape[0])
    sorted_cuts = sorted(
        {
            max(0, min(safe_bin_count - 1, int(bin_index)))
            for bin_index in cut_bins
        }
    )
    boundaries = [0, *sorted_cuts, safe_bin_count]
    segment_count = max(1, len(boundaries) - 1)
    expected_span = max(1.0, float(safe_bin_count) / float(segment_count))
    indexes = {name: index for index, name in enumerate(FEATURE_NAMES)}
    envelope = _scrististics_geometry_envelope()
    segments = []
    penalties = []

    for segment_index in range(segment_count):
        start = int(boundaries[segment_index])
        end = int(boundaries[segment_index + 1])
        if segment_index > 0:
            start = min(safe_bin_count - 1, start + 1)
        end = max(start + 1, min(safe_bin_count, end))
        ink_bbox = _ink_bbox_for_bin_span(image, start, end, safe_bin_count)
        rows = features[start:end]
        span_bins = max(1, end - start)
        span_ratio = float(span_bins) / expected_span
        ink_density = float(rows[:, indexes["ink_density"]].mean()) if rows.size else 0.0
        skeleton_density = float(rows[:, indexes["skeleton_density"]].mean()) if rows.size else 0.0
        endpoint_sum = float(rows[:, indexes["endpoint_density"]].sum()) if rows.size else 0.0
        junction_sum = float(rows[:, indexes["junction_density"]].sum()) if rows.size else 0.0
        vertical_complexity = float(rows[:, indexes["vertical_transition_density"]].mean()) if rows.size else 0.0

        too_wide_score = max(0.0, min(1.0, (span_ratio - 1.45) / 0.75))
        too_narrow_score = max(0.0, min(1.0, (0.55 - span_ratio) / 0.35))
        empty_score = max(0.0, min(1.0, (0.015 - ink_density) / 0.015))
        complexity_score = max(0.0, min(1.0, (endpoint_sum + junction_sum - 2.8) / 4.0))
        profile_too_wide = 0.0
        profile_too_narrow = 0.0
        profile_aspect_outlier = 0.0
        if envelope.get("available") and ink_bbox.get("available"):
            scale = float(ink_bbox.get("image_height", 32)) / max(
                1.0,
                float(envelope.get("reference_image_height", 64.0)),
            )
            width = float(ink_bbox.get("ink_bbox_width", 0))
            aspect = float(ink_bbox.get("ink_aspect_ratio", 0.0))
            width_max = float(envelope.get("width_max_p90", 46.0)) * scale
            width_min = float(envelope.get("width_min_p10", 9.0)) * scale
            aspect_max = float(envelope.get("aspect_max_p90", 2.4))
            aspect_min = float(envelope.get("aspect_min_p10", 0.25))
            profile_too_wide = max(0.0, min(1.0, (width - width_max * 1.12) / max(1.0, width_max * 0.55)))
            profile_too_narrow = (
                max(0.0, min(1.0, (width_min * 0.55 - width) / max(1.0, width_min * 0.55)))
                if width > 0
                else 1.0
            )
            profile_aspect_outlier = max(
                max(0.0, min(1.0, (aspect - aspect_max * 1.10) / max(0.25, aspect_max * 0.40))),
                max(0.0, min(1.0, (aspect_min * 0.80 - aspect) / max(0.10, aspect_min * 0.80))),
            )
            too_wide_score = max(too_wide_score, profile_too_wide, 0.65 * profile_aspect_outlier)
            too_narrow_score = max(too_narrow_score, profile_too_narrow)
        likely_joined_score = max(too_wide_score, min(1.0, 0.55 * too_wide_score + 0.45 * complexity_score))
        fragment_score = max(too_narrow_score, empty_score)
        plausible_score = max(
            0.0,
            1.0
            - max(too_wide_score, too_narrow_score)
            - 0.35 * empty_score
            - 0.20 * complexity_score,
        )
        penalty = max(likely_joined_score, fragment_score)
        penalties.append(penalty)
        segments.append(
            {
                "segment_index": segment_index,
                "start_bin": start,
                "end_bin": end,
                "span_bins": span_bins,
                "span_ratio": span_ratio,
                "ink_density": ink_density,
                "skeleton_density": skeleton_density,
                "endpoint_sum": endpoint_sum,
                "junction_sum": junction_sum,
                "vertical_complexity": vertical_complexity,
                "ink_bbox": ink_bbox,
                "scrististics_profile": {
                    "available": bool(envelope.get("available")),
                    "profile_too_wide": profile_too_wide,
                    "profile_too_narrow": profile_too_narrow,
                    "profile_aspect_outlier": profile_aspect_outlier,
                },
                "too_wide": too_wide_score,
                "too_narrow": too_narrow_score,
                "likely_joined_letters": likely_joined_score,
                "likely_fragment": fragment_score,
                "plausible_single_letter": plausible_score,
            }
        )

    mean_penalty = float(sum(penalties) / max(1, len(penalties)))
    max_penalty = float(max(penalties) if penalties else 0.0)
    path_penalty = 0.65 * mean_penalty + 0.35 * max_penalty
    return {
        "available": True,
        "segment_count": segment_count,
        "expected_span_bins": expected_span,
        "path_geometry_penalty": path_penalty,
        "path_geometry_score": max(0.0, 1.0 - path_penalty),
        "suspicious_segment_count": sum(1 for value in penalties if value >= 0.50),
        "segments": segments,
    }


def build_segmentation_paths(
    boundary_probabilities: torch.Tensor,
    bridge_probabilities: torch.Tensor,
    cut_safety_probabilities: torch.Tensor,
    threshold: float,
    min_distance: int,
    top_k: int = 5,
    expected_split_count: int | None = None,
    length_confidence: float = 0.0,
    length_weight: float = 0.25,
    length_penalty_strength: float = 2.0,
    image: np.ndarray | None = None,
    snap_radius_bins: int = 0,
    recall_threshold: float | None = None,
    recall_candidate_limit: int = 32,
    sequence_features: torch.Tensor | np.ndarray | None = None,
    segment_geometry_weight: float = 0.15,
) -> list[dict]:
    """Build ranked segmentation paths from ScribeTrain head outputs.

    Args:
        boundary_probabilities: Per-bin probability of a split neighborhood.
        bridge_probabilities: Per-bin probability that a cut crosses joined ink.
        cut_safety_probabilities: Per-bin probability that a cut is safe.
        threshold: Minimum boundary probability for peak candidates.
        min_distance: Non-maximum suppression distance in bins.
        top_k: Maximum number of alternative paths to return.
        expected_split_count: Optional length-aware target cut count.
        length_confidence: Confidence for the expected split count.
        length_weight: Score penalty weight for length mismatch.
        length_penalty_strength: Multiplier for punishing wrong cut counts.
        image: Optional rendered word image used to snap cuts to low-ink valleys.
        snap_radius_bins: Maximum left/right bin search for geometric snapping.
        recall_threshold: Optional lower threshold for recall-oriented candidates.
        recall_candidate_limit: Maximum lower-threshold candidates to expose.
        sequence_features: Optional per-bin features for segment plausibility.
        segment_geometry_weight: Score penalty weight for implausible segments.

    Returns:
        Ranked JSON-safe path candidates. Each path stores selected cut bins
        plus the bridge/safety evidence used for scoring.
    """

    boundary = boundary_probabilities.detach().cpu().float()
    bridge = bridge_probabilities.detach().cpu().float()
    safety = cut_safety_probabilities.detach().cpu().float()
    bin_count = int(boundary.numel())
    peak_mask = decode_boundary_peaks(
        boundary.unsqueeze(0),
        threshold=threshold,
        min_distance=min_distance,
    )[0]
    peaks = _positive_bin_indices(peak_mask)
    recall_peaks: list[int] = []
    if recall_threshold is not None and float(recall_threshold) < float(threshold):
        recall_mask = decode_boundary_peaks(
            boundary.unsqueeze(0),
            threshold=float(recall_threshold),
            min_distance=max(1, min_distance),
        )[0]
        recall_peaks = _positive_bin_indices(recall_mask)

    def make_cut_candidate(bin_index: int, source: str) -> dict:
        """Create one cut candidate from peak decoding or the raw boundary map."""

        boundary_score = float(boundary[bin_index].item())
        bridge_risk = float(bridge[bin_index].item())
        safety_score = float(safety[bin_index].item())
        score = 0.55 * boundary_score + 0.35 * safety_score + 0.10 * (1.0 - bridge_risk)
        return {
            "bin": int(bin_index),
            "score": float(score),
            "boundary_probability": boundary_score,
            "bridge_risk": bridge_risk,
            "cut_safety": safety_score,
            "source": source,
        }

    candidates = []
    for bin_index in peaks:
        candidates.append(make_cut_candidate(int(bin_index), "decoded_peak"))

    candidates.sort(key=lambda item: (-item["score"], item["bin"]))
    primary_bins = {int(item["bin"]) for item in candidates}

    recall_candidates = []
    for bin_index in recall_peaks:
        if any(abs(int(bin_index) - primary_bin) < min_distance for primary_bin in primary_bins):
            continue
        recall_candidates.append(make_cut_candidate(int(bin_index), "recall_peak"))
    recall_candidates.sort(key=lambda item: (-item["score"], item["bin"]))
    recall_candidates = recall_candidates[: max(0, int(recall_candidate_limit))]

    expanded_candidates = sorted(
        [*candidates, *recall_candidates],
        key=lambda item: (-item["score"], item["bin"]),
    )

    def bin_is_available(bin_index: int, existing_cuts: list[dict]) -> bool:
        """Keep raw-fill cuts separated from already selected cuts."""

        return all(abs(int(item["bin"]) - int(bin_index)) >= min_distance for item in existing_cuts)

    def build_length_complete_raw_fill_path(target_count: int) -> list[dict]:
        """Fill missing cuts from raw boundary probabilities to satisfy length."""

        if target_count <= 0:
            return []
        filled = list(expanded_candidates[: min(len(expanded_candidates), target_count)])
        if len(filled) >= target_count:
            return sorted(filled, key=lambda item: item["bin"])

        raw_candidates = []
        used_bins = {int(item["bin"]) for item in filled}
        for bin_index in range(bin_count):
            if bin_index in used_bins or not bin_is_available(bin_index, filled):
                continue
            raw_candidates.append(make_cut_candidate(bin_index, "raw_boundary_fill"))
        raw_candidates.sort(key=lambda item: (-item["score"], item["bin"]))

        for raw_candidate in raw_candidates:
            if len(filled) >= target_count:
                break
            if not bin_is_available(int(raw_candidate["bin"]), filled):
                continue
            filled.append(raw_candidate)

        return sorted(filled, key=lambda item: item["bin"])

    def score_path(path_cuts: list[dict]) -> tuple[float, dict]:
        """Score one path using local evidence plus a length-aware prior."""

        evidence_score = float(
            sum(item["score"] for item in path_cuts) / max(1, len(path_cuts))
        )
        length_delta = None
        length_penalty = 0.0
        if expected_split_count is not None:
            length_delta = abs(len(path_cuts) - int(expected_split_count))
            normalized_delta = float(length_delta) / max(
                1.0,
                math.sqrt(float((expected_split_count or 0) + 1)),
            )
            length_penalty = (
                float(length_weight)
                * float(length_penalty_strength)
                * max(0.0, min(1.0, float(length_confidence)))
                * normalized_delta
                * (1.0 + 0.25 * float(length_delta))
            )
        return evidence_score - length_penalty, {
            "evidence_score": evidence_score,
            "length_delta": length_delta,
            "length_penalty": length_penalty,
        }

    candidate_sets: list[tuple[str, list[dict], str]] = []
    if expected_split_count == 0:
        candidate_sets.append(
            (
                "length_prior_zero_cut",
                [],
                "length-aware single-letter path with no cuts",
            )
        )
    elif not candidates:
        candidate_sets.append(
            (
                "no_decoded_peaks_zero_path",
                [],
                "fallback path when the boundary head decoded no cuts",
            )
        )

    selected = sorted(candidates, key=lambda item: item["bin"])
    if selected:
        candidate_sets.append(
            (
                "all_ranked_peaks",
                selected,
                "all decoded peaks sorted left-to-right",
            )
        )

    expanded_selected = sorted(expanded_candidates, key=lambda item: item["bin"])
    if recall_candidates and expanded_selected:
        candidate_sets.append(
            (
                "recall_expanded_peaks",
                expanded_selected,
                "primary peaks plus lower-threshold recall peaks",
            )
        )

    if expected_split_count is not None:
        target = max(0, int(expected_split_count))
        raw_fill_path = build_length_complete_raw_fill_path(target)
        if raw_fill_path:
            candidate_sets.append(
                (
                    "length_complete_raw_boundary_fill",
                    raw_fill_path,
                    "length-complete path filled from raw boundary probabilities",
                )
            )
        if recall_candidates:
            for delta in (0, -1, 1):
                keep_count = target + delta
                if keep_count <= 0 or keep_count > len(expanded_candidates):
                    continue
                kept = sorted(expanded_candidates[:keep_count], key=lambda item: item["bin"])
                candidate_sets.append(
                    (
                        f"recall_length_prior_keep_{keep_count}",
                        kept,
                        f"length-aware path using recall pool keeping {keep_count} strongest cuts",
                    )
                )
    if expected_split_count is not None and candidates:
        target = max(0, int(expected_split_count))
        for delta in (0, -1, 1, -2, 2):
            keep_count = target + delta
            if keep_count <= 0 or keep_count > len(candidates):
                continue
            kept = sorted(candidates[:keep_count], key=lambda item: item["bin"])
            candidate_sets.append(
                (
                    f"length_prior_keep_{keep_count}",
                    kept,
                    f"length-aware path keeping {keep_count} strongest cuts",
                )
            )

    for drop_count in range(1, min(top_k, len(candidates))):
        kept_by_score = candidates[:-drop_count]
        kept = sorted(kept_by_score, key=lambda item: item["bin"])
        if kept:
            candidate_sets.append(
                (
                    f"drop_{drop_count}_weakest",
                    kept,
                    f"alternative path after dropping {drop_count} weakest cuts",
                )
            )

    paths = []
    seen = set()
    for name, cuts, reason in candidate_sets:
        snapped_cuts = _snap_cut_records_to_valleys(
            cuts,
            image=image,
            bin_count=int(boundary.numel()),
            radius=snap_radius_bins,
        )
        cut_bins = tuple(item["bin"] for item in snapped_cuts)
        if cut_bins in seen:
            continue
        seen.add(cut_bins)
        path_score, score_parts = score_path(snapped_cuts)
        segment_geometry = _segment_geometry_for_path(
            list(cut_bins),
            sequence_features,
            bin_count=int(boundary.numel()),
            image=image,
        )
        geometry_penalty = float(segment_geometry.get("path_geometry_penalty", 0.0))
        path_score -= float(segment_geometry_weight) * geometry_penalty
        paths.append(
            {
                "path_id": f"p{len(paths)}_{name}",
                "cut_bins": list(cut_bins),
                "score": float(path_score),
                "candidate_count": len(snapped_cuts),
                "expected_split_count": expected_split_count,
                "length_confidence": float(length_confidence),
                "snap_radius_bins": int(max(0, snap_radius_bins)),
                "segment_geometry_weight": float(segment_geometry_weight),
                "segment_geometry": segment_geometry,
                **score_parts,
                "cuts": snapped_cuts,
                "reason": reason,
            }
        )
    paths.sort(key=lambda item: (-item["score"], item["candidate_count"], item["cut_bins"]))
    return paths[:top_k]


def _ink_density_by_bin(image: np.ndarray, bin_count: int) -> list[float]:
    """Measure how much ink occupies each horizontal boundary bin.

    The synthetic renderer uses black ink on a light background, but this
    helper also tolerates inverted previews by choosing the darker polarity as
    ink. These densities are only used for snapping proposed cut positions onto
    nearby valleys, not for creating new cut candidates.
    """

    if image is None or bin_count <= 0:
        return [0.0] * max(0, bin_count)
    gray = np.asarray(image)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.uint8)
    dark_pixels = gray < 128
    light_pixels = gray > 127
    dark_count = int(dark_pixels.sum())
    light_count = int(light_pixels.sum())
    ink_mask = dark_pixels if dark_count <= light_count else light_pixels
    height, width = ink_mask.shape
    densities = []
    for bin_index in range(bin_count):
        x1 = int(round((bin_index / max(1, bin_count)) * width))
        x2 = int(round(((bin_index + 1) / max(1, bin_count)) * width))
        x2 = max(x1 + 1, min(width, x2))
        region = ink_mask[:, x1:x2]
        densities.append(float(region.sum()) / max(1.0, float(height * (x2 - x1))))
    return densities


def _snap_cut_records_to_valleys(
    cuts: list[dict],
    image: np.ndarray | None,
    bin_count: int,
    radius: int,
) -> list[dict]:
    """Snap proposed cuts to the least-inky nearby bin and keep audit metadata."""

    if not cuts:
        return []
    safe_radius = max(0, int(radius))
    densities = _ink_density_by_bin(image, bin_count) if safe_radius > 0 else []
    snapped_by_bin: dict[int, dict] = {}
    for cut in cuts:
        original_bin = int(cut["bin"])
        snapped_bin = original_bin
        snap_density = None
        if densities:
            left = max(0, original_bin - safe_radius)
            right = min(bin_count - 1, original_bin + safe_radius)
            snapped_bin = min(
                range(left, right + 1),
                key=lambda index: (densities[index], abs(index - original_bin), index),
            )
            snap_density = float(densities[snapped_bin])

        snapped = dict(cut)
        snapped["original_bin"] = original_bin
        snapped["bin"] = int(snapped_bin)
        snapped["snapped_bin"] = int(snapped_bin)
        snapped["snap_delta"] = int(snapped_bin - original_bin)
        if snap_density is not None:
            snapped["snap_ink_density"] = snap_density

        existing = snapped_by_bin.get(snapped_bin)
        if existing is None or float(snapped["score"]) > float(existing["score"]):
            snapped_by_bin[snapped_bin] = snapped

    return [snapped_by_bin[index] for index in sorted(snapped_by_bin)]


def _path_bins_exact_match(predicted_bins: list[int], true_bins: list[int]) -> bool:
    """Return True when the candidate path exactly matches true cut bins."""

    return list(predicted_bins) == list(true_bins)


def _path_bins_within_tolerance(
    predicted_bins: list[int],
    true_bins: list[int],
    tolerance: int = 1,
) -> bool:
    """Return True when every cut matches one true cut within tolerance.

    Args:
        predicted_bins: Candidate path cut bins.
        true_bins: Ground-truth cut bins.
        tolerance: Allowed absolute bin distance.

    Returns:
        True for one-to-one matching within tolerance. Empty predicted and true
        lists are valid matches, which covers single-letter zero-cut cases.
    """

    if len(predicted_bins) != len(true_bins):
        return False
    if not predicted_bins and not true_bins:
        return True
    used_true: set[int] = set()
    for predicted in predicted_bins:
        available_true = [
            true
            for true in true_bins
            if true not in used_true
        ]
        if not available_true:
            return False
        best_true = min(available_true, key=lambda value: (abs(value - predicted), value))
        if abs(best_true - predicted) > tolerance:
            return False
        used_true.add(best_true)
    return True


def _path_topk_metric_hits(paths: list[dict], true_bins: list[int]) -> dict:
    """Compute path top-k exact, within-1, and count-only hits."""

    metrics = {}
    for k in (1, 3, 5):
        selected_paths = paths[:k]
        metrics[f"path_top{k}_exact_hit"] = any(
            _path_bins_exact_match(path.get("cut_bins", []), true_bins)
            for path in selected_paths
        )
        metrics[f"path_top{k}_within_1_bin_hit"] = any(
            _path_bins_within_tolerance(path.get("cut_bins", []), true_bins, tolerance=1)
            for path in selected_paths
        )
        metrics[f"path_top{k}_count_exact_hit"] = any(
            len(path.get("cut_bins", [])) == len(true_bins)
            for path in selected_paths
        )
    metrics["path_zero_cut_truth"] = len(true_bins) == 0
    metrics["path_zero_cut_predicted_top1"] = (
        not paths or len(paths[0].get("cut_bins", [])) == 0
    )
    metrics["path_zero_cut_top1_exact_hit"] = (
        metrics["path_zero_cut_truth"] and metrics["path_zero_cut_predicted_top1"]
    )
    return metrics


def _path_candidate_cut_coverage(
    paths: list[dict],
    true_bins: list[int],
    tolerance: int = 2,
) -> dict:
    """Measure whether true cuts exist anywhere in the proposed path universe.

    This separates a path-ranking failure from a proposal-recall failure. If a
    true cut has no candidate within tolerance in any top-k path, reranking the
    paths cannot recover the correct split.
    """

    candidate_bins = sorted(
        {
            int(bin_index)
            for path in paths
            for bin_index in path.get("cut_bins", [])
        }
    )
    if not true_bins:
        return {
            "candidate_cut_count": len(candidate_bins),
            "true_cut_count": 0,
            "covered_true_cut_count": 0,
            "missing_true_cut_count": 0,
            "coverage_ratio": 1.0,
            "all_true_cuts_covered": True,
            "missing_true_bins": [],
        }

    missing = []
    covered = 0
    for true_bin in true_bins:
        if any(abs(candidate_bin - true_bin) <= tolerance for candidate_bin in candidate_bins):
            covered += 1
        else:
            missing.append(int(true_bin))

    return {
        "candidate_cut_count": len(candidate_bins),
        "true_cut_count": len(true_bins),
        "covered_true_cut_count": covered,
        "missing_true_cut_count": len(missing),
        "coverage_ratio": covered / max(1, len(true_bins)),
        "all_true_cuts_covered": len(missing) == 0,
        "missing_true_bins": missing,
    }


def scribetrain_utility_score(metrics: dict, weights: dict | None = None) -> float:
    """Score a checkpoint by the way N05 actually uses ScribeTrain.

    Peak quality matters, but a useful splitter also needs plausible split
    counts, reasonable length predictions, and at least one good path in the
    top-k set. This utility score keeps checkpoint selection aligned with that
    multi-signal job instead of overfitting to a single metric.
    """

    active_weights = {
        "peak_f1": 0.55,
        "peak_count_pm1": 0.25,
        "length_accuracy": 0.15,
        "path_top5_count_exact": 0.05,
    }
    if weights:
        active_weights.update({str(key): float(value) for key, value in weights.items()})
    return float(
        active_weights["peak_f1"] * float(metrics.get("peak_f1", 0.0))
        + active_weights["peak_count_pm1"]
        * float(metrics.get("peak_split_count_within_1_accuracy", 0.0))
        + active_weights["length_accuracy"] * float(metrics.get("length_accuracy", 0.0))
        + active_weights["path_top5_count_exact"]
        * float(metrics.get("path_top5_count_exact", 0.0))
    )


def _evaluation_counts_for_train_size(
    train_count: int,
    dataset_settings: dict,
    limited_run: bool,
) -> tuple[int, int, str]:
    """Choose validation/test sizes that match the run's seriousness.

    Small `--limit` runs are still quick smoke tests. Larger limited runs,
    especially 5k/20k experiments, need enough validation and test examples to
    make the splitter metrics meaningful instead of 64-sample dice rolls.
    """

    if not limited_run:
        validation_count = int(
            dataset_settings.get("validation_samples", max(200, train_count // 8))
        )
        test_count = int(dataset_settings.get("test_samples", max(200, train_count // 8)))
        return validation_count, test_count, "settings/default evaluation split"

    if train_count < 1000:
        count = max(4, min(64, train_count // 2))
        return count, count, "small smoke evaluation split"

    # Use roughly 10% per evaluation split, capped so 20k runs get 2k/2k
    # without accidentally turning quick experiments into evaluation marathons.
    count = max(256, min(2000, train_count // 10))
    return count, count, "scaled limited-run evaluation split"


def _positive_bin_indices(values: torch.Tensor) -> list[int]:
    """Return sorted indices where a binary boundary vector is active."""

    return [
        int(index)
        for index, value in enumerate(values.detach().cpu().tolist())
        if int(value) == 1
    ]


def _boundary_bin_to_x(bin_index: int, bin_count: int, width: int) -> int:
    """Map a boundary bin index back to an approximate image x-coordinate."""

    if bin_count <= 1:
        return width // 2
    ratio = (float(bin_index) + 0.5) / float(bin_count)
    return max(0, min(width - 1, int(round(ratio * width))))


def _draw_boundary_debug_image(
    image: np.ndarray,
    true_bins: torch.Tensor,
    predicted_bins: torch.Tensor,
    probability_bins: torch.Tensor,
    text: str,
    output_path: Path,
) -> None:
    """Save one predicted-vs-true boundary debug preview.

    Args:
        image: Original rendered word image.
        true_bins: Binary target boundary vector.
        predicted_bins: Binary predicted boundary vector.
        probability_bins: Sigmoid probabilities per bin.
        text: Source word label.
        output_path: Destination PNG path.
    """

    gray = np.asarray(image)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale = 4
    image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    preview = cv2.resize(
        image_bgr,
        (width * scale, height * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    true_indices = _positive_bin_indices(true_bins)
    predicted_indices = _positive_bin_indices(predicted_bins)
    bin_count = int(true_bins.numel())

    for index in true_indices:
        x = _boundary_bin_to_x(index, bin_count, width) * scale
        cv2.line(preview, (x, 0), (x, preview.shape[0] - 1), (0, 180, 0), 1)
    for index in predicted_indices:
        x = _boundary_bin_to_x(index, bin_count, width) * scale
        cv2.line(preview, (x, 0), (x, preview.shape[0] - 1), (0, 0, 220), 1)

    strip_height = 14
    margin = 8
    canvas_height = preview.shape[0] + strip_height * 3 + margin * 4 + 24
    canvas_width = max(preview.shape[1], bin_count * 6)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:, :] = (18, 22, 29)
    canvas[margin : margin + preview.shape[0], 0 : preview.shape[1]] = preview
    cv2.putText(
        canvas,
        f"{text} | green=true red=pred",
        (6, canvas_height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (210, 220, 235),
        1,
        cv2.LINE_AA,
    )

    y_true = margin + preview.shape[0] + margin
    y_pred = y_true + strip_height + margin
    y_prob = y_pred + strip_height + margin
    bin_width = max(1, canvas_width // max(1, bin_count))
    probabilities = probability_bins.detach().cpu().numpy()
    true_values = true_bins.detach().cpu().numpy()
    predicted_values = predicted_bins.detach().cpu().numpy()
    for index in range(bin_count):
        x1 = index * bin_width
        x2 = min(canvas_width - 1, (index + 1) * bin_width - 1)
        if true_values[index] > 0:
            cv2.rectangle(canvas, (x1, y_true), (x2, y_true + strip_height), (0, 180, 0), -1)
        if predicted_values[index] > 0:
            cv2.rectangle(canvas, (x1, y_pred), (x2, y_pred + strip_height), (0, 0, 220), -1)
        intensity = int(max(0.0, min(1.0, float(probabilities[index]))) * 255)
        cv2.rectangle(canvas, (x1, y_prob), (x2, y_prob + strip_height), (intensity, intensity, 0), -1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    debug_dir: Path | None = None,
    split_name: str = "split",
    max_debug_images: int = 0,
    peak_threshold: float = 0.35,
    peak_min_distance: int = 2,
    max_path_examples: int = 0,
    path_top_k: int = 5,
    length_path_weight: float = 0.25,
    length_penalty_strength: float = 2.0,
    snap_radius_bins: int = 0,
    recall_threshold: float | None = None,
    recall_candidate_limit: int = 32,
    segment_geometry_weight: float = 0.15,
) -> dict:
    """Evaluate the model on one split.

    Args:
        model: Sequence model.
        loader: DataLoader for a split.
        device: Torch device.
        debug_dir: Optional folder for predicted-vs-true boundary previews.
        split_name: Label used in debug filenames.
        max_debug_images: Maximum previews to save for this evaluation.
        peak_threshold: Minimum probability used by peak decoding.
        peak_min_distance: Non-maximum suppression distance in bins.
        max_path_examples: Maximum path examples to include in metrics.
        path_top_k: Number of alternative paths per example.
        length_path_weight: How strongly predicted length influences paths.
        length_penalty_strength: Multiplier for wrong-cut-count penalties.
        snap_radius_bins: Local bin radius for snapping cuts to ink valleys.
        recall_threshold: Optional lower threshold for recall-oriented candidates.
        recall_candidate_limit: Maximum lower-threshold candidates to expose.
        segment_geometry_weight: Score penalty weight for implausible segments.

    Returns:
        JSON-safe metrics.
    """

    model.eval()
    boundary_logits_all = []
    hard_boundary_targets_all = []
    length_correct = 0
    length_total = 0
    loss_total = 0.0
    batch_total = 0
    debug_written = 0
    path_examples = []
    path_metric_counts = {
        "path_top1_exact": 0,
        "path_top3_exact": 0,
        "path_top5_exact": 0,
        "path_top1_within_1_bin": 0,
        "path_top3_within_1_bin": 0,
        "path_top5_within_1_bin": 0,
        "path_top1_count_exact": 0,
        "path_top3_count_exact": 0,
        "path_top5_count_exact": 0,
        "path_zero_cut_truth_count": 0,
        "path_zero_cut_top1_exact": 0,
    }
    coverage_counts = {
        "coverage_within_1_sum": 0.0,
        "coverage_within_2_sum": 0.0,
        "all_covered_within_1": 0,
        "all_covered_within_2": 0,
        "missing_within_1_sum": 0,
        "missing_within_2_sum": 0,
        "true_cut_total": 0,
        "covered_true_cut_within_1_total": 0,
        "covered_true_cut_within_2_total": 0,
        "candidate_cut_count_sum": 0,
    }
    path_metric_total = 0

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            boundaries = batch["boundaries"].to(device)
            hard_boundaries = batch["hard_boundaries"].to(device)
            bridge_targets = batch["bridge_targets"].to(device)
            cut_safety_targets = batch["cut_safety_targets"].to(device)
            lengths = batch["lengths"].to(device)
            outputs = model(features)
            boundary_loss = F.binary_cross_entropy_with_logits(
                outputs["boundary_logits"],
                boundaries,
            )
            bridge_loss = F.binary_cross_entropy_with_logits(
                outputs["bridge_logits"],
                bridge_targets,
            )
            cut_safety_loss = F.binary_cross_entropy_with_logits(
                outputs["cut_safety_logits"],
                cut_safety_targets,
            )
            length_loss = F.cross_entropy(outputs["length_logits"], lengths)
            loss_total += float(
                (boundary_loss + bridge_loss + cut_safety_loss + length_loss).item()
            )
            batch_total += 1
            length_predictions = outputs["length_logits"].argmax(dim=1)
            length_correct += int((length_predictions == lengths).sum().item())
            length_total += int(lengths.numel())
            boundary_logits_all.append(outputs["boundary_logits"].cpu())
            hard_boundary_targets_all.append(hard_boundaries.cpu())
            if debug_dir is not None and debug_written < max_debug_images:
                probabilities = torch.sigmoid(outputs["boundary_logits"]).cpu()
                predictions = decode_boundary_peaks(
                    probabilities,
                    threshold=peak_threshold,
                    min_distance=peak_min_distance,
                )
                truth = hard_boundaries.cpu().to(torch.int64)
                for item_index, image in enumerate(batch["images"]):
                    if debug_written >= max_debug_images:
                        break
                    output_path = debug_dir / (
                        f"{split_name}_{debug_written:03d}_"
                        f"{batch['texts'][item_index]}.png"
                    )
                    _draw_boundary_debug_image(
                        image=image,
                        true_bins=truth[item_index],
                        predicted_bins=predictions[item_index],
                        probability_bins=probabilities[item_index],
                        text=batch["texts"][item_index],
                        output_path=output_path,
                    )
                    debug_written += 1
            boundary_probabilities = torch.sigmoid(outputs["boundary_logits"]).cpu()
            bridge_probabilities = torch.sigmoid(outputs["bridge_logits"]).cpu()
            cut_safety_probabilities = torch.sigmoid(outputs["cut_safety_logits"]).cpu()
            length_probabilities = F.softmax(outputs["length_logits"], dim=1).cpu()
            truth = hard_boundaries.cpu().to(torch.int64)
            true_lengths = lengths.detach().cpu()
            for item_index, text in enumerate(batch["texts"]):
                length_row = length_probabilities[item_index]
                predicted_length = int(torch.argmax(length_row).item())
                expected_split_count = max(0, predicted_length - 1)
                length_confidence = float(length_row[predicted_length].item())
                paths = build_segmentation_paths(
                    boundary_probabilities[item_index],
                    bridge_probabilities[item_index],
                    cut_safety_probabilities[item_index],
                    threshold=peak_threshold,
                    min_distance=peak_min_distance,
                    top_k=path_top_k,
                    expected_split_count=expected_split_count,
                    length_confidence=length_confidence,
                    length_weight=length_path_weight,
                    length_penalty_strength=length_penalty_strength,
                    image=batch["images"][item_index],
                    snap_radius_bins=snap_radius_bins,
                    recall_threshold=recall_threshold,
                    recall_candidate_limit=recall_candidate_limit,
                    sequence_features=features[item_index].detach().cpu(),
                    segment_geometry_weight=segment_geometry_weight,
                )
                true_cut_bins = _positive_bin_indices(truth[item_index])
                path_hits = _path_topk_metric_hits(paths, true_cut_bins)
                coverage_within_1 = _path_candidate_cut_coverage(
                    paths,
                    true_cut_bins,
                    tolerance=1,
                )
                coverage_within_2 = _path_candidate_cut_coverage(
                    paths,
                    true_cut_bins,
                    tolerance=max(2, snap_radius_bins),
                )
                path_metric_total += 1
                coverage_counts["coverage_within_1_sum"] += coverage_within_1[
                    "coverage_ratio"
                ]
                coverage_counts["coverage_within_2_sum"] += coverage_within_2[
                    "coverage_ratio"
                ]
                coverage_counts["all_covered_within_1"] += int(
                    coverage_within_1["all_true_cuts_covered"]
                )
                coverage_counts["all_covered_within_2"] += int(
                    coverage_within_2["all_true_cuts_covered"]
                )
                coverage_counts["missing_within_1_sum"] += int(
                    coverage_within_1["missing_true_cut_count"]
                )
                coverage_counts["missing_within_2_sum"] += int(
                    coverage_within_2["missing_true_cut_count"]
                )
                coverage_counts["true_cut_total"] += int(coverage_within_2["true_cut_count"])
                coverage_counts["covered_true_cut_within_1_total"] += int(
                    coverage_within_1["covered_true_cut_count"]
                )
                coverage_counts["covered_true_cut_within_2_total"] += int(
                    coverage_within_2["covered_true_cut_count"]
                )
                coverage_counts["candidate_cut_count_sum"] += int(
                    coverage_within_2["candidate_cut_count"]
                )
                for k in (1, 3, 5):
                    path_metric_counts[f"path_top{k}_exact"] += int(
                        path_hits[f"path_top{k}_exact_hit"]
                    )
                    path_metric_counts[f"path_top{k}_within_1_bin"] += int(
                        path_hits[f"path_top{k}_within_1_bin_hit"]
                    )
                    path_metric_counts[f"path_top{k}_count_exact"] += int(
                        path_hits[f"path_top{k}_count_exact_hit"]
                    )
                path_metric_counts["path_zero_cut_truth_count"] += int(
                    path_hits["path_zero_cut_truth"]
                )
                path_metric_counts["path_zero_cut_top1_exact"] += int(
                    path_hits["path_zero_cut_top1_exact_hit"]
                )
                if len(path_examples) >= max_path_examples:
                    continue
                path_examples.append(
                    {
                        "split": split_name,
                        "text": text,
                        "true_cut_bins": true_cut_bins,
                        "true_length": int(true_lengths[item_index].item()),
                        "predicted_length": predicted_length,
                        "expected_split_count": expected_split_count,
                        "length_confidence": length_confidence,
                        "length_prior_source": "ScribeTrain length head / Scrististics-style path prior",
                        "path_metric_hits": path_hits,
                        "candidate_cut_coverage_within_1": coverage_within_1,
                        "candidate_cut_coverage_within_2": coverage_within_2,
                        "predicted_paths": paths,
                    }
                )

    boundary_metrics = _boundary_metrics(
        torch.cat(boundary_logits_all, dim=0),
        torch.cat(hard_boundary_targets_all, dim=0),
    )
    peak_metrics = _peak_distance_metrics(
        torch.cat(boundary_logits_all, dim=0),
        torch.cat(hard_boundary_targets_all, dim=0),
        threshold=peak_threshold,
        min_distance=peak_min_distance,
    )
    return {
        "loss": loss_total / max(1, batch_total),
        "length_accuracy": length_correct / max(1, length_total),
        "path_examples": path_examples,
        "path_metric_sample_count": path_metric_total,
        "path_zero_cut_truth_count": path_metric_counts["path_zero_cut_truth_count"],
        "path_zero_cut_top1_exact": (
            path_metric_counts["path_zero_cut_top1_exact"]
            / max(1, path_metric_counts["path_zero_cut_truth_count"])
        ),
        "candidate_true_cut_coverage_within_1": (
            coverage_counts["coverage_within_1_sum"] / max(1, path_metric_total)
        ),
        "candidate_true_cut_coverage_within_2": (
            coverage_counts["coverage_within_2_sum"] / max(1, path_metric_total)
        ),
        "candidate_all_true_cuts_covered_within_1": (
            coverage_counts["all_covered_within_1"] / max(1, path_metric_total)
        ),
        "candidate_all_true_cuts_covered_within_2": (
            coverage_counts["all_covered_within_2"] / max(1, path_metric_total)
        ),
        "candidate_missing_true_cut_mean_within_1": (
            coverage_counts["missing_within_1_sum"] / max(1, path_metric_total)
        ),
        "candidate_missing_true_cut_mean_within_2": (
            coverage_counts["missing_within_2_sum"] / max(1, path_metric_total)
        ),
        "candidate_true_cut_micro_coverage_within_1": (
            coverage_counts["covered_true_cut_within_1_total"]
            / max(1, coverage_counts["true_cut_total"])
        ),
        "candidate_true_cut_micro_coverage_within_2": (
            coverage_counts["covered_true_cut_within_2_total"]
            / max(1, coverage_counts["true_cut_total"])
        ),
        "candidate_cut_count_mean": (
            coverage_counts["candidate_cut_count_sum"] / max(1, path_metric_total)
        ),
        **{
            key: value / max(1, path_metric_total)
            for key, value in path_metric_counts.items()
            if key.startswith("path_top")
        },
        **boundary_metrics,
        **peak_metrics,
    }


def train(settings: dict, limit: int | None = None) -> dict:
    """Train the ScribeTrace word sequence splitter.

    Args:
        settings: Root settings dictionary.
        limit: Optional train-sample override for smoke tests.

    Returns:
        Final training report.
    """

    started = time.time()
    dataset_settings = settings["dataset"]
    training_settings = settings["training"]
    output_settings = settings["output"]
    train_count = int(limit or dataset_settings.get("samples", 8000))
    validation_count, test_count, evaluation_split_policy = _evaluation_counts_for_train_size(
        train_count,
        dataset_settings,
        limited_run=bool(limit),
    )
    batch_size = int(training_settings.get("batch_size", 96))
    workers = int(training_settings.get("num_workers", 0))
    device = _choose_device(settings)
    label_map = load_label_map(dataset_settings["label_map_path"])
    char_to_token, token_to_char = build_token_maps(label_map)
    all_words = load_word_samples(settings, char_to_token)
    train_words, validation_words, test_words = split_word_pools(
        all_words,
        seed=int(settings.get("random_seed", 42)),
    )
    tail_profiles = build_tail_profiles(token_to_char)
    glyph_paths = collect_glyph_paths(dataset_settings["matenadata_dir"], label_map)

    train_set = ScribeTrainSyntheticWordDataset(
        settings,
        "train",
        train_count,
        seed_offset=0,
        words=train_words,
        glyph_paths=glyph_paths,
        tail_profiles=tail_profiles,
    )
    validation_set = ScribeTrainSyntheticWordDataset(
        settings,
        "validation",
        validation_count,
        seed_offset=1_000_000,
        words=validation_words,
        glyph_paths=glyph_paths,
        tail_profiles=tail_profiles,
    )
    test_set = ScribeTrainSyntheticWordDataset(
        settings,
        "test",
        test_count,
        seed_offset=2_000_000,
        words=test_words,
        glyph_paths=glyph_paths,
        tail_profiles=tail_profiles,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_sequence_samples,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_sequence_samples,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_sequence_samples,
    )

    model = ScribeTrainWordTraceModel(
        feature_count=len(FEATURE_NAMES),
        hidden_size=int(training_settings.get("hidden_size", 96)),
        num_layers=int(training_settings.get("num_layers", 2)),
        dropout=float(training_settings.get("dropout", 0.15)),
        max_length_class=int(dataset_settings.get("max_sequence_length", 18)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_settings.get("learning_rate", 1e-3)),
        weight_decay=float(training_settings.get("weight_decay", 1e-4)),
    )
    positive_weight = torch.tensor(
        float(training_settings.get("boundary_positive_weight", 2.0)),
        dtype=torch.float32,
        device=device,
    )
    length_loss_weight = float(training_settings.get("length_loss_weight", 0.35))
    bridge_loss_weight = float(training_settings.get("bridge_loss_weight", 0.20))
    cut_safety_loss_weight = float(training_settings.get("cut_safety_loss_weight", 0.20))
    peak_threshold = float(training_settings.get("peak_threshold", 0.35))
    peak_min_distance = int(training_settings.get("peak_min_distance_bins", 2))
    length_path_weight = float(training_settings.get("length_path_weight", 0.25))
    length_penalty_strength = float(training_settings.get("length_penalty_strength", 2.0))
    snap_radius_bins = int(training_settings.get("snap_radius_bins", 2))
    recall_threshold = training_settings.get("recall_threshold", 0.12)
    recall_threshold = None if recall_threshold is None else float(recall_threshold)
    recall_candidate_limit = int(training_settings.get("recall_candidate_limit", 32))
    segment_geometry_weight = float(training_settings.get("segment_geometry_weight", 0.15))
    utility_weights = dict(
        training_settings.get(
            "utility_weights",
            {
                "peak_f1": 0.55,
                "peak_count_pm1": 0.25,
                "length_accuracy": 0.15,
                "path_top5_count_exact": 0.05,
            },
        )
    )
    epochs = int(training_settings.get("epochs", 12))
    best_metric = -math.inf
    best_peak_f1 = 0.0
    best_epoch = 0
    history = []
    model_dir = resolve_path(output_settings["model_dir"])
    report_dir = resolve_path(output_settings["report_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{settings.get('model_name', 'scribetrace_word_sequence_v0_1')}.pt"
    debug_root = report_dir / "boundary_debug"
    max_debug_images = int(output_settings.get("max_boundary_debug_images", 12))
    max_path_examples = int(output_settings.get("max_path_examples", 12))
    path_top_k = int(output_settings.get("path_top_k", 5))
    print(
        "split:",
        f"train={train_count}",
        f"validation={validation_count}",
        f"test={test_count}",
        f"policy={evaluation_split_policy}",
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batch_total = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            boundaries = batch["boundaries"].to(device)
            bridge_targets = batch["bridge_targets"].to(device)
            cut_safety_targets = batch["cut_safety_targets"].to(device)
            lengths = batch["lengths"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features)
            boundary_loss = F.binary_cross_entropy_with_logits(
                outputs["boundary_logits"],
                boundaries,
                pos_weight=positive_weight,
            )
            bridge_loss = F.binary_cross_entropy_with_logits(
                outputs["bridge_logits"],
                bridge_targets,
            )
            cut_safety_loss = F.binary_cross_entropy_with_logits(
                outputs["cut_safety_logits"],
                cut_safety_targets,
            )
            length_loss = F.cross_entropy(outputs["length_logits"], lengths)
            loss = (
                boundary_loss
                + bridge_loss_weight * bridge_loss
                + cut_safety_loss_weight * cut_safety_loss
                + length_loss_weight * length_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.item())
            batch_total += 1

        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            debug_dir=debug_root / f"epoch_{epoch:02d}_validation",
            split_name=f"epoch_{epoch:02d}_validation",
            max_debug_images=max_debug_images,
            peak_threshold=peak_threshold,
            peak_min_distance=peak_min_distance,
            max_path_examples=min(3, max_path_examples),
            path_top_k=path_top_k,
            length_path_weight=length_path_weight,
            length_penalty_strength=length_penalty_strength,
            snap_radius_bins=snap_radius_bins,
            recall_threshold=recall_threshold,
            recall_candidate_limit=recall_candidate_limit,
            segment_geometry_weight=segment_geometry_weight,
        )
        validation_utility = scribetrain_utility_score(validation_metrics, utility_weights)
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, batch_total),
            "validation_utility_score": validation_utility,
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch:02d}: "
            f"loss={epoch_record['train_loss']:.4f} "
            f"val_boundary_f1={validation_metrics['boundary_f1']:.4f} "
            f"val_boundary_recall={validation_metrics['boundary_recall']:.4f} "
            f"val_peak_f1={validation_metrics['peak_f1']:.4f} "
            f"val_peak_count_pm1={validation_metrics['peak_split_count_within_1_accuracy']:.4f} "
            f"val_len_acc={validation_metrics['length_accuracy']:.4f} "
            f"val_utility={validation_utility:.4f}"
        )
        if validation_utility > best_metric:
            best_metric = validation_utility
            best_peak_f1 = float(validation_metrics["peak_f1"])
            best_epoch = epoch
            torch.save(
                {
                    "model_name": settings.get("model_name", "scribetrace_word_sequence_v0_1"),
                    "state_dict": model.state_dict(),
                    "feature_names": list(FEATURE_NAMES),
                    "settings": settings,
                    "best_epoch": best_epoch,
                    "best_validation_utility_score": best_metric,
                    "best_validation_peak_f1": best_peak_f1,
                    "utility_weights": utility_weights,
                },
                model_path,
            )
            print(f"  saved best model: {model_path} utility={best_metric:.4f}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        debug_dir=debug_root / "final_test",
        split_name="final_test",
        max_debug_images=max_debug_images,
        peak_threshold=peak_threshold,
        peak_min_distance=peak_min_distance,
        max_path_examples=max_path_examples,
        path_top_k=path_top_k,
        length_path_weight=length_path_weight,
        length_penalty_strength=length_penalty_strength,
        snap_radius_bins=snap_radius_bins,
        recall_threshold=recall_threshold,
        recall_candidate_limit=recall_candidate_limit,
        segment_geometry_weight=segment_geometry_weight,
    )
    test_utility = scribetrain_utility_score(test_metrics, utility_weights)
    report = {
        "model_name": settings.get("model_name", "scribetrace_word_sequence_v0_1"),
        "status": "completed",
        "model_path": str(model_path),
        "feature_names": list(FEATURE_NAMES),
        "split": {
            "train": train_count,
            "validation": validation_count,
            "test": test_count,
            "evaluation_policy": evaluation_split_policy,
        },
        "word_pool_split": {
            "source_word_count": len(all_words),
            "train_unique_words": len(train_words),
            "validation_unique_words": len(validation_words),
            "test_unique_words": len(test_words),
            "policy": "unique word text split before synthetic sampling",
        },
        "boundary_debug_dir": str(debug_root),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_utility_score": best_metric,
        "best_validation_peak_f1": best_peak_f1,
        "test_utility_score": test_utility,
        "utility_weights": utility_weights,
        "loss_weights": {
            "boundary": 1.0,
            "bridge": bridge_loss_weight,
            "cut_safety": cut_safety_loss_weight,
            "length": length_loss_weight,
            "length_path_prior": length_path_weight,
            "length_penalty_strength": length_penalty_strength,
            "snap_radius_bins": snap_radius_bins,
            "recall_threshold": recall_threshold,
            "recall_candidate_limit": recall_candidate_limit,
            "segment_geometry_weight": segment_geometry_weight,
        },
        "top_k_path_examples": test_metrics.get("path_examples", []),
        "test": test_metrics,
        "history": history,
        "elapsed_seconds": round(time.time() - started, 3),
        "notes": [
            "ScribeTrain monster v0.1 predicts boundaries, bridge risk, cut safety, and length.",
            "Path examples are built from boundary peaks penalized by bridge risk and boosted by cut safety.",
            "The model is intentionally dependency-light: no CRF package yet.",
        ],
    }
    report_path = save_json(report, report_dir / "training_report.json")
    print(f"model:  {model_path}")
    print(f"report: {report_path}")
    print(
        "test:",
        f"boundary_f1={test_metrics['boundary_f1']:.4f}",
        f"boundary_recall={test_metrics['boundary_recall']:.4f}",
        f"peak_f1={test_metrics['peak_f1']:.4f}",
        f"peak_count_pm1={test_metrics['peak_split_count_within_1_accuracy']:.4f}",
        f"length={test_metrics['length_accuracy']:.4f}",
        f"utility={test_utility:.4f}",
    )
    return report


def write_checkpoint_report(
    settings: dict,
    checkpoint_path: Path | None = None,
    limit: int | None = None,
) -> dict:
    """Evaluate a saved ScribeTrain checkpoint and write a report.

    This is the recovery path for interrupted training runs. Checkpoints are
    saved during training, but the full JSON report is only written after final
    test evaluation. This function lets a surviving ``.pt`` checkpoint produce
    validation/test metrics later without retraining.
    """

    started = time.time()
    output_settings = settings["output"]
    report_dir = resolve_path(output_settings["report_dir"])
    model_dir = resolve_path(output_settings["model_dir"])
    model_name = settings.get("model_name", "scribetrace_word_sequence_v0_1")
    checkpoint_path = checkpoint_path or (model_dir / f"{model_name}.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_settings = checkpoint.get("settings")
    if isinstance(checkpoint_settings, dict):
        settings = checkpoint_settings
        output_settings = settings["output"]
        report_dir = resolve_path(output_settings["report_dir"])
        model_dir = resolve_path(output_settings["model_dir"])
        model_name = settings.get("model_name", model_name)

    dataset_settings = settings["dataset"]
    training_settings = settings["training"]
    train_count = int(limit or dataset_settings.get("samples", 8000))
    validation_count, test_count, evaluation_split_policy = _evaluation_counts_for_train_size(
        train_count,
        dataset_settings,
        limited_run=bool(limit),
    )
    batch_size = int(training_settings.get("batch_size", 96))
    workers = int(training_settings.get("num_workers", 0))
    device = _choose_device(settings)
    label_map = load_label_map(dataset_settings["label_map_path"])
    char_to_token, token_to_char = build_token_maps(label_map)
    all_words = load_word_samples(settings, char_to_token)
    train_words, validation_words, test_words = split_word_pools(
        all_words,
        seed=int(settings.get("random_seed", 42)),
    )
    tail_profiles = build_tail_profiles(token_to_char)
    glyph_paths = collect_glyph_paths(dataset_settings["matenadata_dir"], label_map)

    validation_set = ScribeTrainSyntheticWordDataset(
        settings,
        "validation",
        validation_count,
        seed_offset=1_000_000,
        words=validation_words,
        glyph_paths=glyph_paths,
        tail_profiles=tail_profiles,
    )
    test_set = ScribeTrainSyntheticWordDataset(
        settings,
        "test",
        test_count,
        seed_offset=2_000_000,
        words=test_words,
        glyph_paths=glyph_paths,
        tail_profiles=tail_profiles,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_sequence_samples,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_sequence_samples,
    )

    model = ScribeTrainWordTraceModel(
        feature_count=len(FEATURE_NAMES),
        hidden_size=int(training_settings.get("hidden_size", 96)),
        num_layers=int(training_settings.get("num_layers", 2)),
        dropout=float(training_settings.get("dropout", 0.15)),
        max_length_class=int(dataset_settings.get("max_sequence_length", 18)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    peak_threshold = float(training_settings.get("peak_threshold", 0.35))
    peak_min_distance = int(training_settings.get("peak_min_distance_bins", 2))
    length_path_weight = float(training_settings.get("length_path_weight", 0.25))
    length_penalty_strength = float(training_settings.get("length_penalty_strength", 2.0))
    snap_radius_bins = int(training_settings.get("snap_radius_bins", 2))
    recall_threshold = training_settings.get("recall_threshold", 0.12)
    recall_threshold = None if recall_threshold is None else float(recall_threshold)
    recall_candidate_limit = int(training_settings.get("recall_candidate_limit", 32))
    segment_geometry_weight = float(training_settings.get("segment_geometry_weight", 0.15))
    utility_weights = dict(
        checkpoint.get(
            "utility_weights",
            training_settings.get(
                "utility_weights",
                {
                    "peak_f1": 0.55,
                    "peak_count_pm1": 0.25,
                    "length_accuracy": 0.15,
                    "path_top5_count_exact": 0.05,
                },
            ),
        )
    )
    max_debug_images = int(output_settings.get("max_boundary_debug_images", 12))
    max_path_examples = int(output_settings.get("max_path_examples", 12))
    path_top_k = int(output_settings.get("path_top_k", 5))
    debug_root = report_dir / "checkpoint_boundary_debug"

    print(
        "checkpoint split:",
        f"train_reference={train_count}",
        f"validation={validation_count}",
        f"test={test_count}",
        f"policy={evaluation_split_policy}",
    )
    validation_metrics = evaluate(
        model,
        validation_loader,
        device,
        debug_dir=debug_root / "validation",
        split_name="checkpoint_validation",
        max_debug_images=max_debug_images,
        peak_threshold=peak_threshold,
        peak_min_distance=peak_min_distance,
        max_path_examples=min(3, max_path_examples),
        path_top_k=path_top_k,
        length_path_weight=length_path_weight,
        length_penalty_strength=length_penalty_strength,
        snap_radius_bins=snap_radius_bins,
        recall_threshold=recall_threshold,
        recall_candidate_limit=recall_candidate_limit,
        segment_geometry_weight=segment_geometry_weight,
    )
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        debug_dir=debug_root / "test",
        split_name="checkpoint_test",
        max_debug_images=max_debug_images,
        peak_threshold=peak_threshold,
        peak_min_distance=peak_min_distance,
        max_path_examples=max_path_examples,
        path_top_k=path_top_k,
        length_path_weight=length_path_weight,
        length_penalty_strength=length_penalty_strength,
        snap_radius_bins=snap_radius_bins,
        recall_threshold=recall_threshold,
        recall_candidate_limit=recall_candidate_limit,
        segment_geometry_weight=segment_geometry_weight,
    )
    validation_utility = scribetrain_utility_score(validation_metrics, utility_weights)
    test_utility = scribetrain_utility_score(test_metrics, utility_weights)
    report = {
        "model_name": model_name,
        "status": "checkpoint_evaluated",
        "checkpoint_path": str(checkpoint_path),
        "feature_names": list(checkpoint.get("feature_names", FEATURE_NAMES)),
        "checkpoint": {
            "best_epoch": checkpoint.get("best_epoch"),
            "best_validation_utility_score": checkpoint.get("best_validation_utility_score"),
            "best_validation_peak_f1": checkpoint.get("best_validation_peak_f1"),
        },
        "split": {
            "train_reference": train_count,
            "validation": validation_count,
            "test": test_count,
            "evaluation_policy": evaluation_split_policy,
        },
        "word_pool_split": {
            "source_word_count": len(all_words),
            "train_unique_words": len(train_words),
            "validation_unique_words": len(validation_words),
            "test_unique_words": len(test_words),
            "policy": "unique word text split before synthetic sampling",
        },
        "boundary_debug_dir": str(debug_root),
        "device": str(device),
        "validation_utility_score": validation_utility,
        "test_utility_score": test_utility,
        "utility_weights": utility_weights,
        "validation": validation_metrics,
        "test": test_metrics,
        "top_k_path_examples": test_metrics.get("path_examples", []),
        "elapsed_seconds": round(time.time() - started, 3),
        "notes": [
            "Generated from a saved checkpoint after an interrupted training run.",
            "No training was performed while creating this report.",
        ],
    }
    report_path = save_json(report, report_dir / "checkpoint_report.json")
    print(f"checkpoint: {checkpoint_path}")
    print(f"report:     {report_path}")
    print(
        "checkpoint test:",
        f"boundary_f1={test_metrics['boundary_f1']:.4f}",
        f"boundary_recall={test_metrics['boundary_recall']:.4f}",
        f"peak_f1={test_metrics['peak_f1']:.4f}",
        f"peak_count_pm1={test_metrics['peak_split_count_within_1_accuracy']:.4f}",
        f"length={test_metrics['length_accuracy']:.4f}",
        f"utility={test_utility:.4f}",
    )
    return report


def main() -> None:
    """CLI entrypoint for the sequence splitter trainer."""

    parser = argparse.ArgumentParser(description="Train ScribeTrace word sequence splitter.")
    parser.add_argument("--settings", default=str(DEFAULT_SETTINGS_PATH))
    parser.add_argument("--limit", type=int, default=0, help="Override train sample count for smoke tests.")
    parser.add_argument("--epochs", type=int, default=0, help="Override epoch count.")
    parser.add_argument(
        "--checkpoint-report",
        action="store_true",
        help="Evaluate a saved checkpoint and write checkpoint_report.json without training.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional checkpoint path for --checkpoint-report.",
    )
    args = parser.parse_args()
    settings = load_json(args.settings)
    if args.epochs:
        settings.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.checkpoint_report:
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
        write_checkpoint_report(
            settings,
            checkpoint_path=checkpoint_path,
            limit=args.limit or None,
        )
        return
    train(settings, limit=args.limit or None)


if __name__ == "__main__":
    main()
