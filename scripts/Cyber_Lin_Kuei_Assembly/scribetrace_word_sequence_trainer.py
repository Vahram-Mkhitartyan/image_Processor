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
        "PyTorch is required for scribetrace_word_sequence_trainer.py. "
        "Run it inside the project .venv."
    ) from error

LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(LOCAL_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

try:
    from .word_level_ocr_trainer import (
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
    / "scribetrace_word_sequence_settings.json"
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


@dataclass(frozen=True)
class SequenceSample:
    """One generated training row returned by the dataset.

    Args:
        features: Float tensor shaped ``[bin_count, feature_count]``.
        boundaries: Soft boundary heatmap target shaped ``[bin_count]``.
        hard_boundaries: One-hot true split bins shaped ``[bin_count]``.
        length: Integer token count target.
        text: Human-readable synthetic word text for debugging.
        image: Original rendered word image for debug previews.
    """

    features: torch.Tensor
    boundaries: torch.Tensor
    hard_boundaries: torch.Tensor
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


class SyntheticScribeTraceSequenceDataset(Dataset):
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
        return SequenceSample(
            features=torch.tensor(features, dtype=torch.float32),
            boundaries=torch.tensor(boundaries, dtype=torch.float32),
            hard_boundaries=torch.tensor(hard_boundaries, dtype=torch.float32),
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
        "lengths": torch.stack([sample.length for sample in samples]),
        "texts": [sample.text for sample in samples],
        "images": [sample.image for sample in samples],
    }


class ScribeTraceWordSequenceModel(nn.Module):
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
        mean_pool = encoded.mean(dim=1)
        max_pool = encoded.max(dim=1).values
        length_logits = self.length_head(torch.cat([mean_pool, max_pool], dim=1))
        return {"boundary_logits": boundary_logits, "length_logits": length_logits}


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

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            boundaries = batch["boundaries"].to(device)
            hard_boundaries = batch["hard_boundaries"].to(device)
            lengths = batch["lengths"].to(device)
            outputs = model(features)
            boundary_loss = F.binary_cross_entropy_with_logits(
                outputs["boundary_logits"],
                boundaries,
            )
            length_loss = F.cross_entropy(outputs["length_logits"], lengths)
            loss_total += float((boundary_loss + length_loss).item())
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
    if limit:
        # Smoke runs should validate the plumbing, not quietly render thousands
        # of extra words after a tiny train split.
        validation_count = max(4, min(64, train_count // 2))
        test_count = max(4, min(64, train_count // 2))
    else:
        validation_count = int(dataset_settings.get("validation_samples", max(200, train_count // 8)))
        test_count = int(dataset_settings.get("test_samples", max(200, train_count // 8)))
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

    train_set = SyntheticScribeTraceSequenceDataset(
        settings,
        "train",
        train_count,
        seed_offset=0,
        words=train_words,
        glyph_paths=glyph_paths,
        tail_profiles=tail_profiles,
    )
    validation_set = SyntheticScribeTraceSequenceDataset(
        settings,
        "validation",
        validation_count,
        seed_offset=1_000_000,
        words=validation_words,
        glyph_paths=glyph_paths,
        tail_profiles=tail_profiles,
    )
    test_set = SyntheticScribeTraceSequenceDataset(
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

    model = ScribeTraceWordSequenceModel(
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
    peak_threshold = float(training_settings.get("peak_threshold", 0.35))
    peak_min_distance = int(training_settings.get("peak_min_distance_bins", 2))
    epochs = int(training_settings.get("epochs", 12))
    best_metric = -math.inf
    best_epoch = 0
    history = []
    model_dir = resolve_path(output_settings["model_dir"])
    report_dir = resolve_path(output_settings["report_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{settings.get('model_name', 'scribetrace_word_sequence_v0_1')}.pt"
    debug_root = report_dir / "boundary_debug"
    max_debug_images = int(output_settings.get("max_boundary_debug_images", 12))

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batch_total = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            boundaries = batch["boundaries"].to(device)
            lengths = batch["lengths"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features)
            boundary_loss = F.binary_cross_entropy_with_logits(
                outputs["boundary_logits"],
                boundaries,
                pos_weight=positive_weight,
            )
            length_loss = F.cross_entropy(outputs["length_logits"], lengths)
            loss = boundary_loss + length_loss_weight * length_loss
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
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, batch_total),
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
            f"val_len_acc={validation_metrics['length_accuracy']:.4f}"
        )
        if validation_metrics["peak_f1"] > best_metric:
            best_metric = validation_metrics["peak_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model_name": settings.get("model_name", "scribetrace_word_sequence_v0_1"),
                    "state_dict": model.state_dict(),
                    "feature_names": list(FEATURE_NAMES),
                    "settings": settings,
                    "best_epoch": best_epoch,
                    "best_validation_peak_f1": best_metric,
                },
                model_path,
            )
            print(f"  saved best model: {model_path}")

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
    )
    report = {
        "model_name": settings.get("model_name", "scribetrace_word_sequence_v0_1"),
        "status": "completed",
        "model_path": str(model_path),
        "feature_names": list(FEATURE_NAMES),
        "split": {
            "train": train_count,
            "validation": validation_count,
            "test": test_count,
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
        "best_validation_peak_f1": best_metric,
        "test": test_metrics,
        "history": history,
        "elapsed_seconds": round(time.time() - started, 3),
        "notes": [
            "v0.1 predicts word split boundaries from local ScribeTrace-style features.",
            "This is splitter-first; OCR character decisions stay downstream.",
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
    )
    return report


def main() -> None:
    """CLI entrypoint for the sequence splitter trainer."""

    parser = argparse.ArgumentParser(description="Train ScribeTrace word sequence splitter.")
    parser.add_argument("--settings", default=str(DEFAULT_SETTINGS_PATH))
    parser.add_argument("--limit", type=int, default=0, help="Override train sample count for smoke tests.")
    parser.add_argument("--epochs", type=int, default=0, help="Override epoch count.")
    args = parser.parse_args()
    settings = load_json(args.settings)
    if args.epochs:
        settings.setdefault("training", {})["epochs"] = int(args.epochs)
    train(settings, limit=args.limit or None)


if __name__ == "__main__":
    main()
