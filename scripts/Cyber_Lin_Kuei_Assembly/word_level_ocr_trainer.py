"""Train a synthetic Armenian word-level CRNN/CTC recognizer.

This is the Cyber Lin Kuei arena entry for the N05 word-level OCR expert. It
uses Matenadata glyphs plus an Armenian word-frequency list to synthesize word
crops, then trains a compact CRNN with CTC loss.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "Cyber_Lin_Kuei_Assembly"
    / "word_level_ocr_settings.json"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

TAIL_PROFILES = {
    "no_tail": {"ա", "ո", "ս", "ու", "օ", "ռ"},
    "lower_tail": {
        "բ",
        "գ",
        "դ",
        "զ",
        "ը",
        "թ",
        "լ",
        "ձ",
        "յ",
        "շ",
        "չ",
        "պ",
        "ջ",
        "տ",
        "ր",
        "ց",
        "Ը",
        "Լ",
        "Ջ",
        "Յ",
        "ղ",
        "Ղ",
    },
    "upper_tail": {
        "ե",
        "ժ",
        "ծ",
        "հ",
        "ճ",
        "մ",
        "ն",
        "և",
        "Ա",
        "Ե",
        "Զ",
        "Ժ",
        "Ծ",
        "Հ",
        "Ձ",
        "Ճ",
        "Մ",
        "Ն",
        "Շ",
        "Ո",
        "Չ",
        "Ռ",
        "Ս",
        "Տ",
        "Ու",
        "Օ",
        "Եվ",
    },
    "both_tail": {
        "է",
        "ի",
        "խ",
        "կ",
        "վ",
        "փ",
        "ք",
        "ֆ",
        "Բ",
        "Գ",
        "Դ",
        "Է",
        "Թ",
        "Ի",
        "Խ",
        "Կ",
        "Պ",
        "Վ",
        "Ր",
        "Ց",
        "Փ",
        "Ք",
        "Ֆ",
    },
}


@dataclass(frozen=True)
class WordSample:
    """One selected training word and its token IDs.

    Args:
        text: Original Armenian word.
        token_ids: CTC class IDs, with blank reserved for zero.

    Returns:
        Immutable sample consumed by the synthetic dataset.
    """

    text: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class RenderedWord:
    """Synthetic word image plus structure labels known by the generator.

    Args:
        image: Black-on-white rendered word crop.
        split_x_positions: Approximate x positions between neighboring glyphs.
        bridge_count: Number of synthetic joins drawn between neighbors.
        transition_count: Number of possible neighboring glyph transitions.

    Returns:
        Immutable render result consumed by the dataset.
    """

    image: np.ndarray
    split_x_positions: tuple[int, ...]
    bridge_count: int
    transition_count: int


def resolve_path(path: str | Path) -> Path:
    """Resolve absolute or project-relative paths."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def load_json(path: str | Path) -> dict:
    """Load JSON from disk."""

    with resolve_path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path: str | Path) -> Path:
    """Save JSON with Armenian-safe encoding."""

    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return output_path


def seed_everything(seed: int) -> None:
    """Make synthetic rendering and model training deterministic enough."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_label_map(path: str | Path) -> dict[int, str]:
    """Load numeric glyph labels from the shared Matenadata label map."""

    data = load_json(path)
    return {int(key): str(value) for key, value in data.items()}


def build_token_maps(label_map: dict[int, str]) -> tuple[dict[str, int], dict[int, str]]:
    """Build CTC token maps with zero reserved as blank.

    Args:
        label_map: Existing class-id to Armenian glyph map.

    Returns:
        ``char_to_token`` and ``token_to_char`` maps.
    """

    token_to_char = {class_id + 1: label for class_id, label in label_map.items()}
    char_to_token = {label: token_id for token_id, label in token_to_char.items()}
    return char_to_token, token_to_char


def build_tail_profiles(token_to_char: dict[int, str]) -> dict[int, str]:
    """Map CTC token IDs to vertical tail families.

    Args:
        token_to_char: Token ID to Armenian glyph label map.

    Returns:
        Token ID to one of ``no_tail``, ``lower_tail``, ``upper_tail``, or
        ``both_tail``.
    """

    lookup = {}
    for profile, letters in TAIL_PROFILES.items():
        for letter in letters:
            lookup[letter] = profile
    return {
        token_id: lookup.get(label, "both_tail")
        for token_id, label in token_to_char.items()
    }


def tokenize_word(word: str, char_to_token: dict[str, int]) -> tuple[int, ...] | None:
    """Greedily tokenize an Armenian word using the known glyph inventory.

    Args:
        word: Word from the corpus.
        char_to_token: Glyph/digraph to token ID map.

    Returns:
        Tuple of token IDs, or ``None`` if any part is unsupported.
    """

    units = sorted(char_to_token, key=len, reverse=True)
    tokens: list[int] = []
    index = 0
    while index < len(word):
        match = None
        for unit in units:
            if word.startswith(unit, index):
                match = unit
                break
        if match is None:
            return None
        tokens.append(char_to_token[match])
        index += len(match)
    return tuple(tokens)


def load_word_samples(settings: dict, char_to_token: dict[str, int]) -> list[WordSample]:
    """Load and tokenize Armenian word candidates from the frequency table."""

    dataset = settings["dataset"]
    word_path = resolve_path(dataset["word_frequency_path"])
    max_words = int(dataset.get("max_words", 50000))
    min_len = int(dataset.get("min_word_length", 2))
    max_len = int(dataset.get("max_word_length", 18))

    samples: list[WordSample] = []
    with word_path.open("r", encoding="utf-8") as file:
        header = file.readline()
        if not header.startswith("rank\tword\tcount"):
            file.seek(0)
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word = parts[1]
            if not (min_len <= len(word) <= max_len):
                continue
            tokens = tokenize_word(word, char_to_token)
            if tokens is None:
                continue
            samples.append(WordSample(word, tokens))
            if len(samples) >= max_words:
                break

    if not samples:
        raise ValueError(f"No tokenizable words found in {word_path}")
    return samples


def collect_glyph_paths(matenadata_dir: str | Path, label_map: dict[int, str]) -> dict[int, list[Path]]:
    """Collect available glyph images by CTC token ID."""

    root = resolve_path(matenadata_dir)
    glyph_paths: dict[int, list[Path]] = {}
    for class_id in sorted(label_map):
        folder = root / str(class_id)
        paths = [
            path
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not paths:
            raise FileNotFoundError(f"No glyph images found for class {class_id}: {folder}")
        glyph_paths[class_id + 1] = paths
    return glyph_paths


def _load_glyph_mask(path: Path) -> np.ndarray:
    """Load one glyph as a binary mask where ink is 255."""

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read glyph image: {path}")
    border = np.concatenate([image[0, :], image[-1, :], image[:, 0], image[:, -1]])
    if float(np.median(border)) > 128.0:
        ink_source = 255 - image
    else:
        ink_source = image
    _, mask = cv2.threshold(ink_source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return np.zeros((1, 1), dtype=np.uint8)
    x, y, width, height = cv2.boundingRect(coords)
    return mask[y : y + height, x : x + width]


def _resize_glyph(mask: np.ndarray, target_height: int) -> np.ndarray:
    """Resize one glyph mask to a target height while preserving width."""

    height, width = mask.shape
    if height <= 0 or width <= 0:
        return np.zeros((target_height, 1), dtype=np.uint8)
    scale = target_height / float(height)
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)


def _rotate_glyph_mask(mask: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate a glyph mask by a small angle while preserving all ink."""

    if abs(angle_degrees) < 0.01:
        return mask
    height, width = mask.shape
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = max(1, int(height * sin + width * cos))
    new_height = max(1, int(height * cos + width * sin))
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    rotated = cv2.warpAffine(
        mask,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    coords = cv2.findNonZero(rotated)
    if coords is None:
        return rotated
    x, y, width, height = cv2.boundingRect(coords)
    return rotated[y : y + height, x : x + width]


def _fit_glyph_to_canvas_height(mask: np.ndarray, image_height: int) -> np.ndarray:
    """Shrink rotated glyphs that grew beyond the word canvas height.

    Small rotations expand the bounding box. Without this guard, a 30px glyph can
    become 33px tall and break placement on a 32px training canvas.
    """

    height, width = mask.shape
    if height <= image_height:
        return mask
    scale = image_height / float(height)
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(
        mask,
        (target_width, image_height),
        interpolation=cv2.INTER_NEAREST,
    )


def _word_rotation_angle(rendering: dict, rng: random.Random) -> float:
    """Pick a small shared glyph rotation angle for one synthetic word."""

    if rng.random() > float(rendering.get("glyph_rotation_probability", 0.0)):
        return 0.0
    min_degrees = float(rendering.get("min_glyph_rotation_degrees", 1.0))
    max_degrees = float(rendering.get("max_glyph_rotation_degrees", 4.0))
    magnitude = rng.uniform(min(min_degrees, max_degrees), max(min_degrees, max_degrees))
    return magnitude * (-1.0 if rng.random() < 0.5 else 1.0)


def _rightmost_ink_point(mask: np.ndarray, origin_x: int, origin_y: int) -> tuple[int, int] | None:
    """Find the rightmost ink point of a placed glyph mask."""

    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    points = coords.reshape(-1, 2)
    max_x = int(points[:, 0].max())
    edge_points = points[points[:, 0] >= max_x - 1]
    median_y = int(np.median(edge_points[:, 1]))
    return origin_x + max_x, origin_y + median_y


def _leftmost_ink_point(mask: np.ndarray, origin_x: int, origin_y: int) -> tuple[int, int] | None:
    """Find the leftmost ink point of a placed glyph mask."""

    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    points = coords.reshape(-1, 2)
    min_x = int(points[:, 0].min())
    edge_points = points[points[:, 0] <= min_x + 1]
    median_y = int(np.median(edge_points[:, 1]))
    return origin_x + min_x, origin_y + median_y


def _draw_connection_bridge(
    canvas: np.ndarray,
    left_point: tuple[int, int],
    right_point: tuple[int, int],
    rendering: dict,
    rng: random.Random,
) -> None:
    """Draw a small handwriting-like connector between neighboring glyphs."""

    max_vertical_gap = int(rendering.get("bridge_max_vertical_gap_px", 7))
    thickness = int(rendering.get("bridge_thickness_px", 1))
    curve = int(rendering.get("bridge_curve_px", 2))
    ink_value = int(rendering.get("ink_value", 0))
    x1, y1 = left_point
    x2, y2 = right_point
    if x2 <= x1:
        return
    if abs(y2 - y1) > max_vertical_gap:
        return

    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2 + rng.randint(-curve, curve)
    points = np.array([[x1, y1], [mid_x, mid_y], [x2, y2]], dtype=np.int32)
    cv2.polylines(canvas, [points], isClosed=False, color=ink_value, thickness=thickness)


def _apply_word_slant(image: np.ndarray, rendering: dict, rng: random.Random) -> np.ndarray:
    """Apply a tiny whole-word shear to imitate handwriting slant."""

    if rng.random() > float(rendering.get("slant_probability", 0.0)):
        return image
    max_slant = int(rendering.get("max_slant_px", 0))
    if max_slant <= 0:
        return image
    slant = rng.randint(-max_slant, max_slant)
    if slant == 0:
        return image
    height, width = image.shape
    matrix = np.float32([[1, slant / max(1, height), 0], [0, 1, 0]])
    extra = abs(slant)
    warped = cv2.warpAffine(
        image,
        matrix,
        dsize=(width + extra, height),
        borderValue=int(rendering.get("background_value", 255)),
    )
    return warped


def _apply_n02_style_degradation(
    image: np.ndarray,
    rendering: dict,
    rng: random.Random,
    split_x_positions: tuple[int, ...] | None = None,
):
    """Add mild real-crop noise while keeping a binary black-on-white contract."""

    degraded = image.copy()
    adjusted_positions = list(split_x_positions or ())
    background = int(rendering.get("background_value", 255))
    ink_value = int(rendering.get("ink_value", 0))

    if rng.random() < float(rendering.get("blur_probability", 0.0)):
        degraded = cv2.GaussianBlur(degraded, (3, 3), 0)

    if rng.random() < float(rendering.get("morph_probability", 0.0)):
        kernel = np.ones((2, 2), dtype=np.uint8)
        if rng.random() < 0.5:
            degraded = cv2.erode(degraded, kernel, iterations=1)
        else:
            degraded = cv2.dilate(degraded, kernel, iterations=1)

    if rng.random() < float(rendering.get("noise_probability", 0.0)):
        max_noise = int(rendering.get("max_noise_pixels", 0))
        for _ in range(rng.randint(1, max(1, max_noise))):
            x = rng.randrange(max(1, degraded.shape[1]))
            y = rng.randrange(max(1, degraded.shape[0]))
            degraded[y, x] = ink_value if rng.random() < 0.65 else background

    _, degraded = cv2.threshold(degraded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pad_jitter = int(rendering.get("crop_padding_jitter_px", 0))
    if pad_jitter > 0:
        top = rng.randint(0, pad_jitter)
        bottom = rng.randint(0, pad_jitter)
        left = rng.randint(0, pad_jitter)
        right = rng.randint(0, pad_jitter)
        adjusted_positions = [position + left for position in adjusted_positions]
        old_height = degraded.shape[0]
        old_width = degraded.shape[1]
        degraded = cv2.copyMakeBorder(
            degraded,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=background,
        )
        if degraded.shape[0] != int(rendering.get("image_height", 32)):
            new_height = int(rendering.get("image_height", 32))
            new_width = max(4, int(degraded.shape[1] * new_height / degraded.shape[0]))
            scale_x = new_width / max(1, old_width + left + right)
            adjusted_positions = [int(round(position * scale_x)) for position in adjusted_positions]
            degraded = cv2.resize(
                degraded,
                (new_width, new_height),
                interpolation=cv2.INTER_NEAREST,
            )
    if split_x_positions is None:
        return degraded
    width = degraded.shape[1]
    clamped = tuple(
        sorted(
            {
                max(0, min(width - 1, int(position)))
                for position in adjusted_positions
            }
        )
    )
    return degraded, clamped


def render_synthetic_word(
    sample: WordSample,
    glyph_paths: dict[int, list[Path]],
    tail_profiles: dict[int, str],
    rendering: dict,
    rng: random.Random,
) -> RenderedWord:
    """Render one synthetic word by stitching real Matenadata glyphs.

    Args:
        sample: Tokenized word.
        glyph_paths: Mapping from token IDs to glyph image paths.
        rendering: Rendering settings.
        rng: Local deterministic RNG.

    Returns:
        Black-on-white uint8 word image.
    """

    image_height = int(rendering.get("image_height", 32))
    min_height = int(rendering.get("min_glyph_height", 22))
    max_height = int(rendering.get("max_glyph_height", 30))
    min_gap = int(rendering.get("min_gap_px", -2))
    max_gap = int(rendering.get("max_gap_px", 3))
    max_jitter = int(rendering.get("max_vertical_jitter_px", 2))
    baseline_jitter = int(rendering.get("baseline_jitter_px", max_jitter))
    padding = int(rendering.get("horizontal_padding_px", 8))
    background = int(rendering.get("background_value", 255))
    ink_value = int(rendering.get("ink_value", 0))
    bridge_probability = float(rendering.get("bridge_probability", 0.0))

    glyphs: list[tuple[np.ndarray, int, str]] = []
    total_width = padding * 2
    shared_rotation = _word_rotation_angle(rendering, rng)
    for token_id in sample.token_ids:
        glyph_path = rng.choice(glyph_paths[token_id])
        target_height = _target_height_for_tail_profile(
            tail_profiles.get(token_id, "both_tail"),
            image_height,
            rendering,
            rng,
            fallback_min=min_height,
            fallback_max=max_height,
        )
        glyph = _resize_glyph(_load_glyph_mask(glyph_path), target_height)
        angle = shared_rotation
        if not bool(rendering.get("shared_word_rotation", True)):
            angle = _word_rotation_angle(rendering, rng)
        glyph = _rotate_glyph_mask(glyph, angle)
        glyph = _fit_glyph_to_canvas_height(glyph, image_height)
        gap = rng.randint(min_gap, max_gap)
        glyphs.append((glyph, gap, tail_profiles.get(token_id, "both_tail")))
        total_width += glyph.shape[1] + gap

    total_width = max(4, total_width)
    canvas = np.ones((image_height, total_width), dtype=np.uint8) * background
    x = padding
    baseline = image_height - rng.randint(2, 5)
    placed: list[tuple[np.ndarray, int, int]] = []
    for glyph, gap, tail_profile in glyphs:
        if bool(rendering.get("tail_layout_enabled", True)):
            y = _y_for_tail_profile(
                tail_profile,
                glyph.shape[0],
                image_height,
                rng,
                jitter=max_jitter,
            )
        else:
            baseline_y = baseline - glyph.shape[0] + rng.randint(-baseline_jitter, baseline_jitter)
            center_y = (image_height - glyph.shape[0]) // 2 + rng.randint(-max_jitter, max_jitter)
            y = int(round(0.70 * baseline_y + 0.30 * center_y))
        y = max(0, min(image_height - glyph.shape[0], y))
        x = max(0, min(canvas.shape[1] - glyph.shape[1], x))
        region = canvas[y : y + glyph.shape[0], x : x + glyph.shape[1]]
        region[glyph > 0] = ink_value
        placed.append((glyph, x, y))
        x += glyph.shape[1] + gap

    split_x_positions = []
    for left, right in zip(placed, placed[1:]):
        left_glyph, left_x, _ = left
        _, right_x, _ = right
        split_x = int(round((left_x + left_glyph.shape[1] + right_x) / 2.0))
        split_x_positions.append(split_x)

    bridge_count = 0
    for left, right in zip(placed, placed[1:]):
        if rng.random() > bridge_probability:
            continue
        left_point = _rightmost_ink_point(*left)
        right_point = _leftmost_ink_point(*right)
        if left_point is not None and right_point is not None:
            before = canvas.copy()
            _draw_connection_bridge(canvas, left_point, right_point, rendering, rng)
            if not np.array_equal(before, canvas):
                bridge_count += 1

    canvas = _apply_word_slant(canvas, rendering, rng)
    degraded, adjusted_splits = _apply_n02_style_degradation(
        canvas,
        rendering,
        rng,
        tuple(split_x_positions),
    )
    return RenderedWord(
        image=degraded,
        split_x_positions=adjusted_splits,
        bridge_count=bridge_count,
        transition_count=max(0, len(sample.token_ids) - 1),
    )


def _ratio_range(settings: dict, key: str, default: tuple[float, float]) -> tuple[float, float]:
    """Read a two-number ratio setting safely."""

    value = settings.get(key, list(default))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return default
    low = float(value[0])
    high = float(value[1])
    return max(0.05, min(low, high)), min(1.0, max(low, high))


def _target_height_for_tail_profile(
    profile: str,
    image_height: int,
    rendering: dict,
    rng: random.Random,
    fallback_min: int,
    fallback_max: int,
) -> int:
    """Pick glyph height based on how many horizontal bands it occupies."""

    if not bool(rendering.get("tail_layout_enabled", True)):
        return rng.randint(fallback_min, fallback_max)
    ratio_key = {
        "no_tail": "no_tail_height_ratio",
        "upper_tail": "single_tail_height_ratio",
        "lower_tail": "single_tail_height_ratio",
        "both_tail": "both_tail_height_ratio",
    }.get(profile, "both_tail_height_ratio")
    low, high = _ratio_range(rendering, ratio_key, (0.70, 0.95))
    min_px = max(3, int(round(image_height * low)))
    max_px = max(min_px, int(round(image_height * high)))
    return rng.randint(min_px, max_px)


def _y_for_tail_profile(
    profile: str,
    glyph_height: int,
    image_height: int,
    rng: random.Random,
    jitter: int,
) -> int:
    """Place a glyph inside the top/middle/bottom band model."""

    band = image_height / 3.0
    if profile == "no_tail":
        target_top = band + (band - glyph_height) / 2.0
    elif profile == "upper_tail":
        target_top = (2.0 * band - glyph_height) / 2.0
    elif profile == "lower_tail":
        target_top = band + (2.0 * band - glyph_height) / 2.0
    else:
        target_top = (image_height - glyph_height) / 2.0
    return int(round(target_top + rng.randint(-jitter, jitter)))


def _auxiliary_settings(settings: dict, key: str, defaults: dict) -> dict:
    """Return normalized auxiliary-head settings."""

    configured = settings.get("training", {}).get(key, {})
    result = dict(defaults)
    result.update(configured)
    result["enabled"] = bool(result.get("enabled", True))
    result["loss_weight"] = float(result.get("loss_weight", 0.0))
    return result


def prepare_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert black-on-white word image to a tensor for CRNN input."""

    arr = image.astype(np.float32) / 255.0
    arr = 1.0 - arr
    return torch.from_numpy(arr).unsqueeze(0)


class SyntheticWordDataset(Dataset):
    """On-the-fly synthetic Armenian word dataset."""

    def __init__(
        self,
        samples: list[WordSample],
        glyph_paths: dict[int, list[Path]],
        tail_profiles: dict[int, str],
        rendering: dict,
        seed: int,
        epoch_size: int,
    ) -> None:
        self.samples = samples
        self.glyph_paths = glyph_paths
        self.tail_profiles = tail_profiles
        self.rendering = rendering
        self.seed = seed
        self.epoch_size = epoch_size

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index: int):
        rng = random.Random(self.seed + index)
        sample = rng.choice(self.samples)
        rendered = render_synthetic_word(
            sample,
            self.glyph_paths,
            self.tail_profiles,
            self.rendering,
            rng,
        )
        return (
            prepare_tensor(rendered.image),
            torch.tensor(sample.token_ids, dtype=torch.long),
            sample.text,
            torch.tensor(rendered.bridge_count, dtype=torch.long),
            torch.tensor(rendered.transition_count, dtype=torch.long),
            torch.tensor(rendered.split_x_positions, dtype=torch.long),
        )


def collate_word_batch(batch, boundary_radius_steps: int = 1):
    """Pad variable-width word images and concatenate CTC targets."""

    images, targets, texts, bridge_counts, transition_counts, split_positions = zip(*batch)
    heights = [image.shape[1] for image in images]
    widths = [image.shape[2] for image in images]
    max_width = max(widths)
    if len(set(heights)) != 1:
        raise ValueError("All word images in a batch must have the same height.")

    padded = []
    for image in images:
        pad_width = max_width - image.shape[2]
        padded.append(F.pad(image, (0, pad_width, 0, 0), value=0.0))

    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    flat_targets = torch.cat(targets)
    max_steps = max(1, int(max_width // 4))
    boundary_targets = torch.zeros((len(images), max_steps), dtype=torch.float32)
    for batch_index, positions in enumerate(split_positions):
        for position in positions.tolist():
            step = max(0, min(max_steps - 1, int(round(position / 4.0))))
            left = max(0, step - boundary_radius_steps)
            right = min(max_steps, step + boundary_radius_steps + 1)
            boundary_targets[batch_index, left:right] = 1.0
    return {
        "images": torch.stack(padded, dim=0),
        "targets": flat_targets,
        "target_lengths": target_lengths,
        "length_targets": target_lengths.clone(),
        "bridge_count_targets": torch.stack(list(bridge_counts)).long(),
        "transition_count_targets": torch.stack(list(transition_counts)).long(),
        "boundary_targets": boundary_targets,
        "texts": list(texts),
        "widths": torch.tensor(widths, dtype=torch.long),
    }


class WordCRNN(nn.Module):
    """Compact CNN + BiLSTM + CTC word recognizer."""

    def __init__(
        self,
        num_tokens: int,
        max_length_class: int = 24,
        length_auxiliary_enabled: bool = True,
        max_bridge_count_class: int = 24,
        bridge_auxiliary_enabled: bool = True,
        boundary_auxiliary_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.length_auxiliary_enabled = length_auxiliary_enabled
        self.max_length_class = max_length_class
        self.bridge_auxiliary_enabled = bridge_auxiliary_enabled
        self.boundary_auxiliary_enabled = boundary_auxiliary_enabled
        self.max_bridge_count_class = max_bridge_count_class
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(
            input_size=128 * 2,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )
        self.classifier = nn.Linear(512, num_tokens)
        self.length_head = (
            nn.Linear(512, max_length_class + 1)
            if length_auxiliary_enabled
            else None
        )
        self.bridge_head = (
            nn.Linear(512, max_bridge_count_class + 1)
            if bridge_auxiliary_enabled
            else None
        )
        self.boundary_head = nn.Linear(512, 1) if boundary_auxiliary_enabled else None

    def forward(self, images: torch.Tensor):
        features = self.features(images)
        features = features.mean(dim=2) if features.shape[2] != 2 else features.flatten(1, 2)
        sequence = features.permute(0, 2, 1)
        encoded, _ = self.rnn(sequence)
        logits = self.classifier(encoded)
        length_logits = None
        if self.length_head is not None:
            length_logits = self.length_head(encoded.mean(dim=1))
        bridge_logits = None
        if self.bridge_head is not None:
            bridge_logits = self.bridge_head(encoded.mean(dim=1))
        boundary_logits = None
        if self.boundary_head is not None:
            boundary_logits = self.boundary_head(encoded).squeeze(-1)
        return log_probs_and_aux(logits.log_softmax(dim=2), length_logits, bridge_logits, boundary_logits)


@dataclass(frozen=True)
class log_probs_and_aux:
    """Container for CRNN sequence output and auxiliary predictions."""

    log_probs: torch.Tensor
    length_logits: torch.Tensor | None
    bridge_logits: torch.Tensor | None
    boundary_logits: torch.Tensor | None


def sequence_lengths_from_widths(widths: torch.Tensor) -> torch.Tensor:
    """Estimate CRNN time-step lengths after the horizontal pooling stack."""

    return torch.clamp(torch.div(widths, 4, rounding_mode="floor"), min=1)


def decode_greedy(log_probs: torch.Tensor, token_to_char: dict[int, str]) -> list[str]:
    """Decode CTC output by collapsing repeats and blanks."""

    predictions = log_probs.argmax(dim=2).detach().cpu().numpy()
    decoded = []
    for row in predictions:
        tokens = []
        previous = 0
        for token in row:
            token = int(token)
            if token != 0 and token != previous:
                tokens.append(token_to_char.get(token, ""))
            previous = token
        decoded.append("".join(tokens))
    return decoded


def levenshtein(a: str, b: str) -> int:
    """Compute edit distance without extra dependencies."""

    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    token_to_char: dict[int, str],
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    """Evaluate greedy word accuracy and character error rate."""

    model.eval()
    total_words = 0
    correct_words = 0
    total_edits = 0
    total_chars = 0
    length_total = 0
    length_correct = 0
    bridge_total = 0
    bridge_correct = 0
    boundary_total = 0
    boundary_hits = 0
    examples = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images = batch["images"].to(device)
            outputs = model(images)
            decoded = decode_greedy(outputs.log_probs, token_to_char)
            if outputs.length_logits is not None:
                length_targets = torch.clamp(
                    batch["length_targets"].to(device),
                    max=outputs.length_logits.shape[1] - 1,
                )
                length_predictions = outputs.length_logits.argmax(dim=1)
                length_correct += int((length_predictions == length_targets).sum().item())
                length_total += int(length_targets.numel())
            if outputs.bridge_logits is not None:
                bridge_targets = torch.clamp(
                    batch["bridge_count_targets"].to(device),
                    max=outputs.bridge_logits.shape[1] - 1,
                )
                bridge_predictions = outputs.bridge_logits.argmax(dim=1)
                bridge_correct += int((bridge_predictions == bridge_targets).sum().item())
                bridge_total += int(bridge_targets.numel())
            if outputs.boundary_logits is not None:
                boundary_targets = batch["boundary_targets"].to(device)
                steps = min(boundary_targets.shape[1], outputs.boundary_logits.shape[1])
                boundary_probs = outputs.boundary_logits[:, :steps].sigmoid()
                boundary_targets = boundary_targets[:, :steps]
                predicted = boundary_probs > 0.45
                boundary_hits += int(((predicted == 1) & (boundary_targets == 1)).sum().item())
                boundary_total += int((boundary_targets == 1).sum().item())
            for truth, predicted in zip(batch["texts"], decoded):
                total_words += 1
                correct_words += int(truth == predicted)
                total_edits += levenshtein(truth, predicted)
                total_chars += max(1, len(truth))
                if len(examples) < 12:
                    examples.append({"truth": truth, "predicted": predicted})
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    return {
        "word_accuracy": correct_words / max(1, total_words),
        "character_error_rate": total_edits / max(1, total_chars),
        "length_accuracy": length_correct / max(1, length_total),
        "bridge_count_accuracy": bridge_correct / max(1, bridge_total),
        "boundary_recall": boundary_hits / max(1, boundary_total),
        "evaluated_words": total_words,
        "examples": examples,
    }


def length_auxiliary_settings(settings: dict) -> dict:
    """Return normalized auxiliary word-length settings."""

    configured = settings.get("training", {}).get("length_auxiliary", {})
    return {
        "enabled": bool(configured.get("enabled", True)),
        "loss_weight": float(configured.get("loss_weight", 0.12)),
        "max_length_class": int(
            configured.get(
                "max_length_class",
                settings.get("dataset", {}).get("max_word_length", 24),
            )
        ),
    }


def bridge_auxiliary_settings(settings: dict) -> dict:
    """Return normalized bridge-count auxiliary settings."""

    configured = settings.get("training", {}).get("bridge_auxiliary", {})
    return {
        "enabled": bool(configured.get("enabled", True)),
        "loss_weight": float(configured.get("loss_weight", 0.08)),
        "max_bridge_count_class": int(
            configured.get(
                "max_bridge_count_class",
                settings.get("dataset", {}).get("max_word_length", 24),
            )
        ),
    }


def boundary_auxiliary_settings(settings: dict) -> dict:
    """Return normalized split-boundary auxiliary settings."""

    configured = settings.get("training", {}).get("boundary_auxiliary", {})
    return {
        "enabled": bool(configured.get("enabled", True)),
        "loss_weight": float(configured.get("loss_weight", 0.05)),
        "probability_threshold": float(configured.get("probability_threshold", 0.45)),
    }


def split_word_samples(samples: list[WordSample], seed: int) -> tuple[list[WordSample], list[WordSample], list[WordSample]]:
    """Split unique words into train/validation/test pools."""

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * 0.80)
    val_end = int(len(shuffled) * 0.90)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def make_loader(
    samples: list[WordSample],
    glyph_paths: dict[int, list[Path]],
    tail_profiles: dict[int, str],
    settings: dict,
    seed: int,
    epoch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create one synthetic word DataLoader."""

    dataset = SyntheticWordDataset(
        samples=samples,
        glyph_paths=glyph_paths,
        tail_profiles=tail_profiles,
        rendering=settings["rendering"],
        seed=seed,
        epoch_size=epoch_size,
    )
    training = settings["training"]
    boundary_radius_steps = int(
        training.get("boundary_auxiliary", {}).get("target_radius_steps", 1)
    )
    return DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 64)),
        shuffle=shuffle,
        num_workers=int(training.get("num_workers", 0)),
        collate_fn=lambda batch: collate_word_batch(
            batch,
            boundary_radius_steps=boundary_radius_steps,
        ),
    )


def choose_device(settings: dict) -> torch.device:
    """Pick the requested torch device."""

    requested = settings["training"].get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train(settings: dict, limit_batches: int | None = None) -> dict:
    """Train the synthetic word-level CRNN."""

    seed = int(settings.get("random_seed", 42))
    seed_everything(seed)
    label_map = load_label_map(settings["dataset"]["label_map_path"])
    char_to_token, token_to_char = build_token_maps(label_map)
    tail_profiles = build_tail_profiles(token_to_char)
    samples = load_word_samples(settings, char_to_token)
    glyph_paths = collect_glyph_paths(settings["dataset"]["matenadata_dir"], label_map)
    train_words, val_words, test_words = split_word_samples(samples, seed)
    output = settings["output"]
    model_dir = resolve_path(output["model_dir"])
    report_dir = resolve_path(output["report_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(
        train_words,
        glyph_paths,
        tail_profiles,
        settings,
        seed + 1000,
        int(settings["dataset"].get("train_samples", 20000)),
        shuffle=True,
    )
    val_loader = make_loader(
        val_words,
        glyph_paths,
        tail_profiles,
        settings,
        seed + 2000,
        int(settings["dataset"].get("validation_samples", 2000)),
        shuffle=False,
    )
    test_loader = make_loader(
        test_words,
        glyph_paths,
        tail_profiles,
        settings,
        seed + 3000,
        int(settings["dataset"].get("test_samples", 2000)),
        shuffle=False,
    )

    device = choose_device(settings)
    length_aux = length_auxiliary_settings(settings)
    bridge_aux = bridge_auxiliary_settings(settings)
    boundary_aux = boundary_auxiliary_settings(settings)
    model = WordCRNN(
        num_tokens=len(token_to_char) + 1,
        max_length_class=length_aux["max_length_class"],
        length_auxiliary_enabled=length_aux["enabled"],
        max_bridge_count_class=bridge_aux["max_bridge_count_class"],
        bridge_auxiliary_enabled=bridge_aux["enabled"],
        boundary_auxiliary_enabled=boundary_aux["enabled"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(settings["training"].get("learning_rate", 0.001)),
    )
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    gradient_clip = float(settings["training"].get("gradient_clip", 5.0))
    history = []
    start = time.time()

    for epoch in range(1, int(settings["training"].get("epochs", 12)) + 1):
        model.train()
        losses = []
        for batch_index, batch in enumerate(train_loader, start=1):
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            length_targets = torch.clamp(
                batch["length_targets"].to(device),
                max=length_aux["max_length_class"],
            )
            bridge_targets = torch.clamp(
                batch["bridge_count_targets"].to(device),
                max=bridge_aux["max_bridge_count_class"],
            )
            boundary_targets = batch["boundary_targets"].to(device)
            input_lengths = sequence_lengths_from_widths(batch["widths"]).to(device)
            outputs = model(images)
            ctc_component = ctc_loss(
                outputs.log_probs.permute(1, 0, 2),
                targets,
                input_lengths,
                target_lengths,
            )
            length_component = torch.zeros((), device=device)
            if outputs.length_logits is not None and length_aux["loss_weight"] > 0:
                length_component = F.cross_entropy(outputs.length_logits, length_targets)
            bridge_component = torch.zeros((), device=device)
            if outputs.bridge_logits is not None and bridge_aux["loss_weight"] > 0:
                bridge_component = F.cross_entropy(outputs.bridge_logits, bridge_targets)
            boundary_component = torch.zeros((), device=device)
            if outputs.boundary_logits is not None and boundary_aux["loss_weight"] > 0:
                steps = min(boundary_targets.shape[1], outputs.boundary_logits.shape[1])
                boundary_component = F.binary_cross_entropy_with_logits(
                    outputs.boundary_logits[:, :steps],
                    boundary_targets[:, :steps],
                )
            loss = (
                ctc_component
                + length_aux["loss_weight"] * length_component
                + bridge_aux["loss_weight"] * bridge_component
                + boundary_aux["loss_weight"] * boundary_component
            )
            optimizer.zero_grad()
            loss.backward()
            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if limit_batches is not None and batch_index >= limit_batches:
                break

        val_metrics = evaluate_model(
            model,
            val_loader,
            token_to_char,
            device,
            max_batches=limit_batches,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(1, len(losses)),
            "validation": val_metrics,
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch}: loss={epoch_record['train_loss']:.4f} "
            f"val_word_acc={val_metrics['word_accuracy']:.4f} "
            f"val_cer={val_metrics['character_error_rate']:.4f} "
            f"val_len_acc={val_metrics['length_accuracy']:.4f} "
            f"val_bridge_acc={val_metrics['bridge_count_accuracy']:.4f} "
            f"val_boundary_recall={val_metrics['boundary_recall']:.4f}"
        )

    test_metrics = evaluate_model(
        model,
        test_loader,
        token_to_char,
        device,
        max_batches=limit_batches,
    )

    model_name = settings.get("model_name", "word_level_ocr_v0_1")
    model_path = model_dir / f"{model_name}.pt"
    schema_path = model_dir / "word_level_ocr_schema.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "token_to_char": token_to_char,
            "length_auxiliary": length_aux,
            "bridge_auxiliary": bridge_aux,
            "boundary_auxiliary": boundary_aux,
            "settings": settings,
        },
        model_path,
    )
    save_json(
        {
            "model_name": model_name,
            "blank_token": 0,
            "token_to_char": {str(k): v for k, v in token_to_char.items()},
            "length_auxiliary": length_aux,
            "bridge_auxiliary": bridge_aux,
            "boundary_auxiliary": boundary_aux,
            "input_contract": "black_ink_on_white_word_crop",
            "image_height": settings["rendering"].get("image_height", 32),
        },
        schema_path,
    )
    report = {
        "model_name": model_name,
        "elapsed_seconds": round(time.time() - start, 3),
        "device": str(device),
        "word_pool_count": len(samples),
        "split": {
            "train_words": len(train_words),
            "validation_words": len(val_words),
            "test_words": len(test_words),
        },
        "history": history,
        "test": test_metrics,
        "model_path": str(model_path),
        "schema_path": str(schema_path),
    }
    report_path = save_json(report, report_dir / "training_report.json")
    print(f"model: {model_path}")
    print(f"report: {report_path}")
    return report


def smoke(settings: dict, count: int = 12) -> dict:
    """Render debug synthetic words without training."""

    seed = int(settings.get("random_seed", 42))
    seed_everything(seed)
    label_map = load_label_map(settings["dataset"]["label_map_path"])
    char_to_token, _ = build_token_maps(label_map)
    _, token_to_char = build_token_maps(label_map)
    tail_profiles = build_tail_profiles(token_to_char)
    samples = load_word_samples(settings, char_to_token)
    glyph_paths = collect_glyph_paths(settings["dataset"]["matenadata_dir"], label_map)
    debug_dir = resolve_path(settings["output"]["debug_dir"])
    debug_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    outputs = []
    for index in range(count):
        sample = rng.choice(samples)
        rendered = render_synthetic_word(
            sample,
            glyph_paths,
            tail_profiles,
            settings["rendering"],
            rng,
        )
        output_path = debug_dir / f"synthetic_word_{index:03d}_{sample.text}.png"
        Image.fromarray(rendered.image).save(output_path)
        outputs.append(
            {
                "word": sample.text,
                "path": str(output_path),
                "tokens": list(sample.token_ids),
                "split_x_positions": list(rendered.split_x_positions),
                "bridge_count": rendered.bridge_count,
                "transition_count": rendered.transition_count,
            }
        )
    summary_path = save_json({"samples": outputs}, debug_dir / "smoke_summary.json")
    print(f"smoke_summary: {summary_path}")
    return {"samples": outputs, "summary_path": str(summary_path)}


def main() -> None:
    """CLI entrypoint for the word-level OCR training arena."""

    parser = argparse.ArgumentParser(description="Train/smoke Armenian word-level OCR.")
    parser.add_argument("--settings", default=str(DEFAULT_SETTINGS_PATH))
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    parser.add_argument("--smoke-count", type=int, default=12)
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=0,
        help="Debug only: cap train/eval batches per epoch.",
    )
    args = parser.parse_args()

    settings = load_json(args.settings)
    if args.mode == "smoke":
        smoke(settings, count=args.smoke_count)
    else:
        train(settings, limit_batches=args.limit_batches or None)


if __name__ == "__main__":
    main()
