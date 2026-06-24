"""Image preparation utilities for the N05 word-level OCR expert.

The resizing strategy follows the CTC-friendly SimpleHTR idea: keep the crop
height fixed, preserve aspect ratio, optionally allow dynamic width, then
transpose the image so time runs along the original x-axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class WordPreprocessSettings:
    """Configuration for preparing word crops for sequence OCR.

    Args:
        target_height: Fixed model input height in pixels.
        target_width: Fixed width when dynamic width is disabled.
        dynamic_width: Whether to preserve width after height normalization.
        padding_px: Extra horizontal padding when dynamic width is enabled.
        background_value: Canvas color, usually white for black-ink crops.
        normalize_range: Output numeric range, either ``minus_one_to_one`` or
            ``zero_to_one``.

    Returns:
        Immutable settings consumed by :func:`prepare_word_image`.
    """

    target_height: int = 32
    target_width: int = 128
    dynamic_width: bool = True
    padding_px: int = 16
    background_value: int = 255
    normalize_range: str = "minus_one_to_one"


def load_grayscale_image(image_path: str | Path) -> np.ndarray:
    """Load one image as grayscale.

    Args:
        image_path: Path to the crop or mask.

    Returns:
        A uint8 grayscale OpenCV array.

    Raises:
        FileNotFoundError: If the path is missing or unreadable.
    """

    path = Path(image_path).expanduser().resolve()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read word OCR crop: {path}")
    return image


def ensure_black_ink_on_white(image: np.ndarray) -> np.ndarray:
    """Normalize crop polarity to black ink on white background.

    Args:
        image: Grayscale image with unknown polarity.

    Returns:
        A uint8 image where the border/background is bright and ink is dark.
    """

    if image.ndim != 2:
        raise ValueError("Word OCR preprocessing expects a single-channel image.")

    border = np.concatenate(
        [
            image[0, :],
            image[-1, :],
            image[:, 0],
            image[:, -1],
        ]
    )
    if float(np.median(border)) < 128.0:
        return 255 - image
    return image.copy()


def threshold_for_word_ocr(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale crop into a stable black-on-white binary crop.

    Args:
        image: Grayscale crop after polarity normalization.

    Returns:
        Binary uint8 image with values 0 or 255.
    """

    normalized = ensure_black_ink_on_white(image)
    _, thresholded = cv2.threshold(
        normalized,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return thresholded


def _round_to_multiple(value: int, multiple: int = 4) -> int:
    """Round a positive integer up to a multiple used by the CNN stride."""

    return max(multiple, value + ((multiple - value) % multiple))


def prepare_word_image(
    image: np.ndarray,
    settings: WordPreprocessSettings | None = None,
) -> np.ndarray:
    """Resize and normalize one word/text-unit crop for a CTC model.

    Args:
        image: Grayscale crop to prepare.
        settings: Optional preprocessing settings.

    Returns:
        Float32 array shaped like SimpleHTR expects: transposed ``W x H``.
    """

    settings = settings or WordPreprocessSettings()
    binary = threshold_for_word_ocr(image).astype(np.float32)
    height, width = binary.shape
    if height <= 0 or width <= 0:
        raise ValueError("Word OCR crop has empty dimensions.")

    if settings.dynamic_width:
        scale = settings.target_height / float(height)
        target_width = int(width * scale + settings.padding_px)
        target_width = _round_to_multiple(target_width)
        target_height = settings.target_height
        offset_x = (target_width - width * scale) / 2.0
        offset_y = 0.0
    else:
        scale = min(
            settings.target_width / float(width),
            settings.target_height / float(height),
        )
        target_width = settings.target_width
        target_height = settings.target_height
        offset_x = (target_width - width * scale) / 2.0
        offset_y = (target_height - height * scale) / 2.0

    transform = np.float32([[scale, 0, offset_x], [0, scale, offset_y]])
    canvas = np.ones((target_height, target_width), dtype=np.float32)
    canvas *= float(settings.background_value)
    resized = cv2.warpAffine(
        binary,
        transform,
        dsize=(target_width, target_height),
        dst=canvas,
        borderMode=cv2.BORDER_TRANSPARENT,
    )

    transposed = cv2.transpose(resized)
    if settings.normalize_range == "zero_to_one":
        return (transposed / 255.0).astype(np.float32)
    if settings.normalize_range == "minus_one_to_one":
        return (transposed / 255.0 - 0.5).astype(np.float32)
    raise ValueError(f"Unsupported normalize_range: {settings.normalize_range}")


def prepare_word_image_from_path(
    image_path: str | Path,
    settings: WordPreprocessSettings | None = None,
) -> np.ndarray:
    """Load and prepare one word crop from disk.

    Args:
        image_path: Path to a word/text-unit crop.
        settings: Optional preprocessing settings.

    Returns:
        Prepared float32 model input.
    """

    return prepare_word_image(load_grayscale_image(image_path), settings=settings)
