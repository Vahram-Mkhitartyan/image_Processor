from dataclasses import dataclass
from typing import Protocol, Any
import cv2
import numpy as np


def _polarity_values(image):
    """Infer background and foreground values from border and image contrast."""
    border = np.concatenate(
        [image[0, :], image[-1, :], image[:, 0], image[:, -1]]
    )
    background = 0 if float(np.median(border)) < 128 else 255
    foreground = 255 - background
    return background, foreground


def _foreground_mask(image):
    """Return a binary foreground mask independent of source polarity."""
    background, foreground = _polarity_values(image)
    if foreground > background:
        return np.where(image > 127, 255, 0).astype(np.uint8)
    return np.where(image < 128, 255, 0).astype(np.uint8)


class DamageOperation(Protocol):
    name: str

    def apply(
        self,
        image: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        ...


@dataclass
class CutLineDamage:
    name: str = "cut_line"
    thickness: int = 1

    def apply(self, image, rng):
        damaged = image.copy()
        background, _ = _polarity_values(damaged)
        foreground = _foreground_mask(damaged)
        points = cv2.findNonZero(foreground)
        if points is None:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "no_foreground",
            }

        x, y, width, height = cv2.boundingRect(points)
        x1 = int(rng.integers(x, x + width))
        x2 = int(rng.integers(x, x + width))
        y1 = int(rng.integers(y, y + height))
        y2 = int(rng.integers(y, y + height))

        cv2.line(
            damaged,
            (x1, y1),
            (x2, y2),
            int(background),
            self.thickness,
        )

        return damaged, {
            "operation": self.name,
            "applied": True,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "thickness": self.thickness,
        }


@dataclass
class BlurDamage:
    name: str = "blur"
    kernel_size: int = 3

    def apply(self, image, rng):
        kernel = self.kernel_size if self.kernel_size % 2 == 1 else self.kernel_size + 1
        damaged = cv2.GaussianBlur(image, (kernel, kernel), 0)

        return damaged, {
            "operation": self.name,
            "kernel_size": kernel,
        }


@dataclass
class BlackNoiseDamage:
    name: str = "black_noise"
    probability: float = 0.01

    def apply(self, image, rng):
        damaged = image.copy()
        _, foreground = _polarity_values(damaged)
        mask = rng.random(damaged.shape[:2]) < self.probability
        damaged[mask] = foreground

        return damaged, {
            "operation": self.name,
            "probability": self.probability,
            "pixels_added": int(np.count_nonzero(mask)),
            "foreground_value": foreground,
        }


@dataclass
class ErosionDamage:
    name: str = "erosion"
    kernel_size: int = 2
    iterations: int = 1

    def apply(self, image, rng):
        kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)

        background, foreground = _polarity_values(image)
        foreground_mask = _foreground_mask(image)
        eroded = cv2.erode(
            foreground_mask,
            kernel,
            iterations=self.iterations,
        )
        damaged = np.full_like(image, background)
        damaged[eroded > 0] = foreground

        return damaged, {
            "operation": self.name,
            "kernel_size": self.kernel_size,
            "iterations": self.iterations,
            "background_value": background,
            "foreground_value": foreground,
        }
