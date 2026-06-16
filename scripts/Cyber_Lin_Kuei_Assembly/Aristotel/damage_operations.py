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
        return np.where(image > 80, 255, 0).astype(np.uint8)
    return np.where(image < 215, 255, 0).astype(np.uint8)


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
    """
    Border-to-border cut.

    Picks a random point on one glyph bounding-box wall,
    then another random point on the opposite parallel wall,
    and draws a background-colored cut line between them.
    """

    name: str = "cut_line"
    thickness: int = 1
    max_attempts: int = 40
    min_removed_pixels: int = 2
    max_removed_ratio: float = 0.12
    orientation: str = "random"  # "random", "left_right", or "top_bottom"

    def apply(self, image, rng):
        damaged = image.copy()
        background, _ = _polarity_values(damaged)

        foreground_before = _foreground_mask(damaged)
        total_foreground = int(cv2.countNonZero(foreground_before))

        if total_foreground <= 0:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "no_foreground",
            }

        points = cv2.findNonZero(foreground_before)

        if points is None:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "no_foreground_points",
            }

        x, y, width, height = cv2.boundingRect(points)

        if width <= 1 or height <= 1:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "bbox_too_small",
                "bbox": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(width),
                    "height": int(height),
                },
            }

        best_candidate = None

        for attempt in range(self.max_attempts):
            if self.orientation == "left_right":
                chosen_orientation = "left_right"
            elif self.orientation == "top_bottom":
                chosen_orientation = "top_bottom"
            else:
                chosen_orientation = (
                    "left_right"
                    if int(rng.integers(0, 2)) == 0
                    else "top_bottom"
                )

            if chosen_orientation == "left_right":
                # Point on left bbox wall.
                x1 = x
                y1 = int(rng.integers(y, y + height))

                # Point on right bbox wall.
                x2 = x + width - 1
                y2 = int(rng.integers(y, y + height))

            else:
                # Point on top bbox wall.
                x1 = int(rng.integers(x, x + width))
                y1 = y

                # Point on bottom bbox wall.
                x2 = int(rng.integers(x, x + width))
                y2 = y + height - 1

            candidate = damaged.copy()

            cv2.line(
                candidate,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                int(background),
                int(self.thickness),
            )

            foreground_after = _foreground_mask(candidate)

            removed_mask = cv2.bitwise_and(
                foreground_before,
                cv2.bitwise_not(foreground_after),
            )

            removed_pixels = int(cv2.countNonZero(removed_mask))
            removed_ratio = removed_pixels / max(1, total_foreground)

            if removed_pixels < self.min_removed_pixels:
                continue

            if removed_ratio > self.max_removed_ratio:
                continue

            best_candidate = {
                "image": candidate,
                "attempt": attempt,
                "orientation": chosen_orientation,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "removed_foreground_pixels": removed_pixels,
                "removed_foreground_ratio": removed_ratio,
            }

            break

        if best_candidate is None:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "could_not_make_valid_border_cut",
                "attempts": int(self.max_attempts),
                "bbox": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(width),
                    "height": int(height),
                },
                "total_foreground_pixels": int(total_foreground),
                "thickness": int(self.thickness),
                "orientation": self.orientation,
            }

        return best_candidate["image"], {
            "operation": self.name,
            "applied": True,
            "mode": "border_to_border_cut",
            "attempt": best_candidate["attempt"],
            "orientation": best_candidate["orientation"],
            "x1": best_candidate["x1"],
            "y1": best_candidate["y1"],
            "x2": best_candidate["x2"],
            "y2": best_candidate["y2"],
            "thickness": int(self.thickness),
            "removed_foreground_pixels": best_candidate["removed_foreground_pixels"],
            "removed_foreground_ratio": best_candidate["removed_foreground_ratio"],
            "total_foreground_pixels": int(total_foreground),
            "bbox": {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
            },
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
    boundary_remove_probability: float = 0.25
    kernel_size: int = 3

    def apply(self, image, rng):
        background, foreground = _polarity_values(image)

        foreground_mask = _foreground_mask(image)
        total_foreground = int(cv2.countNonZero(foreground_mask))

        if total_foreground <= 0:
            return image.copy(), {
                "operation": self.name,
                "applied": False,
                "reason": "no_foreground",
            }

        kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)

        # Inner core survives. Boundary is what erosion would remove.
        eroded_core = cv2.erode(foreground_mask, kernel, iterations=1)
        boundary = cv2.subtract(foreground_mask, eroded_core)

        boundary_points = boundary > 0
        remove_mask = (
            boundary_points
            & (rng.random(foreground_mask.shape[:2]) < self.boundary_remove_probability)
        )

        damaged_mask = foreground_mask.copy()
        damaged_mask[remove_mask] = 0

        damaged = np.full_like(image, background)
        damaged[damaged_mask > 0] = foreground

        removed_pixels = int(np.count_nonzero(remove_mask))
        removed_ratio = removed_pixels / max(1, total_foreground)

        return damaged, {
            "operation": self.name,
            "applied": True,
            "mode": "probabilistic_boundary_erosion",
            "boundary_remove_probability": self.boundary_remove_probability,
            "kernel_size": self.kernel_size,
            "removed_foreground_pixels": removed_pixels,
            "removed_foreground_ratio": removed_ratio,
            "total_foreground_pixels": total_foreground,
            "background_value": background,
            "foreground_value": foreground,
        }


@dataclass
class StampInterferenceDamage:
    """
    Simulate a stamp mark overlapping handwriting.

    This adds foreign foreground-like ink on top of the glyph:
    - ellipse/ring border
    - fake internal text bars
    - optional diagonal artifact line

    It should contaminate the glyph, not delete it.
    """

    name: str = "stamp_interference"
    opacity: float = 0.55
    ring_thickness: int = 1
    internal_line_count: int = 3
    min_added_pixels: int = 5
    max_added_ratio: float = 0.35
    max_attempts: int = 30

    def apply(self, image, rng):
        damaged = image.copy()
        background, foreground = _polarity_values(damaged)

        foreground_before = _foreground_mask(damaged)
        total_foreground = int(cv2.countNonZero(foreground_before))

        if total_foreground <= 0:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "no_foreground",
            }

        points = cv2.findNonZero(foreground_before)
        if points is None:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "no_foreground_points",
            }

        height, width = image.shape[:2]
        x, y, bbox_w, bbox_h = cv2.boundingRect(points)

        best_candidate = None

        for attempt in range(self.max_attempts):
            stamp_mask = np.zeros((height, width), dtype=np.uint8)

            # Big stamp, larger than the glyph crop.
            stamp_w = int(rng.integers(
                max(12, int(bbox_w * 1.2)),
                max(13, int(bbox_w * 2.4)) + 1,
            ))

            stamp_h = int(rng.integers(
                max(12, int(bbox_h * 1.2)),
                max(13, int(bbox_h * 2.2)) + 1,
            ))

            # Center near the glyph center, with some jitter.
            axes = (
                max(3, stamp_w // 2),
                max(3, stamp_h // 2),
            )

            angle = float(rng.integers(-18, 19))

            # Place the big stamp partly outside/near the crop,
            # so only a sector enters the glyph area.
            side = int(rng.integers(0, 4))

            if side == 0:  # left outside
                cx = int(rng.integers(-axes[0], max(1, x + bbox_w // 2)))
                cy = int(rng.integers(y, y + bbox_h))

            elif side == 1:  # right outside
                cx = int(rng.integers(x + bbox_w // 2, width + axes[0]))
                cy = int(rng.integers(y, y + bbox_h))

            elif side == 2:  # top outside
                cx = int(rng.integers(x, x + bbox_w))
                cy = int(rng.integers(-axes[1], max(1, y + bbox_h // 2)))

            else:  # bottom outside
                cx = int(rng.integers(x, x + bbox_w))
                cy = int(rng.integers(y + bbox_h // 2, height + axes[1]))

            # a sector-like object that mimmics the stamp part
            start_angle = int(rng.integers(0, 360))
            arc_length = int(rng.integers(35, 120))
            end_angle = start_angle + arc_length

            cv2.ellipse(
                stamp_mask,
                (cx, cy),
                axes,
                angle,
                start_angle,
                end_angle,
                255,
                int(self.ring_thickness),
            )

            # Optional second inner arc sometimes.
            if int(rng.integers(0, 2)) == 1:
                inner_axes = (
                    max(2, int(axes[0] * 0.78)),
                    max(2, int(axes[1] * 0.72)),
                )

                cv2.ellipse(
                    stamp_mask,
                    (cx, cy),
                    inner_axes,
                    angle,
                    start_angle,
                    end_angle,
                    255,
                    1,
                )
            # Fake text bars inside stamp.
            line_span = max(4, int(axes[0] * 1.25))
            y_offsets = np.linspace(
                -axes[1] * 0.45,
                axes[1] * 0.45,
                max(1, self.internal_line_count),
            )

            for offset in y_offsets:
                local_y = int(round(cy + offset + rng.integers(-1, 2)))
                local_x1 = int(round(cx - line_span / 2 + rng.integers(-2, 3)))
                local_x2 = int(round(cx + line_span / 2 + rng.integers(-2, 3)))

                local_x1 = int(np.clip(local_x1, 0, width - 1))
                local_x2 = int(np.clip(local_x2, 0, width - 1))
                local_y = int(np.clip(local_y, 0, height - 1))

                cv2.line(
                    stamp_mask,
                    (local_x1, local_y),
                    (local_x2, local_y),
                    255,
                    1,
                )

            # Occasional diagonal/straight stamp artifact.
            if int(rng.integers(0, 3)) == 0:
                x1 = int(np.clip(cx - axes[0], 0, width - 1))
                y1 = int(np.clip(cy - axes[1], 0, height - 1))
                x2 = int(np.clip(cx + axes[0], 0, width - 1))
                y2 = int(np.clip(cy + axes[1], 0, height - 1))

                cv2.line(
                    stamp_mask,
                    (x1, y1),
                    (x2, y2),
                    255,
                    1,
                )

            before_mask = _foreground_mask(damaged)

            candidate = damaged.copy().astype(np.float32)

            # Blend stamp toward foreground value.
            stamp_pixels = stamp_mask > 0
            candidate[stamp_pixels] = (
                candidate[stamp_pixels] * (1.0 - self.opacity)
                + float(foreground) * self.opacity
            )

            candidate = np.clip(candidate, 0, 255).astype(np.uint8)

            after_mask = _foreground_mask(candidate)

            added_mask = cv2.bitwise_and(
                after_mask,
                cv2.bitwise_not(before_mask),
            )

            added_pixels = int(cv2.countNonZero(added_mask))
            added_ratio = added_pixels / max(1, total_foreground)

            if added_pixels < self.min_added_pixels:
                continue

            if added_ratio > self.max_added_ratio:
                continue

            best_candidate = {
                "image": candidate,
                "attempt": attempt,
                "cx": cx,
                "cy": cy,
                "stamp_width": stamp_w,
                "stamp_height": stamp_h,
                "angle": angle,
                "added_foreground_pixels": added_pixels,
                "added_foreground_ratio": added_ratio,
                "stamp_mask_pixels": int(cv2.countNonZero(stamp_mask)),
            }
            break

        if best_candidate is None:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "could_not_make_valid_stamp",
                "attempts": int(self.max_attempts),
                "total_foreground_pixels": int(total_foreground),
            }

        return best_candidate["image"], {
            "operation": self.name,
            "applied": True,
            "mode": "partial_stamp_sector_interference",
            "attempt": best_candidate["attempt"],
            "cx": best_candidate["cx"],
            "cy": best_candidate["cy"],
            "stamp_width": best_candidate["stamp_width"],
            "stamp_height": best_candidate["stamp_height"],
            "angle": best_candidate["angle"],
            "opacity": float(self.opacity),
            "ring_thickness": int(self.ring_thickness),
            "internal_line_count": int(self.internal_line_count),
            "stamp_mask_pixels": best_candidate["stamp_mask_pixels"],
            "added_foreground_pixels": best_candidate["added_foreground_pixels"],
            "added_foreground_ratio": best_candidate["added_foreground_ratio"],
            "total_foreground_pixels": int(total_foreground),
            "background_value": int(background),
            "foreground_value": int(foreground),
        }

@dataclass
class BleedThroughDamage:
    """
    Simulate ink bleeding through from the back side of paper.

    This creates a faint, blurry, offset ghost of foreground-like ink.
    It should be weaker than the main glyph and should mostly act as
    background contamination / threshold confusion.
    """

    name: str = "bleed_through"
    opacity: float = 0.28
    blur_kernel_size: int = 5
    min_shift_px: int = 3
    max_shift_px: int = 8
    min_added_pixels: int = 3
    max_added_ratio: float = 0.30
    max_attempts: int = 30
    allow_flip: bool = True

    def apply(self, image, rng):
        damaged = image.copy()
        background, foreground = _polarity_values(damaged)

        foreground_before = _foreground_mask(damaged)
        total_foreground = int(cv2.countNonZero(foreground_before))

        if total_foreground <= 0:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "no_foreground",
            }

        height, width = image.shape[:2]

        best_candidate = None

        for attempt in range(self.max_attempts):
            ghost_mask = foreground_before.copy()

            # Back-side bleed often appears mirrored or partially shifted.
            if self.allow_flip and int(rng.integers(0, 2)) == 1:
                ghost_mask = cv2.flip(ghost_mask, 1)

            shift_x = int(rng.integers(self.min_shift_px, self.max_shift_px + 1))
            shift_y = int(rng.integers(self.min_shift_px, self.max_shift_px + 1))

            if int(rng.integers(0, 2)) == 0:
                shift_x = -shift_x
            if int(rng.integers(0, 2)) == 0:
                shift_y = -shift_y

            matrix = np.float32([
                [1, 0, shift_x],
                [0, 1, shift_y],
            ])

            shifted = cv2.warpAffine(
                ghost_mask,
                matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            kernel = self.blur_kernel_size
            if kernel % 2 == 0:
                kernel += 1

            ghost_soft = cv2.GaussianBlur(
                shifted,
                (kernel, kernel),
                0,
            )

            ghost_pixels = ghost_soft > 0

            candidate = damaged.copy().astype(np.float32)

            # Blend weak ghost ink toward foreground.
            strength = (ghost_soft.astype(np.float32) / 255.0) * float(self.opacity)

            candidate[ghost_pixels] = (
                candidate[ghost_pixels] * (1.0 - strength[ghost_pixels])
                + float(foreground) * strength[ghost_pixels]
            )

            candidate = np.clip(candidate, 0, 255).astype(np.uint8)

            foreground_after = _foreground_mask(candidate)

            added_mask = cv2.bitwise_and(
                foreground_after,
                cv2.bitwise_not(foreground_before),
            )

            added_pixels = int(cv2.countNonZero(added_mask))
            added_ratio = added_pixels / max(1, total_foreground)

            if added_pixels < self.min_added_pixels:
                continue

            if added_ratio > self.max_added_ratio:
                continue

            best_candidate = {
                "image": candidate,
                "attempt": attempt,
                "shift_x": shift_x,
                "shift_y": shift_y,
                "blur_kernel_size": kernel,
                "added_foreground_pixels": added_pixels,
                "added_foreground_ratio": added_ratio,
                "ghost_mask_pixels": int(cv2.countNonZero(shifted)),
                "ghost_soft_pixels": int(np.count_nonzero(ghost_pixels)),
            }
            break

        if best_candidate is None:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "could_not_make_valid_bleed_through",
                "attempts": int(self.max_attempts),
                "total_foreground_pixels": int(total_foreground),
            }

        return best_candidate["image"], {
            "operation": self.name,
            "applied": True,
            "mode": "faint_offset_ghost_ink",
            "attempt": best_candidate["attempt"],
            "opacity": float(self.opacity),
            "shift_x": best_candidate["shift_x"],
            "shift_y": best_candidate["shift_y"],
            "blur_kernel_size": best_candidate["blur_kernel_size"],
            "ghost_mask_pixels": best_candidate["ghost_mask_pixels"],
            "ghost_soft_pixels": best_candidate["ghost_soft_pixels"],
            "added_foreground_pixels": best_candidate["added_foreground_pixels"],
            "added_foreground_ratio": best_candidate["added_foreground_ratio"],
            "total_foreground_pixels": int(total_foreground),
            "background_value": int(background),
            "foreground_value": int(foreground),
        }


@dataclass
class EdgeCropLossDamage:
    """
    Simulate crop/refiner cutting off part of a glyph at one edge.

    This removes foreground from one side of the glyph bounding box while
    keeping the image canvas size unchanged.
    """

    name: str = "edge_crop_loss"
    min_crop_ratio: float = 0.08
    max_crop_ratio: float = 0.22
    min_removed_pixels: int = 3
    max_removed_ratio: float = 0.30
    max_attempts: int = 30
    side: str = "random"  # random, left, right, top, bottom

    def apply(self, image, rng):
        damaged = image.copy()
        background, foreground = _polarity_values(damaged)

        foreground_before = _foreground_mask(damaged)
        total_foreground = int(cv2.countNonZero(foreground_before))

        if total_foreground <= 0:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "no_foreground",
            }

        points = cv2.findNonZero(foreground_before)
        if points is None:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "no_foreground_points",
            }

        height, width = image.shape[:2]
        x, y, bbox_w, bbox_h = cv2.boundingRect(points)

        valid_sides = ["left", "right", "top", "bottom"]
        best_candidate = None

        for attempt in range(self.max_attempts):
            chosen_side = (
                valid_sides[int(rng.integers(0, len(valid_sides)))]
                if self.side == "random"
                else self.side
            )

            if chosen_side not in valid_sides:
                return damaged, {
                    "operation": self.name,
                    "applied": False,
                    "reason": "invalid_side",
                    "side": self.side,
                }

            candidate = damaged.copy()

            if chosen_side in {"left", "right"}:
                depth = int(round(
                    bbox_w * float(rng.uniform(self.min_crop_ratio, self.max_crop_ratio))
                ))
                depth = max(1, depth)

                if chosen_side == "left":
                    crop_line = int(np.clip(x + depth, 0, width))
                    candidate[:, :crop_line] = background
                else:
                    crop_line = int(np.clip(x + bbox_w - depth, 0, width))
                    candidate[:, crop_line:] = background

            else:
                depth = int(round(
                    bbox_h * float(rng.uniform(self.min_crop_ratio, self.max_crop_ratio))
                ))
                depth = max(1, depth)

                if chosen_side == "top":
                    crop_line = int(np.clip(y + depth, 0, height))
                    candidate[:crop_line, :] = background
                else:
                    crop_line = int(np.clip(y + bbox_h - depth, 0, height))
                    candidate[crop_line:, :] = background

            foreground_after = _foreground_mask(candidate)

            removed_mask = cv2.bitwise_and(
                foreground_before,
                cv2.bitwise_not(foreground_after),
            )

            removed_pixels = int(cv2.countNonZero(removed_mask))
            removed_ratio = removed_pixels / max(1, total_foreground)

            if removed_pixels < self.min_removed_pixels:
                continue

            if removed_ratio > self.max_removed_ratio:
                continue

            best_candidate = {
                "image": candidate,
                "attempt": attempt,
                "side": chosen_side,
                "depth": depth,
                "crop_line": crop_line,
                "removed_foreground_pixels": removed_pixels,
                "removed_foreground_ratio": removed_ratio,
            }
            break

        if best_candidate is None:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "could_not_make_valid_edge_crop_loss",
                "attempts": int(self.max_attempts),
                "total_foreground_pixels": int(total_foreground),
                "bbox": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(bbox_w),
                    "height": int(bbox_h),
                },
            }

        return best_candidate["image"], {
            "operation": self.name,
            "applied": True,
            "mode": "glyph_edge_crop_loss",
            "attempt": best_candidate["attempt"],
            "side": best_candidate["side"],
            "depth": best_candidate["depth"],
            "crop_line": best_candidate["crop_line"],
            "removed_foreground_pixels": best_candidate["removed_foreground_pixels"],
            "removed_foreground_ratio": best_candidate["removed_foreground_ratio"],
            "total_foreground_pixels": int(total_foreground),
            "bbox": {
                "x": int(x),
                "y": int(y),
                "width": int(bbox_w),
                "height": int(bbox_h),
            },
            "background_value": int(background),
            "foreground_value": int(foreground),
        }
    
@dataclass
class ThresholdFailureDamage:
    """
    Simulate bad binarization / threshold failure.

    This does not directly erase with a geometric tool.
    It makes weak ink disappear and/or background dirt become foreground,
    like a bad threshold step would do.
    """

    name: str = "threshold_failure"
    mode: str = "random"  # random, under_threshold, over_threshold, uneven
    min_changed_pixels: int = 5
    max_changed_ratio: float = 0.35
    max_attempts: int = 30

    def apply(self, image, rng):
        damaged = image.copy()
        background, foreground = _polarity_values(damaged)

        foreground_before = _foreground_mask(damaged)
        total_foreground = int(cv2.countNonZero(foreground_before))

        if total_foreground <= 0:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "no_foreground",
            }

        height, width = image.shape[:2]

        best_candidate = None

        for attempt in range(self.max_attempts):
            chosen_mode = self.mode
            if chosen_mode == "random":
                chosen_mode = ["under_threshold", "over_threshold", "uneven"][
                    int(rng.integers(0, 3))
                ]

            gray = image.copy().astype(np.uint8)

            if chosen_mode == "under_threshold":
                # Threshold too strict: weak/faint ink disappears.
                threshold = int(rng.integers(145, 190))

                if foreground > background:
                    failed_mask = np.where(gray > threshold, 255, 0).astype(np.uint8)
                else:
                    failed_mask = np.where(gray < 255 - threshold, 255, 0).astype(np.uint8)

            elif chosen_mode == "over_threshold":
                # Threshold too loose: background dirt becomes foreground.
                threshold = int(rng.integers(25, 75))

                if foreground > background:
                    failed_mask = np.where(gray > threshold, 255, 0).astype(np.uint8)
                else:
                    failed_mask = np.where(gray < 255 - threshold, 255, 0).astype(np.uint8)

            elif chosen_mode == "uneven":
                # Local uneven threshold: one side/zone is harsher than another.
                base_threshold = int(rng.integers(90, 145))
                gradient_strength = float(rng.uniform(25, 70))

                yy, xx = np.mgrid[0:height, 0:width]

                if int(rng.integers(0, 2)) == 0:
                    gradient = (xx / max(1, width - 1)) * gradient_strength
                else:
                    gradient = (yy / max(1, height - 1)) * gradient_strength

                threshold_map = base_threshold + gradient

                if foreground > background:
                    failed_mask = np.where(gray > threshold_map, 255, 0).astype(np.uint8)
                else:
                    failed_mask = np.where(gray < 255 - threshold_map, 255, 0).astype(np.uint8)

            else:
                return damaged, {
                    "operation": self.name,
                    "applied": False,
                    "reason": "invalid_mode",
                    "mode": self.mode,
                }

            candidate = np.full_like(image, background)
            candidate[failed_mask > 0] = foreground

            foreground_after = _foreground_mask(candidate)

            added_mask = cv2.bitwise_and(
                foreground_after,
                cv2.bitwise_not(foreground_before),
            )
            removed_mask = cv2.bitwise_and(
                foreground_before,
                cv2.bitwise_not(foreground_after),
            )

            added_pixels = int(cv2.countNonZero(added_mask))
            removed_pixels = int(cv2.countNonZero(removed_mask))
            changed_pixels = added_pixels + removed_pixels
            changed_ratio = changed_pixels / max(1, total_foreground)

            if changed_pixels < self.min_changed_pixels:
                continue

            if changed_ratio > self.max_changed_ratio:
                continue

            best_candidate = {
                "image": candidate,
                "attempt": attempt,
                "mode": chosen_mode,
                "added_foreground_pixels": added_pixels,
                "removed_foreground_pixels": removed_pixels,
                "changed_foreground_pixels": changed_pixels,
                "changed_foreground_ratio": changed_ratio,
            }
            break

        if best_candidate is None:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "could_not_make_valid_threshold_failure",
                "attempts": int(self.max_attempts),
                "total_foreground_pixels": int(total_foreground),
            }

        return best_candidate["image"], {
            "operation": self.name,
            "applied": True,
            "mode": best_candidate["mode"],
            "attempt": best_candidate["attempt"],
            "added_foreground_pixels": best_candidate["added_foreground_pixels"],
            "removed_foreground_pixels": best_candidate["removed_foreground_pixels"],
            "changed_foreground_pixels": best_candidate["changed_foreground_pixels"],
            "changed_foreground_ratio": best_candidate["changed_foreground_ratio"],
            "total_foreground_pixels": int(total_foreground),
            "background_value": int(background),
            "foreground_value": int(foreground),
        }
    
@dataclass
class CompressionArtifactDamage:
    """
    Simulate compression artifacts from low-quality scanned/saved crops.

    This creates JPEG/block-like degradation:
    - blocky gray noise
    - rough stroke edges
    - weak ringing around foreground
    """

    name: str = "compression_artifacts"
    jpeg_quality_min: int = 12
    jpeg_quality_max: int = 38
    downscale_min: float = 0.65
    downscale_max: float = 0.90
    min_changed_pixels: int = 5
    max_changed_ratio: float = 0.35
    max_attempts: int = 30

    def apply(self, image, rng):
        damaged = image.copy()
        background, foreground = _polarity_values(damaged)

        foreground_before = _foreground_mask(damaged)
        total_foreground = int(cv2.countNonZero(foreground_before))

        if total_foreground <= 0:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "no_foreground",
            }

        height, width = image.shape[:2]
        best_candidate = None

        for attempt in range(self.max_attempts):
            quality = int(rng.integers(
                self.jpeg_quality_min,
                self.jpeg_quality_max + 1,
            ))

            scale = float(rng.uniform(
                self.downscale_min,
                self.downscale_max,
            ))

            small_w = max(4, int(round(width * scale)))
            small_h = max(4, int(round(height * scale)))

            # Downscale then upscale to create block-ish edge loss.
            small = cv2.resize(
                damaged,
                (small_w, small_h),
                interpolation=cv2.INTER_AREA,
            )

            blocky = cv2.resize(
                small,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

            # JPEG encode/decode to create real compression ringing.
            ok, encoded = cv2.imencode(
                ".jpg",
                blocky,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )

            if not ok:
                continue

            decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)

            if decoded is None:
                continue

            candidate = decoded.astype(np.uint8)

            foreground_after = _foreground_mask(candidate)

            added_mask = cv2.bitwise_and(
                foreground_after,
                cv2.bitwise_not(foreground_before),
            )
            removed_mask = cv2.bitwise_and(
                foreground_before,
                cv2.bitwise_not(foreground_after),
            )

            added_pixels = int(cv2.countNonZero(added_mask))
            removed_pixels = int(cv2.countNonZero(removed_mask))
            changed_pixels = added_pixels + removed_pixels
            changed_ratio = changed_pixels / max(1, total_foreground)

            if changed_pixels < self.min_changed_pixels:
                continue

            if changed_ratio > self.max_changed_ratio:
                continue

            best_candidate = {
                "image": candidate,
                "attempt": attempt,
                "jpeg_quality": quality,
                "scale": scale,
                "small_width": small_w,
                "small_height": small_h,
                "added_foreground_pixels": added_pixels,
                "removed_foreground_pixels": removed_pixels,
                "changed_foreground_pixels": changed_pixels,
                "changed_foreground_ratio": changed_ratio,
            }
            break

        if best_candidate is None:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "could_not_make_valid_compression_artifacts",
                "attempts": int(self.max_attempts),
                "total_foreground_pixels": int(total_foreground),
            }

        return best_candidate["image"], {
            "operation": self.name,
            "applied": True,
            "mode": "jpeg_block_compression_artifacts",
            "attempt": best_candidate["attempt"],
            "jpeg_quality": best_candidate["jpeg_quality"],
            "downscale": best_candidate["scale"],
            "small_width": best_candidate["small_width"],
            "small_height": best_candidate["small_height"],
            "added_foreground_pixels": best_candidate["added_foreground_pixels"],
            "removed_foreground_pixels": best_candidate["removed_foreground_pixels"],
            "changed_foreground_pixels": best_candidate["changed_foreground_pixels"],
            "changed_foreground_ratio": best_candidate["changed_foreground_ratio"],
            "total_foreground_pixels": int(total_foreground),
            "background_value": int(background),
            "foreground_value": int(foreground),
        }


@dataclass
class InkOverlapDamage:
    """
    Simulate overlapping handwriting / accidental extra ink.

    This adds a sharp foreground-colored stroke-like artifact across or near
    the glyph. It should look like another handwritten stroke entered the crop.
    """

    name: str = "ink_overlap"
    opacity: float = 0.90
    thickness: int = 1
    min_added_pixels: int = 5
    max_added_ratio: float = 0.30
    max_attempts: int = 30
    stroke_count: int = 1

    def apply(self, image, rng):
        damaged = image.copy()
        background, foreground = _polarity_values(damaged)

        foreground_before = _foreground_mask(damaged)
        total_foreground = int(cv2.countNonZero(foreground_before))

        if total_foreground <= 0:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "no_foreground",
            }

        points = cv2.findNonZero(foreground_before)

        if points is None:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "no_foreground_points",
            }

        height, width = image.shape[:2]
        x, y, bbox_w, bbox_h = cv2.boundingRect(points)

        best_candidate = None

        for attempt in range(self.max_attempts):
            overlap_mask = np.zeros((height, width), dtype=np.uint8)

            for _ in range(max(1, int(self.stroke_count))):
                # Pick one of four crossing directions.
                direction = int(rng.integers(0, 4))

                if direction == 0:
                    # left -> right
                    p1 = (
                        int(np.clip(x - rng.integers(0, max(1, bbox_w // 3 + 1)), 0, width - 1)),
                        int(rng.integers(y, y + bbox_h)),
                    )
                    p3 = (
                        int(np.clip(x + bbox_w + rng.integers(0, max(1, bbox_w // 3 + 1)), 0, width - 1)),
                        int(rng.integers(y, y + bbox_h)),
                    )

                elif direction == 1:
                    # top -> bottom
                    p1 = (
                        int(rng.integers(x, x + bbox_w)),
                        int(np.clip(y - rng.integers(0, max(1, bbox_h // 3 + 1)), 0, height - 1)),
                    )
                    p3 = (
                        int(rng.integers(x, x + bbox_w)),
                        int(np.clip(y + bbox_h + rng.integers(0, max(1, bbox_h // 3 + 1)), 0, height - 1)),
                    )

                elif direction == 2:
                    # diagonal top-left -> bottom-right
                    p1 = (
                        int(np.clip(x - rng.integers(0, max(1, bbox_w // 4 + 1)), 0, width - 1)),
                        int(np.clip(y - rng.integers(0, max(1, bbox_h // 4 + 1)), 0, height - 1)),
                    )
                    p3 = (
                        int(np.clip(x + bbox_w + rng.integers(0, max(1, bbox_w // 4 + 1)), 0, width - 1)),
                        int(np.clip(y + bbox_h + rng.integers(0, max(1, bbox_h // 4 + 1)), 0, height - 1)),
                    )

                else:
                    # diagonal bottom-left -> top-right
                    p1 = (
                        int(np.clip(x - rng.integers(0, max(1, bbox_w // 4 + 1)), 0, width - 1)),
                        int(np.clip(y + bbox_h + rng.integers(0, max(1, bbox_h // 4 + 1)), 0, height - 1)),
                    )
                    p3 = (
                        int(np.clip(x + bbox_w + rng.integers(0, max(1, bbox_w // 4 + 1)), 0, width - 1)),
                        int(np.clip(y - rng.integers(0, max(1, bbox_h // 4 + 1)), 0, height - 1)),
                    )

                # Middle control point makes it look more handwritten/curved.
                mid_x = int((p1[0] + p3[0]) / 2 + rng.integers(-max(1, bbox_w // 4), max(2, bbox_w // 4 + 1)))
                mid_y = int((p1[1] + p3[1]) / 2 + rng.integers(-max(1, bbox_h // 4), max(2, bbox_h // 4 + 1)))

                p2 = (
                    int(np.clip(mid_x, 0, width - 1)),
                    int(np.clip(mid_y, 0, height - 1)),
                )

                # Approximate quadratic curve with polyline points.
                curve_points = []
                for t in np.linspace(0.0, 1.0, 16):
                    one_minus = 1.0 - t
                    px = (
                        one_minus * one_minus * p1[0]
                        + 2 * one_minus * t * p2[0]
                        + t * t * p3[0]
                    )
                    py = (
                        one_minus * one_minus * p1[1]
                        + 2 * one_minus * t * p2[1]
                        + t * t * p3[1]
                    )
                    curve_points.append([int(round(px)), int(round(py))])

                curve_points = np.array(curve_points, dtype=np.int32).reshape((-1, 1, 2))

                cv2.polylines(
                    overlap_mask,
                    [curve_points],
                    isClosed=False,
                    color=255,
                    thickness=int(self.thickness),
                    lineType=cv2.LINE_AA,
                )

            before_mask = _foreground_mask(damaged)

            candidate = damaged.copy().astype(np.float32)

            overlap_pixels = overlap_mask > 0
            candidate[overlap_pixels] = (
                candidate[overlap_pixels] * (1.0 - self.opacity)
                + float(foreground) * self.opacity
            )

            candidate = np.clip(candidate, 0, 255).astype(np.uint8)

            after_mask = _foreground_mask(candidate)

            added_mask = cv2.bitwise_and(
                after_mask,
                cv2.bitwise_not(before_mask),
            )

            added_pixels = int(cv2.countNonZero(added_mask))
            added_ratio = added_pixels / max(1, total_foreground)

            if added_pixels < self.min_added_pixels:
                continue

            if added_ratio > self.max_added_ratio:
                continue

            best_candidate = {
                "image": candidate,
                "attempt": attempt,
                "added_foreground_pixels": added_pixels,
                "added_foreground_ratio": added_ratio,
                "overlap_mask_pixels": int(cv2.countNonZero(overlap_mask)),
            }
            break

        if best_candidate is None:
            return damaged, {
                "operation": self.name,
                "applied": False,
                "reason": "could_not_make_valid_ink_overlap",
                "attempts": int(self.max_attempts),
                "total_foreground_pixels": int(total_foreground),
            }

        return best_candidate["image"], {
            "operation": self.name,
            "applied": True,
            "mode": "foreign_handwriting_stroke_overlap",
            "attempt": best_candidate["attempt"],
            "opacity": float(self.opacity),
            "thickness": int(self.thickness),
            "stroke_count": int(self.stroke_count),
            "overlap_mask_pixels": best_candidate["overlap_mask_pixels"],
            "added_foreground_pixels": best_candidate["added_foreground_pixels"],
            "added_foreground_ratio": best_candidate["added_foreground_ratio"],
            "total_foreground_pixels": int(total_foreground),
            "background_value": int(background),
            "foreground_value": int(foreground),
        }