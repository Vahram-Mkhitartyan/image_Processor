"""Conservative mask repair before ScribeTrace skeletonization."""

import cv2
import numpy as np

from .trace_settings import normalize_trace_settings


class TraceMaskRepairer:
    """Repair tiny mask damage without deleting original ink."""

    def __init__(self, settings=None):
        self.settings = normalize_trace_settings(settings)

    def repair(self, mask):
        original = (np.asarray(mask) > 128).astype(np.uint8) * 255

        if not getattr(self.settings, "enable_mask_repair", True):
            return original, {
                "enabled": False,
                "added_pixels": 0,
                "original_ink_pixels": int(cv2.countNonZero(original)),
                "repaired_ink_pixels": int(cv2.countNonZero(original)),
            }

        kernel = np.ones((3, 3), dtype=np.uint8)

        closed = cv2.morphologyEx(original, cv2.MORPH_CLOSE, kernel, iterations=1)

        horizontal_kernel = np.array([[1, 1, 1]], dtype=np.uint8)
        vertical_kernel = np.array([[1], [1], [1]], dtype=np.uint8)

        horizontal_bridge = cv2.morphologyEx(
            original,
            cv2.MORPH_CLOSE,
            horizontal_kernel,
            iterations=1,
        )
        vertical_bridge = cv2.morphologyEx(
            original,
            cv2.MORPH_CLOSE,
            vertical_kernel,
            iterations=1,
        )

        repaired = cv2.bitwise_or(original, closed)
        repaired = cv2.bitwise_or(repaired, horizontal_bridge)
        repaired = cv2.bitwise_or(repaired, vertical_bridge)

        original_count = int(cv2.countNonZero(original))
        repaired_count = int(cv2.countNonZero(repaired))

        return repaired, {
            "enabled": True,
            "method": "conservative_close_or_original",
            "original_ink_pixels": original_count,
            "repaired_ink_pixels": repaired_count,
            "added_pixels": repaired_count - original_count,
        }