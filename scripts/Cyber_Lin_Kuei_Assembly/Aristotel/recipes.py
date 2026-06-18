import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any
import numpy as np

from .damage_operations import (
    BlackNoiseDamage,
    BlurDamage,
    CutLineDamage,
    DamageOperation,
    ErosionDamage,
    StampInterferenceDamage,
    BleedThroughDamage,
    EdgeCropLossDamage,
    ThresholdFailureDamage,
    CompressionArtifactDamage,
    InkOverlapDamage,
)

@dataclass
class DamageRecipe:
    name: str
    operations: list[DamageOperation]
    severity: float
    trust_label: str

    def apply(self, image, rng: np.random.Generator):
        damaged = image.copy()
        metadata: list[dict[str, Any]] = []

        for operation in self.operations:
            damaged, operation_metadata = operation.apply(damaged, rng)
            metadata.append(operation_metadata)

        return damaged, metadata

    def definition(self):
        """Return a stable JSON-safe description of this recipe."""
        operations = []
        for operation in self.operations:
            values = asdict(operation) if is_dataclass(operation) else {}
            operations.append(
                {
                    "type": type(operation).__name__,
                    "settings": values,
                }
            )
        return {
            "name": self.name,
            "operations": operations,
            "severity": self.severity,
            "trust_label": self.trust_label,
        }

    def signature(self):
        """Return the canonical recipe text used in sample identities."""
        return json.dumps(
            self.definition(),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )


def build_default_recipes():
    return [
        DamageRecipe(
            name="light_cut",
            operations=[CutLineDamage(thickness=2)],
            severity=0.25,
            trust_label="repair_needed",
        ),
        DamageRecipe(
            name="light_blur",
            operations=[BlurDamage(kernel_size=3)],
            severity=0.20,
            trust_label="uncertain",
        ),
        DamageRecipe(
            name="scanner_noise",
            operations=[BlackNoiseDamage(probability=0.01)],
            severity=0.30,
            trust_label="uncertain",
        ),
        DamageRecipe(
            name="light_erosion",
            operations=[
                ErosionDamage(
                    boundary_remove_probability=0.12,
                    kernel_size=2,
                )
            ],
            severity=0.25,
            trust_label="repair_needed",
        ),
        DamageRecipe(
            name="stamp_interference",
            operations=[
                StampInterferenceDamage(
                    opacity=0.55,
                    ring_thickness=1,
                    internal_line_count=3,
                    max_added_ratio=0.35,
                )
            ],
            severity=0.45,
            trust_label="uncertain",
        ),
        DamageRecipe(
            name="bleed_through",
            operations=[
                BleedThroughDamage(
                    opacity=0.28,
                    blur_kernel_size=5,
                    min_shift_px=3,
                    max_shift_px=8,
                    max_added_ratio=0.30,
                )
            ],
            severity=0.35,
            trust_label="uncertain",
        ),

        DamageRecipe(
            name="edge_crop_loss",
            operations=[
                EdgeCropLossDamage(
                    min_crop_ratio=0.08,
                    max_crop_ratio=0.20,
                    max_removed_ratio=0.30,
                    side="random",
                )
            ],
            severity=0.35,
            trust_label="repair_needed",
        ),
        DamageRecipe(
            name="threshold_failure",
            operations=[
                ThresholdFailureDamage(
                    mode="random",
                    min_changed_pixels=5,
                    max_changed_ratio=0.30,
                )
            ],
            severity=0.35,
            trust_label="repair_needed",
        ),
        DamageRecipe(
            name="compression_artifacts",
            operations=[
                CompressionArtifactDamage(
                    jpeg_quality_min=6,
                    jpeg_quality_max=25,
                    downscale_min=0.50,
                    downscale_max=0.90,
                    max_changed_ratio=0.60,
                )
            ],
            severity=0.25,
            trust_label="uncertain",
        ),
        DamageRecipe(
            name="ink_overlap",
            operations=[
                InkOverlapDamage(
                    opacity=0.90,
                    thickness=1,
                    stroke_count=1,
                    max_abs_angle_degrees=35,
                    max_added_ratio=0.40,
                )
            ],
            severity=0.40,
            trust_label="uncertain",
        )
        
    ]
