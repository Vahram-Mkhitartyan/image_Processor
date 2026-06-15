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
            operations=[CutLineDamage(thickness=1)],
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
            operations=[ErosionDamage(kernel_size=2, iterations=1)],
            severity=0.35,
            trust_label="repair_needed",
        ),
    ]
