from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np


@dataclass
class TeacherInput:
    image_path: Path
    label: str
    source_class: str
    source_folder: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data):
        return cls(
            image_path=Path(data["image_path"]),
            label=str(data["label"]),
            source_class=str(data["source_class"]),
            source_folder=Path(data["source_folder"]),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self):
        return {
            "image_path": str(self.image_path),
            "label": self.label,
            "source_class": self.source_class,
            "source_folder": str(self.source_folder),
            "metadata": self.metadata,
        }


@dataclass
class DamagedSample:
    """Hold one reproducible in-memory degradation of a source glyph."""

    sample_id: str
    seed: int
    epoch: int
    variant: int
    teacher_input: TeacherInput
    image: np.ndarray
    damage_recipe: str
    recipe_signature: str
    operations: list[dict[str, Any]]
    changed_pixel_count: int
    changed_pixel_ratio: float
    severity: float
    trust_label: str

    def to_metadata(self, output_image_path: Path | None = None):
        """Return JSON-safe provenance; the image path is optional."""
        metadata = {
            "sample_id": self.sample_id,
            "seed": self.seed,
            "epoch": self.epoch,
            "variant": self.variant,
            "original": self.teacher_input.to_dict(),
            "damage_recipe": self.damage_recipe,
            "recipe_signature": self.recipe_signature,
            "operations": self.operations,
            "changed_pixel_count": self.changed_pixel_count,
            "changed_pixel_ratio": self.changed_pixel_ratio,
            "severity": self.severity,
            "trust_label": self.trust_label,
        }
        metadata["output_image_path"] = (
            str(output_image_path) if output_image_path is not None else None
        )
        return metadata


@dataclass
class OutputFolders:
    root: Path
    images: Path
    metadata: Path
    previews: Path

    @classmethod
    def create(cls, root: Path):
        folders = cls(
            root=root,
            images=root / "images",
            metadata=root / "metadata",
            previews=root / "previews",
        )
        for folder in (folders.root, folders.images, folders.metadata, folders.previews):
            folder.mkdir(parents=True, exist_ok=True)
        return folders
