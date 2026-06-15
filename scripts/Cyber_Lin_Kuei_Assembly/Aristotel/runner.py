import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from .recipes import DamageRecipe
from .router import DatasetRouter
from .teacher_models import DamagedSample, OutputFolders, TeacherInput


class FileCorrupter:
    """Create deterministic degradations without retaining them on disk."""

    def __init__(self, recipes: list[DamageRecipe], seed: int = 42):
        self.recipes = recipes
        self.seed = int(seed)
        self.recipes_by_name = {recipe.name: recipe for recipe in recipes}
        if len(self.recipes_by_name) != len(recipes):
            raise ValueError("Damage recipe names must be unique.")

    def load_image(self, image_path: Path):
        """Load one source glyph as grayscale."""
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return image

    def _sample_identity(
        self,
        teacher_input: TeacherInput,
        recipe: DamageRecipe,
        epoch: int,
        variant: int,
    ) -> tuple[str, int]:
        """Derive a stable sample ID and NumPy seed from sample coordinates."""
        source_id = str(
            teacher_input.metadata.get(
                "source_id",
                f"{teacher_input.source_class}/{teacher_input.image_path.name}",
            )
        )
        identity = (
            f"{self.seed}|{source_id}|{recipe.signature()}|"
            f"{int(epoch)}|{int(variant)}"
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return digest[:20], int(digest[:16], 16)

    def corrupt(
        self,
        teacher_input: TeacherInput,
        epoch: int = 0,
        variant: int = 0,
        recipe_names: list[str] | None = None,
    ) -> list[DamagedSample]:
        """Generate selected recipe variants deterministically in memory."""
        image = self.load_image(teacher_input.image_path)
        samples = []
        recipes = self.recipes
        if recipe_names is not None:
            missing = sorted(set(recipe_names) - set(self.recipes_by_name))
            if missing:
                raise ValueError(f"Unknown damage recipes: {', '.join(missing)}")
            recipes = [self.recipes_by_name[name] for name in recipe_names]

        for recipe in recipes:
            sample_id, sample_seed = self._sample_identity(
                teacher_input,
                recipe,
                epoch,
                variant,
            )
            rng = np.random.default_rng(sample_seed)
            damaged_image, operations = recipe.apply(image, rng)
            changed_pixel_count = int(
                np.count_nonzero(damaged_image != image)
            )
            changed_pixel_ratio = float(
                changed_pixel_count / max(1, image.size)
            )
            sample = DamagedSample(
                sample_id=sample_id,
                seed=sample_seed,
                epoch=int(epoch),
                variant=int(variant),
                teacher_input=teacher_input,
                image=damaged_image,
                damage_recipe=recipe.name,
                recipe_signature=recipe.signature(),
                operations=operations,
                changed_pixel_count=changed_pixel_count,
                changed_pixel_ratio=changed_pixel_ratio,
                severity=recipe.severity,
                trust_label=recipe.trust_label,
            )
            samples.append(sample)

        return samples


class AristotelRunner:
    """Scan source glyphs and expose disk-free or materialized run modes."""

    MODES = {"stream", "manifest", "preview", "export"}

    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        corrupter: FileCorrupter,
        image_extensions=(".png", ".jpg", ".jpeg", ".webp"),
    ):
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.output_folders = None
        self.corrupter = corrupter
        self.router = None
        self.image_extensions = image_extensions

    def _ensure_output(self):
        """Create output folders only when a mode actually writes files."""
        if self.output_folders is None:
            self.output_folders = OutputFolders.create(self.output_root)
            self.router = DatasetRouter(self.output_folders)
        return self.output_folders

    def scan_inputs(self):
        """Return source glyph records in deterministic path order."""
        if not self.input_root.exists():
            raise FileNotFoundError(f"Input root does not exist: {self.input_root}")
        inputs = []

        for class_folder in sorted(self.input_root.iterdir()):
            if not class_folder.is_dir():
                continue

            label = class_folder.name

            for image_path in sorted(class_folder.iterdir()):
                if image_path.suffix.lower() not in self.image_extensions:
                    continue

                inputs.append(
                    TeacherInput(
                        image_path=image_path,
                        label=label,
                        source_class=label,
                        source_folder=class_folder,
                        metadata={
                            "source_id": image_path.relative_to(
                                self.input_root
                            ).as_posix()
                        },
                    )
                )

        return inputs

    def save_sample(self, sample: DamagedSample):
        """Materialize one sample image and its full metadata."""
        self._ensure_output()
        image_path = self.router.image_output_path(sample)
        metadata_path = self.router.metadata_output_path(sample)

        if not cv2.imwrite(str(image_path), sample.image):
            raise RuntimeError(f"Could not save image: {image_path}")

        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(sample.to_metadata(image_path), file, indent=2, ensure_ascii=False)
            file.write("\n")

    def iter_samples(
        self,
        limit: int | None = None,
        epoch: int = 0,
        variants: int = 1,
        recipe_names: list[str] | None = None,
    ) -> Iterator[DamagedSample]:
        """Yield degraded images in RAM for direct consumption by training."""
        if variants < 1:
            raise ValueError("variants must be at least 1.")
        inputs = self.scan_inputs()
        if limit is not None:
            inputs = inputs[:limit]

        for teacher_input in inputs:
            for variant in range(variants):
                yield from self.corrupter.corrupt(
                    teacher_input,
                    epoch=epoch,
                    variant=variant,
                    recipe_names=recipe_names,
                )

    def regenerate(self, metadata: dict[str, Any]) -> DamagedSample:
        """Recreate one sample exactly from a manifest or metadata record."""
        teacher_input = TeacherInput.from_dict(metadata["original"])
        if not teacher_input.image_path.exists():
            source_id = teacher_input.metadata.get("source_id")
            if source_id:
                teacher_input.image_path = self.input_root / str(source_id)
        recipe_name = str(metadata["damage_recipe"])
        expected_signature = metadata.get("recipe_signature")
        active_signature = self.corrupter.recipes_by_name[recipe_name].signature()
        if (
            expected_signature is not None
            and expected_signature != active_signature
        ):
            raise ValueError(
                f"Recipe {recipe_name!r} changed since this manifest was created."
            )
        samples = self.corrupter.corrupt(
            teacher_input,
            epoch=int(metadata.get("epoch", 0)),
            variant=int(metadata.get("variant", 0)),
            recipe_names=[recipe_name],
        )
        sample = samples[0]
        expected_id = metadata.get("sample_id")
        if expected_id is not None and sample.sample_id != expected_id:
            raise ValueError(
                "Manifest sample ID does not match regenerated sample identity."
            )
        return sample

    def _write_manifest_record(self, file, sample: DamagedSample):
        """Append one compact reproducibility record to a JSONL manifest."""
        record = sample.to_metadata()
        record.pop("operations", None)
        file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        file.write("\n")

    def run(
        self,
        mode: str = "stream",
        limit: int | None = None,
        epoch: int = 0,
        variants: int = 1,
        recipe_names: list[str] | None = None,
        preview_count: int = 20,
    ) -> dict[str, Any]:
        """Execute a storage policy while preserving deterministic samples."""
        if mode not in self.MODES:
            raise ValueError(
                f"Unsupported mode {mode!r}; choose from {sorted(self.MODES)}."
            )
        if preview_count < 1:
            raise ValueError("preview_count must be at least 1.")

        produced = 0
        failed = []
        saved_images = 0
        manifest_path = None
        manifest_file = None

        if mode == "manifest":
            folders = self._ensure_output()
            manifest_path = folders.root / "manifest.jsonl"
            manifest_file = manifest_path.open("w", encoding="utf-8")

        try:
            for sample in self.iter_samples(
                limit=limit,
                epoch=epoch,
                variants=variants,
                recipe_names=recipe_names,
            ):
                try:
                    if mode == "export":
                        self.save_sample(sample)
                        saved_images += 1
                    elif mode == "preview" and saved_images < preview_count:
                        folders = self._ensure_output()
                        preview_path = (
                            folders.previews
                            / f"{sample.teacher_input.source_class}_"
                            f"{sample.damage_recipe}_{sample.sample_id}.png"
                        )
                        if not cv2.imwrite(str(preview_path), sample.image):
                            raise RuntimeError(
                                f"Could not save preview: {preview_path}"
                            )
                        saved_images += 1
                    elif mode == "manifest":
                        self._write_manifest_record(manifest_file, sample)

                    # In stream/manifest/preview modes the array becomes
                    # collectible as soon as the consumer advances.
                    produced += 1
                except Exception as error:
                    failed.append(
                        {
                            "sample_id": sample.sample_id,
                            "image_path": str(sample.teacher_input.image_path),
                            "label": sample.teacher_input.label,
                            "error": str(error),
                        }
                    )
        finally:
            if manifest_file is not None:
                manifest_file.close()

        return {
            "mode": mode,
            "input_limit": limit,
            "epoch": epoch,
            "variants": variants,
            "produced_count": produced,
            "saved_image_count": saved_images,
            "failed_count": len(failed),
            "failed": failed,
            "output_root": (
                str(self.output_folders.root)
                if self.output_folders is not None
                else None
            ),
            "manifest_path": (
                str(manifest_path) if manifest_path is not None else None
            ),
        }
