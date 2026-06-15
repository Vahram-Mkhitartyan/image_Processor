from pathlib import Path

from .teacher_models import DamagedSample, OutputFolders


class DatasetRouter:
    def __init__(self, output_folders: OutputFolders):
        self.output_folders = output_folders

    def image_output_path(self, sample: DamagedSample) -> Path:
        class_dir = self.output_folders.images / sample.teacher_input.source_class
        class_dir.mkdir(parents=True, exist_ok=True)
        return class_dir / f"{sample.sample_id}.png"

    def metadata_output_path(self, sample: DamagedSample) -> Path:
        class_dir = self.output_folders.metadata / sample.teacher_input.source_class
        class_dir.mkdir(parents=True, exist_ok=True)
        return class_dir / f"{sample.sample_id}.json"
