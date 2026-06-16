"""Serve the dependency-free local pipeline control room."""

from __future__ import annotations

import json
import mimetypes
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

RUNNABLE_COMMANDS = {
    "pipeline",
    "prep",
    "scribemap",
    "refine",
    "visual",
    "printed_ocr",
    "handwritten_ocr",
    "train",
    "doctor",
    "status",
    "counts",
    "lines",
    "setup",
    "clean",
}

STAGES = (
    {
        "id": "input",
        "label": "SOURCE",
        "title": "Input document",
        "folder": "",
    },
    {
        "id": "n00",
        "label": "N00",
        "title": "File preparation",
        "folder": "n00_file_preparation",
    },
    {
        "id": "n01",
        "label": "N01",
        "title": "ScribeMap",
        "folder": "n01_scribemap",
    },
    {
        "id": "n02",
        "label": "N02",
        "title": "Crop refiner",
        "folder": "n02_crop_refiner",
    },
    {
        "id": "n03",
        "label": "N03",
        "title": "Visual router",
        "folder": "n03_visual_classification",
    },
    {
        "id": "n04",
        "label": "N04",
        "title": "Printed OCR",
        "folder": "n04_printed_ocr",
    },
    {
        "id": "n05",
        "label": "N05",
        "title": "Handwriting experts",
        "folder": "n05_handwritten_ocr",
    },
)

STAGE_FOLDER_LOOKUP = {
    stage["folder"]: stage["id"]
    for stage in STAGES
    if stage["folder"]
}

ARISTOTEL_RECONSTRUCTION_CYCLE = (
    "diagnose",
    "hypothesize",
    "reconstruct",
    "retrace",
    "verify",
    "accept_or_reject",
)

NON_DOCUMENT_TEMP_FOLDERS = {
    "aristotel_ui_preview",
    "matenadata_4_0",
}


def _iso_timestamp(timestamp: float | None = None) -> str:
    """Return a local ISO timestamp suitable for JSON status records."""
    value = timestamp if timestamp is not None else time.time()
    return datetime.fromtimestamp(value).astimezone().isoformat(timespec="seconds")


class PipelineProcessManager:
    """Run one main.py command at a time and retain bounded live logs."""

    MAX_LOG_LINES = 12000

    def __init__(
        self,
        base_dir: Path,
        controller_script: Path,
        python_executable: Path,
    ) -> None:
        self.base_dir = base_dir
        self.controller_script = controller_script
        self.python_executable = python_executable
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._logs: list[str] = []
        self._log_base_offset = 0
        self._state = {
            "status": "idle",
            "command": None,
            "return_code": None,
            "started_at": None,
            "finished_at": None,
            "pid": None,
        }

    def _append_log(self, line: str) -> None:
        """Append one console line while preventing unbounded memory growth."""
        with self._lock:
            self._logs.append(line.rstrip("\n"))
            overflow = len(self._logs) - self.MAX_LOG_LINES
            if overflow > 0:
                del self._logs[:overflow]
                self._log_base_offset += overflow

    def start(self, command: str) -> dict:
        """Launch one allow-listed controller command in the project venv."""
        if command not in RUNNABLE_COMMANDS:
            raise ValueError(f"Unsupported UI command: {command}")

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError(
                    f"{self._state['command']} is already running."
                )

            self._logs = []
            self._log_base_offset = 0
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"

            self._process = subprocess.Popen(
                [
                    str(self.python_executable),
                    "-u",
                    str(self.controller_script),
                    command,
                ],
                cwd=str(self.base_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
                start_new_session=True,
            )
            self._state = {
                "status": "running",
                "command": command,
                "return_code": None,
                "started_at": _iso_timestamp(),
                "finished_at": None,
                "pid": self._process.pid,
            }
            self._append_log(
                f"$ {self.python_executable} -u "
                f"{self.controller_script} {command}"
            )

            self._reader_thread = threading.Thread(
                target=self._consume_output,
                name=f"pipeline-ui-{command}",
                daemon=True,
            )
            self._reader_thread.start()
            return dict(self._state)

    def _consume_output(self) -> None:
        """Read subprocess output until completion and finalize its state."""
        process = self._process
        if process is None:
            return

        if process.stdout is not None:
            for line in iter(process.stdout.readline, ""):
                self._append_log(line)
            process.stdout.close()

        return_code = process.wait()
        with self._lock:
            if self._state["status"] == "stopping":
                status = "stopped"
            else:
                status = "completed" if return_code == 0 else "failed"

            self._state.update(
                {
                    "status": status,
                    "return_code": return_code,
                    "finished_at": _iso_timestamp(),
                    "pid": None,
                }
            )
            self._append_log(
                f"[control-room] {self._state['command']} finished "
                f"with exit code {return_code}."
            )

    def stop(self) -> dict:
        """Terminate the active pipeline subprocess without stopping the UI."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return dict(self._state)
            self._state["status"] = "stopping"
            process = self._process
            self._append_log("[control-room] Stop requested.")

        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
        return self.snapshot()

    def snapshot(self) -> dict:
        """Return the current JSON-safe process state."""
        with self._lock:
            return dict(self._state)

    def logs_since(self, offset: int) -> dict:
        """Return console lines from an absolute client log offset."""
        with self._lock:
            safe_offset = max(offset, self._log_base_offset)
            relative_offset = safe_offset - self._log_base_offset
            lines = self._logs[relative_offset:]
            next_offset = self._log_base_offset + len(self._logs)
            return {
                "lines": list(lines),
                "next_offset": next_offset,
                "truncated": offset < self._log_base_offset,
                "process": dict(self._state),
            }


class PipelineUiApplication:
    """Collect pipeline state and expose safe local UI operations."""

    def __init__(
        self,
        base_dir: Path,
        controller_script: Path,
        python_executable: Path,
    ) -> None:
        self.base_dir = base_dir.resolve()
        self.scripts_dir = self.base_dir / "scripts"
        self.input_dir = self.base_dir / "handwritten_text"
        self.temp_dir = self.base_dir / "temp_processing"
        self.reports_dir = self.base_dir / "reports"
        self.models_dir = self.base_dir / "models"
        self.aristotel_input_dir = self.base_dir / "Matenadata"
        self.aristotel_preview_dir = self.temp_dir / "aristotel_ui_preview"
        self.static_dir = Path(__file__).resolve().parent / "static"
        self.process_manager = PipelineProcessManager(
            base_dir=self.base_dir,
            controller_script=controller_script.resolve(),
            # Keep the venv launcher path instead of resolving its symlink to
            # the system interpreter, otherwise subprocesses lose venv context.
            python_executable=python_executable.absolute(),
        )

    def _document_sources(self) -> dict[str, Path]:
        """Return supported input documents keyed by filename stem."""
        documents: dict[str, Path] = {}
        if not self.input_dir.is_dir():
            return documents

        for path in sorted(self.input_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                documents[path.stem] = path.resolve()
        return documents

    def _stage_complete(self, document_id: str, stage_id: str) -> bool:
        """Return whether the expected stage metadata exists."""
        document_dir = self.temp_dir / document_id
        patterns = {
            "input": ("input_document.*",),
            "n00": ("n00_file_preparation/metadata/metadata.json",),
            "n01": (
                "n01_scribemap/metadata/*_classified_groups.json",
            ),
            "n02": ("n02_crop_refiner/metadata/*_refined_groups.json",),
            "n03": (
                "n03_visual_classification/metadata/"
                "*_n03_visual_classification_routes.json",
            ),
            "n04": ("n04_printed_ocr/metadata/*_printed_text_map.json",),
            "n05": (
                "n05_handwritten_ocr/metadata/"
                "*_handwritten_text_map.json",
            ),
        }
        return any(document_dir.glob(pattern) for pattern in patterns[stage_id])

    def overview(self) -> dict:
        """Build the dashboard overview for documents and stage progress."""
        sources = self._document_sources()
        temp_document_ids = {
            path.name
            for path in self.temp_dir.iterdir()
            if path.is_dir()
            and path.name.lower() not in NON_DOCUMENT_TEMP_FOLDERS
            and not path.name.lower().startswith("aristotel_")
            and not path.name.lower().startswith("matenadata_")
        } if self.temp_dir.is_dir() else set()
        document_ids = sorted(set(sources) | temp_document_ids)
        documents = []

        for document_id in document_ids:
            source = sources.get(document_id)
            stages = {
                stage["id"]: self._stage_complete(document_id, stage["id"])
                for stage in STAGES
            }
            completed_count = sum(stages.values())
            documents.append(
                {
                    "id": document_id,
                    "source_path": (
                        str(source.relative_to(self.base_dir))
                        if source
                        else None
                    ),
                    "stages": stages,
                    "completed_count": completed_count,
                    "total_count": len(STAGES),
                }
            )

        return {
            "project_name": "Armenian OCR Pipeline",
            "base_dir": str(self.base_dir),
            "documents": documents,
            "stages": list(STAGES),
            "commands": sorted(RUNNABLE_COMMANDS),
            "process": self.process_manager.snapshot(),
            "updated_at": _iso_timestamp(),
        }

    def _stage_for_path(self, document_dir: Path, path: Path) -> str:
        """Map one per-document artifact path to its pipeline stage."""
        relative_parts = path.relative_to(document_dir).parts
        if not relative_parts or relative_parts[0].startswith("input_document"):
            return "input"
        return STAGE_FOLDER_LOOKUP.get(relative_parts[0], "input")

    @staticmethod
    def _artifact_kind(relative_path: str) -> str:
        """Classify an image for visual filtering and labels."""
        lowered = relative_path.lower()
        name = Path(relative_path).name.lower()
        if "input_document" in lowered:
            return "source"
        if "/debug/" in lowered or "preview" in name:
            return "debug"
        if "/masks/" in lowered or "_mask" in name:
            return "mask"
        if "/full_images/" in lowered:
            return "transform"
        if "/classified/" in lowered:
            return "classification"
        if "/segments/" in lowered:
            return "segment"
        if "/groups/" in lowered or "/crops/" in lowered:
            return "crop"
        return "image"

    @staticmethod
    def _artifact_priority(kind: str) -> int:
        """Prefer overview and debug evidence before high-volume crops."""
        return {
            "source": 0,
            "transform": 10,
            "debug": 20,
            "mask": 30,
            "classification": 40,
            "image": 50,
            "crop": 60,
            "segment": 70,
        }.get(kind, 80)

    def artifacts(
        self,
        document_id: str,
        stage_id: str = "all",
        query: str = "",
        limit: int = 240,
    ) -> dict:
        """Return a bounded, deterministic artifact catalog."""
        if not document_id or Path(document_id).name != document_id:
            raise ValueError("Invalid document id.")
        valid_stages = {"all"} | {stage["id"] for stage in STAGES}
        if stage_id not in valid_stages:
            raise ValueError("Invalid stage id.")

        document_dir = (self.temp_dir / document_id).resolve()
        if not document_dir.is_relative_to(self.temp_dir.resolve()):
            raise ValueError("Document path escaped temp_processing.")

        candidates: list[dict] = []
        if document_dir.is_dir():
            for path in document_dir.rglob("*"):
                if (
                    not path.is_file()
                    or path.suffix.lower() not in IMAGE_EXTENSIONS
                ):
                    continue
                artifact_stage = self._stage_for_path(document_dir, path)
                if stage_id != "all" and artifact_stage != stage_id:
                    continue

                relative_document_path = path.relative_to(document_dir).as_posix()
                if query and query.lower() not in relative_document_path.lower():
                    continue

                project_relative_path = path.relative_to(self.base_dir).as_posix()
                kind = self._artifact_kind(f"/{relative_document_path}")
                stat = path.stat()
                candidates.append(
                    {
                        "name": path.name,
                        "stage": artifact_stage,
                        "kind": kind,
                        "relative_path": project_relative_path,
                        "document_path": relative_document_path,
                        "url": (
                            "/artifact?path="
                            + quote(project_relative_path, safe="")
                        ),
                        "size_bytes": stat.st_size,
                        "modified_at": _iso_timestamp(stat.st_mtime),
                        "_priority": self._artifact_priority(kind),
                    }
                )

        # Include the source file before a pipeline run has created its copy.
        source = self._document_sources().get(document_id)
        if (
            source
            and stage_id in {"all", "input"}
            and (not query or query.lower() in source.name.lower())
        ):
            relative_path = source.relative_to(self.base_dir).as_posix()
            stat = source.stat()
            candidates.append(
                {
                    "name": source.name,
                    "stage": "input",
                    "kind": "source",
                    "relative_path": relative_path,
                    "document_path": source.name,
                    "url": "/artifact?path=" + quote(relative_path, safe=""),
                    "size_bytes": stat.st_size,
                    "modified_at": _iso_timestamp(stat.st_mtime),
                    "_priority": -1,
                }
            )

        candidates.sort(
            key=lambda item: (
                item["_priority"],
                item["stage"],
                item["document_path"],
            )
        )
        total_count = len(candidates)
        safe_limit = max(1, min(int(limit), 500))
        visible = candidates[:safe_limit]
        for item in visible:
            item.pop("_priority", None)

        counts: dict[str, int] = {}
        for item in candidates:
            counts[item["stage"]] = counts.get(item["stage"], 0) + 1

        return {
            "document_id": document_id,
            "stage": stage_id,
            "query": query,
            "artifacts": visible,
            "returned_count": len(visible),
            "total_count": total_count,
            "stage_counts": counts,
            "truncated": total_count > len(visible),
        }

    def resolve_artifact(self, relative_path: str) -> Path:
        """Resolve one image path while enforcing the project boundary."""
        decoded = unquote(relative_path)
        candidate = (self.base_dir / decoded).resolve()
        if not candidate.is_relative_to(self.base_dir):
            raise ValueError("Artifact path escaped the project.")
        if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("Only image artifacts may be served.")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    @staticmethod
    def _safe_preview_name(value: str) -> str:
        """Return a compact filesystem-safe name for generated preview files."""
        safe = [
            char if char.isalnum() or char in {"-", "_", "."} else "_"
            for char in str(value)
        ]
        return "".join(safe).strip("_")[:96] or "sample"

    def _artifact_url_for_path(self, path: Path | str | None) -> str | None:
        """Return a UI artifact URL for an existing image inside the project."""
        if not path:
            return None
        candidate = Path(path).resolve()
        if (
            not candidate.is_file()
            or candidate.suffix.lower() not in IMAGE_EXTENSIONS
            or not candidate.is_relative_to(self.base_dir)
        ):
            return None
        relative = candidate.relative_to(self.base_dir).as_posix()
        return "/artifact?path=" + quote(relative, safe="")

    def _aristotel_source_inputs(self, limit: int):
        """Yield the first deterministic glyph records without scanning all data."""
        scripts_dir = str(self.scripts_dir)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from Cyber_Lin_Kuei_Assembly.Aristotel.teacher_models import (
            TeacherInput,
        )

        if not self.aristotel_input_dir.is_dir():
            return []

        inputs = []
        for class_folder in sorted(self.aristotel_input_dir.iterdir()):
            if not class_folder.is_dir():
                continue
            for image_path in sorted(class_folder.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                inputs.append(
                    TeacherInput(
                        image_path=image_path,
                        label=class_folder.name,
                        source_class=class_folder.name,
                        source_folder=class_folder,
                        metadata={
                            "source_id": image_path.relative_to(
                                self.aristotel_input_dir
                            ).as_posix()
                        },
                    )
                )
                if len(inputs) >= limit:
                    return inputs
        return inputs

    @staticmethod
    def _aristotel_defense_preview(sample) -> dict:
        """Explain how existing reconstruction concepts would inspect damage."""
        if sample.trust_label == "repair_needed":
            expected_action = "topology_repair_candidate"
            note = (
                "This recipe is labeled as repair-needed evidence; current "
                "ScribeTrace reconstruction would diagnose topology and try "
                "minimal endpoint-bridge hypotheses when endpoints agree."
            )
        else:
            expected_action = "diagnose_before_repair"
            note = (
                "This recipe is uncertain evidence; current ScribeTrace should "
                "first diagnose topology and avoid synthetic repair unless the "
                "trace actually shows damaged structure."
            )

        return {
            "engine": "theoretical_reconstruction",
            "currently_available_tool": "endpoint_bridge",
            "expected_action": expected_action,
            "training_label": sample.trust_label,
            "cycle": list(ARISTOTEL_RECONSTRUCTION_CYCLE),
            "note": note,
        }

    def _aristotel_reconstruction_preview(
        self,
        sample,
        damaged_path: Path,
        sample_index: int,
    ) -> dict:
        """Run the current ScribeTrace reconstruction preview on one sample."""
        scripts_dir = str(self.scripts_dir)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from N05handwritten_ocr.scribetrace.expert import run_scribetrace
        from N05handwritten_ocr.scribetrace.trace_models import TraceInput
        import N05handwritten_ocr.scribetrace.trace_reconstruction as reconstruction_module

        reconstruction_root = self.aristotel_preview_dir / "reconstruction"
        reconstruction_root.mkdir(parents=True, exist_ok=True)
        stable_id = self._safe_preview_name(
            f"{sample_index:02d}_{sample.teacher_input.label}_{sample.damage_recipe}"
        )
        output_dir = reconstruction_root / stable_id
        output_dir.mkdir(parents=True, exist_ok=True)

        trace_input = TraceInput(
            crop_path=str(damaged_path),
            mask_crop_path=str(damaged_path),
            visual_crop_path=str(damaged_path),
            output_dir=str(output_dir),
            text_unit_id=stable_id,
            known_damage_recipes=[sample.damage_recipe],
        )
        trace_settings = {
            "enabled": True,
            "save_debug": True,
            "save_json": True,
            "enable_theoretical_reconstruction": True,
            "enable_mask_repair": False,
            "ink_threshold_mode": "binary",
            "fixed_threshold_value": 128,
            "minimum_ink_pixels": 4,
            "maximum_component_count_for_full_trace": 50,
            "minimum_trace_path_points": 4,
            "reconstruction_use_recognition_verification": False,
            "reconstruction_max_hypotheses": 5,
            "reconstruction_max_accepted": 3,
        }

        original_predictor = reconstruction_module.predict_rf_candidates
        reconstruction_module.predict_rf_candidates = lambda *args, **kwargs: []
        try:
            trace_result = run_scribetrace(trace_input, settings=trace_settings)
        finally:
            reconstruction_module.predict_rf_candidates = original_predictor

        reconstruction = trace_result.reconstruction or {}
        selected_path = reconstruction.get("selected_reconstructed_mask_path")
        selected_hypothesis_id = reconstruction.get("selected_hypothesis_id")
        selected_hypothesis = None
        for hypothesis in reconstruction.get("hypotheses", []):
            if hypothesis.get("hypothesis_id") == selected_hypothesis_id:
                selected_hypothesis = hypothesis
                break
        selected_hypothesis = selected_hypothesis or (
            reconstruction.get("hypotheses", [None])[0]
            if reconstruction.get("hypotheses")
            else None
        )
        overlay_path = (
            selected_hypothesis.get("reconstruction_overlay_path")
            if selected_hypothesis
            else None
        )
        after_url = self._artifact_url_for_path(selected_path)
        overlay_url = self._artifact_url_for_path(overlay_path)
        process_images = [
            {
                "step": "00_damaged",
                "label": "damaged input",
                "url": self._artifact_url_for_path(damaged_path),
            }
        ]
        debug_steps = (
            ("01_components", "component debug", "component_debug_image"),
            ("02_skeleton", "skeleton", "skeleton_debug_image"),
            ("03_graph", "skeleton graph", "skeleton_graph_debug_image"),
            ("04_trace_paths", "trace paths", "trace_paths_debug_image"),
            ("05_landmarks", "landmarks", "landmarks_debug_image"),
        )
        for step, label, key in debug_steps:
            url = self._artifact_url_for_path(trace_result.debug_paths.get(key))
            if url:
                process_images.append(
                    {
                        "step": step,
                        "label": label,
                        "url": url,
                    }
                )
        if after_url:
            process_images.append(
                {
                    "step": "06_reconstructed",
                    "label": "selected reconstruction",
                    "url": after_url,
                }
            )
        if overlay_url:
            process_images.append(
                {
                    "step": "07_overlay",
                    "label": "reconstruction overlay",
                    "url": overlay_url,
                }
            )

        return {
            "status": trace_result.status,
            "error": trace_result.error,
            "result_json_path": trace_result.result_json_path,
            "result_json_url": self._artifact_url_for_path(
                trace_result.result_json_path
            ),
            "recognition_bypassed_for_ui": True,
            "reconstruction_status": reconstruction.get("status"),
            "selected_hypothesis_id": selected_hypothesis_id,
            "selected_feature_source": reconstruction.get(
                "selected_feature_source"
            ),
            "candidate_count": reconstruction.get("candidate_count", 0),
            "accepted_count": reconstruction.get("accepted_count", 0),
            "allowed_defense_types": reconstruction.get(
                "allowed_defense_types",
                [],
            ),
            "implemented_defense_types": reconstruction.get(
                "implemented_defense_types",
                [],
            ),
            "unsupported_defense_types": reconstruction.get(
                "unsupported_defense_types",
                [],
            ),
            "damage_reasons": (
                reconstruction.get("diagnosis", {}).get("damage_reasons", [])
            ),
            "after_url": after_url,
            "overlay_url": overlay_url,
            "after_caption": (
                "selected reconstruction"
                if after_url
                else "kept damaged mask"
            ),
            "overlay_caption": (
                "green added / red removed"
                if overlay_url
                else "no overlay emitted"
            ),
            "process_images": process_images,
            "hypotheses": reconstruction.get("hypotheses", []),
        }

    def aristotel_preview(self, limit: int = 10) -> dict:
        """Build a tiny deterministic Aristotel lab preview for the UI."""
        import cv2

        scripts_dir = str(self.scripts_dir)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from Cyber_Lin_Kuei_Assembly.Aristotel import (
            FileCorrupter,
            build_default_recipes,
        )

        safe_limit = max(1, min(int(limit), 10))
        recipes = build_default_recipes()
        source_limit = max(1, (safe_limit + len(recipes) - 1) // len(recipes))
        source_inputs = self._aristotel_source_inputs(source_limit)
        if not source_inputs:
            return {
                "status": "missing_input",
                "message": f"No glyph images found in {self.aristotel_input_dir}.",
                "samples": [],
                "sample_count": 0,
                "recipe_count": len(recipes),
                "updated_at": _iso_timestamp(),
            }

        sample_dir = self.aristotel_preview_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        corrupter = FileCorrupter(recipes=recipes, seed=404)
        samples = []

        for teacher_input in source_inputs:
            for sample in corrupter.corrupt(teacher_input, epoch=0, variant=0):
                index = len(samples) + 1
                label = self._safe_preview_name(sample.teacher_input.label)
                recipe = self._safe_preview_name(sample.damage_recipe)
                output_name = (
                    f"{index:02d}_{label}_{recipe}_{sample.sample_id}.png"
                )
                output_path = sample_dir / output_name
                if not cv2.imwrite(str(output_path), sample.image):
                    raise RuntimeError(f"Could not save preview: {output_path}")

                original_path = sample.teacher_input.image_path.resolve()
                original_relative = original_path.relative_to(
                    self.base_dir
                ).as_posix()
                damaged_relative = output_path.relative_to(
                    self.base_dir
                ).as_posix()
                metadata = sample.to_metadata(output_path)
                reconstruction_preview = self._aristotel_reconstruction_preview(
                    sample=sample,
                    damaged_path=output_path,
                    sample_index=index,
                )

                samples.append(
                    {
                        "sample_index": index,
                        "sample_id": sample.sample_id,
                        "label": sample.teacher_input.label,
                        "source_id": sample.teacher_input.metadata.get(
                            "source_id"
                        ),
                        "damage_recipe": sample.damage_recipe,
                        "severity": sample.severity,
                        "trust_label": sample.trust_label,
                        "changed_pixel_count": sample.changed_pixel_count,
                        "changed_pixel_ratio": sample.changed_pixel_ratio,
                        "operations": sample.operations,
                        "original_path": str(original_path),
                        "damaged_path": str(output_path.resolve()),
                        "original_url": (
                            "/artifact?path="
                            + quote(original_relative, safe="")
                        ),
                        "damaged_url": (
                            "/artifact?path="
                            + quote(damaged_relative, safe="")
                        ),
                        "metadata": metadata,
                        "defense_preview": self._aristotel_defense_preview(
                            sample
                        ),
                        "reconstruction_preview": reconstruction_preview,
                    }
                )
                if len(samples) >= safe_limit:
                    break
            if len(samples) >= safe_limit:
                break

        return {
            "status": "completed",
            "input_root": str(self.aristotel_input_dir),
            "output_root": str(self.aristotel_preview_dir),
            "sample_count": len(samples),
            "recipe_count": len(recipes),
            "samples": samples,
            "updated_at": _iso_timestamp(),
        }

    @staticmethod
    def _load_json_if_exists(path: Path):
        """Load a JSON report if it exists; otherwise return None."""
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _percent_metric(value) -> float | None:
        """Normalize stored 0..1 metrics into display-friendly percentages."""
        if value is None:
            return None
        try:
            return round(float(value) * 100.0, 2)
        except (TypeError, ValueError):
            return None

    def _training_run_record(self, report_dir: Path) -> dict | None:
        """Build one compact training-run card from known report contracts."""
        training_report = self._load_json_if_exists(
            report_dir / "training_report.json"
        )
        evaluation_summary = self._load_json_if_exists(
            report_dir / "evaluation_summary.json"
        )
        split_report = self._load_json_if_exists(report_dir / "split_report.json")
        training_history = self._load_json_if_exists(
            report_dir / "training_history.json"
        )
        if training_report is None and evaluation_summary is None:
            return None

        payload = training_report or evaluation_summary or {}
        model_name = payload.get("model_name") or report_dir.name
        model_path = payload.get("model_path")
        if not model_path:
            candidate_models = sorted(
                self.models_dir.glob(f"{report_dir.name}/**/*"),
                key=lambda path: path.stat().st_mtime if path.is_file() else 0,
                reverse=True,
            )
            model_path = str(next((path for path in candidate_models if path.is_file()), ""))

        modified_times = [
            path.stat().st_mtime
            for path in report_dir.rglob("*")
            if path.is_file()
        ]
        latest_modified = max(modified_times) if modified_times else report_dir.stat().st_mtime
        damage_metrics = self._load_json_if_exists(
            report_dir / "damage_recipe_metrics.json"
        )
        recipe_scores = []
        if isinstance(damage_metrics, dict):
            test_metrics = damage_metrics.get("test", {})
            for recipe_name, metrics in sorted(test_metrics.items()):
                if not isinstance(metrics, dict):
                    continue
                recipe_scores.append(
                    {
                        "name": recipe_name,
                        "count": metrics.get("count"),
                        "top1": self._percent_metric(metrics.get("top1")),
                        "top5": self._percent_metric(metrics.get("top5")),
                    }
                )

        return {
            "id": report_dir.name,
            "model_name": model_name,
            "model_type": payload.get("model_type", "unknown"),
            "model_path": model_path,
            "model_exists": bool(model_path and Path(model_path).is_file()),
            "report_dir": str(report_dir),
            "updated_at": _iso_timestamp(latest_modified),
            "dataset_rows": payload.get("dataset_rows"),
            "dataset_jsonl": payload.get("dataset_jsonl"),
            "num_classes": payload.get("num_classes"),
            "test_samples": payload.get("test_samples"),
            "split": payload.get("split") or split_report,
            "validation_top1": self._percent_metric(
                payload.get("validation_top1")
                or payload.get("checkpoint_val_top1")
            ),
            "validation_top5": self._percent_metric(
                payload.get("validation_top5")
                or payload.get("checkpoint_val_top5")
            ),
            "test_top1": self._percent_metric(payload.get("test_top1")),
            "test_top5": self._percent_metric(payload.get("test_top5")),
            "recipe_scores": recipe_scores[:18],
            "training_history": (
                training_history
                if isinstance(training_history, list)
                else []
            ),
            "notes": payload.get("notes", []),
            "primary_report": (
                str(report_dir / "training_report.json")
                if (report_dir / "training_report.json").is_file()
                else str(report_dir / "evaluation_summary.json")
            ),
        }

    def training_overview(self) -> dict:
        """Summarize existing model-training reports for the UI."""
        runs = []
        if self.reports_dir.is_dir():
            for report_dir in sorted(self.reports_dir.iterdir()):
                if not report_dir.is_dir():
                    continue
                record = self._training_run_record(report_dir)
                if record is not None:
                    runs.append(record)

        runs.sort(key=lambda item: item["updated_at"], reverse=True)
        active_scribetrace_model = self._load_json_if_exists(
            self.models_dir / "scribetrace_active_model.json"
        )
        commands = [
            {
                "id": "train",
                "label": "Train Minos",
                "command": "train",
                "description": "Run the existing main.py train command.",
                "enabled": True,
            },
            {
                "id": "scribetrace_rf",
                "label": "ScribeTrace RF",
                "command": None,
                "description": (
                    "Report viewer ready. Dedicated launcher will be added "
                    "after we lock the exact export/train command."
                ),
                "enabled": False,
            },
        ]
        return {
            "status": "completed",
            "report_root": str(self.reports_dir),
            "model_root": str(self.models_dir),
            "active_scribetrace_model": active_scribetrace_model,
            "run_count": len(runs),
            "runs": runs,
            "commands": commands,
            "process": self.process_manager.snapshot(),
            "updated_at": _iso_timestamp(),
        }


def _handler_factory(application: PipelineUiApplication):
    """Create a request handler bound to one UI application."""

    class PipelineUiHandler(BaseHTTPRequestHandler):
        server_version = "PipelineControlRoom/0.1"

        def log_message(self, _format: str, *_args) -> None:
            """Suppress noisy per-request HTTP logs in the pipeline console."""

        def _send_bytes(
            self,
            payload: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
            cache_control: str = "no-store",
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", cache_control)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(
            self,
            payload: dict,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_bytes(
                data,
                "application/json; charset=utf-8",
                status=status,
            )

        def _send_error_json(
            self,
            error: Exception | str,
            status: HTTPStatus,
        ) -> None:
            self._send_json({"error": str(error)}, status=status)

        def _read_json_body(self) -> dict:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                return {}
            if content_length > 64 * 1024:
                raise ValueError("Request body is too large.")
            body = self.rfile.read(content_length)
            return json.loads(body.decode("utf-8"))

        def _serve_static(self, relative_name: str) -> None:
            requested = (
                application.static_dir / relative_name
            ).resolve()
            if not requested.is_relative_to(application.static_dir.resolve()):
                self._send_error_json(
                    "Invalid static path.",
                    HTTPStatus.BAD_REQUEST,
                )
                return
            if not requested.is_file():
                self._send_error_json(
                    "Static asset not found.",
                    HTTPStatus.NOT_FOUND,
                )
                return
            content_type, _ = mimetypes.guess_type(str(requested))
            self._send_bytes(
                requested.read_bytes(),
                content_type or "application/octet-stream",
                cache_control="no-cache",
            )

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    self._serve_static("index.html")
                elif parsed.path.startswith("/static/"):
                    self._serve_static(parsed.path.removeprefix("/static/"))
                elif parsed.path == "/api/overview":
                    self._send_json(application.overview())
                elif parsed.path == "/api/logs":
                    offset = int(query.get("offset", ["0"])[0])
                    self._send_json(
                        application.process_manager.logs_since(offset)
                    )
                elif parsed.path == "/api/artifacts":
                    self._send_json(
                        application.artifacts(
                            document_id=query.get("document", [""])[0],
                            stage_id=query.get("stage", ["all"])[0],
                            query=query.get("query", [""])[0],
                            limit=int(query.get("limit", ["240"])[0]),
                        )
                    )
                elif parsed.path == "/api/aristotel-preview":
                    self._send_json(
                        application.aristotel_preview(
                            limit=int(query.get("limit", ["10"])[0]),
                        )
                    )
                elif parsed.path == "/api/training-overview":
                    self._send_json(application.training_overview())
                elif parsed.path == "/artifact":
                    artifact = application.resolve_artifact(
                        query.get("path", [""])[0]
                    )
                    content_type, _ = mimetypes.guess_type(str(artifact))
                    self._send_bytes(
                        artifact.read_bytes(),
                        content_type or "application/octet-stream",
                        cache_control="no-cache",
                    )
                else:
                    self._send_error_json(
                        "Route not found.",
                        HTTPStatus.NOT_FOUND,
                    )
            except FileNotFoundError as error:
                self._send_error_json(error, HTTPStatus.NOT_FOUND)
            except (ValueError, json.JSONDecodeError) as error:
                self._send_error_json(error, HTTPStatus.BAD_REQUEST)
            except Exception as error:
                self._send_error_json(
                    error,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                payload = self._read_json_body()
                if parsed.path == "/api/run":
                    state = application.process_manager.start(
                        str(payload.get("command", ""))
                    )
                    self._send_json({"process": state}, HTTPStatus.ACCEPTED)
                elif parsed.path == "/api/stop":
                    state = application.process_manager.stop()
                    self._send_json({"process": state})
                else:
                    self._send_error_json(
                        "Route not found.",
                        HTTPStatus.NOT_FOUND,
                    )
            except RuntimeError as error:
                self._send_error_json(error, HTTPStatus.CONFLICT)
            except (ValueError, json.JSONDecodeError) as error:
                self._send_error_json(error, HTTPStatus.BAD_REQUEST)
            except Exception as error:
                self._send_error_json(
                    error,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    return PipelineUiHandler


def _build_server(
    application: PipelineUiApplication,
    host: str,
    preferred_port: int,
) -> tuple[ThreadingHTTPServer, int]:
    """Bind the first available localhost port near the preferred port."""
    handler = _handler_factory(application)
    last_error: OSError | None = None
    for port in range(preferred_port, preferred_port + 20):
        try:
            return ThreadingHTTPServer((host, port), handler), port
        except OSError as error:
            last_error = error
    raise RuntimeError(
        f"Could not bind pipeline UI near port {preferred_port}: {last_error}"
    )


def launch_pipeline_ui(
    base_dir: str,
    controller_script: str,
    python_executable: str,
    initial_stage: str | None = None,
    initial_query: str | None = None,
) -> None:
    """Launch the local control room and block until Ctrl+C."""
    host = "127.0.0.1"
    preferred_port = int(os.environ.get("OCR_PIPELINE_UI_PORT", "8765"))
    application = PipelineUiApplication(
        base_dir=Path(base_dir),
        controller_script=Path(controller_script),
        python_executable=Path(python_executable),
    )
    server, port = _build_server(application, host, preferred_port)
    query_parameters = []
    if initial_stage:
        query_parameters.append("stage=" + quote(initial_stage, safe=""))
    if initial_query:
        query_parameters.append("query=" + quote(initial_query, safe=""))
    query_string = "?" + "&".join(query_parameters) if query_parameters else ""
    url = f"http://{host}:{port}{query_string}"

    print("Armenian OCR Pipeline Control Room")
    print("----------------------------------")
    print("URL:", url)
    print("Press Ctrl+C to stop the UI server.")

    if os.environ.get("OCR_PIPELINE_UI_NO_BROWSER") != "1":
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping pipeline control room...")
    finally:
        application.process_manager.stop()
        server.server_close()


__all__ = [
    "PipelineProcessManager",
    "PipelineUiApplication",
    "launch_pipeline_ui",
]
