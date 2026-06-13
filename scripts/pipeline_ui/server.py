"""Serve the dependency-free local pipeline control room."""

from __future__ import annotations

import json
import mimetypes
import os
import signal
import subprocess
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
        self.input_dir = self.base_dir / "handwritten_text"
        self.temp_dir = self.base_dir / "temp_processing"
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
