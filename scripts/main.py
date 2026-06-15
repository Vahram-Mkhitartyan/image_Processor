import argparse
import glob
import json
import os
import shutil
import subprocess
import sys


BASE_DIR = "/home/vahram/Desktop/image_Processor"

SCRIPTS_DIR = f"{BASE_DIR}/scripts"
TESTS_DIR = f"{BASE_DIR}/tests"

TRAIN_SCRIPT = f"{SCRIPTS_DIR}/Cyber_Lin_Kuei_Assembly/train_minos_classifier.py"
BATCH_SCRIPT = f"{SCRIPTS_DIR}/pipeline_control/batch_processor.py"
FILE_PREPARATION_SCRIPT = f"{SCRIPTS_DIR}/N00_file_preparation/file_preparation.py"
SCRIBEMAP_SCRIPT = f"{SCRIPTS_DIR}/N01_scribemap/scribemap_detector.py"
CROP_REFINER_SCRIPT = f"{SCRIPTS_DIR}/N02_crop_refiner/crop_refiner.py"
FILE_PREPARATION_DIR = f"{SCRIPTS_DIR}/N00_file_preparation"
SCRIBEMAP_DIR = f"{SCRIPTS_DIR}/N01_scribemap"
CROP_REFINER_DIR = f"{SCRIPTS_DIR}/N02_crop_refiner"
VISUAL_CLASSIFICATION_SCRIPT = f"{SCRIPTS_DIR}/N03_visual_classification_router/classifier.py"
VISUAL_CLASSIFICATION_DIR = f"{SCRIPTS_DIR}/N03_visual_classification_router"
PRINTED_OCR_SCRIPT = f"{SCRIPTS_DIR}/N04_printed_ocr/printed_ocr.py"
PRINTED_OCR_DIR = f"{SCRIPTS_DIR}/N04_printed_ocr"
N05_ORCHESTRATOR_SCRIPT = f"{SCRIPTS_DIR}/N05handwritten_ocr/expert_orchestrator.py"
N05_DIR = f"{SCRIPTS_DIR}/N05handwritten_ocr"


DATASET_DIR = f"{BASE_DIR}/classifier_dataset_presence"
INPUT_DOCUMENTS_DIR = f"{BASE_DIR}/handwritten_text"
MODELS_DIR = f"{BASE_DIR}/models"
FINAL_RESULTS_DIR = f"{BASE_DIR}/final_results"
FAILED_RESULTS_DIR = f"{BASE_DIR}/failed_results"
TEMP_PROCESSING_DIR = f"{BASE_DIR}/temp_processing"


CLASS_FOLDERS = [
    "mixed",
    "printed_only",
    "empty_or_noise",
    "handwriting_only"
]

LINE_COUNT_EXCLUDED_DIRS = {
    "__pycache__",
    "scan_report",
}

SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

DOCTOR_REQUIRED_PATHS = {
    "main controller": f"{SCRIPTS_DIR}/main.py",
    "batch controller": BATCH_SCRIPT,
    "N00 orchestrator": FILE_PREPARATION_SCRIPT,
    "N01 orchestrator": SCRIBEMAP_SCRIPT,
    "N02 orchestrator": CROP_REFINER_SCRIPT,
    "N03 orchestrator": VISUAL_CLASSIFICATION_SCRIPT,
    "N04 orchestrator": PRINTED_OCR_SCRIPT,
    "N05 orchestrator": N05_ORCHESTRATOR_SCRIPT,
    "active Minos model": f"{MODELS_DIR}/minos_v2_0_best.keras",
}

DOCTOR_RETIRED_ARTIFACT_NAMES = {
    "blue_candidates",
    "dense_candidates",
    "scribemap_blue_candidates.json",
    "scribemap_dense_candidates.json",
}

MAIN_COMMANDS = (
    "ui",
    "status",
    "counts",
    "lines",
    "doctor",
    "train",
    "batch",
    "pipeline",
    "prep",
    "scribemap",
    "refine",
    "visual",
    "visual_classification",
    "n03",
    "printed_ocr",
    "printed",
    "n04",
    "handwritten_ocr",
    "handwritten",
    "n05",
    "splits",
    "letter_splits",
    "clean",
    "setup",
    "completion",
)


def run_pipeline_ui(initial_stage=None, initial_query=None):
    """Launch the local browser control room for pipeline commands.

    Args:
        initial_stage: Optional artifact stage selected when the UI opens.
        initial_query: Optional artifact-path filter selected when the UI opens.

    Returns:
        None. The server runs until interrupted with Ctrl+C.
    """
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)

    from pipeline_ui import launch_pipeline_ui

    launch_pipeline_ui(
        base_dir=BASE_DIR,
        controller_script=os.path.abspath(__file__),
        python_executable=sys.executable,
        initial_stage=initial_stage,
        initial_query=initial_query,
    )


def add_doctor_result(results, level, check_name, detail=None):
    """Record and print one project-doctor result.

    Args:
        results: Mutable list receiving structured result dictionaries.
        level: One of PASS, WARN, or FAIL.
        check_name: Short human-readable check description.
        detail: Optional supporting information.

    Returns:
        None.
    """
    record = {
        "level": level,
        "check": check_name,
        "detail": detail,
    }
    results.append(record)

    message = f"[{level}] {check_name}"
    if detail:
        message += f": {detail}"
    print(message)


def iter_active_python_files():
    """Yield active Python files while excluding legacy and cache folders."""
    for root, dirs, files in os.walk(SCRIPTS_DIR):
        dirs[:] = [
            directory
            for directory in dirs
            if directory not in {"__pycache__", "legacy"}
        ]

        for file_name in sorted(files):
            if file_name.endswith(".py"):
                yield os.path.join(root, file_name)


def iter_test_python_files():
    """Yield Python files from the centralized test suite."""
    if not os.path.isdir(TESTS_DIR):
        return

    for root, dirs, files in os.walk(TESTS_DIR):
        dirs[:] = [
            directory
            for directory in dirs
            if directory != "__pycache__"
        ]
        for file_name in sorted(files):
            if file_name.endswith(".py"):
                yield os.path.join(root, file_name)


def validate_json_file(json_path):
    """Load one JSON file and return an error string when invalid."""
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            json.load(file)
    except Exception as error:
        return str(error)

    return None


def get_input_document_ids():
    """Return sorted document ids for supported files in handwritten_text."""
    if not os.path.isdir(INPUT_DOCUMENTS_DIR):
        return []

    document_ids = []

    for file_name in os.listdir(INPUT_DOCUMENTS_DIR):
        file_path = os.path.join(INPUT_DOCUMENTS_DIR, file_name)
        extension = os.path.splitext(file_name)[1].lower()

        if os.path.isfile(file_path) and extension in SUPPORTED_DOCUMENT_EXTENSIONS:
            document_ids.append(os.path.splitext(file_name)[0])

    return sorted(document_ids)


def collect_missing_path_references(payload):
    """Find missing absolute paths referenced by *_path and *_dir JSON fields."""
    missing = []

    def visit(value, parent_key=""):
        """Recursively inspect nested JSON while retaining the owning field name."""
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, key)
            return

        if isinstance(value, list):
            for child in value:
                visit(child, parent_key)
            return

        if (
            isinstance(value, str)
            and parent_key.endswith(("_path", "_dir"))
            and os.path.isabs(value)
            and not os.path.exists(value)
        ):
            missing.append((parent_key, value))

    visit(payload)
    return missing


def inspect_document_health(document_id, results):
    """Check phase artifacts and metadata contracts for one input document."""
    print()
    print(f"Document: {document_id}")

    document_dir = os.path.join(TEMP_PROCESSING_DIR, document_id)
    final_result_path = os.path.join(
        FINAL_RESULTS_DIR,
        f"{document_id}_handwriting_result.json",
    )

    if not os.path.isdir(document_dir):
        add_doctor_result(
            results,
            "WARN",
            f"{document_id} runtime state",
            "not processed yet",
        )
        return

    n00_metadata_path = os.path.join(
        document_dir,
        "n00_file_preparation",
        "metadata",
        "metadata.json",
    )
    n01_groups_path = os.path.join(
        document_dir,
        "n01_scribemap",
        "metadata",
        f"{document_id}_classified_groups.json",
    )
    n02_groups_path = os.path.join(
        document_dir,
        "n02_crop_refiner",
        "metadata",
        f"{document_id}_refined_groups.json",
    )

    phase_paths = [
        ("N00 metadata", n00_metadata_path),
        ("N01 groups", n01_groups_path),
        ("N02 refined groups", n02_groups_path),
    ]

    for phase_name, phase_path in phase_paths:
        if os.path.isfile(phase_path):
            add_doctor_result(results, "PASS", f"{document_id} {phase_name}")
        else:
            add_doctor_result(
                results,
                "WARN",
                f"{document_id} {phase_name}",
                "not generated",
            )

    if os.path.isfile(n01_groups_path):
        try:
            with open(n01_groups_path, "r", encoding="utf-8") as file:
                n01_payload = json.load(file)

            groups = n01_payload.get("classified_groups", [])
            declared_count = n01_payload.get("scribemap_group_count")

            if declared_count is not None and declared_count != len(groups):
                add_doctor_result(
                    results,
                    "FAIL",
                    f"{document_id} N01 group count",
                    f"declared {declared_count}, stored {len(groups)}",
                )
            else:
                add_doctor_result(
                    results,
                    "PASS",
                    f"{document_id} N01 group count",
                    str(len(groups)),
                )
        except Exception as error:
            add_doctor_result(
                results,
                "FAIL",
                f"{document_id} N01 JSON",
                str(error),
            )

    if os.path.isfile(n02_groups_path):
        try:
            with open(n02_groups_path, "r", encoding="utf-8") as file:
                n02_payload = json.load(file)

            refined_groups = n02_payload.get("refined_groups", [])
            declared_count = n02_payload.get("summary", {}).get("group_count")
            missing_crops = []

            for record in refined_groups:
                for crop_key in (
                    "analysis_crop_path",
                    "classification_crop_path",
                    "context_crop_path",
                    "analysis_mask_crop_path",
                ):
                    crop_path = record.get(crop_key)
                    if crop_path and not os.path.isfile(crop_path):
                        missing_crops.append(crop_path)

            if declared_count is not None and declared_count != len(refined_groups):
                add_doctor_result(
                    results,
                    "FAIL",
                    f"{document_id} N02 group count",
                    f"declared {declared_count}, stored {len(refined_groups)}",
                )
            else:
                add_doctor_result(
                    results,
                    "PASS",
                    f"{document_id} N02 group count",
                    str(len(refined_groups)),
                )

            if missing_crops:
                add_doctor_result(
                    results,
                    "FAIL",
                    f"{document_id} N02 crop references",
                    f"{len(missing_crops)} missing",
                )
            else:
                add_doctor_result(
                    results,
                    "PASS",
                    f"{document_id} N02 crop references",
                    "all present",
                )
        except Exception as error:
            add_doctor_result(
                results,
                "FAIL",
                f"{document_id} N02 JSON",
                str(error),
            )

    if not os.path.isfile(final_result_path):
        add_doctor_result(
            results,
            "WARN",
            f"{document_id} final result",
            "not generated",
        )
        return

    try:
        with open(final_result_path, "r", encoding="utf-8") as file:
            final_payload = json.load(file)

        missing_references = collect_missing_path_references(final_payload)
        if missing_references:
            missing_keys = sorted({
                key
                for key, _ in missing_references
            })
            key_preview = ", ".join(missing_keys[:4])
            if len(missing_keys) > 4:
                key_preview += ", ..."

            add_doctor_result(
                results,
                "WARN",
                f"{document_id} final-result references",
                f"{len(missing_references)} stale paths ({key_preview})",
            )
        else:
            add_doctor_result(
                results,
                "PASS",
                f"{document_id} final-result references",
                "all present",
            )
    except Exception as error:
        add_doctor_result(
            results,
            "FAIL",
            f"{document_id} final result JSON",
            str(error),
        )


def run_project_doctor():
    """Run read-only structural, syntax, dependency, and artifact checks.

    Returns:
        True when no failing checks were found.
    """
    results = []

    print("Image Processor Doctor")
    print("-------------------------")
    print("Mode: read-only")
    print()
    print("Core structure")

    for label, path in DOCTOR_REQUIRED_PATHS.items():
        if os.path.exists(path):
            add_doctor_result(results, "PASS", label, path)
        else:
            add_doctor_result(results, "FAIL", label, f"missing: {path}")

    for label, path in (
        ("input folder", INPUT_DOCUMENTS_DIR),
        ("temp folder", TEMP_PROCESSING_DIR),
        ("final-results folder", FINAL_RESULTS_DIR),
        ("failed-results folder", FAILED_RESULTS_DIR),
    ):
        if os.path.isdir(path):
            add_doctor_result(results, "PASS", label, path)
        else:
            add_doctor_result(results, "FAIL", label, f"missing: {path}")

    print()
    print("Python environment")
    python_executable = resolve_python_executable(
        required_modules=["cv2", "tensorflow"]
    )
    version_result = subprocess.run(
        [python_executable, "-c", "import sys; print(sys.version.split()[0])"],
        capture_output=True,
        text=True,
        check=False,
    )

    if version_result.returncode == 0:
        add_doctor_result(
            results,
            "PASS",
            "pipeline Python",
            f"{python_executable} ({version_result.stdout.strip()})",
        )
    else:
        add_doctor_result(
            results,
            "FAIL",
            "pipeline Python",
            python_executable,
        )

    for module_name in ("cv2", "numpy", "tensorflow"):
        module_result = subprocess.run(
            [python_executable, "-c", f"import {module_name}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if module_result.returncode == 0:
            add_doctor_result(results, "PASS", f"import {module_name}")
        else:
            detail = module_result.stderr.strip().splitlines()[-1]
            add_doctor_result(
                results,
                "FAIL",
                f"import {module_name}",
                detail,
            )

    print()
    print("Source validation")
    python_files = list(iter_active_python_files())
    syntax_failures = []

    for python_path in python_files:
        try:
            with open(python_path, "r", encoding="utf-8") as file:
                source = file.read()
            compile(source, python_path, "exec")
        except Exception as error:
            syntax_failures.append(
                f"{os.path.relpath(python_path, BASE_DIR)}: {error}"
            )

    if syntax_failures:
        for failure in syntax_failures:
            add_doctor_result(results, "FAIL", "Python syntax", failure)
    else:
        add_doctor_result(
            results,
            "PASS",
            "Python syntax",
            f"{len(python_files)} active files",
        )

    test_python_files = list(iter_test_python_files())
    test_syntax_failures = []
    for python_path in test_python_files:
        try:
            with open(python_path, "r", encoding="utf-8") as file:
                source = file.read()
            compile(source, python_path, "exec")
        except Exception as error:
            test_syntax_failures.append(
                f"{os.path.relpath(python_path, BASE_DIR)}: {error}"
            )

    if test_syntax_failures:
        for failure in test_syntax_failures:
            add_doctor_result(results, "FAIL", "Test syntax", failure)
    else:
        add_doctor_result(
            results,
            "PASS",
            "Test syntax",
            f"{len(test_python_files)} files",
        )

    settings_files = []
    for root, dirs, files in os.walk(SCRIPTS_DIR):
        dirs[:] = [
            directory
            for directory in dirs
            if directory not in {"__pycache__", "legacy"}
        ]
        for file_name in files:
            if file_name.endswith(".json"):
                settings_files.append(os.path.join(root, file_name))

    json_failures = []
    for json_path in settings_files:
        error = validate_json_file(json_path)
        if error:
            json_failures.append(
                f"{os.path.relpath(json_path, BASE_DIR)}: {error}"
            )

    if json_failures:
        for failure in json_failures:
            add_doctor_result(results, "FAIL", "settings JSON", failure)
    else:
        add_doctor_result(
            results,
            "PASS",
            "settings JSON",
            f"{len(settings_files)} active files",
        )

    expert_import_code = (
        "import sys; "
        f"sys.path.insert(0, {N05_DIR!r}); "
        "import character_detector, scribetrace, tesseract_ocr, word_level_ocr; "
        "packages = (character_detector, scribetrace, tesseract_ocr, word_level_ocr); "
        "assert all(callable(package.get_expert_manifest) for package in packages); "
        "assert all(callable(package.recognize) for package in packages)"
    )
    expert_import_result = subprocess.run(
        [python_executable, "-c", expert_import_code],
        capture_output=True,
        text=True,
        check=False,
    )

    if expert_import_result.returncode == 0:
        add_doctor_result(
            results,
            "PASS",
            "N05 expert interfaces",
            "4 packages importable",
        )
    else:
        detail = expert_import_result.stderr.strip().splitlines()[-1]
        add_doctor_result(
            results,
            "FAIL",
            "N05 expert interfaces",
            detail,
        )

    retired_artifacts = []
    for search_root in (SCRIPTS_DIR, TEMP_PROCESSING_DIR):
        if not os.path.isdir(search_root):
            continue
        for root, dirs, files in os.walk(search_root):
            for name in dirs + files:
                if name in DOCTOR_RETIRED_ARTIFACT_NAMES:
                    retired_artifacts.append(os.path.join(root, name))

    if retired_artifacts:
        add_doctor_result(
            results,
            "WARN",
            "retired candidate artifacts",
            f"{len(retired_artifacts)} found",
        )
    else:
        add_doctor_result(
            results,
            "PASS",
            "retired candidate artifacts",
            "none",
        )

    document_ids = get_input_document_ids()
    print()
    print("Document contracts")

    if not document_ids:
        add_doctor_result(
            results,
            "WARN",
            "input documents",
            "none found",
        )
    else:
        add_doctor_result(
            results,
            "PASS",
            "input documents",
            str(len(document_ids)),
        )
        for document_id in document_ids:
            inspect_document_health(document_id, results)

    pass_count = sum(item["level"] == "PASS" for item in results)
    warning_count = sum(item["level"] == "WARN" for item in results)
    failure_count = sum(item["level"] == "FAIL" for item in results)

    print()
    print("Doctor summary")
    print("-------------------------")
    print("Passed:", pass_count)
    print("Warnings:", warning_count)
    print("Failures:", failure_count)

    if failure_count:
        print("Diagnosis: needs attention")
    elif warning_count:
        print("Diagnosis: healthy with warnings")
    else:
        print("Diagnosis: healthy")

    return failure_count == 0


def count_text_file_lines(file_path):
    """Count physical lines in one UTF-8 text file.

    Args:
        file_path: Path to the source, documentation, or configuration file.

    Returns:
        Number of physical lines in the file.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        return sum(1 for _ in file)


def collect_project_line_counts():
    """Collect production, test, legacy, documentation, and config totals.

    Args:
        None.

    Returns:
        Dictionary containing line and file counts for each category.
    """
    counts = {
        "active_python": {"lines": 0, "files": 0},
        "test_python": {"lines": 0, "files": 0},
        "legacy_python": {"lines": 0, "files": 0},
        "documentation": {"lines": 0, "files": 0},
        "configuration": {"lines": 0, "files": 0},
    }

    for root, dirs, files in os.walk(SCRIPTS_DIR):
        dirs[:] = [
            directory
            for directory in dirs
            if directory not in LINE_COUNT_EXCLUDED_DIRS
        ]

        relative_root = os.path.relpath(root, SCRIPTS_DIR)
        path_parts = set(relative_root.split(os.sep))
        is_legacy = "legacy" in path_parts

        for file_name in files:
            file_path = os.path.join(root, file_name)
            extension = os.path.splitext(file_name)[1].lower()

            if extension == ".py":
                category = "legacy_python" if is_legacy else "active_python"
            elif extension in {".md", ".txt"}:
                category = "documentation"
            elif extension == ".json":
                category = "configuration"
            else:
                continue

            counts[category]["lines"] += count_text_file_lines(file_path)
            counts[category]["files"] += 1

    if os.path.isdir(TESTS_DIR):
        for root, dirs, files in os.walk(TESTS_DIR):
            dirs[:] = [
                directory
                for directory in dirs
                if directory not in LINE_COUNT_EXCLUDED_DIRS
            ]
            for file_name in files:
                file_path = os.path.join(root, file_name)
                extension = os.path.splitext(file_name)[1].lower()
                if extension == ".py":
                    category = "test_python"
                elif extension in {".md", ".txt"}:
                    category = "documentation"
                elif extension == ".json":
                    category = "configuration"
                else:
                    continue
                counts[category]["lines"] += count_text_file_lines(file_path)
                counts[category]["files"] += 1

    # Include project-level documentation without scanning datasets or outputs.
    for file_name in os.listdir(BASE_DIR):
        file_path = os.path.join(BASE_DIR, file_name)
        extension = os.path.splitext(file_name)[1].lower()

        if os.path.isfile(file_path) and extension in {".md", ".txt"}:
            counts["documentation"]["lines"] += count_text_file_lines(file_path)
            counts["documentation"]["files"] += 1

    return counts


def format_sacred_count(line_count):
    """Format an exact line count together with a compact thousands value.

    Args:
        line_count: Integer number of physical lines.

    Returns:
        Human-readable count such as ``12,345 lines (12.3k)``.
    """
    return f"{line_count:,} lines ({line_count / 1000:.1f}k)"


def show_project_line_counts():
    """Print the project's sacred-question line-count report.

    Args:
        None.

    Returns:
        None.
    """
    counts = collect_project_line_counts()

    active = counts["active_python"]
    tests = counts["test_python"]
    legacy = counts["legacy_python"]
    documentation = counts["documentation"]
    configuration = counts["configuration"]

    all_python_lines = active["lines"] + legacy["lines"]
    all_python_with_tests = all_python_lines + tests["lines"]
    grand_total = (
        all_python_with_tests
        + documentation["lines"]
        + configuration["lines"]
    )

    print("The Sacred Question")
    print("-------------------------")
    print(
        "Active Python:",
        format_sacred_count(active["lines"]),
        f"across {active['files']} files",
    )
    print(
        "Tests:",
        format_sacred_count(tests["lines"]),
        f"across {tests['files']} files",
    )
    print(
        "Legacy Python:",
        format_sacred_count(legacy["lines"]),
        f"across {legacy['files']} files",
    )
    print(
        "Documentation:",
        format_sacred_count(documentation["lines"]),
        f"across {documentation['files']} files",
    )
    print(
        "Configuration:",
        format_sacred_count(configuration["lines"]),
        f"across {configuration['files']} files",
    )
    print("-------------------------")
    print("Active + legacy code:", format_sacred_count(all_python_lines))
    print(
        "All Python including tests:",
        format_sacred_count(all_python_with_tests),
    )
    print("Everything included:", format_sacred_count(grand_total))


def resolve_python_executable(required_modules=None):
    """Pick the Python executable that has required dependencies.
    
    Args:
        required_modules: Optional list of modules the executable must import.
    
    Returns:
        Python executable path string.
    """
    preferred = [
        f"{BASE_DIR}/CNN_environment.venv/bin/python",
        f"{BASE_DIR}/.venv/bin/python",
        sys.executable,
    ]

    if required_modules is None:
        required_modules = ["cv2"]

    def has_required_modules(python_path):
        """Check whether a Python executable can import required modules.
        
        Args:
            python_path: Python executable path to test.
        
        Returns:
            True when required modules import successfully.
        """
        import_statement = "; ".join(f"import {module_name}" for module_name in required_modules)
        check_cmd = [python_path, "-c", import_statement]
        try:
            result = subprocess.run(
                check_cmd,
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except Exception:
            return False

    # Prefer an interpreter that already has runtime deps for the pipeline.
    for candidate in preferred:
        if os.path.exists(candidate) and has_required_modules(candidate):
            return candidate

    # Fallback to first existing interpreter if dependency probe fails.
    for candidate in preferred:
        if os.path.exists(candidate):
            return candidate

    return sys.executable


def run_python_script(script_path, required_modules=None):
    """Run a Python script with the resolved interpreter.
    
    Args:
        script_path: Path to the Python script to execute.
    
    Returns:
        None; raises if subprocess fails.
    """
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    python_executable = resolve_python_executable(required_modules=required_modules)

    subprocess.run(
        [python_executable, script_path],
        check=True
    )

def run_python_script_with_args(script_path, extra_args, required_modules=None):
    """Run a Python script with additional arguments.
    
    Args:
        script_path: Path to the Python script to execute.
        extra_args: Additional command-line arguments for the script.
    
    Returns:
        None; raises if subprocess fails.
    """
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    python_executable = resolve_python_executable(required_modules=required_modules)
    subprocess.run(
        [python_executable, script_path] + extra_args,
        check=True
    )


def show_dataset_counts():
    """Print dataset counts by classifier class.
    
    Args:
        None.
    
    Returns:
        None.
    """
    print("Dataset counts:")
    print("-------------------------")

    total = 0

    for class_name in CLASS_FOLDERS:
        class_dir = f"{DATASET_DIR}/{class_name}"

        if not os.path.exists(class_dir):
            print(class_name, ": folder missing")
            continue

        count = 0

        for file_name in os.listdir(class_dir):
            file_path = f"{class_dir}/{file_name}"

            if os.path.isfile(file_path):
                count += 1

        total += count
        print(class_name, ":", count)

    print("-------------------------")
    print("Total:", total)


def format_storage_size(byte_count):
    """Format a byte count using compact binary units."""
    value = float(max(0, byte_count))
    units = ("B", "KiB", "MiB", "GiB", "TiB")

    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0

    return f"{int(byte_count)} B"


def get_directory_size(directory_path):
    """Return directory size using du when available, with a safe fallback."""
    if not os.path.exists(directory_path):
        return 0

    size_result = subprocess.run(
        ["du", "-s", "-B1", directory_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if size_result.returncode == 0:
        try:
            return int(size_result.stdout.split()[0])
        except (IndexError, ValueError):
            pass

    total_size = 0
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                total_size += os.path.getsize(file_path)
            except OSError:
                continue
    return total_size


def get_document_phase_state(document_id):
    """Return generated phase flags for one input document."""
    document_dir = os.path.join(TEMP_PROCESSING_DIR, document_id)
    phase_paths = {
        "N00": os.path.join(
            document_dir,
            "n00_file_preparation",
            "metadata",
            "metadata.json",
        ),
        "N01": os.path.join(
            document_dir,
            "n01_scribemap",
            "metadata",
            f"{document_id}_classified_groups.json",
        ),
        "N02": os.path.join(
            document_dir,
            "n02_crop_refiner",
            "metadata",
            f"{document_id}_refined_groups.json",
        ),
        "N03": os.path.join(
            document_dir,
            "n03_visual_classification",
            "metadata",
            f"{document_id}_n03_visual_classification_routes.json",
        ),
        "N04": os.path.join(
            document_dir,
            "n04_printed_ocr",
            "metadata",
            f"{document_id}_printed_text_map.json",
        ),
        "N05": os.path.join(
            document_dir,
            "n05_handwritten_ocr",
            "metadata",
            f"{document_id}_handwritten_text_map.json",
        ),
    }
    phase_state = {
        phase_name: os.path.isfile(phase_path)
        for phase_name, phase_path in phase_paths.items()
    }
    phase_state["RESULT"] = os.path.isfile(
        os.path.join(
            FINAL_RESULTS_DIR,
            f"{document_id}_handwriting_result.json",
        )
    )
    return phase_state


def get_n05_expert_state():
    """Return enabled flags from the N05 expert settings file."""
    settings_path = os.path.join(N05_DIR, "settings.json")
    expert_names = (
        "tesseract_ocr",
        "scribetrace",
        "character_detector",
        "word_level_ocr",
    )
    state = {expert_name: False for expert_name in expert_names}

    try:
        with open(settings_path, "r", encoding="utf-8") as file:
            settings = json.load(file)
        configured_experts = settings.get("experts", {})
        for expert_name in expert_names:
            state[expert_name] = bool(
                configured_experts.get(expert_name, {}).get("enabled", False)
            )
    except (OSError, json.JSONDecodeError):
        pass

    return state


def get_git_status_summary():
    """Return the current branch, commit, and working-tree state."""
    repository_result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if repository_result.returncode != 0:
        return {"available": False}

    branch_result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    commit_result = subprocess.run(
        ["git", "log", "-1", "--format=%h %s"],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    changed_paths = [
        line
        for line in status_result.stdout.splitlines()
        if line.strip()
    ]
    return {
        "available": True,
        "branch": branch_result.stdout.strip() or "detached",
        "commit": commit_result.stdout.strip() or "no commits",
        "changed_path_count": len(changed_paths),
    }


def show_project_status():
    """Print a read-only operational snapshot of the complete OCR system.

    Args:
        None.

    Returns:
        None.
    """
    node_scripts = (
        ("N00", "File preparation", FILE_PREPARATION_SCRIPT),
        ("N01", "ScribeMap", SCRIBEMAP_SCRIPT),
        ("N02", "Crop refiner", CROP_REFINER_SCRIPT),
        ("N03", "Visual router", VISUAL_CLASSIFICATION_SCRIPT),
        ("N04", "Printed OCR", PRINTED_OCR_SCRIPT),
        ("N05", "Handwriting experts", N05_ORCHESTRATOR_SCRIPT),
    )
    missing_nodes = [
        node_name
        for node_name, _, script_path in node_scripts
        if not os.path.isfile(script_path)
    ]
    document_ids = get_input_document_ids()
    document_states = {
        document_id: get_document_phase_state(document_id)
        for document_id in document_ids
    }
    active_minos_model = os.path.join(MODELS_DIR, "minos_v2_0_best.keras")
    expert_state = get_n05_expert_state()
    git_state = get_git_status_summary()

    structural_ready = (
        not missing_nodes
        and os.path.isfile(BATCH_SCRIPT)
        and os.path.isfile(active_minos_model)
        and os.path.isdir(INPUT_DOCUMENTS_DIR)
    )
    all_outputs_complete = bool(document_states) and all(
        all(
            phase_state[phase_name]
            for phase_name in ("N00", "N01", "N02", "N03", "N04", "N05")
        )
        for phase_state in document_states.values()
    )

    if structural_ready and document_ids and all_outputs_complete:
        overall_status = "READY"
        overall_detail = "Core pipeline and current document outputs are complete."
    elif structural_ready and document_ids:
        overall_status = "READY"
        overall_detail = (
            "Core pipeline is operational; current document outputs are partial."
        )
    elif structural_ready:
        overall_status = "IDLE"
        overall_detail = "Core pipeline is ready; no input documents found."
    else:
        overall_status = "ATTENTION"
        overall_detail = "One or more required runtime components are missing."

    print("Image Processor System Status")
    print("-------------------------")
    print("Overall:", overall_status)
    print("Detail:", overall_detail)
    print("Python:", sys.executable)
    print(
        "Version:",
        f"{sys.version_info.major}.{sys.version_info.minor}."
        f"{sys.version_info.micro}",
    )
    print("Project:", BASE_DIR)

    print()
    print("Pipeline structure")
    print("-------------------------")
    for node_name, node_title, script_path in node_scripts:
        marker = "OK" if os.path.isfile(script_path) else "MISSING"
        print(f"[{marker:7}] {node_name}  {node_title}")
    batch_marker = "OK" if os.path.isfile(BATCH_SCRIPT) else "MISSING"
    print(f"[{batch_marker:7}] LPH  Batch orchestration")

    print()
    print("Document progress")
    print("-------------------------")
    if not document_ids:
        print("No supported documents in handwritten_text.")
    else:
        header = (
            f"{'DOCUMENT':<18} "
            + " ".join(f"{name:>6}" for name in (
                "N00",
                "N01",
                "N02",
                "N03",
                "N04",
                "N05",
                "RESULT",
            ))
        )
        print(header)
        print("-" * len(header))
        for document_id in document_ids:
            phase_state = document_states[document_id]
            markers = " ".join(
                f"{'yes' if phase_state[name] else '-':>6}"
                for name in ("N00", "N01", "N02", "N03", "N04", "N05", "RESULT")
            )
            print(f"{document_id:<18} {markers}")

    phase_totals = {
        phase_name: sum(
            state[phase_name]
            for state in document_states.values()
        )
        for phase_name in ("N00", "N01", "N02", "N03", "N04", "N05", "RESULT")
    }
    if document_ids:
        print()
        print(
            "Coverage:",
            ", ".join(
                f"{phase_name} {count}/{len(document_ids)}"
                for phase_name, count in phase_totals.items()
            ),
        )

    model_extensions = {".keras", ".joblib", ".pt", ".pth", ".onnx"}
    model_files = []
    if os.path.isdir(MODELS_DIR):
        for root, _, files in os.walk(MODELS_DIR):
            for file_name in files:
                if os.path.splitext(file_name)[1].lower() in model_extensions:
                    model_files.append(os.path.join(root, file_name))

    print()
    print("Runtime assets")
    print("-------------------------")
    print("Input documents:", len(document_ids))
    print("Model artifacts:", len(model_files))
    print(
        "Active Minos:",
        "available" if os.path.isfile(active_minos_model) else "missing",
    )
    print(
        "N05 experts enabled:",
        f"{sum(expert_state.values())}/{len(expert_state)}",
    )
    for expert_name, enabled in expert_state.items():
        print(f"  {'ON ' if enabled else 'off'}  {expert_name}")

    print()
    print("Storage")
    print("-------------------------")
    for label, path in (
        ("Models", MODELS_DIR),
        ("Datasets", os.path.join(BASE_DIR, "datasets")),
        ("Temp processing", TEMP_PROCESSING_DIR),
        ("Final results", FINAL_RESULTS_DIR),
        ("Failed results", FAILED_RESULTS_DIR),
    ):
        print(f"{label:<18} {format_storage_size(get_directory_size(path)):>10}")

    disk_usage = shutil.disk_usage(BASE_DIR)
    print(
        f"{'Disk free':<18} "
        f"{format_storage_size(disk_usage.free):>10} "
        f"of {format_storage_size(disk_usage.total)}"
    )

    print()
    print("Repository")
    print("-------------------------")
    if not git_state["available"]:
        print("Git: not initialized")
    else:
        print("Branch:", git_state["branch"])
        print("Latest:", git_state["commit"])
        changed_count = git_state["changed_path_count"]
        print(
            "Working tree:",
            "clean" if changed_count == 0 else f"{changed_count} changed paths",
        )

    print()
    print("Quick actions")
    print("-------------------------")
    print("Health check:  python main.py doctor")
    print("Control room:  python main.py ui")
    print("Full pipeline: python main.py pipeline")


def train_model():
    """Train the Minos classifier model.
    
    Args:
        None.
    
    Returns:
        None.
    """
    print("Training Minos classifier model...")
    print("Script:", TRAIN_SCRIPT)
    print("-------------------------")

    run_python_script(TRAIN_SCRIPT, required_modules=["cv2", "tensorflow"])


def run_batch():
    """Run the ScribeMap workflow for many documents.
    
    Args:
        None.
    
    Returns:
        Batch summary or None depending on caller context.
    """
    print("Running batch processor (scribemap phase)...")
    print("Script:", BATCH_SCRIPT)
    print("Input folder:", INPUT_DOCUMENTS_DIR)
    print("-------------------------")
    run_python_script_with_args(BATCH_SCRIPT, ["--phase", "scribemap"])

def run_pipeline():
    """Run the default processing pipeline.
    
    Args:
        None.
    
    Returns:
        None.
    """
    print("Running full pipeline: N00 prep -> N01 scribemap -> N02 crop refinement -> N03 visual classification -> N04 printed OCR -> N05 handwriting experts...")
    print("Script:", BATCH_SCRIPT)
    print("Input folder:", INPUT_DOCUMENTS_DIR)
    print("-------------------------")
    run_python_script_with_args(BATCH_SCRIPT, ["--phase", "pipeline"], required_modules=["cv2", "tensorflow"])

def run_crop_refiner_node():
    """Run the N02 crop-refiner phase through batch processing.
    
    Args:
        None.
    
    Returns:
        None.
    """
    print("Running N02 crop-refiner phase on handwritten_text files...")
    print("Script:", BATCH_SCRIPT)
    print("-------------------------")
    run_python_script_with_args(BATCH_SCRIPT, ["--phase", "refine"])

def run_file_preparation_node():
    """Run the file-preparation phase through batch processing.
    
    Args:
        None.
    
    Returns:
        None.
    """
    print("Running prep phase on handwritten_text files...")
    print("Script:", BATCH_SCRIPT)
    print("-------------------------")
    run_python_script_with_args(BATCH_SCRIPT, ["--phase", "prep"])

def run_scribemap_node():
    """Run the ScribeMap phase through batch processing.
    
    Args:
        None.
    
    Returns:
        None.
    """
    print("Running scribemap phase on handwritten_text files...")
    print("Script:", BATCH_SCRIPT)
    print("-------------------------")
    run_python_script_with_args(BATCH_SCRIPT, ["--phase", "scribemap"])


def run_visual_classification_node():
    """Run the N03 visual-classification router through batch processing.

    Args:
        None.

    Returns:
        None.
    """
    print("Running N03 visual-classification phase on handwritten_text files...")
    print("Script:", BATCH_SCRIPT)
    print("Node:", VISUAL_CLASSIFICATION_SCRIPT)
    print("-------------------------")
    run_python_script_with_args(BATCH_SCRIPT, ["--phase", "visual"], required_modules=["cv2", "tensorflow"])


def run_printed_ocr_node():
    """Run the N04 printed-OCR phase through batch processing.

    Args:
        None.

    Returns:
        None.
    """
    print("Running N04 printed-OCR phase on handwritten_text files...")
    print("Script:", BATCH_SCRIPT)
    print("Node:", PRINTED_OCR_SCRIPT)
    print("-------------------------")
    run_python_script_with_args(BATCH_SCRIPT, ["--phase", "printed_ocr"], required_modules=["cv2"])


def run_n05_expert_node():
    """Run the N05 handwriting-expert orchestrator through batch processing.

    Args:
        None.

    Returns:
        None.
    """
    print("Running N05 handwriting-expert phase on handwritten_text files...")
    print("Script:", BATCH_SCRIPT)
    print("Node:", N05_ORCHESTRATOR_SCRIPT)
    print("-------------------------")
    run_python_script_with_args(BATCH_SCRIPT, ["--phase", "handwritten_ocr"])


def run_character_split_viewer():
    """Regenerate N05 character hypotheses and open their debug gallery.

    Args:
        None.

    Returns:
        None. The control-room server runs until interrupted with Ctrl+C.
    """
    run_n05_expert_node()
    run_pipeline_ui(
        initial_stage="n05",
        initial_query="character_unit_proposer/debug",
    )


def clean_outputs():
    # Keep these core folders untouched, per user request and project safety.
    """Remove generated output folders while preserving source assets.
    
    Args:
        None.
    
    Returns:
        None.
    """
    preserved_dirs = {
        "classifier_dataset_presence",
        "models",
        "scripts",
        "handwritten_text",
        ".git",
        ".venv",
        "CNN_environment.venv",
        ".vscode",
        ".codex",
        ".agents",
    }

    removed_paths = []

    # 1) Remove known generated top-level folders.
    generated_dirs = [
        FINAL_RESULTS_DIR,
        FAILED_RESULTS_DIR,
        TEMP_PROCESSING_DIR,
    ]

    # Also remove common generated output folders created during experiments.
    generated_globs = [
        f"{BASE_DIR}/*_output",
        f"{BASE_DIR}/*_test_output",
        f"{BASE_DIR}/test_*_output",
        f"{BASE_DIR}/mask_comparison_debug",
        f"{BASE_DIR}/new_scribemap_from_*",
        f"{BASE_DIR}/old_scribemap_*",
    ]

    for folder in generated_dirs:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            removed_paths.append(folder)

    for pattern in generated_globs:
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                shutil.rmtree(path)
                removed_paths.append(path)

    # 2) Remove caches recursively (safe cleanup).
    for root, dirs, files in os.walk(BASE_DIR):
        # Skip preserved heavy/source folders for speed and safety.
        dirs[:] = [d for d in dirs if d not in preserved_dirs]

        for dir_name in list(dirs):
            if dir_name == "__pycache__":
                cache_path = os.path.join(root, dir_name)
                shutil.rmtree(cache_path)
                removed_paths.append(cache_path)
                dirs.remove(dir_name)

        for file_name in files:
            if file_name.endswith(".pyc"):
                pyc_path = os.path.join(root, file_name)
                os.remove(pyc_path)
                removed_paths.append(pyc_path)

    # 3) Recreate default runtime folders expected by pipeline.
    for folder in [FINAL_RESULTS_DIR, FAILED_RESULTS_DIR, TEMP_PROCESSING_DIR]:
        os.makedirs(folder, exist_ok=True)

    print("Clean complete.")
    print("Preserved: classifier_dataset_presence, models")
    print("Removed items:", len(removed_paths))
    for path in removed_paths:
        print("-", path)


def ensure_project_folders():
    """Create top-level project folders if missing.
    
    Args:
        None.
    
    Returns:
        None.
    """
    folders = [
        DATASET_DIR,
        INPUT_DOCUMENTS_DIR,
        MODELS_DIR,
        FINAL_RESULTS_DIR,
        FAILED_RESULTS_DIR,
        TEMP_PROCESSING_DIR,
        SCRIPTS_DIR
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    for class_name in CLASS_FOLDERS:
        os.makedirs(f"{DATASET_DIR}/{class_name}", exist_ok=True)

    print("Project folders checked/created.")


def show_shell_completion(words_only=False):
    """Print Bash completion data or activation instructions.

    Args:
        words_only: Print only the space-separated command registry when true.

    Returns:
        None.
    """
    if words_only:
        print(" ".join(MAIN_COMMANDS))
        return

    completion_script = f"{SCRIPTS_DIR}/main_completion.bash"
    print("Bash completion script:", completion_script)
    print()
    print("Enable it in the current terminal:")
    print(f"  source {completion_script}")
    print()
    print("Then use either:")
    print("  ocr <Tab>")
    print("  python scripts/main.py <Tab>")
    print("  .venv/bin/python scripts/main.py <Tab>")


def main():
    """Parse the CLI command and run the requested action.
    
    Args:
        None.
    
    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Main controller for Armenian handwriting detection pipeline."
    )

    parser.add_argument(
        "command",
        choices=MAIN_COMMANDS,
        help="Command to run."
    )
    parser.add_argument(
        "--words",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    if args.command == "ui":
        run_pipeline_ui()

    elif args.command == "status":
        show_project_status()

    elif args.command == "counts":
        show_dataset_counts()

    elif args.command == "lines":
        show_project_line_counts()

    elif args.command == "doctor":
        if not run_project_doctor():
            raise SystemExit(1)

    elif args.command == "train":
        train_model()

    elif args.command == "batch":
        run_batch()

    elif args.command == "pipeline":
        run_pipeline()

    elif args.command == "prep":
        run_file_preparation_node()

    elif args.command == "scribemap":
        run_scribemap_node()

    elif args.command == "refine":
        run_crop_refiner_node()

    elif args.command in ["visual", "visual_classification", "n03"]:
        run_visual_classification_node()

    elif args.command in ["printed_ocr", "printed", "n04"]:
        run_printed_ocr_node()

    elif args.command in ["handwritten_ocr", "handwritten", "n05"]:
        run_n05_expert_node()

    elif args.command in ["splits", "letter_splits"]:
        run_character_split_viewer()

    elif args.command == "clean":
        clean_outputs()

    elif args.command == "setup":
        ensure_project_folders()

    elif args.command == "completion":
        show_shell_completion(words_only=args.words)


if __name__ == "__main__":
    main()
    
