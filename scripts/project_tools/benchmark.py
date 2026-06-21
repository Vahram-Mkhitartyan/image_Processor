"""Deterministic geometry regression benchmark for ScribeTrace."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from datetime import datetime
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
BENCHMARK_SETTINGS = {
    "enabled": True,
    "save_debug": False,
    "save_json": False,
    "enable_mask_repair": False,
    "ink_threshold_mode": "otsu",
    "minimum_ink_pixels": 4,
    "enable_theoretical_reconstruction": False,
}


def _natural_key(value: str) -> tuple:
    import re
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _representatives(dataset_dir: Path) -> list[tuple[str, Path]]:
    representatives = []
    for class_dir in sorted((path for path in dataset_dir.iterdir() if path.is_dir()), key=lambda path: _natural_key(path.name)):
        images = sorted(
            (path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: _natural_key(path.name),
        )
        if images:
            representatives.append((class_dir.name, images[0]))
    return representatives


def _stable_vector_hash(feature_vector) -> str | None:
    if feature_vector is None:
        return None
    payload = {
        "feature_names": list(feature_vector.feature_names),
        "vector": [round(float(value), 9) for value in feature_vector.vector],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _run_case(base_dir: Path, class_label: str, image_path: Path) -> dict:
    from N05handwritten_ocr.scribetrace.expert import run_scribetrace
    from N05handwritten_ocr.scribetrace.trace_models import TraceInput

    relative_path = image_path.relative_to(base_dir).as_posix()
    started = time.perf_counter()
    result = run_scribetrace(
        TraceInput(
            crop_path=str(image_path),
            mask_crop_path=str(image_path),
            visual_crop_path=str(image_path),
            output_dir=str(base_dir / "temp_processing" / "benchmark" / class_label),
            text_unit_id=f"benchmark_{class_label}",
        ),
        settings=BENCHMARK_SETTINGS,
    )
    elapsed = time.perf_counter() - started
    observation = result.metrics.get("scrilog_observation") or {}
    return {
        "class_label": class_label,
        "source_path": relative_path,
        "source_sha256": _sha256(image_path),
        "status": result.status,
        "reason": result.reason,
        "elapsed_seconds": elapsed,
        "feature_count": len(result.feature_vector.feature_names) if result.feature_vector else 0,
        "feature_names": list(result.feature_vector.feature_names) if result.feature_vector else [],
        "feature_vector_hash": _stable_vector_hash(result.feature_vector),
        "scrilog_observation": observation,
    }


def _collect(base_dir: Path, source_cases: list[dict] | None = None) -> dict:
    dataset_dir = base_dir / "Matenadata"
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Matenadata directory is missing: {dataset_dir}")
    if source_cases is None:
        sources = _representatives(dataset_dir)
    else:
        sources = [
            (str(case["class_label"]), base_dir / case["source_path"])
            for case in source_cases
        ]
    cases = []
    for index, (class_label, image_path) in enumerate(sources, start=1):
        if not image_path.is_file():
            cases.append({"class_label": class_label, "source_path": str(image_path), "error": "source_missing"})
            continue
        print(f"[{index:02d}/{len(sources):02d}] class {class_label}: {image_path.name}")
        try:
            cases.append(_run_case(base_dir, class_label, image_path))
        except Exception as error:
            cases.append({
                "class_label": class_label,
                "source_path": image_path.relative_to(base_dir).as_posix(),
                "error": str(error),
            })
    durations = [case["elapsed_seconds"] for case in cases if "elapsed_seconds" in case]
    return {
        "case_count": len(cases),
        "successful_case_count": len(durations),
        "median_elapsed_seconds": statistics.median(durations) if durations else None,
        "total_elapsed_seconds": sum(durations),
        "cases": cases,
    }


def _case_differences(expected: dict, current: dict) -> list[dict]:
    differences = []
    for key in ("source_sha256", "status", "reason", "feature_count", "feature_names", "feature_vector_hash", "scrilog_observation"):
        if expected.get(key) != current.get(key):
            differences.append({"field": key, "expected": expected.get(key), "current": current.get(key)})
    return differences


def run_scribetrace_benchmark(base_dir: str | Path, update: bool = False) -> bool:
    """Create or verify the deterministic one-glyph-per-class baseline."""
    base = Path(base_dir).resolve()
    baseline_path = base / "benchmarks" / "scribetrace_geometry_baseline.json"
    reports_dir = base / "reports" / "benchmarks"
    if update:
        current = _collect(base)
        baseline = {
            "version": "scribetrace-geometry-benchmark-v1",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "settings": BENCHMARK_SETTINGS,
            **current,
        }
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Benchmark baseline updated:", baseline_path)
        print("Cases:", baseline["successful_case_count"], "/", baseline["case_count"])
        return baseline["successful_case_count"] == baseline["case_count"]

    if not baseline_path.is_file():
        print("Benchmark baseline is missing.")
        print("Create it with: python scripts/main.py benchmark --update")
        return False

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current = _collect(base, baseline.get("cases") or [])
    current_by_class = {case.get("class_label"): case for case in current["cases"]}
    regressions = []
    for expected in baseline.get("cases", []):
        class_label = expected.get("class_label")
        actual = current_by_class.get(class_label, {"error": "case_missing"})
        differences = _case_differences(expected, actual)
        if differences or actual.get("error"):
            regressions.append({
                "class_label": class_label,
                "source_path": expected.get("source_path"),
                "error": actual.get("error"),
                "differences": differences,
            })

    baseline_median = baseline.get("median_elapsed_seconds")
    current_median = current.get("median_elapsed_seconds")
    performance_warning = None
    if baseline_median and current_median and current_median > baseline_median * 1.5:
        performance_warning = {
            "baseline_median_seconds": baseline_median,
            "current_median_seconds": current_median,
            "ratio": current_median / baseline_median,
        }

    report = {
        "version": "scribetrace-geometry-benchmark-report-v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "baseline_path": str(baseline_path),
        "passed": not regressions,
        "regression_count": len(regressions),
        "performance_warning": performance_warning,
        "current": current,
        "regressions": regressions,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "latest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("ScribeTrace Geometry Benchmark")
    print("------------------------------")
    print("Cases:", current["successful_case_count"], "/", current["case_count"])
    print("Regressions:", len(regressions))
    print("Median time:", f"{current_median:.4f}s" if current_median is not None else "n/a")
    if performance_warning:
        print("Performance warning:", f"{performance_warning['ratio']:.2f}x baseline")
    print("Result:", "PASS" if not regressions else "FAIL")
    print("Report:", report_path)
    return not regressions
