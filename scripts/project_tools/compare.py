"""Compare pipeline JSON, image, or directory artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MAX_RECORDED_DIFFERENCES = 300


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_difference(differences: list[dict], path: str, kind: str, before=None, after=None) -> None:
    if len(differences) >= MAX_RECORDED_DIFFERENCES:
        return
    differences.append({"path": path or "$", "kind": kind, "before": before, "after": after})


def _compare_json_values(before: Any, after: Any, path: str, differences: list[dict]) -> int:
    """Recursively compare JSON-safe values and return the full difference count."""
    if type(before) is not type(after):
        _record_difference(differences, path, "type_changed", type(before).__name__, type(after).__name__)
        return 1
    if isinstance(before, dict):
        count = 0
        before_keys = set(before)
        after_keys = set(after)
        for key in sorted(before_keys - after_keys):
            _record_difference(differences, f"{path}.{key}", "removed", before[key], None)
            count += 1
        for key in sorted(after_keys - before_keys):
            _record_difference(differences, f"{path}.{key}", "added", None, after[key])
            count += 1
        for key in sorted(before_keys & after_keys):
            count += _compare_json_values(before[key], after[key], f"{path}.{key}", differences)
        return count
    if isinstance(before, list):
        count = abs(len(before) - len(after))
        if len(before) != len(after):
            _record_difference(differences, path, "length_changed", len(before), len(after))
        for index, (before_item, after_item) in enumerate(zip(before, after)):
            count += _compare_json_values(before_item, after_item, f"{path}[{index}]", differences)
        return count
    if before != after:
        _record_difference(differences, path, "value_changed", before, after)
        return 1
    return 0


def _compare_json_files(before_path: Path, after_path: Path) -> dict:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    differences: list[dict] = []
    total = _compare_json_values(before, after, "$", differences)
    return {
        "kind": "json",
        "identical": total == 0,
        "difference_count": total,
        "differences": differences,
        "differences_truncated": total > len(differences),
    }


def _ink_mask(grayscale):
    import cv2
    import numpy as np

    _, dark_ink = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, light_ink = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return dark_ink > 0 if np.count_nonzero(dark_ink) <= np.count_nonzero(light_ink) else light_ink > 0


def _compare_image_files(before_path: Path, after_path: Path) -> dict:
    import cv2
    import numpy as np

    before = cv2.imread(str(before_path), cv2.IMREAD_GRAYSCALE)
    after = cv2.imread(str(after_path), cv2.IMREAD_GRAYSCALE)
    if before is None or after is None:
        raise ValueError("OpenCV could not read one of the compared images.")
    if before.shape != after.shape:
        return {
            "kind": "image",
            "identical": False,
            "shape_before": list(before.shape),
            "shape_after": list(after.shape),
            "shape_changed": True,
        }

    absolute = cv2.absdiff(before, after)
    changed = absolute > 0
    before_ink = _ink_mask(before)
    after_ink = _ink_mask(after)
    return {
        "kind": "image",
        "identical": not bool(np.any(changed)),
        "shape_before": list(before.shape),
        "shape_after": list(after.shape),
        "shape_changed": False,
        "changed_pixel_count": int(np.count_nonzero(changed)),
        "changed_pixel_ratio": float(np.count_nonzero(changed) / max(1, changed.size)),
        "mean_absolute_difference": float(np.mean(absolute)),
        "maximum_absolute_difference": int(np.max(absolute)),
        "ink_pixels_before": int(np.count_nonzero(before_ink)),
        "ink_pixels_after": int(np.count_nonzero(after_ink)),
        "added_ink_pixels": int(np.count_nonzero(after_ink & ~before_ink)),
        "removed_ink_pixels": int(np.count_nonzero(before_ink & ~after_ink)),
    }


def _compare_files(before_path: Path, after_path: Path) -> dict:
    result = {
        "before": str(before_path),
        "after": str(after_path),
        "size_before": before_path.stat().st_size,
        "size_after": after_path.stat().st_size,
    }
    suffix = before_path.suffix.lower()
    if suffix == ".json" and after_path.suffix.lower() == ".json":
        result.update(_compare_json_files(before_path, after_path))
    elif suffix in IMAGE_EXTENSIONS and after_path.suffix.lower() in IMAGE_EXTENSIONS:
        result.update(_compare_image_files(before_path, after_path))
    else:
        before_hash = _sha256(before_path)
        after_hash = _sha256(after_path)
        result.update({
            "kind": "binary",
            "identical": before_hash == after_hash,
            "sha256_before": before_hash,
            "sha256_after": after_hash,
        })
    return result


def _directory_files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def _compare_directories(before_path: Path, after_path: Path) -> dict:
    before_files = _directory_files(before_path)
    after_files = _directory_files(after_path)
    before_names = set(before_files)
    after_names = set(after_files)
    comparisons = []
    errors = []
    for relative_path in sorted(before_names & after_names):
        try:
            comparison = _compare_files(before_files[relative_path], after_files[relative_path])
            comparison["relative_path"] = relative_path
            comparisons.append(comparison)
        except Exception as error:
            errors.append({"relative_path": relative_path, "error": str(error)})
    changed = sum(not item.get("identical", False) for item in comparisons)
    return {
        "kind": "directory",
        "identical": not changed and before_names == after_names and not errors,
        "common_file_count": len(comparisons),
        "changed_file_count": changed,
        "unchanged_file_count": len(comparisons) - changed,
        "removed_files": sorted(before_names - after_names),
        "added_files": sorted(after_names - before_names),
        "errors": errors,
        "file_comparisons": [item for item in comparisons if not item.get("identical", False)],
    }


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Comparison path does not exist: {path}")
    return path


def compare_paths(before: str, after: str, base_dir: str | Path, output_path: str | None = None) -> dict:
    """Compare two files or directories and persist a structured report."""
    base = Path(base_dir).resolve()
    before_path = _resolve_path(before, base)
    after_path = _resolve_path(after, base)
    if before_path.is_dir() != after_path.is_dir():
        raise ValueError("Both comparison targets must be files or both must be directories.")

    comparison = (
        _compare_directories(before_path, after_path)
        if before_path.is_dir()
        else _compare_files(before_path, after_path)
    )
    report = {
        "version": "pipeline-compare-v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "before": str(before_path),
        "after": str(after_path),
        "comparison": comparison,
    }
    if output_path:
        destination = _resolve_output_path(output_path, base)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = base / "reports" / "comparisons" / f"compare_{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(destination)
    _print_comparison_summary(report)
    return report


def _resolve_output_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve()


def _print_comparison_summary(report: dict) -> None:
    comparison = report["comparison"]
    print("Pipeline Compare")
    print("----------------")
    print("Before:", report["before"])
    print("After: ", report["after"])
    print("Result:", "IDENTICAL" if comparison.get("identical") else "CHANGED")
    if comparison.get("kind") == "directory":
        print("Changed files:", comparison["changed_file_count"])
        print("Added files:  ", len(comparison["added_files"]))
        print("Removed files:", len(comparison["removed_files"]))
    elif comparison.get("kind") == "json":
        print("JSON differences:", comparison["difference_count"])
    elif comparison.get("kind") == "image":
        print("Changed pixels:", comparison.get("changed_pixel_count", "shape changed"))
        print("Added ink:    ", comparison.get("added_ink_pixels", "n/a"))
        print("Removed ink:  ", comparison.get("removed_ink_pixels", "n/a"))
    print("Report:", report["report_path"])
