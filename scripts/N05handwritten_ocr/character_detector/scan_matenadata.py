from pathlib import Path
import json
from PIL import Image
from collections import Counter


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def scan_dataset(dataset_dir: Path, label_map_path: Path, output_dir: Path) -> dict:
    label_map = load_json(label_map_path)

    expected_folders = {str(i) for i in range(78)}
    found_folders = {p.name for p in dataset_dir.iterdir() if p.is_dir()}

    missing_folders = sorted(expected_folders - found_folders, key=int)
    extra_folders = sorted(found_folders - expected_folders)

    class_reports = []
    total_images = 0
    bad_images = []

    global_size_counter = Counter()

    for class_id in sorted(expected_folders, key=int):
        class_dir = dataset_dir / class_id
        label = label_map.get(class_id)

        image_files = []
        size_counter = Counter()

        if class_dir.exists():
            for file_path in class_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_files.append(file_path)

                    try:
                        with Image.open(file_path) as img:
                            size = img.size
                            size_counter[f"{size[0]}x{size[1]}"] += 1
                            global_size_counter[f"{size[0]}x{size[1]}"] += 1
                    except Exception as e:
                        bad_images.append({
                            "path": str(file_path),
                            "class_id": class_id,
                            "label": label,
                            "error": str(e)
                        })

        count = len(image_files)
        total_images += count

        class_reports.append({
            "class_id": int(class_id),
            "label": label,
            "count": count,
            "top_sizes": dict(size_counter.most_common(10))
        })

    counts = [r["count"] for r in class_reports]

    report = {
        "dataset_dir": str(dataset_dir),
        "label_map_path": str(label_map_path),
        "num_expected_classes": 78,
        "num_found_class_folders": len(found_folders & expected_folders),
        "total_images": total_images,
        "missing_folders": missing_folders,
        "extra_folders": extra_folders,
        "bad_images_count": len(bad_images),
        "bad_images": bad_images[:200],
        "class_count_summary": {
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "average": round(sum(counts) / len(counts), 2) if counts else 0
        },
        "global_top_sizes": dict(global_size_counter.most_common(20)),
        "classes": class_reports
    }

    save_json(report, output_dir / "dataset_scan_report.json")

    # Easy human-readable class counts table.
    lines = ["class_id,label,count"]
    for r in class_reports:
        lines.append(f'{r["class_id"]},{r["label"]},{r["count"]}')

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "class_counts.csv").write_text("\n".join(lines), encoding="utf-8")

    return report


if __name__ == "__main__":
    dataset_dir = Path("Matenadata")
    label_map_path = Path(__file__).resolve().with_name("numeric_label_map.json")
    output_dir = Path("scan_report")

    report = scan_dataset(dataset_dir, label_map_path, output_dir)

    print("Dataset scan complete.")
    print(f"Total images: {report['total_images']}")
    print(f"Found class folders: {report['num_found_class_folders']}/78")
    print(f"Missing folders: {report['missing_folders']}")
    print(f"Extra folders: {report['extra_folders']}")
    print(f"Bad images: {report['bad_images_count']}")
    print(f"Min class count: {report['class_count_summary']['min']}")
    print(f"Max class count: {report['class_count_summary']['max']}")
    print(f"Average class count: {report['class_count_summary']['average']}")
