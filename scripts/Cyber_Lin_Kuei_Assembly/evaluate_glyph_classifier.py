from pathlib import Path
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    top_k_accuracy_score,
)


# ============================================================
# Config
# ============================================================

PROJECT_ROOT = Path("/home/vahram/Desktop/image_Processor")

DATASET_DIR = PROJECT_ROOT / "Matenadata"
LABEL_MAP_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "N05handwritten_ocr"
    / "character_detector"
    / "numeric_label_map.json"
)

MODEL_NAME = "glyph_classifier_v0_3_white_ink"

MODEL_PATH = PROJECT_ROOT / "models" / MODEL_NAME / f"{MODEL_NAME}_best.pt"
REPORT_DIR = PROJECT_ROOT / "reports" / MODEL_NAME

IMAGE_SIZE = 64
NUM_CLASSES = 78
BATCH_SIZE = 64
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"
}


# ============================================================
# IO
# ============================================================

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# Data
# ============================================================

def collect_samples(dataset_dir: Path):
    samples = []

    for class_id in range(NUM_CLASSES):
        class_dir = dataset_dir / str(class_id)

        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((str(path), class_id))

    return samples


def resize_with_padding(image: Image.Image, size: int = 64) -> Image.Image:
    """Resize a white-ink-on-black glyph while preserving black padding."""

    image = image.convert("L")

    width, height = image.size

    scale = min(size / width, size / height)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    image = image.resize((new_width, new_height), Image.BILINEAR)

    canvas = Image.new("L", (size, size), color=0)

    x_offset = (size - new_width) // 2
    y_offset = (size - new_height) // 2

    canvas.paste(image, (x_offset, y_offset))

    return canvas


def threshold_to_binary(image: Image.Image, threshold: int = 128) -> Image.Image:
    """Convert white ink to 255 and black background to 0."""

    arr = np.array(image.convert("L"), dtype=np.uint8)
    binary = np.where(arr >= threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


class GlyphDataset(Dataset):
    def __init__(self, samples, image_size: int = 64, binary_threshold: int = 128):
        self.samples = samples
        self.image_size = image_size
        self.binary_threshold = int(binary_threshold)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        image = Image.open(path)
        image = resize_with_padding(image, self.image_size)
        image = threshold_to_binary(image, self.binary_threshold)

        arr = np.array(image).astype(np.float32) / 255.0

        # Current glyph trainer contract: white ink = 1, black background = 0.

        tensor = torch.from_numpy(arr).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return tensor, label_tensor


# ============================================================
# Model — must match training architecture exactly
# ============================================================

class GlyphClassifier(nn.Module):
    def __init__(self, num_classes: int = 78):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_logits = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            predictions = torch.argmax(logits, dim=1)

            all_logits.append(logits.detach().cpu().numpy())
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)

    average_loss = total_loss / len(dataloader.dataset)

    top1 = accuracy_score(all_labels, all_predictions)

    top5 = top_k_accuracy_score(
        all_labels,
        all_logits,
        k=5,
        labels=list(range(NUM_CLASSES)),
    )

    return average_loss, top1, top5, all_labels, all_predictions, all_logits


def save_top_confusions(y_true, y_pred, label_map, max_items: int = 100):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    rows = []

    for true_id in range(NUM_CLASSES):
        total_true = int(cm[true_id].sum())

        if total_true == 0:
            continue

        for pred_id in range(NUM_CLASSES):
            if true_id == pred_id:
                continue

            count = int(cm[true_id][pred_id])

            if count == 0:
                continue

            rows.append({
                "true_id": true_id,
                "true_label": label_map[str(true_id)],
                "predicted_id": pred_id,
                "predicted_label": label_map[str(pred_id)],
                "count": count,
                "percent_of_true_class": round(count / total_true, 4),
            })

    rows = sorted(rows, key=lambda x: x["count"], reverse=True)

    save_json(
        {
            "max_items": max_items,
            "confusions": rows[:max_items],
        },
        REPORT_DIR / "top_confusions.json",
    )

    return rows[:max_items]


def save_per_class_stats(y_true, y_pred, label_map):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    stats = []

    for class_id in range(NUM_CLASSES):
        true_total = int(cm[class_id].sum())
        predicted_total = int(cm[:, class_id].sum())
        correct = int(cm[class_id, class_id])

        recall = correct / true_total if true_total else 0.0
        precision = correct / predicted_total if predicted_total else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        stats.append({
            "class_id": class_id,
            "label": label_map[str(class_id)],
            "true_total": true_total,
            "predicted_total": predicted_total,
            "correct": correct,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    stats_sorted_by_f1_low = sorted(stats, key=lambda x: x["f1"])
    stats_sorted_by_f1_high = sorted(stats, key=lambda x: x["f1"], reverse=True)

    save_json(
        {
            "classes": stats,
            "weakest_by_f1": stats_sorted_by_f1_low[:20],
            "strongest_by_f1": stats_sorted_by_f1_high[:20],
        },
        REPORT_DIR / "per_class_stats.json",
    )

    return stats_sorted_by_f1_low[:20], stats_sorted_by_f1_high[:20]


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    label_map = load_json(LABEL_MAP_PATH)

    print("=" * 70)
    print("Evaluating saved best glyph classifier")
    print("=" * 70)

    print(f"Model path: {MODEL_PATH}")

    samples = collect_samples(DATASET_DIR)

    paths = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]

    # Must match the original training split:
    # 80 train, 10 validation, 10 test with same RANDOM_SEED.
    train_paths, temporary_paths, train_labels, temporary_labels = train_test_split(
        paths,
        labels,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temporary_paths,
        temporary_labels,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temporary_labels,
    )

    test_samples = list(zip(test_paths, test_labels))

    test_dataset = GlyphDataset(test_samples, IMAGE_SIZE)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GlyphClassifier(NUM_CLASSES).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    test_loss, test_top1, test_top5, y_true, y_pred, y_logits = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    np.savetxt(
        REPORT_DIR / "confusion_matrix.csv",
        cm,
        fmt="%d",
        delimiter=",",
    )

    class_names = [label_map[str(i)] for i in range(NUM_CLASSES)]

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    (REPORT_DIR / "classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    weakest, strongest = save_per_class_stats(y_true, y_pred, label_map)
    top_confusions = save_top_confusions(y_true, y_pred, label_map, max_items=100)

    summary = {
        "model_name": MODEL_NAME,
        "model_path": str(MODEL_PATH),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_top1": checkpoint.get("val_top1"),
        "checkpoint_val_top5": checkpoint.get("val_top5"),
        "test_loss": float(test_loss),
        "test_top1": float(test_top1),
        "test_top5": float(test_top5),
        "test_samples": len(test_samples),
        "outputs": {
            "confusion_matrix": str(REPORT_DIR / "confusion_matrix.csv"),
            "classification_report": str(REPORT_DIR / "classification_report.txt"),
            "per_class_stats": str(REPORT_DIR / "per_class_stats.json"),
            "top_confusions": str(REPORT_DIR / "top_confusions.json"),
        },
    }

    save_json(summary, REPORT_DIR / "evaluation_summary.json")

    print()
    print("Evaluation complete.")
    print(f"Checkpoint epoch:      {checkpoint.get('epoch')}")
    print(f"Checkpoint val top1:   {checkpoint.get('val_top1')}")
    print(f"Checkpoint val top5:   {checkpoint.get('val_top5')}")
    print(f"Test loss:             {test_loss:.4f}")
    print(f"Test top1:             {test_top1:.4f}")
    print(f"Test top5:             {test_top5:.4f}")

    print()
    print("Weakest classes by F1:")
    for item in weakest[:10]:
        print(
            f"  {item['class_id']:02d} {item['label']} | "
            f"precision={item['precision']} recall={item['recall']} f1={item['f1']}"
        )

    print()
    print("Strongest classes by F1:")
    for item in strongest[:10]:
        print(
            f"  {item['class_id']:02d} {item['label']} | "
            f"precision={item['precision']} recall={item['recall']} f1={item['f1']}"
        )

    print()
    print("Top confusions:")
    for item in top_confusions[:15]:
        print(
            f"  {item['true_label']} -> {item['predicted_label']} | "
            f"count={item['count']} "
            f"percent={item['percent_of_true_class']}"
        )

    print()
    print(f"Saved confusion matrix: {REPORT_DIR / 'confusion_matrix.csv'}")
    print(f"Saved class report:     {REPORT_DIR / 'classification_report.txt'}")
    print(f"Saved per-class stats:  {REPORT_DIR / 'per_class_stats.json'}")
    print(f"Saved top confusions:   {REPORT_DIR / 'top_confusions.json'}")


if __name__ == "__main__":
    main()
