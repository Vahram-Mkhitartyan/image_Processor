from pathlib import Path
import argparse
import json
import random
import shutil
from collections import Counter

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

try:
    from Aristotel.recipes import build_default_recipes
except Exception:
    build_default_recipes = None


# ============================================================
# Config
# ============================================================

PROJECT_ROOT = Path("/home/vahram/Desktop/image_Processor")

# Sacred read-only dataset archive.
# Only read image folders 0–77 from here. Do not write anything here.
DATASET_DIR = PROJECT_ROOT / "Matenadata"

# Label map belongs to the N05 character detector expert.
LABEL_MAP_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "N05handwritten_ocr"
    / "character_detector"
    / "numeric_label_map.json"
)

MODEL_NAME = "glyph_classifier_v0_3_white_ink"

# All generated training artifacts go outside Matenadata.
OUTPUT_DIR = PROJECT_ROOT / "models" / MODEL_NAME
REPORT_DIR = PROJECT_ROOT / "reports" / MODEL_NAME

IMAGE_SIZE = 64
NUM_CLASSES = 78

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 3e-4

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

INPUT_POLARITY_MODE = "white_ink_on_black__ink_1_background_0"


# ============================================================
# Utilities
# ============================================================

def seed_everything(seed: int = 42) -> None:
    """Seed all normal random sources used by this training script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path) -> None:
    """Save readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def ensure_dirs() -> None:
    """Create model/report output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def collect_samples(dataset_dir: Path):
    """
    Expected dataset structure:

    Matenadata/
      0/
      1/
      ...
      77/
      label_maps/
        numeric_label_map.json

    Only folders 0-77 are treated as classes.
    """
    samples = []
    class_counts = Counter()

    for class_id in range(NUM_CLASSES):
        class_dir = dataset_dir / str(class_id)

        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        image_paths = []

        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path)

        image_paths = sorted(image_paths)

        for path in image_paths:
            samples.append((str(path), class_id))
            class_counts[str(class_id)] += 1

    return samples, class_counts


def resize_with_padding(image: Image.Image, size: int = 64) -> Image.Image:
    """
    Convert to grayscale, resize while preserving aspect ratio,
    and pad to a square canvas.

    Contract:
      source polarity is white ink on black background.

    If the image is already 64x64, this returns it unchanged after grayscale
    conversion. That avoids unnecessary interpolation on the Matenadata glyphs.
    """
    image = image.convert("L")

    width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError("Invalid image with zero width or height")

    if width == size and height == size:
        return image

    scale = min(size / width, size / height)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    image = image.resize((new_width, new_height), Image.BILINEAR)

    # Black background because source contract is white ink on black.
    canvas = Image.new("L", (size, size), color=0)

    x_offset = (size - new_width) // 2
    y_offset = (size - new_height) // 2

    canvas.paste(image, (x_offset, y_offset))

    return canvas


def threshold_to_binary(image: Image.Image, threshold: int = 128) -> Image.Image:
    """
    Convert a grayscale white-ink-on-black glyph to strict binary pixels.

    Contract:
      - background = black = 0
      - ink = white = 255

    Matenadata glyphs are white ink on black background, but the ink pixels
    may not all be equally white. Thresholding normalizes them.
    """
    arr = np.array(image.convert("L"), dtype=np.uint8)

    # Bright enough = ink, otherwise background.
    binary = np.where(arr >= threshold, 255, 0).astype(np.uint8)

    return Image.fromarray(binary, mode="L")


def build_aristotel_recipe_pool(recipe_names: list[str] | None = None):
    """
    Load the shared Aristotel damage recipes used by the geometry experts.

    Args:
        recipe_names: Optional list of recipe names to keep. None or ["all"]
            keeps every default recipe.

    Returns:
        A deterministic list of DamageRecipe objects.
    """
    if build_default_recipes is None:
        raise RuntimeError(
            "Aristotel recipes could not be imported. Run from the project root "
            "with the project virtual environment active."
        )

    recipes = build_default_recipes()
    recipes = sorted(recipes, key=lambda recipe: recipe.name)

    if not recipe_names or recipe_names == ["all"]:
        return recipes

    wanted = set(recipe_names)
    selected = [recipe for recipe in recipes if recipe.name in wanted]
    missing = sorted(wanted - {recipe.name for recipe in selected})

    if missing:
        raise ValueError(f"Unknown Aristotel recipe names: {', '.join(missing)}")

    return selected


def apply_aristotel_recipe(
    image: Image.Image,
    recipe,
    seed: int,
) -> Image.Image:
    """
    Apply one Aristotel damage recipe to a binary glyph image.

    Input/output contract:
      - white ink = 255
      - black background = 0

    Args:
        image: PIL grayscale glyph.
        recipe: DamageRecipe object.
        seed: Deterministic per-sample seed.

    Returns:
        A PIL grayscale image with the same polarity.
    """
    rng = np.random.default_rng(seed)
    arr = np.array(image.convert("L"), dtype=np.uint8)
    damaged, _ = recipe.apply(arr, rng)
    damaged = np.asarray(damaged, dtype=np.uint8)
    return Image.fromarray(damaged, mode="L")


def make_split_report(samples, train_samples, val_samples, test_samples):
    """Build a compact dataset split report."""
    return {
        "total": len(samples),
        "train": len(train_samples),
        "validation": len(val_samples),
        "test": len(test_samples),
        "train_ratio": round(len(train_samples) / len(samples), 4),
        "validation_ratio": round(len(val_samples) / len(samples), 4),
        "test_ratio": round(len(test_samples) / len(samples), 4),
    }


# ============================================================
# Dataset
# ============================================================

class GlyphDataset(Dataset):
    """
    Dataset for 78-class Armenian glyph classification.

    Source image contract:
      - white ink on black background
      - usually already 64x64

    Tensor contract:
      - ink = 1.0
      - background = 0.0
      - shape = [1, 64, 64]
    """

    def __init__(
        self,
        samples,
        image_size: int = 64,
        aristotel_recipes=None,
        aristotel_variants_per_sample: int = 0,
        seed: int = 42,
        binary_threshold: int = 128,
    ):
        self.samples = samples
        self.image_size = image_size
        self.aristotel_recipes = aristotel_recipes or []
        self.aristotel_variants_per_sample = max(0, int(aristotel_variants_per_sample))
        self.seed = int(seed)
        self.binary_threshold = int(binary_threshold)
        self.views_per_sample = 1 + self.aristotel_variants_per_sample

    def __len__(self):
        return len(self.samples) * self.views_per_sample

    def __getitem__(self, index):
        sample_index = index // self.views_per_sample
        view_index = index % self.views_per_sample
        path, label = self.samples[sample_index]

        try:
            image = Image.open(path)
            image = resize_with_padding(image, self.image_size)
            image = threshold_to_binary(image, self.binary_threshold)

            # view_index 0 is always clean. Later views are deterministic
            # Aristotel variants of the same labeled glyph.
            if view_index > 0 and self.aristotel_recipes:
                recipe_index = (view_index - 1) % len(self.aristotel_recipes)
                recipe = self.aristotel_recipes[recipe_index]
                damage_seed = (
                    self.seed
                    + sample_index * 1009
                    + view_index * 9176
                    + recipe_index * 131
                )
                image = apply_aristotel_recipe(image, recipe, damage_seed)
                image = threshold_to_binary(image, self.binary_threshold)

            arr = np.array(image).astype(np.float32) / 255.0

            # IMPORTANT:
            # Source glyphs are white ink on black background.
            # After thresholding:
            #   ink = 255 -> 1.0
            #   background = 0 -> 0.0
            # Therefore: DO NOT INVERT.
            tensor = torch.from_numpy(arr).unsqueeze(0)

            label_tensor = torch.tensor(label, dtype=torch.long)

            return tensor, label_tensor

        except Exception as e:
            raise RuntimeError(f"Failed to load image: {path}. Error: {e}")


# ============================================================
# Model
# ============================================================

class GlyphClassifier(nn.Module):
    """
    Simple CNN baseline.

    Goal:
    - Armenian glyph top-k candidate generation
    - not final N05 decision
    - not trusted on unsafe/non-character segment crops
    """

    def __init__(self, num_classes: int = 78):
        super().__init__()

        self.features = nn.Sequential(
            # Input: [B, 1, 64, 64]

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # [B, 32, 32, 32]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # [B, 64, 16, 16]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # [B, 128, 8, 8]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            # [B, 256, 1, 1]
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
# Train / Eval
# ============================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    average_loss = total_loss / len(dataloader.dataset)
    top1_accuracy = accuracy_score(all_labels, all_predictions)

    return average_loss, top1_accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model and return loss/top-k metrics plus raw outputs."""
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

    top1_accuracy = accuracy_score(all_labels, all_predictions)

    top5_accuracy = top_k_accuracy_score(
        all_labels,
        all_logits,
        k=5,
        labels=list(range(NUM_CLASSES)),
    )

    return (
        average_loss,
        top1_accuracy,
        top5_accuracy,
        all_labels,
        all_predictions,
        all_logits,
    )


def get_top_k_predictions(logits: np.ndarray, k: int = 5):
    """
    Return top-k class indexes and probabilities for each sample.

    Used later when N05 needs candidate letters, not only one winner.
    """
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    probabilities = torch.softmax(logits_tensor, dim=1).numpy()

    top_indexes = np.argsort(probabilities, axis=1)[:, ::-1][:, :k]
    top_probs = np.take_along_axis(probabilities, top_indexes, axis=1)

    return top_indexes, top_probs


def save_confusion_outputs(y_true, y_pred, label_map):
    """Save confusion matrix and classification report."""
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


def save_topk_examples(y_true, y_logits, label_map, k: int = 5, max_examples: int = 300):
    """Save a small top-k prediction probe report."""
    top_indexes, top_probs = get_top_k_predictions(y_logits, k=k)

    examples = []

    limit = min(len(y_true), max_examples)

    for i in range(limit):
        true_class = int(y_true[i])

        candidates = []

        for class_index, prob in zip(top_indexes[i], top_probs[i]):
            class_index = int(class_index)

            candidates.append({
                "class_id": class_index,
                "label": label_map[str(class_index)],
                "probability": float(prob),
            })

        examples.append({
            "sample_index": i,
            "true_class_id": true_class,
            "true_label": label_map[str(true_class)],
            "top_candidates": candidates,
        })

    save_json(
        {
            "k": k,
            "note": (
                "Only a limited sample of top-k examples is saved here. "
                "Full top-k candidate export can be added later."
            ),
            "examples_saved": len(examples),
            "examples": examples,
        },
        REPORT_DIR / "topk_examples.json",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the N05 character-detector CNN on Matenadata glyphs."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--binary-threshold", type=int, default=128)
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=-1,
        help="Debug limiter. Use -1 for the full Matenadata class folders.",
    )
    parser.add_argument(
        "--use-aristotel",
        action="store_true",
        help="Enable simple Aristotel damage variants for training samples only.",
    )
    parser.add_argument(
        "--aristotel-variants-per-sample",
        type=int,
        default=2,
        help="How many damaged virtual views to add per clean training glyph.",
    )
    parser.add_argument(
        "--aristotel-recipes",
        nargs="+",
        default=["light_cut", "light_erosion", "threshold_failure", "light_blur"],
        help="Recipe names to use, or 'all'. Keep this light for the CNN.",
    )
    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    global DATASET_DIR, MODEL_NAME, OUTPUT_DIR, REPORT_DIR
    global IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_SEED

    args = parse_args()

    DATASET_DIR = args.dataset_dir
    MODEL_NAME = args.model_name
    OUTPUT_DIR = PROJECT_ROOT / "models" / MODEL_NAME
    REPORT_DIR = PROJECT_ROOT / "reports" / MODEL_NAME
    IMAGE_SIZE = args.image_size
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    RANDOM_SEED = args.seed

    seed_everything(RANDOM_SEED)
    ensure_dirs()

    print("=" * 70)
    print(f"Starting training: {MODEL_NAME}")
    print("=" * 70)
    print("Input contract:")
    print("  Source: white ink on black background")
    print("  Thresholded: ink=255, background=0")
    print("  Tensor: ink=1.0, background=0.0")
    print("  Inversion: disabled")
    print("=" * 70)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}")

    label_map = load_json(LABEL_MAP_PATH)

    if len(label_map) != NUM_CLASSES:
        raise ValueError(
            f"Label map has {len(label_map)} entries, expected {NUM_CLASSES}"
        )

    samples, class_counts = collect_samples(DATASET_DIR)

    if args.limit_per_class is not None and args.limit_per_class > 0:
        limited_samples = []
        limited_counts = Counter()
        for class_id in range(NUM_CLASSES):
            class_samples = [
                sample for sample in samples if int(sample[1]) == class_id
            ][: args.limit_per_class]
            limited_samples.extend(class_samples)
            limited_counts[str(class_id)] = len(class_samples)
        samples = limited_samples
        class_counts = limited_counts

    print(f"Collected samples: {len(samples)}")
    print(f"Classes found: {len(class_counts)}")

    if len(class_counts) != NUM_CLASSES:
        raise ValueError(
            f"Found {len(class_counts)} classes, expected {NUM_CLASSES}"
        )

    paths = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]

    # 80% train, 10% validation, 10% test.
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

    train_samples = list(zip(train_paths, train_labels))
    val_samples = list(zip(val_paths, val_labels))
    test_samples = list(zip(test_paths, test_labels))

    split_report = make_split_report(
        samples,
        train_samples,
        val_samples,
        test_samples,
    )

    save_json(split_report, REPORT_DIR / "split_report.json")
    save_json(dict(class_counts), REPORT_DIR / "class_counts.json")

    # Save label map copy with model artifacts.
    shutil.copyfile(
        LABEL_MAP_PATH,
        OUTPUT_DIR / "numeric_label_map.json",
    )

    print("Split:")
    print(f"  Train:      {len(train_samples)}")
    print(f"  Validation: {len(val_samples)}")
    print(f"  Test:       {len(test_samples)}")

    aristotel_recipes = []
    aristotel_training_enabled = bool(args.use_aristotel)

    if aristotel_training_enabled:
        aristotel_recipes = build_aristotel_recipe_pool(args.aristotel_recipes)
        print("Aristotel CNN augmentation:")
        print(f"  Variants/sample: {args.aristotel_variants_per_sample}")
        print("  Recipes:")
        for recipe in aristotel_recipes:
            print(f"    - {recipe.name}")

    train_dataset = GlyphDataset(
        train_samples,
        IMAGE_SIZE,
        aristotel_recipes=aristotel_recipes,
        aristotel_variants_per_sample=(
            args.aristotel_variants_per_sample if aristotel_training_enabled else 0
        ),
        seed=RANDOM_SEED,
        binary_threshold=args.binary_threshold,
    )

    val_dataset = GlyphDataset(
        val_samples,
        IMAGE_SIZE,
        binary_threshold=args.binary_threshold,
    )

    test_dataset = GlyphDataset(
        test_samples,
        IMAGE_SIZE,
        binary_threshold=args.binary_threshold,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    model = GlyphClassifier(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_val_top1 = 0.0
    best_epoch = None
    history = []

    best_model_path = OUTPUT_DIR / f"{MODEL_NAME}_best.pt"
    last_model_path = OUTPUT_DIR / f"{MODEL_NAME}_last.pt"

    print("=" * 70)
    print("Training loop")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_top1 = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss, val_top1, val_top5, _, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_top1)

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_top1": float(train_top1),
            "val_loss": float(val_loss),
            "val_top1": float(val_top1),
            "val_top5": float(val_top5),
            "learning_rate": float(current_lr),
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} "
            f"train_top1={train_top1:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_top1={val_top1:.4f} "
            f"val_top5={val_top5:.4f} | "
            f"lr={current_lr:.6f}"
        )

        checkpoint = {
            "model_name": MODEL_NAME,
            "model_state_dict": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
            "input_polarity_mode": INPUT_POLARITY_MODE,
            "tensor_contract": {
                "source": "white_ink_on_black",
                "thresholded_ink_value": 255,
                "thresholded_background_value": 0,
                "tensor_ink_value": 1.0,
                "tensor_background_value": 0.0,
                "invert": False,
            },
            "binary_threshold": args.binary_threshold,
            "label_map": label_map,
            "epoch": epoch,
            "val_top1": float(val_top1),
            "val_top5": float(val_top5),
            "torch_version": torch.__version__,
        }

        # Always save last model.
        torch.save(checkpoint, last_model_path)

        # Save best model.
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_epoch = epoch
            torch.save(checkpoint, best_model_path)

            print(f"  Saved new best model: val_top1={best_val_top1:.4f}")

    save_json(history, REPORT_DIR / "training_history.json")

    print("=" * 70)
    print("Final test evaluation")
    print("=" * 70)

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    (
        test_loss,
        test_top1,
        test_top5,
        y_true,
        y_pred,
        y_logits,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    save_confusion_outputs(y_true, y_pred, label_map)
    save_topk_examples(y_true, y_logits, label_map, k=5, max_examples=300)

    final_report = {
        "model_name": MODEL_NAME,
        "framework": "PyTorch",
        "torch_version": str(torch.__version__),
        "device": str(device),
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "random_seed": RANDOM_SEED,
        "binary_threshold": args.binary_threshold,
        "input_polarity_mode": INPUT_POLARITY_MODE,
        "tensor_contract": {
            "source": "white_ink_on_black",
            "thresholded_ink_value": 255,
            "thresholded_background_value": 0,
            "tensor_ink_value": 1.0,
            "tensor_background_value": 0.0,
            "invert": False,
        },
        "dataset_dir": str(DATASET_DIR),
        "label_map_path": str(LABEL_MAP_PATH),
        "dataset_total_images": len(samples),
        "train_virtual_images": len(train_dataset),
        "aristotel_training": {
            "enabled": aristotel_training_enabled,
            "variants_per_sample": (
                args.aristotel_variants_per_sample
                if aristotel_training_enabled
                else 0
            ),
            "recipes": [recipe.name for recipe in aristotel_recipes],
            "recipe_definitions": [
                recipe.definition() for recipe in aristotel_recipes
            ],
        },
        "split": split_report,
        "best_epoch": best_epoch,
        "best_val_top1": float(best_val_top1),
        "test_loss": float(test_loss),
        "test_top1": float(test_top1),
        "test_top5": float(test_top5),
        "best_model_path": str(best_model_path),
        "last_model_path": str(last_model_path),
        "notes": [
            "White-ink-on-black training contract.",
            "Images are thresholded to binary: ink=255, background=0.",
            "Tensor values are ink=1.0, background=0.0.",
            "No inversion is applied.",
            "Optional Aristotel augmentation is applied only to training samples.",
            "Top-k output is important because N05 needs candidate letters, not only one winner.",
            "This is still a 78-class classifier; unsafe fragments/multi-character crops need external gating or reject classes later.",
        ],
    }

    save_json(final_report, REPORT_DIR / "training_report.json")

    print()
    print("Training complete.")
    print(f"Best epoch:    {best_epoch}")
    print(f"Best val top1: {best_val_top1:.4f}")
    print(f"Test top1:     {test_top1:.4f}")
    print(f"Test top5:     {test_top5:.4f}")
    print()
    print(f"Best model:    {best_model_path}")
    print(f"Last model:    {last_model_path}")
    print(f"Report:        {REPORT_DIR / 'training_report.json'}")
    print(f"Confusion:     {REPORT_DIR / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()