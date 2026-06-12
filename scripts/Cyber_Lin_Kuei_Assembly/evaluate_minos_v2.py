import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf


IMAGE_SIZE = 128
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

BASE_DIR = "/home/vahram/Desktop/image_Processor"
DATASET_DIR = f"{BASE_DIR}/classifier_dataset_presence"

MODEL_PATH = f"{BASE_DIR}/models/minos_v2_best.keras"

CLASS_NAMES = [
    "mixed",
    "printed_only",
    "empty_or_noise",
    "handwriting_only"
]

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


def class_name_to_multilabel(class_name):
    if class_name == "mixed":
        return [1.0, 1.0, 0.0]

    if class_name == "printed_only":
        return [1.0, 0.0, 0.0]

    if class_name == "handwriting_only":
        return [0.0, 1.0, 0.0]

    if class_name == "empty_or_noise":
        return [0.0, 0.0, 1.0]

    raise ValueError(f"Unknown class name: {class_name}")


def get_image_paths_for_class(class_name):
    class_dir = f"{DATASET_DIR}/{class_name}"

    if not os.path.isdir(class_dir):
        raise FileNotFoundError(f"Class folder does not exist: {class_dir}")

    image_paths = []

    for file_name in os.listdir(class_dir):
        file_path = f"{class_dir}/{file_name}"

        if not os.path.isfile(file_path):
            continue

        extension = os.path.splitext(file_name)[1].lower()

        if extension not in SUPPORTED_EXTENSIONS:
            continue

        image_paths.append(file_path)

    return image_paths


def scan_dataset():
    dataset_items = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        image_paths = get_image_paths_for_class(class_name)
        print(class_name, ":", len(image_paths))

        for image_path in image_paths:
            dataset_items.append({
                "image_path": image_path,
                "class_name": class_name,
                "class_index": class_index,
                "multilabel": class_name_to_multilabel(class_name)
            })

    return dataset_items


def resize_with_padding(image, target_size=IMAGE_SIZE):
    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError("Invalid image size for padding.")

    scale = target_size / max(height, width)

    new_width = max(int(width * scale), 1)
    new_height = max(int(height * scale), 1)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    canvas = 255 * np.ones((target_size, target_size), dtype=np.uint8)

    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return canvas


def load_and_prepare_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    padded = resize_with_padding(image, target_size=IMAGE_SIZE)
    normalized = padded.astype("float32") / 255.0
    normalized = np.expand_dims(normalized, axis=-1)

    return normalized


def load_dataset(dataset_items):
    images = []
    labels = []
    failed_items = []

    for item in dataset_items:
        try:
            image = load_and_prepare_image(item["image_path"])
            images.append(image)
            labels.append(item["multilabel"])
        except Exception as error:
            failed_items.append({
                "image_path": item["image_path"],
                "error": str(error)
            })

    X = np.array(images, dtype="float32")
    y = np.array(labels, dtype="float32")

    return X, y, failed_items


def split_dataset(X, y, dataset_items):
    class_indices = np.array(
        [item["class_index"] for item in dataset_items],
        dtype="int64"
    )

    X_train, X_val, y_train, y_val, class_train, class_val = train_test_split(
        X,
        y,
        class_indices,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=class_indices
    )

    return X_train, X_val, y_train, y_val, class_train, class_val


def derive_minos_class_with_thresholds(
    scores,
    printed_threshold,
    handwriting_threshold,
    noise_threshold
):
    printed_score = float(scores[0])
    handwriting_score = float(scores[1])
    noise_score = float(scores[2])

    printed_present = printed_score >= printed_threshold
    handwriting_present = handwriting_score >= handwriting_threshold
    noise_present = noise_score >= noise_threshold

    if noise_present and not printed_present and not handwriting_present:
        return "empty_or_noise"

    if printed_present and handwriting_present:
        return "mixed"

    if printed_present:
        return "printed_only"

    if handwriting_present:
        return "handwriting_only"

    return "review"


def evaluate_predictions_for_thresholds(
    true_classes,
    y_pred_probs,
    printed_threshold,
    handwriting_threshold,
    noise_threshold
):
    predicted_classes = []

    for scores in y_pred_probs:
        predicted_classes.append(
            derive_minos_class_with_thresholds(
                scores=scores,
                printed_threshold=printed_threshold,
                handwriting_threshold=handwriting_threshold,
                noise_threshold=noise_threshold
            )
        )

    report_labels = [
        "mixed",
        "printed_only",
        "empty_or_noise",
        "handwriting_only",
        "review"
    ]

    print("-------------------------")
    print(
        "Thresholds:",
        "printed=", printed_threshold,
        "handwriting=", handwriting_threshold,
        "noise=", noise_threshold
    )

    print(classification_report(
        true_classes,
        predicted_classes,
        labels=report_labels,
        zero_division=0
    ))

    print("Labels:", report_labels)
    print(confusion_matrix(
        true_classes,
        predicted_classes,
        labels=report_labels
    ))


def main():
    print("-------------------------")
    print("Evaluating saved Minos v2.0 model only. No training.")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model does not exist: {MODEL_PATH}")

    dataset_items = scan_dataset()
    X, y, failed_items = load_dataset(dataset_items)

    X_train, X_val, y_train, y_val, class_train, class_val = split_dataset(
        X,
        y,
        dataset_items
    )

    print("Validation images:", X_val.shape)
    print("Failed items:", len(failed_items))

    model = tf.keras.models.load_model(MODEL_PATH)

    eval_results = model.evaluate(X_val, y_val, return_dict=True)

    print("-------------------------")
    print("Validation results:")
    for key, value in eval_results.items():
        print(f"{key}: {value}")

    y_pred_probs = model.predict(X_val)

    true_classes = [CLASS_NAMES[int(item)] for item in class_val]

    threshold_tests = [
        (0.45, 0.45, 0.65),
        (0.45, 0.50, 0.65),
        (0.45, 0.55, 0.65),
        (0.45, 0.60, 0.65),
        (0.50, 0.50, 0.65),
        (0.50, 0.55, 0.65),
        (0.50, 0.60, 0.65),
        (0.55, 0.55, 0.65),
        (0.55, 0.60, 0.65),
        (0.45, 0.55, 0.70),
        (0.45, 0.60, 0.70),
    ]

    for printed_t, handwriting_t, noise_t in threshold_tests:
        evaluate_predictions_for_thresholds(
            true_classes=true_classes,
            y_pred_probs=y_pred_probs,
            printed_threshold=printed_t,
            handwriting_threshold=handwriting_t,
            noise_threshold=noise_t
        )


if __name__ == "__main__":
    main()
