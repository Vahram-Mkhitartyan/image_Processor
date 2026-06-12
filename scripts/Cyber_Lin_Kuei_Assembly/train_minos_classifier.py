# IMPORTS-----------------------------------------------------
import os
import cv2
import numpy as np

# ML ITS HAPPENING!!!!!
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf


# CONSTANTS---------------------------------------------------
IMAGE_SIZE = 128

VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42


# DATASET PARSING--------------------------------------------

BASE_DIR = "/home/vahram/Desktop/image_Processor"
DATASET_DIR = f"{BASE_DIR}/classifier_dataset_presence"

# These are the real folder names inside the dataset directory.
CLASS_NAMES = [
    "mixed",
    "printed_only",
    "empty_or_noise",
    "handwriting_only"
]

# These are the actual Minos model outputs.
# Minos is now multi-label, not softmax.
OUTPUT_LABELS = [
    "printed_present",
    "handwriting_present",
    "noise"
]

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# LABEL LOGIC-------------------------------------------------

def class_name_to_multilabel(class_name):
    """
    Convert folder class into Minos multi-label target.

    Output order:
    [printed_present, handwriting_present, noise]
    """
    if class_name == "mixed":
        return [1.0, 1.0, 0.0]

    if class_name == "printed_only":
        return [1.0, 0.0, 0.0]

    if class_name == "handwriting_only":
        return [0.0, 1.0, 0.0]

    if class_name == "empty_or_noise":
        return [0.0, 0.0, 1.0]

    raise ValueError(f"Unknown class name: {class_name}")


def derive_minos_class(scores):
    """
    Convert model output scores into final Minos route class.

    scores order:
    [printed_present, handwriting_present, noise]
    """
    printed_score = float(scores[0])
    handwriting_score = float(scores[1])
    noise_score = float(scores[2])

    printed_threshold = 0.45
    handwriting_threshold = 0.45
    noise_threshold = 0.65

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


# DATASET SCAN-----------------------------------------------

def get_image_paths_for_class(class_name):
    """
    Collect training image paths for one class.
    """
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
    """
    Scan the classifier dataset and build item metadata.
    """
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


# IMAGE PREPARATION------------------------------------------

def resize_with_padding(image, target_size=IMAGE_SIZE):
    """
    Resize an image into a padded square canvas.
    """
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
    """
    Load one training image and convert it to model input.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    padded = resize_with_padding(image, target_size=IMAGE_SIZE)

    normalized = padded.astype("float32") / 255.0

    # Shape becomes: (128, 128, 1)
    normalized = np.expand_dims(normalized, axis=-1)

    return normalized


def load_dataset(dataset_items):
    """
    Load all dataset images and multi-label targets into arrays.
    """
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


# TRAIN / VALIDATION SPLIT-----------------------------------

def split_dataset(X, y, dataset_items):
    """
    Split arrays into train/validation sets.

    We stratify by the original folder class, not by multi-label arrays.
    """
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


# MODEL ------------------------------------------------------

def build_cnn_model():
    """
    Build the Minos v2.0 multi-label CNN model.

    Output:
    printed_present score
    handwriting_present score
    noise score
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),

        # Light augmentation only. Heavy augmentation can distort OCR crops.
        tf.keras.layers.RandomRotation(0.02),
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.RandomTranslation(0.03, 0.03),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        # Multi-label output, not softmax.
        tf.keras.layers.Dense(len(OUTPUT_LABELS), activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model


def train_model(X_train, y_train, X_val, y_val):
    """
    Train the Minos v2.0 multi-label crop-type classifier.
    """
    model = build_cnn_model()

    models_dir = f"{BASE_DIR}/models"
    os.makedirs(models_dir, exist_ok=True)

    checkpoint_path = f"{models_dir}/minos_v2_0_best.keras"

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[
            early_stopping,
            model_checkpoint
        ]
    )

    return model, history


def save_model(model):
    """
    Save the final restored Minos model.
    """
    models_dir = f"{BASE_DIR}/models"
    os.makedirs(models_dir, exist_ok=True)

    model_path = f"{models_dir}/minos_v2_0.keras"
    model.save(model_path)

    return model_path


# EVALUATION -------------------------------------------------

def print_raw_multilabel_samples(y_true, y_pred_probs, sample_count=10):
    """
    Print a few raw multi-label predictions for sanity checking.
    """
    print("-------------------------")
    print("Raw prediction samples:")

    limit = min(sample_count, len(y_pred_probs))

    for index in range(limit):
        scores = y_pred_probs[index]
        true_labels = y_true[index]

        print(
            f"sample={index} "
            f"true={true_labels.tolist()} "
            f"pred={{"
            f"printed:{float(scores[0]):.3f}, "
            f"handwriting:{float(scores[1]):.3f}, "
            f"noise:{float(scores[2]):.3f}"
            f"}} "
            f"derived={derive_minos_class(scores)}"
        )


def evaluate_minos(model, X_val, y_val, class_val):
    """
    Evaluate Minos both as a multi-label model and as a derived router.
    """
    print("-------------------------")
    print("Evaluating Minos...")

    eval_results = model.evaluate(X_val, y_val, return_dict=True)

    print("-------------------------")
    print("Validation results:")
    for key, value in eval_results.items():
        print(f"{key}: {value}")

    y_pred_probs = model.predict(X_val)

    print_raw_multilabel_samples(y_val, y_pred_probs)

    predicted_classes = []
    for scores in y_pred_probs:
        predicted_classes.append(derive_minos_class(scores))

    true_classes = []
    for item in class_val:
        true_classes.append(CLASS_NAMES[int(item)])

    # Includes review because thresholding may create review predictions.
    report_labels = [
        "mixed",
        "printed_only",
        "empty_or_noise",
        "handwriting_only",
        "review"
    ]

    print("-------------------------")
    print("Derived Minos classification report:")
    print(classification_report(
        true_classes,
        predicted_classes,
        labels=report_labels,
        zero_division=0
    ))

    print("-------------------------")
    print("Derived Minos confusion matrix:")
    print("Labels:", report_labels)
    print(confusion_matrix(
        true_classes,
        predicted_classes,
        labels=report_labels
    ))

    return {
        "eval_results": eval_results,
        "predicted_classes": predicted_classes,
        "true_classes": true_classes
    }


# EXECUTION --------------------------------------------------

if __name__ == "__main__":
    print("-------------------------")
    print("Minos v2.0 multi-label training started.")

    dataset_items = scan_dataset()

    print("-------------------------")
    print("Total images:", len(dataset_items))

    X, y, failed_items = load_dataset(dataset_items)

    if len(X) == 0:
        raise RuntimeError("Dataset loaded 0 images. Check DATASET_DIR and class folders.")

    X_train, X_val, y_train, y_val, class_train, class_val = split_dataset(
        X,
        y,
        dataset_items
    )

    print("-------------------------")
    print("Dataset shapes:")
    print("Train images shape:", X_train.shape)
    print("Validation images shape:", X_val.shape)
    print("Train labels shape:", y_train.shape)
    print("Validation labels shape:", y_val.shape)
    print("Loaded images shape:", X.shape)
    print("Loaded labels shape:", y.shape)
    print("Failed items:", len(failed_items))

    if len(failed_items) > 0:
        print("First failed item:", failed_items[0])

    print("-------------------------")
    print("Building and training Minos CNN model...")

    model, history = train_model(
        X_train,
        y_train,
        X_val,
        y_val
    )

    print("-------------------------")
    print("Training complete.")

    evaluate_minos(
        model=model,
        X_val=X_val,
        y_val=y_val,
        class_val=class_val
    )

    model_path = save_model(model)

    print("-------------------------")
    print("Model saved to:", model_path)
    print("Best checkpoint saved to:", f"{BASE_DIR}/models/minos_v2_0_best.keras")
    print("Minos v2.0 training finished.")