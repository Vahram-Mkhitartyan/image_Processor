#IMPORTS-------------------------------------------------------
import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
FILE_PREPARATION_DIR = f"{SCRIPTS_DIR}/N00_file_preparation"
for path in [SCRIPTS_DIR, FILE_PREPARATION_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from N00_file_preparation.file_preparation import prepare_file


#HANDWRITING DETECTOR CONFIG-----------------------------------
MODEL_NAME = "handwriting_detector_v0_image_loader"



PROCESSING_MODE = "full_document"
# allowed values:
# "handwriting_sample"
# "full_document"

BASE_DIR = "/home/vahram/Desktop/image_Processor"
MODEL_PATH = f"{BASE_DIR}/models/handwriting_presence_v1_4.keras"

IMAGE_SIZE = 128

CLASS_NAMES = [
    "handwriting_present",
    "printed_only",
    "empty_or_noise"
]

CLASSIFIER_MODEL = None

#PREPARATION MODES---------------------------------------------


HANDWRITING_SAMPLE_STEPS = [                    #processes small samples of handwriting
    "load_image",
    "rotate_major",
    "convert_to_grayscale",
    "denoise_image",
    "improve_contrast",
    "threshold_image",
    "crop_white_margins",
    "calculate_quality",
    "save_outputs"
]


FULL_DOCUMENT_STEPS = [                         #processes the entire document 
    "load_image",
    "rotate_major",
    "convert_to_grayscale",
    "denoise_image",
    "improve_contrast",
    "threshold_image",
    "deskew_image",
    "crop_white_margins",
    "calculate_quality",
    "detect_regions",
    "detect_lines",
    "split_rows",
    "split_fields",
    "create_scribemap_masks",
    "save_outputs"
]


def get_preparation_steps(processing_mode):                 #picks the processing mode, returns the steps
    """Return preparation steps for the requested processing mode.
    
    Args:
        processing_mode: Pipeline mode controlling how candidates are produced.
    
    Returns:
        Ordered list of preparation step names.
    """
    if processing_mode == "handwriting_sample":
        return HANDWRITING_SAMPLE_STEPS

    if processing_mode == "full_document":
        return FULL_DOCUMENT_STEPS

    raise ValueError("processing_mode must be 'handwriting_sample' or 'full_document'")


#BASIC UTILITIES-----------------------------------------------

def check_file_exists(input_path):                          #check if the input file exisits
    """Validate that a path exists.
    
    Args:
        input_path: Path to the input document image.
    
    Returns:
        None; raises if the path is missing.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")


def ensure_output_dir(output_dir):                          #ensure the output directory exists
    """Create an output directory if needed.
    
    Args:
        output_dir: Folder where generated files should be written.
    
    Returns:
        None.
    """
    os.makedirs(output_dir, exist_ok=True)


def load_image(input_path):                                 #load the image
    """Load an image from disk.
    
    Args:
        input_path: Path to the input document image.
    
    Returns:
        Loaded image array.
    """
    image = cv2.imread(input_path)

    if image is None:
        raise ValueError(f"Could not load image from path: {input_path}")

    return image


def save_json(data, output_path):                           #save report as json file
    """Write JSON data to disk.
    
    Args:
        data: Serializable data object.
        output_path: Path where the result should be saved.
    
    Returns:
        Path to the saved JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    return output_path


#CANDIDATE CREATION--------------------------------------------

def create_sample_candidate(preparation_state):             #creates a single handwriting candidate using the cropped image from preparation. this is a placeholder for now, as we are not running an actual model yet.
    """Create a fallback full-page handwriting candidate.
    
    Args:
        preparation_state: State dictionary returned by node 00 file preparation.
    
    Returns:
        Single full-page candidate dictionary.
    """
    cropped_path = preparation_state["artifacts"].get("cropped")                #get the path to the cropped image from preparation artifacts
    cropped_shape = preparation_state["metadata"]["shapes"].get("cropped")      #get the shape of the cropped image from preparation metadata

    if cropped_path is None or cropped_shape is None:
        return []

    height = cropped_shape[0]
    width = cropped_shape[1]

    candidate = {
        "candidate_id": 1,
        "source_image": cropped_path,
        "label": "handwritten",
        "confidence": 1.0,
        "method": "known_handwriting_sample",
        "bbox": {
            "x1": 0,
            "y1": 0,
            "x2": width,
            "y2": height
        }
    }

    return [candidate]


def create_candidates_from_prepared_fields(preparation_metadata):               #creates handwriting candidates using the field crops detected in preparation. this is a placeholder for now, as we are not running an actual model yet.
    """Build handwriting candidates from prepared field metadata.
    
    Args:
        preparation_metadata: Metadata dictionary produced by file preparation.
    
    Returns:
        List of field candidate dictionaries.
    """
    field_data = preparation_metadata.get("field_crops", {}).get("fields", [])

    candidates = []

    for index, field in enumerate(field_data, start=1):
        candidate = {
            "candidate_id": index,
            "source_image": field["path"],
            "label": "unknown",
            "confidence": None,
            "method": "prepared_field_crop_placeholder",
            "row_index": field["row_index"],
            "field_index": field["field_index"],
            "bbox": {
                "x1": field["x1"],
                "y1": field["y1"],
                "x2": field["x2"],
                "y2": field["y2"]
            }
        }

        candidates.append(candidate)

    return candidates


def create_candidates_by_mode(processing_mode, preparation_state):             #creates candidates based on the processing mode
    """Choose candidate generation strategy for a mode.
    
    Args:
        processing_mode: Pipeline mode controlling how candidates are produced.
        preparation_state: State dictionary returned by node 00 file preparation.
    
    Returns:
        List of candidate dictionaries.
    """
    if processing_mode == "handwriting_sample":
        return create_sample_candidate(preparation_state)

    if processing_mode == "full_document":
        return create_candidates_from_prepared_fields(
            preparation_state["metadata"]
        )

    raise ValueError("processing_mode must be 'handwriting_sample' or 'full_document'")

#CLASSIFIER PLACEHOLDER----------------------------------------

def load_classifier_model():
    """Load the handwriting-presence classifier model.
    
    Args:
        None.
    
    Returns:
        Loaded Keras model or None if unavailable.
    """
    global CLASSIFIER_MODEL

    if CLASSIFIER_MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Classifier model not found: {MODEL_PATH}")

        CLASSIFIER_MODEL = tf.keras.models.load_model(MODEL_PATH)

    return CLASSIFIER_MODEL


def resize_with_padding(image, target_size=IMAGE_SIZE):
    """Resize an image into a padded square canvas.
    
    Args:
        image: Input image array.
        target_size: Desired square image size in pixels.
    
    Returns:
        Padded square image array.
    """
    height, width = image.shape[:2]

    scale = target_size / max(height, width)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(image, (new_width, new_height))

    canvas = 255 * np.ones((target_size, target_size), dtype=np.uint8)

    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return canvas


def prepare_candidate_image(candidate_image_path):
    """Load and normalize a candidate image for inference.
    
    Args:
        candidate_image_path: Path to a candidate crop image.
    
    Returns:
        Prepared model input batch array.
    """
    image = cv2.imread(candidate_image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not load candidate image: {candidate_image_path}")

    padded = resize_with_padding(image, target_size=IMAGE_SIZE)

    normalized = padded.astype("float32") / 255.0

    normalized = np.expand_dims(normalized, axis=-1)
    normalized = np.expand_dims(normalized, axis=0)

    return normalized


def classify_candidate_placeholder(candidate, processing_mode):                 #this is a placeholder function for classifying a candidate. it assigns a label and confidence based on the processing mode, but does not run an actual model yet.
    """Return a placeholder classification result.
    
    Args:
        candidate: Candidate crop metadata dictionary.
        processing_mode: Pipeline mode controlling how candidates are produced.
    
    Returns:
        Placeholder classification dictionary.
    """
    if processing_mode == "handwriting_sample":
        candidate["label"] = "handwritten"
        candidate["confidence"] = 1.0
        candidate["classification_method"] = "known_handwriting_sample_placeholder"

    elif processing_mode == "full_document":
        candidate["label"] = "unknown"
        candidate["confidence"] = None
        candidate["classification_method"] = "not_classified_yet"

    else:
        raise ValueError("processing_mode must be 'handwriting_sample' or 'full_document'")

    return candidate

def classify_candidate_with_model(candidate):
    """Classify one candidate using the ML model.
    
    Args:
        candidate: Candidate crop metadata dictionary.
    
    Returns:
        Model classification dictionary.
    """
    model = load_classifier_model()

    candidate_image = prepare_candidate_image(candidate["source_image"])

    predictions = model.predict(candidate_image, verbose=0)[0]

    handwriting_index = CLASS_NAMES.index("handwriting_present")
    handwriting_score = float(predictions[handwriting_index])

    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index])
    label = CLASS_NAMES[class_index]

    candidate["label"] = label
    candidate["confidence"] = confidence
    candidate["handwriting_score"] = handwriting_score
    candidate["contains_handwriting"] = handwriting_score >= 0.35
    candidate["send_to_ocr_later"] = handwriting_score >= 0.35
    candidate["classification_method"] = "handwriting_presence_cnn_model_v1_4"
    candidate["class_scores"] = {
        CLASS_NAMES[index]: float(score)
        for index, score in enumerate(predictions)
    }

    return candidate

def classify_candidates(candidates, processing_mode):
    """Classify a list of candidate crops.
    
    Args:
        candidates: List of candidate crop metadata dictionaries.
        processing_mode: Pipeline mode controlling how candidates are produced.
    
    Returns:
        List of classified candidate dictionaries.
    """
    classified_candidates = []

    for candidate in candidates:
        if processing_mode == "handwriting_sample":
            classified_candidate = classify_candidate_placeholder(
                candidate,
                processing_mode
            )
        else:
            classified_candidate = classify_candidate_with_model(candidate)

        classified_candidates.append(classified_candidate)

    return classified_candidates


#PROCESS DOCUMENT----------------------------------------------

def process_document_for_handwriting(input_path, output_dir, processing_mode=PROCESSING_MODE):   #main function to process a document for handwriting detection. runs the preparation steps, creates candidates, and compiles the result.
    """Run handwriting detection for one document.
    
    Args:
        input_path: Path to the input document image.
        output_dir: Folder where generated files should be written.
        processing_mode: Pipeline mode controlling how candidates are produced.
    
    Returns:
        Handwriting detection result dictionary.
    """
    check_file_exists(input_path)
    ensure_output_dir(output_dir)

    preparation_steps = get_preparation_steps(processing_mode)

    preparation_state = prepare_file(
        input_path=input_path,
        output_dir=output_dir,
        steps=preparation_steps,
        settings={
            "manual_major_rotation": 0
        }
    )

    handwriting_candidates = create_candidates_by_mode(
        processing_mode,
        preparation_state
    )

    handwriting_candidates = classify_candidates(
    handwriting_candidates,
    processing_mode
    )   

    image = load_image(input_path)
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1

    result = {
        "model_name": MODEL_NAME,
        "processing_mode": processing_mode,
        "input_path": input_path,
        "output_dir": output_dir,
        "status": "processed",
        "preparation": {
            "metadata": preparation_state["metadata"],
            "artifacts": preparation_state["artifacts"]
        },
        "image_info": {
            "height": height,
            "width": width,
            "channels": channels,
            "shape": list(image.shape)
        },
        "candidate_summary": {
            "total_candidates": len(handwriting_candidates),
            "classification_status": "classified_by_handwriting_presence_cnn_model_v1_4"    
        },
        "handwriting_candidates": handwriting_candidates,
        "warnings": []
    }

    return result
