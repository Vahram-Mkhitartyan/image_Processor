"""Document discovery, temp input, and result JSON helpers."""

import json
import os
import shutil

from .paths import (
    FAILED_RESULTS_DIR,
    FINAL_RESULTS_DIR,
    INPUT_DOCUMENTS_DIR,
    SUPPORTED_EXTENSIONS,
    TEMP_PROCESSING_DIR,
)


def ensure_batch_folders():
    """
    Create folders required by batch processing.
    """
    os.makedirs(INPUT_DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(TEMP_PROCESSING_DIR, exist_ok=True)
    os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)
    os.makedirs(FAILED_RESULTS_DIR, exist_ok=True)


def get_document_paths(input_folder):
    """
    Collect supported document paths from a folder.

    Args:
        input_folder: Folder containing candidate input documents.

    Returns:
        Sorted list of supported document paths.
    """
    document_paths = []

    for file_name in os.listdir(input_folder):
        file_path = f"{input_folder}/{file_name}"

        if not os.path.isfile(file_path):
            continue

        extension = os.path.splitext(file_name)[1].lower()

        if extension not in SUPPORTED_EXTENSIONS:
            continue

        document_paths.append(file_path)

    return sorted(document_paths)


def get_document_id(document_path):
    """
    Derive a document id from a file path.

    Args:
        document_path: Path to one input document.

    Returns:
        Document id string.
    """
    file_name = os.path.basename(document_path)
    document_id = os.path.splitext(file_name)[0]

    return document_id


def save_json(data, output_path):
    """
    Write JSON data to disk.

    Args:
        data: Serializable data object.
        output_path: Path where the result should be saved.

    Returns:
        Path to the saved JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    return output_path


def get_result_path(document_id):
    """
    Build the final result JSON path for a document.

    Args:
        document_id: Stable identifier derived from the document filename.

    Returns:
        Path to the final result JSON file.
    """
    return f"{FINAL_RESULTS_DIR}/{document_id}_handwriting_result.json"


def load_existing_result(document_id):
    """
    Load an existing result file or create an empty payload.

    Args:
        document_id: Stable identifier derived from the document filename.

    Returns:
        Existing or initialized result payload.
    """
    result_path = get_result_path(document_id)

    if not os.path.exists(result_path):
        return {
            "document_id": document_id,
            "phases": {}
        }

    with open(result_path, "r", encoding="utf-8") as file:
        return json.load(file)


def ensure_temp_input(document_path, document_output_dir):
    """
    Copy a source document into its temp processing folder.

    Args:
        document_path: Path to one input document.
        document_output_dir: Temp folder for one document.

    Returns:
        Path to the copied temp input file.
    """
    os.makedirs(document_output_dir, exist_ok=True)

    file_extension = os.path.splitext(document_path)[1].lower()
    temp_document_path = f"{document_output_dir}/input_document{file_extension}"

    shutil.copy2(document_path, temp_document_path)

    return temp_document_path
