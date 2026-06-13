"""Basic JSON and output-folder helpers for N04."""

import json
import os
import shutil

def load_json(input_path):
    """
    Load a JSON file from disk.

    Used for N03 visual routes, settings, and output metadata.
    """
    with open(input_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, output_path):
    """
    Save Python data as readable JSON.

    Used for:
    - printed text map JSON
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    return output_path


def ensure_dir(path):
    """
    Create a directory if it does not already exist.
    """
    os.makedirs(path, exist_ok=True)


def check_file_exists(path, label="file"):
    """
    Fail early if a required file does not exist.

    This keeps errors clear when N04 is missing:
    - N03 routes JSON
    - routed crop image
    - refined crop image
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_settings(settings_path=None):
    """
    Load optional N04 settings.

    If no settings file exists, return an empty dict.
    """
    if settings_path is None:
        return {}

    if not os.path.exists(settings_path):
        return {}

    return load_json(settings_path)


def create_output_folders(output_dir):
    """
    Create the debug-friendly N04 output structure.

    N04 output structure:

    output_dir/
        crops/
            printed_only/
            mixed/
        metadata/
        debug/

    crops/
        Human-inspection copies of the crops N04 selected for printed OCR.

    metadata/
        Machine-readable printed text map JSON.

    debug/
        Optional visual-inspection artifacts.
    """
    folders = {
        "root": output_dir,

        "crops": f"{output_dir}/crops",
        "printed_only": f"{output_dir}/crops/printed_only",
        "mixed": f"{output_dir}/crops/mixed",

        "tesseract_ready": f"{output_dir}/tesseract_ready",
        "tesseract_ready_printed_only": f"{output_dir}/tesseract_ready/printed_only",
        "tesseract_ready_mixed": f"{output_dir}/tesseract_ready/mixed",

        "metadata": f"{output_dir}/metadata",
        "debug": f"{output_dir}/debug"

    }

    for folder in (
        folders["root"],
        folders["tesseract_ready"],
        folders["tesseract_ready_printed_only"],
        folders["tesseract_ready_mixed"],
        folders["metadata"],
    ):
        ensure_dir(folder)

    return folders


def reset_output_dir(output_dir):
    """
    Delete previous N04 output for this document and recreate it.

    This keeps reruns clean.

    Example:
    This prevents stale copied crops from previous N04 runs.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
