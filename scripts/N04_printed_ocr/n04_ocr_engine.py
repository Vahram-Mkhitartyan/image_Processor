"""Tesseract execution and raw printed-OCR result builders."""

import subprocess

from n04_constants import (
    DEFAULT_OCR_PSM,
    OCR_CANDIDATE_CONFIGS,
    OCR_ENGINE_NAME,
    OCR_ENGINE_VERSION,
    PRIMARY_OCR_LANGUAGE,
)
from n04_io import check_file_exists

def run_tesseract_command(
    crop_path,
    language=PRIMARY_OCR_LANGUAGE,
    psm=DEFAULT_OCR_PSM
):
    """
    Run Tesseract on one prepared crop and return raw OCR text.

    Command:
        tesseract crop.png stdout -l hye --psm 3
    """
    check_file_exists(
        crop_path,
        label="Tesseract-ready crop"
    )

    command = [
        "tesseract",
        crop_path,
        "stdout",
        "-l",
        language,
        "--psm",
        str(psm)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0:
        return {
            "success": False,
            "text": "",
            "error": result.stderr.strip()
        }

    return {
        "success": True,
        "text": result.stdout.strip(),
        "error": None
    }

def split_ocr_lines(text):
    """
    Split raw OCR text into preserved line records.

    N04 should not decide that ugly OCR text is garbage.
    It preserves every non-empty OCR line for downstream reconstruction.
    """
    if text is None:
        return []

    lines = []

    for index, line in enumerate(text.splitlines()):
        cleaned = line.strip()

        if cleaned == "":
            continue

        lines.append({
            "line_index": index,
            "text": cleaned,
            "trusted_as_final": False
        })

    return lines


def build_single_ocr_candidate(crop_path, config):
    """
    Run one OCR candidate configuration.

    Example configs:
        hye-calfa-n + psm 3
        hye + psm 3
    """
    language = config["language"]
    psm = config["psm"]

    raw_result = run_tesseract_command(
        crop_path=crop_path,
        language=language,
        psm=psm
    )

    if raw_result["success"] is False:
        return {
            "engine": config["engine"],
            "engine_version": OCR_ENGINE_VERSION,
            "language": language,
            "psm": psm,
            "role": config.get("role"),
            "text": None,
            "lines": [],
            "status": "failed",
            "trusted_as_final": False,
            "error": raw_result["error"]
        }

    text = raw_result["text"]

    if text == "":
        status = "empty"
    else:
        status = "raw_candidate"

    return {
        "engine": config["engine"],
        "engine_version": OCR_ENGINE_VERSION,
        "language": language,
        "psm": psm,
        "role": config.get("role"),
        "text": text,
        "lines": split_ocr_lines(text),
        "status": status,
        "trusted_as_final": False,
        "error": None
    }


def build_raw_printed_ocr_candidates(crop_path):
    """
    Build the printed OCR candidate object for one printed text unit.

    N04 does not choose final truth.
    N04 stores raw OCR candidates for downstream schema/language reconstruction.
    """
    if crop_path is None:
        return {
            "attempted": False,
            "engine_family": OCR_ENGINE_NAME,
            "engine_version": OCR_ENGINE_VERSION,
            "status": "missing_crop",
            "trusted_as_final": False,
            "primary_language": PRIMARY_OCR_LANGUAGE,
            "candidates": [],
            "error": "No OCR-ready crop path provided."
        }

    candidates = []

    for config in OCR_CANDIDATE_CONFIGS:
        candidate = build_single_ocr_candidate(
            crop_path=crop_path,
            config=config
        )

        candidates.append(candidate)

    any_raw = any(
        candidate["status"] == "raw_candidate"
        for candidate in candidates
    )

    any_failed = any(
        candidate["status"] == "failed"
        for candidate in candidates
    )

    if any_raw:
        status = "raw_candidates"
    elif any_failed:
        status = "failed"
    else:
        status = "empty"

    return {
        "attempted": True,
        "engine_family": OCR_ENGINE_NAME,
        "engine_version": OCR_ENGINE_VERSION,
        "status": status,
        "trusted_as_final": False,
        "primary_language": PRIMARY_OCR_LANGUAGE,
        "candidates": candidates,
        "error": None
    }


def build_tesseract_printed_ocr_result(
    crop_path,
    language=PRIMARY_OCR_LANGUAGE,
    psm=DEFAULT_OCR_PSM
):
    """
    Build a raw printed OCR result using Tesseract.

    Important:
    This output is a raw candidate, not trusted final text.
    Printed schema reconstruction validates this raw result downstream.
    """
    if crop_path is None:
        return {
            "attempted": False,
            "engine": OCR_ENGINE_NAME,
            "engine_version": OCR_ENGINE_VERSION,
            "language": language,
            "psm": psm,
            "text": None,
            "confidence": None,
            "status": "missing_crop",
            "trusted_as_final": False,
            "error": "No Tesseract-ready crop path provided.",
            "word_boxes": []
        }

    result = run_tesseract_command(
        crop_path=crop_path,
        language=language,
        psm=psm
    )

    if result["success"] is False:
        return {
            "attempted": True,
            "engine": OCR_ENGINE_NAME,
            "engine_version": OCR_ENGINE_VERSION,
            "language": language,
            "psm": psm,
            "text": None,
            "confidence": None,
            "status": "failed",
            "trusted_as_final": False,
            "error": result["error"],
            "word_boxes": []
        }

    text = result["text"]

    if text == "":
        status = "empty"
    else:
        status = "raw_candidate"

    return {
        "attempted": True,
        "engine": OCR_ENGINE_NAME,
        "engine_version": OCR_ENGINE_VERSION,
        "language": language,
        "psm": psm,
        "text": text,
        "confidence": None,
        "status": status,
        "trusted_as_final": False,
        "error": None,
        "word_boxes": []
    }


def build_placeholder_printed_ocr_result():
    """
    Build a non-attempted OCR result for records that have no OCR-ready crop.
    """
    return {
        "attempted": False,
        "engine": OCR_ENGINE_NAME,
        "engine_version": OCR_ENGINE_VERSION,
        "text": None,
        "confidence": None,
        "status": "placeholder",
        "word_boxes": []
    }
