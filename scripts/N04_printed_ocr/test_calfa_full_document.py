"""Run Calfa Tesseract OCR directly on one complete, unprocessed document."""

import argparse
import csv
import io
import json
import os
import subprocess
from datetime import datetime, timezone

BASE_DIR = "/home/vahram/Desktop/image_Processor"
DEFAULT_IMAGE_PATH = os.path.join(BASE_DIR, "temp_processing", "test_1", "n00_file_preparation", "full_images", "03_denoised.jpeg")
DEFAULT_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "temp_processing",
    "test_1",
    "n04_printed_ocr",
    "full_document_test",
)
DEFAULT_LANGUAGE = "hye-calfa-n"
DEFAULT_PSM = 6


def run_tesseract(image_path, language, psm, output_format=None):
    """Run Tesseract and return its captured process result.

    Args:
        image_path: Full raw document image.
        language: Installed Tesseract language model name.
        psm: Tesseract page-segmentation mode.
        output_format: Optional renderer such as ``tsv``.

    Returns:
        Completed subprocess result with stdout and stderr.
    """
    command = [
        "tesseract",
        image_path,
        "stdout",
        "-l",
        language,
        "--psm",
        str(psm),
        "-c",
        "preserve_interword_spaces=1",
    ]
    if output_format:
        command.append(output_format)

    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )


def summarize_tsv(tsv_text):
    """Calculate confidence statistics from recognized non-empty words."""
    words = []
    reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")

    for row in reader:
        text = (row.get("text") or "").strip()
        try:
            confidence = float(row.get("conf", -1))
        except (TypeError, ValueError):
            confidence = -1

        if text and confidence >= 0:
            words.append({
                "text": text,
                "confidence": confidence,
                "left": int(row.get("left", 0)),
                "top": int(row.get("top", 0)),
                "width": int(row.get("width", 0)),
                "height": int(row.get("height", 0)),
            })

    confidences = [word["confidence"] for word in words]
    return {
        "recognized_word_count": len(words),
        "mean_word_confidence": (
            round(sum(confidences) / len(confidences), 3)
            if confidences
            else None
        ),
        "minimum_word_confidence": min(confidences) if confidences else None,
        "maximum_word_confidence": max(confidences) if confidences else None,
        "words": words,
    }


def run_full_document_test(
    image_path=DEFAULT_IMAGE_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    language=DEFAULT_LANGUAGE,
    psm=DEFAULT_PSM,
):
    """Run Calfa text and TSV passes and save a compact report."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Raw document not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    document_id = os.path.splitext(os.path.basename(image_path))[0]
    text_path = os.path.join(output_dir, f"{document_id}_calfa_full_document.txt")
    report_path = os.path.join(output_dir, f"{document_id}_calfa_full_document.json")

    text_result = run_tesseract(image_path, language, psm)
    tsv_result = run_tesseract(image_path, language, psm, output_format="tsv")

    if text_result.returncode != 0:
        raise RuntimeError(text_result.stderr.strip() or "Calfa OCR failed.")
    if tsv_result.returncode != 0:
        raise RuntimeError(tsv_result.stderr.strip() or "Calfa TSV pass failed.")

    recognized_text = text_result.stdout.strip()
    confidence_summary = summarize_tsv(tsv_result.stdout)

    with open(text_path, "w", encoding="utf-8") as file:
        file.write(recognized_text)
        file.write("\n")

    report = {
        "test_name": "calfa_full_raw_document",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_path": os.path.abspath(image_path),
        "source_policy": "complete_unprocessed_input_document",
        "language": language,
        "psm": int(psm),
        "recognized_text": recognized_text,
        "text_output_path": text_path,
        "confidence": confidence_summary,
        "stderr": text_result.stderr.strip() or None,
    }

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("Calfa full-document test complete.")
    print("Image:", image_path)
    print("Language:", language)
    print("PSM:", psm)
    print("Recognized words:", confidence_summary["recognized_word_count"])
    print("Mean word confidence:", confidence_summary["mean_word_confidence"])
    print("Text output:", text_path)
    print("JSON report:", report_path)
    print()
    print("Recognized text")
    print("-------------------------")
    print(recognized_text or "[empty]")

    return report


def main():
    """Parse optional source and output overrides for the test."""
    parser = argparse.ArgumentParser(
        description="Run hye-calfa-n OCR on a complete raw document."
    )
    parser.add_argument("image_path", nargs="?", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--psm", type=int, default=DEFAULT_PSM)
    args = parser.parse_args()

    run_full_document_test(
        image_path=args.image_path,
        output_dir=args.output_dir,
        language=args.language,
        psm=args.psm,
    )


if __name__ == "__main__":
    main()
