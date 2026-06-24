"""Word-level Armenian OCR expert contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inference import predict_word_level  # type: ignore
else:
    from .inference import predict_word_level

EXPERT_NAME = "word_level_ocr"


def get_expert_manifest(settings=None):
    """Describe the word-level OCR expert without loading model weights.

    Args:
        settings: Optional expert settings dictionary.

    Returns:
        Expert capability and implementation-status metadata.
    """
    settings = settings or {}
    return {
        "expert_name": EXPERT_NAME,
        "display_name": "Word-Level OCR",
        "enabled": bool(settings.get("enabled", False)),
        "implemented": True,
        "status": "preprocessor_and_language_assets_ready",
        "unit_level": "word",
        "returns_text": True,
        "candidate_schema": "n05_word_candidate_evidence_v1",
        "backend": settings.get("backend", "simplehtr_ctc"),
        "decoder": settings.get("decoder", "bestpath"),
        "recommended_input": "ocr_ready_word_or_text_unit_crop",
    }


def recognize(crop_path, context=None, settings=None):
    """Run one word/text-unit crop through the word-level OCR contract.

    Args:
        crop_path: Path to the OCR-ready word or text-unit crop.
        context: Optional document and routing evidence.
        settings: Optional expert settings dictionary.

    Returns:
        Standard expert-result dictionary.
    """
    settings = settings or {}
    if not bool(settings.get("enabled", False)):
        return {
            "expert_name": EXPERT_NAME,
            "attempted": False,
            "status": "disabled",
            "crop_path": crop_path,
            "context": context,
            "candidates": [],
            "evidence": None,
            "error": None,
        }
    try:
        evidence = predict_word_level(crop_path, settings=settings)
        status = evidence.get("status", "completed")
        return {
            "expert_name": EXPERT_NAME,
            "attempted": True,
            "status": status,
            "crop_path": str(Path(crop_path).expanduser().resolve()),
            "context": context,
            "candidates": evidence.get("candidates", []),
            "evidence": evidence,
            "error": None,
        }
    except Exception as error:
        return {
            "expert_name": EXPERT_NAME,
            "attempted": True,
            "status": "failed",
            "crop_path": str(crop_path),
            "context": context,
            "candidates": [],
            "evidence": None,
            "error": str(error),
        }


def main():
    """Run the word-level OCR expert from the command line."""

    parser = argparse.ArgumentParser(description="Run N05 word-level OCR evidence.")
    parser.add_argument("image", help="Word or text-unit crop path.")
    parser.add_argument("--out", default="", help="Optional result JSON path.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--language-prior",
        action="store_true",
        help="Return corpus-prior debug candidates if model weights are missing.",
    )
    args = parser.parse_args()

    result = recognize(
        args.image,
        settings={
            "enabled": True,
            "top_k": args.top_k,
            "return_language_prior_when_model_missing": args.language_prior,
        },
    )
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        output_path = Path(args.out).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
        print(output_path)
    else:
        print(serialized, end="")
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
