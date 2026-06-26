"""Character-level Armenian pixel-CNN expert and standalone JSON CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from N05handwritten_ocr.character_detector.inference import predict
else:
    from .inference import predict


EXPERT_NAME = "character_detector"


def get_expert_manifest(settings=None):
    """Describe the implemented letter-level CNN expert."""
    settings = settings or {}
    model_path = settings.get("model_path")
    model_name = (
        settings.get("model_name")
        or (Path(model_path).stem.replace("_best", "") if model_path else None)
        or "glyph_classifier_v0_1"
    )
    return {
        "expert_name": EXPERT_NAME,
        "display_name": "Character Detector CNN",
        "enabled": bool(settings.get("enabled", False)),
        "implemented": True,
        "status": "pixel_cnn_ready",
        "unit_level": "character",
        "returns_text": True,
        "candidate_schema": "n05_candidate_evidence_v1",
        "model_name": model_name,
        "model_path": model_path,
        "recommended_input": "single_character_analysis_mask",
    }


def recognize(crop_path, context=None, settings=None):
    """Recognize one glyph and return the shared N05 expert contract."""
    settings = settings or {}
    if not bool(settings.get("enabled", False)):
        return {
            "expert_name": EXPERT_NAME,
            "attempted": False,
            "status": "disabled",
            "crop_path": crop_path,
            "candidates": [],
            "evidence": None,
            "error": None,
        }
    try:
        evidence = predict(crop_path, settings=settings)
        return {
            "expert_name": EXPERT_NAME,
            "attempted": True,
            "status": "completed",
            "crop_path": str(Path(crop_path).expanduser().resolve()),
            "candidates": evidence["candidates"],
            "evidence": evidence,
            "error": None,
        }
    except Exception as error:
        return {
            "expert_name": EXPERT_NAME,
            "attempted": True,
            "status": "failed",
            "crop_path": str(crop_path),
            "candidates": [],
            "evidence": None,
            "error": str(error),
        }


def main():
    """Run one image through the CNN and optionally save JSON evidence."""
    parser = argparse.ArgumentParser(description="Run Armenian glyph CNN inference.")
    parser.add_argument("image", help="Character image or binary-mask path.")
    parser.add_argument("--out", default="", help="Optional result JSON path.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = recognize(
        args.image,
        settings={"enabled": True, "top_k": args.top_k, "device": args.device},
    )
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        output_path = Path(args.out).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
        print(output_path)
    else:
        print(serialized, end="")
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
