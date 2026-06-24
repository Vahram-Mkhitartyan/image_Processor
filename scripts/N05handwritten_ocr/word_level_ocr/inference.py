"""Inference contract for the N05 word-level OCR expert.

This module is intentionally model-agnostic for now. It prepares crops and
validates language assets today, while leaving a narrow adapter seam for a
future CRNN/CTC model trained on Armenian word images.
"""

from __future__ import annotations

from pathlib import Path

try:
    from .language_assets import (
        corpus_path_for_settings,
        load_word_frequencies,
        top_language_candidates,
    )
    from .preprocessing import WordPreprocessSettings, prepare_word_image_from_path
    from .model_runtime import WordCRNNRuntime
except ImportError:
    from language_assets import (  # type: ignore
        corpus_path_for_settings,
        load_word_frequencies,
        top_language_candidates,
    )
    from preprocessing import (  # type: ignore
        WordPreprocessSettings,
        prepare_word_image_from_path,
    )
    from model_runtime import WordCRNNRuntime  # type: ignore


def _settings_to_preprocess(settings: dict | None = None) -> WordPreprocessSettings:
    """Convert JSON settings into preprocessing settings.

    Args:
        settings: Optional word-level expert settings.

    Returns:
        Preprocessing settings dataclass.
    """

    settings = settings or {}
    preprocessing = settings.get("preprocessing", {})
    return WordPreprocessSettings(
        target_height=int(preprocessing.get("target_height", 32)),
        target_width=int(preprocessing.get("target_width", 128)),
        dynamic_width=bool(preprocessing.get("dynamic_width", True)),
        padding_px=int(preprocessing.get("padding_px", 16)),
        background_value=int(preprocessing.get("background_value", 255)),
        normalize_range=str(
            preprocessing.get("normalize_range", "minus_one_to_one")
        ),
    )


def _model_status(settings: dict | None = None) -> dict:
    """Inspect configured model files without importing heavy ML frameworks.

    Args:
        settings: Optional word-level expert settings.

    Returns:
        JSON-safe model availability metadata.
    """

    settings = settings or {}
    model_path = settings.get("model_path") or (
        "models/word_level_ocr_v0_1/word_level_ocr_v0_1.pt"
    )
    char_list_path = settings.get("char_list_path")
    resolved_model = str(Path(model_path).expanduser().resolve()) if model_path else None
    resolved_chars = (
        str(Path(char_list_path).expanduser().resolve()) if char_list_path else None
    )
    return {
        "backend": settings.get("backend", "simplehtr_ctc"),
        "model_path": resolved_model,
        "model_exists": bool(resolved_model and Path(resolved_model).is_file()),
        "char_list_path": resolved_chars,
        "char_list_exists": bool(resolved_chars and Path(resolved_chars).is_file()),
        "decoder": settings.get("decoder", "bestpath"),
    }


def predict_word_level(crop_path: str | Path, settings: dict | None = None) -> dict:
    """Prepare one crop and return word-level OCR evidence.

    Args:
        crop_path: OCR-ready word/text-unit crop.
        settings: Optional word-level expert settings.

    Returns:
        Standard evidence dictionary consumed by ``expert.recognize``.
    """

    settings = settings or {}
    crop = Path(crop_path).expanduser().resolve()
    prepared = prepare_word_image_from_path(crop, _settings_to_preprocess(settings))
    model_status = _model_status(settings)
    frequency_records = load_word_frequencies(
        settings.get("word_frequency_path"),
        int(settings.get("max_language_words", 250000)),
    )
    corpus_path = corpus_path_for_settings(settings)

    evidence = {
        "schema_version": "n05_word_level_ocr_evidence_v1",
        "source_crop_path": str(crop),
        "prepared_shape": list(prepared.shape),
        "model": model_status,
        "language_assets": {
            "corpus_path": str(corpus_path),
            "corpus_exists": corpus_path.is_file(),
            "word_frequency_count": len(frequency_records),
        },
        "candidates": [],
        "notes": [],
    }

    if not model_status["model_exists"]:
        evidence["status"] = "model_missing"
        evidence["notes"].append(
            "Word crop preprocessing and language assets are ready; CRNN/CTC weights are not configured yet."
        )
        if bool(settings.get("return_language_prior_when_model_missing", False)):
            evidence["candidates"] = top_language_candidates(
                settings,
                limit=int(settings.get("top_k", 5)),
            )
        return evidence

    runtime = WordCRNNRuntime(
        model_status["model_path"],
        device=str(settings.get("device", "auto")),
    )
    prediction = runtime.predict(crop, _settings_to_preprocess(settings))
    structure_evidence = {
        "decoded_length": prediction.get("decoded_length"),
        "predicted_length": prediction.get("predicted_length"),
        "length_confidence": prediction.get("length_confidence"),
        "predicted_bridge_count": prediction.get("predicted_bridge_count"),
        "bridge_confidence": prediction.get("bridge_confidence"),
        "split_line_candidates": prediction.get("split_line_candidates", []),
        "tokens": prediction.get("tokens", []),
    }
    evidence["status"] = "completed"
    evidence["prepared_shape"] = prediction["prepared_shape"]
    evidence["prediction"] = prediction
    evidence["structure_evidence"] = structure_evidence
    evidence["candidates"] = [
        {
            "rank": 1,
            "text": prediction["text"],
            "confidence": prediction["confidence"],
            "source": "word_level_crnn_ctc_greedy",
            "evidence_kind": "word_sequence",
            "model_name": prediction["model_name"],
            "decoded_length": prediction.get("decoded_length"),
            "predicted_length": prediction.get("predicted_length"),
            "length_confidence": prediction.get("length_confidence"),
            "tokens": prediction.get("tokens", []),
            "predicted_bridge_count": prediction.get("predicted_bridge_count"),
            "bridge_confidence": prediction.get("bridge_confidence"),
            "split_line_candidates": prediction.get("split_line_candidates", []),
        }
    ]
    return evidence
