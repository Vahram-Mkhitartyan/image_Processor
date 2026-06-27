"""N06 Armenian linguistic resolver orchestrator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.N06_linguistic_resolver.input.word_token_builder import (
    build_tokens_from_n05_payload,
    build_tokens_from_words,
)
from scripts.N06_linguistic_resolver.normalization import (
    ArmenianLetterNgramStats,
    normalize_armenian_word,
)
from scripts.N06_linguistic_resolver.reconstruction import suggest_ngram_repairs
from scripts.N06_linguistic_resolver.schemas import (
    NODE_NAME,
    NODE_VERSION,
    make_ngram_evidence,
    make_resolved_word,
)


DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "scripts" / "N06_linguistic_resolver" / "settings.json"


def load_json(path: str | Path) -> dict:
    """Load a JSON file as UTF-8."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: dict, path: str | Path) -> str:
    """Save readable JSON and return the path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return str(output_path)


def resolve_word_tokens(tokens: list[dict], settings: dict | None = None) -> dict:
    """Run N06 v0.1 evidence generation over word tokens."""

    settings = settings or load_json(DEFAULT_SETTINGS_PATH)
    normalization_settings = settings.get("normalization", {})
    ngram_settings = dict(settings.get("letter_ngram_stats", {}))
    ngram_settings["normalization"] = normalization_settings
    repair_settings = dict(settings.get("ngram_candidate_repair", {}))
    ngram_model = ArmenianLetterNgramStats(ngram_settings).fit_from_corpus()

    resolved_words = []
    for token in tokens:
        text = token.get("text", "")
        normalized = normalize_armenian_word(text, normalization_settings)
        evidence_sources = []
        warnings = []
        if ngram_settings.get("enabled", True):
            score = ngram_model.score_word(text, normalization_settings)
            status = "gibberish_like" if score["is_gibberish_like"] else "armenian_like"
            if score["is_gibberish_like"]:
                warnings.append("letter_ngram_stats_flagged_gibberish_like")
            evidence_sources.append(
                make_ngram_evidence(
                    text=text,
                    normalized_text=normalized,
                    status=status,
                    score=score["z_score"],
                    confidence=score["confidence"],
                    details=score,
                )
            )
        ngram_repairs = {}
        if repair_settings.get("enabled", True):
            ngram_repairs = suggest_ngram_repairs(
                text=text,
                character_candidates=token.get("character_candidates", []),
                ngram_model=ngram_model,
                settings=repair_settings,
                normalization_settings=normalization_settings,
            )
        resolved_word = make_resolved_word(
            token=token,
            normalized_text=normalized,
            evidence_sources=evidence_sources,
            warnings=warnings,
        )
        resolved_word["ngram_candidate_repair"] = ngram_repairs
        resolved_words.append(resolved_word)

    return {
        "node": NODE_NAME,
        "node_version": NODE_VERSION,
        "status": "completed",
        "token_count": len(tokens),
        "settings": settings,
        "ngram_model_status": ngram_model.status,
        "ngram_training_word_count": ngram_model.word_count,
        "resolved_words": resolved_words,
        "trusted_as_final": False,
    }


def build_resolver_output_from_words(words: list[str], settings: dict | None = None) -> dict:
    """Convenience wrapper for manual smoke tests."""

    return resolve_word_tokens(build_tokens_from_words(words), settings=settings)


def main() -> None:
    """CLI for quick N06 smoke runs."""

    parser = argparse.ArgumentParser(description="Run N06 linguistic resolver.")
    parser.add_argument("--settings", default=str(DEFAULT_SETTINGS_PATH))
    parser.add_argument("--input-json", default="")
    parser.add_argument("--words", nargs="*", default=[])
    parser.add_argument("--output-json", default="temp_processing/n06_linguistic_resolver/result.json")
    args = parser.parse_args()

    settings = load_json(args.settings)
    if args.input_json:
        payload = load_json(args.input_json)
        tokens = build_tokens_from_n05_payload(payload)
    else:
        tokens = build_tokens_from_words(args.words, source="cli_words")
    result = resolve_word_tokens(tokens, settings=settings)
    output_path = save_json(result, args.output_json)
    print("N06 resolver output:", output_path)
    print("Tokens:", result["token_count"])
    print("N-gram model:", result["ngram_model_status"], result["ngram_training_word_count"])


if __name__ == "__main__":
    main()
