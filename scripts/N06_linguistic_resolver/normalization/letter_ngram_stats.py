"""Letter n-gram statistics for Armenian word plausibility.

This is an anti-gibberish signal, not a dictionary validator. A word can be
unknown and still Armenian-like; a word can also be in OCR output and look
statistically impossible. N06 keeps both facts separate.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

from .armenian_word_normalizer import (
    armenian_character_report,
    normalize_armenian_word,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(path_value: str | Path) -> Path:
    """Resolve repo-relative paths without requiring callers to know cwd."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _weight_from_count(count: int, mode: str) -> float:
    """Return a stable corpus weight for one dictionary entry."""

    count = max(1, int(count or 1))
    if mode == "raw_count":
        return float(count)
    if mode == "sqrt_count":
        return math.sqrt(count)
    return math.log1p(count)


def _iter_ngrams(word: str, order: int) -> list[str]:
    """Return padded character n-grams for one word."""

    padded = "^" * (order - 1) + word + "$"
    return [padded[index : index + order] for index in range(len(padded) - order + 1)]


class ArmenianLetterNgramStats:
    """Smoothed letter n-gram model trained from Armenian word forms."""

    def __init__(self, settings: dict | None = None):
        self.settings = settings or {}
        self.orders = [
            int(order)
            for order in self.settings.get("orders", [2, 3, 4])
            if int(order) >= 1
        ]
        self.smoothing = float(self.settings.get("smoothing", 0.1))
        self.weight_mode = str(self.settings.get("weight_mode", "log_count"))
        self.ngram_counts: dict[int, Counter] = {order: Counter() for order in self.orders}
        self.total_counts: dict[int, float] = {order: 0.0 for order in self.orders}
        self.vocabularies: dict[int, set[str]] = {order: set() for order in self.orders}
        self.training_scores: list[float] = []
        self.training_mean = 0.0
        self.training_std = 1.0
        self.word_count = 0
        self.status = "empty"

    def fit_from_corpus(self, corpus_path: str | Path | None = None) -> "ArmenianLetterNgramStats":
        """Load Armenian word-frequency TSV and build n-gram counts."""

        path = _resolve_path(
            corpus_path or self.settings.get("corpus_path", "datasets/word_level_ocr/armenian_word_frequencies.tsv")
        )
        max_words = int(self.settings.get("max_words", 250000))
        min_word_count = int(self.settings.get("min_word_count", 1))
        normalization_settings = self.settings.get("normalization", {})

        if not path.exists():
            self.status = "missing_corpus"
            return self

        rows: list[tuple[str, int]] = []
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                raw_word = row.get("word") or ""
                try:
                    count = int(row.get("count") or 1)
                except ValueError:
                    count = 1
                if count < min_word_count:
                    continue
                word = normalize_armenian_word(raw_word, normalization_settings)
                report = armenian_character_report(word)
                if not word or report["unknown_character_count"]:
                    continue
                rows.append((word, count))
                if len(rows) >= max_words:
                    break

        for word, count in rows:
            weight = _weight_from_count(count, self.weight_mode)
            for order in self.orders:
                for ngram in _iter_ngrams(word, order):
                    self.ngram_counts[order][ngram] += weight
                    self.total_counts[order] += weight
                    self.vocabularies[order].add(ngram)

        self.word_count = len(rows)
        if not rows:
            self.status = "empty_corpus"
            return self

        self.training_scores = [
            self._average_log_probability(word)
            for word, _count in rows[: min(len(rows), 50000)]
        ]
        self.training_mean = sum(self.training_scores) / max(1, len(self.training_scores))
        variance = sum(
            (score - self.training_mean) ** 2 for score in self.training_scores
        ) / max(1, len(self.training_scores))
        self.training_std = max(math.sqrt(variance), 1e-6)
        self.status = "trained"
        return self

    def _ngram_log_probability(self, ngram: str, order: int) -> float:
        """Return add-k smoothed log probability for one n-gram."""

        vocabulary_size = max(1, len(self.vocabularies.get(order, set())))
        total = self.total_counts.get(order, 0.0)
        count = self.ngram_counts.get(order, Counter()).get(ngram, 0.0)
        probability = (count + self.smoothing) / (
            total + self.smoothing * vocabulary_size
        )
        return math.log(max(probability, 1e-12))

    def _average_log_probability(self, word: str) -> float:
        """Return mean log probability across configured n-gram orders."""

        values = []
        for order in self.orders:
            ngrams = _iter_ngrams(word, order)
            if not ngrams:
                continue
            values.extend(
                self._ngram_log_probability(ngram, order)
                for ngram in ngrams
            )
        if not values:
            return -99.0
        return sum(values) / len(values)

    def score_word(self, text: str, normalization_settings: dict | None = None) -> dict:
        """Score one word candidate for Armenian letter-sequence plausibility."""

        normalization_settings = normalization_settings or self.settings.get("normalization", {})
        normalized = normalize_armenian_word(text, normalization_settings)
        char_report = armenian_character_report(normalized)
        unknown_ratio = float(char_report["unknown_character_ratio"])
        if not normalized:
            return {
                "status": "empty",
                "text": text,
                "normalized_text": normalized,
                "average_log_probability": -99.0,
                "z_score": -99.0,
                "confidence": 0.0,
                "is_gibberish_like": True,
                "character_report": char_report,
            }

        average_log_probability = self._average_log_probability(normalized)
        z_score = (average_log_probability - self.training_mean) / self.training_std
        confidence = 1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, z_score))))
        gibberish_threshold = float(self.settings.get("gibberish_z_threshold", -2.0))
        unknown_threshold = float(
            self.settings.get("unknown_character_ratio_threshold", 0.25)
        )
        is_gibberish_like = (
            self.status != "trained"
            or z_score < gibberish_threshold
            or unknown_ratio > unknown_threshold
        )
        return {
            "status": self.status,
            "text": text,
            "normalized_text": normalized,
            "average_log_probability": float(average_log_probability),
            "z_score": float(z_score),
            "confidence": float(confidence),
            "is_gibberish_like": bool(is_gibberish_like),
            "character_report": char_report,
            "model": {
                "orders": self.orders,
                "word_count": self.word_count,
                "training_mean": self.training_mean,
                "training_std": self.training_std,
            },
        }
