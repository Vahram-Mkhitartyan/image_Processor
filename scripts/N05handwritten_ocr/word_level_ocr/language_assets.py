"""Language assets for Armenian word-level OCR decoding and scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[2]
DEFAULT_ASSET_DIR = PROJECT_ROOT / "datasets" / "word_level_ocr"
DEFAULT_WORD_FREQUENCIES = DEFAULT_ASSET_DIR / "armenian_word_frequencies.tsv"
DEFAULT_CTC_CORPUS = DEFAULT_ASSET_DIR / "armenian_ctc_corpus.txt"


@dataclass(frozen=True)
class WordFrequencyRecord:
    """One frequency-ranked Armenian word.

    Args:
        rank: Rank in the source corpus.
        word: Armenian word/token.
        count: Observed corpus frequency.

    Returns:
        Immutable record used by the language prior.
    """

    rank: int
    word: str
    count: int


def resolve_asset_path(path: str | Path | None, default_path: Path) -> Path:
    """Resolve an optional asset path against the project root.

    Args:
        path: Optional absolute or project-relative path.
        default_path: Fallback path.

    Returns:
        Absolute asset path.
    """

    if not path:
        return default_path.resolve()
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


@lru_cache(maxsize=8)
def load_word_frequencies(
    frequency_path: str | Path | None = None,
    max_words: int = 250000,
) -> tuple[WordFrequencyRecord, ...]:
    """Load the cleaned Armenian word-frequency table.

    Args:
        frequency_path: Optional TSV path.
        max_words: Maximum number of ranked words to keep in memory.

    Returns:
        Tuple of frequency records sorted by rank.
    """

    path = resolve_asset_path(frequency_path, DEFAULT_WORD_FREQUENCIES)
    if not path.is_file():
        return tuple()

    records: list[WordFrequencyRecord] = []
    with path.open("r", encoding="utf-8") as file:
        header = file.readline()
        if not header.startswith("rank\tword\tcount"):
            file.seek(0)
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            try:
                record = WordFrequencyRecord(
                    rank=int(parts[0]),
                    word=parts[1],
                    count=int(parts[2]),
                )
            except ValueError:
                continue
            records.append(record)
            if max_words > 0 and len(records) >= max_words:
                break
    return tuple(records)


def corpus_path_for_settings(settings: dict | None = None) -> Path:
    """Return the CTC corpus path configured for the expert.

    Args:
        settings: Optional word-level expert settings.

    Returns:
        Absolute corpus path.
    """

    settings = settings or {}
    return resolve_asset_path(settings.get("corpus_path"), DEFAULT_CTC_CORPUS)


def language_prior_for_word(
    word: str,
    frequency_records: tuple[WordFrequencyRecord, ...],
) -> float:
    """Estimate a normalized prior score from corpus frequency.

    Args:
        word: Candidate word.
        frequency_records: Loaded frequency records.

    Returns:
        Score in ``[0, 1]`` where common words receive larger values.
    """

    if not word or not frequency_records:
        return 0.0
    total = sum(max(0, record.count) for record in frequency_records)
    if total <= 0:
        return 0.0
    lookup = {record.word: record.count for record in frequency_records}
    count = lookup.get(word, 0)
    if count <= 0:
        return 0.0
    return min(1.0, math.log1p(count) / math.log1p(total))


def top_language_candidates(
    settings: dict | None = None,
    limit: int = 10,
) -> list[dict]:
    """Return the most common Armenian words as a debug language baseline.

    Args:
        settings: Optional expert settings.
        limit: Maximum candidates to return.

    Returns:
        JSON-safe ranked language-prior candidates.
    """

    settings = settings or {}
    records = load_word_frequencies(
        settings.get("word_frequency_path"),
        int(settings.get("max_language_words", 250000)),
    )
    if not records:
        return []
    top_records = records[: max(0, limit)]
    max_count = max(record.count for record in top_records) if top_records else 1
    candidates = []
    for index, record in enumerate(top_records, start=1):
        candidates.append(
            {
                "rank": index,
                "text": record.word,
                "confidence": record.count / max(1, max_count),
                "source": "armenian_language_prior",
                "evidence_kind": "word_prior",
                "corpus_rank": record.rank,
                "corpus_count": record.count,
            }
        )
    return candidates
