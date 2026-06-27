"""Generic lexicon store placeholder for N06."""

from __future__ import annotations


class LexiconStore:
    """Small in-memory lexicon facade; persistent data comes later."""

    def __init__(self, entries: dict | None = None):
        self.entries = entries or {}

    def contains(self, word: str) -> bool:
        """Return whether ``word`` exists in the current store."""

        return word in self.entries
