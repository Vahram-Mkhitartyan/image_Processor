"""Input adapters for N06 linguistic resolver."""

from .word_token_builder import build_tokens_from_n05_payload, build_tokens_from_words

__all__ = ["build_tokens_from_n05_payload", "build_tokens_from_words"]
