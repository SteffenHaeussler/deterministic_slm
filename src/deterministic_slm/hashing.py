"""Stable hashing helpers."""

import hashlib


def stable_text_hash(text: str) -> str:
    """Return a deterministic SHA-256 hex digest for UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
