import hashlib

from deterministic_slm.hashing import stable_text_hash


def test_stable_text_hash_returns_sha256_hex_digest_for_utf8_text():
    text = "deterministic SLM cafe"

    assert stable_text_hash(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_stable_text_hash_handles_non_ascii_text_as_utf8():
    text = "Grusse aus Koln: äöü"

    assert stable_text_hash(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()
