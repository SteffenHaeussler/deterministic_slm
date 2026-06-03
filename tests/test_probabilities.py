import math

import pytest

from deterministic_slm.probabilities import logprob_to_probability, softmax


def test_softmax_preserves_keys_and_normalizes_probabilities():
    logits = {"cat": 1.0, "dog": 2.0, "owl": 3.0}

    probabilities = softmax(logits)

    assert list(probabilities) == ["cat", "dog", "owl"]
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["owl"] > probabilities["dog"] > probabilities["cat"]


def test_softmax_is_numerically_stable_for_large_logits():
    probabilities = softmax({"a": 1000.0, "b": 1001.0})

    expected_b = math.exp(1.0) / (1.0 + math.exp(1.0))
    assert probabilities["b"] == pytest.approx(expected_b)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_softmax_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        softmax({})


def test_logprob_to_probability_exponentiates_logprob():
    assert logprob_to_probability(-0.25) == pytest.approx(math.exp(-0.25))

