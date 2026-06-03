"""Probability helpers for deterministic inference reporting."""

import math


def softmax(logits: dict[str, float]) -> dict[str, float]:
    """Return normalized probabilities for a mapping of token logits."""
    if not logits:
        raise ValueError("softmax requires non-empty logits")

    max_logit = max(logits.values())
    exp_values = {
        key: math.exp(logit - max_logit)
        for key, logit in logits.items()
    }
    total = sum(exp_values.values())

    return {
        key: exp_value / total
        for key, exp_value in exp_values.items()
    }


def logprob_to_probability(logprob: float) -> float:
    """Convert a natural-log probability to a probability."""
    return math.exp(logprob)
