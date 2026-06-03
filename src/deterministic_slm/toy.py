from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from collections.abc import Callable


@dataclass(frozen=True)
class ToyScenario:
    name: str
    logits: dict[str, float]
    probabilities: dict[str, float]
    selected_token: str
    output_hash: str


@dataclass(frozen=True)
class ToyDemoResult:
    scenarios: tuple[ToyScenario, ...]


_TOKEN_CONTRIBUTIONS: dict[str, tuple[float, float, float]] = {
    "A": (1.0, 2**-25, -(2**-24)),
    "B": (1.0, -(2**-27), -(2**-25)),
}


def run_toy_demo() -> ToyDemoResult:
    return ToyDemoResult(
        scenarios=(
            _build_scenario("left_associative", _reduce_left_associative),
            _build_scenario("nested_pair", _reduce_nested_pair),
        )
    )


def _build_scenario(
    name: str,
    reducer: Callable[[tuple[float, float, float]], float],
) -> ToyScenario:
    logits = {
        token: reducer(contributions)
        for token, contributions in _TOKEN_CONTRIBUTIONS.items()
    }
    probabilities = _softmax(logits)
    selected_token = _greedy_select(logits)
    output_hash = _hash_output(name, logits, probabilities, selected_token)

    return ToyScenario(
        name=name,
        logits=logits,
        probabilities=probabilities,
        selected_token=selected_token,
        output_hash=output_hash,
    )


def _reduce_left_associative(values: tuple[float, float, float]) -> float:
    total = _float32(0.0)
    for value in values:
        total = _float32_add(total, value)
    return total


def _reduce_nested_pair(values: tuple[float, float, float]) -> float:
    first, second, third = values
    return _float32_add(first, _float32_add(second, third))


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    largest = max(logits.values())
    weights = {
        token: math.exp(logit - largest)
        for token, logit in logits.items()
    }
    total = math.fsum(weights.values())

    probabilities: dict[str, float] = {}
    remaining = 1.0
    items = list(weights.items())
    for token, weight in items[:-1]:
        probability = weight / total
        probabilities[token] = probability
        remaining -= probability
    probabilities[items[-1][0]] = remaining

    return probabilities


def _greedy_select(logits: dict[str, float]) -> str:
    return max(logits.items(), key=lambda item: item[1])[0]


def _hash_output(
    name: str,
    logits: dict[str, float],
    probabilities: dict[str, float],
    selected_token: str,
) -> str:
    payload = {
        "name": name,
        "logits": logits,
        "probabilities": probabilities,
        "selected_token": selected_token,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _float32_add(left: float, right: float) -> float:
    return _float32(_float32(left) + _float32(right))


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]
