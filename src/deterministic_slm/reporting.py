from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal


Status = Literal["no divergence observed", "divergence observed"]


@dataclass(frozen=True)
class RunRecord:
    text: str
    output_hash: str
    latency_seconds: float
    token_probabilities: Mapping[str, float] | None = None


@dataclass(frozen=True)
class RunSummary:
    unique_output_count: int
    output_hashes: tuple[str, ...]
    first_differing_position: int | None
    token_probabilities: tuple[Mapping[str, float], ...] | None
    status: Status


def summarize_runs(records: Iterable[Any]) -> RunSummary:
    runs = tuple(records)
    texts = tuple(str(run.text) for run in runs)
    output_hashes = _unique_in_order(str(run.output_hash) for run in runs)
    first_differing_position = _first_differing_position(texts)
    token_probabilities = _token_probabilities(runs)
    status = (
        "no divergence observed"
        if first_differing_position is None
        else "divergence observed"
    )

    return RunSummary(
        unique_output_count=len(set(texts)),
        output_hashes=output_hashes,
        first_differing_position=first_differing_position,
        token_probabilities=token_probabilities,
        status=status,
    )


def _unique_in_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)

    return tuple(unique)


def _first_differing_position(texts: tuple[str, ...]) -> int | None:
    if len(set(texts)) <= 1:
        return None

    shortest_length = min(len(text) for text in texts)
    for index in range(shortest_length):
        if len({text[index] for text in texts}) > 1:
            return index

    return shortest_length


def _token_probabilities(runs: tuple[Any, ...]) -> tuple[Mapping[str, float], ...] | None:
    probabilities = tuple(
        run.token_probabilities
        for run in runs
        if getattr(run, "token_probabilities", None) is not None
    )
    return probabilities or None
