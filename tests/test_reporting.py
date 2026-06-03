from __future__ import annotations

from dataclasses import dataclass

from deterministic_slm.reporting import RunRecord, summarize_runs


@dataclass(frozen=True)
class FakeRunRecord:
    text: str
    output_hash: str
    latency_seconds: float
    token_probabilities: dict[str, float] | None = None


def test_summarize_identical_runs_reports_no_divergence() -> None:
    runs = [
        FakeRunRecord(text="same output", output_hash="hash-a", latency_seconds=0.12),
        FakeRunRecord(text="same output", output_hash="hash-a", latency_seconds=0.15),
    ]

    summary = summarize_runs(runs)

    assert summary.unique_output_count == 1
    assert summary.output_hashes == ("hash-a",)
    assert summary.first_differing_position is None
    assert summary.token_probabilities is None
    assert summary.status == "no divergence observed"


def test_summarize_divergent_runs_reports_first_differing_position() -> None:
    runs = [
        FakeRunRecord(text="answer: alpha", output_hash="hash-a", latency_seconds=0.12),
        FakeRunRecord(text="answer: alpine", output_hash="hash-b", latency_seconds=0.15),
        FakeRunRecord(text="answer: alpha", output_hash="hash-a", latency_seconds=0.11),
    ]

    summary = summarize_runs(runs)

    assert summary.unique_output_count == 2
    assert summary.output_hashes == ("hash-a", "hash-b")
    assert summary.first_differing_position == 11
    assert summary.status == "divergence observed"


def test_summarize_divergence_when_one_output_is_prefix() -> None:
    runs = [
        FakeRunRecord(text="token", output_hash="hash-short", latency_seconds=0.1),
        FakeRunRecord(text="tokens", output_hash="hash-long", latency_seconds=0.1),
    ]

    summary = summarize_runs(runs)

    assert summary.first_differing_position == 5
    assert summary.status == "divergence observed"


def test_summarize_includes_token_probabilities_when_present() -> None:
    runs = [
        RunRecord(
            text="A",
            output_hash="hash-a",
            latency_seconds=0.12,
            token_probabilities={"A": 0.75, "B": 0.25},
        ),
        FakeRunRecord(
            text="B",
            output_hash="hash-b",
            latency_seconds=0.13,
            token_probabilities={"A": 0.49, "B": 0.51},
        ),
    ]

    summary = summarize_runs(runs)

    assert summary.token_probabilities == (
        {"A": 0.75, "B": 0.25},
        {"A": 0.49, "B": 0.51},
    )


def test_summarize_empty_runs_reports_no_divergence() -> None:
    summary = summarize_runs([])

    assert summary.unique_output_count == 0
    assert summary.output_hashes == ()
    assert summary.first_differing_position is None
    assert summary.token_probabilities is None
    assert summary.status == "no divergence observed"
