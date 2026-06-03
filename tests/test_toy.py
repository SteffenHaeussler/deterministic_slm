from __future__ import annotations

from deterministic_slm.toy import ToyDemoResult, run_toy_demo


def test_run_toy_demo_returns_two_probability_scenarios() -> None:
    result = run_toy_demo()

    assert isinstance(result, ToyDemoResult)
    assert len(result.scenarios) == 2

    for scenario in result.scenarios:
        assert scenario.name
        assert set(scenario.logits) == {"A", "B"}
        assert set(scenario.probabilities) == {"A", "B"}
        assert scenario.selected_token in {"A", "B"}
        assert len(scenario.output_hash) == 64
        assert all(0.0 <= value <= 1.0 for value in scenario.probabilities.values())
        assert sum(scenario.probabilities.values()) == 1.0


def test_toy_demo_exposes_reduction_grouping_divergence() -> None:
    first, second = run_toy_demo().scenarios

    assert first.name != second.name
    assert first.logits != second.logits
    assert first.probabilities != second.probabilities
    assert first.selected_token != second.selected_token
    assert first.output_hash != second.output_hash


def test_toy_demo_is_deterministic() -> None:
    assert run_toy_demo() == run_toy_demo()
