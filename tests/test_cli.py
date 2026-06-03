from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from deterministic_slm import cli


def test_hello_prints_logits_probabilities_selected_token_and_hash(capsys):
    exit_code = cli.main(["hello"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "left_associative" in captured.out
    assert "nested_pair" in captured.out
    assert "logits" in captured.out
    assert "probabilities" in captured.out
    assert "selected_token" in captured.out
    assert "output_hash" in captured.out
    assert "expectation: repeating the same grouping gives the same hashes." in captured.out
    assert "actual: repeated same-grouping hashes match." in captured.out
    assert (
        "expectation: changing only the reduction grouping can change the greedy answer."
        in captured.out
    )
    assert "actual: hashes differ and selected tokens differ." in captured.out
    assert "result: the same math grouped differently picked a different token." in captured.out


def test_prompt_demo_prints_constructed_outputs_and_live_probe_summary(
    capsys,
    monkeypatch,
):
    created_backends = []

    class FakeBackend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = []
            created_backends.append(self)

        def complete(self, prompt: str, *, max_tokens: int):
            self.calls.append((prompt, max_tokens))
            return SimpleNamespace(
                text="Hello! I am stable.",
                text_hash="hash-stable",
                latency_seconds=0.01,
                tokens=[],
            )

    monkeypatch.setattr(cli, "OpenAICompatibleBackend", FakeBackend)

    exit_code = cli.main(["prompt-demo", "--repeat", "2", "--max-tokens", "7"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "constructed demo" in captured.out
    assert "live Ollama probe" in captured.out
    assert "prompt: hi, how are you?" in captured.out
    assert "temperature: 0" in captured.out
    assert "Hi, I'm doing well." in captured.out
    assert "Hello, I'm good, thanks for asking." in captured.out
    assert "status: no divergence observed" in captured.out
    assert "unique_output_count: 1" in captured.out
    assert "final_analysis: got 1 unique answer across 2 live runs." in captured.out
    assert created_backends[0].kwargs == {
        "model": "qwen2.5:0.5b",
        "base_url": "http://localhost:11434/v1",
        "seed": 0,
        "timeout": 120.0,
        "allow_missing_probs": False,
    }
    assert created_backends[0].calls == [
        ("hi, how are you?", 7),
        ("hi, how are you?", 7),
    ]


def test_prompt_demo_final_analysis_reports_multiple_unique_live_answers(
    capsys,
    monkeypatch,
):
    answers = ["alpha", "beta", "alpha", "gamma", "beta"]

    class FakeBackend:
        def __init__(self, **kwargs):
            pass

        def complete(self, prompt: str, *, max_tokens: int):
            text = answers.pop(0)
            return SimpleNamespace(
                text=text,
                text_hash=f"hash-{text}",
                latency_seconds=0.01,
                tokens=[],
            )

    monkeypatch.setattr(cli, "OpenAICompatibleBackend", FakeBackend)

    exit_code = cli.main(["prompt-demo", "--repeat", "5"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "final_analysis: got 3 different answers across 5 live runs." in captured.out


def test_prompt_demo_reports_unavailable_live_backend_without_failing(
    capsys,
    monkeypatch,
):
    class FakeBackend:
        def __init__(self, **kwargs):
            pass

        def complete(self, prompt: str, *, max_tokens: int):
            request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
            raise httpx.ConnectError("[Errno 111] Connection refused", request=request)

    monkeypatch.setattr(cli, "OpenAICompatibleBackend", FakeBackend)

    exit_code = cli.main(["prompt-demo", "--repeat", "1"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "constructed demo" in captured.out
    assert "constructed_status: divergence observed" in captured.out
    assert "live Ollama probe" in captured.out
    assert "live_status: unavailable" in captured.out
    assert "backend request failed: [Errno 111] Connection refused" in captured.out
    assert (
        "final_analysis: no live backend result; no live nondeterminism evidence collected."
        in captured.out
    )
    assert captured.err == ""


def test_ollama_probe_uses_default_backend_settings_and_reports_summary(
    capsys,
    monkeypatch,
):
    created_backends = []

    class FakeBackend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = []
            created_backends.append(self)

        def complete(self, prompt: str, *, max_tokens: int):
            self.calls.append((prompt, max_tokens))
            return SimpleNamespace(
                text="Hello.",
                text_hash="hash-hello",
                latency_seconds=0.01,
                tokens=[
                    {"token": "Hello", "logprob": -0.2, "probability": 0.818730753},
                    {"token": ".", "logprob": -0.1, "probability": 0.904837418},
                ],
            )

    monkeypatch.setattr(cli, "OpenAICompatibleBackend", FakeBackend)

    exit_code = cli.main(["ollama-probe", "--repeat", "2", "--max-tokens", "7"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert created_backends[0].kwargs == {
        "model": "qwen2.5:0.5b",
        "base_url": "http://localhost:11434/v1",
        "seed": 0,
        "timeout": 120.0,
        "allow_missing_probs": False,
    }
    assert created_backends[0].calls == [
        ("Say hello in one short sentence.", 7),
        ("Say hello in one short sentence.", 7),
    ]
    assert "no divergence observed" in captured.out
    assert "hash-hello" in captured.out
    assert "probability" in captured.out


def test_ollama_probe_supports_allow_missing_probabilities(monkeypatch):
    created_backends = []

    class FakeBackend:
        def __init__(self, **kwargs):
            created_backends.append(kwargs)

        def complete(self, prompt: str, *, max_tokens: int):
            return SimpleNamespace(
                text="Hello.",
                text_hash="hash-hello",
                latency_seconds=0.01,
                tokens=[],
            )

    monkeypatch.setattr(cli, "OpenAICompatibleBackend", FakeBackend)

    exit_code = cli.main(["ollama-probe", "--repeat", "1", "--allow-missing-probs"])

    assert exit_code == 0
    assert created_backends[0]["allow_missing_probs"] is True


def test_ollama_probe_returns_error_when_probabilities_are_missing(capsys, monkeypatch):
    class FakeBackend:
        def __init__(self, **kwargs):
            pass

        def complete(self, prompt: str, *, max_tokens: int):
            raise cli.MissingProbabilitiesError("response did not include token logprobs")

    monkeypatch.setattr(cli, "OpenAICompatibleBackend", FakeBackend)

    exit_code = cli.main(["ollama-probe", "--repeat", "1"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "response did not include token logprobs" in captured.err


def test_ollama_probe_returns_clean_error_for_backend_http_failure(
    capsys,
    monkeypatch,
):
    class FakeBackend:
        def __init__(self, **kwargs):
            pass

        def complete(self, prompt: str, *, max_tokens: int):
            request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
            response = httpx.Response(
                404,
                request=request,
                json={"error": {"message": "model 'qwen2.5:0.5b' not found"}},
            )
            raise httpx.HTTPStatusError(
                "Client error '404 Not Found'",
                request=request,
                response=response,
            )

    monkeypatch.setattr(cli, "OpenAICompatibleBackend", FakeBackend)

    exit_code = cli.main(["ollama-probe", "--repeat", "1"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "backend request failed: 404" in captured.err
    assert "model 'qwen2.5:0.5b' not found" in captured.err


def test_ollama_probe_accepts_custom_timeout(monkeypatch):
    created_backends = []

    class FakeBackend:
        def __init__(self, **kwargs):
            created_backends.append(kwargs)

        def complete(self, prompt: str, *, max_tokens: int):
            return SimpleNamespace(
                text="Hello.",
                text_hash="hash-hello",
                latency_seconds=0.01,
                tokens=[],
            )

    monkeypatch.setattr(cli, "OpenAICompatibleBackend", FakeBackend)

    exit_code = cli.main(
        ["ollama-probe", "--repeat", "1", "--allow-missing-probs", "--timeout", "3.5"]
    )

    assert exit_code == 0
    assert created_backends[0]["timeout"] == 3.5


def test_ollama_probe_rejects_non_positive_repeat():
    with pytest.raises(SystemExit) as error:
        cli.main(["ollama-probe", "--repeat", "0"])

    assert error.value.code == 2


def test_ollama_probe_rejects_non_positive_max_tokens():
    with pytest.raises(SystemExit) as error:
        cli.main(["ollama-probe", "--max-tokens", "0"])

    assert error.value.code == 2
