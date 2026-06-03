import hashlib
import json
import math

import httpx
import pytest

from deterministic_slm.backend import (
    MissingProbabilitiesError,
    OpenAICompatibleBackend,
)


def test_chat_completion_posts_deterministic_logprob_request_and_normalizes_response():
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "POST"
        assert request.url == "http://localhost:11434/v1/chat/completions"

        payload = json.loads(request.content)
        assert payload == {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Say hi"}],
            "temperature": 0,
            "seed": 1234,
            "max_tokens": 5,
            "logprobs": True,
            "top_logprobs": 0,
        }

        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hi"},
                        "logprobs": {
                            "content": [
                                {"token": "H", "logprob": -0.1},
                                {"token": "i", "logprob": -0.2},
                            ]
                        },
                    }
                ],
            },
        )

    backend = OpenAICompatibleBackend(
        model="llama3.2",
        seed=1234,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = backend.complete("Say hi", max_tokens=5)

    assert len(requests) == 1
    assert result.text == "Hi"
    assert result.text_hash == hashlib.sha256(b"Hi").hexdigest()
    assert result.latency_seconds >= 0
    assert result.raw_response["id"] == "chatcmpl-test"
    assert result.tokens == [
        {"token": "H", "logprob": -0.1, "probability": math.exp(-0.1)},
        {"token": "i", "logprob": -0.2, "probability": math.exp(-0.2)},
    ]


def test_missing_probabilities_raise_typed_error_by_default():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hi"},
                    }
                ],
            },
        )

    backend = OpenAICompatibleBackend(
        model="llama3.2",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(MissingProbabilitiesError, match="logprobs"):
        backend.complete("Say hi", max_tokens=5)


def test_missing_probabilities_can_be_allowed_explicitly():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hi"},
                    }
                ],
            },
        )

    backend = OpenAICompatibleBackend(
        model="llama3.2",
        allow_missing_probs=True,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = backend.complete("Say hi", max_tokens=5)

    assert result.text == "Hi"
    assert result.tokens == []


def test_backend_configures_http_timeout_when_client_is_not_injected(monkeypatch):
    captured_kwargs = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr("deterministic_slm.backend.httpx.Client", FakeClient)

    OpenAICompatibleBackend(model="llama3.2", timeout=42.5)

    assert captured_kwargs["timeout"] == 42.5
