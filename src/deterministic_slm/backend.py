from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any

import httpx


class MissingProbabilitiesError(RuntimeError):
    """Raised when a backend response omits required token probabilities."""


@dataclass(frozen=True)
class CompletionRecord:
    text: str
    tokens: list[dict[str, Any]]
    raw_response: dict[str, Any]
    latency_seconds: float
    text_hash: str


class OpenAICompatibleBackend:
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        endpoint: str = "/chat/completions",
        seed: int = 0,
        timeout: float = 120.0,
        allow_missing_probs: bool = False,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.seed = seed
        self.allow_missing_probs = allow_missing_probs
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, prompt: str, *, max_tokens: int) -> CompletionRecord:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": self.seed,
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": 0,
        }

        started = time.perf_counter()
        response = self._client.post(self._url, json=payload)
        latency_seconds = time.perf_counter() - started
        response.raise_for_status()
        raw_response = response.json()

        choice = self._first_choice(raw_response)
        text = choice.get("message", {}).get("content", "")
        tokens = self._extract_tokens(choice)

        return CompletionRecord(
            text=text,
            tokens=tokens,
            raw_response=raw_response,
            latency_seconds=latency_seconds,
            text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    @property
    def _url(self) -> str:
        endpoint = self.endpoint if self.endpoint.startswith("/") else f"/{self.endpoint}"
        return f"{self.base_url}{endpoint}"

    def _first_choice(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        choices = raw_response.get("choices")
        if not choices:
            raise ValueError("OpenAI-compatible response did not include choices")
        return choices[0]

    def _extract_tokens(self, choice: dict[str, Any]) -> list[dict[str, Any]]:
        content_logprobs = choice.get("logprobs", {}).get("content")
        if content_logprobs is None:
            if self.allow_missing_probs:
                return []
            raise MissingProbabilitiesError("response did not include token logprobs")

        tokens: list[dict[str, Any]] = []
        for item in content_logprobs:
            token = item.get("token")
            logprob = item.get("logprob")
            if token is None or logprob is None:
                if self.allow_missing_probs:
                    continue
                raise MissingProbabilitiesError(
                    "response token logprobs did not include token and logprob"
                )

            tokens.append(
                {
                    "token": token,
                    "logprob": logprob,
                    "probability": math.exp(logprob),
                }
            )
        return tokens
