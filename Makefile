.PHONY: help sync test hello ollama-probe ollama-probe-allow-missing-probs

help:
	@printf "Available targets:\n"
	@printf "  make sync                         Install dependencies with uv\n"
	@printf "  make test                         Run the test suite\n"
	@printf "  make hello                        Run the toy determinism demo\n"
	@printf "  make ollama-probe                 Run the Ollama probe\n"
	@printf "  make ollama-probe-allow-missing-probs  Run the Ollama probe without requiring probabilities\n"

sync:
	uv sync

test:
	uv run pytest -q

hello:
	uv run det-slm hello

ollama-probe:
	uv run det-slm ollama-probe

ollama-probe-allow-missing-probs:
	uv run det-slm ollama-probe --allow-missing-probs
