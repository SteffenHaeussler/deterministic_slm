# deterministic_slm

Exploration of deterministic SLM inference behavior, based on:

https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

The first milestone is a local `uv` harness. It demonstrates the numerical
shape of the problem without requiring a GPU, then optionally probes a local
Ollama model through an OpenAI-compatible API.

## Setup

```bash
uv sync
# or
make sync
```

## Toy Demo

Run the deterministic hello-world demo:

```bash
uv run det-slm hello
# or
make hello
```

This prints two fixed reduction groupings for the same toy logits. Each scenario
includes:

- the prompt
- `temperature=0`
- logits
- softmax probabilities
- the greedy `temperature=0` selected token
- the mapped output text
- an output hash

The toy demo is the reliable local reproduction. It shows how deterministic
floating-point work can still produce different greedy outputs when reduction
grouping changes.

## Combined Prompt Demo

Run a combined demo for the prompt `hi, how are you?`:

```bash
uv run det-slm prompt-demo
# or
make demo
# or
make prompt-demo
```

This first prints the reliable constructed temperature-0 divergence, then runs
the live Ollama probe with the same prompt. The live probe reports whether the
local backend produced multiple full text outputs. A stable local model reports
`no divergence observed` and the command still exits successfully. The command
ends with a final analysis line such as `got 1 unique answer across 5 live runs`
or `got 3 different answers across 5 live runs`.

## Optional Ollama Probe

Install and start Ollama, then pull the default small model:

```bash
ollama pull qwen2.5:0.5b
```

Run the probe:

```bash
uv run det-slm ollama-probe
# or
make ollama-probe
```

Useful options:

```bash
uv run det-slm prompt-demo --repeat 5 --max-tokens 32
uv run det-slm ollama-probe --repeat 5 --max-tokens 32
uv run det-slm ollama-probe --allow-missing-probs
uv run det-slm ollama-probe --model qwen2.5:0.5b --base-url http://localhost:11434/v1
make ollama-probe-allow-missing-probs
```

Run tests with:

```bash
make test
```

By default the probe requests token log probabilities and fails if the backend
does not return them. When log probabilities are present, the harness converts
each `logprob` to `probability = exp(logprob)` and includes those values in the
run output.

## Limitation

Local Ollama on a Mac can measure repeatability, but it does not validate the
batch-invariant vLLM deployment path from the blog. That later phase needs
suitable NVIDIA hardware and a vLLM server configured with:

```bash
VLLM_BATCH_INVARIANT=1
```

Current vLLM batch-invariant execution is intended for compute capability 9.0+
GPUs such as H100/H200/B100/B200-class hardware.
