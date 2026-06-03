# deterministic_slm

Exploration of deterministic SLM inference behavior, based on:

https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

The first milestone is a local `uv` harness. It demonstrates the numerical
shape of the problem without requiring a GPU, then optionally probes a local
Ollama model through an OpenAI-compatible API.

## What this repo is

This repository is meant to make the determinism problem easy to inspect on a
normal CPU machine. The `make demo` path intentionally works without GPU access:
it shows a constructed floating-point reduction case where the same
temperature-0 greedy setup can pick different outputs when the reduction
grouping changes.

This is not yet a full validation of the production GPU path described in the
blog post. Validating that requires suitable NVIDIA hardware and a vLLM server
configured for batch-invariant execution. I do not currently have access to that
GPU hardware, so the repo documents and demonstrates the problem locally while
leaving the full vLLM validation as the open blocker.

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

This first prints the reliable constructed temperature-0 divergence, then tries
the live Ollama probe with the same prompt. There are three different outcomes
to read separately:

- constructed toy divergence: reliable local demonstration
- unavailable live Ollama backend: no live evidence collected
- repeated live run with one output hash: negative or inconclusive live result

If no local Ollama/OpenAI-compatible backend is running, `prompt-demo` reports
`live_status: unavailable`, prints a final analysis saying no live
nondeterminism evidence was collected, and still exits successfully. If the live
probe runs, it reports whether the backend produced multiple full text outputs.

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

By default the direct `ollama-probe` command requests token log probabilities
and fails if the backend is unavailable or does not return them. When log
probabilities are present, the harness converts each `logprob` to
`probability = exp(logprob)` and includes those values in the run output.

## Limitation

Local CPU execution, notebook Transformers runs, and Ollama on a Mac are useful
for showing the numerical shape of the problem and measuring repeatability of a
local backend. A repeated greedy Transformers run that returns one output hash
means no divergence was observed in that run; it does not prove backend-wide
determinism across fresh processes, hardware, batching, or serving stacks. These
paths do not prove the batch-invariant vLLM deployment path from the blog.

That validation is the tricky part: it needs suitable NVIDIA hardware that I do
not currently have access to, plus a vLLM server configured with:

```bash
VLLM_BATCH_INVARIANT=1
```

Current vLLM batch-invariant execution is intended for compute capability 9.0+
GPUs such as H100/H200/B100/B200-class hardware.
