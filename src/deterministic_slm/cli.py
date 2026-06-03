from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import sys

import httpx

from deterministic_slm.backend import MissingProbabilitiesError, OpenAICompatibleBackend
from deterministic_slm.reporting import RunRecord, summarize_runs
from deterministic_slm.toy import DEFAULT_PROMPT, run_toy_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="det-slm",
        description="Explore deterministic SLM inference behavior.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("hello", help="Run the local toy determinism demo.")

    prompt_demo = subparsers.add_parser(
        "prompt-demo",
        help="Show constructed temperature-0 divergence, then probe Ollama.",
    )
    _add_probe_options(prompt_demo, default_prompt=DEFAULT_PROMPT)

    ollama = subparsers.add_parser(
        "ollama-probe",
        help="Probe an OpenAI-compatible local Ollama endpoint.",
    )
    _add_probe_options(ollama, default_prompt="Say hello in one short sentence.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "hello":
        return _run_hello()

    if args.command == "prompt-demo":
        return _run_prompt_demo(args)

    if args.command == "ollama-probe":
        return _run_ollama_probe(args)

    parser.error(f"unknown command: {args.command}")
    return 2


def _run_hello() -> int:
    result = run_toy_demo()
    print("Deterministic toy demo: reduction grouping can change greedy output.")
    for scenario in result.scenarios:
        print(f"\nscenario: {scenario.name}")
        print(f"prompt: {scenario.prompt}")
        print(f"temperature: {scenario.temperature}")
        print(f"logits: {_json(scenario.logits)}")
        print(f"probabilities: {_json(scenario.probabilities)}")
        print(f"selected_token: {scenario.selected_token}")
        print(f"output_text: {scenario.output_text}")
        print(f"output_hash: {scenario.output_hash}")

    repeated_result = run_toy_demo()
    hashes = tuple(scenario.output_hash for scenario in result.scenarios)
    repeated_hashes = tuple(
        scenario.output_hash for scenario in repeated_result.scenarios
    )
    selected_tokens = tuple(scenario.selected_token for scenario in result.scenarios)
    hashes_differ = len(set(hashes)) > 1
    selected_tokens_differ = len(set(selected_tokens)) > 1

    print("\nexpectation: repeating the same grouping gives the same hashes.")
    if hashes == repeated_hashes:
        print("actual: repeated same-grouping hashes match.")
    else:
        print("actual: repeated same-grouping hashes differ.")

    print(
        "expectation: changing only the reduction grouping can change "
        "the greedy answer."
    )
    if hashes_differ and selected_tokens_differ:
        print("actual: hashes differ and selected tokens differ.")
        print("result: the same math grouped differently picked a different token.")
    else:
        print("actual: hashes and selected tokens match.")
        print("result: grouping did not change the picked token in this demo.")
    return 0


def _run_prompt_demo(args: argparse.Namespace) -> int:
    print("constructed demo")
    _print_constructed_demo(args.prompt)

    print("\nlive Ollama probe")
    return _run_ollama_probe(args, final_analysis=True, optional_unavailable=True)


def _print_constructed_demo(prompt: str) -> None:
    result = run_toy_demo(prompt=prompt)
    for scenario in result.scenarios:
        print(f"\nscenario: {scenario.name}")
        print(f"prompt: {scenario.prompt}")
        print(f"temperature: {scenario.temperature}")
        print(f"logits: {_json(scenario.logits)}")
        print(f"probabilities: {_json(scenario.probabilities)}")
        print(f"selected_token: {scenario.selected_token}")
        print(f"output_text: {scenario.output_text}")
        print(f"output_hash: {scenario.output_hash}")

    hashes = tuple(scenario.output_hash for scenario in result.scenarios)
    selected_tokens = tuple(scenario.selected_token for scenario in result.scenarios)
    output_texts = tuple(scenario.output_text for scenario in result.scenarios)
    print("\nconstructed_status: divergence observed")
    print(f"constructed_unique_output_count: {len(set(output_texts))}")
    if len(set(hashes)) > 1 and len(set(selected_tokens)) > 1:
        print("constructed_result: temperature-0 greedy outputs differ.")
    else:
        print("constructed_result: temperature-0 greedy outputs match.")


def _run_ollama_probe(
    args: argparse.Namespace,
    *,
    final_analysis: bool = False,
    optional_unavailable: bool = False,
) -> int:
    backend = OpenAICompatibleBackend(
        model=args.model,
        base_url=args.base_url,
        seed=args.seed,
        timeout=args.timeout,
        allow_missing_probs=args.allow_missing_probs,
    )

    completions = []
    try:
        for _ in range(args.repeat):
            completions.append(backend.complete(args.prompt, max_tokens=args.max_tokens))
    except MissingProbabilitiesError as error:
        print(str(error), file=sys.stderr)
        print(
            "Use --allow-missing-probs to run anyway, but probability reporting "
            "will be incomplete.",
            file=sys.stderr,
        )
        return 1
    except httpx.HTTPStatusError as error:
        print(
            f"backend request failed: {error.response.status_code} "
            f"{error.response.reason_phrase}",
            file=sys.stderr,
        )
        print(error.response.text, file=sys.stderr)
        return 1
    except httpx.RequestError as error:
        if optional_unavailable:
            print("live_status: unavailable")
            print(f"backend: {args.base_url}")
            print(f"backend request failed: {error}")
            if final_analysis:
                print(
                    "\nfinal_analysis: no live backend result; "
                    "no live nondeterminism evidence collected."
                )
            return 0
        print(f"backend request failed: {error}", file=sys.stderr)
        return 1

    records = [
        RunRecord(
            text=completion.text,
            output_hash=completion.text_hash,
            latency_seconds=completion.latency_seconds,
            token_probabilities=_token_probability_map(completion.tokens),
        )
        for completion in completions
    ]
    summary = summarize_runs(records)

    print(f"backend: {args.base_url}")
    print(f"model: {args.model}")
    print(f"repeat: {args.repeat}")
    print(f"status: {summary.status}")
    print(f"unique_output_count: {summary.unique_output_count}")
    print(f"output_hashes: {_json(list(summary.output_hashes))}")
    print(f"first_differing_position: {summary.first_differing_position}")

    for index, completion in enumerate(completions, start=1):
        print(f"\nrun: {index}")
        print(f"text: {completion.text}")
        print(f"output_hash: {completion.text_hash}")
        print(f"latency_seconds: {completion.latency_seconds:.6f}")
        if completion.tokens:
            print(f"token_probabilities: {_json(completion.tokens)}")
        else:
            print("token_probabilities: unavailable")

    if final_analysis:
        print(f"\nfinal_analysis: {_final_analysis(summary.unique_output_count, args.repeat)}")

    return 0


def _final_analysis(unique_output_count: int, repeat: int) -> str:
    if unique_output_count == 1:
        return f"got 1 unique answer across {repeat} live runs."
    return f"got {unique_output_count} different answers across {repeat} live runs."


def _add_probe_options(
    parser: argparse.ArgumentParser,
    *,
    default_prompt: str,
) -> None:
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--prompt", default=default_prompt)
    parser.add_argument("--repeat", type=_positive_int, default=5)
    parser.add_argument("--max-tokens", type=_positive_int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=_positive_float, default=120.0)
    parser.add_argument("--allow-missing-probs", action="store_true")


def _token_probability_map(tokens: list[dict[str, object]]) -> dict[str, float] | None:
    probabilities = {
        f"{index}:{token['token']}": float(token["probability"])
        for index, token in enumerate(tokens)
        if "token" in token and "probability" in token
    }
    return probabilities or None


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
