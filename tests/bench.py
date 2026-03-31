#!/usr/bin/env python3
"""Benchmark prompt processing and generation speed against a running Gallama server.

The script uses the OpenAI-compatible streaming endpoint because the streamed usage
chunk includes both prompt and completion token counts. That lets us derive:

- prompt processing speed ~= prompt_tokens / time_to_first_token
- generation speed ~= completion_tokens / (total_time - time_to_first_token)

It runs two suites:

1. Unique prompt runs at 4k / 8k / 16k target context sizes to avoid prompt cache hits.
2. Incremental prefix runs that grow the same prompt in small steps to expose prompt
   caching behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator, Optional

import httpx


DEFAULT_CONTEXT_TARGETS = [4096, 8192, 16384]
DEFAULT_CACHE_STEP = 2048
DEFAULT_OUTPUT_WORDS = 64
DEFAULT_CALIBRATION_WORDS = [1024, 4096]

WORD_BANK = [
    "amber", "ash", "atlas", "bamboo", "bay", "birch", "cinder", "cobalt",
    "copper", "coral", "dawn", "delta", "ember", "fern", "field", "flint",
    "frost", "garden", "glade", "granite", "grove", "harbor", "horizon",
    "iris", "jade", "jet", "lagoon", "laurel", "linen", "maple", "marble",
    "meadow", "meridian", "mist", "monsoon", "moss", "nectar", "oak",
    "obsidian", "ocean", "orchard", "pearl", "pine", "prairie", "quartz",
    "rain", "reef", "ripple", "river", "sage", "sand", "scarlet", "shadow",
    "shore", "silk", "silver", "slate", "spruce", "stone", "storm",
    "summit", "sunrise", "tide", "timber", "topaz", "vale", "velvet",
    "vine", "violet", "water", "willow", "wind", "zephyr",
]


@dataclass
class BenchResult:
    suite: str
    label: str
    target_prompt_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ttft_s: float
    total_s: float
    prompt_tok_s: float
    gen_tok_s: float
    content_chars: int
    reasoning_chars: int


def format_target_label(target_tokens: int) -> str:
    return f"{target_tokens / 1024:.1f}k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Gallama base URL")
    parser.add_argument("--model", help="Model name served by Gallama. Defaults to the first model from /v1/models")
    parser.add_argument(
        "--context-targets",
        nargs="+",
        type=int,
        default=DEFAULT_CONTEXT_TARGETS,
        help="Target prompt token counts for unique uncached runs",
    )
    parser.add_argument("--cache-start", type=int, default=4096, help="Prompt-cache sweep start tokens")
    parser.add_argument("--cache-end", type=int, default=16384, help="Prompt-cache sweep end tokens")
    parser.add_argument("--cache-step", type=int, default=DEFAULT_CACHE_STEP, help="Prompt-cache sweep step size")
    parser.add_argument("--output-words", type=int, default=DEFAULT_OUTPUT_WORDS, help="Requested output word count")
    parser.add_argument("--warmup", dest="warmup", action="store_true", default=True, help="Run one warmup request before benchmarking")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", help="Skip the warmup request")
    parser.add_argument("--skip-unique", action="store_true", help="Skip unique-prompt benchmark runs")
    parser.add_argument("--skip-cache", action="store_true", help="Skip incremental prompt-cache runs")
    parser.add_argument("--json-out", help="Optional path to save raw benchmark rows as JSON")
    return parser.parse_args()


def resolve_model(client: httpx.Client, base_url: str, requested_model: Optional[str]) -> str:
    if requested_model:
        return requested_model

    response = client.get(f"{base_url.rstrip('/')}/v1/models")
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") or []

    if not data:
        raise RuntimeError("No models returned by /v1/models. Pass --model or start the server with a loaded model.")

    model_id = data[0].get("id")
    if not model_id:
        raise RuntimeError("Could not resolve a model id from /v1/models.")

    return model_id


def batched_words(word_count: int, seed: int) -> list[str]:
    words: list[str] = []
    bank_len = len(WORD_BANK)
    for i in range(word_count):
        words.append(WORD_BANK[(seed + (i * 7) + (i // 11) * 3) % bank_len])
    return words


def join_words(words: Iterable[str], chunk_size: int = 40) -> str:
    words = list(words)
    lines = []
    for index in range(0, len(words), chunk_size):
        lines.append(" ".join(words[index:index + chunk_size]))
    return "\n".join(lines)


def build_unique_prompt(body_word_count: int, label: str, seed: int, output_words: int) -> str:
    intro = (
        f"Benchmark request {label}.\n"
        "Read the following context carefully. The content is intentionally synthetic and unique.\n"
    )
    body = join_words(batched_words(body_word_count, seed))
    instruction = (
        "\nAfter reading the full context, output exactly the word BENCH "
        f"{output_words} times separated by a single space and nothing else."
    )
    return intro + body + instruction


def build_incremental_prompt(all_body_words: list[str], body_word_count: int, output_words: int) -> str:
    intro = (
        "Prompt cache benchmark request.\n"
        "Read the following growing context carefully.\n"
    )
    body = join_words(all_body_words[:body_word_count])
    instruction = (
        "\nAfter reading the full context, output exactly the word BENCH "
        f"{output_words} times separated by a single space and nothing else."
    )
    return intro + body + instruction


def iter_sse_events(lines: Iterable[str]) -> Iterator[tuple[Optional[str], str]]:
    event_name: Optional[str] = None
    data_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if line == "":
            if data_lines:
                yield event_name, "\n".join(data_lines)
            event_name = None
            data_lines = []
            continue

        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())

    if data_lines:
        yield event_name, "\n".join(data_lines)


def stream_chat_completion(
    client: httpx.Client,
    base_url: str,
    model: str,
    prompt: str,
    *,
    max_completion_tokens: int,
) -> BenchResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.01,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_completion_tokens": max_completion_tokens,
    }

    start = time.perf_counter()
    first_content_at: Optional[float] = None
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    content_chars = 0
    reasoning_chars = 0

    with client.stream("POST", f"{base_url.rstrip('/')}/v1/chat/completions", json=payload) as response:
        response.raise_for_status()

        for _event_name, data in iter_sse_events(response.iter_lines()):
            if data == "[DONE]":
                break

            chunk = json.loads(data)
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                total_tokens = usage.get("total_tokens", total_tokens)

            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    if first_content_at is None:
                        first_content_at = time.perf_counter()
                    content_chars += len(content)

                reasoning = (
                    delta.get("reasoning")
                    or delta.get("reasoning_content")
                    or delta.get("thinking")
                )
                if reasoning:
                    if first_content_at is None:
                        first_content_at = time.perf_counter()
                    reasoning_chars += len(reasoning)

                tool_calls = delta.get("tool_calls")
                if tool_calls and first_content_at is None:
                    first_content_at = time.perf_counter()

    end = time.perf_counter()

    if first_content_at is None:
        first_content_at = end

    if prompt_tokens <= 0 and total_tokens <= 0:
        raise RuntimeError(
            "Stream completed without a usage chunk. Ensure the server returns "
            "stream usage data for /v1/chat/completions."
        )

    ttft_s = max(first_content_at - start, 1e-9)
    total_s = max(end - start, ttft_s)
    generation_window_s = max(total_s - ttft_s, 1e-9)
    prompt_tok_s = (prompt_tokens / ttft_s) if prompt_tokens else 0.0
    gen_tok_s = (completion_tokens / generation_window_s) if completion_tokens else 0.0

    return BenchResult(
        suite="",
        label="",
        target_prompt_tokens=0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        ttft_s=ttft_s,
        total_s=total_s,
        prompt_tok_s=prompt_tok_s,
        gen_tok_s=gen_tok_s,
        content_chars=content_chars,
        reasoning_chars=reasoning_chars,
    )


def fit_prompt_token_model(client: httpx.Client, base_url: str, model: str) -> tuple[float, float]:
    measurements: list[tuple[int, int]] = []

    for index, word_count in enumerate(DEFAULT_CALIBRATION_WORDS):
        prompt = build_unique_prompt(
            body_word_count=word_count,
            label=f"calibration-{index}",
            seed=1000 + index * 17,
            output_words=1,
        )
        result = stream_chat_completion(
            client,
            base_url,
            model,
            prompt,
            max_completion_tokens=4,
        )
        measurements.append((word_count, result.prompt_tokens))

    (x1, y1), (x2, y2) = measurements
    if x1 == x2:
        raise RuntimeError("Calibration failed: duplicate x values")

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - (slope * x1)
    if slope <= 0:
        raise RuntimeError(f"Calibration failed: non-positive slope {slope:.6f}")

    return slope, intercept


def body_words_for_target(target_prompt_tokens: int, slope: float, intercept: float) -> int:
    estimated = (target_prompt_tokens - intercept) / slope
    return max(256, int(round(estimated)))


def print_table(title: str, rows: list[BenchResult]) -> None:
    if not rows:
        return

    headers = [
        "suite",
        "label",
        "target",
        "prompt_tok",
        "out_tok",
        "ttft_s",
        "total_s",
        "prompt_tok_s",
        "gen_tok_s",
        "content_chars",
        "reason_chars",
    ]
    table_rows = [
        [
            row.suite,
            row.label,
            str(row.target_prompt_tokens),
            str(row.prompt_tokens),
            str(row.completion_tokens),
            f"{row.ttft_s:.2f}",
            f"{row.total_s:.2f}",
            f"{row.prompt_tok_s:.1f}",
            f"{row.gen_tok_s:.1f}",
            str(row.content_chars),
            str(row.reasoning_chars),
        ]
        for row in rows
    ]

    widths = [len(header) for header in headers]
    for table_row in table_rows:
        for index, value in enumerate(table_row):
            widths[index] = max(widths[index], len(value))

    print()
    print(title)
    print("-" * len(title))
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for table_row in table_rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(table_row)))


def main() -> int:
    args = parse_args()

    if args.cache_step <= 0:
        raise SystemExit("--cache-step must be > 0")
    if args.cache_end < args.cache_start:
        raise SystemExit("--cache-end must be >= --cache-start")

    results: list[BenchResult] = []

    with httpx.Client(timeout=None) as client:
        model = resolve_model(client, args.base_url, args.model)
        print(f"Base URL: {args.base_url}")
        print(f"Model: {model}")
        print(f"Unique contexts: {', '.join(format_target_label(value) for value in args.context_targets)}")
        print(
            "Cache sweep: "
            f"{format_target_label(args.cache_start)} -> {format_target_label(args.cache_end)} "
            f"(step {args.cache_step} tokens)"
        )

        if args.warmup:
            warmup_prompt = build_unique_prompt(
                body_word_count=512,
                label="warmup",
                seed=777,
                output_words=16,
            )
            _ = stream_chat_completion(
                client,
                args.base_url,
                model,
                warmup_prompt,
                max_completion_tokens=32,
            )

        slope, intercept = fit_prompt_token_model(client, args.base_url, model)
        print(f"Calibration: prompt_tokens ~= {intercept:.1f} + {slope:.3f} * body_words")

        if not args.skip_unique:
            for index, target_tokens in enumerate(args.context_targets):
                body_words = body_words_for_target(target_tokens, slope, intercept)
                prompt = build_unique_prompt(
                    body_word_count=body_words,
                    label=f"unique-{target_tokens}",
                    seed=2000 + index * 29,
                    output_words=args.output_words,
                )
                result = stream_chat_completion(
                    client,
                    args.base_url,
                    model,
                    prompt,
                    max_completion_tokens=max(args.output_words * 2, 64),
                )
                result.suite = "unique"
                result.label = format_target_label(target_tokens)
                result.target_prompt_tokens = target_tokens
                results.append(result)
                print(
                    f"[unique] target={target_tokens} prompt_tok={result.prompt_tokens} "
                    f"ttft={result.ttft_s:.2f}s prompt_tok_s={result.prompt_tok_s:.1f} "
                    f"gen_tok_s={result.gen_tok_s:.1f}"
                )

        if not args.skip_cache:
            cache_targets = list(range(args.cache_start, args.cache_end + 1, args.cache_step))
            max_body_words = body_words_for_target(args.cache_end, slope, intercept)
            incremental_body_words = batched_words(max_body_words, seed=5000)

            for target_tokens in cache_targets:
                body_words = body_words_for_target(target_tokens, slope, intercept)
                prompt = build_incremental_prompt(
                    incremental_body_words,
                    body_word_count=body_words,
                    output_words=args.output_words,
                )
                result = stream_chat_completion(
                    client,
                    args.base_url,
                    model,
                    prompt,
                    max_completion_tokens=max(args.output_words * 2, 64),
                )
                result.suite = "cache"
                result.label = format_target_label(target_tokens)
                result.target_prompt_tokens = target_tokens
                results.append(result)
                print(
                    f"[cache ] target={target_tokens} prompt_tok={result.prompt_tokens} "
                    f"ttft={result.ttft_s:.2f}s prompt_tok_s={result.prompt_tok_s:.1f} "
                    f"gen_tok_s={result.gen_tok_s:.1f}"
                )

    unique_rows = [row for row in results if row.suite == "unique"]
    cache_rows = [row for row in results if row.suite == "cache"]

    print_table("Unique Prompt Context Benchmark", unique_rows)
    print_table("Incremental Prompt Cache Benchmark", cache_rows)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump([asdict(row) for row in results], handle, indent=2)
        print()
        print(f"Saved raw results to {args.json_out}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.", file=sys.stderr)
        raise SystemExit(130)
