"""Benchmark inference APIs — latency, throughput, time-to-first-token."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import click
from rich.console import Console
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Presets — pre-configured endpoint/model/benchmark combos
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "wandb-kimi": {
        "base_url": "https://api.inference.wandb.ai/v1",
        "model": "moonshotai/Kimi-K2.5",
        "api_key_env": "WANDB_API_KEY",
        "description": "W&B Inference — Kimi K2.5",
    },
    "wandb-kimi-large": {
        "base_url": "https://api.inference.wandb.ai/v1",
        "model": "moonshotai/Kimi-K2.5",
        "api_key_env": "WANDB_API_KEY",
        "data": "prompt_tokens=256,output_tokens=128",
        "num_requests": 200,
        "concurrency": 1,
        "max_seconds": 300,
        "description": "W&B Inference — Kimi K2.5 — autollm 'large' benchmark (200 sync reqs, 256in/128out)",
    },
}

# ---------------------------------------------------------------------------
# Dataset loading (HuggingFace, local files, synthetic)
# ---------------------------------------------------------------------------

GUIDELLM_DATASETS: dict[str, dict] = {
    "humaneval": {
        "path": "openai/openai_humaneval",
        "column": "prompt",
        "split": "test",
    },
    "cnn_dailymail": {
        "path": "abisee/cnn_dailymail",
        "column": "article",
        "split": "test",
        "args": {"name": "3.0.0"},
    },
}

# ~4 chars per token for English text
FILLER_WORDS = (
    "The quick brown fox jumps over the lazy dog near the river bank where "
    "several ducks were swimming peacefully under the warm afternoon sun while "
    "children played in the nearby park and their parents watched from wooden "
    "benches shaded by old oak trees whose leaves rustled gently in the breeze "
)


def parse_synthetic_spec(spec: str) -> dict[str, int]:
    """Parse 'prompt_tokens=256,output_tokens=128' into a dict."""
    result = {}
    for part in spec.split(","):
        k, v = part.strip().split("=")
        result[k.strip()] = int(v.strip())
    return result


def generate_synthetic_prompt(num_tokens: int) -> str:
    """Generate a prompt of approximately num_tokens tokens."""
    target_chars = num_tokens * 4
    repeats = (target_chars // len(FILLER_WORDS)) + 2
    text = (FILLER_WORDS * repeats)[:target_chars]
    return f"Repeat and continue this text for exactly the requested length:\n\n{text}"


def is_synthetic_spec(data_source: str) -> bool:
    return bool(re.match(r"^(\w+=\d+)(,\s*\w+=\d+)*$", data_source))


def load_prompts(data_source: str, num_prompts: int) -> tuple[list[str], int | None]:
    """Load prompts. Returns (prompts, output_tokens_override).

    output_tokens_override is set only for synthetic specs (prompt_tokens=N,output_tokens=M).
    """
    if is_synthetic_spec(data_source):
        spec = parse_synthetic_spec(data_source)
        prompt_tokens = spec.get("prompt_tokens", 256)
        output_tokens = spec.get("output_tokens", None)
        prompt = generate_synthetic_prompt(prompt_tokens)
        return [prompt], output_tokens

    from datasets import load_dataset

    path_obj = Path(data_source)
    if path_obj.exists():
        if data_source.endswith(".jsonl"):
            prompts = []
            for line in path_obj.read_text().strip().splitlines():
                row = json.loads(line)
                prompts.append(row.get("prompt") or row.get("text") or next(iter(row.values())))
            return prompts[:num_prompts], None
        return path_obj.read_text().strip().splitlines()[:num_prompts], None

    if data_source in GUIDELLM_DATASETS:
        cfg = GUIDELLM_DATASETS[data_source]
        ds = load_dataset(cfg["path"], split=cfg["split"], **cfg.get("args", {}))
        column = cfg["column"]
    else:
        ds = load_dataset(data_source, split="test")
        for col in ("prompt", "text", "question", "instruction", "content"):
            if col in ds.column_names:
                column = col
                break
        else:
            column = ds.column_names[0]

    prompts = [row[column] for row in ds]
    random.shuffle(prompts)
    prompts = [p[:8000] if len(p) > 8000 else p for p in prompts]
    return prompts[:num_prompts], None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    status: int
    ttft: float | None
    total_time: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None

    @property
    def output_tok_per_s(self) -> float | None:
        if self.error or self.total_time == 0 or self.completion_tokens == 0:
            return None
        return self.completion_tokens / self.total_time


@dataclass
class BenchmarkResult:
    requests: list[RequestResult] = field(default_factory=list)
    wall_time: float = 0.0

    @property
    def successes(self) -> list[RequestResult]:
        return [r for r in self.requests if r.error is None]

    @property
    def failures(self) -> list[RequestResult]:
        return [r for r in self.requests if r.error is not None]

    def summary(self) -> dict:
        ok = self.successes
        if not ok:
            return {"total": len(self.requests), "failures": len(self.failures)}

        latencies = [r.total_time for r in ok]
        ttfts = [r.ttft for r in ok if r.ttft is not None]
        total_completion_tokens = sum(r.completion_tokens for r in ok)
        total_prompt_tokens = sum(r.prompt_tokens for r in ok)

        per_req_output_tps = [r.output_tok_per_s for r in ok if r.output_tok_per_s is not None]

        latencies.sort()
        if ttfts:
            ttfts.sort()
        if per_req_output_tps:
            per_req_output_tps.sort()

        def percentile(vals: list[float], p: float) -> float:
            idx = int(len(vals) * p)
            return vals[min(idx, len(vals) - 1)]

        result = {
            "total": len(self.requests),
            "successes": len(ok),
            "failures": len(self.failures),
            "wall_time_s": round(self.wall_time, 3),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "throughput_req_per_s": round(len(ok) / self.wall_time, 2) if self.wall_time else 0,
            "throughput_tok_per_s": round(total_completion_tokens / self.wall_time, 2) if self.wall_time else 0,
        }

        if per_req_output_tps:
            result.update({
                "output_tok_per_s_mean": round(sum(per_req_output_tps) / len(per_req_output_tps), 1),
                "output_tok_per_s_p50": round(percentile(per_req_output_tps, 0.50), 1),
                "output_tok_per_s_p90": round(percentile(per_req_output_tps, 0.90), 1),
            })

        result.update({
            "latency_mean_s": round(sum(latencies) / len(latencies), 3),
            "latency_p50_s": round(percentile(latencies, 0.50), 3),
            "latency_p90_s": round(percentile(latencies, 0.90), 3),
            "latency_p99_s": round(percentile(latencies, 0.99), 3),
        })

        if ttfts:
            result.update({
                "ttft_mean_s": round(sum(ttfts) / len(ttfts), 3),
                "ttft_p50_s": round(percentile(ttfts, 0.50), 3),
                "ttft_p90_s": round(percentile(ttfts, 0.90), 3),
            })

        return result

# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    payload: dict,
    stream: bool,
) -> RequestResult:
    t0 = time.perf_counter()
    ttft = None
    prompt_tokens = 0
    completion_tokens = 0

    try:
        async with session.post(url, headers=headers, json=payload) as resp:
            if stream:
                first_chunk = True
                async for line in resp.content:
                    text = line.decode("utf-8").strip()
                    if not text or not text.startswith("data:"):
                        continue
                    if text == "data: [DONE]":
                        break
                    if first_chunk:
                        ttft = time.perf_counter() - t0
                        first_chunk = False
                    try:
                        chunk = json.loads(text[len("data:"):].strip())
                        usage = chunk.get("usage")
                        if usage:
                            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                            completion_tokens = usage.get("completion_tokens", completion_tokens)
                        for choice in chunk.get("choices", []):
                            delta = choice.get("delta", {})
                            if delta.get("content"):
                                completion_tokens = max(completion_tokens, 1)
                    except json.JSONDecodeError:
                        pass
            else:
                body = await resp.json()
                usage = body.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

            total_time = time.perf_counter() - t0
            return RequestResult(
                status=resp.status,
                ttft=ttft,
                total_time=total_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error=None if resp.status == 200 else f"HTTP {resp.status}",
            )
    except Exception as e:
        return RequestResult(
            status=0,
            ttft=None,
            total_time=time.perf_counter() - t0,
            prompt_tokens=0,
            completion_tokens=0,
            error=str(e),
        )


async def run_benchmark(
    base_url: str,
    api_key: str | None,
    model: str,
    prompts: list[str],
    max_tokens: int,
    num_requests: int,
    concurrency: int,
    stream: bool,
    max_seconds: float | None = None,
) -> BenchmarkResult:
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    result = BenchmarkResult()
    sem = asyncio.Semaphore(concurrency)
    deadline = time.perf_counter() + max_seconds if max_seconds else None

    async def task(prompt: str):
        if deadline and time.perf_counter() >= deadline:
            return
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        async with sem:
            if deadline and time.perf_counter() >= deadline:
                return
            r = await send_request(session, url, headers, payload, stream)
            result.requests.append(r)
            ok = [rr for rr in result.requests if rr.error is None]
            if ok:
                done = len(result.requests)
                elapsed = time.perf_counter() - t0
                latest_otps = r.output_tok_per_s
                otps_str = f"{latest_otps:.0f}" if latest_otps else "?"
                console.print(
                    f"  [{done}/{num_requests}] {r.total_time:.2f}s  "
                    f"output_tok/s={otps_str}  "
                    f"(wall {elapsed:.1f}s)",
                    style="dim",
                )

    request_prompts = [prompts[i % len(prompts)] for i in range(num_requests)]

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        t0 = time.perf_counter()
        await asyncio.gather(*[task(p) for p in request_prompts])
        result.wall_time = time.perf_counter() - t0

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_results(summary: dict) -> None:
    table = Table(title="Benchmark Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for k, v in summary.items():
        table.add_row(k, str(v))
    console.print(table)


@click.command()
@click.option("--preset", type=click.Choice(list(PRESETS.keys())), default=None,
              help="Pre-configured endpoint/model/benchmark combo")
@click.option("--base-url", default=None, help="API base URL (e.g. http://localhost:8000/v1)")
@click.option("--api-key", default=None, envvar="API_KEY", help="API key (or set API_KEY env var)")
@click.option("--model", default=None, help="Model name to benchmark")
@click.option("--data", "data_source", default=None,
              help="Prompt source: 'prompt_tokens=256,output_tokens=128' (synthetic), "
                   "HF shortcut (humaneval), HF id, or local file")
@click.option("--prompt", default=None, help="Single prompt to repeat (ignored if --data is set)")
@click.option("--max-tokens", default=None, type=int, help="Max tokens per request (default: 256, or from synthetic spec)")
@click.option("--num-requests", "-n", default=None, type=int, help="Total requests (default: 10)")
@click.option("--concurrency", "-c", default=None, type=int, help="Concurrent requests (default: 1)")
@click.option("--max-seconds", default=None, type=float, help="Stop after this many seconds")
@click.option("--stream/--no-stream", default=True, help="Use streaming (measures TTFT)")
@click.option("--output", "-o", default=None, type=click.Path(), help="Save JSON results to file")
def cli(
    preset: str | None,
    base_url: str | None,
    api_key: str | None,
    model: str | None,
    data_source: str | None,
    prompt: str | None,
    max_tokens: int | None,
    num_requests: int | None,
    concurrency: int | None,
    max_seconds: float | None,
    stream: bool,
    output: str | None,
):
    """Benchmark an OpenAI-compatible inference API."""

    # Apply preset defaults (CLI flags override)
    if preset:
        cfg = PRESETS[preset]
        console.print(f"[bold green]Preset:[/bold green] {cfg['description']}")
        base_url = base_url or cfg["base_url"]
        model = model or cfg["model"]
        data_source = data_source or cfg.get("data")
        num_requests = num_requests or cfg.get("num_requests")
        concurrency = concurrency or cfg.get("concurrency")
        max_seconds = max_seconds or cfg.get("max_seconds")
        if not api_key:
            api_key = os.environ.get(cfg["api_key_env"])
            if not api_key:
                console.print(f"[red]Set {cfg['api_key_env']} or pass --api-key[/red]")
                raise SystemExit(1)

    # Defaults for anything not set by preset or CLI
    num_requests = num_requests or 10
    concurrency = concurrency or 1

    if not base_url:
        console.print("[red]--base-url is required (or use --preset)[/red]")
        raise SystemExit(1)
    if not model:
        console.print("[red]--model is required (or use --preset)[/red]")
        raise SystemExit(1)

    # Load prompts
    output_tokens_override = None
    if data_source:
        console.print(f"[bold]Data:[/bold] {data_source}")
        prompts, output_tokens_override = load_prompts(data_source, num_requests)
        if is_synthetic_spec(data_source):
            console.print(f"  Synthetic: ~{parse_synthetic_spec(data_source).get('prompt_tokens', '?')} prompt tokens, "
                          f"{output_tokens_override or '?'} output tokens")
        else:
            console.print(f"  Loaded {len(prompts)} unique prompts")
    else:
        prompts = [prompt or "Write a short poem about recursion."]

    effective_max_tokens = max_tokens or output_tokens_override or 256

    console.print(f"\n[bold]Benchmarking[/bold] {model} @ {base_url}")
    console.print(
        f"  requests={num_requests}  concurrency={concurrency}  "
        f"stream={stream}  max_tokens={effective_max_tokens}"
        + (f"  max_seconds={max_seconds}" if max_seconds else "")
    )
    console.print()

    result = asyncio.run(
        run_benchmark(
            base_url, api_key, model, prompts, effective_max_tokens,
            num_requests, concurrency, stream, max_seconds,
        )
    )

    summary = result.summary()
    console.print()
    print_results(summary)

    if result.failures:
        console.print(f"\n[red]{len(result.failures)} failed requests[/red]")
        for f in result.failures[:5]:
            console.print(f"  - {f.error}")

    if output:
        Path(output).write_text(json.dumps(summary, indent=2))
        console.print(f"\nResults saved to {output}")


if __name__ == "__main__":
    cli()
