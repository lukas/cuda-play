#!/usr/bin/env python3
"""
Benchmark TensorRT-LLM inference: latency and throughput.

Requires: TensorRT-LLM (use NGC container nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6)
Usage:
  MODEL=Qwen/Qwen3.5-0.8B python trtllm_bench.py
  python trtllm_bench.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""
from __future__ import annotations

import argparse
import os
import time

from tensorrt_llm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorRT-LLM latency & throughput benchmark")
    parser.add_argument("--model", default=None, help="Override MODEL env")
    parser.add_argument("--prompt", default="Explain quantum computing in one sentence.", help="Benchmark prompt")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max new tokens per request")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--throughput-requests", type=int, default=10, help="Requests for throughput test")
    args = parser.parse_args()

    model_id = args.model or os.environ.get("MODEL") or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Model: {model_id}")
    print(f"Prompt: {args.prompt[:60]}...")
    print(f"Max tokens: {args.max_tokens}")
    print()

    print("Loading model...")
    llm = LLM(model=model_id)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    print("Ready.\n")

    # Warmup
    print(f"Warmup ({args.warmup} iters)...")
    for _ in range(args.warmup):
        list(llm.generate([args.prompt], SamplingParams(temperature=0.0, max_tokens=min(8, args.max_tokens))))
    print("Done.\n")

    # TTFT: time for first token (max_tokens=1)
    print("=== Time to first token (TTFT) ===")
    ttft_params = SamplingParams(temperature=0.0, max_tokens=1)
    ttft_times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        list(llm.generate([args.prompt], ttft_params))
        ttft_times.append(time.perf_counter() - t0)
    ttft_ms = [t * 1000 for t in ttft_times]
    print(f"  TTFT: {sum(ttft_ms)/len(ttft_ms):.1f} ms (avg over {args.iters} iters)")
    print()

    # Latency benchmark
    print(f"=== Latency ({args.iters} iters, max {args.max_tokens} tokens) ===")
    total_times = []
    token_counts = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        outputs = list(llm.generate([args.prompt], sampling_params))
        total_times.append(time.perf_counter() - t0)
        token_counts.append(len(outputs[0].outputs[0].token_ids))
    total_ms = [t * 1000 for t in total_times]
    avg_tokens = sum(token_counts) / len(token_counts)
    print(f"  End-to-end latency:  {sum(total_ms)/len(total_ms):.1f} ms (avg)")
    print(f"  Per-token latency:   {sum(total_ms)/len(total_ms) / avg_tokens:.1f} ms/tok")
    print(f"  Output tokens:       {avg_tokens:.1f} (avg)")
    print()

    # Throughput
    print(f"=== Throughput ({args.throughput_requests} sequential requests) ===")
    t0 = time.perf_counter()
    total_tokens = 0
    for _ in range(args.throughput_requests):
        outputs = list(llm.generate([args.prompt], sampling_params))
        total_tokens += len(outputs[0].outputs[0].token_ids)
    elapsed = time.perf_counter() - t0
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    requests_per_sec = args.throughput_requests / elapsed if elapsed > 0 else 0
    print(f"  Total time:    {elapsed:.2f} s")
    print(f"  Total tokens:  {total_tokens}")
    print(f"  Throughput:    {tokens_per_sec:.1f} tokens/sec")
    print(f"  Requests/sec:  {requests_per_sec:.2f}")
    print()

    print("=== Summary ===")
    print(f"  TTFT (avg):     {sum(ttft_ms)/len(ttft_ms):.1f} ms")
    print(f"  E2E (avg):      {sum(total_ms)/len(total_ms):.1f} ms")
    print(f"  Throughput:     {tokens_per_sec:.1f} tok/s")


if __name__ == "__main__":
    main()
