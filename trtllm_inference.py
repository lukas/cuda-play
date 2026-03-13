#!/usr/bin/env python3
"""
TensorRT-LLM inference using the LLM API.

Requires: TensorRT-LLM (use NGC container nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6)
Usage:
  python trtllm_inference.py [--prompt "Hello"] [--max-new-tokens 32]
  MODEL=Qwen/Qwen3.5-0.8B python trtllm_inference.py

Note: Qwen3.5-0.8B may not be supported; use TinyLlama/TinyLlama-1.1B-Chat-v1.0
or nvidia/Qwen3-8B-FP8 for known-good models.
"""
from __future__ import annotations

import argparse
import os

from tensorrt_llm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorRT-LLM inference")
    parser.add_argument("--prompt", default="Explain quantum computing in one sentence.", help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Max tokens to generate")
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model ID or nvidia/* FP8 checkpoint (default: MODEL env or TinyLlama)",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    args = parser.parse_args()

    model_id = args.model or os.environ.get("MODEL") or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading model: {model_id}")
    llm = LLM(model=model_id)

    prompts = [args.prompt]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    print("Generating...")
    outputs = list(llm.generate(prompts, sampling_params))
    response = outputs[0].outputs[0].text
    print("Response:", response)


if __name__ == "__main__":
    main()
