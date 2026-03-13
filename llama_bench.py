#!/usr/bin/env python3
"""
Benchmark LLM inference: latency (TTFT, time/token) and throughput (tokens/sec).
Uses MODEL and HF_TOKEN from environment.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str):
	hf_token = os.environ.get("HF_TOKEN") or None
	tokenizer = AutoTokenizer.from_pretrained(
		model_id,
		token=hf_token,
		trust_remote_code=True,
	)
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		torch_dtype=torch.bfloat16,
		device_map="cuda",
		token=hf_token,
		trust_remote_code=True,
	)
	model.eval()
	return model, tokenizer


def run_inference(
	model,
	tokenizer,
	prompt: str,
	max_new_tokens: int,
	do_sample: bool = False,
) -> tuple[str, float, float, int]:
	"""Returns (response, ttft_sec, total_sec, num_output_tokens)."""
	messages = [{"role": "user", "content": prompt}]
	text = tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True,
	)
	input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

	torch.cuda.synchronize()
	t0 = time.perf_counter()

	with torch.no_grad():
		output_ids = model.generate(
			input_ids,
			max_new_tokens=max_new_tokens,
			do_sample=do_sample,
			pad_token_id=tokenizer.eos_token_id,
		)

	torch.cuda.synchronize()
	total_sec = time.perf_counter() - t0

	num_output_tokens = output_ids.shape[1] - input_ids.shape[1]
	response = tokenizer.decode(output_ids[0][input_ids.shape[1] :], skip_special_tokens=True)

	# TTFT: without streaming we can't measure directly; return 0 (measured separately via max_tokens=1)
	ttft_sec = 0.0

	return response, ttft_sec, total_sec, num_output_tokens


def main() -> None:
	parser = argparse.ArgumentParser(description="LLM latency & throughput benchmark")
	parser.add_argument("--model", default=None, help="Override MODEL env (default: from .env)")
	parser.add_argument("--prompt", default="Explain quantum computing in one sentence.", help="Benchmark prompt")
	parser.add_argument("--max-tokens", type=int, default=64, help="Max new tokens per request")
	parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
	parser.add_argument("--iters", type=int, default=20, help="Benchmark iterations")
	parser.add_argument("--throughput-requests", type=int, default=10, help="Requests for throughput test")
	args = parser.parse_args()

	model_id = args.model or os.environ.get("MODEL") or "HuggingFaceTB/SmolLM2-1.7B-Instruct"
	print(f"Model: {model_id}")
	print(f"Prompt: {args.prompt[:60]}...")
	print(f"Max tokens: {args.max_tokens}")
	print()

	print("Loading model...")
	model, tokenizer = load_model(model_id)
	print("Ready.\n")

	# Warmup
	print(f"Warmup ({args.warmup} iters)...")
	for _ in range(args.warmup):
		run_inference(model, tokenizer, args.prompt, min(8, args.max_tokens))
	torch.cuda.synchronize()
	print("Done.\n")

	# TTFT: time for first token (max_new_tokens=1)
	print("=== Time to first token (TTFT) ===")
	ttft_times = []
	for _ in range(args.iters):
		_, _, total, _ = run_inference(model, tokenizer, args.prompt, max_new_tokens=1)
		ttft_times.append(total)
	ttft_ms = [t * 1000 for t in ttft_times]
	print(f"  TTFT: {sum(ttft_ms)/len(ttft_ms):.1f} ms (avg over {args.iters} iters)")
	print()

	# Latency benchmark (full generation)
	print(f"=== Latency ({args.iters} iters, max {args.max_tokens} tokens) ===")
	total_times = []
	token_counts = []

	for i in range(args.iters):
		_, _, total, n = run_inference(model, tokenizer, args.prompt, args.max_tokens)
		total_times.append(total)
		token_counts.append(n)

	total_ms = [t * 1000 for t in total_times]
	avg_tokens = sum(token_counts) / len(token_counts)

	print(f"  End-to-end latency:  {sum(total_ms)/len(total_ms):.1f} ms (avg)")
	print(f"  Per-token latency:   {sum(total_ms)/len(total_ms) / avg_tokens:.1f} ms/tok")
	print(f"  Output tokens:       {avg_tokens:.1f} (avg)")
	print(f"  End-to-end latency:         {sum(total_ms)/len(total_ms):.1f} ms (avg)")
	print(f"  Per-token latency:          {sum(total_ms)/len(total_ms) / (sum(token_counts)/len(token_counts)):.1f} ms/tok")
	print(f"  Output tokens:              {sum(token_counts)/len(token_counts):.1f} (avg)")
	print()

	# Throughput: sequential requests, total tokens / total time
	print(f"=== Throughput ({args.throughput_requests} sequential requests) ===")
	t0 = time.perf_counter()
	total_tokens = 0
	for _ in range(args.throughput_requests):
		_, _, _, n = run_inference(model, tokenizer, args.prompt, args.max_tokens)
		total_tokens += n
	torch.cuda.synchronize()
	elapsed = time.perf_counter() - t0

	tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
	requests_per_sec = args.throughput_requests / elapsed if elapsed > 0 else 0

	print(f"  Total time:    {elapsed:.2f} s")
	print(f"  Total tokens:  {total_tokens}")
	print(f"  Throughput:    {tokens_per_sec:.1f} tokens/sec")
	print(f"  Requests/sec:  {requests_per_sec:.2f}")
	print()

	# Summary
	print("=== Summary ===")
	print(f"  TTFT (avg):     {sum(ttft_ms)/len(ttft_ms):.1f} ms")
	print(f"  E2E (avg):      {sum(total_ms)/len(total_ms):.1f} ms")
	print(f"  Throughput:     {tokens_per_sec:.1f} tok/s")


if __name__ == "__main__":
	main()
