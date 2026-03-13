#!/usr/bin/env python3
"""
Run Llama 3.2 1B Instruct inference with NVTX markers for Nsight profiling.

Requires: transformers, accelerate, torch
Usage:
  python llama_inference.py [--prompt "Hello"] [--max-new-tokens 32]
  HF_TOKEN=xxx python llama_inference.py   # for gated meta-llama model
"""
from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
	parser = argparse.ArgumentParser(description="Llama 3.2 1B inference with NVTX for profiling")
	parser.add_argument("--prompt", default="Explain quantum computing in one sentence.", help="Input prompt")
	parser.add_argument("--max-new-tokens", type=int, default=32, help="Max tokens to generate")
	parser.add_argument(
		"--model",
		default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
		help="HuggingFace model ID (use meta-llama/Llama-3.2-1B-Instruct with HF_TOKEN for Llama)",
	)
	parser.add_argument("--warmup", type=int, default=2, help="Warmup forward passes before generation")
	args = parser.parse_args()

	# NVTX: mark model loading
	if torch.cuda.is_available():
		torch.cuda.nvtx.range_push("load_model")

	from transformers import AutoModelForCausalLM, AutoTokenizer

	hf_token = os.environ.get("HF_TOKEN") or None  # None if unset/empty (avoids "Bearer " header)
	tokenizer = AutoTokenizer.from_pretrained(
		args.model,
		token=hf_token,
		trust_remote_code=True,
	)
	model = AutoModelForCausalLM.from_pretrained(
		args.model,
		torch_dtype=torch.bfloat16,
		device_map="cuda",
		token=hf_token,
		trust_remote_code=True,
	)
	model.eval()

	if torch.cuda.is_available():
		torch.cuda.nvtx.range_pop()

	# Prepare input
	messages = [{"role": "user", "content": args.prompt}]
	text = tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True,
	)
	input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

	# Warmup (NVTX-marked)
	if torch.cuda.is_available():
		torch.cuda.nvtx.range_push("warmup")
	for _ in range(args.warmup):
		with torch.no_grad():
			_ = model.generate(input_ids, max_new_tokens=4, do_sample=False)
	if torch.cuda.is_available():
		torch.cuda.synchronize()
	if torch.cuda.is_available():
		torch.cuda.nvtx.range_pop()

	# Generation (NVTX-marked for nsys timeline)
	if torch.cuda.is_available():
		torch.cuda.nvtx.range_push("generate")
	with torch.no_grad():
		output_ids = model.generate(
			input_ids,
			max_new_tokens=args.max_new_tokens,
			do_sample=False,
			pad_token_id=tokenizer.eos_token_id,
		)
	if torch.cuda.is_available():
		torch.cuda.synchronize()
		torch.cuda.nvtx.range_pop()

	response = tokenizer.decode(output_ids[0][input_ids.shape[1] :], skip_special_tokens=True)
	print("Response:", response)


if __name__ == "__main__":
	main()
