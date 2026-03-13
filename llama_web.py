#!/usr/bin/env python3
"""
Gradio web UI for LLM inference. Uses MODEL and HF_TOKEN from environment.
"""
from __future__ import annotations

import os

import gradio as gr
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


def generate(
	prompt: str,
	max_new_tokens: int,
	model,
	tokenizer,
) -> str:
	if not prompt.strip():
		return "Enter a prompt."
	messages = [{"role": "user", "content": prompt}]
	text = tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True,
	)
	input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
	with torch.no_grad():
		output_ids = model.generate(
			input_ids,
			max_new_tokens=max_new_tokens,
			do_sample=False,
			pad_token_id=tokenizer.eos_token_id,
		)
	return tokenizer.decode(output_ids[0][input_ids.shape[1] :], skip_special_tokens=True)


def main() -> None:
	model_id = os.environ.get("MODEL") or "HuggingFaceTB/SmolLM2-1.7B-Instruct"
	print(f"Loading {model_id}...")
	model, tokenizer = load_model(model_id)
	print("Ready.")

	with gr.Blocks(title="LLM Playground") as demo:
		gr.Markdown(f"# {model_id}")
		with gr.Row():
			prompt = gr.Textbox(
				label="Prompt",
				placeholder="Ask me anything...",
				lines=3,
			)
		with gr.Row():
			max_tokens = gr.Slider(8, 256, value=64, step=8, label="Max new tokens")
			btn = gr.Button("Generate")
		output = gr.Textbox(label="Response", lines=6, interactive=False)

		def run(p, m):
			return generate(p, int(m), model, tokenizer)

		btn.click(fn=run, inputs=[prompt, max_tokens], outputs=output)
		prompt.submit(fn=run, inputs=[prompt, max_tokens], outputs=output)

	port = int(os.environ.get("GRADIO_SERVER_PORT", "7861"))
	demo.launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
	main()
