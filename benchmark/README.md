# benchmark

Benchmark OpenAI-compatible inference APIs — measures latency, throughput, and time-to-first-token.

Supports loading diverse prompts from HuggingFace datasets (like [GuideLLM](https://github.com/vllm-project/guidellm) does) and ships with presets for common providers.

## Setup

```bash
cd benchmark
uv venv && uv pip install -e .
```

## Presets

| Preset | Endpoint | Model | API Key Env |
|--------|----------|-------|-------------|
| `wandb-kimi` | `api.inference.wandb.ai` | `moonshotai/Kimi-K2.5` | `WANDB_API_KEY` |

### Benchmark W&B Kimi K2.5 with GuideLLM data

```bash
export WANDB_API_KEY=<your-key>

# Quick sanity check (10 sequential requests, HumanEval prompts)
python bench.py --preset wandb-kimi --data humaneval

# Throughput test (50 requests, 10 concurrent, CNN/DailyMail articles)
python bench.py --preset wandb-kimi --data cnn_dailymail -n 50 -c 10 -o results.json

# Custom HuggingFace dataset
python bench.py --preset wandb-kimi --data openai/openai_humaneval -n 20 -c 5
```

## General Usage

```bash
# Any OpenAI-compatible endpoint
python bench.py --base-url http://localhost:8000/v1 --model my-model -n 50 -c 10

# With a HuggingFace dataset
python bench.py --base-url https://api.openai.com/v1 --api-key $OPENAI_API_KEY \
  --model gpt-4o --data humaneval -n 20 -c 5

# Non-streaming
python bench.py --base-url http://localhost:8000/v1 --model my-model --no-stream

# Local prompt file (one per line)
python bench.py --base-url http://localhost:8000/v1 --model my-model --data prompts.txt
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--preset` | — | Pre-configured endpoint (`wandb-kimi`) |
| `--base-url` | *(required unless preset)* | API base URL |
| `--model` | *(required unless preset)* | Model name |
| `--api-key` | `$API_KEY` | API key (presets check their own env var) |
| `--data` | — | Prompt source: shortcut (`humaneval`, `cnn_dailymail`), HF dataset id, or local file |
| `--prompt` | *"Write a short poem…"* | Single prompt (ignored if `--data` set) |
| `--max-tokens` | 256 | Max tokens per request |
| `-n` | 10 | Total requests |
| `-c` | 1 | Concurrent requests |
| `--stream/--no-stream` | stream | Streaming mode (enables TTFT) |
| `-o` | — | Save JSON results to file |

## Built-in Dataset Shortcuts

| Shortcut | HuggingFace Dataset | Description |
|----------|-------------------|-------------|
| `humaneval` | `openai/openai_humaneval` | 164 code-generation prompts |
| `cnn_dailymail` | `abisee/cnn_dailymail` | News article summarization |

## Metrics

- **throughput_req_per_s** — requests completed per second
- **throughput_tok_per_s** — completion tokens generated per second
- **latency p50/p90/p99** — end-to-end request latency percentiles
- **ttft p50/p90** — time-to-first-token percentiles (streaming only)
