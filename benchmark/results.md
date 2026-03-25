# Kimi K2.5 Inference Benchmark Results

Benchmark: autollm "large" — 200 synchronous requests, 256 prompt tokens, 128 output tokens.

## Summary

| Setup | Output tok/s | Latency p50 | TTFT p50 | Throughput (total tok/s) | Req/s |
|-------|-------------|-------------|----------|--------------------------|-------|
| **Self-hosted SGLang + EAGLE3 (best)** | **216** | ~596ms | ~0ms | 661 | 1.7 |
| Self-hosted SGLang + EAGLE3 + SpecV2 | 214 | ~601ms | ~0ms | 656 | 1.7 |
| Self-hosted SGLang + EAGLE3 | 201 | ~642ms | ~0ms | 614 | 1.6 |
| Self-hosted SGLang baseline | 118 | ~1086ms | ~0ms | 363 | 0.9 |
| **W&B Inference API** | **104.7** | **1231ms** | **180ms** | 104.5 | 0.82 |

## W&B Inference API Details

Run date: 2026-03-19

```
Endpoint:  https://api.inference.wandb.ai/v1
Model:     moonshotai/Kimi-K2.5
Requests:  200 (sequential, concurrency=1)
Prompt:    synthetic ~256 tokens
Output:    128 max tokens
Streaming: yes
```

```json
{
  "total": 200,
  "successes": 200,
  "failures": 0,
  "wall_time_s": 244.94,
  "total_prompt_tokens": 39800,
  "total_completion_tokens": 25600,
  "throughput_req_per_s": 0.82,
  "throughput_tok_per_s": 104.52,
  "output_tok_per_s_mean": 104.7,
  "output_tok_per_s_p50": 104.0,
  "output_tok_per_s_p90": 108.2,
  "latency_mean_s": 1.224,
  "latency_p50_s": 1.231,
  "latency_p90_s": 1.27,
  "latency_p99_s": 1.404,
  "ttft_mean_s": 0.203,
  "ttft_p50_s": 0.18,
  "ttft_p90_s": 0.237
}
```

## Self-hosted Results (from autollm sweeps)

### sweep-kimi-sglang-large (baseline → EAGLE3 + SpecV2)

| Run | Description | Output tok/s | Latency | Throughput |
|-----|-------------|-------------|---------|------------|
| 20260319_001215 | SpecV2 overlap scheduler | 214 | 601ms | 656 tok/s |
| 20260318_233957 | Deeper EAGLE3 chain | 206 | 623ms | 631 tok/s |
| 20260318_230949 | EAGLE3 speculative decoding | 201 | 642ms | 614 tok/s |
| 20260318_235525 | Deeper speculative steps | 194 | 665ms | 593 tok/s |
| 20260318_232408 | Wider draft tree | 191 | 673ms | 586 tok/s |
| baseline | SGLang default | 118 | 1086ms | 363 tok/s |

### sweep-kimi-sglang-eagle-large (EAGLE3 baseline → tuning)

| Run | Description | Output tok/s | Latency | Throughput |
|-----|-------------|-------------|---------|------------|
| 20260319_000932 | Increased draft steps | 216 | 596ms | 661 tok/s |
| 20260318_235419 | Radix cache disabled | 213 | 605ms | 651 tok/s |
| 20260318_232604 | Deeper speculative decoding | 211 | 610ms | 646 tok/s |
| 20260318_234024 | Reduced speculative depth | 201 | 642ms | 614 tok/s |
| baseline | EAGLE3 default | 117 | 1102ms | 358 tok/s |

## Notes

- Self-hosted runs use in-cluster localhost (TTFT ≈ 0ms). W&B Inference has network overhead (TTFT ≈ 180ms).
- Self-hosted "throughput" includes prompt + completion tokens; W&B throughput_tok_per_s is completion-only.
- W&B Inference output tok/s is very consistent: p50=104, p90=108, std dev is low.
- The optimized self-hosted setup (EAGLE3 speculative decoding) is ~2x faster on output tok/s than W&B Inference.
