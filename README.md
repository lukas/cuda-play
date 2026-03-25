# cuda-play

A collection of experiments around speeding up LLM inference and benchmarking GPU performance. The repo spans low-level CUDA/Triton kernel work, end-to-end inference benchmarking, and automated LLM serving optimization.

## Repository layout

### `autollm/`

Standalone project (has its own git history) for **AI-driven LLM serving optimization on Kubernetes**. Deploys vLLM or SGLang serving pods, benchmarks them with guidellm, then uses an AI agent (Claude/GPT) to iteratively tune the serving config. The main workflow is `make sweep` which runs a baseline benchmark followed by N improvement iterations. Supports multiple model families (Qwen, Kimi-K2.5), tensorizer-based fast loading, speculative decoding (EAGLE-3), and remote in-cluster sweeps. See `autollm/README.md` and `autollm/docs/AGENT_HANDOFF.md` for details.

### `benchmark/`

Portable **CLI benchmarking tool for OpenAI-compatible inference APIs**. Measures latency (p50/p90/p99), throughput (requests/sec, tokens/sec), and time-to-first-token via streaming. Supports concurrent requests, HuggingFace dataset loading, synthetic prompt generation, and preset configurations. Independent of Kubernetes — point it at any API endpoint. Entry point: `benchmark/bench.py`.

### `lllm/`

Placeholder for a new project (currently empty).

### Root-level files

Low-level GPU kernel experiments and inference scripts that run on a Kubernetes GPU pod:

| File | What it does |
|------|-------------|
| `vadd.cu` | CUDA vector-add kernel, benchmarks across block sizes, reports memory bandwidth |
| `triton_vadd_bench.py` | Triton vector-add benchmark |
| `triton_mlp.py` | Triton MLP kernel benchmark |
| `triton_mlp_sweep.py` | Parametric sweep of Triton MLP configs |
| `llama_inference.py` | HuggingFace Transformers inference (SmolLM2/Llama) |
| `llama_bench.py` | Benchmarking for Llama/SmolLM2 inference |
| `llama_web.py` | Gradio web UI for interactive LLM chat |
| `trtllm_inference.py` | TensorRT-LLM inference |
| `trtllm_bench.py` | TensorRT-LLM benchmarking |

## Setup

1. Create a `cuda-dev` pod on a GPU cluster (see `cuda-dev.yaml`).
2. Set `KUBECONFIG` to your cluster config (not checked in for security).
3. `make sync` to push code to the pod.

Python deps use [uv](https://docs.astral.sh/uv/). `make py-deps` installs uv in the container and syncs deps from `pyproject.toml`.

## Make targets

| Target | Description |
|--------|-------------|
| `make sync` | Sync local files to the pod |
| `make build` | Compile all `.cu` files |
| `make run`   | Build and run CUDA binaries |
| `make shell` | Exec into the pod |
| `make py-deps` | Install uv + sync Python deps |
| `make triton-run` | Run Triton vadd benchmark |
| `make triton-mlp` | Run Triton MLP benchmark |
| `make profile` | Profile with nsys |
| `make profile-pull` | Copy .nsys-rep to local `profiles/` |
| `make ncu-profile` | Profile with NCU (kernel-level metrics) |
| `make ncu-pull` | Copy .ncu-rep to local `profiles/` |

**Profiling:** `make profile` runs nsys; `make ncu-profile` runs Nsight Compute for per-kernel metrics. NCU 2025.4+ required for H200 — install with `apt-get install nsight-compute-2025.4.1` if missing.

## Config

- `POD` — pod name (default: `cuda-dev`)
- `ARCH` — GPU architecture for nvcc (default: `sm_90` for H100/H200)
