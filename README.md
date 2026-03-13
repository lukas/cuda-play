# cuda-play

CUDA experiments, GPU benchmarks, and AI-driven LLM optimization on Kubernetes.

## Subprojects

- **[autollm/](autollm/)** — AI-driven vLLM optimization. Deploys vLLM on K8s, benchmarks with guidellm, and uses an AI agent to iteratively tune configs. Supports local and remote (in-cluster) sweeps.
- **autollm/runllm/** — Per-model vLLM deployment configs (Qwen2.5-1.5B, Qwen3-235B, Kimi-K2.5). Each model has a K8s pod spec, Makefile, and smoke tests.

See each subproject's README for details.

## CUDA / Triton experiments

Low-level GPU programming experiments on a `cuda-dev` pod:

1. Create a `cuda-dev` pod on a GPU cluster (see `cuda-dev.yaml`).
2. Set `KUBECONFIG` to your cluster config (not checked in for security).
3. `make sync` to push code to the pod.

Python deps use [uv](https://docs.astral.sh/uv/). `make py-deps` installs uv in the container and syncs deps from `pyproject.toml`.

### Targets

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

### autollm shortcuts

| Target | Description |
|--------|-------------|
| `make runllm-apply` | Deploy vLLM pod |
| `make runllm-forward` | Port-forward localhost:8000 to vLLM |
| `make runllm-test` | Test vLLM completion |
| `make autollm-dashboard` | Start benchmark dashboard at http://localhost:8765/ |
| `make autollm-benchmark` | Run benchmark harness (use `DESCRIPTION="..."`) |

**Profiling:** `make profile` runs nsys; `make ncu-profile` runs Nsight Compute for per-kernel metrics. NCU 2025.4+ required for H200 — install with `apt-get install nsight-compute-2025.4.1` if missing.

### Config

- `POD` — pod name (default: `cuda-dev`)
- `ARCH` — GPU architecture for nvcc (default: `sm_90` for H100/H200)
