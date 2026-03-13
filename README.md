# cuda-play

CUDA experiments and GPU benchmarks on Kubernetes.

## Related projects

- **[autollm](https://github.com/lukas/autollm)** — AI-driven vLLM optimization on Kubernetes (separate repo).

## Setup

1. Create a `cuda-dev` pod on a GPU cluster (see `cuda-dev.yaml`).
2. Set `KUBECONFIG` to your cluster config (not checked in for security).
3. `make sync` to push code to the pod.

Python deps use [uv](https://docs.astral.sh/uv/). `make py-deps` installs uv in the container and syncs deps from `pyproject.toml`.

## Targets

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
