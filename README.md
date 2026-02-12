# cuda-play

CUDA experiments and benchmarks. Includes a raw CUDA vadd kernel and a Triton vadd benchmark for comparison.

## Setup

1. Create a `cuda-dev` pod on a GPU cluster (see `cuda-dev.yaml`, `pvc.yaml`).
2. Set `KUBECONFIG` to your cluster config (not checked in for security).
3. `make sync` to push code to the pod.

## Targets

| Target | Description |
|--------|-------------|
| `make sync` | Sync local files to the pod |
| `make build` | Compile all `.cu` files |
| `make run`   | Build and run CUDA binaries |
| `make shell` | Exec into the pod |
| `make triton-run` | Run Triton vadd benchmark |

## Config

- `POD` – pod name (default: `cuda-dev`)
- `ARCH` – GPU architecture for nvcc (default: `sm_90` for H100/H200)
