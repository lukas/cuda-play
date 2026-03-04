# ---- Config ----
POD ?= cuda-dev
REMOTE_DIR ?= /workspace/work
ARCH ?= sm_90
NVCC_FLAGS = -O3 -arch=$(ARCH) -lineinfo

# If you keep running into "localhost:8080", pin kubeconfig here:
KUBECONFIG ?= $(HOME)/cuda-play/CWKubeconfig_new-cluster
export KUBECONFIG

# macOS tar noise suppression + exclude local venv
TARFLAGS = --no-xattrs --no-mac-metadata --exclude=.venv --exclude=venv

# ---- Helpers ----
.PHONY: sync shell lsremote build run clean py-setup py-deps triton-run triton-mlp triton-mlp-sweep triton-clean

sync:
	@echo "Syncing $(CURDIR) -> $(POD):$(REMOTE_DIR)..."
	tar $(TARFLAGS) -czf - -C $(CURDIR) . | \
		kubectl exec -i $(POD) -- bash -lc 'mkdir -p $(REMOTE_DIR) && tar xzf - -C $(REMOTE_DIR)'
	@echo "Sync complete."

shell:
	kubectl exec -it $(POD) -- bash

lsremote:
	kubectl exec -it $(POD) -- bash -lc 'ls -la $(REMOTE_DIR)'

# Compile all .cu files (incl. vadd.cu)
build: sync
	@echo "Building in pod..."
	kubectl exec -it $(POD) -- bash -lc 'cd $(REMOTE_DIR) && for f in *.cu; do nvcc $(NVCC_FLAGS) "$$f" -o "$${f%.cu}"; done'

# Build and run all .cu binaries
run: build
	@echo "Running binaries..."
	kubectl exec -it $(POD) -- bash -lc 'cd $(REMOTE_DIR) && for f in *.cu; do b="$${f%.cu}"; echo "---- Running $$b ----"; ./$$b; done'

clean:
	kubectl exec -it $(POD) -- bash -lc 'cd $(REMOTE_DIR) && for f in *.cu; do rm -f "$${f%.cu}"; done'

# Install uv inside the container if missing
py-setup:
	kubectl exec -it $(POD) -- bash -lc '\
		set -e; \
		export PATH="$$HOME/.local/bin:$$PATH"; \
		if command -v uv >/dev/null 2>&1; then echo "uv OK"; else \
			echo "Installing uv..."; \
			apt-get update -qq && apt-get install -y -qq curl >/dev/null; \
			curl -LsSf https://astral.sh/uv/install.sh | sh; \
			export PATH="$$HOME/.local/bin:$$PATH"; \
			uv --version; \
		fi'

# Sync + install deps with uv
py-deps: sync py-setup
	kubectl exec -it $(POD) -- bash -lc '\
		set -e; \
		export PATH="$$HOME/.local/bin:$$PATH"; \
		cd $(REMOTE_DIR); \
		uv sync'

# Run your Triton benchmark
triton-run: py-deps
	kubectl exec -it $(POD) -- bash -lc '\
		set -e; \
		export PATH="$$HOME/.local/bin:$$PATH"; \
		cd $(REMOTE_DIR); \
		uv run triton_vadd_bench.py'

triton-mlp: py-deps
	kubectl exec -it $(POD) -- bash -lc '\
		export PATH="$$HOME/.local/bin:$$PATH"; \
		export PYTHONUNBUFFERED=1; \
		cd $(REMOTE_DIR); \
		uv run triton_mlp.py 2>&1'

triton-mlp-sweep: py-deps
	kubectl exec -it $(POD) -- bash -lc '\
		set -e; \
		export PATH="$$HOME/.local/bin:$$PATH"; \
		cd $(REMOTE_DIR); \
		uv run triton_mlp_sweep.py --M 4096 --din 1024 --dhid 2048 --dout 1024 --iters 80 --warmup 30'

# Clean .venv if you want a fresh reinstall
triton-clean:
	kubectl exec -it $(POD) -- bash -lc 'rm -rf $(REMOTE_DIR)/.venv'
