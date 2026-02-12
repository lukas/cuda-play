# ---- Config ----
POD ?= cuda-dev
REMOTE_DIR ?= /workspace/work
VENV_DIR ?= /workspace/venv
ARCH ?= sm_90
NVCC_FLAGS = -O3 -arch=$(ARCH) -lineinfo

# If you keep running into "localhost:8080", pin kubeconfig here:
KUBECONFIG ?= $(HOME)/cuda-play/CWKubeconfig_new-cluster
export KUBECONFIG

# macOS tar noise suppression
TARFLAGS = --no-xattrs --no-mac-metadata

# ---- Helpers ----
.PHONY: sync shell lsremote build run clean py-setup py-deps triton-run triton-clean

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

# Install python + venv tooling inside the container if missing
py-setup:
	kubectl exec -it $(POD) -- bash -lc '\
		set -e; \
		if command -v python3 >/dev/null 2>&1; then echo "python3 OK"; else \
			echo "Installing python3..."; \
			apt-get update && apt-get install -y python3 python3-pip; \
		fi; \
		if python3 -c "import venv" >/dev/null 2>&1; then echo "venv OK"; else \
			echo "Installing python3-venv..."; \
			apt-get update && apt-get install -y python3-venv; \
		fi'

# Create venv (idempotent) + install deps
py-deps: sync py-setup
	kubectl exec -it $(POD) -- bash -lc '\
		set -e; \
		if [ ! -d "$(VENV_DIR)" ]; then python3 -m venv $(VENV_DIR); fi; \
		. $(VENV_DIR)/bin/activate; \
		python -m pip install -U pip; \
		python -m pip install -U torch triton'

# Run your Triton benchmark
triton-run: py-deps
	kubectl exec -it $(POD) -- bash -lc '\
		set -e; \
		. $(VENV_DIR)/bin/activate; \
		cd $(REMOTE_DIR); \
		python triton_vadd_bench.py'

# Clean venv if you want a fresh reinstall
triton-clean:
	kubectl exec -it $(POD) -- bash -lc 'rm -rf $(VENV_DIR)'
