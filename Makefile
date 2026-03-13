# Load .env if present (HF_TOKEN, MODEL, etc.)
-include .env

# ---- Config ----
POD ?= cuda-dev
REMOTE_DIR ?= /workspace/work
ARCH ?= sm_90
NVCC_FLAGS = -O3 -arch=$(ARCH) -lineinfo -lnvToolsExt

# Profiling
PROFILE_TARGET ?= vadd
PROFILE_OUT ?= $(PROFILE_TARGET)_nsys
NSYS_CMD ?= /opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64/nsys
NSYS_FLAGS ?= -t cuda,nvtx --force-overwrite=true -o

# NCU (Nsight Compute) - kernel-level profiling
NCU_CMD ?= ncu
NCU_OUT ?= $(PROFILE_TARGET)_ncu
NCU_FLAGS ?= -k vadd -c 4 --set basic --nvtx --print-nvtx-rename kernel -f -o

# If you keep running into "localhost:8080", pin kubeconfig here:
KUBECONFIG ?= $(HOME)/cuda-play/CWKubeconfig_new-cluster
export KUBECONFIG

# macOS tar noise suppression + exclude local venv and secrets
TARFLAGS = --no-xattrs --no-mac-metadata --exclude=.venv --exclude=venv --exclude=.env

# ---- Helpers ----
.PHONY: sync shell lsremote build run clean profile profile-pull profile-new ncu-setup ncu-profile ncu-pull ncu-new py-setup py-deps triton-run triton-mlp triton-mlp-sweep triton-clean llama-deps llama-run llama-nsys llama-ncu llama-web llama-web-forward

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

# Profile with nsys (e.g. make profile PROFILE_TARGET=vadd)
profile: build
	@echo "Profiling $(PROFILE_TARGET) with $(NSYS_CMD)..."
	kubectl exec -it $(POD) -- bash -lc 'cd $(REMOTE_DIR) && $(NSYS_CMD) profile $(NSYS_FLAGS) $(PROFILE_OUT) ./$(PROFILE_TARGET)'
	@echo "Profile saved: $(PROFILE_OUT).nsys-rep (use profile-pull to copy locally)"

# Pull profile from pod to ./profiles/
profile-pull:
	@mkdir -p profiles
	kubectl cp $(POD):$(REMOTE_DIR)/$(PROFILE_OUT).nsys-rep profiles/$(PROFILE_OUT).nsys-rep
	@echo "Copied to profiles/$(PROFILE_OUT).nsys-rep"

# Profile + pull to new timestamped file
profile-new: build
	@ts=$$(date +%Y%m%d_%H%M%S); \
	echo "Profiling $(PROFILE_TARGET) -> $$ts.nsys-rep..."; \
	kubectl exec -it $(POD) -- bash -lc 'cd $(REMOTE_DIR) && $(NSYS_CMD) profile $(NSYS_FLAGS) '$$ts' ./$(PROFILE_TARGET)'; \
	mkdir -p profiles; \
	kubectl cp $(POD):$(REMOTE_DIR)/$$ts.nsys-rep profiles/$$ts.nsys-rep; \
	echo "Saved to profiles/$$ts.nsys-rep"

# Install Nsight Systems (for nsys profile)
nsys-setup:
	kubectl exec -it $(POD) -- bash -lc '\
		set -e; \
		if command -v nsys >/dev/null 2>&1; then echo "nsys OK"; else \
			echo "Installing nsight-systems..."; \
			apt-get update -qq && apt-get install -y -qq nsight-systems; \
		fi'

# Install NCU 2025.4+ (required for H200; 2024.1 fails with "Failed to prepare kernel")
ncu-setup:
	kubectl exec -it $(POD) -- bash -lc '\
		set -e; \
		if ncu --version 2>/dev/null | grep -q "2025"; then echo "ncu 2025 OK"; else \
			echo "Installing nsight-compute-2025.4.1..."; \
			apt-get update -qq && apt-get install -y -qq nsight-compute-2025.4.1; \
		fi'

# Profile with NCU (kernel-level metrics) - timestamped output
ncu-profile: build ncu-setup
	@ts=$$(date +%Y%m%d_%H%M%S); \
	echo "NCU profiling $(PROFILE_TARGET) -> $$ts.ncu-rep..."; \
	kubectl exec -it $(POD) -- bash -lc 'cd $(REMOTE_DIR) && NCU_PROFILE=1 $(NCU_CMD) $(NCU_FLAGS) '$$ts' ./$(PROFILE_TARGET)'; \
	mkdir -p profiles; \
	kubectl cp $(POD):$(REMOTE_DIR)/$$ts.ncu-rep profiles/$$ts.ncu-rep; \
	echo "Saved to profiles/$$ts.ncu-rep"

ncu-pull:
	@echo "Usage: kubectl cp $(POD):$(REMOTE_DIR)/<timestamp>.ncu-rep profiles/"
	@echo "  Or use: make ncu-profile (auto-pulls to timestamped file)"

ncu-new: ncu-profile

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
		uv sync --extra gpu'

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

# ---- Llama (SmolLM2 default; set HF_TOKEN and optionally MODEL in .env for Llama 3.2 1B) ----
# Install llama deps (transformers, accelerate)
llama-deps: sync py-setup
	kubectl exec -it $(POD) -- bash -lc '\
		set -e; \
		export PATH="$$HOME/.local/bin:$$PATH"; \
		cd $(REMOTE_DIR); \
		uv sync --extra gpu --extra llama'

# Login to Hugging Face (stores token in pod; add HF_TOKEN to .env then: make llama-login)
llama-login: llama-deps
	@set -a; [ -f $(CURDIR)/.env ] && . $(CURDIR)/.env; set +a; \
	if [ -z "$$HF_TOKEN" ]; then echo "Usage: add HF_TOKEN to .env, then make llama-login"; exit 1; fi; \
	kubectl exec -i $(POD) -- env HF_TOKEN="$$HF_TOKEN" bash -lc '\
		set -e; \
		export PATH="$$HOME/.local/bin:$$PATH"; \
		cd $(REMOTE_DIR); \
		uv run hf auth login --token "$$HF_TOKEN"'

# Run inference (default: SmolLM2; set HF_TOKEN and MODEL in .env for Llama 3.2 1B)
llama-run: llama-deps
	@set -a; [ -f $(CURDIR)/.env ] && . $(CURDIR)/.env; set +a; \
	kubectl exec -i $(POD) -- env HF_TOKEN="$$HF_TOKEN" bash -lc '\
		set -e; \
		export PATH="$$HOME/.local/bin:$$PATH"; \
		export PYTHONUNBUFFERED=1; \
		model_opt=""; [ -n "$(MODEL)" ] && model_opt="--model $(MODEL)"; \
		cd $(REMOTE_DIR); \
		uv run python llama_inference.py '"$$model_opt"''

# Profile Llama with Nsight Systems (timeline: load, warmup, generate)
llama-nsys: llama-deps nsys-setup
	@set -a; [ -f $(CURDIR)/.env ] && . $(CURDIR)/.env; set +a; \
	ts=$$(date +%Y%m%d_%H%M%S); \
	echo "Nsight Systems profiling Llama -> $$ts.nsys-rep..."; \
	model_opt=""; [ -n "$(MODEL)" ] && model_opt="--model $(MODEL)"; \
	kubectl exec -i $(POD) -- env HF_TOKEN="$$HF_TOKEN" bash -lc '\
		set -e; \
		export PATH="$$HOME/.local/bin:$$PATH"; \
		export PYTHONUNBUFFERED=1; \
		cd $(REMOTE_DIR); \
		nsys profile -t cuda,nvtx --force-overwrite=true -o '$$ts' \
			uv run python llama_inference.py --max-new-tokens 16 '"$$model_opt"''; \
	mkdir -p profiles; \
	kubectl cp $(POD):$(REMOTE_DIR)/$$ts.nsys-rep profiles/llama_$$ts.nsys-rep 2>/dev/null || true; \
	echo "Saved to profiles/llama_$$ts.nsys-rep (or check pod for $$ts.nsys-rep)"

# Profile Llama with Nsight Compute (kernel-level; samples first ~20 kernels)
llama-ncu: llama-deps ncu-setup
	@set -a; [ -f $(CURDIR)/.env ] && . $(CURDIR)/.env; set +a; \
	ts=$$(date +%Y%m%d_%H%M%S); \
	echo "Nsight Compute profiling Llama -> $$ts.ncu-rep..."; \
	model_opt=""; [ -n "$(MODEL)" ] && model_opt="--model $(MODEL)"; \
	kubectl exec -i $(POD) -- env HF_TOKEN="$$HF_TOKEN" bash -lc '\
		set -e; \
		export PATH="$$HOME/.local/bin:$$PATH"; \
		export PYTHONUNBUFFERED=1; \
		cd $(REMOTE_DIR); \
		ncu --set basic --nvtx --print-nvtx-rename kernel -c 20 -f -o '$$ts' \
			uv run python llama_inference.py --max-new-tokens 8 --warmup 0 '"$$model_opt"''; \
	mkdir -p profiles; \
	kubectl cp $(POD):$(REMOTE_DIR)/$$ts.ncu-rep profiles/llama_$$ts.ncu-rep 2>/dev/null || true; \
	echo "Saved to profiles/llama_$$ts.ncu-rep (or check pod for $$ts.ncu-rep)"

# Web UI for LLM (Gradio on port 7861). In another terminal: make llama-web-forward
llama-web: llama-deps
	@set -a; [ -f $(CURDIR)/.env ] && . $(CURDIR)/.env; set +a; \
	echo "Starting Gradio on port 7861..."; \
	echo "In another terminal run: make llama-web-forward"; \
	echo "Then open http://localhost:7861"; \
	kubectl exec -i $(POD) -- env HF_TOKEN="$$HF_TOKEN" MODEL="$(MODEL)" GRADIO_SERVER_PORT=7861 bash -lc '\
		set -e; \
		export PATH="$$HOME/.local/bin:$$PATH"; \
		cd $(REMOTE_DIR); \
		uv run python llama_web.py'

# Port-forward to access llama-web (run in separate terminal)
llama-web-forward:
	@echo "Forwarding localhost:7861 -> $(POD):7861"; \
	echo "Open http://localhost:7861"; \
	kubectl port-forward $(POD) 7861:7861

