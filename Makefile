CARGO ?= cargo
FEATURES ?= metal

.PHONY: all build run test bench lint fmt clean gpu-check cuda-check colab-sync colab-clear

all: build

build:
	$(CARGO) build --release --features $(FEATURES)

run:
	$(CARGO) run --release --features $(FEATURES)

test:
	$(CARGO) test --features $(FEATURES)

bench:
	$(CARGO) run --example bench --features $(FEATURES)

lint:
	$(CARGO) clippy --features $(FEATURES) --all-targets -- -D warnings
	$(CARGO) fmt --check

fmt:
	$(CARGO) fmt

setup-hooks:
	git config core.hooksPath .githooks
	chmod +x .githooks/*
	@echo "Git pre-commit hooks installed successfully!"

clean:
	$(CARGO) clean


# Explicit pretrained Metal inference; requires local Qwen3-0.6B weights.
gpu-check:
	RLSE_GPU_DEVICE=metal $(CARGO) test --locked --features metal --test test_gpu_inference --test test_pretrained_numerics -- --ignored --nocapture

# Run inside a CUDA host (e.g. Colab), with local Qwen3 pretrained weights.
cuda-check:
	RLSE_GPU_DEVICE=cuda $(CARGO) test --locked --features cuda --test test_gpu_inference --test test_pretrained_quality --test test_pretrained_numerics -- --ignored --nocapture

# Pack current source tree and embed into test_colab.ipynb for zero-popup VS Code -> Colab execution.
colab-sync:
	@sync_dir=$$(mktemp -d /tmp/rlse-colab-sync.XXXXXX) && \
		python3 scripts/gpu_validation.py pack --output "$$sync_dir/rlse-source.tar.gz" --embed-notebook

# Clear embedded archive payload from test_colab.ipynb before committing.
colab-clear:
	python3 scripts/gpu_validation.py clear-notebook
