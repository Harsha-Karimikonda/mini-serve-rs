CARGO ?= cargo
FEATURES ?= metal

.PHONY: all build run test bench lint fmt clean gpu-check

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
	$(CARGO) test --locked --features metal --test test_gpu_inference -- --ignored --nocapture
