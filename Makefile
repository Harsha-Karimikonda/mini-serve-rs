CARGO ?= cargo
FEATURES ?= metal

.PHONY: all build run test lint fmt clean

all: build

build:
	$(CARGO) build --release --features $(FEATURES)

run:
	$(CARGO) run --release --features $(FEATURES)

test:
	$(CARGO) test --features $(FEATURES)

lint:
	$(CARGO) clippy --features $(FEATURES) --all-targets -- -D warnings
	$(CARGO) fmt --check

fmt:
	$(CARGO) fmt

clean:
	$(CARGO) clean

