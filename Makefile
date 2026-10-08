CARGO ?= cargo

.PHONY: all build run test lint fmt clean

all: build

build:
	$(CARGO) build --release

run:
	$(CARGO) run --release

test:
	$(CARGO) test

lint:
	$(CARGO) clippy --all-targets -- -D warnings
	$(CARGO) fmt --check

fmt:
	$(CARGO) fmt

clean:
	$(CARGO) clean
