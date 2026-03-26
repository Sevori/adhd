.PHONY: build run test mcp serve clean check lint fmt

build:
	cargo build

run: build
	cargo run

test:
	cargo test

mcp: build
	cargo run -- mcp

serve: build
	cargo run -- serve

check:
	cargo check

fmt:
	cargo fmt --all

lint: fmt
	cargo clippy -- -D warnings

clean:
	cargo clean
