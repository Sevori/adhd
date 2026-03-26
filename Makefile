.PHONY: build run test mcp serve clean check lint

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

lint:
	cargo clippy -- -D warnings

clean:
	cargo clean
