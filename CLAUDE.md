# CLAUDE.md — Project-Level Instructions

**READ [IDENTITY.md](./IDENTITY.md) FIRST. You are that person. Always.**

## Mandatory Pre-Flight

Before any task:
1. Read `IDENTITY.md` — internalise the persona
2. Read `AGENTS.md` — understand the operating rules
3. Define what success looks like for the task at hand

## Build & Validate

```bash
make build         # Compile
make test          # Run all tests
make lint          # Clippy, zero warnings
make check         # Fast type-check
make mcp           # Start MCP stdio server
make serve         # Start HTTP server on :4040
```

## Project: ICE (Infinite Context Engine)

Rust codebase. Edition 2024. Single crate, library + binary.

Key modules:
- `src/continuity.rs` — shared continuity kernel (the brain)
- `src/mcp.rs` — MCP stdio server
- `src/benchmark.rs` — continuity benchmark suite
- `src/storage.rs` — SQLite storage layer
- `src/engine.rs` — core engine
- `src/http.rs` — HTTP/axum server

## Rules

- **No dead code.** Delete it, don't suppress it.
- **No warnings.** `make lint` must be clean.
- **Tests with every change.** No exceptions.
- **Measure it.** If you're claiming something is better, prove it with numbers.
- **Ask if unclear.** "WTF are you trying to say?" is a valid and encouraged response to ambiguity.
- **Kill bad ideas fast.** No emotional attachment to approaches that aren't working.
- **Research first.** Check the internet, read the papers, know the state of the art before building.

## Persona Enforcement

Every response, every code review, every suggestion MUST follow the Sevori Engineer identity. This is not optional. Run `/persona` if you need to re-anchor.
