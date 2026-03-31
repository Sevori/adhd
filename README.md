# Infinite Context Engine

ICE is a local, self-hostable continuity kernel for AI agents.

It persists events, derives typed continuity items, compiles bounded context packs, and exposes the shared state over MCP or HTTP so agent swaps, handoffs, and cold starts do not reset the work.

This repository is the executable engine. The protocol/spec work may live elsewhere, but this repo is the Rust implementation and benchmark harness.

## What This Repo Contains

- `SPEC.md` is the source of truth for behavior.
- `src/` contains the engine, storage, continuity kernel, MCP server, HTTP server, benchmarks, and client installers.
- `scripts/install.sh` installs the `ice` binary from GitHub releases or from source.
- `docs/` currently contains benchmark and tribunal artifacts, not a standalone protocol manual.

## Core Ideas

- Shared continuity survives across sessions, agents, and crashes.
- Context identity matters more than agent identity.
- ICE stores raw events and derived continuity items such as facts, decisions, constraints, work claims, lessons, incidents, and operational scars.
- Recall is bounded and scored so live operational state can beat stale lexical debris.
- The engine can coordinate multiple attached agents through work claims, signals, snapshots, and handoffs.

## Install

Install the latest release:

```bash
gh api repos/Sevori/adhd/contents/scripts/install.sh -H 'Accept: application/vnd.github.raw' > install-ice.sh
sh install-ice.sh --version latest
```

Useful variants:

```bash
sh install-ice.sh --dry-run --version latest
sh install-ice.sh --bin-dir "$HOME/.local/bin"
sh install-ice.sh --from-source --version latest
```

Requirements:

- `gh` must be installed.
- `gh auth login` must have access to `Sevori/adhd`.

## Build And Validate

```bash
make build
make check
make test
make lint
```

## Quick Start

Write a note into a shared root:

```bash
ice --root .ice ingest \
  --kind note \
  --agent me \
  --session study-1 \
  --namespace personal \
  --text "Renato prefers direct feedback and TDD."
```

Query it from a cold session:

```bash
ice --root .ice query \
  --agent someone-else \
  --session cold-session \
  --namespace personal \
  --text "What do you know about Renato?" \
  --budget-tokens 128
```

Run the MCP bridge:

```bash
ice --root .ice mcp
```

Run the HTTP API:

```bash
ice --root .ice serve --addr 127.0.0.1:4040
```

## Agent Client Installs

ICE can install managed continuity entries for several clients.

Claude Code:

```bash
ice claude install-global
ice claude status
ice claude uninstall
```

OpenHands:

```bash
ice openhands install-global
ice openhands status
ice openhands uninstall
```

OpenCode:

```bash
ice opencode install-global
ice opencode status
ice opencode uninstall
```

Goose:

```bash
ice goose install-global
ice goose status
ice goose uninstall
```

Codex:

```bash
ice codex install-global
```

Codex currently exposes only `install-global`.

## Benchmarks

ICE ships its own continuity benchmarks plus a `LongMemEval` runner.

Main entry points:

```bash
ice bench --help
ice bench-market --help
ice longmemeval run --help
ice longmemeval evaluate --help
make longmemeval-oracle
```

## Reading Order

If you need to understand the project instead of just running it:

1. `IDENTITY.md`
2. `SPEC.md`
3. `AGENTS.md`
4. `CLAUDE.md`

## Current Status

Current crate version: `0.4.0-public-beta`.

The engine is active and heavily test-driven. On this branch, `make check` and `make test` pass.
