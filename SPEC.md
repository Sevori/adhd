# ICE Specification

This is the regenerable artifact. The source code is downstream from this document. If this spec is complete and correct, any sufficiently capable agent can produce a conforming implementation. If the implementation diverges from this spec, the implementation is wrong.

Reference: [Bootstrapping Coding Agents](https://www.monperrus.net/martin/coding-agent-bootstrap) (Monperrus, 2026) — "The specification is the program."

## What ICE Is

ICE (Infinite Context Engine) is a **shared continuity kernel** that gives AI agents persistent memory across sessions, handoffs, and crashes. It stores events, derives continuity items (facts, decisions, constraints, scars, lessons, hypotheses), compiles them into bounded context packs, and serves them via MCP or HTTP.

The core problem: agents lose everything when a session ends. ICE makes them remember.

## Architecture

```
Agent (Claude Code, etc.)
  |
  v
MCP stdio / HTTP API
  |
  v
Engine (thread-safe wrapper)
  |
  v
SharedContinuityKernel
  |
  ├── Event ingestion (write_events)
  ├── Continuity derivation (write_derivations)
  ├── Context compilation (read_context, handoff)
  ├── Recall (vector + continuity-priority scoring)
  ├── Snapshots (point-in-time captures)
  └── Organism state (machine-level coordination)
  |
  v
SQLite (WAL mode) + zstd compressed log
```

## Data Model

### Events
Raw observations from agents. Kind: `Observation`, `Action`, `Request`, `Response`, `Reflection`. Stored in `events` table. Each event produces memories in Hot and Episodic layers.

### Continuity Items
Derived from events or written directly. The durable knowledge graph.

| Kind | Half-life (hrs) | Floor salience | Recall boost | Purpose |
|------|----------------|----------------|--------------|---------|
| WorkingState | 18 | 0.02 | — | Ephemeral agent state |
| WorkClaim | 8 | 0.02 | — | Coordination locks |
| Signal | 12 | 0.01 | +0.22 | Cross-agent coordination |
| Hypothesis | 36 | 0.03 | — | Unvalidated ideas |
| Outcome | 48 | 0.04 | — | Results of actions |
| Derivation | 72 | 0.05 | — | Inferred knowledge |
| Summary | 96 | 0.08 | — | Compressed history |
| Fact | 144 | 0.06 | — | Verified truths |
| Lesson | 192 | 0.10 | +0.20 | Learned patterns |
| Decision | 240 | 0.14 | +0.34 | Choices with rationale |
| Constraint | 336 | 0.22 | +0.31 | Rules and limitations |
| Incident | 432 | 0.20 | +0.28 | Failures and incidents |
| OperationalScar | 720 | 0.36 | +0.42 | Trauma from past failures |

Decay: `effective_salience = salience * 0.5^(age / half_life)`, clamped to `[floor, 1.0]`. Resolved items decay 3.3x faster with 5x lower floor.

### Memory Layers
`Hot` (rank 2) > `Episodic` (rank 3) > `Semantic` (rank 4) > `Summary` (rank 5) > `Cold` (rank 1). Higher-rank layers win deduplication.

### Retention States
`Open` > `Active` > `Resolved` > `Superseded` > `Rejected`. Open and Active items get full half-life. Others get compressed decay.

## MCP Tools

The MCP interface exposes these tools via stdio:

| Tool | Purpose |
|------|---------|
| `continuity_bootstrap` | Attach agent + open context + read pack in one call |
| `continuity_identify_machine` | Return canonical machine identity |
| `continuity_read_context` | Read current context pack with constraints, scars, decisions |
| `continuity_write_event` | Ingest a raw observation |
| `continuity_write_item` | Write a continuity item directly |
| `continuity_recall` | Vector + continuity-priority recall |
| `continuity_explain` | Explain compiler state and chunk composition |
| `continuity_handoff` | Generate handoff proof for agent transitions |
| `continuity_snapshot` | Point-in-time capture |
| `continuity_resume` | Resume from snapshot |
| `continuity_replay` | Replay event history |
| `continuity_claim_work` | Coordination: claim a work scope |
| `continuity_upsert_agent_badge` | Register agent capabilities |
| `continuity_publish_coordination_signal` | Cross-agent signals |
| `continuity_publish_signal` | Publish signal to subscribers |

## Benchmark Correctness

The benchmark suite defines correctness. A conforming implementation MUST pass all benchmark scenarios.

### Key Metrics (what "correct" means)
| Metric | Abbrev | Target | Meaning |
|--------|--------|--------|---------|
| Critical Fact Survival Rate | CFSR | > 0.80 | Facts survive handoffs |
| Constraint Survival Rate | CSR | > 0.80 | Constraints persist |
| Decision Lineage Fidelity | DLF | > 0.60 | Decisions retain provenance |
| Operational Scar Retention | OSR | > 0.80 | Scars are not forgotten |
| Resume Accuracy Score | RAS | > 0.70 | Overall handoff quality |
| Memory Pollution Rate | MPR | < 0.20 | Low noise in output |
| Mistake Recurrence Rate | MRR | = 0.00 | Past mistakes not repeated |

### Benchmark Classes
`AgentSwapSurvival`, `StrongToSmallContinuation`, `SmallToSmallRelay`, `InterruptionStress`, `OperationalScar`, `CrossAgentCollaborative`, `CrashRecovery`, `MemoryPollution`, `ContextBudgetCompression`, `BaselineIsolation`.

### Survival Analysis (Phase 1 metacognitive)
Every benchmark run produces a `SurvivalReport` classifying each ground truth item as SURVIVED or LOST with extracted features (keyword density, file path presence, framing analysis). This data feeds future phases of self-improving extraction.

## CLI

```
ice --root <path>     # Storage root (default: .ice)
ice ingest            # Ingest events from stdin
ice query             # Query memories
ice mcp               # Start MCP stdio server
ice serve             # Start HTTP server (default :4040)
ice bench             # Run benchmark suite
ice metrics           # Show engine metrics
ice handoff           # Generate handoff proof
ice explain           # Explain context state
```

## Validation

```bash
make build    # Must compile with zero warnings
make test     # All tests must pass
make lint     # cargo fmt + clippy -D warnings
make check    # Fast type-check
```

## What This Spec Does NOT Cover

- Deployment topology (that's ops, not spec)
- Specific model choices (the adapter layer abstracts this)
- UI/UX (ICE is infrastructure)
- Pricing or licensing (see Cargo.toml for license)

## Bootstrap Property

A conforming implementation of ICE, acting as a coding agent via its own MCP interface, should be able to modify its own source code and still pass its own benchmark suite. If this property holds, the spec is the fixed point — not the code.

This is the meta-circular test: `ICE(spec) -> implementation -> ICE(spec) -> implementation'`, where `implementation'` is observationally equivalent (passes same benchmarks).
