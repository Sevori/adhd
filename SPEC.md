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

### Survival Analysis (metacognitive)

#### Phase 1: Survival Analytics
Every benchmark run produces a `SurvivalReport` classifying each ground truth item as SURVIVED or LOST with extracted features (keyword density, file path presence, framing analysis).

#### Phase 2: Candidate Hypothesis Generation

After a suite completes, offline meta-analysis generates candidate hypotheses about survival patterns:

1. Collect all `SurvivalRecord`s across classes and baselines
2. For each (category, feature) pair, run a chi-squared proportion test (2×2 contingency table)
3. Apply Benjamini-Hochberg FDR correction across all tests
4. If adjusted p < 0.05 AND sample size ≥ 10, emit a candidate hypothesis
5. Write as `ContinuityKind::Hypothesis` with `Scope::Project` (not Lesson, not Global — unvalidated)

Output: `MetaLessonReport` serialized to `meta-lessons.json` alongside the suite report.

**What this does NOT do (yet):** These hypotheses are correlational observations, not proven improvements. Promotion to `Lesson` or `Constraint` requires Phase 3 (closed-loop A/B validation showing the hypothesis improves downstream continuity metrics on a holdout set).

**Statistical limitations:**
- Chi-squared with Yates correction is unreliable for sparse cells (expected count < 5). Hypotheses with `sparse_cells: true` need Fisher's exact test (not yet implemented).
- With few benchmark classes, the sample of independent observations is small regardless of item count.

**Kill criterion:** If after 5 closed-loop cycles no survival metric improves by > 5% absolute, the patterns are noise.

**Prior art:** Reflexion (Shinn et al., 2023) reflects on task failures; Meta-Policy Reflexion (arXiv 2509.03990) extracts corrective action rules; MemMA (arXiv 2603.18718) does backward error propagation through memory. None analyse memory forgetting patterns to mutate extraction behaviour — but none claim to until the loop is closed.

#### Phase 3: Closed-Loop Hypothesis Injection

Phase 3 closes the metacognitive loop by making the extraction prompt dynamic. Survival hypotheses from Phase 2 are injected into the prompt as extraction guidance, and their impact on survival rates is measured via A/B comparison.

**Mechanism:**

1. Before building the extraction prompt, load prior `MetaLessonReport` from the output directory. Extract eligible hypotheses (non-sparse, adjusted p < 0.05).
2. Convert qualifying hypotheses into `SurvivalHypothesis` directives: each carries a feature name, category, direction (survived_more / lost_more), and a natural-language hint for prompt injection.
3. `render_structured_resume_prompt` accepts an optional `&[SurvivalHypothesis]` slice. When non-empty, a `[SURVIVAL HINTS]` section is appended to the prompt instructing the model to bias extraction toward features correlated with survival (e.g., "Include file paths in critical facts — items with file paths survive at 90% vs 10% without").
4. During benchmark runs, each class runs two variants of the `SharedContinuity` baseline on the same envelope: one **without** hypothesis injection (control) and one **with** injection (treatment).
5. The `BenchmarkClassReport` gains an optional `hypothesis_injection` field containing the `Phase3ClassResult` with control and treatment survival reports plus per-metric deltas.
6. A/B delta is computed per metric (CFSR, CSR, OSR, RAS) as `treatment - control`. Positive delta = hypothesis helped.

**Data types:**
- `SurvivalHypothesis`: extracted from `MetaLesson`, contains `feature_name`, `category`, `direction`, and a human-readable `hint` for prompt injection.
- `Phase3ValidationReport`: contains per-hypothesis treatment vs control survival rates, absolute improvement, and promotion/rejection decisions.
- `Phase3ClassResult`: per-class A/B result with control and treatment survival, hypotheses, and metric deltas.
- `HypothesisValidationResult`: per-hypothesis validation comparing treatment vs control across classes.

**Promotion criteria:**
- If treatment survival rate > control by >= 5% absolute across >= 3 benchmark classes, and positive in > 50% of classes, the hypothesis is promoted from `Hypothesis` to `Lesson` (status: `Active`).
- If treatment <= control across >= 3 classes, the hypothesis is rejected (status: `Rejected`).
- Otherwise the hypothesis remains `Open` for further cycles.

**Statistical safeguards:**
- Only hypotheses with `sparse_cells: false` and `adjusted_p_value < 0.05` are eligible for injection.
- A hypothesis is only promoted if the improvement is consistent (positive in > 50% of individual benchmark classes).

**Convergence:** After 5 validation cycles with no hypothesis producing > 5% improvement, the metacognitive loop is declared converged and stops generating new hypotheses.

**Output:** `Phase3ValidationReport` serialized to `phase3-report.json` alongside the suite report.

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

### Managed Client Integration

ICE may install managed MCP entries for external agent clients so they can mount the shared continuity kernel without manual JSON or YAML editing.

**OpenHands integration:**

```
ice openhands install-global   # Write managed MCP entry to ~/.openhands/mcp.json
ice openhands status           # Report binary/config/root status for the managed entry
ice openhands uninstall        # Remove only the managed entry
```

Behavior:

- Default OpenHands config path: `~/.openhands/mcp.json`
- Default OpenHands organism root: `~/.openhands/organisms/ice`
- Install writes a stdio MCP server entry that executes the current `ice` binary as `ice --root <organism-root> mcp`
- The entry MUST be marked as ICE-managed so uninstall removes only ICE-owned entries
- Install MUST refuse to overwrite an unmanaged same-name entry
- Install MUST be idempotent when the managed entry already matches the desired command and args
- Status MUST report whether the binary exists, whether the organism root exists, whether the config file exists, and whether the named entry is present

**OpenCode integration:**

```
ice opencode install-global    # Write managed MCP entry to ~/.config/opencode/opencode.json
ice opencode status            # Report binary/config/root status for the managed entry
ice opencode uninstall         # Remove only the managed entry
```

Behavior:

- Default OpenCode config path: `~/.config/opencode/opencode.json`
- Default OpenCode organism root: `~/.config/opencode/organisms/ice`
- Install writes a local MCP server entry under `mcp.<server-name>` that executes the current `ice` binary as `ice --root <organism-root> mcp`
- The local MCP command MUST be represented in OpenCode's array form: `["/path/to/ice", "--root", "...", "mcp"]`
- The entry MUST be marked as ICE-managed so uninstall removes only ICE-owned entries
- Install MUST refuse to overwrite an unmanaged same-name entry
- Install MUST be idempotent when the managed entry already matches the desired command and args
- Status MUST report whether the binary exists, whether the organism root exists, whether the config file exists, and whether the named entry is present

**Goose integration:**

```
ice goose install-global       # Write managed stdio extension into ~/.config/goose/config.yaml
ice goose status               # Report binary/config/root status for the managed entry
ice goose uninstall            # Remove only the managed entry
```

Behavior:

- Default Goose config path: `~/.config/goose/config.yaml`
- Default Goose organism root: `~/.config/goose/organisms/ice`
- Install writes a managed stdio extension under `extensions.<server-name>` that executes the current `ice` binary as `ice --root <organism-root> mcp`
- The extension MUST set Goose `type: stdio`, `cmd`, `args`, `enabled`, and empty `envs`
- The entry MUST be marked as ICE-managed so uninstall removes only ICE-owned entries
- Install MUST refuse to overwrite an unmanaged same-name entry
- Install MUST be idempotent when the managed entry already matches the desired command and args
- Status MUST report whether the binary exists, whether the organism root exists, whether the config file exists, and whether the named entry is present

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
