# DECISION REPORT: Phase 3 Metacognitive Closed-Loop Implementation

## 1. Final Verdict

**PR-A = GO**

PR-A (Agent A, branch `feat/metacognitive-phase3-agent-a`) is approved. PR-B (Agent B, branch `feat/metacognitive-phase3-agent-b`) should be closed.

---

## 2. Executive Summary

**Winner:** PR-A
**Why:** PR-A is the only implementation that closes the metacognitive loop. Its Phase 3 code executes in production (`ice bench`) — `run_phase3_injection` is called from `run_class` (line 1748), `generate_phase3_report` from `run_continuity_suite` (line 779), and `analyze_with_hypotheses` from `run_phase3_injection` (line 2203). PR-B's Phase 3 code is dead — all functions are called exclusively from `#[cfg(test)]` blocks.

**Minimum decisive factor:** PR-A wires Phase 3 into the benchmark execution path. PR-B does not.

**Why PR-B lost:** Every Phase 3 function (`validate_hypotheses`, `write_phase3_outcomes_to_kernel`, `extract_eligible_hypotheses`) has zero production call sites. By the project's own "no dead code" rule (AGENTS.md), all of PR-B's Phase 3 code is in violation. The validation logic is correct but orphaned.

**Why alternative outcomes were rejected:**
- BOTH TIED: impossible. PR-A executes, PR-B does not. This is not a matter of quality preference.
- BOTH NO GO: PR-A has real flaws (temperature=0 A/B concern, unsanitised prompt injection, no kernel write tests) but these are addressable in follow-up. The core loop closure is functional.
- PR-B = GO: PR-B cannot close the loop without ~50-100 lines of wiring that do not exist.

---

## 3. Real Problem Definition

### PR-A
- **Claimed problem:** Close the metacognitive loop — inject survival hypotheses into extraction prompts and measure improvement.
- **Actual problem solved:** Provides the full pipeline: hypothesis loading from prior runs → prompt injection → A/B control/treatment execution → per-class comparison → promotion/rejection → report generation. The loop is structurally closed.
- **Mismatch:** The A/B comparison at temperature=0 may produce identical outputs, making the measurement a near-no-op in practice. The loop is closed structurally but may not produce meaningful signal with deterministic models.

### PR-B
- **Claimed problem:** Same as PR-A.
- **Actual problem solved:** Provides validation logic, type definitions, and adapter modifications for hypothesis injection. Does NOT provide execution wiring.
- **Mismatch:** Critical. Claims to close the loop but does not connect the machinery to the benchmark execution path.

---

## 4. Reality vs Fantasy

### PR-A

| Category | Claims |
|----------|--------|
| **Directly proven by code** | Phase 3 functions are called from `run_class` (line 1748) and `run_continuity_suite` (line 779). `analyze_with_hypotheses` is called at line 2203. Division-by-zero is guarded. Promotion requires ≥3 classes and majority vote. Adapter backward compatible via default trait impl. |
| **Strongly inferred** | The promotion criteria (>5% absolute, majority classes) provide reasonable false-positive protection when combined with Phase 2's BH-corrected significance gate. |
| **Plausible but unproven** | A/B comparison produces meaningful signal at temperature=0. The `[SURVIVAL HINTS]` actually influence model output. |
| **Speculative** | That promoted Lessons will improve downstream continuity quality. |
| **Fantasy** | None identified in the code claims. SPEC.md is honest about limitations. |

### PR-B

| Category | Claims |
|----------|--------|
| **Directly proven by code** | Validation logic is correct. Promotion/rejection criteria match PR-A's. Adapter changes are backward compatible. 169 tests pass. |
| **Strongly inferred** | The code could be wired in with ~50-100 lines of changes. |
| **Fantasy** | The implicit claim that this "closes the loop." It does not. `ice bench` executes zero Phase 3 lines. |

---

## 5. Deep Analysis: PR-A (Agent A)

### Real scope
Adds ~400 lines to benchmark.rs + adapter modifications. Phase 3 activates conditionally when prior `meta-lessons.json` exists from a previous run. First run produces hypotheses, second run validates them.

### Technical design
1. `load_prior_hypotheses` reads `meta-lessons.json` from prior runs
2. `extract_eligible_hypotheses` filters to non-sparse, BH-adjusted p < 0.05
3. `run_phase3_injection` runs control (no hints) and treatment (with hints) on same envelope
4. `generate_phase3_report` aggregates per-class deltas, applies promotion criteria
5. `write_phase3_outcomes_to_kernel` promotes validated hypotheses to Lesson kind

### Strengths
- **The loop is closed.** This is the single most important property.
- Clean integration: Phase 3 is post-baseline, no interference with existing execution
- Adapter backward compatible via default trait implementation
- Per-class delta tracking enables post-hoc diagnostics
- Convergence detection (cycle ≥ 5 with no promotions)

### Weaknesses
1. **temperature=0 A/B concern** (plausible, unverified): deterministic model may produce identical output for both arms, collapsing measurement to noise
2. **Unsanitised prompt injection**: hypothesis text interpolated verbatim into prompt. Shared with PR-B. Requires kernel write access to exploit.
3. **Promoted Lessons have no expiry or re-test trigger**: a false positive becomes permanent
4. **Borderline hypotheses never get rejected**: only improvement ≤ 0.0 triggers rejection; hypotheses between 0% and 5% improvement survive indefinitely as open Hypothesis items
5. **`write_phase3_outcomes_to_kernel` untested**
6. **No e2e test with real kernel**
7. **Direction string matching fragility**: `hypothesis.direction.contains("Survived")` silently maps malformed values

### Risks
- False positive promotion at early cycles with few classes
- Prompt injection if kernel access is compromised (low probability, high impact)
- Zombie hypotheses accumulating between 0-5% improvement range

### Missing evidence
- No measurement of whether `[SURVIVAL HINTS]` actually change model output at temperature=0
- No e2e test proving the full cycle (Phase 2 → meta-lessons.json → Phase 3 → promotion)
- No backward compatibility test for reports without phase3 fields

### Trade-offs
- Chose execution completeness over statistical rigour (correct trade-off for v1)
- Chose conditional activation over always-on (correct — no hypotheses = no Phase 3)

---

## 6. Deep Analysis: PR-B (Agent B)

### Real scope
Adds Phase 3 types, validation logic, and adapter modifications. Does NOT wire into execution.

### Technical design
Same validation logic as PR-A (both independently converged on identical promotion criteria). Better standalone function design (`validate_hypotheses` takes pre-built slices). Same adapter changes (byte-for-byte identical adapters.rs).

### Strengths
- Clean standalone validation function
- 169 tests pass (more than main's 132)
- Correct convergence detection
- Well-structured types

### Weaknesses
1. **ALL PHASE 3 CODE IS DEAD.** No production call sites. This is the fatal flaw.
2. `Phase3Arm` enum is declared but never dispatched
3. `write_phase3_outcomes_to_kernel` has zero call sites (not even in tests)

### Why PR-B cannot win
The project's AGENTS.md states: "No dead code. Delete it, don't suppress it." PR-B adds ~300 lines of dead code. The validation logic is correct but unexercised. A library of correct functions that nothing calls is not a feature — it is technical debt with a positive test suite.

---

## 7. Decision Matrix

| Criterion | PR-A | PR-B | Evidence | Decisive? |
|-----------|------|------|----------|-----------|
| Alignment to real problem | **Closes the loop** | Types only, no execution | grep for call sites in non-test code | **DECISIVE** |
| Scope accuracy | Correct | Too narrow (missing wiring) | Code analysis | DECISIVE |
| Technical correctness | Correct with minor fragilities | Correct | Both verified by correctness analysts | Tie |
| Completeness | Structurally complete | Incomplete (dead code) | Production call site analysis | DECISIVE |
| Implementation clarity | Clean, post-baseline isolation | Clean standalone functions | Architecture review | Tie |
| Useful simplicity | Conditional activation | N/A (doesn't activate) | Code flow analysis | Non-decisive |
| Accidental complexity | ~400 lines, moderate | ~300 lines, lower | Line count | Non-decisive |
| Regression risk | Low (Phase 3 is post-baseline) | Zero (code never runs) | Architecture analysis | Non-decisive |
| Operational risk | Prompt injection (shared), temp=0 A/B concern | N/A (code never runs) | Critical opponent analysis | Non-decisive |
| Test quality | 5 Phase 3 tests, no e2e | 4 Phase 3 tests, no e2e | Test count and analysis | Tie |
| Evidence quality | Execution path verified with line numbers | Dead code verified with line numbers | Multiple independent reviewers | DECISIVE |
| Maintainability | 7800-line file (pre-existing concern) | 7400-line file | wc -l | Non-decisive |
| Architectural fit | Follows existing patterns | Follows existing patterns | Architecture review | Tie |
| Edge-case robustness | Direction string fragility, zombie hypotheses | N/A | Critical opponent | Non-decisive |
| Future evolution cost | Phase 4 can build on this directly | Would need wiring first | Code analysis | Non-decisive |
| Confidence in conclusions | HIGH | HIGH | 10 agents + Hawk | N/A |
| Unsupported narrative | temp=0 concern is unverified | "Closes the loop" claim is false | Hawk process review | N/A |

**Decisive criteria: 4 of 4 favour PR-A.** The decision is not close.

---

## 8. Ambiguities Found

### Ambiguity 1: Does PR-A actually wire Phase 3 into production code?

- **Detected by:** Hawk (process integrity monitor)
- **Investigated by:** Direct grep before tribunal launch + PR-A scope analyst
- **Resolution:** Confirmed. `run_phase3_injection` called at benchmark.rs:1748 (inside `run_class`), `generate_phase3_report` at line 779 (inside `run_continuity_suite`), `analyze_with_hypotheses` at line 2203 (inside `run_phase3_injection`). None are in test code.
- **Challenger attack:** None possible — line numbers are verifiable.
- **Status:** RESOLVED.

### Ambiguity 2: Does the A/B comparison produce meaningful signal at temperature=0?

- **Detected by:** PR-A critical opponent
- **Investigated by:** Not empirically verified (requires Ollama)
- **Resolution:** UNRESOLVED. Plausible concern. The model may produce identical output for both arms, making survival rate deltas zero. This would prevent false promotions (safe) but also prevent true promotions (system never learns).
- **Effect on verdict:** Does not change the PR-A vs PR-B decision. Both PRs share this architectural limitation. It affects whether Phase 3 produces useful results, not which PR should continue.

---

## 9. Internal Contests

### Contest 1: Is dead code an automatic disqualifier?

- **Thesis (attacker):** Dead code violates AGENTS.md. PR-B should be NO-GO.
- **Counter (defender):** The code could be wired in with ~50-100 lines. Ship the logic, wire later.
- **Evidence:** Defender-pragmatist examined the code and confirmed the attack. Defender said: "Do not ship dead code with a promise to connect it later."
- **Winner:** Attacker. The defender conceded.

### Contest 2: Are the statistics rigorous enough for promotion?

- **Thesis (attacker):** 15-25% false positive rate. No real significance test.
- **Counter (defender):** Phase 3 is gate 6-8 in an 8-gate pipeline. Phase 2 handles statistical significance. Phase 3 handles effect size.
- **Winner:** Defender. The pipeline-level argument is sound.

### Contest 3: Is prompt injection a real risk?

- **Thesis (attacker):** Zero sanitisation = trivial injection.
- **Counter (defender):** Requires kernel write access. If attacker has that, they can poison any of 13 kinds. Hypotheses decay fastest (36h).
- **Winner:** Defender. The threat model doesn't support hypothesis-specific sanitisation as a priority.

---

## 10. Discarded Reasoning Branches

### Branch: "PR-B's validation logic is cleaner, so PR-B should win on quality"
- **Why discarded:** Code quality is not a decisive criterion when one PR executes and the other doesn't. A clean function that nothing calls has zero value.
- **Evidence:** PR-B correctness analyst confirmed both PRs have identical adapter code and equivalent validation logic.
- **Effect:** None. This branch could not change the outcome.

### Branch: "The temperature=0 concern makes PR-A's A/B comparison invalid, so PR-A should also be NO-GO"
- **Why discarded:** Unverified claim (flagged by Hawk as "drunken talk"). Even if true, it makes Phase 3 a no-op (safe), not a harm. And PR-B shares the same limitation.
- **Effect:** None on PR selection. Noted as follow-up concern.

---

## 11. Final Reasoning Path

1. **Define the problem:** Phase 3 must close the metacognitive loop — inject hypotheses into prompts and measure whether they improve survival.
2. **Identify the decisive criterion:** Does the code execute in production (`ice bench`)?
3. **Verify PR-A:** grep confirms 3 production call sites (lines 1748, 779, 2203). Phase 3 activates conditionally. Verified by scope analyst independently.
4. **Verify PR-B:** grep confirms 0 production call sites. All Phase 3 functions test-only. Confirmed by 3 independent reviewers (pragmatist attacker, pragmatist defender, scope analyst).
5. **Check if quality difference could override:** Both PRs have identical adapter code. Both have correct validation logic. PR-A has richer per-class deltas. Neither has e2e tests. Quality is a tie at best, slight PR-A advantage.
6. **Check for disqualifying flaws in PR-A:** temperature=0 concern is unverified and shared by both PRs. Prompt injection requires kernel access (not unique to Phase 3). No blockers found.
7. **Check if PR-B could be wired in easily:** Defender estimated 50-100 lines. But that work hasn't been done. Shipping dead code with a promise to wire later violates project rules.
8. **Converge:** PR-A is the only implementation that satisfies the problem statement. The decision is not ambiguous.

---

## 12. Limits of the Conclusion

- **What could not be proven:** Whether the A/B comparison produces meaningful signal at temperature=0. This requires running `ice bench` with Ollama, which was not available.
- **What remains inference:** That the 8-gate pipeline provides sufficient false-positive protection. This is strongly inferred from the gate design but not empirically validated.
- **What new evidence could change the verdict:** If PR-A's wiring were found to be broken (e.g., `run_phase3_injection` always short-circuits due to empty hypotheses on first run), the verdict would shift to BOTH NO GO. However, the conditional activation on `!hypotheses.is_empty()` combined with `load_prior_hypotheses` reading from `meta-lessons.json` (which Phase 2 produces) makes this unlikely.

---

## 13. Actionable Conclusion

- **PR-A remains open.** Branch: `feat/metacognitive-phase3-agent-a`
- **PR-B should be closed.** Branch: `feat/metacognitive-phase3-agent-b`

### Required follow-up before merge:
1. **Test `write_phase3_outcomes_to_kernel`** with a real kernel (currently untested)
2. **Add backward compatibility test** for reports without `phase3` field
3. **Document the temperature=0 limitation** in SPEC.md — if the model is deterministic, A/B produces identical output and no promotion fires. This is safe but means Phase 3 only produces signal with temperature > 0.
4. **Consider Lesson expiry** — promoted Lessons currently have no re-test trigger or TTL

### Not required before merge (follow-up work):
- Prompt sanitisation (threat model doesn't justify it at this stage)
- Fisher's exact test for sparse cells
- Module extraction of benchmark.rs
- Zombie hypothesis cleanup (borderline 0-5% improvement range)

---

## Appendix: Evidence Sources

All tribunal findings are persisted in the ICE continuity kernel with prefix `tribunal:`. 10 agents + 1 Hawk produced 11 continuity items. 5 prior attackers (`phase3-review:`) and 5 prior defenders (`phase3-defense:`) produced 10 additional items. Total: 21 continuity items documenting the full adversarial review.

### Agents involved
| Agent | Role | Model | Assigned to |
|-------|------|-------|-------------|
| tribunal-a-scope | Problem/scope | Haiku | PR-A |
| tribunal-a-correctness | Implementation | Sonnet | PR-A |
| tribunal-a-testing | Testing/evidence | Haiku | PR-A |
| tribunal-a-opponent | Critical opponent | Sonnet | PR-A |
| tribunal-a-architecture | Architecture | Haiku | PR-A |
| tribunal-b-scope | Problem/scope | Haiku | PR-B |
| tribunal-b-correctness | Implementation quality | Sonnet | PR-B |
| tribunal-b-testing | Testing comparison | Haiku | PR-B |
| tribunal-b-opponent | Viability assessment | Sonnet | PR-B |
| hawk | Process integrity | Opus | Observer |

### Verification commands
```bash
# Verify PR-A wiring (decisive evidence)
cd .claude/worktrees/agent-a057a123
grep -n "run_phase3_injection\|generate_phase3_report\|analyze_with_hypotheses" src/benchmark.rs | grep -v "test\|Test\|#\[test\]\|mod tests\|pub fn\|///\|fn "

# Verify PR-B dead code (decisive evidence)
cd .claude/worktrees/agent-a82d4eec
grep -n "validate_hypotheses\|write_phase3_outcomes\|extract_eligible\|Phase3Arm" src/benchmark.rs | grep -v "test\|Test\|#\[test\]\|mod tests\|pub fn\|pub struct\|pub enum\|///\|fn "
```

---

*Report generated 2026-03-26. Tribunal process: 10 analysis agents + 1 Hawk observer, balanced across Haiku/Sonnet/Opus model tiers. All findings persisted to ICE continuity kernel. Decision is deterministic and reproducible from the evidence cited.*
