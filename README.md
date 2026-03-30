# Infinite Continuity Engine

Local, self-hostable continuity substrate for interchangeable agents.

One persistent working mind on disk while agents come and go. Context identity is primary, agent identity is disposable, and continuity survives swaps, restarts, and role changes.

> **Disclaimer**: This repository is a brute-force validation of a hypothesis that drives the [Sevori Continuity Protocol](https://github.com/Sevori/continuity-protocol) spec and research. It has no intention of building a real product or producing quality code — it exists solely to stress-test an idea until it either proves itself or breaks.

> **Why "adhd"?** This was built around one person's very specific way of learning. A brain with ADHD doesn't fit the normal flow — thoughts need to be typed, structured, and externalised because the default doesn't work. This engine is the tool that emerged from that constraint.

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

The installer now expects `gh` to be installed and authenticated for the target repository.

## Claude Code

Claude Code is now a repo-owned install path.

Install the managed MCP entry:

```bash
ice claude install-global
```

Check the install:

```bash
ice claude status
```

Update later by rerunning the installer. That replaces the `ice` binary in place and keeps the same Claude MCP entry and organism root.

What this does:

- writes a managed MCP entry into `~/.claude.json`
- points it at `ice --root ~/.claude/organisms/ice mcp`
- refuses to overwrite an unmanaged same-name entry

To use a custom organism root:

```bash
ice claude install-global --root /path/to/shared-organism
```

To remove the managed entry:

```bash
ice claude uninstall
```

Restart Claude Code after installation so it reloads MCP servers.

## Open Source Agents

ICE can now install managed continuity entries for three open-source agent clients: OpenHands, OpenCode, and Goose.

### OpenHands

Install the managed MCP entry:

```bash
ice openhands install-global
ice openhands status
ice openhands uninstall
```

What this does:

- writes a managed MCP entry into `~/.openhands/mcp.json`
- points it at `ice --root ~/.openhands/organisms/ice mcp`
- refuses to overwrite an unmanaged same-name entry

### OpenCode

Install the managed MCP entry:

```bash
ice opencode install-global
ice opencode status
ice opencode uninstall
```

What this does:

- writes a managed local MCP entry into `~/.config/opencode/opencode.json`
- points it at `ice --root ~/.config/opencode/organisms/ice mcp`
- refuses to overwrite an unmanaged same-name entry

### Goose

Install the managed MCP entry:

```bash
ice goose install-global
ice goose status
ice goose uninstall
```

What this does:

- writes a managed stdio extension into `~/.config/goose/config.yaml`
- points it at `ice --root ~/.config/goose/organisms/ice mcp`
- refuses to overwrite an unmanaged same-name entry

Restart the client after installation so it reloads MCP servers.

## Local Models on M2 Max 64GB

For local open-source models on an M2 Max with 64 GB RAM, the safe default is:

- main model: `qwen2.5:14b`
- small/helper model: `qwen2.5:3b` or `qwen2.5:1.5b`
- slower/stronger option: `qwen2.5:32b`

OpenCode example with a local Ollama-compatible endpoint:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (local)",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "qwen2.5:14b": { "name": "Qwen 2.5 14B" },
        "qwen2.5:3b": { "name": "Qwen 2.5 3B" }
      }
    }
  },
  "model": "ollama/qwen2.5:14b",
  "small_model": "ollama/qwen2.5:3b"
}
```

Goose: run `goose configure`, choose `Ollama`, keep `localhost:11434` unless your daemon is elsewhere, and select a tool-calling model such as `qwen2.5:14b`.

## Quick Start

### The idea

Open two terminals (or two agents). One ices knowledge, the other recalls it. They share the same brain on disk.

### From the terminal

**Terminal A — ice knowledge:**

```bash
ice --root .ice ingest \
  --kind note \
  --agent "me" \
  --session "study-1" \
  --namespace "personal" \
  --text "My name is Renato. I am a TDD developer — Trauma Driven Development. My sense of humour is not common."
```

**Terminal B — a cold session recalls it:**

```bash
ice --root .ice query \
  --agent "someone-else" \
  --session "cold-session" \
  --namespace "personal" \
  --text "What do you know about Renato?" \
  --budget-tokens 128
```

The cold session has never seen Renato, but the iced knowledge surfaces immediately.

### From Claude Code (via MCP)

After installing the MCP entry (`ice claude install-global`), restart Claude Code. The continuity tools become native MCP calls.

**Terminal 1 — tell Claude to ice it:**

> "Hey Claude, my name is Renato, I'm a TDD developer (Trauma Driven Development) and my sense of humour isn't common. Ice it."

Claude bootstraps a context and writes the facts:

```json
// Claude calls continuity_bootstrap to open a context
continuity_bootstrap({
  "agent_id": "claude-code",
  "agent_type": "claude",
  "namespace": "personal",
  "task_id": "getting-to-know",
  "session_id": "session-1",
  "objective": "Learn about the human and ice it for future agents"
})

// Then ices the knowledge as typed facts
continuity_write_items({
  "context_id": "<from bootstrap>",
  "author_agent_id": "claude-code",
  "items": [
    {"kind": "fact", "title": "Human identity", "body": "The human's name is Renato."},
    {"kind": "fact", "title": "Development philosophy", "body": "Renato is a TDD developer — Trauma Driven Development. He builds robust systems because past trauma taught him what breaks."},
    {"kind": "fact", "title": "Sense of humour", "body": "Renato's sense of humour is not common. Expect dry, dark, or absurd jokes. Do not over-explain or soften them."}
  ]
})
```

**Terminal 2 — a cold agent asks what it knows:**

> "What do you know about Renato?"

A completely fresh Claude session (or Codex, or any agent) bootstraps into the same namespace and gets everything back:

```json
// Cold agent bootstraps into the same namespace/task
continuity_bootstrap({
  "agent_id": "cold-agent",
  "agent_type": "claude",
  "namespace": "personal",
  "task_id": "getting-to-know",
  "session_id": "fresh-session",
  "objective": "What do you know about Renato?"
})

// The bootstrap response already contains the recalled facts:
// - "The human's name is Renato."
// - "Renato is a TDD developer — Trauma Driven Development."
// - "Renato's sense of humour is not common. Expect dry, dark, or absurd jokes."
```

The cold agent never saw the original conversation. It just bootstrapped into the same shared brain and got the iced knowledge back, ranked by relevance.

The MCP server reads from the same on-disk root, so knowledge iced from the terminal, from Claude, or from Codex all lives in the same place.

## Other Ways to Run

Run the MCP bridge directly (without the managed install):

```bash
ice --root .ice mcp
```

Run the HTTP API:

```bash
ice --root .ice serve --addr 127.0.0.1:4040
```

Run the local benchmark:

```bash
ice --root .ice-bench bench \
  --mode continuity \
  --budget-tokens 192 \
  --candidate-limit 12 \
  --recent-window 6
```

## LongMemEval

ICE now ships a native `LongMemEval` runner for generating benchmark predictions from an isolated continuity replay, plus an optional wrapper around the official evaluator.

### What it does

- replays each LongMemEval history into a fresh ICE root
- preserves the original dataset timestamps during ingest
- retrieves a bounded continuity pack for the benchmark question
- can run a question-conditioned reading-notes pass over each retrieved session before the final answer
- recovers full retrieved session transcripts for the final answer when they fit the prompt budget, instead of relying only on clipped excerpts
- asks a reader model for the final answer
- writes `jsonl` predictions in the official `{"question_id","hypothesis"}` shape

### Get the official dataset

```bash
mkdir -p data/longmemeval
cd data/longmemeval
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json
cd ../..
```

### Generate predictions

This example uses an Ollama-served reader model and the lightweight `oracle` split first.

```bash
ice --root .ice-longmemeval longmemeval run \
  --dataset data/longmemeval/longmemeval_oracle.json \
  --output artifacts/longmemeval/oracle-predictions.jsonl \
  --reader-endpoint http://127.0.0.1:11434 \
  --reader-model qwen2.5:14b \
  --budget-tokens 512 \
  --candidate-limit 24
```

You can also target any OpenAI-compatible `/v1/chat/completions` endpoint:

```bash
export OPENAI_API_KEY=YOUR_KEY
ice --root .ice-longmemeval longmemeval run \
  --dataset data/longmemeval/longmemeval_oracle.json \
  --output artifacts/longmemeval/oracle-openai-compatible.jsonl \
  --reader-provider openai-compatible \
  --reader-endpoint https://api.openai.com/v1 \
  --reader-model gpt-4.1-mini
```

Useful controls:

- `--max-cases 20` to smoke-test on a subset
- `--offset 20` to resume on the next slice
- `--question-id q1,q2` to target specific cases
- `--question-type temporal-reasoning,multi-session` to focus on one family
- `--work-dir /tmp/ice-longmemeval-work` to store per-case replay roots elsewhere
- `--reader-provider ollama|openai-compatible` to switch the answer generator
- `--reader-method con-separate|direct` to choose between the default two-stage reader and single-pass answering
- `--reader-api-key-env OPENAI_API_KEY` to source a bearer token for compatible hosted endpoints
- `--reader-max-retries 4` and `--reader-retry-backoff-secs 2` to survive transient 429/5xx failures on hosted readers

The command writes:

- the official predictions file you can submit to the evaluator
- a JSON report next to it
- a debug directory with one prompt, one response, one serialized context pack, and optional reading notes per case

Implementation note:

- the final answer prompt now prefers the full recovered session transcripts when they stay within a bounded prompt budget; the reading notes are treated as a guide, not the source of truth
- the default LongMemEval reader output budget is `256` tokens so arithmetic and counting answers do not get truncated mid-response

### Run the official evaluator

Clone the benchmark repository first:

```bash
git clone https://github.com/xiaowu0162/LongMemEval.git
python3.12 -m venv /tmp/LongMemEval-venv
/tmp/LongMemEval-venv/bin/pip install -r ./LongMemEval/requirements-lite.txt
/tmp/LongMemEval-venv/bin/pip install 'httpx<0.28'
```

Then run:

```bash
export OPENAI_API_KEY=YOUR_KEY
ice longmemeval evaluate \
  --repo ./LongMemEval \
  --predictions artifacts/longmemeval/oracle-predictions.jsonl \
  --dataset data/longmemeval/longmemeval_oracle.json \
  --python-bin /tmp/LongMemEval-venv/bin/python \
  --judge-model gpt-4o
```

Notes:

- the official `print_qa_metrics.py` script currently assumes `gpt-4o`; ICE skips that summary step automatically for other judge models
- the upstream evaluator stack is currently reliable on Python 3.12; Python 3.14 and `httpx 0.28+` break `openai==1.35.1`
- the runner does not ingest benchmark-only supervision labels such as `answer_session_ids` or `has_answer`
- `ice ingest` now accepts `--timestamp <RFC3339>` when you need to replay historical events with real time ordering

### Repeatable Oracle Runs

For a full reproducible `oracle` benchmark pass, use:

```bash
make longmemeval-oracle
```

The helper script:

- downloads the requested LongMemEval split
- clones the official evaluator repository
- creates a Python 3.12 evaluator venv with the known-good dependency pins
- runs `ice longmemeval run`
- runs `ice longmemeval evaluate`
- writes a Markdown summary under `.artifacts/longmemeval/`

GitHub Actions also ships a dedicated `LongMemEval Oracle` workflow for manual dispatch. It runs the same script and uploads the resulting benchmark artifacts.
