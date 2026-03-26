use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::{Context, Result};
use serde_json::{Value, json};
use tempfile::tempdir;

struct McpHarness {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl McpHarness {
    fn spawn(root: &Path) -> Result<Self> {
        let mut child = Command::new(env!("CARGO_BIN_EXE_ice"))
            .arg("--root")
            .arg(root)
            .arg("mcp")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("spawning ice mcp")?;
        let stdin = child.stdin.take().context("capturing child stdin")?;
        let stdout = child.stdout.take().context("capturing child stdout")?;
        Ok(Self {
            child,
            stdin: Some(stdin),
            stdout: BufReader::new(stdout),
            next_id: 1,
        })
    }

    fn initialize(&mut self) -> Result<()> {
        let id = self.take_id();
        let response = self.request(json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {
                    "name": "mcp-stdio-test",
                    "version": "0.1.0"
                }
            }
        }))?;
        assert_eq!(
            response["result"]["serverInfo"]["name"],
            "ice-shared-continuity-kernel"
        );

        self.notify(json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }))
    }

    fn tool_call(&mut self, name: &str, arguments: Value) -> Result<Value> {
        let id = self.take_id();
        let response = self.request(json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }))?;
        let result = response
            .get("result")
            .context("missing tools/call result envelope")?;
        let is_error = result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if is_error {
            anyhow::bail!(
                "MCP tool {name} failed: {}",
                result["content"][0]["text"]
                    .as_str()
                    .unwrap_or("unknown error")
            );
        }
        Ok(result["structuredContent"].clone())
    }

    fn request(&mut self, request: Value) -> Result<Value> {
        let stdin = self.stdin.as_mut().context("stdin already closed")?;
        writeln!(stdin, "{request}").context("writing JSON-RPC request")?;
        stdin.flush().context("flushing JSON-RPC request")?;

        let mut line = String::new();
        self.stdout
            .read_line(&mut line)
            .context("reading JSON-RPC response")?;
        serde_json::from_str(&line).with_context(|| format!("parsing JSON-RPC response: {line}"))
    }

    fn notify(&mut self, request: Value) -> Result<()> {
        let stdin = self.stdin.as_mut().context("stdin already closed")?;
        writeln!(stdin, "{request}").context("writing JSON-RPC notification")?;
        stdin.flush().context("flushing JSON-RPC notification")?;
        Ok(())
    }

    fn shutdown(mut self) -> Result<()> {
        drop(self.stdin.take());
        let status = self.child.wait().context("waiting for mcp process exit")?;
        assert!(
            status.success(),
            "expected clean mcp shutdown, got {status}"
        );
        Ok(())
    }

    fn take_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

fn titles(items: &Value) -> Vec<String> {
    items
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|item| {
            item.get("title")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect()
}

#[test]
fn mcp_stdio_survives_handoff_and_runtime_restart() -> Result<()> {
    let dir = tempdir()?;
    let objective = "Keep continuity stable across disposable external heads";

    let mut first = McpHarness::spawn(dir.path())?;
    first.initialize()?;

    let bootstrap = first.tool_call(
        "continuity_bootstrap",
        json!({
            "agent_id": "planner-a",
            "agent_type": "codex",
            "namespace": "shared-lab",
            "task_id": "bridge-proof",
            "session_id": "session-a",
            "objective": objective,
            "role": "planner",
            "capabilities": ["read", "write", "handoff"],
            "token_budget": 256,
            "candidate_limit": 12
        }),
    )?;
    let context_id = bootstrap["context"]["id"]
        .as_str()
        .context("bootstrap missing context id")?
        .to_string();

    let decision = first.tool_call(
        "continuity_write_item",
        json!({
            "context_id": context_id,
            "author_agent_id": "planner-a",
            "kind": "decision",
            "title": "Use the MCP bridge as the shared head interface",
            "body": "External agents must attach through stdio JSON-RPC so continuity survives model swaps.",
            "scope": "shared",
            "importance": 0.98,
            "confidence": 0.94
        }),
    )?;
    let decision_id = decision["id"]
        .as_str()
        .context("decision write missing id")?
        .to_string();

    first.tool_call(
        "continuity_write_item",
        json!({
            "context_id": context_id,
            "author_agent_id": "planner-a",
            "kind": "constraint",
            "title": "Do not leak the raw transcript",
            "body": "The next head may only inherit continuity through the sanctioned UCI and MCP path.",
            "scope": "shared",
            "importance": 0.99,
            "confidence": 0.97
        }),
    )?;

    first.tool_call(
        "continuity_write_item",
        json!({
            "context_id": context_id,
            "author_agent_id": "planner-a",
            "kind": "operational_scar",
            "title": "Nested JSON contracts broke small models",
            "body": "The earlier nested schema caused total failure on small local models. Keep the flattened string-array contract.",
            "scope": "project",
            "importance": 1.0,
            "confidence": 0.99
        }),
    )?;

    let handoff = first.tool_call(
        "continuity_handoff",
        json!({
            "from_agent_id": "planner-a",
            "to_agent_id": "coder-b",
            "context_id": context_id,
            "objective": objective,
            "reason": "Swap to the next external head without losing the thread",
            "token_budget": 256,
            "candidate_limit": 12
        }),
    )?;
    let snapshot_id = handoff["snapshot"]["id"]
        .as_str()
        .context("handoff missing snapshot id")?
        .to_string();
    let handoff_id = handoff["handoff"]["id"]
        .as_str()
        .context("handoff missing handoff id")?
        .to_string();

    let handoff_decisions = titles(&handoff["context"]["decisions"]);
    let handoff_constraints = titles(&handoff["context"]["constraints"]);
    let handoff_scars = titles(&handoff["context"]["operational_scars"]);
    assert!(
        handoff_decisions.contains(&"Use the MCP bridge as the shared head interface".to_string())
    );
    assert!(handoff_constraints.contains(&"Do not leak the raw transcript".to_string()));
    assert!(handoff_scars.contains(&"Nested JSON contracts broke small models".to_string()));

    first.shutdown()?;

    let mut second = McpHarness::spawn(dir.path())?;
    second.initialize()?;

    let resumed = second.tool_call(
        "continuity_resume",
        json!({
            "snapshot_id": snapshot_id,
            "objective": objective,
            "agent_id": "coder-b",
            "token_budget": 256,
            "candidate_limit": 12
        }),
    )?;

    let resumed_decisions = titles(&resumed["context"]["decisions"]);
    let resumed_constraints = titles(&resumed["context"]["constraints"]);
    let resumed_scars = titles(&resumed["context"]["operational_scars"]);
    assert!(
        resumed_decisions.contains(&"Use the MCP bridge as the shared head interface".to_string())
    );
    assert!(resumed_constraints.contains(&"Do not leak the raw transcript".to_string()));
    assert!(resumed_scars.contains(&"Nested JSON contracts broke small models".to_string()));

    let explained = second.tool_call(
        "continuity_explain",
        json!({
            "kind": "handoff",
            "id": handoff_id
        }),
    )?;
    assert_eq!(explained["handoff"]["to_agent_id"], "coder-b");

    let decision_explained = second.tool_call(
        "continuity_explain",
        json!({
            "kind": "continuity_item",
            "id": decision_id
        }),
    )?;
    assert_eq!(
        decision_explained["item"]["title"],
        "Use the MCP bridge as the shared head interface"
    );

    second.shutdown()?;
    Ok(())
}
