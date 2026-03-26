use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::Engine;
use crate::continuity::{
    AttachAgentInput, ClaimWorkInput, ContinuityHandoffInput, ContinuityItemInput, ContinuityKind,
    ContinuityStatus, CoordinationLane, CoordinationProjectedLane, CoordinationSeverity,
    CoordinationSignalInput, ExplainTarget, MACHINE_NAMESPACE_ALIAS, OpenContextInput,
    ReadContextInput, ResumeInput, SignalInput, SnapshotInput, SupportRef,
    UnifiedContinuityInterface, UpsertAgentBadgeInput, WriteEventInput,
};
use crate::model::{DimensionValue, EventKind, MemoryLayer, Scope, Selector, SnapshotResolution};

const JSON_RPC_VERSION: &str = "2.0";
const MCP_PROTOCOL_VERSION: &str = "2025-06-18";
const MCP_SERVER_NAME: &str = "ice-shared-continuity-kernel";
const DEFAULT_TOKEN_BUDGET: usize = 512;
const DEFAULT_CANDIDATE_LIMIT: usize = 24;
const DEFAULT_REPLAY_LIMIT: usize = 50;

pub fn serve_stdio(engine: Arc<Engine>) -> Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = BufWriter::new(stdout.lock());
    let mut line = String::new();

    loop {
        line.clear();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            writer.flush()?;
            return Ok(());
        }

        let payload = line.trim();
        if payload.is_empty() {
            continue;
        }

        let request: JsonRpcRequest = serde_json::from_str(payload)
            .with_context(|| format!("parsing MCP request: {payload}"))?;
        if let Some(response) = handle_jsonrpc_request(engine.as_ref(), request)? {
            serde_json::to_writer(&mut writer, &response)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
    }
}

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: Option<String>,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Deserialize)]
struct ToolCallRequest {
    name: String,
    #[serde(default)]
    arguments: Value,
}

#[derive(Debug, Deserialize)]
struct BootstrapInput {
    agent_id: String,
    agent_type: String,
    namespace: Option<String>,
    task_id: Option<String>,
    session_id: String,
    objective: String,
    #[serde(default)]
    capabilities: Vec<String>,
    role: Option<String>,
    #[serde(default)]
    metadata: Value,
    selector: Option<Selector>,
    #[serde(default = "default_token_budget_value")]
    token_budget: usize,
    #[serde(default = "default_candidate_limit_value")]
    candidate_limit: usize,
}

#[derive(Debug, Deserialize)]
struct ReadToolInput {
    context_id: Option<String>,
    namespace: Option<String>,
    task_id: Option<String>,
    objective: String,
    selector: Option<Selector>,
    agent_id: Option<String>,
    session_id: Option<String>,
    view_id: Option<String>,
    #[serde(default)]
    include_resolved: bool,
    #[serde(default = "default_token_budget_value")]
    token_budget: usize,
    #[serde(default = "default_candidate_limit_value")]
    candidate_limit: usize,
}

#[derive(Debug, Deserialize)]
struct RecallToolInput {
    context_id: Option<String>,
    namespace: Option<String>,
    task_id: Option<String>,
    objective: String,
    #[serde(default)]
    include_resolved: bool,
    #[serde(default = "default_candidate_limit_value")]
    candidate_limit: usize,
}

#[derive(Debug, Deserialize)]
struct SnapshotToolInput {
    context_id: Option<String>,
    namespace: Option<String>,
    task_id: Option<String>,
    objective: Option<String>,
    selector: Option<Selector>,
    resolution: Option<crate::model::SnapshotResolution>,
    #[serde(default = "default_token_budget_value")]
    token_budget: usize,
    #[serde(default = "default_candidate_limit_value")]
    candidate_limit: usize,
    owner_agent_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResumeToolInput {
    snapshot_id: Option<String>,
    context_id: Option<String>,
    namespace: Option<String>,
    task_id: Option<String>,
    objective: String,
    agent_id: Option<String>,
    #[serde(default = "default_token_budget_value")]
    token_budget: usize,
    #[serde(default = "default_candidate_limit_value")]
    candidate_limit: usize,
}

#[derive(Debug, Deserialize)]
struct HandoffToolInput {
    from_agent_id: String,
    to_agent_id: String,
    context_id: Option<String>,
    namespace: Option<String>,
    task_id: Option<String>,
    objective: String,
    reason: String,
    selector: Option<Selector>,
    resolution: Option<crate::model::SnapshotResolution>,
    #[serde(default = "default_token_budget_value")]
    token_budget: usize,
    #[serde(default = "default_candidate_limit_value")]
    candidate_limit: usize,
}

#[derive(Debug, Deserialize)]
struct ClaimWorkToolInput {
    context_id: String,
    agent_id: String,
    title: String,
    body: String,
    scope: crate::model::Scope,
    #[serde(default)]
    resources: Vec<String>,
    #[serde(default)]
    exclusive: bool,
    attachment_id: Option<String>,
    lease_seconds: Option<u64>,
    #[serde(default)]
    extra: Value,
}

#[derive(Debug, Deserialize)]
struct AgentBadgeToolInput {
    attachment_id: Option<String>,
    agent_id: Option<String>,
    namespace: Option<String>,
    context_id: Option<String>,
    display_name: Option<String>,
    status: Option<String>,
    focus: Option<String>,
    headline: Option<String>,
    resource: Option<String>,
    repo_root: Option<String>,
    branch: Option<String>,
    #[serde(default)]
    metadata: Value,
}

#[derive(Debug, Deserialize)]
struct CoordinationSignalToolInput {
    context_id: String,
    agent_id: String,
    title: String,
    body: String,
    lane: CoordinationLane,
    target_agent_id: Option<String>,
    target_projected_lane: Option<CoordinationProjectedLane>,
    claim_id: Option<String>,
    resource: Option<String>,
    severity: Option<CoordinationSeverity>,
    #[serde(default)]
    projection_ids: Vec<String>,
    #[serde(default)]
    projected_lanes: Vec<CoordinationProjectedLane>,
    #[serde(default)]
    extra: Value,
}

#[derive(Debug, Deserialize)]
struct ReplayToolInput {
    selector: Selector,
    #[serde(default = "default_replay_limit_value")]
    limit: usize,
}

#[derive(Debug, Deserialize)]
struct EventToolInput {
    context_id: Option<String>,
    kind: String,
    agent_id: String,
    agent_role: Option<String>,
    session_id: String,
    task_id: Option<String>,
    project_id: Option<String>,
    goal_id: Option<String>,
    run_id: Option<String>,
    namespace: Option<String>,
    environment: Option<String>,
    source: String,
    scope: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    dimensions: Vec<DimensionValue>,
    content: String,
    #[serde(default)]
    attributes: Value,
}

#[derive(Debug, Deserialize)]
struct ContinuityItemToolInput {
    context_id: String,
    author_agent_id: String,
    kind: String,
    title: String,
    body: String,
    scope: Option<String>,
    status: Option<String>,
    importance: Option<f64>,
    confidence: Option<f64>,
    salience: Option<f64>,
    layer: Option<String>,
    #[serde(default)]
    supports: Vec<SupportRef>,
    #[serde(default)]
    dimensions: Vec<DimensionValue>,
    #[serde(default)]
    extra: Value,
}

#[derive(Debug, Deserialize)]
struct ExplainToolInput {
    kind: String,
    id: Option<String>,
}

fn default_token_budget_value() -> usize {
    DEFAULT_TOKEN_BUDGET
}

fn default_candidate_limit_value() -> usize {
    DEFAULT_CANDIDATE_LIMIT
}

fn default_replay_limit_value() -> usize {
    DEFAULT_REPLAY_LIMIT
}

fn parse_scope_alias(value: Option<&str>) -> Scope {
    match value
        .unwrap_or("shared")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "agent" => Scope::Agent,
        "session" => Scope::Session,
        "project" => Scope::Project,
        "global" => Scope::Global,
        "shared" | "" => Scope::Shared,
        _ => Scope::Shared,
    }
}

fn parse_event_kind_alias(value: &str) -> Result<EventKind> {
    match value.trim().to_ascii_lowercase().as_str() {
        "prompt" => Ok(EventKind::Prompt),
        "response" => Ok(EventKind::Response),
        "tool_call" => Ok(EventKind::ToolCall),
        "tool_result" => Ok(EventKind::ToolResult),
        "shell_command" => Ok(EventKind::ShellCommand),
        "shell_output" => Ok(EventKind::ShellOutput),
        "file_diff" => Ok(EventKind::FileDiff),
        "error" => Ok(EventKind::Error),
        "exception" => Ok(EventKind::Exception),
        "document" => Ok(EventKind::Document),
        "trace" => Ok(EventKind::Trace),
        "api_request" => Ok(EventKind::ApiRequest),
        "api_response" => Ok(EventKind::ApiResponse),
        "note" | "observation" | "log" => Ok(EventKind::Note),
        other => Err(anyhow!("unsupported continuity_write_event kind: {other}")),
    }
}

fn parse_continuity_kind_alias(value: &str) -> Result<ContinuityKind> {
    match value.trim().to_ascii_lowercase().as_str() {
        "working_state" => Ok(ContinuityKind::WorkingState),
        "work_claim" => Ok(ContinuityKind::WorkClaim),
        "derivation" => Ok(ContinuityKind::Derivation),
        "fact" => Ok(ContinuityKind::Fact),
        "decision" => Ok(ContinuityKind::Decision),
        "constraint" => Ok(ContinuityKind::Constraint),
        "hypothesis" => Ok(ContinuityKind::Hypothesis),
        "incident" => Ok(ContinuityKind::Incident),
        "operational_scar" | "scar" => Ok(ContinuityKind::OperationalScar),
        "outcome" => Ok(ContinuityKind::Outcome),
        "signal" => Ok(ContinuityKind::Signal),
        "summary" => Ok(ContinuityKind::Summary),
        "lesson" => Ok(ContinuityKind::Lesson),
        other => Err(anyhow!("unsupported continuity_write_item kind: {other}")),
    }
}

fn parse_continuity_status_alias(value: Option<&str>) -> Result<Option<ContinuityStatus>> {
    match value.map(|value| value.trim().to_ascii_lowercase()) {
        None => Ok(None),
        Some(value) if value.is_empty() => Ok(None),
        Some(value) => match value.as_str() {
            "open" => Ok(Some(ContinuityStatus::Open)),
            "active" => Ok(Some(ContinuityStatus::Active)),
            "resolved" => Ok(Some(ContinuityStatus::Resolved)),
            "superseded" => Ok(Some(ContinuityStatus::Superseded)),
            "rejected" => Ok(Some(ContinuityStatus::Rejected)),
            other => Err(anyhow!("unsupported continuity status: {other}")),
        },
    }
}

fn parse_memory_layer_alias(value: Option<&str>) -> Result<Option<MemoryLayer>> {
    match value.map(|value| value.trim().to_ascii_lowercase()) {
        None => Ok(None),
        Some(value) if value.is_empty() => Ok(None),
        Some(value) => match value.as_str() {
            "hot" => Ok(Some(MemoryLayer::Hot)),
            "episodic" => Ok(Some(MemoryLayer::Episodic)),
            "semantic" => Ok(Some(MemoryLayer::Semantic)),
            "summary" => Ok(Some(MemoryLayer::Summary)),
            "cold" => Ok(Some(MemoryLayer::Cold)),
            other => Err(anyhow!("unsupported memory layer: {other}")),
        },
    }
}

fn parse_explain_target_alias(kind: &str, id: Option<String>) -> Result<ExplainTarget> {
    let id = id.ok_or_else(|| anyhow!("continuity_explain requires id"))?;
    match kind.trim().to_ascii_lowercase().as_str() {
        "context" => Ok(ExplainTarget::Context { id }),
        "item" | "continuity_item" => Ok(ExplainTarget::ContinuityItem { id }),
        "snapshot" => Ok(ExplainTarget::Snapshot { id }),
        "handoff" => Ok(ExplainTarget::Handoff { id }),
        "pack" => Ok(ExplainTarget::Pack { id }),
        "view" => Ok(ExplainTarget::View { id }),
        other => Err(anyhow!("unsupported continuity_explain kind: {other}")),
    }
}

fn resolve_machine_scope(
    engine: &Engine,
    namespace: Option<String>,
    task_id: Option<String>,
    apply_defaults: bool,
) -> Result<(Option<String>, Option<String>, Value)> {
    let machine = serde_json::to_value(engine.identify_machine()?)?;
    let machine_namespace = machine
        .get("namespace")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let default_task_id = machine
        .get("default_task_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let namespace = match namespace.as_deref() {
        Some(MACHINE_NAMESPACE_ALIAS) => Some(machine_namespace.clone()),
        Some(other) => Some(other.to_string()),
        None if apply_defaults => Some(machine_namespace.clone()),
        None => None,
    };
    let task_id = match task_id {
        Some(task_id) => Some(task_id),
        None if apply_defaults => Some(default_task_id),
        None => None,
    };
    Ok((namespace, task_id, machine))
}

fn default_snapshot_resolution_value() -> SnapshotResolution {
    SnapshotResolution::Medium
}

fn handle_jsonrpc_request(engine: &Engine, request: JsonRpcRequest) -> Result<Option<Value>> {
    let id = request.id.clone();
    let response = match request.method.as_str() {
        "initialize" => success_response(
            id,
            json!({
                "protocolVersion": request
                    .params
                    .get("protocolVersion")
                    .and_then(Value::as_str)
                    .unwrap_or(MCP_PROTOCOL_VERSION),
                "capabilities": {
                    "tools": {
                        "listChanged": false
                    }
                },
                "serverInfo": {
                    "name": MCP_SERVER_NAME,
                    "title": "Shared Continuity Kernel",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        ),
        "notifications/initialized" => return Ok(None),
        "ping" => success_response(id, json!({})),
        "tools/list" => success_response(id, json!({ "tools": tool_definitions() })),
        "resources/list" => success_response(id, json!({ "resources": [] })),
        "prompts/list" => success_response(id, json!({ "prompts": [] })),
        "tools/call" => {
            let call: ToolCallRequest =
                serde_json::from_value(request.params).context("parsing tools/call params")?;
            let result = match dispatch_tool(engine, &call.name, call.arguments) {
                Ok(value) => tool_success(value),
                Err(error) => tool_error(error),
            };
            success_response(id, result)
        }
        other => error_response(id, -32601, format!("unsupported MCP method: {other}"), None),
    };

    Ok(Some(response))
}

fn dispatch_tool(engine: &Engine, name: &str, arguments: Value) -> Result<Value> {
    match name {
        "continuity_identify_machine" => continuity_identify_machine(engine),
        "continuity_bootstrap" => continuity_bootstrap(engine, arguments),
        "continuity_read_context" => continuity_read_context(engine, arguments),
        "continuity_recall" => continuity_recall(engine, arguments),
        "continuity_write_event" => continuity_write_event(engine, arguments),
        "continuity_write_item" => continuity_write_item(engine, arguments),
        "continuity_claim_work" => continuity_claim_work(engine, arguments),
        "continuity_upsert_agent_badge" => continuity_upsert_agent_badge(engine, arguments),
        "continuity_publish_coordination_signal" => {
            continuity_publish_coordination_signal(engine, arguments)
        }
        "continuity_publish_signal" => continuity_publish_signal(engine, arguments),
        "continuity_handoff" => continuity_handoff(engine, arguments),
        "continuity_snapshot" => continuity_snapshot(engine, arguments),
        "continuity_resume" => continuity_resume(engine, arguments),
        "continuity_explain" => continuity_explain(engine, arguments),
        "continuity_replay" => continuity_replay(engine, arguments),
        other => Err(anyhow!("unknown MCP tool: {other}")),
    }
}

fn continuity_identify_machine(engine: &Engine) -> Result<Value> {
    Ok(serde_json::to_value(engine.identify_machine()?)?)
}

fn continuity_bootstrap(engine: &Engine, arguments: Value) -> Result<Value> {
    let input: BootstrapInput =
        serde_json::from_value(arguments).context("parsing continuity_bootstrap arguments")?;
    let objective = input.objective.clone();
    let (namespace, task_id, machine) =
        resolve_machine_scope(engine, input.namespace.clone(), input.task_id.clone(), true)?;
    let attachment = engine.attach_agent(AttachAgentInput {
        agent_id: input.agent_id.clone(),
        agent_type: input.agent_type,
        capabilities: input.capabilities,
        namespace: namespace
            .clone()
            .ok_or_else(|| anyhow!("missing namespace after machine resolution"))?,
        role: input.role,
        metadata: input.metadata,
    })?;
    let context = engine.open_context(OpenContextInput {
        namespace: namespace
            .ok_or_else(|| anyhow!("missing namespace after machine resolution"))?,
        task_id: task_id.ok_or_else(|| anyhow!("missing task_id after machine resolution"))?,
        session_id: input.session_id,
        objective: input.objective.clone(),
        selector: input.selector.clone(),
        agent_id: Some(input.agent_id.clone()),
        attachment_id: Some(attachment.id.clone()),
    })?;
    let read = engine.read_context(ReadContextInput {
        context_id: Some(context.id.clone()),
        namespace: None,
        task_id: None,
        objective: objective.clone(),
        token_budget: input.token_budget,
        selector: input.selector,
        agent_id: Some(input.agent_id),
        session_id: Some(context.session_id.clone()),
        view_id: None,
        include_resolved: false,
        candidate_limit: input.candidate_limit,
    })?;
    let recall = engine.recall(crate::continuity::RecallInput {
        context_id: Some(context.id.clone()),
        namespace: None,
        task_id: None,
        objective,
        include_resolved: false,
        candidate_limit: input.candidate_limit.min(8).max(4),
    })?;
    Ok(json!({
        "machine": machine,
        "attachment": attachment,
        "context": context,
        "read": read,
        "recall": recall
    }))
}

fn continuity_read_context(engine: &Engine, arguments: Value) -> Result<Value> {
    let input: ReadToolInput =
        serde_json::from_value(arguments).context("parsing continuity_read_context arguments")?;
    let (namespace, task_id, machine) = if input.context_id.is_some() {
        (input.namespace, input.task_id, serde_json::Value::Null)
    } else {
        resolve_machine_scope(engine, input.namespace, input.task_id, true)?
    };
    let read = engine.read_context(ReadContextInput {
        context_id: input.context_id,
        namespace,
        task_id,
        objective: input.objective,
        token_budget: input.token_budget,
        selector: input.selector,
        agent_id: input.agent_id,
        session_id: input.session_id,
        view_id: input.view_id,
        include_resolved: input.include_resolved,
        candidate_limit: input.candidate_limit,
    })?;
    let mut value = serde_json::to_value(read)?;
    if !machine.is_null() {
        value
            .as_object_mut()
            .expect("context read should serialize as object")
            .insert("machine".to_string(), machine);
    }
    Ok(value)
}

fn continuity_recall(engine: &Engine, arguments: Value) -> Result<Value> {
    let input: RecallToolInput =
        serde_json::from_value(arguments).context("parsing continuity_recall arguments")?;
    let (namespace, task_id, machine) = if input.context_id.is_some() {
        (input.namespace, input.task_id, serde_json::Value::Null)
    } else {
        resolve_machine_scope(engine, input.namespace, input.task_id, true)?
    };
    let recall = engine.recall(crate::continuity::RecallInput {
        context_id: input.context_id,
        namespace,
        task_id,
        objective: input.objective,
        include_resolved: input.include_resolved,
        candidate_limit: input.candidate_limit,
    })?;
    let mut value = serde_json::to_value(recall)?;
    if !machine.is_null() {
        value
            .as_object_mut()
            .expect("continuity recall should serialize as object")
            .insert("machine".to_string(), machine);
    }
    Ok(value)
}

fn continuity_write_event(engine: &Engine, arguments: Value) -> Result<Value> {
    let input: EventToolInput =
        serde_json::from_value(arguments).context("parsing continuity_write_event arguments")?;
    let input = WriteEventInput {
        context_id: input.context_id,
        event: crate::model::EventInput {
            kind: parse_event_kind_alias(&input.kind)?,
            agent_id: input.agent_id,
            agent_role: input.agent_role,
            session_id: input.session_id,
            task_id: input.task_id,
            project_id: input.project_id,
            goal_id: input.goal_id,
            run_id: input.run_id,
            namespace: input.namespace,
            environment: input.environment,
            source: input.source,
            scope: parse_scope_alias(input.scope.as_deref()),
            tags: input.tags,
            dimensions: input.dimensions,
            content: input.content,
            attributes: input.attributes,
        },
    };
    let manifest = engine.write_events(vec![input])?;
    Ok(serde_json::to_value(
        manifest
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("no ingest manifest returned"))?,
    )?)
}

fn continuity_write_item(engine: &Engine, arguments: Value) -> Result<Value> {
    let input: ContinuityItemToolInput =
        serde_json::from_value(arguments).context("parsing continuity_write_item arguments")?;
    let input = ContinuityItemInput {
        context_id: input.context_id,
        author_agent_id: input.author_agent_id,
        kind: parse_continuity_kind_alias(&input.kind)?,
        title: input.title,
        body: input.body,
        scope: parse_scope_alias(input.scope.as_deref()),
        status: parse_continuity_status_alias(input.status.as_deref())?,
        importance: input.importance,
        confidence: input.confidence,
        salience: input.salience,
        layer: parse_memory_layer_alias(input.layer.as_deref())?,
        supports: input.supports,
        dimensions: input.dimensions,
        extra: input.extra,
    };
    let record = match input.kind {
        ContinuityKind::WorkClaim => {
            return Err(anyhow!(
                "work_claim requires continuity_claim_work so lease and resource coordination are preserved"
            ));
        }
        ContinuityKind::Decision => engine.mark_decision(input)?,
        ContinuityKind::Constraint => engine.mark_constraint(input)?,
        ContinuityKind::Hypothesis => engine.mark_hypothesis(input)?,
        ContinuityKind::Incident => engine.mark_incident(input)?,
        ContinuityKind::OperationalScar => engine.mark_operational_scar(input)?,
        _ => engine
            .write_derivations(vec![input])?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("no continuity item returned"))?,
    };
    Ok(serde_json::to_value(record)?)
}

fn continuity_claim_work(engine: &Engine, arguments: Value) -> Result<Value> {
    let input: ClaimWorkToolInput =
        serde_json::from_value(arguments).context("parsing continuity_claim_work arguments")?;
    Ok(serde_json::to_value(engine.claim_work(
        ClaimWorkInput {
            context_id: input.context_id,
            agent_id: input.agent_id,
            title: input.title,
            body: input.body,
            scope: input.scope,
            resources: input.resources,
            exclusive: input.exclusive,
            attachment_id: input.attachment_id,
            lease_seconds: input.lease_seconds,
            extra: input.extra,
        },
    )?)?)
}

fn continuity_upsert_agent_badge(engine: &Engine, arguments: Value) -> Result<Value> {
    let input: AgentBadgeToolInput = serde_json::from_value(arguments)
        .context("parsing continuity_upsert_agent_badge arguments")?;
    Ok(serde_json::to_value(engine.upsert_agent_badge(
        UpsertAgentBadgeInput {
            attachment_id: input.attachment_id,
            agent_id: input.agent_id,
            namespace: input.namespace,
            context_id: input.context_id,
            display_name: input.display_name,
            status: input.status,
            focus: input.focus,
            headline: input.headline,
            resource: input.resource,
            repo_root: input.repo_root,
            branch: input.branch,
            metadata: input.metadata,
        },
    )?)?)
}

fn continuity_publish_coordination_signal(engine: &Engine, arguments: Value) -> Result<Value> {
    let input: CoordinationSignalToolInput = serde_json::from_value(arguments)
        .context("parsing continuity_publish_coordination_signal arguments")?;
    Ok(serde_json::to_value(engine.publish_coordination_signal(
        CoordinationSignalInput {
            context_id: input.context_id,
            agent_id: input.agent_id,
            title: input.title,
            body: input.body,
            lane: input.lane,
            target_agent_id: input.target_agent_id,
            target_projected_lane: input.target_projected_lane,
            claim_id: input.claim_id,
            resource: input.resource,
            severity: input.severity,
            projection_ids: input.projection_ids,
            projected_lanes: input.projected_lanes,
            extra: input.extra,
        },
    )?)?)
}

fn continuity_publish_signal(engine: &Engine, arguments: Value) -> Result<Value> {
    let input: SignalInput =
        serde_json::from_value(arguments).context("parsing continuity_publish_signal arguments")?;
    Ok(serde_json::to_value(engine.publish_signal(input)?)?)
}

fn continuity_handoff(engine: &Engine, arguments: Value) -> Result<Value> {
    let input: HandoffToolInput =
        serde_json::from_value(arguments).context("parsing continuity_handoff arguments")?;
    let (namespace, task_id, machine) = if input.context_id.is_some() {
        (input.namespace, input.task_id, serde_json::Value::Null)
    } else {
        resolve_machine_scope(engine, input.namespace, input.task_id, true)?
    };
    let handoff = engine.handoff(ContinuityHandoffInput {
        from_agent_id: input.from_agent_id,
        to_agent_id: input.to_agent_id,
        context_id: input.context_id,
        namespace,
        task_id,
        objective: input.objective,
        reason: input.reason,
        selector: input.selector,
        resolution: input
            .resolution
            .unwrap_or_else(default_snapshot_resolution_value),
        token_budget: input.token_budget,
        candidate_limit: input.candidate_limit,
    })?;
    let mut value = serde_json::to_value(handoff)?;
    if !machine.is_null() {
        value
            .as_object_mut()
            .expect("handoff should serialize as object")
            .insert("machine".to_string(), machine);
    }
    Ok(value)
}

fn continuity_snapshot(engine: &Engine, arguments: Value) -> Result<Value> {
    let input: SnapshotToolInput =
        serde_json::from_value(arguments).context("parsing continuity_snapshot arguments")?;
    let (namespace, task_id, machine) = if input.context_id.is_some() {
        (input.namespace, input.task_id, serde_json::Value::Null)
    } else {
        resolve_machine_scope(engine, input.namespace, input.task_id, true)?
    };
    let snapshot = engine.snapshot(SnapshotInput {
        context_id: input.context_id,
        namespace,
        task_id,
        objective: input.objective,
        selector: input.selector,
        resolution: input
            .resolution
            .unwrap_or_else(default_snapshot_resolution_value),
        token_budget: input.token_budget,
        candidate_limit: input.candidate_limit,
        owner_agent_id: input.owner_agent_id,
    })?;
    let mut value = serde_json::to_value(snapshot)?;
    if !machine.is_null() {
        value
            .as_object_mut()
            .expect("snapshot should serialize as object")
            .insert("machine".to_string(), machine);
    }
    Ok(value)
}

fn continuity_resume(engine: &Engine, arguments: Value) -> Result<Value> {
    let input: ResumeToolInput =
        serde_json::from_value(arguments).context("parsing continuity_resume arguments")?;
    let (namespace, task_id, machine) = if input.context_id.is_some() || input.snapshot_id.is_some()
    {
        (input.namespace, input.task_id, serde_json::Value::Null)
    } else {
        resolve_machine_scope(engine, input.namespace, input.task_id, true)?
    };
    let resume = engine.resume(ResumeInput {
        snapshot_id: input.snapshot_id,
        context_id: input.context_id,
        namespace,
        task_id,
        objective: input.objective,
        token_budget: input.token_budget,
        candidate_limit: input.candidate_limit,
        agent_id: input.agent_id,
    })?;
    let mut value = serde_json::to_value(resume)?;
    if !machine.is_null() {
        value
            .as_object_mut()
            .expect("resume should serialize as object")
            .insert("machine".to_string(), machine);
    }
    Ok(value)
}

fn continuity_explain(engine: &Engine, arguments: Value) -> Result<Value> {
    let input: ExplainToolInput =
        serde_json::from_value(arguments).context("parsing continuity_explain arguments")?;
    let target = parse_explain_target_alias(&input.kind, input.id)?;
    engine.explain(target)
}

fn continuity_replay(engine: &Engine, arguments: Value) -> Result<Value> {
    let input: ReplayToolInput =
        serde_json::from_value(arguments).context("parsing continuity_replay arguments")?;
    Ok(serde_json::to_value(
        engine.replay_selector(input.selector, input.limit)?,
    )?)
}

fn success_response(id: Option<Value>, result: Value) -> Value {
    json!({
        "jsonrpc": JSON_RPC_VERSION,
        "id": id.unwrap_or(Value::Null),
        "result": result
    })
}

fn error_response(id: Option<Value>, code: i64, message: String, data: Option<Value>) -> Value {
    json!({
        "jsonrpc": JSON_RPC_VERSION,
        "id": id.unwrap_or(Value::Null),
        "error": {
            "code": code,
            "message": message,
            "data": data
        }
    })
}

fn tool_success(value: Value) -> Value {
    let text = serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string());
    json!({
        "content": [
            {
                "type": "text",
                "text": text
            }
        ],
        "structuredContent": value,
        "isError": false
    })
}

fn tool_error(error: anyhow::Error) -> Value {
    let text = error.to_string();
    json!({
        "content": [
            {
                "type": "text",
                "text": text
            }
        ],
        "isError": true
    })
}

fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "continuity_identify_machine",
            "description": "Return the canonical machine identity, machine namespace, and default organism task so a fresh head can discover the shared brain before attaching.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "continuity_bootstrap",
            "description": "Attach an agent, open the continuity context, and read the bounded context pack in one call. If namespace/task are omitted, the machine organism is used by default.",
            "inputSchema": {
                "type": "object",
                "required": ["agent_id", "agent_type", "session_id", "objective"],
                "properties": {
                    "agent_id": {"type": "string"},
                    "agent_type": {"type": "string"},
                    "namespace": {"type": "string", "default": MACHINE_NAMESPACE_ALIAS},
                    "task_id": {"type": "string"},
                    "session_id": {"type": "string"},
                    "objective": {"type": "string"},
                    "role": {"type": "string"},
                    "capabilities": {"type": "array", "items": {"type": "string"}},
                    "selector": {"type": "object"},
                    "metadata": {"type": "object"},
                    "token_budget": {"type": "integer", "minimum": 1, "default": DEFAULT_TOKEN_BUDGET},
                    "candidate_limit": {"type": "integer", "minimum": 1, "default": DEFAULT_CANDIDATE_LIMIT}
                }
            }
        }),
        json!({
            "name": "continuity_read_context",
            "description": "Read the current context pack, active constraints, scars, decisions, and open threads. If no context is specified, the machine organism is used by default.",
            "inputSchema": {
                "type": "object",
                "required": ["objective"],
                "properties": {
                    "context_id": {"type": "string"},
                    "namespace": {"type": "string"},
                    "task_id": {"type": "string"},
                    "objective": {"type": "string"},
                    "selector": {"type": "object"},
                    "agent_id": {"type": "string"},
                    "session_id": {"type": "string"},
                    "view_id": {"type": "string"},
                    "include_resolved": {"type": "boolean", "default": false},
                    "token_budget": {"type": "integer", "minimum": 1, "default": DEFAULT_TOKEN_BUDGET},
                    "candidate_limit": {"type": "integer", "minimum": 1, "default": DEFAULT_CANDIDATE_LIMIT}
                }
            }
        }),
        json!({
            "name": "continuity_recall",
            "description": "Run a fast bounded recall over high-signal continuity state such as scars, decisions, incidents, and hot working memory without building the full context pack.",
            "inputSchema": {
                "type": "object",
                "required": ["objective"],
                "properties": {
                    "context_id": {"type": "string"},
                    "namespace": {"type": "string"},
                    "task_id": {"type": "string"},
                    "objective": {"type": "string"},
                    "include_resolved": {"type": "boolean", "default": false},
                    "candidate_limit": {"type": "integer", "minimum": 1, "default": DEFAULT_CANDIDATE_LIMIT}
                }
            }
        }),
        json!({
            "name": "continuity_write_event",
            "description": "Append one raw event into the immutable journal with provenance such as prompts, tool calls, terminal I/O, or errors.",
            "inputSchema": {
                "type": "object",
                "required": ["kind", "agent_id", "session_id", "source", "content"],
                "properties": {
                    "context_id": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "description": "Event kind. Accepts canonical values like prompt/response/tool_call and operator aliases like observation or log."
                    },
                    "agent_id": {"type": "string"},
                    "agent_role": {"type": "string"},
                    "session_id": {"type": "string"},
                    "task_id": {"type": "string"},
                    "project_id": {"type": "string"},
                    "goal_id": {"type": "string"},
                    "run_id": {"type": "string"},
                    "namespace": {"type": "string"},
                    "environment": {"type": "string"},
                    "source": {"type": "string"},
                    "scope": {
                        "type": "string",
                        "default": "shared",
                        "description": "Logical memory scope. Prefer shared/project/session/agent/global. Unknown values fall back to shared."
                    },
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "dimensions": {"type": "array", "items": {"type": "object"}},
                    "content": {"type": "string"},
                    "attributes": {"type": "object"}
                }
            }
        }),
        json!({
            "name": "continuity_write_item",
            "description": "Write one typed continuity item such as a fact, decision, constraint, incident, lesson, or operational scar.",
            "inputSchema": {
                "type": "object",
                "required": ["context_id", "author_agent_id", "kind", "title", "body"],
                "properties": {
                    "context_id": {"type": "string"},
                    "author_agent_id": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "description": "Continuity kind. Accepts fact/decision/constraint/incident/lesson/operational_scar and the alias scar."
                    },
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "scope": {
                        "type": "string",
                        "default": "shared",
                        "description": "Logical memory scope. Prefer shared/project/session/agent/global. Unknown values fall back to shared."
                    },
                    "status": {"type": "string", "description": "Optional item lifecycle status such as open, active, resolved, superseded, or rejected."},
                    "importance": {"type": "number"},
                    "confidence": {"type": "number"},
                    "salience": {"type": "number"},
                    "layer": {"type": "string", "description": "Optional memory layer: hot, episodic, semantic, summary, or cold."},
                    "supports": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["support_type", "support_id"],
                            "properties": {
                                "support_type": {"type": "string"},
                                "support_id": {"type": "string"},
                                "reason": {"type": "string"},
                                "weight": {"type": "number"}
                            }
                        }
                    },
                    "dimensions": {"type": "array", "items": {"type": "object"}},
                    "extra": {"type": "object"}
                }
            }
        }),
        json!({
            "name": "continuity_claim_work",
            "description": "Claim a resource or work area through the organism so other agents can see ownership, back off, and detect conflicts without side channels.",
            "inputSchema": {
                "type": "object",
                "required": ["context_id", "agent_id", "title", "body", "scope"],
                "properties": {
                    "context_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "scope": {"type": "string"},
                    "resources": {"type": "array", "items": {"type": "string"}},
                    "exclusive": {"type": "boolean", "default": false},
                    "attachment_id": {"type": "string"},
                    "lease_seconds": {"type": "integer", "minimum": 1, "default": 180},
                    "extra": {"type": "object"}
                }
            }
        }),
        json!({
            "name": "continuity_upsert_agent_badge",
            "description": "Register or refresh a live attachment-linked agent badge so the machine organism and Grafana can see who is connected and what they are working on.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "attachment_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "namespace": {"type": "string"},
                    "context_id": {"type": "string"},
                    "display_name": {"type": "string"},
                    "status": {"type": "string"},
                    "focus": {"type": "string"},
                    "headline": {"type": "string"},
                    "resource": {"type": "string"},
                    "repo_root": {"type": "string"},
                    "branch": {"type": "string"},
                    "metadata": {"type": "object"}
                }
            }
        }),
        json!({
            "name": "continuity_publish_coordination_signal",
            "description": "Publish coordination anxiety, review, backoff, or coaching pressure linked to a claim so agents can coordinate through the organism itself.",
            "inputSchema": {
                "type": "object",
                "required": ["context_id", "agent_id", "title", "body", "lane"],
                "properties": {
                    "context_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "lane": {"type": "string"},
                    "target_agent_id": {"type": "string"},
                    "target_projected_lane": {
                        "type": "object",
                        "properties": {
                            "projection_id": {"type": "string"},
                            "projection_kind": {"type": "string"},
                            "label": {"type": "string"},
                            "resource": {"type": "string"},
                            "repo_root": {"type": "string"},
                            "branch": {"type": "string"},
                            "task_id": {"type": "string"}
                        }
                    },
                    "claim_id": {"type": "string"},
                    "resource": {"type": "string"},
                    "severity": {"type": "string"},
                    "projection_ids": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "projected_lanes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "projection_id": {"type": "string"},
                                "projection_kind": {"type": "string"},
                                "label": {"type": "string"},
                                "resource": {"type": "string"},
                                "repo_root": {"type": "string"},
                                "branch": {"type": "string"},
                                "task_id": {"type": "string"}
                            }
                        }
                    },
                    "extra": {"type": "object"}
                }
            }
        }),
        json!({
            "name": "continuity_publish_signal",
            "description": "Publish a live signal into the shared namespace so another attached agent can consume it through subscriptions or context reads.",
            "inputSchema": {
                "type": "object",
                "required": ["context_id", "agent_id", "title", "body"],
                "properties": {
                    "context_id": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "dimensions": {"type": "array", "items": {"type": "object"}},
                    "extra": {"type": "object"}
                }
            }
        }),
        json!({
            "name": "continuity_handoff",
            "description": "Create a provenance-backed handoff from one agent to another, including a snapshot and bounded context pack. If no context is specified, the machine organism is used by default.",
            "inputSchema": {
                "type": "object",
                "required": ["from_agent_id", "to_agent_id", "objective", "reason"],
                "properties": {
                    "from_agent_id": {"type": "string"},
                    "to_agent_id": {"type": "string"},
                    "context_id": {"type": "string"},
                    "namespace": {"type": "string"},
                    "task_id": {"type": "string"},
                    "objective": {"type": "string"},
                    "reason": {"type": "string"},
                    "selector": {"type": "object"},
                    "resolution": {"type": "string"},
                    "token_budget": {"type": "integer", "minimum": 1, "default": DEFAULT_TOKEN_BUDGET},
                    "candidate_limit": {"type": "integer", "minimum": 1, "default": DEFAULT_CANDIDATE_LIMIT}
                }
            }
        }),
        json!({
            "name": "continuity_snapshot",
            "description": "Capture a replayable snapshot of the current context state at bounded resolution. If no context is specified, the machine organism is used by default.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "context_id": {"type": "string"},
                    "namespace": {"type": "string"},
                    "task_id": {"type": "string"},
                    "objective": {"type": "string"},
                    "selector": {"type": "object"},
                    "resolution": {"type": "string"},
                    "token_budget": {"type": "integer", "minimum": 1, "default": DEFAULT_TOKEN_BUDGET},
                    "candidate_limit": {"type": "integer", "minimum": 1, "default": DEFAULT_CANDIDATE_LIMIT},
                    "owner_agent_id": {"type": "string"}
                }
            }
        }),
        json!({
            "name": "continuity_resume",
            "description": "Resume from a snapshot or directly from a namespace/task identity after interruption, swap, or crash. If no context is specified, the machine organism is used by default.",
            "inputSchema": {
                "type": "object",
                "required": ["objective"],
                "properties": {
                    "snapshot_id": {"type": "string"},
                    "context_id": {"type": "string"},
                    "namespace": {"type": "string"},
                    "task_id": {"type": "string"},
                    "objective": {"type": "string"},
                    "agent_id": {"type": "string"},
                    "token_budget": {"type": "integer", "minimum": 1, "default": DEFAULT_TOKEN_BUDGET},
                    "candidate_limit": {"type": "integer", "minimum": 1, "default": DEFAULT_CANDIDATE_LIMIT}
                }
            }
        }),
        json!({
            "name": "continuity_explain",
            "description": "Explain why a context, snapshot, handoff, pack, or continuity item exists and what supports it.",
            "inputSchema": {
                "type": "object",
                "required": ["kind", "id"],
                "properties": {
                    "kind": {
                        "type": "string",
                        "description": "Explain target kind. Accepts context, item, continuity_item, snapshot, handoff, pack, or view."
                    },
                    "id": {"type": "string"}
                }
            }
        }),
        json!({
            "name": "continuity_replay",
            "description": "Replay raw journal rows by selector when you need the event trail instead of the compressed context pack.",
            "inputSchema": {
                "type": "object",
                "required": ["selector"],
                "properties": {
                    "selector": {"type": "object"},
                    "limit": {"type": "integer", "minimum": 1, "default": DEFAULT_REPLAY_LIMIT}
                }
            }
        }),
    ]
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn tool_definitions_include_bootstrap_and_handoff() {
        let names = tool_definitions()
            .into_iter()
            .filter_map(|tool| tool.get("name").and_then(Value::as_str).map(str::to_string))
            .collect::<Vec<_>>();
        assert!(names.contains(&"continuity_identify_machine".to_string()));
        assert!(names.contains(&"continuity_bootstrap".to_string()));
        assert!(names.contains(&"continuity_handoff".to_string()));
        assert!(names.contains(&"continuity_claim_work".to_string()));
        assert!(names.contains(&"continuity_recall".to_string()));
        assert!(names.contains(&"continuity_upsert_agent_badge".to_string()));
        assert!(names.contains(&"continuity_publish_coordination_signal".to_string()));
    }

    #[test]
    fn identify_machine_tool_returns_canonical_namespace() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let machine = dispatch_tool(&engine, "continuity_identify_machine", json!({})).unwrap();
        assert!(
            machine["namespace"]
                .as_str()
                .unwrap()
                .starts_with("machine:")
        );
        assert_eq!(
            machine["default_task_id"].as_str(),
            Some(crate::continuity::DEFAULT_MACHINE_TASK_ID)
        );
    }

    #[test]
    fn bootstrap_tool_attaches_agent_and_reads_context() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let value = dispatch_tool(
            &engine,
            "continuity_bootstrap",
            json!({
                "agent_id": "codex-a",
                "agent_type": "codex",
                "namespace": "demo",
                "task_id": "swap-proof",
                "session_id": "session-1",
                "objective": "Resume the same task after an agent swap",
                "role": "coder",
                "capabilities": ["read", "write"]
            }),
        )
        .unwrap();

        assert_eq!(value["attachment"]["input"]["agent_id"], "codex-a");
        assert_eq!(value["context"]["namespace"], "demo");
        assert_eq!(value["read"]["context"]["task_id"], "swap-proof");
        assert!(value["recall"]["summary"].is_string());
    }

    #[test]
    fn recall_tool_surfaces_high_signal_scars() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let value = dispatch_tool(
            &engine,
            "continuity_bootstrap",
            json!({
                "agent_id": "codex-a",
                "agent_type": "codex",
                "namespace": "demo",
                "task_id": "recall-proof",
                "session_id": "session-1",
                "objective": "keep trauma searchable"
            }),
        )
        .unwrap();
        let context_id = value["context"]["id"].as_str().unwrap();
        engine
            .mark_operational_scar(crate::continuity::ContinuityItemInput {
                context_id: context_id.to_string(),
                author_agent_id: "debugger".into(),
                kind: crate::continuity::ContinuityKind::OperationalScar,
                title: "Hot reload is worthless if it forks the brain.".into(),
                body: "Split runtime roots make the dashboard lie.".into(),
                scope: crate::model::Scope::Project,
                status: Some(crate::continuity::ContinuityStatus::Open),
                importance: Some(0.98),
                confidence: Some(0.9),
                salience: Some(0.99),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: json!({}),
            })
            .unwrap();
        let recall = dispatch_tool(
            &engine,
            "continuity_recall",
            json!({
                "context_id": context_id,
                "objective": "Which trauma matters when observability lies?"
            }),
        )
        .unwrap();
        assert_eq!(
            recall["items"][0]["kind"].as_str(),
            Some("operational_scar")
        );
        assert!(recall["answer_hint"].is_string());
        assert!(
            recall["summary"]
                .as_str()
                .unwrap()
                .contains("operational_scar")
        );
    }

    #[test]
    fn bootstrap_defaults_to_machine_scope_when_omitted() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let machine = engine.identify_machine().unwrap();
        let value = dispatch_tool(
            &engine,
            "continuity_bootstrap",
            json!({
                "agent_id": "codex-b",
                "agent_type": "codex",
                "session_id": "session-2",
                "objective": "Wake up inside the machine organism"
            }),
        )
        .unwrap();

        assert_eq!(value["context"]["namespace"], machine.namespace);
        assert_eq!(value["context"]["task_id"], machine.default_task_id);
    }

    #[test]
    fn write_item_tool_accepts_namespace_like_scope_aliases() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let bootstrap = dispatch_tool(
            &engine,
            "continuity_bootstrap",
            json!({
                "agent_id": "claude-code-test",
                "agent_type": "claude",
                "namespace": "amendments-squad",
                "task_id": "incident-inc-1400",
                "session_id": "session-1",
                "objective": "Track the live amendments incident"
            }),
        )
        .unwrap();
        let context_id = bootstrap["context"]["id"].as_str().unwrap();

        let item = dispatch_tool(
            &engine,
            "continuity_write_item",
            json!({
                "context_id": context_id,
                "author_agent_id": "claude-code-test",
                "kind": "fact",
                "title": "CreatePreQuotedAmendmentTicket is dead code in amendment-service",
                "body": "CreatePreQuotedAmendmentTicket has zero accept_quote metrics in production.",
                "scope": "amendments-squad",
                "importance": 0.5,
                "confidence": 0.95
            }),
        )
        .unwrap();

        assert_eq!(item["kind"], "fact");
        assert_eq!(item["scope"], "shared");
    }

    #[test]
    fn write_event_tool_accepts_observation_kind_and_namespace_like_scope() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();

        let event = dispatch_tool(
            &engine,
            "continuity_write_event",
            json!({
                "kind": "observation",
                "agent_id": "claude-code-test",
                "session_id": "amendment-service-2026-03-23",
                "source": "slack-incident-channel",
                "scope": "amendments-squad",
                "content": "INC-1400 incident active. 73 spurious tickets from legacy cloning bug."
            }),
        )
        .unwrap();

        assert_eq!(event["event"]["kind"], "note");
        assert_eq!(event["event"]["scope"], "shared");
    }

    #[test]
    fn explain_tool_accepts_item_alias() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let bootstrap = dispatch_tool(
            &engine,
            "continuity_bootstrap",
            json!({
                "agent_id": "claude-code-test",
                "agent_type": "claude",
                "namespace": "amendments-squad",
                "task_id": "incident-inc-1400",
                "session_id": "session-1",
                "objective": "Track the live amendments incident"
            }),
        )
        .unwrap();
        let context_id = bootstrap["context"]["id"].as_str().unwrap();

        let item = dispatch_tool(
            &engine,
            "continuity_write_item",
            json!({
                "context_id": context_id,
                "author_agent_id": "claude-code-test",
                "kind": "incident",
                "title": "INC-1400: Spike in cloned add-pax tickets",
                "body": "73 spurious Async tickets appeared in managing_backoffice.",
                "scope": "amendments-squad"
            }),
        )
        .unwrap();

        let explained = dispatch_tool(
            &engine,
            "continuity_explain",
            json!({
                "kind": "item",
                "id": item["id"].as_str().unwrap()
            }),
        )
        .unwrap();

        assert_eq!(explained["item"]["id"], item["id"]);
    }

    #[test]
    fn claim_work_tool_persists_visible_claim() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let bootstrap = dispatch_tool(
            &engine,
            "continuity_bootstrap",
            json!({
                "agent_id": "planner-a",
                "agent_type": "codex",
                "namespace": "demo",
                "task_id": "claim-work",
                "session_id": "session-1",
                "objective": "Coordinate ownership through the organism"
            }),
        )
        .unwrap();
        let context_id = bootstrap["context"]["id"].as_str().unwrap();
        let claim = dispatch_tool(
            &engine,
            "continuity_claim_work",
            json!({
                "context_id": context_id,
                "agent_id": "planner-a",
                "title": "Own the kernel coordination slice",
                "body": "I am rewriting the continuity core. Back off this scope until I publish a handoff.",
                "scope": "project",
                "resources": ["src/continuity.rs"],
                "exclusive": true
            }),
        )
        .unwrap();

        assert_eq!(claim["kind"], "work_claim");
        assert_eq!(claim["author_agent_id"], "planner-a");

        let read = dispatch_tool(
            &engine,
            "continuity_read_context",
            json!({
                "context_id": context_id,
                "objective": "Resume the same work"
            }),
        )
        .unwrap();
        assert_eq!(read["work_claims"].as_array().map(Vec::len), Some(1));
    }

    #[test]
    fn upsert_agent_badge_tool_surfaces_visible_badge() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let bootstrap = dispatch_tool(
            &engine,
            "continuity_bootstrap",
            json!({
                "agent_id": "planner-a",
                "agent_type": "codex",
                "namespace": "demo",
                "task_id": "badge-work",
                "session_id": "session-1",
                "objective": "Show badge state in the organism"
            }),
        )
        .unwrap();
        let attachment_id = bootstrap["attachment"]["id"].as_str().unwrap();
        let context_id = bootstrap["context"]["id"].as_str().unwrap();

        let badge = dispatch_tool(
            &engine,
            "continuity_upsert_agent_badge",
            json!({
                "attachment_id": attachment_id,
                "context_id": context_id,
                "display_name": "Planner A",
                "status": "writing",
                "focus": "rewiring src/storage.rs",
                "headline": "real-time badge lane",
                "resource": "src/storage.rs"
            }),
        )
        .unwrap();

        assert_eq!(badge["display_name"], "Planner A");
        let read = dispatch_tool(
            &engine,
            "continuity_read_context",
            json!({
                "context_id": context_id,
                "objective": "inspect badge state"
            }),
        )
        .unwrap();
        assert_eq!(
            read["organism"]["agent_badges"][0]["focus"].as_str(),
            Some("rewiring src/storage.rs")
        );
    }

    #[test]
    fn coordination_signal_tool_persists_anxiety_pressure() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let bootstrap = dispatch_tool(
            &engine,
            "continuity_bootstrap",
            json!({
                "agent_id": "planner-a",
                "agent_type": "codex",
                "namespace": "demo",
                "task_id": "coordination-signal",
                "session_id": "session-1",
                "objective": "Let the organism carry review anxiety"
            }),
        )
        .unwrap();
        let context_id = bootstrap["context"]["id"].as_str().unwrap();
        let claim = dispatch_tool(
            &engine,
            "continuity_claim_work",
            json!({
                "context_id": context_id,
                "agent_id": "planner-a",
                "title": "Own src/storage.rs",
                "body": "Metrics surgery in progress.",
                "scope": "project",
                "resources": ["src/storage.rs"],
                "exclusive": true
            }),
        )
        .unwrap();
        let claim_id = claim["id"].as_str().unwrap();

        let signal = dispatch_tool(
            &engine,
            "continuity_publish_coordination_signal",
            json!({
                "context_id": context_id,
                "agent_id": "therapist",
                "title": "Anxiety spike over storage surgery",
                "body": "Watch metrics cardinality before this lands.",
                "lane": "anxiety",
                "target_agent_id": "planner-a",
                "claim_id": claim_id,
                "resource": "src/storage.rs"
            }),
        )
        .unwrap();

        assert_eq!(signal["kind"], "signal");
        let read = dispatch_tool(
            &engine,
            "continuity_read_context",
            json!({
                "context_id": context_id,
                "objective": "Inspect live anxiety"
            }),
        )
        .unwrap();
        assert_eq!(read["organism"]["anxiety_signal_count"].as_u64(), Some(1));
        assert_eq!(
            read["coordination_signals"].as_array().map(Vec::len),
            Some(1)
        );
    }
}
