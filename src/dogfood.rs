use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::time::{Duration, sleep};

use crate::Engine;
use crate::continuity::{
    AttachAgentInput, ClaimWorkInput, ContinuityItemRecord, ContinuityStatus, CoordinationLane,
    CoordinationProjectedLane, CoordinationSignalInput, CoordinationSignalRecord, HeartbeatInput,
    OpenContextInput, ReadContextInput, ResolveOrSupersedeInput, UnifiedContinuityInterface,
    UpsertAgentBadgeInput, coordination_signal,
};
use crate::model::Scope;

const ORGANISM_SOURCE: &str = "dogfood_organism_choir";
const DEFAULT_ORGANISM_LEASE_SECS: u64 = 180;
const DEFAULT_ORGANISM_TOKEN_BUDGET: usize = 256;
const DEFAULT_ORGANISM_CANDIDATE_LIMIT: usize = 24;

#[derive(Debug, Clone)]
pub struct OrganismChorusConfig {
    pub namespace: String,
    pub task_id: String,
    pub objective: String,
    pub session_id: String,
    pub pulse_secs: Option<u64>,
    pub pulse_count: Option<u64>,
    pub lease_secs: u64,
    pub token_budget: usize,
    pub candidate_limit: usize,
}

impl OrganismChorusConfig {
    pub fn with_defaults(
        namespace: String,
        task_id: String,
        objective: String,
        session_id: String,
    ) -> Self {
        Self {
            namespace,
            task_id,
            objective,
            session_id,
            pulse_secs: None,
            pulse_count: None,
            lease_secs: DEFAULT_ORGANISM_LEASE_SECS,
            token_budget: DEFAULT_ORGANISM_TOKEN_BUDGET,
            candidate_limit: DEFAULT_ORGANISM_CANDIDATE_LIMIT,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct OrganismAgentStatus {
    pub agent_id: String,
    pub role: String,
    pub attachment_id: String,
    pub context_id: String,
    pub claim_id: String,
    pub tick_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct OrganismChorusStatus {
    pub context_id: String,
    pub namespace: String,
    pub task_id: String,
    pub iteration: u64,
    pub agents: Vec<OrganismAgentStatus>,
    pub published_signal_ids: Vec<String>,
    pub resolved_signal_ids: Vec<String>,
    pub organism: serde_json::Value,
}

#[derive(Debug, Clone)]
struct OrganismAgentSpec {
    agent_id: &'static str,
    role: &'static str,
    claim_title: &'static str,
    claim_body: &'static str,
    resource: &'static str,
    capabilities: &'static [&'static str],
}

#[derive(Debug, Clone)]
struct OrganismAgentRuntime {
    spec: OrganismAgentSpec,
    attachment_id: String,
    context_id: String,
    claim_id: String,
    tick_count: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct ClaimConflictSummary {
    resource: String,
    #[serde(default)]
    claim_ids: Vec<String>,
    #[serde(default)]
    agents: Vec<String>,
    #[serde(default)]
    projection_ids: Vec<String>,
    #[serde(default)]
    projected_lanes: Vec<ClaimConflictLaneSummary>,
}

#[derive(Debug, Clone, Deserialize)]
struct ClaimConflictLaneSummary {
    claim_id: String,
    agent_id: String,
    display_name: Option<String>,
    projection_id: String,
    projection_kind: String,
    label: String,
    resource: Option<String>,
    repo_root: Option<String>,
    branch: Option<String>,
    task_id: Option<String>,
}

#[derive(Debug, Clone)]
struct ConflictTarget {
    agent_id: String,
    display_name: Option<String>,
    claim_id: Option<String>,
    projection_id: Option<String>,
    lane_label: Option<String>,
    projected_lane: Option<CoordinationProjectedLane>,
}

fn badge_status_for_role(role: &str) -> &'static str {
    match role {
        "anxiety" => "watching_conflicts",
        "boundary" => "guarding_boundaries",
        "review" => "triaging_reviews",
        "therapist" => "cooling_pressure",
        "scar_curator" => "curating_scars",
        _ => "attached",
    }
}

pub async fn run_organism_choir(
    engine: Arc<Engine>,
    config: OrganismChorusConfig,
) -> Result<serde_json::Value> {
    let mut runtime = bootstrap_organism_choir(engine.as_ref(), &config)?;
    let mut iteration = 0_u64;
    let pulse_limit = config.pulse_count.filter(|count| *count > 0);
    let pulse_secs = config.pulse_secs.filter(|secs| *secs > 0);

    loop {
        iteration += 1;
        let status = pulse_organism_choir(engine.as_ref(), &config, &mut runtime, iteration)?;
        let value = serde_json::to_value(&status)?;
        let should_continue = pulse_limit
            .map(|limit| iteration < limit)
            .unwrap_or_else(|| pulse_secs.is_some());
        if !should_continue {
            return Ok(value);
        }
        if let Some(pulse_secs) = pulse_secs {
            println!("{}", serde_json::to_string(&value)?);
            sleep(Duration::from_secs(pulse_secs)).await;
        }
    }
}

fn bootstrap_organism_choir(
    engine: &Engine,
    config: &OrganismChorusConfig,
) -> Result<Vec<OrganismAgentRuntime>> {
    let mut runtime = Vec::new();

    for spec in default_organism_specs() {
        let attachment = engine.attach_agent(AttachAgentInput {
            agent_id: spec.agent_id.to_string(),
            agent_type: "state_agent".to_string(),
            capabilities: spec
                .capabilities
                .iter()
                .map(|item| item.to_string())
                .collect(),
            namespace: config.namespace.clone(),
            role: Some(spec.role.to_string()),
            metadata: serde_json::json!({
                "source": ORGANISM_SOURCE,
                "role": spec.role,
                "resource": spec.resource,
            }),
        })?;
        let context = engine.open_context(OpenContextInput {
            namespace: config.namespace.clone(),
            task_id: config.task_id.clone(),
            session_id: format!("{}::{}", config.session_id, spec.agent_id),
            objective: config.objective.clone(),
            selector: None,
            agent_id: Some(spec.agent_id.to_string()),
            attachment_id: Some(attachment.id.clone()),
        })?;
        let claim = engine.claim_work(ClaimWorkInput {
            context_id: context.id.clone(),
            agent_id: spec.agent_id.to_string(),
            title: spec.claim_title.to_string(),
            body: spec.claim_body.to_string(),
            scope: Scope::Shared,
            resources: vec![spec.resource.to_string()],
            exclusive: true,
            attachment_id: Some(attachment.id.clone()),
            lease_seconds: Some(config.lease_secs),
            extra: serde_json::json!({
                "source": ORGANISM_SOURCE,
                "role": spec.role,
                "resource": spec.resource,
            }),
        })?;
        engine.upsert_agent_badge(UpsertAgentBadgeInput {
            attachment_id: Some(attachment.id.clone()),
            agent_id: None,
            namespace: None,
            context_id: Some(context.id.clone()),
            display_name: Some(spec.agent_id.to_string()),
            status: Some(badge_status_for_role(spec.role).to_string()),
            focus: Some(spec.claim_title.to_string()),
            headline: Some(spec.claim_body.to_string()),
            resource: Some(spec.resource.to_string()),
            repo_root: None,
            branch: None,
            metadata: serde_json::json!({
                "source": ORGANISM_SOURCE,
                "role": spec.role,
                "resource": spec.resource,
            }),
        })?;
        runtime.push(OrganismAgentRuntime {
            spec,
            attachment_id: attachment.id,
            context_id: context.id,
            claim_id: claim.id,
            tick_count: attachment.tick_count,
        });
    }

    Ok(runtime)
}

fn pulse_organism_choir(
    engine: &Engine,
    config: &OrganismChorusConfig,
    runtime: &mut [OrganismAgentRuntime],
    iteration: u64,
) -> Result<OrganismChorusStatus> {
    for agent in runtime.iter_mut() {
        let attachment = engine.heartbeat(HeartbeatInput {
            attachment_id: Some(agent.attachment_id.clone()),
            agent_id: None,
            namespace: None,
            context_id: Some(agent.context_id.clone()),
        })?;
        agent.tick_count = attachment.tick_count;
        if let Some(context_id) = attachment.context_id {
            agent.context_id = context_id;
        }
        engine.upsert_agent_badge(UpsertAgentBadgeInput {
            attachment_id: Some(agent.attachment_id.clone()),
            agent_id: None,
            namespace: None,
            context_id: Some(agent.context_id.clone()),
            display_name: Some(agent.spec.agent_id.to_string()),
            status: Some(badge_status_for_role(agent.spec.role).to_string()),
            focus: Some(agent.spec.claim_title.to_string()),
            headline: Some(agent.spec.claim_body.to_string()),
            resource: Some(agent.spec.resource.to_string()),
            repo_root: None,
            branch: None,
            metadata: serde_json::json!({
                "source": ORGANISM_SOURCE,
                "role": agent.spec.role,
                "resource": agent.spec.resource,
                "iteration": iteration,
            }),
        })?;
    }

    let context_id = runtime
        .first()
        .map(|agent| agent.context_id.clone())
        .unwrap_or_default();
    let before = read_shared_context(engine, config, &context_id)?;

    let mut published_signal_ids = Vec::new();
    let mut resolved_signal_ids = Vec::new();

    if let Some(anxiety_agent) = runtime.iter().find(|agent| agent.spec.role == "anxiety") {
        published_signal_ids.extend(publish_conflict_signals(
            engine,
            &before,
            anxiety_agent,
            CoordinationLane::Anxiety,
        )?);
    }
    if let Some(boundary_agent) = runtime.iter().find(|agent| agent.spec.role == "boundary") {
        published_signal_ids.extend(publish_conflict_signals(
            engine,
            &before,
            boundary_agent,
            CoordinationLane::Backoff,
        )?);
        resolved_signal_ids.extend(resolve_cleared_signals(
            engine,
            &before,
            boundary_agent.spec.agent_id,
            boundary_agent.spec.agent_id,
            boundary_agent.spec.role,
            CoordinationLane::Backoff,
        )?);
    }
    if let Some(review_agent) = runtime.iter().find(|agent| agent.spec.role == "review") {
        published_signal_ids.extend(publish_persistent_review_signals(
            engine,
            &before,
            review_agent,
        )?);
        resolved_signal_ids.extend(resolve_cleared_signals(
            engine,
            &before,
            review_agent.spec.agent_id,
            review_agent.spec.agent_id,
            review_agent.spec.role,
            CoordinationLane::Review,
        )?);
    }
    if let Some(therapist) = runtime.iter().find(|agent| agent.spec.role == "therapist") {
        resolved_signal_ids.extend(resolve_cleared_signals(
            engine,
            &before,
            therapist.spec.agent_id,
            "anxiety-sentinel",
            therapist.spec.role,
            CoordinationLane::Anxiety,
        )?);
    }
    if let Some(curator) = runtime
        .iter()
        .find(|agent| agent.spec.role == "scar_curator")
    {
        published_signal_ids.extend(sync_scar_coaching(engine, &before, curator)?);
        resolved_signal_ids.extend(resolve_scar_coaching_if_quiet(engine, &before, curator)?);
    }

    let after = read_shared_context(engine, config, &context_id)?;
    Ok(OrganismChorusStatus {
        context_id,
        namespace: config.namespace.clone(),
        task_id: config.task_id.clone(),
        iteration,
        agents: runtime
            .iter()
            .map(|agent| OrganismAgentStatus {
                agent_id: agent.spec.agent_id.to_string(),
                role: agent.spec.role.to_string(),
                attachment_id: agent.attachment_id.clone(),
                context_id: agent.context_id.clone(),
                claim_id: agent.claim_id.clone(),
                tick_count: agent.tick_count,
            })
            .collect(),
        published_signal_ids,
        resolved_signal_ids,
        organism: after.organism,
    })
}

fn read_shared_context(
    engine: &Engine,
    config: &OrganismChorusConfig,
    context_id: &str,
) -> Result<crate::continuity::ContextRead> {
    engine.read_context(ReadContextInput {
        context_id: Some(context_id.to_string()),
        namespace: Some(config.namespace.clone()),
        task_id: Some(config.task_id.clone()),
        objective: config.objective.clone(),
        token_budget: config.token_budget,
        selector: None,
        agent_id: Some("boundary-warden".to_string()),
        session_id: Some(config.session_id.clone()),
        view_id: None,
        include_resolved: true,
        candidate_limit: config.candidate_limit,
    })
}

fn publish_conflict_signals(
    engine: &Engine,
    read: &crate::continuity::ContextRead,
    runtime: &OrganismAgentRuntime,
    lane: CoordinationLane,
) -> Result<Vec<String>> {
    let mut published = Vec::new();
    for conflict in claim_conflicts(read) {
        for target in conflict_targets(&conflict) {
            if has_source_signal(
                read,
                runtime.spec.agent_id,
                lane,
                Some(conflict.resource.as_str()),
                Some(target.agent_id.as_str()),
                target.projected_lane.as_ref(),
            ) {
                continue;
            }
            let signal = engine.publish_coordination_signal(CoordinationSignalInput {
                context_id: runtime.context_id.clone(),
                agent_id: runtime.spec.agent_id.to_string(),
                title: conflict_signal_title(lane, &conflict.resource, &target),
                body: conflict_signal_body(lane, &conflict, &target),
                lane,
                target_agent_id: Some(target.agent_id.clone()),
                target_projected_lane: target.projected_lane.clone(),
                claim_id: target
                    .claim_id
                    .clone()
                    .or_else(|| conflict.claim_ids.first().cloned()),
                resource: Some(conflict.resource.clone()),
                severity: None,
                projection_ids: target
                    .projection_id
                    .clone()
                    .into_iter()
                    .chain(conflict.projection_ids.clone().into_iter())
                    .fold(Vec::<String>::new(), |mut ids, projection_id| {
                        if !ids.iter().any(|existing| existing == &projection_id) {
                            ids.push(projection_id);
                        }
                        ids
                    }),
                projected_lanes: projected_lanes_for_signal(&conflict),
                extra: serde_json::json!({
                    "source": ORGANISM_SOURCE,
                    "role": runtime.spec.role,
                    "policy": "claim_conflict",
                    "target_lane_label": target.lane_label,
                }),
            })?;
            published.push(signal.id);
        }
    }
    Ok(published)
}

fn publish_persistent_review_signals(
    engine: &Engine,
    read: &crate::continuity::ContextRead,
    runtime: &OrganismAgentRuntime,
) -> Result<Vec<String>> {
    let mut published = Vec::new();
    for conflict in claim_conflicts(read) {
        for target in conflict_targets(&conflict) {
            if !has_source_signal(
                read,
                "boundary-warden",
                CoordinationLane::Backoff,
                Some(conflict.resource.as_str()),
                Some(target.agent_id.as_str()),
                target.projected_lane.as_ref(),
            ) {
                continue;
            }
            if has_source_signal(
                read,
                runtime.spec.agent_id,
                CoordinationLane::Review,
                Some(conflict.resource.as_str()),
                Some(target.agent_id.as_str()),
                target.projected_lane.as_ref(),
            ) {
                continue;
            }
            let signal = engine.publish_coordination_signal(CoordinationSignalInput {
                context_id: runtime.context_id.clone(),
                agent_id: runtime.spec.agent_id.to_string(),
                title: conflict_signal_title(CoordinationLane::Review, &conflict.resource, &target),
                body: conflict_signal_body(CoordinationLane::Review, &conflict, &target),
                lane: CoordinationLane::Review,
                target_agent_id: Some(target.agent_id.clone()),
                target_projected_lane: target.projected_lane.clone(),
                claim_id: target
                    .claim_id
                    .clone()
                    .or_else(|| conflict.claim_ids.first().cloned()),
                resource: Some(conflict.resource.clone()),
                severity: None,
                projection_ids: target
                    .projection_id
                    .clone()
                    .into_iter()
                    .chain(conflict.projection_ids.clone().into_iter())
                    .fold(Vec::<String>::new(), |mut ids, projection_id| {
                        if !ids.iter().any(|existing| existing == &projection_id) {
                            ids.push(projection_id);
                        }
                        ids
                    }),
                projected_lanes: projected_lanes_for_signal(&conflict),
                extra: serde_json::json!({
                    "source": ORGANISM_SOURCE,
                    "role": runtime.spec.role,
                    "policy": "persistent_claim_conflict",
                    "target_lane_label": target.lane_label,
                }),
            })?;
            published.push(signal.id);
        }
    }
    Ok(published)
}

fn resolve_cleared_signals(
    engine: &Engine,
    read: &crate::continuity::ContextRead,
    actor_agent_id: &str,
    target_author_agent_id: &str,
    actor_role: &str,
    lane: CoordinationLane,
) -> Result<Vec<String>> {
    let active_conflicts = claim_conflicts(read);
    let mut resolved = Vec::new();
    for item in read.coordination_signals.iter().filter(|item| {
        item.author_agent_id == target_author_agent_id
            && source_is(item, ORGANISM_SOURCE)
            && coordination_signal(item)
                .map(|signal| signal.lane == lane.as_str())
                .unwrap_or(false)
    }) {
        let Some(signal) = coordination_signal(item) else {
            continue;
        };
        if signal_still_matches_active_conflict(&signal, &active_conflicts) {
            continue;
        }
        engine.resolve_or_supersede(ResolveOrSupersedeInput {
            continuity_id: item.id.clone(),
            actor_agent_id: actor_agent_id.to_string(),
            new_status: ContinuityStatus::Resolved,
            supersedes_id: None,
            resolution_note: Some("claim conflict pressure cooled off".to_string()),
            extra: serde_json::json!({
                "source": ORGANISM_SOURCE,
                "role": actor_role,
                "policy": "claim_conflict",
            }),
        })?;
        resolved.push(item.id.clone());
    }
    Ok(resolved)
}

fn sync_scar_coaching(
    engine: &Engine,
    read: &crate::continuity::ContextRead,
    runtime: &OrganismAgentRuntime,
) -> Result<Vec<String>> {
    let active_scars = read
        .operational_scars
        .iter()
        .filter(|item| item.status.is_open())
        .collect::<Vec<_>>();
    if active_scars.is_empty() {
        return Ok(Vec::new());
    }
    let title = format!("Scar pressure: {} active scars", active_scars.len());
    let body = format!(
        "Operational scar pressure remains active: {}.",
        active_scars
            .iter()
            .take(3)
            .map(|item| item.title.clone())
            .collect::<Vec<_>>()
            .join(" | ")
    );
    let existing = active_source_signal(
        read,
        runtime.spec.agent_id,
        CoordinationLane::Coach,
        None,
        None,
        None,
    );
    if existing
        .map(|item| item.title == title && item.body == body)
        .unwrap_or(false)
    {
        return Ok(Vec::new());
    }
    let mut published = Vec::new();
    let signal = engine.publish_coordination_signal(CoordinationSignalInput {
        context_id: runtime.context_id.clone(),
        agent_id: runtime.spec.agent_id.to_string(),
        title,
        body,
        lane: CoordinationLane::Coach,
        target_agent_id: None,
        target_projected_lane: None,
        claim_id: None,
        resource: None,
        severity: None,
        projection_ids: Vec::new(),
        projected_lanes: Vec::new(),
        extra: serde_json::json!({
            "source": ORGANISM_SOURCE,
            "role": runtime.spec.role,
            "policy": "scar_pressure",
        }),
    })?;
    published.push(signal.id);
    Ok(published)
}

fn resolve_scar_coaching_if_quiet(
    engine: &Engine,
    read: &crate::continuity::ContextRead,
    runtime: &OrganismAgentRuntime,
) -> Result<Vec<String>> {
    let active_scars = read
        .operational_scars
        .iter()
        .filter(|item| item.status.is_open())
        .count();
    if active_scars > 0 {
        return Ok(Vec::new());
    }

    let mut resolved = Vec::new();
    for item in read.coordination_signals.iter().filter(|item| {
        item.author_agent_id == runtime.spec.agent_id
            && source_is(item, ORGANISM_SOURCE)
            && coordination_signal(item)
                .map(|signal| signal.lane == CoordinationLane::Coach.as_str())
                .unwrap_or(false)
    }) {
        engine.resolve_or_supersede(ResolveOrSupersedeInput {
            continuity_id: item.id.clone(),
            actor_agent_id: runtime.spec.agent_id.to_string(),
            new_status: ContinuityStatus::Resolved,
            supersedes_id: None,
            resolution_note: Some("scar pressure is quiet right now".to_string()),
            extra: serde_json::json!({
                "source": ORGANISM_SOURCE,
                "role": runtime.spec.role,
                "policy": "scar_pressure",
            }),
        })?;
        resolved.push(item.id.clone());
    }
    Ok(resolved)
}

fn active_source_signal<'a>(
    read: &'a crate::continuity::ContextRead,
    author_agent_id: &str,
    lane: CoordinationLane,
    resource: Option<&str>,
    target_agent_id: Option<&str>,
    target_projected_lane: Option<&CoordinationProjectedLane>,
) -> Option<&'a ContinuityItemRecord> {
    read.coordination_signals.iter().find(|item| {
        item.author_agent_id == author_agent_id
            && source_is(item, ORGANISM_SOURCE)
            && coordination_signal(item)
                .map(|signal| {
                    signal.lane == lane.as_str()
                        && resource
                            .map(|expected| signal.resource.as_deref() == Some(expected))
                            .unwrap_or(true)
                        && target_agent_id
                            .map(|expected| signal.target_agent_id.as_deref() == Some(expected))
                            .unwrap_or(true)
                        && target_projected_lane
                            .map(|expected| {
                                signal
                                    .target_projected_lane
                                    .as_ref()
                                    .map(|lane| lane.projection_id.as_str())
                                    == Some(expected.projection_id.as_str())
                            })
                            .unwrap_or(true)
                })
                .unwrap_or(false)
    })
}

fn has_source_signal(
    read: &crate::continuity::ContextRead,
    author_agent_id: &str,
    lane: CoordinationLane,
    resource: Option<&str>,
    target_agent_id: Option<&str>,
    target_projected_lane: Option<&CoordinationProjectedLane>,
) -> bool {
    active_source_signal(
        read,
        author_agent_id,
        lane,
        resource,
        target_agent_id,
        target_projected_lane,
    )
    .is_some()
}

fn source_is(item: &ContinuityItemRecord, source: &str) -> bool {
    item.extra
        .get("user")
        .and_then(|value| value.get("source"))
        .and_then(serde_json::Value::as_str)
        .or_else(|| item.extra.get("source").and_then(serde_json::Value::as_str))
        == Some(source)
}

fn claim_conflicts(read: &crate::continuity::ContextRead) -> Vec<ClaimConflictSummary> {
    serde_json::from_value(
        read.organism
            .get("claim_conflicts")
            .cloned()
            .unwrap_or_else(|| serde_json::json!([])),
    )
    .unwrap_or_default()
}

fn signal_still_matches_active_conflict(
    signal: &CoordinationSignalRecord,
    active_conflicts: &[ClaimConflictSummary],
) -> bool {
    if active_conflicts.is_empty() {
        return false;
    }

    let has_specific_target = signal.resource.is_some()
        || signal.target_agent_id.is_some()
        || signal.target_projected_lane.is_some()
        || !signal.projection_ids.is_empty()
        || !signal.projected_lanes.is_empty();
    if !has_specific_target {
        return true;
    }

    active_conflicts
        .iter()
        .any(|conflict| conflict_matches_signal(conflict, signal))
}

fn conflict_matches_signal(
    conflict: &ClaimConflictSummary,
    signal: &CoordinationSignalRecord,
) -> bool {
    if signal
        .resource
        .as_deref()
        .is_some_and(|resource| conflict.resource != resource)
    {
        return false;
    }
    if signal
        .target_agent_id
        .as_deref()
        .is_some_and(|target_agent_id| {
            !conflict
                .agents
                .iter()
                .any(|agent_id| agent_id == target_agent_id)
                && !conflict
                    .projected_lanes
                    .iter()
                    .any(|lane| lane.agent_id == target_agent_id)
        })
    {
        return false;
    }
    if signal
        .target_projected_lane
        .as_ref()
        .is_some_and(|target_lane| {
            !conflict
                .projected_lanes
                .iter()
                .any(|lane| lane.projection_id == target_lane.projection_id)
        })
    {
        return false;
    }
    if !signal.projection_ids.is_empty()
        && !signal.projection_ids.iter().any(|projection_id| {
            conflict
                .projection_ids
                .iter()
                .any(|active_id| active_id == projection_id)
        })
    {
        return false;
    }
    if !signal.projected_lanes.is_empty()
        && !signal.projected_lanes.iter().any(|signal_lane| {
            conflict
                .projected_lanes
                .iter()
                .any(|lane| lane.projection_id == signal_lane.projection_id)
        })
    {
        return false;
    }
    true
}

fn conflict_signal_title(
    lane: CoordinationLane,
    resource: &str,
    target: &ConflictTarget,
) -> String {
    let subject = target
        .display_name
        .as_deref()
        .unwrap_or(target.agent_id.as_str());
    match lane {
        CoordinationLane::Anxiety => format!("Anxiety spike on {resource} for {subject}"),
        CoordinationLane::Backoff => format!("Back off {subject} on {resource}"),
        CoordinationLane::Coach => format!("Coach {subject} on {resource}"),
        CoordinationLane::Review => format!("Review {subject} on {resource}"),
        CoordinationLane::Warning => format!("Warning {subject} on {resource}"),
    }
}

fn conflict_signal_body(
    lane: CoordinationLane,
    conflict: &ClaimConflictSummary,
    target: &ConflictTarget,
) -> String {
    let agents = if conflict.agents.is_empty() {
        "unknown agents".to_string()
    } else {
        conflict.agents.join(", ")
    };
    let lanes = conflict_lane_labels(conflict)
        .map(|labels| format!(" between lanes [{}]", labels))
        .unwrap_or_default();
    let subject = target
        .display_name
        .as_deref()
        .unwrap_or(target.agent_id.as_str());
    let target_lane = target
        .lane_label
        .as_deref()
        .map(|label| format!(" on lane [{label}]"))
        .unwrap_or_default();
    match lane {
        CoordinationLane::Anxiety => format!(
            "Exclusive claims are colliding on {} across [{}]{}. {} is directly involved{}. Slow down and confirm ownership before editing again.",
            conflict.resource, agents, lanes, subject, target_lane
        ),
        CoordinationLane::Backoff => format!(
            "Exclusive claims are colliding on {} across [{}]{}. {} must stop pushing{} until ownership is renegotiated through the organism.",
            conflict.resource, agents, lanes, subject, target_lane
        ),
        CoordinationLane::Coach => format!(
            "Conflict pressure on {} needs coaching across [{}]{}. Start with {}{}.",
            conflict.resource, agents, lanes, subject, target_lane
        ),
        CoordinationLane::Review => format!(
            "Conflict pressure on {} now needs explicit review across [{}]{}. Start with {}{}.",
            conflict.resource, agents, lanes, subject, target_lane
        ),
        CoordinationLane::Warning => format!(
            "Conflict pressure on {} is drifting toward a warning state across [{}]{}. Watch {}{}.",
            conflict.resource, agents, lanes, subject, target_lane
        ),
    }
}

fn conflict_targets(conflict: &ClaimConflictSummary) -> Vec<ConflictTarget> {
    let mut targets = Vec::<ConflictTarget>::new();
    for lane in &conflict.projected_lanes {
        if targets.iter().any(|target| {
            target.claim_id.as_deref() == Some(lane.claim_id.as_str())
                || target
                    .projection_id
                    .as_deref()
                    .map(|projection_id| {
                        projection_id == lane.projection_id && target.agent_id == lane.agent_id
                    })
                    .unwrap_or(false)
        }) {
            continue;
        }
        targets.push(ConflictTarget {
            agent_id: lane.agent_id.clone(),
            display_name: lane.display_name.clone(),
            claim_id: Some(lane.claim_id.clone()),
            projection_id: Some(lane.projection_id.clone()),
            lane_label: Some(lane.label.clone()),
            projected_lane: Some(CoordinationProjectedLane {
                projection_id: lane.projection_id.clone(),
                projection_kind: lane.projection_kind.clone(),
                label: lane.label.clone(),
                resource: lane.resource.clone(),
                repo_root: lane.repo_root.clone(),
                branch: lane.branch.clone(),
                task_id: lane.task_id.clone(),
            }),
        });
    }
    for agent_id in &conflict.agents {
        if targets.iter().any(|target| target.agent_id == *agent_id) {
            continue;
        }
        targets.push(ConflictTarget {
            agent_id: agent_id.clone(),
            display_name: None,
            claim_id: None,
            projection_id: None,
            lane_label: None,
            projected_lane: None,
        });
    }
    targets
}

fn conflict_lane_labels(conflict: &ClaimConflictSummary) -> Option<String> {
    let labels = conflict
        .projected_lanes
        .iter()
        .map(|lane| lane.label.clone())
        .fold(Vec::<String>::new(), |mut labels, label| {
            if !labels.iter().any(|existing| existing == &label) {
                labels.push(label);
            }
            labels
        });
    if labels.is_empty() {
        None
    } else {
        Some(labels.join(", "))
    }
}

fn projected_lanes_for_signal(conflict: &ClaimConflictSummary) -> Vec<CoordinationProjectedLane> {
    conflict
        .projected_lanes
        .iter()
        .map(|lane| CoordinationProjectedLane {
            projection_id: lane.projection_id.clone(),
            projection_kind: lane.projection_kind.clone(),
            label: lane.label.clone(),
            resource: lane.resource.clone(),
            repo_root: lane.repo_root.clone(),
            branch: lane.branch.clone(),
            task_id: lane.task_id.clone(),
        })
        .collect()
}

fn default_organism_specs() -> Vec<OrganismAgentSpec> {
    vec![
        OrganismAgentSpec {
            agent_id: "anxiety-sentinel",
            role: "anxiety",
            claim_title: "Hold anxiety lane",
            claim_body: "Monitor live claim conflicts and translate them into explicit anxiety signals.",
            resource: "organism/lane/anxiety",
            capabilities: &[
                "heartbeat",
                "claim_work",
                "read_context",
                "publish_coordination_signal",
            ],
        },
        OrganismAgentSpec {
            agent_id: "therapist",
            role: "therapist",
            claim_title: "Hold therapy lane",
            claim_body: "Treat disproven anxiety and keep old pressure from haunting active context forever.",
            resource: "organism/lane/therapy",
            capabilities: &[
                "heartbeat",
                "claim_work",
                "read_context",
                "resolve_or_supersede",
            ],
        },
        OrganismAgentSpec {
            agent_id: "review-coordinator",
            role: "review",
            claim_title: "Hold review lane",
            claim_body: "Escalate persistent ownership collisions into explicit review before parallel heads keep editing blind.",
            resource: "organism/lane/review",
            capabilities: &[
                "heartbeat",
                "claim_work",
                "read_context",
                "publish_coordination_signal",
                "resolve_or_supersede",
            ],
        },
        OrganismAgentSpec {
            agent_id: "scar-curator",
            role: "scar_curator",
            claim_title: "Hold scar lane",
            claim_body: "Keep operational scar pressure visible without turning it into random noise.",
            resource: "organism/lane/scars",
            capabilities: &[
                "heartbeat",
                "claim_work",
                "read_context",
                "publish_coordination_signal",
            ],
        },
        OrganismAgentSpec {
            agent_id: "boundary-warden",
            role: "boundary",
            claim_title: "Hold boundary lane",
            claim_body: "Publish explicit backoff pressure when ownership is colliding and the organism needs hard boundaries.",
            resource: "organism/lane/boundary",
            capabilities: &[
                "heartbeat",
                "claim_work",
                "read_context",
                "publish_coordination_signal",
            ],
        },
    ]
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::continuity::{ContinuityItemInput, ContinuityKind};
    use crate::model::MemoryLayer;
    use uuid::Uuid;

    #[tokio::test]
    async fn organism_choir_attaches_visible_agents_and_claims_scope() {
        let dir = tempdir().unwrap();
        let engine = Arc::new(Engine::open(dir.path()).unwrap());

        run_organism_choir(
            engine.clone(),
            OrganismChorusConfig {
                pulse_secs: None,
                ..OrganismChorusConfig::with_defaults(
                    "demo".into(),
                    "organism-choir".into(),
                    "keep the organism alive".into(),
                    format!("organism-{}", Uuid::now_v7()),
                )
            },
        )
        .await
        .unwrap();

        let metrics = engine.metrics_snapshot().unwrap().prometheus_text;
        assert!(metrics.contains(
            "ice_agent_active{agent_id=\"anxiety-sentinel\",agent_type=\"state_agent\",namespace=\"demo\",role=\"anxiety\"} 1"
        ));
        assert!(metrics.contains(
            "ice_agent_active{agent_id=\"therapist\",agent_type=\"state_agent\",namespace=\"demo\",role=\"therapist\"} 1"
        ));
        assert!(metrics.contains("ice_agent_badge_connected{attachment_id=\"attach:"));
        assert!(metrics.contains("display_name=\"anxiety-sentinel\""));
        assert!(metrics.contains("focus=\"Hold anxiety lane\""));

        let read = engine
            .read_context(ReadContextInput {
                context_id: None,
                namespace: Some("demo".into()),
                task_id: Some("organism-choir".into()),
                objective: "inspect choir".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        let choir_claims = read
            .work_claims
            .iter()
            .filter(|item| source_is(item, ORGANISM_SOURCE))
            .count();
        assert_eq!(choir_claims, 5);
        assert_eq!(
            read.organism["agent_badges"].as_array().map(Vec::len),
            Some(5)
        );
    }

    #[tokio::test]
    async fn organism_choir_turns_conflicts_into_anxiety_and_backoff() {
        let dir = tempdir().unwrap();
        let engine = Arc::new(Engine::open(dir.path()).unwrap());
        let planner = engine
            .attach_agent(AttachAgentInput {
                agent_id: "planner-a".into(),
                agent_type: "local".into(),
                capabilities: vec!["write".into()],
                namespace: "demo".into(),
                role: Some("planner".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo", "branch": "main"}),
            })
            .unwrap();
        let debugger = engine
            .attach_agent(AttachAgentInput {
                agent_id: "debugger-b".into(),
                agent_type: "local".into(),
                capabilities: vec!["write".into()],
                namespace: "demo".into(),
                role: Some("debugger".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo", "branch": "feature/shadow"}),
            })
            .unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "organism-conflict".into(),
                session_id: "session-1".into(),
                objective: "fight over one file".into(),
                selector: None,
                agent_id: Some("planner-a".into()),
                attachment_id: Some(planner.id.clone()),
            })
            .unwrap();
        engine
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(planner.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Planner".into()),
                status: Some("editing".into()),
                focus: Some("hold main lane".into()),
                headline: Some("planner main worktree".into()),
                resource: Some("repo/demo/main".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        engine
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(debugger.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Debugger".into()),
                status: Some("editing".into()),
                focus: Some("hold shadow lane".into()),
                headline: Some("debugger shadow worktree".into()),
                resource: Some("repo/demo/feature/shadow".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("feature/shadow".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "planner-a".into(),
                title: "Own src/continuity.rs".into(),
                body: "Planner lane".into(),
                scope: Scope::Project,
                resources: vec!["file/src/continuity.rs".into()],
                exclusive: true,
                attachment_id: Some(planner.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "debugger-b".into(),
                title: "Also own src/continuity.rs".into(),
                body: "Debugger lane".into(),
                scope: Scope::Project,
                resources: vec!["file/src/continuity.rs".into()],
                exclusive: true,
                attachment_id: Some(debugger.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();

        run_organism_choir(
            engine.clone(),
            OrganismChorusConfig {
                pulse_secs: None,
                ..OrganismChorusConfig::with_defaults(
                    "demo".into(),
                    "organism-conflict".into(),
                    "watch conflict pressure".into(),
                    "organism-conflict-session".into(),
                )
            },
        )
        .await
        .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect conflict pressure".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        let anxiety_targets = read
            .coordination_signals
            .iter()
            .filter(|item| {
                item.author_agent_id == "anxiety-sentinel" && source_is(item, ORGANISM_SOURCE)
            })
            .filter_map(coordination_signal)
            .filter(|signal| signal.lane == CoordinationLane::Anxiety.as_str())
            .filter_map(|signal| signal.target_agent_id)
            .collect::<Vec<_>>();
        let backoff_targets = read
            .coordination_signals
            .iter()
            .filter(|item| {
                item.author_agent_id == "boundary-warden" && source_is(item, ORGANISM_SOURCE)
            })
            .filter_map(coordination_signal)
            .filter(|signal| signal.lane == CoordinationLane::Backoff.as_str())
            .filter_map(|signal| signal.target_agent_id)
            .collect::<Vec<_>>();

        assert!(
            anxiety_targets
                .iter()
                .any(|agent_id| agent_id == "planner-a")
        );
        assert!(
            anxiety_targets
                .iter()
                .any(|agent_id| agent_id == "debugger-b")
        );
        assert!(
            backoff_targets
                .iter()
                .any(|agent_id| agent_id == "planner-a")
        );
        assert!(
            backoff_targets
                .iter()
                .any(|agent_id| agent_id == "debugger-b")
        );
        assert!(read.coordination_signals.iter().any(|item| {
            item.author_agent_id == "anxiety-sentinel"
                && source_is(item, ORGANISM_SOURCE)
                && coordination_signal(item)
                    .map(|signal| {
                        signal.lane == CoordinationLane::Anxiety.as_str()
                            && signal.resource.as_deref() == Some("file/src/continuity.rs")
                            && signal.target_agent_id.as_deref() == Some("planner-a")
                            && signal
                                .target_projected_lane
                                .as_ref()
                                .map(|lane| lane.label.as_str())
                                == Some("demo @ main")
                            && signal.projected_lanes.len() == 2
                            && signal
                                .projected_lanes
                                .iter()
                                .any(|lane| lane.label == "demo @ main")
                            && signal
                                .projected_lanes
                                .iter()
                                .any(|lane| lane.label == "demo @ feature/shadow")
                    })
                    .unwrap_or(false)
                && item.body.contains("demo @ main")
                && item.body.contains("demo @ feature/shadow")
        }));
        assert!(read.coordination_signals.iter().any(|item| {
            item.author_agent_id == "boundary-warden"
                && source_is(item, ORGANISM_SOURCE)
                && coordination_signal(item)
                    .map(|signal| {
                        signal.lane == CoordinationLane::Backoff.as_str()
                            && signal.resource.as_deref() == Some("file/src/continuity.rs")
                            && signal.target_agent_id.as_deref() == Some("debugger-b")
                            && signal
                                .target_projected_lane
                                .as_ref()
                                .map(|lane| lane.label.as_str())
                                == Some("demo @ feature/shadow")
                            && signal.projected_lanes.len() == 2
                    })
                    .unwrap_or(false)
        }));
        assert!(
            read.organism["coordination_pressure"]
                .as_array()
                .unwrap()
                .iter()
                .any(|signal| {
                    signal["lane"].as_str() == Some("anxiety")
                        && signal["target_projected_lane"]["label"].as_str() == Some("demo @ main")
                        && signal["projected_lanes"]
                            .as_array()
                            .map(|lanes| {
                                lanes
                                    .iter()
                                    .any(|lane| lane["label"].as_str() == Some("demo @ main"))
                                    && lanes.iter().any(|lane| {
                                        lane["label"].as_str() == Some("demo @ feature/shadow")
                                    })
                            })
                            .unwrap_or(false)
                })
        );
        let projections = read.organism["lane_projections"].as_array().unwrap();
        assert!(projections.iter().any(|projection| {
            projection["label"].as_str() == Some("demo @ main")
                && projection["coordination_signal_count"].as_u64() == Some(2)
                && projection["blocking_signal_count"].as_u64() == Some(1)
        }));
        assert!(projections.iter().any(|projection| {
            projection["label"].as_str() == Some("demo @ feature/shadow")
                && projection["coordination_signal_count"].as_u64() == Some(2)
                && projection["blocking_signal_count"].as_u64() == Some(1)
        }));
    }

    #[tokio::test]
    async fn persistent_conflicts_escalate_into_targeted_review_pressure() {
        let dir = tempdir().unwrap();
        let engine = Arc::new(Engine::open(dir.path()).unwrap());
        let planner = engine
            .attach_agent(AttachAgentInput {
                agent_id: "planner-a".into(),
                agent_type: "local".into(),
                capabilities: vec!["write".into()],
                namespace: "demo".into(),
                role: Some("planner".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo", "branch": "main"}),
            })
            .unwrap();
        let debugger = engine
            .attach_agent(AttachAgentInput {
                agent_id: "debugger-b".into(),
                agent_type: "local".into(),
                capabilities: vec!["write".into()],
                namespace: "demo".into(),
                role: Some("debugger".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo", "branch": "feature/shadow"}),
            })
            .unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "organism-review".into(),
                session_id: "session-1".into(),
                objective: "escalate persistent ownership fights into review".into(),
                selector: None,
                agent_id: Some("planner-a".into()),
                attachment_id: Some(planner.id.clone()),
            })
            .unwrap();
        engine
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(planner.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Planner".into()),
                status: Some("editing".into()),
                focus: Some("hold main lane".into()),
                headline: Some("planner main worktree".into()),
                resource: Some("repo/demo/main".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        engine
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(debugger.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Debugger".into()),
                status: Some("editing".into()),
                focus: Some("hold shadow lane".into()),
                headline: Some("debugger shadow worktree".into()),
                resource: Some("repo/demo/feature/shadow".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("feature/shadow".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "planner-a".into(),
                title: "Own src/storage.rs".into(),
                body: "Planner storage lane".into(),
                scope: Scope::Project,
                resources: vec!["file/src/storage.rs".into()],
                exclusive: true,
                attachment_id: Some(planner.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "debugger-b".into(),
                title: "Also own src/storage.rs".into(),
                body: "Debugger storage lane".into(),
                scope: Scope::Project,
                resources: vec!["file/src/storage.rs".into()],
                exclusive: true,
                attachment_id: Some(debugger.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let config = OrganismChorusConfig {
            pulse_secs: None,
            pulse_count: Some(2),
            ..OrganismChorusConfig::with_defaults(
                "demo".into(),
                "organism-review".into(),
                "escalate persistent ownership fights into review".into(),
                "organism-review-session".into(),
            )
        };
        let status = run_organism_choir(engine.clone(), config).await.unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect review pressure".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 32,
            })
            .unwrap();

        let review_targets = read
            .coordination_signals
            .iter()
            .filter(|item| {
                item.author_agent_id == "review-coordinator" && source_is(item, ORGANISM_SOURCE)
            })
            .filter_map(coordination_signal)
            .filter(|signal| signal.lane == CoordinationLane::Review.as_str())
            .collect::<Vec<_>>();

        assert_eq!(review_targets.len(), 2);
        assert!(review_targets.iter().any(|signal| {
            signal.target_agent_id.as_deref() == Some("planner-a")
                && signal
                    .target_projected_lane
                    .as_ref()
                    .map(|lane| lane.label.as_str())
                    == Some("demo @ main")
        }));
        assert!(review_targets.iter().any(|signal| {
            signal.target_agent_id.as_deref() == Some("debugger-b")
                && signal
                    .target_projected_lane
                    .as_ref()
                    .map(|lane| lane.label.as_str())
                    == Some("demo @ feature/shadow")
        }));
        assert_eq!(status["iteration"].as_u64(), Some(2));
        assert!(status["agents"].as_array().is_some_and(|agents| {
            agents.iter().any(|agent| {
                agent["agent_id"].as_str() == Some("review-coordinator")
                    && agent["tick_count"].as_u64().unwrap_or_default() > 3
            })
        }));
        let projections = read.organism["lane_projections"].as_array().unwrap();
        assert!(projections.iter().any(|projection| {
            projection["label"].as_str() == Some("demo @ main")
                && projection["coordination_signal_count"].as_u64() == Some(3)
                && projection["blocking_signal_count"].as_u64() == Some(1)
                && projection["review_signal_count"].as_u64() == Some(1)
        }));
        assert!(projections.iter().any(|projection| {
            projection["label"].as_str() == Some("demo @ feature/shadow")
                && projection["coordination_signal_count"].as_u64() == Some(3)
                && projection["blocking_signal_count"].as_u64() == Some(1)
                && projection["review_signal_count"].as_u64() == Some(1)
        }));
    }

    #[tokio::test]
    async fn bounded_pulse_runs_exit_even_with_pulse_spacing() {
        let dir = tempdir().unwrap();
        let engine = Arc::new(Engine::open(dir.path()).unwrap());
        let config = OrganismChorusConfig {
            pulse_secs: Some(5),
            pulse_count: Some(1),
            ..OrganismChorusConfig::with_defaults(
                "demo".into(),
                "organism-bounded-exit".into(),
                "exit after one spaced pulse".into(),
                "organism-bounded-exit-session".into(),
            )
        };

        let status =
            tokio::time::timeout(Duration::from_secs(2), run_organism_choir(engine, config))
                .await
                .expect("bounded pulse run should finish before the inter-pulse sleep");

        assert_eq!(status.unwrap()["iteration"].as_u64(), Some(1));
    }

    #[tokio::test]
    async fn therapist_resolves_conflict_pressure_after_the_fight_cools() {
        let dir = tempdir().unwrap();
        let engine = Arc::new(Engine::open(dir.path()).unwrap());
        let planner = engine
            .attach_agent(AttachAgentInput {
                agent_id: "planner-a".into(),
                agent_type: "local".into(),
                capabilities: vec!["write".into()],
                namespace: "demo".into(),
                role: Some("planner".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let debugger = engine
            .attach_agent(AttachAgentInput {
                agent_id: "debugger-b".into(),
                agent_type: "local".into(),
                capabilities: vec!["write".into()],
                namespace: "demo".into(),
                role: Some("debugger".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "organism-therapy".into(),
                session_id: "session-1".into(),
                objective: "fight and cool down".into(),
                selector: None,
                agent_id: Some("planner-a".into()),
                attachment_id: Some(planner.id.clone()),
            })
            .unwrap();
        let planner_claim = engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "planner-a".into(),
                title: "Own src/storage.rs".into(),
                body: "Planner lane".into(),
                scope: Scope::Project,
                resources: vec!["src/storage.rs".into()],
                exclusive: true,
                attachment_id: Some(planner.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "debugger-b".into(),
                title: "Also own src/storage.rs".into(),
                body: "Debugger lane".into(),
                scope: Scope::Project,
                resources: vec!["src/storage.rs".into()],
                exclusive: true,
                attachment_id: Some(debugger.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let config = OrganismChorusConfig {
            pulse_secs: None,
            ..OrganismChorusConfig::with_defaults(
                "demo".into(),
                "organism-therapy".into(),
                "watch conflict pressure".into(),
                "organism-therapy-session".into(),
            )
        };
        run_organism_choir(engine.clone(), config.clone())
            .await
            .unwrap();
        engine
            .resolve_or_supersede(ResolveOrSupersedeInput {
                continuity_id: planner_claim.id,
                actor_agent_id: "planner-a".into(),
                new_status: ContinuityStatus::Resolved,
                supersedes_id: None,
                resolution_note: Some("ownership released".into()),
                extra: serde_json::json!({}),
            })
            .unwrap();

        run_organism_choir(engine.clone(), config).await.unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect therapy".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert!(
            read.organism["anxiety_signal_count"]
                .as_u64()
                .unwrap_or_default()
                == 0
        );
        assert!(read.signals.iter().any(|item| {
            item.author_agent_id == "anxiety-sentinel"
                && source_is(item, ORGANISM_SOURCE)
                && !item.status.is_open()
        }));
        assert!(read.signals.iter().any(|item| {
            item.author_agent_id == "boundary-warden"
                && source_is(item, ORGANISM_SOURCE)
                && !item.status.is_open()
        }));
    }

    #[tokio::test]
    async fn therapist_resolves_only_cleared_targeted_conflicts() {
        let dir = tempdir().unwrap();
        let engine = Arc::new(Engine::open(dir.path()).unwrap());
        let planner = engine
            .attach_agent(AttachAgentInput {
                agent_id: "planner-a".into(),
                agent_type: "local".into(),
                capabilities: vec!["write".into()],
                namespace: "demo".into(),
                role: Some("planner".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let debugger = engine
            .attach_agent(AttachAgentInput {
                agent_id: "debugger-b".into(),
                agent_type: "local".into(),
                capabilities: vec!["write".into()],
                namespace: "demo".into(),
                role: Some("debugger".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let reviewer = engine
            .attach_agent(AttachAgentInput {
                agent_id: "reviewer-c".into(),
                agent_type: "local".into(),
                capabilities: vec!["write".into()],
                namespace: "demo".into(),
                role: Some("reviewer".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "organism-partial-therapy".into(),
                session_id: "session-1".into(),
                objective: "cool one conflict while another stays active".into(),
                selector: None,
                agent_id: Some("planner-a".into()),
                attachment_id: Some(planner.id.clone()),
            })
            .unwrap();
        engine
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(planner.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Planner".into()),
                status: Some("editing".into()),
                focus: Some("hold main lane".into()),
                headline: Some("planner main worktree".into()),
                resource: Some("repo/demo/main".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        engine
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(debugger.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Debugger".into()),
                status: Some("editing".into()),
                focus: Some("hold shadow lane".into()),
                headline: Some("debugger shadow worktree".into()),
                resource: Some("repo/demo/feature/shadow".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("feature/shadow".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        engine
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(reviewer.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Reviewer".into()),
                status: Some("editing".into()),
                focus: Some("hold review lane".into()),
                headline: Some("reviewer review worktree".into()),
                resource: Some("repo/demo/feature/review".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("feature/review".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let planner_storage_claim = engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "planner-a".into(),
                title: "Own src/storage.rs".into(),
                body: "Planner storage lane".into(),
                scope: Scope::Project,
                resources: vec!["file/src/storage.rs".into()],
                exclusive: true,
                attachment_id: Some(planner.id.clone()),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "debugger-b".into(),
                title: "Also own src/storage.rs".into(),
                body: "Debugger storage lane".into(),
                scope: Scope::Project,
                resources: vec!["file/src/storage.rs".into()],
                exclusive: true,
                attachment_id: Some(debugger.id.clone()),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "planner-a".into(),
                title: "Own src/query.rs".into(),
                body: "Planner query lane".into(),
                scope: Scope::Project,
                resources: vec!["file/src/query.rs".into()],
                exclusive: true,
                attachment_id: Some(planner.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "reviewer-c".into(),
                title: "Also own src/query.rs".into(),
                body: "Reviewer query lane".into(),
                scope: Scope::Project,
                resources: vec!["file/src/query.rs".into()],
                exclusive: true,
                attachment_id: Some(reviewer.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let config = OrganismChorusConfig {
            pulse_secs: None,
            ..OrganismChorusConfig::with_defaults(
                "demo".into(),
                "organism-partial-therapy".into(),
                "cool one conflict while another stays active".into(),
                "organism-partial-therapy-session".into(),
            )
        };
        run_organism_choir(engine.clone(), config.clone())
            .await
            .unwrap();
        engine
            .resolve_or_supersede(ResolveOrSupersedeInput {
                continuity_id: planner_storage_claim.id,
                actor_agent_id: "planner-a".into(),
                new_status: ContinuityStatus::Resolved,
                supersedes_id: None,
                resolution_note: Some("storage ownership released".into()),
                extra: serde_json::json!({}),
            })
            .unwrap();

        run_organism_choir(engine.clone(), config).await.unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect partial therapy".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 32,
            })
            .unwrap();

        let open_storage_signals = read
            .coordination_signals
            .iter()
            .filter_map(coordination_signal)
            .filter(|signal| signal.resource.as_deref() == Some("file/src/storage.rs"))
            .count();
        let open_query_signals = read
            .coordination_signals
            .iter()
            .filter_map(coordination_signal)
            .filter(|signal| signal.resource.as_deref() == Some("file/src/query.rs"))
            .collect::<Vec<_>>();
        let resolved_storage_signals = read
            .signals
            .iter()
            .filter(|item| source_is(item, ORGANISM_SOURCE) && !item.status.is_open())
            .filter_map(|item| crate::continuity::coordination_signal_from_extra(&item.extra))
            .filter(|signal| signal.resource.as_deref() == Some("file/src/storage.rs"))
            .count();

        assert_eq!(open_storage_signals, 0);
        assert_eq!(resolved_storage_signals, 4);
        assert_eq!(open_query_signals.len(), 6);
        assert!(open_query_signals.iter().any(|signal| {
            signal.lane == CoordinationLane::Anxiety.as_str()
                && signal.target_agent_id.as_deref() == Some("planner-a")
        }));
        assert!(open_query_signals.iter().any(|signal| {
            signal.lane == CoordinationLane::Anxiety.as_str()
                && signal.target_agent_id.as_deref() == Some("reviewer-c")
        }));
        assert!(open_query_signals.iter().any(|signal| {
            signal.lane == CoordinationLane::Backoff.as_str()
                && signal.target_agent_id.as_deref() == Some("planner-a")
        }));
        assert!(open_query_signals.iter().any(|signal| {
            signal.lane == CoordinationLane::Backoff.as_str()
                && signal.target_agent_id.as_deref() == Some("reviewer-c")
        }));
        assert!(open_query_signals.iter().any(|signal| {
            signal.lane == CoordinationLane::Review.as_str()
                && signal.target_agent_id.as_deref() == Some("planner-a")
        }));
        assert!(open_query_signals.iter().any(|signal| {
            signal.lane == CoordinationLane::Review.as_str()
                && signal.target_agent_id.as_deref() == Some("reviewer-c")
        }));
    }

    #[tokio::test]
    async fn scar_curator_surfaces_open_scars_as_coaching_pressure() {
        let dir = tempdir().unwrap();
        let engine = Arc::new(Engine::open(dir.path()).unwrap());
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "organism-scars".into(),
                session_id: "session-1".into(),
                objective: "remember the pain".into(),
                selector: None,
                agent_id: Some("operator".into()),
                attachment_id: None,
            })
            .unwrap();
        engine
            .mark_operational_scar(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "operator".into(),
                kind: ContinuityKind::OperationalScar,
                title: "Do not trust naked transcript carryover".into(),
                body: "Swaps need the kernel, not accidental transcript leakage.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.99),
                confidence: Some(0.99),
                salience: Some(0.99),
                layer: Some(MemoryLayer::Semantic),
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();

        run_organism_choir(
            engine.clone(),
            OrganismChorusConfig {
                pulse_secs: None,
                ..OrganismChorusConfig::with_defaults(
                    "demo".into(),
                    "organism-scars".into(),
                    "watch scar pressure".into(),
                    "organism-scar-session".into(),
                )
            },
        )
        .await
        .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect scar coaching".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert!(read.coordination_signals.iter().any(|item| {
            item.author_agent_id == "scar-curator"
                && source_is(item, ORGANISM_SOURCE)
                && coordination_signal(item)
                    .map(|signal| signal.lane == CoordinationLane::Coach.as_str())
                    .unwrap_or(false)
        }));
    }
}
