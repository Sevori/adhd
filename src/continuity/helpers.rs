use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};

use crate::dispatch;
use crate::model::{DimensionValue, EventInput, Scope, Selector};

use super::schema::*;
use super::types::*;

pub(crate) fn compile_handoff_proof(read: &ContextRead) -> HandoffProof {
    let mut registers = Vec::new();

    let primary_fact = read
        .incidents
        .first()
        .map(|item| {
            (
                "pf1",
                "fact",
                item.id.clone(),
                item.title.clone(),
                item.body.clone(),
                !item.supports.is_empty(),
            )
        })
        .or_else(|| {
            read.recall.items.first().map(|item| {
                (
                    "pf1",
                    "fact",
                    item.id.clone(),
                    item.title.clone(),
                    item.preview.clone(),
                    item.support_count > 0,
                )
            })
        })
        .unwrap_or_else(|| {
            (
                "pf1",
                "fact",
                read.context.id.clone(),
                "Primary context".to_string(),
                format!(
                    "Resume inside {} / {}.",
                    read.context.namespace, read.context.task_id
                ),
                true,
            )
        });
    registers.push(HandoffProofRegister {
        label: primary_fact.0.to_string(),
        register_kind: primary_fact.1.to_string(),
        source_id: primary_fact.2,
        title: primary_fact.3,
        body: trim_text(&primary_fact.4, PROOF_TRIM_LIMIT),
        has_provenance: primary_fact.5,
    });

    let push_item_register = |regs: &mut Vec<HandoffProofRegister>,
                              label: &str,
                              kind: &str,
                              item: &ContinuityItemRecord| {
        regs.push(HandoffProofRegister {
            label: label.to_string(),
            register_kind: kind.to_string(),
            source_id: item.id.clone(),
            title: item.title.clone(),
            body: trim_text(&item.body, PROOF_TRIM_LIMIT),
            has_provenance: !item.supports.is_empty(),
        });
    };

    if let Some(item) = read.decisions.first() {
        push_item_register(&mut registers, "pd1", "decision", item);
    }
    if let Some(item) = read.constraints.first() {
        push_item_register(&mut registers, "pk1", "constraint", item);
    }
    if let Some(item) = read.operational_scars.first() {
        push_item_register(&mut registers, "ps1", "scar", item);
    }
    if let Some(item) = read
        .working_state
        .iter()
        .find(|item| {
            item.title == "model-next-step" || item.extra["next_step"].as_bool() == Some(true)
        })
        .or_else(|| read.working_state.first())
    {
        push_item_register(&mut registers, "pn1", "next_step", item);
    }

    let digest = registers
        .iter()
        .map(|item| format!("{}:{}:{}", item.label, item.register_kind, item.title))
        .collect::<Vec<_>>()
        .join(" | ");

    HandoffProof { digest, registers }
}

pub(crate) fn trim_text(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.trim().to_string();
    }
    let mut trimmed = String::new();
    for ch in text.chars().take(limit.saturating_sub(3)) {
        trimmed.push(ch);
    }
    trimmed.push_str("...");
    trimmed.trim().to_string()
}

pub(crate) fn inject_context(event: &mut EventInput, context: &ContextRecord) {
    event.namespace = Some(context.namespace.clone());
    event.task_id = Some(context.task_id.clone());
    event.dimensions = augment_dimensions(
        std::mem::take(&mut event.dimensions),
        vec![
            DimensionValue {
                key: "context".to_string(),
                value: context.id.clone(),
                weight: DEFAULT_DIMENSION_WEIGHT,
            },
            DimensionValue {
                key: "context_namespace".to_string(),
                value: context.namespace.clone(),
                weight: DEFAULT_DIMENSION_WEIGHT,
            },
            DimensionValue {
                key: "context_task".to_string(),
                value: context.task_id.clone(),
                weight: DEFAULT_DIMENSION_WEIGHT,
            },
        ],
    );
}

pub(crate) fn resolve_namespace(
    storage: &crate::storage::Storage,
    namespace: Option<String>,
) -> anyhow::Result<Option<String>> {
    match namespace {
        Some(namespace) => storage.resolve_namespace_alias(Some(namespace.as_str())),
        None => Ok(None),
    }
}

pub(crate) fn resolve_selector_namespace(
    storage: &crate::storage::Storage,
    selector: Option<Selector>,
) -> anyhow::Result<Option<Selector>> {
    let Some(mut selector) = selector else {
        return Ok(None);
    };
    selector.namespace = storage.resolve_namespace_alias(selector.namespace.as_deref())?;
    Ok(Some(selector))
}

pub(crate) fn merge_context_selector(
    context: &ContextRecord,
    selector: Option<Selector>,
) -> Selector {
    let mut selector = selector.unwrap_or_else(|| context.selector.clone());
    selector.namespace = Some(context.namespace.clone());
    selector.all.push(crate::model::DimensionFilter {
        key: "context".to_string(),
        values: vec![context.id.clone()],
    });
    selector
}

pub(crate) fn filter_kind(
    items: &[ContinuityItemRecord],
    kind: ContinuityKind,
) -> Vec<ContinuityItemRecord> {
    items
        .iter()
        .filter(|item| item.kind == kind)
        .cloned()
        .collect()
}

pub(crate) fn default_work_claim_lease_seconds() -> u64 {
    180
}

pub(crate) fn normalize_work_claim_resources(resources: &[String]) -> Vec<String> {
    let mut normalized = resources
        .iter()
        .map(|resource| resource.trim())
        .filter(|resource| !resource.is_empty())
        .map(|resource| resource.to_ascii_lowercase())
        .collect::<Vec<_>>();
    normalized.sort();
    normalized.dedup();
    normalized
}

pub(crate) fn work_claim_key(
    context_id: &str,
    scope: Scope,
    agent_id: &str,
    title: &str,
    resources: &[String],
) -> String {
    let normalized_title = title
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    let subject = if resources.is_empty() {
        normalized_title
    } else {
        resources.join("|")
    };
    format!("{context_id}:{scope}:{agent_id}:{subject}")
}

pub(crate) fn merge_work_claim_extra(
    extra: serde_json::Value,
    coordination: &WorkClaimCoordination,
) -> serde_json::Value {
    let coordination_value =
        serde_json::to_value(coordination).unwrap_or_else(|_| serde_json::json!({}));
    match extra {
        serde_json::Value::Object(mut map) => {
            map.insert("coordination".to_string(), coordination_value);
            serde_json::Value::Object(map)
        }
        serde_json::Value::Null => serde_json::json!({ "coordination": coordination_value }),
        other => serde_json::json!({
            "coordination": coordination_value,
            "payload": other,
        }),
    }
}

pub(crate) fn work_claim_coordination_from_extra(
    extra: &serde_json::Value,
) -> Option<WorkClaimCoordination> {
    extra
        .get("user")
        .and_then(|value| value.get("coordination"))
        .or_else(|| extra.get("coordination"))
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
}

pub(crate) fn work_claim_coordination(
    item: &ContinuityItemRecord,
) -> Option<WorkClaimCoordination> {
    if item.kind != ContinuityKind::WorkClaim {
        return None;
    }
    work_claim_coordination_from_extra(&item.extra)
}

pub(crate) fn work_claim_is_live(item: &ContinuityItemRecord, now: DateTime<Utc>) -> bool {
    if item.kind != ContinuityKind::WorkClaim || !item.status.is_open() {
        return false;
    }
    work_claim_coordination(item)
        .and_then(|coordination| coordination.lease_expires_at)
        .map(|deadline| deadline > now)
        .unwrap_or(true)
}

pub(crate) fn counts_as_open_thread(item: &ContinuityItemRecord, now: DateTime<Utc>) -> bool {
    if !item.status.is_open() {
        return false;
    }
    if item.kind == ContinuityKind::WorkClaim {
        return work_claim_is_live(item, now);
    }
    true
}

pub(crate) fn work_claims_conflict(
    left: &ContinuityItemRecord,
    right: &ContinuityItemRecord,
    now: DateTime<Utc>,
) -> bool {
    if left.id == right.id || !work_claim_is_live(left, now) || !work_claim_is_live(right, now) {
        return false;
    }
    let Some(left_coordination) = work_claim_coordination(left) else {
        return false;
    };
    let Some(right_coordination) = work_claim_coordination(right) else {
        return false;
    };
    if !(left_coordination.exclusive || right_coordination.exclusive) {
        return false;
    }
    left_coordination.resources.iter().any(|resource| {
        right_coordination
            .resources
            .iter()
            .any(|other| other == resource)
    })
}

pub(crate) fn default_coordination_severity(lane: CoordinationLane) -> CoordinationSeverity {
    match lane {
        CoordinationLane::Review | CoordinationLane::Coach => CoordinationSeverity::Info,
        CoordinationLane::Warning | CoordinationLane::Anxiety => CoordinationSeverity::Warn,
        CoordinationLane::Backoff => CoordinationSeverity::Block,
    }
}

pub(crate) fn merge_coordination_signal_extra(
    extra: serde_json::Value,
    lane: CoordinationLane,
    severity: CoordinationSeverity,
    target_agent_id: Option<String>,
    target_projected_lane: Option<CoordinationProjectedLane>,
    claim_id: Option<String>,
    resource: Option<String>,
    projection_ids: Vec<String>,
    projected_lanes: Vec<CoordinationProjectedLane>,
) -> serde_json::Value {
    let coordination_value = serde_json::json!({
        "lane": lane.as_str(),
        "severity": severity.as_str(),
        "target_agent_id": target_agent_id,
        "target_projected_lane": target_projected_lane,
        "claim_id": claim_id,
        "resource": resource,
        "projection_ids": projection_ids,
        "projected_lanes": projected_lanes,
    });
    match extra {
        serde_json::Value::Object(mut map) => {
            map.insert("coordination_signal".to_string(), coordination_value);
            serde_json::Value::Object(map)
        }
        serde_json::Value::Null => {
            serde_json::json!({ "coordination_signal": coordination_value })
        }
        other => serde_json::json!({
            "coordination_signal": coordination_value,
            "payload": other,
        }),
    }
}

pub(crate) fn coordination_signal_from_extra(
    extra: &serde_json::Value,
) -> Option<CoordinationSignalRecord> {
    extra
        .get("user")
        .and_then(|value| value.get("coordination_signal"))
        .or_else(|| extra.get("coordination_signal"))
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
}

pub(crate) fn coordination_signal(item: &ContinuityItemRecord) -> Option<CoordinationSignalRecord> {
    if item.kind != ContinuityKind::Signal || !item.status.is_open() {
        return None;
    }
    coordination_signal_from_extra(&item.extra)
}

pub(crate) fn organism_state(
    items: &[ContinuityItemRecord],
    now: DateTime<Utc>,
    agent_badges: &[AgentBadgeRecord],
    lane_projections: &[LaneProjectionRecord],
) -> serde_json::Value {
    let mut retention_classes = BTreeMap::<String, usize>::new();
    let mut kinds = BTreeMap::<String, usize>::new();
    let mut open_pressure = Vec::new();
    let mut treated_count = 0usize;
    let mut active_claims = Vec::new();
    let mut active_claim_records = Vec::new();
    let mut expired_claims = 0usize;
    let mut coordination_pressure = Vec::new();
    let mut anxiety_pressure = Vec::new();

    for item in items {
        *retention_classes
            .entry(item.retention.class.clone())
            .or_insert(0) += 1;
        *kinds.entry(item.kind.as_str().to_string()).or_insert(0) += 1;
        if item.retention.class.starts_with("treated_") {
            treated_count += 1;
        }
        if item.kind == ContinuityKind::WorkClaim {
            if work_claim_is_live(item, now) {
                if let Some(coordination) = work_claim_coordination(item) {
                    active_claims.push(serde_json::json!({
                        "id": item.id,
                        "agent_id": item.author_agent_id,
                        "title": item.title,
                        "resources": coordination.resources,
                        "exclusive": coordination.exclusive,
                        "lease_expires_at": coordination.lease_expires_at,
                    }));
                    active_claim_records.push(item);
                }
            } else if item.status.is_open() {
                expired_claims += 1;
            }
        }
        if counts_as_open_thread(item, now) {
            open_pressure.push(serde_json::json!({
                "id": item.id,
                "kind": item.kind.as_str(),
                "status": item.status.as_str(),
                "title": item.title,
                "retention_class": item.retention.class,
                "effective_salience": item.retention.effective_salience,
            }));
        }
        if let Some(signal) = coordination_signal(item) {
            let summary = serde_json::json!({
                "id": item.id,
                "lane": signal.lane,
                "severity": signal.severity,
                "title": item.title,
                "target_agent_id": signal.target_agent_id,
                "target_projected_lane": signal.target_projected_lane,
                "claim_id": signal.claim_id,
                "resource": signal.resource,
                "projection_ids": signal.projection_ids,
                "projected_lanes": signal.projected_lanes,
            });
            if summary["lane"].as_str() == Some(CoordinationLane::Anxiety.as_str()) {
                anxiety_pressure.push(summary.clone());
            }
            coordination_pressure.push(summary);
        }
    }
    open_pressure.truncate(8);
    active_claims.truncate(8);
    coordination_pressure.truncate(8);
    anxiety_pressure.truncate(8);

    let attachment_projection_by_id = attachment_projection_map(agent_badges, lane_projections);
    let mut seen_resources = BTreeMap::<String, ClaimConflictSummary>::new();
    for (index, left) in active_claim_records.iter().enumerate() {
        let Some(left_coordination) = work_claim_coordination(left) else {
            continue;
        };
        for right in active_claim_records.iter().skip(index + 1) {
            if !work_claims_conflict(left, right, now) {
                continue;
            }
            let Some(right_coordination) = work_claim_coordination(right) else {
                continue;
            };
            for resource in left_coordination.resources.iter().filter(|resource| {
                right_coordination
                    .resources
                    .iter()
                    .any(|other| other == *resource)
            }) {
                let conflict = seen_resources
                    .entry(resource.clone())
                    .or_insert_with(|| ClaimConflictSummary::new(resource.clone()));
                conflict.absorb_claim(
                    left,
                    projected_lane_for_claim(
                        left,
                        &left_coordination,
                        &attachment_projection_by_id,
                        lane_projections,
                    ),
                );
                conflict.absorb_claim(
                    right,
                    projected_lane_for_claim(
                        right,
                        &right_coordination,
                        &attachment_projection_by_id,
                        lane_projections,
                    ),
                );
            }
        }
    }
    let claim_conflicts = seen_resources.into_values().collect::<Vec<_>>();

    serde_json::json!({
        "continuity_items": items.len(),
        "treated_items": treated_count,
        "retention_classes": retention_classes,
        "kinds": kinds,
        "open_pressure": open_pressure,
        "active_claim_count": active_claim_records.len(),
        "expired_claim_count": expired_claims,
        "active_claims": active_claims,
        "claim_conflict_count": claim_conflicts.len(),
        "claim_conflicts": claim_conflicts,
        "coordination_signal_count": coordination_pressure.len(),
        "coordination_pressure": coordination_pressure,
        "anxiety_signal_count": anxiety_pressure.len(),
        "anxiety_pressure": anxiety_pressure,
    })
}

pub(crate) fn merge_dispatch_worker_lane_projections(
    lane_projections: &mut Vec<LaneProjectionRecord>,
    dispatch_state: &dispatch::DispatchOrganismSnapshot,
    namespace_fallback: &str,
) {
    if dispatch_state.workers.is_empty() {
        return;
    }
    let mut merged = lane_projections
        .drain(..)
        .map(|projection| (projection.projection_id.clone(), projection))
        .collect::<BTreeMap<_, _>>();
    for worker in &dispatch_state.workers {
        let dispatch_projection = dispatch_worker_lane_projection(worker, namespace_fallback);
        if let Some(existing) = merged.get_mut(&dispatch_projection.projection_id) {
            absorb_dispatch_lane_projection(existing, dispatch_projection);
        } else {
            merged.insert(
                dispatch_projection.projection_id.clone(),
                dispatch_projection,
            );
        }
    }
    let mut values = merged.into_values().collect::<Vec<_>>();
    values.sort_by(|left, right| {
        right
            .updated_at
            .cmp(&left.updated_at)
            .then_with(|| left.projection_kind.cmp(&right.projection_kind))
            .then_with(|| left.label.cmp(&right.label))
    });
    *lane_projections = values;
}

pub(crate) fn merge_dispatch_assignment_pressure(
    lane_projections: &mut Vec<LaneProjectionRecord>,
    dispatch_state: &dispatch::DispatchOrganismSnapshot,
    namespace_fallback: &str,
) {
    if dispatch_state.assignments.is_empty() {
        return;
    }
    let mut merged = lane_projections
        .drain(..)
        .map(|projection| (projection.projection_id.clone(), projection))
        .collect::<BTreeMap<_, _>>();
    for assignment in &dispatch_state.assignments {
        let projection = merged
            .entry(assignment.projected_lane.projection_id.clone())
            .or_insert_with(|| {
                dispatch_assignment_lane_projection(
                    assignment,
                    &assignment.projected_lane,
                    namespace_fallback,
                    true,
                )
            });
        absorb_dispatch_assignment_pressure(projection, assignment, true);
        if let Some(attached_lane) = assignment.attached_projected_lane.as_ref() {
            if attached_lane.projection_id != assignment.projected_lane.projection_id {
                let projection = merged
                    .entry(attached_lane.projection_id.clone())
                    .or_insert_with(|| {
                        dispatch_assignment_lane_projection(
                            assignment,
                            attached_lane,
                            namespace_fallback,
                            false,
                        )
                    });
                absorb_dispatch_assignment_pressure(projection, assignment, false);
            }
        }
    }
    let mut values = merged.into_values().collect::<Vec<_>>();
    values.sort_by(|left, right| {
        right
            .updated_at
            .cmp(&left.updated_at)
            .then_with(|| left.projection_kind.cmp(&right.projection_kind))
            .then_with(|| left.label.cmp(&right.label))
    });
    *lane_projections = values;
}

fn dispatch_worker_lane_projection(
    worker: &dispatch::DispatchWorkerPresence,
    namespace_fallback: &str,
) -> LaneProjectionRecord {
    let headline = if worker.active_assignment_count > 0 {
        format!(
            "{} is {} with {} active dispatch assignment(s).",
            worker.display_name, worker.status, worker.active_assignment_count
        )
    } else {
        format!(
            "{} is {} via the dispatch spine.",
            worker.display_name, worker.status
        )
    };
    LaneProjectionRecord {
        projection_id: worker.projected_lane.projection_id.clone(),
        namespace: worker
            .namespace
            .clone()
            .unwrap_or_else(|| namespace_fallback.to_string()),
        projection_kind: worker.projected_lane.projection_kind.clone(),
        label: worker.projected_lane.label.clone(),
        resource: worker.projected_lane.resource.clone(),
        repo_root: worker.projected_lane.repo_root.clone(),
        branch: worker.projected_lane.branch.clone(),
        task_id: worker.projected_lane.task_id.clone(),
        connected_agents: 1,
        live_claims: 0,
        claim_conflicts: 0,
        coordination_signal_count: 0,
        blocking_signal_count: 0,
        review_signal_count: 0,
        dispatch_assignment_count: 0,
        dispatch_assignment_anxiety_max: 0.0,
        dispatch_assignment_explicit_cli_count: 0,
        dispatch_assignment_live_badge_opt_in_count: 0,
        coordination_lanes: Vec::new(),
        agent_ids: vec![worker.worker_id.clone()],
        display_names: vec![worker.display_name.clone()],
        focus: if worker.focus.trim().is_empty() {
            "dispatch ready".to_string()
        } else {
            worker.focus.clone()
        },
        headline,
        updated_at: worker.last_seen_at,
    }
}

fn absorb_dispatch_lane_projection(
    existing: &mut LaneProjectionRecord,
    dispatch_projection: LaneProjectionRecord,
) {
    if existing.resource.is_none() {
        existing.resource = dispatch_projection.resource.clone();
    }
    if existing.task_id.is_none() {
        existing.task_id = dispatch_projection.task_id.clone();
    }
    if existing.repo_root.is_none() {
        existing.repo_root = dispatch_projection.repo_root.clone();
    }
    if existing.branch.is_none() {
        existing.branch = dispatch_projection.branch.clone();
    }
    let agent_ids = existing
        .agent_ids
        .iter()
        .cloned()
        .chain(dispatch_projection.agent_ids)
        .collect::<BTreeSet<_>>();
    existing.agent_ids = agent_ids.iter().cloned().collect();
    existing.connected_agents = existing.agent_ids.len();
    let display_names = existing
        .display_names
        .iter()
        .cloned()
        .chain(dispatch_projection.display_names)
        .collect::<BTreeSet<_>>();
    existing.display_names = display_names.iter().cloned().collect();
    if dispatch_projection.updated_at >= existing.updated_at
        || existing.focus.is_empty()
        || existing.headline.is_empty()
    {
        existing.focus = dispatch_projection.focus;
        existing.headline = dispatch_projection.headline;
        existing.updated_at = dispatch_projection.updated_at;
    }
}

fn dispatch_assignment_lane_projection(
    assignment: &dispatch::DispatchAssignmentPresence,
    lane: &CoordinationProjectedLane,
    namespace_fallback: &str,
    carry_worker_identity: bool,
) -> LaneProjectionRecord {
    let connected_agents = usize::from(carry_worker_identity);
    let agent_ids = if carry_worker_identity {
        vec![assignment.worker_id.clone()]
    } else {
        Vec::new()
    };
    let display_names = if carry_worker_identity {
        vec![assignment.worker_id.clone()]
    } else {
        Vec::new()
    };
    LaneProjectionRecord {
        projection_id: lane.projection_id.clone(),
        namespace: assignment
            .namespace
            .clone()
            .if_empty_then(namespace_fallback),
        projection_kind: lane.projection_kind.clone(),
        label: lane.label.clone(),
        resource: lane.resource.clone(),
        repo_root: lane.repo_root.clone(),
        branch: lane.branch.clone(),
        task_id: lane.task_id.clone(),
        connected_agents,
        live_claims: 0,
        claim_conflicts: 0,
        coordination_signal_count: 0,
        blocking_signal_count: 0,
        review_signal_count: 0,
        dispatch_assignment_count: 0,
        dispatch_assignment_anxiety_max: 0.0,
        dispatch_assignment_explicit_cli_count: 0,
        dispatch_assignment_live_badge_opt_in_count: 0,
        coordination_lanes: Vec::new(),
        agent_ids,
        display_names,
        focus: assignment.objective.clone(),
        headline: format!(
            "{} active dispatch assignment(s); max anxiety {:.2}",
            1, assignment.pressure.anxiety
        ),
        updated_at: assignment.created_at,
    }
}

fn absorb_dispatch_assignment_pressure(
    projection: &mut LaneProjectionRecord,
    assignment: &dispatch::DispatchAssignmentPresence,
    carry_worker_identity: bool,
) {
    projection.dispatch_assignment_count += 1;
    projection.dispatch_assignment_anxiety_max = projection
        .dispatch_assignment_anxiety_max
        .max(assignment.pressure.anxiety);
    if !carry_worker_identity {
        match assignment.attached_projected_lane_source {
            Some(dispatch::DispatchAttachedLaneSource::ExplicitCli) => {
                projection.dispatch_assignment_explicit_cli_count += 1;
            }
            Some(dispatch::DispatchAttachedLaneSource::LiveBadgeOptIn) => {
                projection.dispatch_assignment_live_badge_opt_in_count += 1;
            }
            None => {}
        }
    }
    if carry_worker_identity {
        if !projection
            .agent_ids
            .iter()
            .any(|agent_id| agent_id == &assignment.worker_id)
        {
            projection.agent_ids.push(assignment.worker_id.clone());
            projection.connected_agents = projection.agent_ids.len();
        }
        if !projection
            .display_names
            .iter()
            .any(|display_name| display_name == &assignment.worker_id)
        {
            projection.display_names.push(assignment.worker_id.clone());
        }
    }
    if assignment.created_at >= projection.updated_at
        || projection.focus.is_empty()
        || projection.headline.is_empty()
    {
        projection.focus = assignment.objective.clone();
        projection.headline = format!(
            "{} active dispatch assignment(s); max anxiety {:.2}",
            projection.dispatch_assignment_count, projection.dispatch_assignment_anxiety_max
        );
        projection.updated_at = assignment.created_at;
    }
}

pub(super) trait IfEmptyThen {
    fn if_empty_then(self, fallback: &str) -> String;
}

impl IfEmptyThen for String {
    fn if_empty_then(self, fallback: &str) -> String {
        if self.trim().is_empty() {
            fallback.to_string()
        } else {
            self
        }
    }
}

fn attachment_projection_map(
    agent_badges: &[AgentBadgeRecord],
    lane_projections: &[LaneProjectionRecord],
) -> BTreeMap<String, (Option<String>, LaneProjectionRecord)> {
    let mut attachment_projection_by_id = BTreeMap::new();
    for badge in agent_badges.iter().filter(|badge| badge.connected) {
        let Some(projection) = projection_for_badge(badge, lane_projections) else {
            continue;
        };
        attachment_projection_by_id.insert(
            badge.attachment_id.clone(),
            (Some(badge.display_name.clone()), projection.clone()),
        );
    }
    attachment_projection_by_id
}

fn projection_for_badge<'a>(
    badge: &AgentBadgeRecord,
    lane_projections: &'a [LaneProjectionRecord],
) -> Option<&'a LaneProjectionRecord> {
    lane_projections
        .iter()
        .find(|projection| {
            projection.namespace == badge.namespace
                && projection.resource == badge.resource
                && projection.repo_root == badge.repo_root
                && projection.branch == badge.branch
                && projection.task_id == badge.task_id
        })
        .or_else(|| {
            lane_projections.iter().find(|projection| {
                projection.namespace == badge.namespace
                    && projection.repo_root == badge.repo_root
                    && projection.branch == badge.branch
                    && projection.task_id == badge.task_id
            })
        })
        .or_else(|| {
            lane_projections.iter().find(|projection| {
                projection.namespace == badge.namespace
                    && projection.resource == badge.resource
                    && projection.task_id == badge.task_id
            })
        })
}

fn projected_lane_for_claim(
    item: &ContinuityItemRecord,
    coordination: &WorkClaimCoordination,
    attachment_projection_by_id: &BTreeMap<String, (Option<String>, LaneProjectionRecord)>,
    lane_projections: &[LaneProjectionRecord],
) -> Option<ClaimConflictLaneSummary> {
    if let Some((display_name, projection)) = coordination
        .attachment_id
        .as_ref()
        .and_then(|attachment_id| attachment_projection_by_id.get(attachment_id))
    {
        return Some(claim_conflict_lane_summary(
            item,
            display_name.clone(),
            projection,
        ));
    }

    let resource = coordination
        .resources
        .iter()
        .find(|resource| !resource.trim().is_empty())?;
    let projection = lane_projections.iter().find(|projection| {
        projection.namespace == item.namespace
            && projection.resource.as_deref() == Some(resource.as_str())
            && projection.task_id.as_deref() == Some(item.task_id.as_str())
    })?;
    Some(claim_conflict_lane_summary(item, None, projection))
}

fn claim_conflict_lane_summary(
    item: &ContinuityItemRecord,
    display_name: Option<String>,
    projection: &LaneProjectionRecord,
) -> ClaimConflictLaneSummary {
    ClaimConflictLaneSummary {
        claim_id: item.id.clone(),
        agent_id: item.author_agent_id.clone(),
        title: item.title.clone(),
        display_name,
        projection_id: projection.projection_id.clone(),
        projection_kind: projection.projection_kind.clone(),
        label: projection.label.clone(),
        resource: projection.resource.clone(),
        repo_root: projection.repo_root.clone(),
        branch: projection.branch.clone(),
        task_id: projection.task_id.clone(),
    }
}

pub(crate) fn augment_dimensions(
    base: Vec<DimensionValue>,
    extra: Vec<DimensionValue>,
) -> Vec<DimensionValue> {
    let mut merged = base;
    for item in extra {
        if !merged
            .iter()
            .any(|existing| existing.key == item.key && existing.value == item.value)
        {
            merged.push(item);
        }
    }
    merged
}
