use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::model::{
    ContextPack, ContextPackManifest, DimensionValue, HandoffRecord, MemoryLayer, Scope, Selector,
    SnapshotResolution, ViewManifest,
};

use super::types::{
    ContextStatus, ContinuityKind, ContinuityStatus, CoordinationLane, CoordinationSeverity,
};

fn default_support_weight() -> f64 {
    1.0
}

pub(crate) fn default_candidate_limit() -> usize {
    24
}

pub(crate) fn default_token_budget() -> usize {
    384
}

pub(crate) fn default_snapshot_resolution() -> SnapshotResolution {
    SnapshotResolution::Medium
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportRef {
    pub support_type: String,
    pub support_id: String,
    pub reason: Option<String>,
    #[serde(default = "default_support_weight")]
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachAgentInput {
    pub agent_id: String,
    pub agent_type: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    pub namespace: String,
    pub role: Option<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAttachmentRecord {
    pub id: String,
    pub attached_at: DateTime<Utc>,
    pub last_seen_at: DateTime<Utc>,
    pub input: AttachAgentInput,
    pub tick_count: usize,
    pub active: bool,
    pub context_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertAgentBadgeInput {
    pub attachment_id: Option<String>,
    pub agent_id: Option<String>,
    pub namespace: Option<String>,
    pub context_id: Option<String>,
    pub display_name: Option<String>,
    pub status: Option<String>,
    pub focus: Option<String>,
    pub headline: Option<String>,
    pub resource: Option<String>,
    pub repo_root: Option<String>,
    pub branch: Option<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBadgeRecord {
    pub attachment_id: String,
    pub agent_id: String,
    pub agent_type: String,
    pub namespace: String,
    pub task_id: Option<String>,
    pub context_id: Option<String>,
    pub role: Option<String>,
    pub display_name: String,
    pub status: String,
    pub focus: String,
    pub headline: String,
    pub resource: Option<String>,
    pub repo_root: Option<String>,
    pub branch: Option<String>,
    pub metadata: serde_json::Value,
    pub updated_at: DateTime<Utc>,
    pub last_seen_at: DateTime<Utc>,
    pub tick_count: usize,
    pub connected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneProjectionRecord {
    pub projection_id: String,
    pub namespace: String,
    pub projection_kind: String,
    pub label: String,
    pub resource: Option<String>,
    pub repo_root: Option<String>,
    pub branch: Option<String>,
    pub task_id: Option<String>,
    pub connected_agents: usize,
    pub live_claims: usize,
    pub claim_conflicts: usize,
    #[serde(default)]
    pub coordination_signal_count: usize,
    #[serde(default)]
    pub blocking_signal_count: usize,
    #[serde(default)]
    pub review_signal_count: usize,
    #[serde(default)]
    pub dispatch_assignment_count: usize,
    #[serde(default)]
    pub dispatch_assignment_anxiety_max: f64,
    #[serde(default)]
    pub dispatch_assignment_explicit_cli_count: usize,
    #[serde(default)]
    pub dispatch_assignment_live_badge_opt_in_count: usize,
    #[serde(default)]
    pub coordination_lanes: Vec<String>,
    pub agent_ids: Vec<String>,
    pub display_names: Vec<String>,
    pub focus: String,
    pub headline: String,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineProfile {
    pub machine_id: String,
    pub label: String,
    pub namespace: String,
    pub default_task_id: String,
    pub host_name: String,
    pub os_name: String,
    pub kernel_version: Option<String>,
    pub storage_root: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenContextInput {
    pub namespace: String,
    pub task_id: String,
    pub session_id: String,
    pub objective: String,
    pub selector: Option<Selector>,
    pub agent_id: Option<String>,
    pub attachment_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextRecord {
    pub id: String,
    pub namespace: String,
    pub task_id: String,
    pub session_id: String,
    pub objective: String,
    pub selector: Selector,
    pub status: ContextStatus,
    pub opened_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub current_agent_id: Option<String>,
    pub current_attachment_id: Option<String>,
    pub last_snapshot_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteEventInput {
    pub context_id: Option<String>,
    #[serde(flatten)]
    pub event: crate::model::EventInput,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuityItemInput {
    pub context_id: String,
    pub author_agent_id: String,
    pub kind: ContinuityKind,
    pub title: String,
    pub body: String,
    pub scope: Scope,
    pub status: Option<ContinuityStatus>,
    pub importance: Option<f64>,
    pub confidence: Option<f64>,
    pub salience: Option<f64>,
    pub layer: Option<MemoryLayer>,
    #[serde(default)]
    pub supports: Vec<SupportRef>,
    #[serde(default)]
    pub dimensions: Vec<DimensionValue>,
    #[serde(default)]
    pub extra: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuityRetentionState {
    pub class: String,
    pub age_hours: f64,
    pub half_life_hours: f64,
    pub floor: f64,
    pub decay_multiplier: f64,
    pub effective_salience: f64,
}

fn default_plasticity_spacing_interval_hours() -> f64 {
    6.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuityPlasticityState {
    pub activation_count: usize,
    pub successful_use_count: usize,
    pub confirmation_count: usize,
    pub contradiction_count: usize,
    pub independent_source_count: usize,
    #[serde(default)]
    pub spaced_reactivation_count: usize,
    pub stability_score: f64,
    pub prediction_error: f64,
    #[serde(default = "default_plasticity_spacing_interval_hours")]
    pub spacing_interval_hours: f64,
    pub last_reactivated_at: Option<DateTime<Utc>>,
    pub last_confirmed_at: Option<DateTime<Utc>>,
    pub last_contradicted_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub last_strengthened_at: Option<DateTime<Utc>>,
}

impl Default for ContinuityPlasticityState {
    fn default() -> Self {
        Self {
            activation_count: 0,
            successful_use_count: 0,
            confirmation_count: 0,
            contradiction_count: 0,
            independent_source_count: 0,
            spaced_reactivation_count: 0,
            stability_score: 0.0,
            prediction_error: 0.0,
            spacing_interval_hours: default_plasticity_spacing_interval_hours(),
            last_reactivated_at: None,
            last_confirmed_at: None,
            last_contradicted_at: None,
            last_strengthened_at: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuityItemRecord {
    pub id: String,
    pub memory_id: String,
    pub context_id: String,
    pub namespace: String,
    pub task_id: String,
    pub author_agent_id: String,
    pub kind: ContinuityKind,
    pub scope: Scope,
    pub status: ContinuityStatus,
    pub title: String,
    pub body: String,
    pub importance: f64,
    pub confidence: f64,
    pub salience: f64,
    pub retention: ContinuityRetentionState,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub supersedes_id: Option<String>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub supports: Vec<SupportRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub practice_state: Option<PracticeLifecycleState>,
    pub extra: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadContextInput {
    pub context_id: Option<String>,
    pub namespace: Option<String>,
    pub task_id: Option<String>,
    pub objective: String,
    pub token_budget: usize,
    pub selector: Option<Selector>,
    pub agent_id: Option<String>,
    pub session_id: Option<String>,
    pub view_id: Option<String>,
    #[serde(default)]
    pub include_resolved: bool,
    #[serde(default = "default_candidate_limit")]
    pub candidate_limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatInput {
    pub attachment_id: Option<String>,
    pub agent_id: Option<String>,
    pub namespace: Option<String>,
    pub context_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextRead {
    pub context: ContextRecord,
    pub objective: String,
    pub pack: ContextPack,
    pub latest_snapshot_id: Option<String>,
    pub organism: serde_json::Value,
    pub recall: ContinuityRecall,
    pub working_state: Vec<ContinuityItemRecord>,
    pub work_claims: Vec<ContinuityItemRecord>,
    pub coordination_signals: Vec<ContinuityItemRecord>,
    pub decisions: Vec<ContinuityItemRecord>,
    pub constraints: Vec<ContinuityItemRecord>,
    pub hypotheses: Vec<ContinuityItemRecord>,
    pub incidents: Vec<ContinuityItemRecord>,
    pub operational_scars: Vec<ContinuityItemRecord>,
    pub outcomes: Vec<ContinuityItemRecord>,
    pub lessons: Vec<ContinuityItemRecord>,
    pub current_practice: PracticeView,
    pub learning: LearningView,
    pub signals: Vec<ContinuityItemRecord>,
    pub open_threads: Vec<ContinuityItemRecord>,
    pub rationale: serde_json::Value,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PracticeLifecycleState {
    Current,
    Aging,
    Stale,
    Retired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PracticeView {
    pub summary: String,
    pub items: Vec<ContinuityItemRecord>,
    #[serde(default)]
    pub evidence: Vec<PracticeEvidenceRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PracticeEvidenceRecord {
    pub practice_id: String,
    pub support_signal: f64,
    pub evidence_count: usize,
    pub evidence: Vec<ContinuityItemRecord>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LearningViewMode {
    Recent,
    Lineage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningView {
    pub mode: LearningViewMode,
    pub summary: String,
    pub items: Vec<ContinuityItemRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuityRecallItem {
    pub id: String,
    pub memory_id: String,
    pub kind: ContinuityKind,
    pub status: ContinuityStatus,
    pub title: String,
    pub preview: String,
    pub author_agent_id: String,
    pub updated_at: DateTime<Utc>,
    pub effective_salience: f64,
    pub support_count: usize,
    pub score: f64,
    pub why: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContinuityRecallCompiler {
    #[serde(default)]
    pub compiled_hit_count: usize,
    #[serde(default)]
    pub lexical_fallback_count: usize,
    #[serde(default)]
    pub stale_debris_demoted_count: usize,
    #[serde(default)]
    pub priority_seed_count: usize,
    #[serde(default)]
    pub operational_seed_count: usize,
    #[serde(default)]
    pub recent_update_seed_count: usize,
    #[serde(default)]
    pub active_thread_seed_count: usize,
    pub dominant_band: Option<String>,
    #[serde(default)]
    pub band_hit_counts: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuityRecall {
    pub query: String,
    pub summary: String,
    pub answer_hint: Option<String>,
    pub total_candidates: usize,
    pub timings_ms: serde_json::Value,
    #[serde(default)]
    pub compiler: ContinuityRecallCompiler,
    pub items: Vec<ContinuityRecallItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuityCompilerStateRecord {
    pub context_id: String,
    pub dirty: bool,
    pub item_count: usize,
    pub refreshed_at: DateTime<Utc>,
    pub compiled_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuityCompiledChunkRecord {
    pub chunk_id: String,
    pub band: String,
    pub compiled_at: DateTime<Utc>,
    pub item_count: usize,
    pub item_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallInput {
    pub context_id: Option<String>,
    pub namespace: Option<String>,
    pub task_id: Option<String>,
    pub objective: String,
    #[serde(default)]
    pub include_resolved: bool,
    #[serde(default = "default_candidate_limit")]
    pub candidate_limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalInput {
    pub context_id: String,
    pub agent_id: String,
    pub title: String,
    pub body: String,
    #[serde(default)]
    pub dimensions: Vec<DimensionValue>,
    #[serde(default)]
    pub extra: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CoordinationProjectedLane {
    pub projection_id: String,
    pub projection_kind: String,
    pub label: String,
    pub resource: Option<String>,
    pub repo_root: Option<String>,
    pub branch: Option<String>,
    pub task_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationSignalInput {
    pub context_id: String,
    pub agent_id: String,
    pub title: String,
    pub body: String,
    pub lane: CoordinationLane,
    pub target_agent_id: Option<String>,
    pub target_projected_lane: Option<CoordinationProjectedLane>,
    pub claim_id: Option<String>,
    pub resource: Option<String>,
    pub severity: Option<CoordinationSeverity>,
    #[serde(default)]
    pub projection_ids: Vec<String>,
    #[serde(default)]
    pub projected_lanes: Vec<CoordinationProjectedLane>,
    #[serde(default)]
    pub extra: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimWorkInput {
    pub context_id: String,
    pub agent_id: String,
    pub title: String,
    pub body: String,
    pub scope: Scope,
    #[serde(default)]
    pub resources: Vec<String>,
    #[serde(default)]
    pub exclusive: bool,
    pub attachment_id: Option<String>,
    pub lease_seconds: Option<u64>,
    #[serde(default)]
    pub extra: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct WorkClaimConflict {
    pub id: String,
    pub agent_id: String,
    pub title: String,
    #[serde(default)]
    pub resources: Vec<String>,
    #[serde(default)]
    pub exclusive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct WorkClaimCoordination {
    #[serde(default)]
    pub claim_key: String,
    #[serde(default)]
    pub resources: Vec<String>,
    #[serde(default)]
    pub exclusive: bool,
    pub attachment_id: Option<String>,
    #[serde(default = "super::helpers::default_work_claim_lease_seconds")]
    pub lease_seconds: u64,
    pub lease_expires_at: Option<DateTime<Utc>>,
    pub renewed_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub conflict_count: usize,
    #[serde(default)]
    pub conflicts_with: Vec<WorkClaimConflict>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotInput {
    pub context_id: Option<String>,
    pub namespace: Option<String>,
    pub task_id: Option<String>,
    pub objective: Option<String>,
    pub selector: Option<Selector>,
    #[serde(default = "default_snapshot_resolution")]
    pub resolution: SnapshotResolution,
    #[serde(default = "default_token_budget")]
    pub token_budget: usize,
    #[serde(default = "default_candidate_limit")]
    pub candidate_limit: usize,
    pub owner_agent_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotRecord {
    pub id: String,
    pub context_id: String,
    pub created_at: DateTime<Utc>,
    pub resolution: SnapshotResolution,
    pub objective: String,
    pub selector: Selector,
    pub view_id: String,
    pub pack_id: String,
    pub manifest_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotManifest {
    pub snapshot: SnapshotRecord,
    pub context: ContextRecord,
    pub view: ViewManifest,
    pub pack: ContextPackManifest,
    pub continuity: Vec<ContinuityItemRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeInput {
    pub snapshot_id: Option<String>,
    pub context_id: Option<String>,
    pub namespace: Option<String>,
    pub task_id: Option<String>,
    pub objective: String,
    #[serde(default = "default_token_budget")]
    pub token_budget: usize,
    #[serde(default = "default_candidate_limit")]
    pub candidate_limit: usize,
    pub agent_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeRecord {
    pub snapshot: Option<SnapshotRecord>,
    pub context: ContextRead,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuityHandoffInput {
    pub from_agent_id: String,
    pub to_agent_id: String,
    pub context_id: Option<String>,
    pub namespace: Option<String>,
    pub task_id: Option<String>,
    pub objective: String,
    pub reason: String,
    pub selector: Option<Selector>,
    #[serde(default = "default_snapshot_resolution")]
    pub resolution: SnapshotResolution,
    #[serde(default = "default_token_budget")]
    pub token_budget: usize,
    #[serde(default = "default_candidate_limit")]
    pub candidate_limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuityHandoffRecord {
    pub handoff: HandoffRecord,
    pub snapshot: SnapshotRecord,
    pub context: ContextRead,
    pub proof: HandoffProof,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffProofRegister {
    pub label: String,
    pub register_kind: String,
    pub source_id: String,
    pub title: String,
    pub body: String,
    pub has_provenance: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HandoffProof {
    pub digest: String,
    #[serde(default)]
    pub registers: Vec<HandoffProofRegister>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeInput {
    pub context_id: String,
    pub agent_id: String,
    pub title: String,
    pub result: String,
    pub quality: f64,
    #[serde(default)]
    pub pack_id: Option<String>,
    #[serde(default)]
    pub used_memory_ids: Vec<String>,
    #[serde(default)]
    pub confirmed_memory_ids: Vec<String>,
    #[serde(default)]
    pub contradicted_memory_ids: Vec<String>,
    #[serde(default)]
    pub failures: Vec<String>,
    #[serde(default)]
    pub dimensions: Vec<DimensionValue>,
    #[serde(default)]
    pub extra: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolveOrSupersedeInput {
    pub continuity_id: String,
    pub actor_agent_id: String,
    pub new_status: ContinuityStatus,
    pub supersedes_id: Option<String>,
    pub resolution_note: Option<String>,
    #[serde(default)]
    pub extra: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEventInput {
    pub context_id: Option<String>,
    pub namespace: Option<String>,
    pub task_id: Option<String>,
    pub agent_id: String,
    pub level: String,
    pub message: String,
    #[serde(default)]
    pub attributes: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExplainTarget {
    Context { id: String },
    ContinuityItem { id: String },
    Snapshot { id: String },
    Handoff { id: String },
    Pack { id: String },
    View { id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct CoordinationSignalRecord {
    pub lane: String,
    pub severity: String,
    pub target_agent_id: Option<String>,
    pub target_projected_lane: Option<CoordinationProjectedLane>,
    pub claim_id: Option<String>,
    pub resource: Option<String>,
    #[serde(default)]
    pub projection_ids: Vec<String>,
    #[serde(default)]
    pub projected_lanes: Vec<CoordinationProjectedLane>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct ClaimConflictLaneSummary {
    pub claim_id: String,
    pub agent_id: String,
    pub title: String,
    pub display_name: Option<String>,
    pub projection_id: String,
    pub projection_kind: String,
    pub label: String,
    pub resource: Option<String>,
    pub repo_root: Option<String>,
    pub branch: Option<String>,
    pub task_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(super) struct ClaimConflictSummary {
    pub resource: String,
    #[serde(default)]
    pub claim_ids: Vec<String>,
    #[serde(default)]
    pub agents: Vec<String>,
    #[serde(default)]
    pub titles: Vec<String>,
    #[serde(default)]
    pub projection_ids: Vec<String>,
    #[serde(default)]
    pub projected_lanes: Vec<ClaimConflictLaneSummary>,
}

impl ClaimConflictSummary {
    pub fn new(resource: String) -> Self {
        Self {
            resource,
            ..Self::default()
        }
    }

    pub fn absorb_claim(
        &mut self,
        item: &ContinuityItemRecord,
        projected_lane: Option<ClaimConflictLaneSummary>,
    ) {
        if !self.claim_ids.iter().any(|claim_id| claim_id == &item.id) {
            self.claim_ids.push(item.id.clone());
        }
        if !self
            .agents
            .iter()
            .any(|agent_id| agent_id == &item.author_agent_id)
        {
            self.agents.push(item.author_agent_id.clone());
        }
        if !self.titles.iter().any(|title| title == &item.title) {
            self.titles.push(item.title.clone());
        }
        if let Some(projected_lane) = projected_lane {
            if !self
                .projection_ids
                .iter()
                .any(|projection_id| projection_id == &projected_lane.projection_id)
            {
                self.projection_ids
                    .push(projected_lane.projection_id.clone());
            }
            if !self
                .projected_lanes
                .iter()
                .any(|lane| lane.claim_id == projected_lane.claim_id)
            {
                self.projected_lanes.push(projected_lane);
            }
        }
    }
}
