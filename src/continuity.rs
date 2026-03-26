use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::dispatch;
use crate::engine::Engine;
use crate::model::{
    ContextPack, ContextPackManifest, DimensionValue, EventInput, EventKind, HandoffRecord,
    IngestManifest, MemoryLayer, QueryInput, ReplayRow, Scope, Selector, SnapshotResolution,
    SubscriptionInput, SubscriptionRecord, ViewInput, ViewManifest, ViewOp,
};
use crate::query::build_context_pack;

pub const MACHINE_NAMESPACE_ALIAS: &str = "@machine";
pub const DEFAULT_MACHINE_TASK_ID: &str = "machine-organism";

const PROOF_TRIM_LIMIT: usize = 220;
const DEFAULT_DIMENSION_WEIGHT: i32 = 100;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContinuityKind {
    WorkingState,
    WorkClaim,
    Derivation,
    Fact,
    Decision,
    Constraint,
    Hypothesis,
    Incident,
    OperationalScar,
    Outcome,
    Signal,
    Summary,
    Lesson,
}

impl ContinuityKind {
    pub fn default_layer(self) -> MemoryLayer {
        match self {
            Self::WorkingState | Self::WorkClaim | Self::Signal => MemoryLayer::Hot,
            Self::Derivation | Self::Fact | Self::Decision | Self::Constraint => {
                MemoryLayer::Semantic
            }
            Self::Hypothesis | Self::Incident | Self::Outcome | Self::Lesson => {
                MemoryLayer::Episodic
            }
            Self::OperationalScar => MemoryLayer::Semantic,
            Self::Summary => MemoryLayer::Summary,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::WorkingState => "working_state",
            Self::WorkClaim => "work_claim",
            Self::Derivation => "derivation",
            Self::Fact => "fact",
            Self::Decision => "decision",
            Self::Constraint => "constraint",
            Self::Hypothesis => "hypothesis",
            Self::Incident => "incident",
            Self::OperationalScar => "operational_scar",
            Self::Outcome => "outcome",
            Self::Signal => "signal",
            Self::Summary => "summary",
            Self::Lesson => "lesson",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContinuityStatus {
    Open,
    Active,
    Resolved,
    Superseded,
    Rejected,
}

impl ContinuityStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Active => "active",
            Self::Resolved => "resolved",
            Self::Superseded => "superseded",
            Self::Rejected => "rejected",
        }
    }

    pub fn is_open(self) -> bool {
        matches!(self, Self::Open | Self::Active)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextStatus {
    Open,
    Paused,
    Closed,
}

impl ContextStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Paused => "paused",
            Self::Closed => "closed",
        }
    }
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
    pub event: EventInput,
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
    pub signals: Vec<ContinuityItemRecord>,
    pub open_threads: Vec<ContinuityItemRecord>,
    pub rationale: serde_json::Value,
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
    pub priority_seed_count: usize,
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CoordinationLane {
    Anxiety,
    Review,
    Warning,
    Backoff,
    Coach,
}

impl CoordinationLane {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Anxiety => "anxiety",
            Self::Review => "review",
            Self::Warning => "warning",
            Self::Backoff => "backoff",
            Self::Coach => "coach",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CoordinationSeverity {
    Info,
    Warn,
    Block,
}

impl CoordinationSeverity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Block => "block",
        }
    }
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
    #[serde(default = "default_work_claim_lease_seconds")]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum UciRequest {
    IdentifyMachine,
    AttachAgent { input: AttachAgentInput },
    UpsertAgentBadge { input: UpsertAgentBadgeInput },
    Heartbeat { input: HeartbeatInput },
    OpenContext { input: OpenContextInput },
    ReadContext { input: ReadContextInput },
    Recall { input: RecallInput },
    WriteEvents { inputs: Vec<WriteEventInput> },
    WriteDerivations { inputs: Vec<ContinuityItemInput> },
    ClaimWork { input: ClaimWorkInput },
    PublishCoordinationSignal { input: CoordinationSignalInput },
    PublishSignal { input: SignalInput },
    Subscribe { input: SubscriptionInput },
    Handoff { input: ContinuityHandoffInput },
    Snapshot { input: SnapshotInput },
    Resume { input: ResumeInput },
    Explain { target: ExplainTarget },
    Replay { selector: Selector, limit: usize },
    RecordOutcome { input: OutcomeInput },
    MarkDecision { input: ContinuityItemInput },
    MarkConstraint { input: ContinuityItemInput },
    MarkHypothesis { input: ContinuityItemInput },
    MarkIncident { input: ContinuityItemInput },
    MarkOperationalScar { input: ContinuityItemInput },
    ResolveOrSupersede { input: ResolveOrSupersedeInput },
    EmitTelemetry { input: TelemetryEventInput },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", content = "data", rename_all = "snake_case")]
pub enum UciResponse {
    MachineProfile(MachineProfile),
    AgentAttachment(AgentAttachmentRecord),
    AgentBadge(AgentBadgeRecord),
    Context(ContextRecord),
    ContextRead(ContextRead),
    Recall(ContinuityRecall),
    IngestManifests(Vec<IngestManifest>),
    ContinuityItems(Vec<ContinuityItemRecord>),
    Subscription(SubscriptionRecord),
    Handoff(ContinuityHandoffRecord),
    Snapshot(SnapshotRecord),
    Resume(ResumeRecord),
    Explain(serde_json::Value),
    Replay(Vec<ReplayRow>),
    Telemetry(serde_json::Value),
}

pub trait UnifiedContinuityInterface {
    fn identify_machine(&self) -> Result<MachineProfile>;
    fn attach_agent(&self, input: AttachAgentInput) -> Result<AgentAttachmentRecord>;
    fn upsert_agent_badge(&self, input: UpsertAgentBadgeInput) -> Result<AgentBadgeRecord>;
    fn heartbeat(&self, input: HeartbeatInput) -> Result<AgentAttachmentRecord>;
    fn open_context(&self, input: OpenContextInput) -> Result<ContextRecord>;
    fn read_context(&self, input: ReadContextInput) -> Result<ContextRead>;
    fn recall(&self, input: RecallInput) -> Result<ContinuityRecall>;
    fn write_events(&self, inputs: Vec<WriteEventInput>) -> Result<Vec<IngestManifest>>;
    fn write_derivations(
        &self,
        inputs: Vec<ContinuityItemInput>,
    ) -> Result<Vec<ContinuityItemRecord>>;
    fn claim_work(&self, input: ClaimWorkInput) -> Result<ContinuityItemRecord>;
    fn publish_coordination_signal(
        &self,
        input: CoordinationSignalInput,
    ) -> Result<ContinuityItemRecord>;
    fn publish_signal(&self, input: SignalInput) -> Result<ContinuityItemRecord>;
    fn subscribe(&self, input: SubscriptionInput) -> Result<SubscriptionRecord>;
    fn handoff(&self, input: ContinuityHandoffInput) -> Result<ContinuityHandoffRecord>;
    fn snapshot(&self, input: SnapshotInput) -> Result<SnapshotRecord>;
    fn resume(&self, input: ResumeInput) -> Result<ResumeRecord>;
    fn explain(&self, target: ExplainTarget) -> Result<serde_json::Value>;
    fn replay_selector(&self, selector: Selector, limit: usize) -> Result<Vec<ReplayRow>>;
    fn record_outcome(&self, input: OutcomeInput) -> Result<ContinuityItemRecord>;
    fn mark_decision(&self, input: ContinuityItemInput) -> Result<ContinuityItemRecord>;
    fn mark_constraint(&self, input: ContinuityItemInput) -> Result<ContinuityItemRecord>;
    fn mark_hypothesis(&self, input: ContinuityItemInput) -> Result<ContinuityItemRecord>;
    fn mark_incident(&self, input: ContinuityItemInput) -> Result<ContinuityItemRecord>;
    fn mark_operational_scar(&self, input: ContinuityItemInput) -> Result<ContinuityItemRecord>;
    fn resolve_or_supersede(&self, input: ResolveOrSupersedeInput) -> Result<ContinuityItemRecord>;
    fn emit_telemetry(&self, input: TelemetryEventInput) -> Result<serde_json::Value>;

    fn handle_request(&self, request: UciRequest) -> Result<UciResponse> {
        Ok(match request {
            UciRequest::IdentifyMachine => UciResponse::MachineProfile(self.identify_machine()?),
            UciRequest::AttachAgent { input } => {
                UciResponse::AgentAttachment(self.attach_agent(input)?)
            }
            UciRequest::UpsertAgentBadge { input } => {
                UciResponse::AgentBadge(self.upsert_agent_badge(input)?)
            }
            UciRequest::Heartbeat { input } => UciResponse::AgentAttachment(self.heartbeat(input)?),
            UciRequest::OpenContext { input } => UciResponse::Context(self.open_context(input)?),
            UciRequest::ReadContext { input } => {
                UciResponse::ContextRead(self.read_context(input)?)
            }
            UciRequest::Recall { input } => UciResponse::Recall(self.recall(input)?),
            UciRequest::WriteEvents { inputs } => {
                UciResponse::IngestManifests(self.write_events(inputs)?)
            }
            UciRequest::WriteDerivations { inputs } => {
                UciResponse::ContinuityItems(self.write_derivations(inputs)?)
            }
            UciRequest::ClaimWork { input } => {
                UciResponse::ContinuityItems(vec![self.claim_work(input)?])
            }
            UciRequest::PublishCoordinationSignal { input } => {
                UciResponse::ContinuityItems(vec![self.publish_coordination_signal(input)?])
            }
            UciRequest::PublishSignal { input } => {
                UciResponse::ContinuityItems(vec![self.publish_signal(input)?])
            }
            UciRequest::Subscribe { input } => UciResponse::Subscription(self.subscribe(input)?),
            UciRequest::Handoff { input } => UciResponse::Handoff(self.handoff(input)?),
            UciRequest::Snapshot { input } => UciResponse::Snapshot(self.snapshot(input)?),
            UciRequest::Resume { input } => UciResponse::Resume(self.resume(input)?),
            UciRequest::Explain { target } => UciResponse::Explain(self.explain(target)?),
            UciRequest::Replay { selector, limit } => {
                UciResponse::Replay(self.replay_selector(selector, limit)?)
            }
            UciRequest::RecordOutcome { input } => {
                UciResponse::ContinuityItems(vec![self.record_outcome(input)?])
            }
            UciRequest::MarkDecision { input } => {
                UciResponse::ContinuityItems(vec![self.mark_decision(input)?])
            }
            UciRequest::MarkConstraint { input } => {
                UciResponse::ContinuityItems(vec![self.mark_constraint(input)?])
            }
            UciRequest::MarkHypothesis { input } => {
                UciResponse::ContinuityItems(vec![self.mark_hypothesis(input)?])
            }
            UciRequest::MarkIncident { input } => {
                UciResponse::ContinuityItems(vec![self.mark_incident(input)?])
            }
            UciRequest::MarkOperationalScar { input } => {
                UciResponse::ContinuityItems(vec![self.mark_operational_scar(input)?])
            }
            UciRequest::ResolveOrSupersede { input } => {
                UciResponse::ContinuityItems(vec![self.resolve_or_supersede(input)?])
            }
            UciRequest::EmitTelemetry { input } => {
                UciResponse::Telemetry(self.emit_telemetry(input)?)
            }
        })
    }
}

pub type SharedContinuityKernel = Engine;

impl UnifiedContinuityInterface for Engine {
    fn identify_machine(&self) -> Result<MachineProfile> {
        self.with_storage(|storage, _| storage.machine_profile())
    }

    fn attach_agent(&self, input: AttachAgentInput) -> Result<AgentAttachmentRecord> {
        self.with_storage(|storage, _| {
            let mut input = input;
            input.namespace = resolve_namespace(storage, Some(input.namespace))?
                .expect("namespace should remain present");
            storage.attach_agent(input)
        })
    }

    fn upsert_agent_badge(&self, input: UpsertAgentBadgeInput) -> Result<AgentBadgeRecord> {
        self.with_storage(|storage, _| {
            let mut input = input;
            input.namespace = resolve_namespace(storage, input.namespace)?;
            storage.upsert_agent_badge(input)
        })
    }

    fn heartbeat(&self, input: HeartbeatInput) -> Result<AgentAttachmentRecord> {
        self.with_storage(|storage, _| {
            let mut input = input;
            input.namespace = resolve_namespace(storage, input.namespace)?;
            storage.heartbeat_attachment(input)
        })
    }

    fn open_context(&self, input: OpenContextInput) -> Result<ContextRecord> {
        self.with_storage(|storage, _| {
            let mut input = input;
            input.namespace = resolve_namespace(storage, Some(input.namespace))?
                .expect("namespace should remain present");
            input.selector = resolve_selector_namespace(storage, input.selector)?;
            storage.open_context(input)
        })
    }

    fn read_context(&self, input: ReadContextInput) -> Result<ContextRead> {
        self.with_storage(|storage, _| {
            let namespace = resolve_namespace(storage, input.namespace.clone())?;
            let context = storage.resolve_context(
                input.context_id.as_deref(),
                namespace.as_deref(),
                input.task_id.as_deref(),
            )?;
            let selector = merge_context_selector(
                &context,
                resolve_selector_namespace(storage, input.selector.clone())?,
            );
            let continuity = storage.list_continuity_items(&context.id, input.include_resolved)?;
            let recall = storage.recall_continuity(
                &context.id,
                &input.objective,
                input.include_resolved,
                input.candidate_limit.min(8).max(4),
            )?;
            let pack = build_context_pack(
                storage,
                QueryInput {
                    agent_id: input.agent_id.clone(),
                    session_id: input
                        .session_id
                        .clone()
                        .or_else(|| Some(context.session_id.clone())),
                    task_id: Some(context.task_id.clone()),
                    namespace: Some(context.namespace.clone()),
                    objective: Some(input.objective.clone()),
                    selector: Some(selector.clone()),
                    view_id: input.view_id.clone(),
                    query_text: input.objective.clone(),
                    budget_tokens: input.token_budget,
                    candidate_limit: input.candidate_limit,
                },
            )?;
            if let Some(agent_id) = input.agent_id.as_deref() {
                storage.touch_active_attachment(
                    agent_id,
                    Some(context.namespace.as_str()),
                    Some(context.id.as_str()),
                )?;
            }
            let latest_snapshot_id = storage.latest_snapshot_id(&context.id)?;
            let pack_item_count = pack.items.len();
            let now = Utc::now();
            let agent_badges = storage.list_agent_badges(Some(context.namespace.as_str()), None)?;
            let mut lane_projections =
                storage.list_lane_projections(Some(context.namespace.as_str()), None)?;
            let dispatch_state =
                dispatch::organism_snapshot(&storage.config.root, Some(context.namespace.as_str()));
            merge_dispatch_worker_lane_projections(
                &mut lane_projections,
                &dispatch_state,
                context.namespace.as_str(),
            );
            merge_dispatch_assignment_pressure(
                &mut lane_projections,
                &dispatch_state,
                context.namespace.as_str(),
            );
            let mut organism = organism_state(&continuity, now, &agent_badges, &lane_projections);
            if let Some(object) = organism.as_object_mut() {
                object.insert(
                    "agent_badges".to_string(),
                    serde_json::to_value(&agent_badges)?,
                );
                object.insert(
                    "lane_projections".to_string(),
                    serde_json::to_value(&lane_projections)?,
                );
                object.insert(
                    "dispatch".to_string(),
                    serde_json::to_value(&dispatch_state)?,
                );
            }
            Ok(ContextRead {
                context,
                objective: input.objective,
                pack,
                latest_snapshot_id,
                organism,
                recall: recall.clone(),
                working_state: filter_kind(&continuity, ContinuityKind::WorkingState),
                work_claims: continuity
                    .iter()
                    .filter(|item| work_claim_is_live(item, now))
                    .cloned()
                    .collect(),
                coordination_signals: continuity
                    .iter()
                    .filter(|item| coordination_signal(item).is_some())
                    .cloned()
                    .collect(),
                decisions: filter_kind(&continuity, ContinuityKind::Decision),
                constraints: filter_kind(&continuity, ContinuityKind::Constraint),
                hypotheses: filter_kind(&continuity, ContinuityKind::Hypothesis),
                incidents: filter_kind(&continuity, ContinuityKind::Incident),
                operational_scars: filter_kind(&continuity, ContinuityKind::OperationalScar),
                outcomes: filter_kind(&continuity, ContinuityKind::Outcome),
                signals: filter_kind(&continuity, ContinuityKind::Signal),
                open_threads: continuity
                    .iter()
                    .filter(|item| counts_as_open_thread(item, now))
                    .cloned()
                    .collect(),
                rationale: serde_json::json!({
                    "selector": selector,
                    "continuity_count": continuity.len(),
                    "pack_item_count": pack_item_count,
                    "continuity_recall_count": recall.items.len(),
                    "continuity_recall_timings_ms": recall.timings_ms,
                    "continuity_recall_compiler": recall.compiler,
                    "continuity_recall_top_why": recall.items.first().map(|item| item.why.clone()).unwrap_or_default(),
                }),
            })
        })
    }

    fn recall(&self, input: RecallInput) -> Result<ContinuityRecall> {
        self.with_storage(|storage, _| {
            let namespace = resolve_namespace(storage, input.namespace.clone())?;
            let context = storage.resolve_context(
                input.context_id.as_deref(),
                namespace.as_deref(),
                input.task_id.as_deref(),
            )?;
            storage.recall_continuity(
                &context.id,
                &input.objective,
                input.include_resolved,
                input.candidate_limit,
            )
        })
    }

    fn write_events(&self, inputs: Vec<WriteEventInput>) -> Result<Vec<IngestManifest>> {
        self.with_storage_mut(|storage, telemetry| {
            let mut manifests = Vec::new();
            for mut input in inputs {
                if let Some(context_id) = &input.context_id {
                    let context = storage.get_context(context_id)?;
                    inject_context(&mut input.event, &context);
                } else {
                    input.event.namespace =
                        resolve_namespace(storage, input.event.namespace.clone())?;
                }
                manifests.push(storage.ingest(input.event, telemetry)?);
            }
            Ok(manifests)
        })
    }

    fn write_derivations(
        &self,
        inputs: Vec<ContinuityItemInput>,
    ) -> Result<Vec<ContinuityItemRecord>> {
        self.with_storage(|storage, _| {
            inputs
                .into_iter()
                .map(|input| storage.persist_continuity_item(input))
                .collect()
        })
    }

    fn claim_work(&self, input: ClaimWorkInput) -> Result<ContinuityItemRecord> {
        self.with_storage(|storage, _| storage.claim_work(input))
    }

    fn publish_coordination_signal(
        &self,
        input: CoordinationSignalInput,
    ) -> Result<ContinuityItemRecord> {
        let severity = input
            .severity
            .unwrap_or_else(|| default_coordination_severity(input.lane));
        let mut dimensions = vec![
            DimensionValue {
                key: "signal.lane".to_string(),
                value: input.lane.as_str().to_string(),
                weight: DEFAULT_DIMENSION_WEIGHT,
            },
            DimensionValue {
                key: "signal.severity".to_string(),
                value: severity.as_str().to_string(),
                weight: DEFAULT_DIMENSION_WEIGHT,
            },
        ];
        if let Some(target_agent_id) = &input.target_agent_id {
            dimensions.push(DimensionValue {
                key: "signal.target_agent".to_string(),
                value: target_agent_id.clone(),
                weight: DEFAULT_DIMENSION_WEIGHT,
            });
        }
        if let Some(claim_id) = &input.claim_id {
            dimensions.push(DimensionValue {
                key: "claim.id".to_string(),
                value: claim_id.clone(),
                weight: DEFAULT_DIMENSION_WEIGHT,
            });
        }
        if let Some(resource) = &input.resource {
            dimensions.push(DimensionValue {
                key: "claim.resource".to_string(),
                value: resource.clone(),
                weight: DEFAULT_DIMENSION_WEIGHT,
            });
        }
        self.publish_signal(SignalInput {
            context_id: input.context_id,
            agent_id: input.agent_id,
            title: input.title,
            body: input.body,
            dimensions,
            extra: merge_coordination_signal_extra(
                input.extra,
                input.lane,
                severity,
                input.target_agent_id,
                input.target_projected_lane,
                input.claim_id,
                input.resource,
                input.projection_ids,
                input.projected_lanes,
            ),
        })
    }

    fn publish_signal(&self, input: SignalInput) -> Result<ContinuityItemRecord> {
        self.with_storage(|storage, _| {
            storage.persist_continuity_item(ContinuityItemInput {
                context_id: input.context_id,
                author_agent_id: input.agent_id,
                kind: ContinuityKind::Signal,
                title: input.title,
                body: input.body,
                scope: Scope::Shared,
                status: Some(ContinuityStatus::Active),
                importance: Some(0.9),
                confidence: Some(0.9),
                salience: Some(0.95),
                layer: Some(MemoryLayer::Hot),
                supports: Vec::new(),
                dimensions: input.dimensions,
                extra: input.extra,
            })
        })
    }

    fn subscribe(&self, input: SubscriptionInput) -> Result<SubscriptionRecord> {
        self.with_storage(|storage, _| storage.create_subscription(input))
    }

    fn handoff(&self, input: ContinuityHandoffInput) -> Result<ContinuityHandoffRecord> {
        let machine = self.identify_machine()?;
        let namespace = match input.namespace.clone() {
            Some(namespace) => {
                self.with_storage(|storage, _| resolve_namespace(storage, Some(namespace)))?
            }
            None => Some(machine.namespace.clone()),
        };
        let task_id = input
            .task_id
            .clone()
            .unwrap_or_else(|| machine.default_task_id.clone());
        let selector = self.with_storage(|storage, _| {
            resolve_selector_namespace(storage, input.selector.clone())
        })?;
        let context = self.open_context(OpenContextInput {
            namespace: namespace
                .clone()
                .expect("handoff namespace should stay present"),
            task_id,
            session_id: format!("handoff-{}", Uuid::now_v7()),
            objective: input.objective.clone(),
            selector: selector.clone(),
            agent_id: Some(input.from_agent_id.clone()),
            attachment_id: None,
        })?;
        let snapshot = self.snapshot(SnapshotInput {
            context_id: Some(
                input
                    .context_id
                    .clone()
                    .unwrap_or_else(|| context.id.clone()),
            ),
            namespace: None,
            task_id: None,
            objective: Some(input.objective.clone()),
            selector: selector.clone(),
            resolution: input.resolution,
            token_budget: input.token_budget,
            candidate_limit: input.candidate_limit,
            owner_agent_id: Some(input.from_agent_id.clone()),
        })?;
        let context_read = self.read_context(ReadContextInput {
            context_id: Some(snapshot.context_id.clone()),
            namespace: None,
            task_id: None,
            objective: input.objective.clone(),
            token_budget: input.token_budget,
            selector,
            agent_id: Some(input.to_agent_id.clone()),
            session_id: None,
            view_id: Some(snapshot.view_id.clone()),
            include_resolved: false,
            candidate_limit: input.candidate_limit,
        })?;
        let proof = compile_handoff_proof(&context_read);
        let handoff = self.with_storage(|storage, _| {
            let record = HandoffRecord {
                id: format!("handoff:{}", Uuid::now_v7()),
                created_at: Utc::now(),
                from_agent_id: input.from_agent_id.clone(),
                to_agent_id: input.to_agent_id.clone(),
                reason: input.reason.clone(),
                view_id: snapshot.view_id.clone(),
                pack_id: snapshot.pack_id.clone(),
                conflict_count: 0,
                manifest_path: storage
                    .paths
                    .debug_dir
                    .join("handoffs")
                    .join(format!("{}.json", Uuid::now_v7()))
                    .display()
                    .to_string(),
            };
            let pack_manifest = storage.explain_context_pack(&snapshot.pack_id)?;
            let snapshot_manifest = storage.explain_snapshot(&snapshot.id)?;
            storage.persist_handoff(
                &record,
                &serde_json::json!({
                    "handoff": record.clone(),
                    "reason": input.reason,
                    "snapshot": snapshot_manifest,
                    "pack": pack_manifest,
                    "context": context_read,
                    "proof": proof,
                }),
                &input.objective,
            )?;
            storage.get_handoff(&record.id)
        })?;
        Ok(ContinuityHandoffRecord {
            handoff,
            snapshot,
            context: context_read,
            proof,
        })
    }

    fn snapshot(&self, input: SnapshotInput) -> Result<SnapshotRecord> {
        self.with_storage(|storage, _| {
            let namespace = resolve_namespace(storage, input.namespace.clone())?;
            let context = storage.resolve_context(
                input.context_id.as_deref(),
                namespace.as_deref(),
                input.task_id.as_deref(),
            )?;
            let objective = input
                .objective
                .clone()
                .unwrap_or_else(|| context.objective.clone());
            let selector = merge_context_selector(
                &context,
                resolve_selector_namespace(storage, input.selector.clone())?,
            );
            let view = storage.materialize_view(ViewInput {
                op: ViewOp::Snapshot,
                owner_agent_id: input.owner_agent_id.clone(),
                namespace: Some(context.namespace.clone()),
                objective: Some(objective.clone()),
                selectors: vec![selector.clone()],
                source_view_ids: Vec::new(),
                resolution: Some(input.resolution),
                limit: Some(input.candidate_limit.max(1)),
            })?;
            let pack = build_context_pack(
                storage,
                QueryInput {
                    agent_id: input.owner_agent_id.clone(),
                    session_id: Some(context.session_id.clone()),
                    task_id: Some(context.task_id.clone()),
                    namespace: Some(context.namespace.clone()),
                    objective: Some(objective.clone()),
                    selector: Some(selector.clone()),
                    view_id: Some(view.id.clone()),
                    query_text: objective.clone(),
                    budget_tokens: input.token_budget,
                    candidate_limit: input.candidate_limit,
                },
            )?;
            storage.persist_snapshot(
                &context,
                selector,
                objective,
                input.resolution,
                &view.id,
                &pack.id,
            )
        })
    }

    fn resume(&self, input: ResumeInput) -> Result<ResumeRecord> {
        let snapshot = if let Some(snapshot_id) = &input.snapshot_id {
            Some(self.with_storage(|storage, _| storage.get_snapshot(snapshot_id))?)
        } else {
            None
        };
        let namespace =
            self.with_storage(|storage, _| resolve_namespace(storage, input.namespace.clone()))?;
        let context = self.read_context(ReadContextInput {
            context_id: input
                .context_id
                .clone()
                .or_else(|| snapshot.as_ref().map(|item| item.context_id.clone())),
            namespace,
            task_id: input.task_id.clone(),
            objective: input.objective,
            token_budget: input.token_budget,
            selector: snapshot.as_ref().map(|item| item.selector.clone()),
            agent_id: input.agent_id,
            session_id: None,
            view_id: snapshot.as_ref().map(|item| item.view_id.clone()),
            include_resolved: false,
            candidate_limit: input.candidate_limit,
        })?;
        Ok(ResumeRecord { snapshot, context })
    }

    fn explain(&self, target: ExplainTarget) -> Result<serde_json::Value> {
        self.with_storage(|storage, _| match target {
            ExplainTarget::Context { id } => {
                Ok(serde_json::to_value(storage.explain_context(&id)?)?)
            }
            ExplainTarget::ContinuityItem { id } => {
                Ok(serde_json::to_value(storage.explain_continuity_item(&id)?)?)
            }
            ExplainTarget::Snapshot { id } => Ok(storage.explain_snapshot(&id)?),
            ExplainTarget::Handoff { id } => storage.explain_handoff(&id),
            ExplainTarget::Pack { id } => {
                Ok(serde_json::to_value(storage.explain_context_pack(&id)?)?)
            }
            ExplainTarget::View { id } => Ok(serde_json::to_value(storage.explain_view(&id)?)?),
        })
    }

    fn replay_selector(&self, selector: Selector, limit: usize) -> Result<Vec<ReplayRow>> {
        self.with_storage(|storage, telemetry| {
            storage.replay_by_selector(telemetry, &selector, limit)
        })
    }

    fn record_outcome(&self, input: OutcomeInput) -> Result<ContinuityItemRecord> {
        self.with_storage(|storage, _| {
            storage.persist_continuity_item(ContinuityItemInput {
                context_id: input.context_id,
                author_agent_id: input.agent_id,
                kind: ContinuityKind::Outcome,
                title: input.title,
                body: input.result,
                scope: Scope::Project,
                status: Some(ContinuityStatus::Resolved),
                importance: Some(input.quality.clamp(0.0, 1.0)),
                confidence: Some(0.9),
                salience: Some(input.quality.clamp(0.0, 1.0)),
                layer: Some(MemoryLayer::Episodic),
                supports: Vec::new(),
                dimensions: augment_dimensions(
                    input.dimensions,
                    vec![DimensionValue {
                        key: "outcome_quality".to_string(),
                        value: format!("{:.2}", input.quality),
                        weight: DEFAULT_DIMENSION_WEIGHT,
                    }],
                ),
                extra: serde_json::json!({
                    "failures": input.failures,
                    "extra": input.extra,
                }),
            })
        })
    }

    fn mark_decision(&self, mut input: ContinuityItemInput) -> Result<ContinuityItemRecord> {
        input.kind = ContinuityKind::Decision;
        self.with_storage(|storage, _| storage.persist_continuity_item(input))
    }

    fn mark_constraint(&self, mut input: ContinuityItemInput) -> Result<ContinuityItemRecord> {
        input.kind = ContinuityKind::Constraint;
        self.with_storage(|storage, _| storage.persist_continuity_item(input))
    }

    fn mark_hypothesis(&self, mut input: ContinuityItemInput) -> Result<ContinuityItemRecord> {
        input.kind = ContinuityKind::Hypothesis;
        self.with_storage(|storage, _| storage.persist_continuity_item(input))
    }

    fn mark_incident(&self, mut input: ContinuityItemInput) -> Result<ContinuityItemRecord> {
        input.kind = ContinuityKind::Incident;
        self.with_storage(|storage, _| storage.persist_continuity_item(input))
    }

    fn mark_operational_scar(
        &self,
        mut input: ContinuityItemInput,
    ) -> Result<ContinuityItemRecord> {
        input.kind = ContinuityKind::OperationalScar;
        input.salience = Some(input.salience.unwrap_or(0.95));
        input.importance = Some(input.importance.unwrap_or(0.95));
        self.with_storage(|storage, _| storage.persist_continuity_item(input))
    }

    fn resolve_or_supersede(&self, input: ResolveOrSupersedeInput) -> Result<ContinuityItemRecord> {
        self.with_storage(|storage, _| storage.resolve_continuity_item(input))
    }

    fn emit_telemetry(&self, input: TelemetryEventInput) -> Result<serde_json::Value> {
        let namespace =
            self.with_storage(|storage, _| resolve_namespace(storage, input.namespace.clone()))?;
        let event = WriteEventInput {
            context_id: input.context_id.clone(),
            event: EventInput {
                kind: EventKind::Trace,
                agent_id: input.agent_id,
                agent_role: Some("telemetry".to_string()),
                session_id: format!("telemetry-{}", Uuid::now_v7()),
                task_id: input.task_id,
                project_id: None,
                goal_id: None,
                run_id: None,
                namespace,
                environment: Some("local".to_string()),
                source: "uci".to_string(),
                scope: Scope::Shared,
                tags: vec![format!("level:{}", input.level)],
                dimensions: Vec::new(),
                content: input.message,
                attributes: input.attributes,
            },
        };
        let manifests = self.write_events(vec![event])?;
        Ok(serde_json::json!({
            "ingested": manifests.len(),
            "event_id": manifests.first().map(|item| item.event.id.clone()),
        }))
    }
}

fn compile_handoff_proof(read: &ContextRead) -> HandoffProof {
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

    if let Some(item) = read.decisions.first() {
        registers.push(HandoffProofRegister {
            label: "pd1".to_string(),
            register_kind: "decision".to_string(),
            source_id: item.id.clone(),
            title: item.title.clone(),
            body: trim_text(&item.body, PROOF_TRIM_LIMIT),
            has_provenance: !item.supports.is_empty(),
        });
    }
    if let Some(item) = read.constraints.first() {
        registers.push(HandoffProofRegister {
            label: "pk1".to_string(),
            register_kind: "constraint".to_string(),
            source_id: item.id.clone(),
            title: item.title.clone(),
            body: trim_text(&item.body, PROOF_TRIM_LIMIT),
            has_provenance: !item.supports.is_empty(),
        });
    }
    if let Some(item) = read.operational_scars.first() {
        registers.push(HandoffProofRegister {
            label: "ps1".to_string(),
            register_kind: "scar".to_string(),
            source_id: item.id.clone(),
            title: item.title.clone(),
            body: trim_text(&item.body, PROOF_TRIM_LIMIT),
            has_provenance: !item.supports.is_empty(),
        });
    }
    if let Some(item) = read
        .working_state
        .iter()
        .find(|item| {
            item.title == "model-next-step" || item.extra["next_step"].as_bool() == Some(true)
        })
        .or_else(|| read.working_state.first())
    {
        registers.push(HandoffProofRegister {
            label: "pn1".to_string(),
            register_kind: "next_step".to_string(),
            source_id: item.id.clone(),
            title: item.title.clone(),
            body: trim_text(&item.body, PROOF_TRIM_LIMIT),
            has_provenance: !item.supports.is_empty(),
        });
    }

    let digest = registers
        .iter()
        .map(|item| format!("{}:{}:{}", item.label, item.register_kind, item.title))
        .collect::<Vec<_>>()
        .join(" | ");

    HandoffProof { digest, registers }
}

fn trim_text(text: &str, limit: usize) -> String {
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

fn inject_context(event: &mut EventInput, context: &ContextRecord) {
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

fn resolve_namespace(
    storage: &crate::storage::Storage,
    namespace: Option<String>,
) -> Result<Option<String>> {
    match namespace {
        Some(namespace) => storage.resolve_namespace_alias(Some(namespace.as_str())),
        None => Ok(None),
    }
}

fn resolve_selector_namespace(
    storage: &crate::storage::Storage,
    selector: Option<Selector>,
) -> Result<Option<Selector>> {
    let Some(mut selector) = selector else {
        return Ok(None);
    };
    selector.namespace = storage.resolve_namespace_alias(selector.namespace.as_deref())?;
    Ok(Some(selector))
}

fn merge_context_selector(context: &ContextRecord, selector: Option<Selector>) -> Selector {
    let mut selector = selector.unwrap_or_else(|| context.selector.clone());
    selector.namespace = Some(context.namespace.clone());
    selector.all.push(crate::model::DimensionFilter {
        key: "context".to_string(),
        values: vec![context.id.clone()],
    });
    selector
}

fn filter_kind(items: &[ContinuityItemRecord], kind: ContinuityKind) -> Vec<ContinuityItemRecord> {
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
struct ClaimConflictLaneSummary {
    claim_id: String,
    agent_id: String,
    title: String,
    display_name: Option<String>,
    projection_id: String,
    projection_kind: String,
    label: String,
    resource: Option<String>,
    repo_root: Option<String>,
    branch: Option<String>,
    task_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ClaimConflictSummary {
    resource: String,
    #[serde(default)]
    claim_ids: Vec<String>,
    #[serde(default)]
    agents: Vec<String>,
    #[serde(default)]
    titles: Vec<String>,
    #[serde(default)]
    projection_ids: Vec<String>,
    #[serde(default)]
    projected_lanes: Vec<ClaimConflictLaneSummary>,
}

impl ClaimConflictSummary {
    fn new(resource: String) -> Self {
        Self {
            resource,
            ..Self::default()
        }
    }

    fn absorb_claim(
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
        serde_json::Value::Null => serde_json::json!({ "coordination_signal": coordination_value }),
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

fn organism_state(
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

fn merge_dispatch_worker_lane_projections(
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

fn merge_dispatch_assignment_pressure(
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

trait IfEmptyThen {
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

fn augment_dimensions(
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

fn default_support_weight() -> f64 {
    1.0
}

fn default_candidate_limit() -> usize {
    24
}

fn default_token_budget() -> usize {
    384
}

fn default_snapshot_resolution() -> SnapshotResolution {
    SnapshotResolution::Medium
}

#[cfg(test)]
mod tests {
    use chrono::Duration;
    use rusqlite::{Connection, params};
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn continuity_round_trip_supports_snapshot_and_resume() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let attachment = engine
            .attach_agent(AttachAgentInput {
                agent_id: "planner-a".into(),
                agent_type: "ollama".into(),
                capabilities: vec!["plan".into(), "reason".into()],
                namespace: "demo".into(),
                role: Some("planner".into()),
                metadata: serde_json::json!({"model": "glm-4.7-flash"}),
            })
            .unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "shared-kernel".into(),
                session_id: "session-1".into(),
                objective: "stabilize continuity".into(),
                selector: None,
                agent_id: Some("planner-a".into()),
                attachment_id: Some(attachment.id.clone()),
            })
            .unwrap();

        let decision = engine
            .mark_decision(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "planner-a".into(),
                kind: ContinuityKind::Decision,
                title: "Context identity wins".into(),
                body: "Use context namespace and task lineage as the stable identity, not the current agent.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Active),
                importance: Some(0.95),
                confidence: Some(0.9),
                salience: Some(0.95),
                layer: None,
                supports: Vec::new(),
                dimensions: vec![DimensionValue {
                    key: "decision.identity".into(),
                    value: "context-first".into(),
                    weight: DEFAULT_DIMENSION_WEIGHT,
                }],
                extra: serde_json::json!({}),
            })
            .unwrap();
        let scar = engine
            .mark_operational_scar(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "debugger-a".into(),
                kind: ContinuityKind::OperationalScar,
                title: "Do not probe Ollama blindly".into(),
                body: "Ad hoc model probes hung; use the controlled adapter path and capture telemetry.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.98),
                confidence: Some(0.9),
                salience: Some(0.99),
                layer: None,
                supports: vec![SupportRef {
                    support_type: "continuity".into(),
                    support_id: decision.id.clone(),
                    reason: Some("same runtime path".into()),
                    weight: 0.8,
                }],
                dimensions: vec![DimensionValue {
                    key: "scar.runtime".into(),
                    value: "ollama-probe".into(),
                    weight: DEFAULT_DIMENSION_WEIGHT,
                }],
                extra: serde_json::json!({}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id.clone()),
                namespace: None,
                task_id: None,
                objective: "resume continuity work".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("coder-a".into()),
                session_id: Some("session-2".into()),
                view_id: None,
                include_resolved: false,
                candidate_limit: 16,
            })
            .unwrap();

        assert_eq!(read.decisions.len(), 1);
        assert_eq!(read.operational_scars.len(), 1);
        assert!(
            read.decisions
                .iter()
                .any(|item| item.title.contains("Context identity wins"))
        );

        let snapshot = engine
            .snapshot(SnapshotInput {
                context_id: Some(context.id.clone()),
                namespace: None,
                task_id: None,
                objective: Some("checkpoint before handoff".into()),
                selector: None,
                resolution: SnapshotResolution::Medium,
                token_budget: 256,
                candidate_limit: 16,
                owner_agent_id: Some("planner-a".into()),
            })
            .unwrap();
        let resumed = engine
            .resume(ResumeInput {
                snapshot_id: Some(snapshot.id.clone()),
                context_id: None,
                namespace: None,
                task_id: None,
                objective: "resume from snapshot".into(),
                token_budget: 256,
                candidate_limit: 16,
                agent_id: Some("small-agent".into()),
            })
            .unwrap();

        assert_eq!(resumed.snapshot.unwrap().id, snapshot.id);
        assert_eq!(resumed.context.operational_scars[0].id, scar.id);
    }

    #[test]
    fn heartbeat_keeps_attachment_alive_without_fake_work() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let attachment = engine
            .attach_agent(AttachAgentInput {
                agent_id: "idle-agent".into(),
                agent_type: "local".into(),
                capabilities: vec!["observe".into()],
                namespace: "demo".into(),
                role: Some("observer".into()),
                metadata: serde_json::json!({"mode": "heartbeat-test"}),
            })
            .unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "heartbeat".into(),
                session_id: "session-heartbeat".into(),
                objective: "stay visible while idle".into(),
                selector: None,
                agent_id: Some("idle-agent".into()),
                attachment_id: Some(attachment.id.clone()),
            })
            .unwrap();

        let beat = engine
            .heartbeat(HeartbeatInput {
                attachment_id: Some(attachment.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
            })
            .unwrap();

        assert_eq!(beat.id, attachment.id);
        assert!(beat.tick_count >= 3);
        assert_eq!(beat.context_id.as_deref(), Some(context.id.as_str()));

        let metrics = engine.metrics_snapshot().unwrap().prometheus_text;
        assert!(metrics.contains(
            "ice_agent_active{agent_id=\"idle-agent\",agent_type=\"local\",namespace=\"demo\",role=\"observer\"} 1"
        ));
    }

    #[test]
    fn stale_working_state_fades_below_operational_scar() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "retention-test".into(),
                session_id: "session-1".into(),
                objective: "let scars outlive stale scratch state".into(),
                selector: None,
                agent_id: Some("operator".into()),
                attachment_id: None,
            })
            .unwrap();
        let mut working_items = engine
            .write_derivations(vec![ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "operator".into(),
                kind: ContinuityKind::WorkingState,
                title: "Old scratch".into(),
                body: "Temporary scratch note that should fade.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Active),
                importance: Some(0.95),
                confidence: Some(0.95),
                salience: Some(0.98),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            }])
            .unwrap();
        let working = working_items.remove(0);
        let scar = engine
            .mark_operational_scar(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "operator".into(),
                kind: ContinuityKind::OperationalScar,
                title: "Expensive failure".into(),
                body: "This failure should stay sticky.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.70),
                confidence: Some(0.80),
                salience: Some(0.72),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let sqlite = dir.path().join("data/ice.sqlite");
        let conn = Connection::open(sqlite).unwrap();
        let stale_at = (Utc::now() - Duration::days(21)).to_rfc3339();
        conn.execute(
            "UPDATE continuity_items SET ts = ?2, updated_at = ?2 WHERE id = ?1",
            params![working.id, stale_at],
        )
        .unwrap();
        conn.execute(
            "UPDATE continuity_items SET ts = ?2, updated_at = ?2 WHERE id = ?1",
            params![scar.id, stale_at],
        )
        .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "resume without repeating failure".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("operator".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert_eq!(read.open_threads[0].kind, ContinuityKind::OperationalScar);
        assert!(
            read.operational_scars[0].retention.effective_salience
                > read.working_state[0].retention.effective_salience
        );
    }

    #[test]
    fn resolved_items_enter_treated_retention_and_decay_faster() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "therapy-test".into(),
                session_id: "session-1".into(),
                objective: "treated memories should still exist but interfere less".into(),
                selector: None,
                agent_id: Some("therapist".into()),
                attachment_id: None,
            })
            .unwrap();
        let open_constraint = engine
            .mark_constraint(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "therapist".into(),
                kind: ContinuityKind::Constraint,
                title: "Keep logs portable".into(),
                body: "Observability must stay portable.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.85),
                confidence: Some(0.85),
                salience: Some(0.80),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        let treated_constraint = engine
            .mark_constraint(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "therapist".into(),
                kind: ContinuityKind::Constraint,
                title: "Old workaround".into(),
                body: "No longer required after the fix.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.85),
                confidence: Some(0.85),
                salience: Some(0.80),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .resolve_or_supersede(ResolveOrSupersedeInput {
                continuity_id: treated_constraint.id.clone(),
                actor_agent_id: "therapist".into(),
                new_status: ContinuityStatus::Resolved,
                supersedes_id: None,
                resolution_note: Some("therapy completed".into()),
                extra: serde_json::json!({"mode": "therapy"}),
            })
            .unwrap();

        let sqlite = dir.path().join("data/ice.sqlite");
        let conn = Connection::open(sqlite).unwrap();
        let aged_at = (Utc::now() - Duration::days(14)).to_rfc3339();
        conn.execute(
            "UPDATE continuity_items SET ts = ?2, updated_at = ?2 WHERE id = ?1",
            params![open_constraint.id, aged_at],
        )
        .unwrap();
        conn.execute(
            "UPDATE continuity_items SET ts = ?2, updated_at = ?2, resolved_at = ?2 WHERE id = ?1",
            params![treated_constraint.id, aged_at],
        )
        .unwrap();

        let items = engine
            .with_storage(|storage, _| storage.list_continuity_items(&context.id, true))
            .unwrap();
        let open_item = items
            .iter()
            .find(|item| item.id == open_constraint.id)
            .unwrap();
        let treated_item = items
            .iter()
            .find(|item| item.id == treated_constraint.id)
            .unwrap();

        assert_eq!(treated_item.retention.class, "treated_constraint");
        assert!(open_item.retention.effective_salience > treated_item.retention.effective_salience);
    }

    #[test]
    fn read_context_exposes_organism_state_summary() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "organism-summary".into(),
                session_id: "session-1".into(),
                objective: "summarize organism state".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();
        engine
            .mark_operational_scar(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::OperationalScar,
                title: "Do not repeat the hang".into(),
                body: "A prior hang should remain visible.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.9),
                confidence: Some(0.9),
                salience: Some(0.9),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        let treated = engine
            .mark_constraint(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Constraint,
                title: "Legacy workaround".into(),
                body: "This constraint has been retired.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.8),
                confidence: Some(0.8),
                salience: Some(0.7),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .resolve_or_supersede(ResolveOrSupersedeInput {
                continuity_id: treated.id.clone(),
                actor_agent_id: "observer".into(),
                new_status: ContinuityStatus::Resolved,
                supersedes_id: None,
                resolution_note: Some("recovery complete".into()),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect organism".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert_eq!(
            read.organism["retention_classes"]["operational_scar"].as_u64(),
            Some(1)
        );
        assert_eq!(
            read.organism["retention_classes"]["treated_constraint"].as_u64(),
            Some(1)
        );
        assert_eq!(read.organism["treated_items"].as_u64(), Some(1));
        assert_eq!(
            read.organism["open_pressure"][0]["retention_class"].as_str(),
            Some("operational_scar")
        );
        assert_eq!(read.recall.items[0].kind, ContinuityKind::OperationalScar);
        assert!(read.recall.summary.contains("operational_scar"));
    }

    #[test]
    fn read_context_recall_surfaces_trauma_before_stale_working_state() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "recall-fastpath".into(),
                session_id: "session-1".into(),
                objective: "keep trauma searchable".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();

        for index in 0..64 {
            engine
                .write_derivations(vec![ContinuityItemInput {
                    context_id: context.id.clone(),
                    author_agent_id: "observer".into(),
                    kind: ContinuityKind::WorkingState,
                    title: format!("Scratch note {index}"),
                    body: "low-signal scratch state".into(),
                    scope: Scope::Project,
                    status: Some(ContinuityStatus::Active),
                    importance: Some(0.2),
                    confidence: Some(0.6),
                    salience: Some(0.12),
                    layer: None,
                    supports: Vec::new(),
                    dimensions: Vec::new(),
                    extra: serde_json::json!({}),
                }])
                .unwrap();
        }

        engine
            .mark_operational_scar(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "debugger".into(),
                kind: ContinuityKind::OperationalScar,
                title: "Hot reload is worthless if it forks the brain.".into(),
                body: "Split runtime roots make the dashboard lie.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.98),
                confidence: Some(0.9),
                salience: Some(0.99),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id.clone()),
                namespace: None,
                task_id: None,
                objective: "What trauma matters if observability lies after a restart?".into(),
                token_budget: PROOF_TRIM_LIMIT,
                selector: None,
                agent_id: Some("reader".into()),
                session_id: Some("session-2".into()),
                view_id: None,
                include_resolved: false,
                candidate_limit: 12,
            })
            .unwrap();

        assert_eq!(read.recall.items[0].kind, ContinuityKind::OperationalScar);
        assert!(read.recall.items[0].title.contains("forks the brain"));
        assert!(read.recall.items.len() <= 12);
        assert!(
            read.recall.answer_hint.is_some(),
            "expected a safe answer hint for the top trauma"
        );
    }

    #[test]
    fn read_context_pack_reflex_surfaces_recalled_fact_without_manual_tool_choice() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "machine-demo".into(),
                task_id: "reflex-pack".into(),
                session_id: "session-1".into(),
                objective: "keep machine facts instantly available".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();

        for index in 0..48 {
            engine
                .write_derivations(vec![ContinuityItemInput {
                    context_id: context.id.clone(),
                    author_agent_id: "observer".into(),
                    kind: ContinuityKind::WorkingState,
                    title: format!("Scratch note {index}"),
                    body: "low-signal scratch state".into(),
                    scope: Scope::Project,
                    status: Some(ContinuityStatus::Active),
                    importance: Some(0.2),
                    confidence: Some(0.55),
                    salience: Some(0.1),
                    layer: None,
                    supports: Vec::new(),
                    dimensions: Vec::new(),
                    extra: serde_json::json!({}),
                }])
                .unwrap();
        }

        engine
            .write_derivations(vec![ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "archivist".into(),
                kind: ContinuityKind::Fact,
                title: "Alice smell baseline".into(),
                body: "The smell of Alice is flowers.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.97),
                confidence: Some(0.99),
                salience: Some(0.98),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            }])
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id.clone()),
                namespace: None,
                task_id: None,
                objective: "What does Alice smell like?".into(),
                token_budget: 240,
                selector: None,
                agent_id: Some("reader".into()),
                session_id: Some("session-2".into()),
                view_id: None,
                include_resolved: false,
                candidate_limit: 12,
            })
            .unwrap();

        let reflex_item = read
            .pack
            .items
            .iter()
            .take(3)
            .find(|item| item.body.contains("flowers"))
            .expect("expected recalled fact to surface near the front of the pack");
        assert!(reflex_item.breakdown.continuity > 0.0);
        assert!(
            reflex_item
                .why
                .iter()
                .any(|why| why.starts_with("continuity#"))
        );
        assert_eq!(
            read.recall.answer_hint.as_deref(),
            Some("The smell of Alice is flowers.")
        );
        assert_eq!(read.recall.compiler.dominant_band.as_deref(), Some("hot"));
        assert!(read.recall.compiler.compiled_hit_count > 0);
        assert!(
            read.recall.items[0]
                .why
                .iter()
                .any(|why| why.starts_with("compiled_"))
        );
        assert_eq!(
            read.rationale["continuity_recall_compiler"]["dominant_band"].as_str(),
            Some("hot")
        );
        assert!(
            read.rationale["continuity_recall_compiler"]["compiled_hit_count"]
                .as_u64()
                .unwrap_or_default()
                > 0
        );
        assert_eq!(
            read.rationale["continuity_recall_top_why"][0].as_str(),
            Some("lexical")
        );
    }

    #[test]
    fn read_context_recall_refreshes_compiled_bands_when_memory_changes() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "machine-demo".into(),
                task_id: "compiler-refresh".into(),
                session_id: "session-1".into(),
                objective: "keep durable facts cheap to recall".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();

        engine
            .write_derivations(vec![ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "archivist".into(),
                kind: ContinuityKind::Fact,
                title: "Baseline smell".into(),
                body: "Alice smells like rain.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.92),
                confidence: Some(0.95),
                salience: Some(0.91),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            }])
            .unwrap();

        let first = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id.clone()),
                namespace: None,
                task_id: None,
                objective: "What does Alice smell like?".into(),
                token_budget: PROOF_TRIM_LIMIT,
                selector: None,
                agent_id: Some("reader-a".into()),
                session_id: Some("session-a".into()),
                view_id: None,
                include_resolved: false,
                candidate_limit: 12,
            })
            .unwrap();
        assert_eq!(
            first.recall.answer_hint.as_deref(),
            Some("Alice smells like rain.")
        );

        engine
            .write_derivations(vec![ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "archivist".into(),
                kind: ContinuityKind::Fact,
                title: "Upgraded smell".into(),
                body: "Alice smells like flowers.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.97),
                confidence: Some(0.99),
                salience: Some(0.99),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            }])
            .unwrap();

        let second = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "What does Alice smell like now?".into(),
                token_budget: PROOF_TRIM_LIMIT,
                selector: None,
                agent_id: Some("reader-b".into()),
                session_id: Some("session-b".into()),
                view_id: None,
                include_resolved: false,
                candidate_limit: 12,
            })
            .unwrap();
        assert!(second.recall.items[0].preview.contains("flowers"));
        assert!(
            second.recall.items[0]
                .why
                .iter()
                .any(|why| why.starts_with("compiled_"))
        );
        assert_eq!(second.recall.compiler.dominant_band.as_deref(), Some("hot"));
    }

    #[test]
    fn explain_context_surfaces_continuity_compiler_state_and_chunks() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "machine-demo".into(),
                task_id: "compiler-explain".into(),
                session_id: "session-1".into(),
                objective: "make compiler state explainable".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();

        engine
            .write_derivations(vec![ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "archivist".into(),
                kind: ContinuityKind::Fact,
                title: "Compiler explain smell".into(),
                body: "Alice smells like flowers.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.97),
                confidence: Some(0.99),
                salience: Some(0.99),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            }])
            .unwrap();

        engine
            .read_context(ReadContextInput {
                context_id: Some(context.id.clone()),
                namespace: None,
                task_id: None,
                objective: "What does Alice smell like?".into(),
                token_budget: PROOF_TRIM_LIMIT,
                selector: None,
                agent_id: Some("reader".into()),
                session_id: Some("session-a".into()),
                view_id: None,
                include_resolved: false,
                candidate_limit: 12,
            })
            .unwrap();

        let explained = engine
            .explain(ExplainTarget::Context { id: context.id })
            .unwrap();
        assert_eq!(
            explained["compiler"]["state"]["dirty"].as_bool(),
            Some(false)
        );
        assert_eq!(
            explained["compiler"]["state"]["item_count"].as_u64(),
            Some(1)
        );
        assert_eq!(
            explained["compiler"]["chunks"][0]["band"].as_str(),
            Some("hot")
        );
        assert_eq!(
            explained["compiler"]["chunks"][0]["item_count"].as_u64(),
            Some(1)
        );
    }

    #[test]
    fn read_context_marks_dispatch_as_unconfigured_when_absent() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "dispatch-unconfigured".into(),
                session_id: "session-1".into(),
                objective: "inspect dispatch state".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect organism".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert_eq!(
            read.organism["dispatch"]["configured"].as_bool(),
            Some(false)
        );
        assert_eq!(
            read.organism["dispatch"]["workers_active"].as_u64(),
            Some(0)
        );
        assert_eq!(
            read.organism["dispatch"]["assignments_active"].as_u64(),
            Some(0)
        );
    }

    #[test]
    fn read_context_exposes_lane_projections() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let attachment = engine
            .attach_agent(AttachAgentInput {
                agent_id: "observer".into(),
                agent_type: "codex".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "demo".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo", "branch": "main"}),
            })
            .unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: DEFAULT_MACHINE_TASK_ID.into(),
                session_id: "session-projection".into(),
                objective: "project repo lanes".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: Some(attachment.id.clone()),
            })
            .unwrap();
        engine
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(attachment.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Observer".into()),
                status: Some("syncing".into()),
                focus: Some("project repo lanes".into()),
                headline: Some("surface repo worktree projections".into()),
                resource: Some("repo/demo/main".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({"source": "test"}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "observer".into(),
                title: "Own repo lane".into(),
                body: "Project the repo lane from the machine brain.".into(),
                scope: Scope::Shared,
                resources: vec!["repo/demo/main".into()],
                exclusive: true,
                attachment_id: Some(attachment.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .publish_coordination_signal(CoordinationSignalInput {
                context_id: context.id.clone(),
                agent_id: "boundary-warden".into(),
                title: "Back off the repo lane".into(),
                body: "The repo lane is under active coordination pressure.".into(),
                lane: CoordinationLane::Backoff,
                target_agent_id: Some("observer".into()),
                target_projected_lane: Some(CoordinationProjectedLane {
                    projection_id: "repo:/tmp/demo:main".into(),
                    projection_kind: "repo".into(),
                    label: "demo @ main".into(),
                    resource: Some("repo/demo/main".into()),
                    repo_root: Some("/tmp/demo".into()),
                    branch: Some("main".into()),
                    task_id: Some(DEFAULT_MACHINE_TASK_ID.into()),
                }),
                claim_id: None,
                resource: Some("repo/demo/main".into()),
                severity: None,
                projection_ids: vec!["repo:/tmp/demo:main".into()],
                projected_lanes: vec![CoordinationProjectedLane {
                    projection_id: "repo:/tmp/demo:main".into(),
                    projection_kind: "repo".into(),
                    label: "demo @ main".into(),
                    resource: Some("repo/demo/main".into()),
                    repo_root: Some("/tmp/demo".into()),
                    branch: Some("main".into()),
                    task_id: Some(DEFAULT_MACHINE_TASK_ID.into()),
                }],
                extra: serde_json::json!({"source": "test"}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect projected lanes".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: false,
                candidate_limit: 32,
            })
            .unwrap();

        let projections = read.organism["lane_projections"].as_array().unwrap();
        assert_eq!(projections.len(), 1);
        assert_eq!(projections[0]["projection_kind"].as_str(), Some("repo"));
        assert_eq!(projections[0]["label"].as_str(), Some("demo @ main"));
        assert_eq!(projections[0]["connected_agents"].as_u64(), Some(1));
        assert_eq!(projections[0]["live_claims"].as_u64(), Some(1));
        assert_eq!(
            projections[0]["coordination_signal_count"].as_u64(),
            Some(1)
        );
        assert_eq!(projections[0]["blocking_signal_count"].as_u64(), Some(1));
        assert_eq!(
            projections[0]["coordination_lanes"][0].as_str(),
            Some("backoff")
        );
        assert_eq!(
            projections[0]["focus"].as_str(),
            Some("Back off the repo lane")
        );
    }

    #[test]
    fn read_context_keeps_same_repo_branch_worktrees_separate() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let attachment_a = engine
            .attach_agent(AttachAgentInput {
                agent_id: "observer-a".into(),
                agent_type: "codex".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "demo".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/worktrees/adhd", "branch": "main"}),
            })
            .unwrap();
        let attachment_b = engine
            .attach_agent(AttachAgentInput {
                agent_id: "observer-b".into(),
                agent_type: "codex".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "demo".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/shadow/adhd", "branch": "main"}),
            })
            .unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: DEFAULT_MACHINE_TASK_ID.into(),
                session_id: "session-worktrees".into(),
                objective: "separate same-branch repo worktrees".into(),
                selector: None,
                agent_id: Some("observer-a".into()),
                attachment_id: Some(attachment_a.id.clone()),
            })
            .unwrap();
        engine
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(attachment_a.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Observer A".into()),
                status: Some("syncing".into()),
                focus: Some("watch first worktree".into()),
                headline: Some("keep worktree A separate".into()),
                resource: Some("repo/adhd/main".into()),
                repo_root: Some("/tmp/worktrees/adhd".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({"source": "test"}),
            })
            .unwrap();
        engine
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(attachment_b.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Observer B".into()),
                status: Some("syncing".into()),
                focus: Some("watch second worktree".into()),
                headline: Some("keep worktree B separate".into()),
                resource: Some("repo/adhd/main".into()),
                repo_root: Some("/tmp/shadow/adhd".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({"source": "test"}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "observer-a".into(),
                title: "Hold first repo lane".into(),
                body: "Project the first worktree lane.".into(),
                scope: Scope::Shared,
                resources: vec!["repo/adhd/main".into()],
                exclusive: true,
                attachment_id: Some(attachment_a.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "observer-b".into(),
                title: "Hold second repo lane".into(),
                body: "Project the second worktree lane.".into(),
                scope: Scope::Shared,
                resources: vec!["repo/adhd/main".into()],
                exclusive: true,
                attachment_id: Some(attachment_b.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect repo worktree lanes".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer-a".into()),
                session_id: None,
                view_id: None,
                include_resolved: false,
                candidate_limit: 32,
            })
            .unwrap();

        let projections = read.organism["lane_projections"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|projection| projection["projection_kind"].as_str() == Some("repo"))
            .collect::<Vec<_>>();
        assert_eq!(projections.len(), 2);
        assert!(projections.iter().any(|projection| {
            projection["projection_id"].as_str() == Some("repo:/tmp/worktrees/adhd:main")
                && projection["repo_root"].as_str() == Some("/tmp/worktrees/adhd")
                && projection["connected_agents"].as_u64() == Some(1)
                && projection["live_claims"].as_u64() == Some(1)
        }));
        assert!(projections.iter().any(|projection| {
            projection["projection_id"].as_str() == Some("repo:/tmp/shadow/adhd:main")
                && projection["repo_root"].as_str() == Some("/tmp/shadow/adhd")
                && projection["connected_agents"].as_u64() == Some(1)
                && projection["live_claims"].as_u64() == Some(1)
        }));
    }

    #[test]
    fn claim_conflicts_include_projected_lane_identity() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let main_attachment = engine
            .attach_agent(AttachAgentInput {
                agent_id: "agent-main".into(),
                agent_type: "local".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "demo".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo", "branch": "main"}),
            })
            .unwrap();
        let shadow_attachment = engine
            .attach_agent(AttachAgentInput {
                agent_id: "agent-shadow".into(),
                agent_type: "local".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "demo".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo", "branch": "feature/shadow"}),
            })
            .unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: DEFAULT_MACHINE_TASK_ID.into(),
                session_id: "session-projected-conflict".into(),
                objective: "surface lane-aware conflicts".into(),
                selector: None,
                agent_id: Some("agent-main".into()),
                attachment_id: Some(main_attachment.id.clone()),
            })
            .unwrap();
        engine
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(main_attachment.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Agent Main".into()),
                status: Some("editing".into()),
                focus: Some("touch file".into()),
                headline: Some("main lane".into()),
                resource: Some("repo/demo/main".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        engine
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(shadow_attachment.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Agent Shadow".into()),
                status: Some("editing".into()),
                focus: Some("touch file".into()),
                headline: Some("shadow lane".into()),
                resource: Some("repo/demo/feature/shadow".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("feature/shadow".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "agent-main".into(),
                title: "Edit storage from main".into(),
                body: "Touch file/src/storage.rs from main.".into(),
                scope: Scope::Project,
                resources: vec!["file/src/storage.rs".into()],
                exclusive: true,
                attachment_id: Some(main_attachment.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "agent-shadow".into(),
                title: "Edit storage from shadow".into(),
                body: "Touch file/src/storage.rs from shadow.".into(),
                scope: Scope::Project,
                resources: vec!["file/src/storage.rs".into()],
                exclusive: true,
                attachment_id: Some(shadow_attachment.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect lane-aware conflict state".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("agent-main".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 24,
            })
            .unwrap();

        let conflicts = read.organism["claim_conflicts"].as_array().unwrap();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(
            conflicts[0]["resource"].as_str(),
            Some("file/src/storage.rs")
        );
        let lane_projection_ids = read.organism["lane_projections"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|projection| projection["projection_id"].as_str())
            .collect::<Vec<_>>();
        let projection_ids = conflicts[0]["projection_ids"].as_array().unwrap();
        assert_eq!(projection_ids.len(), 2);
        assert!(projection_ids.iter().all(|projection_id| {
            projection_id
                .as_str()
                .map(|projection_id| lane_projection_ids.contains(&projection_id))
                .unwrap_or(false)
        }));
        let lanes = conflicts[0]["projected_lanes"].as_array().unwrap();
        assert_eq!(lanes.len(), 2);
        assert!(lanes.iter().any(|lane| {
            lane["label"].as_str() == Some("demo @ main")
                && lane["projection_kind"].as_str() == Some("repo")
                && lane["display_name"].as_str() == Some("Agent Main")
        }));
        assert!(lanes.iter().any(|lane| {
            lane["label"].as_str() == Some("demo @ feature/shadow")
                && lane["projection_kind"].as_str() == Some("repo")
                && lane["display_name"].as_str() == Some("Agent Shadow")
        }));
    }

    #[test]
    fn handoff_manifest_persists_organism_state() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "handoff-organism".into(),
                session_id: "session-1".into(),
                objective: "handoff should carry organism state".into(),
                selector: None,
                agent_id: Some("planner-a".into()),
                attachment_id: None,
            })
            .unwrap();
        engine
            .mark_operational_scar(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "planner-a".into(),
                kind: ContinuityKind::OperationalScar,
                title: "Prior failure".into(),
                body: "This scar should survive the handoff manifest.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.9),
                confidence: Some(0.9),
                salience: Some(0.9),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let handoff = engine
            .handoff(ContinuityHandoffInput {
                context_id: Some(context.id),
                namespace: Some("demo".into()),
                task_id: Some("handoff-organism".into()),
                from_agent_id: "planner-a".into(),
                to_agent_id: "coder-b".into(),
                objective: "resume with organism state".into(),
                reason: "swap heads".into(),
                selector: None,
                resolution: SnapshotResolution::Medium,
                token_budget: 256,
                candidate_limit: 16,
            })
            .unwrap();

        let explained = engine
            .explain(ExplainTarget::Handoff {
                id: handoff.handoff.id,
            })
            .unwrap();
        assert_eq!(
            explained["context"]["organism"]["retention_classes"]["operational_scar"].as_u64(),
            Some(1)
        );
        assert_eq!(
            explained["proof"]["registers"]
                .as_array()
                .unwrap()
                .iter()
                .find(|item| item["label"].as_str() == Some("ps1"))
                .and_then(|item| item["register_kind"].as_str()),
            Some("scar")
        );
        assert!(
            handoff
                .proof
                .registers
                .iter()
                .any(|item| item.label == "ps1" && item.register_kind == "scar")
        );
    }

    #[test]
    fn work_claims_surface_conflicts_in_context_reads() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let planner = engine
            .attach_agent(AttachAgentInput {
                agent_id: "planner-a".into(),
                agent_type: "local".into(),
                capabilities: vec!["plan".into()],
                namespace: "demo".into(),
                role: Some("planner".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let debugger = engine
            .attach_agent(AttachAgentInput {
                agent_id: "debugger-b".into(),
                agent_type: "local".into(),
                capabilities: vec!["debug".into()],
                namespace: "demo".into(),
                role: Some("debugger".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "claim-conflicts".into(),
                session_id: "session-1".into(),
                objective: "coordinate files without collisions".into(),
                selector: None,
                agent_id: Some("planner-a".into()),
                attachment_id: Some(planner.id.clone()),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "planner-a".into(),
                title: "Own src/continuity.rs".into(),
                body: "I am rewriting the kernel coordination path.".into(),
                scope: Scope::Project,
                resources: vec!["src/continuity.rs".into()],
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
                title: "Also touching src/continuity.rs".into(),
                body: "I think the planner is doing something cursed.".into(),
                scope: Scope::Project,
                resources: vec!["src/continuity.rs".into()],
                exclusive: true,
                attachment_id: Some(debugger.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "resume the same claim state".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert_eq!(read.work_claims.len(), 2);
        assert_eq!(read.organism["active_claim_count"].as_u64(), Some(2));
        assert_eq!(read.organism["claim_conflict_count"].as_u64(), Some(1));
        assert!(
            read.open_threads
                .iter()
                .any(|item| item.kind == ContinuityKind::WorkClaim)
        );
    }

    #[test]
    fn heartbeat_renews_attached_work_claim_lease() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let attachment = engine
            .attach_agent(AttachAgentInput {
                agent_id: "coder-a".into(),
                agent_type: "local".into(),
                capabilities: vec!["write".into()],
                namespace: "demo".into(),
                role: Some("coder".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "claim-heartbeat".into(),
                session_id: "session-1".into(),
                objective: "keep the claim alive while idle".into(),
                selector: None,
                agent_id: Some("coder-a".into()),
                attachment_id: Some(attachment.id.clone()),
            })
            .unwrap();
        let claim = engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "coder-a".into(),
                title: "Own src/mcp.rs".into(),
                body: "Implement MCP plumbing without collisions.".into(),
                scope: Scope::Project,
                resources: vec!["src/mcp.rs".into()],
                exclusive: true,
                attachment_id: Some(attachment.id.clone()),
                lease_seconds: Some(60),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let sqlite = dir.path().join("data/ice.sqlite");
        let conn = Connection::open(sqlite).unwrap();
        let (memory_id, extra_json) = conn
            .query_row(
                "SELECT memory_id, extra_json FROM continuity_items WHERE id = ?1",
                params![claim.id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .unwrap();
        let mut extra: serde_json::Value = serde_json::from_str(&extra_json).unwrap();
        extra["user"]["coordination"]["lease_expires_at"] =
            serde_json::json!((Utc::now() - Duration::seconds(5)).to_rfc3339());
        conn.execute(
            "UPDATE continuity_items SET extra_json = ?2 WHERE id = ?1",
            params![claim.id, extra.to_string()],
        )
        .unwrap();
        conn.execute(
            "UPDATE memory_items SET extra_json = ?2 WHERE id = ?1",
            params![memory_id, extra.to_string()],
        )
        .unwrap();

        let expired = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id.clone()),
                namespace: None,
                task_id: None,
                objective: "before heartbeat".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();
        assert!(expired.work_claims.is_empty());

        engine
            .heartbeat(HeartbeatInput {
                attachment_id: Some(attachment.id),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
            })
            .unwrap();

        let renewed = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "after heartbeat".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();
        assert_eq!(renewed.work_claims.len(), 1);
    }

    #[test]
    fn coordination_anxiety_surfaces_in_organism_pressure() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "anxiety-pressure".into(),
                session_id: "session-1".into(),
                objective: "let the organism worry out loud".into(),
                selector: None,
                agent_id: Some("planner-a".into()),
                attachment_id: None,
            })
            .unwrap();
        let claim = engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "planner-a".into(),
                title: "Own src/storage.rs".into(),
                body: "Metrics surgery in progress.".into(),
                scope: Scope::Project,
                resources: vec!["src/storage.rs".into()],
                exclusive: true,
                attachment_id: None,
                lease_seconds: Some(180),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .publish_coordination_signal(CoordinationSignalInput {
                context_id: context.id.clone(),
                agent_id: "therapist".into(),
                title: "Anxiety spike over storage surgery".into(),
                body: "Watch metrics cardinality before this lands.".into(),
                lane: CoordinationLane::Anxiety,
                target_agent_id: Some("planner-a".into()),
                target_projected_lane: Some(CoordinationProjectedLane {
                    projection_id: "repo:demo:main".into(),
                    projection_kind: "repo".into(),
                    label: "demo @ main".into(),
                    resource: Some("repo/demo/main".into()),
                    repo_root: Some("/tmp/demo".into()),
                    branch: Some("main".into()),
                    task_id: Some("anxiety-pressure".into()),
                }),
                claim_id: Some(claim.id),
                resource: Some("src/storage.rs".into()),
                severity: None,
                projection_ids: vec!["repo:demo:main".into()],
                projected_lanes: vec![CoordinationProjectedLane {
                    projection_id: "repo:demo:main".into(),
                    projection_kind: "repo".into(),
                    label: "demo @ main".into(),
                    resource: Some("repo/demo/main".into()),
                    repo_root: Some("/tmp/demo".into()),
                    branch: Some("main".into()),
                    task_id: Some("anxiety-pressure".into()),
                }],
                extra: serde_json::json!({}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect anxiety".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert_eq!(read.coordination_signals.len(), 1);
        assert_eq!(read.organism["coordination_signal_count"].as_u64(), Some(1));
        assert_eq!(read.organism["anxiety_signal_count"].as_u64(), Some(1));
        assert_eq!(
            read.organism["anxiety_pressure"][0]["target_agent_id"].as_str(),
            Some("planner-a")
        );
        assert_eq!(
            read.organism["anxiety_pressure"][0]["target_projected_lane"]["label"].as_str(),
            Some("demo @ main")
        );
        assert_eq!(
            read.organism["anxiety_pressure"][0]["projection_ids"]
                .as_array()
                .map(Vec::len),
            Some(1)
        );
        assert_eq!(
            read.organism["anxiety_pressure"][0]["projected_lanes"][0]["label"].as_str(),
            Some("demo @ main")
        );
    }

    #[test]
    fn therapy_resolves_anxiety_into_treated_memory() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "therapy-anxiety".into(),
                session_id: "session-1".into(),
                objective: "treat false coordination fear".into(),
                selector: None,
                agent_id: Some("planner-a".into()),
                attachment_id: None,
            })
            .unwrap();
        let signal = engine
            .publish_coordination_signal(CoordinationSignalInput {
                context_id: context.id.clone(),
                agent_id: "therapist".into(),
                title: "Anxiety over phantom risk".into(),
                body: "This might break invariants, but it is still unproven.".into(),
                lane: CoordinationLane::Anxiety,
                target_agent_id: Some("planner-a".into()),
                target_projected_lane: None,
                claim_id: None,
                resource: Some("src/continuity.rs".into()),
                severity: Some(CoordinationSeverity::Warn),
                projection_ids: Vec::new(),
                projected_lanes: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .resolve_or_supersede(ResolveOrSupersedeInput {
                continuity_id: signal.id.clone(),
                actor_agent_id: "therapist".into(),
                new_status: ContinuityStatus::Resolved,
                supersedes_id: None,
                resolution_note: Some("Therapy proved the fear was not reality.".into()),
                extra: serde_json::json!({"therapy": true}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect treated anxiety".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert!(read.coordination_signals.is_empty());
        assert_eq!(read.organism["anxiety_signal_count"].as_u64(), Some(0));
        assert_eq!(
            read.organism["retention_classes"]["treated_signal"].as_u64(),
            Some(1)
        );
    }

    #[test]
    fn resolve_updates_status_dimension() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "resolve-test".into(),
                session_id: "session-1".into(),
                objective: "track scar status".into(),
                selector: None,
                agent_id: Some("debugger-a".into()),
                attachment_id: None,
            })
            .unwrap();
        let incident = engine
            .mark_incident(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "debugger-a".into(),
                kind: ContinuityKind::Incident,
                title: "Regression".into(),
                body: "A known regression reappeared.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.9),
                confidence: Some(0.8),
                salience: Some(0.9),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let resolved = engine
            .resolve_or_supersede(ResolveOrSupersedeInput {
                continuity_id: incident.id.clone(),
                actor_agent_id: "debugger-a".into(),
                new_status: ContinuityStatus::Resolved,
                supersedes_id: None,
                resolution_note: Some("fixed in kernel".into()),
                extra: serde_json::json!({}),
            })
            .unwrap();

        assert_eq!(resolved.status, ContinuityStatus::Resolved);
        let explained = engine
            .explain(ExplainTarget::ContinuityItem { id: incident.id })
            .unwrap();
        assert_eq!(
            explained["item"]["status"].as_str(),
            Some(ContinuityStatus::Resolved.as_str())
        );
    }

    #[test]
    fn continuity_kind_default_layer_maps_correctly() {
        assert_eq!(
            ContinuityKind::WorkingState.default_layer(),
            MemoryLayer::Hot
        );
        assert_eq!(ContinuityKind::WorkClaim.default_layer(), MemoryLayer::Hot);
        assert_eq!(ContinuityKind::Signal.default_layer(), MemoryLayer::Hot);
        assert_eq!(
            ContinuityKind::Derivation.default_layer(),
            MemoryLayer::Semantic
        );
        assert_eq!(ContinuityKind::Fact.default_layer(), MemoryLayer::Semantic);
        assert_eq!(
            ContinuityKind::Decision.default_layer(),
            MemoryLayer::Semantic
        );
        assert_eq!(
            ContinuityKind::Constraint.default_layer(),
            MemoryLayer::Semantic
        );
        assert_eq!(
            ContinuityKind::OperationalScar.default_layer(),
            MemoryLayer::Semantic
        );
        assert_eq!(
            ContinuityKind::Hypothesis.default_layer(),
            MemoryLayer::Episodic
        );
        assert_eq!(
            ContinuityKind::Incident.default_layer(),
            MemoryLayer::Episodic
        );
        assert_eq!(
            ContinuityKind::Outcome.default_layer(),
            MemoryLayer::Episodic
        );
        assert_eq!(
            ContinuityKind::Lesson.default_layer(),
            MemoryLayer::Episodic
        );
        assert_eq!(
            ContinuityKind::Summary.default_layer(),
            MemoryLayer::Summary
        );
    }

    #[test]
    fn continuity_kind_as_str_round_trips_through_serde() {
        let all_kinds = [
            ContinuityKind::WorkingState,
            ContinuityKind::WorkClaim,
            ContinuityKind::Derivation,
            ContinuityKind::Fact,
            ContinuityKind::Decision,
            ContinuityKind::Constraint,
            ContinuityKind::Hypothesis,
            ContinuityKind::Incident,
            ContinuityKind::OperationalScar,
            ContinuityKind::Outcome,
            ContinuityKind::Signal,
            ContinuityKind::Summary,
            ContinuityKind::Lesson,
        ];
        for kind in all_kinds {
            let json = serde_json::to_string(&kind).unwrap();
            let deserialized: ContinuityKind = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, kind);
            assert_eq!(
                json.trim_matches('"'),
                kind.as_str(),
                "as_str should match serde serialization for {:?}",
                kind
            );
        }
    }

    #[test]
    fn continuity_status_as_str_all_variants() {
        assert_eq!(ContinuityStatus::Open.as_str(), "open");
        assert_eq!(ContinuityStatus::Active.as_str(), "active");
        assert_eq!(ContinuityStatus::Resolved.as_str(), "resolved");
        assert_eq!(ContinuityStatus::Superseded.as_str(), "superseded");
        assert_eq!(ContinuityStatus::Rejected.as_str(), "rejected");
    }

    #[test]
    fn continuity_status_is_open_distinguishes_active_from_closed() {
        assert!(ContinuityStatus::Open.is_open());
        assert!(ContinuityStatus::Active.is_open());
        assert!(!ContinuityStatus::Resolved.is_open());
        assert!(!ContinuityStatus::Superseded.is_open());
        assert!(!ContinuityStatus::Rejected.is_open());
    }

    #[test]
    fn context_status_as_str_all_variants() {
        assert_eq!(ContextStatus::Open.as_str(), "open");
        assert_eq!(ContextStatus::Paused.as_str(), "paused");
        assert_eq!(ContextStatus::Closed.as_str(), "closed");
    }

    #[test]
    fn coordination_lane_as_str_all_variants() {
        assert_eq!(CoordinationLane::Anxiety.as_str(), "anxiety");
        assert_eq!(CoordinationLane::Review.as_str(), "review");
        assert_eq!(CoordinationLane::Warning.as_str(), "warning");
        assert_eq!(CoordinationLane::Backoff.as_str(), "backoff");
        assert_eq!(CoordinationLane::Coach.as_str(), "coach");
    }

    #[test]
    fn coordination_severity_as_str_all_variants() {
        assert_eq!(CoordinationSeverity::Info.as_str(), "info");
        assert_eq!(CoordinationSeverity::Warn.as_str(), "warn");
        assert_eq!(CoordinationSeverity::Block.as_str(), "block");
    }

    #[test]
    fn default_coordination_severity_maps_lanes() {
        assert_eq!(
            default_coordination_severity(CoordinationLane::Review),
            CoordinationSeverity::Info
        );
        assert_eq!(
            default_coordination_severity(CoordinationLane::Coach),
            CoordinationSeverity::Info
        );
        assert_eq!(
            default_coordination_severity(CoordinationLane::Warning),
            CoordinationSeverity::Warn
        );
        assert_eq!(
            default_coordination_severity(CoordinationLane::Anxiety),
            CoordinationSeverity::Warn
        );
        assert_eq!(
            default_coordination_severity(CoordinationLane::Backoff),
            CoordinationSeverity::Block
        );
    }

    #[test]
    fn normalize_work_claim_resources_deduplicates_and_sorts() {
        let input = vec![
            "  SRC/main.rs ".to_string(),
            "src/lib.rs".to_string(),
            "SRC/MAIN.RS".to_string(),
            "".to_string(),
            "  ".to_string(),
        ];
        let result = normalize_work_claim_resources(&input);
        assert_eq!(result, vec!["src/lib.rs", "src/main.rs"]);
    }

    #[test]
    fn normalize_work_claim_resources_empty_input() {
        let result = normalize_work_claim_resources(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn work_claim_key_uses_title_when_no_resources() {
        let key = work_claim_key("ctx-1", Scope::Project, "agent-a", "Fix the bug!", &[]);
        assert!(key.starts_with("ctx-1:"));
        assert!(key.contains("agent-a"));
        assert!(key.contains("fix-the-bug"));
        assert!(!key.contains("!"));
    }

    #[test]
    fn work_claim_key_uses_resources_when_present() {
        let resources = vec!["src/main.rs".to_string(), "src/lib.rs".to_string()];
        let key = work_claim_key("ctx-1", Scope::Shared, "agent-a", "title", &resources);
        assert!(key.contains("src/main.rs|src/lib.rs"));
        assert!(!key.contains("title"));
    }

    #[test]
    fn merge_work_claim_extra_into_null() {
        let coordination = WorkClaimCoordination {
            claim_key: "key".into(),
            resources: vec!["file.rs".into()],
            exclusive: true,
            ..Default::default()
        };
        let result = merge_work_claim_extra(serde_json::Value::Null, &coordination);
        assert!(result["coordination"]["claim_key"].as_str() == Some("key"));
        assert!(result["coordination"]["exclusive"].as_bool() == Some(true));
    }

    #[test]
    fn merge_work_claim_extra_into_object() {
        let coordination = WorkClaimCoordination::default();
        let existing = serde_json::json!({"user_data": 42});
        let result = merge_work_claim_extra(existing, &coordination);
        assert_eq!(result["user_data"].as_u64(), Some(42));
        assert!(result["coordination"].is_object());
    }

    #[test]
    fn merge_work_claim_extra_into_non_object() {
        let coordination = WorkClaimCoordination::default();
        let result = merge_work_claim_extra(serde_json::json!("string payload"), &coordination);
        assert_eq!(result["payload"].as_str(), Some("string payload"));
        assert!(result["coordination"].is_object());
    }

    #[test]
    fn work_claim_coordination_from_extra_finds_nested_and_top_level() {
        let with_user = serde_json::json!({
            "user": { "coordination": { "claim_key": "from-user", "resources": [], "exclusive": false } }
        });
        let result = work_claim_coordination_from_extra(&with_user).unwrap();
        assert_eq!(result.claim_key, "from-user");

        let top_level = serde_json::json!({
            "coordination": { "claim_key": "from-top", "resources": [], "exclusive": true }
        });
        let result = work_claim_coordination_from_extra(&top_level).unwrap();
        assert_eq!(result.claim_key, "from-top");

        let empty = serde_json::json!({});
        assert!(work_claim_coordination_from_extra(&empty).is_none());
    }

    #[test]
    fn merge_coordination_signal_extra_into_null() {
        let result = merge_coordination_signal_extra(
            serde_json::Value::Null,
            CoordinationLane::Anxiety,
            CoordinationSeverity::Warn,
            Some("target-agent".into()),
            None,
            Some("claim-1".into()),
            Some("src/file.rs".into()),
            vec!["proj-1".into()],
            Vec::new(),
        );
        assert_eq!(
            result["coordination_signal"]["lane"].as_str(),
            Some("anxiety")
        );
        assert_eq!(
            result["coordination_signal"]["severity"].as_str(),
            Some("warn")
        );
        assert_eq!(
            result["coordination_signal"]["target_agent_id"].as_str(),
            Some("target-agent")
        );
        assert_eq!(
            result["coordination_signal"]["claim_id"].as_str(),
            Some("claim-1")
        );
    }

    #[test]
    fn merge_coordination_signal_extra_into_object_preserves_fields() {
        let existing = serde_json::json!({"metadata": "keep"});
        let result = merge_coordination_signal_extra(
            existing,
            CoordinationLane::Review,
            CoordinationSeverity::Info,
            None,
            None,
            None,
            None,
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(result["metadata"].as_str(), Some("keep"));
        assert_eq!(
            result["coordination_signal"]["lane"].as_str(),
            Some("review")
        );
    }

    #[test]
    fn merge_coordination_signal_extra_wraps_non_object() {
        let result = merge_coordination_signal_extra(
            serde_json::json!(42),
            CoordinationLane::Backoff,
            CoordinationSeverity::Block,
            None,
            None,
            None,
            None,
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(result["payload"].as_u64(), Some(42));
        assert_eq!(
            result["coordination_signal"]["lane"].as_str(),
            Some("backoff")
        );
    }

    #[test]
    fn coordination_signal_from_extra_finds_nested_and_top_level() {
        let with_user = serde_json::json!({
            "user": { "coordination_signal": { "lane": "anxiety", "severity": "warn" } }
        });
        let result = coordination_signal_from_extra(&with_user).unwrap();
        assert_eq!(result.lane, "anxiety");

        let top_level = serde_json::json!({
            "coordination_signal": { "lane": "review", "severity": "info" }
        });
        let result = coordination_signal_from_extra(&top_level).unwrap();
        assert_eq!(result.lane, "review");

        let empty = serde_json::json!({});
        assert!(coordination_signal_from_extra(&empty).is_none());
    }

    fn make_test_item(
        id: &str,
        kind: ContinuityKind,
        status: ContinuityStatus,
        extra: serde_json::Value,
    ) -> ContinuityItemRecord {
        ContinuityItemRecord {
            id: id.into(),
            memory_id: format!("mem-{id}"),
            context_id: "ctx-1".into(),
            namespace: "ns".into(),
            task_id: "task".into(),
            author_agent_id: "agent-a".into(),
            kind,
            scope: Scope::Project,
            status,
            title: format!("Item {id}"),
            body: "body".into(),
            importance: 0.9,
            confidence: 0.9,
            salience: 0.9,
            retention: ContinuityRetentionState {
                class: kind.as_str().to_string(),
                age_hours: 1.0,
                half_life_hours: 36.0,
                floor: 0.03,
                decay_multiplier: 1.0,
                effective_salience: 0.9,
            },
            created_at: Utc::now(),
            updated_at: Utc::now(),
            supersedes_id: None,
            resolved_at: None,
            supports: Vec::new(),
            extra,
        }
    }

    #[test]
    fn filter_kind_returns_only_matching() {
        let items = vec![
            make_test_item(
                "1",
                ContinuityKind::Fact,
                ContinuityStatus::Open,
                serde_json::json!({}),
            ),
            make_test_item(
                "2",
                ContinuityKind::Decision,
                ContinuityStatus::Open,
                serde_json::json!({}),
            ),
            make_test_item(
                "3",
                ContinuityKind::Fact,
                ContinuityStatus::Active,
                serde_json::json!({}),
            ),
        ];
        let facts = filter_kind(&items, ContinuityKind::Fact);
        assert_eq!(facts.len(), 2);
        assert!(facts.iter().all(|item| item.kind == ContinuityKind::Fact));

        let empty = filter_kind(&items, ContinuityKind::Lesson);
        assert!(empty.is_empty());
    }

    #[test]
    fn work_claim_is_live_requires_kind_and_open_status() {
        let now = Utc::now();
        let non_claim = make_test_item(
            "1",
            ContinuityKind::Fact,
            ContinuityStatus::Open,
            serde_json::json!({}),
        );
        assert!(!work_claim_is_live(&non_claim, now));

        let resolved_claim = make_test_item(
            "2",
            ContinuityKind::WorkClaim,
            ContinuityStatus::Resolved,
            serde_json::json!({}),
        );
        assert!(!work_claim_is_live(&resolved_claim, now));
    }

    #[test]
    fn work_claim_is_live_with_no_coordination_defaults_to_true() {
        let now = Utc::now();
        let claim = make_test_item(
            "1",
            ContinuityKind::WorkClaim,
            ContinuityStatus::Open,
            serde_json::json!({}),
        );
        assert!(work_claim_is_live(&claim, now));
    }

    #[test]
    fn work_claim_is_live_respects_lease_expiry() {
        let now = Utc::now();
        let future = now + Duration::hours(1);
        let past = now - Duration::hours(1);

        let live_claim = make_test_item(
            "1",
            ContinuityKind::WorkClaim,
            ContinuityStatus::Active,
            serde_json::json!({
                "coordination": {
                    "claim_key": "k",
                    "resources": [],
                    "exclusive": false,
                    "lease_expires_at": future.to_rfc3339()
                }
            }),
        );
        assert!(work_claim_is_live(&live_claim, now));

        let expired_claim = make_test_item(
            "2",
            ContinuityKind::WorkClaim,
            ContinuityStatus::Active,
            serde_json::json!({
                "coordination": {
                    "claim_key": "k",
                    "resources": [],
                    "exclusive": false,
                    "lease_expires_at": past.to_rfc3339()
                }
            }),
        );
        assert!(!work_claim_is_live(&expired_claim, now));
    }

    #[test]
    fn counts_as_open_thread_filters_resolved_and_expired_claims() {
        let now = Utc::now();
        let resolved = make_test_item(
            "1",
            ContinuityKind::Fact,
            ContinuityStatus::Resolved,
            serde_json::json!({}),
        );
        assert!(!counts_as_open_thread(&resolved, now));

        let open_fact = make_test_item(
            "2",
            ContinuityKind::Fact,
            ContinuityStatus::Open,
            serde_json::json!({}),
        );
        assert!(counts_as_open_thread(&open_fact, now));

        let open_claim_no_coord = make_test_item(
            "3",
            ContinuityKind::WorkClaim,
            ContinuityStatus::Open,
            serde_json::json!({}),
        );
        assert!(counts_as_open_thread(&open_claim_no_coord, now));

        let past = now - Duration::hours(1);
        let expired_claim = make_test_item(
            "4",
            ContinuityKind::WorkClaim,
            ContinuityStatus::Open,
            serde_json::json!({
                "coordination": {
                    "claim_key": "k",
                    "resources": [],
                    "exclusive": false,
                    "lease_expires_at": past.to_rfc3339()
                }
            }),
        );
        assert!(!counts_as_open_thread(&expired_claim, now));
    }

    #[test]
    fn work_claims_conflict_requires_exclusive_and_shared_resource() {
        let now = Utc::now();
        let future = now + Duration::hours(1);

        let make_claim = |id: &str, resources: Vec<&str>, exclusive: bool| {
            make_test_item(
                id,
                ContinuityKind::WorkClaim,
                ContinuityStatus::Active,
                serde_json::json!({
                    "coordination": {
                        "claim_key": format!("k-{id}"),
                        "resources": resources,
                        "exclusive": exclusive,
                        "lease_expires_at": future.to_rfc3339()
                    }
                }),
            )
        };

        let claim_a = make_claim("a", vec!["file.rs"], true);
        let claim_b = make_claim("b", vec!["file.rs"], false);
        assert!(work_claims_conflict(&claim_a, &claim_b, now));

        let claim_c = make_claim("c", vec!["file.rs"], false);
        let claim_d = make_claim("d", vec!["file.rs"], false);
        assert!(!work_claims_conflict(&claim_c, &claim_d, now));

        let claim_e = make_claim("e", vec!["a.rs"], true);
        let claim_f = make_claim("f", vec!["b.rs"], true);
        assert!(!work_claims_conflict(&claim_e, &claim_f, now));

        assert!(!work_claims_conflict(&claim_a, &claim_a, now));
    }

    #[test]
    fn coordination_signal_requires_signal_kind_and_open_status() {
        let signal_item = make_test_item(
            "1",
            ContinuityKind::Signal,
            ContinuityStatus::Active,
            serde_json::json!({
                "coordination_signal": {
                    "lane": "anxiety",
                    "severity": "warn"
                }
            }),
        );
        assert!(coordination_signal(&signal_item).is_some());

        let resolved_signal = make_test_item(
            "2",
            ContinuityKind::Signal,
            ContinuityStatus::Resolved,
            serde_json::json!({
                "coordination_signal": { "lane": "anxiety", "severity": "warn" }
            }),
        );
        assert!(coordination_signal(&resolved_signal).is_none());

        let wrong_kind = make_test_item(
            "3",
            ContinuityKind::Fact,
            ContinuityStatus::Open,
            serde_json::json!({
                "coordination_signal": { "lane": "anxiety", "severity": "warn" }
            }),
        );
        assert!(coordination_signal(&wrong_kind).is_none());
    }

    #[test]
    fn trim_text_within_limit_returns_trimmed() {
        assert_eq!(trim_text("  hello  ", 100), "hello");
    }

    #[test]
    fn trim_text_beyond_limit_truncates_with_ellipsis() {
        let long = "a".repeat(300);
        let trimmed = trim_text(&long, 10);
        assert!(trimmed.ends_with("..."));
        assert!(trimmed.chars().count() <= 10);
    }

    #[test]
    fn augment_dimensions_avoids_duplicates() {
        let base = vec![DimensionValue {
            key: "k1".into(),
            value: "v1".into(),
            weight: DEFAULT_DIMENSION_WEIGHT,
        }];
        let extra = vec![
            DimensionValue {
                key: "k1".into(),
                value: "v1".into(),
                weight: DEFAULT_DIMENSION_WEIGHT,
            },
            DimensionValue {
                key: "k2".into(),
                value: "v2".into(),
                weight: 50,
            },
        ];
        let merged = augment_dimensions(base, extra);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].key, "k1");
        assert_eq!(merged[1].key, "k2");
    }

    #[test]
    fn default_functions_return_expected_values() {
        assert_eq!(default_support_weight(), 1.0);
        assert_eq!(default_candidate_limit(), 24);
        assert_eq!(default_token_budget(), 384);
        assert_eq!(default_snapshot_resolution(), SnapshotResolution::Medium);
        assert_eq!(default_work_claim_lease_seconds(), 180);
    }

    #[test]
    fn if_empty_then_uses_fallback_for_blank() {
        assert_eq!("".to_string().if_empty_then("fallback"), "fallback");
        assert_eq!("  ".to_string().if_empty_then("fallback"), "fallback");
        assert_eq!("value".to_string().if_empty_then("fallback"), "value");
    }

    #[test]
    fn continuity_kind_serde_roundtrip() {
        let kind = ContinuityKind::OperationalScar;
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(json, "\"operational_scar\"");
        let back: ContinuityKind = serde_json::from_str(&json).unwrap();
        assert_eq!(back, kind);
    }

    #[test]
    fn continuity_status_serde_roundtrip() {
        for status in [
            ContinuityStatus::Open,
            ContinuityStatus::Active,
            ContinuityStatus::Resolved,
            ContinuityStatus::Superseded,
            ContinuityStatus::Rejected,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: ContinuityStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status);
            assert_eq!(json.trim_matches('"'), status.as_str());
        }
    }

    #[test]
    fn coordination_lane_serde_roundtrip() {
        for lane in [
            CoordinationLane::Anxiety,
            CoordinationLane::Review,
            CoordinationLane::Warning,
            CoordinationLane::Backoff,
            CoordinationLane::Coach,
        ] {
            let json = serde_json::to_string(&lane).unwrap();
            let back: CoordinationLane = serde_json::from_str(&json).unwrap();
            assert_eq!(back, lane);
        }
    }

    #[test]
    fn coordination_severity_serde_roundtrip() {
        for severity in [
            CoordinationSeverity::Info,
            CoordinationSeverity::Warn,
            CoordinationSeverity::Block,
        ] {
            let json = serde_json::to_string(&severity).unwrap();
            let back: CoordinationSeverity = serde_json::from_str(&json).unwrap();
            assert_eq!(back, severity);
        }
    }
}
