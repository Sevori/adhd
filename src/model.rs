use std::fmt;

use chrono::{DateTime, Utc};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

macro_rules! impl_snake_case_display {
    ($ty:ty, $($variant:ident => $label:expr),+ $(,)?) => {
        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(match self { $(Self::$variant => $label),+ })
            }
        }
    };
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    Prompt,
    Response,
    ToolCall,
    ToolResult,
    ShellCommand,
    ShellOutput,
    FileDiff,
    Error,
    Exception,
    Document,
    Trace,
    ApiRequest,
    ApiResponse,
    Note,
}

impl_snake_case_display!(EventKind,
    Prompt => "prompt", Response => "response", ToolCall => "tool_call",
    ToolResult => "tool_result", ShellCommand => "shell_command",
    ShellOutput => "shell_output", FileDiff => "file_diff", Error => "error",
    Exception => "exception", Document => "document", Trace => "trace",
    ApiRequest => "api_request", ApiResponse => "api_response", Note => "note",
);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    Agent,
    Session,
    Shared,
    Project,
    Global,
}

impl_snake_case_display!(Scope,
    Agent => "agent", Session => "session", Shared => "shared",
    Project => "project", Global => "global",
);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum MemoryLayer {
    Hot = 1,
    Episodic = 2,
    Semantic = 3,
    Summary = 4,
    Cold = 5,
}

impl MemoryLayer {
    pub fn as_i64(self) -> i64 {
        self as i64
    }
}

impl_snake_case_display!(MemoryLayer,
    Hot => "hot", Episodic => "episodic", Semantic => "semantic",
    Summary => "summary", Cold => "cold",
);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DimensionValue {
    pub key: String,
    pub value: String,
    #[serde(default = "default_dimension_weight")]
    pub weight: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DimensionFilter {
    pub key: String,
    #[serde(default)]
    pub values: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct Selector {
    #[serde(default)]
    pub all: Vec<DimensionFilter>,
    #[serde(default)]
    pub any: Vec<DimensionFilter>,
    #[serde(default)]
    pub exclude: Vec<DimensionFilter>,
    #[serde(default)]
    pub layers: Vec<MemoryLayer>,
    pub start_ts: Option<DateTime<Utc>>,
    pub end_ts: Option<DateTime<Utc>>,
    pub limit: Option<usize>,
    pub namespace: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ViewOp {
    Project,
    Slice,
    Intersect,
    Union,
    Snapshot,
    Fork,
    Merge,
}

impl_snake_case_display!(ViewOp,
    Project => "project", Slice => "slice", Intersect => "intersect",
    Union => "union", Snapshot => "snapshot", Fork => "fork", Merge => "merge",
);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotResolution {
    Coarse,
    Medium,
    Fine,
}

impl_snake_case_display!(SnapshotResolution,
    Coarse => "coarse", Medium => "medium", Fine => "fine",
);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventInput {
    pub kind: EventKind,
    pub agent_id: String,
    pub agent_role: Option<String>,
    #[serde(default)]
    pub timestamp: Option<DateTime<Utc>>,
    pub session_id: String,
    pub task_id: Option<String>,
    pub project_id: Option<String>,
    pub goal_id: Option<String>,
    pub run_id: Option<String>,
    pub namespace: Option<String>,
    pub environment: Option<String>,
    pub source: String,
    pub scope: Scope,
    pub tags: Vec<String>,
    #[serde(default)]
    pub dimensions: Vec<DimensionValue>,
    pub content: String,
    pub attributes: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRecord {
    pub id: String,
    pub ts: DateTime<Utc>,
    #[serde(flatten)]
    pub input: EventInput,
    pub content_hash: String,
    pub byte_size: usize,
    pub token_estimate: usize,
    pub importance: f64,
    pub segment_seq: i64,
    pub segment_line: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub layer: MemoryLayer,
    pub scope: Scope,
    pub agent_id: String,
    pub session_id: String,
    pub task_id: Option<String>,
    pub ts: DateTime<Utc>,
    pub importance: f64,
    pub confidence: f64,
    pub token_estimate: usize,
    pub source_event_id: Option<String>,
    pub scope_key: String,
    pub body: String,
    pub extra: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageRecord {
    pub parent_id: String,
    pub child_id: String,
    pub edge_kind: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationRecord {
    pub id: String,
    pub ts: DateTime<Utc>,
    pub source_id: String,
    pub target_id: String,
    pub relation: String,
    pub weight: f64,
    pub attributes: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestManifest {
    pub event: EventRecord,
    pub hot_memory_id: String,
    pub episodic_memory_id: String,
    pub semantic_memory_id: Option<String>,
    pub summary_memory_id: Option<String>,
    pub entities: Vec<SemanticEntity>,
    pub timings_ms: IngestTimings,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IngestTimings {
    pub raw_log_append: u128,
    pub sqlite_insert: u128,
    pub hot_promotion: u128,
    pub episode_promotion: u128,
    pub semantic_promotion: u128,
    pub summary_promotion: u128,
    pub total: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub prometheus_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayRow {
    pub event: EventRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEntity {
    pub value: String,
    pub normalized: String,
    pub kind: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewInput {
    pub op: ViewOp,
    pub owner_agent_id: Option<String>,
    pub namespace: Option<String>,
    pub objective: Option<String>,
    #[serde(default)]
    pub selectors: Vec<Selector>,
    #[serde(default)]
    pub source_view_ids: Vec<String>,
    pub resolution: Option<SnapshotResolution>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewConflict {
    pub key: String,
    pub values: Vec<String>,
    pub memory_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewItem {
    pub memory_id: String,
    pub layer: MemoryLayer,
    pub token_estimate: usize,
    pub score: f64,
    pub matched_dimensions: Vec<String>,
    pub why: Vec<String>,
    pub provenance: serde_json::Value,
    pub body: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewManifest {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub input: ViewInput,
    pub item_count: usize,
    pub conflict_count: usize,
    pub selected: Vec<ViewItem>,
    pub conflicts: Vec<ViewConflict>,
    pub timings_ms: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewRecord {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub input: ViewInput,
    pub item_count: usize,
    pub conflict_count: usize,
    pub manifest_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffInput {
    pub from_agent_id: String,
    pub to_agent_id: String,
    pub reason: String,
    pub query_text: String,
    pub budget_tokens: usize,
    pub view_id: Option<String>,
    pub selector: Option<Selector>,
    pub objective: Option<String>,
    pub namespace: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffRecord {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub from_agent_id: String,
    pub to_agent_id: String,
    pub reason: String,
    pub view_id: String,
    pub pack_id: String,
    pub conflict_count: usize,
    pub manifest_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionInput {
    pub agent_id: String,
    pub name: Option<String>,
    pub selector: Selector,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionRecord {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub input: SubscriptionInput,
    pub cursor_ts: Option<DateTime<Utc>>,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionPoll {
    pub subscription_id: String,
    pub cursor_ts: Option<DateTime<Utc>>,
    pub items: Vec<ViewItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryInput {
    pub agent_id: Option<String>,
    pub session_id: Option<String>,
    pub task_id: Option<String>,
    pub namespace: Option<String>,
    pub objective: Option<String>,
    pub selector: Option<Selector>,
    pub view_id: Option<String>,
    pub query_text: String,
    pub budget_tokens: usize,
    pub candidate_limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScoreBreakdown {
    pub continuity: f64,
    pub practice_evidence: f64,
    pub continuity_kind: f64,
    pub continuity_status: f64,
    pub selector: f64,
    pub lexical: f64,
    pub vector: f64,
    pub entity: f64,
    pub temporal: f64,
    pub recency: f64,
    pub salience: f64,
    pub lineage: f64,
    pub scope: f64,
    pub view: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateRecord {
    pub memory: MemoryRecord,
    pub final_score: f64,
    pub breakdown: ScoreBreakdown,
    pub why: Vec<String>,
    pub provenance: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPackItem {
    pub memory_id: String,
    pub layer: MemoryLayer,
    pub token_estimate: usize,
    pub final_score: f64,
    pub why: Vec<String>,
    pub breakdown: ScoreBreakdown,
    pub provenance: serde_json::Value,
    pub body: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectedCandidate {
    pub memory_id: String,
    pub layer: MemoryLayer,
    pub token_estimate: usize,
    pub final_score: f64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPackManifest {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub query: QueryInput,
    pub used_tokens: usize,
    pub selected: Vec<ContextPackItem>,
    pub rejected: Vec<RejectedCandidate>,
    pub timings_ms: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPack {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub query: QueryInput,
    pub used_tokens: usize,
    pub items: Vec<ContextPackItem>,
    pub manifest_path: String,
}

fn default_dimension_weight() -> i32 {
    100
}
