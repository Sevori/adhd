use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};
use regex::Regex;
use rusqlite::{Connection, OptionalExtension, params};
use sysinfo::System;
use tracing::debug;
use uuid::Uuid;
use zstd::stream::write::Encoder;

use crate::config::{EngineConfig, EnginePaths};
use crate::continuity::{
    AgentAttachmentRecord, AgentBadgeRecord, AttachAgentInput, ClaimWorkInput, ContextRecord,
    ContextStatus, ContinuityCompiledChunkRecord, ContinuityCompilerStateRecord,
    ContinuityItemInput, ContinuityItemRecord, ContinuityKind, ContinuityPlasticityState,
    ContinuityRecall, ContinuityRecallCompiler, ContinuityRecallItem, ContinuityRetentionState,
    ContinuityStatus, CoordinationLane, CoordinationSeverity, DEFAULT_MACHINE_TASK_ID,
    HeartbeatInput, LaneProjectionRecord, MACHINE_NAMESPACE_ALIAS, MachineProfile,
    OpenContextInput, OutcomeInput, PracticeLifecycleState, ResolveOrSupersedeInput,
    SnapshotManifest, SnapshotRecord, SupportRef, UpsertAgentBadgeInput, WorkClaimConflict,
    WorkClaimCoordination, coordination_signal, default_work_claim_lease_seconds,
    merge_work_claim_extra, normalize_work_claim_resources, work_claim_coordination,
    work_claim_is_live, work_claim_key, work_claims_conflict,
};
use crate::embedding::{EmbeddingRuntime, l2_norm};
use crate::model::{
    ContextPack, ContextPackManifest, DimensionFilter, DimensionValue, EventInput, EventRecord,
    HandoffRecord, IngestManifest, IngestTimings, LineageRecord, MemoryLayer, MemoryRecord,
    RejectedCandidate, RelationRecord, ReplayRow, Scope, Selector, SemanticEntity,
    SnapshotResolution, SubscriptionInput, SubscriptionPoll, SubscriptionRecord, ViewConflict,
    ViewInput, ViewItem, ViewManifest, ViewOp, ViewRecord,
};
use crate::telemetry::EngineTelemetry;

pub struct Storage {
    pub config: EngineConfig,
    pub paths: EnginePaths,
    embedding: EmbeddingRuntime,
    conn: Connection,
}

#[cfg(test)]
struct StorageBytesMetricsTestHook {
    log_dir: PathBuf,
    entered_tx: std::sync::mpsc::Sender<()>,
    release_rx: std::sync::mpsc::Receiver<()>,
}

#[cfg(test)]
static STORAGE_BYTES_METRICS_TEST_HOOK: std::sync::Mutex<Option<StorageBytesMetricsTestHook>> =
    std::sync::Mutex::new(None);

#[cfg(test)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct MetricsLiveStateQueryCounts {
    active_attachment_reads: usize,
    continuity_snapshot_reads: usize,
    live_work_claim_reads: usize,
    active_signal_reads: usize,
}

#[cfg(test)]
struct MetricsLiveStateQueryCounterHook {
    root: PathBuf,
    counts: MetricsLiveStateQueryCounts,
}

#[cfg(test)]
static METRICS_LIVE_STATE_QUERY_COUNTER_HOOK: std::sync::Mutex<
    Option<MetricsLiveStateQueryCounterHook>,
> = std::sync::Mutex::new(None);

#[derive(Debug)]
struct ActiveAttachmentMetric {
    attachment_id: String,
    agent_id: String,
    agent_type: String,
    namespace: String,
    role: Option<String>,
    metadata: serde_json::Value,
    context_id: Option<String>,
    last_seen_at: DateTime<Utc>,
    tick_count: i64,
}

#[derive(Debug, Default)]
struct LiveContinuityMetricSnapshot {
    live_work_claims: Vec<ContinuityItemRecord>,
    active_signals: Vec<ContinuityItemRecord>,
}

#[derive(Debug, Clone)]
struct StoredAgentBadge {
    display_name: String,
    status: String,
    focus: String,
    headline: String,
    resource: Option<String>,
    repo_root: Option<String>,
    branch: Option<String>,
    metadata: serde_json::Value,
    context_id: Option<String>,
    task_id: Option<String>,
    updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct DerivedAgentBadge {
    context_id: Option<String>,
    task_id: Option<String>,
    display_name: String,
    status: String,
    focus: String,
    headline: String,
    resource: Option<String>,
    repo_root: Option<String>,
    branch: Option<String>,
    metadata: serde_json::Value,
}

#[derive(Debug, Clone)]
struct LaneProjectionIdentity {
    namespace: String,
    projection_id: String,
    projection_kind: String,
    label: String,
    resource: Option<String>,
    repo_root: Option<String>,
    branch: Option<String>,
    task_id: Option<String>,
}

#[derive(Debug, Clone)]
struct LaneProjectionAccumulator {
    identity: LaneProjectionIdentity,
    agent_ids: BTreeSet<String>,
    display_names: BTreeSet<String>,
    focus: String,
    headline: String,
    updated_at: DateTime<Utc>,
    live_claims: usize,
    claim_conflicts: usize,
    coordination_signal_count: usize,
    blocking_signal_count: usize,
    review_signal_count: usize,
    dispatch_assignment_count: usize,
    dispatch_assignment_anxiety_max: f64,
    dispatch_assignment_explicit_cli_count: usize,
    dispatch_assignment_live_badge_opt_in_count: usize,
    coordination_lanes: BTreeSet<String>,
}

impl LaneProjectionAccumulator {
    fn new(identity: LaneProjectionIdentity) -> Self {
        Self {
            identity,
            agent_ids: BTreeSet::new(),
            display_names: BTreeSet::new(),
            focus: String::new(),
            headline: String::new(),
            updated_at: Utc::now(),
            live_claims: 0,
            claim_conflicts: 0,
            coordination_signal_count: 0,
            blocking_signal_count: 0,
            review_signal_count: 0,
            dispatch_assignment_count: 0,
            dispatch_assignment_anxiety_max: 0.0,
            dispatch_assignment_explicit_cli_count: 0,
            dispatch_assignment_live_badge_opt_in_count: 0,
            coordination_lanes: BTreeSet::new(),
        }
    }

    fn absorb_badge(&mut self, badge: &AgentBadgeRecord) {
        self.agent_ids.insert(badge.agent_id.clone());
        self.display_names.insert(badge.display_name.clone());
        if badge.updated_at >= self.updated_at || self.focus.is_empty() {
            self.focus = badge.focus.clone();
            self.headline = badge.headline.clone();
            self.updated_at = badge.updated_at;
        }
    }

    fn absorb_claim(&mut self, claim: &ContinuityItemRecord) {
        self.live_claims += 1;
        if claim.updated_at >= self.updated_at {
            self.updated_at = claim.updated_at;
            if self.focus.is_empty() {
                self.focus = claim.title.clone();
            }
            if self.headline.is_empty() {
                self.headline = claim.body.clone();
            }
        }
    }

    fn absorb_signal(&mut self, signal: &ContinuityItemRecord, lane: &str) {
        self.coordination_signal_count += 1;
        if coordination_signal(signal)
            .map(|record| record.severity == CoordinationSeverity::Block.as_str())
            .unwrap_or(false)
        {
            self.blocking_signal_count += 1;
        }
        if lane == CoordinationLane::Review.as_str() {
            self.review_signal_count += 1;
        }
        self.coordination_lanes.insert(lane.to_string());
        if signal.updated_at >= self.updated_at || self.focus.is_empty() || self.headline.is_empty()
        {
            self.focus = signal.title.clone();
            self.headline = signal.body.clone();
            self.updated_at = signal.updated_at;
        }
    }

    fn finalize(self) -> LaneProjectionRecord {
        LaneProjectionRecord {
            projection_id: self.identity.projection_id,
            namespace: self.identity.namespace,
            projection_kind: self.identity.projection_kind,
            label: self.identity.label,
            resource: self.identity.resource,
            repo_root: self.identity.repo_root,
            branch: self.identity.branch,
            task_id: self.identity.task_id,
            connected_agents: self.agent_ids.len(),
            live_claims: self.live_claims,
            claim_conflicts: self.claim_conflicts,
            coordination_signal_count: self.coordination_signal_count,
            blocking_signal_count: self.blocking_signal_count,
            review_signal_count: self.review_signal_count,
            dispatch_assignment_count: self.dispatch_assignment_count,
            dispatch_assignment_anxiety_max: self.dispatch_assignment_anxiety_max,
            dispatch_assignment_explicit_cli_count: self.dispatch_assignment_explicit_cli_count,
            dispatch_assignment_live_badge_opt_in_count: self
                .dispatch_assignment_live_badge_opt_in_count,
            coordination_lanes: self.coordination_lanes.into_iter().collect(),
            agent_ids: self.agent_ids.into_iter().collect(),
            display_names: self.display_names.into_iter().collect(),
            focus: self.focus,
            headline: self.headline,
            updated_at: self.updated_at,
        }
    }
}

impl Storage {
    pub fn open(config: EngineConfig) -> Result<Self> {
        let paths = config.ensure_dirs()?;
        let conn = Connection::open(&paths.sqlite_path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.pragma_update(None, "temp_store", "MEMORY")?;
        conn.pragma_update(None, "mmap_size", 268_435_456_i64)?;
        let embedding = EmbeddingRuntime::from_config(&config.embedding_backend)?;
        let storage = Self {
            config,
            paths,
            embedding,
            conn,
        };
        storage.init_schema()?;
        storage.ensure_column(
            "memory_vectors",
            "backend_key",
            "TEXT NOT NULL DEFAULT 'hash:128'",
        )?;
        let _ = storage.machine_profile()?;
        storage.canonicalize_machine_aliases()?;
        Ok(storage)
    }

    pub fn embedding_backend_key(&self) -> String {
        self.embedding.backend_key()
    }

    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );

            INSERT OR IGNORE INTO meta(key, value) VALUES
              ('segment_seq', '0'),
              ('segment_line', '0');

            CREATE TABLE IF NOT EXISTS events (
              id TEXT PRIMARY KEY,
              ts TEXT NOT NULL,
              kind TEXT NOT NULL,
              scope TEXT NOT NULL,
              agent_id TEXT NOT NULL,
              agent_role TEXT,
              session_id TEXT NOT NULL,
              task_id TEXT,
              project_id TEXT,
              goal_id TEXT,
              run_id TEXT,
              namespace TEXT,
              environment TEXT,
              source TEXT NOT NULL,
              tags_json TEXT NOT NULL,
              dimensions_json TEXT NOT NULL,
              attributes_json TEXT NOT NULL,
              content TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              byte_size INTEGER NOT NULL,
              token_estimate INTEGER NOT NULL,
              importance REAL NOT NULL,
              segment_seq INTEGER NOT NULL,
              segment_line INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_events_session_ts ON events(session_id, ts DESC);
            CREATE INDEX IF NOT EXISTS idx_events_agent_ts ON events(agent_id, ts DESC);
            CREATE INDEX IF NOT EXISTS idx_events_task_ts ON events(task_id, ts DESC);

            CREATE TABLE IF NOT EXISTS memory_items (
              id TEXT PRIMARY KEY,
              layer INTEGER NOT NULL,
              scope TEXT NOT NULL,
              agent_id TEXT NOT NULL,
              session_id TEXT NOT NULL,
              task_id TEXT,
              ts TEXT NOT NULL,
              importance REAL NOT NULL,
              confidence REAL NOT NULL,
              token_estimate INTEGER NOT NULL,
              source_event_id TEXT,
              scope_key TEXT NOT NULL,
              body TEXT NOT NULL,
              extra_json TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_layer_scope_key
              ON memory_items(layer, scope_key);
            CREATE INDEX IF NOT EXISTS idx_memory_layer_ts ON memory_items(layer, ts DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_session_layer ON memory_items(session_id, layer, ts DESC);

            DROP TRIGGER IF EXISTS memory_items_ai;
            DROP TRIGGER IF EXISTS memory_items_ad;
            DROP TRIGGER IF EXISTS memory_items_au;

            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
              memory_id UNINDEXED,
              body,
              tokenize = 'unicode61 remove_diacritics 2 tokenchars ''_-./'''
            );

            CREATE TABLE IF NOT EXISTS lineage (
              parent_id TEXT NOT NULL,
              child_id TEXT NOT NULL,
              edge_kind TEXT NOT NULL,
              weight REAL NOT NULL,
              PRIMARY KEY(parent_id, child_id, edge_kind)
            );

            CREATE INDEX IF NOT EXISTS idx_lineage_child ON lineage(child_id);

            CREATE TABLE IF NOT EXISTS entities (
              id TEXT PRIMARY KEY,
              normalized TEXT NOT NULL UNIQUE,
              value TEXT NOT NULL,
              kind TEXT NOT NULL,
              weight REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entity_mentions (
              entity_id TEXT NOT NULL,
              memory_id TEXT NOT NULL,
              event_id TEXT,
              role TEXT NOT NULL,
              ts TEXT NOT NULL,
              weight REAL NOT NULL,
              PRIMARY KEY(entity_id, memory_id, role)
            );

            CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id, ts DESC);
            CREATE INDEX IF NOT EXISTS idx_entity_mentions_memory ON entity_mentions(memory_id);

            CREATE TABLE IF NOT EXISTS memory_vectors (
              memory_id TEXT PRIMARY KEY,
              backend_key TEXT NOT NULL DEFAULT 'hash:128',
              dim INTEGER NOT NULL,
              norm REAL NOT NULL,
              data BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_vectors_backend ON memory_vectors(backend_key);

            CREATE TABLE IF NOT EXISTS context_packs (
              id TEXT PRIMARY KEY,
              ts TEXT NOT NULL,
              agent_id TEXT,
              session_id TEXT,
              task_id TEXT,
              query_text TEXT NOT NULL,
              budget_tokens INTEGER NOT NULL,
              total_candidate_count INTEGER NOT NULL,
              total_selected_count INTEGER NOT NULL,
              manifest_path TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS context_pack_items (
              pack_id TEXT NOT NULL,
              memory_id TEXT NOT NULL,
              included INTEGER NOT NULL,
              final_score REAL NOT NULL,
              rank INTEGER NOT NULL,
              token_estimate INTEGER NOT NULL,
              reason_json TEXT NOT NULL,
              PRIMARY KEY(pack_id, memory_id)
            );

            CREATE TABLE IF NOT EXISTS continuity_plasticity (
              continuity_id TEXT PRIMARY KEY,
              belief_key TEXT,
              source_role TEXT,
              activation_count INTEGER NOT NULL,
              successful_use_count INTEGER NOT NULL,
              confirmation_count INTEGER NOT NULL,
              contradiction_count INTEGER NOT NULL,
              independent_source_count INTEGER NOT NULL,
              spaced_reactivation_count INTEGER NOT NULL DEFAULT 0,
              spacing_interval_hours REAL NOT NULL DEFAULT 6.0,
              last_reactivated_at TEXT,
              last_confirmed_at TEXT,
              last_contradicted_at TEXT,
              last_strengthened_at TEXT,
              stability_score REAL NOT NULL,
              prediction_error REAL NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_continuity_plasticity_belief_key
              ON continuity_plasticity(belief_key);

            CREATE TABLE IF NOT EXISTS item_dimensions (
              item_id TEXT NOT NULL,
              item_type TEXT NOT NULL,
              ts TEXT NOT NULL,
              layer INTEGER,
              key TEXT NOT NULL,
              value TEXT NOT NULL,
              weight INTEGER NOT NULL,
              PRIMARY KEY(item_id, item_type, key, value)
            );

            CREATE INDEX IF NOT EXISTS idx_item_dimensions_lookup
              ON item_dimensions(item_type, key, value, ts DESC);
            CREATE INDEX IF NOT EXISTS idx_item_dimensions_item
              ON item_dimensions(item_id, item_type);

            CREATE TABLE IF NOT EXISTS fabric_relations (
              id TEXT PRIMARY KEY,
              ts TEXT NOT NULL,
              source_id TEXT NOT NULL,
              target_id TEXT NOT NULL,
              relation TEXT NOT NULL,
              weight REAL NOT NULL,
              attributes_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_fabric_relations_source
              ON fabric_relations(source_id, relation, ts DESC);
            CREATE INDEX IF NOT EXISTS idx_fabric_relations_target
              ON fabric_relations(target_id, relation, ts DESC);

            CREATE TABLE IF NOT EXISTS views (
              id TEXT PRIMARY KEY,
              ts TEXT NOT NULL,
              op TEXT NOT NULL,
              owner_agent_id TEXT,
              namespace TEXT,
              objective TEXT,
              selector_json TEXT NOT NULL,
              source_view_ids_json TEXT NOT NULL,
              resolution TEXT,
              item_count INTEGER NOT NULL,
              conflict_count INTEGER NOT NULL,
              manifest_path TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_views_ts ON views(ts DESC);

            CREATE TABLE IF NOT EXISTS view_items (
              view_id TEXT NOT NULL,
              memory_id TEXT NOT NULL,
              rank INTEGER NOT NULL,
              score REAL NOT NULL,
              token_estimate INTEGER NOT NULL,
              matched_dimensions_json TEXT NOT NULL,
              why_json TEXT NOT NULL,
              PRIMARY KEY(view_id, memory_id)
            );

            CREATE INDEX IF NOT EXISTS idx_view_items_memory ON view_items(memory_id);

            CREATE TABLE IF NOT EXISTS handoffs (
              id TEXT PRIMARY KEY,
              ts TEXT NOT NULL,
              from_agent_id TEXT NOT NULL,
              to_agent_id TEXT NOT NULL,
              reason TEXT NOT NULL,
              query_text TEXT NOT NULL,
              view_id TEXT NOT NULL,
              pack_id TEXT NOT NULL,
              conflict_count INTEGER NOT NULL,
              manifest_path TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_handoffs_agents
              ON handoffs(from_agent_id, to_agent_id, ts DESC);

            CREATE TABLE IF NOT EXISTS subscriptions (
              id TEXT PRIMARY KEY,
              ts TEXT NOT NULL,
              agent_id TEXT NOT NULL,
              name TEXT,
              selector_json TEXT NOT NULL,
              cursor_ts TEXT,
              active INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_subscriptions_agent
              ON subscriptions(agent_id, active, ts DESC);

            CREATE TABLE IF NOT EXISTS agent_attachments (
              id TEXT PRIMARY KEY,
              ts TEXT NOT NULL,
              agent_id TEXT NOT NULL,
              agent_type TEXT NOT NULL,
              namespace TEXT NOT NULL,
              role TEXT,
              capabilities_json TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              active INTEGER NOT NULL,
              last_seen_at TEXT,
              tick_count INTEGER NOT NULL DEFAULT 0,
              context_id TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_agent_attachments_agent
              ON agent_attachments(agent_id, active, ts DESC);
            CREATE INDEX IF NOT EXISTS idx_agent_attachments_namespace
              ON agent_attachments(namespace, active, ts DESC);

            CREATE TABLE IF NOT EXISTS agent_badges (
              attachment_id TEXT PRIMARY KEY,
              updated_at TEXT NOT NULL,
              agent_id TEXT NOT NULL,
              display_name TEXT NOT NULL,
              status TEXT NOT NULL,
              focus TEXT NOT NULL,
              headline TEXT NOT NULL,
              resource TEXT,
              repo_root TEXT,
              branch TEXT,
              metadata_json TEXT NOT NULL,
              context_id TEXT,
              task_id TEXT,
              FOREIGN KEY(attachment_id) REFERENCES agent_attachments(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_agent_badges_agent
              ON agent_badges(agent_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_agent_badges_context
              ON agent_badges(context_id, task_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS contexts (
              id TEXT PRIMARY KEY,
              opened_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              namespace TEXT NOT NULL,
              task_id TEXT NOT NULL,
              session_id TEXT NOT NULL,
              objective TEXT NOT NULL,
              selector_json TEXT NOT NULL,
              status TEXT NOT NULL,
              current_agent_id TEXT,
              current_attachment_id TEXT,
              last_snapshot_id TEXT,
              UNIQUE(namespace, task_id)
            );

            CREATE INDEX IF NOT EXISTS idx_contexts_namespace_task
              ON contexts(namespace, task_id);

            CREATE TABLE IF NOT EXISTS continuity_items (
              id TEXT PRIMARY KEY,
              memory_id TEXT NOT NULL UNIQUE,
              ts TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              context_id TEXT NOT NULL,
              namespace TEXT NOT NULL,
              task_id TEXT NOT NULL,
              author_agent_id TEXT NOT NULL,
              kind TEXT NOT NULL,
              scope TEXT NOT NULL,
              status TEXT NOT NULL,
              title TEXT NOT NULL,
              body TEXT NOT NULL,
              importance REAL NOT NULL,
              confidence REAL NOT NULL,
              salience REAL NOT NULL,
              supersedes_id TEXT,
              resolved_at TEXT,
              extra_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_continuity_context_kind_status
              ON continuity_items(context_id, kind, status, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_continuity_context_status_salience
              ON continuity_items(context_id, status, salience DESC, updated_at DESC, ts DESC);

            CREATE INDEX IF NOT EXISTS idx_continuity_context_salience
              ON continuity_items(context_id, salience DESC, updated_at DESC, ts DESC);

            CREATE TABLE IF NOT EXISTS continuity_support (
              continuity_id TEXT NOT NULL,
              support_type TEXT NOT NULL,
              support_id TEXT NOT NULL,
              reason TEXT,
              weight REAL NOT NULL,
              PRIMARY KEY(continuity_id, support_type, support_id, reason)
            );

            CREATE INDEX IF NOT EXISTS idx_continuity_support_continuity
              ON continuity_support(continuity_id, weight DESC);

            CREATE TABLE IF NOT EXISTS continuity_compiler_state (
              context_id TEXT PRIMARY KEY,
              dirty INTEGER NOT NULL,
              item_count INTEGER NOT NULL,
              refreshed_at TEXT NOT NULL,
              compiled_at TEXT
            );

            CREATE TABLE IF NOT EXISTS continuity_compiled_chunks (
              chunk_id TEXT PRIMARY KEY,
              context_id TEXT NOT NULL,
              band TEXT NOT NULL,
              compiled_at TEXT NOT NULL,
              item_count INTEGER NOT NULL,
              item_ids_json TEXT NOT NULL,
              body TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_continuity_compiled_context_band
              ON continuity_compiled_chunks(context_id, band);

            CREATE VIRTUAL TABLE IF NOT EXISTS continuity_compiled_fts USING fts5(
              chunk_id UNINDEXED,
              context_id UNINDEXED,
              band UNINDEXED,
              body
            );

            CREATE TABLE IF NOT EXISTS context_snapshots (
              id TEXT PRIMARY KEY,
              ts TEXT NOT NULL,
              context_id TEXT NOT NULL,
              resolution TEXT NOT NULL,
              objective TEXT NOT NULL,
              selector_json TEXT NOT NULL,
              view_id TEXT NOT NULL,
              pack_id TEXT NOT NULL,
              manifest_path TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_context_snapshots_context
              ON context_snapshots(context_id, ts DESC);
            "#
        )?;
        self.ensure_column("events", "agent_role", "TEXT")?;
        self.ensure_column("events", "project_id", "TEXT")?;
        self.ensure_column("events", "goal_id", "TEXT")?;
        self.ensure_column("events", "run_id", "TEXT")?;
        self.ensure_column("events", "namespace", "TEXT")?;
        self.ensure_column("events", "environment", "TEXT")?;
        self.ensure_column("events", "dimensions_json", "TEXT NOT NULL DEFAULT '[]'")?;
        self.ensure_column(
            "continuity_plasticity",
            "spaced_reactivation_count",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        self.ensure_column(
            "continuity_plasticity",
            "spacing_interval_hours",
            "REAL NOT NULL DEFAULT 6.0",
        )?;
        self.ensure_column("continuity_plasticity", "last_strengthened_at", "TEXT")?;
        self.ensure_column("agent_attachments", "last_seen_at", "TEXT")?;
        self.ensure_column(
            "agent_attachments",
            "tick_count",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        self.ensure_continuity_support_reason_identity()?;
        Ok(())
    }

    fn ensure_continuity_support_reason_identity(&self) -> Result<()> {
        let mut stmt = self.conn.prepare("PRAGMA table_info(continuity_support)")?;
        let mut rows = stmt.query([])?;
        let mut has_reason_in_primary_key = false;
        while let Some(row) = rows.next()? {
            let column_name = row.get::<_, String>(1)?;
            let pk_ordinal = row.get::<_, i64>(5)?;
            if column_name == "reason" && pk_ordinal > 0 {
                has_reason_in_primary_key = true;
                break;
            }
        }
        if has_reason_in_primary_key {
            return Ok(());
        }

        self.conn.execute_batch(
            r#"
            ALTER TABLE continuity_support RENAME TO continuity_support_legacy;

            CREATE TABLE continuity_support (
              continuity_id TEXT NOT NULL,
              support_type TEXT NOT NULL,
              support_id TEXT NOT NULL,
              reason TEXT,
              weight REAL NOT NULL,
              PRIMARY KEY(continuity_id, support_type, support_id, reason)
            );

            INSERT INTO continuity_support(continuity_id, support_type, support_id, reason, weight)
            SELECT continuity_id, support_type, support_id, reason, weight
            FROM continuity_support_legacy;

            DROP TABLE continuity_support_legacy;

            CREATE INDEX IF NOT EXISTS idx_continuity_support_continuity
              ON continuity_support(continuity_id, weight DESC);
            "#,
        )?;

        Ok(())
    }

    pub fn machine_profile(&self) -> Result<MachineProfile> {
        if let Some(value) = self.meta_string("machine_profile_json")? {
            return Ok(serde_json::from_str(&value)?);
        }

        let host_name = System::host_name()
            .or_else(|| std::env::var("HOSTNAME").ok())
            .unwrap_or_else(|| "localhost".to_string());
        let os_name = System::name().unwrap_or_else(|| std::env::consts::OS.to_string());
        let kernel_version = System::kernel_version();
        let machine_id = Uuid::now_v7().to_string();
        let profile = MachineProfile {
            machine_id: machine_id.clone(),
            label: format!("adhd@{host_name}"),
            namespace: format!("machine:{machine_id}"),
            default_task_id: DEFAULT_MACHINE_TASK_ID.to_string(),
            host_name,
            os_name,
            kernel_version,
            storage_root: self.paths.root.display().to_string(),
        };
        self.set_meta_string("machine_profile_json", &serde_json::to_string(&profile)?)?;
        Ok(profile)
    }

    pub fn resolve_namespace_alias(&self, namespace: Option<&str>) -> Result<Option<String>> {
        let Some(namespace) = namespace else {
            return Ok(None);
        };
        if namespace == MACHINE_NAMESPACE_ALIAS {
            return Ok(Some(self.machine_profile()?.namespace));
        }
        Ok(Some(namespace.to_string()))
    }

    fn canonicalize_machine_aliases(&self) -> Result<()> {
        let canonical_namespace = self.machine_profile()?.namespace;
        let json_alias = format!("\"{MACHINE_NAMESPACE_ALIAS}\"");
        let json_namespace = format!("\"{canonical_namespace}\"");

        let mut stmt = self.conn.prepare(
            "SELECT id, task_id FROM contexts WHERE namespace = ?1 ORDER BY opened_at ASC",
        )?;
        let alias_contexts = stmt
            .query_map(params![MACHINE_NAMESPACE_ALIAS], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        for (alias_context_id, task_id) in alias_contexts {
            let canonical_context_id = self
                .conn
                .query_row(
                    "SELECT id FROM contexts WHERE namespace = ?1 AND task_id = ?2",
                    params![canonical_namespace, task_id],
                    |row| row.get::<_, String>(0),
                )
                .optional()?;
            if let Some(canonical_context_id) = canonical_context_id {
                self.conn.execute(
                    "UPDATE agent_attachments SET context_id = ?1 WHERE context_id = ?2",
                    params![canonical_context_id, alias_context_id],
                )?;
                self.conn.execute(
                    "UPDATE continuity_items SET context_id = ?1, namespace = ?2 WHERE context_id = ?3",
                    params![canonical_context_id, canonical_namespace, alias_context_id],
                )?;
                self.conn.execute(
                    "UPDATE context_snapshots SET context_id = ?1 WHERE context_id = ?2",
                    params![canonical_context_id, alias_context_id],
                )?;
                self.conn.execute(
                    "DELETE FROM contexts WHERE id = ?1",
                    params![alias_context_id],
                )?;
            } else {
                self.conn.execute(
                    "UPDATE contexts SET namespace = ?1, selector_json = REPLACE(selector_json, ?2, ?3) WHERE id = ?4",
                    params![
                        canonical_namespace,
                        json_alias,
                        json_namespace,
                        alias_context_id
                    ],
                )?;
            }
        }

        self.conn.execute(
            "UPDATE agent_attachments SET namespace = ?1 WHERE namespace = ?2",
            params![canonical_namespace, MACHINE_NAMESPACE_ALIAS],
        )?;
        self.conn.execute(
            "UPDATE continuity_items SET namespace = ?1 WHERE namespace = ?2",
            params![canonical_namespace, MACHINE_NAMESPACE_ALIAS],
        )?;
        self.conn.execute(
            "UPDATE events SET namespace = ?1 WHERE namespace = ?2",
            params![canonical_namespace, MACHINE_NAMESPACE_ALIAS],
        )?;
        self.conn.execute(
            "UPDATE views SET namespace = CASE WHEN namespace = ?2 THEN ?1 ELSE namespace END, selector_json = REPLACE(selector_json, ?3, ?4) WHERE namespace = ?2 OR instr(selector_json, ?3) > 0",
            params![
                canonical_namespace,
                MACHINE_NAMESPACE_ALIAS,
                json_alias,
                json_namespace
            ],
        )?;
        self.conn.execute(
            "UPDATE subscriptions SET selector_json = REPLACE(selector_json, ?1, ?2) WHERE instr(selector_json, ?1) > 0",
            params![json_alias, json_namespace],
        )?;
        self.conn.execute(
            "UPDATE context_snapshots SET selector_json = REPLACE(selector_json, ?1, ?2) WHERE instr(selector_json, ?1) > 0",
            params![json_alias, json_namespace],
        )?;
        self.conn.execute(
            "UPDATE item_dimensions SET value = ?1 WHERE key = 'namespace' AND value = ?2",
            params![canonical_namespace, MACHINE_NAMESPACE_ALIAS],
        )?;

        self.deactivate_duplicate_attachments(&canonical_namespace)?;
        self.supersede_duplicate_work_claims(&canonical_namespace)?;
        Ok(())
    }

    fn deactivate_duplicate_attachments(&self, canonical_namespace: &str) -> Result<()> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, agent_id
            FROM agent_attachments
            WHERE namespace = ?1 AND active = 1
            ORDER BY agent_id ASC, COALESCE(last_seen_at, ts) DESC, ts DESC
            "#,
        )?;
        let mut rows = stmt.query(params![canonical_namespace])?;
        let mut seen = HashSet::new();
        while let Some(row) = rows.next()? {
            let attachment_id = row.get::<_, String>(0)?;
            let agent_id = row.get::<_, String>(1)?;
            if !seen.insert(agent_id) {
                self.conn.execute(
                    "UPDATE agent_attachments SET active = 0 WHERE id = ?1",
                    params![attachment_id],
                )?;
            }
        }
        Ok(())
    }

    fn supersede_duplicate_work_claims(&self, canonical_namespace: &str) -> Result<()> {
        let now = Utc::now();
        let mut claims = self
            .list_live_work_claims()?
            .into_iter()
            .filter(|item| item.namespace == canonical_namespace)
            .collect::<Vec<_>>();
        claims.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
        let mut seen = HashMap::<(String, String, String, String), String>::new();
        for claim in claims {
            let resources = work_claim_coordination(&claim)
                .map(|coordination| coordination.resources.join("|"))
                .unwrap_or_default();
            let key = (
                claim.task_id.clone(),
                claim.author_agent_id.clone(),
                claim.scope.to_string(),
                format!("{}::{resources}", claim.title.trim().to_ascii_lowercase()),
            );
            if let Some(keep_id) = seen.get(&key) {
                self.conn.execute(
                    r#"
                    UPDATE continuity_items
                    SET status = ?2,
                        updated_at = ?3,
                        resolved_at = ?3,
                        supersedes_id = ?4
                    WHERE id = ?1
                    "#,
                    params![
                        claim.id,
                        ContinuityStatus::Superseded.as_str(),
                        now.to_rfc3339(),
                        keep_id
                    ],
                )?;
            } else {
                seen.insert(key, claim.id);
            }
        }
        Ok(())
    }

    pub fn ingest(
        &mut self,
        input: EventInput,
        telemetry: &EngineTelemetry,
    ) -> Result<IngestManifest> {
        let start_total = Instant::now();
        let now = input.timestamp.unwrap_or_else(Utc::now);
        let event_id = Uuid::now_v7().to_string();
        let content_hash = blake3::hash(input.content.as_bytes()).to_hex().to_string();
        let byte_size = input.content.len();
        let token_estimate = estimate_tokens(&input.content);
        let importance = infer_importance(&input);

        let start_append = Instant::now();
        let (segment_seq, segment_line) = self.append_raw_log(
            &event_id,
            now,
            &input,
            &content_hash,
            byte_size,
            token_estimate,
            importance,
        )?;
        telemetry.observe_raw_log_append_seconds(start_append.elapsed().as_secs_f64());

        let event = EventRecord {
            id: event_id.clone(),
            ts: now,
            input,
            content_hash,
            byte_size,
            token_estimate,
            importance,
            segment_seq,
            segment_line,
        };
        let entities = extract_entities(&event);

        let start_sql = Instant::now();
        self.insert_event(&event)?;
        self.index_item_dimensions(
            "event",
            &event.id,
            None,
            event.ts,
            &event_dimensions(&event, &entities),
        )?;
        telemetry.observe_sqlite_insert_seconds(start_sql.elapsed().as_secs_f64());

        let start_hot = Instant::now();
        let hot_memory_id = self.promote_hot(&event)?;
        telemetry.observe_promotion_seconds(start_hot.elapsed().as_secs_f64());

        let start_episode = Instant::now();
        let episodic_memory_id = self.promote_episode(&event)?;
        telemetry.observe_promotion_seconds(start_episode.elapsed().as_secs_f64());

        let start_semantic = Instant::now();
        let semantic_memory_id = self.promote_semantic(&event, &entities)?;
        telemetry.observe_promotion_seconds(start_semantic.elapsed().as_secs_f64());

        let start_summary = Instant::now();
        let summary_memory_id = self.promote_summary(&event, &entities)?;
        telemetry.observe_promotion_seconds(start_summary.elapsed().as_secs_f64());

        self.insert_lineage(&LineageRecord {
            parent_id: event.id.clone(),
            child_id: hot_memory_id.clone(),
            edge_kind: "promoted_hot".to_string(),
            weight: 1.0,
        })?;
        self.insert_lineage(&LineageRecord {
            parent_id: event.id.clone(),
            child_id: episodic_memory_id.clone(),
            edge_kind: "promoted_episodic".to_string(),
            weight: 0.9,
        })?;
        if let Some(semantic_memory_id) = &semantic_memory_id {
            self.insert_lineage(&LineageRecord {
                parent_id: event.id.clone(),
                child_id: semantic_memory_id.clone(),
                edge_kind: "promoted_semantic".to_string(),
                weight: 0.8,
            })?;
        }
        if let Some(summary_memory_id) = &summary_memory_id {
            self.insert_lineage(&LineageRecord {
                parent_id: event.id.clone(),
                child_id: summary_memory_id.clone(),
                edge_kind: "promoted_summary".to_string(),
                weight: 0.7,
            })?;
        }
        self.index_memory_dimensions(&hot_memory_id, MemoryLayer::Hot, &event, &[], "hot")?;
        self.index_memory_dimensions(
            &episodic_memory_id,
            MemoryLayer::Episodic,
            &event,
            &[],
            "episodic",
        )?;
        if let Some(semantic_memory_id) = &semantic_memory_id {
            self.index_memory_dimensions(
                semantic_memory_id,
                MemoryLayer::Semantic,
                &event,
                &entities,
                "semantic",
            )?;
            self.insert_relation(&RelationRecord {
                id: format!("rel:{}", Uuid::now_v7()),
                ts: event.ts,
                source_id: event.id.clone(),
                target_id: semantic_memory_id.clone(),
                relation: "derives_semantic".to_string(),
                weight: 0.8,
                attributes: serde_json::json!({}),
            })?;
        }
        if let Some(summary_memory_id) = &summary_memory_id {
            self.index_memory_dimensions(
                summary_memory_id,
                MemoryLayer::Summary,
                &event,
                &entities,
                "summary",
            )?;
            self.insert_relation(&RelationRecord {
                id: format!("rel:{}", Uuid::now_v7()),
                ts: event.ts,
                source_id: event.id.clone(),
                target_id: summary_memory_id.clone(),
                relation: "derives_summary".to_string(),
                weight: 0.7,
                attributes: serde_json::json!({}),
            })?;
        }
        self.insert_relation(&RelationRecord {
            id: format!("rel:{}", Uuid::now_v7()),
            ts: event.ts,
            source_id: event.id.clone(),
            target_id: hot_memory_id.clone(),
            relation: "derives_hot".to_string(),
            weight: 1.0,
            attributes: serde_json::json!({}),
        })?;
        self.insert_relation(&RelationRecord {
            id: format!("rel:{}", Uuid::now_v7()),
            ts: event.ts,
            source_id: event.id.clone(),
            target_id: episodic_memory_id.clone(),
            relation: "derives_episode".to_string(),
            weight: 0.9,
            attributes: serde_json::json!({}),
        })?;

        self.touch_active_attachment(
            &event.input.agent_id,
            event.input.namespace.as_deref(),
            None,
        )?;
        self.refresh_layer_counts(telemetry)?;
        telemetry.observe_ingest_event(event.byte_size as u64);

        let timings = IngestTimings {
            raw_log_append: start_append.elapsed().as_millis(),
            sqlite_insert: start_sql.elapsed().as_millis(),
            hot_promotion: start_hot.elapsed().as_millis(),
            episode_promotion: start_episode.elapsed().as_millis(),
            semantic_promotion: start_semantic.elapsed().as_millis(),
            summary_promotion: start_summary.elapsed().as_millis(),
            total: start_total.elapsed().as_millis(),
        };
        let manifest = IngestManifest {
            event,
            hot_memory_id,
            episodic_memory_id,
            semantic_memory_id,
            summary_memory_id,
            entities,
            timings_ms: timings,
        };
        self.write_ingest_manifest(&manifest)?;
        debug!(
            op = "persist_event",
            event_id = %manifest.event.id,
            event_kind = %manifest.event.input.kind,
            agent_id = %manifest.event.input.agent_id,
            session_id = %manifest.event.input.session_id,
            namespace = manifest.event.input.namespace.as_deref().unwrap_or(""),
            task_id = manifest.event.input.task_id.as_deref().unwrap_or(""),
            bytes = manifest.event.byte_size,
            token_estimate = manifest.event.token_estimate,
            hot_memory_id = %manifest.hot_memory_id,
            episodic_memory_id = %manifest.episodic_memory_id,
            semantic_memory_id = ?manifest.semantic_memory_id,
            summary_memory_id = ?manifest.summary_memory_id,
            total_ms = manifest.timings_ms.total,
            "persisted event into continuity storage"
        );
        Ok(manifest)
    }

    pub fn list_memory(
        &self,
        layer: Option<MemoryLayer>,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>> {
        let mut rows = Vec::new();
        let sql = if layer.is_some() {
            r#"
            SELECT id, layer, scope, agent_id, session_id, task_id, ts, importance, confidence,
                   token_estimate, source_event_id, scope_key, body, extra_json
            FROM memory_items
            WHERE layer = ?1
            ORDER BY ts DESC
            LIMIT ?2
            "#
        } else {
            r#"
            SELECT id, layer, scope, agent_id, session_id, task_id, ts, importance, confidence,
                   token_estimate, source_event_id, scope_key, body, extra_json
            FROM memory_items
            ORDER BY ts DESC
            LIMIT ?1
            "#
        };
        let mut stmt = self.conn.prepare(sql)?;
        let mut query = if let Some(layer) = layer {
            stmt.query(params![layer.as_i64(), limit as i64])?
        } else {
            stmt.query(params![limit as i64])?
        };
        while let Some(row) = query.next()? {
            rows.push(read_memory_row(row)?);
        }
        Ok(rows)
    }

    pub fn search_lexical(&self, query_text: &str, limit: usize) -> Result<Vec<MemoryRecord>> {
        let fts_query = fts_query(query_text);
        if fts_query.is_empty() {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn.prepare(
            r#"
            SELECT m.id, m.layer, m.scope, m.agent_id, m.session_id, m.task_id, m.ts,
                   m.importance, m.confidence, m.token_estimate, m.source_event_id,
                   m.scope_key, m.body, m.extra_json
            FROM memory_fts
            JOIN memory_items m ON m.id = memory_fts.memory_id
            WHERE memory_fts MATCH ?1
            ORDER BY bm25(memory_fts)
            LIMIT ?2
            "#,
        )?;
        let mut rows = stmt.query(params![fts_query, limit as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(read_memory_row(row)?);
        }
        Ok(out)
    }

    pub fn recent_memories(
        &self,
        session_id: Option<&str>,
        task_id: Option<&str>,
        agent_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>> {
        let sql = r#"
        SELECT id, layer, scope, agent_id, session_id, task_id, ts, importance, confidence,
               token_estimate, source_event_id, scope_key, body, extra_json
        FROM memory_items
        WHERE (?1 IS NULL OR session_id = ?1)
          AND (?2 IS NULL OR task_id = ?2)
          AND (?3 IS NULL OR agent_id = ?3)
        ORDER BY ts DESC
        LIMIT ?4
        "#;
        let mut stmt = self.conn.prepare(sql)?;
        let mut rows = stmt.query(params![session_id, task_id, agent_id, limit as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(read_memory_row(row)?);
        }
        Ok(out)
    }

    pub fn entity_memories(&self, query_text: &str, limit: usize) -> Result<Vec<MemoryRecord>> {
        let entities = query_entities(query_text);
        if entities.is_empty() {
            return Ok(Vec::new());
        }
        let normalized = entities
            .iter()
            .map(|entity| format!("'{}'", entity.replace('\'', "''")))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            r#"
            SELECT m.id, m.layer, m.scope, m.agent_id, m.session_id, m.task_id, m.ts,
                   m.importance, m.confidence, m.token_estimate, m.source_event_id,
                   m.scope_key, m.body, m.extra_json
            FROM entity_mentions em
            JOIN entities e ON e.id = em.entity_id
            JOIN memory_items m ON m.id = em.memory_id
            WHERE e.normalized IN ({normalized})
            ORDER BY em.weight DESC, m.ts DESC
            LIMIT {limit}
            "#
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(read_memory_row(row)?);
        }
        Ok(out)
    }

    pub fn vector_memories(&self) -> Result<Vec<(MemoryRecord, Vec<f32>)>> {
        let backend_key = self.embedding.backend_key();
        let mut stmt = self.conn.prepare(
            r#"
            SELECT m.id, m.layer, m.scope, m.agent_id, m.session_id, m.task_id, m.ts,
                   m.importance, m.confidence, m.token_estimate, m.source_event_id,
                   m.scope_key, m.body, m.extra_json, v.data
            FROM memory_vectors v
            JOIN memory_items m ON m.id = v.memory_id
            WHERE v.backend_key = ?1
            ORDER BY m.ts DESC
            "#,
        )?;
        let mut rows = stmt.query(params![backend_key])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let memory = read_memory_row(row)?;
            let blob: Vec<u8> = row.get(14)?;
            out.push((memory, decode_vector(&blob)));
        }
        Ok(out)
    }

    pub fn embed_query_vector(&self, text: &str) -> Result<Vec<f32>> {
        self.embedding.embed(text)
    }

    pub fn lineage_neighbors(
        &self,
        seed_ids: &[String],
        limit: usize,
    ) -> Result<Vec<MemoryRecord>> {
        if seed_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = (0..seed_ids.len())
            .map(|_| "?".to_string())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            r#"
            SELECT m.id, m.layer, m.scope, m.agent_id, m.session_id, m.task_id, m.ts,
                   m.importance, m.confidence, m.token_estimate, m.source_event_id,
                   m.scope_key, m.body, m.extra_json
            FROM lineage l
            JOIN memory_items m ON m.id = l.child_id
            WHERE l.parent_id IN ({placeholders})
            ORDER BY m.ts DESC
            LIMIT {limit}
            "#
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(rusqlite::params_from_iter(seed_ids.iter()))?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(read_memory_row(row)?);
        }
        Ok(out)
    }

    pub fn provenance_for_memory(&self, memory: &MemoryRecord) -> Result<serde_json::Value> {
        let source_event_id = memory.source_event_id.clone();
        let source_event = if let Some(source_event_id) = &memory.source_event_id {
            self.conn
                .query_row(
                    r#"
                    SELECT ts, kind, source, segment_seq, segment_line
                    FROM events
                    WHERE id = ?1
                    "#,
                    params![source_event_id],
                    |row| {
                        Ok(serde_json::json!({
                            "id": source_event_id,
                            "ts": row.get::<_, String>(0)?,
                            "kind": row.get::<_, String>(1)?,
                            "source": row.get::<_, String>(2)?,
                            "segment_seq": row.get::<_, i64>(3)?,
                            "segment_line": row.get::<_, i64>(4)?,
                        }))
                    },
                )
                .optional()?
        } else {
            None
        };
        let mut stmt = self.conn.prepare(
            "SELECT parent_id, edge_kind, weight FROM lineage WHERE child_id = ?1 ORDER BY weight DESC, parent_id ASC",
        )?;
        let mut rows = stmt.query(params![memory.id.clone()])?;
        let mut parents = Vec::new();
        while let Some(row) = rows.next()? {
            parents.push(serde_json::json!({
                "parent_id": row.get::<_, String>(0)?,
                "edge_kind": row.get::<_, String>(1)?,
                "weight": row.get::<_, f64>(2)?,
            }));
        }
        let dimensions = self.dimensions_for_item("memory", &memory.id)?;
        let relations = self.relations_for_item(&memory.id)?;
        Ok(serde_json::json!({
            "memory_id": memory.id,
            "layer": memory.layer,
            "source_event_id": source_event_id,
            "source_event": source_event,
            "parents": parents,
            "dimensions": dimensions,
            "relations": relations,
            "extra": memory.extra,
        }))
    }

    pub fn persist_context_pack(
        &self,
        pack: &ContextPack,
        manifest: &ContextPackManifest,
        rejected: &[RejectedCandidate],
    ) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO context_packs(
              id, ts, agent_id, session_id, task_id, query_text, budget_tokens,
              total_candidate_count, total_selected_count, manifest_path
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            "#,
            params![
                pack.id,
                pack.created_at.to_rfc3339(),
                pack.query.agent_id,
                pack.query.session_id,
                pack.query.task_id,
                pack.query.query_text,
                pack.query.budget_tokens as i64,
                (pack.items.len() + rejected.len()) as i64,
                pack.items.len() as i64,
                pack.manifest_path,
            ],
        )?;
        for (rank, item) in pack.items.iter().enumerate() {
            self.conn.execute(
                r#"
                INSERT OR REPLACE INTO context_pack_items(
                  pack_id, memory_id, included, final_score, rank, token_estimate, reason_json
                ) VALUES (?1, ?2, 1, ?3, ?4, ?5, ?6)
                "#,
                params![
                    pack.id,
                    item.memory_id,
                    item.final_score,
                    rank as i64,
                    item.token_estimate as i64,
                    serde_json::to_string(&item.why)?,
                ],
            )?;
        }
        for (rank, item) in rejected.iter().enumerate() {
            self.conn.execute(
                r#"
                INSERT OR REPLACE INTO context_pack_items(
                  pack_id, memory_id, included, final_score, rank, token_estimate, reason_json
                ) VALUES (?1, ?2, 0, ?3, ?4, ?5, ?6)
                "#,
                params![
                    pack.id,
                    item.memory_id,
                    item.final_score,
                    rank as i64,
                    item.token_estimate as i64,
                    serde_json::to_string(&item.reason)?,
                ],
            )?;
        }
        let path = PathBuf::from(&pack.manifest_path);
        fs::write(path, serde_json::to_vec_pretty(manifest)?)?;
        debug!(
            op = "persist_context_pack",
            pack_id = %pack.id,
            agent_id = pack.query.agent_id.as_deref().unwrap_or(""),
            session_id = pack.query.session_id.as_deref().unwrap_or(""),
            task_id = pack.query.task_id.as_deref().unwrap_or(""),
            namespace = pack.query.namespace.as_deref().unwrap_or(""),
            budget_tokens = pack.query.budget_tokens,
            selected_count = pack.items.len(),
            rejected_count = rejected.len(),
            manifest_path = %pack.manifest_path,
            "persisted bounded context pack"
        );
        Ok(())
    }

    fn selected_memory_ids_for_pack(&self, pack_id: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT memory_id
            FROM context_pack_items
            WHERE pack_id = ?1 AND included = 1
            ORDER BY rank ASC, final_score DESC, memory_id ASC
            "#,
        )?;
        let mut rows = stmt.query(params![pack_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row.get(0)?);
        }
        Ok(out)
    }

    fn continuity_ids_for_memory_ids(
        &self,
        memory_ids: &[String],
    ) -> Result<HashMap<String, String>> {
        let mut out = HashMap::new();
        if memory_ids.is_empty() {
            return Ok(out);
        }
        for chunk in memory_ids.chunks(200) {
            let placeholders = (0..chunk.len())
                .map(|index| format!("?{}", index + 1))
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!(
                r#"
                SELECT memory_id, id
                FROM continuity_items
                WHERE memory_id IN ({placeholders})
                "#
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let mut rows = stmt.query(rusqlite::params_from_iter(
                chunk.iter().map(|id| id.as_str()),
            ))?;
            while let Some(row) = rows.next()? {
                out.insert(row.get::<_, String>(0)?, row.get::<_, String>(1)?);
            }
        }
        Ok(out)
    }

    fn plasticity_for_continuity_many(
        &self,
        continuity_ids: &[String],
    ) -> Result<HashMap<String, StoredContinuityPlasticity>> {
        let mut out = HashMap::new();
        if continuity_ids.is_empty() {
            return Ok(out);
        }
        for chunk in continuity_ids.chunks(200) {
            let placeholders = (0..chunk.len())
                .map(|index| format!("?{}", index + 1))
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!(
                r#"
                SELECT continuity_id, belief_key, source_role, activation_count, successful_use_count,
                       confirmation_count, contradiction_count, independent_source_count,
                       spaced_reactivation_count, spacing_interval_hours, stability_score,
                       prediction_error, last_reactivated_at, last_confirmed_at,
                       last_contradicted_at, last_strengthened_at
                FROM continuity_plasticity
                WHERE continuity_id IN ({placeholders})
                "#
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let mut rows = stmt.query(rusqlite::params_from_iter(
                chunk.iter().map(|id| id.as_str()),
            ))?;
            while let Some(row) = rows.next()? {
                let last_reactivated_at = row
                    .get::<_, Option<String>>(12)?
                    .map(|value| DateTime::parse_from_rfc3339(&value))
                    .transpose()?
                    .map(|ts| ts.with_timezone(&Utc));
                let last_confirmed_at = row
                    .get::<_, Option<String>>(13)?
                    .map(|value| DateTime::parse_from_rfc3339(&value))
                    .transpose()?
                    .map(|ts| ts.with_timezone(&Utc));
                let last_contradicted_at = row
                    .get::<_, Option<String>>(14)?
                    .map(|value| DateTime::parse_from_rfc3339(&value))
                    .transpose()?
                    .map(|ts| ts.with_timezone(&Utc));
                let last_strengthened_at = row
                    .get::<_, Option<String>>(15)?
                    .map(|value| DateTime::parse_from_rfc3339(&value))
                    .transpose()?
                    .map(|ts| ts.with_timezone(&Utc));
                out.insert(
                    row.get::<_, String>(0)?,
                    StoredContinuityPlasticity {
                        belief_key: row.get(1)?,
                        source_role: row.get(2)?,
                        state: ContinuityPlasticityState {
                            activation_count: row.get::<_, i64>(3)? as usize,
                            successful_use_count: row.get::<_, i64>(4)? as usize,
                            confirmation_count: row.get::<_, i64>(5)? as usize,
                            contradiction_count: row.get::<_, i64>(6)? as usize,
                            independent_source_count: row.get::<_, i64>(7)? as usize,
                            spaced_reactivation_count: row.get::<_, i64>(8)? as usize,
                            spacing_interval_hours: row.get(9)?,
                            stability_score: row.get(10)?,
                            prediction_error: row.get(11)?,
                            last_reactivated_at,
                            last_confirmed_at,
                            last_contradicted_at,
                            last_strengthened_at,
                        },
                    },
                );
            }
        }
        Ok(out)
    }

    fn sync_belief_cluster_independent_sources(
        &self,
        belief_key: &str,
        updated_at: DateTime<Utc>,
    ) -> Result<()> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT source_role
            FROM continuity_plasticity
            WHERE belief_key = ?1
            "#,
        )?;
        let mut rows = stmt.query(params![belief_key])?;
        let mut explicit_roles = BTreeSet::new();
        let mut row_count = 0usize;
        while let Some(row) = rows.next()? {
            row_count += 1;
            if let Some(role) = row.get::<_, Option<String>>(0)? {
                let trimmed = role.trim();
                if !trimmed.is_empty() {
                    explicit_roles.insert(trimmed.to_string());
                }
            }
        }
        let independent_source_count = if explicit_roles.is_empty() {
            row_count.max(1)
        } else {
            explicit_roles.len()
        };
        self.conn.execute(
            r#"
            UPDATE continuity_plasticity
            SET independent_source_count = ?2,
                updated_at = ?3
            WHERE belief_key = ?1
            "#,
            params![
                belief_key,
                independent_source_count as i64,
                updated_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    fn ensure_continuity_plasticity(
        &self,
        continuity_id: &str,
        extra: &serde_json::Value,
        updated_at: DateTime<Utc>,
    ) -> Result<StoredContinuityPlasticity> {
        let belief_key = continuity_belief_key(extra);
        let source_role = continuity_source_role(extra);
        self.conn.execute(
            r#"
            INSERT OR IGNORE INTO continuity_plasticity(
              continuity_id, belief_key, source_role, activation_count, successful_use_count,
              confirmation_count, contradiction_count, independent_source_count,
              spaced_reactivation_count, spacing_interval_hours, stability_score,
              prediction_error, created_at, updated_at
            ) VALUES (?1, ?2, ?3, 0, 0, 0, 0, 1, 0, 6.0, 0.0, 0.0, ?4, ?4)
            "#,
            params![
                continuity_id,
                belief_key,
                source_role,
                updated_at.to_rfc3339(),
            ],
        )?;
        self.conn.execute(
            r#"
            UPDATE continuity_plasticity
            SET belief_key = COALESCE(?2, belief_key),
                source_role = COALESCE(?3, source_role),
                updated_at = ?4
            WHERE continuity_id = ?1
            "#,
            params![
                continuity_id,
                belief_key,
                source_role,
                updated_at.to_rfc3339(),
            ],
        )?;
        if let Some(belief_key) = continuity_belief_key(extra) {
            self.sync_belief_cluster_independent_sources(&belief_key, updated_at)?;
        }
        Ok(self
            .plasticity_for_continuity_many(&[continuity_id.to_string()])?
            .remove(continuity_id)
            .unwrap_or_default())
    }

    fn bump_continuity_plasticity(
        &self,
        continuity_id: &str,
        updated_at: DateTime<Utc>,
        activated: bool,
        successful: bool,
        confirmed: bool,
        contradicted: bool,
    ) -> Result<()> {
        let current = self
            .plasticity_for_continuity_many(&[continuity_id.to_string()])?
            .remove(continuity_id)
            .unwrap_or_default();
        let mut state = current.state;
        state.spacing_interval_hours =
            normalize_spacing_interval_hours(state.spacing_interval_hours);
        let last_strength_signal_at = state
            .last_strengthened_at
            .or(state.last_confirmed_at)
            .or(state.last_reactivated_at);
        let spacing_ready = last_strength_signal_at
            .map(|ts| spacing_elapsed_hours(ts, updated_at) >= state.spacing_interval_hours)
            .unwrap_or(true);
        if activated {
            state.activation_count += 1;
            state.last_reactivated_at = Some(updated_at);
        }
        if successful {
            state.successful_use_count += 1;
        }
        if confirmed {
            state.confirmation_count += 1;
            state.last_confirmed_at = Some(updated_at);
        }
        if contradicted {
            state.contradiction_count += 1;
            state.last_contradicted_at = Some(updated_at);
            state.spacing_interval_hours =
                (state.spacing_interval_hours * 0.5_f64).clamp(2.0_f64, 24.0_f64 * 21.0_f64);
        }
        if (activated || successful || confirmed) && spacing_ready {
            state.spaced_reactivation_count += 1;
            state.last_strengthened_at = Some(updated_at);
            let growth = if confirmed || successful {
                1.8_f64
            } else {
                1.25_f64
            };
            state.spacing_interval_hours =
                normalize_spacing_interval_hours(state.spacing_interval_hours * growth);
        }
        state.prediction_error = if state.confirmation_count + state.contradiction_count == 0 {
            0.0
        } else {
            state.contradiction_count as f64
                / (state.confirmation_count + state.contradiction_count) as f64
        };
        state.stability_score = 0.35 * (state.activation_count as f64 + 1.0).ln()
            + 1.1 * (state.successful_use_count as f64 + 1.0).ln()
            + 1.3 * (state.confirmation_count as f64 + 1.0).ln()
            + 1.6 * (state.spaced_reactivation_count as f64 + 1.0).ln()
            + 0.2 * spacing_interval_signal(&state)
            + 0.6 * (state.independent_source_count as f64 + 1.0).ln()
            - 1.8 * state.contradiction_count as f64;
        self.conn.execute(
            r#"
            INSERT INTO continuity_plasticity(
              continuity_id, belief_key, source_role, activation_count, successful_use_count,
              confirmation_count, contradiction_count, independent_source_count,
              spaced_reactivation_count, spacing_interval_hours, stability_score,
              prediction_error, last_reactivated_at, last_confirmed_at, last_contradicted_at,
              last_strengthened_at, created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?17)
            ON CONFLICT(continuity_id) DO UPDATE SET
              activation_count = excluded.activation_count,
              successful_use_count = excluded.successful_use_count,
              confirmation_count = excluded.confirmation_count,
              contradiction_count = excluded.contradiction_count,
              independent_source_count = excluded.independent_source_count,
              spaced_reactivation_count = excluded.spaced_reactivation_count,
              spacing_interval_hours = excluded.spacing_interval_hours,
              stability_score = excluded.stability_score,
              prediction_error = excluded.prediction_error,
              last_reactivated_at = excluded.last_reactivated_at,
              last_confirmed_at = excluded.last_confirmed_at,
              last_contradicted_at = excluded.last_contradicted_at,
              last_strengthened_at = excluded.last_strengthened_at,
              updated_at = excluded.updated_at
            "#,
            params![
                continuity_id,
                current.belief_key,
                current.source_role,
                state.activation_count as i64,
                state.successful_use_count as i64,
                state.confirmation_count as i64,
                state.contradiction_count as i64,
                state.independent_source_count as i64,
                state.spaced_reactivation_count as i64,
                state.spacing_interval_hours,
                state.stability_score,
                state.prediction_error,
                state
                    .last_reactivated_at
                    .map(|ts: DateTime<Utc>| ts.to_rfc3339()),
                state
                    .last_confirmed_at
                    .map(|ts: DateTime<Utc>| ts.to_rfc3339()),
                state
                    .last_contradicted_at
                    .map(|ts: DateTime<Utc>| ts.to_rfc3339()),
                state
                    .last_strengthened_at
                    .map(|ts: DateTime<Utc>| ts.to_rfc3339()),
                updated_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn record_outcome(&self, input: OutcomeInput) -> Result<ContinuityItemRecord> {
        let outcome_ts = Utc::now();
        let mut activated_memory_ids = BTreeSet::<String>::new();
        if let Some(pack_id) = &input.pack_id {
            activated_memory_ids.extend(self.selected_memory_ids_for_pack(pack_id)?);
        }
        activated_memory_ids.extend(input.used_memory_ids.iter().cloned());
        activated_memory_ids.extend(input.confirmed_memory_ids.iter().cloned());
        activated_memory_ids.extend(input.contradicted_memory_ids.iter().cloned());

        let continuity_by_memory = self.continuity_ids_for_memory_ids(
            &activated_memory_ids.iter().cloned().collect::<Vec<_>>(),
        )?;
        let used_set = input
            .used_memory_ids
            .iter()
            .filter_map(|memory_id| continuity_by_memory.get(memory_id).cloned())
            .collect::<BTreeSet<_>>();
        let confirmed_set = input
            .confirmed_memory_ids
            .iter()
            .filter_map(|memory_id| continuity_by_memory.get(memory_id).cloned())
            .collect::<BTreeSet<_>>();
        let contradicted_set = input
            .contradicted_memory_ids
            .iter()
            .filter_map(|memory_id| continuity_by_memory.get(memory_id).cloned())
            .collect::<BTreeSet<_>>();
        let activated_set = activated_memory_ids
            .iter()
            .filter_map(|memory_id| continuity_by_memory.get(memory_id).cloned())
            .collect::<BTreeSet<_>>();

        for continuity_id in &activated_set {
            self.bump_continuity_plasticity(
                continuity_id,
                outcome_ts,
                true,
                input.quality >= 0.65 && used_set.contains(continuity_id),
                confirmed_set.contains(continuity_id),
                contradicted_set.contains(continuity_id),
            )?;
        }

        let mut support_refs = Vec::<SupportRef>::new();
        for memory_id in &input.used_memory_ids {
            if let Some(continuity_id) = continuity_by_memory.get(memory_id) {
                support_refs.push(SupportRef {
                    support_type: "continuity".to_string(),
                    support_id: continuity_id.clone(),
                    reason: Some("outcome_used".to_string()),
                    weight: 1.0,
                });
            } else {
                support_refs.push(SupportRef {
                    support_type: "memory".to_string(),
                    support_id: memory_id.clone(),
                    reason: Some("outcome_used".to_string()),
                    weight: 1.0,
                });
            }
        }
        for memory_id in &input.confirmed_memory_ids {
            if let Some(continuity_id) = continuity_by_memory.get(memory_id) {
                support_refs.push(SupportRef {
                    support_type: "continuity".to_string(),
                    support_id: continuity_id.clone(),
                    reason: Some("outcome_confirmed".to_string()),
                    weight: 1.2,
                });
            }
        }
        for memory_id in &input.contradicted_memory_ids {
            if let Some(continuity_id) = continuity_by_memory.get(memory_id) {
                support_refs.push(SupportRef {
                    support_type: "continuity".to_string(),
                    support_id: continuity_id.clone(),
                    reason: Some("outcome_contradicted".to_string()),
                    weight: 1.2,
                });
            }
        }
        support_refs.sort_by(|a, b| {
            a.support_type
                .cmp(&b.support_type)
                .then_with(|| a.support_id.cmp(&b.support_id))
                .then_with(|| a.reason.cmp(&b.reason))
        });
        support_refs.dedup_by(|left, right| {
            left.support_type == right.support_type
                && left.support_id == right.support_id
                && left.reason == right.reason
        });

        let mut dimensions = input.dimensions;
        dimensions.push(DimensionValue {
            key: "outcome_quality".to_string(),
            value: format!("{:.2}", input.quality),
            weight: 100,
        });

        self.persist_continuity_item(ContinuityItemInput {
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
            supports: support_refs,
            dimensions,
            extra: serde_json::json!({
                "pack_id": input.pack_id,
                "used_memory_ids": input.used_memory_ids,
                "confirmed_memory_ids": input.confirmed_memory_ids,
                "contradicted_memory_ids": input.contradicted_memory_ids,
                "failures": input.failures,
                "extra": input.extra,
            }),
        })
    }

    pub fn explain_context_pack(&self, id: &str) -> Result<ContextPackManifest> {
        let manifest_path = self
            .conn
            .query_row(
                "SELECT manifest_path FROM context_packs WHERE id = ?1",
                params![id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .ok_or_else(|| anyhow!("unknown context pack {id}"))?;
        Ok(serde_json::from_slice(&fs::read(manifest_path)?)?)
    }

    pub fn dimensions_for_item(
        &self,
        item_type: &str,
        item_id: &str,
    ) -> Result<Vec<DimensionValue>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT key, value, weight
            FROM item_dimensions
            WHERE item_type = ?1 AND item_id = ?2
            ORDER BY key ASC, value ASC
            "#,
        )?;
        let mut rows = stmt.query(params![item_type, item_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(DimensionValue {
                key: row.get(0)?,
                value: row.get(1)?,
                weight: row.get(2)?,
            });
        }
        Ok(out)
    }

    pub fn event_by_id(&self, id: &str) -> Result<Option<EventRecord>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, ts, kind, scope, agent_id, agent_role, session_id, task_id, project_id,
                   goal_id, run_id, namespace, environment, source, tags_json, dimensions_json,
                   attributes_json, content, content_hash, byte_size, token_estimate, importance,
                   segment_seq, segment_line
            FROM events
            WHERE id = ?1
            "#,
        )?;
        stmt.query_row(params![id], |row| {
            read_event_row(row).map_err(to_sqlite_anyhow)
        })
        .optional()
        .map_err(Into::into)
    }

    pub fn relations_for_item(&self, item_id: &str) -> Result<Vec<RelationRecord>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, ts, source_id, target_id, relation, weight, attributes_json
            FROM fabric_relations
            WHERE source_id = ?1 OR target_id = ?1
            ORDER BY ts DESC, relation ASC
            "#,
        )?;
        let mut rows = stmt.query(params![item_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(RelationRecord {
                id: row.get(0)?,
                ts: DateTime::parse_from_rfc3339(&row.get::<_, String>(1)?)?.with_timezone(&Utc),
                source_id: row.get(2)?,
                target_id: row.get(3)?,
                relation: row.get(4)?,
                weight: row.get(5)?,
                attributes: serde_json::from_str(&row.get::<_, String>(6)?)?,
            });
        }
        Ok(out)
    }

    pub fn annotate_item(
        &self,
        item_type: &str,
        item_id: &str,
        dimensions: &[DimensionValue],
    ) -> Result<Vec<DimensionValue>> {
        let (ts, layer) = self.item_metadata(item_type, item_id)?;
        self.index_item_dimensions(item_type, item_id, layer, ts, dimensions)?;
        self.dimensions_for_item(item_type, item_id)
    }

    pub fn relate_items(
        &self,
        source_id: &str,
        target_id: &str,
        relation: &str,
        weight: f64,
        attributes: serde_json::Value,
    ) -> Result<RelationRecord> {
        let record = RelationRecord {
            id: format!("rel:{}", Uuid::now_v7()),
            ts: Utc::now(),
            source_id: source_id.to_string(),
            target_id: target_id.to_string(),
            relation: relation.to_string(),
            weight,
            attributes,
        };
        self.insert_relation(&record)?;
        Ok(record)
    }

    pub fn view_memories(&self, view_id: &str) -> Result<Vec<MemoryRecord>> {
        let ids = self.view_memory_ids(view_id)?;
        self.memories_by_ids(&ids)
    }

    pub fn selector_memories(
        &self,
        selector: &Selector,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>> {
        Ok(self
            .selector_candidates(selector, limit)?
            .into_iter()
            .map(|candidate| candidate.memory)
            .collect())
    }

    pub fn materialize_view(&self, input: ViewInput) -> Result<ViewRecord> {
        let total_start = Instant::now();
        let created_at = Utc::now();
        let id = format!("view:{}", Uuid::now_v7());
        let limit = input.limit.unwrap_or(48).max(1);

        let mut sources = Vec::new();
        for selector in &input.selectors {
            sources.push(self.selector_candidates(selector, limit.saturating_mul(4))?);
        }
        for source_view_id in &input.source_view_ids {
            sources.push(self.candidates_from_view(source_view_id)?);
        }
        if sources.is_empty() {
            let fallback = Selector {
                namespace: input.namespace.clone(),
                limit: Some(limit.saturating_mul(4)),
                ..Selector::default()
            };
            sources.push(self.selector_candidates(&fallback, limit.saturating_mul(4))?);
        }

        let source_count = sources.len().max(1);
        let mut merged: HashMap<String, ViewAccumulator> = HashMap::new();
        for (source_index, source) in sources.into_iter().enumerate() {
            for (rank, candidate) in source.into_iter().enumerate() {
                let entry = merged
                    .entry(candidate.memory.id.clone())
                    .or_insert_with(|| ViewAccumulator::new(candidate.memory.clone()));
                entry.score += candidate.score.max(0.0) + 1.0 - (rank as f64 / limit.max(1) as f64);
                entry.sources.insert(source_index);
                entry
                    .matched_dimensions
                    .extend(candidate.matched_dimensions.into_iter());
                entry.why.extend(candidate.why.into_iter());
            }
        }

        let mut candidates = merged
            .into_values()
            .filter(|candidate| {
                !matches!(input.op, ViewOp::Intersect) || candidate.sources.len() == source_count
            })
            .map(|candidate| {
                let recency = recency_score(Utc::now(), candidate.memory.ts);
                let score = candidate.score
                    + candidate.memory.importance * 0.35
                    + candidate.memory.confidence * 0.25
                    + recency * 0.20;
                ScoredMemory {
                    memory: candidate.memory,
                    score,
                    matched_dimensions: candidate.matched_dimensions.into_iter().collect(),
                    why: candidate.why.into_iter().collect(),
                }
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if matches!(input.op, ViewOp::Snapshot) {
            let resolution = input.resolution.unwrap_or(SnapshotResolution::Medium);
            candidates = snapshot_candidates(candidates, resolution);
        }
        candidates.truncate(limit);

        let conflicts = self.detect_view_conflicts(&candidates)?;
        for conflict in &conflicts {
            for pair in conflict.memory_ids.windows(2) {
                if let [left, right] = pair {
                    self.insert_relation(&RelationRecord {
                        id: format!("rel:{}", Uuid::now_v7()),
                        ts: created_at,
                        source_id: left.clone(),
                        target_id: right.clone(),
                        relation: "contradicts".to_string(),
                        weight: 0.8,
                        attributes: serde_json::json!({
                            "key": conflict.key,
                            "values": conflict.values,
                        }),
                    })?;
                }
            }
        }

        let selected = candidates
            .iter()
            .map(|candidate| -> Result<ViewItem> {
                Ok(ViewItem {
                    memory_id: candidate.memory.id.clone(),
                    layer: candidate.memory.layer,
                    token_estimate: candidate.memory.token_estimate,
                    score: candidate.score,
                    matched_dimensions: candidate.matched_dimensions.clone(),
                    why: candidate.why.clone(),
                    provenance: self.provenance_for_memory(&candidate.memory)?,
                    body: candidate.memory.body.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let manifest = ViewManifest {
            id: id.clone(),
            created_at,
            input: input.clone(),
            item_count: selected.len(),
            conflict_count: conflicts.len(),
            selected: selected.clone(),
            conflicts: conflicts.clone(),
            timings_ms: serde_json::json!({
                "total": total_start.elapsed().as_millis(),
            }),
        };
        let manifest_path = self.persist_view(&manifest)?;
        Ok(ViewRecord {
            id,
            created_at,
            input,
            item_count: selected.len(),
            conflict_count: conflicts.len(),
            manifest_path,
        })
    }

    pub fn get_view(&self, id: &str) -> Result<ViewRecord> {
        self.conn
            .query_row(
                r#"
                SELECT ts, op, owner_agent_id, namespace, objective, selector_json,
                       source_view_ids_json, resolution, item_count, conflict_count, manifest_path
                FROM views
                WHERE id = ?1
                "#,
                params![id],
                |row| {
                    let op = parse_view_op(&row.get::<_, String>(1)?).map_err(to_sqlite_anyhow)?;
                    let selectors =
                        serde_json::from_str(&row.get::<_, String>(5)?).map_err(to_sqlite_error)?;
                    let source_view_ids =
                        serde_json::from_str(&row.get::<_, String>(6)?).map_err(to_sqlite_error)?;
                    let resolution = match row.get::<_, Option<String>>(7)? {
                        Some(value) => {
                            Some(parse_snapshot_resolution(&value).map_err(to_sqlite_anyhow)?)
                        }
                        None => None,
                    };
                    Ok(ViewRecord {
                        id: id.to_string(),
                        created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(0)?)
                            .map_err(to_sqlite_error)?
                            .with_timezone(&Utc),
                        input: ViewInput {
                            op,
                            owner_agent_id: row.get(2)?,
                            namespace: row.get(3)?,
                            objective: row.get(4)?,
                            selectors,
                            source_view_ids,
                            resolution,
                            limit: Some(row.get::<_, i64>(8)? as usize),
                        },
                        item_count: row.get::<_, i64>(8)? as usize,
                        conflict_count: row.get::<_, i64>(9)? as usize,
                        manifest_path: row.get(10)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| anyhow!("unknown view {id}"))
    }

    pub fn explain_view(&self, id: &str) -> Result<ViewManifest> {
        let view = self.get_view(id)?;
        Ok(serde_json::from_slice(&fs::read(view.manifest_path)?)?)
    }

    pub fn fork_view(&self, view_id: &str, owner_agent_id: Option<String>) -> Result<ViewRecord> {
        let base = self.get_view(view_id)?;
        let mut input = base.input.clone();
        input.op = ViewOp::Fork;
        input.owner_agent_id = owner_agent_id.or(input.owner_agent_id);
        input.source_view_ids = vec![view_id.to_string()];
        self.materialize_view(input)
    }

    pub fn persist_handoff(
        &self,
        record: &HandoffRecord,
        manifest: &serde_json::Value,
        query_text: &str,
    ) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO handoffs(
              id, ts, from_agent_id, to_agent_id, reason, query_text, view_id, pack_id, conflict_count, manifest_path
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            "#,
            params![
                record.id,
                record.created_at.to_rfc3339(),
                record.from_agent_id,
                record.to_agent_id,
                record.reason,
                query_text,
                record.view_id,
                record.pack_id,
                record.conflict_count as i64,
                record.manifest_path,
            ],
        )?;
        fs::write(&record.manifest_path, serde_json::to_vec_pretty(manifest)?)?;
        self.insert_relation(&RelationRecord {
            id: format!("rel:{}", Uuid::now_v7()),
            ts: record.created_at,
            source_id: record.view_id.clone(),
            target_id: record.pack_id.clone(),
            relation: "handoff_pack".to_string(),
            weight: 1.0,
            attributes: serde_json::json!({"from_agent": record.from_agent_id, "to_agent": record.to_agent_id}),
        })?;
        debug!(
            op = "persist_handoff",
            handoff_id = %record.id,
            from_agent_id = %record.from_agent_id,
            to_agent_id = %record.to_agent_id,
            view_id = %record.view_id,
            pack_id = %record.pack_id,
            conflict_count = record.conflict_count,
            manifest_path = %record.manifest_path,
            "persisted handoff manifest"
        );
        Ok(())
    }

    pub fn get_handoff(&self, id: &str) -> Result<HandoffRecord> {
        self.conn
            .query_row(
                r#"
                SELECT ts, from_agent_id, to_agent_id, reason, view_id, pack_id, conflict_count, manifest_path
                FROM handoffs
                WHERE id = ?1
                "#,
                params![id],
                |row| {
                    Ok(HandoffRecord {
                        id: id.to_string(),
                        created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(0)?)
                            .map_err(to_sqlite_error)?
                            .with_timezone(&Utc),
                        from_agent_id: row.get(1)?,
                        to_agent_id: row.get(2)?,
                        reason: row.get(3)?,
                        view_id: row.get(4)?,
                        pack_id: row.get(5)?,
                        conflict_count: row.get::<_, i64>(6)? as usize,
                        manifest_path: row.get(7)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| anyhow!("unknown handoff {id}"))
    }

    pub fn explain_handoff(&self, id: &str) -> Result<serde_json::Value> {
        let record = self.get_handoff(id)?;
        Ok(serde_json::from_slice(&fs::read(record.manifest_path)?)?)
    }

    pub fn attach_agent(&self, input: AttachAgentInput) -> Result<AgentAttachmentRecord> {
        let namespace = self
            .resolve_namespace_alias(Some(input.namespace.as_str()))?
            .expect("attachment namespace should remain present");
        let id = format!("attach:{}", Uuid::now_v7());
        let attached_at = Utc::now();
        self.conn.execute(
            "UPDATE agent_attachments SET active = 0 WHERE agent_id = ?1 AND namespace = ?2",
            params![input.agent_id, namespace],
        )?;
        self.conn.execute(
            r#"
            INSERT INTO agent_attachments(
              id, ts, agent_id, agent_type, namespace, role, capabilities_json, metadata_json, active, last_seen_at, tick_count, context_id
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1, ?9, 1, NULL)
            "#,
            params![
                id,
                attached_at.to_rfc3339(),
                input.agent_id,
                input.agent_type,
                namespace,
                input.role,
                serde_json::to_string(&input.capabilities)?,
                serde_json::to_string(&input.metadata)?,
                attached_at.to_rfc3339(),
            ],
        )?;
        let record = AgentAttachmentRecord {
            id,
            attached_at,
            last_seen_at: attached_at,
            input,
            tick_count: 1,
            active: true,
            context_id: None,
        };
        debug!(
            op = "attach_agent",
            attachment_id = %record.id,
            agent_id = %record.input.agent_id,
            agent_type = %record.input.agent_type,
            namespace = %record.input.namespace,
            role = ?record.input.role,
            capability_count = record.input.capabilities.len(),
            "attached agent to continuity organism"
        );
        Ok(record)
    }

    pub fn upsert_agent_badge(&self, input: UpsertAgentBadgeInput) -> Result<AgentBadgeRecord> {
        let attachment = match input.attachment_id.as_deref() {
            Some(attachment_id) => self.get_attachment(attachment_id)?,
            None => {
                let agent_id = input
                    .agent_id
                    .as_deref()
                    .ok_or_else(|| anyhow!("agent badge requires attachment_id or agent_id"))?;
                let namespace = input.namespace.as_deref().ok_or_else(|| {
                    anyhow!("agent badge requires namespace when attachment_id is omitted")
                })?;
                self.get_active_attachment(agent_id, namespace)?
            }
        };
        let derived = self.derive_agent_badge(&attachment, input.context_id.as_deref())?;
        let existing = self.get_stored_agent_badge(&attachment.id)?;
        let now = Utc::now();
        let display_name = input
            .display_name
            .or_else(|| existing.as_ref().map(|badge| badge.display_name.clone()))
            .unwrap_or_else(|| derived.display_name.clone());
        let status = input
            .status
            .or_else(|| existing.as_ref().map(|badge| badge.status.clone()))
            .unwrap_or_else(|| derived.status.clone());
        let focus = input
            .focus
            .or_else(|| existing.as_ref().map(|badge| badge.focus.clone()))
            .unwrap_or_else(|| derived.focus.clone());
        let headline = input
            .headline
            .or_else(|| existing.as_ref().map(|badge| badge.headline.clone()))
            .unwrap_or_else(|| derived.headline.clone());
        let resource = input
            .resource
            .or_else(|| existing.as_ref().and_then(|badge| badge.resource.clone()))
            .or_else(|| derived.resource.clone());
        let repo_root = input
            .repo_root
            .or_else(|| existing.as_ref().and_then(|badge| badge.repo_root.clone()))
            .or_else(|| derived.repo_root.clone());
        let branch = input
            .branch
            .or_else(|| existing.as_ref().and_then(|badge| badge.branch.clone()))
            .or_else(|| derived.branch.clone());
        let metadata = if !input.metadata.is_null() {
            input.metadata
        } else if let Some(existing) = existing.as_ref() {
            existing.metadata.clone()
        } else {
            derived.metadata.clone()
        };
        let context_id = input
            .context_id
            .or_else(|| existing.as_ref().and_then(|badge| badge.context_id.clone()))
            .or_else(|| derived.context_id.clone());
        let task_id = existing
            .as_ref()
            .and_then(|badge| badge.task_id.clone())
            .or_else(|| derived.task_id.clone());
        self.conn.execute(
            r#"
            INSERT INTO agent_badges(
              attachment_id, updated_at, agent_id, display_name, status, focus, headline,
              resource, repo_root, branch, metadata_json, context_id, task_id
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
            ON CONFLICT(attachment_id) DO UPDATE SET
              updated_at = excluded.updated_at,
              agent_id = excluded.agent_id,
              display_name = excluded.display_name,
              status = excluded.status,
              focus = excluded.focus,
              headline = excluded.headline,
              resource = excluded.resource,
              repo_root = excluded.repo_root,
              branch = excluded.branch,
              metadata_json = excluded.metadata_json,
              context_id = excluded.context_id,
              task_id = excluded.task_id
            "#,
            params![
                attachment.id,
                now.to_rfc3339(),
                attachment.input.agent_id,
                display_name,
                status,
                focus,
                headline,
                resource,
                repo_root,
                branch,
                serde_json::to_string(&metadata)?,
                context_id,
                task_id,
            ],
        )?;
        let badge = self.get_agent_badge(&attachment.id)?;
        debug!(
            op = "upsert_agent_badge",
            attachment_id = %badge.attachment_id,
            agent_id = %badge.agent_id,
            namespace = %badge.namespace,
            task_id = badge.task_id.as_deref().unwrap_or(""),
            repo_root = badge.repo_root.as_deref().unwrap_or(""),
            branch = badge.branch.as_deref().unwrap_or(""),
            resource = badge.resource.as_deref().unwrap_or(""),
            status = %badge.status,
            "persisted agent badge"
        );
        Ok(badge)
    }

    pub fn get_agent_badge(&self, attachment_id: &str) -> Result<AgentBadgeRecord> {
        self.list_agent_badges(None, None)?
            .into_iter()
            .find(|badge| badge.attachment_id == attachment_id)
            .ok_or_else(|| anyhow!("unknown agent badge for attachment {attachment_id}"))
    }

    pub fn heartbeat_attachment(&self, input: HeartbeatInput) -> Result<AgentAttachmentRecord> {
        if let Some(attachment_id) = input.attachment_id.as_deref() {
            self.touch_attachment(attachment_id, input.context_id.as_deref())?;
            return self.get_attachment(attachment_id);
        }

        let agent_id = input
            .agent_id
            .as_deref()
            .ok_or_else(|| anyhow!("heartbeat requires attachment_id or agent_id"))?;
        let namespace = input
            .namespace
            .as_deref()
            .ok_or_else(|| anyhow!("heartbeat requires namespace when attachment_id is omitted"))?;
        let namespace = self
            .resolve_namespace_alias(Some(namespace))?
            .expect("heartbeat namespace should remain present");
        self.touch_active_attachment(
            agent_id,
            Some(namespace.as_str()),
            input.context_id.as_deref(),
        )?;
        self.get_active_attachment(agent_id, &namespace)
    }

    pub fn touch_active_attachment(
        &self,
        agent_id: &str,
        namespace: Option<&str>,
        context_id: Option<&str>,
    ) -> Result<()> {
        let Some(namespace) = self.resolve_namespace_alias(namespace)? else {
            return Ok(());
        };
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            r#"
            UPDATE agent_attachments
            SET last_seen_at = ?3,
                tick_count = tick_count + 1,
                context_id = COALESCE(?4, context_id)
            WHERE agent_id = ?1 AND namespace = ?2 AND active = 1
            "#,
            params![agent_id, namespace, now, context_id],
        )?;
        self.refresh_claim_leases(None, Some(agent_id), Some(namespace.as_str()), context_id)?;
        Ok(())
    }

    pub fn touch_attachment(&self, attachment_id: &str, context_id: Option<&str>) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let attachment = self.get_attachment(attachment_id)?;
        self.conn.execute(
            r#"
            UPDATE agent_attachments
            SET last_seen_at = ?2,
                tick_count = tick_count + 1,
                context_id = COALESCE(?3, context_id)
            WHERE id = ?1
            "#,
            params![attachment_id, now, context_id],
        )?;
        self.refresh_claim_leases(
            Some(attachment_id),
            Some(attachment.input.agent_id.as_str()),
            Some(attachment.input.namespace.as_str()),
            context_id.or(attachment.context_id.as_deref()),
        )?;
        Ok(())
    }

    pub fn get_attachment(&self, attachment_id: &str) -> Result<AgentAttachmentRecord> {
        self.conn
            .query_row(
                r#"
                SELECT ts, agent_id, agent_type, capabilities_json, namespace, role, metadata_json,
                       active, COALESCE(last_seen_at, ts), tick_count, context_id
                FROM agent_attachments
                WHERE id = ?1
                "#,
                params![attachment_id],
                |row| read_attachment_row(attachment_id, row).map_err(to_sqlite_anyhow),
            )
            .optional()?
            .ok_or_else(|| anyhow!("unknown attachment {attachment_id}"))
    }

    pub fn get_active_attachment(
        &self,
        agent_id: &str,
        namespace: &str,
    ) -> Result<AgentAttachmentRecord> {
        let namespace = self
            .resolve_namespace_alias(Some(namespace))?
            .expect("active attachment namespace should remain present");
        self.conn
            .query_row(
                r#"
                SELECT id, ts, agent_id, agent_type, capabilities_json, namespace, role,
                       metadata_json, active, COALESCE(last_seen_at, ts), tick_count, context_id
                FROM agent_attachments
                WHERE agent_id = ?1 AND namespace = ?2 AND active = 1
                ORDER BY ts DESC
                LIMIT 1
                "#,
                params![agent_id, namespace],
                |row| {
                    let id = row.get::<_, String>(0)?;
                    read_attachment_row_with_offset(&id, row, 1).map_err(to_sqlite_anyhow)
                },
            )
            .optional()?
            .ok_or_else(|| {
                anyhow!("no active attachment for agent {agent_id} in namespace {namespace}")
            })
    }

    pub fn open_context(&self, input: OpenContextInput) -> Result<ContextRecord> {
        let namespace = self
            .resolve_namespace_alias(Some(input.namespace.as_str()))?
            .expect("context namespace should remain present");
        let opened_at = Utc::now();
        let selector = input.selector.unwrap_or_else(|| Selector {
            all: vec![DimensionFilter {
                key: "task".to_string(),
                values: vec![input.task_id.clone()],
            }],
            any: Vec::new(),
            exclude: Vec::new(),
            layers: Vec::new(),
            start_ts: None,
            end_ts: None,
            limit: Some(48),
            namespace: Some(namespace.clone()),
        });
        let existing_id = self
            .conn
            .query_row(
                "SELECT id FROM contexts WHERE namespace = ?1 AND task_id = ?2",
                params![namespace, input.task_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        let id = existing_id.unwrap_or_else(|| format!("ctx:{}", Uuid::now_v7()));
        self.conn.execute(
            r#"
            INSERT INTO contexts(
              id, opened_at, updated_at, namespace, task_id, session_id, objective, selector_json,
              status, current_agent_id, current_attachment_id, last_snapshot_id
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, NULL)
            ON CONFLICT(id) DO UPDATE SET
              updated_at = excluded.updated_at,
              session_id = excluded.session_id,
              objective = excluded.objective,
              selector_json = excluded.selector_json,
              status = excluded.status,
              current_agent_id = excluded.current_agent_id,
              current_attachment_id = excluded.current_attachment_id
            "#,
            params![
                id,
                opened_at.to_rfc3339(),
                opened_at.to_rfc3339(),
                namespace,
                input.task_id,
                input.session_id,
                input.objective,
                serde_json::to_string(&selector)?,
                ContextStatus::Open.as_str(),
                input.agent_id,
                input.attachment_id,
            ],
        )?;
        if let Some(attachment_id) = &input.attachment_id {
            self.touch_attachment(attachment_id, Some(&id))?;
        } else if let Some(agent_id) = input.agent_id.as_deref() {
            let namespace = self.get_context(&id)?.namespace;
            self.touch_active_attachment(agent_id, Some(namespace.as_str()), Some(&id))?;
        }
        let context = self.get_context(&id)?;
        debug!(
            op = "open_context",
            context_id = %context.id,
            namespace = %context.namespace,
            task_id = %context.task_id,
            session_id = %context.session_id,
            current_agent_id = ?context.current_agent_id,
            current_attachment_id = ?context.current_attachment_id,
            "opened or refreshed context"
        );
        Ok(context)
    }

    pub fn get_context(&self, id: &str) -> Result<ContextRecord> {
        self.conn
            .query_row(
                r#"
                SELECT opened_at, updated_at, namespace, task_id, session_id, objective, selector_json,
                       status, current_agent_id, current_attachment_id, last_snapshot_id
                FROM contexts
                WHERE id = ?1
                "#,
                params![id],
                |row| {
                    Ok(ContextRecord {
                        id: id.to_string(),
                        opened_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(0)?)
                            .map_err(to_sqlite_error)?
                            .with_timezone(&Utc),
                        updated_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(1)?)
                            .map_err(to_sqlite_error)?
                            .with_timezone(&Utc),
                        namespace: row.get(2)?,
                        task_id: row.get(3)?,
                        session_id: row.get(4)?,
                        objective: row.get(5)?,
                        selector: serde_json::from_str(&row.get::<_, String>(6)?)
                            .map_err(to_sqlite_error)?,
                        status: parse_context_status(&row.get::<_, String>(7)?)
                            .map_err(to_sqlite_anyhow)?,
                        current_agent_id: row.get(8)?,
                        current_attachment_id: row.get(9)?,
                        last_snapshot_id: row.get(10)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| anyhow!("unknown context {id}"))
    }

    pub fn resolve_context(
        &self,
        context_id: Option<&str>,
        namespace: Option<&str>,
        task_id: Option<&str>,
    ) -> Result<ContextRecord> {
        if let Some(context_id) = context_id {
            return self.get_context(context_id);
        }
        let namespace = self
            .resolve_namespace_alias(namespace)?
            .ok_or_else(|| anyhow!("namespace is required to resolve context"))?;
        let task_id = task_id.ok_or_else(|| anyhow!("task_id is required to resolve context"))?;
        let context_id = self
            .conn
            .query_row(
                "SELECT id FROM contexts WHERE namespace = ?1 AND task_id = ?2",
                params![namespace, task_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .ok_or_else(|| {
                anyhow!("unknown context for namespace={namespace} task_id={task_id}")
            })?;
        self.get_context(&context_id)
    }

    pub fn explain_context(&self, id: &str) -> Result<serde_json::Value> {
        let context = self.get_context(id)?;
        let continuity = self.list_continuity_items(id, true)?;
        let latest_snapshot = context
            .last_snapshot_id
            .clone()
            .map(|snapshot_id| self.explain_snapshot(&snapshot_id))
            .transpose()?;
        let compiler = self.explain_continuity_compiler(id)?;
        Ok(serde_json::json!({
            "context": context,
            "continuity": continuity,
            "latest_snapshot": latest_snapshot,
            "compiler": compiler,
        }))
    }

    pub fn explain_continuity_compiler(&self, context_id: &str) -> Result<serde_json::Value> {
        let state = self.continuity_compiler_state_record(context_id)?;
        let chunks = self
            .load_compiled_continuity_chunks(context_id)?
            .into_iter()
            .map(|chunk| ContinuityCompiledChunkRecord {
                chunk_id: chunk.chunk_id,
                band: chunk.band,
                compiled_at: chunk.compiled_at,
                item_count: chunk.item_count,
                item_ids: chunk.item_ids,
            })
            .collect::<Vec<_>>();
        Ok(serde_json::json!({
            "state": state,
            "chunks": chunks,
        }))
    }

    pub fn persist_continuity_item(
        &self,
        input: ContinuityItemInput,
    ) -> Result<ContinuityItemRecord> {
        let context = self.get_context(&input.context_id)?;
        let id = format!("continuity:{}", Uuid::now_v7());
        let memory_id = format!("continuity-memory:{}", Uuid::now_v7());
        let created_at = Utc::now();
        let status = input.status.unwrap_or(match input.kind {
            ContinuityKind::Signal => ContinuityStatus::Active,
            ContinuityKind::Outcome => ContinuityStatus::Resolved,
            _ => ContinuityStatus::Open,
        });
        let layer = input.layer.unwrap_or_else(|| input.kind.default_layer());
        let importance = input.importance.unwrap_or(match input.kind {
            ContinuityKind::OperationalScar => 0.95,
            ContinuityKind::Decision | ContinuityKind::Constraint => 0.9,
            ContinuityKind::Signal => 0.85,
            _ => 0.7,
        });
        let confidence = input.confidence.unwrap_or(0.85);
        let salience = input.salience.unwrap_or(importance);
        let supports = input.supports.clone();
        let extra_json = serde_json::json!({
            "title": input.title,
            "kind": input.kind.as_str(),
            "status": status.as_str(),
            "context_id": context.id,
            "namespace": context.namespace,
            "task_id": context.task_id,
            "supports": supports,
            "user": input.extra,
        });
        let body = continuity_body(&input.title, &input.body, input.kind, status, &supports);
        self.conn.execute(
            r#"
            INSERT INTO memory_items (
              id, layer, scope, agent_id, session_id, task_id, ts, importance, confidence,
              token_estimate, source_event_id, scope_key, body, extra_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, NULL, ?11, ?12, ?13)
            "#,
            params![
                memory_id,
                layer.as_i64(),
                input.scope.to_string(),
                input.author_agent_id,
                context.session_id,
                context.task_id,
                created_at.to_rfc3339(),
                importance,
                confidence,
                estimate_tokens(&body) as i64,
                format!("{}:{}", context.id, id),
                body,
                extra_json.to_string(),
            ],
        )?;
        self.refresh_fts_entry(&memory_id, &body)?;
        self.refresh_vector(&memory_id, &body)?;
        self.index_item_dimensions(
            "memory",
            &memory_id,
            Some(layer),
            created_at,
            &continuity_dimensions(&context, &input, status),
        )?;
        self.conn.execute(
            r#"
            INSERT INTO continuity_items(
              id, memory_id, ts, updated_at, context_id, namespace, task_id, author_agent_id, kind,
              scope, status, title, body, importance, confidence, salience, supersedes_id, resolved_at, extra_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, NULL, NULL, ?17)
            "#,
            params![
                id,
                memory_id,
                created_at.to_rfc3339(),
                created_at.to_rfc3339(),
                context.id,
                context.namespace,
                context.task_id,
                input.author_agent_id,
                input.kind.as_str(),
                input.scope.to_string(),
                status.as_str(),
                input.title,
                input.body,
                importance,
                confidence,
                salience,
                extra_json.to_string(),
            ],
        )?;
        for support in &supports {
            self.conn.execute(
                r#"
                INSERT OR REPLACE INTO continuity_support(
                  continuity_id, support_type, support_id, reason, weight
                ) VALUES (?1, ?2, ?3, ?4, ?5)
                "#,
                params![
                    id,
                    support.support_type,
                    support.support_id,
                    support.reason,
                    support.weight,
                ],
            )?;
            if let Some(parent_id) =
                self.support_parent_id(&support.support_type, &support.support_id)?
            {
                self.insert_lineage(&LineageRecord {
                    parent_id,
                    child_id: memory_id.clone(),
                    edge_kind: format!("supports_{}", input.kind.as_str()),
                    weight: support.weight,
                })?;
            }
            self.insert_relation(&RelationRecord {
                id: format!("rel:{}", Uuid::now_v7()),
                ts: created_at,
                source_id: support.support_id.clone(),
                target_id: id.clone(),
                relation: "supports".to_string(),
                weight: support.weight,
                attributes: serde_json::json!({
                    "support_type": support.support_type,
                    "reason": support.reason,
                }),
            })?;
        }
        self.insert_relation(&RelationRecord {
            id: format!("rel:{}", Uuid::now_v7()),
            ts: created_at,
            source_id: context.id.clone(),
            target_id: id.clone(),
            relation: "context_item".to_string(),
            weight: 1.0,
            attributes: serde_json::json!({
                "kind": input.kind.as_str(),
                "status": status.as_str(),
            }),
        })?;
        self.touch_active_attachment(
            &input.author_agent_id,
            Some(context.namespace.as_str()),
            Some(context.id.as_str()),
        )?;
        self.ensure_continuity_plasticity(&id, &extra_json, created_at)?;
        self.mark_continuity_context_dirty(&context.id)?;
        let record = self
            .list_continuity_items(&context.id, true)?
            .into_iter()
            .find(|item| item.id == id)
            .ok_or_else(|| anyhow!("failed to load stored continuity item"))?;
        debug!(
            op = "persist_continuity_item",
            continuity_id = %record.id,
            memory_id = %record.memory_id,
            kind = %record.kind.as_str(),
            status = %record.status.as_str(),
            namespace = %record.namespace,
            task_id = %record.task_id,
            context_id = %record.context_id,
            author_agent_id = %record.author_agent_id,
            scope = %record.scope,
            support_count = supports.len(),
            title = %record.title,
            "persisted continuity item"
        );
        Ok(record)
    }

    pub fn claim_work(&self, input: ClaimWorkInput) -> Result<ContinuityItemRecord> {
        let context = self.get_context(&input.context_id)?;
        let now = Utc::now();
        let resources = normalize_work_claim_resources(&input.resources);
        let scope = input.scope.clone();
        let lease_seconds = input
            .lease_seconds
            .unwrap_or_else(default_work_claim_lease_seconds)
            .clamp(15, 3600);
        let claim_key = work_claim_key(
            &context.id,
            scope.clone(),
            &input.agent_id,
            &input.title,
            &resources,
        );
        let existing = self.list_continuity_items(&context.id, true)?;
        let existing_claim = existing.iter().find(|item| {
            item.kind == ContinuityKind::WorkClaim
                && item.author_agent_id == input.agent_id
                && work_claim_coordination(item)
                    .map(|coordination| coordination.claim_key == claim_key)
                    .unwrap_or(false)
        });
        let conflicts = existing
            .iter()
            .filter(|item| {
                item.kind == ContinuityKind::WorkClaim
                    && item.author_agent_id != input.agent_id
                    && work_claim_is_live(item, now)
            })
            .filter_map(|item| {
                let coordination = work_claim_coordination(item)?;
                let overlaps = coordination
                    .resources
                    .iter()
                    .any(|resource| resources.iter().any(|other| other == resource));
                if !overlaps || !(coordination.exclusive || input.exclusive) {
                    return None;
                }
                Some(WorkClaimConflict {
                    id: item.id.clone(),
                    agent_id: item.author_agent_id.clone(),
                    title: item.title.clone(),
                    resources: coordination.resources,
                    exclusive: coordination.exclusive,
                })
            })
            .collect::<Vec<_>>();
        let coordination = WorkClaimCoordination {
            claim_key,
            resources: resources.clone(),
            exclusive: input.exclusive,
            attachment_id: input.attachment_id.clone(),
            lease_seconds,
            lease_expires_at: Some(now + chrono::Duration::seconds(lease_seconds as i64)),
            renewed_at: Some(now),
            conflict_count: conflicts.len(),
            conflicts_with: conflicts,
        };
        if let Some(existing_claim) = existing_claim {
            return self.refresh_work_claim(existing_claim, &input, &coordination);
        }
        self.persist_continuity_item(ContinuityItemInput {
            context_id: input.context_id,
            author_agent_id: input.agent_id,
            kind: ContinuityKind::WorkClaim,
            title: input.title,
            body: input.body,
            scope,
            status: Some(ContinuityStatus::Active),
            importance: Some(if input.exclusive { 0.9 } else { 0.78 }),
            confidence: Some(0.92),
            salience: Some(if input.exclusive { 0.94 } else { 0.86 }),
            layer: Some(MemoryLayer::Hot),
            supports: Vec::new(),
            dimensions: claim_dimensions(&resources, input.exclusive, &coordination.claim_key),
            extra: merge_work_claim_extra(input.extra, &coordination),
        })
    }

    pub fn list_continuity_items(
        &self,
        context_id: &str,
        include_resolved: bool,
    ) -> Result<Vec<ContinuityItemRecord>> {
        let sql = if include_resolved {
            r#"
            SELECT id, memory_id, ts, updated_at, context_id, namespace, task_id, author_agent_id,
                   kind, scope, status, title, body, importance, confidence, salience,
                   supersedes_id, resolved_at, extra_json
            FROM continuity_items
            WHERE context_id = ?1
            ORDER BY salience DESC, updated_at DESC, ts DESC
            "#
        } else {
            r#"
            SELECT id, memory_id, ts, updated_at, context_id, namespace, task_id, author_agent_id,
                   kind, scope, status, title, body, importance, confidence, salience,
                   supersedes_id, resolved_at, extra_json
            FROM continuity_items
            WHERE context_id = ?1 AND status IN ('open', 'active')
            ORDER BY salience DESC, updated_at DESC, ts DESC
            "#
        };
        let mut stmt = self.conn.prepare(sql)?;
        let mut rows = stmt.query(params![context_id])?;
        let mut raw = Vec::new();
        while let Some(row) = rows.next()? {
            raw.push(read_continuity_row(row)?);
        }
        let now = Utc::now();
        let plasticity_map = self.plasticity_for_continuity_many(
            &raw.iter().map(|item| item.id.clone()).collect::<Vec<_>>(),
        )?;
        let support_map = self.supports_for_continuity_many(
            &raw.iter().map(|item| item.id.clone()).collect::<Vec<_>>(),
        )?;
        let mut out = raw
            .into_iter()
            .map(|item| {
                let item_id = item.id.clone();
                build_continuity_record(
                    item,
                    support_map.get(&item_id).cloned().unwrap_or_default(),
                    plasticity_map.get(&item_id).cloned(),
                    now,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        out.sort_by(|a, b| {
            b.retention
                .effective_salience
                .partial_cmp(&a.retention.effective_salience)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.updated_at.cmp(&a.updated_at))
                .then_with(|| b.created_at.cmp(&a.created_at))
        });
        Ok(out)
    }

    fn mark_continuity_context_dirty(&self, context_id: &str) -> Result<()> {
        let item_count = self.conn.query_row(
            "SELECT COUNT(*) FROM continuity_items WHERE context_id = ?1",
            params![context_id],
            |row| row.get::<_, i64>(0),
        )?;
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            r#"
            INSERT INTO continuity_compiler_state(context_id, dirty, item_count, refreshed_at, compiled_at)
            VALUES (?1, 1, ?2, ?3, NULL)
            ON CONFLICT(context_id) DO UPDATE SET
              dirty = 1,
              item_count = excluded.item_count,
              refreshed_at = excluded.refreshed_at
            "#,
            params![context_id, item_count, now],
        )?;
        Ok(())
    }

    fn maybe_refresh_compiled_continuity(
        &self,
        context_id: &str,
        include_resolved: bool,
        now: DateTime<Utc>,
    ) -> Result<Vec<CompiledContinuityChunk>> {
        let state = self
            .conn
            .query_row(
                "SELECT dirty FROM continuity_compiler_state WHERE context_id = ?1",
                params![context_id],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        if state != Some(0) {
            self.compile_continuity_context(context_id, include_resolved, now)?;
        }
        self.load_compiled_continuity_chunks(context_id)
    }

    fn continuity_compiler_state_record(
        &self,
        context_id: &str,
    ) -> Result<Option<ContinuityCompilerStateRecord>> {
        self.conn
            .query_row(
                r#"
                SELECT context_id, dirty, item_count, refreshed_at, compiled_at
                FROM continuity_compiler_state
                WHERE context_id = ?1
                "#,
                params![context_id],
                |row| {
                    let compiled_at = row
                        .get::<_, Option<String>>(4)?
                        .map(|value| {
                            DateTime::parse_from_rfc3339(&value)
                                .map(|dt| dt.with_timezone(&Utc))
                                .map_err(to_sqlite_error)
                        })
                        .transpose()?;
                    Ok(ContinuityCompilerStateRecord {
                        context_id: row.get(0)?,
                        dirty: row.get::<_, i64>(1)? != 0,
                        item_count: row.get::<_, i64>(2)? as usize,
                        refreshed_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(3)?)
                            .map_err(to_sqlite_error)?
                            .with_timezone(&Utc),
                        compiled_at,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    fn load_compiled_continuity_chunks(
        &self,
        context_id: &str,
    ) -> Result<Vec<CompiledContinuityChunk>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT chunk_id, band, compiled_at, item_count, item_ids_json
            FROM continuity_compiled_chunks
            WHERE context_id = ?1
            ORDER BY CASE band
              WHEN 'hot' THEN 0
              WHEN 'warm' THEN 1
              ELSE 2
            END ASC
            "#,
        )?;
        let mut rows = stmt.query(params![context_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(CompiledContinuityChunk {
                chunk_id: row.get(0)?,
                band: row.get(1)?,
                compiled_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(2)?)
                    .map_err(to_sqlite_error)?
                    .with_timezone(&Utc),
                item_count: row.get::<_, i64>(3)? as usize,
                item_ids: serde_json::from_str(&row.get::<_, String>(4)?)?,
            });
        }
        Ok(out)
    }

    fn compile_continuity_context(
        &self,
        context_id: &str,
        include_resolved: bool,
        now: DateTime<Utc>,
    ) -> Result<()> {
        let hot = self.compiler_band_items_with_direct_supports(
            self.compiler_band_items(
                context_id,
                include_resolved,
                "status IN ('open', 'active')",
                "ORDER BY salience DESC, updated_at DESC, ts DESC",
                16,
            )?,
            8,
            now,
        )?;
        let warm = self.compiler_band_items_with_direct_supports(
            self.compiler_band_items(
            context_id,
            true,
            "kind IN ('fact', 'decision', 'constraint', 'incident', 'operational_scar', 'lesson', 'summary', 'derivation')",
            "ORDER BY salience DESC, updated_at DESC, ts DESC",
            28,
        )?,
            10,
            now,
        )?;
        let cold = self.compiler_band_items_with_direct_supports(
            self.compiler_band_items(
            context_id,
            true,
            "kind IN ('fact', 'decision', 'constraint', 'incident', 'operational_scar', 'lesson', 'summary', 'outcome', 'derivation')",
            "ORDER BY importance DESC, salience DESC, updated_at DESC, ts DESC",
            48,
        )?,
            12,
            now,
        )?;

        let mut used = HashSet::new();
        let bands = [("hot", hot), ("warm", warm), ("cold", cold)]
            .into_iter()
            .map(|(band, items)| {
                let filtered = items
                    .into_iter()
                    .filter(|item| used.insert(item.id.clone()))
                    .collect::<Vec<_>>();
                let ranked = compiler_rank_items_by_support_graph(filtered);
                (band, ranked)
            })
            .collect::<Vec<_>>();

        self.conn.execute(
            "DELETE FROM continuity_compiled_fts WHERE context_id = ?1",
            params![context_id],
        )?;
        self.conn.execute(
            "DELETE FROM continuity_compiled_chunks WHERE context_id = ?1",
            params![context_id],
        )?;

        let compiled_at = now.to_rfc3339();
        let mut compiled_item_count = 0usize;
        for (band, items) in bands {
            if items.is_empty() {
                continue;
            }
            compiled_item_count += items.len();
            let chunk_id = format!("continuity-compiled:{context_id}:{band}");
            let item_ids = items.iter().map(|item| item.id.clone()).collect::<Vec<_>>();
            let body = compiler_chunk_body(band, &items);
            self.conn.execute(
                r#"
                INSERT INTO continuity_compiled_chunks(
                  chunk_id, context_id, band, compiled_at, item_count, item_ids_json, body
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                "#,
                params![
                    chunk_id.as_str(),
                    context_id,
                    band,
                    compiled_at,
                    items.len() as i64,
                    serde_json::to_string(&item_ids)?,
                    body.as_str(),
                ],
            )?;
            self.conn.execute(
                "INSERT INTO continuity_compiled_fts(chunk_id, context_id, band, body) VALUES (?1, ?2, ?3, ?4)",
                params![chunk_id.as_str(), context_id, band, body.as_str()],
            )?;
        }

        let total_items = self.conn.query_row(
            "SELECT COUNT(*) FROM continuity_items WHERE context_id = ?1",
            params![context_id],
            |row| row.get::<_, i64>(0),
        )?;
        self.conn.execute(
            r#"
            INSERT INTO continuity_compiler_state(context_id, dirty, item_count, refreshed_at, compiled_at)
            VALUES (?1, 0, ?2, ?3, ?3)
            ON CONFLICT(context_id) DO UPDATE SET
              dirty = 0,
              item_count = excluded.item_count,
              refreshed_at = excluded.refreshed_at,
              compiled_at = excluded.compiled_at
            "#,
            params![context_id, total_items, compiled_at],
        )?;
        debug!(
            op = "compile_continuity_context",
            context_id = %context_id,
            compiled_item_count,
            total_item_count = total_items,
            "compiled continuity bands"
        );
        Ok(())
    }

    fn compiler_band_items(
        &self,
        context_id: &str,
        include_resolved: bool,
        predicate_sql: &str,
        order_sql: &str,
        limit: usize,
    ) -> Result<Vec<ContinuityItemRecord>> {
        let status_clause = if include_resolved {
            String::new()
        } else {
            " AND status IN ('open', 'active')".to_string()
        };
        let sql = format!(
            r#"
            SELECT id, memory_id, ts, updated_at, context_id, namespace, task_id, author_agent_id,
                   kind, scope, status, title, body, importance, confidence, salience,
                   supersedes_id, resolved_at, extra_json
            FROM continuity_items
            WHERE context_id = ?1 AND {predicate}{status_clause}
            {order}
            LIMIT ?2
            "#,
            predicate = predicate_sql,
            order = order_sql
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(params![context_id, limit as i64])?;
        let mut raw = Vec::new();
        while let Some(row) = rows.next()? {
            raw.push(read_continuity_row(row)?);
        }
        let support_map = self.supports_for_continuity_many(
            &raw.iter().map(|item| item.id.clone()).collect::<Vec<_>>(),
        )?;
        let plasticity_map = self.plasticity_for_continuity_many(
            &raw.iter().map(|item| item.id.clone()).collect::<Vec<_>>(),
        )?;
        raw.into_iter()
            .map(|item| {
                let item_id = item.id.clone();
                build_continuity_record(
                    item,
                    support_map.get(&item_id).cloned().unwrap_or_default(),
                    plasticity_map.get(&item_id).cloned(),
                    Utc::now(),
                )
            })
            .collect()
    }

    fn compiler_band_items_with_direct_supports(
        &self,
        mut items: Vec<ContinuityItemRecord>,
        max_promoted_supports: usize,
        now: DateTime<Utc>,
    ) -> Result<Vec<ContinuityItemRecord>> {
        if items.is_empty() || max_promoted_supports == 0 {
            return Ok(items);
        }

        let existing_ids = items
            .iter()
            .map(|item| item.id.clone())
            .collect::<HashSet<_>>();
        let mut ranked_support_ids = Vec::<(usize, usize, f64, String)>::new();
        for (anchor_rank, item) in items.iter().enumerate() {
            for (support_rank, support) in item.supports.iter().enumerate() {
                if support.support_type != "continuity"
                    || existing_ids.contains(&support.support_id)
                {
                    continue;
                }
                ranked_support_ids.push((
                    anchor_rank,
                    support_rank,
                    support.weight,
                    support.support_id.clone(),
                ));
            }
        }
        if ranked_support_ids.is_empty() {
            return Ok(items);
        }

        ranked_support_ids.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| {
                    right
                        .2
                        .partial_cmp(&left.2)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.3.cmp(&right.3))
        });

        let mut seen = HashSet::new();
        let support_ids = ranked_support_ids
            .into_iter()
            .filter_map(|(_, _, _, support_id)| {
                if seen.insert(support_id.clone()) {
                    Some(support_id)
                } else {
                    None
                }
            })
            .take(max_promoted_supports)
            .collect::<Vec<_>>();
        if support_ids.is_empty() {
            return Ok(items);
        }

        let support_map = self.supports_for_continuity_many(&support_ids)?;
        let plasticity_map = self.plasticity_for_continuity_many(&support_ids)?;
        let support_rows = self
            .raw_continuity_rows_by_ids(&support_ids)?
            .into_iter()
            .map(|row| {
                let item_id = row.id.clone();
                build_continuity_record(
                    row,
                    support_map.get(&item_id).cloned().unwrap_or_default(),
                    plasticity_map.get(&item_id).cloned(),
                    now,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let support_by_id = support_rows
            .into_iter()
            .map(|item| (item.id.clone(), item))
            .collect::<HashMap<_, _>>();

        items.extend(
            support_ids
                .into_iter()
                .filter_map(|support_id| support_by_id.get(&support_id).cloned()),
        );
        Ok(items)
    }

    fn raw_continuity_rows_by_ids(&self, ids: &[String]) -> Result<Vec<RawContinuityRow>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = std::iter::repeat_n("?", ids.len())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            r#"
            SELECT id, memory_id, ts, updated_at, context_id, namespace, task_id, author_agent_id,
                   kind, scope, status, title, body, importance, confidence, salience,
                   supersedes_id, resolved_at, extra_json
            FROM continuity_items
            WHERE id IN ({placeholders})
            "#
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(rusqlite::params_from_iter(ids.iter()))?;
        let mut raw = Vec::new();
        while let Some(row) = rows.next()? {
            raw.push(read_continuity_row(row)?);
        }
        Ok(raw)
    }

    pub fn recall_continuity(
        &self,
        context_id: &str,
        objective: &str,
        include_resolved: bool,
        limit: usize,
    ) -> Result<ContinuityRecall> {
        let started = Instant::now();
        let limit = limit.max(1).min(24);
        let priority_window = limit.max(8) * 8;
        let lexical_window = limit.max(8) * 6;
        let now = Utc::now();

        let mut seeds = BTreeMap::<String, RecallSeed>::new();
        let compiled_started = Instant::now();
        let compiled_chunks =
            self.maybe_refresh_compiled_continuity(context_id, include_resolved, now)?;
        let mut compiled_len = 0usize;

        let priority_sql = if include_resolved {
            r#"
            SELECT id, memory_id, ts, updated_at, context_id, namespace, task_id, author_agent_id,
                   kind, scope, status, title, body, importance, confidence, salience,
                   supersedes_id, resolved_at, extra_json
            FROM continuity_items
            WHERE context_id = ?1
            ORDER BY salience DESC, updated_at DESC, ts DESC
            LIMIT ?2
            "#
        } else {
            r#"
            SELECT id, memory_id, ts, updated_at, context_id, namespace, task_id, author_agent_id,
                   kind, scope, status, title, body, importance, confidence, salience,
                   supersedes_id, resolved_at, extra_json
            FROM continuity_items
            WHERE context_id = ?1 AND status IN ('open', 'active')
            ORDER BY salience DESC, updated_at DESC, ts DESC
            LIMIT ?2
            "#
        };
        let mut stmt = self.conn.prepare(priority_sql)?;
        let mut rows = stmt.query(params![context_id, priority_window as i64])?;
        let mut priority_len = 0usize;
        while let Some(row) = rows.next()? {
            let raw = read_continuity_row(row)?;
            let seed = seeds.entry(raw.id.clone()).or_default();
            seed.raw = Some(raw);
            seed.priority_rank = Some(priority_len);
            priority_len += 1;
        }

        let lexical_started = Instant::now();
        let lexical_query = fts_query(objective);
        let mut lexical_len = 0usize;
        if !lexical_query.is_empty() && !compiled_chunks.is_empty() {
            let compiled_sql = r#"
                SELECT chunk_id, band
                FROM continuity_compiled_fts
                WHERE context_id = ?1 AND continuity_compiled_fts MATCH ?2
                LIMIT ?3
            "#;
            let mut stmt = self.conn.prepare(compiled_sql)?;
            let mut rows = stmt.query(params![context_id, lexical_query.clone(), 3_i64])?;
            let chunk_by_id = compiled_chunks
                .iter()
                .map(|chunk| (chunk.chunk_id.clone(), chunk))
                .collect::<BTreeMap<_, _>>();
            let mut matched_item_ids = Vec::new();
            while let Some(row) = rows.next()? {
                let chunk_id = row.get::<_, String>(0)?;
                let Some(chunk) = chunk_by_id.get(&chunk_id) else {
                    continue;
                };
                for item_id in &chunk.item_ids {
                    let seed = seeds.entry(item_id.clone()).or_default();
                    seed.compiled_rank.get_or_insert(compiled_len);
                    seed.compiled_band.get_or_insert_with(|| chunk.band.clone());
                    matched_item_ids.push(item_id.clone());
                }
                compiled_len += chunk.item_count.max(1);
            }
            let missing_ids = matched_item_ids
                .into_iter()
                .filter(|id| seeds.get(id).and_then(|seed| seed.raw.as_ref()).is_none())
                .collect::<Vec<_>>();
            for raw in self.raw_continuity_rows_by_ids(&missing_ids)? {
                let seed = seeds.entry(raw.id.clone()).or_default();
                seed.raw = Some(raw);
            }
        }
        if !lexical_query.is_empty() {
            let lexical_sql = if include_resolved {
                r#"
                SELECT c.id, c.memory_id, c.ts, c.updated_at, c.context_id, c.namespace, c.task_id,
                       c.author_agent_id, c.kind, c.scope, c.status, c.title, c.body,
                       c.importance, c.confidence, c.salience, c.supersedes_id, c.resolved_at,
                       c.extra_json
                FROM continuity_items c
                JOIN memory_fts ON memory_fts.memory_id = c.memory_id
                WHERE c.context_id = ?1 AND memory_fts MATCH ?2
                ORDER BY bm25(memory_fts)
                LIMIT ?3
                "#
            } else {
                r#"
                SELECT c.id, c.memory_id, c.ts, c.updated_at, c.context_id, c.namespace, c.task_id,
                       c.author_agent_id, c.kind, c.scope, c.status, c.title, c.body,
                       c.importance, c.confidence, c.salience, c.supersedes_id, c.resolved_at,
                       c.extra_json
                FROM continuity_items c
                JOIN memory_fts ON memory_fts.memory_id = c.memory_id
                WHERE c.context_id = ?1
                  AND c.status IN ('open', 'active')
                  AND memory_fts MATCH ?2
                ORDER BY bm25(memory_fts)
                LIMIT ?3
                "#
            };
            let mut stmt = self.conn.prepare(lexical_sql)?;
            let mut rows = stmt.query(params![context_id, lexical_query, lexical_window as i64])?;
            while let Some(row) = rows.next()? {
                let raw = read_continuity_row(row)?;
                let seed = seeds.entry(raw.id.clone()).or_default();
                seed.raw = Some(raw);
                seed.lexical_rank = Some(lexical_len);
                lexical_len += 1;
            }
        }
        let lexical_ms = lexical_started.elapsed().as_millis();

        let support_map =
            self.supports_for_continuity_many(&seeds.keys().cloned().collect::<Vec<_>>())?;
        let plasticity_map =
            self.plasticity_for_continuity_many(&seeds.keys().cloned().collect::<Vec<_>>())?;
        let seed_ids = seeds.keys().cloned().collect::<HashSet<_>>();
        let (anchor_support_scores, support_backlink_scores) =
            continuity_recall_support_scores(&support_map, &seed_ids);
        let compiled_hit_count = seeds
            .values()
            .filter(|seed| seed.compiled_band.is_some())
            .count();
        let lexical_fallback_count = seeds
            .values()
            .filter(|seed| seed.compiled_band.is_none() && seed.lexical_rank.is_some())
            .count();
        let priority_seed_count = seeds
            .values()
            .filter(|seed| seed.priority_rank.is_some())
            .count();
        let mut recall_items = seeds
            .into_values()
            .filter_map(|mut seed| seed.raw.take().map(|raw| (raw, seed)))
            .map(|(raw, seed)| {
                let raw_id = raw.id.clone();
                let item = build_continuity_record(
                    raw,
                    support_map.get(&raw_id).cloned().unwrap_or_default(),
                    plasticity_map.get(&raw_id).cloned(),
                    now,
                )?;
                let belief_key = continuity_belief_key(&item.extra);
                let practice_key = continuity_practice_key(&item.extra);
                let source_role = continuity_source_role(&item.extra);
                let plasticity = plasticity_map
                    .get(&item.id)
                    .map(|value| value.state.clone());
                let lexical_score = rank_score(seed.lexical_rank, lexical_len);
                let priority_score = rank_score(seed.priority_rank, priority_len);
                let compiled_score = rank_score(seed.compiled_rank, compiled_len);
                let lexical_weight = if seed.compiled_band.is_some() {
                    0.75
                } else {
                    1.05
                };
                let score = lexical_score * lexical_weight
                    + compiled_score * 1.2
                    + priority_score * 0.65
                    + item.retention.effective_salience * 0.9
                    + continuity_kind_boost(item.kind)
                    + continuity_status_boost(item.status)
                    + continuity_retention_adjustment(&item)
                    + continuity_source_role_adjustment(
                        belief_key.as_deref(),
                        source_role.as_deref(),
                        item.kind,
                    )
                    + anchor_support_scores
                        .get(&item.id)
                        .copied()
                        .unwrap_or_default()
                    + support_backlink_scores
                        .get(&item.id)
                        .copied()
                        .unwrap_or_default()
                    + plasticity_recall_adjustment(plasticity.as_ref());
                let mut why = Vec::new();
                if seed.lexical_rank.is_some() {
                    why.push("lexical".to_string());
                }
                if let Some(band) = seed.compiled_band.as_deref() {
                    why.push(format!("compiled_{band}"));
                }
                if seed.priority_rank.is_some() {
                    why.push("retention_priority".to_string());
                }
                if anchor_support_scores.contains_key(&item.id) {
                    why.push("causal_anchor".to_string());
                }
                if support_backlink_scores.contains_key(&item.id) {
                    why.push("supports_anchor".to_string());
                }
                why.push(item.retention.class.clone());
                Ok(ScoredContinuityRecallItem {
                    item: ContinuityRecallItem {
                        id: item.id.clone(),
                        memory_id: item.memory_id.clone(),
                        kind: item.kind,
                        status: item.status,
                        title: item.title.clone(),
                        preview: continuity_preview(&item.body, 180),
                        author_agent_id: item.author_agent_id.clone(),
                        updated_at: item.updated_at,
                        effective_salience: item.retention.effective_salience,
                        support_count: item.supports.len(),
                        score,
                        why,
                    },
                    belief_key,
                    practice_key,
                    source_role,
                    plasticity,
                    practice_state: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        apply_belief_key_competition(&mut recall_items);
        apply_practice_lifecycle_competition(
            &mut recall_items,
            crate::continuity::objective_requests_history_context(objective),
            now,
        );
        recall_items.sort_by(|a, b| {
            b.item
                .score
                .partial_cmp(&a.item.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.item.updated_at.cmp(&a.item.updated_at))
        });
        let total_candidates = recall_items.len();
        let mut recall_items = recall_items
            .into_iter()
            .map(|mut item| {
                let mut seen = HashSet::new();
                item.item.why.retain(|why| seen.insert(why.clone()));
                item.item
            })
            .collect::<Vec<_>>();
        recall_items.truncate(limit);
        let mut band_hit_counts = BTreeMap::<String, usize>::new();
        for item in &recall_items {
            if let Some(band) = item
                .why
                .iter()
                .find_map(|why| why.strip_prefix("compiled_"))
                .map(ToString::to_string)
            {
                *band_hit_counts.entry(band).or_insert(0) += 1;
            }
        }
        let dominant_band = band_hit_counts
            .iter()
            .max_by(|left, right| {
                left.1
                    .cmp(right.1)
                    .then_with(|| left.0.cmp(right.0).reverse())
            })
            .map(|(band, _)| band.clone());

        Ok(ContinuityRecall {
            query: objective.to_string(),
            summary: continuity_recall_summary(objective, &recall_items),
            answer_hint: continuity_recall_answer_hint(&recall_items),
            total_candidates,
            timings_ms: serde_json::json!({
                "compiled": compiled_started.elapsed().as_millis(),
                "lexical": lexical_ms,
                "total": started.elapsed().as_millis(),
            }),
            compiler: ContinuityRecallCompiler {
                compiled_hit_count,
                lexical_fallback_count,
                priority_seed_count,
                dominant_band,
                band_hit_counts,
            },
            items: recall_items,
        })
    }

    pub fn explain_continuity_item(&self, id: &str) -> Result<serde_json::Value> {
        let item = self
            .list_continuity_items(
                &self
                    .conn
                    .query_row(
                        "SELECT context_id FROM continuity_items WHERE id = ?1",
                        params![id],
                        |row| row.get::<_, String>(0),
                    )
                    .optional()?
                    .ok_or_else(|| anyhow!("unknown continuity item {id}"))?,
                true,
            )?
            .into_iter()
            .find(|item| item.id == id)
            .ok_or_else(|| anyhow!("unknown continuity item {id}"))?;
        let memory = self
            .memories_by_ids(std::slice::from_ref(&item.memory_id))?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("missing continuity memory {}", item.memory_id))?;
        Ok(serde_json::json!({
            "item": item,
            "provenance": self.provenance_for_memory(&memory)?,
        }))
    }

    pub fn resolve_continuity_item(
        &self,
        input: ResolveOrSupersedeInput,
    ) -> Result<ContinuityItemRecord> {
        let existing_context = self
            .conn
            .query_row(
                "SELECT context_id, memory_id, title, body, extra_json FROM continuity_items WHERE id = ?1",
                params![input.continuity_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| anyhow!("unknown continuity item {}", input.continuity_id))?;
        let updated_at = Utc::now();
        let mut extra: serde_json::Value = serde_json::from_str(&existing_context.4)?;
        extra["resolution"] = serde_json::json!({
            "actor_agent_id": input.actor_agent_id,
            "note": input.resolution_note,
            "extra": input.extra,
        });
        self.conn.execute(
            r#"
            UPDATE continuity_items
            SET updated_at = ?2,
                status = ?3,
                supersedes_id = ?4,
                resolved_at = ?5,
                extra_json = ?6
            WHERE id = ?1
            "#,
            params![
                input.continuity_id,
                updated_at.to_rfc3339(),
                input.new_status.as_str(),
                input.supersedes_id,
                if input.new_status.is_open() {
                    None::<String>
                } else {
                    Some(updated_at.to_rfc3339())
                },
                extra.to_string(),
            ],
        )?;
        let updated_body = format!(
            "[{}] {}\n{}",
            input.new_status.as_str(),
            existing_context.2,
            existing_context.3
        );
        self.conn.execute(
            "UPDATE memory_items SET ts = ?2, body = ?3, extra_json = ?4 WHERE id = ?1",
            params![
                existing_context.1,
                updated_at.to_rfc3339(),
                updated_body,
                extra.to_string(),
            ],
        )?;
        self.refresh_fts_entry(&existing_context.1, &updated_body)?;
        self.refresh_vector(&existing_context.1, &updated_body)?;
        self.replace_dimension(
            "memory",
            &existing_context.1,
            "continuity_status",
            input.new_status.as_str(),
        )?;
        if let Some(supersedes_id) = &input.supersedes_id {
            self.insert_relation(&RelationRecord {
                id: format!("rel:{}", Uuid::now_v7()),
                ts: updated_at,
                source_id: input.continuity_id.clone(),
                target_id: supersedes_id.clone(),
                relation: "supersedes".to_string(),
                weight: 1.0,
                attributes: serde_json::json!({}),
            })?;
            if let Some(target_extra_json) = self
                .conn
                .query_row(
                    "SELECT extra_json FROM continuity_items WHERE id = ?1",
                    params![supersedes_id],
                    |row| row.get::<_, String>(0),
                )
                .optional()?
            {
                let target_extra = serde_json::from_str::<serde_json::Value>(&target_extra_json)?;
                if continuity_belief_key(&extra).is_some()
                    && continuity_belief_key(&extra) == continuity_belief_key(&target_extra)
                {
                    self.ensure_continuity_plasticity(&input.continuity_id, &extra, updated_at)?;
                    self.ensure_continuity_plasticity(supersedes_id, &target_extra, updated_at)?;
                    self.bump_continuity_plasticity(
                        &input.continuity_id,
                        updated_at,
                        false,
                        false,
                        false,
                        true,
                    )?;
                    self.bump_continuity_plasticity(
                        supersedes_id,
                        updated_at,
                        true,
                        false,
                        true,
                        false,
                    )?;
                }
            }
        }
        if input.new_status == ContinuityStatus::Superseded {
            if let Some(supersedes_id) = input.supersedes_id.as_deref() {
                let items = self.list_continuity_items(&existing_context.0, true)?;
                if let (Some(previous), Some(replacement)) = (
                    items.iter().find(|item| item.id == input.continuity_id),
                    items.iter().find(|item| item.id == supersedes_id),
                ) {
                    self.maybe_emit_belief_update_lesson(
                        previous,
                        replacement,
                        &input.actor_agent_id,
                        input.resolution_note.as_deref(),
                    )?;
                }
            }
        }
        self.mark_continuity_context_dirty(&existing_context.0)?;
        self.list_continuity_items(&existing_context.0, true)?
            .into_iter()
            .find(|item| item.id == input.continuity_id)
            .ok_or_else(|| anyhow!("failed to reload continuity item"))
    }

    fn maybe_emit_belief_update_lesson(
        &self,
        previous: &ContinuityItemRecord,
        replacement: &ContinuityItemRecord,
        actor_agent_id: &str,
        resolution_note: Option<&str>,
    ) -> Result<Option<ContinuityItemRecord>> {
        let Some(previous_key) = continuity_belief_key(&previous.extra) else {
            return Ok(None);
        };
        let Some(replacement_key) = continuity_belief_key(&replacement.extra) else {
            return Ok(None);
        };
        if previous_key != replacement_key || previous.id == replacement.id {
            return Ok(None);
        }

        let reason = resolution_note
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("A newer belief with the same key replaced the prior state.");
        let previous_title = compact_learning_text(&previous.title, 72);
        let replacement_title = compact_learning_text(&replacement.title, 72);
        let previous_body = compact_learning_text(&previous.body, 220);
        let replacement_body = compact_learning_text(&replacement.body, 220);
        let lesson_title = format!("Belief update: {previous_title} -> {replacement_title}");
        let lesson_body = format!(
            "Stable belief changed for {belief_key}.\nPrevious: {previous_body}\nCurrent: {replacement_body}\nReason: {reason}",
            belief_key = previous_key,
        );

        let mut lesson_extra = serde_json::json!({
            "learning_trigger": "prediction_error_reconsolidation",
            "learning_belief_key": previous_key,
            "previous_continuity_id": previous.id,
            "current_continuity_id": replacement.id,
            "resolution_note": resolution_note,
        });
        if let Some(source_role) = continuity_source_role(&replacement.extra)
            .or_else(|| continuity_source_role(&previous.extra))
        {
            lesson_extra["learning_source_role"] = serde_json::json!(source_role);
        }

        let lesson = self.persist_continuity_item(ContinuityItemInput {
            context_id: previous.context_id.clone(),
            author_agent_id: actor_agent_id.to_string(),
            kind: ContinuityKind::Lesson,
            title: lesson_title,
            body: lesson_body,
            scope: Scope::Project,
            status: Some(ContinuityStatus::Resolved),
            importance: Some(
                ((previous.importance + replacement.importance) * 0.5).clamp(0.75, 0.95),
            ),
            confidence: Some(replacement.confidence.max(0.85)),
            salience: Some(replacement.salience.max(0.78)),
            layer: Some(MemoryLayer::Semantic),
            supports: vec![
                SupportRef {
                    support_type: "continuity".to_string(),
                    support_id: previous.id.clone(),
                    reason: Some("belief_update_previous".to_string()),
                    weight: 1.0,
                },
                SupportRef {
                    support_type: "continuity".to_string(),
                    support_id: replacement.id.clone(),
                    reason: Some("belief_update_current".to_string()),
                    weight: 1.2,
                },
            ],
            dimensions: vec![
                DimensionValue {
                    key: "learning_trigger".to_string(),
                    value: "prediction_error_reconsolidation".to_string(),
                    weight: 95,
                },
                DimensionValue {
                    key: "learning_belief_key".to_string(),
                    value: previous_key,
                    weight: 90,
                },
            ],
            extra: lesson_extra,
        })?;
        Ok(Some(lesson))
    }

    fn refresh_work_claim(
        &self,
        existing: &ContinuityItemRecord,
        input: &ClaimWorkInput,
        coordination: &WorkClaimCoordination,
    ) -> Result<ContinuityItemRecord> {
        let updated_at = Utc::now();
        let title = input.title.clone();
        let extra = merge_work_claim_extra(input.extra.clone(), coordination);
        let body = input.body.clone();
        let importance = if input.exclusive { 0.9 } else { 0.78 };
        let salience = if input.exclusive { 0.94 } else { 0.86 };
        self.conn.execute(
            r#"
            UPDATE continuity_items
            SET updated_at = ?2,
                status = ?3,
                title = ?4,
                body = ?5,
                importance = ?6,
                confidence = ?7,
                salience = ?8,
                resolved_at = NULL,
                extra_json = ?9
            WHERE id = ?1
            "#,
            params![
                existing.id,
                updated_at.to_rfc3339(),
                ContinuityStatus::Active.as_str(),
                title.as_str(),
                body.as_str(),
                importance,
                0.92_f64,
                salience,
                serde_json::json!({
                    "title": title.as_str(),
                    "kind": ContinuityKind::WorkClaim.as_str(),
                    "status": ContinuityStatus::Active.as_str(),
                    "context_id": existing.context_id,
                    "namespace": existing.namespace,
                    "task_id": existing.task_id,
                    "supports": existing.supports.clone(),
                    "user": extra.clone(),
                })
                .to_string(),
            ],
        )?;
        let memory_body = continuity_body(
            &input.title,
            &body,
            ContinuityKind::WorkClaim,
            ContinuityStatus::Active,
            &existing.supports,
        );
        self.conn.execute(
            r#"
            UPDATE memory_items
            SET ts = ?2,
                importance = ?3,
                confidence = ?4,
                token_estimate = ?5,
                body = ?6,
                extra_json = ?7
            WHERE id = ?1
            "#,
            params![
                existing.memory_id,
                updated_at.to_rfc3339(),
                importance,
                0.92_f64,
                estimate_tokens(&memory_body) as i64,
                memory_body,
                serde_json::json!({
                    "title": title.as_str(),
                    "kind": ContinuityKind::WorkClaim.as_str(),
                    "status": ContinuityStatus::Active.as_str(),
                    "context_id": existing.context_id,
                    "namespace": existing.namespace,
                    "task_id": existing.task_id,
                    "supports": existing.supports.clone(),
                    "user": extra,
                })
                .to_string(),
            ],
        )?;
        self.refresh_fts_entry(&existing.memory_id, &memory_body)?;
        self.refresh_vector(&existing.memory_id, &memory_body)?;
        self.replace_dimension(
            "memory",
            &existing.memory_id,
            "continuity_status",
            ContinuityStatus::Active.as_str(),
        )?;
        self.replace_dimension(
            "memory",
            &existing.memory_id,
            "claim.exclusive",
            if input.exclusive { "true" } else { "false" },
        )?;
        self.mark_continuity_context_dirty(&existing.context_id)?;
        self.list_continuity_items(&existing.context_id, true)?
            .into_iter()
            .find(|item| item.id == existing.id)
            .ok_or_else(|| anyhow!("failed to reload work claim {}", existing.id))
    }

    fn refresh_claim_leases(
        &self,
        attachment_id: Option<&str>,
        agent_id: Option<&str>,
        namespace: Option<&str>,
        context_id: Option<&str>,
    ) -> Result<()> {
        let Some(agent_id) = agent_id else {
            return Ok(());
        };
        let context_id = context_id.map(str::to_string).or_else(|| {
            namespace.and_then(|namespace| {
                self.get_active_attachment(agent_id, namespace)
                    .ok()
                    .and_then(|attachment| attachment.context_id)
            })
        });
        let Some(context_id) = context_id else {
            return Ok(());
        };
        let now = Utc::now();
        let claims = self.list_continuity_items(&context_id, true)?;
        for claim in claims.into_iter().filter(|item| {
            item.kind == ContinuityKind::WorkClaim
                && item.author_agent_id == agent_id
                && item.status.is_open()
        }) {
            let Some(mut coordination) = work_claim_coordination(&claim) else {
                continue;
            };
            if let Some(expected_attachment) = coordination.attachment_id.as_deref() {
                if attachment_id != Some(expected_attachment) {
                    continue;
                }
            }
            coordination.renewed_at = Some(now);
            coordination.lease_expires_at =
                Some(now + chrono::Duration::seconds(coordination.lease_seconds as i64));
            let extra = merge_work_claim_extra(
                claim
                    .extra
                    .get("user")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({})),
                &coordination,
            );
            let persisted = serde_json::json!({
                "title": claim.title,
                "kind": ContinuityKind::WorkClaim.as_str(),
                "status": claim.status.as_str(),
                "context_id": claim.context_id,
                "namespace": claim.namespace,
                "task_id": claim.task_id,
                "supports": claim.supports.clone(),
                "user": extra,
            });
            self.conn.execute(
                "UPDATE continuity_items SET updated_at = ?2, extra_json = ?3 WHERE id = ?1",
                params![claim.id, now.to_rfc3339(), persisted.to_string()],
            )?;
            self.conn.execute(
                "UPDATE memory_items SET ts = ?2, extra_json = ?3 WHERE id = ?1",
                params![claim.memory_id, now.to_rfc3339(), persisted.to_string()],
            )?;
        }
        Ok(())
    }

    pub fn latest_snapshot_id(&self, context_id: &str) -> Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT id FROM context_snapshots WHERE context_id = ?1 ORDER BY ts DESC LIMIT 1",
                params![context_id],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn persist_snapshot(
        &self,
        context: &ContextRecord,
        selector: Selector,
        objective: String,
        resolution: SnapshotResolution,
        view_id: &str,
        pack_id: &str,
    ) -> Result<SnapshotRecord> {
        let id = format!("snapshot:{}", Uuid::now_v7());
        let created_at = Utc::now();
        let manifest_path = self
            .paths
            .debug_dir
            .join("snapshots")
            .join(format!("{id}.json"))
            .display()
            .to_string();
        let snapshot = SnapshotRecord {
            id: id.clone(),
            context_id: context.id.clone(),
            created_at,
            resolution,
            objective: objective.clone(),
            selector: selector.clone(),
            view_id: view_id.to_string(),
            pack_id: pack_id.to_string(),
            manifest_path: manifest_path.clone(),
        };
        let manifest = SnapshotManifest {
            snapshot: snapshot.clone(),
            context: context.clone(),
            view: self.explain_view(view_id)?,
            pack: self.explain_context_pack(pack_id)?,
            continuity: self.list_continuity_items(&context.id, true)?,
        };
        self.conn.execute(
            r#"
            INSERT INTO context_snapshots(
              id, ts, context_id, resolution, objective, selector_json, view_id, pack_id, manifest_path
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            "#,
            params![
                snapshot.id,
                snapshot.created_at.to_rfc3339(),
                snapshot.context_id,
                snapshot.resolution.to_string(),
                snapshot.objective,
                serde_json::to_string(&snapshot.selector)?,
                snapshot.view_id,
                snapshot.pack_id,
                snapshot.manifest_path,
            ],
        )?;
        self.conn.execute(
            "UPDATE contexts SET updated_at = ?2, last_snapshot_id = ?3 WHERE id = ?1",
            params![context.id, created_at.to_rfc3339(), id],
        )?;
        fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
        self.persist_snapshot_memory(context, &snapshot)?;
        debug!(
            op = "persist_snapshot",
            snapshot_id = %snapshot.id,
            context_id = %snapshot.context_id,
            namespace = %context.namespace,
            task_id = %context.task_id,
            view_id = %snapshot.view_id,
            pack_id = %snapshot.pack_id,
            resolution = %snapshot.resolution,
            manifest_path = %snapshot.manifest_path,
            "persisted continuity snapshot"
        );
        Ok(snapshot)
    }

    pub fn get_snapshot(&self, id: &str) -> Result<SnapshotRecord> {
        self.conn
            .query_row(
                r#"
                SELECT ts, context_id, resolution, objective, selector_json, view_id, pack_id, manifest_path
                FROM context_snapshots
                WHERE id = ?1
                "#,
                params![id],
                |row| {
                    Ok(SnapshotRecord {
                        id: id.to_string(),
                        created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(0)?)
                            .map_err(to_sqlite_error)?
                            .with_timezone(&Utc),
                        context_id: row.get(1)?,
                        resolution: parse_snapshot_resolution(&row.get::<_, String>(2)?)
                            .map_err(to_sqlite_anyhow)?,
                        objective: row.get(3)?,
                        selector: serde_json::from_str(&row.get::<_, String>(4)?)
                            .map_err(to_sqlite_error)?,
                        view_id: row.get(5)?,
                        pack_id: row.get(6)?,
                        manifest_path: row.get(7)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| anyhow!("unknown snapshot {id}"))
    }

    pub fn explain_snapshot(&self, id: &str) -> Result<serde_json::Value> {
        let snapshot = self.get_snapshot(id)?;
        Ok(serde_json::from_slice(&fs::read(snapshot.manifest_path)?)?)
    }

    pub fn create_subscription(&self, input: SubscriptionInput) -> Result<SubscriptionRecord> {
        let id = format!("sub:{}", Uuid::now_v7());
        let created_at = Utc::now();
        self.conn.execute(
            r#"
            INSERT INTO subscriptions(id, ts, agent_id, name, selector_json, cursor_ts, active)
            VALUES (?1, ?2, ?3, ?4, ?5, NULL, 1)
            "#,
            params![
                id,
                created_at.to_rfc3339(),
                input.agent_id,
                input.name,
                serde_json::to_string(&input.selector)?,
            ],
        )?;
        Ok(SubscriptionRecord {
            id,
            created_at,
            input,
            cursor_ts: None,
            active: true,
        })
    }

    pub fn poll_subscription(&self, id: &str, limit: usize) -> Result<SubscriptionPoll> {
        let (selector_json, cursor_ts) = self
            .conn
            .query_row(
                "SELECT selector_json, cursor_ts FROM subscriptions WHERE id = ?1 AND active = 1",
                params![id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
            )
            .optional()?
            .ok_or_else(|| anyhow!("unknown subscription {id}"))?;
        let mut selector: Selector = serde_json::from_str(&selector_json)?;
        selector.start_ts = cursor_ts
            .as_deref()
            .map(DateTime::parse_from_rfc3339)
            .transpose()?
            .map(|ts| ts.with_timezone(&Utc));
        selector.limit = Some(limit.max(1));
        let candidates = self.selector_candidates(&selector, limit.max(1))?;
        let items = candidates
            .iter()
            .map(|candidate| -> Result<ViewItem> {
                Ok(ViewItem {
                    memory_id: candidate.memory.id.clone(),
                    layer: candidate.memory.layer,
                    token_estimate: candidate.memory.token_estimate,
                    score: candidate.score,
                    matched_dimensions: candidate.matched_dimensions.clone(),
                    why: candidate.why.clone(),
                    provenance: self.provenance_for_memory(&candidate.memory)?,
                    body: candidate.memory.body.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let new_cursor_ts = candidates.first().map(|candidate| candidate.memory.ts);
        if let Some(cursor) = new_cursor_ts {
            self.conn.execute(
                "UPDATE subscriptions SET cursor_ts = ?2 WHERE id = ?1",
                params![id, cursor.to_rfc3339()],
            )?;
        }
        Ok(SubscriptionPoll {
            subscription_id: id.to_string(),
            cursor_ts: new_cursor_ts,
            items,
        })
    }

    pub fn replay_by_selector(
        &self,
        telemetry: &EngineTelemetry,
        selector: &Selector,
        limit: usize,
    ) -> Result<Vec<ReplayRow>> {
        telemetry.observe_replay();
        let ids = self.select_item_ids("event", selector, limit.max(1))?;
        let mut out = Vec::new();
        for event_id in ids {
            let mut stmt = self.conn.prepare(
                r#"
                SELECT id, ts, kind, scope, agent_id, agent_role, session_id, task_id, project_id,
                       goal_id, run_id, namespace, environment, source, tags_json, dimensions_json,
                       attributes_json, content, content_hash, byte_size, token_estimate, importance,
                       segment_seq, segment_line
                FROM events
                WHERE id = ?1
                "#,
            )?;
            let row = stmt.query_row(params![event_id], |row| {
                read_event_row(row).map_err(to_sqlite_anyhow)
            })?;
            out.push(ReplayRow { event: row });
        }
        out.sort_by(|a, b| b.event.ts.cmp(&a.event.ts));
        out.truncate(limit);
        Ok(out)
    }

    fn selector_candidates(&self, selector: &Selector, limit: usize) -> Result<Vec<ScoredMemory>> {
        let ids = self.select_item_ids("memory", selector, limit.max(1))?;
        let mut candidates = self
            .memories_by_ids(&ids)?
            .into_iter()
            .filter(|memory| selector.layers.is_empty() || selector.layers.contains(&memory.layer))
            .filter(|memory| selector.start_ts.is_none_or(|start| memory.ts >= start))
            .filter(|memory| selector.end_ts.is_none_or(|end| memory.ts <= end))
            .map(|memory| {
                let matched = self.selector_match_labels("memory", &memory.id, selector)?;
                let namespace_match = selector
                    .namespace
                    .as_deref()
                    .map(|namespace| {
                        matched
                            .iter()
                            .any(|entry| entry == &format!("namespace={namespace}"))
                    })
                    .unwrap_or(false);
                let score = matched.len() as f64 * 0.22
                    + memory.importance * 0.35
                    + memory.confidence * 0.25
                    + recency_score(Utc::now(), memory.ts) * 0.2
                    + if namespace_match { 0.15 } else { 0.0 };
                Ok(ScoredMemory {
                    memory,
                    score,
                    matched_dimensions: matched,
                    why: vec!["selector".to_string()],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(selector.limit.unwrap_or(limit).min(limit));
        Ok(candidates)
    }

    fn candidates_from_view(&self, view_id: &str) -> Result<Vec<ScoredMemory>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT m.id, m.layer, m.scope, m.agent_id, m.session_id, m.task_id, m.ts,
                   m.importance, m.confidence, m.token_estimate, m.source_event_id,
                   m.scope_key, m.body, m.extra_json, vi.score, vi.matched_dimensions_json,
                   vi.why_json
            FROM view_items vi
            JOIN memory_items m ON m.id = vi.memory_id
            WHERE vi.view_id = ?1
            ORDER BY vi.rank ASC
            "#,
        )?;
        let mut rows = stmt.query(params![view_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(ScoredMemory {
                memory: read_memory_row(row)?,
                score: row.get(14)?,
                matched_dimensions: serde_json::from_str(&row.get::<_, String>(15)?)?,
                why: serde_json::from_str(&row.get::<_, String>(16)?)?,
            });
        }
        Ok(out)
    }

    fn detect_view_conflicts(&self, candidates: &[ScoredMemory]) -> Result<Vec<ViewConflict>> {
        let mut groups: BTreeMap<String, BTreeMap<String, Vec<String>>> = BTreeMap::new();
        for candidate in candidates {
            for dimension in self.dimensions_for_item("memory", &candidate.memory.id)? {
                if !is_conflict_dimension(&dimension.key) {
                    continue;
                }
                groups
                    .entry(dimension.key)
                    .or_default()
                    .entry(dimension.value)
                    .or_default()
                    .push(candidate.memory.id.clone());
            }
        }
        Ok(groups
            .into_iter()
            .filter(|(_, values)| values.len() > 1)
            .map(|(key, values)| ViewConflict {
                key,
                values: values.keys().cloned().collect(),
                memory_ids: values.into_values().flatten().collect(),
            })
            .collect())
    }

    fn persist_view(&self, manifest: &ViewManifest) -> Result<String> {
        let manifest_path = self
            .paths
            .debug_dir
            .join("views")
            .join(format!("{}.json", manifest.id))
            .display()
            .to_string();
        self.conn.execute(
            r#"
            INSERT INTO views(
              id, ts, op, owner_agent_id, namespace, objective, selector_json,
              source_view_ids_json, resolution, item_count, conflict_count, manifest_path
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
            "#,
            params![
                manifest.id,
                manifest.created_at.to_rfc3339(),
                manifest.input.op.to_string(),
                manifest.input.owner_agent_id,
                manifest.input.namespace,
                manifest.input.objective,
                serde_json::to_string(&manifest.input.selectors)?,
                serde_json::to_string(&manifest.input.source_view_ids)?,
                manifest.input.resolution.map(|value| value.to_string()),
                manifest.item_count as i64,
                manifest.conflict_count as i64,
                manifest_path,
            ],
        )?;
        for (rank, item) in manifest.selected.iter().enumerate() {
            self.conn.execute(
                r#"
                INSERT INTO view_items(
                  view_id, memory_id, rank, score, token_estimate, matched_dimensions_json, why_json
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                "#,
                params![
                    manifest.id,
                    item.memory_id,
                    rank as i64,
                    item.score,
                    item.token_estimate as i64,
                    serde_json::to_string(&item.matched_dimensions)?,
                    serde_json::to_string(&item.why)?,
                ],
            )?;
        }
        fs::write(&manifest_path, serde_json::to_vec_pretty(manifest)?)?;
        for source_view_id in &manifest.input.source_view_ids {
            self.insert_relation(&RelationRecord {
                id: format!("rel:{}", Uuid::now_v7()),
                ts: manifest.created_at,
                source_id: source_view_id.clone(),
                target_id: manifest.id.clone(),
                relation: "view_source".to_string(),
                weight: 1.0,
                attributes: serde_json::json!({}),
            })?;
        }
        Ok(manifest_path)
    }

    fn select_item_ids(
        &self,
        item_type: &str,
        selector: &Selector,
        limit: usize,
    ) -> Result<Vec<String>> {
        let mut required_sets = Vec::new();
        for filter in &selector.all {
            required_sets.push(self.filter_item_ids(item_type, filter)?);
        }
        let had_required = !required_sets.is_empty();
        let mut any_set = HashSet::new();
        for filter in &selector.any {
            any_set.extend(self.filter_item_ids(item_type, filter)?);
        }
        let mut selected = if !required_sets.is_empty() {
            let mut intersection = required_sets.remove(0);
            for set in required_sets {
                intersection = intersection
                    .intersection(&set)
                    .cloned()
                    .collect::<HashSet<_>>();
            }
            intersection
        } else if !selector.any.is_empty() {
            any_set.clone()
        } else {
            self.default_item_ids(item_type, limit.saturating_mul(8))?
        };
        if !selector.any.is_empty() && had_required {
            selected = selected
                .intersection(&any_set)
                .cloned()
                .collect::<HashSet<_>>();
        }
        for filter in &selector.exclude {
            let excluded = self.filter_item_ids(item_type, filter)?;
            selected = selected
                .difference(&excluded)
                .cloned()
                .collect::<HashSet<_>>();
        }
        let mut ordered = selected.into_iter().collect::<Vec<_>>();
        ordered.sort();
        let mut records = match item_type {
            "memory" => self
                .memories_by_ids(&ordered)?
                .into_iter()
                .map(|memory| (memory.id.clone(), memory.ts))
                .collect::<Vec<_>>(),
            "event" => {
                let mut rows = Vec::new();
                for event_id in ordered {
                    let ts = self.conn.query_row(
                        "SELECT ts FROM events WHERE id = ?1",
                        params![event_id],
                        |row| row.get::<_, String>(0),
                    )?;
                    rows.push((
                        event_id,
                        DateTime::parse_from_rfc3339(&ts)?.with_timezone(&Utc),
                    ));
                }
                rows
            }
            other => return Err(anyhow!("unsupported selector item_type {other}")),
        };
        records.retain(|(_, ts)| selector.start_ts.is_none_or(|start| *ts >= start));
        records.retain(|(_, ts)| selector.end_ts.is_none_or(|end| *ts <= end));
        records.sort_by(|a, b| b.1.cmp(&a.1));
        Ok(records.into_iter().map(|(id, _)| id).take(limit).collect())
    }

    fn filter_item_ids(
        &self,
        item_type: &str,
        filter: &DimensionFilter,
    ) -> Result<HashSet<String>> {
        let values = if filter.values.is_empty() {
            vec!["*".to_string()]
        } else {
            filter.values.clone()
        };
        let wildcard = values.len() == 1 && values[0] == "*";
        let mut out = HashSet::new();
        let mut stmt = if wildcard {
            self.conn
                .prepare("SELECT item_id FROM item_dimensions WHERE item_type = ?1 AND key = ?2")?
        } else {
            let placeholders = (0..values.len())
                .map(|index| format!("?{}", index + 3))
                .collect::<Vec<_>>()
                .join(",");
            self.conn.prepare(&format!(
                "SELECT item_id FROM item_dimensions WHERE item_type = ?1 AND key = ?2 AND value IN ({placeholders})"
            ))?
        };
        let params = if wildcard {
            rusqlite::params_from_iter(vec![item_type.to_string(), filter.key.clone()])
        } else {
            rusqlite::params_from_iter(
                [vec![item_type.to_string(), filter.key.clone()], values].concat(),
            )
        };
        let mut rows = stmt.query(params)?;
        while let Some(row) = rows.next()? {
            out.insert(row.get(0)?);
        }
        Ok(out)
    }

    fn default_item_ids(&self, item_type: &str, limit: usize) -> Result<HashSet<String>> {
        let sql = match item_type {
            "memory" => "SELECT id FROM memory_items ORDER BY ts DESC LIMIT ?1",
            "event" => "SELECT id FROM events ORDER BY ts DESC LIMIT ?1",
            other => return Err(anyhow!("unsupported default item_type {other}")),
        };
        let mut stmt = self.conn.prepare(sql)?;
        let mut rows = stmt.query(params![limit as i64])?;
        let mut out = HashSet::new();
        while let Some(row) = rows.next()? {
            out.insert(row.get(0)?);
        }
        Ok(out)
    }

    fn selector_match_labels(
        &self,
        item_type: &str,
        item_id: &str,
        selector: &Selector,
    ) -> Result<Vec<String>> {
        let dims = self.dimensions_for_item(item_type, item_id)?;
        let mut out = Vec::new();
        for filter in selector.all.iter().chain(selector.any.iter()) {
            for dimension in dims.iter().filter(|dimension| dimension.key == filter.key) {
                if filter.values.is_empty() || filter.values.contains(&dimension.value) {
                    out.push(format!("{}={}", dimension.key, dimension.value));
                }
            }
        }
        Ok(out)
    }

    fn view_memory_ids(&self, view_id: &str) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT memory_id FROM view_items WHERE view_id = ?1 ORDER BY rank ASC")?;
        let mut rows = stmt.query(params![view_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row.get(0)?);
        }
        Ok(out)
    }

    pub(crate) fn memories_by_ids(&self, ids: &[String]) -> Result<Vec<MemoryRecord>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = (0..ids.len())
            .map(|_| "?".to_string())
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            r#"
            SELECT id, layer, scope, agent_id, session_id, task_id, ts, importance, confidence,
                   token_estimate, source_event_id, scope_key, body, extra_json
            FROM memory_items
            WHERE id IN ({placeholders})
            "#
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(rusqlite::params_from_iter(ids.iter()))?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(read_memory_row(row)?);
        }
        let mut order = HashMap::new();
        for (idx, id) in ids.iter().enumerate() {
            order.insert(id.clone(), idx);
        }
        out.sort_by_key(|memory| order.get(&memory.id).copied().unwrap_or(usize::MAX));
        Ok(out)
    }

    fn item_metadata(
        &self,
        item_type: &str,
        item_id: &str,
    ) -> Result<(DateTime<Utc>, Option<MemoryLayer>)> {
        match item_type {
            "memory" => self
                .conn
                .query_row(
                    "SELECT ts, layer FROM memory_items WHERE id = ?1",
                    params![item_id],
                    |row| {
                        let layer = match row.get::<_, i64>(1)? {
                            1 => MemoryLayer::Hot,
                            2 => MemoryLayer::Episodic,
                            3 => MemoryLayer::Semantic,
                            4 => MemoryLayer::Summary,
                            5 => MemoryLayer::Cold,
                            layer => {
                                return Err(rusqlite::Error::ToSqlConversionFailure(
                                    anyhow!("unknown memory layer {layer}").into(),
                                ));
                            }
                        };
                        Ok((
                            DateTime::parse_from_rfc3339(&row.get::<_, String>(0)?)
                                .map_err(to_sqlite_error)?
                                .with_timezone(&Utc),
                            Some(layer),
                        ))
                    },
                )
                .optional()?
                .ok_or_else(|| anyhow!("unknown memory item {item_id}")),
            "event" => self
                .conn
                .query_row(
                    "SELECT ts FROM events WHERE id = ?1",
                    params![item_id],
                    |row| {
                        Ok((
                            DateTime::parse_from_rfc3339(&row.get::<_, String>(0)?)
                                .map_err(to_sqlite_error)?
                                .with_timezone(&Utc),
                            None,
                        ))
                    },
                )
                .optional()?
                .ok_or_else(|| anyhow!("unknown event item {item_id}")),
            other => Err(anyhow!("unsupported item_type {other}")),
        }
    }

    fn index_memory_dimensions(
        &self,
        memory_id: &str,
        layer: MemoryLayer,
        event: &EventRecord,
        entities: &[SemanticEntity],
        derived_kind: &str,
    ) -> Result<()> {
        let mut dimensions = event_dimensions(event, entities);
        dimensions.push(DimensionValue {
            key: "memory_layer".to_string(),
            value: layer.to_string(),
            weight: 100,
        });
        dimensions.push(DimensionValue {
            key: "derived_kind".to_string(),
            value: derived_kind.to_string(),
            weight: 100,
        });
        self.index_item_dimensions("memory", memory_id, Some(layer), event.ts, &dimensions)
    }

    fn index_item_dimensions(
        &self,
        item_type: &str,
        item_id: &str,
        layer: Option<MemoryLayer>,
        ts: DateTime<Utc>,
        dimensions: &[DimensionValue],
    ) -> Result<()> {
        let mut dedup = BTreeMap::new();
        for dimension in dimensions {
            if dimension.key.trim().is_empty() || dimension.value.trim().is_empty() {
                continue;
            }
            dedup.insert(
                (dimension.key.clone(), dimension.value.clone()),
                dimension.weight,
            );
        }
        for ((key, value), weight) in dedup {
            self.conn.execute(
                r#"
                INSERT OR REPLACE INTO item_dimensions(item_id, item_type, ts, layer, key, value, weight)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                "#,
                params![
                    item_id,
                    item_type,
                    ts.to_rfc3339(),
                    layer.map(MemoryLayer::as_i64),
                    key,
                    value,
                    weight,
                ],
            )?;
        }
        Ok(())
    }

    fn insert_relation(&self, row: &RelationRecord) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO fabric_relations(
              id, ts, source_id, target_id, relation, weight, attributes_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            params![
                row.id,
                row.ts.to_rfc3339(),
                row.source_id,
                row.target_id,
                row.relation,
                row.weight,
                serde_json::to_string(&row.attributes)?,
            ],
        )?;
        Ok(())
    }

    fn ensure_column(&self, table: &str, column: &str, spec: &str) -> Result<()> {
        let pragma = format!("PRAGMA table_info({table})");
        let mut stmt = self.conn.prepare(&pragma)?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            if row.get::<_, String>(1)? == column {
                return Ok(());
            }
        }
        self.conn.execute(
            &format!("ALTER TABLE {table} ADD COLUMN {column} {spec}"),
            [],
        )?;
        Ok(())
    }

    pub fn replay(
        &self,
        telemetry: &EngineTelemetry,
        session_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ReplayRow>> {
        telemetry.observe_replay();
        let sql = if session_id.is_some() {
            r#"
            SELECT id, ts, kind, scope, agent_id, agent_role, session_id, task_id, project_id,
                   goal_id, run_id, namespace, environment, source, tags_json, dimensions_json,
                   attributes_json, content, content_hash, byte_size, token_estimate, importance,
                   segment_seq, segment_line
            FROM events
            WHERE session_id = ?1
            ORDER BY ts DESC
            LIMIT ?2
            "#
        } else {
            r#"
            SELECT id, ts, kind, scope, agent_id, agent_role, session_id, task_id, project_id,
                   goal_id, run_id, namespace, environment, source, tags_json, dimensions_json,
                   attributes_json, content, content_hash, byte_size, token_estimate, importance,
                   segment_seq, segment_line
            FROM events
            ORDER BY ts DESC
            LIMIT ?1
            "#
        };
        let mut stmt = self.conn.prepare(sql)?;
        let mut rows = if let Some(session_id) = session_id {
            stmt.query(params![session_id, limit as i64])?
        } else {
            stmt.query(params![limit as i64])?
        };
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(ReplayRow {
                event: read_event_row(row)?,
            });
        }
        Ok(out)
    }

    pub fn metrics_text(&self, telemetry: &EngineTelemetry) -> Result<String> {
        let mut text = telemetry.render_prometheus()?;
        self.append_metrics_text(&mut text)?;
        self.append_storage_bytes_metric_text(&mut text)?;
        Ok(text)
    }

    pub fn append_metrics_text(&self, text: &mut String) -> Result<()> {
        let machine = self.machine_profile()?;
        let persisted_events = self.count_rows("events")?;
        let persisted_memories = self.count_rows("memory_items")?;
        let persisted_hot = self.count_layer(MemoryLayer::Hot)?;
        let persisted_episodic = self.count_layer(MemoryLayer::Episodic)?;
        let lineage_edges = self.count_rows("lineage")?;
        let persisted_dimensions = self.count_rows("item_dimensions")?;
        let persisted_relations = self.count_rows("fabric_relations")?;
        let persisted_views = self.count_rows("views")?;
        let persisted_handoffs = self.count_rows("handoffs")?;
        let persisted_subscriptions = self.count_rows("subscriptions")?;
        let compiler_contexts = self.count_rows("continuity_compiler_state")?;
        let dirty_compiler_contexts = self.count_where("continuity_compiler_state", "dirty = 1")?;
        let compiler_band_totals = self.continuity_compiler_band_totals()?;
        let active_cutoff = Utc::now() - chrono::Duration::seconds(30);
        let mut active_by_namespace = BTreeMap::new();
        let mut claim_scopes = BTreeMap::<(String, String), Vec<ContinuityItemRecord>>::new();
        let now = Utc::now();
        let live_continuity = self.list_live_continuity_metric_snapshot()?;
        let agent_attachments = self.list_active_agent_attachments()?;
        let agent_badges = self.list_agent_badges_from_attachments(
            &agent_attachments,
            &live_continuity.live_work_claims,
            None,
            None,
        )?;
        let lane_projections = self.list_lane_projections_from_live_state(
            None,
            None,
            &agent_badges,
            &live_continuity.live_work_claims,
            &live_continuity.active_signals,
            now,
        )?;

        for claim in live_continuity.live_work_claims {
            claim_scopes
                .entry((claim.namespace.clone(), claim.task_id.clone()))
                .or_default()
                .push(claim);
        }

        text.push_str(&format!(
            "# HELP ice_events_persisted Persisted event rows.\n# TYPE ice_events_persisted gauge\nice_events_persisted {}\n",
            persisted_events
        ));
        text.push_str(&format!(
            "# HELP ice_memory_items_persisted Persisted memory rows.\n# TYPE ice_memory_items_persisted gauge\nice_memory_items_persisted {}\n",
            persisted_memories
        ));
        text.push_str(&format!(
            "# HELP ice_hot_items_persisted Persisted hot memory rows.\n# TYPE ice_hot_items_persisted gauge\nice_hot_items_persisted {}\n",
            persisted_hot
        ));
        text.push_str(&format!(
            "# HELP ice_episodic_items_persisted Persisted episodic memory rows.\n# TYPE ice_episodic_items_persisted gauge\nice_episodic_items_persisted {}\n",
            persisted_episodic
        ));
        text.push_str(&format!(
            "# HELP ice_lineage_edges_persisted Persisted lineage edges.\n# TYPE ice_lineage_edges_persisted gauge\nice_lineage_edges_persisted {}\n",
            lineage_edges
        ));
        text.push_str(&format!(
            "# HELP ice_item_dimensions_persisted Persisted item dimensions.\n# TYPE ice_item_dimensions_persisted gauge\nice_item_dimensions_persisted {}\n",
            persisted_dimensions
        ));
        text.push_str(&format!(
            "# HELP ice_fabric_relations_persisted Persisted fabric relations.\n# TYPE ice_fabric_relations_persisted gauge\nice_fabric_relations_persisted {}\n",
            persisted_relations
        ));
        text.push_str(&format!(
            "# HELP ice_views_persisted Persisted materialized views.\n# TYPE ice_views_persisted gauge\nice_views_persisted {}\n",
            persisted_views
        ));
        text.push_str(&format!(
            "# HELP ice_handoffs_persisted Persisted handoff manifests.\n# TYPE ice_handoffs_persisted gauge\nice_handoffs_persisted {}\n",
            persisted_handoffs
        ));
        text.push_str(&format!(
            "# HELP ice_subscriptions_persisted Persisted subscriptions.\n# TYPE ice_subscriptions_persisted gauge\nice_subscriptions_persisted {}\n",
            persisted_subscriptions
        ));
        text.push_str(
            "# HELP ice_continuity_compiler_contexts Continuity contexts tracked by the bounded continuity compiler.\n# TYPE ice_continuity_compiler_contexts gauge\n",
        );
        text.push_str(&format!(
            "ice_continuity_compiler_contexts {}\n",
            compiler_contexts
        ));
        text.push_str(
            "# HELP ice_continuity_compiler_dirty_contexts Continuity contexts waiting for on-demand recompilation.\n# TYPE ice_continuity_compiler_dirty_contexts gauge\n",
        );
        text.push_str(&format!(
            "ice_continuity_compiler_dirty_contexts {}\n",
            dirty_compiler_contexts
        ));
        text.push_str(
            "# HELP ice_continuity_compiler_chunks Compiled continuity chunks currently materialized by compiler band.\n# TYPE ice_continuity_compiler_chunks gauge\n",
        );
        text.push_str(
            "# HELP ice_continuity_compiler_items Continuity items represented inside compiled continuity chunks by band.\n# TYPE ice_continuity_compiler_items gauge\n",
        );
        for band in ["hot", "warm", "cold"] {
            let (chunks, items) = compiler_band_totals
                .get(band)
                .copied()
                .unwrap_or((0_i64, 0_i64));
            text.push_str(&format!(
                "ice_continuity_compiler_chunks{{band=\"{}\"}} {}\n",
                band, chunks
            ));
            text.push_str(&format!(
                "ice_continuity_compiler_items{{band=\"{}\"}} {}\n",
                band, items
            ));
        }
        text.push_str(
            "# HELP ice_machine_info Canonical machine identity for this continuity spine.\n# TYPE ice_machine_info gauge\n",
        );
        text.push_str(&format!(
            "ice_machine_info{{machine_id=\"{}\",label=\"{}\",namespace=\"{}\",host_name=\"{}\",os_name=\"{}\",default_task_id=\"{}\"}} 1\n",
            prometheus_label_value(&machine.machine_id),
            prometheus_label_value(&machine.label),
            prometheus_label_value(&machine.namespace),
            prometheus_label_value(&machine.host_name),
            prometheus_label_value(&machine.os_name),
            prometheus_label_value(&machine.default_task_id),
        ));
        text.push_str(
            "# HELP ice_agent_active Agent attachments seen within the freshness window.\n# TYPE ice_agent_active gauge\n",
        );
        text.push_str(
            "# HELP ice_agent_last_seen_unix_seconds Last observed activity timestamp for each active attachment.\n# TYPE ice_agent_last_seen_unix_seconds gauge\n",
        );
        text.push_str(
            "# HELP ice_agent_ticks_total Cumulative presence ticks observed for each active attachment.\n# TYPE ice_agent_ticks_total gauge\n",
        );
        for attachment in &agent_attachments {
            let freshness = if attachment.last_seen_at >= active_cutoff {
                1
            } else {
                0
            };
            if freshness == 1 {
                *active_by_namespace
                    .entry(attachment.namespace.clone())
                    .or_insert(0_i64) += 1;
            }
            let labels = format!(
                "agent_id=\"{}\",agent_type=\"{}\",namespace=\"{}\",role=\"{}\"",
                prometheus_label_value(&attachment.agent_id),
                prometheus_label_value(&attachment.agent_type),
                prometheus_label_value(&attachment.namespace),
                prometheus_label_value(attachment.role.as_deref().unwrap_or("unknown")),
            );
            text.push_str(&format!("ice_agent_active{{{labels}}} {freshness}\n"));
            text.push_str(&format!(
                "ice_agent_last_seen_unix_seconds{{{labels}}} {}\n",
                attachment.last_seen_at.timestamp()
            ));
            text.push_str(&format!(
                "ice_agent_ticks_total{{{labels}}} {}\n",
                attachment.tick_count
            ));
        }
        text.push_str(
            "# HELP ice_agent_badge_connected Live agent badge state linked to an attachment and current work.\n# TYPE ice_agent_badge_connected gauge\n",
        );
        text.push_str(
            "# HELP ice_agent_badge_last_updated_unix_seconds Last badge update timestamp for each attachment-linked agent badge.\n# TYPE ice_agent_badge_last_updated_unix_seconds gauge\n",
        );
        for badge in &agent_badges {
            let labels = format!(
                "attachment_id=\"{}\",agent_id=\"{}\",display_name=\"{}\",agent_type=\"{}\",namespace=\"{}\",task_id=\"{}\",role=\"{}\",status=\"{}\",focus=\"{}\",headline=\"{}\",resource=\"{}\",repo_root=\"{}\",branch=\"{}\"",
                prometheus_label_value(&badge.attachment_id),
                prometheus_label_value(&badge.agent_id),
                prometheus_label_value(&compact_metric_label(&badge.display_name, 64)),
                prometheus_label_value(&badge.agent_type),
                prometheus_label_value(&badge.namespace),
                prometheus_label_value(badge.task_id.as_deref().unwrap_or("")),
                prometheus_label_value(badge.role.as_deref().unwrap_or("unknown")),
                prometheus_label_value(&compact_metric_label(&badge.status, 32)),
                prometheus_label_value(&compact_metric_label(&badge.focus, 120)),
                prometheus_label_value(&compact_metric_label(&badge.headline, 120)),
                prometheus_label_value(&compact_metric_label(
                    badge.resource.as_deref().unwrap_or(""),
                    96,
                )),
                prometheus_label_value(&compact_metric_label(
                    badge.repo_root.as_deref().unwrap_or(""),
                    96,
                )),
                prometheus_label_value(&compact_metric_label(
                    badge.branch.as_deref().unwrap_or(""),
                    64,
                )),
            );
            text.push_str(&format!(
                "ice_agent_badge_connected{{{labels}}} {}\n",
                if badge.connected { 1 } else { 0 }
            ));
            text.push_str(&format!(
                "ice_agent_badge_last_updated_unix_seconds{{{labels}}} {}\n",
                badge.updated_at.timestamp()
            ));
        }
        text.push_str(
            "# HELP ice_lane_projection_agents Connected agents visible inside each derived machine lane projection.\n# TYPE ice_lane_projection_agents gauge\n",
        );
        text.push_str(
            "# HELP ice_lane_projection_claims Live work claims visible inside each derived machine lane projection.\n# TYPE ice_lane_projection_claims gauge\n",
        );
        text.push_str(
            "# HELP ice_lane_projection_conflicts Conflicting live work claims visible inside each derived machine lane projection.\n# TYPE ice_lane_projection_conflicts gauge\n",
        );
        text.push_str(
            "# HELP ice_lane_projection_coordination_signals Active coordination signals visible inside each derived machine lane projection.\n# TYPE ice_lane_projection_coordination_signals gauge\n",
        );
        text.push_str(
            "# HELP ice_lane_projection_blocking_signals Block-severity coordination signals visible inside each derived machine lane projection.\n# TYPE ice_lane_projection_blocking_signals gauge\n",
        );
        text.push_str(
            "# HELP ice_lane_projection_review_signals Review-lane coordination signals visible inside each derived machine lane projection.\n# TYPE ice_lane_projection_review_signals gauge\n",
        );
        for projection in &lane_projections {
            let labels = format!(
                "namespace=\"{}\",projection_id=\"{}\",projection_kind=\"{}\",label=\"{}\",resource=\"{}\",repo_root=\"{}\",branch=\"{}\",task_id=\"{}\"",
                prometheus_label_value(&projection.namespace),
                prometheus_label_value(&compact_metric_label(&projection.projection_id, 120)),
                prometheus_label_value(&projection.projection_kind),
                prometheus_label_value(&compact_metric_label(&projection.label, 80)),
                prometheus_label_value(&compact_metric_label(
                    projection.resource.as_deref().unwrap_or(""),
                    96,
                )),
                prometheus_label_value(&compact_metric_label(
                    projection.repo_root.as_deref().unwrap_or(""),
                    96,
                )),
                prometheus_label_value(&compact_metric_label(
                    projection.branch.as_deref().unwrap_or(""),
                    64,
                )),
                prometheus_label_value(projection.task_id.as_deref().unwrap_or("")),
            );
            text.push_str(&format!(
                "ice_lane_projection_agents{{{labels}}} {}\n",
                projection.connected_agents
            ));
            text.push_str(&format!(
                "ice_lane_projection_claims{{{labels}}} {}\n",
                projection.live_claims
            ));
            text.push_str(&format!(
                "ice_lane_projection_conflicts{{{labels}}} {}\n",
                projection.claim_conflicts
            ));
            text.push_str(&format!(
                "ice_lane_projection_coordination_signals{{{labels}}} {}\n",
                projection.coordination_signal_count
            ));
            text.push_str(&format!(
                "ice_lane_projection_blocking_signals{{{labels}}} {}\n",
                projection.blocking_signal_count
            ));
            text.push_str(&format!(
                "ice_lane_projection_review_signals{{{labels}}} {}\n",
                projection.review_signal_count
            ));
        }
        text.push_str(
            "# HELP ice_active_agents Active agents per namespace inside the freshness window.\n# TYPE ice_active_agents gauge\n",
        );
        for (namespace, count) in active_by_namespace {
            text.push_str(&format!(
                "ice_active_agents{{namespace=\"{}\"}} {}\n",
                prometheus_label_value(&namespace),
                count
            ));
        }
        text.push_str(
            "# HELP ice_work_claims_active Live work claims per namespace/task.\n# TYPE ice_work_claims_active gauge\n",
        );
        text.push_str(
            "# HELP ice_work_claim_conflicts Live conflicting work-claim resources per namespace/task.\n# TYPE ice_work_claim_conflicts gauge\n",
        );
        for ((namespace, task_id), claims) in claim_scopes {
            let mut conflict_resources = BTreeSet::new();
            for (index, left) in claims.iter().enumerate() {
                let Some(left_coordination) = work_claim_coordination(left) else {
                    continue;
                };
                for right in claims.iter().skip(index + 1) {
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
                        conflict_resources.insert(resource.clone());
                    }
                }
            }
            let labels = format!(
                "namespace=\"{}\",task_id=\"{}\"",
                prometheus_label_value(&namespace),
                prometheus_label_value(&task_id),
            );
            text.push_str(&format!(
                "ice_work_claims_active{{{labels}}} {}\n",
                claims.len()
            ));
            text.push_str(&format!(
                "ice_work_claim_conflicts{{{labels}}} {}\n",
                conflict_resources.len()
            ));
        }
        Ok(())
    }

    pub fn append_storage_bytes_metric_text(&self, text: &mut String) -> Result<()> {
        append_storage_bytes_metric_text(text, &self.paths.sqlite_path, &self.paths.log_dir)
    }

    fn insert_event(&self, event: &EventRecord) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO events (
              id, ts, kind, scope, agent_id, agent_role, session_id, task_id, project_id, goal_id,
              run_id, namespace, environment, source, tags_json, dimensions_json, attributes_json,
              content, content_hash, byte_size, token_estimate, importance, segment_seq, segment_line
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24)
            "#,
            params![
                event.id,
                event.ts.to_rfc3339(),
                event.input.kind.to_string(),
                event.input.scope.to_string(),
                event.input.agent_id,
                event.input.agent_role,
                event.input.session_id,
                event.input.task_id,
                event.input.project_id,
                event.input.goal_id,
                event.input.run_id,
                event.input.namespace,
                event.input.environment,
                event.input.source,
                serde_json::to_string(&event.input.tags)?,
                serde_json::to_string(&event.input.dimensions)?,
                serde_json::to_string(&event.input.attributes)?,
                event.input.content,
                event.content_hash,
                event.byte_size as i64,
                event.token_estimate as i64,
                event.importance,
                event.segment_seq,
                event.segment_line,
            ],
        )?;
        Ok(())
    }

    fn promote_hot(&self, event: &EventRecord) -> Result<String> {
        let id = format!("hot:{}", event.id);
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO memory_items (
              id, layer, scope, agent_id, session_id, task_id, ts, importance, confidence,
              token_estimate, source_event_id, scope_key, body, extra_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
            "#,
            params![
                id,
                MemoryLayer::Hot.as_i64(),
                event.input.scope.to_string(),
                event.input.agent_id,
                event.input.session_id,
                event.input.task_id,
                event.ts.to_rfc3339(),
                event.importance,
                1.0_f64,
                event.token_estimate as i64,
                event.id,
                event.id,
                hot_body(event),
                serde_json::json!({
                    "kind": event.input.kind,
                    "source": event.input.source,
                    "namespace": event.input.namespace,
                    "project_id": event.input.project_id,
                    "goal_id": event.input.goal_id,
                    "run_id": event.input.run_id,
                    "agent_role": event.input.agent_role,
                    "segment": { "seq": event.segment_seq, "line": event.segment_line },
                    "hash": event.content_hash,
                    "tags": event.input.tags,
                })
                .to_string(),
            ],
        )?;
        self.refresh_fts_entry(&id, &hot_body(event))?;
        Ok(format!("hot:{}", event.id))
    }

    fn promote_episode(&self, event: &EventRecord) -> Result<String> {
        let slot = event.ts.timestamp() / 900;
        let scope_key = format!(
            "{}:{}:{}:{}",
            event.input.agent_id,
            event.input.session_id,
            event.input.task_id.as_deref().unwrap_or("_"),
            slot
        );
        let id = format!("episode:{}", scope_key);
        let existing = self
            .conn
            .query_row(
                r#"
                SELECT body, token_estimate, importance
                FROM memory_items
                WHERE layer = ?1 AND scope_key = ?2
                "#,
                params![MemoryLayer::Episodic.as_i64(), scope_key],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, f64>(2)?,
                    ))
                },
            )
            .optional()?;
        let merged_body = match &existing {
            Some((body, _, _)) => merge_episode_body(
                body,
                &episode_fragment(event),
                self.config.episode_max_chars,
            ),
            None => episode_fragment(event),
        };
        let token_estimate = estimate_tokens(&merged_body);
        let importance = existing
            .as_ref()
            .map(|(_, _, importance)| importance.max(event.importance))
            .unwrap_or(event.importance);
        let extra_json = serde_json::json!({
            "slot": slot,
            "source_event_id": event.id,
            "kind": "rolling_episode",
            "namespace": event.input.namespace,
            "project_id": event.input.project_id,
            "goal_id": event.input.goal_id,
            "run_id": event.input.run_id,
        })
        .to_string();
        if existing.is_some() {
            self.conn.execute(
                r#"
                UPDATE memory_items
                SET ts = ?2,
                    importance = ?3,
                    confidence = ?4,
                    token_estimate = ?5,
                    source_event_id = ?6,
                    body = ?7,
                    extra_json = ?8
                WHERE id = ?1
                "#,
                params![
                    id,
                    event.ts.to_rfc3339(),
                    importance,
                    0.85_f64,
                    token_estimate as i64,
                    event.id,
                    merged_body,
                    extra_json,
                ],
            )?;
        } else {
            self.conn.execute(
                r#"
                INSERT INTO memory_items (
                  id, layer, scope, agent_id, session_id, task_id, ts, importance, confidence,
                  token_estimate, source_event_id, scope_key, body, extra_json
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
                "#,
                params![
                    id,
                    MemoryLayer::Episodic.as_i64(),
                    event.input.scope.to_string(),
                    event.input.agent_id,
                    event.input.session_id,
                    event.input.task_id,
                    event.ts.to_rfc3339(),
                    importance,
                    0.85_f64,
                    token_estimate as i64,
                    event.id,
                    scope_key,
                    merged_body,
                    extra_json,
                ],
            )?;
        }
        self.refresh_fts_entry(&id, &merged_body)?;
        Ok(format!(
            "episode:{}:{}:{}:{}",
            event.input.agent_id,
            event.input.session_id,
            event.input.task_id.as_deref().unwrap_or("_"),
            slot
        ))
    }

    fn promote_semantic(
        &self,
        event: &EventRecord,
        entities: &[SemanticEntity],
    ) -> Result<Option<String>> {
        if entities.is_empty()
            && !matches!(
                event.input.kind,
                crate::model::EventKind::Error | crate::model::EventKind::Exception
            )
        {
            return Ok(None);
        }

        let body = semantic_body(event, entities);
        let id = format!("semantic:{}", event.id);
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO memory_items (
              id, layer, scope, agent_id, session_id, task_id, ts, importance, confidence,
              token_estimate, source_event_id, scope_key, body, extra_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
            "#,
            params![
                id,
                MemoryLayer::Semantic.as_i64(),
                event.input.scope.to_string(),
                event.input.agent_id,
                event.input.session_id,
                event.input.task_id,
                event.ts.to_rfc3339(),
                (event.importance + 0.1).min(1.0),
                0.75_f64,
                estimate_tokens(&body) as i64,
                event.id,
                event.id,
                body,
                serde_json::json!({
                    "entity_count": entities.len(),
                    "entities": entities,
                    "namespace": event.input.namespace,
                    "project_id": event.input.project_id,
                    "goal_id": event.input.goal_id,
                    "run_id": event.input.run_id,
                })
                .to_string(),
            ],
        )?;
        self.refresh_fts_entry(&id, &semantic_body(event, entities))?;
        self.refresh_vector(&id, &semantic_body(event, entities))?;
        self.upsert_entity_mentions(&id, &event.id, event.ts, entities)?;
        Ok(Some(format!("semantic:{}", event.id)))
    }

    fn promote_summary(
        &self,
        event: &EventRecord,
        entities: &[SemanticEntity],
    ) -> Result<Option<String>> {
        let slot = event.ts.timestamp() / 3600;
        let scope_key = format!(
            "{}:{}:{}:{}",
            event.input.agent_id,
            event.input.session_id,
            event.input.task_id.as_deref().unwrap_or("_"),
            slot
        );
        let id = format!("summary:{}", scope_key);
        let fragment = summary_fragment(event, entities);
        let existing = self
            .conn
            .query_row(
                r#"
                SELECT body, importance
                FROM memory_items
                WHERE layer = ?1 AND scope_key = ?2
                "#,
                params![MemoryLayer::Summary.as_i64(), scope_key],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?)),
            )
            .optional()?;
        let body = match &existing {
            Some((current, _)) => {
                merge_episode_body(current, &fragment, self.config.episode_max_chars)
            }
            None => fragment,
        };
        let importance = existing
            .as_ref()
            .map(|(_, existing_importance)| existing_importance.max(event.importance))
            .unwrap_or(event.importance);
        if existing.is_some() {
            self.conn.execute(
                r#"
                UPDATE memory_items
                SET ts = ?2,
                    importance = ?3,
                    confidence = ?4,
                    token_estimate = ?5,
                    source_event_id = ?6,
                    body = ?7,
                    extra_json = ?8
                WHERE id = ?1
                "#,
                params![
                    id,
                    event.ts.to_rfc3339(),
                    importance,
                    0.65_f64,
                    estimate_tokens(&body) as i64,
                    event.id,
                    body,
                    serde_json::json!({
                        "slot": slot,
                        "kind": "rolling_summary",
                        "namespace": event.input.namespace,
                        "project_id": event.input.project_id,
                        "goal_id": event.input.goal_id,
                        "run_id": event.input.run_id,
                    })
                    .to_string(),
                ],
            )?;
        } else {
            self.conn.execute(
                r#"
                INSERT INTO memory_items (
                  id, layer, scope, agent_id, session_id, task_id, ts, importance, confidence,
                  token_estimate, source_event_id, scope_key, body, extra_json
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
                "#,
                params![
                    id,
                    MemoryLayer::Summary.as_i64(),
                    event.input.scope.to_string(),
                    event.input.agent_id,
                    event.input.session_id,
                    event.input.task_id,
                    event.ts.to_rfc3339(),
                    importance,
                    0.65_f64,
                    estimate_tokens(&body) as i64,
                    event.id,
                    scope_key,
                    body,
                    serde_json::json!({
                        "slot": slot,
                        "kind": "rolling_summary",
                        "namespace": event.input.namespace,
                        "project_id": event.input.project_id,
                        "goal_id": event.input.goal_id,
                        "run_id": event.input.run_id,
                    })
                    .to_string(),
                ],
            )?;
        }
        self.refresh_fts_entry(&id, &summary_fragment(event, entities))?;
        self.refresh_vector(&id, &body)?;
        Ok(Some(format!("summary:{}", scope_key)))
    }

    fn refresh_layer_counts(&self, telemetry: &EngineTelemetry) -> Result<()> {
        let hot_count = self.count_layer(MemoryLayer::Hot)?;
        let episodic_count = self.count_layer(MemoryLayer::Episodic)?;
        telemetry.set_hot_items(hot_count);
        telemetry.set_episodic_items(episodic_count);
        Ok(())
    }

    fn count_layer(&self, layer: MemoryLayer) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM memory_items WHERE layer = ?1",
            params![layer.as_i64()],
            |row| row.get(0),
        )?)
    }

    fn count_rows(&self, table: &str) -> Result<i64> {
        let sql = format!("SELECT COUNT(*) FROM {table}");
        Ok(self.conn.query_row(&sql, [], |row| row.get(0))?)
    }

    fn count_where(&self, table: &str, predicate_sql: &str) -> Result<i64> {
        let sql = format!("SELECT COUNT(*) FROM {table} WHERE {predicate_sql}");
        Ok(self.conn.query_row(&sql, [], |row| row.get(0))?)
    }

    pub fn list_agent_badges(
        &self,
        namespace: Option<&str>,
        task_id: Option<&str>,
    ) -> Result<Vec<AgentBadgeRecord>> {
        let namespace = self.resolve_namespace_alias(namespace)?;
        let attachments = self.list_active_agent_attachments()?;
        let live_work_claims = self.list_live_work_claims()?;
        self.list_agent_badges_from_attachments(
            &attachments,
            &live_work_claims,
            namespace.as_deref(),
            task_id,
        )
    }

    fn list_agent_badges_from_attachments(
        &self,
        attachments: &[ActiveAttachmentMetric],
        live_work_claims: &[ContinuityItemRecord],
        namespace: Option<&str>,
        task_id: Option<&str>,
    ) -> Result<Vec<AgentBadgeRecord>> {
        let active_cutoff = Utc::now() - chrono::Duration::seconds(30);
        let mut badges = Vec::new();
        for attachment in attachments {
            if let Some(namespace) = namespace {
                if attachment.namespace != namespace {
                    continue;
                }
            }
            let stored = self.get_stored_agent_badge(&attachment.attachment_id)?;
            let derived = self.derive_agent_badge_from_metric(&attachment, live_work_claims)?;
            let task = stored
                .as_ref()
                .and_then(|badge| badge.task_id.clone())
                .or_else(|| derived.task_id.clone());
            if let Some(filter_task) = task_id {
                if task.as_deref() != Some(filter_task) {
                    continue;
                }
            }
            let updated_at = stored
                .as_ref()
                .map(|badge| badge.updated_at)
                .unwrap_or(attachment.last_seen_at);
            let display_name = stored
                .as_ref()
                .map(|badge| badge.display_name.clone())
                .unwrap_or_else(|| derived.display_name.clone());
            let status = stored
                .as_ref()
                .map(|badge| badge.status.clone())
                .unwrap_or_else(|| derived.status.clone());
            let focus = stored
                .as_ref()
                .map(|badge| badge.focus.clone())
                .unwrap_or_else(|| derived.focus.clone());
            let headline = stored
                .as_ref()
                .map(|badge| badge.headline.clone())
                .unwrap_or_else(|| derived.headline.clone());
            let resource = stored
                .as_ref()
                .and_then(|badge| badge.resource.clone())
                .or_else(|| derived.resource.clone());
            let repo_root = stored
                .as_ref()
                .and_then(|badge| badge.repo_root.clone())
                .or_else(|| derived.repo_root.clone());
            let branch = stored
                .as_ref()
                .and_then(|badge| badge.branch.clone())
                .or_else(|| derived.branch.clone());
            let metadata = stored
                .as_ref()
                .map(|badge| badge.metadata.clone())
                .unwrap_or_else(|| derived.metadata.clone());
            let context_id = stored
                .as_ref()
                .and_then(|badge| badge.context_id.clone())
                .or_else(|| derived.context_id.clone());
            badges.push(AgentBadgeRecord {
                attachment_id: attachment.attachment_id.clone(),
                agent_id: attachment.agent_id.clone(),
                agent_type: attachment.agent_type.clone(),
                namespace: attachment.namespace.clone(),
                task_id: task,
                context_id,
                role: attachment.role.clone(),
                display_name,
                status,
                focus,
                headline,
                resource,
                repo_root,
                branch,
                metadata,
                updated_at,
                last_seen_at: attachment.last_seen_at,
                tick_count: attachment.tick_count as usize,
                connected: attachment.last_seen_at >= active_cutoff,
            });
        }
        badges.sort_by(|left, right| {
            right
                .connected
                .cmp(&left.connected)
                .then_with(|| right.last_seen_at.cmp(&left.last_seen_at))
                .then_with(|| left.agent_id.cmp(&right.agent_id))
        });
        Ok(badges)
    }

    pub fn list_lane_projections(
        &self,
        namespace: Option<&str>,
        task_id: Option<&str>,
    ) -> Result<Vec<LaneProjectionRecord>> {
        let namespace = self.resolve_namespace_alias(namespace)?;
        let now = Utc::now();
        let attachments = self.list_active_agent_attachments()?;
        let live_work_claims = self.list_live_work_claims()?;
        let badges = self.list_agent_badges_from_attachments(
            &attachments,
            &live_work_claims,
            namespace.as_deref(),
            None,
        )?;
        let active_signals = self.list_active_coordination_signals()?;
        self.list_lane_projections_from_live_state(
            namespace.as_deref(),
            task_id,
            &badges,
            &live_work_claims,
            &active_signals,
            now,
        )
    }

    fn list_lane_projections_from_live_state(
        &self,
        namespace: Option<&str>,
        task_id: Option<&str>,
        badges: &[AgentBadgeRecord],
        live_work_claims: &[ContinuityItemRecord],
        active_signals: &[ContinuityItemRecord],
        now: DateTime<Utc>,
    ) -> Result<Vec<LaneProjectionRecord>> {
        let mut projections: BTreeMap<String, LaneProjectionAccumulator> = BTreeMap::new();
        let mut projection_identity_by_id: BTreeMap<String, LaneProjectionIdentity> =
            BTreeMap::new();
        let mut repo_projection_by_attachment: BTreeMap<String, LaneProjectionIdentity> =
            BTreeMap::new();
        let mut repo_projection_hints: BTreeMap<
            (String, String, String),
            BTreeSet<(String, String)>,
        > = BTreeMap::new();

        for badge in badges.iter().filter(|badge| badge.connected) {
            let Some(identity) = self.lane_projection_from_badge(&badge) else {
                continue;
            };
            if let Some(filter_task) = task_id {
                if identity.task_id.as_deref() != Some(filter_task) {
                    continue;
                }
            }
            projection_identity_by_id
                .entry(identity.projection_id.clone())
                .or_insert_with(|| identity.clone());
            let entry = projections
                .entry(identity.projection_id.clone())
                .or_insert_with(|| {
                    let mut accumulator = LaneProjectionAccumulator::new(identity.clone());
                    accumulator.updated_at = badge.updated_at;
                    accumulator
                });
            entry.absorb_badge(&badge);
            if identity.projection_kind == "repo" {
                repo_projection_by_attachment.insert(badge.attachment_id.clone(), identity.clone());
                if let (Some(resource), Some(repo_root)) =
                    (identity.resource.clone(), identity.repo_root.clone())
                {
                    repo_projection_hints
                        .entry((
                            identity.namespace.clone(),
                            identity.task_id.clone().unwrap_or_default(),
                            resource,
                        ))
                        .or_default()
                        .insert((repo_root, identity.branch.clone().unwrap_or_default()));
                }
            }
        }

        let mut projected_claims = Vec::<(String, ContinuityItemRecord)>::new();
        for claim in live_work_claims.iter().cloned().filter(|claim| {
            namespace
                .map(|filter_namespace| claim.namespace == filter_namespace)
                .unwrap_or(true)
        }) {
            let coordination = work_claim_coordination(&claim);
            let repo_projection_hint = coordination.as_ref().and_then(|coordination| {
                coordination
                    .resources
                    .iter()
                    .find(|resource| !resource.trim().is_empty())
                    .and_then(|resource| {
                        repo_projection_hints
                            .get(&(
                                claim.namespace.clone(),
                                claim.task_id.clone(),
                                resource.clone(),
                            ))
                            .filter(|hints| hints.len() == 1)
                            .and_then(|hints| hints.iter().next().cloned())
                    })
            });
            let attachment_projection = coordination
                .as_ref()
                .and_then(|coordination| coordination.attachment_id.as_deref())
                .and_then(|attachment_id| repo_projection_by_attachment.get(attachment_id));
            let Some(identity) = self.lane_projection_from_claim(
                &claim,
                attachment_projection,
                repo_projection_hint
                    .as_ref()
                    .map(|(repo_root, _)| repo_root.as_str()),
                repo_projection_hint
                    .as_ref()
                    .map(|(_, branch)| branch.as_str())
                    .filter(|branch| !branch.is_empty()),
            ) else {
                continue;
            };
            if let Some(filter_task) = task_id {
                if identity.task_id.as_deref() != Some(filter_task) {
                    continue;
                }
            }
            projection_identity_by_id
                .entry(identity.projection_id.clone())
                .or_insert_with(|| identity.clone());
            let entry = projections
                .entry(identity.projection_id.clone())
                .or_insert_with(|| {
                    let mut accumulator = LaneProjectionAccumulator::new(identity.clone());
                    accumulator.updated_at = claim.updated_at;
                    accumulator
                });
            entry.absorb_claim(&claim);
            projected_claims.push((identity.projection_id.clone(), claim));
        }

        let mut projection_conflicts = BTreeMap::<String, usize>::new();
        for (left_index, (left_projection_id, left_claim)) in projected_claims.iter().enumerate() {
            for (right_projection_id, right_claim) in projected_claims.iter().skip(left_index + 1) {
                if !work_claims_conflict(left_claim, right_claim, now) {
                    continue;
                }
                if left_projection_id != right_projection_id
                    && Self::claim_is_repo_scoped(left_claim)
                    && Self::claim_is_repo_scoped(right_claim)
                {
                    continue;
                }
                if left_projection_id == right_projection_id {
                    *projection_conflicts
                        .entry(left_projection_id.clone())
                        .or_default() += 1;
                    continue;
                }
                *projection_conflicts
                    .entry(left_projection_id.clone())
                    .or_default() += 1;
                *projection_conflicts
                    .entry(right_projection_id.clone())
                    .or_default() += 1;
            }
        }

        for (projection_id, conflicts) in projection_conflicts {
            if let Some(entry) = projections.get_mut(&projection_id) {
                entry.claim_conflicts = conflicts;
            }
        }

        for signal_item in active_signals.iter().cloned().filter(|item| {
            namespace
                .map(|filter_namespace| item.namespace == filter_namespace)
                .unwrap_or(true)
        }) {
            let Some(signal) = coordination_signal(&signal_item) else {
                continue;
            };
            for identity in self.lane_projection_identities_from_signal(
                &signal_item,
                &signal,
                &projection_identity_by_id,
                &repo_projection_hints,
            ) {
                if let Some(filter_task) = task_id {
                    if identity.task_id.as_deref() != Some(filter_task) {
                        continue;
                    }
                }
                projection_identity_by_id
                    .entry(identity.projection_id.clone())
                    .or_insert_with(|| identity.clone());
                let entry = projections
                    .entry(identity.projection_id.clone())
                    .or_insert_with(|| {
                        let mut accumulator = LaneProjectionAccumulator::new(identity.clone());
                        accumulator.updated_at = signal_item.updated_at;
                        accumulator
                    });
                entry.absorb_signal(&signal_item, &signal.lane);
            }
        }

        let mut records = projections
            .into_values()
            .map(LaneProjectionAccumulator::finalize)
            .collect::<Vec<_>>();
        records.sort_by(|left, right| {
            right
                .updated_at
                .cmp(&left.updated_at)
                .then_with(|| left.projection_kind.cmp(&right.projection_kind))
                .then_with(|| left.label.cmp(&right.label))
        });
        Ok(records)
    }

    fn lane_projection_from_badge(
        &self,
        badge: &AgentBadgeRecord,
    ) -> Option<LaneProjectionIdentity> {
        let resource = badge.resource.clone();
        let task_id = badge.task_id.clone();
        let projection_kind = Self::lane_projection_kind(resource.as_deref(), task_id.as_deref())?;
        let repo_root = Self::normalized_projection_text(badge.repo_root.clone());
        let branch = Self::normalized_projection_text(
            badge
                .branch
                .clone()
                .or_else(|| Self::branch_from_repo_resource(resource.as_deref())),
        );
        let projection_id = Self::lane_projection_id(
            &projection_kind,
            resource.as_deref(),
            repo_root.as_deref(),
            branch.as_deref(),
            task_id.as_deref(),
        )?;
        Some(LaneProjectionIdentity {
            namespace: badge.namespace.clone(),
            label: Self::lane_projection_label(
                &projection_kind,
                resource.as_deref(),
                repo_root.as_deref(),
                branch.as_deref(),
                task_id.as_deref(),
            ),
            projection_id,
            projection_kind,
            resource,
            repo_root,
            branch,
            task_id,
        })
    }

    fn lane_projection_from_claim(
        &self,
        claim: &ContinuityItemRecord,
        attachment_projection: Option<&LaneProjectionIdentity>,
        hinted_repo_root: Option<&str>,
        hinted_branch: Option<&str>,
    ) -> Option<LaneProjectionIdentity> {
        let coordination = work_claim_coordination(claim)?;
        let resource = coordination
            .resources
            .iter()
            .find(|resource| !resource.trim().is_empty())
            .cloned();
        if let Some(identity) = attachment_projection.filter(|_| {
            !matches!(
                resource.as_deref(),
                Some(resource) if resource.starts_with("repo/") || resource.starts_with("organism/")
            )
        }) {
            return Some(identity.clone());
        }
        let task_id = Some(claim.task_id.clone());
        let projection_kind = Self::lane_projection_kind(resource.as_deref(), task_id.as_deref())?;
        let (repo_root, branch) = self.repo_projection_metadata_from_claim(
            claim,
            &coordination,
            resource.as_deref(),
            hinted_repo_root,
            hinted_branch,
        );
        let projection_id = Self::lane_projection_id(
            &projection_kind,
            resource.as_deref(),
            repo_root.as_deref(),
            branch.as_deref(),
            task_id.as_deref(),
        )?;
        Some(LaneProjectionIdentity {
            namespace: claim.namespace.clone(),
            label: Self::lane_projection_label(
                &projection_kind,
                resource.as_deref(),
                repo_root.as_deref(),
                branch.as_deref(),
                task_id.as_deref(),
            ),
            projection_id,
            projection_kind,
            resource,
            repo_root,
            branch,
            task_id,
        })
    }

    fn repo_projection_metadata_from_claim(
        &self,
        claim: &ContinuityItemRecord,
        coordination: &WorkClaimCoordination,
        resource: Option<&str>,
        hinted_repo_root: Option<&str>,
        hinted_branch: Option<&str>,
    ) -> (Option<String>, Option<String>) {
        let (attachment_repo_root, attachment_branch) = coordination
            .attachment_id
            .as_deref()
            .map(|attachment_id| self.repo_projection_metadata_from_attachment(attachment_id))
            .unwrap_or((None, None));
        let (extra_repo_root, extra_branch) =
            Self::repo_projection_metadata_from_extra(&claim.extra);
        (
            Self::normalized_projection_text(
                attachment_repo_root
                    .or(extra_repo_root)
                    .or_else(|| hinted_repo_root.map(ToString::to_string)),
            ),
            Self::normalized_projection_text(
                attachment_branch
                    .or(extra_branch)
                    .or_else(|| hinted_branch.map(ToString::to_string))
                    .or_else(|| Self::branch_from_repo_resource(resource)),
            ),
        )
    }

    fn repo_projection_metadata_from_attachment(
        &self,
        attachment_id: &str,
    ) -> (Option<String>, Option<String>) {
        let attachment = self.get_attachment(attachment_id).ok();
        let stored_badge = self.get_stored_agent_badge(attachment_id).ok().flatten();
        let repo_root = attachment
            .as_ref()
            .and_then(|attachment| Self::metadata_string(&attachment.input.metadata, "repo_root"))
            .or_else(|| {
                stored_badge
                    .as_ref()
                    .and_then(|badge| badge.repo_root.clone())
            });
        let branch = attachment
            .as_ref()
            .and_then(|attachment| Self::metadata_string(&attachment.input.metadata, "branch"))
            .or_else(|| stored_badge.as_ref().and_then(|badge| badge.branch.clone()));
        (
            Self::normalized_projection_text(repo_root),
            Self::normalized_projection_text(branch),
        )
    }

    fn repo_projection_metadata_from_extra(
        extra: &serde_json::Value,
    ) -> (Option<String>, Option<String>) {
        let user = extra.get("user").unwrap_or(extra);
        (
            Self::metadata_string(user, "repo_root"),
            Self::metadata_string(user, "branch"),
        )
    }

    fn lane_projection_from_signal(
        &self,
        signal: &ContinuityItemRecord,
        resource: Option<&str>,
        hinted_repo_root: Option<&str>,
        hinted_branch: Option<&str>,
    ) -> Option<LaneProjectionIdentity> {
        let task_id = Some(signal.task_id.clone());
        let projection_kind = Self::lane_projection_kind(resource, task_id.as_deref())?;
        let repo_root = Self::normalized_projection_text(hinted_repo_root.map(ToString::to_string));
        let branch = Self::normalized_projection_text(
            hinted_branch
                .map(ToString::to_string)
                .or_else(|| Self::branch_from_repo_resource(resource)),
        );
        let projection_id = Self::lane_projection_id(
            &projection_kind,
            resource,
            repo_root.as_deref(),
            branch.as_deref(),
            task_id.as_deref(),
        )?;
        Some(LaneProjectionIdentity {
            namespace: signal.namespace.clone(),
            label: Self::lane_projection_label(
                &projection_kind,
                resource,
                repo_root.as_deref(),
                branch.as_deref(),
                task_id.as_deref(),
            ),
            projection_id,
            projection_kind,
            resource: resource.map(ToString::to_string),
            repo_root,
            branch,
            task_id,
        })
    }

    fn lane_projection_identity_from_projected_lane(
        namespace: &str,
        task_id: &str,
        lane: &crate::continuity::CoordinationProjectedLane,
    ) -> LaneProjectionIdentity {
        LaneProjectionIdentity {
            namespace: namespace.to_string(),
            projection_id: lane.projection_id.clone(),
            projection_kind: lane.projection_kind.clone(),
            label: lane.label.clone(),
            resource: lane.resource.clone(),
            repo_root: lane.repo_root.clone(),
            branch: lane.branch.clone(),
            task_id: lane.task_id.clone().or_else(|| Some(task_id.to_string())),
        }
    }

    fn lane_projection_identities_from_signal(
        &self,
        signal_item: &ContinuityItemRecord,
        signal: &crate::continuity::CoordinationSignalRecord,
        known_projections: &BTreeMap<String, LaneProjectionIdentity>,
        repo_projection_hints: &BTreeMap<(String, String, String), BTreeSet<(String, String)>>,
    ) -> Vec<LaneProjectionIdentity> {
        let mut identities = BTreeMap::<String, LaneProjectionIdentity>::new();

        if let Some(target_lane) = signal.target_projected_lane.as_ref() {
            let identity = Self::lane_projection_identity_from_projected_lane(
                &signal_item.namespace,
                &signal_item.task_id,
                target_lane,
            );
            identities.insert(identity.projection_id.clone(), identity);
        }

        if identities.is_empty() {
            for lane in &signal.projected_lanes {
                let identity = Self::lane_projection_identity_from_projected_lane(
                    &signal_item.namespace,
                    &signal_item.task_id,
                    lane,
                );
                identities.insert(identity.projection_id.clone(), identity);
            }
        }

        if identities.is_empty() {
            for projection_id in &signal.projection_ids {
                if let Some(identity) = known_projections.get(projection_id) {
                    identities
                        .entry(identity.projection_id.clone())
                        .or_insert_with(|| identity.clone());
                }
            }
        }

        if identities.is_empty() {
            let repo_projection_hint = signal.resource.as_ref().and_then(|resource| {
                repo_projection_hints
                    .get(&(
                        signal_item.namespace.clone(),
                        signal_item.task_id.clone(),
                        resource.clone(),
                    ))
                    .filter(|hints| hints.len() == 1)
                    .and_then(|hints| hints.iter().next().cloned())
            });
            if let Some(identity) = self.lane_projection_from_signal(
                signal_item,
                signal.resource.as_deref(),
                repo_projection_hint
                    .as_ref()
                    .map(|(repo_root, _)| repo_root.as_str()),
                repo_projection_hint
                    .as_ref()
                    .map(|(_, branch)| branch.as_str())
                    .filter(|branch| !branch.is_empty()),
            ) {
                identities.insert(identity.projection_id.clone(), identity);
            }
        }

        identities.into_values().collect()
    }

    fn lane_projection_kind(resource: Option<&str>, task_id: Option<&str>) -> Option<String> {
        if let Some(resource) = resource {
            if resource.starts_with("repo/") {
                return Some("repo".to_string());
            }
            if resource.starts_with("organism/") {
                return Some("organism".to_string());
            }
            return Some("lane".to_string());
        }
        if task_id.is_some_and(|task_id| task_id != DEFAULT_MACHINE_TASK_ID) {
            return Some("task".to_string());
        }
        None
    }

    fn lane_projection_id(
        projection_kind: &str,
        resource: Option<&str>,
        repo_root: Option<&str>,
        branch: Option<&str>,
        task_id: Option<&str>,
    ) -> Option<String> {
        if projection_kind == "repo" {
            if let Some(repo_root) = repo_root {
                return Some(format!(
                    "repo:{}:{}",
                    repo_root,
                    branch
                        .map(ToString::to_string)
                        .or_else(|| Self::branch_from_repo_resource(resource))
                        .unwrap_or_default()
                ));
            }
        }
        if let Some(resource) = resource {
            return Some(format!("{projection_kind}:{resource}"));
        }
        if projection_kind == "task" {
            return task_id.map(|task_id| format!("task:{task_id}"));
        }
        if let Some(repo_root) = repo_root {
            return Some(format!("repo:{}:{}", repo_root, branch.unwrap_or_default()));
        }
        None
    }

    fn lane_projection_label(
        projection_kind: &str,
        resource: Option<&str>,
        repo_root: Option<&str>,
        branch: Option<&str>,
        task_id: Option<&str>,
    ) -> String {
        match projection_kind {
            "repo" => {
                let repo_name = repo_root
                    .and_then(Self::repo_root_leaf)
                    .or_else(|| Self::repo_name_from_resource(resource))
                    .unwrap_or_else(|| "repo".to_string());
                match branch
                    .map(ToString::to_string)
                    .or_else(|| Self::branch_from_repo_resource(resource))
                {
                    Some(branch) if !branch.is_empty() => format!("{repo_name} @ {branch}"),
                    _ => repo_name,
                }
            }
            "organism" => resource
                .and_then(|resource| resource.rsplit('/').next())
                .map(ToString::to_string)
                .unwrap_or_else(|| "organism".to_string()),
            "task" => task_id
                .map(ToString::to_string)
                .unwrap_or_else(|| "task".to_string()),
            _ => resource
                .map(ToString::to_string)
                .unwrap_or_else(|| "lane".to_string()),
        }
    }

    fn repo_root_leaf(repo_root: &str) -> Option<String> {
        PathBuf::from(repo_root)
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .filter(|name| !name.is_empty())
    }

    fn repo_name_from_resource(resource: Option<&str>) -> Option<String> {
        let rest = resource?.strip_prefix("repo/")?;
        rest.split('/').next().map(ToString::to_string)
    }

    fn branch_from_repo_resource(resource: Option<&str>) -> Option<String> {
        let rest = resource?.strip_prefix("repo/")?;
        let (_, branch) = rest.split_once('/')?;
        Some(branch.to_string())
    }

    fn metadata_string(metadata: &serde_json::Value, key: &str) -> Option<String> {
        metadata
            .get(key)
            .and_then(|value| value.as_str())
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    }

    fn claim_is_repo_scoped(claim: &ContinuityItemRecord) -> bool {
        work_claim_coordination(claim)
            .and_then(|coordination| {
                coordination
                    .resources
                    .into_iter()
                    .find(|resource| !resource.trim().is_empty())
            })
            .is_some_and(|resource| resource.starts_with("repo/"))
    }

    fn normalized_projection_text(value: Option<String>) -> Option<String> {
        value
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    }

    fn list_active_agent_attachments(&self) -> Result<Vec<ActiveAttachmentMetric>> {
        #[cfg(test)]
        {
            if let Some(hook) = METRICS_LIVE_STATE_QUERY_COUNTER_HOOK
                .lock()
                .unwrap()
                .as_mut()
            {
                if hook.root == self.paths.root {
                    hook.counts.active_attachment_reads += 1;
                }
            }
        }
        let mut stmt = self.conn.prepare(
            r#"
            SELECT id, agent_id, agent_type, namespace, role, metadata_json, context_id, COALESCE(last_seen_at, ts), tick_count
            FROM agent_attachments
            WHERE active = 1
            ORDER BY namespace ASC, agent_id ASC, ts DESC
            "#,
        )?;
        let mut rows = stmt.query([])?;
        let mut attachments = Vec::new();
        while let Some(row) = rows.next()? {
            attachments.push(ActiveAttachmentMetric {
                attachment_id: row.get(0)?,
                agent_id: row.get(1)?,
                agent_type: row.get(2)?,
                namespace: row.get(3)?,
                role: row.get(4)?,
                metadata: serde_json::from_str(&row.get::<_, String>(5)?)?,
                context_id: row.get(6)?,
                last_seen_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(7)?)
                    .map_err(to_sqlite_error)?
                    .with_timezone(&Utc),
                tick_count: row.get(8)?,
            });
        }
        Ok(attachments)
    }

    fn list_live_work_claims(&self) -> Result<Vec<ContinuityItemRecord>> {
        #[cfg(test)]
        {
            if let Some(hook) = METRICS_LIVE_STATE_QUERY_COUNTER_HOOK
                .lock()
                .unwrap()
                .as_mut()
            {
                if hook.root == self.paths.root {
                    hook.counts.live_work_claim_reads += 1;
                }
            }
        }
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM contexts ORDER BY namespace ASC, task_id ASC")?;
        let mut rows = stmt.query([])?;
        let now = Utc::now();
        let mut claims = Vec::new();
        while let Some(row) = rows.next()? {
            let context_id = row.get::<_, String>(0)?;
            claims.extend(
                self.list_continuity_items(&context_id, false)?
                    .into_iter()
                    .filter(|item| item.kind == ContinuityKind::WorkClaim)
                    .filter(|item| work_claim_is_live(item, now)),
            );
        }
        Ok(claims)
    }

    fn list_active_coordination_signals(&self) -> Result<Vec<ContinuityItemRecord>> {
        #[cfg(test)]
        {
            if let Some(hook) = METRICS_LIVE_STATE_QUERY_COUNTER_HOOK
                .lock()
                .unwrap()
                .as_mut()
            {
                if hook.root == self.paths.root {
                    hook.counts.active_signal_reads += 1;
                }
            }
        }
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM contexts ORDER BY namespace ASC, task_id ASC")?;
        let mut rows = stmt.query([])?;
        let mut signals = Vec::new();
        while let Some(row) = rows.next()? {
            let context_id = row.get::<_, String>(0)?;
            signals.extend(
                self.list_continuity_items(&context_id, false)?
                    .into_iter()
                    .filter(|item| coordination_signal(item).is_some()),
            );
        }
        Ok(signals)
    }

    fn get_stored_agent_badge(&self, attachment_id: &str) -> Result<Option<StoredAgentBadge>> {
        self.conn
            .query_row(
                r#"
                SELECT updated_at, agent_id, display_name, status, focus, headline, resource,
                       repo_root, branch, metadata_json, context_id, task_id
                FROM agent_badges
                WHERE attachment_id = ?1
                "#,
                params![attachment_id],
                |row| {
                    Ok(StoredAgentBadge {
                        updated_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(0)?)
                            .map_err(to_sqlite_error)?
                            .with_timezone(&Utc),
                        display_name: row.get(2)?,
                        status: row.get(3)?,
                        focus: row.get(4)?,
                        headline: row.get(5)?,
                        resource: row.get(6)?,
                        repo_root: row.get(7)?,
                        branch: row.get(8)?,
                        metadata: serde_json::from_str(&row.get::<_, String>(9)?)
                            .map_err(to_sqlite_error)?,
                        context_id: row.get(10)?,
                        task_id: row.get(11)?,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    fn derive_agent_badge(
        &self,
        attachment: &AgentAttachmentRecord,
        preferred_context_id: Option<&str>,
    ) -> Result<DerivedAgentBadge> {
        let attachment_metric = ActiveAttachmentMetric {
            attachment_id: attachment.id.clone(),
            agent_id: attachment.input.agent_id.clone(),
            agent_type: attachment.input.agent_type.clone(),
            namespace: attachment.input.namespace.clone(),
            role: attachment.input.role.clone(),
            metadata: attachment.input.metadata.clone(),
            context_id: preferred_context_id
                .map(ToString::to_string)
                .or_else(|| attachment.context_id.clone()),
            last_seen_at: attachment.last_seen_at,
            tick_count: attachment.tick_count as i64,
        };
        let live_work_claims = self.list_live_work_claims()?;
        self.derive_agent_badge_from_metric(&attachment_metric, &live_work_claims)
    }

    fn derive_agent_badge_from_metric(
        &self,
        attachment: &ActiveAttachmentMetric,
        live_work_claims: &[ContinuityItemRecord],
    ) -> Result<DerivedAgentBadge> {
        let context = attachment
            .context_id
            .as_deref()
            .map(|context_id| self.get_context(context_id))
            .transpose()?;
        let live_claim = self.primary_live_claim_for_attachment(attachment, live_work_claims);
        let display_name = attachment
            .metadata
            .get("display_name")
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string)
            .unwrap_or_else(|| attachment.agent_id.clone());
        let status = if live_claim.is_some() {
            "working".to_string()
        } else {
            "attached".to_string()
        };
        let focus = live_claim
            .as_ref()
            .map(|claim| claim.title.clone())
            .or_else(|| context.as_ref().map(|context| context.objective.clone()))
            .unwrap_or_else(|| "connected to machine organism".to_string());
        let headline = live_claim
            .as_ref()
            .map(|claim| claim.body.clone())
            .or_else(|| {
                context
                    .as_ref()
                    .map(|context| format!("{} in task {}", context.objective, context.task_id))
            })
            .unwrap_or_else(|| {
                format!(
                    "{} is attached to {}",
                    attachment.agent_id, attachment.namespace
                )
            });
        let resource = live_claim.as_ref().and_then(|claim| {
            work_claim_coordination(claim)
                .and_then(|coordination| coordination.resources.into_iter().next())
        });
        let repo_root = attachment
            .metadata
            .get("repo_root")
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string);
        let branch = attachment
            .metadata
            .get("branch")
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string);
        Ok(DerivedAgentBadge {
            context_id: attachment.context_id.clone(),
            task_id: context.map(|context| context.task_id),
            display_name,
            status,
            focus,
            headline,
            resource,
            repo_root,
            branch,
            metadata: attachment.metadata.clone(),
        })
    }

    fn primary_live_claim_for_attachment(
        &self,
        attachment: &ActiveAttachmentMetric,
        live_work_claims: &[ContinuityItemRecord],
    ) -> Option<ContinuityItemRecord> {
        let mut claims = live_work_claims
            .iter()
            .cloned()
            .filter(|claim| claim.namespace == attachment.namespace)
            .filter(|claim| claim.author_agent_id == attachment.agent_id)
            .filter(|claim| {
                work_claim_coordination(claim)
                    .and_then(|coordination| coordination.attachment_id)
                    .map(|attachment_id| attachment_id == attachment.attachment_id)
                    .unwrap_or(false)
                    || attachment
                        .context_id
                        .as_deref()
                        .map(|context_id| claim.context_id == context_id)
                        .unwrap_or(false)
            })
            .collect::<Vec<_>>();
        claims.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
        claims.into_iter().next()
    }

    fn append_raw_log(
        &mut self,
        event_id: &str,
        ts: DateTime<Utc>,
        input: &EventInput,
        content_hash: &str,
        byte_size: usize,
        token_estimate: usize,
        importance: f64,
    ) -> Result<(i64, i64)> {
        let mut seq = self.meta_i64("segment_seq")?;
        let mut line = self.meta_i64("segment_line")?;
        let path = self.segment_path(seq);
        if fs::metadata(&path).map(|m| m.len()).unwrap_or(0) > self.config.segment_target_bytes {
            seq += 1;
            line = 0;
            self.set_meta_i64("segment_seq", seq)?;
            self.set_meta_i64("segment_line", line)?;
        }

        line += 1;
        let path = self.segment_path(seq);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        let mut encoder = Encoder::new(file, 3)?;
        let payload = serde_json::json!({
            "id": event_id,
            "ts": ts,
            "kind": input.kind,
            "scope": input.scope,
            "agent_id": input.agent_id,
            "agent_role": input.agent_role,
            "session_id": input.session_id,
            "task_id": input.task_id,
            "project_id": input.project_id,
            "goal_id": input.goal_id,
            "run_id": input.run_id,
            "namespace": input.namespace,
            "environment": input.environment,
            "source": input.source,
            "tags": input.tags,
            "dimensions": input.dimensions,
            "attributes": input.attributes,
            "content": input.content,
            "content_hash": content_hash,
            "byte_size": byte_size,
            "token_estimate": token_estimate,
            "importance": importance,
            "segment_seq": seq,
            "segment_line": line,
        });
        let line_bytes = serde_json::to_vec(&payload)?;
        encoder.write_all(&line_bytes)?;
        encoder.write_all(b"\n")?;
        encoder.finish()?;

        self.set_meta_i64("segment_seq", seq)?;
        self.set_meta_i64("segment_line", line)?;
        Ok((seq, line))
    }

    fn write_ingest_manifest(&self, manifest: &IngestManifest) -> Result<()> {
        let path = self
            .paths
            .debug_dir
            .join("ingest")
            .join(format!("{}.json", manifest.event.id));
        fs::write(path, serde_json::to_vec_pretty(manifest)?)?;
        Ok(())
    }

    fn insert_lineage(&self, row: &LineageRecord) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO lineage(parent_id, child_id, edge_kind, weight)
            VALUES (?1, ?2, ?3, ?4)
            "#,
            params![row.parent_id, row.child_id, row.edge_kind, row.weight],
        )?;
        Ok(())
    }

    fn refresh_fts_entry(&self, memory_id: &str, body: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM memory_fts WHERE memory_id = ?1",
            params![memory_id],
        )?;
        self.conn.execute(
            "INSERT INTO memory_fts(memory_id, body) VALUES (?1, ?2)",
            params![memory_id, body],
        )?;
        Ok(())
    }

    fn refresh_vector(&self, memory_id: &str, body: &str) -> Result<()> {
        let vector = self.embedding.embed(body)?;
        let norm = l2_norm(&vector);
        let backend_key = self.embedding.backend_key();
        self.conn.execute(
            r#"
            INSERT INTO memory_vectors(memory_id, backend_key, dim, norm, data)
            VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(memory_id) DO UPDATE SET
              backend_key = excluded.backend_key,
              dim = excluded.dim,
              norm = excluded.norm,
              data = excluded.data
            "#,
            params![
                memory_id,
                backend_key,
                vector.len() as i64,
                norm,
                encode_vector(&vector)
            ],
        )?;
        Ok(())
    }

    fn upsert_entity_mentions(
        &self,
        memory_id: &str,
        event_id: &str,
        ts: DateTime<Utc>,
        entities: &[SemanticEntity],
    ) -> Result<()> {
        for entity in entities {
            let entity_id = self.upsert_entity(entity)?;
            self.conn.execute(
                r#"
                INSERT OR REPLACE INTO entity_mentions(entity_id, memory_id, event_id, role, ts, weight)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                "#,
                params![
                    entity_id,
                    memory_id,
                    event_id,
                    entity.kind,
                    ts.to_rfc3339(),
                    entity.weight
                ],
            )?;
        }
        Ok(())
    }

    fn upsert_entity(&self, entity: &SemanticEntity) -> Result<String> {
        let existing = self
            .conn
            .query_row(
                "SELECT id FROM entities WHERE normalized = ?1",
                params![entity.normalized],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        if let Some(id) = existing {
            self.conn.execute(
                "UPDATE entities SET value = ?2, kind = ?3, weight = MAX(weight, ?4) WHERE id = ?1",
                params![id, entity.value, entity.kind, entity.weight],
            )?;
            return Ok(id);
        }
        let id = format!("ent:{}", Uuid::now_v7());
        self.conn.execute(
            "INSERT INTO entities(id, normalized, value, kind, weight) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                id,
                entity.normalized,
                entity.value,
                entity.kind,
                entity.weight
            ],
        )?;
        Ok(id)
    }

    fn meta_string(&self, key: &str) -> Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                params![key],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(Into::into)
    }

    fn set_meta_string(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    fn meta_i64(&self, key: &str) -> Result<i64> {
        let value = self
            .meta_string(key)?
            .ok_or_else(|| anyhow!("missing meta key {key}"))?;
        value.parse::<i64>().map_err(Into::into)
    }

    fn set_meta_i64(&self, key: &str, value: i64) -> Result<()> {
        self.set_meta_string(key, &value.to_string())
    }

    fn segment_path(&self, seq: i64) -> PathBuf {
        self.paths
            .log_dir
            .join(format!("events-{seq:08}.jsonl.zst"))
    }

    #[cfg(test)]
    pub(crate) fn install_storage_bytes_metrics_test_hook(
        log_dir: PathBuf,
        entered_tx: std::sync::mpsc::Sender<()>,
        release_rx: std::sync::mpsc::Receiver<()>,
    ) {
        let mut hook = STORAGE_BYTES_METRICS_TEST_HOOK.lock().unwrap();
        *hook = Some(StorageBytesMetricsTestHook {
            log_dir,
            entered_tx,
            release_rx,
        });
    }

    #[cfg(test)]
    pub(crate) fn install_metrics_live_state_query_counter(root: PathBuf) {
        *METRICS_LIVE_STATE_QUERY_COUNTER_HOOK.lock().unwrap() =
            Some(MetricsLiveStateQueryCounterHook {
                root,
                counts: MetricsLiveStateQueryCounts::default(),
            });
    }

    #[cfg(test)]
    pub(crate) fn metrics_live_state_query_counts() -> (usize, usize, usize, usize) {
        let counts = METRICS_LIVE_STATE_QUERY_COUNTER_HOOK
            .lock()
            .unwrap()
            .as_ref()
            .map(|hook| hook.counts)
            .unwrap_or_default();
        (
            counts.active_attachment_reads,
            counts.continuity_snapshot_reads,
            counts.live_work_claim_reads,
            counts.active_signal_reads,
        )
    }

    fn list_live_continuity_metric_snapshot(&self) -> Result<LiveContinuityMetricSnapshot> {
        #[cfg(test)]
        {
            if let Some(hook) = METRICS_LIVE_STATE_QUERY_COUNTER_HOOK
                .lock()
                .unwrap()
                .as_mut()
            {
                if hook.root == self.paths.root {
                    hook.counts.continuity_snapshot_reads += 1;
                }
            }
        }
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM contexts ORDER BY namespace ASC, task_id ASC")?;
        let mut rows = stmt.query([])?;
        let now = Utc::now();
        let mut snapshot = LiveContinuityMetricSnapshot::default();
        while let Some(row) = rows.next()? {
            let context_id = row.get::<_, String>(0)?;
            for item in self.list_continuity_items(&context_id, false)? {
                if item.kind == ContinuityKind::WorkClaim && work_claim_is_live(&item, now) {
                    snapshot.live_work_claims.push(item);
                } else if coordination_signal(&item).is_some() {
                    snapshot.active_signals.push(item);
                }
            }
        }
        Ok(snapshot)
    }

    fn continuity_compiler_band_totals(&self) -> Result<BTreeMap<String, (i64, i64)>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT band, COUNT(*), COALESCE(SUM(item_count), 0)
            FROM continuity_compiled_chunks
            GROUP BY band
            "#,
        )?;
        let mut rows = stmt.query([])?;
        let mut out = BTreeMap::new();
        while let Some(row) = rows.next()? {
            out.insert(
                row.get::<_, String>(0)?,
                (row.get::<_, i64>(1)?, row.get::<_, i64>(2)?),
            );
        }
        Ok(out)
    }

    fn supports_for_continuity_many(
        &self,
        continuity_ids: &[String],
    ) -> Result<HashMap<String, Vec<SupportRef>>> {
        let mut out = HashMap::<String, Vec<SupportRef>>::new();
        if continuity_ids.is_empty() {
            return Ok(out);
        }
        for chunk in continuity_ids.chunks(200) {
            let placeholders = (0..chunk.len())
                .map(|index| format!("?{}", index + 1))
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!(
                r#"
                SELECT continuity_id, support_type, support_id, reason, weight
                FROM continuity_support
                WHERE continuity_id IN ({placeholders})
                ORDER BY continuity_id ASC, weight DESC, support_type ASC, support_id ASC, reason ASC
                "#
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let mut rows = stmt.query(rusqlite::params_from_iter(
                chunk.iter().map(|id| id.as_str()),
            ))?;
            while let Some(row) = rows.next()? {
                out.entry(row.get::<_, String>(0)?)
                    .or_default()
                    .push(SupportRef {
                        support_type: row.get(1)?,
                        support_id: row.get(2)?,
                        reason: row.get(3)?,
                        weight: row.get(4)?,
                    });
            }
        }
        Ok(out)
    }

    fn support_parent_id(&self, support_type: &str, support_id: &str) -> Result<Option<String>> {
        Ok(match support_type {
            "event" | "memory" => Some(support_id.to_string()),
            "continuity" => self
                .conn
                .query_row(
                    "SELECT memory_id FROM continuity_items WHERE id = ?1",
                    params![support_id],
                    |row| row.get::<_, String>(0),
                )
                .optional()?,
            _ => None,
        })
    }

    fn replace_dimension(
        &self,
        item_type: &str,
        item_id: &str,
        key: &str,
        value: &str,
    ) -> Result<()> {
        self.conn.execute(
            "DELETE FROM item_dimensions WHERE item_type = ?1 AND item_id = ?2 AND key = ?3",
            params![item_type, item_id, key],
        )?;
        self.index_item_dimensions(
            item_type,
            item_id,
            self.item_metadata(item_type, item_id)?.1,
            Utc::now(),
            &[DimensionValue {
                key: key.to_string(),
                value: value.to_string(),
                weight: 100,
            }],
        )
    }

    fn persist_snapshot_memory(
        &self,
        context: &ContextRecord,
        snapshot: &SnapshotRecord,
    ) -> Result<()> {
        let body = format!(
            "[snapshot:{}] objective={}\nview={}\npack={}\nselector={}",
            snapshot.resolution,
            snapshot.objective,
            snapshot.view_id,
            snapshot.pack_id,
            serde_json::to_string(&snapshot.selector)?,
        );
        let memory_id = format!("cold:{}", snapshot.id);
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO memory_items (
              id, layer, scope, agent_id, session_id, task_id, ts, importance, confidence,
              token_estimate, source_event_id, scope_key, body, extra_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, NULL, ?11, ?12, ?13)
            "#,
            params![
                memory_id,
                MemoryLayer::Cold.as_i64(),
                Scope::Project.to_string(),
                context
                    .current_agent_id
                    .clone()
                    .unwrap_or_else(|| "kernel".to_string()),
                context.session_id,
                context.task_id,
                snapshot.created_at.to_rfc3339(),
                0.9_f64,
                0.95_f64,
                estimate_tokens(&body) as i64,
                snapshot.id,
                body,
                serde_json::json!({
                    "snapshot_id": snapshot.id,
                    "context_id": context.id,
                    "namespace": context.namespace,
                    "task_id": context.task_id,
                    "view_id": snapshot.view_id,
                    "pack_id": snapshot.pack_id,
                })
                .to_string(),
            ],
        )?;
        self.refresh_fts_entry(&memory_id, &body)?;
        self.refresh_vector(&memory_id, &body)?;
        self.index_item_dimensions(
            "memory",
            &memory_id,
            Some(MemoryLayer::Cold),
            snapshot.created_at,
            &[
                DimensionValue {
                    key: "context".to_string(),
                    value: context.id.clone(),
                    weight: 100,
                },
                DimensionValue {
                    key: "snapshot".to_string(),
                    value: snapshot.id.clone(),
                    weight: 100,
                },
                DimensionValue {
                    key: "continuity_kind".to_string(),
                    value: "snapshot".to_string(),
                    weight: 100,
                },
            ],
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct ScoredMemory {
    memory: MemoryRecord,
    score: f64,
    matched_dimensions: Vec<String>,
    why: Vec<String>,
}

#[derive(Debug)]
struct ViewAccumulator {
    memory: MemoryRecord,
    score: f64,
    matched_dimensions: HashSet<String>,
    why: HashSet<String>,
    sources: HashSet<usize>,
}

impl ViewAccumulator {
    fn new(memory: MemoryRecord) -> Self {
        Self {
            memory,
            score: 0.0,
            matched_dimensions: HashSet::new(),
            why: HashSet::new(),
            sources: HashSet::new(),
        }
    }
}

fn parse_scope(value: &str) -> Result<Scope> {
    Ok(match value {
        "agent" => Scope::Agent,
        "session" => Scope::Session,
        "shared" => Scope::Shared,
        "project" => Scope::Project,
        "global" => Scope::Global,
        other => return Err(anyhow!("unknown scope {other}")),
    })
}

fn parse_context_status(value: &str) -> Result<ContextStatus> {
    Ok(match value {
        "open" => ContextStatus::Open,
        "paused" => ContextStatus::Paused,
        "closed" => ContextStatus::Closed,
        other => return Err(anyhow!("unknown context status {other}")),
    })
}

fn parse_continuity_kind(value: &str) -> Result<ContinuityKind> {
    Ok(match value {
        "working_state" => ContinuityKind::WorkingState,
        "work_claim" => ContinuityKind::WorkClaim,
        "derivation" => ContinuityKind::Derivation,
        "fact" => ContinuityKind::Fact,
        "decision" => ContinuityKind::Decision,
        "constraint" => ContinuityKind::Constraint,
        "hypothesis" => ContinuityKind::Hypothesis,
        "incident" => ContinuityKind::Incident,
        "operational_scar" => ContinuityKind::OperationalScar,
        "outcome" => ContinuityKind::Outcome,
        "signal" => ContinuityKind::Signal,
        "summary" => ContinuityKind::Summary,
        "lesson" => ContinuityKind::Lesson,
        other => return Err(anyhow!("unknown continuity kind {other}")),
    })
}

fn parse_continuity_status(value: &str) -> Result<ContinuityStatus> {
    Ok(match value {
        "open" => ContinuityStatus::Open,
        "active" => ContinuityStatus::Active,
        "resolved" => ContinuityStatus::Resolved,
        "superseded" => ContinuityStatus::Superseded,
        "rejected" => ContinuityStatus::Rejected,
        other => return Err(anyhow!("unknown continuity status {other}")),
    })
}

fn continuity_metadata_str(extra: &serde_json::Value, key: &str) -> Option<String> {
    extra
        .get(key)
        .and_then(|value| value.as_str())
        .or_else(|| {
            extra
                .get("user")
                .and_then(|user| user.get(key))
                .and_then(|value| value.as_str())
        })
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn continuity_belief_key(extra: &serde_json::Value) -> Option<String> {
    continuity_metadata_str(extra, "belief_key")
}

fn continuity_practice_key(extra: &serde_json::Value) -> Option<String> {
    continuity_metadata_str(extra, "practice_key")
}

fn continuity_source_role(extra: &serde_json::Value) -> Option<String> {
    continuity_metadata_str(extra, "source_role")
}

fn continuity_dimensions(
    context: &ContextRecord,
    input: &ContinuityItemInput,
    status: ContinuityStatus,
) -> Vec<DimensionValue> {
    let mut dimensions = vec![
        DimensionValue {
            key: "context".to_string(),
            value: context.id.clone(),
            weight: 100,
        },
        DimensionValue {
            key: "task".to_string(),
            value: context.task_id.clone(),
            weight: 100,
        },
        DimensionValue {
            key: "namespace".to_string(),
            value: context.namespace.clone(),
            weight: 100,
        },
        DimensionValue {
            key: "continuity_kind".to_string(),
            value: input.kind.as_str().to_string(),
            weight: 100,
        },
        DimensionValue {
            key: "continuity_status".to_string(),
            value: status.as_str().to_string(),
            weight: 100,
        },
        DimensionValue {
            key: "continuity_author".to_string(),
            value: input.author_agent_id.clone(),
            weight: 100,
        },
    ];
    if matches!(input.kind, ContinuityKind::OperationalScar) {
        dimensions.push(DimensionValue {
            key: "scar".to_string(),
            value: "true".to_string(),
            weight: 100,
        });
    }
    if let Some(belief_key) = continuity_belief_key(&input.extra) {
        dimensions.push(DimensionValue {
            key: "belief_key".to_string(),
            value: belief_key,
            weight: 100,
        });
    }
    if let Some(practice_key) = continuity_practice_key(&input.extra) {
        dimensions.push(DimensionValue {
            key: "practice_key".to_string(),
            value: practice_key,
            weight: 100,
        });
    }
    if let Some(source_role) = continuity_source_role(&input.extra) {
        dimensions.push(DimensionValue {
            key: "source_role".to_string(),
            value: source_role,
            weight: 90,
        });
    }
    dimensions.extend(input.dimensions.clone());
    dimensions
}

fn claim_dimensions(resources: &[String], exclusive: bool, claim_key: &str) -> Vec<DimensionValue> {
    let mut dimensions = vec![
        DimensionValue {
            key: "claim.key".to_string(),
            value: claim_key.to_string(),
            weight: 100,
        },
        DimensionValue {
            key: "claim.exclusive".to_string(),
            value: if exclusive { "true" } else { "false" }.to_string(),
            weight: 100,
        },
    ];
    for resource in resources {
        dimensions.push(DimensionValue {
            key: "claim.resource".to_string(),
            value: resource.clone(),
            weight: 100,
        });
    }
    dimensions
}

fn continuity_body(
    title: &str,
    body: &str,
    kind: ContinuityKind,
    status: ContinuityStatus,
    supports: &[SupportRef],
) -> String {
    let mut lines = vec![
        format!("kind={}", kind.as_str()),
        format!("status={}", status.as_str()),
        format!("title={title}"),
    ];
    if !supports.is_empty() {
        lines.push(format!("supports={}", supports.len()));
    }
    lines.push(format!("body={body}"));
    lines.join("\n")
}

fn parse_view_op(value: &str) -> Result<ViewOp> {
    Ok(match value {
        "project" => ViewOp::Project,
        "slice" => ViewOp::Slice,
        "intersect" => ViewOp::Intersect,
        "union" => ViewOp::Union,
        "snapshot" => ViewOp::Snapshot,
        "fork" => ViewOp::Fork,
        "merge" => ViewOp::Merge,
        other => return Err(anyhow!("unknown view op {other}")),
    })
}

fn parse_snapshot_resolution(value: &str) -> Result<SnapshotResolution> {
    Ok(match value {
        "coarse" => SnapshotResolution::Coarse,
        "medium" => SnapshotResolution::Medium,
        "fine" => SnapshotResolution::Fine,
        other => return Err(anyhow!("unknown snapshot resolution {other}")),
    })
}

fn snapshot_candidates(
    candidates: Vec<ScoredMemory>,
    resolution: SnapshotResolution,
) -> Vec<ScoredMemory> {
    if matches!(resolution, SnapshotResolution::Fine) {
        return candidates;
    }
    let mut grouped = BTreeMap::<String, ScoredMemory>::new();
    for candidate in candidates {
        let group_key = match resolution {
            SnapshotResolution::Coarse => candidate.memory.scope_key.clone(),
            SnapshotResolution::Medium => format!(
                "{}:{}",
                candidate
                    .memory
                    .source_event_id
                    .clone()
                    .unwrap_or_else(|| candidate.memory.scope_key.clone()),
                if matches!(
                    candidate.memory.layer,
                    MemoryLayer::Summary | MemoryLayer::Semantic
                ) {
                    "high"
                } else {
                    "low"
                }
            ),
            SnapshotResolution::Fine => unreachable!(),
        };
        match grouped.get_mut(&group_key) {
            Some(existing)
                if layer_resolution_rank(candidate.memory.layer)
                    > layer_resolution_rank(existing.memory.layer)
                    || candidate.score > existing.score =>
            {
                *existing = candidate;
            }
            None => {
                grouped.insert(group_key, candidate);
            }
            _ => {}
        }
    }
    grouped.into_values().collect()
}

fn layer_resolution_rank(layer: MemoryLayer) -> i32 {
    match layer {
        MemoryLayer::Summary => 5,
        MemoryLayer::Semantic => 4,
        MemoryLayer::Episodic => 3,
        MemoryLayer::Hot => 2,
        MemoryLayer::Cold => 1,
    }
}

fn is_conflict_dimension(key: &str) -> bool {
    key.starts_with("claim.")
        || key.starts_with("hypothesis.")
        || key.starts_with("decision.")
        || key.starts_with("constraint.")
}

fn event_dimensions(event: &EventRecord, entities: &[SemanticEntity]) -> Vec<DimensionValue> {
    let mut out = vec![
        dim("agent", &event.input.agent_id),
        dim("session", &event.input.session_id),
        dim("kind", &event.input.kind.to_string()),
        dim("scope", &event.input.scope.to_string()),
        dim("source", &event.input.source),
    ];
    if let Some(agent_role) = &event.input.agent_role {
        out.push(dim("agent_role", agent_role));
    }
    if let Some(task_id) = &event.input.task_id {
        out.push(dim("task", task_id));
    }
    if let Some(project_id) = &event.input.project_id {
        out.push(dim("project", project_id));
    }
    if let Some(goal_id) = &event.input.goal_id {
        out.push(dim("goal", goal_id));
    }
    if let Some(run_id) = &event.input.run_id {
        out.push(dim("run", run_id));
    }
    if let Some(namespace) = &event.input.namespace {
        out.push(dim("namespace", namespace));
    }
    if let Some(environment) = &event.input.environment {
        out.push(dim("environment", environment));
    }
    for tag in &event.input.tags {
        out.push(dim("tag", tag));
    }
    for dimension in &event.input.dimensions {
        out.push(dimension.clone());
    }
    for entity in entities {
        out.push(DimensionValue {
            key: entity.kind.clone(),
            value: entity.value.clone(),
            weight: (entity.weight * 100.0).round() as i32,
        });
    }
    out
}

fn dim(key: &str, value: &str) -> DimensionValue {
    DimensionValue {
        key: key.to_string(),
        value: value.to_string(),
        weight: 100,
    }
}

fn to_sqlite_error<E>(err: E) -> rusqlite::Error
where
    E: std::error::Error + Send + Sync + 'static,
{
    rusqlite::Error::ToSqlConversionFailure(Box::new(err))
}

fn to_sqlite_anyhow(err: anyhow::Error) -> rusqlite::Error {
    rusqlite::Error::ToSqlConversionFailure(err.into())
}

fn recency_score(now: chrono::DateTime<Utc>, ts: chrono::DateTime<Utc>) -> f64 {
    let age_hours = (now - ts).num_seconds().max(0) as f64 / 3600.0;
    1.0 / (1.0 + age_hours / 6.0)
}

struct ContinuityRetentionInput<'a> {
    kind: ContinuityKind,
    status: ContinuityStatus,
    salience: f64,
    importance: f64,
    confidence: f64,
    updated_at: chrono::DateTime<Utc>,
    plasticity: Option<&'a ContinuityPlasticityState>,
    now: chrono::DateTime<Utc>,
}

fn continuity_retention_state(input: ContinuityRetentionInput<'_>) -> ContinuityRetentionState {
    let (base_class, base_half_life_hours, base_floor) = match input.kind {
        ContinuityKind::WorkingState => ("working", 18.0, 0.02),
        ContinuityKind::WorkClaim => ("work_claim", 8.0, 0.02),
        ContinuityKind::Signal => ("signal", 12.0, 0.01),
        ContinuityKind::Hypothesis => ("hypothesis", 36.0, 0.03),
        ContinuityKind::Outcome => ("outcome", 48.0, 0.04),
        ContinuityKind::Derivation => ("derivation", 72.0, 0.05),
        ContinuityKind::Summary => ("summary", 96.0, 0.08),
        ContinuityKind::Fact => ("fact", 144.0, 0.06),
        ContinuityKind::Lesson => ("lesson", 192.0, 0.10),
        ContinuityKind::Decision => ("decision", 240.0, 0.14),
        ContinuityKind::Constraint => ("constraint", 336.0, 0.22),
        ContinuityKind::Incident => ("incident", 432.0, 0.20),
        ContinuityKind::OperationalScar => ("operational_scar", 720.0, 0.36),
    };
    let importance_boost = 0.8 + input.importance.clamp(0.0, 1.0) * 0.8;
    let confidence_boost = 0.8 + input.confidence.clamp(0.0, 1.0) * 0.4;
    let mut half_life_hours = base_half_life_hours * importance_boost * confidence_boost;
    let mut floor = base_floor;
    let class = if input.status.is_open() {
        if matches!(
            input.kind,
            ContinuityKind::OperationalScar | ContinuityKind::Incident
        ) {
            half_life_hours *= 1.15;
        }
        if matches!(
            input.kind,
            ContinuityKind::Constraint | ContinuityKind::Decision
        ) {
            half_life_hours *= 1.05;
        }
        base_class.to_string()
    } else {
        half_life_hours *= 0.3;
        floor *= 0.2;
        format!("treated_{base_class}")
    };
    if let Some(plasticity) = input.plasticity {
        let reinforcement = 0.06 * (plasticity.activation_count as f64 + 1.0).ln()
            + 0.12 * (plasticity.successful_use_count as f64 + 1.0).ln()
            + 0.16 * (plasticity.confirmation_count as f64 + 1.0).ln()
            + 0.22 * (plasticity.spaced_reactivation_count as f64 + 1.0).ln()
            + 0.04 * spacing_interval_signal(plasticity)
            + 0.05 * (plasticity.independent_source_count as f64 + 1.0).ln();
        let contradiction_penalty = 0.22 * plasticity.contradiction_count as f64;
        let half_life_multiplier =
            (1.0_f64 + reinforcement - contradiction_penalty).clamp(0.35_f64, 3.5_f64);
        half_life_hours *= half_life_multiplier;
        if plasticity.prediction_error > 0.5 {
            floor *= (1.0_f64 - (plasticity.prediction_error - 0.5_f64).min(0.4_f64))
                .clamp(0.5_f64, 1.0_f64);
        } else if plasticity.confirmation_count > 0 {
            floor = (floor
                + 0.02_f64 * (plasticity.confirmation_count as f64 + 1.0_f64).ln()
                + 0.01_f64 * (plasticity.successful_use_count as f64 + 1.0_f64).ln())
            .min(0.95_f64);
        }
    }
    let age_hours = (input.now - input.updated_at).num_seconds().max(0) as f64 / 3600.0;
    let decay_multiplier = 0.5_f64.powf(age_hours / half_life_hours.max(1.0));
    let plasticity_salience_bonus = input
        .plasticity
        .map(|plasticity| {
            (0.03 * (plasticity.successful_use_count as f64 + 1.0).ln()
                + 0.05 * (plasticity.confirmation_count as f64 + 1.0).ln()
                + 0.07 * (plasticity.spaced_reactivation_count as f64 + 1.0).ln()
                + 0.02 * spacing_interval_signal(plasticity)
                - 0.12 * plasticity.prediction_error
                - 0.04 * plasticity.contradiction_count as f64)
                .clamp(-0.35_f64, 0.25_f64)
        })
        .unwrap_or(0.0);
    let effective_salience = input
        .salience
        .clamp(0.0, 1.0)
        .mul_add(decay_multiplier, plasticity_salience_bonus);
    ContinuityRetentionState {
        class,
        age_hours,
        half_life_hours,
        floor,
        decay_multiplier,
        effective_salience: effective_salience.max(floor).min(1.0),
    }
}

fn read_attachment_row(
    attachment_id: &str,
    row: &rusqlite::Row<'_>,
) -> Result<AgentAttachmentRecord> {
    read_attachment_row_with_offset(attachment_id, row, 0)
}

fn read_attachment_row_with_offset(
    attachment_id: &str,
    row: &rusqlite::Row<'_>,
    offset: usize,
) -> Result<AgentAttachmentRecord> {
    let attached_at =
        DateTime::parse_from_rfc3339(&row.get::<_, String>(offset)?)?.with_timezone(&Utc);
    let last_seen_at =
        DateTime::parse_from_rfc3339(&row.get::<_, String>(offset + 8)?)?.with_timezone(&Utc);
    Ok(AgentAttachmentRecord {
        id: attachment_id.to_string(),
        attached_at,
        last_seen_at,
        input: AttachAgentInput {
            agent_id: row.get(offset + 1)?,
            agent_type: row.get(offset + 2)?,
            capabilities: serde_json::from_str(&row.get::<_, String>(offset + 3)?)?,
            namespace: row.get(offset + 4)?,
            role: row.get(offset + 5)?,
            metadata: serde_json::from_str(&row.get::<_, String>(offset + 6)?)?,
        },
        active: row.get::<_, i64>(offset + 7)? != 0,
        tick_count: row.get::<_, i64>(offset + 9)? as usize,
        context_id: row.get(offset + 10)?,
    })
}

fn read_memory_row(row: &rusqlite::Row<'_>) -> Result<MemoryRecord> {
    Ok(MemoryRecord {
        id: row.get(0)?,
        layer: match row.get::<_, i64>(1)? {
            1 => MemoryLayer::Hot,
            2 => MemoryLayer::Episodic,
            3 => MemoryLayer::Semantic,
            4 => MemoryLayer::Summary,
            5 => MemoryLayer::Cold,
            layer => return Err(anyhow!("unknown memory layer {layer}")),
        },
        scope: match row.get::<_, String>(2)?.as_str() {
            "agent" => crate::model::Scope::Agent,
            "session" => crate::model::Scope::Session,
            "shared" => crate::model::Scope::Shared,
            "project" => crate::model::Scope::Project,
            "global" => crate::model::Scope::Global,
            scope => return Err(anyhow!("unknown scope {scope}")),
        },
        agent_id: row.get(3)?,
        session_id: row.get(4)?,
        task_id: row.get(5)?,
        ts: DateTime::parse_from_rfc3339(&row.get::<_, String>(6)?)?.with_timezone(&Utc),
        importance: row.get(7)?,
        confidence: row.get(8)?,
        token_estimate: row.get::<_, i64>(9)? as usize,
        source_event_id: row.get(10)?,
        scope_key: row.get(11)?,
        body: row.get(12)?,
        extra: serde_json::from_str::<serde_json::Value>(&row.get::<_, String>(13)?)?,
    })
}

#[derive(Debug, Clone)]
struct RawContinuityRow {
    id: String,
    memory_id: String,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    context_id: String,
    namespace: String,
    task_id: String,
    author_agent_id: String,
    kind: ContinuityKind,
    scope: Scope,
    status: ContinuityStatus,
    title: String,
    body: String,
    importance: f64,
    confidence: f64,
    salience: f64,
    supersedes_id: Option<String>,
    resolved_at: Option<DateTime<Utc>>,
    extra: serde_json::Value,
}

#[derive(Debug, Default, Clone)]
struct RecallSeed {
    raw: Option<RawContinuityRow>,
    lexical_rank: Option<usize>,
    priority_rank: Option<usize>,
    compiled_rank: Option<usize>,
    compiled_band: Option<String>,
}

#[derive(Debug, Clone)]
struct CompiledContinuityChunk {
    chunk_id: String,
    band: String,
    compiled_at: DateTime<Utc>,
    item_ids: Vec<String>,
    item_count: usize,
}

#[derive(Debug, Clone, Default)]
struct StoredContinuityPlasticity {
    belief_key: Option<String>,
    source_role: Option<String>,
    state: ContinuityPlasticityState,
}

#[derive(Debug, Clone)]
struct ScoredContinuityRecallItem {
    item: ContinuityRecallItem,
    belief_key: Option<String>,
    practice_key: Option<String>,
    source_role: Option<String>,
    plasticity: Option<ContinuityPlasticityState>,
    practice_state: Option<PracticeLifecycleState>,
}

fn read_continuity_row(row: &rusqlite::Row<'_>) -> Result<RawContinuityRow> {
    Ok(RawContinuityRow {
        id: row.get(0)?,
        memory_id: row.get(1)?,
        created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(2)?)?.with_timezone(&Utc),
        updated_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(3)?)?.with_timezone(&Utc),
        context_id: row.get(4)?,
        namespace: row.get(5)?,
        task_id: row.get(6)?,
        author_agent_id: row.get(7)?,
        kind: parse_continuity_kind(&row.get::<_, String>(8)?)?,
        scope: parse_scope(&row.get::<_, String>(9)?)?,
        status: parse_continuity_status(&row.get::<_, String>(10)?)?,
        title: row.get(11)?,
        body: row.get(12)?,
        importance: row.get(13)?,
        confidence: row.get(14)?,
        salience: row.get(15)?,
        supersedes_id: row.get(16)?,
        resolved_at: row
            .get::<_, Option<String>>(17)?
            .map(|value| DateTime::parse_from_rfc3339(&value))
            .transpose()?
            .map(|ts| ts.with_timezone(&Utc)),
        extra: serde_json::from_str::<serde_json::Value>(&row.get::<_, String>(18)?)?,
    })
}

fn build_continuity_record(
    raw: RawContinuityRow,
    supports: Vec<SupportRef>,
    plasticity: Option<StoredContinuityPlasticity>,
    now: DateTime<Utc>,
) -> Result<ContinuityItemRecord> {
    let mut extra = raw.extra;
    if let Some(ref plasticity) = plasticity {
        extra["plasticity"] = serde_json::to_value(&plasticity.state)?;
        if let Some(belief_key) = &plasticity.belief_key {
            extra["belief_key"] = serde_json::json!(belief_key);
        }
        if let Some(source_role) = &plasticity.source_role {
            extra["source_role"] = serde_json::json!(source_role);
        }
    }
    let retention = continuity_retention_state(ContinuityRetentionInput {
        kind: raw.kind,
        status: raw.status,
        salience: raw.salience,
        importance: raw.importance,
        confidence: raw.confidence,
        updated_at: raw.updated_at,
        plasticity: plasticity.as_ref().map(|value| &value.state),
        now,
    });
    Ok(ContinuityItemRecord {
        id: raw.id,
        memory_id: raw.memory_id,
        context_id: raw.context_id,
        namespace: raw.namespace,
        task_id: raw.task_id,
        author_agent_id: raw.author_agent_id,
        kind: raw.kind,
        scope: raw.scope,
        status: raw.status,
        title: raw.title,
        body: raw.body,
        importance: raw.importance,
        confidence: raw.confidence,
        salience: raw.salience,
        retention,
        created_at: raw.created_at,
        updated_at: raw.updated_at,
        supersedes_id: raw.supersedes_id,
        resolved_at: raw.resolved_at,
        supports,
        practice_state: None,
        extra,
    })
}

fn continuity_kind_boost(kind: ContinuityKind) -> f64 {
    match kind {
        ContinuityKind::OperationalScar => 0.42,
        ContinuityKind::Decision => 0.34,
        ContinuityKind::Constraint => 0.31,
        ContinuityKind::Incident => 0.28,
        ContinuityKind::Signal => 0.22,
        ContinuityKind::Outcome => 0.18,
        ContinuityKind::Hypothesis => 0.14,
        ContinuityKind::WorkClaim => 0.1,
        ContinuityKind::WorkingState => 0.08,
        ContinuityKind::Fact => 0.16,
        ContinuityKind::Derivation => 0.12,
        ContinuityKind::Summary => 0.11,
        ContinuityKind::Lesson => 0.2,
    }
}

fn continuity_status_boost(status: ContinuityStatus) -> f64 {
    match status {
        ContinuityStatus::Open => 0.18,
        ContinuityStatus::Active => 0.14,
        ContinuityStatus::Resolved => 0.03,
        ContinuityStatus::Superseded => 0.01,
        ContinuityStatus::Rejected => 0.0,
    }
}

fn continuity_retention_adjustment(item: &ContinuityItemRecord) -> f64 {
    let mut score = 0.0;
    if item.retention.class.starts_with("treated_") {
        score -= 0.18;
    }
    if matches!(item.kind, ContinuityKind::WorkingState) {
        score -= 0.08;
    }
    if matches!(
        item.status,
        ContinuityStatus::Superseded | ContinuityStatus::Rejected
    ) {
        score -= 0.12;
    }
    score
}

fn compact_learning_text(text: &str, max_chars: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= max_chars {
        return compact;
    }
    let keep = max_chars.saturating_sub(3);
    let mut trimmed = compact.chars().take(keep).collect::<String>();
    trimmed.push_str("...");
    trimmed
}

fn normalize_spacing_interval_hours(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(2.0, 24.0 * 21.0)
    } else {
        6.0
    }
}

fn spacing_elapsed_hours(start: DateTime<Utc>, end: DateTime<Utc>) -> f64 {
    (end - start).num_seconds().max(0) as f64 / 3600.0
}

fn spacing_interval_signal(plasticity: &ContinuityPlasticityState) -> f64 {
    (normalize_spacing_interval_hours(plasticity.spacing_interval_hours) / 6.0)
        .max(1.0)
        .ln()
}

fn plasticity_recall_adjustment(plasticity: Option<&ContinuityPlasticityState>) -> f64 {
    plasticity
        .map(|plasticity| {
            (0.04 * (plasticity.activation_count as f64 + 1.0).ln()
                + 0.08 * (plasticity.successful_use_count as f64 + 1.0).ln()
                + 0.12 * (plasticity.confirmation_count as f64 + 1.0).ln()
                + 0.16 * (plasticity.spaced_reactivation_count as f64 + 1.0).ln()
                + 0.04 * spacing_interval_signal(plasticity)
                - 0.18 * plasticity.prediction_error
                - 0.05 * plasticity.contradiction_count as f64)
                .clamp(-0.45_f64, 0.35_f64)
        })
        .unwrap_or(0.0)
}

fn continuity_source_role_adjustment(
    belief_key: Option<&str>,
    source_role: Option<&str>,
    kind: ContinuityKind,
) -> f64 {
    let is_user_belief = belief_key
        .map(|value| value.starts_with("user."))
        .unwrap_or(false);
    match source_role {
        Some("user") => {
            if is_user_belief {
                0.2
            } else {
                0.08
            }
        }
        Some("assistant") => {
            if is_user_belief {
                -0.18
            } else if matches!(
                kind,
                ContinuityKind::Fact | ContinuityKind::Derivation | ContinuityKind::Lesson
            ) {
                -0.08
            } else {
                -0.03
            }
        }
        Some("importer") | Some("system") | Some("tool") => {
            if is_user_belief {
                0.03
            } else {
                0.06
            }
        }
        Some(_) | None => 0.0,
    }
}

fn continuity_belief_competition_rank(item: &ScoredContinuityRecallItem) -> f64 {
    let mut score = item.item.score
        + continuity_source_role_adjustment(
            item.belief_key.as_deref(),
            item.source_role.as_deref(),
            item.item.kind,
        );
    score += match item.item.status {
        ContinuityStatus::Open => 0.08,
        ContinuityStatus::Active => 0.06,
        ContinuityStatus::Resolved => 0.0,
        ContinuityStatus::Superseded => -0.22,
        ContinuityStatus::Rejected => -0.28,
    };
    if let Some(plasticity) = item.plasticity.as_ref() {
        score += 0.03 * (plasticity.confirmation_count as f64 + 1.0).ln();
        score += 0.02 * (plasticity.successful_use_count as f64 + 1.0).ln();
        score += 0.06 * (plasticity.spaced_reactivation_count as f64 + 1.0).ln();
        score += 0.02 * spacing_interval_signal(plasticity);
        score -= 0.05 * plasticity.contradiction_count as f64;
        score -= 0.12 * plasticity.prediction_error;
    }
    score
}

fn is_guidance_like_kind(kind: ContinuityKind) -> bool {
    matches!(
        kind,
        ContinuityKind::Decision
            | ContinuityKind::Constraint
            | ContinuityKind::Lesson
            | ContinuityKind::Outcome
    )
}

fn continuity_practice_anchor(
    updated_at: DateTime<Utc>,
    plasticity: Option<&ContinuityPlasticityState>,
) -> DateTime<Utc> {
    let mut anchor = updated_at;
    if let Some(plasticity) = plasticity {
        for ts in [
            plasticity.last_strengthened_at,
            plasticity.last_confirmed_at,
            plasticity.last_reactivated_at,
        ]
        .into_iter()
        .flatten()
        {
            if ts > anchor {
                anchor = ts;
            }
        }
    }
    anchor
}

fn derive_recall_practice_state(
    item: &ContinuityRecallItem,
    plasticity: Option<&ContinuityPlasticityState>,
    now: DateTime<Utc>,
) -> Option<PracticeLifecycleState> {
    if !is_guidance_like_kind(item.kind) {
        return None;
    }
    if matches!(
        item.status,
        ContinuityStatus::Superseded | ContinuityStatus::Rejected
    ) {
        return Some(PracticeLifecycleState::Retired);
    }
    let anchor = continuity_practice_anchor(item.updated_at, plasticity);
    let age_hours = (now - anchor).num_seconds().max(0) as f64 / 3600.0;
    let base_half_life: f64 = match item.kind {
        ContinuityKind::Outcome => 96.0,
        ContinuityKind::Lesson => 168.0,
        ContinuityKind::Decision => 216.0,
        ContinuityKind::Constraint => 288.0,
        _ => 96.0,
    };
    let fresh_window_hours = base_half_life.clamp(12.0, 24.0 * 14.0) * 0.18;
    let stale_window_hours = base_half_life.clamp(24.0, 24.0 * 45.0) * 0.55;
    let retirement_window_hours = (stale_window_hours * 2.2).clamp(24.0 * 5.0, 24.0 * 120.0);
    if item.status == ContinuityStatus::Resolved {
        if age_hours <= fresh_window_hours * 0.75 {
            Some(PracticeLifecycleState::Aging)
        } else {
            Some(PracticeLifecycleState::Stale)
        }
    } else if item.status.is_open() && age_hours > retirement_window_hours {
        Some(PracticeLifecycleState::Retired)
    } else if age_hours <= fresh_window_hours {
        Some(PracticeLifecycleState::Current)
    } else if age_hours <= stale_window_hours {
        Some(PracticeLifecycleState::Aging)
    } else {
        Some(PracticeLifecycleState::Stale)
    }
}

fn practice_state_adjustment(
    state: Option<PracticeLifecycleState>,
    status: ContinuityStatus,
    history_requested: bool,
) -> f64 {
    match (state, status, history_requested) {
        (Some(PracticeLifecycleState::Current), _, false) => 0.16,
        (Some(PracticeLifecycleState::Aging), _, false) => 0.03,
        (Some(PracticeLifecycleState::Stale), state, false) if state.is_open() => -0.38,
        (Some(PracticeLifecycleState::Stale), _, false) => -0.2,
        (Some(PracticeLifecycleState::Retired), state, false) if state.is_open() => -0.48,
        (Some(PracticeLifecycleState::Retired), _, false) => -0.34,
        (Some(PracticeLifecycleState::Current), _, true) => 0.05,
        (Some(PracticeLifecycleState::Aging), _, true) => 0.0,
        (Some(PracticeLifecycleState::Stale), _, true) => -0.02,
        (Some(PracticeLifecycleState::Retired), _, true) => -0.06,
        (None, _, _) => 0.0,
    }
}

fn practice_state_why(state: Option<PracticeLifecycleState>) -> Option<&'static str> {
    match state {
        Some(PracticeLifecycleState::Current) => Some("practice_current"),
        Some(PracticeLifecycleState::Aging) => Some("practice_aging"),
        Some(PracticeLifecycleState::Stale) => Some("practice_stale"),
        Some(PracticeLifecycleState::Retired) => Some("practice_retired"),
        None => None,
    }
}

fn continuity_practice_competition_rank(item: &ScoredContinuityRecallItem) -> f64 {
    let mut score = item.item.score;
    if let Some(plasticity) = item.plasticity.as_ref() {
        score += 0.04 * (plasticity.confirmation_count as f64 + 1.0).ln();
        score += 0.03 * (plasticity.successful_use_count as f64 + 1.0).ln();
        score += 0.05 * (plasticity.spaced_reactivation_count as f64 + 1.0).ln();
        score -= 0.08 * plasticity.contradiction_count as f64;
        score -= 0.12 * plasticity.prediction_error;
    }
    score
}

fn apply_practice_lifecycle_competition(
    items: &mut [ScoredContinuityRecallItem],
    history_requested: bool,
    now: DateTime<Utc>,
) {
    for item in items.iter_mut() {
        let state = derive_recall_practice_state(&item.item, item.plasticity.as_ref(), now);
        item.practice_state = state;
        item.item.score += practice_state_adjustment(state, item.item.status, history_requested);
        if let Some(why) = practice_state_why(state) {
            item.item.why.push(why.to_string());
        }
        if item.item.status.is_open()
            && matches!(
                state,
                Some(PracticeLifecycleState::Stale | PracticeLifecycleState::Retired)
            )
        {
            item.item.why.push("stale_open_guidance".to_string());
        }
    }

    if history_requested {
        return;
    }

    let mut practice_groups = HashMap::<String, Vec<usize>>::new();
    for (index, item) in items.iter().enumerate() {
        let Some(practice_key) = item.practice_key.as_ref() else {
            continue;
        };
        practice_groups
            .entry(practice_key.clone())
            .or_default()
            .push(index);
    }

    for indices in practice_groups.into_values() {
        if indices.len() < 2 {
            continue;
        }
        let mut ordered = indices;
        ordered.sort_by(|left, right| {
            continuity_practice_competition_rank(&items[*right])
                .partial_cmp(&continuity_practice_competition_rank(&items[*left]))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    items[*right]
                        .item
                        .updated_at
                        .cmp(&items[*left].item.updated_at)
                })
        });
        let winner_index = ordered[0];
        let winner_rank = continuity_practice_competition_rank(&items[winner_index]);
        let winner = &mut items[winner_index].item;
        winner.score += 0.08;
        winner.why.push("practice_key_winner".to_string());

        for loser_index in ordered.into_iter().skip(1) {
            let loser_rank = continuity_practice_competition_rank(&items[loser_index]);
            let penalty = (0.1 + (winner_rank - loser_rank).clamp(0.0, 0.18)).clamp(0.1, 0.24);
            let loser = &mut items[loser_index].item;
            loser.score -= penalty;
            loser.why.push("practice_key_competitor".to_string());
        }
    }
}

fn apply_belief_key_competition(items: &mut [ScoredContinuityRecallItem]) {
    let mut belief_groups = HashMap::<String, Vec<usize>>::new();
    for (index, item) in items.iter().enumerate() {
        let Some(belief_key) = item.belief_key.as_ref() else {
            continue;
        };
        belief_groups
            .entry(belief_key.clone())
            .or_default()
            .push(index);
    }

    for indices in belief_groups.into_values() {
        if indices.len() < 2 {
            continue;
        }
        let mut ordered = indices;
        ordered.sort_by(|left, right| {
            continuity_belief_competition_rank(&items[*right])
                .partial_cmp(&continuity_belief_competition_rank(&items[*left]))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    items[*right]
                        .item
                        .updated_at
                        .cmp(&items[*left].item.updated_at)
                })
        });
        let winner_index = ordered[0];
        let winner_rank = continuity_belief_competition_rank(&items[winner_index]);
        let runner_up_rank = ordered
            .get(1)
            .map(|index| continuity_belief_competition_rank(&items[*index]))
            .unwrap_or(winner_rank);
        let winner_bonus =
            (0.08 + (winner_rank - runner_up_rank).max(0.0).min(0.12)).clamp(0.08, 0.2);
        let winner_source_role = items[winner_index].source_role.clone();
        let winner = &mut items[winner_index].item;
        winner.score += winner_bonus;
        winner.why.push("belief_key_winner".to_string());
        if let Some(source_role) = winner_source_role.as_deref() {
            winner.why.push(format!("source_role:{source_role}"));
        }

        for loser_index in ordered.into_iter().skip(1) {
            let loser_rank = continuity_belief_competition_rank(&items[loser_index]);
            let penalty = (0.1 + (winner_rank - loser_rank).max(0.0).min(0.18)).clamp(0.1, 0.28);
            let loser_source_role = items[loser_index].source_role.clone();
            let loser = &mut items[loser_index].item;
            loser.score -= penalty;
            loser.why.push("belief_key_competitor".to_string());
            if let Some(source_role) = loser_source_role.as_deref() {
                loser.why.push(format!("source_role:{source_role}"));
            }
        }
    }
}

fn rank_score(rank: Option<usize>, len: usize) -> f64 {
    rank.map(|value| {
        let len = len.max(1) as f64;
        1.0 - (value as f64 / len)
    })
    .unwrap_or(0.0)
}

fn continuity_preview(body: &str, max_chars: usize) -> String {
    let compact = body.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out = compact.chars().take(max_chars).collect::<String>();
    if compact.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}

fn compiler_chunk_body(band: &str, items: &[ContinuityItemRecord]) -> String {
    let mut body = format!("band {band}\n");
    let items_by_id = items
        .iter()
        .map(|item| (item.id.as_str(), item))
        .collect::<HashMap<_, _>>();
    let bundled_support_ids = items
        .iter()
        .flat_map(|item| item.supports.iter())
        .filter(|support| {
            support.support_type == "continuity"
                && items_by_id.contains_key(support.support_id.as_str())
        })
        .map(|support| support.support_id.clone())
        .collect::<HashSet<_>>();
    for item in items {
        let in_band_supports = item
            .supports
            .iter()
            .filter(|support| support.support_type == "continuity")
            .filter_map(|support| {
                items_by_id
                    .get(support.support_id.as_str())
                    .copied()
                    .map(|supported| (support, supported))
            })
            .collect::<Vec<_>>();
        if in_band_supports.is_empty() {
            if bundled_support_ids.contains(&item.id) {
                continue;
            }
            body.push_str(&compiler_standalone_body(item));
            continue;
        }
        body.push_str(&compiler_bundle_body(item, &in_band_supports));
    }
    body
}

fn compiler_rank_items_by_support_graph(
    mut items: Vec<ContinuityItemRecord>,
) -> Vec<ContinuityItemRecord> {
    if items.len() <= 1 {
        return items;
    }

    let item_ids = items
        .iter()
        .map(|item| item.id.as_str())
        .collect::<HashSet<_>>();
    let original_order = items
        .iter()
        .enumerate()
        .map(|(idx, item)| (item.id.clone(), idx))
        .collect::<HashMap<_, _>>();
    let mut outgoing_support_scores = HashMap::<String, f64>::new();
    let mut incoming_support_scores = HashMap::<String, f64>::new();

    for item in &items {
        for support in &item.supports {
            if support.support_type != "continuity"
                || !item_ids.contains(support.support_id.as_str())
            {
                continue;
            }
            let weight = support.weight.clamp(0.0, 1.5);
            *outgoing_support_scores
                .entry(item.id.clone())
                .or_insert(0.0) += weight;
            *incoming_support_scores
                .entry(support.support_id.clone())
                .or_insert(0.0) += weight;
        }
    }

    let score_of = |id: &str| -> (bool, f64) {
        let outgoing = outgoing_support_scores.get(id).copied().unwrap_or_default();
        let incoming = incoming_support_scores.get(id).copied().unwrap_or_default();
        (outgoing > 0.0, outgoing * 1.2 + incoming * 0.35)
    };

    items.sort_by(|left, right| {
        let (left_anchor, left_graph) = score_of(&left.id);
        let (right_anchor, right_graph) = score_of(&right.id);

        right_anchor
            .cmp(&left_anchor)
            .then_with(|| {
                right_graph
                    .partial_cmp(&left_graph)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                right
                    .salience
                    .partial_cmp(&left.salience)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| right.updated_at.cmp(&left.updated_at))
            .then_with(|| {
                original_order
                    .get(&left.id)
                    .cmp(&original_order.get(&right.id))
            })
    });

    items
}

fn compiler_bundle_body(
    item: &ContinuityItemRecord,
    supports: &[(&SupportRef, &ContinuityItemRecord)],
) -> String {
    let mut body = format!(
        "[{}] {}\nstatus {} class {}\nrationale: {}\nsupported-by:\n",
        item.kind.as_str().to_uppercase(),
        item.title,
        item.status.as_str(),
        item.retention.class,
        compiler_compact_text(&item.body)
    );
    for (support_ref, support_item) in supports {
        body.push_str("  - ");
        body.push_str(&format!(
            "[{}] {} | status {} class {}",
            support_item.kind.as_str().to_uppercase(),
            support_item.title,
            support_item.status.as_str(),
            support_item.retention.class,
        ));
        if let Some(reason) = support_ref.reason.as_deref().map(str::trim) {
            if !reason.is_empty() {
                body.push_str(&format!(" | reason {reason}"));
            }
        }
        body.push_str(&format!(" | weight {:.2}", support_ref.weight));
        body.push_str(&format!(
            " | {}\n",
            compiler_compact_text(&support_item.body)
        ));
    }
    body.push('\n');
    body
}

fn compiler_standalone_body(item: &ContinuityItemRecord) -> String {
    format!(
        "[{}] {}\nstatus {} class {}\ndetail: {}\n\n",
        item.kind.as_str().to_uppercase(),
        item.title,
        item.status.as_str(),
        item.retention.class,
        compiler_compact_text(&item.body)
    )
}

fn compiler_compact_text(body: &str) -> String {
    body.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn continuity_recall_support_scores(
    support_map: &HashMap<String, Vec<SupportRef>>,
    seed_ids: &HashSet<String>,
) -> (HashMap<String, f64>, HashMap<String, f64>) {
    let mut anchor_scores = HashMap::<String, f64>::new();
    let mut support_scores = HashMap::<String, f64>::new();

    for (anchor_id, supports) in support_map {
        if !seed_ids.contains(anchor_id) {
            continue;
        }
        let mut anchor_total = 0.0;
        for support in supports {
            if support.support_type != "continuity" || !seed_ids.contains(&support.support_id) {
                continue;
            }
            let weight = support.weight.clamp(0.0, 1.5);
            anchor_total += weight * 0.12;
            *support_scores
                .entry(support.support_id.clone())
                .or_insert(0.0) += weight * 0.1;
        }
        if anchor_total > 0.0 {
            anchor_scores.insert(anchor_id.clone(), anchor_total.min(0.36));
        }
    }

    for score in support_scores.values_mut() {
        *score = score.min(0.3);
    }

    (anchor_scores, support_scores)
}

fn continuity_recall_summary(query: &str, items: &[ContinuityRecallItem]) -> String {
    if items.is_empty() {
        return format!("No high-signal continuity matched \"{query}\".");
    }
    let top = items
        .iter()
        .take(3)
        .map(|item| format!("{}: {}", item.kind.as_str(), item.title))
        .collect::<Vec<_>>()
        .join(" | ");
    format!("Top continuity recall for \"{query}\": {top}")
}

fn continuity_recall_answer_hint(items: &[ContinuityRecallItem]) -> Option<String> {
    let top = items.first()?;
    if top.effective_salience < 0.72 || top.score < 1.0 {
        return None;
    }
    if matches!(
        top.kind,
        ContinuityKind::WorkingState | ContinuityKind::Signal | ContinuityKind::WorkClaim
    ) {
        return None;
    }
    if let Some(next) = items.get(1) {
        if (top.score - next.score) < 0.1 {
            return None;
        }
    }
    let answer = if matches!(
        top.kind,
        ContinuityKind::Fact | ContinuityKind::Derivation | ContinuityKind::Lesson
    ) {
        top.preview.trim()
    } else {
        top.title.trim()
    };
    if answer.is_empty() {
        None
    } else {
        Some(answer.to_string())
    }
}

fn read_event_row(row: &rusqlite::Row<'_>) -> Result<EventRecord> {
    let kind = match row.get::<_, String>(2)?.as_str() {
        "prompt" => crate::model::EventKind::Prompt,
        "response" => crate::model::EventKind::Response,
        "tool_call" => crate::model::EventKind::ToolCall,
        "tool_result" => crate::model::EventKind::ToolResult,
        "shell_command" => crate::model::EventKind::ShellCommand,
        "shell_output" => crate::model::EventKind::ShellOutput,
        "file_diff" => crate::model::EventKind::FileDiff,
        "error" => crate::model::EventKind::Error,
        "exception" => crate::model::EventKind::Exception,
        "document" => crate::model::EventKind::Document,
        "trace" => crate::model::EventKind::Trace,
        "api_request" => crate::model::EventKind::ApiRequest,
        "api_response" => crate::model::EventKind::ApiResponse,
        "note" => crate::model::EventKind::Note,
        value => return Err(anyhow!("unknown event kind {value}")),
    };
    let scope = match row.get::<_, String>(3)?.as_str() {
        "agent" => crate::model::Scope::Agent,
        "session" => crate::model::Scope::Session,
        "shared" => crate::model::Scope::Shared,
        "project" => crate::model::Scope::Project,
        "global" => crate::model::Scope::Global,
        value => return Err(anyhow!("unknown scope {value}")),
    };
    Ok(EventRecord {
        id: row.get(0)?,
        ts: DateTime::parse_from_rfc3339(&row.get::<_, String>(1)?)?.with_timezone(&Utc),
        input: EventInput {
            kind,
            scope,
            agent_id: row.get(4)?,
            agent_role: row.get(5)?,
            timestamp: Some(
                DateTime::parse_from_rfc3339(&row.get::<_, String>(1)?)?.with_timezone(&Utc),
            ),
            session_id: row.get(6)?,
            task_id: row.get(7)?,
            project_id: row.get(8)?,
            goal_id: row.get(9)?,
            run_id: row.get(10)?,
            namespace: row.get(11)?,
            environment: row.get(12)?,
            source: row.get(13)?,
            tags: serde_json::from_str(&row.get::<_, String>(14)?)?,
            dimensions: serde_json::from_str(&row.get::<_, String>(15)?)?,
            attributes: serde_json::from_str(&row.get::<_, String>(16)?)?,
            content: row.get(17)?,
        },
        content_hash: row.get(18)?,
        byte_size: row.get::<_, i64>(19)? as usize,
        token_estimate: row.get::<_, i64>(20)? as usize,
        importance: row.get(21)?,
        segment_seq: row.get(22)?,
        segment_line: row.get(23)?,
    })
}

fn hot_body(event: &EventRecord) -> String {
    let preview = trim_for_preview(&event.input.content, 1_024);
    format!(
        "[{}:{}:{}] {}",
        event.input.source,
        event.input.kind,
        event.ts.to_rfc3339(),
        preview
    )
}

fn episode_fragment(event: &EventRecord) -> String {
    format!(
        "- {} {} {}",
        event.ts.format("%H:%M:%S"),
        event.input.kind,
        trim_for_preview(&event.input.content, 512)
    )
}

fn merge_episode_body(current: &str, fragment: &str, max_chars: usize) -> String {
    let mut merged = if current.is_empty() {
        fragment.to_string()
    } else if current.contains(fragment) {
        current.to_string()
    } else {
        format!("{current}\n{fragment}")
    };
    if merged.len() <= max_chars {
        return merged;
    }
    let keep = merged.len() - max_chars;
    merged.drain(..keep);
    merged
}

fn infer_importance(input: &EventInput) -> f64 {
    let mut score = 0.45_f64;
    if matches!(
        input.kind,
        crate::model::EventKind::Error | crate::model::EventKind::Exception
    ) {
        score += 0.35;
    }
    if input.content.contains("must")
        || input.content.contains("never")
        || input.content.contains("non-negotiable")
    {
        score += 0.20;
    }
    if input.content.contains("TODO") || input.content.contains("FIXME") {
        score += 0.10;
    }
    score.min(1.0)
}

fn estimate_tokens(value: &str) -> usize {
    (value.len() / 4).max(1)
}

fn trim_for_preview(value: &str, max_chars: usize) -> String {
    if value.len() <= max_chars {
        return value.to_string();
    }
    let trimmed: String = value.chars().take(max_chars).collect();
    format!("{trimmed} …")
}

fn compact_metric_label(value: &str, max_chars: usize) -> String {
    let single_line = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if single_line.chars().count() <= max_chars {
        return single_line;
    }
    let keep: String = single_line.chars().take(max_chars).collect();
    format!("{keep}...")
}

fn semantic_body(event: &EventRecord, entities: &[SemanticEntity]) -> String {
    let mut lines = vec![
        format!("kind={}", event.input.kind),
        format!("source={}", event.input.source),
        format!("scope={}", event.input.scope),
    ];
    if !event.input.tags.is_empty() {
        lines.push(format!("tags={}", event.input.tags.join(",")));
    }
    if let Some(agent_role) = &event.input.agent_role {
        lines.push(format!("agent_role={agent_role}"));
    }
    if let Some(task_id) = &event.input.task_id {
        lines.push(format!("task={task_id}"));
    }
    if let Some(project_id) = &event.input.project_id {
        lines.push(format!("project={project_id}"));
    }
    if let Some(goal_id) = &event.input.goal_id {
        lines.push(format!("goal={goal_id}"));
    }
    if let Some(run_id) = &event.input.run_id {
        lines.push(format!("run={run_id}"));
    }
    if let Some(namespace) = &event.input.namespace {
        lines.push(format!("namespace={namespace}"));
    }
    if let Some(environment) = &event.input.environment {
        lines.push(format!("environment={environment}"));
    }
    for entity in entities {
        lines.push(format!("entity:{}={}", entity.kind, entity.value));
    }
    lines.push(format!(
        "content={}",
        trim_for_preview(&event.input.content, 320)
    ));
    lines.join("\n")
}

fn summary_fragment(event: &EventRecord, entities: &[SemanticEntity]) -> String {
    let entity_rollup = entities
        .iter()
        .take(4)
        .map(|entity| format!("{}:{}", entity.kind, entity.value))
        .collect::<Vec<_>>()
        .join(", ");
    if entity_rollup.is_empty() {
        format!(
            "- {} {} {}",
            event.ts.format("%H:%M:%S"),
            event.input.kind,
            trim_for_preview(&event.input.content, 200)
        )
    } else {
        format!(
            "- {} {} {} | entities: {}",
            event.ts.format("%H:%M:%S"),
            event.input.kind,
            trim_for_preview(&event.input.content, 180),
            entity_rollup
        )
    }
}

fn extract_entities(event: &EventRecord) -> Vec<SemanticEntity> {
    let mut hits = Vec::new();
    let content = &event.input.content;

    for capture in path_regex().captures_iter(content) {
        if let Some(value) = capture.get(1) {
            hits.push(entity("path", value.as_str(), 0.9));
        }
    }
    for capture in url_regex().captures_iter(content) {
        if let Some(value) = capture.get(0) {
            hits.push(entity("url", value.as_str(), 0.8));
        }
    }
    for capture in backtick_regex().captures_iter(content) {
        if let Some(value) = capture.get(1) {
            hits.push(entity("code", value.as_str(), 0.75));
        }
    }
    for capture in error_regex().captures_iter(content) {
        if let Some(value) = capture.get(0) {
            hits.push(entity("error", value.as_str(), 0.95));
        }
    }
    for tag in &event.input.tags {
        hits.push(entity("tag", tag, 0.5));
    }
    if let Some(task_id) = &event.input.task_id {
        hits.push(entity("task", task_id, 0.7));
    }

    hits.sort_by(|a, b| a.normalized.cmp(&b.normalized));
    hits.dedup_by(|a, b| a.normalized == b.normalized && a.kind == b.kind);
    hits.truncate(12);
    hits
}

fn entity(kind: &str, value: &str, weight: f64) -> SemanticEntity {
    SemanticEntity {
        value: value.to_string(),
        normalized: value
            .trim_matches(|c: char| {
                !c.is_alphanumeric() && c != '/' && c != '.' && c != '_' && c != '-'
            })
            .to_lowercase(),
        kind: kind.to_string(),
        weight,
    }
}

fn path_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"(?i)(?:^|[\s(])([./~]?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)").unwrap()
    })
}

fn url_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"https?://[^\s)]+").unwrap())
}

fn backtick_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"`([^`]{2,128})`").unwrap())
}

fn error_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"\b[A-Z][A-Za-z0-9]*(?:Error|Exception|Failure|Timeout)\b").unwrap()
    })
}

fn encode_vector(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len() * std::mem::size_of::<f32>());
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub(crate) fn decode_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn fts_query(query_text: &str) -> String {
    query_text
        .split(|c: char| !(c.is_alphanumeric() || matches!(c, '/' | '.' | '_' | '-')))
        .filter(|token| token.len() > 1)
        .map(|token| format!("\"{}\"", token.to_lowercase().replace('"', "")))
        .collect::<Vec<_>>()
        .join(" OR ")
}

fn query_entities(query_text: &str) -> Vec<String> {
    query_text
        .split(|c: char| !(c.is_alphanumeric() || matches!(c, '/' | '.' | '_' | '-')))
        .filter(|token| token.len() > 2)
        .map(|token| token.to_lowercase())
        .collect::<Vec<_>>()
}

fn prometheus_label_value(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}

pub(crate) fn append_storage_bytes_metric_text(
    text: &mut String,
    sqlite_path: &Path,
    log_dir: &Path,
) -> Result<()> {
    let storage_bytes = storage_bytes_from_paths(sqlite_path, log_dir)?;
    text.push_str(&format!(
        "# HELP ice_storage_bytes On-disk storage used by SQLite and raw log segments.\n# TYPE ice_storage_bytes gauge\nice_storage_bytes {}\n",
        storage_bytes
    ));
    Ok(())
}

fn storage_bytes_from_paths(sqlite_path: &Path, log_dir: &Path) -> Result<u64> {
    #[cfg(test)]
    if let Some(hook) = {
        let mut hook = STORAGE_BYTES_METRICS_TEST_HOOK.lock().unwrap();
        if hook
            .as_ref()
            .is_some_and(|hook| hook.log_dir.as_path() == log_dir)
        {
            hook.take()
        } else {
            None
        }
    } {
        let _ = hook.entered_tx.send(());
        let _ = hook.release_rx.recv();
    }

    let mut total = fs::metadata(sqlite_path).map(|m| m.len()).unwrap_or(0);
    for entry in fs::read_dir(log_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            total += entry.metadata()?.len();
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use chrono::Duration;
    use tempfile::tempdir;

    use super::*;
    use crate::continuity::{
        CoordinationLane, CoordinationProjectedLane, CoordinationSeverity,
        merge_coordination_signal_extra,
    };
    use crate::embedding::EmbeddingBackendConfig;
    use crate::model::{EventInput, EventKind, QueryInput, Scope};
    use crate::telemetry::EngineTelemetry;

    #[test]
    fn ingest_creates_event_and_both_memory_layers() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let telemetry = EngineTelemetry::new();
        let mut storage = Storage::open(config).unwrap();
        let manifest = storage
            .ingest(
                EventInput {
                    kind: EventKind::Prompt,
                    agent_id: "agent-a".into(),
                    agent_role: None,
                    timestamp: None,
                    session_id: "session-a".into(),
                    task_id: Some("task-a".into()),
                    project_id: Some("project-a".into()),
                    goal_id: Some("goal-a".into()),
                    run_id: Some("run-a".into()),
                    namespace: Some("test".into()),
                    environment: Some("test".into()),
                    source: "cli".into(),
                    scope: Scope::Shared,
                    tags: vec!["demo".into()],
                    dimensions: vec![DimensionValue {
                        key: "claim.invariant".into(),
                        value: "remember-this".into(),
                        weight: 100,
                    }],
                    content: "remember this invariant".into(),
                    attributes: serde_json::json!({"turn": 1}),
                },
                &telemetry,
            )
            .unwrap();

        assert!(manifest.hot_memory_id.starts_with("hot:"));
        assert!(manifest.episodic_memory_id.starts_with("episode:"));
        assert_eq!(
            storage
                .list_memory(Some(MemoryLayer::Hot), 10)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            storage
                .list_memory(Some(MemoryLayer::Episodic), 10)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            storage
                .replay(&telemetry, Some("session-a"), 10)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn ingest_uses_explicit_timestamp_when_provided() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let telemetry = EngineTelemetry::new();
        let mut storage = Storage::open(config).unwrap();
        let timestamp = chrono::DateTime::parse_from_rfc3339("2023-04-10T23:07:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let manifest = storage
            .ingest(
                EventInput {
                    kind: EventKind::Document,
                    agent_id: "importer".into(),
                    agent_role: Some("importer".into()),
                    timestamp: Some(timestamp),
                    session_id: "historical-session".into(),
                    task_id: Some("historical-task".into()),
                    project_id: None,
                    goal_id: None,
                    run_id: None,
                    namespace: Some("history".into()),
                    environment: Some("longmemeval".into()),
                    source: "import".into(),
                    scope: Scope::Shared,
                    tags: vec!["history".into()],
                    dimensions: Vec::new(),
                    content: "Historical note".into(),
                    attributes: serde_json::json!({}),
                },
                &telemetry,
            )
            .unwrap();

        assert_eq!(manifest.event.ts, timestamp);
        let replay = storage
            .replay(&telemetry, Some("historical-session"), 10)
            .unwrap();
        assert_eq!(replay.len(), 1);
        assert_eq!(replay[0].event.ts, timestamp);
        assert_eq!(replay[0].event.input.timestamp, Some(timestamp));
    }

    #[test]
    fn continuity_priority_query_uses_salience_index() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();
        let mut stmt = storage
            .conn
            .prepare(
                r#"
                EXPLAIN QUERY PLAN
                SELECT id
                FROM continuity_items
                WHERE context_id = ?1 AND status IN ('open', 'active')
                ORDER BY salience DESC, updated_at DESC, ts DESC
                LIMIT ?2
                "#,
            )
            .unwrap();
        let rows = stmt
            .query_map(params!["ctx:test", 16_i64], |row| row.get::<_, String>(3))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert!(
            rows.iter()
                .any(|detail| detail.contains("idx_continuity_context_salience")
                    || detail.contains("idx_continuity_context_status_salience")),
            "query plan did not use a continuity salience index: {rows:?}"
        );
    }

    #[test]
    fn second_ingest_updates_episode_without_failing() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let telemetry = EngineTelemetry::new();
        let mut storage = Storage::open(config).unwrap();

        for kind in [EventKind::Prompt, EventKind::Error] {
            storage
                .ingest(
                    EventInput {
                        kind: kind.clone(),
                        agent_id: "agent-a".into(),
                        agent_role: None,
                        timestamp: None,
                        session_id: "session-a".into(),
                        task_id: Some("task-a".into()),
                        project_id: Some("project-a".into()),
                        goal_id: Some("goal-a".into()),
                        run_id: Some("run-a".into()),
                        namespace: Some("test".into()),
                        environment: Some("test".into()),
                        source: "cli".into(),
                        scope: Scope::Shared,
                        tags: vec!["demo".into()],
                        dimensions: Vec::new(),
                        content: format!("event for {kind}"),
                        attributes: serde_json::json!({}),
                    },
                    &telemetry,
                )
                .unwrap();
        }

        assert_eq!(
            storage
                .replay(&telemetry, Some("session-a"), 10)
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            storage
                .list_memory(Some(MemoryLayer::Hot), 10)
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            storage
                .list_memory(Some(MemoryLayer::Episodic), 10)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn materialized_view_detects_conflicting_claims() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let telemetry = EngineTelemetry::new();
        let mut storage = Storage::open(config).unwrap();

        for (agent, value) in [("planner", "healthy"), ("debugger", "degraded")] {
            storage
                .ingest(
                    EventInput {
                        kind: EventKind::Note,
                        agent_id: agent.into(),
                        agent_role: Some(agent.into()),
                        timestamp: None,
                        session_id: "session-a".into(),
                        task_id: Some("task-a".into()),
                        project_id: Some("project-a".into()),
                        goal_id: Some("goal-a".into()),
                        run_id: Some("run-a".into()),
                        namespace: Some("test".into()),
                        environment: Some("test".into()),
                        source: "cli".into(),
                        scope: Scope::Project,
                        tags: vec!["demo".into()],
                        dimensions: vec![
                            DimensionValue {
                                key: "endpoint".into(),
                                value: "/v1/demo".into(),
                                weight: 100,
                            },
                            DimensionValue {
                                key: "claim.api_status".into(),
                                value: value.into(),
                                weight: 100,
                            },
                        ],
                        content: format!("{agent} says api status is {value}"),
                        attributes: serde_json::json!({}),
                    },
                    &telemetry,
                )
                .unwrap();
        }

        let view = storage
            .materialize_view(ViewInput {
                op: ViewOp::Merge,
                owner_agent_id: Some("planner".into()),
                namespace: Some("test".into()),
                objective: Some("merge conflict".into()),
                selectors: vec![Selector {
                    all: vec![
                        DimensionFilter {
                            key: "project".into(),
                            values: vec!["project-a".into()],
                        },
                        DimensionFilter {
                            key: "endpoint".into(),
                            values: vec!["/v1/demo".into()],
                        },
                    ],
                    any: Vec::new(),
                    exclude: Vec::new(),
                    layers: Vec::new(),
                    start_ts: None,
                    end_ts: None,
                    limit: Some(8),
                    namespace: Some("test".into()),
                }],
                source_view_ids: Vec::new(),
                resolution: Some(SnapshotResolution::Fine),
                limit: Some(8),
            })
            .unwrap();

        assert_eq!(view.conflict_count, 1);
        assert!(view.item_count >= 2);
    }

    #[test]
    fn subscription_poll_returns_matching_items() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let telemetry = EngineTelemetry::new();
        let mut storage = Storage::open(config).unwrap();

        let subscription = storage
            .create_subscription(SubscriptionInput {
                agent_id: "watcher".into(),
                name: Some("project-watch".into()),
                selector: Selector {
                    all: vec![DimensionFilter {
                        key: "project".into(),
                        values: vec!["project-a".into()],
                    }],
                    any: Vec::new(),
                    exclude: Vec::new(),
                    layers: Vec::new(),
                    start_ts: None,
                    end_ts: None,
                    limit: Some(8),
                    namespace: Some("test".into()),
                },
            })
            .unwrap();

        storage
            .ingest(
                EventInput {
                    kind: EventKind::Note,
                    agent_id: "agent-a".into(),
                    agent_role: Some("planner".into()),
                    timestamp: None,
                    session_id: "session-a".into(),
                    task_id: Some("task-a".into()),
                    project_id: Some("project-a".into()),
                    goal_id: Some("goal-a".into()),
                    run_id: Some("run-a".into()),
                    namespace: Some("test".into()),
                    environment: Some("test".into()),
                    source: "cli".into(),
                    scope: Scope::Project,
                    tags: vec!["demo".into()],
                    dimensions: vec![DimensionValue {
                        key: "endpoint".into(),
                        value: "/v1/demo".into(),
                        weight: 100,
                    }],
                    content: "project-a update".into(),
                    attributes: serde_json::json!({}),
                },
                &telemetry,
            )
            .unwrap();

        let poll = storage.poll_subscription(&subscription.id, 8).unwrap();
        assert!(!poll.items.is_empty());
        assert!(
            poll.items
                .iter()
                .any(|item| item.body.contains("project-a update"))
        );
    }

    #[test]
    fn metrics_include_active_agent_presence() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let telemetry = EngineTelemetry::new();
        let mut storage = Storage::open(config).unwrap();

        let attachment = storage
            .attach_agent(AttachAgentInput {
                agent_id: "agent-a".into(),
                agent_type: "codex".into(),
                capabilities: vec!["read_context".into(), "write_events".into()],
                namespace: "test".into(),
                role: Some("planner".into()),
                metadata: serde_json::json!({"source": "test"}),
            })
            .unwrap();
        storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: "task-a".into(),
                session_id: "session-a".into(),
                objective: "observe agent presence".into(),
                selector: None,
                agent_id: Some("agent-a".into()),
                attachment_id: Some(attachment.id.clone()),
            })
            .unwrap();
        storage
            .ingest(
                EventInput {
                    kind: EventKind::Note,
                    agent_id: "agent-a".into(),
                    agent_role: Some("planner".into()),
                    timestamp: None,
                    session_id: "session-a".into(),
                    task_id: Some("task-a".into()),
                    project_id: Some("project-a".into()),
                    goal_id: Some("goal-a".into()),
                    run_id: Some("run-a".into()),
                    namespace: Some("test".into()),
                    environment: Some("test".into()),
                    source: "cli".into(),
                    scope: Scope::Project,
                    tags: vec!["presence".into()],
                    dimensions: Vec::new(),
                    content: "agent touched the brain".into(),
                    attributes: serde_json::json!({}),
                },
                &telemetry,
            )
            .unwrap();

        let metrics = storage.metrics_text(&telemetry).unwrap();
        let labels = r#"agent_id="agent-a",agent_type="codex",namespace="test",role="planner""#;
        assert!(metrics.contains(&format!("ice_agent_active{{{labels}}} 1")));
        assert!(metrics.contains(&format!("ice_agent_ticks_total{{{labels}}} 3")));
        assert!(metrics.contains(r#"ice_active_agents{namespace="test"} 1"#));
    }

    #[test]
    fn metrics_include_live_agent_badges() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let telemetry = EngineTelemetry::new();
        let storage = Storage::open(config).unwrap();

        let attachment = storage
            .attach_agent(AttachAgentInput {
                agent_id: "agent-a".into(),
                agent_type: "codex".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "test".into(),
                role: Some("planner".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo"}),
            })
            .unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: "task-a".into(),
                session_id: "session-a".into(),
                objective: "show the live badge".into(),
                selector: None,
                agent_id: Some("agent-a".into()),
                attachment_id: Some(attachment.id.clone()),
            })
            .unwrap();
        storage
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(attachment.id),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id),
                display_name: Some("Agent A".into()),
                status: Some("writing".into()),
                focus: Some("rewiring src/storage.rs".into()),
                headline: Some("badge registry".into()),
                resource: Some("src/storage.rs".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({"source": "test"}),
            })
            .unwrap();

        let metrics = storage.metrics_text(&telemetry).unwrap();
        assert!(metrics.contains(r#"ice_agent_badge_connected{attachment_id="attach:"#));
        assert!(metrics.contains(r#"agent_id="agent-a""#));
        assert!(metrics.contains(r#"display_name="Agent A""#));
        assert!(metrics.contains(r#"focus="rewiring src/storage.rs""#));
        assert!(metrics.contains(r#"resource="src/storage.rs""#));
        assert!(metrics.contains(r#"branch="main""#));
    }

    #[test]
    fn metrics_include_lane_projections() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let telemetry = EngineTelemetry::new();
        let storage = Storage::open(config).unwrap();

        let attachment = storage
            .attach_agent(AttachAgentInput {
                agent_id: "agent-a".into(),
                agent_type: "codex".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "test".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo", "branch": "main"}),
            })
            .unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: DEFAULT_MACHINE_TASK_ID.into(),
                session_id: "session-projection".into(),
                objective: "show live lane projections".into(),
                selector: None,
                agent_id: Some("agent-a".into()),
                attachment_id: Some(attachment.id.clone()),
            })
            .unwrap();
        storage
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(attachment.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Agent A".into()),
                status: Some("syncing".into()),
                focus: Some("project repo lanes".into()),
                headline: Some("machine-first repo lane".into()),
                resource: Some("repo/demo/main".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({"source": "test"}),
            })
            .unwrap();
        storage
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "agent-a".into(),
                title: "Own repo lane".into(),
                body: "Project the repo lane through the machine brain.".into(),
                scope: Scope::Shared,
                resources: vec!["repo/demo/main".into()],
                exclusive: true,
                attachment_id: Some(attachment.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();
        storage
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "agent-b".into(),
                title: "Collide on repo lane".into(),
                body: "Test projection conflict counting.".into(),
                scope: Scope::Shared,
                resources: vec!["repo/demo/main".into()],
                exclusive: true,
                attachment_id: None,
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();
        storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id,
                author_agent_id: "boundary-warden".into(),
                kind: ContinuityKind::Signal,
                title: "Back off the repo lane".into(),
                body: "The repo lane is under active coordination pressure.".into(),
                scope: Scope::Shared,
                status: Some(ContinuityStatus::Active),
                importance: None,
                confidence: None,
                salience: None,
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: merge_coordination_signal_extra(
                    serde_json::json!({}),
                    CoordinationLane::Backoff,
                    CoordinationSeverity::Block,
                    None,
                    Some(CoordinationProjectedLane {
                        projection_id: "repo:/tmp/demo:main".into(),
                        projection_kind: "repo".into(),
                        label: "demo @ main".into(),
                        resource: Some("repo/demo/main".into()),
                        repo_root: Some("/tmp/demo".into()),
                        branch: Some("main".into()),
                        task_id: Some(DEFAULT_MACHINE_TASK_ID.into()),
                    }),
                    None,
                    Some("repo/demo/main".into()),
                    vec!["repo:/tmp/demo:main".into()],
                    vec![CoordinationProjectedLane {
                        projection_id: "repo:/tmp/demo:main".into(),
                        projection_kind: "repo".into(),
                        label: "demo @ main".into(),
                        resource: Some("repo/demo/main".into()),
                        repo_root: Some("/tmp/demo".into()),
                        branch: Some("main".into()),
                        task_id: Some(DEFAULT_MACHINE_TASK_ID.into()),
                    }],
                ),
            })
            .unwrap();

        let metrics = storage.metrics_text(&telemetry).unwrap();
        assert!(metrics.contains(r#"ice_lane_projection_agents{namespace="test""#));
        assert!(metrics.contains(r#"projection_kind="repo""#));
        assert!(metrics.contains(r#"resource="repo/demo/main""#));
        assert!(metrics.contains(r#"branch="main""#));
        assert!(metrics.contains(
            r#"ice_lane_projection_agents{namespace="test",projection_id="repo:/tmp/demo:main""#
        ));
        assert!(metrics.contains(r#"} 1"#));
        assert!(metrics.contains(
            r#"ice_lane_projection_claims{namespace="test",projection_id="repo:/tmp/demo:main""#
        ));
        assert!(metrics.contains(r#"ice_lane_projection_claims{namespace="test",projection_id="repo:/tmp/demo:main",projection_kind="repo",label="demo @ main",resource="repo/demo/main",repo_root="/tmp/demo",branch="main",task_id="machine-organism"} 2"#));
        assert!(metrics.contains(r#"ice_lane_projection_conflicts{namespace="test",projection_id="repo:/tmp/demo:main",projection_kind="repo",label="demo @ main",resource="repo/demo/main",repo_root="/tmp/demo",branch="main",task_id="machine-organism"} 1"#));
        assert!(metrics.contains(r#"ice_lane_projection_coordination_signals{namespace="test",projection_id="repo:/tmp/demo:main",projection_kind="repo",label="demo @ main",resource="repo/demo/main",repo_root="/tmp/demo",branch="main",task_id="machine-organism"} 1"#));
        assert!(metrics.contains(r#"ice_lane_projection_blocking_signals{namespace="test",projection_id="repo:/tmp/demo:main",projection_kind="repo",label="demo @ main",resource="repo/demo/main",repo_root="/tmp/demo",branch="main",task_id="machine-organism"} 1"#));
        assert!(metrics.contains(r#"ice_lane_projection_review_signals{namespace="test",projection_id="repo:/tmp/demo:main",projection_kind="repo",label="demo @ main",resource="repo/demo/main",repo_root="/tmp/demo",branch="main",task_id="machine-organism"} 0"#));
    }

    #[test]
    fn metrics_text_reuses_one_live_state_snapshot() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let telemetry = EngineTelemetry::new();
        let storage = Storage::open(config).unwrap();

        let attachment = storage
            .attach_agent(AttachAgentInput {
                agent_id: "agent-a".into(),
                agent_type: "codex".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "test".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo", "branch": "main"}),
            })
            .unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: DEFAULT_MACHINE_TASK_ID.into(),
                session_id: "session-snapshot".into(),
                objective: "count live state metric reads".into(),
                selector: None,
                agent_id: Some("agent-a".into()),
                attachment_id: Some(attachment.id.clone()),
            })
            .unwrap();
        storage
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(attachment.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Agent A".into()),
                status: Some("syncing".into()),
                focus: Some("dedupe live state reads".into()),
                headline: Some("metrics snapshot".into()),
                resource: Some("repo/demo/main".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({"source": "test"}),
            })
            .unwrap();
        storage
            .claim_work(ClaimWorkInput {
                context_id: context.id,
                agent_id: "agent-a".into(),
                title: "Own repo lane".into(),
                body: "Exercise live state snapshot reuse.".into(),
                scope: Scope::Shared,
                resources: vec!["repo/demo/main".into()],
                exclusive: true,
                attachment_id: Some(attachment.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();

        Storage::install_metrics_live_state_query_counter(dir.path().to_path_buf());
        let _ = storage.metrics_text(&telemetry).unwrap();
        let counts = Storage::metrics_live_state_query_counts();
        assert_eq!(
            counts,
            (1, 1, 0, 0),
            "expected one attachment read and one shared continuity snapshot during metrics snapshot"
        );
    }

    #[test]
    fn lane_projections_keep_same_repo_branch_worktrees_separate() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();

        let attachment_a = storage
            .attach_agent(AttachAgentInput {
                agent_id: "agent-a".into(),
                agent_type: "codex".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "test".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/worktrees/adhd", "branch": "main"}),
            })
            .unwrap();
        let attachment_b = storage
            .attach_agent(AttachAgentInput {
                agent_id: "agent-b".into(),
                agent_type: "codex".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "test".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/shadow/adhd", "branch": "main"}),
            })
            .unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: DEFAULT_MACHINE_TASK_ID.into(),
                session_id: "session-projection".into(),
                objective: "keep same-branch worktrees separate".into(),
                selector: None,
                agent_id: Some("agent-a".into()),
                attachment_id: Some(attachment_a.id.clone()),
            })
            .unwrap();
        storage
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(attachment_a.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Agent A".into()),
                status: Some("syncing".into()),
                focus: Some("watch main worktree a".into()),
                headline: Some("first worktree".into()),
                resource: Some("repo/adhd/main".into()),
                repo_root: Some("/tmp/worktrees/adhd".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({"source": "test"}),
            })
            .unwrap();
        storage
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(attachment_b.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Agent B".into()),
                status: Some("syncing".into()),
                focus: Some("watch main worktree b".into()),
                headline: Some("second worktree".into()),
                resource: Some("repo/adhd/main".into()),
                repo_root: Some("/tmp/shadow/adhd".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({"source": "test"}),
            })
            .unwrap();
        storage
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "agent-a".into(),
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
        storage
            .claim_work(ClaimWorkInput {
                context_id: context.id,
                agent_id: "agent-b".into(),
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

        let projections = storage
            .list_lane_projections(Some("test"), Some(DEFAULT_MACHINE_TASK_ID))
            .unwrap()
            .into_iter()
            .filter(|projection| projection.projection_kind == "repo")
            .collect::<Vec<_>>();
        assert_eq!(projections.len(), 2);
        assert!(projections.iter().any(|projection| {
            projection.projection_id == "repo:/tmp/worktrees/adhd:main"
                && projection.repo_root.as_deref() == Some("/tmp/worktrees/adhd")
                && projection.branch.as_deref() == Some("main")
                && projection.connected_agents == 1
                && projection.live_claims == 1
                && projection.claim_conflicts == 0
        }));
        assert!(projections.iter().any(|projection| {
            projection.projection_id == "repo:/tmp/shadow/adhd:main"
                && projection.repo_root.as_deref() == Some("/tmp/shadow/adhd")
                && projection.branch.as_deref() == Some("main")
                && projection.connected_agents == 1
                && projection.live_claims == 1
                && projection.claim_conflicts == 0
        }));
    }

    #[test]
    fn lane_projections_keep_multiple_repo_lanes_separate() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();

        let attachment_a = storage
            .attach_agent(AttachAgentInput {
                agent_id: "agent-a".into(),
                agent_type: "codex".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "test".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo", "branch": "main"}),
            })
            .unwrap();
        let attachment_b = storage
            .attach_agent(AttachAgentInput {
                agent_id: "agent-b".into(),
                agent_type: "codex".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "test".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo", "branch": "feature/shadow"}),
            })
            .unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: DEFAULT_MACHINE_TASK_ID.into(),
                session_id: "session-projection".into(),
                objective: "show separate repo worktree lanes".into(),
                selector: None,
                agent_id: Some("agent-a".into()),
                attachment_id: Some(attachment_a.id.clone()),
            })
            .unwrap();
        storage
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(attachment_a.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Agent A".into()),
                status: Some("syncing".into()),
                focus: Some("watch main lane".into()),
                headline: Some("main worktree".into()),
                resource: Some("repo/demo/main".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({"source": "test"}),
            })
            .unwrap();
        storage
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(attachment_b.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Agent B".into()),
                status: Some("syncing".into()),
                focus: Some("watch feature lane".into()),
                headline: Some("feature worktree".into()),
                resource: Some("repo/demo/feature/shadow".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("feature/shadow".into()),
                metadata: serde_json::json!({"source": "test"}),
            })
            .unwrap();
        storage
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "agent-a".into(),
                title: "Hold main lane".into(),
                body: "Project main worktree lane.".into(),
                scope: Scope::Shared,
                resources: vec!["repo/demo/main".into()],
                exclusive: true,
                attachment_id: Some(attachment_a.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();
        storage
            .claim_work(ClaimWorkInput {
                context_id: context.id,
                agent_id: "agent-b".into(),
                title: "Hold feature lane".into(),
                body: "Project feature worktree lane.".into(),
                scope: Scope::Shared,
                resources: vec!["repo/demo/feature/shadow".into()],
                exclusive: true,
                attachment_id: Some(attachment_b.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let projections = storage
            .list_lane_projections(Some("test"), Some(DEFAULT_MACHINE_TASK_ID))
            .unwrap()
            .into_iter()
            .filter(|projection| projection.projection_kind == "repo")
            .collect::<Vec<_>>();
        assert_eq!(projections.len(), 2);
        assert!(projections.iter().any(|projection| {
            projection.label == "demo @ main"
                && projection.connected_agents == 1
                && projection.live_claims == 1
                && projection.claim_conflicts == 0
        }));
        assert!(projections.iter().any(|projection| {
            projection.label == "demo @ feature/shadow"
                && projection.connected_agents == 1
                && projection.live_claims == 1
                && projection.claim_conflicts == 0
        }));
    }

    #[test]
    fn lane_projections_surface_cross_lane_file_conflicts() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();

        let attachment_a = storage
            .attach_agent(AttachAgentInput {
                agent_id: "agent-a".into(),
                agent_type: "codex".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "test".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo", "branch": "main"}),
            })
            .unwrap();
        let attachment_b = storage
            .attach_agent(AttachAgentInput {
                agent_id: "agent-b".into(),
                agent_type: "codex".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "test".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo", "branch": "feature/shadow"}),
            })
            .unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: DEFAULT_MACHINE_TASK_ID.into(),
                session_id: "session-projection-conflict".into(),
                objective: "surface cross-lane file conflicts".into(),
                selector: None,
                agent_id: Some("agent-a".into()),
                attachment_id: Some(attachment_a.id.clone()),
            })
            .unwrap();
        storage
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(attachment_a.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Agent A".into()),
                status: Some("editing".into()),
                focus: Some("touch src/storage.rs".into()),
                headline: Some("main worktree".into()),
                resource: Some("repo/demo/main".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({"source": "test"}),
            })
            .unwrap();
        storage
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(attachment_b.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Agent B".into()),
                status: Some("editing".into()),
                focus: Some("touch src/storage.rs".into()),
                headline: Some("feature worktree".into()),
                resource: Some("repo/demo/feature/shadow".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("feature/shadow".into()),
                metadata: serde_json::json!({"source": "test"}),
            })
            .unwrap();
        storage
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "agent-a".into(),
                title: "Edit storage from main".into(),
                body: "Mutate src/storage.rs from main.".into(),
                scope: Scope::Shared,
                resources: vec!["file/src/storage.rs".into()],
                exclusive: true,
                attachment_id: Some(attachment_a.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();
        storage
            .claim_work(ClaimWorkInput {
                context_id: context.id,
                agent_id: "agent-b".into(),
                title: "Edit storage from feature".into(),
                body: "Mutate src/storage.rs from feature.".into(),
                scope: Scope::Shared,
                resources: vec!["file/src/storage.rs".into()],
                exclusive: true,
                attachment_id: Some(attachment_b.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let projections = storage
            .list_lane_projections(Some("test"), Some(DEFAULT_MACHINE_TASK_ID))
            .unwrap()
            .into_iter()
            .filter(|projection| projection.projection_kind == "repo")
            .collect::<Vec<_>>();
        assert_eq!(projections.len(), 2);
        assert!(projections.iter().any(|projection| {
            projection.label == "demo @ main"
                && projection.connected_agents == 1
                && projection.live_claims == 1
                && projection.claim_conflicts == 1
        }));
        assert!(projections.iter().any(|projection| {
            projection.label == "demo @ feature/shadow"
                && projection.connected_agents == 1
                && projection.live_claims == 1
                && projection.claim_conflicts == 1
        }));
    }

    #[test]
    fn lane_projections_materialize_signal_only_pressure() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: DEFAULT_MACHINE_TASK_ID.into(),
                session_id: "session-signals".into(),
                objective: "surface coordination-only lane pressure".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();
        storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id,
                author_agent_id: "reviewer".into(),
                kind: ContinuityKind::Signal,
                title: "Review src/storage.rs".into(),
                body: "The storage lane needs explicit human review.".into(),
                scope: Scope::Shared,
                status: Some(ContinuityStatus::Active),
                importance: None,
                confidence: None,
                salience: None,
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: merge_coordination_signal_extra(
                    serde_json::json!({}),
                    CoordinationLane::Review,
                    CoordinationSeverity::Info,
                    None,
                    None,
                    None,
                    Some("src/storage.rs".into()),
                    Vec::new(),
                    Vec::new(),
                ),
            })
            .unwrap();

        let projections = storage
            .list_lane_projections(Some("test"), Some(DEFAULT_MACHINE_TASK_ID))
            .unwrap();

        assert!(projections.iter().any(|projection| {
            projection.projection_kind == "lane"
                && projection.label == "src/storage.rs"
                && projection.live_claims == 0
                && projection.claim_conflicts == 0
                && projection.coordination_signal_count == 1
                && projection.blocking_signal_count == 0
                && projection.review_signal_count == 1
                && projection.coordination_lanes == vec!["review".to_string()]
                && projection.focus == "Review src/storage.rs"
        }));
    }

    #[test]
    fn lane_projections_count_blocking_signal_pressure_by_target_lane() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();

        let attachment_a = storage
            .attach_agent(AttachAgentInput {
                agent_id: "agent-a".into(),
                agent_type: "codex".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "test".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo", "branch": "main"}),
            })
            .unwrap();
        let attachment_b = storage
            .attach_agent(AttachAgentInput {
                agent_id: "agent-b".into(),
                agent_type: "codex".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "test".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo", "branch": "feature/shadow"}),
            })
            .unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: DEFAULT_MACHINE_TASK_ID.into(),
                session_id: "session-projection-blocking-signal".into(),
                objective: "count blocking coordination pressure by target lane".into(),
                selector: None,
                agent_id: Some("agent-a".into()),
                attachment_id: Some(attachment_a.id.clone()),
            })
            .unwrap();
        storage
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(attachment_a.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Agent A".into()),
                status: Some("editing".into()),
                focus: Some("hold main lane".into()),
                headline: Some("main worktree".into()),
                resource: Some("repo/demo/main".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({"source": "test"}),
            })
            .unwrap();
        storage
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(attachment_b.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Agent B".into()),
                status: Some("editing".into()),
                focus: Some("hold shadow lane".into()),
                headline: Some("shadow worktree".into()),
                resource: Some("repo/demo/feature/shadow".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("feature/shadow".into()),
                metadata: serde_json::json!({"source": "test"}),
            })
            .unwrap();
        storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "boundary-warden".into(),
                kind: ContinuityKind::Signal,
                title: "Back off Agent A on src/storage.rs".into(),
                body: "Main lane is blocked until ownership is renegotiated.".into(),
                scope: Scope::Shared,
                status: Some(ContinuityStatus::Active),
                importance: None,
                confidence: None,
                salience: None,
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: merge_coordination_signal_extra(
                    serde_json::json!({}),
                    CoordinationLane::Backoff,
                    CoordinationSeverity::Block,
                    Some("agent-a".into()),
                    Some(CoordinationProjectedLane {
                        projection_id: "repo:/tmp/demo:main".into(),
                        projection_kind: "repo".into(),
                        label: "demo @ main".into(),
                        resource: Some("repo/demo/main".into()),
                        repo_root: Some("/tmp/demo".into()),
                        branch: Some("main".into()),
                        task_id: Some(DEFAULT_MACHINE_TASK_ID.into()),
                    }),
                    None,
                    Some("src/storage.rs".into()),
                    vec![
                        "repo:/tmp/demo:main".into(),
                        "repo:/tmp/demo:feature/shadow".into(),
                    ],
                    vec![
                        CoordinationProjectedLane {
                            projection_id: "repo:/tmp/demo:main".into(),
                            projection_kind: "repo".into(),
                            label: "demo @ main".into(),
                            resource: Some("repo/demo/main".into()),
                            repo_root: Some("/tmp/demo".into()),
                            branch: Some("main".into()),
                            task_id: Some(DEFAULT_MACHINE_TASK_ID.into()),
                        },
                        CoordinationProjectedLane {
                            projection_id: "repo:/tmp/demo:feature/shadow".into(),
                            projection_kind: "repo".into(),
                            label: "demo @ feature/shadow".into(),
                            resource: Some("repo/demo/feature/shadow".into()),
                            repo_root: Some("/tmp/demo".into()),
                            branch: Some("feature/shadow".into()),
                            task_id: Some(DEFAULT_MACHINE_TASK_ID.into()),
                        },
                    ],
                ),
            })
            .unwrap();
        storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id,
                author_agent_id: "review-coordinator".into(),
                kind: ContinuityKind::Signal,
                title: "Review Agent B on src/storage.rs".into(),
                body: "Shadow lane needs review before it keeps editing.".into(),
                scope: Scope::Shared,
                status: Some(ContinuityStatus::Active),
                importance: None,
                confidence: None,
                salience: None,
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: merge_coordination_signal_extra(
                    serde_json::json!({}),
                    CoordinationLane::Review,
                    CoordinationSeverity::Info,
                    Some("agent-b".into()),
                    Some(CoordinationProjectedLane {
                        projection_id: "repo:/tmp/demo:feature/shadow".into(),
                        projection_kind: "repo".into(),
                        label: "demo @ feature/shadow".into(),
                        resource: Some("repo/demo/feature/shadow".into()),
                        repo_root: Some("/tmp/demo".into()),
                        branch: Some("feature/shadow".into()),
                        task_id: Some(DEFAULT_MACHINE_TASK_ID.into()),
                    }),
                    None,
                    Some("src/storage.rs".into()),
                    vec![
                        "repo:/tmp/demo:main".into(),
                        "repo:/tmp/demo:feature/shadow".into(),
                    ],
                    vec![
                        CoordinationProjectedLane {
                            projection_id: "repo:/tmp/demo:main".into(),
                            projection_kind: "repo".into(),
                            label: "demo @ main".into(),
                            resource: Some("repo/demo/main".into()),
                            repo_root: Some("/tmp/demo".into()),
                            branch: Some("main".into()),
                            task_id: Some(DEFAULT_MACHINE_TASK_ID.into()),
                        },
                        CoordinationProjectedLane {
                            projection_id: "repo:/tmp/demo:feature/shadow".into(),
                            projection_kind: "repo".into(),
                            label: "demo @ feature/shadow".into(),
                            resource: Some("repo/demo/feature/shadow".into()),
                            repo_root: Some("/tmp/demo".into()),
                            branch: Some("feature/shadow".into()),
                            task_id: Some(DEFAULT_MACHINE_TASK_ID.into()),
                        },
                    ],
                ),
            })
            .unwrap();

        let projections = storage
            .list_lane_projections(Some("test"), Some(DEFAULT_MACHINE_TASK_ID))
            .unwrap()
            .into_iter()
            .filter(|projection| projection.projection_kind == "repo")
            .collect::<Vec<_>>();

        assert_eq!(projections.len(), 2);
        assert!(projections.iter().any(|projection| {
            projection.label == "demo @ main"
                && projection.coordination_signal_count == 1
                && projection.blocking_signal_count == 1
                && projection.review_signal_count == 0
                && projection.coordination_lanes == vec!["backoff".to_string()]
        }));
        assert!(projections.iter().any(|projection| {
            projection.label == "demo @ feature/shadow"
                && projection.coordination_signal_count == 1
                && projection.blocking_signal_count == 0
                && projection.review_signal_count == 1
                && projection.coordination_lanes == vec!["review".to_string()]
        }));
        let metrics = storage.metrics_text(&EngineTelemetry::new()).unwrap();
        assert!(metrics.contains(r#"ice_lane_projection_review_signals{namespace="test",projection_id="repo:/tmp/demo:main",projection_kind="repo",label="demo @ main",resource="repo/demo/main",repo_root="/tmp/demo",branch="main",task_id="machine-organism"} 0"#));
        assert!(metrics.contains(r#"ice_lane_projection_review_signals{namespace="test",projection_id="repo:/tmp/demo:feature/shadow",projection_kind="repo",label="demo @ feature/shadow",resource="repo/demo/feature/shadow",repo_root="/tmp/demo",branch="feature/shadow",task_id="machine-organism"} 1"#));
    }

    #[test]
    fn metrics_include_live_work_claim_pressure() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let telemetry = EngineTelemetry::new();
        let storage = Storage::open(config).unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: "task-claims".into(),
                session_id: "session-a".into(),
                objective: "surface live coordination pressure".into(),
                selector: None,
                agent_id: Some("planner".into()),
                attachment_id: None,
            })
            .unwrap();
        storage
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "planner".into(),
                title: "Own src/continuity.rs".into(),
                body: "Kernel rewrite in progress.".into(),
                scope: Scope::Project,
                resources: vec!["src/continuity.rs".into()],
                exclusive: true,
                attachment_id: None,
                lease_seconds: Some(180),
                extra: serde_json::json!({}),
            })
            .unwrap();
        storage
            .claim_work(ClaimWorkInput {
                context_id: context.id,
                agent_id: "reviewer".into(),
                title: "Review src/continuity.rs".into(),
                body: "This overlaps the same file.".into(),
                scope: Scope::Project,
                resources: vec!["src/continuity.rs".into()],
                exclusive: true,
                attachment_id: None,
                lease_seconds: Some(180),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let metrics = storage.metrics_text(&telemetry).unwrap();
        assert!(
            metrics.contains(r#"ice_work_claims_active{namespace="test",task_id="task-claims"} 2"#)
        );
        assert!(
            metrics
                .contains(r#"ice_work_claim_conflicts{namespace="test",task_id="task-claims"} 1"#)
        );
    }

    #[test]
    fn metrics_include_continuity_compiler_state() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let telemetry = EngineTelemetry::new();
        let storage = Storage::open(config).unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: "task-compiler".into(),
                session_id: "session-a".into(),
                objective: "compile bounded continuity bands".into(),
                selector: None,
                agent_id: Some("compiler".into()),
                attachment_id: None,
            })
            .unwrap();
        storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "compiler".into(),
                kind: ContinuityKind::Fact,
                title: "Smell fact".into(),
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
            })
            .unwrap();

        let dirty_metrics = storage.metrics_text(&telemetry).unwrap();
        assert!(dirty_metrics.contains("ice_continuity_compiler_contexts 1"));
        assert!(dirty_metrics.contains("ice_continuity_compiler_dirty_contexts 1"));
        assert!(dirty_metrics.contains(r#"ice_continuity_compiler_chunks{band="hot"} 0"#));
        assert!(dirty_metrics.contains(r#"ice_continuity_compiler_items{band="hot"} 0"#));

        storage
            .recall_continuity(&context.id, "What does Alice smell like?", false, 12)
            .unwrap();

        let compiled_metrics = storage.metrics_text(&telemetry).unwrap();
        assert!(compiled_metrics.contains("ice_continuity_compiler_contexts 1"));
        assert!(compiled_metrics.contains("ice_continuity_compiler_dirty_contexts 0"));
        assert!(compiled_metrics.contains(r#"ice_continuity_compiler_chunks{band="hot"} 1"#));
        assert!(compiled_metrics.contains(r#"ice_continuity_compiler_items{band="hot"} 1"#));
        assert!(compiled_metrics.contains(r#"ice_continuity_compiler_chunks{band="warm"} 0"#));
        assert!(compiled_metrics.contains(r#"ice_continuity_compiler_chunks{band="cold"} 0"#));
    }

    #[test]
    fn compiler_promotes_direct_continuity_supports_into_hot_chunk() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: "task-proof-bundle".into(),
                session_id: "session-a".into(),
                objective: "preserve decision rationale in compiled continuity".into(),
                selector: None,
                agent_id: Some("planner".into()),
                attachment_id: None,
            })
            .unwrap();
        let fact = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "planner".into(),
                kind: ContinuityKind::Fact,
                title: "Latency requirement".into(),
                body: "Service latency must stay below 100ms.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Resolved),
                importance: Some(0.55),
                confidence: Some(0.95),
                salience: Some(0.4),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        let decision = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "planner".into(),
                kind: ContinuityKind::Decision,
                title: "Use request cache".into(),
                body: "Cache request results on the hot path.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.95),
                confidence: Some(0.95),
                salience: Some(0.95),
                layer: None,
                supports: vec![SupportRef {
                    support_type: "continuity".into(),
                    support_id: fact.id.clone(),
                    reason: Some("latency_rationale".into()),
                    weight: 0.9,
                }],
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();

        storage
            .recall_continuity(&context.id, "how do we keep latency under 100ms?", false, 8)
            .unwrap();

        let (item_ids_json, body): (String, String) = storage
            .conn
            .query_row(
                r#"
                SELECT item_ids_json, body
                FROM continuity_compiled_chunks
                WHERE context_id = ?1 AND band = 'hot'
                "#,
                params![context.id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        let item_ids = serde_json::from_str::<Vec<String>>(&item_ids_json).unwrap();

        assert!(item_ids.contains(&decision.id));
        assert!(item_ids.contains(&fact.id));
        assert!(body.contains("[DECISION] Use request cache"));
        assert!(body.contains("supported-by:"));
        assert!(body.contains("[FACT] Latency requirement"));
        assert!(body.contains("latency_rationale"));
        assert!(body.contains("Service latency must stay below 100ms."));
    }

    #[test]
    fn recall_continuity_marks_causal_anchor_and_support_items() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: "task-causal-recall".into(),
                session_id: "session-a".into(),
                objective: "retain rationale links during recall".into(),
                selector: None,
                agent_id: Some("planner".into()),
                attachment_id: None,
            })
            .unwrap();
        let support = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "planner".into(),
                kind: ContinuityKind::Fact,
                title: "Latency target".into(),
                body: "Requests must stay below 100ms.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Resolved),
                importance: Some(0.45),
                confidence: Some(0.95),
                salience: Some(0.35),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        let anchor = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "planner".into(),
                kind: ContinuityKind::Decision,
                title: "Enable request cache".into(),
                body: "Cache the hottest request path to hit the latency target.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.92),
                confidence: Some(0.95),
                salience: Some(0.92),
                layer: None,
                supports: vec![SupportRef {
                    support_type: "continuity".into(),
                    support_id: support.id.clone(),
                    reason: Some("latency_target".into()),
                    weight: 1.0,
                }],
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "planner".into(),
                kind: ContinuityKind::Fact,
                title: "Cache size note".into(),
                body: "Keep the cache size bounded for portability.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Resolved),
                importance: Some(0.55),
                confidence: Some(0.95),
                salience: Some(0.55),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let recall = storage
            .recall_continuity(
                &context.id,
                "why are we caching the request path?",
                false,
                8,
            )
            .unwrap();
        let recalled_anchor = recall
            .items
            .iter()
            .find(|item| item.id == anchor.id)
            .unwrap();
        let recalled_support = recall
            .items
            .iter()
            .find(|item| item.id == support.id)
            .unwrap();

        assert!(recalled_anchor.why.iter().any(|why| why == "causal_anchor"));
        assert!(
            recalled_support
                .why
                .iter()
                .any(|why| why == "supports_anchor")
        );
        assert!(recalled_anchor.score > recalled_support.score);
    }

    #[test]
    fn compiler_ranks_causal_bundle_anchor_before_standalone_items() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: "task-bundle-rank".into(),
                session_id: "session-a".into(),
                objective: "prefer causal bundles first".into(),
                selector: None,
                agent_id: Some("planner".into()),
                attachment_id: None,
            })
            .unwrap();
        let support = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "planner".into(),
                kind: ContinuityKind::Fact,
                title: "Latency budget".into(),
                body: "Keep hot path latency under 100ms.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Resolved),
                importance: Some(0.60),
                confidence: Some(0.95),
                salience: Some(0.35),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        let anchor = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "planner".into(),
                kind: ContinuityKind::Decision,
                title: "Cache request path".into(),
                body: "Cache the request path because the latency budget is tight.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.88),
                confidence: Some(0.95),
                salience: Some(0.70),
                layer: None,
                supports: vec![SupportRef {
                    support_type: "continuity".into(),
                    support_id: support.id.clone(),
                    reason: Some("latency_budget".into()),
                    weight: 1.0,
                }],
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        let standalone = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "planner".into(),
                kind: ContinuityKind::Incident,
                title: "Dashboard color drift".into(),
                body: "A dashboard color mismatch is open but unrelated to the cache decision."
                    .into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.90),
                confidence: Some(0.95),
                salience: Some(0.96),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();

        storage
            .recall_continuity(
                &context.id,
                "why are we caching the request path?",
                false,
                8,
            )
            .unwrap();

        let (item_ids_json, body): (String, String) = storage
            .conn
            .query_row(
                r#"
                SELECT item_ids_json, body
                FROM continuity_compiled_chunks
                WHERE context_id = ?1 AND band = 'hot'
                "#,
                params![context.id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        let item_ids = serde_json::from_str::<Vec<String>>(&item_ids_json).unwrap();

        assert_eq!(item_ids.first(), Some(&anchor.id));
        assert!(item_ids.contains(&standalone.id));
        let first_entry = body.lines().nth(1).unwrap_or_default();
        assert!(first_entry.contains("[DECISION] Cache request path"));
    }

    #[test]
    fn machine_profile_persists_and_resolves_alias() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();

        let machine_a = storage.machine_profile().unwrap();
        let machine_b = storage.machine_profile().unwrap();

        assert_eq!(machine_a.machine_id, machine_b.machine_id);
        assert_eq!(
            storage
                .resolve_namespace_alias(Some(MACHINE_NAMESPACE_ALIAS))
                .unwrap(),
            Some(machine_a.namespace.clone())
        );
    }

    #[test]
    fn metrics_include_machine_identity() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let telemetry = EngineTelemetry::new();
        let storage = Storage::open(config).unwrap();
        let machine = storage.machine_profile().unwrap();

        let metrics = storage.metrics_text(&telemetry).unwrap();
        assert!(metrics.contains("ice_machine_info{"));
        assert!(metrics.contains(&format!(r#"machine_id="{}""#, machine.machine_id)));
        assert!(metrics.contains(&format!(r#"namespace="{}""#, machine.namespace)));
    }

    #[test]
    fn storage_open_canonicalizes_machine_alias_ghosts() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let telemetry = EngineTelemetry::new();
        let storage = Storage::open(config.clone()).unwrap();
        let machine = storage.machine_profile().unwrap();
        let canonical_context = storage
            .open_context(OpenContextInput {
                namespace: machine.namespace.clone(),
                task_id: machine.default_task_id.clone(),
                session_id: "session-canonical".into(),
                objective: "canonical machine organism".into(),
                selector: None,
                agent_id: Some("codex-observer".into()),
                attachment_id: None,
            })
            .unwrap();
        storage
            .claim_work(ClaimWorkInput {
                context_id: canonical_context.id.clone(),
                agent_id: "anxiety-sentinel".into(),
                title: "Own the anxiety lane".into(),
                body: "Canonical organism lane.".into(),
                scope: Scope::Shared,
                resources: vec!["organism/anxiety".into()],
                exclusive: true,
                attachment_id: None,
                lease_seconds: Some(180),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let now = Utc::now();
        let alias_context_id = "ctx:alias-machine".to_string();
        let alias_selector = Selector {
            all: vec![DimensionFilter {
                key: "task".into(),
                values: vec![machine.default_task_id.clone()],
            }],
            any: Vec::new(),
            exclude: Vec::new(),
            layers: Vec::new(),
            start_ts: None,
            end_ts: None,
            limit: Some(48),
            namespace: Some(MACHINE_NAMESPACE_ALIAS.into()),
        };
        storage
            .conn
            .execute(
                r#"
                INSERT INTO contexts(
                  id, opened_at, updated_at, namespace, task_id, session_id, objective, selector_json,
                  status, current_agent_id, current_attachment_id, last_snapshot_id
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'open', ?9, NULL, NULL)
                "#,
                params![
                    alias_context_id,
                    now.to_rfc3339(),
                    now.to_rfc3339(),
                    MACHINE_NAMESPACE_ALIAS,
                    machine.default_task_id,
                    "session-alias",
                    "legacy alias organism",
                    serde_json::to_string(&alias_selector).unwrap(),
                    "codex-observer",
                ],
            )
            .unwrap();
        storage
            .conn
            .execute(
                r#"
                INSERT INTO agent_attachments(
                  id, ts, agent_id, agent_type, namespace, role, capabilities_json, metadata_json, active, last_seen_at, tick_count, context_id
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, '[]', '{}', 1, ?2, 1, ?7)
                "#,
                params![
                    "attach:alias-machine",
                    now.to_rfc3339(),
                    "codex-observer",
                    "codex",
                    MACHINE_NAMESPACE_ALIAS,
                    "operator",
                    alias_context_id,
                ],
            )
            .unwrap();
        let alias_coordination = WorkClaimCoordination {
            claim_key: "alias-claim".into(),
            resources: vec!["organism/anxiety".into()],
            exclusive: true,
            attachment_id: Some("attach:alias-machine".into()),
            lease_seconds: 180,
            lease_expires_at: Some(now + chrono::Duration::seconds(180)),
            renewed_at: Some(now),
            conflict_count: 0,
            conflicts_with: Vec::new(),
        };
        storage
            .conn
            .execute(
                r#"
                INSERT INTO continuity_items(
                  id, memory_id, ts, updated_at, context_id, namespace, task_id, author_agent_id, kind,
                  scope, status, title, body, importance, confidence, salience, supersedes_id, resolved_at, extra_json
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'work_claim', ?9, 'open', ?10, ?11, 0.95, 0.9, 0.95, NULL, NULL, ?12)
                "#,
                params![
                    "continuity:alias-claim",
                    "continuity-memory:alias-claim",
                    now.to_rfc3339(),
                    now.to_rfc3339(),
                    alias_context_id,
                    MACHINE_NAMESPACE_ALIAS,
                    machine.default_task_id,
                    "anxiety-sentinel",
                    Scope::Shared.to_string(),
                    "Own the anxiety lane",
                    "Legacy alias organism lane.",
                    merge_work_claim_extra(serde_json::json!({}), &alias_coordination).to_string(),
                ],
            )
            .unwrap();
        drop(storage);

        let storage = Storage::open(config).unwrap();
        let metrics = storage.metrics_text(&telemetry).unwrap();
        assert!(!metrics.contains(r#"namespace="@machine""#));
        assert!(metrics.contains(&format!(
            r#"ice_work_claims_active{{namespace="{}",task_id="{}"}} 1"#,
            machine.namespace, machine.default_task_id
        )));
    }

    #[test]
    fn refresh_vector_persists_active_backend_key() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path())
            .with_embedding_backend(EmbeddingBackendConfig::Hash { dim: 64 });
        let storage = Storage::open(config).unwrap();

        storage
            .refresh_vector("memory:test", "database crashed at 3am")
            .unwrap();

        let (backend_key, dim): (String, i64) = storage
            .conn
            .query_row(
                "SELECT backend_key, dim FROM memory_vectors WHERE memory_id = ?1",
                params!["memory:test"],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(backend_key, "hash:64");
        assert_eq!(dim, 64);
    }

    #[test]
    fn vector_memories_only_returns_rows_for_active_backend() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path())
            .with_embedding_backend(EmbeddingBackendConfig::Hash { dim: 64 });
        let storage = Storage::open(config).unwrap();
        let now = Utc::now().to_rfc3339();

        for memory_id in ["memory:active", "memory:stale"] {
            storage
                .conn
                .execute(
                    r#"
                    INSERT INTO memory_items(
                      id, layer, scope, agent_id, session_id, task_id, ts, importance, confidence,
                      token_estimate, source_event_id, scope_key, body, extra_json
                    ) VALUES (?1, ?2, ?3, ?4, ?5, NULL, ?6, ?7, ?8, ?9, NULL, ?10, ?11, '{}')
                    "#,
                    params![
                        memory_id,
                        MemoryLayer::Hot as i64,
                        Scope::Shared.to_string(),
                        "agent-a",
                        "session-a",
                        now,
                        0.8_f64,
                        0.9_f64,
                        12_i64,
                        format!("{memory_id}:scope"),
                        format!("{memory_id} body"),
                    ],
                )
                .unwrap();
        }

        storage
            .conn
            .execute(
                "INSERT INTO memory_vectors(memory_id, backend_key, dim, norm, data) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    "memory:active",
                    "hash:64",
                    64_i64,
                    1.0_f64,
                    encode_vector(&vec![1.0_f32; 64]),
                ],
            )
            .unwrap();
        storage
            .conn
            .execute(
                "INSERT INTO memory_vectors(memory_id, backend_key, dim, norm, data) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    "memory:stale",
                    "ollama:embeddinggemma",
                    3_i64,
                    1.0_f64,
                    encode_vector(&[0.0_f32, 1.0_f32, 0.0_f32]),
                ],
            )
            .unwrap();

        let memories = storage.vector_memories().unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].0.id, "memory:active");
    }

    #[test]
    fn record_outcome_reinforces_confirmed_continuity_items() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: "task-plasticity-confirm".into(),
                session_id: "session-confirm".into(),
                objective: "track reinforced continuity".into(),
                selector: None,
                agent_id: Some("agent-a".into()),
                attachment_id: None,
            })
            .unwrap();
        let decision = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "agent-a".into(),
                kind: ContinuityKind::Decision,
                title: "Use Neovim".into(),
                body: "The user prefers Neovim for repo work.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.75),
                confidence: Some(0.8),
                salience: Some(0.75),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "belief_key": "user.preference.editor",
                    "source_role": "user",
                }),
            })
            .unwrap();
        let before = storage
            .list_continuity_items(&context.id, true)
            .unwrap()
            .into_iter()
            .find(|item| item.id == decision.id)
            .unwrap();
        let pack = crate::query::build_context_pack(
            &storage,
            QueryInput {
                agent_id: Some("agent-a".into()),
                session_id: Some(context.session_id.clone()),
                task_id: Some(context.task_id.clone()),
                namespace: Some(context.namespace.clone()),
                objective: Some("resume editor preference".into()),
                selector: None,
                view_id: None,
                query_text: "editor preference".into(),
                budget_tokens: 192,
                candidate_limit: 8,
            },
        )
        .unwrap();
        assert!(
            pack.items
                .iter()
                .any(|item| item.memory_id == decision.memory_id)
        );

        let outcome = storage
            .record_outcome(OutcomeInput {
                context_id: context.id.clone(),
                agent_id: "agent-a".into(),
                title: "Applied editor preference".into(),
                result: "Neovim was used successfully.".into(),
                quality: 0.95,
                pack_id: Some(pack.id),
                used_memory_ids: vec![decision.memory_id.clone()],
                confirmed_memory_ids: vec![decision.memory_id.clone()],
                contradicted_memory_ids: Vec::new(),
                failures: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        assert!(outcome.supports.iter().any(|support| {
            support.support_type == "continuity"
                && support.support_id == decision.id
                && support.reason.as_deref() == Some("outcome_confirmed")
        }));

        let after = storage
            .list_continuity_items(&context.id, true)
            .unwrap()
            .into_iter()
            .find(|item| item.id == decision.id)
            .unwrap();
        let plasticity = after.extra.get("plasticity").unwrap();
        assert_eq!(plasticity["activation_count"], serde_json::json!(1));
        assert_eq!(plasticity["successful_use_count"], serde_json::json!(1));
        assert_eq!(plasticity["confirmation_count"], serde_json::json!(1));
        assert_eq!(plasticity["contradiction_count"], serde_json::json!(0));
        assert!(
            after.retention.effective_salience > before.retention.effective_salience,
            "expected confirmed continuity to gain effective salience"
        );
    }

    #[test]
    fn repeated_immediate_outcomes_do_not_count_as_extra_spaced_reactivation() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: "task-plasticity-massed".into(),
                session_id: "session-massed".into(),
                objective: "avoid fake spacing gains".into(),
                selector: None,
                agent_id: Some("agent-a".into()),
                attachment_id: None,
            })
            .unwrap();
        let decision = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "agent-a".into(),
                kind: ContinuityKind::Decision,
                title: "Guard latest learning".into(),
                body: "Recent learning should show up before the full line.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.78),
                confidence: Some(0.81),
                salience: Some(0.79),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();

        for title in ["First success", "Second immediate success"] {
            storage
                .record_outcome(OutcomeInput {
                    context_id: context.id.clone(),
                    agent_id: "agent-a".into(),
                    title: title.into(),
                    result: "The latest learning was used correctly.".into(),
                    quality: 0.95,
                    pack_id: None,
                    used_memory_ids: vec![decision.memory_id.clone()],
                    confirmed_memory_ids: vec![decision.memory_id.clone()],
                    contradicted_memory_ids: Vec::new(),
                    failures: Vec::new(),
                    dimensions: Vec::new(),
                    extra: serde_json::json!({}),
                })
                .unwrap();
        }

        let after = storage
            .list_continuity_items(&context.id, true)
            .unwrap()
            .into_iter()
            .find(|item| item.id == decision.id)
            .unwrap();
        let plasticity = after.extra.get("plasticity").unwrap();
        let spacing_interval_hours = plasticity["spacing_interval_hours"].as_f64().unwrap();

        assert_eq!(plasticity["activation_count"], serde_json::json!(2));
        assert_eq!(plasticity["successful_use_count"], serde_json::json!(2));
        assert_eq!(plasticity["confirmation_count"], serde_json::json!(2));
        assert_eq!(
            plasticity["spaced_reactivation_count"],
            serde_json::json!(1)
        );
        assert!(spacing_interval_hours > 6.0);
    }

    #[test]
    fn spaced_reactivation_outgrows_massed_repetition() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: "task-plasticity-spacing".into(),
                session_id: "session-spacing".into(),
                objective: "reward spaced reactivation".into(),
                selector: None,
                agent_id: Some("agent-a".into()),
                attachment_id: None,
            })
            .unwrap();
        let spaced = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "agent-a".into(),
                kind: ContinuityKind::Decision,
                title: "Spaced learning path".into(),
                body: "This item should only strengthen when it comes back after a useful gap."
                    .into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.8),
                confidence: Some(0.82),
                salience: Some(0.8),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        let massed = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "agent-a".into(),
                kind: ContinuityKind::Decision,
                title: "Massed learning path".into(),
                body: "This item gets hammered twice in a row.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.8),
                confidence: Some(0.82),
                salience: Some(0.8),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();

        for item in [&spaced, &massed] {
            storage
                .record_outcome(OutcomeInput {
                    context_id: context.id.clone(),
                    agent_id: "agent-a".into(),
                    title: format!("Prime {}", item.title),
                    result: "The memory was used and confirmed.".into(),
                    quality: 0.94,
                    pack_id: None,
                    used_memory_ids: vec![item.memory_id.clone()],
                    confirmed_memory_ids: vec![item.memory_id.clone()],
                    contradicted_memory_ids: Vec::new(),
                    failures: Vec::new(),
                    dimensions: Vec::new(),
                    extra: serde_json::json!({}),
                })
                .unwrap();
        }

        let rewind = (Utc::now() - Duration::hours(12)).to_rfc3339();
        storage
            .conn
            .execute(
                r#"
                UPDATE continuity_plasticity
                SET last_reactivated_at = ?2,
                    last_confirmed_at = ?2,
                    last_strengthened_at = ?2,
                    updated_at = ?2
                WHERE continuity_id = ?1
                "#,
                params![spaced.id, rewind],
            )
            .unwrap();

        storage
            .record_outcome(OutcomeInput {
                context_id: context.id.clone(),
                agent_id: "agent-a".into(),
                title: "Immediate massed repeat".into(),
                result: "The same memory was used again right away.".into(),
                quality: 0.94,
                pack_id: None,
                used_memory_ids: vec![massed.memory_id.clone()],
                confirmed_memory_ids: vec![massed.memory_id.clone()],
                contradicted_memory_ids: Vec::new(),
                failures: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        storage
            .record_outcome(OutcomeInput {
                context_id: context.id.clone(),
                agent_id: "agent-a".into(),
                title: "Delayed spaced repeat".into(),
                result: "The memory came back after enough time had passed.".into(),
                quality: 0.94,
                pack_id: None,
                used_memory_ids: vec![spaced.memory_id.clone()],
                confirmed_memory_ids: vec![spaced.memory_id.clone()],
                contradicted_memory_ids: Vec::new(),
                failures: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let items = storage.list_continuity_items(&context.id, true).unwrap();
        let spaced_after = items.iter().find(|item| item.id == spaced.id).unwrap();
        let massed_after = items.iter().find(|item| item.id == massed.id).unwrap();

        assert_eq!(
            spaced_after.extra["plasticity"]["spaced_reactivation_count"],
            serde_json::json!(2)
        );
        assert_eq!(
            massed_after.extra["plasticity"]["spaced_reactivation_count"],
            serde_json::json!(1)
        );
        assert!(spaced_after.retention.half_life_hours > massed_after.retention.half_life_hours);
        assert!(
            spaced_after.retention.effective_salience > massed_after.retention.effective_salience
        );
    }

    #[test]
    fn recall_continuity_prefers_user_sourced_belief_over_assistant_competitor() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: "task-belief-competition-recall".into(),
                session_id: "session-belief-competition-recall".into(),
                objective: "pick the current user preference".into(),
                selector: None,
                agent_id: Some("agent-a".into()),
                attachment_id: None,
            })
            .unwrap();
        let user_item = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "agent-a".into(),
                kind: ContinuityKind::Fact,
                title: "User prefers Neovim".into(),
                body: "The user explicitly said they prefer Neovim for repo work.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.76),
                confidence: Some(0.78),
                salience: Some(0.76),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "belief_key": "user.preference.editor",
                    "source_role": "user",
                }),
            })
            .unwrap();
        let assistant_item = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "agent-a".into(),
                kind: ContinuityKind::Fact,
                title: "Assistant guessed Emacs".into(),
                body: "The assistant speculated that the user probably prefers Emacs.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.76),
                confidence: Some(0.78),
                salience: Some(0.76),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "belief_key": "user.preference.editor",
                    "source_role": "assistant",
                }),
            })
            .unwrap();

        let recall = storage
            .recall_continuity(&context.id, "what editor does the user prefer", false, 8)
            .unwrap();
        assert_eq!(
            recall.items.first().map(|item| item.id.as_str()),
            Some(user_item.id.as_str())
        );

        let competitor = recall
            .items
            .iter()
            .find(|item| item.id == assistant_item.id)
            .unwrap();
        assert!(
            competitor
                .why
                .iter()
                .any(|why| why == "belief_key_competitor")
        );
        assert!(
            competitor
                .why
                .iter()
                .any(|why| why == "source_role:assistant")
        );
    }

    #[test]
    fn build_context_pack_rejects_weaker_same_belief_continuity_item() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: "task-belief-competition-pack".into(),
                session_id: "session-belief-competition-pack".into(),
                objective: "select the strongest editor preference".into(),
                selector: None,
                agent_id: Some("agent-a".into()),
                attachment_id: None,
            })
            .unwrap();
        let user_item = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "agent-a".into(),
                kind: ContinuityKind::Fact,
                title: "User prefers Neovim".into(),
                body: "The user explicitly said they prefer Neovim for repo work.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.76),
                confidence: Some(0.78),
                salience: Some(0.76),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "belief_key": "user.preference.editor",
                    "source_role": "user",
                }),
            })
            .unwrap();
        let assistant_item = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "agent-a".into(),
                kind: ContinuityKind::Fact,
                title: "Assistant guessed Emacs".into(),
                body: "The assistant speculated that the user probably prefers Emacs.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.76),
                confidence: Some(0.78),
                salience: Some(0.76),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "belief_key": "user.preference.editor",
                    "source_role": "assistant",
                }),
            })
            .unwrap();

        let pack = crate::query::build_context_pack(
            &storage,
            QueryInput {
                agent_id: Some("agent-a".into()),
                session_id: Some(context.session_id.clone()),
                task_id: Some(context.task_id.clone()),
                namespace: Some(context.namespace.clone()),
                objective: Some("resume the user's editor preference".into()),
                selector: None,
                view_id: None,
                query_text: "user editor preference".into(),
                budget_tokens: 192,
                candidate_limit: 8,
            },
        )
        .unwrap();

        assert!(
            pack.items
                .iter()
                .any(|item| item.memory_id == user_item.memory_id)
        );
        assert!(
            !pack
                .items
                .iter()
                .any(|item| item.memory_id == assistant_item.memory_id)
        );

        let manifest = storage.explain_context_pack(&pack.id).unwrap();
        assert!(manifest.rejected.iter().any(|candidate| {
            candidate.memory_id == assistant_item.memory_id
                && candidate.reason == "belief_key_competitor"
        }));
    }

    #[test]
    fn recall_continuity_demotes_stale_open_guidance_without_reinforcement() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: "task-stale-open-guidance".into(),
                session_id: "session-stale-open-guidance".into(),
                objective: "track live review guidance".into(),
                selector: None,
                agent_id: Some("agent-a".into()),
                attachment_id: None,
            })
            .unwrap();
        let stale = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "agent-a".into(),
                kind: ContinuityKind::Decision,
                title: "Old review guidance".into(),
                body: "The old review guidance said to refresh the old MCP session first.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.84),
                confidence: Some(0.84),
                salience: Some(0.84),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        let current = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "agent-a".into(),
                kind: ContinuityKind::Decision,
                title: "Current review guidance".into(),
                body: "The current review guidance says to start from current practice.".into(),
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

        let sqlite = dir.path().join("data/ice.sqlite");
        let conn = Connection::open(sqlite).unwrap();
        let stale_ts = (Utc::now() - chrono::Duration::days(12)).to_rfc3339();
        let current_ts = (Utc::now() - chrono::Duration::hours(3)).to_rfc3339();
        conn.execute(
            "UPDATE continuity_items SET ts = ?2, updated_at = ?2 WHERE id = ?1",
            params![stale.id.as_str(), stale_ts],
        )
        .unwrap();
        conn.execute(
            "UPDATE memory_items SET ts = ?2 WHERE id = ?1",
            params![
                stale.memory_id.as_str(),
                (Utc::now() - chrono::Duration::days(12)).to_rfc3339()
            ],
        )
        .unwrap();
        conn.execute(
            "UPDATE continuity_items SET ts = ?2, updated_at = ?2 WHERE id = ?1",
            params![current.id.as_str(), current_ts],
        )
        .unwrap();
        conn.execute(
            "UPDATE memory_items SET ts = ?2 WHERE id = ?1",
            params![
                current.memory_id.as_str(),
                (Utc::now() - chrono::Duration::hours(3)).to_rfc3339()
            ],
        )
        .unwrap();

        let recall = storage
            .recall_continuity(&context.id, "what is the current review guidance", false, 8)
            .unwrap();
        assert_eq!(
            recall.items.first().map(|item| item.id.as_str()),
            Some(current.id.as_str())
        );

        let stale_item = recall
            .items
            .iter()
            .find(|item| item.id == stale.id)
            .unwrap();
        assert!(
            stale_item
                .why
                .iter()
                .any(|why| why == "stale_open_guidance")
        );
        assert!(
            stale_item
                .why
                .iter()
                .any(|why| why == "practice_retired" || why == "practice_stale")
        );
    }

    #[test]
    fn record_outcome_penalizes_contradicted_continuity_items() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: "task-plasticity-contradict".into(),
                session_id: "session-contradict".into(),
                objective: "track contradicted continuity".into(),
                selector: None,
                agent_id: Some("agent-a".into()),
                attachment_id: None,
            })
            .unwrap();
        let fact = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "agent-a".into(),
                kind: ContinuityKind::Fact,
                title: "Alice lives in London".into(),
                body: "Alice currently lives in London.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.8),
                confidence: Some(0.8),
                salience: Some(0.8),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "belief_key": "user.location.city",
                    "source_role": "user",
                }),
            })
            .unwrap();
        let before = storage
            .list_continuity_items(&context.id, true)
            .unwrap()
            .into_iter()
            .find(|item| item.id == fact.id)
            .unwrap();

        let outcome = storage
            .record_outcome(OutcomeInput {
                context_id: context.id.clone(),
                agent_id: "agent-a".into(),
                title: "Location update invalidated".into(),
                result: "The prior location memory was wrong.".into(),
                quality: 0.2,
                pack_id: None,
                used_memory_ids: vec![fact.memory_id.clone()],
                confirmed_memory_ids: Vec::new(),
                contradicted_memory_ids: vec![fact.memory_id.clone()],
                failures: vec!["stale-memory".into()],
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        assert!(outcome.supports.iter().any(|support| {
            support.support_type == "continuity"
                && support.support_id == fact.id
                && support.reason.as_deref() == Some("outcome_contradicted")
        }));

        let after = storage
            .list_continuity_items(&context.id, true)
            .unwrap()
            .into_iter()
            .find(|item| item.id == fact.id)
            .unwrap();
        let plasticity = after.extra.get("plasticity").unwrap();
        assert_eq!(plasticity["activation_count"], serde_json::json!(1));
        assert_eq!(plasticity["successful_use_count"], serde_json::json!(0));
        assert_eq!(plasticity["confirmation_count"], serde_json::json!(0));
        assert_eq!(plasticity["contradiction_count"], serde_json::json!(1));
        assert!(
            after.retention.effective_salience < before.retention.effective_salience,
            "expected contradicted continuity to lose effective salience"
        );
    }

    #[test]
    fn resolve_or_supersede_updates_belief_key_plasticity_on_both_items() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: "task-plasticity-supersede".into(),
                session_id: "session-supersede".into(),
                objective: "track superseded belief continuity".into(),
                selector: None,
                agent_id: Some("agent-a".into()),
                attachment_id: None,
            })
            .unwrap();
        let stale = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "agent-a".into(),
                kind: ContinuityKind::Fact,
                title: "Alice lives in London".into(),
                body: "Alice currently lives in London.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.8),
                confidence: Some(0.8),
                salience: Some(0.8),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "belief_key": "user.location.city",
                    "source_role": "user",
                }),
            })
            .unwrap();
        let replacement = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "agent-a".into(),
                kind: ContinuityKind::Fact,
                title: "Alice lives in Berlin".into(),
                body: "Alice currently lives in Berlin.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.82),
                confidence: Some(0.82),
                salience: Some(0.82),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "belief_key": "user.location.city",
                    "source_role": "user",
                }),
            })
            .unwrap();

        storage
            .resolve_continuity_item(ResolveOrSupersedeInput {
                continuity_id: stale.id.clone(),
                actor_agent_id: "agent-a".into(),
                new_status: ContinuityStatus::Superseded,
                supersedes_id: Some(replacement.id.clone()),
                resolution_note: Some("The user moved.".into()),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let items = storage.list_continuity_items(&context.id, true).unwrap();
        let stale_after = items.iter().find(|item| item.id == stale.id).unwrap();
        let replacement_after = items.iter().find(|item| item.id == replacement.id).unwrap();
        assert_eq!(stale_after.status, ContinuityStatus::Superseded);
        assert_eq!(
            stale_after.extra["plasticity"]["contradiction_count"],
            serde_json::json!(1)
        );
        assert_eq!(
            replacement_after.extra["plasticity"]["confirmation_count"],
            serde_json::json!(1)
        );
    }

    #[test]
    fn resolve_or_supersede_emits_learning_lesson_for_belief_updates() {
        let dir = tempdir().unwrap();
        let config = EngineConfig::with_root(dir.path());
        let storage = Storage::open(config).unwrap();
        let context = storage
            .open_context(OpenContextInput {
                namespace: "test".into(),
                task_id: "task-belief-learning".into(),
                session_id: "session-belief-learning".into(),
                objective: "surface belief learning updates".into(),
                selector: None,
                agent_id: Some("agent-a".into()),
                attachment_id: None,
            })
            .unwrap();
        let stale = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "agent-a".into(),
                kind: ContinuityKind::Fact,
                title: "Alice lives in London".into(),
                body: "Alice currently lives in London.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.8),
                confidence: Some(0.8),
                salience: Some(0.8),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "belief_key": "user.location.city",
                    "source_role": "user",
                }),
            })
            .unwrap();
        let replacement = storage
            .persist_continuity_item(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "agent-a".into(),
                kind: ContinuityKind::Fact,
                title: "Alice lives in Berlin".into(),
                body: "Alice currently lives in Berlin.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.82),
                confidence: Some(0.82),
                salience: Some(0.82),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "belief_key": "user.location.city",
                    "source_role": "user",
                }),
            })
            .unwrap();

        storage
            .resolve_continuity_item(ResolveOrSupersedeInput {
                continuity_id: stale.id.clone(),
                actor_agent_id: "agent-a".into(),
                new_status: ContinuityStatus::Superseded,
                supersedes_id: Some(replacement.id.clone()),
                resolution_note: Some("The user moved.".into()),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let items = storage.list_continuity_items(&context.id, true).unwrap();
        let lesson = items
            .iter()
            .find(|item| {
                item.kind == ContinuityKind::Lesson && item.title.starts_with("Belief update:")
            })
            .unwrap();

        assert!(lesson.body.contains("Alice currently lives in London."));
        assert!(lesson.body.contains("Alice currently lives in Berlin."));
        assert!(lesson.body.contains("The user moved."));
        assert_eq!(
            lesson.extra["user"]["learning_trigger"],
            serde_json::json!("prediction_error_reconsolidation")
        );
        assert_eq!(
            lesson.extra["user"]["learning_belief_key"],
            serde_json::json!("user.location.city")
        );
        assert!(lesson.supports.iter().any(|support| {
            support.support_id == stale.id
                && support.reason.as_deref() == Some("belief_update_previous")
        }));
        assert!(lesson.supports.iter().any(|support| {
            support.support_id == replacement.id
                && support.reason.as_deref() == Some("belief_update_current")
        }));
    }
}
