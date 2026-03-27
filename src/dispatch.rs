use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::ValueEnum;
use postgres::fallible_iterator::FallibleIterator;
use postgres::{Client, NoTls, Row};
use serde::{Deserialize, Serialize};
use tokio::runtime::Handle;
use tracing::debug;
use uuid::Uuid;

use crate::continuity::{
    CoordinationLane, CoordinationProjectedLane, CoordinationSeverity, CoordinationSignalInput,
    MachineProfile, OutcomeInput, ReadContextInput, ResumeInput, ResumeRecord,
    SharedContinuityKernel, SignalInput, SnapshotInput, UnifiedContinuityInterface,
};
use crate::model::{Selector, SnapshotResolution};

pub const DEFAULT_DISPATCH_NOTIFY_CHANNEL: &str = "ice_dispatch_signal";
pub const DEFAULT_DISPATCH_CHANNEL: &str = DEFAULT_DISPATCH_NOTIFY_CHANNEL;
const DEFAULT_WORKER_STALE_SECS: u64 = 120;
const DEFAULT_ASSIGNMENT_TOKEN_BUDGET: usize = 224;
const DEFAULT_ASSIGNMENT_CANDIDATE_LIMIT: usize = 24;

#[cfg(test)]
struct AppendMetricsTestHook {
    root: PathBuf,
    entered_tx: std::sync::mpsc::Sender<()>,
    release_rx: std::sync::mpsc::Receiver<()>,
}

#[cfg(test)]
static APPEND_METRICS_TEST_HOOK: std::sync::Mutex<Option<AppendMetricsTestHook>> =
    std::sync::Mutex::new(None);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchConfig {
    pub database_url: String,
    #[serde(default = "default_dispatch_notify_channel")]
    pub notify_channel: String,
    #[serde(default = "default_worker_stale_secs")]
    pub worker_stale_secs: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum DispatchWorkerTier {
    Small,
    Medium,
    Large,
    Script,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DispatchAttachedLaneSource {
    ExplicitCli,
    LiveBadgeOptIn,
}

impl DispatchAttachedLaneSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitCli => "explicit_cli",
            Self::LiveBadgeOptIn => "live_badge_opt_in",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum DispatchSignalKind {
    TaskComplete,
    HandoffReady,
}

pub type DispatchMessageKind = DispatchSignalKind;
pub type DispatchTargetTier = DispatchWorkerTier;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DispatchStatus {
    Queued,
    Assigned,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchWorkerUpsertInput {
    pub worker_id: String,
    pub display_name: String,
    pub role: String,
    pub agent_type: String,
    pub tier: DispatchWorkerTier,
    pub model: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default = "default_max_parallelism")]
    pub max_parallelism: usize,
    #[serde(default)]
    pub focus: String,
    pub namespace: Option<String>,
    pub task_id: Option<String>,
    #[serde(default = "default_worker_status")]
    pub status: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchWorkerRecord {
    pub worker_id: String,
    pub display_name: String,
    pub role: String,
    pub tier: DispatchWorkerTier,
    pub agent_type: String,
    pub model: String,
    pub capabilities: Vec<String>,
    pub max_parallelism: usize,
    pub status: String,
    pub focus: String,
    pub namespace: Option<String>,
    pub task_id: Option<String>,
    pub metadata: serde_json::Value,
    pub last_seen_at: DateTime<Utc>,
    pub active_assignment_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishDispatchSignalInput {
    pub kind: DispatchSignalKind,
    pub from_agent_id: String,
    pub context_id: String,
    pub namespace: String,
    pub task_id: String,
    pub snapshot_id: Option<String>,
    pub objective: String,
    pub target_role: Option<String>,
    pub preferred_tier: Option<DispatchWorkerTier>,
    pub reason: Option<String>,
    #[serde(default)]
    pub extra: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchSignalRecord {
    pub id: String,
    pub kind: DispatchSignalKind,
    pub status: DispatchStatus,
    pub machine_id: String,
    pub namespace: String,
    pub task_id: String,
    pub context_id: String,
    pub snapshot_id: Option<String>,
    pub from_agent_id: String,
    pub objective: String,
    pub target_role: Option<String>,
    pub preferred_tier: Option<DispatchWorkerTier>,
    pub reason: Option<String>,
    pub extra: serde_json::Value,
    pub assigned_worker_id: Option<String>,
    pub assignment_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub queued_at: DateTime<Utc>,
    pub assigned_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchPressureProfile {
    pub anxiety: f64,
    pub fear: f64,
    pub confidence: f64,
    pub discipline: f64,
    pub habit: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DispatchAffectOverride {
    pub anxiety: Option<f64>,
    pub confidence: Option<f64>,
    pub fear: Option<f64>,
    pub discipline: Option<f64>,
    pub habit: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchCompleteInput {
    pub kind: DispatchSignalKind,
    pub context_id: Option<String>,
    pub namespace: Option<String>,
    pub task_id: Option<String>,
    pub agent_id: String,
    pub title: String,
    pub result: String,
    pub objective: Option<String>,
    pub quality: f64,
    #[serde(default)]
    pub failures: Vec<String>,
    pub selector: Option<Selector>,
    pub target_tier: DispatchWorkerTier,
    #[serde(default)]
    pub affect_override: DispatchAffectOverride,
    #[serde(default)]
    pub extra: serde_json::Value,
    #[serde(default = "default_snapshot_resolution")]
    pub snapshot_resolution: SnapshotResolution,
    #[serde(default = "default_assignment_token_budget")]
    pub token_budget: usize,
    #[serde(default = "default_assignment_candidate_limit")]
    pub candidate_limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchClaimInput {
    pub worker: DispatchWorkerUpsertInput,
    pub lease_seconds: i64,
    #[serde(default = "default_assignment_token_budget")]
    pub token_budget: usize,
    #[serde(default = "default_assignment_candidate_limit")]
    pub candidate_limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchResumeHandle {
    pub context_id: String,
    pub snapshot_id: Option<String>,
    pub namespace: String,
    pub task_id: String,
    pub objective: String,
    pub token_budget: usize,
    pub candidate_limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchContextPreview {
    pub pack_id: String,
    pub latest_snapshot_id: Option<String>,
    pub pack_used_tokens: usize,
    pub hot_memory: Vec<String>,
    pub decisions: Vec<String>,
    pub constraints: Vec<String>,
    pub operational_scars: Vec<String>,
    pub open_threads: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchAssignmentEnvelope {
    pub machine: MachineProfile,
    pub signal_id: String,
    pub worker_id: String,
    pub worker_display_name: String,
    pub worker_role: String,
    pub worker_tier: DispatchWorkerTier,
    pub objective: String,
    pub start_summary: String,
    pub resume: DispatchResumeHandle,
    pub context_preview: DispatchContextPreview,
    pub pressure: DispatchPressureProfile,
    #[serde(default)]
    pub attached_projected_lane: Option<CoordinationProjectedLane>,
    #[serde(default)]
    pub attached_projected_lane_source: Option<DispatchAttachedLaneSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchAssignmentRecord {
    pub id: String,
    pub signal_id: String,
    pub worker_id: String,
    pub machine_id: String,
    pub namespace: String,
    pub task_id: String,
    pub context_id: String,
    pub snapshot_id: Option<String>,
    pub objective: String,
    pub worker_role: String,
    pub worker_tier: DispatchWorkerTier,
    pub status: DispatchStatus,
    pub pressure: DispatchPressureProfile,
    pub envelope: DispatchAssignmentEnvelope,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchWorkerPresence {
    pub worker_id: String,
    pub display_name: String,
    pub role: String,
    pub tier: DispatchWorkerTier,
    pub agent_type: String,
    pub model: String,
    pub status: String,
    pub focus: String,
    pub namespace: Option<String>,
    pub task_id: Option<String>,
    pub active_assignment_count: usize,
    pub projected_lane: CoordinationProjectedLane,
    #[serde(default)]
    pub attached_projected_lane: Option<CoordinationProjectedLane>,
    #[serde(default)]
    pub attached_projected_lane_source: Option<DispatchAttachedLaneSource>,
    pub last_seen_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchAssignmentPresence {
    pub assignment_id: String,
    pub signal_id: String,
    pub worker_id: String,
    pub worker_role: String,
    pub worker_tier: DispatchWorkerTier,
    pub namespace: String,
    pub task_id: String,
    pub context_id: String,
    pub objective: String,
    pub status: DispatchStatus,
    pub pressure: DispatchPressureProfile,
    pub projected_lane: CoordinationProjectedLane,
    #[serde(default)]
    pub attached_projected_lane: Option<CoordinationProjectedLane>,
    #[serde(default)]
    pub attached_projected_lane_source: Option<DispatchAttachedLaneSource>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchOrganismSnapshot {
    pub configured: bool,
    pub reachable: bool,
    pub notify_channel: Option<String>,
    pub worker_active_window_secs: Option<u64>,
    pub workers_active: usize,
    pub signals_queued: usize,
    pub assignments_active: usize,
    #[serde(default)]
    pub workers: Vec<DispatchWorkerPresence>,
    #[serde(default)]
    pub assignments: Vec<DispatchAssignmentPresence>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteDispatchInput {
    pub router_id: String,
    #[serde(default = "default_assignment_token_budget")]
    pub token_budget: usize,
    #[serde(default = "default_assignment_candidate_limit")]
    pub candidate_limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaitDispatchInput {
    #[serde(flatten)]
    pub route: RouteDispatchInput,
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteAssignmentInput {
    pub assignment_id: String,
    pub worker_id: Option<String>,
    #[serde(default)]
    pub failed: bool,
}

pub struct DispatchSpine {
    config: DispatchConfig,
}

impl DispatchConfig {
    pub fn load(root: impl AsRef<Path>) -> Result<Option<Self>> {
        if let Ok(database_url) = std::env::var("ICE_DISPATCH_DATABASE_URL") {
            let database_url = database_url.trim().to_string();
            if !database_url.is_empty() {
                let config = Self {
                    database_url,
                    notify_channel: std::env::var("ICE_DISPATCH_NOTIFY_CHANNEL")
                        .ok()
                        .filter(|value| !value.trim().is_empty())
                        .unwrap_or_else(default_dispatch_notify_channel),
                    worker_stale_secs: std::env::var("ICE_DISPATCH_WORKER_STALE_SECS")
                        .ok()
                        .and_then(|value| value.parse::<u64>().ok())
                        .unwrap_or_else(default_worker_stale_secs),
                };
                config.validate()?;
                return Ok(Some(config));
            }
        }
        let path = root.as_ref().join("data/dispatch-config.json");
        if !path.exists() {
            return Ok(None);
        }
        let text =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let config: Self =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        config.validate()?;
        Ok(Some(config))
    }

    pub fn save(&self, root: impl AsRef<Path>) -> Result<PathBuf> {
        self.validate()?;
        let path = root.as_ref().join("data/dispatch-config.json");
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let body = serde_json::to_string_pretty(self)?;
        fs::write(&path, body).with_context(|| format!("writing {}", path.display()))?;
        Ok(path)
    }

    fn validate(&self) -> Result<()> {
        if self.database_url.trim().is_empty() {
            anyhow::bail!("dispatch database url must not be empty");
        }
        if !is_valid_pg_identifier(&self.notify_channel) {
            anyhow::bail!(
                "dispatch notify channel '{}' must be a safe PostgreSQL identifier",
                self.notify_channel
            );
        }
        Ok(())
    }
}

impl DispatchSpine {
    pub fn from_root(root: impl AsRef<Path>) -> Result<Option<Self>> {
        let Some(config) = DispatchConfig::load(root)? else {
            return Ok(None);
        };
        Ok(Some(Self { config }))
    }

    pub fn from_root_required(root: impl AsRef<Path>) -> Result<Self> {
        Self::from_root(root)?.context(
            "dispatch spine is not configured; run `ice dispatch init --database-url ...` or set ICE_DISPATCH_DATABASE_URL",
        )
    }

    pub fn init(root: impl AsRef<Path>, config: DispatchConfig) -> Result<Self> {
        config.save(root)?;
        let spine = Self { config };
        let mut client = spine.connect()?;
        spine.ensure_schema(&mut client)?;
        Ok(spine)
    }

    pub fn upsert_worker(&self, input: DispatchWorkerUpsertInput) -> Result<DispatchWorkerRecord> {
        let mut client = self.connect()?;
        self.ensure_schema(&mut client)?;
        let capabilities = serde_json::to_value(&input.capabilities)?;
        let now = Utc::now();
        client.execute(
            r#"
            INSERT INTO dispatch_workers (
              worker_id, display_name, role, tier, agent_type, model,
              capabilities_json, max_parallelism, status, focus, namespace, task_id,
              metadata_json, last_seen_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
            ON CONFLICT (worker_id) DO UPDATE SET
              display_name = EXCLUDED.display_name,
              role = EXCLUDED.role,
              tier = EXCLUDED.tier,
              agent_type = EXCLUDED.agent_type,
              model = EXCLUDED.model,
              capabilities_json = EXCLUDED.capabilities_json,
              max_parallelism = EXCLUDED.max_parallelism,
              status = EXCLUDED.status,
              focus = EXCLUDED.focus,
              namespace = EXCLUDED.namespace,
              task_id = EXCLUDED.task_id,
              metadata_json = EXCLUDED.metadata_json,
              last_seen_at = EXCLUDED.last_seen_at
            "#,
            &[
                &input.worker_id,
                &input.display_name,
                &input.role,
                &input.tier.as_str(),
                &input.agent_type,
                &input.model,
                &capabilities,
                &(input.max_parallelism as i32),
                &input.status,
                &input.focus,
                &input.namespace,
                &input.task_id,
                &input.metadata,
                &now,
            ],
        )?;
        let record = self.worker_record(&mut client, &input.worker_id)?;
        debug!(
            op = "dispatch_upsert_worker",
            worker_id = %record.worker_id,
            display_name = %record.display_name,
            role = %record.role,
            tier = %record.tier.as_str(),
            agent_type = %record.agent_type,
            namespace = record.namespace.as_deref().unwrap_or(""),
            task_id = record.task_id.as_deref().unwrap_or(""),
            active_assignment_count = record.active_assignment_count,
            "persisted dispatch worker heartbeat"
        );
        Ok(record)
    }

    pub fn publish_signal(
        &self,
        machine: &MachineProfile,
        input: PublishDispatchSignalInput,
    ) -> Result<DispatchSignalRecord> {
        let mut client = self.connect()?;
        self.ensure_schema(&mut client)?;
        let record = DispatchSignalRecord {
            id: format!("dispatch-signal:{}", Uuid::now_v7()),
            kind: input.kind,
            status: DispatchStatus::Queued,
            machine_id: machine.machine_id.clone(),
            namespace: input.namespace,
            task_id: input.task_id,
            context_id: input.context_id,
            snapshot_id: input.snapshot_id,
            from_agent_id: input.from_agent_id,
            objective: input.objective,
            target_role: input.target_role,
            preferred_tier: input.preferred_tier,
            reason: input.reason,
            extra: input.extra,
            assigned_worker_id: None,
            assignment_id: None,
            created_at: Utc::now(),
            queued_at: Utc::now(),
            assigned_at: None,
            completed_at: None,
        };
        let mut transaction = client.transaction()?;
        transaction.execute(
            r#"
            INSERT INTO dispatch_signals (
              id, signal_kind, status, machine_id, namespace, task_id, context_id, snapshot_id,
              from_agent_id, objective, target_role, preferred_tier, reason, payload_json,
              created_at, queued_at, assigned_worker_id, assignment_id, assigned_at, completed_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20)
            "#,
            &[
                &record.id,
                &record.kind.as_str(),
                &record.status.as_str(),
                &record.machine_id,
                &record.namespace,
                &record.task_id,
                &record.context_id,
                &record.snapshot_id,
                &record.from_agent_id,
                &record.objective,
                &record.target_role,
                &record.preferred_tier.map(|tier| tier.as_str().to_string()),
                &record.reason,
                &record.extra,
                &record.created_at,
                &record.queued_at,
                &record.assigned_worker_id,
                &record.assignment_id,
                &record.assigned_at,
                &record.completed_at,
            ],
        )?;
        let notify_sql = format!("SELECT pg_notify('{}', $1)", self.config.notify_channel);
        transaction.query(&notify_sql, &[&record.id])?;
        transaction.commit()?;
        debug!(
            op = "dispatch_publish_signal",
            signal_id = %record.id,
            signal_kind = %record.kind.as_str(),
            namespace = %record.namespace,
            task_id = %record.task_id,
            context_id = %record.context_id,
            snapshot_id = ?record.snapshot_id,
            from_agent_id = %record.from_agent_id,
            target_role = ?record.target_role,
            preferred_tier = ?record.preferred_tier.map(|tier| tier.as_str().to_string()),
            "persisted dispatch signal and emitted notify"
        );
        Ok(record)
    }

    pub fn route_once(
        &self,
        engine: &SharedContinuityKernel,
        input: RouteDispatchInput,
    ) -> Result<Option<DispatchAssignmentRecord>> {
        let machine = engine.identify_machine()?;
        let mut client = self.connect()?;
        self.ensure_schema(&mut client)?;
        let mut transaction = client.transaction()?;
        let Some(signal_row) = transaction.query_opt(
            r#"
            SELECT id, signal_kind, status, machine_id, namespace, task_id, context_id, snapshot_id,
                   from_agent_id, objective, target_role, preferred_tier, reason, payload_json,
                   created_at, queued_at, assigned_worker_id, assignment_id, assigned_at, completed_at
            FROM dispatch_signals
            WHERE status = 'queued'
            ORDER BY queued_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
            "#,
            &[],
        )? else {
            transaction.commit()?;
            return Ok(None);
        };
        let signal = read_signal_record(&signal_row)?;
        let candidates = self.list_candidate_workers(&mut transaction, &signal)?;
        let Some(worker) = select_best_worker(&signal, &candidates) else {
            transaction.commit()?;
            return Ok(None);
        };
        let resume = engine.resume(ResumeInput {
            snapshot_id: signal.snapshot_id.clone(),
            context_id: Some(signal.context_id.clone()),
            namespace: Some(signal.namespace.clone()),
            task_id: Some(signal.task_id.clone()),
            objective: signal.objective.clone(),
            token_budget: input.token_budget,
            candidate_limit: input.candidate_limit,
            agent_id: Some(worker.worker_id.clone()),
        })?;
        let pressure = apply_affect_override(
            derive_pressure_profile(worker.tier, signal.preferred_tier, signal.kind),
            signal.extra.get("affect_override"),
        );
        let envelope = build_assignment_envelope(
            &machine,
            &signal,
            &worker,
            &resume,
            pressure.clone(),
            input.token_budget,
            input.candidate_limit,
        );
        let assignment = DispatchAssignmentRecord {
            id: format!("dispatch-assignment:{}", Uuid::now_v7()),
            signal_id: signal.id.clone(),
            worker_id: worker.worker_id.clone(),
            machine_id: machine.machine_id.clone(),
            namespace: signal.namespace.clone(),
            task_id: signal.task_id.clone(),
            context_id: signal.context_id.clone(),
            snapshot_id: signal.snapshot_id.clone(),
            objective: signal.objective.clone(),
            worker_role: worker.role.clone(),
            worker_tier: worker.tier,
            status: DispatchStatus::Assigned,
            pressure: pressure.clone(),
            envelope,
            created_at: Utc::now(),
            completed_at: None,
        };
        let envelope_json = serde_json::to_value(&assignment.envelope)?;
        let pressure_json = serde_json::to_value(&assignment.pressure)?;
        transaction.execute(
            r#"
            INSERT INTO dispatch_assignments (
              id, signal_id, worker_id, machine_id, namespace, task_id, context_id, snapshot_id,
              objective, worker_role, worker_tier, status, pressure_json, envelope_json,
              created_at, completed_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
            "#,
            &[
                &assignment.id,
                &assignment.signal_id,
                &assignment.worker_id,
                &assignment.machine_id,
                &assignment.namespace,
                &assignment.task_id,
                &assignment.context_id,
                &assignment.snapshot_id,
                &assignment.objective,
                &assignment.worker_role,
                &assignment.worker_tier.as_str(),
                &assignment.status.as_str(),
                &pressure_json,
                &envelope_json,
                &assignment.created_at,
                &assignment.completed_at,
            ],
        )?;
        transaction.execute(
            r#"
            UPDATE dispatch_signals
            SET status = 'assigned',
                assigned_worker_id = $2,
                assignment_id = $3,
                assigned_at = $4
            WHERE id = $1
            "#,
            &[
                &signal.id,
                &assignment.worker_id,
                &assignment.id,
                &assignment.created_at,
            ],
        )?;
        transaction.execute(
            r#"
            UPDATE dispatch_workers
            SET focus = $2,
                namespace = $3,
                task_id = $4,
                status = 'busy'
            WHERE worker_id = $1
            "#,
            &[
                &assignment.worker_id,
                &assignment.envelope.start_summary,
                &assignment.namespace,
                &assignment.task_id,
            ],
        )?;
        transaction.commit()?;
        emit_assignment_signal(engine, &assignment, &resume, &input.router_id)?;
        debug!(
            op = "dispatch_route_assignment",
            assignment_id = %assignment.id,
            signal_id = %assignment.signal_id,
            worker_id = %assignment.worker_id,
            namespace = %assignment.namespace,
            task_id = %assignment.task_id,
            context_id = %assignment.context_id,
            snapshot_id = ?assignment.snapshot_id,
            attached_projected_lane = ?assignment.envelope.attached_projected_lane.as_ref().map(|lane| lane.projection_id.clone()),
            attached_projected_lane_source = ?assignment.envelope.attached_projected_lane_source.map(|source| source.as_str().to_string()),
            anxiety = assignment.pressure.anxiety,
            fear = assignment.pressure.fear,
            confidence = assignment.pressure.confidence,
            "assigned dispatch signal to worker"
        );
        Ok(Some(assignment))
    }

    pub fn claim_for_worker(
        &self,
        engine: &SharedContinuityKernel,
        worker_input: DispatchWorkerUpsertInput,
        token_budget: usize,
        candidate_limit: usize,
    ) -> Result<Option<DispatchAssignmentRecord>> {
        let worker = self.upsert_worker(worker_input)?;
        let machine = engine.identify_machine()?;
        let mut client = self.connect()?;
        self.ensure_schema(&mut client)?;
        let mut transaction = client.transaction()?;
        let signal_rows = transaction.query(
            r#"
            SELECT id, signal_kind, status, machine_id, namespace, task_id, context_id, snapshot_id,
                   from_agent_id, objective, target_role, preferred_tier, reason, payload_json,
                   created_at, queued_at, assigned_worker_id, assignment_id, assigned_at, completed_at
            FROM dispatch_signals
            WHERE status = 'queued'
            ORDER BY queued_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 16
            "#,
            &[],
        )?;
        let mut chosen_signal = None;
        for row in signal_rows {
            let signal = read_signal_record(&row)?;
            if worker_can_take_signal(&worker, &signal) {
                chosen_signal = Some(signal);
                break;
            }
        }
        let Some(signal) = chosen_signal else {
            transaction.commit()?;
            return Ok(None);
        };
        let (assignment, resume) = persist_assignment_for_worker(
            engine,
            &machine,
            &mut transaction,
            &signal,
            &worker,
            token_budget,
            candidate_limit,
        )?;
        transaction.commit()?;
        emit_assignment_signal(engine, &assignment, &resume, &worker.worker_id)?;
        debug!(
            op = "dispatch_claim_assignment",
            assignment_id = %assignment.id,
            signal_id = %assignment.signal_id,
            worker_id = %assignment.worker_id,
            namespace = %assignment.namespace,
            task_id = %assignment.task_id,
            context_id = %assignment.context_id,
            snapshot_id = ?assignment.snapshot_id,
            attached_projected_lane = ?assignment.envelope.attached_projected_lane.as_ref().map(|lane| lane.projection_id.clone()),
            attached_projected_lane_source = ?assignment.envelope.attached_projected_lane_source.map(|source| source.as_str().to_string()),
            anxiety = assignment.pressure.anxiety,
            fear = assignment.pressure.fear,
            confidence = assignment.pressure.confidence,
            "worker claimed dispatch assignment"
        );
        Ok(Some(assignment))
    }

    pub fn wait_and_route(
        &self,
        engine: &SharedContinuityKernel,
        input: WaitDispatchInput,
    ) -> Result<Option<DispatchAssignmentRecord>> {
        if let Some(record) = self.route_once(engine, input.route.clone())? {
            return Ok(Some(record));
        }
        let timeout = Duration::from_secs(input.timeout_secs.unwrap_or(30));
        let start = Instant::now();
        let mut client = self.connect()?;
        self.ensure_schema(&mut client)?;
        let listen_sql = format!("LISTEN {}", self.config.notify_channel);
        client.batch_execute(&listen_sql)?;
        let mut notifications = client.notifications();
        while start.elapsed() < timeout {
            let remaining = timeout.saturating_sub(start.elapsed());
            let messages = notifications.timeout_iter(remaining).collect::<Vec<_>>()?;
            if !messages.is_empty() {
                if let Some(record) = self.route_once(engine, input.route.clone())? {
                    return Ok(Some(record));
                }
            }
        }
        Ok(None)
    }

    pub fn wait_and_claim_for_worker(
        &self,
        engine: &SharedContinuityKernel,
        worker_input: DispatchWorkerUpsertInput,
        token_budget: usize,
        candidate_limit: usize,
        timeout_secs: u64,
    ) -> Result<Option<DispatchAssignmentRecord>> {
        if let Some(record) =
            self.claim_for_worker(engine, worker_input.clone(), token_budget, candidate_limit)?
        {
            return Ok(Some(record));
        }
        let timeout = Duration::from_secs(timeout_secs);
        let start = Instant::now();
        let mut client = self.connect()?;
        self.ensure_schema(&mut client)?;
        let listen_sql = format!("LISTEN {}", self.config.notify_channel);
        client.batch_execute(&listen_sql)?;
        let mut notifications = client.notifications();
        while start.elapsed() < timeout {
            let remaining = timeout.saturating_sub(start.elapsed());
            let messages = notifications.timeout_iter(remaining).collect::<Vec<_>>()?;
            if !messages.is_empty() {
                if let Some(record) = self.claim_for_worker(
                    engine,
                    worker_input.clone(),
                    token_budget,
                    candidate_limit,
                )? {
                    return Ok(Some(record));
                }
            }
        }
        Ok(None)
    }

    pub fn complete_assignment(
        &self,
        input: CompleteAssignmentInput,
    ) -> Result<DispatchAssignmentRecord> {
        let mut client = self.connect()?;
        self.ensure_schema(&mut client)?;
        let mut transaction = client.transaction()?;
        let Some(row) = transaction.query_opt(
            r#"
            SELECT id, signal_id, worker_id, machine_id, namespace, task_id, context_id, snapshot_id,
                   objective, worker_role, worker_tier, status, pressure_json, envelope_json,
                   created_at, completed_at
            FROM dispatch_assignments
            WHERE id = $1
            FOR UPDATE
            "#,
            &[&input.assignment_id],
        )? else {
            anyhow::bail!("assignment {} not found", input.assignment_id);
        };
        let mut assignment = read_assignment_record(&row)?;
        if let Some(worker_id) = &input.worker_id {
            if worker_id != &assignment.worker_id {
                anyhow::bail!(
                    "assignment {} belongs to worker {}, not {}",
                    assignment.id,
                    assignment.worker_id,
                    worker_id
                );
            }
        }
        let completed_at = Utc::now();
        let final_status = if input.failed {
            DispatchStatus::Failed
        } else {
            DispatchStatus::Completed
        };
        transaction.execute(
            "UPDATE dispatch_assignments SET status = $2, completed_at = $3 WHERE id = $1",
            &[&assignment.id, &final_status.as_str(), &completed_at],
        )?;
        transaction.execute(
            "UPDATE dispatch_signals SET status = $2, completed_at = $3 WHERE assignment_id = $1",
            &[&assignment.id, &final_status.as_str(), &completed_at],
        )?;
        transaction.execute(
            "UPDATE dispatch_workers SET status = 'idle', focus = '' WHERE worker_id = $1",
            &[&assignment.worker_id],
        )?;
        transaction.commit()?;
        assignment.status = final_status;
        assignment.completed_at = Some(completed_at);
        debug!(
            op = "dispatch_complete_assignment",
            assignment_id = %assignment.id,
            signal_id = %assignment.signal_id,
            worker_id = %assignment.worker_id,
            namespace = %assignment.namespace,
            task_id = %assignment.task_id,
            final_status = %assignment.status.as_str(),
            completed_at = ?assignment.completed_at,
            "completed dispatch assignment"
        );
        Ok(assignment)
    }

    pub fn snapshot_for_organism(
        &self,
        namespace_filter: Option<&str>,
    ) -> Result<DispatchOrganismSnapshot> {
        let mut client = self.connect()?;
        self.ensure_schema(&mut client)?;
        let live_cutoff =
            Utc::now() - chrono::Duration::seconds(self.config.worker_stale_secs as i64);
        let workers = self
            .list_workers_with_load(&mut client)?
            .into_iter()
            .filter(|worker| worker.last_seen_at >= live_cutoff)
            .filter(|worker| {
                worker_visible_in_namespace(worker.namespace.as_deref(), namespace_filter)
            })
            .map(|worker| dispatch_worker_presence(worker, namespace_filter))
            .collect::<Vec<_>>();
        let assignments = self
            .list_active_assignments(&mut client, namespace_filter)?
            .into_iter()
            .map(dispatch_assignment_presence)
            .collect::<Vec<_>>();
        let signals_queued = self.count_queued_signals(&mut client, namespace_filter)?;
        Ok(DispatchOrganismSnapshot {
            configured: true,
            reachable: true,
            notify_channel: Some(self.config.notify_channel.clone()),
            worker_active_window_secs: Some(self.config.worker_stale_secs),
            workers_active: workers.len(),
            signals_queued,
            assignments_active: assignments.len(),
            workers,
            assignments,
            error: None,
        })
    }

    pub fn render_metrics(root: impl AsRef<Path>) -> String {
        let root = root.as_ref().to_path_buf();
        match run_dispatch_blocking(move || {
            match Self::from_root(&root) {
            Ok(Some(spine)) => match spine.render_metrics_inner() {
                Ok(mut body) => {
                    body.insert_str(
                        0,
                        "# HELP ice_dispatch_up Whether the PostgreSQL dispatch spine is reachable.\n# TYPE ice_dispatch_up gauge\nice_dispatch_up 1\n",
                    );
                    Ok(body)
                }
                Err(error) => Ok(dispatch_down_metrics(&format!("{error:#}"))),
            },
            Ok(None) => Ok(
                "# HELP ice_dispatch_up Whether the PostgreSQL dispatch spine is reachable.\n# TYPE ice_dispatch_up gauge\nice_dispatch_up 0\n".to_string(),
            ),
            Err(error) => Ok(dispatch_down_metrics(&format!("{error:#}"))),
        }
        }) {
            Ok(text) => text,
            Err(error) => dispatch_down_metrics(&format!("{error:#}")),
        }
    }

    fn render_metrics_inner(&self) -> Result<String> {
        let mut client = self.connect()?;
        self.ensure_schema(&mut client)?;
        let mut text = String::new();
        text.push_str(
            "# HELP ice_dispatch_workers_active Active dispatch workers within the freshness window.\n# TYPE ice_dispatch_workers_active gauge\n",
        );
        text.push_str(
            "# HELP ice_dispatch_worker_connected Live dispatch worker badge rows from the PostgreSQL dispatch spine.\n# TYPE ice_dispatch_worker_connected gauge\n",
        );
        text.push_str(
            "# HELP ice_dispatch_signals Durable dispatch signals grouped by kind, target, and status.\n# TYPE ice_dispatch_signals gauge\n",
        );
        text.push_str(
            "# HELP ice_dispatch_assignments Durable dispatch assignments grouped by worker and status.\n# TYPE ice_dispatch_assignments gauge\n",
        );
        text.push_str(
            "# HELP ice_dispatch_projection_workers Live dispatch-worker lane projections visible to the machine-level observability path.\n# TYPE ice_dispatch_projection_workers gauge\n",
        );
        text.push_str(
            "# HELP ice_dispatch_projection_assignments Active dispatch assignments projected onto dispatch-worker lanes.\n# TYPE ice_dispatch_projection_assignments gauge\n",
        );
        text.push_str(
            "# HELP ice_dispatch_projection_assignment_anxiety Max active dispatch anxiety projected onto each dispatch-worker lane.\n# TYPE ice_dispatch_projection_assignment_anxiety gauge\n",
        );
        text.push_str(
            "# HELP ice_dispatch_projection_assignment_sources Active dispatch assignments projected onto each lane, split by attached-lane provenance source.\n# TYPE ice_dispatch_projection_assignment_sources gauge\n",
        );

        for row in client.query(
            r#"
            SELECT role, tier, COUNT(*)::BIGINT AS worker_count
            FROM dispatch_workers
            WHERE last_seen_at >= NOW() - ($1::BIGINT * INTERVAL '1 second')
            GROUP BY role, tier
            ORDER BY role, tier
            "#,
            &[&(self.config.worker_stale_secs as i64)],
        )? {
            let role: String = row.get("role");
            let tier: String = row.get("tier");
            let count: i64 = row.get("worker_count");
            text.push_str(&format!(
                "ice_dispatch_workers_active{{role=\"{}\",tier=\"{}\"}} {}\n",
                prometheus_label_value(&role),
                prometheus_label_value(&tier),
                count
            ));
        }

        for worker in self.list_workers_with_load(&mut client)? {
            let labels = format!(
                "worker_id=\"{}\",display_name=\"{}\",role=\"{}\",tier=\"{}\",agent_type=\"{}\",model=\"{}\",focus=\"{}\",namespace=\"{}\",task_id=\"{}\"",
                prometheus_label_value(&worker.worker_id),
                prometheus_label_value(&compact_metric_label(&worker.display_name, 64)),
                prometheus_label_value(&worker.role),
                prometheus_label_value(worker.tier.as_str()),
                prometheus_label_value(&worker.agent_type),
                prometheus_label_value(&compact_metric_label(&worker.model, 64)),
                prometheus_label_value(&compact_metric_label(&worker.focus, 120)),
                prometheus_label_value(worker.namespace.as_deref().unwrap_or("")),
                prometheus_label_value(worker.task_id.as_deref().unwrap_or("")),
            );
            text.push_str(&format!("ice_dispatch_worker_connected{{{labels}}} 1\n"));
        }

        for row in client.query(
            r#"
            SELECT signal_kind, COALESCE(target_role, '') AS target_role,
                   COALESCE(preferred_tier, '') AS preferred_tier, status,
                   COUNT(*)::BIGINT AS signal_count
            FROM dispatch_signals
            GROUP BY signal_kind, COALESCE(target_role, ''), COALESCE(preferred_tier, ''), status
            ORDER BY signal_kind, status
            "#,
            &[],
        )? {
            let kind: String = row.get("signal_kind");
            let target_role: String = row.get("target_role");
            let preferred_tier: String = row.get("preferred_tier");
            let status: String = row.get("status");
            let count: i64 = row.get("signal_count");
            text.push_str(&format!(
                "ice_dispatch_signals{{kind=\"{}\",target_role=\"{}\",preferred_tier=\"{}\",status=\"{}\"}} {}\n",
                prometheus_label_value(&kind),
                prometheus_label_value(&target_role),
                prometheus_label_value(&preferred_tier),
                prometheus_label_value(&status),
                count
            ));
        }

        for row in client.query(
            r#"
            SELECT a.worker_id, w.role, w.tier, a.status, COUNT(*)::BIGINT AS assignment_count
            FROM dispatch_assignments a
            JOIN dispatch_workers w ON w.worker_id = a.worker_id
            GROUP BY a.worker_id, w.role, w.tier, a.status
            ORDER BY a.worker_id, a.status
            "#,
            &[],
        )? {
            let worker_id: String = row.get("worker_id");
            let role: String = row.get("role");
            let tier: String = row.get("tier");
            let status: String = row.get("status");
            let count: i64 = row.get("assignment_count");
            text.push_str(&format!(
                "ice_dispatch_assignments{{worker_id=\"{}\",role=\"{}\",tier=\"{}\",status=\"{}\"}} {}\n",
                prometheus_label_value(&worker_id),
                prometheus_label_value(&role),
                prometheus_label_value(&tier),
                prometheus_label_value(&status),
                count
            ));
        }

        let live_cutoff =
            Utc::now() - chrono::Duration::seconds(self.config.worker_stale_secs as i64);
        let workers = self
            .list_workers_with_load(&mut client)?
            .into_iter()
            .filter(|worker| worker.last_seen_at >= live_cutoff)
            .map(|worker| dispatch_worker_presence(worker, None))
            .collect::<Vec<_>>();
        let assignments = self
            .list_active_assignments(&mut client, None)?
            .into_iter()
            .map(dispatch_assignment_presence)
            .collect::<Vec<_>>();
        let mut projections = BTreeMap::<String, DispatchProjectionMetricState>::new();
        for worker in &workers {
            let entry = projections
                .entry(worker.projected_lane.projection_id.clone())
                .or_insert_with(|| {
                    DispatchProjectionMetricState::new(
                        &worker.projected_lane,
                        worker.namespace.as_deref(),
                        worker.task_id.as_deref(),
                    )
                });
            entry.worker_count += 1;
        }
        for assignment in &assignments {
            record_dispatch_projection_assignment(
                &mut projections,
                &assignment.projected_lane,
                Some(assignment.namespace.as_str()),
                Some(assignment.task_id.as_str()),
                assignment.pressure.anxiety,
                None,
            );
            if let Some(attached_lane) = assignment.attached_projected_lane.as_ref() {
                if attached_lane.projection_id != assignment.projected_lane.projection_id {
                    record_dispatch_projection_assignment(
                        &mut projections,
                        attached_lane,
                        Some(assignment.namespace.as_str()),
                        Some(assignment.task_id.as_str()),
                        assignment.pressure.anxiety,
                        assignment.attached_projected_lane_source,
                    );
                }
            }
        }
        for projection in projections.into_values() {
            let labels = projection_metric_labels(&projection);
            text.push_str(&format!(
                "ice_dispatch_projection_workers{{{labels}}} {}\n",
                projection.worker_count
            ));
            text.push_str(&format!(
                "ice_dispatch_projection_assignments{{{labels}}} {}\n",
                projection.assignment_count
            ));
            text.push_str(&format!(
                "ice_dispatch_projection_assignment_anxiety{{{labels}}} {:.6}\n",
                projection.assignment_anxiety_max
            ));
            text.push_str(&format!(
                "ice_dispatch_projection_assignment_sources{{{labels},source=\"explicit_cli\"}} {}\n",
                projection.assignment_explicit_cli_count
            ));
            text.push_str(&format!(
                "ice_dispatch_projection_assignment_sources{{{labels},source=\"live_badge_opt_in\"}} {}\n",
                projection.assignment_live_badge_opt_in_count
            ));
        }

        Ok(text)
    }

    fn connect(&self) -> Result<Client> {
        Client::connect(&self.config.database_url, NoTls)
            .with_context(|| "connecting to PostgreSQL dispatch spine")
    }

    fn ensure_schema(&self, client: &mut Client) -> Result<()> {
        client.batch_execute(
            r#"
            CREATE TABLE IF NOT EXISTS dispatch_workers (
              worker_id TEXT PRIMARY KEY,
              display_name TEXT NOT NULL,
              role TEXT NOT NULL,
              tier TEXT NOT NULL,
              agent_type TEXT NOT NULL,
              model TEXT NOT NULL,
              capabilities_json JSONB NOT NULL DEFAULT '[]'::jsonb,
              max_parallelism INTEGER NOT NULL,
              status TEXT NOT NULL,
              focus TEXT NOT NULL DEFAULT '',
              namespace TEXT,
              task_id TEXT,
              metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
              last_seen_at TIMESTAMPTZ NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dispatch_signals (
              id TEXT PRIMARY KEY,
              signal_kind TEXT NOT NULL,
              status TEXT NOT NULL,
              machine_id TEXT NOT NULL,
              namespace TEXT NOT NULL,
              task_id TEXT NOT NULL,
              context_id TEXT NOT NULL,
              snapshot_id TEXT,
              from_agent_id TEXT NOT NULL,
              objective TEXT NOT NULL,
              target_role TEXT,
              preferred_tier TEXT,
              reason TEXT,
              payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
              created_at TIMESTAMPTZ NOT NULL,
              queued_at TIMESTAMPTZ NOT NULL,
              assigned_worker_id TEXT,
              assignment_id TEXT,
              assigned_at TIMESTAMPTZ,
              completed_at TIMESTAMPTZ
            );

            CREATE INDEX IF NOT EXISTS idx_dispatch_signals_status_queue
              ON dispatch_signals(status, queued_at);
            CREATE INDEX IF NOT EXISTS idx_dispatch_signals_context
              ON dispatch_signals(namespace, task_id, context_id);

            CREATE TABLE IF NOT EXISTS dispatch_assignments (
              id TEXT PRIMARY KEY,
              signal_id TEXT NOT NULL REFERENCES dispatch_signals(id) ON DELETE CASCADE,
              worker_id TEXT NOT NULL REFERENCES dispatch_workers(worker_id) ON DELETE CASCADE,
              machine_id TEXT NOT NULL,
              namespace TEXT NOT NULL,
              task_id TEXT NOT NULL,
              context_id TEXT NOT NULL,
              snapshot_id TEXT,
              objective TEXT NOT NULL,
              worker_role TEXT NOT NULL,
              worker_tier TEXT NOT NULL,
              status TEXT NOT NULL,
              pressure_json JSONB NOT NULL,
              envelope_json JSONB NOT NULL,
              created_at TIMESTAMPTZ NOT NULL,
              completed_at TIMESTAMPTZ
            );

            CREATE INDEX IF NOT EXISTS idx_dispatch_assignments_worker_status
              ON dispatch_assignments(worker_id, status, created_at DESC);
            "#,
        )?;
        Ok(())
    }

    fn worker_record(&self, client: &mut Client, worker_id: &str) -> Result<DispatchWorkerRecord> {
        let row = client
            .query_one(
                r#"
                SELECT w.worker_id, w.display_name, w.role, w.tier, w.agent_type, w.model,
                       w.capabilities_json, w.max_parallelism, w.status, w.focus, w.namespace,
                       w.task_id, w.metadata_json, w.last_seen_at,
                       COALESCE((
                         SELECT COUNT(*)::BIGINT
                         FROM dispatch_assignments a
                         WHERE a.worker_id = w.worker_id
                           AND a.status = 'assigned'
                       ), 0) AS active_assignment_count
                FROM dispatch_workers w
                WHERE w.worker_id = $1
                "#,
                &[&worker_id],
            )
            .with_context(|| format!("reading worker {worker_id}"))?;
        read_worker_record(&row)
    }

    fn list_workers_with_load(&self, client: &mut Client) -> Result<Vec<DispatchWorkerRecord>> {
        client
            .query(
                r#"
                SELECT w.worker_id, w.display_name, w.role, w.tier, w.agent_type, w.model,
                       w.capabilities_json, w.max_parallelism, w.status, w.focus, w.namespace,
                       w.task_id, w.metadata_json, w.last_seen_at,
                       COALESCE((
                         SELECT COUNT(*)::BIGINT
                         FROM dispatch_assignments a
                         WHERE a.worker_id = w.worker_id
                           AND a.status = 'assigned'
                       ), 0) AS active_assignment_count
                FROM dispatch_workers w
                ORDER BY w.worker_id
                "#,
                &[],
            )?
            .iter()
            .map(read_worker_record)
            .collect()
    }

    fn list_active_assignments(
        &self,
        client: &mut Client,
        namespace_filter: Option<&str>,
    ) -> Result<Vec<DispatchAssignmentRecord>> {
        let rows = match namespace_filter {
            Some(namespace) => client.query(
                r#"
                SELECT id, signal_id, worker_id, machine_id, namespace, task_id, context_id, snapshot_id,
                       objective, worker_role, worker_tier, status, pressure_json, envelope_json,
                       created_at, completed_at
                FROM dispatch_assignments
                WHERE status = 'assigned'
                  AND namespace = $1
                ORDER BY created_at DESC
                "#,
                &[&namespace],
            )?,
            None => client.query(
                r#"
                SELECT id, signal_id, worker_id, machine_id, namespace, task_id, context_id, snapshot_id,
                       objective, worker_role, worker_tier, status, pressure_json, envelope_json,
                       created_at, completed_at
                FROM dispatch_assignments
                WHERE status = 'assigned'
                ORDER BY created_at DESC
                "#,
                &[],
            )?,
        };
        rows.iter().map(read_assignment_record).collect()
    }

    fn count_queued_signals(
        &self,
        client: &mut Client,
        namespace_filter: Option<&str>,
    ) -> Result<usize> {
        let count = match namespace_filter {
            Some(namespace) => client.query_one(
                "SELECT COUNT(*)::BIGINT AS count FROM dispatch_signals WHERE status = 'queued' AND namespace = $1",
                &[&namespace],
            )?,
            None => client.query_one(
                "SELECT COUNT(*)::BIGINT AS count FROM dispatch_signals WHERE status = 'queued'",
                &[],
            )?,
        }
        .get::<_, i64>("count");
        Ok(count.max(0) as usize)
    }

    fn list_candidate_workers(
        &self,
        transaction: &mut postgres::Transaction<'_>,
        signal: &DispatchSignalRecord,
    ) -> Result<Vec<DispatchWorkerRecord>> {
        transaction
            .query(
                r#"
                SELECT w.worker_id, w.display_name, w.role, w.tier, w.agent_type, w.model,
                       w.capabilities_json, w.max_parallelism, w.status, w.focus, w.namespace,
                       w.task_id, w.metadata_json, w.last_seen_at,
                       COALESCE((
                         SELECT COUNT(*)::BIGINT
                         FROM dispatch_assignments a
                         WHERE a.worker_id = w.worker_id
                           AND a.status = 'assigned'
                       ), 0) AS active_assignment_count
                FROM dispatch_workers w
                WHERE w.last_seen_at >= NOW() - ($1::BIGINT * INTERVAL '1 second')
                  AND (
                    SELECT COUNT(*)::BIGINT
                    FROM dispatch_assignments a
                    WHERE a.worker_id = w.worker_id
                      AND a.status = 'assigned'
                  ) < w.max_parallelism
                ORDER BY w.last_seen_at DESC, w.worker_id ASC
                "#,
                &[&(self.config.worker_stale_secs as i64)],
            )?
            .iter()
            .map(read_worker_record)
            .filter_map(|record| match record {
                Ok(worker) if worker_can_take_signal(&worker, signal) => Some(Ok(worker)),
                Ok(_) => None,
                Err(error) => Some(Err(error)),
            })
            .collect()
    }
}

pub fn append_metrics(text: &mut String, root: impl AsRef<Path>) {
    #[cfg(test)]
    {
        let root = root.as_ref();
        let maybe_hook = APPEND_METRICS_TEST_HOOK.lock().unwrap().take();
        if let Some(hook) = maybe_hook {
            if hook.root == root {
                hook.entered_tx.send(()).unwrap();
                hook.release_rx
                    .recv_timeout(Duration::from_secs(1))
                    .unwrap();
            } else {
                *APPEND_METRICS_TEST_HOOK.lock().unwrap() = Some(hook);
            }
        }
    }
    text.push_str(&DispatchSpine::render_metrics(root));
}

#[cfg(test)]
pub(crate) fn install_append_metrics_test_hook(
    root: PathBuf,
    entered_tx: std::sync::mpsc::Sender<()>,
    release_rx: std::sync::mpsc::Receiver<()>,
) {
    *APPEND_METRICS_TEST_HOOK.lock().unwrap() = Some(AppendMetricsTestHook {
        root,
        entered_tx,
        release_rx,
    });
}

pub fn organism_snapshot(
    root: impl AsRef<Path>,
    namespace_filter: Option<&str>,
) -> DispatchOrganismSnapshot {
    let root = root.as_ref();
    match DispatchConfig::load(root) {
        Ok(Some(config)) => {
            let spine = DispatchSpine {
                config: config.clone(),
            };
            match spine.snapshot_for_organism(namespace_filter) {
                Ok(snapshot) => snapshot,
                Err(error) => DispatchOrganismSnapshot::unreachable(config, error),
            }
        }
        Ok(None) => DispatchOrganismSnapshot::unconfigured(),
        Err(error) => DispatchOrganismSnapshot::config_error(error),
    }
}

fn run_dispatch_blocking<T, F>(f: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    if Handle::try_current().is_ok() {
        thread::spawn(f)
            .join()
            .map_err(|_| anyhow::anyhow!("dispatch blocking task panicked"))?
    } else {
        f()
    }
}

pub fn default_worker_active_window_secs() -> i64 {
    DEFAULT_WORKER_STALE_SECS as i64
}

pub async fn init_dispatch_schema(database_url: &str) -> Result<()> {
    let config = DispatchConfig {
        database_url: database_url.to_string(),
        notify_channel: default_dispatch_notify_channel(),
        worker_stale_secs: default_worker_stale_secs(),
    };
    let spine = DispatchSpine { config };
    let mut client = spine.connect()?;
    spine.ensure_schema(&mut client)
}

pub async fn upsert_dispatch_worker(
    database_url: &str,
    input: DispatchWorkerUpsertInput,
) -> Result<DispatchWorkerRecord> {
    let spine = DispatchSpine {
        config: DispatchConfig {
            database_url: database_url.to_string(),
            notify_channel: default_dispatch_notify_channel(),
            worker_stale_secs: default_worker_stale_secs(),
        },
    };
    spine.upsert_worker(input)
}

pub async fn dispatch_stats(
    database_url: &str,
    channel: &str,
    active_window_secs: i64,
) -> Result<serde_json::Value> {
    let spine = DispatchSpine {
        config: DispatchConfig {
            database_url: database_url.to_string(),
            notify_channel: channel.to_string(),
            worker_stale_secs: active_window_secs.max(1) as u64,
        },
    };
    let mut client = spine.connect()?;
    spine.ensure_schema(&mut client)?;
    let workers = spine.list_workers_with_load(&mut client)?;
    let signals = client
        .query_one(
            "SELECT COUNT(*)::BIGINT AS count FROM dispatch_signals WHERE status = 'queued'",
            &[],
        )?
        .get::<_, i64>("count");
    let assignments = client
        .query_one(
            "SELECT COUNT(*)::BIGINT AS count FROM dispatch_assignments WHERE status = 'assigned'",
            &[],
        )?
        .get::<_, i64>("count");
    Ok(serde_json::json!({
        "channel": channel,
        "worker_active_window_secs": active_window_secs,
        "workers_active": workers.len(),
        "signals_queued": signals,
        "assignments_active": assignments,
        "workers": workers,
    }))
}

pub async fn dispatch_complete(
    engine: &SharedContinuityKernel,
    database_url: &str,
    channel: &str,
    input: DispatchCompleteInput,
) -> Result<serde_json::Value> {
    let machine = engine.identify_machine()?;
    let objective = input
        .objective
        .clone()
        .unwrap_or_else(|| input.title.clone());
    let context = engine.read_context(ReadContextInput {
        context_id: input.context_id.clone(),
        namespace: input.namespace.clone(),
        task_id: input.task_id.clone(),
        objective: objective.clone(),
        token_budget: input.token_budget,
        selector: input.selector.clone(),
        agent_id: Some(input.agent_id.clone()),
        session_id: None,
        view_id: None,
        include_resolved: true,
        candidate_limit: input.candidate_limit,
    })?;
    let outcome = engine.record_outcome(OutcomeInput {
        context_id: context.context.id.clone(),
        agent_id: input.agent_id.clone(),
        title: input.title.clone(),
        result: input.result.clone(),
        quality: input.quality,
        failures: input.failures.clone(),
        dimensions: Vec::new(),
        extra: input.extra.clone(),
    })?;
    let snapshot = engine.snapshot(SnapshotInput {
        context_id: Some(context.context.id.clone()),
        namespace: Some(context.context.namespace.clone()),
        task_id: Some(context.context.task_id.clone()),
        objective: Some(objective.clone()),
        selector: input.selector.clone(),
        resolution: input.snapshot_resolution,
        token_budget: input.token_budget,
        candidate_limit: input.candidate_limit,
        owner_agent_id: Some(input.agent_id.clone()),
    })?;
    let extra = serde_json::json!({
        "result": input.result,
        "quality": input.quality,
        "failures": input.failures,
        "affect_override": input.affect_override,
        "target_tier": input.target_tier,
        "extra": input.extra,
    });
    let spine = DispatchSpine {
        config: DispatchConfig {
            database_url: database_url.to_string(),
            notify_channel: channel.to_string(),
            worker_stale_secs: default_worker_stale_secs(),
        },
    };
    let signal = spine.publish_signal(
        &machine,
        PublishDispatchSignalInput {
            kind: input.kind,
            from_agent_id: input.agent_id,
            context_id: context.context.id.clone(),
            namespace: context.context.namespace.clone(),
            task_id: context.context.task_id.clone(),
            snapshot_id: Some(snapshot.id.clone()),
            objective,
            target_role: None,
            preferred_tier: Some(input.target_tier),
            reason: Some(input.title),
            extra,
        },
    )?;
    Ok(serde_json::json!({
        "outcome": outcome,
        "snapshot": snapshot,
        "signal": signal,
    }))
}

pub async fn claim_dispatch_assignment(
    engine: &SharedContinuityKernel,
    database_url: &str,
    input: DispatchClaimInput,
) -> Result<Option<DispatchAssignmentRecord>> {
    let spine = DispatchSpine {
        config: DispatchConfig {
            database_url: database_url.to_string(),
            notify_channel: default_dispatch_notify_channel(),
            worker_stale_secs: default_worker_stale_secs(),
        },
    };
    spine.claim_for_worker(
        engine,
        input.worker,
        input.token_budget,
        input.candidate_limit,
    )
}

pub async fn listen_for_dispatch_assignment(
    engine: &SharedContinuityKernel,
    database_url: &str,
    channel: &str,
    input: DispatchClaimInput,
    wait_secs: u64,
) -> Result<Option<DispatchAssignmentRecord>> {
    let spine = DispatchSpine {
        config: DispatchConfig {
            database_url: database_url.to_string(),
            notify_channel: channel.to_string(),
            worker_stale_secs: default_worker_stale_secs(),
        },
    };
    spine.wait_and_claim_for_worker(
        engine,
        input.worker,
        input.token_budget,
        input.candidate_limit,
        wait_secs,
    )
}

pub async fn acknowledge_dispatch_assignment(
    database_url: &str,
    worker_id: &str,
    message_id: &str,
) -> Result<DispatchAssignmentRecord> {
    let spine = DispatchSpine {
        config: DispatchConfig {
            database_url: database_url.to_string(),
            notify_channel: default_dispatch_notify_channel(),
            worker_stale_secs: default_worker_stale_secs(),
        },
    };
    spine.complete_assignment(CompleteAssignmentInput {
        assignment_id: message_id.to_string(),
        worker_id: Some(worker_id.to_string()),
        failed: false,
    })
}

fn persist_assignment_for_worker(
    engine: &SharedContinuityKernel,
    machine: &MachineProfile,
    transaction: &mut postgres::Transaction<'_>,
    signal: &DispatchSignalRecord,
    worker: &DispatchWorkerRecord,
    token_budget: usize,
    candidate_limit: usize,
) -> Result<(DispatchAssignmentRecord, ResumeRecord)> {
    let resume = engine.resume(ResumeInput {
        snapshot_id: signal.snapshot_id.clone(),
        context_id: Some(signal.context_id.clone()),
        namespace: Some(signal.namespace.clone()),
        task_id: Some(signal.task_id.clone()),
        objective: signal.objective.clone(),
        token_budget,
        candidate_limit,
        agent_id: Some(worker.worker_id.clone()),
    })?;
    let pressure = apply_affect_override(
        derive_pressure_profile(worker.tier, signal.preferred_tier, signal.kind),
        signal.extra.get("affect_override"),
    );
    let envelope = build_assignment_envelope(
        machine,
        signal,
        worker,
        &resume,
        pressure.clone(),
        token_budget,
        candidate_limit,
    );
    let assignment = DispatchAssignmentRecord {
        id: format!("dispatch-assignment:{}", Uuid::now_v7()),
        signal_id: signal.id.clone(),
        worker_id: worker.worker_id.clone(),
        machine_id: machine.machine_id.clone(),
        namespace: signal.namespace.clone(),
        task_id: signal.task_id.clone(),
        context_id: signal.context_id.clone(),
        snapshot_id: signal.snapshot_id.clone(),
        objective: signal.objective.clone(),
        worker_role: worker.role.clone(),
        worker_tier: worker.tier,
        status: DispatchStatus::Assigned,
        pressure: pressure.clone(),
        envelope,
        created_at: Utc::now(),
        completed_at: None,
    };
    let envelope_json = serde_json::to_value(&assignment.envelope)?;
    let pressure_json = serde_json::to_value(&assignment.pressure)?;
    transaction.execute(
        r#"
        INSERT INTO dispatch_assignments (
          id, signal_id, worker_id, machine_id, namespace, task_id, context_id, snapshot_id,
          objective, worker_role, worker_tier, status, pressure_json, envelope_json,
          created_at, completed_at
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
        "#,
        &[
            &assignment.id,
            &assignment.signal_id,
            &assignment.worker_id,
            &assignment.machine_id,
            &assignment.namespace,
            &assignment.task_id,
            &assignment.context_id,
            &assignment.snapshot_id,
            &assignment.objective,
            &assignment.worker_role,
            &assignment.worker_tier.as_str(),
            &assignment.status.as_str(),
            &pressure_json,
            &envelope_json,
            &assignment.created_at,
            &assignment.completed_at,
        ],
    )?;
    transaction.execute(
        r#"
        UPDATE dispatch_signals
        SET status = 'assigned',
            assigned_worker_id = $2,
            assignment_id = $3,
            assigned_at = $4
        WHERE id = $1
        "#,
        &[
            &signal.id,
            &assignment.worker_id,
            &assignment.id,
            &assignment.created_at,
        ],
    )?;
    transaction.execute(
        r#"
        UPDATE dispatch_workers
        SET focus = $2,
            status = 'busy',
            namespace = $3,
            task_id = $4
        WHERE worker_id = $1
        "#,
        &[
            &assignment.worker_id,
            &assignment.envelope.start_summary,
            &assignment.namespace,
            &assignment.task_id,
        ],
    )?;
    Ok((assignment, resume))
}

fn emit_assignment_signal(
    engine: &SharedContinuityKernel,
    assignment: &DispatchAssignmentRecord,
    resume: &ResumeRecord,
    router_id: &str,
) -> Result<()> {
    engine.publish_signal(SignalInput {
        context_id: assignment.context_id.clone(),
        agent_id: router_id.to_string(),
        title: format!(
            "Dispatch assigned {} ({})",
            assignment.worker_id,
            assignment.worker_tier.as_str()
        ),
        body: format!(
            "{} should resume {} in {} / {} from {}.",
            assignment.worker_id,
            assignment.objective,
            assignment.namespace,
            assignment.task_id,
            assignment
                .snapshot_id
                .as_deref()
                .unwrap_or(assignment.context_id.as_str())
        ),
        dimensions: Vec::new(),
        extra: serde_json::json!({
            "dispatch": {
                "assignment_id": assignment.id,
                "signal_id": assignment.signal_id,
                "worker_id": assignment.worker_id,
                "worker_tier": assignment.worker_tier,
                "router_id": router_id,
                "pack_id": resume.context.pack.id,
            }
        }),
    })?;
    if assignment.pressure.anxiety >= 0.6 {
        engine.publish_coordination_signal(CoordinationSignalInput {
            context_id: assignment.context_id.clone(),
            agent_id: router_id.to_string(),
            title: format!("Dispatch anxiety for {}", assignment.worker_id),
            body: format!(
                "{} is a {}-tier worker. Resume from {} and verify constraints plus scars before changing code.",
                assignment.worker_id,
                assignment.worker_tier.as_str(),
                assignment
                    .snapshot_id
                    .as_deref()
                    .unwrap_or(assignment.context_id.as_str())
            ),
            lane: CoordinationLane::Anxiety,
            target_agent_id: Some(assignment.worker_id.clone()),
            target_projected_lane: Some(CoordinationProjectedLane {
                projection_id: format!("dispatch:worker:{}", assignment.worker_id),
                projection_kind: "dispatch_worker".to_string(),
                label: format!("dispatch {}", assignment.worker_id),
                resource: Some(format!("dispatch/worker/{}", assignment.worker_id)),
                repo_root: None,
                branch: None,
                task_id: Some(assignment.task_id.clone()),
            }),
            claim_id: None,
            resource: Some(format!("dispatch/assignment/{}", assignment.id)),
            severity: Some(CoordinationSeverity::Warn),
            projection_ids: vec![format!("dispatch:worker:{}", assignment.worker_id)],
            projected_lanes: Vec::new(),
            extra: serde_json::json!({
                "dispatch": {
                    "assignment_id": assignment.id,
                    "signal_id": assignment.signal_id,
                    "pressure": assignment.pressure,
                }
            }),
        })?;
    }
    Ok(())
}

fn build_assignment_envelope(
    machine: &MachineProfile,
    signal: &DispatchSignalRecord,
    worker: &DispatchWorkerRecord,
    resume: &ResumeRecord,
    pressure: DispatchPressureProfile,
    token_budget: usize,
    candidate_limit: usize,
) -> DispatchAssignmentEnvelope {
    let attached_projected_lane =
        attached_projected_lane_from_metadata(&worker.metadata, worker.task_id.as_deref());
    let attached_projected_lane_source =
        attached_projected_lane_source_from_metadata(&worker.metadata);
    let preview = DispatchContextPreview {
        pack_id: resume.context.pack.id.clone(),
        latest_snapshot_id: resume.context.latest_snapshot_id.clone(),
        pack_used_tokens: resume.context.pack.used_tokens,
        hot_memory: resume
            .context
            .pack
            .items
            .iter()
            .take(6)
            .map(|item| compact_line(&item.body, 140))
            .collect(),
        decisions: resume
            .context
            .decisions
            .iter()
            .take(4)
            .map(|item| item.title.clone())
            .collect(),
        constraints: resume
            .context
            .constraints
            .iter()
            .take(4)
            .map(|item| item.title.clone())
            .collect(),
        operational_scars: resume
            .context
            .operational_scars
            .iter()
            .take(4)
            .map(|item| item.title.clone())
            .collect(),
        open_threads: resume
            .context
            .open_threads
            .iter()
            .take(4)
            .map(|item| item.title.clone())
            .collect(),
    };
    let start_summary = preview
        .constraints
        .first()
        .cloned()
        .or_else(|| preview.operational_scars.first().cloned())
        .or_else(|| preview.hot_memory.first().cloned())
        .unwrap_or_else(|| signal.objective.clone());
    DispatchAssignmentEnvelope {
        machine: machine.clone(),
        signal_id: signal.id.clone(),
        worker_id: worker.worker_id.clone(),
        worker_display_name: worker.display_name.clone(),
        worker_role: worker.role.clone(),
        worker_tier: worker.tier,
        objective: signal.objective.clone(),
        start_summary,
        resume: DispatchResumeHandle {
            context_id: signal.context_id.clone(),
            snapshot_id: signal.snapshot_id.clone(),
            namespace: signal.namespace.clone(),
            task_id: signal.task_id.clone(),
            objective: signal.objective.clone(),
            token_budget,
            candidate_limit,
        },
        context_preview: preview,
        pressure,
        attached_projected_lane,
        attached_projected_lane_source,
    }
}

fn derive_pressure_profile(
    worker_tier: DispatchWorkerTier,
    preferred_tier: Option<DispatchWorkerTier>,
    kind: DispatchSignalKind,
) -> DispatchPressureProfile {
    let (mut anxiety, mut fear, mut confidence, mut discipline): (f64, f64, f64, f64) =
        match worker_tier {
            DispatchWorkerTier::Small => (0.78, 0.42, 0.34, 0.64),
            DispatchWorkerTier::Medium => (0.52, 0.20, 0.58, 0.60),
            DispatchWorkerTier::Large => (0.26, 0.08, 0.88, 0.48),
            DispatchWorkerTier::Script => (0.08, 0.01, 0.98, 0.99),
        };
    if Some(worker_tier) == preferred_tier {
        confidence += 0.05;
        discipline += 0.03;
    } else if preferred_tier.is_some() {
        anxiety += 0.06;
        fear += 0.04;
    }
    let mut habit: f64 = match worker_tier {
        DispatchWorkerTier::Small => 0.16,
        DispatchWorkerTier::Medium => 0.34,
        DispatchWorkerTier::Large => 0.48,
        DispatchWorkerTier::Script => 0.99,
    };
    match kind {
        DispatchSignalKind::TaskComplete => {
            discipline += 0.04;
            habit += 0.05;
        }
        DispatchSignalKind::HandoffReady => {
            anxiety += 0.03;
        }
    }
    DispatchPressureProfile {
        anxiety: anxiety.clamp(0.0, 1.0),
        fear: fear.clamp(0.0, 1.0),
        confidence: confidence.clamp(0.0, 1.0),
        discipline: discipline.clamp(0.0, 1.0),
        habit: habit.clamp(0.0, 1.0),
    }
}

fn apply_affect_override(
    mut profile: DispatchPressureProfile,
    override_value: Option<&serde_json::Value>,
) -> DispatchPressureProfile {
    let Some(override_value) = override_value else {
        return profile;
    };
    let Ok(override_profile) =
        serde_json::from_value::<DispatchAffectOverride>(override_value.clone())
    else {
        return profile;
    };
    if let Some(value) = override_profile.anxiety {
        profile.anxiety = value.clamp(0.0, 1.0);
    }
    if let Some(value) = override_profile.fear {
        profile.fear = value.clamp(0.0, 1.0);
    }
    if let Some(value) = override_profile.confidence {
        profile.confidence = value.clamp(0.0, 1.0);
    }
    if let Some(value) = override_profile.discipline {
        profile.discipline = value.clamp(0.0, 1.0);
    }
    if let Some(value) = override_profile.habit {
        profile.habit = value.clamp(0.0, 1.0);
    }
    profile
}

fn select_best_worker(
    signal: &DispatchSignalRecord,
    workers: &[DispatchWorkerRecord],
) -> Option<DispatchWorkerRecord> {
    workers
        .iter()
        .cloned()
        .max_by(|left, right| worker_score(signal, left).total_cmp(&worker_score(signal, right)))
}

fn worker_score(signal: &DispatchSignalRecord, worker: &DispatchWorkerRecord) -> f64 {
    let mut score = 0.0;
    if signal
        .target_role
        .as_deref()
        .map(|role| role == worker.role)
        .unwrap_or(true)
    {
        score += 50.0;
    }
    if signal.preferred_tier == Some(worker.tier) {
        score += 20.0;
    }
    if worker.tier == DispatchWorkerTier::Script
        && signal.preferred_tier != Some(DispatchWorkerTier::Script)
    {
        score -= 25.0;
    }
    score += (worker
        .max_parallelism
        .saturating_sub(worker.active_assignment_count) as f64)
        * 5.0;
    score -= worker.active_assignment_count as f64 * 3.0;
    score += match worker.tier {
        DispatchWorkerTier::Small => 4.0,
        DispatchWorkerTier::Medium => 6.0,
        DispatchWorkerTier::Large => 8.0,
        DispatchWorkerTier::Script => 2.0,
    };
    score
}

fn worker_can_take_signal(worker: &DispatchWorkerRecord, signal: &DispatchSignalRecord) -> bool {
    if worker.status.eq_ignore_ascii_case("offline") {
        return false;
    }
    if let Some(target_role) = signal.target_role.as_deref() {
        if worker.role != target_role {
            return false;
        }
    }
    if worker.tier == DispatchWorkerTier::Script
        && signal.preferred_tier != Some(DispatchWorkerTier::Script)
    {
        return false;
    }
    worker.active_assignment_count < worker.max_parallelism
}

fn read_worker_record(row: &Row) -> Result<DispatchWorkerRecord> {
    let capabilities: serde_json::Value = row.get("capabilities_json");
    Ok(DispatchWorkerRecord {
        worker_id: row.get("worker_id"),
        display_name: row.get("display_name"),
        role: row.get("role"),
        tier: DispatchWorkerTier::from_db(row.get::<_, String>("tier").as_str())?,
        agent_type: row.get("agent_type"),
        model: row.get("model"),
        capabilities: serde_json::from_value(capabilities).unwrap_or_default(),
        max_parallelism: row.get::<_, i32>("max_parallelism") as usize,
        status: row.get("status"),
        focus: row.get("focus"),
        namespace: row.get("namespace"),
        task_id: row.get("task_id"),
        metadata: row.get("metadata_json"),
        last_seen_at: row.get("last_seen_at"),
        active_assignment_count: row.get::<_, i64>("active_assignment_count") as usize,
    })
}

fn read_signal_record(row: &Row) -> Result<DispatchSignalRecord> {
    let preferred_tier = row
        .get::<_, Option<String>>("preferred_tier")
        .map(|value| DispatchWorkerTier::from_db(value.as_str()))
        .transpose()?;
    Ok(DispatchSignalRecord {
        id: row.get("id"),
        kind: DispatchSignalKind::from_db(row.get::<_, String>("signal_kind").as_str())?,
        status: DispatchStatus::from_db(row.get::<_, String>("status").as_str())?,
        machine_id: row.get("machine_id"),
        namespace: row.get("namespace"),
        task_id: row.get("task_id"),
        context_id: row.get("context_id"),
        snapshot_id: row.get("snapshot_id"),
        from_agent_id: row.get("from_agent_id"),
        objective: row.get("objective"),
        target_role: row.get("target_role"),
        preferred_tier,
        reason: row.get("reason"),
        extra: row.get("payload_json"),
        assigned_worker_id: row.get("assigned_worker_id"),
        assignment_id: row.get("assignment_id"),
        created_at: row.get("created_at"),
        queued_at: row.get("queued_at"),
        assigned_at: row.get("assigned_at"),
        completed_at: row.get("completed_at"),
    })
}

fn read_assignment_record(row: &Row) -> Result<DispatchAssignmentRecord> {
    let envelope: serde_json::Value = row.get("envelope_json");
    let pressure: serde_json::Value = row.get("pressure_json");
    Ok(DispatchAssignmentRecord {
        id: row.get("id"),
        signal_id: row.get("signal_id"),
        worker_id: row.get("worker_id"),
        machine_id: row.get("machine_id"),
        namespace: row.get("namespace"),
        task_id: row.get("task_id"),
        context_id: row.get("context_id"),
        snapshot_id: row.get("snapshot_id"),
        objective: row.get("objective"),
        worker_role: row.get("worker_role"),
        worker_tier: DispatchWorkerTier::from_db(row.get::<_, String>("worker_tier").as_str())?,
        status: DispatchStatus::from_db(row.get::<_, String>("status").as_str())?,
        pressure: serde_json::from_value(pressure)?,
        envelope: serde_json::from_value(envelope)?,
        created_at: row.get("created_at"),
        completed_at: row.get("completed_at"),
    })
}

fn worker_visible_in_namespace(
    worker_namespace: Option<&str>,
    namespace_filter: Option<&str>,
) -> bool {
    match namespace_filter {
        Some(namespace) => worker_namespace.is_none_or(|value| value == namespace),
        None => true,
    }
}

fn dispatch_worker_presence(
    worker: DispatchWorkerRecord,
    namespace_filter: Option<&str>,
) -> DispatchWorkerPresence {
    let attached_projected_lane =
        attached_projected_lane_from_metadata(&worker.metadata, worker.task_id.as_deref());
    let attached_projected_lane_source =
        attached_projected_lane_source_from_metadata(&worker.metadata);
    let projected_lane = dispatch_worker_projected_lane(&worker.worker_id, worker.task_id.clone());
    DispatchWorkerPresence {
        worker_id: worker.worker_id,
        display_name: worker.display_name,
        role: worker.role,
        tier: worker.tier,
        agent_type: worker.agent_type,
        model: worker.model,
        status: worker.status,
        focus: worker.focus,
        namespace: worker
            .namespace
            .or_else(|| namespace_filter.map(ToString::to_string)),
        task_id: worker.task_id,
        active_assignment_count: worker.active_assignment_count,
        projected_lane,
        attached_projected_lane,
        attached_projected_lane_source,
        last_seen_at: worker.last_seen_at,
    }
}

fn dispatch_assignment_presence(
    assignment: DispatchAssignmentRecord,
) -> DispatchAssignmentPresence {
    DispatchAssignmentPresence {
        assignment_id: assignment.id,
        signal_id: assignment.signal_id,
        worker_id: assignment.worker_id.clone(),
        worker_role: assignment.worker_role,
        worker_tier: assignment.worker_tier,
        namespace: assignment.namespace,
        task_id: assignment.task_id.clone(),
        context_id: assignment.context_id,
        objective: assignment.objective,
        status: assignment.status,
        pressure: assignment.pressure,
        projected_lane: dispatch_worker_projected_lane(
            &assignment.worker_id,
            Some(assignment.task_id),
        ),
        attached_projected_lane: assignment.envelope.attached_projected_lane,
        attached_projected_lane_source: assignment.envelope.attached_projected_lane_source,
        created_at: assignment.created_at,
    }
}

fn attached_projected_lane_from_metadata(
    metadata: &serde_json::Value,
    task_id_fallback: Option<&str>,
) -> Option<CoordinationProjectedLane> {
    let value = metadata.get("attached_lane")?.clone();
    let mut lane: CoordinationProjectedLane = serde_json::from_value(value).ok()?;
    if lane.projection_id.trim().is_empty()
        || lane.projection_kind.trim().is_empty()
        || lane.label.trim().is_empty()
    {
        return None;
    }
    if lane
        .task_id
        .as_deref()
        .is_none_or(|task_id| task_id.trim().is_empty())
    {
        lane.task_id = task_id_fallback
            .filter(|task_id| !task_id.trim().is_empty())
            .map(ToString::to_string);
    }
    Some(lane)
}

fn attached_projected_lane_source_from_metadata(
    metadata: &serde_json::Value,
) -> Option<DispatchAttachedLaneSource> {
    serde_json::from_value(metadata.get("attached_lane_source")?.clone()).ok()
}

fn dispatch_worker_projected_lane(
    worker_id: &str,
    task_id: Option<String>,
) -> CoordinationProjectedLane {
    CoordinationProjectedLane {
        projection_id: format!("dispatch:worker:{worker_id}"),
        projection_kind: "dispatch_worker".to_string(),
        label: format!("dispatch {worker_id}"),
        resource: Some(format!("dispatch/worker/{worker_id}")),
        repo_root: None,
        branch: None,
        task_id,
    }
}

fn compact_line(text: &str, max_chars: usize) -> String {
    let first = text
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("");
    let compact = first.trim();
    if compact.chars().count() <= max_chars {
        compact.to_string()
    } else {
        let mut value = compact
            .chars()
            .take(max_chars.saturating_sub(1))
            .collect::<String>();
        value.push('…');
        value
    }
}

fn compact_metric_label(value: &str, max_chars: usize) -> String {
    if value.is_empty() {
        return String::new();
    }
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut shortened = value
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    shortened.push('…');
    shortened
}

#[derive(Debug, Clone)]
struct DispatchProjectionMetricState {
    projection_id: String,
    projection_kind: String,
    label: String,
    resource: Option<String>,
    repo_root: Option<String>,
    branch: Option<String>,
    task_id: Option<String>,
    namespace: String,
    worker_count: usize,
    assignment_count: usize,
    assignment_anxiety_max: f64,
    assignment_explicit_cli_count: usize,
    assignment_live_badge_opt_in_count: usize,
}

impl DispatchProjectionMetricState {
    fn new(
        lane: &CoordinationProjectedLane,
        namespace: Option<&str>,
        task_id_fallback: Option<&str>,
    ) -> Self {
        Self {
            projection_id: lane.projection_id.clone(),
            projection_kind: lane.projection_kind.clone(),
            label: lane.label.clone(),
            resource: lane.resource.clone(),
            repo_root: lane.repo_root.clone(),
            branch: lane.branch.clone(),
            task_id: lane
                .task_id
                .clone()
                .or_else(|| task_id_fallback.map(ToString::to_string)),
            namespace: namespace.unwrap_or("").to_string(),
            worker_count: 0,
            assignment_count: 0,
            assignment_anxiety_max: 0.0,
            assignment_explicit_cli_count: 0,
            assignment_live_badge_opt_in_count: 0,
        }
    }
}

fn record_dispatch_projection_assignment(
    projections: &mut BTreeMap<String, DispatchProjectionMetricState>,
    lane: &CoordinationProjectedLane,
    namespace: Option<&str>,
    task_id_fallback: Option<&str>,
    anxiety: f64,
    attached_lane_source: Option<DispatchAttachedLaneSource>,
) {
    let entry = projections
        .entry(lane.projection_id.clone())
        .or_insert_with(|| DispatchProjectionMetricState::new(lane, namespace, task_id_fallback));
    entry.assignment_count += 1;
    entry.assignment_anxiety_max = entry.assignment_anxiety_max.max(anxiety);
    match attached_lane_source {
        Some(DispatchAttachedLaneSource::ExplicitCli) => {
            entry.assignment_explicit_cli_count += 1;
        }
        Some(DispatchAttachedLaneSource::LiveBadgeOptIn) => {
            entry.assignment_live_badge_opt_in_count += 1;
        }
        None => {}
    }
}

fn projection_metric_labels(projection: &DispatchProjectionMetricState) -> String {
    format!(
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
    )
}

fn dispatch_down_metrics(error: &str) -> String {
    format!(
        "# HELP ice_dispatch_up Whether the PostgreSQL dispatch spine is reachable.\n# TYPE ice_dispatch_up gauge\nice_dispatch_up 0\n# HELP ice_dispatch_status PostgreSQL dispatch spine status.\n# TYPE ice_dispatch_status gauge\nice_dispatch_status{{state=\"{}\"}} 0\n",
        prometheus_label_value(&compact_metric_label(error, 120))
    )
}

fn prometheus_label_value(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}

fn is_valid_pg_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    match chars.next() {
        Some(first) if first == '_' || first.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn default_dispatch_notify_channel() -> String {
    DEFAULT_DISPATCH_NOTIFY_CHANNEL.to_string()
}

fn default_worker_stale_secs() -> u64 {
    DEFAULT_WORKER_STALE_SECS
}

fn default_assignment_token_budget() -> usize {
    DEFAULT_ASSIGNMENT_TOKEN_BUDGET
}

fn default_assignment_candidate_limit() -> usize {
    DEFAULT_ASSIGNMENT_CANDIDATE_LIMIT
}

fn default_worker_status() -> String {
    "idle".to_string()
}

fn default_max_parallelism() -> usize {
    1
}

fn default_snapshot_resolution() -> SnapshotResolution {
    SnapshotResolution::Medium
}

impl DispatchOrganismSnapshot {
    fn unconfigured() -> Self {
        Self {
            configured: false,
            reachable: false,
            notify_channel: None,
            worker_active_window_secs: None,
            workers_active: 0,
            signals_queued: 0,
            assignments_active: 0,
            workers: Vec::new(),
            assignments: Vec::new(),
            error: None,
        }
    }

    fn config_error(error: anyhow::Error) -> Self {
        Self {
            configured: false,
            reachable: false,
            notify_channel: None,
            worker_active_window_secs: None,
            workers_active: 0,
            signals_queued: 0,
            assignments_active: 0,
            workers: Vec::new(),
            assignments: Vec::new(),
            error: Some(format!("{error:#}")),
        }
    }

    fn unreachable(config: DispatchConfig, error: anyhow::Error) -> Self {
        Self {
            configured: true,
            reachable: false,
            notify_channel: Some(config.notify_channel),
            worker_active_window_secs: Some(config.worker_stale_secs),
            workers_active: 0,
            signals_queued: 0,
            assignments_active: 0,
            workers: Vec::new(),
            assignments: Vec::new(),
            error: Some(format!("{error:#}")),
        }
    }
}

impl DispatchWorkerTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
            Self::Script => "script",
        }
    }

    fn from_db(value: &str) -> Result<Self> {
        match value {
            "small" => Ok(Self::Small),
            "medium" => Ok(Self::Medium),
            "large" => Ok(Self::Large),
            "script" => Ok(Self::Script),
            other => anyhow::bail!("unknown dispatch worker tier '{other}'"),
        }
    }
}

impl DispatchSignalKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TaskComplete => "task_complete",
            Self::HandoffReady => "handoff_ready",
        }
    }

    fn from_db(value: &str) -> Result<Self> {
        match value {
            "task_complete" => Ok(Self::TaskComplete),
            "handoff_ready" => Ok(Self::HandoffReady),
            other => anyhow::bail!("unknown dispatch signal kind '{other}'"),
        }
    }
}

impl DispatchStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Assigned => "assigned",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }

    fn from_db(value: &str) -> Result<Self> {
        match value {
            "queued" => Ok(Self::Queued),
            "assigned" => Ok(Self::Assigned),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            other => anyhow::bail!("unknown dispatch status '{other}'"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        DispatchAffectOverride, DispatchAttachedLaneSource, DispatchConfig,
        DispatchOrganismSnapshot, DispatchPressureProfile, DispatchProjectionMetricState,
        DispatchSignalKind, DispatchSignalRecord, DispatchStatus, DispatchWorkerRecord,
        DispatchWorkerTier, apply_affect_override, attached_projected_lane_from_metadata,
        attached_projected_lane_source_from_metadata, compact_line, compact_metric_label,
        derive_pressure_profile, dispatch_down_metrics, dispatch_worker_projected_lane,
        is_valid_pg_identifier, projection_metric_labels, prometheus_label_value,
        record_dispatch_projection_assignment, select_best_worker, worker_can_take_signal,
        worker_score, worker_visible_in_namespace,
    };
    use crate::continuity::CoordinationProjectedLane;
    use chrono::Utc;

    fn make_worker(
        id: &str,
        role: &str,
        tier: DispatchWorkerTier,
        active: usize,
        max_par: usize,
    ) -> DispatchWorkerRecord {
        DispatchWorkerRecord {
            worker_id: id.into(),
            display_name: id.into(),
            role: role.into(),
            tier,
            agent_type: "test".into(),
            model: "test-model".into(),
            capabilities: Vec::new(),
            max_parallelism: max_par,
            status: "idle".into(),
            focus: String::new(),
            namespace: None,
            task_id: None,
            metadata: serde_json::json!({}),
            last_seen_at: Utc::now(),
            active_assignment_count: active,
        }
    }

    fn make_signal(
        target_role: Option<&str>,
        preferred_tier: Option<DispatchWorkerTier>,
    ) -> DispatchSignalRecord {
        DispatchSignalRecord {
            id: "signal-1".into(),
            kind: DispatchSignalKind::TaskComplete,
            status: DispatchStatus::Queued,
            machine_id: "machine".into(),
            namespace: "ns".into(),
            task_id: "task".into(),
            context_id: "ctx".into(),
            snapshot_id: None,
            from_agent_id: "agent".into(),
            objective: "do stuff".into(),
            target_role: target_role.map(Into::into),
            preferred_tier,
            reason: None,
            extra: serde_json::json!({}),
            assigned_worker_id: None,
            assignment_id: None,
            created_at: Utc::now(),
            queued_at: Utc::now(),
            assigned_at: None,
            completed_at: None,
        }
    }

    #[test]
    fn pressure_profile_adds_more_anxiety_to_small_tiers() {
        let small = derive_pressure_profile(
            DispatchWorkerTier::Small,
            Some(DispatchWorkerTier::Small),
            DispatchSignalKind::TaskComplete,
        );
        let large = derive_pressure_profile(
            DispatchWorkerTier::Large,
            Some(DispatchWorkerTier::Large),
            DispatchSignalKind::TaskComplete,
        );
        assert!(small.anxiety > large.anxiety);
        assert!(small.fear > large.fear);
        assert!(small.confidence < large.confidence);
    }

    #[test]
    fn worker_selection_prefers_role_match_and_free_capacity() {
        let signal = DispatchSignalRecord {
            id: "signal-1".into(),
            kind: DispatchSignalKind::TaskComplete,
            status: DispatchStatus::Queued,
            machine_id: "machine".into(),
            namespace: "machine:test".into(),
            task_id: "machine-organism".into(),
            context_id: "ctx".into(),
            snapshot_id: None,
            from_agent_id: "planner".into(),
            objective: "continue fixing dispatch".into(),
            target_role: Some("coder".into()),
            preferred_tier: Some(DispatchWorkerTier::Small),
            reason: None,
            extra: serde_json::json!({}),
            assigned_worker_id: None,
            assignment_id: None,
            created_at: Utc::now(),
            queued_at: Utc::now(),
            assigned_at: None,
            completed_at: None,
        };
        let blocked = DispatchWorkerRecord {
            worker_id: "large-debugger".into(),
            display_name: "large-debugger".into(),
            role: "debugger".into(),
            tier: DispatchWorkerTier::Large,
            agent_type: "ollama".into(),
            model: "llama".into(),
            capabilities: Vec::new(),
            max_parallelism: 1,
            status: "idle".into(),
            focus: String::new(),
            namespace: None,
            task_id: None,
            metadata: serde_json::json!({}),
            last_seen_at: Utc::now(),
            active_assignment_count: 0,
        };
        let chosen = DispatchWorkerRecord {
            worker_id: "small-coder".into(),
            display_name: "small-coder".into(),
            role: "coder".into(),
            tier: DispatchWorkerTier::Small,
            agent_type: "ollama".into(),
            model: "qwen".into(),
            capabilities: Vec::new(),
            max_parallelism: 1,
            status: "idle".into(),
            focus: String::new(),
            namespace: None,
            task_id: None,
            metadata: serde_json::json!({}),
            last_seen_at: Utc::now(),
            active_assignment_count: 0,
        };
        let selected = select_best_worker(&signal, &[blocked, chosen.clone()]).unwrap();
        assert_eq!(selected.worker_id, chosen.worker_id);
    }

    #[test]
    fn pg_identifier_validation_is_strict() {
        assert!(is_valid_pg_identifier("ice_dispatch_signal"));
        assert!(!is_valid_pg_identifier("ice-dispatch-signal"));
        assert!(!is_valid_pg_identifier("9dispatch"));
    }

    // ---- pressure profile coverage ----

    #[test]
    fn pressure_profile_all_tiers_produce_valid_ranges() {
        for tier in [
            DispatchWorkerTier::Small,
            DispatchWorkerTier::Medium,
            DispatchWorkerTier::Large,
            DispatchWorkerTier::Script,
        ] {
            for kind in [
                DispatchSignalKind::TaskComplete,
                DispatchSignalKind::HandoffReady,
            ] {
                let profile = derive_pressure_profile(tier, Some(tier), kind);
                assert!(
                    (0.0..=1.0).contains(&profile.anxiety),
                    "anxiety out of range for {tier:?}/{kind:?}"
                );
                assert!(
                    (0.0..=1.0).contains(&profile.fear),
                    "fear out of range for {tier:?}/{kind:?}"
                );
                assert!(
                    (0.0..=1.0).contains(&profile.confidence),
                    "confidence out of range for {tier:?}/{kind:?}"
                );
                assert!(
                    (0.0..=1.0).contains(&profile.discipline),
                    "discipline out of range for {tier:?}/{kind:?}"
                );
                assert!(
                    (0.0..=1.0).contains(&profile.habit),
                    "habit out of range for {tier:?}/{kind:?}"
                );
            }
        }
    }

    #[test]
    fn pressure_profile_preferred_tier_mismatch_increases_anxiety() {
        let matched = derive_pressure_profile(
            DispatchWorkerTier::Small,
            Some(DispatchWorkerTier::Small),
            DispatchSignalKind::TaskComplete,
        );
        let mismatched = derive_pressure_profile(
            DispatchWorkerTier::Small,
            Some(DispatchWorkerTier::Large),
            DispatchSignalKind::TaskComplete,
        );
        assert!(
            mismatched.anxiety > matched.anxiety,
            "tier mismatch should increase anxiety"
        );
        assert!(
            mismatched.fear > matched.fear,
            "tier mismatch should increase fear"
        );
    }

    #[test]
    fn pressure_profile_no_preferred_tier() {
        let profile = derive_pressure_profile(
            DispatchWorkerTier::Medium,
            None,
            DispatchSignalKind::TaskComplete,
        );
        assert!((0.0..=1.0).contains(&profile.anxiety));
    }

    #[test]
    fn pressure_profile_handoff_adds_anxiety() {
        let task = derive_pressure_profile(
            DispatchWorkerTier::Medium,
            None,
            DispatchSignalKind::TaskComplete,
        );
        let handoff = derive_pressure_profile(
            DispatchWorkerTier::Medium,
            None,
            DispatchSignalKind::HandoffReady,
        );
        assert!(
            handoff.anxiety > task.anxiety,
            "handoff should add anxiety vs task_complete"
        );
    }

    #[test]
    fn pressure_profile_script_tier_has_high_confidence_and_discipline() {
        let profile = derive_pressure_profile(
            DispatchWorkerTier::Script,
            Some(DispatchWorkerTier::Script),
            DispatchSignalKind::TaskComplete,
        );
        assert!(profile.confidence > 0.9);
        assert!(profile.discipline > 0.9);
        assert!(profile.habit > 0.9);
    }

    // ---- apply_affect_override coverage ----

    #[test]
    fn apply_affect_override_with_none_returns_original() {
        let profile = derive_pressure_profile(
            DispatchWorkerTier::Small,
            None,
            DispatchSignalKind::TaskComplete,
        );
        let result = apply_affect_override(profile.clone(), None);
        assert_eq!(result.anxiety, profile.anxiety);
        assert_eq!(result.fear, profile.fear);
    }

    #[test]
    fn apply_affect_override_applies_partial_overrides() {
        let profile = derive_pressure_profile(
            DispatchWorkerTier::Small,
            None,
            DispatchSignalKind::TaskComplete,
        );
        let override_val = serde_json::json!({"anxiety": 0.1, "confidence": 0.99});
        let result = apply_affect_override(profile.clone(), Some(&override_val));
        assert!((result.anxiety - 0.1).abs() < f64::EPSILON);
        assert!((result.confidence - 0.99).abs() < f64::EPSILON);
        assert_eq!(
            result.fear, profile.fear,
            "unspecified fields should be unchanged"
        );
    }

    #[test]
    fn apply_affect_override_clamps_out_of_range_values() {
        let profile = derive_pressure_profile(
            DispatchWorkerTier::Medium,
            None,
            DispatchSignalKind::TaskComplete,
        );
        let override_val = serde_json::json!({"anxiety": 5.0, "fear": -2.0});
        let result = apply_affect_override(profile, Some(&override_val));
        assert!((result.anxiety - 1.0).abs() < f64::EPSILON);
        assert!(result.fear.abs() < f64::EPSILON);
    }

    #[test]
    fn apply_affect_override_ignores_invalid_json() {
        let profile = derive_pressure_profile(
            DispatchWorkerTier::Medium,
            None,
            DispatchSignalKind::TaskComplete,
        );
        let bad_val = serde_json::json!("not an object");
        let result = apply_affect_override(profile.clone(), Some(&bad_val));
        assert_eq!(result.anxiety, profile.anxiety);
    }

    // ---- worker_can_take_signal coverage ----

    #[test]
    fn worker_can_take_signal_basic_idle_worker_accepts() {
        let worker = make_worker("w1", "coder", DispatchWorkerTier::Small, 0, 1);
        let signal = make_signal(None, None);
        assert!(worker_can_take_signal(&worker, &signal));
    }

    #[test]
    fn worker_can_take_signal_offline_worker_rejects() {
        let mut worker = make_worker("w1", "coder", DispatchWorkerTier::Small, 0, 1);
        worker.status = "offline".into();
        let signal = make_signal(None, None);
        assert!(!worker_can_take_signal(&worker, &signal));
    }

    #[test]
    fn worker_can_take_signal_wrong_role_rejects() {
        let worker = make_worker("w1", "debugger", DispatchWorkerTier::Small, 0, 1);
        let signal = make_signal(Some("coder"), None);
        assert!(!worker_can_take_signal(&worker, &signal));
    }

    #[test]
    fn worker_can_take_signal_at_capacity_rejects() {
        let worker = make_worker("w1", "coder", DispatchWorkerTier::Small, 1, 1);
        let signal = make_signal(None, None);
        assert!(!worker_can_take_signal(&worker, &signal));
    }

    #[test]
    fn worker_can_take_signal_script_tier_only_takes_script_signals() {
        let worker = make_worker("w1", "runner", DispatchWorkerTier::Script, 0, 1);
        let non_script = make_signal(None, Some(DispatchWorkerTier::Small));
        assert!(!worker_can_take_signal(&worker, &non_script));

        let script_signal = make_signal(None, Some(DispatchWorkerTier::Script));
        assert!(worker_can_take_signal(&worker, &script_signal));
    }

    #[test]
    fn worker_can_take_signal_parallel_capacity() {
        let worker = make_worker("w1", "coder", DispatchWorkerTier::Medium, 2, 4);
        let signal = make_signal(None, None);
        assert!(worker_can_take_signal(&worker, &signal));
    }

    // ---- worker_score coverage ----

    #[test]
    fn worker_score_role_match_adds_points() {
        let signal = make_signal(Some("coder"), None);
        let matching = make_worker("w1", "coder", DispatchWorkerTier::Medium, 0, 1);
        let non_matching = make_worker("w2", "debugger", DispatchWorkerTier::Medium, 0, 1);
        assert!(worker_score(&signal, &matching) > worker_score(&signal, &non_matching));
    }

    #[test]
    fn worker_score_preferred_tier_adds_points() {
        let signal = make_signal(None, Some(DispatchWorkerTier::Small));
        let preferred = make_worker("w1", "coder", DispatchWorkerTier::Small, 0, 1);
        let non_preferred = make_worker("w2", "coder", DispatchWorkerTier::Medium, 0, 1);
        assert!(worker_score(&signal, &preferred) > worker_score(&signal, &non_preferred));
    }

    #[test]
    fn worker_score_script_tier_penalty_when_not_preferred() {
        let signal = make_signal(None, Some(DispatchWorkerTier::Small));
        let script = make_worker("w1", "runner", DispatchWorkerTier::Script, 0, 1);
        let small = make_worker("w2", "runner", DispatchWorkerTier::Small, 0, 1);
        assert!(worker_score(&signal, &script) < worker_score(&signal, &small));
    }

    #[test]
    fn worker_score_free_capacity_increases_score() {
        let signal = make_signal(None, None);
        let free = make_worker("w1", "coder", DispatchWorkerTier::Medium, 0, 4);
        let busy = make_worker("w2", "coder", DispatchWorkerTier::Medium, 3, 4);
        assert!(worker_score(&signal, &free) > worker_score(&signal, &busy));
    }

    // ---- select_best_worker coverage ----

    #[test]
    fn select_best_worker_returns_none_for_empty_list() {
        let signal = make_signal(None, None);
        assert!(select_best_worker(&signal, &[]).is_none());
    }

    #[test]
    fn select_best_worker_single_worker() {
        let signal = make_signal(None, None);
        let worker = make_worker("w1", "coder", DispatchWorkerTier::Medium, 0, 1);
        let selected = select_best_worker(&signal, &[worker.clone()]).unwrap();
        assert_eq!(selected.worker_id, "w1");
    }

    #[test]
    fn select_best_worker_picks_matching_role_over_larger_tier() {
        let signal = make_signal(Some("coder"), Some(DispatchWorkerTier::Small));
        let large_wrong_role = make_worker("w1", "debugger", DispatchWorkerTier::Large, 0, 1);
        let small_right_role = make_worker("w2", "coder", DispatchWorkerTier::Small, 0, 1);
        let selected = select_best_worker(&signal, &[large_wrong_role, small_right_role]).unwrap();
        assert_eq!(selected.worker_id, "w2");
    }

    // ---- compact_line coverage ----

    #[test]
    fn compact_line_short_text_unchanged() {
        assert_eq!(compact_line("hello world", 20), "hello world");
    }

    #[test]
    fn compact_line_truncates_long_text() {
        let result = compact_line("this is a very long text that should be truncated", 10);
        assert!(
            result.chars().count() <= 10,
            "expected at most 10 chars, got {} ('{}')",
            result.chars().count(),
            result
        );
        assert!(result.ends_with('…'));
    }

    #[test]
    fn compact_line_uses_first_non_empty_line() {
        let result = compact_line("\n\n  actual content\nsecond line", 50);
        assert_eq!(result, "actual content");
    }

    #[test]
    fn compact_line_empty_text() {
        assert_eq!(compact_line("", 10), "");
    }

    // ---- compact_metric_label coverage ----

    #[test]
    fn compact_metric_label_empty_returns_empty() {
        assert_eq!(compact_metric_label("", 10), "");
    }

    #[test]
    fn compact_metric_label_short_returns_unchanged() {
        assert_eq!(compact_metric_label("short", 10), "short");
    }

    #[test]
    fn compact_metric_label_truncates_long_value() {
        let result = compact_metric_label("a very long label value here", 10);
        assert!(result.ends_with('…'));
        assert!(result.chars().count() <= 10);
    }

    // ---- prometheus_label_value coverage ----

    #[test]
    fn prometheus_label_value_escapes_backslash() {
        assert_eq!(prometheus_label_value(r"a\b"), r"a\\b");
    }

    #[test]
    fn prometheus_label_value_escapes_newline() {
        assert_eq!(prometheus_label_value("a\nb"), r"a\nb");
    }

    #[test]
    fn prometheus_label_value_escapes_quotes() {
        assert_eq!(prometheus_label_value(r#"a"b"#), r#"a\"b"#);
    }

    #[test]
    fn prometheus_label_value_plain_text_unchanged() {
        assert_eq!(prometheus_label_value("hello"), "hello");
    }

    // ---- pg identifier coverage ----

    #[test]
    fn pg_identifier_underscore_prefix_valid() {
        assert!(is_valid_pg_identifier("_my_channel"));
    }

    #[test]
    fn pg_identifier_empty_string_invalid() {
        assert!(!is_valid_pg_identifier(""));
    }

    #[test]
    fn pg_identifier_with_spaces_invalid() {
        assert!(!is_valid_pg_identifier("ice dispatch"));
    }

    #[test]
    fn pg_identifier_alphanumeric_with_underscore_valid() {
        assert!(is_valid_pg_identifier("a1_b2_c3"));
    }

    // ---- dispatch_down_metrics coverage ----

    #[test]
    fn dispatch_down_metrics_contains_up_gauge() {
        let text = dispatch_down_metrics("connection refused");
        assert!(text.contains("ice_dispatch_up 0"));
        assert!(text.contains("ice_dispatch_status"));
        assert!(text.contains("connection refused"));
    }

    // ---- DispatchConfig validation coverage ----

    #[test]
    fn dispatch_config_validate_empty_url_fails() {
        let config = DispatchConfig {
            database_url: "  ".to_string(),
            notify_channel: "ice_dispatch_signal".to_string(),
            worker_stale_secs: 120,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn dispatch_config_validate_bad_channel_fails() {
        let config = DispatchConfig {
            database_url: "postgres://localhost/test".to_string(),
            notify_channel: "bad-channel-name".to_string(),
            worker_stale_secs: 120,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn dispatch_config_validate_good_config_passes() {
        let config = DispatchConfig {
            database_url: "postgres://localhost/test".to_string(),
            notify_channel: "ice_dispatch_signal".to_string(),
            worker_stale_secs: 120,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn dispatch_config_save_and_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let config = DispatchConfig {
            database_url: "postgres://localhost/test".to_string(),
            notify_channel: "ice_dispatch_signal".to_string(),
            worker_stale_secs: 60,
        };
        let path = config.save(dir.path()).unwrap();
        assert!(path.exists());

        let loaded = DispatchConfig::load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.database_url, config.database_url);
        assert_eq!(loaded.notify_channel, config.notify_channel);
        assert_eq!(loaded.worker_stale_secs, config.worker_stale_secs);
    }

    #[test]
    fn dispatch_config_load_returns_none_when_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let loaded = DispatchConfig::load(dir.path()).unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn dispatch_config_serde_round_trip() {
        let config = DispatchConfig {
            database_url: "postgres://localhost/test".to_string(),
            notify_channel: "ice_dispatch_signal".to_string(),
            worker_stale_secs: 90,
        };
        let json = serde_json::to_string(&config).unwrap();
        let parsed: DispatchConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.database_url, config.database_url);
        assert_eq!(parsed.worker_stale_secs, 90);
    }

    // ---- worker_visible_in_namespace coverage ----

    #[test]
    fn worker_visible_no_filter_always_visible() {
        assert!(worker_visible_in_namespace(Some("ns"), None));
        assert!(worker_visible_in_namespace(None, None));
    }

    #[test]
    fn worker_visible_matching_namespace() {
        assert!(worker_visible_in_namespace(Some("ns"), Some("ns")));
    }

    #[test]
    fn worker_visible_no_namespace_matches_any_filter() {
        assert!(worker_visible_in_namespace(None, Some("ns")));
    }

    #[test]
    fn worker_visible_wrong_namespace_hidden() {
        assert!(!worker_visible_in_namespace(Some("other"), Some("ns")));
    }

    // ---- DispatchOrganismSnapshot factory methods ----

    #[test]
    fn organism_snapshot_unconfigured() {
        let snap = DispatchOrganismSnapshot::unconfigured();
        assert!(!snap.configured);
        assert!(!snap.reachable);
        assert!(snap.error.is_none());
    }

    #[test]
    fn organism_snapshot_config_error() {
        let snap = DispatchOrganismSnapshot::config_error(anyhow::anyhow!("test error"));
        assert!(!snap.configured);
        assert!(!snap.reachable);
        assert!(snap.error.as_ref().unwrap().contains("test error"));
    }

    #[test]
    fn organism_snapshot_unreachable() {
        let config = DispatchConfig {
            database_url: "postgres://localhost/test".to_string(),
            notify_channel: "ice_dispatch_signal".to_string(),
            worker_stale_secs: 120,
        };
        let snap = DispatchOrganismSnapshot::unreachable(config, anyhow::anyhow!("conn refused"));
        assert!(snap.configured);
        assert!(!snap.reachable);
        assert!(snap.error.as_ref().unwrap().contains("conn refused"));
        assert_eq!(snap.notify_channel.as_deref(), Some("ice_dispatch_signal"));
    }

    // ---- tier/kind/status round-trips ----

    #[test]
    fn worker_tier_as_str_from_db_round_trip() {
        for tier in [
            DispatchWorkerTier::Small,
            DispatchWorkerTier::Medium,
            DispatchWorkerTier::Large,
            DispatchWorkerTier::Script,
        ] {
            let s = tier.as_str();
            let back = DispatchWorkerTier::from_db(s).unwrap();
            assert_eq!(back, tier);
        }
    }

    #[test]
    fn worker_tier_from_db_unknown_fails() {
        assert!(DispatchWorkerTier::from_db("unknown").is_err());
    }

    #[test]
    fn signal_kind_as_str_from_db_round_trip() {
        for kind in [
            DispatchSignalKind::TaskComplete,
            DispatchSignalKind::HandoffReady,
        ] {
            let s = kind.as_str();
            let back = DispatchSignalKind::from_db(s).unwrap();
            assert_eq!(back, kind);
        }
    }

    #[test]
    fn signal_kind_from_db_unknown_fails() {
        assert!(DispatchSignalKind::from_db("unknown").is_err());
    }

    #[test]
    fn dispatch_status_as_str_from_db_round_trip() {
        for status in [
            DispatchStatus::Queued,
            DispatchStatus::Assigned,
            DispatchStatus::Completed,
            DispatchStatus::Failed,
        ] {
            let s = status.as_str();
            let back = DispatchStatus::from_db(s).unwrap();
            assert_eq!(back, status);
        }
    }

    #[test]
    fn dispatch_status_from_db_unknown_fails() {
        assert!(DispatchStatus::from_db("unknown").is_err());
    }

    // ---- DispatchAttachedLaneSource ----

    #[test]
    fn attached_lane_source_as_str() {
        assert_eq!(
            DispatchAttachedLaneSource::ExplicitCli.as_str(),
            "explicit_cli"
        );
        assert_eq!(
            DispatchAttachedLaneSource::LiveBadgeOptIn.as_str(),
            "live_badge_opt_in"
        );
    }

    // ---- attached_projected_lane_from_metadata ----

    #[test]
    fn attached_lane_from_metadata_returns_none_without_key() {
        let metadata = serde_json::json!({});
        assert!(attached_projected_lane_from_metadata(&metadata, None).is_none());
    }

    #[test]
    fn attached_lane_from_metadata_returns_none_for_empty_fields() {
        let metadata = serde_json::json!({
            "attached_lane": {
                "projection_id": "",
                "projection_kind": "",
                "label": "",
            }
        });
        assert!(attached_projected_lane_from_metadata(&metadata, None).is_none());
    }

    #[test]
    fn attached_lane_from_metadata_fills_task_id_from_fallback() {
        let metadata = serde_json::json!({
            "attached_lane": {
                "projection_id": "proj:1",
                "projection_kind": "branch",
                "label": "my branch",
            }
        });
        let lane = attached_projected_lane_from_metadata(&metadata, Some("task-42")).unwrap();
        assert_eq!(lane.projection_id, "proj:1");
        assert_eq!(lane.task_id.as_deref(), Some("task-42"));
    }

    #[test]
    fn attached_lane_from_metadata_preserves_existing_task_id() {
        let metadata = serde_json::json!({
            "attached_lane": {
                "projection_id": "proj:1",
                "projection_kind": "branch",
                "label": "my branch",
                "task_id": "original-task",
            }
        });
        let lane = attached_projected_lane_from_metadata(&metadata, Some("fallback")).unwrap();
        assert_eq!(lane.task_id.as_deref(), Some("original-task"));
    }

    #[test]
    fn attached_lane_source_from_metadata_returns_none_without_key() {
        let metadata = serde_json::json!({});
        assert!(attached_projected_lane_source_from_metadata(&metadata).is_none());
    }

    #[test]
    fn attached_lane_source_from_metadata_returns_explicit_cli() {
        let metadata = serde_json::json!({"attached_lane_source": "explicit_cli"});
        let source = attached_projected_lane_source_from_metadata(&metadata).unwrap();
        assert_eq!(source, DispatchAttachedLaneSource::ExplicitCli);
    }

    // ---- dispatch_worker_projected_lane ----

    #[test]
    fn dispatch_worker_projected_lane_format() {
        let lane = dispatch_worker_projected_lane("worker-1", Some("task-1".to_string()));
        assert_eq!(lane.projection_id, "dispatch:worker:worker-1");
        assert_eq!(lane.projection_kind, "dispatch_worker");
        assert_eq!(lane.label, "dispatch worker-1");
        assert_eq!(lane.task_id.as_deref(), Some("task-1"));
    }

    // ---- record_dispatch_projection_assignment ----

    #[test]
    fn record_dispatch_projection_assignment_accumulates() {
        let mut projections = BTreeMap::new();
        let lane = CoordinationProjectedLane {
            projection_id: "proj:1".to_string(),
            projection_kind: "test".to_string(),
            label: "test lane".to_string(),
            resource: None,
            repo_root: None,
            branch: None,
            task_id: None,
        };
        record_dispatch_projection_assignment(
            &mut projections,
            &lane,
            Some("ns"),
            Some("task"),
            0.5,
            None,
        );
        record_dispatch_projection_assignment(
            &mut projections,
            &lane,
            Some("ns"),
            Some("task"),
            0.8,
            Some(DispatchAttachedLaneSource::ExplicitCli),
        );
        record_dispatch_projection_assignment(
            &mut projections,
            &lane,
            Some("ns"),
            Some("task"),
            0.3,
            Some(DispatchAttachedLaneSource::LiveBadgeOptIn),
        );

        let state = projections.get("proj:1").unwrap();
        assert_eq!(state.assignment_count, 3);
        assert!((state.assignment_anxiety_max - 0.8).abs() < f64::EPSILON);
        assert_eq!(state.assignment_explicit_cli_count, 1);
        assert_eq!(state.assignment_live_badge_opt_in_count, 1);
    }

    // ---- projection_metric_labels ----

    #[test]
    fn projection_metric_labels_contains_all_fields() {
        let state = DispatchProjectionMetricState {
            projection_id: "proj:1".to_string(),
            projection_kind: "test".to_string(),
            label: "test lane".to_string(),
            resource: Some("resource/path".to_string()),
            repo_root: None,
            branch: Some("main".to_string()),
            task_id: Some("task-1".to_string()),
            namespace: "ns".to_string(),
            worker_count: 2,
            assignment_count: 1,
            assignment_anxiety_max: 0.5,
            assignment_explicit_cli_count: 0,
            assignment_live_badge_opt_in_count: 0,
        };
        let labels = projection_metric_labels(&state);
        assert!(labels.contains("namespace=\"ns\""));
        assert!(labels.contains("projection_id=\"proj:1\""));
        assert!(labels.contains("task_id=\"task-1\""));
        assert!(labels.contains("branch=\"main\""));
    }

    // ---- organism_snapshot via public function ----

    #[test]
    fn organism_snapshot_without_config_returns_unconfigured() {
        let dir = tempfile::tempdir().unwrap();
        let snap = super::organism_snapshot(dir.path(), None);
        assert!(!snap.configured);
        assert!(!snap.reachable);
    }

    // ---- default functions ----

    #[test]
    fn default_values_are_sane() {
        assert_eq!(
            super::default_dispatch_notify_channel(),
            "ice_dispatch_signal"
        );
        assert_eq!(super::default_worker_stale_secs(), 120);
        assert_eq!(super::default_assignment_token_budget(), 224);
        assert_eq!(super::default_assignment_candidate_limit(), 24);
        assert_eq!(super::default_worker_status(), "idle");
        assert_eq!(super::default_max_parallelism(), 1);
        assert_eq!(
            super::default_snapshot_resolution(),
            super::super::model::SnapshotResolution::Medium
        );
        assert_eq!(super::default_worker_active_window_secs(), 120);
    }

    // ---- append_metrics without config ----

    #[test]
    fn append_metrics_without_config_outputs_down_gauge() {
        let dir = tempfile::tempdir().unwrap();
        let mut text = String::new();
        super::append_metrics(&mut text, dir.path());
        assert!(text.contains("ice_dispatch_up 0"));
    }

    // ---- DispatchPressureProfile serde ----

    #[test]
    fn pressure_profile_serde_round_trip() {
        let profile = DispatchPressureProfile {
            anxiety: 0.5,
            fear: 0.3,
            confidence: 0.7,
            discipline: 0.6,
            habit: 0.4,
        };
        let json = serde_json::to_string(&profile).unwrap();
        let parsed: DispatchPressureProfile = serde_json::from_str(&json).unwrap();
        assert!((parsed.anxiety - 0.5).abs() < f64::EPSILON);
        assert!((parsed.habit - 0.4).abs() < f64::EPSILON);
    }

    // ---- DispatchAffectOverride serde ----

    #[test]
    fn affect_override_default_is_all_none() {
        let override_val = DispatchAffectOverride::default();
        assert!(override_val.anxiety.is_none());
        assert!(override_val.confidence.is_none());
        assert!(override_val.fear.is_none());
        assert!(override_val.discipline.is_none());
        assert!(override_val.habit.is_none());
    }

    // ---- dispatch_worker_presence transformation ----

    #[test]
    fn dispatch_worker_presence_copies_core_fields() {
        let worker = make_worker("w1", "coder", DispatchWorkerTier::Medium, 2, 4);
        let presence = super::dispatch_worker_presence(worker.clone(), None);
        assert_eq!(presence.worker_id, "w1");
        assert_eq!(presence.role, "coder");
        assert_eq!(presence.tier, DispatchWorkerTier::Medium);
        assert_eq!(presence.active_assignment_count, 2);
        assert_eq!(presence.projected_lane.projection_id, "dispatch:worker:w1");
        assert_eq!(presence.projected_lane.projection_kind, "dispatch_worker");
    }

    #[test]
    fn dispatch_worker_presence_fills_namespace_from_filter_when_none() {
        let worker = make_worker("w1", "coder", DispatchWorkerTier::Small, 0, 1);
        let presence = super::dispatch_worker_presence(worker, Some("ns-filter"));
        assert_eq!(presence.namespace.as_deref(), Some("ns-filter"));
    }

    #[test]
    fn dispatch_worker_presence_preserves_worker_namespace() {
        let mut worker = make_worker("w1", "coder", DispatchWorkerTier::Small, 0, 1);
        worker.namespace = Some("worker-ns".to_string());
        let presence = super::dispatch_worker_presence(worker, Some("ns-filter"));
        assert_eq!(presence.namespace.as_deref(), Some("worker-ns"));
    }

    #[test]
    fn dispatch_worker_presence_with_attached_lane_in_metadata() {
        let mut worker = make_worker("w1", "coder", DispatchWorkerTier::Small, 0, 1);
        worker.metadata = serde_json::json!({
            "attached_lane": {
                "projection_id": "branch:feat/x",
                "projection_kind": "branch",
                "label": "feat/x",
            },
            "attached_lane_source": "explicit_cli",
        });
        worker.task_id = Some("task-99".to_string());
        let presence = super::dispatch_worker_presence(worker, None);
        let lane = presence.attached_projected_lane.unwrap();
        assert_eq!(lane.projection_id, "branch:feat/x");
        assert_eq!(lane.task_id.as_deref(), Some("task-99"));
        assert_eq!(
            presence.attached_projected_lane_source,
            Some(DispatchAttachedLaneSource::ExplicitCli)
        );
    }

    #[test]
    fn dispatch_worker_presence_no_attached_lane() {
        let worker = make_worker("w1", "coder", DispatchWorkerTier::Small, 0, 1);
        let presence = super::dispatch_worker_presence(worker, None);
        assert!(presence.attached_projected_lane.is_none());
        assert!(presence.attached_projected_lane_source.is_none());
    }

    // ---- dispatch_assignment_presence transformation ----

    fn make_assignment_record() -> super::DispatchAssignmentRecord {
        super::DispatchAssignmentRecord {
            id: "assign-1".to_string(),
            signal_id: "signal-1".to_string(),
            worker_id: "w1".to_string(),
            machine_id: "machine-1".to_string(),
            namespace: "ns".to_string(),
            task_id: "task-1".to_string(),
            context_id: "ctx-1".to_string(),
            snapshot_id: Some("snap-1".to_string()),
            objective: "do stuff".to_string(),
            worker_role: "coder".to_string(),
            worker_tier: DispatchWorkerTier::Medium,
            status: DispatchStatus::Assigned,
            pressure: DispatchPressureProfile {
                anxiety: 0.5,
                fear: 0.2,
                confidence: 0.7,
                discipline: 0.6,
                habit: 0.4,
            },
            envelope: super::DispatchAssignmentEnvelope {
                machine: super::super::continuity::MachineProfile {
                    machine_id: "machine-1".to_string(),
                    label: "test-machine".to_string(),
                    namespace: "ns".to_string(),
                    default_task_id: "task-1".to_string(),
                    host_name: "localhost".to_string(),
                    os_name: "linux".to_string(),
                    kernel_version: None,
                    storage_root: "/tmp/test".to_string(),
                },
                signal_id: "signal-1".to_string(),
                worker_id: "w1".to_string(),
                worker_display_name: "Worker 1".to_string(),
                worker_role: "coder".to_string(),
                worker_tier: DispatchWorkerTier::Medium,
                objective: "do stuff".to_string(),
                start_summary: "start here".to_string(),
                resume: super::DispatchResumeHandle {
                    context_id: "ctx-1".to_string(),
                    snapshot_id: Some("snap-1".to_string()),
                    namespace: "ns".to_string(),
                    task_id: "task-1".to_string(),
                    objective: "do stuff".to_string(),
                    token_budget: 224,
                    candidate_limit: 24,
                },
                context_preview: super::DispatchContextPreview {
                    pack_id: "pack-1".to_string(),
                    latest_snapshot_id: Some("snap-1".to_string()),
                    pack_used_tokens: 100,
                    hot_memory: vec!["memory-1".to_string()],
                    decisions: vec!["decision-1".to_string()],
                    constraints: vec!["constraint-1".to_string()],
                    operational_scars: vec!["scar-1".to_string()],
                    open_threads: vec!["thread-1".to_string()],
                },
                pressure: DispatchPressureProfile {
                    anxiety: 0.5,
                    fear: 0.2,
                    confidence: 0.7,
                    discipline: 0.6,
                    habit: 0.4,
                },
                attached_projected_lane: None,
                attached_projected_lane_source: None,
            },
            created_at: Utc::now(),
            completed_at: None,
        }
    }

    #[test]
    fn dispatch_assignment_presence_copies_fields() {
        let assignment = make_assignment_record();
        let presence = super::dispatch_assignment_presence(assignment);
        assert_eq!(presence.assignment_id, "assign-1");
        assert_eq!(presence.signal_id, "signal-1");
        assert_eq!(presence.worker_id, "w1");
        assert_eq!(presence.worker_role, "coder");
        assert_eq!(presence.worker_tier, DispatchWorkerTier::Medium);
        assert_eq!(presence.namespace, "ns");
        assert_eq!(presence.task_id, "task-1");
        assert_eq!(presence.context_id, "ctx-1");
        assert_eq!(presence.objective, "do stuff");
        assert_eq!(presence.status, DispatchStatus::Assigned);
        assert!(
            (presence.pressure.anxiety - 0.5).abs() < f64::EPSILON,
            "pressure should be copied"
        );
        assert_eq!(presence.projected_lane.projection_id, "dispatch:worker:w1");
        assert_eq!(presence.projected_lane.task_id.as_deref(), Some("task-1"));
    }

    #[test]
    fn dispatch_assignment_presence_with_attached_lane() {
        let mut assignment = make_assignment_record();
        assignment.envelope.attached_projected_lane = Some(CoordinationProjectedLane {
            projection_id: "branch:feat/y".to_string(),
            projection_kind: "branch".to_string(),
            label: "feat/y".to_string(),
            resource: None,
            repo_root: None,
            branch: Some("feat/y".to_string()),
            task_id: None,
        });
        assignment.envelope.attached_projected_lane_source =
            Some(DispatchAttachedLaneSource::LiveBadgeOptIn);
        let presence = super::dispatch_assignment_presence(assignment);
        let lane = presence.attached_projected_lane.unwrap();
        assert_eq!(lane.projection_id, "branch:feat/y");
        assert_eq!(
            presence.attached_projected_lane_source,
            Some(DispatchAttachedLaneSource::LiveBadgeOptIn)
        );
    }

    // ---- DispatchProjectionMetricState::new ----

    #[test]
    fn projection_metric_state_new_defaults() {
        let lane = CoordinationProjectedLane {
            projection_id: "proj:test".to_string(),
            projection_kind: "test".to_string(),
            label: "test".to_string(),
            resource: Some("res/path".to_string()),
            repo_root: Some("/repo".to_string()),
            branch: Some("main".to_string()),
            task_id: Some("task-1".to_string()),
        };
        let state = DispatchProjectionMetricState::new(&lane, Some("ns"), Some("fallback"));
        assert_eq!(state.projection_id, "proj:test");
        assert_eq!(state.projection_kind, "test");
        assert_eq!(state.label, "test");
        assert_eq!(state.resource.as_deref(), Some("res/path"));
        assert_eq!(state.repo_root.as_deref(), Some("/repo"));
        assert_eq!(state.branch.as_deref(), Some("main"));
        assert_eq!(state.task_id.as_deref(), Some("task-1"));
        assert_eq!(state.namespace, "ns");
        assert_eq!(state.worker_count, 0);
        assert_eq!(state.assignment_count, 0);
        assert!(state.assignment_anxiety_max.abs() < f64::EPSILON);
        assert_eq!(state.assignment_explicit_cli_count, 0);
        assert_eq!(state.assignment_live_badge_opt_in_count, 0);
    }

    #[test]
    fn projection_metric_state_new_uses_task_id_fallback() {
        let lane = CoordinationProjectedLane {
            projection_id: "proj:test".to_string(),
            projection_kind: "test".to_string(),
            label: "test".to_string(),
            resource: None,
            repo_root: None,
            branch: None,
            task_id: None,
        };
        let state = DispatchProjectionMetricState::new(&lane, None, Some("fallback-task"));
        assert_eq!(state.task_id.as_deref(), Some("fallback-task"));
        assert_eq!(state.namespace, "");
    }

    #[test]
    fn projection_metric_state_new_lane_task_id_wins_over_fallback() {
        let lane = CoordinationProjectedLane {
            projection_id: "proj:test".to_string(),
            projection_kind: "test".to_string(),
            label: "test".to_string(),
            resource: None,
            repo_root: None,
            branch: None,
            task_id: Some("lane-task".to_string()),
        };
        let state = DispatchProjectionMetricState::new(&lane, None, Some("fallback"));
        assert_eq!(state.task_id.as_deref(), Some("lane-task"));
    }

    // ---- comprehensive serde round-trips for struct coverage ----

    #[test]
    fn dispatch_config_serde_with_defaults() {
        let json = r#"{"database_url":"postgres://localhost/test"}"#;
        let config: DispatchConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.notify_channel, "ice_dispatch_signal");
        assert_eq!(config.worker_stale_secs, 120);
    }

    #[test]
    fn dispatch_worker_tier_serde_round_trip() {
        for tier in [
            DispatchWorkerTier::Small,
            DispatchWorkerTier::Medium,
            DispatchWorkerTier::Large,
            DispatchWorkerTier::Script,
        ] {
            let json = serde_json::to_string(&tier).unwrap();
            let parsed: DispatchWorkerTier = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, tier);
        }
    }

    #[test]
    fn dispatch_signal_kind_serde_round_trip() {
        for kind in [
            DispatchSignalKind::TaskComplete,
            DispatchSignalKind::HandoffReady,
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let parsed: DispatchSignalKind = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, kind);
        }
    }

    #[test]
    fn dispatch_status_serde_round_trip() {
        for status in [
            DispatchStatus::Queued,
            DispatchStatus::Assigned,
            DispatchStatus::Completed,
            DispatchStatus::Failed,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let parsed: DispatchStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, status);
        }
    }

    #[test]
    fn dispatch_attached_lane_source_serde_round_trip() {
        for source in [
            DispatchAttachedLaneSource::ExplicitCli,
            DispatchAttachedLaneSource::LiveBadgeOptIn,
        ] {
            let json = serde_json::to_string(&source).unwrap();
            let parsed: DispatchAttachedLaneSource = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, source);
        }
    }

    #[test]
    fn dispatch_worker_upsert_input_serde_with_defaults() {
        let json = serde_json::json!({
            "worker_id": "w1",
            "display_name": "Worker 1",
            "role": "coder",
            "agent_type": "claude",
            "tier": "small",
            "model": "sonnet",
        });
        let input: super::DispatchWorkerUpsertInput = serde_json::from_value(json).unwrap();
        assert_eq!(input.worker_id, "w1");
        assert_eq!(input.max_parallelism, 1);
        assert_eq!(input.status, "idle");
        assert!(input.capabilities.is_empty());
        assert_eq!(input.focus, "");
        assert!(input.namespace.is_none());
        assert!(input.task_id.is_none());
        assert!(input.metadata.is_null() || input.metadata == serde_json::json!({}));
    }

    #[test]
    fn dispatch_worker_upsert_input_serde_with_all_fields() {
        let json = serde_json::json!({
            "worker_id": "w1",
            "display_name": "Worker 1",
            "role": "coder",
            "agent_type": "claude",
            "tier": "large",
            "model": "opus",
            "capabilities": ["code", "review"],
            "max_parallelism": 4,
            "focus": "fixing bugs",
            "namespace": "ns",
            "task_id": "task-1",
            "status": "busy",
            "metadata": {"key": "value"},
        });
        let input: super::DispatchWorkerUpsertInput = serde_json::from_value(json).unwrap();
        assert_eq!(input.tier, DispatchWorkerTier::Large);
        assert_eq!(input.max_parallelism, 4);
        assert_eq!(input.capabilities, vec!["code", "review"]);
        assert_eq!(input.status, "busy");
        assert_eq!(input.namespace.as_deref(), Some("ns"));
    }

    #[test]
    fn dispatch_signal_record_serde_round_trip() {
        let record = make_signal(Some("coder"), Some(DispatchWorkerTier::Small));
        let json = serde_json::to_value(&record).unwrap();
        let parsed: DispatchSignalRecord = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.id, record.id);
        assert_eq!(parsed.kind, record.kind);
        assert_eq!(parsed.status, record.status);
        assert_eq!(parsed.target_role.as_deref(), Some("coder"));
        assert_eq!(parsed.preferred_tier, Some(DispatchWorkerTier::Small));
    }

    #[test]
    fn dispatch_worker_record_serde_round_trip() {
        let worker = make_worker("w1", "coder", DispatchWorkerTier::Medium, 2, 4);
        let json = serde_json::to_value(&worker).unwrap();
        let parsed: DispatchWorkerRecord = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.worker_id, "w1");
        assert_eq!(parsed.role, "coder");
        assert_eq!(parsed.tier, DispatchWorkerTier::Medium);
        assert_eq!(parsed.active_assignment_count, 2);
        assert_eq!(parsed.max_parallelism, 4);
    }

    #[test]
    fn dispatch_complete_input_serde_with_defaults() {
        let json = serde_json::json!({
            "kind": "task_complete",
            "agent_id": "agent-1",
            "title": "Fixed the bug",
            "result": "All tests pass",
            "quality": 0.95,
            "target_tier": "medium",
        });
        let input: super::DispatchCompleteInput = serde_json::from_value(json).unwrap();
        assert_eq!(input.kind, DispatchSignalKind::TaskComplete);
        assert_eq!(input.agent_id, "agent-1");
        assert!((input.quality - 0.95).abs() < f64::EPSILON);
        assert!(input.failures.is_empty());
        assert!(input.context_id.is_none());
        assert!(input.namespace.is_none());
        assert!(input.task_id.is_none());
        assert_eq!(input.token_budget, 224);
        assert_eq!(input.candidate_limit, 24);
    }

    #[test]
    fn dispatch_claim_input_serde_with_defaults() {
        let json = serde_json::json!({
            "worker": {
                "worker_id": "w1",
                "display_name": "Worker 1",
                "role": "coder",
                "agent_type": "claude",
                "tier": "small",
                "model": "sonnet",
            },
            "lease_seconds": 300,
        });
        let input: super::DispatchClaimInput = serde_json::from_value(json).unwrap();
        assert_eq!(input.lease_seconds, 300);
        assert_eq!(input.token_budget, 224);
        assert_eq!(input.candidate_limit, 24);
        assert_eq!(input.worker.worker_id, "w1");
    }

    #[test]
    fn dispatch_resume_handle_serde_round_trip() {
        let handle = super::DispatchResumeHandle {
            context_id: "ctx-1".to_string(),
            snapshot_id: Some("snap-1".to_string()),
            namespace: "ns".to_string(),
            task_id: "task-1".to_string(),
            objective: "do stuff".to_string(),
            token_budget: 224,
            candidate_limit: 24,
        };
        let json = serde_json::to_value(&handle).unwrap();
        let parsed: super::DispatchResumeHandle = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.context_id, "ctx-1");
        assert_eq!(parsed.snapshot_id.as_deref(), Some("snap-1"));
        assert_eq!(parsed.token_budget, 224);
    }

    #[test]
    fn dispatch_context_preview_serde_round_trip() {
        let preview = super::DispatchContextPreview {
            pack_id: "pack-1".to_string(),
            latest_snapshot_id: Some("snap-1".to_string()),
            pack_used_tokens: 100,
            hot_memory: vec!["m1".to_string(), "m2".to_string()],
            decisions: vec!["d1".to_string()],
            constraints: vec!["c1".to_string()],
            operational_scars: vec!["s1".to_string()],
            open_threads: vec!["t1".to_string()],
        };
        let json = serde_json::to_value(&preview).unwrap();
        let parsed: super::DispatchContextPreview = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.pack_id, "pack-1");
        assert_eq!(parsed.hot_memory.len(), 2);
        assert_eq!(parsed.pack_used_tokens, 100);
    }

    #[test]
    fn dispatch_publish_signal_input_serde() {
        let json = serde_json::json!({
            "kind": "handoff_ready",
            "from_agent_id": "agent-1",
            "context_id": "ctx-1",
            "namespace": "ns",
            "task_id": "task-1",
            "objective": "continue work",
        });
        let input: super::PublishDispatchSignalInput = serde_json::from_value(json).unwrap();
        assert_eq!(input.kind, DispatchSignalKind::HandoffReady);
        assert!(input.snapshot_id.is_none());
        assert!(input.target_role.is_none());
        assert!(input.preferred_tier.is_none());
        assert!(input.reason.is_none());
    }

    #[test]
    fn dispatch_organism_snapshot_serde_round_trip() {
        let snap = DispatchOrganismSnapshot::unconfigured();
        let json = serde_json::to_value(&snap).unwrap();
        let parsed: DispatchOrganismSnapshot = serde_json::from_value(json).unwrap();
        assert!(!parsed.configured);
        assert!(!parsed.reachable);
        assert!(parsed.workers.is_empty());
        assert!(parsed.assignments.is_empty());
    }

    #[test]
    fn route_dispatch_input_serde_with_defaults() {
        let json = serde_json::json!({
            "router_id": "router-1",
        });
        let input: super::RouteDispatchInput = serde_json::from_value(json).unwrap();
        assert_eq!(input.router_id, "router-1");
        assert_eq!(input.token_budget, 224);
        assert_eq!(input.candidate_limit, 24);
    }

    #[test]
    fn wait_dispatch_input_serde() {
        let json = serde_json::json!({
            "router_id": "router-1",
            "timeout_secs": 60,
        });
        let input: super::WaitDispatchInput = serde_json::from_value(json).unwrap();
        assert_eq!(input.route.router_id, "router-1");
        assert_eq!(input.timeout_secs, Some(60));
    }

    #[test]
    fn complete_assignment_input_serde() {
        let json = serde_json::json!({
            "assignment_id": "assign-1",
        });
        let input: super::CompleteAssignmentInput = serde_json::from_value(json).unwrap();
        assert_eq!(input.assignment_id, "assign-1");
        assert!(input.worker_id.is_none());
        assert!(!input.failed);
    }

    #[test]
    fn complete_assignment_input_serde_with_failed() {
        let json = serde_json::json!({
            "assignment_id": "assign-1",
            "worker_id": "w1",
            "failed": true,
        });
        let input: super::CompleteAssignmentInput = serde_json::from_value(json).unwrap();
        assert_eq!(input.worker_id.as_deref(), Some("w1"));
        assert!(input.failed);
    }

    #[test]
    fn dispatch_assignment_record_serde_round_trip() {
        let assignment = make_assignment_record();
        let json = serde_json::to_value(&assignment).unwrap();
        let parsed: super::DispatchAssignmentRecord = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.id, "assign-1");
        assert_eq!(parsed.signal_id, "signal-1");
        assert_eq!(parsed.worker_id, "w1");
        assert_eq!(parsed.worker_tier, DispatchWorkerTier::Medium);
        assert_eq!(parsed.status, DispatchStatus::Assigned);
        assert!((parsed.pressure.anxiety - 0.5).abs() < f64::EPSILON);
        assert_eq!(parsed.envelope.worker_display_name, "Worker 1");
    }

    #[test]
    fn dispatch_assignment_envelope_serde_round_trip() {
        let assignment = make_assignment_record();
        let json = serde_json::to_value(&assignment.envelope).unwrap();
        let parsed: super::DispatchAssignmentEnvelope = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.signal_id, "signal-1");
        assert_eq!(parsed.worker_id, "w1");
        assert_eq!(parsed.start_summary, "start here");
        assert_eq!(parsed.resume.context_id, "ctx-1");
        assert_eq!(parsed.context_preview.pack_id, "pack-1");
        assert_eq!(parsed.machine.host_name, "localhost");
    }

    // ---- worker_worker_presence serde ----

    #[test]
    fn dispatch_worker_presence_serde_round_trip() {
        let worker = make_worker("w1", "coder", DispatchWorkerTier::Medium, 0, 1);
        let presence = super::dispatch_worker_presence(worker, None);
        let json = serde_json::to_value(&presence).unwrap();
        let parsed: super::DispatchWorkerPresence = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.worker_id, "w1");
        assert_eq!(parsed.tier, DispatchWorkerTier::Medium);
    }

    #[test]
    fn dispatch_assignment_presence_serde_round_trip() {
        let assignment = make_assignment_record();
        let presence = super::dispatch_assignment_presence(assignment);
        let json = serde_json::to_value(&presence).unwrap();
        let parsed: super::DispatchAssignmentPresence = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.assignment_id, "assign-1");
        assert_eq!(parsed.worker_tier, DispatchWorkerTier::Medium);
    }

    // ---- more worker_score edge cases ----

    #[test]
    fn worker_score_no_target_role_still_gets_role_bonus() {
        let signal = make_signal(None, None);
        let worker = make_worker("w1", "coder", DispatchWorkerTier::Medium, 0, 1);
        let score = worker_score(&signal, &worker);
        assert!(
            score >= 50.0,
            "no target_role means any role matches for +50"
        );
    }

    #[test]
    fn worker_score_tier_bonus_ordering() {
        let signal = make_signal(None, None);
        let small = worker_score(
            &signal,
            &make_worker("w1", "coder", DispatchWorkerTier::Small, 0, 1),
        );
        let medium = worker_score(
            &signal,
            &make_worker("w2", "coder", DispatchWorkerTier::Medium, 0, 1),
        );
        let large = worker_score(
            &signal,
            &make_worker("w3", "coder", DispatchWorkerTier::Large, 0, 1),
        );
        let script = worker_score(
            &signal,
            &make_worker("w4", "coder", DispatchWorkerTier::Script, 0, 1),
        );
        assert!(large > medium, "large tier should score higher than medium");
        assert!(medium > small, "medium tier should score higher than small");
        assert!(
            small > script,
            "small tier should score higher than script (script gets -25 penalty)"
        );
    }

    #[test]
    fn worker_score_heavily_loaded_worker_scores_lower() {
        let signal = make_signal(None, None);
        let fresh = make_worker("w1", "coder", DispatchWorkerTier::Medium, 0, 8);
        let loaded = make_worker("w2", "coder", DispatchWorkerTier::Medium, 7, 8);
        assert!(worker_score(&signal, &fresh) > worker_score(&signal, &loaded));
    }

    // ---- select_best_worker more scenarios ----

    #[test]
    fn select_best_worker_prefers_lower_load() {
        let signal = make_signal(None, None);
        let busy = make_worker("w1", "coder", DispatchWorkerTier::Medium, 3, 4);
        let free = make_worker("w2", "coder", DispatchWorkerTier::Medium, 0, 4);
        let selected = select_best_worker(&signal, &[busy, free]).unwrap();
        assert_eq!(selected.worker_id, "w2");
    }

    #[test]
    fn select_best_worker_many_workers() {
        let signal = make_signal(Some("coder"), Some(DispatchWorkerTier::Small));
        let workers = vec![
            make_worker("w1", "debugger", DispatchWorkerTier::Large, 0, 1),
            make_worker("w2", "coder", DispatchWorkerTier::Large, 0, 1),
            make_worker("w3", "coder", DispatchWorkerTier::Small, 0, 1),
            make_worker("w4", "coder", DispatchWorkerTier::Medium, 0, 1),
        ];
        let selected = select_best_worker(&signal, &workers).unwrap();
        assert_eq!(
            selected.worker_id, "w3",
            "should pick worker matching both role and preferred tier"
        );
    }

    // ---- worker_can_take_signal more edge cases ----

    #[test]
    fn worker_can_take_signal_offline_case_insensitive() {
        let mut worker = make_worker("w1", "coder", DispatchWorkerTier::Small, 0, 1);
        worker.status = "OFFLINE".into();
        let signal = make_signal(None, None);
        assert!(!worker_can_take_signal(&worker, &signal));
    }

    #[test]
    fn worker_can_take_signal_script_no_preferred_rejects() {
        let worker = make_worker("w1", "runner", DispatchWorkerTier::Script, 0, 1);
        let signal = make_signal(None, None);
        assert!(
            !worker_can_take_signal(&worker, &signal),
            "script worker should reject signals without preferred_tier=script"
        );
    }

    #[test]
    fn worker_can_take_signal_matching_role_accepted() {
        let worker = make_worker("w1", "coder", DispatchWorkerTier::Medium, 0, 2);
        let signal = make_signal(Some("coder"), None);
        assert!(worker_can_take_signal(&worker, &signal));
    }

    // ---- apply_affect_override full field coverage ----

    #[test]
    fn apply_affect_override_all_fields() {
        let profile = DispatchPressureProfile {
            anxiety: 0.5,
            fear: 0.5,
            confidence: 0.5,
            discipline: 0.5,
            habit: 0.5,
        };
        let override_val = serde_json::json!({
            "anxiety": 0.1,
            "fear": 0.2,
            "confidence": 0.3,
            "discipline": 0.4,
            "habit": 0.9,
        });
        let result = apply_affect_override(profile, Some(&override_val));
        assert!((result.anxiety - 0.1).abs() < f64::EPSILON);
        assert!((result.fear - 0.2).abs() < f64::EPSILON);
        assert!((result.confidence - 0.3).abs() < f64::EPSILON);
        assert!((result.discipline - 0.4).abs() < f64::EPSILON);
        assert!((result.habit - 0.9).abs() < f64::EPSILON);
    }

    // ---- attached_projected_lane_from_metadata edge cases ----

    #[test]
    fn attached_lane_from_metadata_invalid_json_returns_none() {
        let metadata = serde_json::json!({"attached_lane": "not an object"});
        assert!(attached_projected_lane_from_metadata(&metadata, None).is_none());
    }

    #[test]
    fn attached_lane_from_metadata_blank_task_id_uses_fallback() {
        let metadata = serde_json::json!({
            "attached_lane": {
                "projection_id": "proj:1",
                "projection_kind": "branch",
                "label": "my branch",
                "task_id": "  ",
            }
        });
        let lane = attached_projected_lane_from_metadata(&metadata, Some("fallback")).unwrap();
        assert_eq!(lane.task_id.as_deref(), Some("fallback"));
    }

    #[test]
    fn attached_lane_from_metadata_blank_fallback_leaves_none() {
        let metadata = serde_json::json!({
            "attached_lane": {
                "projection_id": "proj:1",
                "projection_kind": "branch",
                "label": "my branch",
            }
        });
        let lane = attached_projected_lane_from_metadata(&metadata, Some("  ")).unwrap();
        assert!(lane.task_id.is_none());
    }

    #[test]
    fn attached_lane_from_metadata_partial_empty_fields_returns_none() {
        let metadata = serde_json::json!({
            "attached_lane": {
                "projection_id": "proj:1",
                "projection_kind": "  ",
                "label": "my branch",
            }
        });
        assert!(attached_projected_lane_from_metadata(&metadata, None).is_none());
    }

    #[test]
    fn attached_lane_source_from_metadata_live_badge() {
        let metadata = serde_json::json!({"attached_lane_source": "live_badge_opt_in"});
        let source = attached_projected_lane_source_from_metadata(&metadata).unwrap();
        assert_eq!(source, DispatchAttachedLaneSource::LiveBadgeOptIn);
    }

    #[test]
    fn attached_lane_source_from_metadata_invalid_returns_none() {
        let metadata = serde_json::json!({"attached_lane_source": "unknown_source"});
        assert!(attached_projected_lane_source_from_metadata(&metadata).is_none());
    }

    // ---- dispatch_worker_projected_lane more ----

    #[test]
    fn dispatch_worker_projected_lane_no_task_id() {
        let lane = dispatch_worker_projected_lane("w-abc", None);
        assert_eq!(lane.projection_id, "dispatch:worker:w-abc");
        assert!(lane.task_id.is_none());
        assert!(lane.repo_root.is_none());
        assert!(lane.branch.is_none());
        assert_eq!(lane.resource.as_deref(), Some("dispatch/worker/w-abc"));
    }

    // ---- compact_line and compact_metric_label edge cases ----

    #[test]
    fn compact_line_single_char_max() {
        let result = compact_line("abcdef", 1);
        assert_eq!(result.chars().count(), 1);
        assert!(result.ends_with('…'));
    }

    #[test]
    fn compact_line_exact_length() {
        let result = compact_line("hello", 5);
        assert_eq!(result, "hello");
    }

    #[test]
    fn compact_line_all_blank_lines_returns_empty() {
        let result = compact_line("\n\n\n", 10);
        assert_eq!(result, "");
    }

    #[test]
    fn compact_metric_label_exact_length() {
        let result = compact_metric_label("hello", 5);
        assert_eq!(result, "hello");
    }

    // ---- prometheus_label_value edge cases ----

    #[test]
    fn prometheus_label_value_multiple_special_chars() {
        let result = prometheus_label_value("a\\b\nc\"d");
        assert_eq!(result, "a\\\\b\\nc\\\"d");
    }

    #[test]
    fn prometheus_label_value_empty_string() {
        assert_eq!(prometheus_label_value(""), "");
    }

    // ---- is_valid_pg_identifier edge cases ----

    #[test]
    fn pg_identifier_single_char_valid() {
        assert!(is_valid_pg_identifier("a"));
        assert!(is_valid_pg_identifier("_"));
    }

    #[test]
    fn pg_identifier_numeric_chars_in_body_valid() {
        assert!(is_valid_pg_identifier("a123"));
    }

    #[test]
    fn pg_identifier_special_chars_invalid() {
        assert!(!is_valid_pg_identifier("a.b"));
        assert!(!is_valid_pg_identifier("a;b"));
        assert!(!is_valid_pg_identifier("a'b"));
    }

    // ---- dispatch_down_metrics edge cases ----

    #[test]
    fn dispatch_down_metrics_escapes_special_chars_in_error() {
        let text = dispatch_down_metrics("error with \"quotes\" and\nnewlines");
        assert!(text.contains("ice_dispatch_up 0"));
        assert!(text.contains("ice_dispatch_status"));
        assert!(!text.contains('\n') || text.matches('\n').count() >= 2);
    }

    #[test]
    fn dispatch_down_metrics_long_error_gets_truncated() {
        let long_error = "a".repeat(200);
        let text = dispatch_down_metrics(&long_error);
        assert!(text.contains("ice_dispatch_up 0"));
        assert!(
            text.contains('…'),
            "long error should be truncated with ellipsis"
        );
    }

    // ---- projection_metric_labels edge cases ----

    #[test]
    fn projection_metric_labels_with_none_fields() {
        let state = DispatchProjectionMetricState {
            projection_id: "proj:1".to_string(),
            projection_kind: "test".to_string(),
            label: "test".to_string(),
            resource: None,
            repo_root: None,
            branch: None,
            task_id: None,
            namespace: "".to_string(),
            worker_count: 0,
            assignment_count: 0,
            assignment_anxiety_max: 0.0,
            assignment_explicit_cli_count: 0,
            assignment_live_badge_opt_in_count: 0,
        };
        let labels = projection_metric_labels(&state);
        assert!(labels.contains("namespace=\"\""));
        assert!(labels.contains("resource=\"\""));
        assert!(labels.contains("repo_root=\"\""));
        assert!(labels.contains("branch=\"\""));
        assert!(labels.contains("task_id=\"\""));
    }

    // ---- record_dispatch_projection_assignment edge cases ----

    #[test]
    fn record_dispatch_projection_assignment_creates_new_entry() {
        let mut projections = BTreeMap::new();
        let lane = CoordinationProjectedLane {
            projection_id: "proj:new".to_string(),
            projection_kind: "test".to_string(),
            label: "new lane".to_string(),
            resource: None,
            repo_root: None,
            branch: None,
            task_id: None,
        };
        record_dispatch_projection_assignment(
            &mut projections,
            &lane,
            Some("ns"),
            None,
            0.7,
            Some(DispatchAttachedLaneSource::LiveBadgeOptIn),
        );
        let state = projections.get("proj:new").unwrap();
        assert_eq!(state.assignment_count, 1);
        assert!((state.assignment_anxiety_max - 0.7).abs() < f64::EPSILON);
        assert_eq!(state.assignment_live_badge_opt_in_count, 1);
        assert_eq!(state.assignment_explicit_cli_count, 0);
    }

    #[test]
    fn record_dispatch_projection_assignment_no_source() {
        let mut projections = BTreeMap::new();
        let lane = CoordinationProjectedLane {
            projection_id: "proj:x".to_string(),
            projection_kind: "test".to_string(),
            label: "x".to_string(),
            resource: None,
            repo_root: None,
            branch: None,
            task_id: None,
        };
        record_dispatch_projection_assignment(&mut projections, &lane, None, None, 0.3, None);
        let state = projections.get("proj:x").unwrap();
        assert_eq!(state.assignment_explicit_cli_count, 0);
        assert_eq!(state.assignment_live_badge_opt_in_count, 0);
    }

    // ---- DispatchConfig::load from file ----

    #[test]
    fn dispatch_config_load_invalid_json_file_fails() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::write(data_dir.join("dispatch-config.json"), "not json").unwrap();
        assert!(DispatchConfig::load(dir.path()).is_err());
    }

    #[test]
    fn dispatch_config_load_file_with_bad_channel_fails() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();
        let json = serde_json::json!({
            "database_url": "postgres://localhost/test",
            "notify_channel": "bad-channel",
        });
        std::fs::write(
            data_dir.join("dispatch-config.json"),
            serde_json::to_string(&json).unwrap(),
        )
        .unwrap();
        assert!(DispatchConfig::load(dir.path()).is_err());
    }

    // ---- DispatchAffectOverride serde ----

    #[test]
    fn affect_override_serde_round_trip() {
        let override_val = DispatchAffectOverride {
            anxiety: Some(0.1),
            confidence: Some(0.9),
            fear: None,
            discipline: Some(0.5),
            habit: None,
        };
        let json = serde_json::to_value(&override_val).unwrap();
        let parsed: DispatchAffectOverride = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.anxiety, Some(0.1));
        assert_eq!(parsed.confidence, Some(0.9));
        assert!(parsed.fear.is_none());
        assert_eq!(parsed.discipline, Some(0.5));
        assert!(parsed.habit.is_none());
    }

    // ---- DispatchConfig::save validation ----

    #[test]
    fn dispatch_config_save_validates_before_writing() {
        let dir = tempfile::tempdir().unwrap();
        let bad_config = DispatchConfig {
            database_url: "  ".to_string(),
            notify_channel: "ice_dispatch_signal".to_string(),
            worker_stale_secs: 120,
        };
        assert!(bad_config.save(dir.path()).is_err());
    }

    // ---- organism_snapshot with config but no postgres ----

    #[test]
    fn organism_snapshot_with_config_but_unreachable() {
        let dir = tempfile::tempdir().unwrap();
        let config = DispatchConfig {
            database_url: "postgres://localhost:59999/nonexistent".to_string(),
            notify_channel: "ice_dispatch_signal".to_string(),
            worker_stale_secs: 120,
        };
        config.save(dir.path()).unwrap();
        let snap = super::organism_snapshot(dir.path(), None);
        assert!(snap.configured);
        assert!(!snap.reachable);
        assert!(snap.error.is_some());
    }
}
