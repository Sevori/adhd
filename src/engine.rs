use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use anyhow::Result;

use crate::config::EngineConfig;
use crate::continuity::AgentBadgeRecord;
use crate::dispatch;
use crate::embedding::cosine_similarity;
use crate::model::{
    ContextPack, ContextPackManifest, DimensionValue, EventInput, HandoffInput, HandoffRecord,
    IngestManifest, MemoryLayer, MemoryRecord, MetricsSnapshot, QueryInput, RelationRecord,
    ReplayRow, Selector, SubscriptionInput, SubscriptionPoll, SubscriptionRecord, ViewInput,
    ViewManifest, ViewRecord,
};
use crate::query::build_context_pack;
use crate::storage::Storage;
use crate::telemetry::EngineTelemetry;
use uuid::Uuid;

pub struct Engine {
    pub(crate) telemetry: std::sync::Arc<EngineTelemetry>,
    pub(crate) storage: Mutex<Storage>,
}

struct EngineStorageGuard<'a> {
    guard: MutexGuard<'a, Storage>,
    telemetry: &'a EngineTelemetry,
    operation: &'static str,
    acquired_at: Instant,
}

impl Deref for EngineStorageGuard<'_> {
    type Target = Storage;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl DerefMut for EngineStorageGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

impl Drop for EngineStorageGuard<'_> {
    fn drop(&mut self) {
        let seconds = self.acquired_at.elapsed().as_secs_f64();
        self.telemetry
            .observe_engine_storage_lock_hold_seconds(seconds);
        self.telemetry
            .observe_engine_storage_lock_hold_seconds_for_operation(self.operation, seconds);
    }
}

impl Engine {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let config = EngineConfig::with_root(root);
        let storage = Storage::open(config)?;
        Ok(Self {
            telemetry: EngineTelemetry::new(),
            storage: Mutex::new(storage),
        })
    }

    pub fn ingest(&self, input: EventInput) -> Result<IngestManifest> {
        let mut storage = self.lock_storage("ingest");
        storage.ingest(input, &self.telemetry)
    }

    fn lock_storage(&self, operation: &'static str) -> EngineStorageGuard<'_> {
        let wait_started_at = Instant::now();
        let guard = match self.storage.lock() {
            Ok(storage) => storage,
            Err(poisoned) => {
                self.telemetry.observe_engine_storage_lock_poison_recovery();
                tracing::warn!(
                    operation,
                    "engine storage lock was poisoned; recovering inner state"
                );
                poisoned.into_inner()
            }
        };
        let seconds = wait_started_at.elapsed().as_secs_f64();
        self.telemetry
            .observe_engine_storage_lock_wait_seconds(seconds);
        self.telemetry
            .observe_engine_storage_lock_wait_seconds_for_operation(operation, seconds);
        EngineStorageGuard {
            guard,
            telemetry: &self.telemetry,
            operation,
            acquired_at: Instant::now(),
        }
    }

    pub(crate) fn with_storage<T>(
        &self,
        f: impl FnOnce(&Storage, &EngineTelemetry) -> Result<T>,
    ) -> Result<T> {
        let storage = self.lock_storage("with_storage");
        f(&storage, &self.telemetry)
    }

    pub(crate) fn with_storage_mut<T>(
        &self,
        f: impl FnOnce(&mut Storage, &EngineTelemetry) -> Result<T>,
    ) -> Result<T> {
        let mut storage = self.lock_storage("with_storage_mut");
        f(&mut storage, &self.telemetry)
    }

    pub fn list_memory(
        &self,
        layer: Option<MemoryLayer>,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>> {
        let storage = self.lock_storage("list_memory");
        storage.list_memory(layer, limit)
    }

    pub fn replay(&self, session_id: Option<&str>, limit: usize) -> Result<Vec<ReplayRow>> {
        let storage = self.lock_storage("replay");
        storage.replay(&self.telemetry, session_id, limit)
    }

    pub fn replay_by_selector(&self, selector: &Selector, limit: usize) -> Result<Vec<ReplayRow>> {
        let storage = self.lock_storage("replay_by_selector");
        storage.replay_by_selector(&self.telemetry, selector, limit)
    }

    pub fn metrics_snapshot(&self) -> Result<MetricsSnapshot> {
        let telemetry_started_at = Instant::now();
        let mut prometheus_text = self.telemetry.render_prometheus()?;
        self.telemetry.observe_metrics_snapshot_phase_seconds(
            "telemetry_render",
            telemetry_started_at.elapsed().as_secs_f64(),
        );

        let storage_started_at = Instant::now();
        let (root, sqlite_path, log_dir) = {
            let storage = self.lock_storage("metrics_snapshot");
            storage.append_metrics_text(&mut prometheus_text)?;
            (
                storage.config.root.clone(),
                storage.paths.sqlite_path.clone(),
                storage.paths.log_dir.clone(),
            )
        };
        self.telemetry.observe_metrics_snapshot_phase_seconds(
            "storage_metrics",
            storage_started_at.elapsed().as_secs_f64(),
        );

        let storage_bytes_started_at = Instant::now();
        crate::storage::append_storage_bytes_metric_text(
            &mut prometheus_text,
            &sqlite_path,
            &log_dir,
        )?;
        self.telemetry.observe_metrics_snapshot_phase_seconds(
            "storage_bytes",
            storage_bytes_started_at.elapsed().as_secs_f64(),
        );

        let dispatch_started_at = Instant::now();
        dispatch::append_metrics(&mut prometheus_text, &root);
        self.telemetry.observe_metrics_snapshot_phase_seconds(
            "dispatch_append",
            dispatch_started_at.elapsed().as_secs_f64(),
        );
        Ok(MetricsSnapshot { prometheus_text })
    }

    pub fn embedding_backend_key(&self) -> String {
        let storage = self.lock_storage("embedding_backend_key");
        storage.embedding_backend_key()
    }

    pub fn list_agent_badges(
        &self,
        namespace: Option<&str>,
        task_id: Option<&str>,
    ) -> Result<Vec<AgentBadgeRecord>> {
        let storage = self.lock_storage("list_agent_badges");
        storage.list_agent_badges(namespace, task_id)
    }

    pub fn build_context_pack(&self, query: QueryInput) -> Result<ContextPack> {
        let storage = self.lock_storage("build_context_pack");
        build_context_pack(&storage, query)
    }

    pub fn explain_context_pack(&self, id: &str) -> Result<ContextPackManifest> {
        let storage = self.lock_storage("explain_context_pack");
        storage.explain_context_pack(id)
    }

    pub fn annotate_item(
        &self,
        item_type: &str,
        item_id: &str,
        dimensions: &[DimensionValue],
    ) -> Result<Vec<DimensionValue>> {
        let storage = self.lock_storage("annotate_item");
        storage.annotate_item(item_type, item_id, dimensions)
    }

    pub fn relate_items(
        &self,
        source_id: &str,
        target_id: &str,
        relation: &str,
        weight: f64,
        attributes: serde_json::Value,
    ) -> Result<RelationRecord> {
        let storage = self.lock_storage("relate_items");
        storage.relate_items(source_id, target_id, relation, weight, attributes)
    }

    pub fn materialize_view(&self, input: ViewInput) -> Result<ViewRecord> {
        let storage = self.lock_storage("materialize_view");
        storage.materialize_view(input)
    }

    pub fn get_view(&self, id: &str) -> Result<ViewRecord> {
        let storage = self.lock_storage("get_view");
        storage.get_view(id)
    }

    pub fn explain_view(&self, id: &str) -> Result<ViewManifest> {
        let storage = self.lock_storage("explain_view");
        storage.explain_view(id)
    }

    pub fn fork_view(&self, id: &str, owner_agent_id: Option<String>) -> Result<ViewRecord> {
        let storage = self.lock_storage("fork_view");
        storage.fork_view(id, owner_agent_id)
    }

    pub fn create_handoff(&self, input: HandoffInput) -> Result<HandoffRecord> {
        let storage = self.lock_storage("create_handoff");
        let view = if let Some(view_id) = &input.view_id {
            storage.get_view(view_id)?
        } else {
            storage.materialize_view(ViewInput {
                op: crate::model::ViewOp::Slice,
                owner_agent_id: Some(input.from_agent_id.clone()),
                namespace: input.namespace.clone(),
                objective: input.objective.clone(),
                selectors: input.selector.clone().into_iter().collect(),
                source_view_ids: Vec::new(),
                resolution: Some(crate::model::SnapshotResolution::Medium),
                limit: Some(48),
            })?
        };
        let pack = build_context_pack(
            &storage,
            QueryInput {
                agent_id: Some(input.to_agent_id.clone()),
                session_id: None,
                task_id: None,
                namespace: input.namespace.clone(),
                objective: input.objective.clone(),
                selector: input.selector.clone(),
                view_id: Some(view.id.clone()),
                query_text: input.query_text.clone(),
                budget_tokens: input.budget_tokens,
                candidate_limit: 32,
            },
        )?;
        let handoff = HandoffRecord {
            id: format!("handoff:{}", Uuid::now_v7()),
            created_at: chrono::Utc::now(),
            from_agent_id: input.from_agent_id.clone(),
            to_agent_id: input.to_agent_id.clone(),
            reason: input.reason.clone(),
            view_id: view.id.clone(),
            pack_id: pack.id.clone(),
            conflict_count: view.conflict_count,
            manifest_path: storage
                .paths
                .debug_dir
                .join("handoffs")
                .join(format!("{}.json", Uuid::now_v7()))
                .display()
                .to_string(),
        };
        let view_manifest = storage.explain_view(&view.id)?;
        let pack_manifest = storage.explain_context_pack(&pack.id)?;
        storage.persist_handoff(
            &handoff,
            &serde_json::json!({
                "handoff": handoff.clone(),
                "input": input,
                "view": view_manifest,
                "pack": pack_manifest,
            }),
            &pack.query.query_text,
        )?;
        storage.get_handoff(&handoff.id)
    }

    pub fn get_handoff(&self, id: &str) -> Result<HandoffRecord> {
        let storage = self.lock_storage("get_handoff");
        storage.get_handoff(id)
    }

    pub fn explain_handoff(&self, id: &str) -> Result<serde_json::Value> {
        let storage = self.lock_storage("explain_handoff");
        storage.explain_handoff(id)
    }

    pub fn create_subscription(&self, input: SubscriptionInput) -> Result<SubscriptionRecord> {
        let storage = self.lock_storage("create_subscription");
        storage.create_subscription(input)
    }

    pub fn poll_subscription(&self, id: &str, limit: usize) -> Result<SubscriptionPoll> {
        let storage = self.lock_storage("poll_subscription");
        storage.poll_subscription(id, limit)
    }

    pub fn vector_baseline(
        &self,
        query_text: &str,
        session_id: Option<&str>,
        task_id: Option<&str>,
        agent_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>> {
        let storage = self.lock_storage("vector_baseline");
        let query_vector = storage.embed_query_vector(query_text)?;
        let mut scored = storage
            .vector_memories()?
            .into_iter()
            .filter(|(memory, _)| session_id.is_none_or(|value| memory.session_id == value))
            .filter(|(memory, _)| {
                task_id.is_none_or(|value| memory.task_id.as_deref() == Some(value))
            })
            .filter(|(memory, _)| agent_id.is_none_or(|value| memory.agent_id == value))
            .map(|(memory, vector)| (memory, cosine_similarity(&query_vector, &vector)))
            .collect::<Vec<_>>();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored
            .into_iter()
            .take(limit)
            .map(|(memory, _)| memory)
            .collect())
    }

    pub fn summary_baseline(
        &self,
        session_id: Option<&str>,
        task_id: Option<&str>,
        agent_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>> {
        let storage = self.lock_storage("summary_baseline");
        Ok(storage
            .list_memory(Some(MemoryLayer::Summary), limit.saturating_mul(8))?
            .into_iter()
            .filter(|memory| session_id.is_none_or(|value| memory.session_id == value))
            .filter(|memory| task_id.is_none_or(|value| memory.task_id.as_deref() == Some(value)))
            .filter(|memory| agent_id.is_none_or(|value| memory.agent_id == value))
            .take(limit)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::time::Duration;

    use tempfile::tempdir;

    use super::*;

    fn poison_storage_mutex(engine: &Arc<Engine>) {
        let engine = Arc::clone(engine);
        let result = std::thread::spawn(move || {
            let _guard = engine.lock_storage("poison_test");
            panic!("poison the engine storage lock");
        })
        .join();
        assert!(result.is_err());
    }

    fn sample_event() -> EventInput {
        EventInput {
            kind: crate::model::EventKind::Note,
            agent_id: "agent-test".to_string(),
            agent_role: Some("tester".to_string()),
            session_id: "session-test".to_string(),
            task_id: Some("task-test".to_string()),
            project_id: None,
            goal_id: None,
            run_id: None,
            namespace: Some("namespace-test".to_string()),
            environment: None,
            source: "engine-test".to_string(),
            scope: crate::model::Scope::Shared,
            tags: vec!["engine".to_string()],
            dimensions: Vec::new(),
            content: "engine poison recovery".to_string(),
            attributes: serde_json::json!({}),
        }
    }

    fn metric_counter(metrics_text: &str, name: &str) -> u64 {
        metrics_text
            .lines()
            .find_map(|line| {
                line.strip_prefix(&format!("{name} "))
                    .and_then(|value| value.parse::<u64>().ok())
            })
            .unwrap_or_else(|| panic!("missing counter metric {name}"))
    }

    fn metric_sum(metrics_text: &str, name: &str) -> f64 {
        metrics_text
            .lines()
            .find_map(|line| {
                line.strip_prefix(&format!("{name} "))
                    .and_then(|value| value.parse::<f64>().ok())
            })
            .unwrap_or_else(|| panic!("missing sum metric {name}"))
    }

    fn labeled_metric_sum(metrics_text: &str, name: &str, operation: &str) -> f64 {
        let prefix = format!("{name}{{operation=\"{operation}\"}} ");
        metrics_text
            .lines()
            .find_map(|line| {
                line.strip_prefix(&prefix)
                    .and_then(|value| value.parse::<f64>().ok())
            })
            .unwrap_or_else(|| panic!("missing labeled sum metric {name} for {operation}"))
    }

    #[test]
    fn engine_recovers_after_storage_mutex_poison_for_reads() {
        let dir = tempdir().unwrap();
        let engine = Arc::new(Engine::open(dir.path()).unwrap());
        poison_storage_mutex(&engine);

        let memories = engine.list_memory(None, 8).unwrap();
        assert!(memories.is_empty());
    }

    #[test]
    fn engine_recovers_after_storage_mutex_poison_for_writes() {
        let dir = tempdir().unwrap();
        let engine = Arc::new(Engine::open(dir.path()).unwrap());
        poison_storage_mutex(&engine);

        let manifest = engine.ingest(sample_event()).unwrap();
        assert_eq!(manifest.event.input.content, "engine poison recovery");

        let memories = engine.list_memory(None, 8).unwrap();
        assert!(!memories.is_empty());
    }

    #[test]
    fn engine_lock_metrics_render_after_poison_recovery() {
        let dir = tempdir().unwrap();
        let engine = Arc::new(Engine::open(dir.path()).unwrap());
        poison_storage_mutex(&engine);

        let _ = engine.list_memory(None, 8).unwrap();
        let metrics = engine.metrics_snapshot().unwrap();

        assert!(
            metrics
                .prometheus_text
                .contains("ice_engine_storage_lock_wait_seconds")
        );
        assert!(
            metrics
                .prometheus_text
                .contains("ice_engine_storage_lock_hold_seconds")
        );
        assert!(
            metrics
                .prometheus_text
                .contains("ice_engine_storage_lock_poison_recoveries")
        );
    }

    #[test]
    fn engine_lock_metrics_capture_real_contention() {
        let dir = tempdir().unwrap();
        let engine = Arc::new(Engine::open(dir.path()).unwrap());
        let (holder_ready_tx, holder_ready_rx) = mpsc::channel();
        let (release_holder_tx, release_holder_rx) = mpsc::channel();
        let holder_engine = Arc::clone(&engine);
        let holder = std::thread::spawn(move || {
            let _guard = holder_engine.lock_storage("contention_holder");
            holder_ready_tx.send(()).unwrap();
            release_holder_rx
                .recv_timeout(Duration::from_secs(1))
                .unwrap();
        });
        holder_ready_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();

        let (waiter_started_tx, waiter_started_rx) = mpsc::channel();
        let waiter_engine = Arc::clone(&engine);
        let waiter = std::thread::spawn(move || {
            waiter_started_tx.send(()).unwrap();
            let _guard = waiter_engine.lock_storage("contention_waiter");
        });
        waiter_started_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();

        // Hold the lock long enough for the waiter path to incur measurable delay.
        std::thread::sleep(Duration::from_millis(60));
        release_holder_tx.send(()).unwrap();

        holder.join().unwrap();
        waiter.join().unwrap();

        let metrics = engine.metrics_snapshot().unwrap();
        let metrics_text = &metrics.prometheus_text;
        let wait_count = metric_counter(metrics_text, "ice_engine_storage_lock_wait_seconds_count");
        let hold_count = metric_counter(metrics_text, "ice_engine_storage_lock_hold_seconds_count");
        let wait_sum = metric_sum(metrics_text, "ice_engine_storage_lock_wait_seconds_sum");
        let hold_sum = metric_sum(metrics_text, "ice_engine_storage_lock_hold_seconds_sum");
        let waiter_wait_sum = labeled_metric_sum(
            metrics_text,
            "ice_engine_storage_lock_wait_seconds_by_operation_sum",
            "contention_waiter",
        );
        let holder_hold_sum = labeled_metric_sum(
            metrics_text,
            "ice_engine_storage_lock_hold_seconds_by_operation_sum",
            "contention_holder",
        );
        // metrics_snapshot now renders telemetry before it takes the engine lock,
        // so the returned snapshot sees the holder/waiter contention but not its
        // own lock observation yet.
        assert_eq!(wait_count, 2);
        assert_eq!(hold_count, 2);
        assert!(
            wait_sum >= 0.04,
            "expected contention wait time to be recorded, got {wait_sum}"
        );
        assert!(
            hold_sum >= 0.04,
            "expected held lock time to be recorded, got {hold_sum}"
        );
        assert!(
            waiter_wait_sum >= 0.04,
            "expected contention waiter wait time to be recorded, got {waiter_wait_sum}"
        );
        assert!(
            holder_hold_sum >= 0.04,
            "expected contention holder hold time to be recorded, got {holder_hold_sum}"
        );
    }

    #[test]
    fn metrics_snapshot_releases_engine_lock_before_dispatch_append() {
        let dir = tempdir().unwrap();
        let engine = Arc::new(Engine::open(dir.path()).unwrap());
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        crate::dispatch::install_append_metrics_test_hook(
            dir.path().to_path_buf(),
            entered_tx,
            release_rx,
        );

        let snapshot_engine = Arc::clone(&engine);
        let snapshot = std::thread::spawn(move || snapshot_engine.metrics_snapshot().unwrap());
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        let (list_done_tx, list_done_rx) = mpsc::channel();
        let list_engine = Arc::clone(&engine);
        let list_thread = std::thread::spawn(move || {
            let _ = list_engine.list_memory(None, 8).unwrap();
            list_done_tx.send(()).unwrap();
        });

        assert!(
            list_done_rx
                .recv_timeout(Duration::from_millis(200))
                .is_ok(),
            "expected list_memory to proceed while dispatch metrics append is blocked"
        );

        release_tx.send(()).unwrap();
        snapshot.join().unwrap();
        list_thread.join().unwrap();
    }

    #[test]
    fn metrics_snapshot_does_not_take_engine_lock_during_telemetry_render() {
        let dir = tempdir().unwrap();
        let engine = Arc::new(Engine::open(dir.path()).unwrap());
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        engine
            .telemetry
            .install_render_prometheus_test_hook(entered_tx, release_rx);

        let snapshot_engine = Arc::clone(&engine);
        let snapshot = std::thread::spawn(move || snapshot_engine.metrics_snapshot().unwrap());
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        let (list_done_tx, list_done_rx) = mpsc::channel();
        let list_engine = Arc::clone(&engine);
        let list_thread = std::thread::spawn(move || {
            let _ = list_engine.list_memory(None, 8).unwrap();
            list_done_tx.send(()).unwrap();
        });

        assert!(
            list_done_rx
                .recv_timeout(Duration::from_millis(200))
                .is_ok(),
            "expected list_memory to proceed while telemetry rendering is blocked"
        );

        release_tx.send(()).unwrap();
        snapshot.join().unwrap();
        list_thread.join().unwrap();
    }

    #[test]
    fn metrics_snapshot_releases_engine_lock_before_storage_bytes_sidecar() {
        let dir = tempdir().unwrap();
        let engine = Arc::new(Engine::open(dir.path()).unwrap());
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        crate::storage::Storage::install_storage_bytes_metrics_test_hook(
            crate::config::EngineConfig::with_root(dir.path())
                .paths()
                .log_dir,
            entered_tx,
            release_rx,
        );

        let snapshot_engine = Arc::clone(&engine);
        let snapshot = std::thread::spawn(move || snapshot_engine.metrics_snapshot().unwrap());
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        let (list_done_tx, list_done_rx) = mpsc::channel();
        let list_engine = Arc::clone(&engine);
        let list_thread = std::thread::spawn(move || {
            let _ = list_engine.list_memory(None, 8).unwrap();
            list_done_tx.send(()).unwrap();
        });

        assert!(
            list_done_rx
                .recv_timeout(Duration::from_millis(200))
                .is_ok(),
            "expected list_memory to proceed while storage-bytes metrics are blocked"
        );

        release_tx.send(()).unwrap();
        snapshot.join().unwrap();
        list_thread.join().unwrap();
    }

    #[test]
    fn metrics_snapshot_phase_metrics_render_on_following_snapshot() {
        let dir = tempdir().unwrap();
        let engine = Arc::new(Engine::open(dir.path()).unwrap());

        let _ = engine.metrics_snapshot().unwrap();
        let metrics = engine.metrics_snapshot().unwrap();

        assert!(
            metrics.prometheus_text.contains(
                "ice_engine_metrics_snapshot_phase_seconds_sum{phase=\"telemetry_render\"}"
            )
        );
        assert!(
            metrics.prometheus_text.contains(
                "ice_engine_metrics_snapshot_phase_seconds_sum{phase=\"storage_metrics\"}"
            )
        );
        assert!(
            metrics
                .prometheus_text
                .contains("ice_engine_metrics_snapshot_phase_seconds_sum{phase=\"storage_bytes\"}")
        );
        assert!(
            metrics.prometheus_text.contains(
                "ice_engine_metrics_snapshot_phase_seconds_sum{phase=\"dispatch_append\"}"
            )
        );
    }
}
