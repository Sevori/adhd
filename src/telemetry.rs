use std::io;
use std::sync::Arc;

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use prometheus_client::registry::Registry;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct EngineLockOperationLabels {
    operation: &'static str,
}

type EngineLockOperationHistogram = Family<EngineLockOperationLabels, Histogram, fn() -> Histogram>;

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct MetricsSnapshotPhaseLabels {
    phase: &'static str,
}

type MetricsSnapshotPhaseHistogram =
    Family<MetricsSnapshotPhaseLabels, Histogram, fn() -> Histogram>;

fn engine_storage_lock_wait_histogram() -> Histogram {
    Histogram::new(exponential_buckets(0.000_005, 2.0, 20))
}

fn engine_storage_lock_hold_histogram() -> Histogram {
    Histogram::new(exponential_buckets(0.000_05, 2.0, 20))
}

fn metrics_snapshot_phase_histogram() -> Histogram {
    Histogram::new(exponential_buckets(0.000_01, 2.0, 20))
}

#[cfg(test)]
#[derive(Debug)]
struct RenderPrometheusTestHook {
    entered_tx: std::sync::mpsc::Sender<()>,
    release_rx: std::sync::mpsc::Receiver<()>,
}

#[derive(Debug)]
pub struct EngineTelemetry {
    registry: std::sync::Mutex<Registry>,
    ingest_events_total: Counter,
    ingest_bytes_total: Counter,
    replay_requests_total: Counter,
    engine_storage_lock_poison_recoveries_total: Counter,
    hot_items: Gauge<i64>,
    episodic_items: Gauge<i64>,
    raw_log_append_seconds: Histogram,
    sqlite_insert_seconds: Histogram,
    promotion_seconds: Histogram,
    engine_storage_lock_wait_seconds: Histogram,
    engine_storage_lock_hold_seconds: Histogram,
    engine_storage_lock_wait_seconds_by_operation: EngineLockOperationHistogram,
    engine_storage_lock_hold_seconds_by_operation: EngineLockOperationHistogram,
    metrics_snapshot_phase_seconds: MetricsSnapshotPhaseHistogram,
    #[cfg(test)]
    render_prometheus_test_hook: std::sync::Mutex<Option<RenderPrometheusTestHook>>,
}

impl EngineTelemetry {
    pub fn new() -> Arc<Self> {
        let mut registry = Registry::default();

        let ingest_events_total = Counter::default();
        let ingest_bytes_total = Counter::default();
        let replay_requests_total = Counter::default();
        let engine_storage_lock_poison_recoveries_total = Counter::default();
        let hot_items = Gauge::default();
        let episodic_items = Gauge::default();
        let raw_log_append_seconds = Histogram::new(exponential_buckets(0.000_5, 2.0, 20));
        let sqlite_insert_seconds = Histogram::new(exponential_buckets(0.000_5, 2.0, 20));
        let promotion_seconds = Histogram::new(exponential_buckets(0.000_5, 2.0, 20));
        let engine_storage_lock_wait_seconds = engine_storage_lock_wait_histogram();
        let engine_storage_lock_hold_seconds = engine_storage_lock_hold_histogram();
        let engine_storage_lock_wait_seconds_by_operation = Family::<
            EngineLockOperationLabels,
            Histogram,
            fn() -> Histogram,
        >::new_with_constructor(
            engine_storage_lock_wait_histogram
        );
        let engine_storage_lock_hold_seconds_by_operation = Family::<
            EngineLockOperationLabels,
            Histogram,
            fn() -> Histogram,
        >::new_with_constructor(
            engine_storage_lock_hold_histogram
        );
        let metrics_snapshot_phase_seconds = Family::<
            MetricsSnapshotPhaseLabels,
            Histogram,
            fn() -> Histogram,
        >::new_with_constructor(
            metrics_snapshot_phase_histogram
        );

        registry.register(
            "ice_ingest_events",
            "Total ingested events",
            ingest_events_total.clone(),
        );
        registry.register(
            "ice_ingest_bytes",
            "Total ingested bytes",
            ingest_bytes_total.clone(),
        );
        registry.register(
            "ice_replay_requests",
            "Total replay requests",
            replay_requests_total.clone(),
        );
        registry.register(
            "ice_engine_storage_lock_poison_recoveries",
            "Total times the engine recovered a poisoned storage lock",
            engine_storage_lock_poison_recoveries_total.clone(),
        );
        registry.register(
            "ice_hot_items",
            "Current hot memory items",
            hot_items.clone(),
        );
        registry.register(
            "ice_episodic_items",
            "Current episodic memory items",
            episodic_items.clone(),
        );
        registry.register(
            "ice_raw_log_append_seconds",
            "Raw log append duration",
            raw_log_append_seconds.clone(),
        );
        registry.register(
            "ice_sqlite_insert_seconds",
            "SQLite insert duration",
            sqlite_insert_seconds.clone(),
        );
        registry.register(
            "ice_promotion_seconds",
            "Promotion duration for memory layers",
            promotion_seconds.clone(),
        );
        registry.register(
            "ice_engine_storage_lock_wait_seconds",
            "Engine storage lock acquisition wait duration",
            engine_storage_lock_wait_seconds.clone(),
        );
        registry.register(
            "ice_engine_storage_lock_hold_seconds",
            "Engine storage lock hold duration",
            engine_storage_lock_hold_seconds.clone(),
        );
        registry.register(
            "ice_engine_storage_lock_wait_seconds_by_operation",
            "Engine storage lock acquisition wait duration grouped by engine operation",
            engine_storage_lock_wait_seconds_by_operation.clone(),
        );
        registry.register(
            "ice_engine_storage_lock_hold_seconds_by_operation",
            "Engine storage lock hold duration grouped by engine operation",
            engine_storage_lock_hold_seconds_by_operation.clone(),
        );
        registry.register(
            "ice_engine_metrics_snapshot_phase_seconds",
            "Engine metrics snapshot phase duration grouped by snapshot phase",
            metrics_snapshot_phase_seconds.clone(),
        );

        Arc::new(Self {
            registry: std::sync::Mutex::new(registry),
            ingest_events_total,
            ingest_bytes_total,
            replay_requests_total,
            engine_storage_lock_poison_recoveries_total,
            hot_items,
            episodic_items,
            raw_log_append_seconds,
            sqlite_insert_seconds,
            promotion_seconds,
            engine_storage_lock_wait_seconds,
            engine_storage_lock_hold_seconds,
            engine_storage_lock_wait_seconds_by_operation,
            engine_storage_lock_hold_seconds_by_operation,
            metrics_snapshot_phase_seconds,
            #[cfg(test)]
            render_prometheus_test_hook: std::sync::Mutex::new(None),
        })
    }

    pub fn init_logging() {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
            .with_current_span(true)
            .with_target(false)
            .with_writer(io::stderr)
            .init();
    }

    pub fn observe_ingest_event(&self, bytes: u64) {
        self.ingest_events_total.inc();
        self.ingest_bytes_total.inc_by(bytes);
    }

    pub fn observe_replay(&self) {
        self.replay_requests_total.inc();
    }

    pub fn observe_engine_storage_lock_poison_recovery(&self) {
        self.engine_storage_lock_poison_recoveries_total.inc();
    }

    pub fn set_hot_items(&self, count: i64) {
        self.hot_items.set(count);
    }

    pub fn set_episodic_items(&self, count: i64) {
        self.episodic_items.set(count);
    }

    pub fn observe_raw_log_append_seconds(&self, seconds: f64) {
        self.raw_log_append_seconds.observe(seconds);
    }

    pub fn observe_sqlite_insert_seconds(&self, seconds: f64) {
        self.sqlite_insert_seconds.observe(seconds);
    }

    pub fn observe_promotion_seconds(&self, seconds: f64) {
        self.promotion_seconds.observe(seconds);
    }

    pub fn observe_engine_storage_lock_wait_seconds(&self, seconds: f64) {
        self.engine_storage_lock_wait_seconds.observe(seconds);
    }

    pub fn observe_engine_storage_lock_hold_seconds(&self, seconds: f64) {
        self.engine_storage_lock_hold_seconds.observe(seconds);
    }

    pub fn observe_engine_storage_lock_wait_seconds_for_operation(
        &self,
        operation: &'static str,
        seconds: f64,
    ) {
        self.engine_storage_lock_wait_seconds_by_operation
            .get_or_create(&EngineLockOperationLabels { operation })
            .observe(seconds);
    }

    pub fn observe_engine_storage_lock_hold_seconds_for_operation(
        &self,
        operation: &'static str,
        seconds: f64,
    ) {
        self.engine_storage_lock_hold_seconds_by_operation
            .get_or_create(&EngineLockOperationLabels { operation })
            .observe(seconds);
    }

    pub fn observe_metrics_snapshot_phase_seconds(&self, phase: &'static str, seconds: f64) {
        self.metrics_snapshot_phase_seconds
            .get_or_create(&MetricsSnapshotPhaseLabels { phase })
            .observe(seconds);
    }

    pub fn render_prometheus(&self) -> anyhow::Result<String> {
        #[cfg(test)]
        {
            if let Some(hook) = self.render_prometheus_test_hook.lock().unwrap().take() {
                hook.entered_tx.send(()).unwrap();
                hook.release_rx
                    .recv_timeout(std::time::Duration::from_secs(1))?;
            }
        }
        let mut body = String::new();
        let registry = self.registry.lock().unwrap();
        encode(&mut body, &registry)?;
        Ok(body)
    }

    #[cfg(test)]
    pub(crate) fn install_render_prometheus_test_hook(
        &self,
        entered_tx: std::sync::mpsc::Sender<()>,
        release_rx: std::sync::mpsc::Receiver<()>,
    ) {
        *self.render_prometheus_test_hook.lock().unwrap() = Some(RenderPrometheusTestHook {
            entered_tx,
            release_rx,
        });
    }
}
