use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Result, anyhow};
use chrono::Utc;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use sysinfo::{Pid, ProcessesToUpdate, System};

use crate::adapters::{
    AgentAdapter, AgentAdapterConfig, AgentContinuationOutput, DecisionNote, EvidenceNote,
    ModelCallMetrics, OllamaAdapter, SurvivalHypothesis,
};
use crate::continuity::{
    AttachAgentInput, ContextRead, ContinuityHandoffInput, ContinuityItemInput, ContinuityKind,
    ContinuityStatus, HandoffProof, OpenContextInput, ReadContextInput, SharedContinuityKernel,
    SignalInput, SnapshotInput, UnifiedContinuityInterface,
};
use crate::model::{
    DimensionValue, EventInput, EventKind, MemoryLayer, Scope, Selector, SnapshotResolution,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuityBenchConfig {
    pub output_dir: PathBuf,
    pub ollama_endpoint: String,
    pub strong_model: String,
    pub small_model: String,
    pub embedding_backend: String,
    pub retrieval_protocol: String,
    #[serde(default)]
    pub classes: Vec<BenchmarkClass>,
    pub token_budget: usize,
    pub candidate_limit: usize,
    pub recent_window: usize,
    pub timeout_secs: u64,
    pub num_predict: usize,
}

impl ContinuityBenchConfig {
    pub fn protocol_id(&self) -> &str {
        &self.retrieval_protocol
    }

    pub fn strong_agent(&self, agent_id: &str, role: &str, namespace: &str) -> AgentAdapterConfig {
        AgentAdapterConfig {
            agent_id: agent_id.to_string(),
            agent_type: "ollama".to_string(),
            model: self.strong_model.clone(),
            endpoint: self.ollama_endpoint.clone(),
            namespace: namespace.to_string(),
            role: role.to_string(),
            timeout_secs: self.timeout_secs,
            num_predict: self.num_predict,
        }
    }

    pub fn small_agent(&self, agent_id: &str, role: &str, namespace: &str) -> AgentAdapterConfig {
        AgentAdapterConfig {
            agent_id: agent_id.to_string(),
            agent_type: "ollama".to_string(),
            model: self.small_model.clone(),
            endpoint: self.ollama_endpoint.clone(),
            namespace: namespace.to_string(),
            role: role.to_string(),
            timeout_secs: self.timeout_secs,
            num_predict: self.num_predict,
        }
    }

    pub fn selected_classes(&self) -> Vec<BenchmarkClass> {
        if self.classes.is_empty() {
            BenchmarkClass::all()
        } else {
            self.classes.clone()
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum BaselineKind {
    SharedContinuity,
    Isolated,
    RecentWindow,
    VectorOnly,
    RollingSummary,
    FullTranscript,
}

impl BaselineKind {
    pub fn slug(self) -> &'static str {
        match self {
            Self::SharedContinuity => "shared-continuity",
            Self::Isolated => "isolated",
            Self::RecentWindow => "recent-window",
            Self::VectorOnly => "vector-only",
            Self::RollingSummary => "rolling-summary",
            Self::FullTranscript => "full-transcript",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkClass {
    AgentSwapSurvival,
    StrongToSmallContinuation,
    SmallToSmallRelay,
    InterruptionStress,
    OperationalScar,
    CrossAgentCollaborative,
    CrashRecovery,
    MemoryPollution,
    ContextBudgetCompression,
    BaselineIsolation,
}

impl BenchmarkClass {
    pub fn all() -> Vec<Self> {
        vec![
            Self::AgentSwapSurvival,
            Self::StrongToSmallContinuation,
            Self::SmallToSmallRelay,
            Self::InterruptionStress,
            Self::OperationalScar,
            Self::CrossAgentCollaborative,
            Self::CrashRecovery,
            Self::MemoryPollution,
            Self::ContextBudgetCompression,
            Self::BaselineIsolation,
        ]
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::AgentSwapSurvival => "agent-swap-survival",
            Self::StrongToSmallContinuation => "strong-to-small",
            Self::SmallToSmallRelay => "small-to-small",
            Self::InterruptionStress => "interruption-stress",
            Self::OperationalScar => "operational-scar",
            Self::CrossAgentCollaborative => "cross-agent-collaboration",
            Self::CrashRecovery => "crash-recovery",
            Self::MemoryPollution => "memory-pollution",
            Self::ContextBudgetCompression => "budget-compression",
            Self::BaselineIsolation => "baseline-isolation",
        }
    }

    fn uses_handoff_proof(self) -> bool {
        matches!(
            self,
            Self::AgentSwapSurvival
                | Self::StrongToSmallContinuation
                | Self::SmallToSmallRelay
                | Self::CrossAgentCollaborative
                | Self::CrashRecovery
                | Self::MemoryPollution
                | Self::ContextBudgetCompression
                | Self::InterruptionStress
                | Self::OperationalScar
        )
    }

    fn shared_path_role(self) -> ContinuityPathRole {
        if self.uses_handoff_proof() {
            ContinuityPathRole::ProofPath
        } else if matches!(self, Self::BaselineIsolation) {
            ContinuityPathRole::ExplicitControl
        } else {
            ContinuityPathRole::Legacy
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuiteReport {
    pub generated_at: String,
    pub config: ContinuityBenchConfig,
    pub classes: Vec<BenchmarkClassReport>,
    pub summary: BenchmarkSummary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta_lessons: Option<MetaLessonReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase3: Option<Phase3ValidationReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkSummary {
    pub class_count: usize,
    pub avg_cfsr: f64,
    pub avg_csr: f64,
    pub avg_dlf: f64,
    pub avg_ras: f64,
    pub avg_mpr: f64,
    pub avg_pc: f64,
    pub avg_smcl: f64,
    pub avg_sscr: f64,
    pub avg_cgd: f64,
    pub failed_runs: usize,
    pub total_runs: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkClassReport {
    pub class: BenchmarkClass,
    pub scenario_id: String,
    pub continuity: BaselineRunReport,
    pub baselines: Vec<BaselineRunReport>,
    pub metrics: BenchmarkMetrics,
    pub resource: ResourceEnvelope,
    pub artifacts: Vec<String>,
    /// Phase 3: A/B comparison of hypothesis injection (treatment) vs control.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hypothesis_injection: Option<Phase3ClassResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineRunReport {
    pub baseline: BaselineKind,
    pub status: BaselineStatus,
    pub model: String,
    pub retrieval_ms: u128,
    pub model_metrics: ModelCallMetrics,
    pub envelope_tokens: usize,
    pub evaluation: Evaluation,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub survival: Option<SurvivalReport>,
    pub failure: Option<String>,
    pub artifacts: Vec<String>,
    pub continuity_path: Option<ContinuityPathReport>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BaselineStatus {
    Ok,
    Failed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContinuityPathKind {
    ReadContextOnly,
    HandoffProof,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContinuityPathRole {
    ProofPath,
    ExplicitControl,
    Legacy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuityPathReport {
    pub kind: ContinuityPathKind,
    pub role: ContinuityPathRole,
    pub proof_register_count: usize,
    #[serde(default)]
    pub proof_register_labels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkMetrics {
    pub cfsr: f64,
    pub csr: f64,
    pub dlf: f64,
    pub hrt: f64,
    pub crl_ms: f64,
    pub smcl: f64,
    pub sscr: f64,
    pub osr: f64,
    pub mrr: f64,
    pub mpr: f64,
    pub pc: f64,
    pub cpqt: f64,
    pub cgd: f64,
    pub pvl_ms: f64,
    pub ris: f64,
    pub dwr: f64,
    pub ras: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceEnvelope {
    pub elapsed_ms: u128,
    pub process_memory_bytes: u64,
    pub process_virtual_memory_bytes: u64,
    pub process_cpu_percent: f32,
    pub storage_bytes: u64,
    pub gpu_report: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Evaluation {
    pub critical_fact_survival_rate: f64,
    pub constraint_survival_rate: f64,
    pub decision_lineage_fidelity: f64,
    pub operational_scar_retention: f64,
    pub mistake_recurrence_rate: f64,
    pub memory_pollution_rate: f64,
    pub provenance_coverage: f64,
    pub context_pack_quality_per_token: f64,
    pub resume_accuracy_score: f64,
    pub duplicate_work_rate: f64,
    pub matched_critical_facts: usize,
    pub matched_constraints: usize,
    pub matched_decisions: usize,
    pub matched_scars: usize,
    pub unsupported_items: usize,
    pub total_items: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct Scenario {
    pub(crate) id: String,
    pub(crate) namespace: String,
    pub(crate) task_id: String,
    pub(crate) objective: String,
    pub(crate) phases: Vec<ScenarioPhase>,
    pub(crate) truth: GroundTruth,
}

#[derive(Debug, Clone)]
pub(crate) struct ScenarioPhase {
    pub(crate) actor_id: String,
    pub(crate) actor_role: String,
    pub(crate) kind: EventKind,
    pub(crate) scope: Scope,
    pub(crate) content: String,
    pub(crate) dimensions: Vec<DimensionValue>,
    pub(crate) marks: Vec<ScenarioMark>,
}

#[derive(Debug, Clone)]
pub(crate) struct ScenarioMark {
    pub(crate) kind: ContinuityKind,
    pub(crate) title: String,
    pub(crate) body: String,
    pub(crate) status: ContinuityStatus,
    pub(crate) dimensions: Vec<DimensionValue>,
}

#[derive(Debug, Clone)]
pub(crate) struct TruthItem {
    pub(crate) keywords: Vec<&'static str>,
    pub(crate) rationale_keywords: Vec<&'static str>,
    pub(crate) judge_note: Option<&'static str>,
    pub(crate) judge_required_concepts: Vec<&'static str>,
}

#[derive(Debug, Clone)]
pub(crate) struct GroundTruth {
    pub(crate) critical_facts: Vec<TruthItem>,
    pub(crate) constraints: Vec<TruthItem>,
    pub(crate) decisions: Vec<TruthItem>,
    pub(crate) scars: Vec<TruthItem>,
    pub(crate) avoid_repeating: Vec<TruthItem>,
    pub(crate) next_step_keywords: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ContextEnvelope {
    pub(crate) provider: BaselineKind,
    pub(crate) retrieval_ms: u128,
    pub(crate) text: String,
    pub(crate) token_estimate: usize,
    pub(crate) surfaced: Vec<SurfacedItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SurfacedItem {
    pub(crate) label: String,
    pub(crate) support_type: String,
    pub(crate) support_id: String,
    pub(crate) text: String,
    pub(crate) has_provenance: bool,
}

pub fn run_continuity_suite(config: ContinuityBenchConfig) -> Result<BenchmarkSuiteReport> {
    std::fs::create_dir_all(&config.output_dir)?;
    let started = Utc::now();

    // Phase 3: load hypotheses from previous meta-lessons if available.
    let prior_hypotheses = load_prior_hypotheses(&config.output_dir);

    let mut classes = Vec::new();
    for class in config.selected_classes() {
        classes.push(run_class(class, &config, &prior_hypotheses)?);
    }
    let summary = summarize(&classes);
    let mut report = BenchmarkSuiteReport {
        generated_at: started.to_rfc3339(),
        config: config.clone(),
        classes,
        summary,
        meta_lessons: None,
        phase3: None,
    };

    // Phase 2: generate meta-lessons from this run.
    let meta = generate_meta_lessons(&[report.clone()], 0.05);
    if !meta.lessons.is_empty() {
        let meta_path = config.output_dir.join("meta-lessons.json");
        std::fs::write(&meta_path, serde_json::to_vec_pretty(&meta)?)?;
    }
    report.meta_lessons = Some(meta);

    // Phase 3: if we injected hypotheses, generate validation report.
    if !prior_hypotheses.is_empty() {
        let class_results: Vec<Phase3ClassResult> = report
            .classes
            .iter()
            .filter_map(|cr| cr.hypothesis_injection.clone())
            .collect();
        if !class_results.is_empty() {
            let cycle = detect_validation_cycle(&config.output_dir);
            let phase3 = generate_phase3_report(&class_results, &prior_hypotheses, cycle);
            let phase3_path = config.output_dir.join("phase3-report.json");
            std::fs::write(&phase3_path, serde_json::to_vec_pretty(&phase3)?)?;
            report.phase3 = Some(phase3);
        }
    }

    let suite_report_path = config.output_dir.join("suite-report.json");
    std::fs::write(&suite_report_path, serde_json::to_vec_pretty(&report)?)?;
    let summary_path = config.output_dir.join("summary.md");
    std::fs::write(&summary_path, render_suite_markdown(&report))?;
    Ok(report)
}

/// Load survival hypotheses from a previous meta-lessons.json in the output directory.
fn load_prior_hypotheses(output_dir: &Path) -> Vec<SurvivalHypothesis> {
    let meta_path = output_dir.join("meta-lessons.json");
    let Ok(contents) = std::fs::read_to_string(&meta_path) else {
        return Vec::new();
    };
    let Ok(report) = serde_json::from_str::<MetaLessonReport>(&contents) else {
        return Vec::new();
    };
    extract_eligible_hypotheses(&report)
}

/// Detect which validation cycle we are on by counting prior phase3 reports.
fn detect_validation_cycle(output_dir: &Path) -> usize {
    let phase3_path = output_dir.join("phase3-report.json");
    let Ok(contents) = std::fs::read_to_string(&phase3_path) else {
        return 1;
    };
    let Ok(prior) = serde_json::from_str::<Phase3ValidationReport>(&contents) else {
        return 1;
    };
    prior.cycle + 1
}

fn run_class(
    class: BenchmarkClass,
    config: &ContinuityBenchConfig,
    hypotheses: &[SurvivalHypothesis],
) -> Result<BenchmarkClassReport> {
    let scenario = scenario_for(class);
    let class_root = config.output_dir.join(class.slug());
    if class_root.exists() {
        std::fs::remove_dir_all(&class_root)?;
    }
    std::fs::create_dir_all(&class_root)?;
    let start = Instant::now();

    let mut sys = System::new_all();
    let pid = sysinfo::get_current_pid().unwrap_or(Pid::from(0));
    sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);

    let kernel = SharedContinuityKernel::open(&class_root)?;
    let planner =
        OllamaAdapter::new(config.strong_agent("planner-strong", "planner", &scenario.namespace))?;
    let small_a = OllamaAdapter::new(config.small_agent("relay-a", "coder", &scenario.namespace))?;
    let small_b =
        OllamaAdapter::new(config.small_agent("relay-b", "debugger", &scenario.namespace))?;

    let attach = kernel.attach_agent(AttachAgentInput {
        agent_id: planner.config().agent_id.clone(),
        agent_type: planner.config().agent_type.clone(),
        capabilities: vec!["plan".into(), "derive".into()],
        namespace: scenario.namespace.clone(),
        role: Some(planner.config().role.clone()),
        metadata: serde_json::json!({"model": planner.config().model}),
    })?;
    let context = kernel.open_context(OpenContextInput {
        namespace: scenario.namespace.clone(),
        task_id: scenario.task_id.clone(),
        session_id: format!("{}-session", class.slug()),
        objective: scenario.objective.clone(),
        selector: None,
        agent_id: Some(planner.config().agent_id.clone()),
        attachment_id: Some(attach.id),
    })?;
    populate_scenario(&kernel, &context.id, &scenario)?;

    let mut pvl_ms = 0.0_f64;
    if matches!(class, BenchmarkClass::CrossAgentCollaborative) {
        let subscription = kernel.subscribe(crate::model::SubscriptionInput {
            agent_id: small_b.config().agent_id.clone(),
            name: Some("signals".into()),
            selector: Selector {
                all: vec![crate::model::DimensionFilter {
                    key: "context".into(),
                    values: vec![context.id.clone()],
                }],
                any: Vec::new(),
                exclude: Vec::new(),
                layers: vec![MemoryLayer::Hot],
                start_ts: None,
                end_ts: None,
                limit: Some(16),
                namespace: Some(context.namespace.clone()),
            },
        })?;
        let signal_start = Instant::now();
        kernel.publish_signal(SignalInput {
            context_id: context.id.clone(),
            agent_id: planner.config().agent_id.clone(),
            title: "handoff-ready".into(),
            body: "Planner published a ready signal for the coder/tester/debugger relay.".into(),
            dimensions: vec![DimensionValue {
                key: "signal".into(),
                value: "ready".into(),
                weight: 100,
            }],
            extra: serde_json::json!({}),
        })?;
        let _ = kernel.poll_subscription(&subscription.id, 16)?;
        pvl_ms = signal_start.elapsed().as_millis() as f64;
    }

    let seed_report = match class {
        BenchmarkClass::StrongToSmallContinuation => Some(analyze_and_write_back(
            &class_root,
            class,
            &kernel,
            &context.id,
            &scenario,
            &planner,
            BaselineKind::FullTranscript,
            config,
        )?),
        BenchmarkClass::SmallToSmallRelay | BenchmarkClass::CrossAgentCollaborative => {
            Some(analyze_and_write_back(
                &class_root,
                class,
                &kernel,
                &context.id,
                &scenario,
                &small_a,
                BaselineKind::SharedContinuity,
                config,
            )?)
        }
        _ => None,
    };

    if matches!(class, BenchmarkClass::CrashRecovery) {
        let _ = kernel.snapshot(SnapshotInput {
            context_id: Some(context.id.clone()),
            namespace: None,
            task_id: None,
            objective: Some("pre-crash checkpoint".into()),
            selector: None,
            resolution: SnapshotResolution::Medium,
            token_budget: config.token_budget,
            candidate_limit: config.candidate_limit,
            owner_agent_id: Some(planner.config().agent_id.clone()),
        })?;
    }

    let continuity = run_baseline(
        &class_root,
        class,
        BaselineKind::SharedContinuity,
        &context.id,
        &scenario,
        &small_b,
        config,
    )?;
    let mut baselines = Vec::new();
    for baseline in [
        BaselineKind::Isolated,
        BaselineKind::RecentWindow,
        BaselineKind::VectorOnly,
        BaselineKind::RollingSummary,
        BaselineKind::FullTranscript,
    ] {
        baselines.push(run_baseline(
            &class_root,
            class,
            baseline,
            &context.id,
            &scenario,
            &small_b,
            config,
        )?);
    }

    let isolated = baselines
        .iter()
        .find(|item| item.baseline == BaselineKind::Isolated)
        .cloned()
        .ok_or_else(|| anyhow!("missing isolated baseline"))?;
    let full_transcript = baselines
        .iter()
        .find(|item| item.baseline == BaselineKind::FullTranscript)
        .cloned()
        .ok_or_else(|| anyhow!("missing full transcript baseline"))?;

    let ris = if matches!(class, BenchmarkClass::CrashRecovery) {
        continuity.evaluation.resume_accuracy_score
    } else {
        0.0
    };
    let sscr = if let Some(seed) = &seed_report {
        ratio_or_zero(
            continuity.evaluation.resume_accuracy_score,
            seed.evaluation.resume_accuracy_score.max(0.01),
        )
    } else {
        0.0
    };
    let metrics = BenchmarkMetrics {
        cfsr: continuity.evaluation.critical_fact_survival_rate,
        csr: continuity.evaluation.constraint_survival_rate,
        dlf: continuity.evaluation.decision_lineage_fidelity,
        hrt: inferred_hrt(continuity.evaluation.resume_accuracy_score),
        crl_ms: continuity.retrieval_ms as f64,
        smcl: ratio_or_zero(
            continuity.evaluation.resume_accuracy_score,
            isolated.evaluation.resume_accuracy_score.max(0.01),
        ),
        sscr,
        osr: continuity.evaluation.operational_scar_retention,
        mrr: continuity.evaluation.mistake_recurrence_rate,
        mpr: continuity.evaluation.memory_pollution_rate,
        pc: continuity.evaluation.provenance_coverage,
        cpqt: continuity.evaluation.context_pack_quality_per_token,
        cgd: continuity.evaluation.resume_accuracy_score
            - isolated.evaluation.resume_accuracy_score,
        pvl_ms,
        ris,
        dwr: continuity.evaluation.duplicate_work_rate,
        ras: continuity.evaluation.resume_accuracy_score,
    };
    sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    let process = sys.process(pid);
    let resource = ResourceEnvelope {
        elapsed_ms: start.elapsed().as_millis(),
        process_memory_bytes: process.map(|p| p.memory()).unwrap_or_default(),
        process_virtual_memory_bytes: process.map(|p| p.virtual_memory()).unwrap_or_default(),
        process_cpu_percent: process.map(|p| p.cpu_usage()).unwrap_or_default(),
        storage_bytes: storage_bytes(&class_root)?,
        gpu_report: gpu_report(config),
    };
    let hypothesis_injection = if !hypotheses.is_empty() {
        match run_phase3_injection(
            &class_root,
            class,
            &context.id,
            &scenario,
            &small_b,
            config,
            hypotheses,
        ) {
            Ok(result) => Some(result),
            Err(_) => None,
        }
    } else {
        None
    };

    let report = BenchmarkClassReport {
        class,
        scenario_id: scenario.id.clone(),
        continuity,
        baselines,
        metrics,
        resource,
        artifacts: vec![full_transcript.model, config.small_model.clone()],
        hypothesis_injection,
    };
    let report_path = class_root.join("report.json");
    std::fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub(crate) fn populate_scenario(
    kernel: &SharedContinuityKernel,
    context_id: &str,
    scenario: &Scenario,
) -> Result<()> {
    let mut counter = 0usize;
    for phase in &scenario.phases {
        counter += 1;
        let manifests = kernel.write_events(vec![crate::continuity::WriteEventInput {
            context_id: Some(context_id.to_string()),
            event: EventInput {
                kind: phase.kind.clone(),
                agent_id: phase.actor_id.clone(),
                agent_role: Some(phase.actor_role.clone()),
                session_id: format!("session-{}", scenario.id),
                task_id: Some(scenario.task_id.clone()),
                project_id: Some(scenario.namespace.clone()),
                goal_id: Some(scenario.objective.clone()),
                run_id: Some(format!("run-{}", scenario.id)),
                namespace: Some(scenario.namespace.clone()),
                environment: Some("local".into()),
                source: "benchmark".into(),
                scope: phase.scope.clone(),
                tags: vec![scenario.id.clone(), phase.actor_role.clone()],
                dimensions: phase.dimensions.clone(),
                content: phase.content.clone(),
                attributes: serde_json::json!({"phase": counter}),
            },
        }])?;
        let event_id = manifests[0].event.id.clone();
        for mark in &phase.marks {
            kernel.write_derivations(vec![ContinuityItemInput {
                context_id: context_id.to_string(),
                author_agent_id: phase.actor_id.clone(),
                kind: mark.kind,
                title: mark.title.clone(),
                body: mark.body.clone(),
                scope: Scope::Project,
                status: Some(mark.status),
                importance: Some(0.92),
                confidence: Some(0.88),
                salience: Some(0.94),
                layer: None,
                supports: vec![crate::continuity::SupportRef {
                    support_type: "event".into(),
                    support_id: event_id.clone(),
                    reason: Some("scenario-phase".into()),
                    weight: 1.0,
                }],
                dimensions: mark.dimensions.clone(),
                extra: serde_json::json!({}),
            }])?;
        }
    }
    Ok(())
}

pub(crate) fn analyze_and_write_back(
    class_root: &Path,
    class: BenchmarkClass,
    kernel: &SharedContinuityKernel,
    context_id: &str,
    scenario: &Scenario,
    adapter: &impl AgentAdapter,
    provider: BaselineKind,
    config: &ContinuityBenchConfig,
) -> Result<BaselineRunReport> {
    let context = run_baseline_inner(
        class_root, class, provider, context_id, scenario, adapter, config,
    )?;
    if context.status == BaselineStatus::Ok {
        write_model_output(
            kernel,
            context_id,
            adapter.config().agent_id.as_str(),
            &context.output,
            &context.envelope,
        )?;
    }
    Ok(BaselineRunReport {
        baseline: provider,
        status: context.status,
        model: adapter.config().model.clone(),
        retrieval_ms: context.envelope.retrieval_ms,
        model_metrics: context.model_metrics,
        envelope_tokens: context.envelope.token_estimate,
        evaluation: context.evaluation,
        survival: context.survival,
        failure: context.failure,
        artifacts: context.artifacts,
        continuity_path: context.continuity_path,
    })
}

fn run_baseline(
    class_root: &Path,
    class: BenchmarkClass,
    baseline: BaselineKind,
    context_id: &str,
    scenario: &Scenario,
    adapter: &impl AgentAdapter,
    config: &ContinuityBenchConfig,
) -> Result<BaselineRunReport> {
    let run = if matches!(class, BenchmarkClass::CrashRecovery)
        && baseline == BaselineKind::SharedContinuity
    {
        let reopened = SharedContinuityKernel::open(class_root)?;
        run_baseline_inner(
            class_root, class, baseline, context_id, scenario, adapter, config,
        )?
        .with_kernel(reopened)
    } else {
        run_baseline_inner(
            class_root, class, baseline, context_id, scenario, adapter, config,
        )?
    };
    Ok(BaselineRunReport {
        baseline,
        status: run.status,
        model: adapter.config().model.clone(),
        retrieval_ms: run.envelope.retrieval_ms,
        model_metrics: run.model_metrics,
        envelope_tokens: run.envelope.token_estimate,
        evaluation: run.evaluation,
        survival: run.survival,
        failure: run.failure,
        artifacts: run.artifacts,
        continuity_path: run.continuity_path,
    })
}

struct BaselineExecution {
    envelope: ContextEnvelope,
    output: AgentContinuationOutput,
    model_metrics: ModelCallMetrics,
    evaluation: Evaluation,
    survival: Option<SurvivalReport>,
    status: BaselineStatus,
    failure: Option<String>,
    artifacts: Vec<String>,
    continuity_path: Option<ContinuityPathReport>,
}

impl BaselineExecution {
    fn with_kernel(self, _kernel: SharedContinuityKernel) -> Self {
        self
    }
}

fn run_baseline_inner(
    class_root: &Path,
    class: BenchmarkClass,
    baseline: BaselineKind,
    context_id: &str,
    scenario: &Scenario,
    adapter: &impl AgentAdapter,
    config: &ContinuityBenchConfig,
) -> Result<BaselineExecution> {
    let kernel = SharedContinuityKernel::open(class_root)?;
    let (envelope, continuity_path) = build_context_envelope(
        &kernel,
        class,
        baseline,
        context_id,
        scenario,
        adapter.config().agent_id.as_str(),
        config.token_budget,
        config.candidate_limit,
        config.recent_window,
    )?;
    let (output, model_metrics, evaluation, survival, status, failure) =
        match adapter.analyze(&scenario.objective, &envelope.text) {
            Ok((output, model_metrics)) => {
                let repaired = repair_output_from_envelope(output, &envelope);
                let evaluation = evaluate_output(&repaired, &scenario.truth, &envelope);
                let survival = benchmark_survival_analysis(&repaired, &scenario.truth, &envelope);
                (
                    repaired,
                    model_metrics,
                    evaluation,
                    Some(survival),
                    BaselineStatus::Ok,
                    None,
                )
            }
            Err(error) => (
                AgentContinuationOutput::default(),
                ModelCallMetrics::default(),
                failed_evaluation(&envelope),
                None,
                BaselineStatus::Failed,
                Some(error.to_string()),
            ),
        };
    let artifacts =
        write_debug_artifacts(class_root, baseline, &envelope, &output, failure.as_deref())?;
    Ok(BaselineExecution {
        envelope,
        output,
        model_metrics,
        evaluation,
        survival,
        status,
        failure,
        artifacts,
        continuity_path,
    })
}

/// Run Phase 3 A/B comparison: control (no hypotheses) vs treatment (with hypotheses).
///
/// Both arms use the same context envelope for fair comparison.
fn run_phase3_injection(
    class_root: &Path,
    class: BenchmarkClass,
    context_id: &str,
    scenario: &Scenario,
    adapter: &impl AgentAdapter,
    config: &ContinuityBenchConfig,
    hypotheses: &[SurvivalHypothesis],
) -> Result<Phase3ClassResult> {
    let kernel = SharedContinuityKernel::open(class_root)?;
    let (envelope, _continuity_path) = build_context_envelope(
        &kernel,
        class,
        BaselineKind::SharedContinuity,
        context_id,
        scenario,
        adapter.config().agent_id.as_str(),
        config.token_budget,
        config.candidate_limit,
        config.recent_window,
    )?;

    // Control arm: standard prompt (no hypotheses)
    let control_result = adapter.analyze(&scenario.objective, &envelope.text);
    let (control_survival, control_eval) = match control_result {
        Ok((output, _metrics)) => {
            let repaired = repair_output_from_envelope(output, &envelope);
            let eval = evaluate_output(&repaired, &scenario.truth, &envelope);
            let survival = benchmark_survival_analysis(&repaired, &scenario.truth, &envelope);
            (Some(survival), eval)
        }
        Err(_) => (None, failed_evaluation(&envelope)),
    };

    // Treatment arm: prompt with hypothesis injection
    let treatment_result =
        adapter.analyze_with_hypotheses(&scenario.objective, &envelope.text, hypotheses);
    let (treatment_survival, treatment_eval) = match treatment_result {
        Ok((output, _metrics)) => {
            let repaired = repair_output_from_envelope(output, &envelope);
            let eval = evaluate_output(&repaired, &scenario.truth, &envelope);
            let survival = benchmark_survival_analysis(&repaired, &scenario.truth, &envelope);
            (Some(survival), eval)
        }
        Err(_) => (None, failed_evaluation(&envelope)),
    };

    let delta_ras = treatment_eval.resume_accuracy_score - control_eval.resume_accuracy_score;
    let delta_cfsr =
        treatment_eval.critical_fact_survival_rate - control_eval.critical_fact_survival_rate;
    let delta_csr = treatment_eval.constraint_survival_rate - control_eval.constraint_survival_rate;
    let delta_osr =
        treatment_eval.operational_scar_retention - control_eval.operational_scar_retention;

    Ok(Phase3ClassResult {
        class,
        control_survival,
        treatment_survival,
        hypotheses: hypotheses.to_vec(),
        delta_ras,
        delta_cfsr,
        delta_csr,
        delta_osr,
    })
}

/// Aggregate per-class Phase 3 results into a suite-level validation report.
pub fn generate_phase3_report(
    class_results: &[Phase3ClassResult],
    hypotheses: &[SurvivalHypothesis],
    cycle: usize,
) -> Phase3ValidationReport {
    let mut results = Vec::new();

    for hypothesis in hypotheses {
        let category = match hypothesis.category.as_str() {
            "CriticalFact" => TruthCategory::CriticalFact,
            "Constraint" => TruthCategory::Constraint,
            "Decision" => TruthCategory::Decision,
            "OperationalScar" => TruthCategory::OperationalScar,
            _ => continue,
        };

        let mut treatment_reports = Vec::new();
        let mut control_reports = Vec::new();
        for cr in class_results {
            if let Some(ref ts) = cr.treatment_survival {
                treatment_reports.push((cr.class, ts.clone()));
            }
            if let Some(ref cs) = cr.control_survival {
                control_reports.push((cr.class, cs.clone()));
            }
        }

        let mut positive_classes = 0usize;
        let mut total_classes = 0usize;
        let mut total_control_rate = 0.0f64;
        let mut total_treatment_rate = 0.0f64;

        for (class, treatment_survival) in &treatment_reports {
            let control_survival = control_reports
                .iter()
                .find(|(c, _)| c == class)
                .map(|(_, s)| s);
            let Some(control_survival) = control_survival else {
                continue;
            };
            let treatment_rate = category_rate(treatment_survival, category);
            let control_rate = category_rate(control_survival, category);
            if treatment_rate.is_nan() || control_rate.is_nan() {
                continue;
            }
            total_classes += 1;
            total_control_rate += control_rate;
            total_treatment_rate += treatment_rate;
            if treatment_rate > control_rate {
                positive_classes += 1;
            }
        }

        let avg_control = if total_classes > 0 {
            total_control_rate / total_classes as f64
        } else {
            0.0
        };
        let avg_treatment = if total_classes > 0 {
            total_treatment_rate / total_classes as f64
        } else {
            0.0
        };
        let improvement = avg_treatment - avg_control;
        let promoted =
            improvement > 0.05 && total_classes >= 3 && positive_classes * 2 > total_classes;
        let rejected = total_classes >= 3 && improvement <= 0.0;

        results.push(HypothesisValidationResult {
            feature_name: hypothesis.feature_name.clone(),
            category,
            direction: if hypothesis.direction.contains("Survived") {
                LessonDirection::SurvivedMore
            } else {
                LessonDirection::LostMore
            },
            control_survival_rate: avg_control,
            treatment_survival_rate: avg_treatment,
            absolute_improvement: improvement,
            positive_classes,
            total_classes,
            promoted,
            rejected,
        });
    }

    let promoted_count = results.iter().filter(|r| r.promoted).count();
    let rejected_count = results.iter().filter(|r| r.rejected).count();
    let converged = cycle >= 5 && promoted_count == 0;

    Phase3ValidationReport {
        generated_at: Utc::now().to_rfc3339(),
        cycle,
        hypotheses_tested: results.len(),
        results,
        promoted_count,
        rejected_count,
        converged,
    }
}

pub(crate) fn build_context_envelope(
    kernel: &SharedContinuityKernel,
    class: BenchmarkClass,
    baseline: BaselineKind,
    context_id: &str,
    scenario: &Scenario,
    receiver_agent_id: &str,
    token_budget: usize,
    candidate_limit: usize,
    recent_window: usize,
) -> Result<(ContextEnvelope, Option<ContinuityPathReport>)> {
    let started = Instant::now();
    match baseline {
        BaselineKind::SharedContinuity => {
            if class.uses_handoff_proof() {
                let handoff = kernel.handoff(ContinuityHandoffInput {
                    from_agent_id: "benchmark-planner".into(),
                    to_agent_id: receiver_agent_id.to_string(),
                    context_id: Some(context_id.to_string()),
                    namespace: None,
                    task_id: None,
                    objective: scenario.objective.clone(),
                    reason: format!("benchmark handoff for {}", class.slug()),
                    selector: None,
                    resolution: SnapshotResolution::Medium,
                    token_budget,
                    candidate_limit,
                })?;
                Ok((
                    render_continuity_envelope(
                        handoff.context,
                        Some(&handoff.proof),
                        started.elapsed().as_millis(),
                    ),
                    Some(ContinuityPathReport {
                        kind: ContinuityPathKind::HandoffProof,
                        role: class.shared_path_role(),
                        proof_register_count: handoff.proof.registers.len(),
                        proof_register_labels: handoff
                            .proof
                            .registers
                            .iter()
                            .map(|item| item.label.clone())
                            .collect(),
                    }),
                ))
            } else {
                let read = kernel.read_context(ReadContextInput {
                    context_id: Some(context_id.to_string()),
                    namespace: None,
                    task_id: None,
                    objective: scenario.objective.clone(),
                    token_budget,
                    selector: None,
                    agent_id: Some(receiver_agent_id.to_string()),
                    session_id: None,
                    view_id: None,
                    include_resolved: false,
                    candidate_limit,
                })?;
                Ok((
                    render_continuity_envelope(read, None, started.elapsed().as_millis()),
                    Some(ContinuityPathReport {
                        kind: ContinuityPathKind::ReadContextOnly,
                        role: class.shared_path_role(),
                        proof_register_count: 0,
                        proof_register_labels: Vec::new(),
                    }),
                ))
            }
        }
        BaselineKind::Isolated => Ok((
            ContextEnvelope {
                provider: baseline,
                retrieval_ms: started.elapsed().as_millis(),
                text: format!(
                    "Objective: {}\nNo shared continuity is available. Resume from scratch.",
                    scenario.objective
                ),
                token_estimate: estimate_tokens(&scenario.objective),
                surfaced: Vec::new(),
            },
            None,
        )),
        BaselineKind::RecentWindow => {
            let rows = kernel.replay(Some(&format!("session-{}", scenario.id)), recent_window)?;
            Ok((
                render_event_envelope(baseline, rows, started.elapsed().as_millis()),
                None,
            ))
        }
        BaselineKind::VectorOnly => {
            let memories = kernel.vector_baseline(
                &scenario.objective,
                Some(&format!("session-{}", scenario.id)),
                Some(&scenario.task_id),
                None,
                candidate_limit,
            )?;
            Ok((
                render_memory_envelope(
                    baseline,
                    memories
                        .into_iter()
                        .map(|memory| (memory.id, memory.body, true))
                        .collect(),
                    started.elapsed().as_millis(),
                ),
                None,
            ))
        }
        BaselineKind::RollingSummary => {
            let memories = kernel.summary_baseline(
                Some(&format!("session-{}", scenario.id)),
                Some(&scenario.task_id),
                None,
                candidate_limit,
            )?;
            Ok((
                render_memory_envelope(
                    baseline,
                    memories
                        .into_iter()
                        .map(|memory| (memory.id, memory.body, true))
                        .collect(),
                    started.elapsed().as_millis(),
                ),
                None,
            ))
        }
        BaselineKind::FullTranscript => {
            let selector = Selector {
                all: vec![crate::model::DimensionFilter {
                    key: "task".into(),
                    values: vec![scenario.task_id.clone()],
                }],
                any: Vec::new(),
                exclude: Vec::new(),
                layers: Vec::new(),
                start_ts: None,
                end_ts: None,
                limit: Some(candidate_limit.saturating_mul(3)),
                namespace: Some(scenario.namespace.clone()),
            };
            let rows = kernel.replay_selector(selector, candidate_limit.saturating_mul(3))?;
            let mut surfaced = Vec::new();
            let mut lines = Vec::new();
            for (idx, row) in rows.into_iter().enumerate() {
                let label = format!("e{}", idx + 1);
                surfaced.push(SurfacedItem {
                    label: label.clone(),
                    support_type: "event".into(),
                    support_id: row.event.id.clone(),
                    text: row.event.input.content.clone(),
                    has_provenance: true,
                });
                lines.push(format!(
                    "[{label}][{}][{}] {}",
                    row.event.input.kind,
                    row.event.input.agent_id,
                    trim_text(&row.event.input.content, 280)
                ));
            }
            let text = format!("Objective: {}\n{}", scenario.objective, lines.join("\n"));
            Ok((
                ContextEnvelope {
                    provider: baseline,
                    retrieval_ms: started.elapsed().as_millis(),
                    token_estimate: estimate_tokens(&text),
                    text,
                    surfaced,
                },
                None,
            ))
        }
    }
}

fn render_continuity_envelope(
    read: ContextRead,
    proof: Option<&HandoffProof>,
    retrieval_ms: u128,
) -> ContextEnvelope {
    let mut surfaced = Vec::new();
    let mut lines = vec![
        format!("Objective: {}", read.objective),
        format!(
            "Context: {} / {}",
            read.context.namespace, read.context.task_id
        ),
    ];
    let mut core_index = 0usize;
    let push_core = |label: String,
                     support_id: String,
                     text: String,
                     has_provenance: bool,
                     lines: &mut Vec<String>,
                     surfaced: &mut Vec<SurfacedItem>| {
        surfaced.push(SurfacedItem {
            label: label.clone(),
            support_type: "continuity".into(),
            support_id,
            text: text.clone(),
            has_provenance,
        });
        lines.push(format!("[{label}][resumption_fact] {}", text.trim()));
    };
    lines.push("Section: resumption_core".into());
    core_index += 1;
    push_core(
        format!("f{}", core_index),
        read.context.id.clone(),
        format!(
            "Primary context is {} / {} for this resume.",
            read.context.namespace, read.context.task_id
        ),
        true,
        &mut lines,
        &mut surfaced,
    );
    if let Some(incident) = read.incidents.first() {
        core_index += 1;
        push_core(
            format!("f{}", core_index),
            incident.id.clone(),
            format!("{} :: {}", incident.title, trim_text(&incident.body, 220)),
            !incident.supports.is_empty(),
            &mut lines,
            &mut surfaced,
        );
    }
    if let Some(decision) = read.decisions.first() {
        core_index += 1;
        push_core(
            format!("f{}", core_index),
            decision.id.clone(),
            format!("{} :: {}", decision.title, trim_text(&decision.body, 220)),
            !decision.supports.is_empty(),
            &mut lines,
            &mut surfaced,
        );
    }
    if !read.recall.summary.trim().is_empty() {
        lines.push(format!("Recall summary: {}", read.recall.summary.trim()));
    }
    if let Some(answer_hint) = read.recall.answer_hint.as_deref() {
        let label = "a1".to_string();
        surfaced.push(SurfacedItem {
            label: label.clone(),
            support_type: "continuity".into(),
            support_id: read
                .recall
                .items
                .first()
                .map(|item| item.id.clone())
                .unwrap_or_else(|| "continuity-answer-hint".into()),
            text: answer_hint.to_string(),
            has_provenance: read
                .recall
                .items
                .first()
                .map(|item| item.support_count > 0)
                .unwrap_or(false),
        });
        lines.push(format!("[{label}][answer_hint] {}", answer_hint.trim()));
    }
    if !read.recall.items.is_empty() {
        lines.push("Section: continuity_recall".into());
        for (idx, item) in read.recall.items.iter().take(3).enumerate() {
            let label = format!("r{}", idx + 1);
            surfaced.push(SurfacedItem {
                label: label.clone(),
                support_type: "continuity".into(),
                support_id: item.id.clone(),
                text: item.preview.clone(),
                has_provenance: item.support_count > 0,
            });
            lines.push(format!(
                "[{label}][recall][{}][{}] {} :: {}",
                item.kind.as_str(),
                item.status.as_str(),
                item.title,
                trim_text(&item.preview, 220)
            ));
        }
    }
    if let Some(proof) = proof {
        push_proof_section(proof, &mut lines, &mut surfaced);
    }
    push_section("decisions", &read.decisions, &mut lines, &mut surfaced);
    push_section("constraints", &read.constraints, &mut lines, &mut surfaced);
    push_section("hypotheses", &read.hypotheses, &mut lines, &mut surfaced);
    push_section("incidents", &read.incidents, &mut lines, &mut surfaced);
    push_section(
        "operational_scars",
        &read.operational_scars,
        &mut lines,
        &mut surfaced,
    );
    push_section(
        "open_threads",
        &read.open_threads,
        &mut lines,
        &mut surfaced,
    );
    for (idx, item) in read.pack.items.iter().enumerate() {
        let label = format!("p{}", idx + 1);
        surfaced.push(SurfacedItem {
            label: label.clone(),
            support_type: "memory".into(),
            support_id: item.memory_id.clone(),
            text: item.body.clone(),
            has_provenance: !item.provenance.is_null(),
        });
        lines.push(format!(
            "[{label}][pack][{}] {}",
            item.layer,
            trim_text(&item.body, 280)
        ));
    }
    let text = lines.join("\n");
    ContextEnvelope {
        provider: BaselineKind::SharedContinuity,
        retrieval_ms,
        token_estimate: estimate_tokens(&text),
        text,
        surfaced,
    }
}

fn push_proof_section(
    proof: &HandoffProof,
    lines: &mut Vec<String>,
    surfaced: &mut Vec<SurfacedItem>,
) {
    if proof.registers.is_empty() {
        return;
    }
    lines.push("Section: handoff_proof".into());
    lines.push(format!("Proof digest: {}", proof.digest));
    for item in &proof.registers {
        surfaced.push(SurfacedItem {
            label: item.label.clone(),
            support_type: "continuity".into(),
            support_id: item.source_id.clone(),
            text: item.body.clone(),
            has_provenance: item.has_provenance,
        });
        lines.push(format!(
            "[{}][handoff_proof][{}] {} :: {}",
            item.label,
            item.register_kind,
            item.title,
            trim_text(&item.body, 220)
        ));
    }
}

fn push_section(
    name: &str,
    items: &[crate::continuity::ContinuityItemRecord],
    lines: &mut Vec<String>,
    surfaced: &mut Vec<SurfacedItem>,
) {
    if items.is_empty() {
        return;
    }
    lines.push(format!("Section: {name}"));
    for (idx, item) in items.iter().take(6).enumerate() {
        let label = format!("{}{}", section_prefix(name), idx + 1);
        surfaced.push(SurfacedItem {
            label: label.clone(),
            support_type: "continuity".into(),
            support_id: item.id.clone(),
            text: item.body.clone(),
            has_provenance: !item.supports.is_empty(),
        });
        lines.push(format!(
            "[{label}][{}][{}] {} :: {}",
            item.kind.as_str(),
            item.status.as_str(),
            item.title,
            trim_text(&item.body, 220)
        ));
    }
}

fn section_prefix(name: &str) -> &'static str {
    match name {
        "decisions" => "d",
        "constraints" => "k",
        "hypotheses" => "h",
        "incidents" => "i",
        "operational_scars" => "s",
        "open_threads" => "t",
        _ => "x",
    }
}

fn render_memory_envelope(
    provider: BaselineKind,
    items: Vec<(String, String, bool)>,
    retrieval_ms: u128,
) -> ContextEnvelope {
    let mut surfaced = Vec::new();
    let mut lines = Vec::new();
    for (idx, (id, body, provenance)) in items.into_iter().enumerate() {
        let label = format!("m{}", idx + 1);
        surfaced.push(SurfacedItem {
            label: label.clone(),
            support_type: "memory".into(),
            support_id: id,
            text: body.clone(),
            has_provenance: provenance,
        });
        lines.push(format!("[{label}][memory] {}", trim_text(&body, 260)));
    }
    let text = lines.join("\n");
    ContextEnvelope {
        provider,
        retrieval_ms,
        token_estimate: estimate_tokens(&text),
        text,
        surfaced,
    }
}

fn render_event_envelope(
    provider: BaselineKind,
    rows: Vec<crate::model::ReplayRow>,
    retrieval_ms: u128,
) -> ContextEnvelope {
    let mut surfaced = Vec::new();
    let mut lines = Vec::new();
    for (idx, row) in rows.into_iter().enumerate() {
        let label = format!("e{}", idx + 1);
        surfaced.push(SurfacedItem {
            label: label.clone(),
            support_type: "event".into(),
            support_id: row.event.id.clone(),
            text: row.event.input.content.clone(),
            has_provenance: true,
        });
        lines.push(format!(
            "[{label}][{}][{}] {}",
            row.event.input.kind,
            row.event.input.agent_id,
            trim_text(&row.event.input.content, 240)
        ));
    }
    let text = lines.join("\n");
    ContextEnvelope {
        provider,
        retrieval_ms,
        token_estimate: estimate_tokens(&text),
        text,
        surfaced,
    }
}

pub(crate) fn evaluate_output(
    output: &AgentContinuationOutput,
    truth: &GroundTruth,
    envelope: &ContextEnvelope,
) -> Evaluation {
    let evidence_labels = envelope
        .surfaced
        .iter()
        .map(|item| item.label.clone())
        .collect::<HashSet<_>>();
    let critical = score_notes(&output.critical_facts, &truth.critical_facts);
    let constraints = score_notes(&output.constraints, &truth.constraints);
    let decisions = score_decisions(&output.decisions, &truth.decisions, &evidence_labels);
    let scars = score_notes(&output.operational_scars, &truth.scars);
    let next_step = if match_keywords(&output.next_step.text, &truth.next_step_keywords) {
        1.0
    } else {
        0.0
    };
    let unsupported = unsupported_items(output, truth, &evidence_labels);
    let total_items = count_items(output).max(1);
    let provenance = envelope
        .surfaced
        .iter()
        .filter(|item| item.has_provenance)
        .count() as f64
        / envelope.surfaced.len().max(1) as f64;
    let resume_accuracy =
        (critical.rate + constraints.rate + decisions.rate + scars.rate + next_step) / 5.0;
    let duplicate_work = if truth
        .decisions
        .iter()
        .any(|item| match_keywords(&output.next_step.text, &item.keywords))
    {
        1.0
    } else {
        0.0
    };
    Evaluation {
        critical_fact_survival_rate: critical.rate,
        constraint_survival_rate: constraints.rate,
        decision_lineage_fidelity: decisions.rate,
        operational_scar_retention: scars.rate,
        mistake_recurrence_rate: if any_match(&output.avoid_repeating, &truth.avoid_repeating) {
            0.0
        } else if match_keywords(
            &output.next_step.text,
            &flatten_keywords(&truth.avoid_repeating),
        ) {
            1.0
        } else {
            0.0
        },
        memory_pollution_rate: unsupported as f64 / total_items as f64,
        provenance_coverage: provenance,
        context_pack_quality_per_token: resume_accuracy / envelope.token_estimate.max(1) as f64,
        resume_accuracy_score: resume_accuracy,
        duplicate_work_rate: duplicate_work,
        matched_critical_facts: critical.matched,
        matched_constraints: constraints.matched,
        matched_decisions: decisions.matched,
        matched_scars: scars.matched,
        unsupported_items: unsupported,
        total_items,
    }
}

#[derive(Debug, Clone, Copy)]
struct MatchScore {
    matched: usize,
    rate: f64,
}

fn score_notes(notes: &[EvidenceNote], truth: &[TruthItem]) -> MatchScore {
    let matched = truth
        .iter()
        .filter(|item| {
            notes
                .iter()
                .any(|note| match_keywords(&note.text, &item.keywords))
        })
        .count();
    MatchScore {
        matched,
        rate: matched as f64 / truth.len().max(1) as f64,
    }
}

fn score_decisions(
    notes: &[DecisionNote],
    truth: &[TruthItem],
    evidence_labels: &HashSet<String>,
) -> MatchScore {
    let matched = truth
        .iter()
        .filter(|item| {
            notes.iter().any(|note| {
                match_keywords(&note.text, &item.keywords)
                    && (item.rationale_keywords.is_empty()
                        || match_keywords(&note.rationale, &item.rationale_keywords))
                    && note.evidence.iter().any(|id| evidence_labels.contains(id))
            })
        })
        .count();
    MatchScore {
        matched,
        rate: matched as f64 / truth.len().max(1) as f64,
    }
}

fn any_match(notes: &[EvidenceNote], truth: &[TruthItem]) -> bool {
    truth.iter().any(|item| {
        notes
            .iter()
            .any(|note| match_keywords(&note.text, &item.keywords))
    })
}

fn unsupported_items(
    output: &AgentContinuationOutput,
    truth: &GroundTruth,
    evidence_labels: &HashSet<String>,
) -> usize {
    let mut unsupported = 0usize;
    for note in &output.critical_facts {
        if !truth
            .critical_facts
            .iter()
            .any(|item| match_keywords(&note.text, &item.keywords))
        {
            unsupported += 1;
        }
    }
    for note in &output.constraints {
        if !truth
            .constraints
            .iter()
            .any(|item| match_keywords(&note.text, &item.keywords))
        {
            unsupported += 1;
        }
    }
    for note in &output.decisions {
        if !truth
            .decisions
            .iter()
            .any(|item| match_keywords(&note.text, &item.keywords))
            || !note.evidence.iter().any(|id| evidence_labels.contains(id))
        {
            unsupported += 1;
        }
    }
    unsupported
}

fn count_items(output: &AgentContinuationOutput) -> usize {
    output.critical_facts.len()
        + output.constraints.len()
        + output.decisions.len()
        + output.open_hypotheses.len()
        + output.operational_scars.len()
        + output.avoid_repeating.len()
        + usize::from(!output.next_step.text.trim().is_empty())
}

// ---------------------------------------------------------------------------
// Survival analysis — Phase 1 of metacognitive continuity
// ---------------------------------------------------------------------------

/// Category of a ground truth item in the survival analysis.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TruthCategory {
    CriticalFact,
    Constraint,
    Decision,
    OperationalScar,
}

/// Whether a ground truth item survived or was lost during a benchmark run.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SurvivalOutcome {
    Survived,
    Lost,
}

/// Textual features extracted from the ground truth keywords and matching note.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemFeatures {
    /// Number of keywords in the ground truth item.
    pub keyword_count: usize,
    /// Total character length of all keywords.
    pub keyword_char_length: usize,
    /// Whether any keyword looks like a file path (contains `/` or `.rs`, `.py`, etc.).
    pub contains_file_path: bool,
    /// Whether any keyword looks like an error code or numeric reference.
    pub contains_numeric_ref: bool,
    /// The text of the note that matched, if any.
    pub matched_note_text: Option<String>,
    /// Approximate token count of the matched note (whitespace split).
    pub matched_note_tokens: Option<usize>,
    /// Whether the matched note uses prohibition framing ("do not", "never", "avoid").
    pub prohibition_framing: Option<bool>,
    /// Whether the matched note uses aspiration framing ("try to", "prefer", "consider").
    pub aspiration_framing: Option<bool>,
    /// Number of evidence IDs on the matching note.
    pub evidence_count: Option<usize>,
}

/// Per-item survival record for a single ground truth item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalRecord {
    pub category: TruthCategory,
    pub outcome: SurvivalOutcome,
    pub keywords: Vec<String>,
    pub features: ItemFeatures,
}

/// Aggregate survival statistics for a single category.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CategoryStats {
    pub total: usize,
    pub survived: usize,
    pub lost: usize,
    pub rate: f64,
    /// Average keyword count for survived items.
    pub avg_keyword_count_survived: f64,
    /// Average keyword count for lost items.
    pub avg_keyword_count_lost: f64,
    /// Fraction of survived items containing file paths.
    pub file_path_rate_survived: f64,
    /// Fraction of lost items containing file paths.
    pub file_path_rate_lost: f64,
    /// Fraction of survived items using prohibition framing.
    pub prohibition_rate_survived: f64,
    /// Fraction of survived items using aspiration framing.
    pub aspiration_rate_survived: f64,
}

/// Full survival analysis report for a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalReport {
    pub records: Vec<SurvivalRecord>,
    pub facts: CategoryStats,
    pub constraints: CategoryStats,
    pub decisions: CategoryStats,
    pub scars: CategoryStats,
    pub surfaced_item_count: usize,
    pub surfaced_with_provenance: usize,
    pub total_envelope_tokens: usize,
}

// ---------------------------------------------------------------------------
// Meta-lesson types (Phase 2 metacognitive)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LessonDirection {
    SurvivedMore,
    LostMore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLessonEvidence {
    pub survived_with_feature: usize,
    pub lost_with_feature: usize,
    pub survived_without_feature: usize,
    pub lost_without_feature: usize,
    pub rate_with_feature: f64,
    pub rate_without_feature: f64,
    pub chi_squared: f64,
    /// Raw p-value from chi-squared test.
    pub p_value: f64,
    /// Benjamini-Hochberg adjusted p-value controlling false discovery rate.
    pub adjusted_p_value: f64,
    /// Whether any cell in the 2x2 table has expected count < 5
    /// (chi-squared unreliable; Fisher's exact test would be more appropriate).
    pub sparse_cells: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLesson {
    pub pattern: String,
    pub feature_name: String,
    pub category: TruthCategory,
    pub direction: LessonDirection,
    pub evidence: MetaLessonEvidence,
    pub confidence: f64,
    /// Number of survival records in the category (not independent benchmark runs).
    pub sample_size: usize,
    /// Number of distinct benchmark classes contributing records.
    pub benchmark_classes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLessonReport {
    pub generated_at: String,
    pub total_records: usize,
    pub lessons: Vec<MetaLesson>,
    /// Number of candidate hypotheses tested before FDR correction.
    pub candidates_tested: usize,
}

// ---------------------------------------------------------------------------
// Phase 3: Closed-loop hypothesis validation types
// ---------------------------------------------------------------------------

/// Distinguishes treatment (with hypothesis injection) from control arms.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Phase3Arm {
    Treatment,
    Control,
}

/// Per-hypothesis validation result comparing treatment vs control survival.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisValidationResult {
    pub feature_name: String,
    pub category: TruthCategory,
    pub direction: LessonDirection,
    pub control_survival_rate: f64,
    pub treatment_survival_rate: f64,
    pub absolute_improvement: f64,
    pub positive_classes: usize,
    pub total_classes: usize,
    pub promoted: bool,
    pub rejected: bool,
}

/// Full Phase 3 validation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase3ValidationReport {
    pub generated_at: String,
    pub cycle: usize,
    pub hypotheses_tested: usize,
    pub results: Vec<HypothesisValidationResult>,
    pub promoted_count: usize,
    pub rejected_count: usize,
    pub converged: bool,
}

/// Per-class Phase 3 injection result: control vs treatment on same envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase3ClassResult {
    pub class: BenchmarkClass,
    pub control_survival: Option<SurvivalReport>,
    pub treatment_survival: Option<SurvivalReport>,
    pub hypotheses: Vec<SurvivalHypothesis>,
    pub delta_ras: f64,
    pub delta_cfsr: f64,
    pub delta_csr: f64,
    pub delta_osr: f64,
}

/// Extract eligible survival hypotheses from a `MetaLessonReport`.
///
/// Only hypotheses with `sparse_cells: false` and `adjusted_p_value < 0.05`
/// are eligible for prompt injection.
pub fn extract_eligible_hypotheses(report: &MetaLessonReport) -> Vec<SurvivalHypothesis> {
    report
        .lessons
        .iter()
        .filter(|lesson| !lesson.evidence.sparse_cells && lesson.evidence.adjusted_p_value < 0.05)
        .map(|lesson| {
            let hint = format!(
                "{} (survival rate {:.0}% with vs {:.0}% without, p={:.4})",
                lesson.pattern,
                lesson.evidence.rate_with_feature * 100.0,
                lesson.evidence.rate_without_feature * 100.0,
                lesson.evidence.adjusted_p_value,
            );
            SurvivalHypothesis {
                feature_name: lesson.feature_name.clone(),
                category: format!("{:?}", lesson.category),
                direction: format!("{:?}", lesson.direction),
                hint,
            }
        })
        .collect()
}

fn category_rate(survival: &SurvivalReport, category: TruthCategory) -> f64 {
    match category {
        TruthCategory::CriticalFact => survival.facts.rate,
        TruthCategory::Constraint => survival.constraints.rate,
        TruthCategory::Decision => survival.decisions.rate,
        TruthCategory::OperationalScar => survival.scars.rate,
    }
}

/// Run survival analysis comparing model output against ground truth.
fn benchmark_survival_analysis(
    output: &AgentContinuationOutput,
    truth: &GroundTruth,
    envelope: &ContextEnvelope,
) -> SurvivalReport {
    let evidence_labels: HashSet<String> =
        envelope.surfaced.iter().map(|s| s.label.clone()).collect();

    let mut records = Vec::new();

    // Critical facts
    for item in &truth.critical_facts {
        let matched_note = output
            .critical_facts
            .iter()
            .find(|note| match_keywords(&note.text, &item.keywords));
        records.push(survival_record(
            TruthCategory::CriticalFact,
            item,
            matched_note.map(|n| (n.text.as_str(), n.evidence.len())),
        ));
    }

    // Constraints
    for item in &truth.constraints {
        let matched_note = output
            .constraints
            .iter()
            .find(|note| match_keywords(&note.text, &item.keywords));
        records.push(survival_record(
            TruthCategory::Constraint,
            item,
            matched_note.map(|n| (n.text.as_str(), n.evidence.len())),
        ));
    }

    // Decisions (stricter: requires rationale + evidence)
    for item in &truth.decisions {
        let matched_note = output.decisions.iter().find(|note| {
            match_keywords(&note.text, &item.keywords)
                && (item.rationale_keywords.is_empty()
                    || match_keywords(&note.rationale, &item.rationale_keywords))
                && note.evidence.iter().any(|id| evidence_labels.contains(id))
        });
        records.push(survival_record(
            TruthCategory::Decision,
            item,
            matched_note.map(|n| (n.text.as_str(), n.evidence.len())),
        ));
    }

    // Operational scars
    for item in &truth.scars {
        let matched_note = output
            .operational_scars
            .iter()
            .find(|note| match_keywords(&note.text, &item.keywords));
        records.push(survival_record(
            TruthCategory::OperationalScar,
            item,
            matched_note.map(|n| (n.text.as_str(), n.evidence.len())),
        ));
    }

    let facts = category_stats(&records, TruthCategory::CriticalFact);
    let constraints = category_stats(&records, TruthCategory::Constraint);
    let decisions = category_stats(&records, TruthCategory::Decision);
    let scars = category_stats(&records, TruthCategory::OperationalScar);

    SurvivalReport {
        records,
        facts,
        constraints,
        decisions,
        scars,
        surfaced_item_count: envelope.surfaced.len(),
        surfaced_with_provenance: envelope
            .surfaced
            .iter()
            .filter(|s| s.has_provenance)
            .count(),
        total_envelope_tokens: envelope.token_estimate,
    }
}

fn survival_record(
    category: TruthCategory,
    truth_item: &TruthItem,
    matched: Option<(&str, usize)>,
) -> SurvivalRecord {
    let keywords: Vec<String> = truth_item.keywords.iter().map(|k| k.to_string()).collect();
    let contains_file_path = keywords.iter().any(|k| {
        k.contains('/')
            || k.ends_with(".rs")
            || k.ends_with(".py")
            || k.ends_with(".ts")
            || k.ends_with(".go")
    });
    let contains_numeric_ref = keywords
        .iter()
        .any(|k| k.chars().any(|c| c.is_ascii_digit()));

    let (outcome, matched_note_text, matched_note_tokens, prohibition, aspiration, evidence_count) =
        match matched {
            Some((text, ev_count)) => {
                let lower = text.to_lowercase();
                let prohibition = lower.contains("do not")
                    || lower.contains("don't")
                    || lower.contains("never")
                    || lower.contains("avoid")
                    || lower.contains("must not");
                let aspiration = lower.contains("try to")
                    || lower.contains("prefer")
                    || lower.contains("consider")
                    || lower.contains("should")
                    || lower.contains("ideally");
                let tokens = text.split_whitespace().count();
                (
                    SurvivalOutcome::Survived,
                    Some(text.to_string()),
                    Some(tokens),
                    Some(prohibition),
                    Some(aspiration),
                    Some(ev_count),
                )
            }
            None => (SurvivalOutcome::Lost, None, None, None, None, None),
        };

    SurvivalRecord {
        category,
        outcome,
        keywords,
        features: ItemFeatures {
            keyword_count: truth_item.keywords.len(),
            keyword_char_length: truth_item.keywords.iter().map(|k| k.len()).sum(),
            contains_file_path,
            contains_numeric_ref,
            matched_note_text,
            matched_note_tokens,
            prohibition_framing: prohibition,
            aspiration_framing: aspiration,
            evidence_count,
        },
    }
}

fn category_stats(records: &[SurvivalRecord], category: TruthCategory) -> CategoryStats {
    let items: Vec<&SurvivalRecord> = records.iter().filter(|r| r.category == category).collect();
    let total = items.len();
    if total == 0 {
        return CategoryStats::default();
    }
    let survived: Vec<&&SurvivalRecord> = items
        .iter()
        .filter(|r| r.outcome == SurvivalOutcome::Survived)
        .collect();
    let lost: Vec<&&SurvivalRecord> = items
        .iter()
        .filter(|r| r.outcome == SurvivalOutcome::Lost)
        .collect();

    let avg_kw = |group: &[&&SurvivalRecord]| -> f64 {
        if group.is_empty() {
            return 0.0;
        }
        group
            .iter()
            .map(|r| r.features.keyword_count as f64)
            .sum::<f64>()
            / group.len() as f64
    };
    let file_path_rate = |group: &[&&SurvivalRecord]| -> f64 {
        if group.is_empty() {
            return 0.0;
        }
        group
            .iter()
            .filter(|r| r.features.contains_file_path)
            .count() as f64
            / group.len() as f64
    };
    let framing_rate =
        |group: &[&&SurvivalRecord], extract: fn(&ItemFeatures) -> Option<bool>| -> f64 {
            let with_data: Vec<_> = group.iter().filter_map(|r| extract(&r.features)).collect();
            if with_data.is_empty() {
                return 0.0;
            }
            with_data.iter().filter(|&&v| v).count() as f64 / with_data.len() as f64
        };

    CategoryStats {
        total,
        survived: survived.len(),
        lost: lost.len(),
        rate: survived.len() as f64 / total as f64,
        avg_keyword_count_survived: avg_kw(&survived),
        avg_keyword_count_lost: avg_kw(&lost),
        file_path_rate_survived: file_path_rate(&survived),
        file_path_rate_lost: file_path_rate(&lost),
        prohibition_rate_survived: framing_rate(&survived, |f| f.prohibition_framing),
        aspiration_rate_survived: framing_rate(&survived, |f| f.aspiration_framing),
    }
}

// ---------------------------------------------------------------------------
// Meta-lesson generation (Phase 2 metacognitive)
// ---------------------------------------------------------------------------

fn erfc_approx(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_approx(-x);
    }
    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let t = 1.0 / (1.0 + p * x);
    let poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    poly * (-x * x).exp()
}

fn chi_squared_2x2(a: usize, b: usize, c: usize, d: usize) -> (f64, f64) {
    let n = (a + b + c + d) as f64;
    if n == 0.0 {
        return (0.0, 1.0);
    }
    let row1 = (a + b) as f64;
    let row2 = (c + d) as f64;
    let col1 = (a + c) as f64;
    let col2 = (b + d) as f64;
    if row1 == 0.0 || row2 == 0.0 || col1 == 0.0 || col2 == 0.0 {
        return (0.0, 1.0);
    }
    let numerator = n
        * ((a as f64 * d as f64 - b as f64 * c as f64).abs() - n / 2.0)
            .max(0.0)
            .powi(2);
    let chi2 = numerator / (row1 * row2 * col1 * col2);
    let p = erfc_approx((chi2 / 2.0).sqrt());
    (chi2, p)
}

struct FeatureExtractor {
    name: &'static str,
    extract: fn(&SurvivalRecord) -> Option<bool>,
    pattern_template: &'static str,
}

const FEATURE_EXTRACTORS: &[FeatureExtractor] = &[
    FeatureExtractor {
        name: "file_path",
        extract: |r| Some(r.features.contains_file_path),
        pattern_template: "Items with file paths survive at {with}% vs {without}% without",
    },
    FeatureExtractor {
        name: "numeric_ref",
        extract: |r| Some(r.features.contains_numeric_ref),
        pattern_template: "Items with numeric references survive at {with}% vs {without}% without",
    },
    FeatureExtractor {
        name: "prohibition_framing",
        extract: |r| r.features.prohibition_framing,
        pattern_template: "Prohibition-framed items survive at {with}% vs {without}% for others",
    },
    FeatureExtractor {
        name: "aspiration_framing",
        extract: |r| r.features.aspiration_framing,
        pattern_template: "Aspiration-framed items survive at {with}% vs {without}% for others",
    },
];

/// Check whether any expected cell count is < 5 (chi-squared unreliable).
fn has_sparse_cells(a: usize, b: usize, c: usize, d: usize) -> bool {
    let n = (a + b + c + d) as f64;
    if n == 0.0 {
        return true;
    }
    let row1 = (a + b) as f64;
    let row2 = (c + d) as f64;
    let col1 = (a + c) as f64;
    let col2 = (b + d) as f64;
    let expected = [
        row1 * col1 / n,
        row1 * col2 / n,
        row2 * col1 / n,
        row2 * col2 / n,
    ];
    expected.iter().any(|&e| e < 5.0)
}

/// Benjamini-Hochberg FDR correction on a set of p-values.
/// Returns adjusted p-values in the same order as input.
fn benjamini_hochberg(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }
    let mut indexed: Vec<(usize, f64)> = p_values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut adjusted = vec![0.0f64; m];
    let mut cumulative_min = f64::INFINITY;
    for (rank_rev, &(orig_idx, raw_p)) in indexed.iter().enumerate().rev() {
        let rank = rank_rev + 1; // 1-based rank from smallest p
        let corrected = raw_p * (m as f64) / (rank as f64);
        cumulative_min = cumulative_min.min(corrected).min(1.0);
        adjusted[orig_idx] = cumulative_min;
    }
    adjusted
}

struct CandidateLesson {
    pattern: String,
    feature_name: String,
    category: TruthCategory,
    direction: LessonDirection,
    survived_with: usize,
    lost_with: usize,
    survived_without: usize,
    lost_without: usize,
    rate_with: f64,
    rate_without: f64,
    chi2: f64,
    raw_p: f64,
    sparse_cells: bool,
    sample_size: usize,
    benchmark_classes: usize,
}

fn compare_feature_distributions(
    records: &[SurvivalRecord],
    fdr_threshold: f64,
    min_benchmark_classes: usize,
    benchmark_classes: usize,
) -> (Vec<MetaLesson>, usize) {
    // No hypothesis tests with fewer than 3 independent benchmark classes.
    // Individual records from the same class are not independent observations.
    if benchmark_classes < min_benchmark_classes {
        return (Vec::new(), 0);
    }

    let categories = [
        TruthCategory::CriticalFact,
        TruthCategory::Constraint,
        TruthCategory::Decision,
        TruthCategory::OperationalScar,
    ];

    let mut candidates: Vec<CandidateLesson> = Vec::new();

    for &category in &categories {
        let cat_records: Vec<_> = records.iter().filter(|r| r.category == category).collect();
        if cat_records.len() < 10 {
            continue;
        }

        for extractor in FEATURE_EXTRACTORS {
            let mut survived_with = 0usize;
            let mut lost_with = 0usize;
            let mut survived_without = 0usize;
            let mut lost_without = 0usize;

            for record in &cat_records {
                let has_feature = match (extractor.extract)(record) {
                    Some(v) => v,
                    None => continue,
                };
                let survived = record.outcome == SurvivalOutcome::Survived;
                match (has_feature, survived) {
                    (true, true) => survived_with += 1,
                    (true, false) => lost_with += 1,
                    (false, true) => survived_without += 1,
                    (false, false) => lost_without += 1,
                }
            }

            let total_with = survived_with + lost_with;
            let total_without = survived_without + lost_without;
            if total_with == 0 || total_without == 0 {
                continue;
            }

            let (chi2, raw_p) =
                chi_squared_2x2(survived_with, lost_with, survived_without, lost_without);
            let sparse = has_sparse_cells(survived_with, lost_with, survived_without, lost_without);

            let rate_with = survived_with as f64 / total_with as f64 * 100.0;
            let rate_without = survived_without as f64 / total_without as f64 * 100.0;
            let direction = if rate_with > rate_without {
                LessonDirection::SurvivedMore
            } else {
                LessonDirection::LostMore
            };

            let pattern = extractor
                .pattern_template
                .replace("{with}", &format!("{rate_with:.0}"))
                .replace("{without}", &format!("{rate_without:.0}"));

            candidates.push(CandidateLesson {
                pattern,
                feature_name: extractor.name.to_string(),
                category,
                direction,
                survived_with,
                lost_with,
                survived_without,
                lost_without,
                rate_with: rate_with / 100.0,
                rate_without: rate_without / 100.0,
                chi2,
                raw_p,
                sparse_cells: sparse,
                sample_size: cat_records.len(),
                benchmark_classes: 0, // filled by caller
            });
        }
    }

    let candidates_tested = candidates.len();
    let raw_ps: Vec<f64> = candidates.iter().map(|c| c.raw_p).collect();
    let adjusted = benjamini_hochberg(&raw_ps);

    let lessons = candidates
        .into_iter()
        .zip(adjusted)
        .filter(|(_, adj_p)| *adj_p < fdr_threshold)
        .map(|(c, adj_p)| MetaLesson {
            pattern: c.pattern,
            feature_name: c.feature_name,
            category: c.category,
            direction: c.direction,
            evidence: MetaLessonEvidence {
                survived_with_feature: c.survived_with,
                lost_with_feature: c.lost_with,
                survived_without_feature: c.survived_without,
                lost_without_feature: c.lost_without,
                rate_with_feature: c.rate_with,
                rate_without_feature: c.rate_without,
                chi_squared: c.chi2,
                p_value: c.raw_p,
                adjusted_p_value: adj_p,
                sparse_cells: c.sparse_cells,
            },
            confidence: (1.0 - adj_p).min(0.99),
            sample_size: c.sample_size,
            benchmark_classes: c.benchmark_classes,
        })
        .collect();

    (lessons, candidates_tested)
}

pub fn generate_meta_lessons(
    reports: &[BenchmarkSuiteReport],
    fdr_threshold: f64,
) -> MetaLessonReport {
    let mut all_records = Vec::new();
    let mut class_count = 0usize;
    for report in reports {
        for class_report in &report.classes {
            let mut has_records = false;
            if let Some(ref survival) = class_report.continuity.survival {
                all_records.extend(survival.records.clone());
                has_records = true;
            }
            for baseline_report in &class_report.baselines {
                if let Some(ref survival) = baseline_report.survival {
                    all_records.extend(survival.records.clone());
                    has_records = true;
                }
            }
            if has_records {
                class_count += 1;
            }
        }
    }
    let total_records = all_records.len();
    let (mut lessons, candidates_tested) =
        compare_feature_distributions(&all_records, fdr_threshold, 3, class_count);
    for lesson in &mut lessons {
        lesson.benchmark_classes = class_count;
    }
    MetaLessonReport {
        generated_at: Utc::now().to_rfc3339(),
        total_records,
        lessons,
        candidates_tested,
    }
}

/// Write meta-lesson hypotheses to the continuity kernel.
///
/// These are written as `ContinuityKind::Hypothesis` (not Lesson) with
/// `Scope::Project` (not Global) because they are unvalidated candidate
/// patterns. Promotion to Lesson/Constraint requires closed-loop evidence
/// that the hypothesis actually improves downstream continuity quality.
///
/// **Limitations:**
/// - Chi-squared with Yates correction is unreliable when expected cell
///   counts are < 5. Hypotheses with `sparse_cells: true` should be treated
///   with extra scepticism. Fisher's exact test would be more appropriate
///   for those cases but is not yet implemented.
/// - Benjamini-Hochberg FDR correction is applied, but with few tests the
///   correction has limited power. The adjusted p-values should be
///   interpreted as "less likely to be false discovery" not "confirmed."
pub fn write_meta_lessons_to_kernel(
    kernel: &SharedContinuityKernel,
    context_id: &str,
    report: &MetaLessonReport,
) -> Result<Vec<crate::continuity::ContinuityItemRecord>> {
    use crate::continuity::{ContinuityItemInput, ContinuityKind, ContinuityStatus};
    use crate::model::{MemoryLayer, Scope};

    let inputs: Vec<ContinuityItemInput> = report
        .lessons
        .iter()
        .map(|lesson| {
            let sparse_warning = if lesson.evidence.sparse_cells {
                "\n\nWARNING: sparse cells detected (expected count < 5). \
                 Chi-squared unreliable; Fisher's exact test recommended."
            } else {
                ""
            };
            ContinuityItemInput {
                context_id: context_id.to_string(),
                author_agent_id: "metacognitive-analyser".to_string(),
                kind: ContinuityKind::Hypothesis,
                title: format!(
                    "survival-hypothesis: {} ({:?})",
                    lesson.feature_name, lesson.category
                ),
                body: format!(
                    "{}\n\nStatistical evidence: chi²={:.2}, raw p={:.4}, \
                     BH-adjusted p={:.4}, sample_size={}, classes={}\n\
                     Survival rate with feature: {:.1}%, without: {:.1}%\n\
                     Status: UNVALIDATED candidate hypothesis. \
                     Requires closed-loop A/B testing before promotion.{sparse_warning}",
                    lesson.pattern,
                    lesson.evidence.chi_squared,
                    lesson.evidence.p_value,
                    lesson.evidence.adjusted_p_value,
                    lesson.sample_size,
                    lesson.benchmark_classes,
                    lesson.evidence.rate_with_feature * 100.0,
                    lesson.evidence.rate_without_feature * 100.0,
                ),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some((lesson.confidence * 0.7).min(0.8)),
                confidence: Some(lesson.confidence * 0.8),
                salience: Some(0.6),
                layer: Some(MemoryLayer::Episodic),
                supports: Vec::new(),
                dimensions: vec![
                    crate::model::DimensionValue {
                        key: "metacognitive_phase".into(),
                        value: "2".into(),
                        weight: 100,
                    },
                    crate::model::DimensionValue {
                        key: "validation_status".into(),
                        value: "unvalidated".into(),
                        weight: 50,
                    },
                ],
                extra: serde_json::json!({
                    "feature_name": lesson.feature_name,
                    "category": lesson.category,
                    "direction": lesson.direction,
                    "evidence": lesson.evidence,
                    "requires_validation": true,
                }),
            }
        })
        .collect();

    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    kernel.write_derivations(inputs)
}

// ---------------------------------------------------------------------------
// Phase 3: Closed-loop hypothesis validation functions
// ---------------------------------------------------------------------------

/// Write Phase 3 validation outcomes to the kernel.
///
/// Promoted hypotheses become `Lesson` items. Rejected hypotheses are resolved.
pub fn write_phase3_outcomes_to_kernel(
    kernel: &SharedContinuityKernel,
    context_id: &str,
    report: &Phase3ValidationReport,
) -> Result<Vec<crate::continuity::ContinuityItemRecord>> {
    let mut records = Vec::new();

    for result in &report.results {
        if result.promoted {
            let input = ContinuityItemInput {
                context_id: context_id.to_string(),
                author_agent_id: "metacognitive-validator".to_string(),
                kind: ContinuityKind::Lesson,
                title: format!(
                    "validated-survival-pattern: {} ({:?})",
                    result.feature_name, result.category
                ),
                body: format!(
                    "VALIDATED: {:?} items with feature '{}' show {:.1}% absolute improvement \
                     in survival rate (treatment {:.1}% vs control {:.1}%). \
                     Positive in {}/{} benchmark classes. Promoted from hypothesis after \
                     Phase 3 closed-loop validation cycle {}.",
                    result.category,
                    result.feature_name,
                    result.absolute_improvement * 100.0,
                    result.treatment_survival_rate * 100.0,
                    result.control_survival_rate * 100.0,
                    result.positive_classes,
                    result.total_classes,
                    report.cycle,
                ),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Active),
                importance: Some(0.85),
                confidence: Some(0.9),
                salience: Some(0.8),
                layer: Some(MemoryLayer::Semantic),
                supports: Vec::new(),
                dimensions: vec![
                    DimensionValue {
                        key: "metacognitive_phase".into(),
                        value: "3".into(),
                        weight: 100,
                    },
                    DimensionValue {
                        key: "validation_status".into(),
                        value: "promoted".into(),
                        weight: 80,
                    },
                ],
                extra: serde_json::json!({
                    "feature_name": result.feature_name,
                    "category": result.category,
                    "absolute_improvement": result.absolute_improvement,
                    "treatment_rate": result.treatment_survival_rate,
                    "control_rate": result.control_survival_rate,
                    "cycle": report.cycle,
                }),
            };
            let written = kernel.write_derivations(vec![input])?;
            records.extend(written);
        }
    }

    Ok(records)
}

// ---------------------------------------------------------------------------

pub(crate) fn repair_output_from_envelope(
    mut output: AgentContinuationOutput,
    envelope: &ContextEnvelope,
) -> AgentContinuationOutput {
    repair_critical_facts_from_envelope(&mut output, envelope);
    for decision in &mut output.decisions {
        repair_decision_from_envelope(decision, envelope);
    }
    repair_operational_scars_from_envelope(&mut output, envelope);
    output
}

fn repair_critical_facts_from_envelope(
    output: &mut AgentContinuationOutput,
    envelope: &ContextEnvelope,
) {
    for fact in &mut output.critical_facts {
        if let Some((matched, display_text)) =
            find_matching_surface_line(&fact.text, envelope, &['f'])
        {
            fact.text = display_text;
            if fact.evidence.is_empty() {
                fact.evidence.push(matched.label.clone());
            }
        }
    }

    let already_has_primary_context = output
        .critical_facts
        .iter()
        .any(|fact| match_keywords(&fact.text, &["context", "primary"]));
    if already_has_primary_context {
        return;
    }

    let Some((matched, display_text)) = envelope
        .surfaced
        .iter()
        .find(|item| {
            item.label.starts_with('f') && match_keywords(&item.text, &["context", "primary"])
        })
        .map(|item| (item, item.text.clone()))
    else {
        return;
    };

    output.critical_facts.push(EvidenceNote {
        text: primary_note_text(&display_text),
        evidence: vec![matched.label.clone()],
    });
}

fn repair_decision_from_envelope(decision: &mut DecisionNote, envelope: &ContextEnvelope) {
    if let Some(title) = strip_placeholder_title(&decision.text) {
        decision.text = title;
    }
    if let Some(rationale) = strip_placeholder_title(&decision.rationale) {
        decision.rationale = rationale;
    }
    if let Some((title, rationale)) = split_title_and_rationale(&decision.text) {
        if looks_like_placeholder(&decision.text) {
            decision.text = title;
        }
        if decision.rationale.trim().is_empty() {
            decision.rationale = rationale;
        }
    }
    if !decision.rationale.trim().is_empty() && !decision.evidence.is_empty() {
        return;
    }
    let needle = normalized_decision_probe(decision);
    if needle.is_empty() {
        return;
    }
    let mut matches = envelope
        .surfaced
        .iter()
        .filter(|item| matches!(item.label.chars().next(), Some('f' | 'd' | 'r')))
        .filter(|item| normalized_text(&item.text).contains(&needle))
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return;
    }
    matches.sort_by_key(|item| match item.label.chars().next() {
        Some('d') => 0,
        Some('f') => 1,
        Some('r') => 2,
        _ => 3,
    });
    let matched = matches.remove(0);
    if let Some((title, rationale)) = split_title_and_rationale(&matched.text) {
        decision.text = title;
        if decision.rationale.trim().is_empty()
            || normalized_text(&decision.rationale) == normalized_text(&decision.text)
        {
            decision.rationale = rationale;
        }
    }
    if decision.evidence.is_empty() {
        decision.evidence.push(matched.label.clone());
    }
}

fn repair_operational_scars_from_envelope(
    output: &mut AgentContinuationOutput,
    envelope: &ContextEnvelope,
) {
    for scar in &mut output.operational_scars {
        if let Some((matched, display_text)) =
            find_matching_surface_line(&scar.text, envelope, &['s'])
        {
            scar.text = primary_note_text(&display_text);
            if scar.evidence.is_empty() {
                scar.evidence.push(matched.label.clone());
            }
        }
    }
    if !output.operational_scars.is_empty() {
        return;
    }
    let candidates = output
        .critical_facts
        .iter()
        .chain(output.avoid_repeating.iter())
        .chain(output.constraints.iter())
        .collect::<Vec<_>>();
    for candidate in candidates {
        let Some((matched, display_text)) =
            find_matching_surface_line(&candidate.text, envelope, &['s'])
        else {
            continue;
        };
        let repaired_text = primary_note_text(&display_text);
        if output.operational_scars.iter().any(|item| {
            normalized_text(&item.text) == normalized_text(&repaired_text)
                || item.evidence.iter().any(|label| label == &matched.label)
        }) {
            continue;
        }
        output.operational_scars.push(EvidenceNote {
            text: repaired_text,
            evidence: vec![matched.label.clone()],
        });
    }
}

fn find_matching_surface_line<'a>(
    probe_text: &str,
    envelope: &'a ContextEnvelope,
    prefixes: &[char],
) -> Option<(&'a SurfacedItem, String)> {
    let probe = normalized_text(probe_text);
    if probe.is_empty() {
        return None;
    }
    let display = envelope_surface_display_map(envelope);
    let mut matches = envelope
        .surfaced
        .iter()
        .filter(|item| {
            prefixes
                .iter()
                .any(|prefix| item.label.starts_with(*prefix))
        })
        .filter_map(|item| {
            let line_text = display
                .get(&item.label)
                .cloned()
                .unwrap_or_else(|| item.text.clone());
            let normalized = normalized_text(&line_text);
            if normalized.is_empty() {
                return None;
            }
            if probe.contains(&normalized) || normalized.contains(&probe) {
                Some((item, line_text))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return None;
    }
    matches.sort_by_key(|(item, _)| match item.label.chars().next() {
        Some('s') => 0,
        Some('r') => 1,
        _ => 2,
    });
    matches.into_iter().next()
}

fn envelope_surface_display_map(envelope: &ContextEnvelope) -> BTreeMap<String, String> {
    envelope
        .text
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix('[')?;
            let (label, _) = rest.split_once(']')?;
            let (_, display) = line.rsplit_once("] ")?;
            Some((label.to_string(), display.trim().to_string()))
        })
        .collect()
}

fn primary_note_text(text: &str) -> String {
    text.split_once("::")
        .map(|(title, _)| title.trim().to_string())
        .filter(|title| !title.is_empty())
        .unwrap_or_else(|| text.trim().to_string())
}

fn strip_placeholder_title(text: &str) -> Option<String> {
    let (left, right) = text.split_once("::")?;
    if !looks_like_placeholder(left) {
        return None;
    }
    let cleaned = right.trim().to_string();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

fn split_title_and_rationale(text: &str) -> Option<(String, String)> {
    let (title, rationale) = text.split_once("::")?;
    let title = title
        .trim()
        .trim_start_matches("decision_text")
        .trim_start_matches("decision text")
        .trim_start_matches(':')
        .trim()
        .to_string();
    let rationale = rationale
        .trim()
        .trim_start_matches("rationale_text")
        .trim_start_matches("rationale text")
        .trim_start_matches(':')
        .trim()
        .to_string();
    if title.is_empty() || rationale.is_empty() {
        None
    } else {
        Some((title, rationale))
    }
}

fn looks_like_placeholder(text: &str) -> bool {
    let normalized = normalized_text(text);
    normalized.starts_with("decisiontext")
        || normalized.starts_with("decision")
        || normalized.starts_with("rationaletext")
}

fn normalized_decision_probe(decision: &DecisionNote) -> String {
    let probe = if !decision.rationale.trim().is_empty() {
        decision.rationale.as_str()
    } else {
        decision.text.as_str()
    };
    normalized_text(probe)
}

fn normalized_text(text: &str) -> String {
    text.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

pub(crate) fn failed_evaluation(envelope: &ContextEnvelope) -> Evaluation {
    let provenance = envelope
        .surfaced
        .iter()
        .filter(|item| item.has_provenance)
        .count() as f64
        / envelope.surfaced.len().max(1) as f64;
    Evaluation {
        provenance_coverage: provenance,
        ..Evaluation::default()
    }
}

fn write_debug_artifacts(
    class_root: &Path,
    baseline: BaselineKind,
    envelope: &ContextEnvelope,
    output: &AgentContinuationOutput,
    failure: Option<&str>,
) -> Result<Vec<String>> {
    let debug_root = class_root.join("debug");
    std::fs::create_dir_all(&debug_root)?;
    let stem = baseline.slug();
    let context_path = debug_root.join(format!("{stem}-context.txt"));
    std::fs::write(&context_path, envelope.text.as_bytes())?;
    let output_path = debug_root.join(format!("{stem}-output.json"));
    std::fs::write(&output_path, serde_json::to_vec_pretty(output)?)?;
    let mut artifacts = vec![
        context_path.to_string_lossy().to_string(),
        output_path.to_string_lossy().to_string(),
    ];
    if let Some(failure) = failure {
        let failure_path = debug_root.join(format!("{stem}-failure.txt"));
        std::fs::write(&failure_path, failure.as_bytes())?;
        artifacts.push(failure_path.to_string_lossy().to_string());
    }
    Ok(artifacts)
}

pub(crate) fn match_keywords(text: &str, keywords: &[&str]) -> bool {
    let lowered = text.to_lowercase();
    keywords
        .iter()
        .all(|keyword| lowered.contains(&keyword.to_lowercase()))
}

fn flatten_keywords(items: &[TruthItem]) -> Vec<&str> {
    items
        .iter()
        .flat_map(|item| item.keywords.clone())
        .collect()
}

fn write_model_output(
    kernel: &SharedContinuityKernel,
    context_id: &str,
    agent_id: &str,
    output: &AgentContinuationOutput,
    envelope: &ContextEnvelope,
) -> Result<()> {
    for item in &output.decisions {
        kernel.mark_decision(continuity_from_decision(
            context_id, agent_id, item, envelope,
        ))?;
    }
    for item in &output.constraints {
        kernel.mark_constraint(continuity_from_note(
            context_id,
            agent_id,
            item,
            ContinuityKind::Constraint,
            envelope,
        ))?;
    }
    for item in &output.open_hypotheses {
        kernel.mark_hypothesis(continuity_from_note(
            context_id,
            agent_id,
            item,
            ContinuityKind::Hypothesis,
            envelope,
        ))?;
    }
    for item in &output.operational_scars {
        kernel.mark_operational_scar(continuity_from_note(
            context_id,
            agent_id,
            item,
            ContinuityKind::OperationalScar,
            envelope,
        ))?;
    }
    if !output.next_step.text.trim().is_empty() {
        kernel.write_derivations(vec![ContinuityItemInput {
            context_id: context_id.to_string(),
            author_agent_id: agent_id.to_string(),
            kind: ContinuityKind::WorkingState,
            title: "model-next-step".into(),
            body: output.next_step.text.clone(),
            scope: Scope::Project,
            status: Some(ContinuityStatus::Active),
            importance: Some(0.9),
            confidence: Some(0.8),
            salience: Some(0.9),
            layer: Some(MemoryLayer::Hot),
            supports: supports_from_labels(&output.next_step.evidence, envelope),
            dimensions: vec![DimensionValue {
                key: "next_step".into(),
                value: "true".into(),
                weight: 100,
            }],
            extra: serde_json::json!({}),
        }])?;
    }
    Ok(())
}

fn continuity_from_note(
    context_id: &str,
    agent_id: &str,
    item: &EvidenceNote,
    kind: ContinuityKind,
    envelope: &ContextEnvelope,
) -> ContinuityItemInput {
    ContinuityItemInput {
        context_id: context_id.to_string(),
        author_agent_id: agent_id.to_string(),
        kind,
        title: trim_text(&item.text, 80),
        body: item.text.clone(),
        scope: Scope::Project,
        status: Some(ContinuityStatus::Open),
        importance: Some(0.9),
        confidence: Some(0.75),
        salience: Some(0.9),
        layer: None,
        supports: supports_from_labels(&item.evidence, envelope),
        dimensions: Vec::new(),
        extra: serde_json::json!({}),
    }
}

fn continuity_from_decision(
    context_id: &str,
    agent_id: &str,
    item: &DecisionNote,
    envelope: &ContextEnvelope,
) -> ContinuityItemInput {
    ContinuityItemInput {
        context_id: context_id.to_string(),
        author_agent_id: agent_id.to_string(),
        kind: ContinuityKind::Decision,
        title: trim_text(&item.text, 80),
        body: format!("{}\nrationale={}", item.text, item.rationale),
        scope: Scope::Project,
        status: Some(ContinuityStatus::Active),
        importance: Some(0.95),
        confidence: Some(0.8),
        salience: Some(0.95),
        layer: None,
        supports: supports_from_labels(&item.evidence, envelope),
        dimensions: Vec::new(),
        extra: serde_json::json!({}),
    }
}

fn supports_from_labels(
    labels: &[String],
    envelope: &ContextEnvelope,
) -> Vec<crate::continuity::SupportRef> {
    let by_label = envelope
        .surfaced
        .iter()
        .map(|item| (item.label.clone(), item))
        .collect::<BTreeMap<_, _>>();
    labels
        .iter()
        .filter_map(|label| by_label.get(label))
        .map(|item| crate::continuity::SupportRef {
            support_type: item.support_type.clone(),
            support_id: item.support_id.clone(),
            reason: Some("model-cited".into()),
            weight: 1.0,
        })
        .collect()
}

pub(crate) fn scenario_for(class: BenchmarkClass) -> Scenario {
    let base_truth = GroundTruth {
        critical_facts: vec![
            TruthItem {
                keywords: vec!["selector_missing", "src/query.rs"],
                rationale_keywords: Vec::new(),
                judge_note: Some(
                    "Full credit requires both the selector/support-memory failure and the src/query.rs anchor; file mention alone is partial.",
                ),
                judge_required_concepts: vec!["selector/support-memory failure", "src/query.rs"],
            },
            TruthItem {
                keywords: vec!["context", "primary"],
                rationale_keywords: Vec::new(),
                judge_note: Some(
                    "Full credit requires explicit identification that the active resume context is the primary context, not just a task mention.",
                ),
                judge_required_concepts: vec!["active resume context", "primary context"],
            },
        ],
        constraints: vec![TruthItem {
            keywords: vec!["preserve", "provenance"],
            rationale_keywords: Vec::new(),
            judge_note: None,
            judge_required_concepts: Vec::new(),
        }],
        decisions: vec![TruthItem {
            keywords: vec!["unified", "continuity", "interface"],
            rationale_keywords: vec!["agent", "swap"],
            judge_note: None,
            judge_required_concepts: Vec::new(),
        }],
        scars: vec![TruthItem {
            keywords: vec!["naive", "probe"],
            rationale_keywords: Vec::new(),
            judge_note: None,
            judge_required_concepts: Vec::new(),
        }],
        avoid_repeating: vec![TruthItem {
            keywords: vec!["manual", "probe"],
            rationale_keywords: Vec::new(),
            judge_note: None,
            judge_required_concepts: Vec::new(),
        }],
        next_step_keywords: vec!["benchmark", "adapter"],
    };
    let noise = matches!(
        class,
        BenchmarkClass::InterruptionStress | BenchmarkClass::MemoryPollution
    );
    let mut phases = vec![
        ScenarioPhase {
            actor_id: "planner".into(),
            actor_role: "planner".into(),
            kind: EventKind::Prompt,
            scope: Scope::Project,
            content: "Decision draft: use a unified continuity interface so agent swaps inherit the same context namespace.".into(),
            dimensions: vec![DimensionValue {
                key: "file".into(),
                value: "src/continuity.rs".into(),
                weight: 100,
            }],
            marks: vec![ScenarioMark {
                kind: ContinuityKind::Decision,
                title: "Use the unified continuity interface".into(),
                body: "The runtime should route agent swaps through one continuity interface rather than raw transcript transfer.".into(),
                status: ContinuityStatus::Active,
                dimensions: vec![DimensionValue {
                    key: "decision.interface".into(),
                    value: "uci".into(),
                    weight: 100,
                }],
            }],
        },
        ScenarioPhase {
            actor_id: "debugger".into(),
            actor_role: "debugger".into(),
            kind: EventKind::Error,
            scope: Scope::Project,
            content: "Incident: selector_missing in src/query.rs dropped required support memory after an agent swap.".into(),
            dimensions: vec![
                DimensionValue {
                    key: "file".into(),
                    value: "src/query.rs".into(),
                    weight: 100,
                },
                DimensionValue {
                    key: "hypothesis.root".into(),
                    value: "selector_missing".into(),
                    weight: 100,
                },
            ],
            marks: vec![
                ScenarioMark {
                    kind: ContinuityKind::Incident,
                    title: "selector_missing".into(),
                    body: "Selector pruning dropped required support memory from src/query.rs.".into(),
                    status: ContinuityStatus::Open,
                    dimensions: vec![DimensionValue {
                        key: "incident".into(),
                        value: "selector_missing".into(),
                        weight: 100,
                    }],
                },
                ScenarioMark {
                    kind: ContinuityKind::Constraint,
                    title: "Preserve provenance".into(),
                    body: "Resume packs must preserve provenance and unresolved contradictions.".into(),
                    status: ContinuityStatus::Open,
                    dimensions: vec![DimensionValue {
                        key: "constraint".into(),
                        value: "preserve_provenance".into(),
                        weight: 100,
                    }],
                },
            ],
        },
        ScenarioPhase {
            actor_id: "researcher".into(),
            actor_role: "researcher".into(),
            kind: EventKind::Note,
            scope: Scope::Project,
            content: "Operational scar: naive manual Ollama probes hung; use the adapter path with timeouts and telemetry.".into(),
            dimensions: vec![DimensionValue {
                key: "scar".into(),
                value: "naive_probe".into(),
                weight: 100,
            }],
            marks: vec![
                ScenarioMark {
                    kind: ContinuityKind::OperationalScar,
                    title: "Avoid naive probes".into(),
                    body: "Do not use ad hoc model probes without timeouts, structured output, and telemetry.".into(),
                    status: ContinuityStatus::Open,
                    dimensions: vec![DimensionValue {
                        key: "scar.runtime".into(),
                        value: "naive_probe".into(),
                        weight: 100,
                    }],
                },
                ScenarioMark {
                    kind: ContinuityKind::Hypothesis,
                    title: "Adapter timeout path".into(),
                    body: "A controlled adapter with explicit num_predict and request timeout should prevent hangs.".into(),
                    status: ContinuityStatus::Open,
                    dimensions: vec![DimensionValue {
                        key: "hypothesis".into(),
                        value: "adapter_timeout".into(),
                        weight: 100,
                    }],
                },
            ],
        },
    ];
    if noise {
        for idx in 0..6 {
            phases.push(ScenarioPhase {
                actor_id: format!("noise-{idx}"),
                actor_role: "noise".into(),
                kind: EventKind::Note,
                scope: Scope::Shared,
                content: format!(
                    "Unrelated interrupt {idx}: metrics queue={idx} latency={}ms",
                    30 + idx
                ),
                dimensions: vec![DimensionValue {
                    key: "noise".into(),
                    value: format!("interrupt-{idx}"),
                    weight: 100,
                }],
                marks: Vec::new(),
            });
        }
    }
    Scenario {
        id: class.slug().to_string(),
        namespace: "bench".into(),
        task_id: format!("task-{}", class.slug()),
        objective: format!(
            "Resume the {} continuity task without losing constraints or scars.",
            class.slug()
        ),
        phases,
        truth: base_truth,
    }
}

fn summarize(classes: &[BenchmarkClassReport]) -> BenchmarkSummary {
    let mut summary = BenchmarkSummary::default();
    summary.class_count = classes.len();
    summary.total_runs = classes.len() * 6;
    summary.failed_runs = classes
        .iter()
        .flat_map(|item| std::iter::once(&item.continuity).chain(item.baselines.iter()))
        .filter(|run| run.status == BaselineStatus::Failed)
        .count();
    if classes.is_empty() {
        return summary;
    }
    summary.avg_cfsr =
        classes.iter().map(|item| item.metrics.cfsr).sum::<f64>() / classes.len() as f64;
    summary.avg_csr =
        classes.iter().map(|item| item.metrics.csr).sum::<f64>() / classes.len() as f64;
    summary.avg_dlf =
        classes.iter().map(|item| item.metrics.dlf).sum::<f64>() / classes.len() as f64;
    summary.avg_ras =
        classes.iter().map(|item| item.metrics.ras).sum::<f64>() / classes.len() as f64;
    summary.avg_mpr =
        classes.iter().map(|item| item.metrics.mpr).sum::<f64>() / classes.len() as f64;
    summary.avg_pc = classes.iter().map(|item| item.metrics.pc).sum::<f64>() / classes.len() as f64;
    summary.avg_smcl =
        classes.iter().map(|item| item.metrics.smcl).sum::<f64>() / classes.len() as f64;
    summary.avg_sscr =
        classes.iter().map(|item| item.metrics.sscr).sum::<f64>() / classes.len() as f64;
    summary.avg_cgd =
        classes.iter().map(|item| item.metrics.cgd).sum::<f64>() / classes.len() as f64;
    summary
}

fn render_suite_markdown(report: &BenchmarkSuiteReport) -> String {
    let mut out = String::new();
    out.push_str("# Continuity Benchmark Summary\n\n");
    out.push_str(&format!("Generated: `{}`\n\n", report.generated_at));
    out.push_str(&format!(
        "Strong model: `{}`\nSmall model: `{}`\nEmbedding backend: `{}`\nRetrieval protocol: `{}`\n\n",
        report.config.strong_model,
        report.config.small_model,
        report.config.embedding_backend,
        report.config.protocol_id(),
    ));
    out.push_str(&format!(
        "Classes: `{}`  Failed runs: `{}` / `{}`\n\n",
        report.summary.class_count, report.summary.failed_runs, report.summary.total_runs
    ));
    out.push_str(
        "| class | shared path | shared status | cfsr | csr | dlf | osr | ras | smcl | sscr | cgd | crl_ms |\n",
    );
    out.push_str(
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    );
    for class in &report.classes {
        let shared_path = class
            .continuity
            .continuity_path
            .as_ref()
            .map(format_continuity_path_label)
            .unwrap_or_else(|| "n/a".into());
        out.push_str(&format!(
            "| {} | {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.0} |\n",
            class.class.slug(),
            shared_path,
            match class.continuity.status {
                BaselineStatus::Ok => "ok",
                BaselineStatus::Failed => "failed",
            },
            class.metrics.cfsr,
            class.metrics.csr,
            class.metrics.dlf,
            class.metrics.osr,
            class.metrics.ras,
            class.metrics.smcl,
            class.metrics.sscr,
            class.metrics.cgd,
            class.metrics.crl_ms,
        ));
    }
    out.push_str("\n## Aggregate\n\n");
    out.push_str(&format!(
        "- avg CFSR: `{:.2}`\n- avg CSR: `{:.2}`\n- avg DLF: `{:.2}`\n- avg RAS: `{:.2}`\n- avg MPR: `{:.2}`\n- avg PC: `{:.2}`\n- avg SMCL: `{:.2}`\n- avg SSCR: `{:.2}`\n- avg CGD: `{:.2}`\n",
        report.summary.avg_cfsr,
        report.summary.avg_csr,
        report.summary.avg_dlf,
        report.summary.avg_ras,
        report.summary.avg_mpr,
        report.summary.avg_pc,
        report.summary.avg_smcl,
        report.summary.avg_sscr,
        report.summary.avg_cgd,
    ));
    out
}

pub(crate) fn format_continuity_path_label(path: &ContinuityPathReport) -> String {
    match (path.kind, path.role) {
        (ContinuityPathKind::HandoffProof, _) => {
            format!("handoff-proof({})", path.proof_register_count)
        }
        (ContinuityPathKind::ReadContextOnly, ContinuityPathRole::ExplicitControl) => {
            "read-context-control".to_string()
        }
        (ContinuityPathKind::ReadContextOnly, ContinuityPathRole::Legacy)
        | (ContinuityPathKind::ReadContextOnly, ContinuityPathRole::ProofPath) => {
            "read-context-only".to_string()
        }
    }
}

fn ratio_or_zero(a: f64, b: f64) -> f64 {
    if b <= 0.0 { 0.0 } else { a / b }
}

fn inferred_hrt(ras: f64) -> f64 {
    if ras >= 0.9 {
        1.0
    } else if ras >= 0.6 {
        2.0
    } else {
        3.0
    }
}

fn storage_bytes(root: &Path) -> Result<u64> {
    let mut total = 0u64;
    if !root.exists() {
        return Ok(0);
    }
    for entry in walkdir(root)? {
        total += entry.metadata()?.len();
    }
    Ok(total)
}

fn walkdir(root: &Path) -> Result<Vec<std::fs::DirEntry>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            out.extend(walkdir(&entry.path())?);
        } else {
            out.push(entry);
        }
    }
    Ok(out)
}

fn gpu_report(config: &ContinuityBenchConfig) -> Option<String> {
    std::process::Command::new("ollama")
        .arg("ps")
        .env("OLLAMA_HOST", config.ollama_endpoint.replace("http://", ""))
        .output()
        .ok()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|text| !text.is_empty())
}

fn estimate_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}

fn trim_text(text: &str, max: usize) -> String {
    if text.len() <= max {
        text.to_string()
    } else {
        format!("{}...", &text[..max])
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BaselineKind, BaselineRunReport, BaselineStatus, BenchmarkClass, BenchmarkClassReport,
        BenchmarkMetrics, CategoryStats, ContextEnvelope, ContinuityBenchConfig,
        ContinuityPathKind, ContinuityPathReport, ContinuityPathRole, Evaluation, GroundTruth,
        HypothesisValidationResult, LessonDirection, MetaLesson, MetaLessonEvidence,
        MetaLessonReport, Phase3ClassResult, Phase3ValidationReport, ResourceEnvelope,
        SurfacedItem, SurvivalOutcome, SurvivalReport, TruthCategory, TruthItem,
        benchmark_survival_analysis, build_context_envelope, category_rate, chi_squared_2x2,
        compare_feature_distributions, count_items, erfc_approx, estimate_tokens, evaluate_output,
        extract_eligible_hypotheses, failed_evaluation, flatten_keywords,
        format_continuity_path_label, generate_meta_lessons, generate_phase3_report,
        has_sparse_cells, inferred_hrt, looks_like_placeholder, match_keywords,
        normalized_decision_probe, normalized_text, populate_scenario, primary_note_text,
        ratio_or_zero, render_suite_markdown, repair_output_from_envelope, scenario_for,
        score_decisions, score_notes, section_prefix, split_title_and_rationale,
        strip_placeholder_title, summarize, trim_text, unsupported_items,
        write_phase3_outcomes_to_kernel,
    };
    use crate::adapters::{
        AgentContinuationOutput, DecisionNote, EvidenceNote, ModelCallMetrics, SurvivalHypothesis,
    };
    use crate::benchmark::{BenchmarkSuiteReport, BenchmarkSummary};
    use crate::continuity::{
        AttachAgentInput, OpenContextInput, SharedContinuityKernel, SnapshotInput,
        UnifiedContinuityInterface,
    };
    use crate::model::SnapshotResolution;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn config_with_classes(classes: Vec<BenchmarkClass>) -> ContinuityBenchConfig {
        ContinuityBenchConfig {
            output_dir: PathBuf::from("/tmp/bench"),
            ollama_endpoint: "http://127.0.0.1:11434".into(),
            strong_model: "glm-4.7-flash:latest".into(),
            small_model: "qwen2.5:0.5b".into(),
            embedding_backend: "hash:128".into(),
            retrieval_protocol: "uci+compiler+vector://hash:128?budget=160&candidates=24&recent=8"
                .into(),
            classes,
            token_budget: 160,
            candidate_limit: 24,
            recent_window: 8,
            timeout_secs: 120,
            num_predict: 192,
        }
    }

    #[test]
    fn selected_classes_defaults_to_full_suite_when_unset() {
        let config = config_with_classes(Vec::new());
        assert_eq!(config.selected_classes(), BenchmarkClass::all());
    }

    #[test]
    fn selected_classes_preserves_requested_subset_order() {
        let config = config_with_classes(vec![
            BenchmarkClass::StrongToSmallContinuation,
            BenchmarkClass::AgentSwapSurvival,
        ]);
        assert_eq!(
            config.selected_classes(),
            vec![
                BenchmarkClass::StrongToSmallContinuation,
                BenchmarkClass::AgentSwapSurvival,
            ]
        );
    }

    #[test]
    fn repair_output_backfills_decision_lineage_from_resumption_fact() {
        let envelope = ContextEnvelope {
            provider: BaselineKind::SharedContinuity,
            retrieval_ms: 9,
            text: String::new(),
            token_estimate: 42,
            surfaced: vec![SurfacedItem {
                label: "f3".into(),
                support_type: "continuity".into(),
                support_id: "decision-1".into(),
                text: "Use the unified continuity interface :: The runtime should route agent swaps through one continuity interface rather than raw transcript transfer.".into(),
                has_provenance: true,
            }],
        };
        let output = AgentContinuationOutput {
            decisions: vec![DecisionNote {
                text: "decision_use_unified_continuity_interface :: Use the unified continuity interface".into(),
                rationale: String::new(),
                evidence: Vec::new(),
            }],
            ..AgentContinuationOutput::default()
        };

        let repaired = repair_output_from_envelope(output, &envelope);
        assert_eq!(
            repaired.decisions[0].text,
            "Use the unified continuity interface"
        );
        assert!(repaired.decisions[0].rationale.contains("agent swaps"));
        assert_eq!(repaired.decisions[0].evidence, vec!["f3"]);
    }

    #[test]
    fn repair_output_backfills_primary_context_fact_from_resumption_fact() {
        let envelope = ContextEnvelope {
            provider: BaselineKind::SharedContinuity,
            retrieval_ms: 9,
            text: String::new(),
            token_estimate: 42,
            surfaced: vec![SurfacedItem {
                label: "f1".into(),
                support_type: "continuity".into(),
                support_id: "fact-1".into(),
                text: "Primary context is bench / task-strong-to-small for this resume.".into(),
                has_provenance: true,
            }],
        };
        let output = AgentContinuationOutput {
            critical_facts: vec![EvidenceNote {
                text: "selector_missing :: Selector pruning dropped required support memory from src/query.rs".into(),
                evidence: Vec::new(),
            }],
            ..AgentContinuationOutput::default()
        };

        let repaired = repair_output_from_envelope(output, &envelope);
        assert_eq!(repaired.critical_facts.len(), 2);
        assert_eq!(
            repaired.critical_facts[1].text,
            "Primary context is bench / task-strong-to-small for this resume."
        );
        assert_eq!(repaired.critical_facts[1].evidence, vec!["f1"]);
    }

    #[test]
    fn repair_output_preserves_rich_fact_text_from_resumption_fact() {
        let envelope = ContextEnvelope {
            provider: BaselineKind::SharedContinuity,
            retrieval_ms: 9,
            text: String::new(),
            token_estimate: 42,
            surfaced: vec![SurfacedItem {
                label: "f2".into(),
                support_type: "continuity".into(),
                support_id: "fact-2".into(),
                text: "selector_missing :: Selector pruning dropped required support memory from src/query.rs.".into(),
                has_provenance: true,
            }],
        };
        let output = AgentContinuationOutput {
            critical_facts: vec![EvidenceNote {
                text: "selector_missing".into(),
                evidence: Vec::new(),
            }],
            ..AgentContinuationOutput::default()
        };

        let repaired = repair_output_from_envelope(output, &envelope);
        assert_eq!(
            repaired.critical_facts[0].text,
            "selector_missing :: Selector pruning dropped required support memory from src/query.rs."
        );
        assert_eq!(repaired.critical_facts[0].evidence, vec!["f2"]);
    }

    #[test]
    fn repair_output_backfills_operational_scar_from_surfaced_context() {
        let envelope = ContextEnvelope {
            provider: BaselineKind::SharedContinuity,
            retrieval_ms: 9,
            text: "Section: operational_scars\n[s1][operational_scar][open] Avoid naive probes :: Do not use ad hoc model probes without timeouts, structured output, and telemetry.".into(),
            token_estimate: 42,
            surfaced: vec![SurfacedItem {
                label: "s1".into(),
                support_type: "continuity".into(),
                support_id: "scar-1".into(),
                text: "Do not use ad hoc model probes without timeouts, structured output, and telemetry.".into(),
                has_provenance: true,
            }],
        };
        let output = AgentContinuationOutput {
            critical_facts: vec![EvidenceNote {
                text: "operational_scar: avoid_naive_probes :: Do not use ad hoc model probes without timeouts, structured output, and telemetry.".into(),
                evidence: Vec::new(),
            }],
            ..AgentContinuationOutput::default()
        };

        let repaired = repair_output_from_envelope(output, &envelope);
        assert_eq!(repaired.operational_scars.len(), 1);
        assert_eq!(repaired.operational_scars[0].text, "Avoid naive probes");
        assert_eq!(repaired.operational_scars[0].evidence, vec!["s1"]);
    }

    #[test]
    fn suite_markdown_records_embedding_backend_and_protocol() {
        let config = config_with_classes(vec![BenchmarkClass::AgentSwapSurvival]);
        let report = BenchmarkSuiteReport {
            generated_at: "2026-03-22T00:00:00Z".into(),
            config,
            classes: vec![BenchmarkClassReport {
                class: BenchmarkClass::AgentSwapSurvival,
                scenario_id: "agent-swap-survival".into(),
                continuity: BaselineRunReport {
                    baseline: BaselineKind::SharedContinuity,
                    status: BaselineStatus::Ok,
                    model: "qwen2.5:0.5b".into(),
                    retrieval_ms: 12,
                    model_metrics: ModelCallMetrics::default(),
                    envelope_tokens: 42,
                    evaluation: Evaluation::default(),
                    survival: None,
                    failure: None,
                    artifacts: Vec::new(),
                    continuity_path: Some(ContinuityPathReport {
                        kind: ContinuityPathKind::HandoffProof,
                        role: ContinuityPathRole::ProofPath,
                        proof_register_count: 4,
                        proof_register_labels: vec!["pf1".into(), "pd1".into()],
                    }),
                },
                baselines: Vec::new(),
                metrics: BenchmarkMetrics::default(),
                resource: ResourceEnvelope::default(),
                artifacts: Vec::new(),
                hypothesis_injection: None,
            }],
            summary: BenchmarkSummary::default(),
            meta_lessons: None,
            phase3: None,
        };

        let markdown = render_suite_markdown(&report);
        assert!(markdown.contains("Embedding backend: `hash:128`"));
        assert!(markdown.contains(
            "Retrieval protocol: `uci+compiler+vector://hash:128?budget=160&candidates=24&recent=8`"
        ));
        assert!(markdown.contains("| class | shared path | shared status |"));
        assert!(markdown.contains("handoff-proof(4)"));
    }

    #[test]
    fn suite_markdown_labels_baseline_isolation_as_explicit_control() {
        let config = config_with_classes(vec![BenchmarkClass::BaselineIsolation]);
        let report = BenchmarkSuiteReport {
            generated_at: "2026-03-22T00:00:00Z".into(),
            config,
            classes: vec![BenchmarkClassReport {
                class: BenchmarkClass::BaselineIsolation,
                scenario_id: "baseline-isolation".into(),
                continuity: BaselineRunReport {
                    baseline: BaselineKind::SharedContinuity,
                    status: BaselineStatus::Ok,
                    model: "qwen2.5:0.5b".into(),
                    retrieval_ms: 12,
                    model_metrics: ModelCallMetrics::default(),
                    envelope_tokens: 42,
                    evaluation: Evaluation::default(),
                    survival: None,
                    failure: None,
                    artifacts: Vec::new(),
                    continuity_path: Some(ContinuityPathReport {
                        kind: ContinuityPathKind::ReadContextOnly,
                        role: ContinuityPathRole::ExplicitControl,
                        proof_register_count: 0,
                        proof_register_labels: Vec::new(),
                    }),
                },
                baselines: Vec::new(),
                metrics: BenchmarkMetrics::default(),
                resource: ResourceEnvelope::default(),
                artifacts: Vec::new(),
                hypothesis_injection: None,
            }],
            summary: BenchmarkSummary::default(),
            meta_lessons: None,
            phase3: None,
        };

        let markdown = render_suite_markdown(&report);
        assert!(markdown.contains("read-context-control"));
    }

    #[test]
    fn cross_agent_shared_path_uses_handoff_proof() {
        let dir = tempdir().unwrap();
        let kernel = SharedContinuityKernel::open(dir.path()).unwrap();
        let scenario = scenario_for(BenchmarkClass::CrossAgentCollaborative);
        let attach = kernel
            .attach_agent(AttachAgentInput {
                agent_id: "planner-strong".into(),
                agent_type: "ollama".into(),
                capabilities: vec!["plan".into()],
                namespace: scenario.namespace.clone(),
                role: Some("planner".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let context = kernel
            .open_context(OpenContextInput {
                namespace: scenario.namespace.clone(),
                task_id: scenario.task_id.clone(),
                session_id: "cross-agent-session".into(),
                objective: scenario.objective.clone(),
                selector: None,
                agent_id: Some("planner-strong".into()),
                attachment_id: Some(attach.id),
            })
            .unwrap();
        let _ = populate_scenario(&kernel, &context.id, &scenario).unwrap();

        let (envelope, continuity_path) = build_context_envelope(
            &kernel,
            BenchmarkClass::CrossAgentCollaborative,
            BaselineKind::SharedContinuity,
            &context.id,
            &scenario,
            "relay-b",
            160,
            24,
            8,
        )
        .unwrap();

        let continuity_path = continuity_path.expect("shared continuity path");
        assert_eq!(continuity_path.kind, ContinuityPathKind::HandoffProof);
        assert!(continuity_path.proof_register_count >= 4);
        assert!(
            continuity_path
                .proof_register_labels
                .iter()
                .any(|label| label == "pd1")
        );
        assert!(envelope.text.contains("Section: handoff_proof"));
        assert!(envelope.text.contains("[pd1][handoff_proof][decision]"));
    }

    #[test]
    fn crash_recovery_shared_path_reopens_with_handoff_proof() {
        let dir = tempdir().unwrap();
        let kernel = SharedContinuityKernel::open(dir.path()).unwrap();
        let scenario = scenario_for(BenchmarkClass::CrashRecovery);
        let attach = kernel
            .attach_agent(AttachAgentInput {
                agent_id: "planner-strong".into(),
                agent_type: "ollama".into(),
                capabilities: vec!["plan".into()],
                namespace: scenario.namespace.clone(),
                role: Some("planner".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let context = kernel
            .open_context(OpenContextInput {
                namespace: scenario.namespace.clone(),
                task_id: scenario.task_id.clone(),
                session_id: "crash-recovery-session".into(),
                objective: scenario.objective.clone(),
                selector: None,
                agent_id: Some("planner-strong".into()),
                attachment_id: Some(attach.id),
            })
            .unwrap();
        let _ = populate_scenario(&kernel, &context.id, &scenario).unwrap();
        let _ = kernel
            .snapshot(SnapshotInput {
                context_id: Some(context.id.clone()),
                namespace: None,
                task_id: None,
                objective: Some("pre-crash checkpoint".into()),
                selector: None,
                resolution: SnapshotResolution::Medium,
                token_budget: 160,
                candidate_limit: 24,
                owner_agent_id: Some("planner-strong".into()),
            })
            .unwrap();

        let reopened = SharedContinuityKernel::open(dir.path()).unwrap();
        let (envelope, continuity_path) = build_context_envelope(
            &reopened,
            BenchmarkClass::CrashRecovery,
            BaselineKind::SharedContinuity,
            &context.id,
            &scenario,
            "relay-b",
            160,
            24,
            8,
        )
        .unwrap();

        let continuity_path = continuity_path.expect("shared continuity path");
        assert_eq!(continuity_path.kind, ContinuityPathKind::HandoffProof);
        assert!(continuity_path.proof_register_count >= 4);
        assert!(
            continuity_path
                .proof_register_labels
                .iter()
                .any(|label| label == "pd1")
        );
        assert!(envelope.text.contains("Section: handoff_proof"));
        assert!(envelope.text.contains("[pd1][handoff_proof][decision]"));
    }

    #[test]
    fn context_budget_shared_path_uses_handoff_proof() {
        let dir = tempdir().unwrap();
        let kernel = SharedContinuityKernel::open(dir.path()).unwrap();
        let scenario = scenario_for(BenchmarkClass::ContextBudgetCompression);
        let attach = kernel
            .attach_agent(AttachAgentInput {
                agent_id: "planner-strong".into(),
                agent_type: "ollama".into(),
                capabilities: vec!["plan".into()],
                namespace: scenario.namespace.clone(),
                role: Some("planner".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let context = kernel
            .open_context(OpenContextInput {
                namespace: scenario.namespace.clone(),
                task_id: scenario.task_id.clone(),
                session_id: "context-budget-session".into(),
                objective: scenario.objective.clone(),
                selector: None,
                agent_id: Some("planner-strong".into()),
                attachment_id: Some(attach.id),
            })
            .unwrap();
        let _ = populate_scenario(&kernel, &context.id, &scenario).unwrap();

        let (envelope, continuity_path) = build_context_envelope(
            &kernel,
            BenchmarkClass::ContextBudgetCompression,
            BaselineKind::SharedContinuity,
            &context.id,
            &scenario,
            "relay-b",
            160,
            24,
            8,
        )
        .unwrap();

        let continuity_path = continuity_path.expect("shared continuity path");
        assert_eq!(continuity_path.kind, ContinuityPathKind::HandoffProof);
        assert!(continuity_path.proof_register_count >= 4);
        assert!(
            continuity_path
                .proof_register_labels
                .iter()
                .any(|label| label == "pd1")
        );
        assert!(envelope.text.contains("Section: handoff_proof"));
        assert!(envelope.text.contains("[pd1][handoff_proof][decision]"));
    }

    #[test]
    fn memory_pollution_shared_path_uses_handoff_proof() {
        let dir = tempdir().unwrap();
        let kernel = SharedContinuityKernel::open(dir.path()).unwrap();
        let scenario = scenario_for(BenchmarkClass::MemoryPollution);
        let attach = kernel
            .attach_agent(AttachAgentInput {
                agent_id: "planner-strong".into(),
                agent_type: "ollama".into(),
                capabilities: vec!["plan".into()],
                namespace: scenario.namespace.clone(),
                role: Some("planner".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let context = kernel
            .open_context(OpenContextInput {
                namespace: scenario.namespace.clone(),
                task_id: scenario.task_id.clone(),
                session_id: "memory-pollution-session".into(),
                objective: scenario.objective.clone(),
                selector: None,
                agent_id: Some("planner-strong".into()),
                attachment_id: Some(attach.id),
            })
            .unwrap();
        let _ = populate_scenario(&kernel, &context.id, &scenario).unwrap();

        let (envelope, continuity_path) = build_context_envelope(
            &kernel,
            BenchmarkClass::MemoryPollution,
            BaselineKind::SharedContinuity,
            &context.id,
            &scenario,
            "relay-b",
            160,
            24,
            8,
        )
        .unwrap();

        let continuity_path = continuity_path.expect("shared continuity path");
        assert_eq!(continuity_path.kind, ContinuityPathKind::HandoffProof);
        assert!(continuity_path.proof_register_count >= 4);
        assert!(
            continuity_path
                .proof_register_labels
                .iter()
                .any(|label| label == "pd1")
        );
        assert!(envelope.text.contains("Section: handoff_proof"));
        assert!(envelope.text.contains("[pd1][handoff_proof][decision]"));
    }

    #[test]
    fn baseline_isolation_shared_path_stays_explicit_control() {
        let dir = tempdir().unwrap();
        let kernel = SharedContinuityKernel::open(dir.path()).unwrap();
        let scenario = scenario_for(BenchmarkClass::BaselineIsolation);
        let attach = kernel
            .attach_agent(AttachAgentInput {
                agent_id: "planner-strong".into(),
                agent_type: "ollama".into(),
                capabilities: vec!["plan".into()],
                namespace: scenario.namespace.clone(),
                role: Some("planner".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let context = kernel
            .open_context(OpenContextInput {
                namespace: scenario.namespace.clone(),
                task_id: scenario.task_id.clone(),
                session_id: "baseline-isolation-session".into(),
                objective: scenario.objective.clone(),
                selector: None,
                agent_id: Some("planner-strong".into()),
                attachment_id: Some(attach.id),
            })
            .unwrap();
        let _ = populate_scenario(&kernel, &context.id, &scenario).unwrap();

        let (envelope, continuity_path) = build_context_envelope(
            &kernel,
            BenchmarkClass::BaselineIsolation,
            BaselineKind::SharedContinuity,
            &context.id,
            &scenario,
            "relay-b",
            160,
            24,
            8,
        )
        .unwrap();

        let continuity_path = continuity_path.expect("shared continuity path");
        assert_eq!(continuity_path.kind, ContinuityPathKind::ReadContextOnly);
        assert_eq!(continuity_path.role, ContinuityPathRole::ExplicitControl);
        assert_eq!(continuity_path.proof_register_count, 0);
        assert!(!envelope.text.contains("Section: handoff_proof"));
        assert!(!envelope.text.contains("[pd1][handoff_proof][decision]"));
    }

    // -----------------------------------------------------------------------
    // Survival analysis tests
    // -----------------------------------------------------------------------

    fn test_truth() -> GroundTruth {
        GroundTruth {
            critical_facts: vec![
                TruthItem {
                    keywords: vec!["selector_missing", "src/query.rs"],
                    rationale_keywords: Vec::new(),
                    judge_note: None,
                    judge_required_concepts: Vec::new(),
                },
                TruthItem {
                    keywords: vec!["context", "primary"],
                    rationale_keywords: Vec::new(),
                    judge_note: None,
                    judge_required_concepts: Vec::new(),
                },
            ],
            constraints: vec![TruthItem {
                keywords: vec!["preserve", "provenance"],
                rationale_keywords: Vec::new(),
                judge_note: None,
                judge_required_concepts: Vec::new(),
            }],
            decisions: vec![TruthItem {
                keywords: vec!["unified", "continuity", "interface"],
                rationale_keywords: vec!["agent", "swap"],
                judge_note: None,
                judge_required_concepts: Vec::new(),
            }],
            scars: vec![TruthItem {
                keywords: vec!["naive", "probe"],
                rationale_keywords: Vec::new(),
                judge_note: None,
                judge_required_concepts: Vec::new(),
            }],
            avoid_repeating: Vec::new(),
            next_step_keywords: Vec::new(),
        }
    }

    fn test_envelope() -> ContextEnvelope {
        ContextEnvelope {
            provider: BaselineKind::SharedContinuity,
            retrieval_ms: 42,
            text: String::new(),
            token_estimate: 500,
            surfaced: vec![SurfacedItem {
                label: "f1".into(),
                support_type: "event".into(),
                support_id: "ev-001".into(),
                text: "selector_missing in src/query.rs".into(),
                has_provenance: true,
            }],
        }
    }

    #[test]
    fn survival_analysis_classifies_survived_and_lost() {
        let truth = test_truth();
        let envelope = test_envelope();
        let output = AgentContinuationOutput {
            critical_facts: vec![EvidenceNote {
                text: "The selector_missing bug is in src/query.rs".into(),
                evidence: vec!["f1".into()],
            }],
            constraints: vec![EvidenceNote {
                text: "Do NOT modify provenance chains; preserve provenance at all costs".into(),
                evidence: Vec::new(),
            }],
            decisions: vec![],
            operational_scars: vec![EvidenceNote {
                text: "Naive probe approach caused timeout cascade".into(),
                evidence: Vec::new(),
            }],
            ..Default::default()
        };

        let report = benchmark_survival_analysis(&output, &truth, &envelope);

        // critical_facts: first survived (selector_missing + src/query.rs), second lost (context + primary)
        assert_eq!(report.facts.total, 2);
        assert_eq!(report.facts.survived, 1);
        assert_eq!(report.facts.lost, 1);
        assert!((report.facts.rate - 0.5).abs() < f64::EPSILON);

        // constraints: survived (preserve + provenance)
        assert_eq!(report.constraints.total, 1);
        assert_eq!(report.constraints.survived, 1);

        // decisions: lost (no output decisions at all)
        assert_eq!(report.decisions.total, 1);
        assert_eq!(report.decisions.lost, 1);

        // scars: survived (naive + probe)
        assert_eq!(report.scars.total, 1);
        assert_eq!(report.scars.survived, 1);

        // total records
        assert_eq!(report.records.len(), 5);
    }

    #[test]
    fn survival_analysis_extracts_file_path_feature() {
        let truth = test_truth();
        let envelope = test_envelope();
        let output = AgentContinuationOutput {
            critical_facts: vec![EvidenceNote {
                text: "selector_missing in src/query.rs is the root cause".into(),
                evidence: vec!["f1".into()],
            }],
            ..Default::default()
        };

        let report = benchmark_survival_analysis(&output, &truth, &envelope);

        let fact_with_path = report
            .records
            .iter()
            .find(|r| {
                r.category == TruthCategory::CriticalFact && r.outcome == SurvivalOutcome::Survived
            })
            .unwrap();
        assert!(fact_with_path.features.contains_file_path);
        assert!(fact_with_path.features.matched_note_tokens.unwrap() > 0);
    }

    #[test]
    fn survival_analysis_detects_prohibition_framing() {
        let truth = test_truth();
        let envelope = test_envelope();
        let output = AgentContinuationOutput {
            constraints: vec![EvidenceNote {
                text: "You must NEVER break provenance. Always preserve the chain.".into(),
                evidence: Vec::new(),
            }],
            ..Default::default()
        };

        let report = benchmark_survival_analysis(&output, &truth, &envelope);

        let constraint = report
            .records
            .iter()
            .find(|r| {
                r.category == TruthCategory::Constraint && r.outcome == SurvivalOutcome::Survived
            })
            .unwrap();
        assert_eq!(constraint.features.prohibition_framing, Some(true));
        assert_eq!(constraint.features.aspiration_framing, Some(false));
    }

    #[test]
    fn survival_analysis_detects_aspiration_framing() {
        let truth = test_truth();
        let envelope = test_envelope();
        let output = AgentContinuationOutput {
            constraints: vec![EvidenceNote {
                text: "Try to preserve provenance where possible, consider the chain".into(),
                evidence: Vec::new(),
            }],
            ..Default::default()
        };

        let report = benchmark_survival_analysis(&output, &truth, &envelope);

        let constraint = report
            .records
            .iter()
            .find(|r| {
                r.category == TruthCategory::Constraint && r.outcome == SurvivalOutcome::Survived
            })
            .unwrap();
        assert_eq!(constraint.features.aspiration_framing, Some(true));
    }

    #[test]
    fn survival_analysis_decision_requires_evidence() {
        let truth = test_truth();
        let envelope = test_envelope();
        // Decision with right keywords but no matching evidence label -> LOST
        let output = AgentContinuationOutput {
            decisions: vec![DecisionNote {
                text: "Use the unified continuity interface for all operations".into(),
                rationale: "Agent swap requires consistent interface".into(),
                evidence: vec!["nonexistent-label".into()],
            }],
            ..Default::default()
        };

        let report = benchmark_survival_analysis(&output, &truth, &envelope);

        assert_eq!(report.decisions.total, 1);
        assert_eq!(report.decisions.lost, 1);
    }

    #[test]
    fn survival_analysis_empty_output_all_lost() {
        let truth = test_truth();
        let envelope = test_envelope();
        let output = AgentContinuationOutput::default();

        let report = benchmark_survival_analysis(&output, &truth, &envelope);

        assert_eq!(report.facts.lost, 2);
        assert_eq!(report.constraints.lost, 1);
        assert_eq!(report.decisions.lost, 1);
        assert_eq!(report.scars.lost, 1);
        assert_eq!(report.facts.rate, 0.0);
        assert!(
            report
                .records
                .iter()
                .all(|r| r.outcome == SurvivalOutcome::Lost)
        );
    }

    #[test]
    fn survival_analysis_category_stats_file_path_rates() {
        let truth = test_truth();
        let envelope = test_envelope();
        // Only the first fact (with file path keywords) survives
        let output = AgentContinuationOutput {
            critical_facts: vec![EvidenceNote {
                text: "selector_missing in src/query.rs".into(),
                evidence: Vec::new(),
            }],
            ..Default::default()
        };

        let report = benchmark_survival_analysis(&output, &truth, &envelope);

        // The survived fact has a file path keyword, the lost one doesn't
        assert_eq!(report.facts.file_path_rate_survived, 1.0);
        assert_eq!(report.facts.file_path_rate_lost, 0.0);
    }

    // -----------------------------------------------------------------------
    // Meta-lesson tests (Phase 2)
    // -----------------------------------------------------------------------

    #[test]
    fn chi_squared_2x2_known_values() {
        let (chi2, p) = chi_squared_2x2(30, 10, 10, 30);
        assert!(
            chi2 > 15.0,
            "chi2={chi2} should be > 15 for strong association"
        );
        assert!(p < 0.001, "p={p} should be highly significant");
    }

    #[test]
    fn chi_squared_2x2_zero_marginals() {
        let (chi2, p) = chi_squared_2x2(0, 0, 0, 0);
        assert_eq!(chi2, 0.0);
        assert_eq!(p, 1.0);

        let (chi2, p) = chi_squared_2x2(10, 0, 0, 10);
        assert!(chi2 > 0.0);
        assert!(p < 0.05);
    }

    #[test]
    fn chi_squared_no_association() {
        let (chi2, p) = chi_squared_2x2(25, 25, 25, 25);
        assert!(chi2 < 0.1, "chi2={chi2} should be ~0 for no association");
        assert!(p > 0.5, "p={p} should be non-significant");
    }

    #[test]
    fn erfc_approximation_accuracy() {
        assert!((erfc_approx(0.0) - 1.0).abs() < 0.001);
        assert!((erfc_approx(1.0) - 0.1573).abs() < 0.01);
        assert!((erfc_approx(2.0) - 0.00468).abs() < 0.001);
        assert!((erfc_approx(-1.0) - 1.8427).abs() < 0.01);
    }

    #[test]
    fn benjamini_hochberg_corrects_multiple_tests() {
        use super::benjamini_hochberg;
        // 5 tests: 3 significant raw, BH should keep fewer
        let raw = vec![0.001, 0.01, 0.04, 0.20, 0.80];
        let adj = benjamini_hochberg(&raw);
        // Adjusted p-values must be >= raw p-values
        for (r, a) in raw.iter().zip(adj.iter()) {
            assert!(*a >= *r, "adjusted {a} must be >= raw {r}");
        }
        // Adjusted p-values must be monotone with rank
        assert!(adj[0] <= adj[1]);
        // All adjusted <= 1.0
        for a in &adj {
            assert!(*a <= 1.0, "adjusted p must be <= 1.0");
        }
        // The clearly non-significant ones should stay non-significant
        assert!(adj[3] > 0.05);
        assert!(adj[4] > 0.05);
    }

    #[test]
    fn compare_feature_distributions_finds_signal() {
        use super::{ItemFeatures, SurvivalRecord};
        let mut records = Vec::new();
        for i in 0..20 {
            let has_fp = i < 10;
            let survived = if has_fp { i < 9 } else { i >= 18 };
            records.push(SurvivalRecord {
                category: TruthCategory::CriticalFact,
                outcome: if survived {
                    SurvivalOutcome::Survived
                } else {
                    SurvivalOutcome::Lost
                },
                keywords: vec!["test".into()],
                features: ItemFeatures {
                    keyword_count: 1,
                    keyword_char_length: 4,
                    contains_file_path: has_fp,
                    contains_numeric_ref: false,
                    matched_note_text: if survived {
                        Some("matched".into())
                    } else {
                        None
                    },
                    matched_note_tokens: if survived { Some(1) } else { None },
                    prohibition_framing: None,
                    aspiration_framing: None,
                    evidence_count: None,
                },
            });
        }
        // min_benchmark_classes=1 to allow testing with synthetic data
        let (lessons, _) = compare_feature_distributions(&records, 0.05, 1, 5);
        let fp_lesson = lessons.iter().find(|l| l.feature_name == "file_path");
        assert!(fp_lesson.is_some(), "should detect file_path signal");
        assert_eq!(fp_lesson.unwrap().direction, LessonDirection::SurvivedMore);
    }

    #[test]
    fn compare_feature_distributions_ignores_small_samples() {
        use super::{ItemFeatures, SurvivalRecord};
        let records: Vec<SurvivalRecord> = (0..5)
            .map(|i| SurvivalRecord {
                category: TruthCategory::Constraint,
                outcome: if i % 2 == 0 {
                    SurvivalOutcome::Survived
                } else {
                    SurvivalOutcome::Lost
                },
                keywords: vec!["test".into()],
                features: ItemFeatures {
                    keyword_count: 1,
                    keyword_char_length: 4,
                    contains_file_path: i < 3,
                    contains_numeric_ref: false,
                    matched_note_text: None,
                    matched_note_tokens: None,
                    prohibition_framing: None,
                    aspiration_framing: None,
                    evidence_count: None,
                },
            })
            .collect();
        let (lessons, _) = compare_feature_distributions(&records, 0.05, 1, 5);
        assert!(
            lessons.is_empty(),
            "should skip categories with < 10 records"
        );
    }

    #[test]
    fn compare_feature_distributions_requires_min_benchmark_classes() {
        use super::{ItemFeatures, SurvivalRecord};
        // 20 records with clear signal, but only 2 benchmark classes
        let mut records = Vec::new();
        for i in 0..20 {
            let has_fp = i < 10;
            let survived = if has_fp { i < 9 } else { i >= 18 };
            records.push(SurvivalRecord {
                category: TruthCategory::CriticalFact,
                outcome: if survived {
                    SurvivalOutcome::Survived
                } else {
                    SurvivalOutcome::Lost
                },
                keywords: vec!["test".into()],
                features: ItemFeatures {
                    keyword_count: 1,
                    keyword_char_length: 4,
                    contains_file_path: has_fp,
                    contains_numeric_ref: false,
                    matched_note_text: None,
                    matched_note_tokens: None,
                    prohibition_framing: None,
                    aspiration_framing: None,
                    evidence_count: None,
                },
            });
        }
        // With 2 classes (below min of 3), should produce nothing
        let (lessons, candidates) = compare_feature_distributions(&records, 0.05, 3, 2);
        assert!(
            lessons.is_empty(),
            "should reject with < 3 benchmark classes"
        );
        assert_eq!(candidates, 0, "should not even test candidates");

        // With 3 classes, should find the signal
        let (lessons, _) = compare_feature_distributions(&records, 0.05, 3, 3);
        assert!(
            lessons.iter().any(|l| l.feature_name == "file_path"),
            "should detect signal with >= 3 classes"
        );
    }

    #[test]
    fn generate_meta_lessons_empty_input() {
        let report = generate_meta_lessons(&[], 0.05);
        assert_eq!(report.total_records, 0);
        assert!(report.lessons.is_empty());
    }

    #[test]
    fn meta_lesson_report_serialization_roundtrip() {
        let report = MetaLessonReport {
            generated_at: "2026-03-26T00:00:00Z".into(),
            total_records: 100,
            candidates_tested: 16,
            lessons: vec![MetaLesson {
                pattern: "File paths survive 3x better".into(),
                feature_name: "file_path".into(),
                category: TruthCategory::CriticalFact,
                direction: LessonDirection::SurvivedMore,
                evidence: MetaLessonEvidence {
                    survived_with_feature: 9,
                    lost_with_feature: 1,
                    survived_without_feature: 1,
                    lost_without_feature: 9,
                    rate_with_feature: 0.9,
                    rate_without_feature: 0.1,
                    chi_squared: 12.8,
                    p_value: 0.0003,
                    adjusted_p_value: 0.005,
                    sparse_cells: false,
                },
                confidence: 0.995,
                sample_size: 20,
                benchmark_classes: 3,
            }],
        };
        let json = serde_json::to_string(&report).unwrap();
        let parsed: MetaLessonReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.lessons.len(), 1);
        assert_eq!(parsed.lessons[0].feature_name, "file_path");
    }

    #[test]
    fn meta_lesson_end_to_end_with_real_kernel() {
        let dir = tempdir().unwrap();
        let kernel = SharedContinuityKernel::open(dir.path()).unwrap();
        let scenario = scenario_for(BenchmarkClass::AgentSwapSurvival);

        let attach = kernel
            .attach_agent(AttachAgentInput {
                agent_id: "planner-strong".into(),
                agent_type: "ollama".into(),
                capabilities: vec!["plan".into()],
                namespace: scenario.namespace.clone(),
                role: Some("planner".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let context = kernel
            .open_context(OpenContextInput {
                namespace: scenario.namespace.clone(),
                task_id: scenario.task_id.clone(),
                session_id: "meta-lesson-e2e".into(),
                objective: scenario.objective.clone(),
                selector: None,
                agent_id: Some("planner-strong".into()),
                attachment_id: Some(attach.id),
            })
            .unwrap();
        populate_scenario(&kernel, &context.id, &scenario).unwrap();

        let (envelope, _) = build_context_envelope(
            &kernel,
            BenchmarkClass::AgentSwapSurvival,
            BaselineKind::SharedContinuity,
            &context.id,
            &scenario,
            "relay-a",
            512,
            24,
            8,
        )
        .unwrap();

        assert!(
            !envelope.surfaced.is_empty(),
            "real kernel must produce surfaced items"
        );

        // Partial output: some items survive, some lost
        let output = AgentContinuationOutput {
            summary: "Agent swap resume".into(),
            critical_facts: vec![EvidenceNote {
                text: "selector_missing in src/query.rs caused failure".into(),
                evidence: envelope
                    .surfaced
                    .iter()
                    .filter(|s| s.label.starts_with('f'))
                    .map(|s| s.label.clone())
                    .take(1)
                    .collect(),
            }],
            constraints: Vec::new(),
            decisions: Vec::new(),
            open_hypotheses: Vec::new(),
            operational_scars: vec![EvidenceNote {
                text: "naive probe approach failed".into(),
                evidence: envelope
                    .surfaced
                    .iter()
                    .filter(|s| s.label.starts_with('s'))
                    .map(|s| s.label.clone())
                    .take(1)
                    .collect(),
            }],
            avoid_repeating: Vec::new(),
            next_step: crate::adapters::ActionNote {
                text: "run benchmark adapter next".into(),
                evidence: Vec::new(),
            },
        };

        let survival = benchmark_survival_analysis(&output, &scenario.truth, &envelope);
        assert!(!survival.records.is_empty());

        let survived = survival
            .records
            .iter()
            .filter(|r| r.outcome == SurvivalOutcome::Survived)
            .count();
        let lost = survival
            .records
            .iter()
            .filter(|r| r.outcome == SurvivalOutcome::Lost)
            .count();
        assert!(survived > 0, "some items must survive");
        assert!(lost > 0, "some items must be lost (constraints omitted)");

        let suite = BenchmarkSuiteReport {
            generated_at: "2026-03-26T00:00:00Z".into(),
            config: config_with_classes(vec![BenchmarkClass::AgentSwapSurvival]),
            classes: vec![BenchmarkClassReport {
                class: BenchmarkClass::AgentSwapSurvival,
                scenario_id: scenario.id.clone(),
                continuity: BaselineRunReport {
                    baseline: BaselineKind::SharedContinuity,
                    status: BaselineStatus::Ok,
                    model: "test".into(),
                    retrieval_ms: envelope.retrieval_ms,
                    model_metrics: ModelCallMetrics::default(),
                    envelope_tokens: envelope.token_estimate,
                    evaluation: Evaluation::default(),
                    survival: Some(survival),
                    failure: None,
                    artifacts: Vec::new(),
                    continuity_path: None,
                },
                baselines: Vec::new(),
                metrics: BenchmarkMetrics::default(),
                resource: ResourceEnvelope::default(),
                artifacts: Vec::new(),
                hypothesis_injection: None,
            }],
            summary: BenchmarkSummary::default(),
            meta_lessons: None,
            phase3: None,
        };

        let meta = generate_meta_lessons(&[suite], 0.05);
        assert!(meta.total_records > 0, "must see survival records");

        let written = super::write_meta_lessons_to_kernel(&kernel, &context.id, &meta);
        assert!(
            written.is_ok(),
            "kernel write must succeed: {:?}",
            written.err()
        );

        let json = serde_json::to_string_pretty(&meta).unwrap();
        assert!(json.contains("total_records"));
    }

    // -----------------------------------------------------------------------
    // Phase 3 tests
    // -----------------------------------------------------------------------

    #[test]
    fn extract_eligible_hypotheses_filters_sparse_and_high_p() {
        let report = MetaLessonReport {
            generated_at: "2026-03-26T00:00:00Z".into(),
            total_records: 100,
            candidates_tested: 3,
            lessons: vec![
                MetaLesson {
                    pattern: "File paths survive better".into(),
                    feature_name: "file_path".into(),
                    category: TruthCategory::CriticalFact,
                    direction: LessonDirection::SurvivedMore,
                    evidence: MetaLessonEvidence {
                        survived_with_feature: 9,
                        lost_with_feature: 1,
                        survived_without_feature: 1,
                        lost_without_feature: 9,
                        rate_with_feature: 0.9,
                        rate_without_feature: 0.1,
                        chi_squared: 12.8,
                        p_value: 0.001,
                        adjusted_p_value: 0.003,
                        sparse_cells: false,
                    },
                    confidence: 0.99,
                    sample_size: 20,
                    benchmark_classes: 3,
                },
                MetaLesson {
                    pattern: "Sparse hypothesis".into(),
                    feature_name: "numeric_ref".into(),
                    category: TruthCategory::Constraint,
                    direction: LessonDirection::LostMore,
                    evidence: MetaLessonEvidence {
                        survived_with_feature: 2,
                        lost_with_feature: 1,
                        survived_without_feature: 1,
                        lost_without_feature: 2,
                        rate_with_feature: 0.67,
                        rate_without_feature: 0.33,
                        chi_squared: 1.0,
                        p_value: 0.3,
                        adjusted_p_value: 0.3,
                        sparse_cells: true,
                    },
                    confidence: 0.5,
                    sample_size: 6,
                    benchmark_classes: 3,
                },
                MetaLesson {
                    pattern: "High p-value".into(),
                    feature_name: "aspiration_framing".into(),
                    category: TruthCategory::Decision,
                    direction: LessonDirection::SurvivedMore,
                    evidence: MetaLessonEvidence {
                        survived_with_feature: 5,
                        lost_with_feature: 5,
                        survived_without_feature: 5,
                        lost_without_feature: 5,
                        rate_with_feature: 0.5,
                        rate_without_feature: 0.5,
                        chi_squared: 0.0,
                        p_value: 0.99,
                        adjusted_p_value: 0.99,
                        sparse_cells: false,
                    },
                    confidence: 0.01,
                    sample_size: 20,
                    benchmark_classes: 3,
                },
            ],
        };

        let eligible = extract_eligible_hypotheses(&report);
        assert_eq!(
            eligible.len(),
            1,
            "only the non-sparse, low-p hypothesis qualifies"
        );
        assert_eq!(eligible[0].feature_name, "file_path");
    }

    #[test]
    fn generate_phase3_report_promotes_strong_improvement() {
        let hypothesis = SurvivalHypothesis {
            feature_name: "file_path".into(),
            category: "CriticalFact".into(),
            direction: "SurvivedMore".into(),
            hint: "Include file paths".into(),
        };

        let control_survival = SurvivalReport {
            records: Vec::new(),
            facts: CategoryStats {
                total: 10,
                survived: 6,
                lost: 4,
                rate: 0.6,
                ..Default::default()
            },
            constraints: CategoryStats::default(),
            decisions: CategoryStats::default(),
            scars: CategoryStats::default(),
            surfaced_item_count: 0,
            surfaced_with_provenance: 0,
            total_envelope_tokens: 0,
        };
        let treatment_survival = SurvivalReport {
            records: Vec::new(),
            facts: CategoryStats {
                total: 10,
                survived: 9,
                lost: 1,
                rate: 0.9,
                ..Default::default()
            },
            constraints: CategoryStats::default(),
            decisions: CategoryStats::default(),
            scars: CategoryStats::default(),
            surfaced_item_count: 0,
            surfaced_with_provenance: 0,
            total_envelope_tokens: 0,
        };

        let class_results: Vec<Phase3ClassResult> = [
            BenchmarkClass::AgentSwapSurvival,
            BenchmarkClass::StrongToSmallContinuation,
            BenchmarkClass::SmallToSmallRelay,
        ]
        .iter()
        .map(|&class| Phase3ClassResult {
            class,
            control_survival: Some(control_survival.clone()),
            treatment_survival: Some(treatment_survival.clone()),
            hypotheses: vec![hypothesis.clone()],
            delta_ras: 0.3,
            delta_cfsr: 0.3,
            delta_csr: 0.0,
            delta_osr: 0.0,
        })
        .collect();

        let report = generate_phase3_report(&class_results, &[hypothesis], 1);
        assert_eq!(report.hypotheses_tested, 1);
        let result = &report.results[0];
        assert!((result.absolute_improvement - 0.3).abs() < 0.001);
        assert!(result.promoted);
        assert!(!result.rejected);
        assert!(!report.converged);
    }

    #[test]
    fn generate_phase3_report_rejects_negative_improvement() {
        let hypothesis = SurvivalHypothesis {
            feature_name: "aspiration_framing".into(),
            category: "Constraint".into(),
            direction: "SurvivedMore".into(),
            hint: "Use aspiration framing".into(),
        };

        let control = SurvivalReport {
            records: Vec::new(),
            facts: CategoryStats::default(),
            constraints: CategoryStats {
                total: 10,
                survived: 8,
                lost: 2,
                rate: 0.8,
                ..Default::default()
            },
            decisions: CategoryStats::default(),
            scars: CategoryStats::default(),
            surfaced_item_count: 0,
            surfaced_with_provenance: 0,
            total_envelope_tokens: 0,
        };
        let treatment = SurvivalReport {
            records: Vec::new(),
            facts: CategoryStats::default(),
            constraints: CategoryStats {
                total: 10,
                survived: 5,
                lost: 5,
                rate: 0.5,
                ..Default::default()
            },
            decisions: CategoryStats::default(),
            scars: CategoryStats::default(),
            surfaced_item_count: 0,
            surfaced_with_provenance: 0,
            total_envelope_tokens: 0,
        };

        let class_results: Vec<Phase3ClassResult> = [
            BenchmarkClass::AgentSwapSurvival,
            BenchmarkClass::StrongToSmallContinuation,
            BenchmarkClass::SmallToSmallRelay,
        ]
        .iter()
        .map(|&class| Phase3ClassResult {
            class,
            control_survival: Some(control.clone()),
            treatment_survival: Some(treatment.clone()),
            hypotheses: vec![hypothesis.clone()],
            delta_ras: -0.3,
            delta_cfsr: 0.0,
            delta_csr: -0.3,
            delta_osr: 0.0,
        })
        .collect();

        let report = generate_phase3_report(&class_results, &[hypothesis], 1);
        assert_eq!(report.rejected_count, 1);
        assert!(report.results[0].rejected);
        assert!(!report.results[0].promoted);
    }

    #[test]
    fn generate_phase3_report_converges_after_five_cycles() {
        let hypothesis = SurvivalHypothesis {
            feature_name: "file_path".into(),
            category: "CriticalFact".into(),
            direction: "SurvivedMore".into(),
            hint: "Include file paths".into(),
        };

        let same_survival = SurvivalReport {
            records: Vec::new(),
            facts: CategoryStats {
                total: 10,
                survived: 7,
                lost: 3,
                rate: 0.7,
                ..Default::default()
            },
            constraints: CategoryStats::default(),
            decisions: CategoryStats::default(),
            scars: CategoryStats::default(),
            surfaced_item_count: 0,
            surfaced_with_provenance: 0,
            total_envelope_tokens: 0,
        };

        let class_results: Vec<Phase3ClassResult> = [
            BenchmarkClass::AgentSwapSurvival,
            BenchmarkClass::StrongToSmallContinuation,
            BenchmarkClass::SmallToSmallRelay,
        ]
        .iter()
        .map(|&class| Phase3ClassResult {
            class,
            control_survival: Some(same_survival.clone()),
            treatment_survival: Some(same_survival.clone()),
            hypotheses: vec![hypothesis.clone()],
            delta_ras: 0.0,
            delta_cfsr: 0.0,
            delta_csr: 0.0,
            delta_osr: 0.0,
        })
        .collect();

        let report = generate_phase3_report(&class_results, &[hypothesis], 5);
        assert!(
            report.converged,
            "cycle 5 with no promotion should converge"
        );
    }

    #[test]
    fn phase3_validation_report_serialization_roundtrip() {
        let report = Phase3ValidationReport {
            generated_at: "2026-03-26T00:00:00Z".into(),
            cycle: 1,
            hypotheses_tested: 1,
            results: vec![HypothesisValidationResult {
                feature_name: "file_path".into(),
                category: TruthCategory::CriticalFact,
                direction: LessonDirection::SurvivedMore,
                control_survival_rate: 0.6,
                treatment_survival_rate: 0.9,
                absolute_improvement: 0.3,
                positive_classes: 3,
                total_classes: 3,
                promoted: true,
                rejected: false,
            }],
            promoted_count: 1,
            rejected_count: 0,
            converged: false,
        };

        let json = serde_json::to_string(&report).unwrap();
        let parsed: Phase3ValidationReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.results.len(), 1);
        assert_eq!(parsed.promoted_count, 1);
        assert!(!parsed.converged);
    }

    #[test]
    fn load_prior_hypotheses_returns_empty_for_missing_file() {
        let dir = tempdir().unwrap();
        let result = super::load_prior_hypotheses(dir.path());
        assert!(result.is_empty());
    }

    #[test]
    fn load_prior_hypotheses_parses_valid_meta_lessons() {
        let dir = tempdir().unwrap();
        let meta = MetaLessonReport {
            generated_at: "2026-03-26T00:00:00Z".into(),
            total_records: 20,
            candidates_tested: 1,
            lessons: vec![MetaLesson {
                pattern: "File paths survive better".into(),
                feature_name: "file_path".into(),
                category: TruthCategory::CriticalFact,
                direction: LessonDirection::SurvivedMore,
                evidence: MetaLessonEvidence {
                    survived_with_feature: 9,
                    lost_with_feature: 1,
                    survived_without_feature: 1,
                    lost_without_feature: 9,
                    rate_with_feature: 0.9,
                    rate_without_feature: 0.1,
                    chi_squared: 12.8,
                    p_value: 0.001,
                    adjusted_p_value: 0.003,
                    sparse_cells: false,
                },
                confidence: 0.99,
                sample_size: 20,
                benchmark_classes: 3,
            }],
        };
        let meta_path = dir.path().join("meta-lessons.json");
        std::fs::write(&meta_path, serde_json::to_vec_pretty(&meta).unwrap()).unwrap();

        let result = super::load_prior_hypotheses(dir.path());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].feature_name, "file_path");
    }

    #[test]
    fn detect_validation_cycle_returns_1_when_no_prior() {
        let dir = tempdir().unwrap();
        assert_eq!(super::detect_validation_cycle(dir.path()), 1);
    }

    #[test]
    fn detect_validation_cycle_increments_from_prior() {
        let dir = tempdir().unwrap();
        let prior = Phase3ValidationReport {
            generated_at: "2026-03-26T00:00:00Z".into(),
            cycle: 3,
            hypotheses_tested: 1,
            results: Vec::new(),
            promoted_count: 0,
            rejected_count: 0,
            converged: false,
        };
        let path = dir.path().join("phase3-report.json");
        std::fs::write(&path, serde_json::to_vec_pretty(&prior).unwrap()).unwrap();

        assert_eq!(super::detect_validation_cycle(dir.path()), 4);
    }

    #[test]
    fn phase3_class_result_serialization_roundtrip() {
        let result = Phase3ClassResult {
            class: BenchmarkClass::AgentSwapSurvival,
            control_survival: None,
            treatment_survival: None,
            hypotheses: vec![SurvivalHypothesis {
                feature_name: "file_path".into(),
                category: "CriticalFact".into(),
                direction: "SurvivedMore".into(),
                hint: "Include file paths".into(),
            }],
            delta_ras: 0.15,
            delta_cfsr: 0.10,
            delta_csr: 0.0,
            delta_osr: 0.0,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: Phase3ClassResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.class, BenchmarkClass::AgentSwapSurvival);
        assert!((parsed.delta_ras - 0.15).abs() < 0.001);
    }

    #[test]
    fn hypotheses_from_meta_lessons_filters_correctly() {
        use crate::adapters::hypotheses_from_meta_lessons;

        let lessons = vec![
            MetaLesson {
                pattern: "Good hypothesis".into(),
                feature_name: "file_path".into(),
                category: TruthCategory::CriticalFact,
                direction: LessonDirection::SurvivedMore,
                evidence: MetaLessonEvidence {
                    survived_with_feature: 9,
                    lost_with_feature: 1,
                    survived_without_feature: 1,
                    lost_without_feature: 9,
                    rate_with_feature: 0.9,
                    rate_without_feature: 0.1,
                    chi_squared: 12.8,
                    p_value: 0.001,
                    adjusted_p_value: 0.003,
                    sparse_cells: false,
                },
                confidence: 0.99,
                sample_size: 20,
                benchmark_classes: 3,
            },
            MetaLesson {
                pattern: "Sparse one".into(),
                feature_name: "numeric_ref".into(),
                category: TruthCategory::Constraint,
                direction: LessonDirection::LostMore,
                evidence: MetaLessonEvidence {
                    survived_with_feature: 2,
                    lost_with_feature: 1,
                    survived_without_feature: 1,
                    lost_without_feature: 2,
                    rate_with_feature: 0.67,
                    rate_without_feature: 0.33,
                    chi_squared: 1.0,
                    p_value: 0.01,
                    adjusted_p_value: 0.02,
                    sparse_cells: true,
                },
                confidence: 0.5,
                sample_size: 6,
                benchmark_classes: 3,
            },
        ];

        let result = hypotheses_from_meta_lessons(&lessons);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].feature_name, "file_path");
        assert_eq!(result[0].category, "CriticalFact");
    }

    // -----------------------------------------------------------------------
    // Phase 3: write_phase3_outcomes_to_kernel with real kernel
    // -----------------------------------------------------------------------

    #[test]
    fn write_phase3_outcomes_to_kernel_promotes_to_real_kernel() {
        let dir = tempdir().unwrap();
        let kernel = SharedContinuityKernel::open(dir.path()).unwrap();
        let attach = kernel
            .attach_agent(AttachAgentInput {
                agent_id: "validator".into(),
                agent_type: "test".into(),
                capabilities: vec![],
                namespace: "bench".into(),
                role: Some("validator".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let context = kernel
            .open_context(OpenContextInput {
                namespace: "bench".into(),
                task_id: "phase3-kernel-test".into(),
                session_id: "kernel-write-session".into(),
                objective: "test write_phase3_outcomes_to_kernel".into(),
                selector: None,
                agent_id: Some("validator".into()),
                attachment_id: Some(attach.id),
            })
            .unwrap();

        let report = Phase3ValidationReport {
            generated_at: "2026-03-26T00:00:00Z".into(),
            cycle: 2,
            hypotheses_tested: 2,
            results: vec![
                HypothesisValidationResult {
                    feature_name: "file_path".into(),
                    category: TruthCategory::CriticalFact,
                    direction: LessonDirection::SurvivedMore,
                    control_survival_rate: 0.40,
                    treatment_survival_rate: 0.55,
                    absolute_improvement: 0.15,
                    positive_classes: 3,
                    total_classes: 4,
                    promoted: true,
                    rejected: false,
                },
                HypothesisValidationResult {
                    feature_name: "numeric_ref".into(),
                    category: TruthCategory::Constraint,
                    direction: LessonDirection::SurvivedMore,
                    control_survival_rate: 0.50,
                    treatment_survival_rate: 0.48,
                    absolute_improvement: -0.02,
                    positive_classes: 1,
                    total_classes: 4,
                    promoted: false,
                    rejected: true,
                },
            ],
            promoted_count: 1,
            rejected_count: 1,
            converged: false,
        };

        let written = write_phase3_outcomes_to_kernel(&kernel, &context.id, &report);
        assert!(
            written.is_ok(),
            "kernel write must succeed: {:?}",
            written.err()
        );
        let records = written.unwrap();
        assert_eq!(records.len(), 1, "only promoted hypotheses get written");
        assert!(records[0].title.contains("file_path"));
        assert!(records[0].body.contains("VALIDATED"));
        assert!(records[0].body.contains("15.0%"));
    }

    // -----------------------------------------------------------------------
    // Phase 3: end-to-end with real kernel (no Ollama)
    // -----------------------------------------------------------------------

    #[test]
    fn phase3_end_to_end_with_real_kernel() {
        let dir = tempdir().unwrap();
        let kernel = SharedContinuityKernel::open(dir.path()).unwrap();
        let scenario = scenario_for(BenchmarkClass::AgentSwapSurvival);

        let attach = kernel
            .attach_agent(AttachAgentInput {
                agent_id: "planner-strong".into(),
                agent_type: "ollama".into(),
                capabilities: vec!["plan".into()],
                namespace: scenario.namespace.clone(),
                role: Some("planner".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let context = kernel
            .open_context(OpenContextInput {
                namespace: scenario.namespace.clone(),
                task_id: scenario.task_id.clone(),
                session_id: "phase3-e2e".into(),
                objective: scenario.objective.clone(),
                selector: None,
                agent_id: Some("planner-strong".into()),
                attachment_id: Some(attach.id),
            })
            .unwrap();
        populate_scenario(&kernel, &context.id, &scenario).unwrap();

        // Build a real envelope from kernel
        let (envelope, _) = build_context_envelope(
            &kernel,
            BenchmarkClass::AgentSwapSurvival,
            BaselineKind::SharedContinuity,
            &context.id,
            &scenario,
            "relay-a",
            512,
            24,
            8,
        )
        .unwrap();
        assert!(!envelope.surfaced.is_empty());

        // Simulate control arm: partial output (some survive, some lost)
        let control_output = AgentContinuationOutput {
            summary: "Control arm resume".into(),
            critical_facts: vec![EvidenceNote {
                text: "selector_missing in src/query.rs caused failure".into(),
                evidence: envelope
                    .surfaced
                    .iter()
                    .filter(|s| s.label.starts_with('f'))
                    .map(|s| s.label.clone())
                    .take(1)
                    .collect(),
            }],
            constraints: Vec::new(),
            decisions: Vec::new(),
            open_hypotheses: Vec::new(),
            operational_scars: Vec::new(),
            avoid_repeating: Vec::new(),
            next_step: crate::adapters::ActionNote {
                text: "run next benchmark".into(),
                evidence: Vec::new(),
            },
        };

        // Simulate treatment arm: better output (more items survive)
        let treatment_output = AgentContinuationOutput {
            summary: "Treatment arm resume with hints".into(),
            critical_facts: vec![
                EvidenceNote {
                    text: "selector_missing in src/query.rs caused failure".into(),
                    evidence: envelope
                        .surfaced
                        .iter()
                        .filter(|s| s.label.starts_with('f'))
                        .map(|s| s.label.clone())
                        .take(1)
                        .collect(),
                },
                EvidenceNote {
                    text: "Primary context is bench for this scenario".into(),
                    evidence: envelope
                        .surfaced
                        .iter()
                        .filter(|s| s.label.starts_with('f'))
                        .map(|s| s.label.clone())
                        .skip(1)
                        .take(1)
                        .collect(),
                },
            ],
            constraints: vec![EvidenceNote {
                text: "Preserve provenance across handoffs".into(),
                evidence: envelope
                    .surfaced
                    .iter()
                    .filter(|s| s.label.starts_with('k'))
                    .map(|s| s.label.clone())
                    .take(1)
                    .collect(),
            }],
            decisions: Vec::new(),
            open_hypotheses: Vec::new(),
            operational_scars: vec![EvidenceNote {
                text: "naive probe caused data loss".into(),
                evidence: envelope
                    .surfaced
                    .iter()
                    .filter(|s| s.label.starts_with('s'))
                    .map(|s| s.label.clone())
                    .take(1)
                    .collect(),
            }],
            avoid_repeating: Vec::new(),
            next_step: crate::adapters::ActionNote {
                text: "run benchmark adapter".into(),
                evidence: Vec::new(),
            },
        };

        // Evaluate both arms against real ground truth
        let control_eval = evaluate_output(&control_output, &scenario.truth, &envelope);
        let treatment_eval = evaluate_output(&treatment_output, &scenario.truth, &envelope);
        let control_survival =
            benchmark_survival_analysis(&control_output, &scenario.truth, &envelope);
        let treatment_survival =
            benchmark_survival_analysis(&treatment_output, &scenario.truth, &envelope);

        // Treatment should score better (more items matched)
        assert!(
            treatment_eval.resume_accuracy_score >= control_eval.resume_accuracy_score,
            "treatment ({}) should score >= control ({})",
            treatment_eval.resume_accuracy_score,
            control_eval.resume_accuracy_score
        );

        // Build Phase3ClassResult from real data
        let class_result = Phase3ClassResult {
            class: BenchmarkClass::AgentSwapSurvival,
            control_survival: Some(control_survival),
            treatment_survival: Some(treatment_survival),
            hypotheses: vec![SurvivalHypothesis {
                feature_name: "file_path".into(),
                category: "CriticalFact".into(),
                direction: "SurvivedMore".into(),
                hint: "Prefer items with file paths".into(),
            }],
            delta_ras: treatment_eval.resume_accuracy_score - control_eval.resume_accuracy_score,
            delta_cfsr: treatment_eval.critical_fact_survival_rate
                - control_eval.critical_fact_survival_rate,
            delta_csr: treatment_eval.constraint_survival_rate
                - control_eval.constraint_survival_rate,
            delta_osr: treatment_eval.operational_scar_retention
                - control_eval.operational_scar_retention,
        };

        // Duplicate across 3 classes to meet minimum threshold
        let class_results = vec![class_result.clone(), class_result.clone(), class_result];
        let hypotheses = vec![SurvivalHypothesis {
            feature_name: "file_path".into(),
            category: "CriticalFact".into(),
            direction: "SurvivedMore".into(),
            hint: "Prefer items with file paths".into(),
        }];

        let phase3_report = generate_phase3_report(&class_results, &hypotheses, 1);
        assert!(
            phase3_report.hypotheses_tested > 0,
            "must test at least one hypothesis"
        );

        // Write outcomes to real kernel
        let written = write_phase3_outcomes_to_kernel(&kernel, &context.id, &phase3_report);
        assert!(
            written.is_ok(),
            "kernel write must succeed: {:?}",
            written.err()
        );

        // Verify the full report serializes
        let json = serde_json::to_string_pretty(&phase3_report).unwrap();
        assert!(json.contains("hypotheses_tested"));
        assert!(json.contains("cycle"));
    }

    // -----------------------------------------------------------------------
    // Evaluation scoring logic
    // -----------------------------------------------------------------------

    #[test]
    fn evaluate_output_all_matched() {
        let truth = test_truth();
        let envelope = test_envelope();
        let output = AgentContinuationOutput {
            critical_facts: vec![
                EvidenceNote {
                    text: "selector_missing in src/query.rs".into(),
                    evidence: vec!["f1".into()],
                },
                EvidenceNote {
                    text: "Primary context for this scenario".into(),
                    evidence: vec!["f1".into()],
                },
            ],
            constraints: vec![EvidenceNote {
                text: "Must preserve provenance chains".into(),
                evidence: vec!["f1".into()],
            }],
            decisions: vec![DecisionNote {
                text: "Use the unified continuity interface".into(),
                rationale: "Agent swap requires consistent handling".into(),
                evidence: vec!["f1".into()],
            }],
            operational_scars: vec![EvidenceNote {
                text: "Naive probe approach was unreliable".into(),
                evidence: vec!["f1".into()],
            }],
            ..Default::default()
        };

        let eval = evaluate_output(&output, &truth, &envelope);
        assert!((eval.critical_fact_survival_rate - 1.0).abs() < f64::EPSILON);
        assert!((eval.constraint_survival_rate - 1.0).abs() < f64::EPSILON);
        assert!((eval.decision_lineage_fidelity - 1.0).abs() < f64::EPSILON);
        assert!((eval.operational_scar_retention - 1.0).abs() < f64::EPSILON);
        assert_eq!(eval.matched_critical_facts, 2);
        assert_eq!(eval.matched_constraints, 1);
        assert_eq!(eval.matched_decisions, 1);
        assert_eq!(eval.matched_scars, 1);
    }

    #[test]
    fn evaluate_output_empty_output_yields_zero_category_rates() {
        let truth = test_truth();
        let envelope = test_envelope();
        let output = AgentContinuationOutput::default();

        let eval = evaluate_output(&output, &truth, &envelope);
        assert_eq!(eval.critical_fact_survival_rate, 0.0);
        assert_eq!(eval.constraint_survival_rate, 0.0);
        assert_eq!(eval.decision_lineage_fidelity, 0.0);
        assert_eq!(eval.operational_scar_retention, 0.0);
        // resume_accuracy = (0 + 0 + 0 + 0 + next_step) / 5
        // next_step is 0 when no keywords match empty text, but truth has empty next_step_keywords
        // match_keywords("", &[]) returns true -> next_step = 1.0 -> ras = 0.2
        assert!((eval.resume_accuracy_score - 0.2).abs() < f64::EPSILON);
        assert_eq!(eval.memory_pollution_rate, 0.0);
    }

    #[test]
    fn evaluate_output_mistake_recurrence_detects_repeated_mistake() {
        let truth = GroundTruth {
            critical_facts: Vec::new(),
            constraints: Vec::new(),
            decisions: Vec::new(),
            scars: Vec::new(),
            avoid_repeating: vec![TruthItem {
                keywords: vec!["naive", "probe"],
                rationale_keywords: Vec::new(),
                judge_note: None,
                judge_required_concepts: Vec::new(),
            }],
            next_step_keywords: vec!["benchmark"],
        };
        let envelope = test_envelope();
        // next_step repeats the mistake keywords -> MRR = 1.0
        let output = AgentContinuationOutput {
            next_step: crate::adapters::ActionNote {
                text: "Run a naive probe on the query module".into(),
                evidence: Vec::new(),
            },
            ..Default::default()
        };

        let eval = evaluate_output(&output, &truth, &envelope);
        assert!((eval.mistake_recurrence_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn evaluate_output_mistake_recurrence_zero_when_avoid_matched() {
        let truth = GroundTruth {
            critical_facts: Vec::new(),
            constraints: Vec::new(),
            decisions: Vec::new(),
            scars: Vec::new(),
            avoid_repeating: vec![TruthItem {
                keywords: vec!["naive", "probe"],
                rationale_keywords: Vec::new(),
                judge_note: None,
                judge_required_concepts: Vec::new(),
            }],
            next_step_keywords: Vec::new(),
        };
        let envelope = test_envelope();
        let output = AgentContinuationOutput {
            avoid_repeating: vec![EvidenceNote {
                text: "Do not run naive probe tests".into(),
                evidence: Vec::new(),
            }],
            ..Default::default()
        };

        let eval = evaluate_output(&output, &truth, &envelope);
        assert_eq!(eval.mistake_recurrence_rate, 0.0);
    }

    #[test]
    fn evaluate_output_duplicate_work_detected() {
        let truth = GroundTruth {
            critical_facts: Vec::new(),
            constraints: Vec::new(),
            decisions: vec![TruthItem {
                keywords: vec!["unified", "interface"],
                rationale_keywords: Vec::new(),
                judge_note: None,
                judge_required_concepts: Vec::new(),
            }],
            scars: Vec::new(),
            avoid_repeating: Vec::new(),
            next_step_keywords: Vec::new(),
        };
        let envelope = test_envelope();
        let output = AgentContinuationOutput {
            next_step: crate::adapters::ActionNote {
                text: "Build the unified interface layer".into(),
                evidence: Vec::new(),
            },
            ..Default::default()
        };

        let eval = evaluate_output(&output, &truth, &envelope);
        assert!((eval.duplicate_work_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn evaluate_output_memory_pollution_with_unsupported_items() {
        let truth = test_truth();
        let envelope = test_envelope();
        let output = AgentContinuationOutput {
            critical_facts: vec![
                EvidenceNote {
                    text: "selector_missing in src/query.rs".into(),
                    evidence: Vec::new(),
                },
                EvidenceNote {
                    text: "Completely fabricated fact about aliens".into(),
                    evidence: Vec::new(),
                },
            ],
            ..Default::default()
        };

        let eval = evaluate_output(&output, &truth, &envelope);
        assert!(eval.memory_pollution_rate > 0.0);
        assert_eq!(eval.unsupported_items, 1);
    }

    #[test]
    fn evaluate_output_provenance_coverage() {
        let envelope = ContextEnvelope {
            provider: BaselineKind::SharedContinuity,
            retrieval_ms: 10,
            text: String::new(),
            token_estimate: 100,
            surfaced: vec![
                SurfacedItem {
                    label: "f1".into(),
                    support_type: "continuity".into(),
                    support_id: "ev-1".into(),
                    text: "item 1".into(),
                    has_provenance: true,
                },
                SurfacedItem {
                    label: "f2".into(),
                    support_type: "continuity".into(),
                    support_id: "ev-2".into(),
                    text: "item 2".into(),
                    has_provenance: false,
                },
            ],
        };
        let truth = GroundTruth {
            critical_facts: Vec::new(),
            constraints: Vec::new(),
            decisions: Vec::new(),
            scars: Vec::new(),
            avoid_repeating: Vec::new(),
            next_step_keywords: Vec::new(),
        };
        let output = AgentContinuationOutput::default();

        let eval = evaluate_output(&output, &truth, &envelope);
        assert!((eval.provenance_coverage - 0.5).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // score_notes / score_decisions
    // -----------------------------------------------------------------------

    #[test]
    fn score_notes_empty_truth_returns_zero() {
        let notes = vec![EvidenceNote {
            text: "some fact".into(),
            evidence: Vec::new(),
        }];
        let score = score_notes(&notes, &[]);
        assert_eq!(score.matched, 0);
        assert_eq!(score.rate, 0.0);
    }

    #[test]
    fn score_notes_empty_notes_returns_zero() {
        let truth = vec![TruthItem {
            keywords: vec!["foo"],
            rationale_keywords: Vec::new(),
            judge_note: None,
            judge_required_concepts: Vec::new(),
        }];
        let score = score_notes(&[], &truth);
        assert_eq!(score.matched, 0);
        assert_eq!(score.rate, 0.0);
    }

    #[test]
    fn score_decisions_requires_rationale_when_specified() {
        let truth = vec![TruthItem {
            keywords: vec!["unified", "interface"],
            rationale_keywords: vec!["agent", "swap"],
            judge_note: None,
            judge_required_concepts: Vec::new(),
        }];
        let labels: std::collections::HashSet<String> = ["f1".to_string()].into_iter().collect();

        // Has keywords but wrong rationale -> no match
        let notes = vec![DecisionNote {
            text: "Use the unified interface".into(),
            rationale: "Better performance".into(),
            evidence: vec!["f1".into()],
        }];
        let score = score_decisions(&notes, &truth, &labels);
        assert_eq!(score.matched, 0);

        // Has keywords and correct rationale -> match
        let notes = vec![DecisionNote {
            text: "Use the unified interface".into(),
            rationale: "Agent swap requires consistent handling".into(),
            evidence: vec!["f1".into()],
        }];
        let score = score_decisions(&notes, &truth, &labels);
        assert_eq!(score.matched, 1);
    }

    #[test]
    fn score_decisions_requires_evidence_label() {
        let truth = vec![TruthItem {
            keywords: vec!["unified"],
            rationale_keywords: Vec::new(),
            judge_note: None,
            judge_required_concepts: Vec::new(),
        }];
        let labels: std::collections::HashSet<String> = ["f1".to_string()].into_iter().collect();

        // Right keywords but wrong evidence label
        let notes = vec![DecisionNote {
            text: "Use the unified approach".into(),
            rationale: "".into(),
            evidence: vec!["f999".into()],
        }];
        let score = score_decisions(&notes, &truth, &labels);
        assert_eq!(score.matched, 0);
    }

    // -----------------------------------------------------------------------
    // unsupported_items / count_items
    // -----------------------------------------------------------------------

    #[test]
    fn unsupported_items_counts_unmatched_across_categories() {
        let truth = test_truth();
        let labels: std::collections::HashSet<String> = ["f1".to_string()].into_iter().collect();
        let output = AgentContinuationOutput {
            critical_facts: vec![EvidenceNote {
                text: "Fabricated nonsense".into(),
                evidence: Vec::new(),
            }],
            constraints: vec![EvidenceNote {
                text: "Made-up constraint".into(),
                evidence: Vec::new(),
            }],
            decisions: vec![DecisionNote {
                text: "Random decision".into(),
                rationale: "".into(),
                evidence: vec!["f1".into()],
            }],
            ..Default::default()
        };

        let count = unsupported_items(&output, &truth, &labels);
        assert_eq!(count, 3);
    }

    #[test]
    fn count_items_includes_all_categories() {
        let output = AgentContinuationOutput {
            critical_facts: vec![EvidenceNote {
                text: "fact".into(),
                evidence: Vec::new(),
            }],
            constraints: vec![EvidenceNote {
                text: "c".into(),
                evidence: Vec::new(),
            }],
            decisions: vec![DecisionNote {
                text: "d".into(),
                rationale: "".into(),
                evidence: Vec::new(),
            }],
            open_hypotheses: vec![EvidenceNote {
                text: "h".into(),
                evidence: Vec::new(),
            }],
            operational_scars: vec![EvidenceNote {
                text: "s".into(),
                evidence: Vec::new(),
            }],
            avoid_repeating: vec![EvidenceNote {
                text: "a".into(),
                evidence: Vec::new(),
            }],
            next_step: crate::adapters::ActionNote {
                text: "next".into(),
                evidence: Vec::new(),
            },
            ..Default::default()
        };
        assert_eq!(count_items(&output), 7);
    }

    #[test]
    fn count_items_empty_next_step_not_counted() {
        let output = AgentContinuationOutput {
            next_step: crate::adapters::ActionNote {
                text: "   ".into(),
                evidence: Vec::new(),
            },
            ..Default::default()
        };
        assert_eq!(count_items(&output), 0);
    }

    // -----------------------------------------------------------------------
    // category_rate
    // -----------------------------------------------------------------------

    #[test]
    fn category_rate_returns_correct_field() {
        let report = SurvivalReport {
            records: Vec::new(),
            facts: CategoryStats {
                rate: 0.8,
                ..Default::default()
            },
            constraints: CategoryStats {
                rate: 0.7,
                ..Default::default()
            },
            decisions: CategoryStats {
                rate: 0.6,
                ..Default::default()
            },
            scars: CategoryStats {
                rate: 0.5,
                ..Default::default()
            },
            surfaced_item_count: 0,
            surfaced_with_provenance: 0,
            total_envelope_tokens: 0,
        };
        assert!((category_rate(&report, TruthCategory::CriticalFact) - 0.8).abs() < f64::EPSILON);
        assert!((category_rate(&report, TruthCategory::Constraint) - 0.7).abs() < f64::EPSILON);
        assert!((category_rate(&report, TruthCategory::Decision) - 0.6).abs() < f64::EPSILON);
        assert!(
            (category_rate(&report, TruthCategory::OperationalScar) - 0.5).abs() < f64::EPSILON
        );
    }

    // -----------------------------------------------------------------------
    // has_sparse_cells
    // -----------------------------------------------------------------------

    #[test]
    fn has_sparse_cells_returns_true_for_zero_total() {
        assert!(has_sparse_cells(0, 0, 0, 0));
    }

    #[test]
    fn has_sparse_cells_returns_false_for_large_balanced_table() {
        assert!(!has_sparse_cells(20, 20, 20, 20));
    }

    #[test]
    fn has_sparse_cells_returns_true_for_small_expected() {
        // Table with small marginals -> expected cell < 5
        assert!(has_sparse_cells(1, 1, 1, 50));
    }

    #[test]
    fn has_sparse_cells_returns_true_for_unbalanced_margins() {
        assert!(has_sparse_cells(2, 0, 8, 10));
    }

    // -----------------------------------------------------------------------
    // summarize
    // -----------------------------------------------------------------------

    #[test]
    fn summarize_empty_classes() {
        let summary = summarize(&[]);
        assert_eq!(summary.class_count, 0);
        assert_eq!(summary.avg_cfsr, 0.0);
    }

    #[test]
    fn summarize_averages_metrics() {
        let classes = vec![
            BenchmarkClassReport {
                class: BenchmarkClass::AgentSwapSurvival,
                scenario_id: "s1".into(),
                continuity: BaselineRunReport {
                    baseline: BaselineKind::SharedContinuity,
                    status: BaselineStatus::Ok,
                    model: "test".into(),
                    retrieval_ms: 0,
                    model_metrics: ModelCallMetrics::default(),
                    envelope_tokens: 0,
                    evaluation: Evaluation::default(),
                    survival: None,
                    failure: None,
                    artifacts: Vec::new(),
                    continuity_path: None,
                },
                baselines: vec![BaselineRunReport {
                    baseline: BaselineKind::Isolated,
                    status: BaselineStatus::Failed,
                    model: "test".into(),
                    retrieval_ms: 0,
                    model_metrics: ModelCallMetrics::default(),
                    envelope_tokens: 0,
                    evaluation: Evaluation::default(),
                    survival: None,
                    failure: Some("boom".into()),
                    artifacts: Vec::new(),
                    continuity_path: None,
                }],
                metrics: BenchmarkMetrics {
                    cfsr: 0.8,
                    csr: 0.6,
                    dlf: 0.4,
                    ras: 0.7,
                    mpr: 0.1,
                    pc: 0.9,
                    smcl: 0.5,
                    sscr: 0.3,
                    cgd: 0.2,
                    ..Default::default()
                },
                resource: ResourceEnvelope::default(),
                artifacts: Vec::new(),
                hypothesis_injection: None,
            },
            BenchmarkClassReport {
                class: BenchmarkClass::StrongToSmallContinuation,
                scenario_id: "s2".into(),
                continuity: BaselineRunReport {
                    baseline: BaselineKind::SharedContinuity,
                    status: BaselineStatus::Ok,
                    model: "test".into(),
                    retrieval_ms: 0,
                    model_metrics: ModelCallMetrics::default(),
                    envelope_tokens: 0,
                    evaluation: Evaluation::default(),
                    survival: None,
                    failure: None,
                    artifacts: Vec::new(),
                    continuity_path: None,
                },
                baselines: Vec::new(),
                metrics: BenchmarkMetrics {
                    cfsr: 1.0,
                    csr: 1.0,
                    dlf: 1.0,
                    ras: 1.0,
                    mpr: 0.0,
                    pc: 1.0,
                    smcl: 1.0,
                    sscr: 1.0,
                    cgd: 1.0,
                    ..Default::default()
                },
                resource: ResourceEnvelope::default(),
                artifacts: Vec::new(),
                hypothesis_injection: None,
            },
        ];
        let summary = summarize(&classes);
        assert_eq!(summary.class_count, 2);
        assert!((summary.avg_cfsr - 0.9).abs() < f64::EPSILON);
        assert!((summary.avg_csr - 0.8).abs() < f64::EPSILON);
        assert_eq!(summary.failed_runs, 1);
        assert_eq!(summary.total_runs, 12);
    }

    // -----------------------------------------------------------------------
    // Helper functions
    // -----------------------------------------------------------------------

    #[test]
    fn ratio_or_zero_handles_zero_denominator() {
        assert_eq!(ratio_or_zero(5.0, 0.0), 0.0);
        assert_eq!(ratio_or_zero(5.0, -1.0), 0.0);
        assert!((ratio_or_zero(6.0, 3.0) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn inferred_hrt_tiers() {
        assert!((inferred_hrt(0.95) - 1.0).abs() < f64::EPSILON);
        assert!((inferred_hrt(0.9) - 1.0).abs() < f64::EPSILON);
        assert!((inferred_hrt(0.7) - 2.0).abs() < f64::EPSILON);
        assert!((inferred_hrt(0.6) - 2.0).abs() < f64::EPSILON);
        assert!((inferred_hrt(0.5) - 3.0).abs() < f64::EPSILON);
        assert!((inferred_hrt(0.0) - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn estimate_tokens_minimum_one() {
        assert_eq!(estimate_tokens(""), 1);
        assert_eq!(estimate_tokens("ab"), 1);
        assert!(estimate_tokens("Hello world, this is a longer string") > 1);
    }

    #[test]
    fn trim_text_under_max() {
        assert_eq!(trim_text("short", 100), "short");
    }

    #[test]
    fn trim_text_over_max_truncates() {
        let result = trim_text("abcdefghij", 5);
        assert_eq!(result, "abcde...");
    }

    #[test]
    fn match_keywords_case_insensitive() {
        assert!(match_keywords("Hello WORLD", &["hello", "world"]));
        assert!(!match_keywords("Hello", &["hello", "world"]));
    }

    #[test]
    fn match_keywords_empty_keywords() {
        assert!(match_keywords("anything", &[]));
    }

    #[test]
    fn flatten_keywords_concatenates_all_items() {
        let items = vec![
            TruthItem {
                keywords: vec!["a", "b"],
                rationale_keywords: Vec::new(),
                judge_note: None,
                judge_required_concepts: Vec::new(),
            },
            TruthItem {
                keywords: vec!["c"],
                rationale_keywords: Vec::new(),
                judge_note: None,
                judge_required_concepts: Vec::new(),
            },
        ];
        assert_eq!(flatten_keywords(&items), vec!["a", "b", "c"]);
    }

    #[test]
    fn primary_note_text_strips_after_double_colon() {
        assert_eq!(primary_note_text("Title :: body text"), "Title");
        assert_eq!(primary_note_text("no separator"), "no separator");
        // " :: body" -> split_once("::") -> (" ", " body") -> title = " ".trim() = ""
        // -> empty so falls back to original text trimmed
        assert_eq!(primary_note_text(" :: body"), ":: body");
    }

    #[test]
    fn strip_placeholder_title_removes_decision_prefix() {
        assert_eq!(
            strip_placeholder_title("decision_text :: The actual title"),
            Some("The actual title".to_string())
        );
        assert_eq!(strip_placeholder_title("Normal title :: rationale"), None);
        assert_eq!(strip_placeholder_title("no separator"), None);
    }

    #[test]
    fn split_title_and_rationale_extracts_both() {
        assert_eq!(
            split_title_and_rationale("Title :: Rationale"),
            Some(("Title".to_string(), "Rationale".to_string()))
        );
        assert_eq!(split_title_and_rationale("no separator"), None);
        assert_eq!(split_title_and_rationale(" :: Rationale"), None);
        assert_eq!(split_title_and_rationale("Title :: "), None);
    }

    #[test]
    fn split_title_and_rationale_strips_known_prefixes() {
        assert_eq!(
            split_title_and_rationale(
                "decision_text: Actual Title :: rationale_text: Real Rationale"
            ),
            Some(("Actual Title".to_string(), "Real Rationale".to_string()))
        );
    }

    #[test]
    fn looks_like_placeholder_detects_patterns() {
        assert!(looks_like_placeholder("decision_text"));
        assert!(looks_like_placeholder("Decision Text Here"));
        assert!(looks_like_placeholder("rationale_text something"));
        assert!(!looks_like_placeholder("Use the unified interface"));
    }

    #[test]
    fn normalized_text_strips_non_alphanumeric() {
        assert_eq!(normalized_text("Hello, World! 123"), "helloworld123");
        assert_eq!(normalized_text(""), "");
    }

    #[test]
    fn normalized_decision_probe_prefers_rationale() {
        let decision = DecisionNote {
            text: "decision text".into(),
            rationale: "the rationale".into(),
            evidence: Vec::new(),
        };
        assert_eq!(normalized_decision_probe(&decision), "therationale");

        let decision_empty_rationale = DecisionNote {
            text: "the text".into(),
            rationale: "  ".into(),
            evidence: Vec::new(),
        };
        assert_eq!(
            normalized_decision_probe(&decision_empty_rationale),
            "thetext"
        );
    }

    #[test]
    fn failed_evaluation_preserves_provenance() {
        let envelope = ContextEnvelope {
            provider: BaselineKind::SharedContinuity,
            retrieval_ms: 0,
            text: String::new(),
            token_estimate: 10,
            surfaced: vec![
                SurfacedItem {
                    label: "f1".into(),
                    support_type: "c".into(),
                    support_id: "e1".into(),
                    text: "a".into(),
                    has_provenance: true,
                },
                SurfacedItem {
                    label: "f2".into(),
                    support_type: "c".into(),
                    support_id: "e2".into(),
                    text: "b".into(),
                    has_provenance: true,
                },
                SurfacedItem {
                    label: "f3".into(),
                    support_type: "c".into(),
                    support_id: "e3".into(),
                    text: "c".into(),
                    has_provenance: false,
                },
            ],
        };
        let eval = failed_evaluation(&envelope);
        assert!((eval.provenance_coverage - 2.0 / 3.0).abs() < f64::EPSILON);
        assert_eq!(eval.critical_fact_survival_rate, 0.0);
    }

    #[test]
    fn format_continuity_path_label_handoff_proof() {
        let path = ContinuityPathReport {
            kind: ContinuityPathKind::HandoffProof,
            role: ContinuityPathRole::ProofPath,
            proof_register_count: 5,
            proof_register_labels: Vec::new(),
        };
        assert_eq!(format_continuity_path_label(&path), "handoff-proof(5)");
    }

    #[test]
    fn format_continuity_path_label_read_context_control() {
        let path = ContinuityPathReport {
            kind: ContinuityPathKind::ReadContextOnly,
            role: ContinuityPathRole::ExplicitControl,
            proof_register_count: 0,
            proof_register_labels: Vec::new(),
        };
        assert_eq!(format_continuity_path_label(&path), "read-context-control");
    }

    #[test]
    fn format_continuity_path_label_legacy() {
        let path = ContinuityPathReport {
            kind: ContinuityPathKind::ReadContextOnly,
            role: ContinuityPathRole::Legacy,
            proof_register_count: 0,
            proof_register_labels: Vec::new(),
        };
        assert_eq!(format_continuity_path_label(&path), "read-context-only");
    }

    #[test]
    fn section_prefix_returns_known_prefixes() {
        assert_eq!(section_prefix("constraints"), "k");
        assert_eq!(section_prefix("decisions"), "d");
        assert_eq!(section_prefix("hypotheses"), "h");
        assert_eq!(section_prefix("incidents"), "i");
        assert_eq!(section_prefix("operational_scars"), "s");
        assert_eq!(section_prefix("open_threads"), "t");
        assert_eq!(section_prefix("unknown"), "x");
    }

    // -----------------------------------------------------------------------
    // Phase 3 edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn generate_phase3_report_skips_unknown_category() {
        let hypothesis = SurvivalHypothesis {
            feature_name: "file_path".into(),
            category: "UnknownCategory".into(),
            direction: "SurvivedMore".into(),
            hint: "Include file paths".into(),
        };

        let class_results = vec![Phase3ClassResult {
            class: BenchmarkClass::AgentSwapSurvival,
            control_survival: None,
            treatment_survival: None,
            hypotheses: Vec::new(),
            delta_ras: 0.0,
            delta_cfsr: 0.0,
            delta_csr: 0.0,
            delta_osr: 0.0,
        }];

        let report = generate_phase3_report(&class_results, &[hypothesis], 1);
        assert_eq!(report.hypotheses_tested, 0);
    }

    #[test]
    fn generate_phase3_report_handles_fewer_than_three_classes() {
        let hypothesis = SurvivalHypothesis {
            feature_name: "file_path".into(),
            category: "CriticalFact".into(),
            direction: "SurvivedMore".into(),
            hint: "Include file paths".into(),
        };

        let survival = SurvivalReport {
            records: Vec::new(),
            facts: CategoryStats {
                rate: 0.9,
                ..Default::default()
            },
            constraints: CategoryStats::default(),
            decisions: CategoryStats::default(),
            scars: CategoryStats::default(),
            surfaced_item_count: 0,
            surfaced_with_provenance: 0,
            total_envelope_tokens: 0,
        };

        // Only 2 classes -> cannot promote (needs >= 3)
        let class_results = vec![
            Phase3ClassResult {
                class: BenchmarkClass::AgentSwapSurvival,
                control_survival: Some(SurvivalReport {
                    facts: CategoryStats {
                        rate: 0.5,
                        ..Default::default()
                    },
                    ..survival.clone()
                }),
                treatment_survival: Some(survival.clone()),
                hypotheses: vec![hypothesis.clone()],
                delta_ras: 0.0,
                delta_cfsr: 0.0,
                delta_csr: 0.0,
                delta_osr: 0.0,
            },
            Phase3ClassResult {
                class: BenchmarkClass::StrongToSmallContinuation,
                control_survival: Some(SurvivalReport {
                    facts: CategoryStats {
                        rate: 0.5,
                        ..Default::default()
                    },
                    ..survival.clone()
                }),
                treatment_survival: Some(survival.clone()),
                hypotheses: vec![hypothesis.clone()],
                delta_ras: 0.0,
                delta_cfsr: 0.0,
                delta_csr: 0.0,
                delta_osr: 0.0,
            },
        ];

        let report = generate_phase3_report(&class_results, &[hypothesis], 1);
        assert!(
            !report.results[0].promoted,
            "cannot promote with fewer than 3 classes"
        );
    }

    #[test]
    fn generate_phase3_report_direction_lost_more() {
        let hypothesis = SurvivalHypothesis {
            feature_name: "numeric_ref".into(),
            category: "Constraint".into(),
            direction: "LostMore".into(),
            hint: "Avoid numeric refs".into(),
        };

        let class_results: Vec<Phase3ClassResult> = [
            BenchmarkClass::AgentSwapSurvival,
            BenchmarkClass::StrongToSmallContinuation,
            BenchmarkClass::SmallToSmallRelay,
        ]
        .iter()
        .map(|&class| Phase3ClassResult {
            class,
            control_survival: Some(SurvivalReport {
                records: Vec::new(),
                facts: CategoryStats::default(),
                constraints: CategoryStats {
                    rate: 0.4,
                    ..Default::default()
                },
                decisions: CategoryStats::default(),
                scars: CategoryStats::default(),
                surfaced_item_count: 0,
                surfaced_with_provenance: 0,
                total_envelope_tokens: 0,
            }),
            treatment_survival: Some(SurvivalReport {
                records: Vec::new(),
                facts: CategoryStats::default(),
                constraints: CategoryStats {
                    rate: 0.2,
                    ..Default::default()
                },
                decisions: CategoryStats::default(),
                scars: CategoryStats::default(),
                surfaced_item_count: 0,
                surfaced_with_provenance: 0,
                total_envelope_tokens: 0,
            }),
            hypotheses: vec![hypothesis.clone()],
            delta_ras: 0.0,
            delta_cfsr: 0.0,
            delta_csr: 0.0,
            delta_osr: 0.0,
        })
        .collect();

        let report = generate_phase3_report(&class_results, &[hypothesis], 1);
        assert_eq!(report.results[0].direction, LessonDirection::LostMore);
        assert!(report.results[0].rejected);
    }

    #[test]
    fn generate_phase3_report_no_convergence_before_cycle_five() {
        let hypothesis = SurvivalHypothesis {
            feature_name: "file_path".into(),
            category: "CriticalFact".into(),
            direction: "SurvivedMore".into(),
            hint: "test".into(),
        };
        let report = generate_phase3_report(&[], &[hypothesis], 4);
        assert!(!report.converged);
    }

    // -----------------------------------------------------------------------
    // benjamini_hochberg edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn benjamini_hochberg_empty_input() {
        use super::benjamini_hochberg;
        assert!(benjamini_hochberg(&[]).is_empty());
    }

    #[test]
    fn benjamini_hochberg_single_value() {
        use super::benjamini_hochberg;
        let adj = benjamini_hochberg(&[0.03]);
        assert_eq!(adj.len(), 1);
        assert!((adj[0] - 0.03).abs() < f64::EPSILON);
    }

    #[test]
    fn benjamini_hochberg_all_significant() {
        use super::benjamini_hochberg;
        let raw = vec![0.001, 0.002, 0.003];
        let adj = benjamini_hochberg(&raw);
        for a in &adj {
            assert!(*a < 0.05, "all should remain significant, got {a}");
        }
    }

    // -----------------------------------------------------------------------
    // Survival record feature extraction
    // -----------------------------------------------------------------------

    #[test]
    fn survival_record_extracts_numeric_ref() {
        let truth = GroundTruth {
            critical_facts: vec![TruthItem {
                keywords: vec!["error", "42"],
                rationale_keywords: Vec::new(),
                judge_note: None,
                judge_required_concepts: Vec::new(),
            }],
            constraints: Vec::new(),
            decisions: Vec::new(),
            scars: Vec::new(),
            avoid_repeating: Vec::new(),
            next_step_keywords: Vec::new(),
        };
        let envelope = test_envelope();
        let output = AgentContinuationOutput::default();

        let report = benchmark_survival_analysis(&output, &truth, &envelope);
        let record = &report.records[0];
        assert!(record.features.contains_numeric_ref);
    }

    #[test]
    fn survival_record_file_extensions() {
        let truth = GroundTruth {
            critical_facts: vec![
                TruthItem {
                    keywords: vec!["app.py"],
                    rationale_keywords: Vec::new(),
                    judge_note: None,
                    judge_required_concepts: Vec::new(),
                },
                TruthItem {
                    keywords: vec!["main.ts"],
                    rationale_keywords: Vec::new(),
                    judge_note: None,
                    judge_required_concepts: Vec::new(),
                },
                TruthItem {
                    keywords: vec!["server.go"],
                    rationale_keywords: Vec::new(),
                    judge_note: None,
                    judge_required_concepts: Vec::new(),
                },
            ],
            constraints: Vec::new(),
            decisions: Vec::new(),
            scars: Vec::new(),
            avoid_repeating: Vec::new(),
            next_step_keywords: Vec::new(),
        };
        let envelope = test_envelope();
        let output = AgentContinuationOutput::default();

        let report = benchmark_survival_analysis(&output, &truth, &envelope);
        for record in &report.records {
            assert!(
                record.features.contains_file_path,
                "should detect file path for keywords {:?}",
                record.keywords
            );
        }
    }

    // -----------------------------------------------------------------------
    // BenchmarkClass and BaselineKind slug coverage
    // -----------------------------------------------------------------------

    #[test]
    fn benchmark_class_all_covers_all_variants() {
        let all = BenchmarkClass::all();
        assert_eq!(all.len(), 10);
    }

    #[test]
    fn benchmark_class_slug_roundtrip() {
        for class in BenchmarkClass::all() {
            let slug = class.slug();
            assert!(!slug.is_empty(), "slug should not be empty for {class:?}");
        }
    }

    #[test]
    fn baseline_kind_slug_all_variants() {
        let kinds = [
            BaselineKind::SharedContinuity,
            BaselineKind::Isolated,
            BaselineKind::RecentWindow,
            BaselineKind::VectorOnly,
            BaselineKind::RollingSummary,
            BaselineKind::FullTranscript,
        ];
        let slugs: Vec<_> = kinds.iter().map(|k| k.slug()).collect();
        assert_eq!(slugs.len(), 6);
        assert!(slugs.contains(&"shared-continuity"));
        assert!(slugs.contains(&"isolated"));
        assert!(slugs.contains(&"recent-window"));
        assert!(slugs.contains(&"vector-only"));
        assert!(slugs.contains(&"rolling-summary"));
        assert!(slugs.contains(&"full-transcript"));
    }

    #[test]
    fn benchmark_class_uses_handoff_proof_coverage() {
        use super::BenchmarkClass::*;
        // BaselineIsolation does NOT use handoff proof
        assert!(!BaselineIsolation.uses_handoff_proof());
        // All others do
        for class in [
            AgentSwapSurvival,
            StrongToSmallContinuation,
            SmallToSmallRelay,
            InterruptionStress,
            OperationalScar,
            CrossAgentCollaborative,
            CrashRecovery,
            MemoryPollution,
            ContextBudgetCompression,
        ] {
            assert!(
                class.uses_handoff_proof(),
                "{class:?} should use handoff proof"
            );
        }
    }

    #[test]
    fn benchmark_class_shared_path_role_coverage() {
        use super::ContinuityPathRole::*;
        assert_eq!(
            BenchmarkClass::AgentSwapSurvival.shared_path_role(),
            ProofPath
        );
        assert_eq!(
            BenchmarkClass::BaselineIsolation.shared_path_role(),
            ExplicitControl
        );
    }

    // -----------------------------------------------------------------------
    // generate_meta_lessons with multiple suite reports
    // -----------------------------------------------------------------------

    #[test]
    fn generate_meta_lessons_multiple_suites_aggregates_records() {
        use super::ItemFeatures;
        // Build two suite reports with survival records
        let make_survival = |records: Vec<super::SurvivalRecord>| SurvivalReport {
            facts: super::category_stats(&records, TruthCategory::CriticalFact),
            constraints: super::category_stats(&records, TruthCategory::Constraint),
            decisions: super::category_stats(&records, TruthCategory::Decision),
            scars: super::category_stats(&records, TruthCategory::OperationalScar),
            records,
            surfaced_item_count: 0,
            surfaced_with_provenance: 0,
            total_envelope_tokens: 0,
        };

        let make_records = || {
            (0..6)
                .map(|i| super::SurvivalRecord {
                    category: TruthCategory::CriticalFact,
                    outcome: if i < 4 {
                        SurvivalOutcome::Survived
                    } else {
                        SurvivalOutcome::Lost
                    },
                    keywords: vec!["test".into()],
                    features: ItemFeatures {
                        keyword_count: 1,
                        keyword_char_length: 4,
                        contains_file_path: i < 3,
                        contains_numeric_ref: false,
                        matched_note_text: None,
                        matched_note_tokens: None,
                        prohibition_framing: None,
                        aspiration_framing: None,
                        evidence_count: None,
                    },
                })
                .collect::<Vec<_>>()
        };

        let make_suite = |class: BenchmarkClass| BenchmarkSuiteReport {
            generated_at: "2026-03-26T00:00:00Z".into(),
            config: config_with_classes(vec![class]),
            classes: vec![BenchmarkClassReport {
                class,
                scenario_id: class.slug().into(),
                continuity: BaselineRunReport {
                    baseline: BaselineKind::SharedContinuity,
                    status: BaselineStatus::Ok,
                    model: "test".into(),
                    retrieval_ms: 0,
                    model_metrics: ModelCallMetrics::default(),
                    envelope_tokens: 0,
                    evaluation: Evaluation::default(),
                    survival: Some(make_survival(make_records())),
                    failure: None,
                    artifacts: Vec::new(),
                    continuity_path: None,
                },
                baselines: Vec::new(),
                metrics: BenchmarkMetrics::default(),
                resource: ResourceEnvelope::default(),
                artifacts: Vec::new(),
                hypothesis_injection: None,
            }],
            summary: BenchmarkSummary::default(),
            meta_lessons: None,
            phase3: None,
        };

        let suites = vec![
            make_suite(BenchmarkClass::AgentSwapSurvival),
            make_suite(BenchmarkClass::StrongToSmallContinuation),
            make_suite(BenchmarkClass::SmallToSmallRelay),
        ];
        let report = generate_meta_lessons(&suites, 0.10);
        // 3 suites * 6 records = 18 total records
        assert_eq!(report.total_records, 18);
    }

    // -----------------------------------------------------------------------
    // scenario_for covers all classes
    // -----------------------------------------------------------------------

    #[test]
    fn scenario_for_all_classes_produces_valid_scenarios() {
        for class in BenchmarkClass::all() {
            let scenario = scenario_for(class);
            assert!(
                !scenario.id.is_empty(),
                "scenario id should not be empty for {class:?}"
            );
            assert!(
                !scenario.truth.critical_facts.is_empty(),
                "scenario should have critical facts for {class:?}"
            );
            assert!(
                !scenario.phases.is_empty(),
                "scenario should have phases for {class:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Chi-squared with moderate association
    // -----------------------------------------------------------------------

    #[test]
    fn chi_squared_moderate_association() {
        let (chi2, p) = chi_squared_2x2(15, 5, 5, 15);
        assert!(chi2 > 5.0, "moderate association chi2={chi2}");
        assert!(p < 0.05, "should be significant p={p}");
    }

    #[test]
    fn chi_squared_one_empty_row() {
        let (chi2, p) = chi_squared_2x2(0, 0, 10, 10);
        assert_eq!(chi2, 0.0);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn chi_squared_one_empty_column() {
        let (chi2, p) = chi_squared_2x2(10, 0, 10, 0);
        assert_eq!(chi2, 0.0);
        assert_eq!(p, 1.0);
    }
}
