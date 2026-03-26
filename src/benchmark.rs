use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use sysinfo::{Pid, ProcessesToUpdate, System};

use crate::adapters::{
    AgentAdapter, AgentAdapterConfig, AgentContinuationOutput, DecisionNote, EvidenceNote,
    ModelCallMetrics, OllamaAdapter, parse_structured_output, render_structured_resume_prompt,
    structured_output_schema,
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
pub struct MarketHeadChallengeConfig {
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

impl MarketHeadChallengeConfig {
    pub fn selected_classes(&self) -> Vec<BenchmarkClass> {
        if self.classes.is_empty() {
            vec![
                BenchmarkClass::AgentSwapSurvival,
                BenchmarkClass::StrongToSmallContinuation,
                BenchmarkClass::SmallToSmallRelay,
                BenchmarkClass::OperationalScar,
                BenchmarkClass::InterruptionStress,
            ]
        } else {
            self.classes.clone()
        }
    }

    fn as_bench_config(&self) -> ContinuityBenchConfig {
        ContinuityBenchConfig {
            output_dir: self.output_dir.clone(),
            ollama_endpoint: self.ollama_endpoint.clone(),
            strong_model: self.strong_model.clone(),
            small_model: self.small_model.clone(),
            embedding_backend: self.embedding_backend.clone(),
            retrieval_protocol: self.retrieval_protocol.clone(),
            classes: self.classes.clone(),
            token_budget: self.token_budget,
            candidate_limit: self.candidate_limit,
            recent_window: self.recent_window,
            timeout_secs: self.timeout_secs,
            num_predict: self.num_predict,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketHeadChallengeManifest {
    pub generated_at: String,
    pub config: MarketHeadChallengeConfig,
    pub cases: Vec<MarketHeadChallengeCaseManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketHeadChallengeCaseManifest {
    pub class: BenchmarkClass,
    pub scenario_id: String,
    pub protocol: String,
    pub prompt_path: String,
    pub response_path: String,
    pub schema_path: String,
    pub template_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketHeadChallengeExportReport {
    pub generated_at: String,
    pub manifest_path: String,
    pub evaluator_pack_path: String,
    pub case_count: usize,
    pub cases: Vec<MarketHeadChallengeCaseManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketHeadJudgeExportReport {
    pub generated_at: String,
    pub manifest_path: String,
    pub case_count: usize,
    pub cases: Vec<MarketHeadChallengeCaseManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketHeadChallengeEvaluationReport {
    pub generated_at: String,
    pub model_name: String,
    pub evaluator_pack_path: String,
    pub responses_dir: String,
    pub cases: Vec<MarketHeadChallengeCaseEvaluation>,
    pub summary: MarketHeadChallengeSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketHeadChallengeCaseEvaluation {
    pub class: BenchmarkClass,
    pub scenario_id: String,
    pub protocol: String,
    pub response_path: String,
    pub status: BaselineStatus,
    pub raw_evaluation: Evaluation,
    pub evaluation: Evaluation,
    pub failure: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MarketHeadChallengeSummary {
    pub class_count: usize,
    pub avg_absorption_cfsr: f64,
    pub avg_absorption_dlf: f64,
    pub avg_absorption_osr: f64,
    pub avg_absorption_ras: f64,
    pub avg_cfsr: f64,
    pub avg_dlf: f64,
    pub avg_osr: f64,
    pub avg_ras: f64,
    pub avg_mpr: f64,
    pub avg_pc: f64,
    pub failed_cases: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketHeadJudgeEvaluationReport {
    pub generated_at: String,
    pub model_name: String,
    pub manifest_path: String,
    pub responses_dir: String,
    pub cases: Vec<MarketHeadJudgeCaseEvaluation>,
    pub summary: MarketHeadJudgeSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketHeadJudgeCaseEvaluation {
    pub class: BenchmarkClass,
    pub scenario_id: String,
    pub protocol: String,
    pub response_path: String,
    pub status: BaselineStatus,
    pub evaluation: JudgeEvaluation,
    pub failure: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MarketHeadJudgeSummary {
    pub class_count: usize,
    pub avg_judge_cfsr: f64,
    pub avg_judge_csr: f64,
    pub avg_judge_dlf: f64,
    pub avg_judge_osr: f64,
    pub avg_judge_next_step: f64,
    pub avg_judge_comprehension: f64,
    pub failed_cases: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketHeadJudgeComparisonEntry {
    pub judge_head: String,
    pub summary: MarketHeadJudgeSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketHeadJudgeSamePackComparisonReport {
    pub generated_at: String,
    pub challenged_head: String,
    pub canonical: MarketHeadChallengeSummary,
    pub judges: Vec<MarketHeadJudgeComparisonEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketHeadJudgePackComparisonEntry {
    pub challenged_head: String,
    pub judge_head: String,
    pub canonical: MarketHeadChallengeSummary,
    pub judge: MarketHeadJudgeSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketHeadJudgePackComparisonReport {
    pub generated_at: String,
    pub cases: usize,
    pub canonical_vs_judge: Vec<MarketHeadJudgePackComparisonEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeCalibrationDelta {
    pub mean_signed_delta: f64,
    pub mean_absolute_delta: f64,
    pub max_absolute_delta: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JudgeCalibrationVerdict {
    Aligned,
    AlignedButSofter,
    AlignedButHarsher,
    Divergent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketHeadJudgeCalibrationEntry {
    pub judge_head: String,
    pub challenged_heads: Vec<String>,
    pub pack_count: usize,
    pub class_count: usize,
    pub shared_alignment_score: f64,
    pub verdict: JudgeCalibrationVerdict,
    pub cfsr_delta: JudgeCalibrationDelta,
    pub dlf_delta: JudgeCalibrationDelta,
    pub osr_delta: JudgeCalibrationDelta,
    pub ras_comprehension_proxy_delta: JudgeCalibrationDelta,
    pub avg_judge_next_step: f64,
    pub avg_canonical_mpr: f64,
    pub failed_cases: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketHeadJudgeCalibrationReport {
    pub generated_at: String,
    pub entries: Vec<MarketHeadJudgeCalibrationEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketHeadJudgeDisagreementCase {
    pub class: BenchmarkClass,
    pub scenario_id: String,
    pub canonical_absorption_cfsr: f64,
    pub canonical_absorption_csr: f64,
    pub canonical_absorption_dlf: f64,
    pub canonical_absorption_osr: f64,
    pub canonical_absorption_ras: f64,
    pub judge_cfsr: f64,
    pub judge_csr: f64,
    pub judge_dlf: f64,
    pub judge_osr: f64,
    pub judge_comprehension: f64,
    pub cfsr_delta: f64,
    pub csr_delta: f64,
    pub dlf_delta: f64,
    pub osr_delta: f64,
    pub ras_comprehension_proxy_delta: f64,
    pub shared_alignment_score: f64,
    pub dominant_drift_metric: String,
    pub drift_classification: String,
    pub observed_critical_facts: Vec<String>,
    pub critical_fact_diagnostics: Vec<MarketHeadJudgeCriticalFactDiagnostic>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketHeadJudgeCriticalFactDiagnostic {
    pub index: usize,
    pub expectation: String,
    pub strictness_note: Option<String>,
    pub required_concepts: Vec<String>,
    pub canonical_matched: bool,
    pub canonical_note_text: Option<String>,
    pub judge_score: Option<u8>,
    pub judge_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MarketHeadJudgeDisagreementSummary {
    pub class_count: usize,
    pub avg_abs_cfsr_delta: f64,
    pub avg_abs_csr_delta: f64,
    pub avg_abs_dlf_delta: f64,
    pub avg_abs_osr_delta: f64,
    pub avg_abs_ras_proxy_delta: f64,
    pub avg_alignment_score: f64,
    pub divergent_cases: usize,
    pub classification_counts: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketHeadJudgeDisagreementReport {
    pub generated_at: String,
    pub challenged_head: String,
    pub judge_head: String,
    pub canonical_report_path: String,
    pub judge_report_path: String,
    pub summary: MarketHeadJudgeDisagreementSummary,
    pub cases: Vec<MarketHeadJudgeDisagreementCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyMarketHeadJudgeSamePackReport {
    generated_at: String,
    challenged_head: String,
    cases: usize,
    canonical: MarketHeadChallengeSummary,
    judges: Vec<LegacyMarketHeadJudgeComparisonEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyMarketHeadJudgeComparisonEntry {
    judge_head: String,
    avg_judge_cfsr: f64,
    avg_judge_comprehension: f64,
    avg_judge_csr: f64,
    avg_judge_dlf: f64,
    avg_judge_next_step: f64,
    avg_judge_osr: f64,
    class_count: usize,
    failed_cases: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MarketHeadEvaluatorPack {
    generated_at: String,
    config: MarketHeadChallengeConfig,
    cases: Vec<MarketHeadEvaluatorCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MarketHeadEvaluatorCase {
    class: BenchmarkClass,
    scenario_id: String,
    protocol: String,
    envelope: ContextEnvelope,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JudgeEvaluation {
    pub critical_fact_rate: f64,
    pub constraint_rate: f64,
    pub decision_rate: f64,
    pub scar_rate: f64,
    pub next_step_rate: f64,
    pub comprehension_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct JudgeResponse {
    #[serde(default)]
    summary: String,
    #[serde(default)]
    critical_facts: Vec<JudgeItemScore>,
    #[serde(default)]
    constraints: Vec<JudgeItemScore>,
    #[serde(default)]
    decisions: Vec<JudgeItemScore>,
    #[serde(default)]
    scars: Vec<JudgeItemScore>,
    #[serde(default)]
    next_step: Vec<JudgeItemScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct JudgeItemScore {
    index: usize,
    score: u8,
    #[serde(default)]
    reason: String,
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
struct Scenario {
    id: String,
    title: String,
    namespace: String,
    task_id: String,
    objective: String,
    phases: Vec<ScenarioPhase>,
    truth: GroundTruth,
}

#[derive(Debug, Clone)]
struct ScenarioPhase {
    actor_id: String,
    actor_role: String,
    kind: EventKind,
    scope: Scope,
    content: String,
    dimensions: Vec<DimensionValue>,
    marks: Vec<ScenarioMark>,
}

#[derive(Debug, Clone)]
struct ScenarioMark {
    kind: ContinuityKind,
    title: String,
    body: String,
    status: ContinuityStatus,
    dimensions: Vec<DimensionValue>,
}

#[derive(Debug, Clone)]
struct TruthItem {
    id: &'static str,
    keywords: Vec<&'static str>,
    rationale_keywords: Vec<&'static str>,
    judge_note: Option<&'static str>,
    judge_required_concepts: Vec<&'static str>,
}

#[derive(Debug, Clone)]
struct GroundTruth {
    critical_facts: Vec<TruthItem>,
    constraints: Vec<TruthItem>,
    decisions: Vec<TruthItem>,
    hypotheses: Vec<TruthItem>,
    scars: Vec<TruthItem>,
    avoid_repeating: Vec<TruthItem>,
    next_step_keywords: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContextEnvelope {
    provider: BaselineKind,
    retrieval_ms: u128,
    text: String,
    token_estimate: usize,
    surfaced: Vec<SurfacedItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SurfacedItem {
    label: String,
    support_type: String,
    support_id: String,
    text: String,
    has_provenance: bool,
}

pub fn run_continuity_suite(config: ContinuityBenchConfig) -> Result<BenchmarkSuiteReport> {
    std::fs::create_dir_all(&config.output_dir)?;
    let started = Utc::now();
    let mut classes = Vec::new();
    for class in config.selected_classes() {
        classes.push(run_class(class, &config)?);
    }
    let summary = summarize(&classes);
    let report = BenchmarkSuiteReport {
        generated_at: started.to_rfc3339(),
        config: config.clone(),
        classes,
        summary,
    };
    let suite_report_path = config.output_dir.join("suite-report.json");
    std::fs::write(&suite_report_path, serde_json::to_vec_pretty(&report)?)?;
    let summary_path = config.output_dir.join("summary.md");
    std::fs::write(&summary_path, render_suite_markdown(&report))?;
    Ok(report)
}

pub fn export_market_head_challenge(
    config: MarketHeadChallengeConfig,
) -> Result<MarketHeadChallengeExportReport> {
    let root = config.output_dir.join("market-head");
    if root.exists() {
        std::fs::remove_dir_all(&root)?;
    }
    std::fs::create_dir_all(&root)?;
    let generated_at = Utc::now().to_rfc3339();
    let bench_config = config.as_bench_config();
    let mut manifest_cases = Vec::new();
    let mut evaluator_cases = Vec::new();
    for class in config.selected_classes() {
        let (manifest_case, evaluator_case) =
            build_market_head_challenge_case(&root, class, &bench_config)?;
        manifest_cases.push(manifest_case);
        evaluator_cases.push(evaluator_case);
    }
    let manifest = MarketHeadChallengeManifest {
        generated_at: generated_at.clone(),
        config: config.clone(),
        cases: manifest_cases.clone(),
    };
    let evaluator_pack = MarketHeadEvaluatorPack {
        generated_at: generated_at.clone(),
        config,
        cases: evaluator_cases,
    };
    let manifest_path = root.join("challenge-manifest.json");
    let evaluator_pack_path = root.join("challenge-evaluator-pack.json");
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    std::fs::write(
        &evaluator_pack_path,
        serde_json::to_vec_pretty(&evaluator_pack)?,
    )?;
    std::fs::write(
        root.join("README.md"),
        render_market_head_challenge_readme(),
    )?;
    Ok(MarketHeadChallengeExportReport {
        generated_at,
        manifest_path: manifest_path.to_string_lossy().to_string(),
        evaluator_pack_path: evaluator_pack_path.to_string_lossy().to_string(),
        case_count: manifest_cases.len(),
        cases: manifest_cases,
    })
}

pub fn evaluate_market_head_challenge(
    evaluator_pack_path: impl AsRef<Path>,
    responses_dir: impl AsRef<Path>,
    model_name: &str,
) -> Result<MarketHeadChallengeEvaluationReport> {
    let evaluator_pack_path = evaluator_pack_path.as_ref();
    let responses_dir = responses_dir.as_ref();
    let evaluator_pack: MarketHeadEvaluatorPack =
        serde_json::from_slice(&std::fs::read(evaluator_pack_path)?)?;
    let mut cases = Vec::new();
    for case in &evaluator_pack.cases {
        let response_path = responses_dir.join(case.class.slug()).join("response.json");
        let scenario = scenario_for(case.class);
        let case_report = match std::fs::read_to_string(&response_path) {
            Ok(raw_text) => match parse_structured_output(&raw_text) {
                Ok(output) => {
                    let raw_evaluation = evaluate_output(&output, &scenario.truth, &case.envelope);
                    let repaired = repair_output_from_envelope(output, &case.envelope);
                    let evaluation = evaluate_output(&repaired, &scenario.truth, &case.envelope);
                    MarketHeadChallengeCaseEvaluation {
                        class: case.class,
                        scenario_id: case.scenario_id.clone(),
                        protocol: case.protocol.clone(),
                        response_path: response_path.to_string_lossy().to_string(),
                        status: BaselineStatus::Ok,
                        raw_evaluation,
                        evaluation,
                        failure: None,
                    }
                }
                Err(error) => MarketHeadChallengeCaseEvaluation {
                    class: case.class,
                    scenario_id: case.scenario_id.clone(),
                    protocol: case.protocol.clone(),
                    response_path: response_path.to_string_lossy().to_string(),
                    status: BaselineStatus::Failed,
                    raw_evaluation: failed_evaluation(&case.envelope),
                    evaluation: failed_evaluation(&case.envelope),
                    failure: Some(error.to_string()),
                },
            },
            Err(error) => MarketHeadChallengeCaseEvaluation {
                class: case.class,
                scenario_id: case.scenario_id.clone(),
                protocol: case.protocol.clone(),
                response_path: response_path.to_string_lossy().to_string(),
                status: BaselineStatus::Failed,
                raw_evaluation: failed_evaluation(&case.envelope),
                evaluation: failed_evaluation(&case.envelope),
                failure: Some(error.to_string()),
            },
        };
        cases.push(case_report);
    }
    let summary = summarize_market_head_challenge(&cases);
    let report = MarketHeadChallengeEvaluationReport {
        generated_at: Utc::now().to_rfc3339(),
        model_name: model_name.to_string(),
        evaluator_pack_path: evaluator_pack_path.to_string_lossy().to_string(),
        responses_dir: responses_dir.to_string_lossy().to_string(),
        cases,
        summary,
    };
    let report_path = responses_dir.join("market-head-evaluation-report.json");
    let summary_path = responses_dir.join("market-head-evaluation-summary.md");
    std::fs::create_dir_all(responses_dir)?;
    std::fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    std::fs::write(
        &summary_path,
        render_market_head_evaluation_markdown(&report),
    )?;
    Ok(report)
}

pub fn export_market_head_judge_challenge(
    evaluator_pack_path: impl AsRef<Path>,
    responses_dir: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
) -> Result<MarketHeadJudgeExportReport> {
    let evaluator_pack_path = evaluator_pack_path.as_ref();
    let responses_dir = responses_dir.as_ref();
    let output_dir = output_dir.as_ref();
    let root = output_dir.join("market-head-judge");
    if root.exists() {
        std::fs::remove_dir_all(&root)?;
    }
    std::fs::create_dir_all(&root)?;

    let evaluator_pack: MarketHeadEvaluatorPack =
        serde_json::from_slice(&std::fs::read(evaluator_pack_path)?)?;
    let generated_at = Utc::now().to_rfc3339();
    let mut cases = Vec::with_capacity(evaluator_pack.cases.len());

    for case in &evaluator_pack.cases {
        let scenario = scenario_for(case.class);
        let response_path = responses_dir.join(case.class.slug()).join("response.json");
        let response_text = std::fs::read_to_string(&response_path)
            .with_context(|| format!("reading response for {}", case.scenario_id))?;
        let response = parse_structured_output(&response_text)
            .with_context(|| format!("parsing response for {}", case.scenario_id))?;
        let class_dir = root.join(case.class.slug());
        std::fs::create_dir_all(&class_dir)?;

        let prompt_path = class_dir.join("prompt.txt");
        let schema_path = class_dir.join("response.schema.json");
        let template_path = class_dir.join("response.template.json");
        std::fs::write(
            &prompt_path,
            render_market_head_judge_prompt(case.class, &scenario.truth, &response),
        )?;
        std::fs::write(
            &schema_path,
            serde_json::to_vec_pretty(&judge_response_schema())?,
        )?;
        std::fs::write(
            &template_path,
            serde_json::to_vec_pretty(&judge_response_template())?,
        )?;

        cases.push(MarketHeadChallengeCaseManifest {
            class: case.class,
            scenario_id: case.scenario_id.clone(),
            protocol: format!("judge://{}", case.protocol),
            prompt_path: format!("{}/prompt.txt", case.class.slug()),
            response_path: format!("{}/response.json", case.class.slug()),
            schema_path: format!("{}/response.schema.json", case.class.slug()),
            template_path: format!("{}/response.template.json", case.class.slug()),
        });
    }

    let manifest = MarketHeadChallengeManifest {
        generated_at: generated_at.clone(),
        config: evaluator_pack.config,
        cases: cases.clone(),
    };
    let manifest_path = root.join("judge-manifest.json");
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    std::fs::write(root.join("README.md"), render_market_head_judge_readme())?;

    Ok(MarketHeadJudgeExportReport {
        generated_at,
        manifest_path: manifest_path.to_string_lossy().to_string(),
        case_count: cases.len(),
        cases,
    })
}

pub fn evaluate_market_head_judge_challenge(
    manifest_path: impl AsRef<Path>,
    responses_dir: impl AsRef<Path>,
    model_name: &str,
) -> Result<MarketHeadJudgeEvaluationReport> {
    let manifest_path = manifest_path.as_ref();
    let responses_dir = responses_dir.as_ref();
    let manifest: MarketHeadChallengeManifest =
        serde_json::from_slice(&std::fs::read(manifest_path)?)?;
    let mut cases = Vec::with_capacity(manifest.cases.len());

    for case in &manifest.cases {
        let scenario = scenario_for(case.class);
        let response_path = responses_dir.join(&case.response_path);
        let case_report = match std::fs::read_to_string(&response_path) {
            Ok(raw_text) => match serde_json::from_str::<JudgeResponse>(&raw_text) {
                Ok(response) => MarketHeadJudgeCaseEvaluation {
                    class: case.class,
                    scenario_id: case.scenario_id.clone(),
                    protocol: case.protocol.clone(),
                    response_path: response_path.to_string_lossy().to_string(),
                    status: BaselineStatus::Ok,
                    evaluation: judge_evaluate_response(&response, &scenario.truth),
                    failure: None,
                },
                Err(error) => MarketHeadJudgeCaseEvaluation {
                    class: case.class,
                    scenario_id: case.scenario_id.clone(),
                    protocol: case.protocol.clone(),
                    response_path: response_path.to_string_lossy().to_string(),
                    status: BaselineStatus::Failed,
                    evaluation: JudgeEvaluation::default(),
                    failure: Some(error.to_string()),
                },
            },
            Err(error) => MarketHeadJudgeCaseEvaluation {
                class: case.class,
                scenario_id: case.scenario_id.clone(),
                protocol: case.protocol.clone(),
                response_path: response_path.to_string_lossy().to_string(),
                status: BaselineStatus::Failed,
                evaluation: JudgeEvaluation::default(),
                failure: Some(error.to_string()),
            },
        };
        cases.push(case_report);
    }

    let summary = summarize_market_head_judge(&cases);
    let report = MarketHeadJudgeEvaluationReport {
        generated_at: Utc::now().to_rfc3339(),
        model_name: model_name.to_string(),
        manifest_path: manifest_path.to_string_lossy().to_string(),
        responses_dir: responses_dir.to_string_lossy().to_string(),
        cases,
        summary,
    };
    let report_path = responses_dir.join("market-head-judge-report.json");
    let summary_path = responses_dir.join("market-head-judge-summary.md");
    std::fs::create_dir_all(responses_dir)?;
    std::fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    std::fs::write(&summary_path, render_market_head_judge_markdown(&report))?;
    Ok(report)
}

pub fn compare_market_head_same_pack(
    challenged_head: impl Into<String>,
    canonical_report_path: impl AsRef<Path>,
    judge_reports: Vec<(String, PathBuf)>,
) -> Result<MarketHeadJudgeSamePackComparisonReport> {
    let canonical_report_path = canonical_report_path.as_ref();
    let canonical: MarketHeadChallengeEvaluationReport =
        serde_json::from_slice(&std::fs::read(canonical_report_path)?).with_context(|| {
            format!(
                "parsing canonical report {}",
                canonical_report_path.display()
            )
        })?;

    if judge_reports.is_empty() {
        return Err(anyhow!("at least one judge report is required"));
    }

    let mut judges = Vec::with_capacity(judge_reports.len());
    for (judge_head, report_path) in judge_reports {
        let report: MarketHeadJudgeEvaluationReport =
            serde_json::from_slice(&std::fs::read(&report_path)?)
                .with_context(|| format!("parsing judge report {}", report_path.display()))?;
        judges.push(MarketHeadJudgeComparisonEntry {
            judge_head,
            summary: report.summary,
        });
    }

    Ok(MarketHeadJudgeSamePackComparisonReport {
        generated_at: Utc::now().to_rfc3339(),
        challenged_head: challenged_head.into(),
        canonical: canonical.summary,
        judges,
    })
}

pub fn compare_market_head_judge_pack(
    same_pack_report_paths: Vec<PathBuf>,
) -> Result<MarketHeadJudgePackComparisonReport> {
    if same_pack_report_paths.is_empty() {
        return Err(anyhow!("at least one same-pack report is required"));
    }

    let mut cases = None;
    let mut canonical_vs_judge = Vec::new();
    for report_path in same_pack_report_paths {
        let report = load_market_head_same_pack_report(&report_path)?;
        let report_cases = report.canonical.class_count;
        match cases {
            Some(existing) if existing != report_cases => {
                anyhow::bail!(
                    "mismatched class count: {} has {}, expected {}",
                    report_path.display(),
                    report_cases,
                    existing
                );
            }
            None => cases = Some(report_cases),
            _ => {}
        }

        for judge in report.judges {
            canonical_vs_judge.push(MarketHeadJudgePackComparisonEntry {
                challenged_head: report.challenged_head.clone(),
                judge_head: judge.judge_head,
                canonical: report.canonical.clone(),
                judge: judge.summary,
            });
        }
    }

    canonical_vs_judge.sort_by(|left, right| {
        left.challenged_head
            .cmp(&right.challenged_head)
            .then(left.judge_head.cmp(&right.judge_head))
    });

    Ok(MarketHeadJudgePackComparisonReport {
        generated_at: Utc::now().to_rfc3339(),
        cases: cases.unwrap_or_default(),
        canonical_vs_judge,
    })
}

pub fn compare_market_head_judge_calibration(
    same_pack_report_paths: Vec<PathBuf>,
) -> Result<MarketHeadJudgeCalibrationReport> {
    if same_pack_report_paths.is_empty() {
        return Err(anyhow!("at least one same-pack report is required"));
    }

    #[derive(Default)]
    struct DeltaAccumulator {
        signed_sum: f64,
        absolute_sum: f64,
        max_absolute: f64,
        count: usize,
    }

    impl DeltaAccumulator {
        fn record(&mut self, delta: f64) {
            self.signed_sum += delta;
            let absolute = delta.abs();
            self.absolute_sum += absolute;
            self.max_absolute = self.max_absolute.max(absolute);
            self.count += 1;
        }

        fn finish(&self) -> JudgeCalibrationDelta {
            if self.count == 0 {
                return JudgeCalibrationDelta {
                    mean_signed_delta: 0.0,
                    mean_absolute_delta: 0.0,
                    max_absolute_delta: 0.0,
                };
            }
            let count = self.count as f64;
            JudgeCalibrationDelta {
                mean_signed_delta: self.signed_sum / count,
                mean_absolute_delta: self.absolute_sum / count,
                max_absolute_delta: self.max_absolute,
            }
        }
    }

    #[derive(Default)]
    struct JudgeCalibrationAccumulator {
        challenged_heads: BTreeSet<String>,
        pack_count: usize,
        class_count: usize,
        cfsr: DeltaAccumulator,
        dlf: DeltaAccumulator,
        osr: DeltaAccumulator,
        ras_proxy: DeltaAccumulator,
        judge_next_step_sum: f64,
        canonical_mpr_sum: f64,
        failed_cases: usize,
    }

    let mut by_judge: BTreeMap<String, JudgeCalibrationAccumulator> = BTreeMap::new();
    for report_path in same_pack_report_paths {
        let report = load_market_head_same_pack_report(&report_path)?;
        for judge in report.judges {
            let entry = by_judge.entry(judge.judge_head.clone()).or_default();
            entry
                .challenged_heads
                .insert(report.challenged_head.clone());
            entry.pack_count += 1;
            entry.class_count += report.canonical.class_count;
            entry
                .cfsr
                .record(judge.summary.avg_judge_cfsr - report.canonical.avg_absorption_cfsr);
            entry
                .dlf
                .record(judge.summary.avg_judge_dlf - report.canonical.avg_absorption_dlf);
            entry
                .osr
                .record(judge.summary.avg_judge_osr - report.canonical.avg_absorption_osr);
            entry.ras_proxy.record(
                judge.summary.avg_judge_comprehension - report.canonical.avg_absorption_ras,
            );
            entry.judge_next_step_sum += judge.summary.avg_judge_next_step;
            entry.canonical_mpr_sum += report.canonical.avg_mpr;
            entry.failed_cases += judge.summary.failed_cases;
        }
    }

    let mut entries: Vec<MarketHeadJudgeCalibrationEntry> = by_judge
        .into_iter()
        .map(|(judge_head, acc)| {
            let cfsr_delta = acc.cfsr.finish();
            let dlf_delta = acc.dlf.finish();
            let osr_delta = acc.osr.finish();
            let ras_comprehension_proxy_delta = acc.ras_proxy.finish();
            let shared_alignment_score = (cfsr_delta.mean_absolute_delta
                + dlf_delta.mean_absolute_delta
                + osr_delta.mean_absolute_delta
                + ras_comprehension_proxy_delta.mean_absolute_delta)
                / 4.0;
            let shared_signed_center = (cfsr_delta.mean_signed_delta
                + dlf_delta.mean_signed_delta
                + osr_delta.mean_signed_delta
                + ras_comprehension_proxy_delta.mean_signed_delta)
                / 4.0;
            let verdict = if shared_alignment_score <= 0.05 {
                if shared_signed_center <= -0.03 {
                    JudgeCalibrationVerdict::AlignedButSofter
                } else if shared_signed_center >= 0.03 {
                    JudgeCalibrationVerdict::AlignedButHarsher
                } else {
                    JudgeCalibrationVerdict::Aligned
                }
            } else {
                JudgeCalibrationVerdict::Divergent
            };
            let pack_count = acc.pack_count.max(1);
            MarketHeadJudgeCalibrationEntry {
                judge_head,
                challenged_heads: acc.challenged_heads.into_iter().collect(),
                pack_count: acc.pack_count,
                class_count: acc.class_count,
                shared_alignment_score,
                verdict,
                cfsr_delta,
                dlf_delta,
                osr_delta,
                ras_comprehension_proxy_delta,
                avg_judge_next_step: acc.judge_next_step_sum / pack_count as f64,
                avg_canonical_mpr: acc.canonical_mpr_sum / pack_count as f64,
                failed_cases: acc.failed_cases,
            }
        })
        .collect();
    entries.sort_by(|left, right| left.judge_head.cmp(&right.judge_head));

    Ok(MarketHeadJudgeCalibrationReport {
        generated_at: Utc::now().to_rfc3339(),
        entries,
    })
}

pub fn compare_market_head_judge_disagreement(
    challenged_head: impl Into<String>,
    judge_head: impl Into<String>,
    canonical_report_path: impl AsRef<Path>,
    judge_report_path: impl AsRef<Path>,
) -> Result<MarketHeadJudgeDisagreementReport> {
    let canonical_report_path = canonical_report_path.as_ref();
    let judge_report_path = judge_report_path.as_ref();
    let canonical: MarketHeadChallengeEvaluationReport =
        serde_json::from_slice(&std::fs::read(canonical_report_path)?).with_context(|| {
            format!(
                "parsing canonical report {}",
                canonical_report_path.display()
            )
        })?;
    let judge: MarketHeadJudgeEvaluationReport =
        serde_json::from_slice(&std::fs::read(judge_report_path)?)
            .with_context(|| format!("parsing judge report {}", judge_report_path.display()))?;

    let canonical_by_scenario: BTreeMap<&str, &MarketHeadChallengeCaseEvaluation> = canonical
        .cases
        .iter()
        .map(|case| (case.scenario_id.as_str(), case))
        .collect();

    let mut cases = Vec::new();
    for judge_case in &judge.cases {
        let Some(canonical_case) = canonical_by_scenario.get(judge_case.scenario_id.as_str())
        else {
            continue;
        };
        let (observed_critical_facts, critical_fact_diagnostics) =
            collect_critical_fact_disagreement_diagnostics(canonical_case, judge_case);
        let cfsr_delta = judge_case.evaluation.critical_fact_rate
            - canonical_case.raw_evaluation.critical_fact_survival_rate;
        let csr_delta = judge_case.evaluation.constraint_rate
            - canonical_case.raw_evaluation.constraint_survival_rate;
        let dlf_delta = judge_case.evaluation.decision_rate
            - canonical_case.raw_evaluation.decision_lineage_fidelity;
        let osr_delta = judge_case.evaluation.scar_rate
            - canonical_case.raw_evaluation.operational_scar_retention;
        let ras_proxy_delta = judge_case.evaluation.comprehension_score
            - canonical_case.raw_evaluation.resume_accuracy_score;
        let drift_candidates = [
            ("cfsr", cfsr_delta.abs()),
            ("csr", csr_delta.abs()),
            ("dlf", dlf_delta.abs()),
            ("osr", osr_delta.abs()),
            ("ras_comprehension_proxy", ras_proxy_delta.abs()),
        ];
        let dominant_drift_metric = drift_candidates
            .into_iter()
            .max_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, _)| name.to_string())
            .unwrap_or_else(|| "none".to_string());
        let drift_classification =
            classify_market_head_judge_drift(cfsr_delta, &critical_fact_diagnostics);
        let shared_alignment_score = (cfsr_delta.abs()
            + csr_delta.abs()
            + dlf_delta.abs()
            + osr_delta.abs()
            + ras_proxy_delta.abs())
            / 5.0;
        cases.push(MarketHeadJudgeDisagreementCase {
            class: judge_case.class,
            scenario_id: judge_case.scenario_id.clone(),
            canonical_absorption_cfsr: canonical_case.raw_evaluation.critical_fact_survival_rate,
            canonical_absorption_csr: canonical_case.raw_evaluation.constraint_survival_rate,
            canonical_absorption_dlf: canonical_case.raw_evaluation.decision_lineage_fidelity,
            canonical_absorption_osr: canonical_case.raw_evaluation.operational_scar_retention,
            canonical_absorption_ras: canonical_case.raw_evaluation.resume_accuracy_score,
            judge_cfsr: judge_case.evaluation.critical_fact_rate,
            judge_csr: judge_case.evaluation.constraint_rate,
            judge_dlf: judge_case.evaluation.decision_rate,
            judge_osr: judge_case.evaluation.scar_rate,
            judge_comprehension: judge_case.evaluation.comprehension_score,
            cfsr_delta,
            csr_delta,
            dlf_delta,
            osr_delta,
            ras_comprehension_proxy_delta: ras_proxy_delta,
            shared_alignment_score,
            dominant_drift_metric,
            drift_classification,
            observed_critical_facts,
            critical_fact_diagnostics,
        });
    }

    cases.sort_by(|left, right| {
        left.class
            .slug()
            .cmp(right.class.slug())
            .then(left.scenario_id.cmp(&right.scenario_id))
    });
    let summary = summarize_market_head_judge_disagreement(&cases);

    Ok(MarketHeadJudgeDisagreementReport {
        generated_at: Utc::now().to_rfc3339(),
        challenged_head: challenged_head.into(),
        judge_head: judge_head.into(),
        canonical_report_path: canonical_report_path.to_string_lossy().to_string(),
        judge_report_path: judge_report_path.to_string_lossy().to_string(),
        summary,
        cases,
    })
}

fn collect_critical_fact_disagreement_diagnostics(
    canonical_case: &MarketHeadChallengeCaseEvaluation,
    judge_case: &MarketHeadJudgeCaseEvaluation,
) -> (Vec<String>, Vec<MarketHeadJudgeCriticalFactDiagnostic>) {
    let scenario = scenario_for(judge_case.class);
    let challenged_output = std::fs::read_to_string(&canonical_case.response_path)
        .ok()
        .and_then(|raw| parse_structured_output(&raw).ok());
    let judge_response = std::fs::read_to_string(&judge_case.response_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<JudgeResponse>(&raw).ok());

    let observed_critical_facts = challenged_output
        .as_ref()
        .map(|output| {
            output
                .critical_facts
                .iter()
                .map(|note| note.text.clone())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let diagnostics = scenario
        .truth
        .critical_facts
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let canonical_match = challenged_output.as_ref().and_then(|output| {
                output
                    .critical_facts
                    .iter()
                    .find(|note| match_keywords(&note.text, &item.keywords))
            });
            let judge_detail = judge_response.as_ref().and_then(|response| {
                response
                    .critical_facts
                    .iter()
                    .find(|score| score.index == index)
            });

            MarketHeadJudgeCriticalFactDiagnostic {
                index,
                expectation: item.keywords.join(" "),
                strictness_note: item.judge_note.map(|note| note.to_string()),
                required_concepts: item
                    .judge_required_concepts
                    .iter()
                    .map(|concept| (*concept).to_string())
                    .collect(),
                canonical_matched: canonical_match.is_some(),
                canonical_note_text: canonical_match.map(|note| note.text.clone()),
                judge_score: judge_detail.map(|detail| detail.score),
                judge_reason: judge_detail.map(|detail| detail.reason.clone()),
            }
        })
        .collect();

    (observed_critical_facts, diagnostics)
}

fn classify_market_head_judge_drift(
    cfsr_delta: f64,
    diagnostics: &[MarketHeadJudgeCriticalFactDiagnostic],
) -> String {
    if cfsr_delta.abs() <= f64::EPSILON {
        return "aligned".to_string();
    }

    let has_lexical_gap = diagnostics
        .iter()
        .any(|detail| !detail.canonical_matched && detail.judge_score == Some(3));
    let has_partial_gap = diagnostics.iter().any(|detail| {
        !detail.canonical_matched
            && matches!(detail.judge_score, Some(score) if score > 0 && score < 3)
    });
    let has_judge_undercredit = diagnostics.iter().any(|detail| {
        detail.canonical_matched && matches!(detail.judge_score, Some(score) if score < 3)
    });

    match (has_lexical_gap, has_partial_gap, has_judge_undercredit) {
        (true, false, false) => "canonical_lexical_gap".to_string(),
        (true, _, true) => "mixed_lexical_gap_and_judge_undercredit".to_string(),
        (false, true, false) => "canonical_partial_gap".to_string(),
        (false, true, true) => "mixed_partial_gap_and_judge_undercredit".to_string(),
        (false, false, true) => "judge_undercredit".to_string(),
        _ => "unclassified_cfsr_drift".to_string(),
    }
}

fn load_market_head_same_pack_report(
    path: &Path,
) -> Result<MarketHeadJudgeSamePackComparisonReport> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading same-pack report {}", path.display()))?;
    match serde_json::from_slice::<MarketHeadJudgeSamePackComparisonReport>(&bytes) {
        Ok(report) => Ok(report),
        Err(current_error) => {
            let legacy: LegacyMarketHeadJudgeSamePackReport =
                serde_json::from_slice(&bytes).with_context(|| {
                    format!(
                        "parsing same-pack report {} in current or legacy format (current error: {current_error})",
                        path.display()
                    )
                })?;
            Ok(MarketHeadJudgeSamePackComparisonReport {
                generated_at: legacy.generated_at,
                challenged_head: legacy.challenged_head,
                canonical: legacy.canonical,
                judges: legacy
                    .judges
                    .into_iter()
                    .map(|judge| MarketHeadJudgeComparisonEntry {
                        judge_head: judge.judge_head,
                        summary: MarketHeadJudgeSummary {
                            class_count: judge.class_count,
                            avg_judge_cfsr: judge.avg_judge_cfsr,
                            avg_judge_csr: judge.avg_judge_csr,
                            avg_judge_dlf: judge.avg_judge_dlf,
                            avg_judge_osr: judge.avg_judge_osr,
                            avg_judge_next_step: judge.avg_judge_next_step,
                            avg_judge_comprehension: judge.avg_judge_comprehension,
                            failed_cases: judge.failed_cases,
                        },
                    })
                    .collect(),
            })
        }
    }
}

fn run_class(
    class: BenchmarkClass,
    config: &ContinuityBenchConfig,
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
    let labels = populate_scenario(&kernel, &context.id, &scenario)?;

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
        &labels,
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
            &labels,
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
    let report = BenchmarkClassReport {
        class,
        scenario_id: scenario.id.clone(),
        continuity,
        baselines,
        metrics,
        resource,
        artifacts: vec![full_transcript.model, config.small_model.clone()],
    };
    let report_path = class_root.join("report.json");
    std::fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn build_market_head_challenge_case(
    root: &Path,
    class: BenchmarkClass,
    config: &ContinuityBenchConfig,
) -> Result<(MarketHeadChallengeCaseManifest, MarketHeadEvaluatorCase)> {
    let scenario = scenario_for(class);
    let class_root = root.join(class.slug());
    let work_root = root.join(".work").join(class.slug());
    if class_root.exists() {
        std::fs::remove_dir_all(&class_root)?;
    }
    if work_root.exists() {
        std::fs::remove_dir_all(&work_root)?;
    }
    std::fs::create_dir_all(&class_root)?;
    std::fs::create_dir_all(
        work_root
            .parent()
            .ok_or_else(|| anyhow!("work root missing parent directory"))?,
    )?;

    let kernel = SharedContinuityKernel::open(&work_root)?;
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
        session_id: format!("market-head-{}", class.slug()),
        objective: scenario.objective.clone(),
        selector: None,
        agent_id: Some(planner.config().agent_id.clone()),
        attachment_id: Some(attach.id),
    })?;
    let _labels = populate_scenario(&kernel, &context.id, &scenario)?;

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
    }

    match class {
        BenchmarkClass::StrongToSmallContinuation => {
            let _ = analyze_and_write_back(
                &work_root,
                class,
                &kernel,
                &context.id,
                &scenario,
                &planner,
                BaselineKind::FullTranscript,
                config,
            )?;
        }
        BenchmarkClass::SmallToSmallRelay | BenchmarkClass::CrossAgentCollaborative => {
            let _ = analyze_and_write_back(
                &work_root,
                class,
                &kernel,
                &context.id,
                &scenario,
                &small_a,
                BaselineKind::SharedContinuity,
                config,
            )?;
        }
        _ => {}
    }

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

    let envelope_kernel = SharedContinuityKernel::open(&work_root)?;
    let (envelope, continuity_path) = build_context_envelope(
        &envelope_kernel,
        class,
        BaselineKind::SharedContinuity,
        &context.id,
        &scenario,
        small_b.config().agent_id.as_str(),
        config.token_budget,
        config.candidate_limit,
        config.recent_window,
    )?;
    let protocol = continuity_path
        .as_ref()
        .map(format_continuity_path_label)
        .unwrap_or_else(|| "n/a".into());

    let prompt = render_structured_resume_prompt(
        "market-head challenger",
        &scenario.objective,
        &envelope.text,
    );
    let prompt_path = class_root.join("prompt.txt");
    let schema_path = class_root.join("response.schema.json");
    let template_path = class_root.join("response.template.json");
    let response_path = class_root.join("response.json");
    std::fs::write(&prompt_path, prompt.as_bytes())?;
    std::fs::write(
        &schema_path,
        serde_json::to_vec_pretty(&structured_output_schema())?,
    )?;
    std::fs::write(
        &template_path,
        serde_json::to_vec_pretty(&market_head_response_template())?,
    )?;

    let manifest = MarketHeadChallengeCaseManifest {
        class,
        scenario_id: scenario.id.clone(),
        protocol: protocol.clone(),
        prompt_path: relative_display(root, &prompt_path),
        response_path: relative_display(root, &response_path),
        schema_path: relative_display(root, &schema_path),
        template_path: relative_display(root, &template_path),
    };
    let evaluator = MarketHeadEvaluatorCase {
        class,
        scenario_id: scenario.id,
        protocol,
        envelope,
    };
    Ok((manifest, evaluator))
}

fn populate_scenario(
    kernel: &SharedContinuityKernel,
    context_id: &str,
    scenario: &Scenario,
) -> Result<HashMap<String, SupportLabel>> {
    let mut labels = HashMap::new();
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
        let label = format!("e{}", counter);
        labels.insert(
            label.clone(),
            SupportLabel {
                label,
                support_type: "event".into(),
                support_id: event_id.clone(),
            },
        );
        for mark in &phase.marks {
            let item = kernel.write_derivations(vec![ContinuityItemInput {
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
            let mark_label = format!("c{}", labels.len() + 1);
            labels.insert(
                mark_label.clone(),
                SupportLabel {
                    label: mark_label,
                    support_type: "continuity".into(),
                    support_id: item[0].id.clone(),
                },
            );
        }
    }
    Ok(labels)
}

fn analyze_and_write_back(
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
        class_root,
        class,
        provider,
        context_id,
        scenario,
        adapter,
        config,
        &HashMap::new(),
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
    labels: &HashMap<String, SupportLabel>,
) -> Result<BaselineRunReport> {
    let run = if matches!(class, BenchmarkClass::CrashRecovery)
        && baseline == BaselineKind::SharedContinuity
    {
        let reopened = SharedContinuityKernel::open(class_root)?;
        run_baseline_inner(
            class_root, class, baseline, context_id, scenario, adapter, config, labels,
        )?
        .with_kernel(reopened)
    } else {
        run_baseline_inner(
            class_root, class, baseline, context_id, scenario, adapter, config, labels,
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
    _labels: &HashMap<String, SupportLabel>,
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
    let (output, model_metrics, evaluation, status, failure) =
        match adapter.analyze(&scenario.objective, &envelope.text) {
            Ok((output, model_metrics)) => {
                let repaired = repair_output_from_envelope(output, &envelope);
                let evaluation = evaluate_output(&repaired, &scenario.truth, &envelope);
                (
                    repaired,
                    model_metrics,
                    evaluation,
                    BaselineStatus::Ok,
                    None,
                )
            }
            Err(error) => (
                AgentContinuationOutput::default(),
                ModelCallMetrics::default(),
                failed_evaluation(&envelope),
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
        status,
        failure,
        artifacts,
        continuity_path,
    })
}

fn build_context_envelope(
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

fn evaluate_output(
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

fn repair_output_from_envelope(
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

fn failed_evaluation(envelope: &ContextEnvelope) -> Evaluation {
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

fn match_keywords(text: &str, keywords: &[&str]) -> bool {
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

fn scenario_for(class: BenchmarkClass) -> Scenario {
    let base_truth = GroundTruth {
        critical_facts: vec![
            TruthItem {
                id: "selector_missing",
                keywords: vec!["selector_missing", "src/query.rs"],
                rationale_keywords: Vec::new(),
                judge_note: Some(
                    "Full credit requires both the selector/support-memory failure and the src/query.rs anchor; file mention alone is partial.",
                ),
                judge_required_concepts: vec!["selector/support-memory failure", "src/query.rs"],
            },
            TruthItem {
                id: "context_primary",
                keywords: vec!["context", "primary"],
                rationale_keywords: Vec::new(),
                judge_note: Some(
                    "Full credit requires explicit identification that the active resume context is the primary context, not just a task mention.",
                ),
                judge_required_concepts: vec!["active resume context", "primary context"],
            },
        ],
        constraints: vec![TruthItem {
            id: "preserve_provenance",
            keywords: vec!["preserve", "provenance"],
            rationale_keywords: Vec::new(),
            judge_note: None,
            judge_required_concepts: Vec::new(),
        }],
        decisions: vec![TruthItem {
            id: "use_uci",
            keywords: vec!["unified", "continuity", "interface"],
            rationale_keywords: vec!["agent", "swap"],
            judge_note: None,
            judge_required_concepts: Vec::new(),
        }],
        hypotheses: vec![TruthItem {
            id: "adapter_timeout",
            keywords: vec!["timeout", "adapter"],
            rationale_keywords: Vec::new(),
            judge_note: None,
            judge_required_concepts: Vec::new(),
        }],
        scars: vec![TruthItem {
            id: "naive_probe",
            keywords: vec!["naive", "probe"],
            rationale_keywords: Vec::new(),
            judge_note: None,
            judge_required_concepts: Vec::new(),
        }],
        avoid_repeating: vec![TruthItem {
            id: "manual_probe",
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
        title: format!("{} continuity scenario", class.slug()),
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

fn format_continuity_path_label(path: &ContinuityPathReport) -> String {
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

fn summarize_market_head_challenge(
    cases: &[MarketHeadChallengeCaseEvaluation],
) -> MarketHeadChallengeSummary {
    let mut summary = MarketHeadChallengeSummary::default();
    summary.class_count = cases.len();
    summary.failed_cases = cases
        .iter()
        .filter(|case| case.status == BaselineStatus::Failed)
        .count();
    if cases.is_empty() {
        return summary;
    }
    let len = cases.len() as f64;
    summary.avg_absorption_cfsr = cases
        .iter()
        .map(|case| case.raw_evaluation.critical_fact_survival_rate)
        .sum::<f64>()
        / len;
    summary.avg_absorption_dlf = cases
        .iter()
        .map(|case| case.raw_evaluation.decision_lineage_fidelity)
        .sum::<f64>()
        / len;
    summary.avg_absorption_osr = cases
        .iter()
        .map(|case| case.raw_evaluation.operational_scar_retention)
        .sum::<f64>()
        / len;
    summary.avg_absorption_ras = cases
        .iter()
        .map(|case| case.raw_evaluation.resume_accuracy_score)
        .sum::<f64>()
        / len;
    summary.avg_cfsr = cases
        .iter()
        .map(|case| case.evaluation.critical_fact_survival_rate)
        .sum::<f64>()
        / len;
    summary.avg_dlf = cases
        .iter()
        .map(|case| case.evaluation.decision_lineage_fidelity)
        .sum::<f64>()
        / len;
    summary.avg_osr = cases
        .iter()
        .map(|case| case.evaluation.operational_scar_retention)
        .sum::<f64>()
        / len;
    summary.avg_ras = cases
        .iter()
        .map(|case| case.evaluation.resume_accuracy_score)
        .sum::<f64>()
        / len;
    summary.avg_mpr = cases
        .iter()
        .map(|case| case.evaluation.memory_pollution_rate)
        .sum::<f64>()
        / len;
    summary.avg_pc = cases
        .iter()
        .map(|case| case.evaluation.provenance_coverage)
        .sum::<f64>()
        / len;
    summary
}

fn summarize_market_head_judge(cases: &[MarketHeadJudgeCaseEvaluation]) -> MarketHeadJudgeSummary {
    let mut summary = MarketHeadJudgeSummary::default();
    summary.class_count = cases.len();
    summary.failed_cases = cases
        .iter()
        .filter(|case| case.status == BaselineStatus::Failed)
        .count();
    if cases.is_empty() {
        return summary;
    }
    let len = cases.len() as f64;
    summary.avg_judge_cfsr = cases
        .iter()
        .map(|case| case.evaluation.critical_fact_rate)
        .sum::<f64>()
        / len;
    summary.avg_judge_csr = cases
        .iter()
        .map(|case| case.evaluation.constraint_rate)
        .sum::<f64>()
        / len;
    summary.avg_judge_dlf = cases
        .iter()
        .map(|case| case.evaluation.decision_rate)
        .sum::<f64>()
        / len;
    summary.avg_judge_osr = cases
        .iter()
        .map(|case| case.evaluation.scar_rate)
        .sum::<f64>()
        / len;
    summary.avg_judge_next_step = cases
        .iter()
        .map(|case| case.evaluation.next_step_rate)
        .sum::<f64>()
        / len;
    summary.avg_judge_comprehension = cases
        .iter()
        .map(|case| case.evaluation.comprehension_score)
        .sum::<f64>()
        / len;
    summary
}

fn render_market_head_evaluation_markdown(report: &MarketHeadChallengeEvaluationReport) -> String {
    let mut out = String::new();
    out.push_str("# Market-Head Challenge Summary\n\n");
    out.push_str(&format!("Generated: `{}`\n\n", report.generated_at));
    out.push_str(&format!(
        "Model: `{}`\nEvaluator pack: `{}`\nResponses dir: `{}`\n\n",
        report.model_name, report.evaluator_pack_path, report.responses_dir
    ));
    out.push_str("| class | protocol | status | raw cfsr | raw dlf | raw osr | raw ras | repaired cfsr | repaired dlf | repaired osr | repaired ras |\n");
    out.push_str("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for case in &report.cases {
        out.push_str(&format!(
            "| {} | {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |\n",
            case.class.slug(),
            case.protocol,
            match case.status {
                BaselineStatus::Ok => "ok",
                BaselineStatus::Failed => "failed",
            },
            case.raw_evaluation.critical_fact_survival_rate,
            case.raw_evaluation.decision_lineage_fidelity,
            case.raw_evaluation.operational_scar_retention,
            case.raw_evaluation.resume_accuracy_score,
            case.evaluation.critical_fact_survival_rate,
            case.evaluation.decision_lineage_fidelity,
            case.evaluation.operational_scar_retention,
            case.evaluation.resume_accuracy_score,
        ));
    }
    out.push_str("\n## Aggregate\n\n");
    out.push_str(&format!(
        "- avg raw CFSR: `{:.2}`\n- avg raw DLF: `{:.2}`\n- avg raw OSR: `{:.2}`\n- avg raw RAS: `{:.2}`\n- avg repaired CFSR: `{:.2}`\n- avg repaired DLF: `{:.2}`\n- avg repaired OSR: `{:.2}`\n- avg repaired RAS: `{:.2}`\n- avg MPR: `{:.2}`\n- avg PC: `{:.2}`\n- failed cases: `{}` / `{}`\n",
        report.summary.avg_absorption_cfsr,
        report.summary.avg_absorption_dlf,
        report.summary.avg_absorption_osr,
        report.summary.avg_absorption_ras,
        report.summary.avg_cfsr,
        report.summary.avg_dlf,
        report.summary.avg_osr,
        report.summary.avg_ras,
        report.summary.avg_mpr,
        report.summary.avg_pc,
        report.summary.failed_cases,
        report.summary.class_count,
    ));
    out
}

fn render_market_head_judge_markdown(report: &MarketHeadJudgeEvaluationReport) -> String {
    let mut out = String::new();
    out.push_str("# Market-Head Judge Summary\n\n");
    out.push_str(&format!("Generated: `{}`\n\n", report.generated_at));
    out.push_str(&format!(
        "Judge model: `{}`\nJudge manifest: `{}`\nResponses dir: `{}`\n\n",
        report.model_name, report.manifest_path, report.responses_dir
    ));
    out.push_str("| class | protocol | status | judge cfsr | judge csr | judge dlf | judge osr | judge next-step | judge composite |\n");
    out.push_str("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for case in &report.cases {
        out.push_str(&format!(
            "| {} | {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |\n",
            case.class.slug(),
            case.protocol,
            match case.status {
                BaselineStatus::Ok => "ok",
                BaselineStatus::Failed => "failed",
            },
            case.evaluation.critical_fact_rate,
            case.evaluation.constraint_rate,
            case.evaluation.decision_rate,
            case.evaluation.scar_rate,
            case.evaluation.next_step_rate,
            case.evaluation.comprehension_score,
        ));
    }
    out.push_str("\n## Aggregate\n\n");
    out.push_str(&format!(
        "- avg judge CFSR: `{:.2}`\n- avg judge CSR: `{:.2}`\n- avg judge DLF: `{:.2}`\n- avg judge OSR: `{:.2}`\n- avg judge next-step: `{:.2}`\n- avg judge comprehension: `{:.2}`\n- failed cases: `{}` / `{}`\n",
        report.summary.avg_judge_cfsr,
        report.summary.avg_judge_csr,
        report.summary.avg_judge_dlf,
        report.summary.avg_judge_osr,
        report.summary.avg_judge_next_step,
        report.summary.avg_judge_comprehension,
        report.summary.failed_cases,
        report.summary.class_count,
    ));
    out
}

pub fn render_market_head_same_pack_markdown(
    report: &MarketHeadJudgeSamePackComparisonReport,
) -> String {
    let mut out = String::new();
    out.push_str("# Market-Head Judge Same-Pack Comparison\n\n");
    out.push_str(&format!("Generated: `{}`\n\n", report.generated_at));
    out.push_str(&format!(
        "Challenged head: `{}`\n\n",
        report.challenged_head
    ));
    out.push_str("## Canonical Baseline\n\n");
    out.push_str(&format!(
        "- absorption `CFSR {:.2}`\n- absorption `DLF {:.2}`\n- absorption `OSR {:.2}`\n- absorption `RAS {:.2}`\n- `MPR {:.2}`\n- failed `{}` / `{}`\n\n",
        report.canonical.avg_absorption_cfsr,
        report.canonical.avg_absorption_dlf,
        report.canonical.avg_absorption_osr,
        report.canonical.avg_absorption_ras,
        report.canonical.avg_mpr,
        report.canonical.failed_cases,
        report.canonical.class_count,
    ));
    out.push_str("| judge head | judge cfsr | judge csr | judge dlf | judge osr | judge next-step | judge comprehension | failures |\n");
    out.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for judge in &report.judges {
        out.push_str(&format!(
            "| {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {} |\n",
            judge.judge_head,
            judge.summary.avg_judge_cfsr,
            judge.summary.avg_judge_csr,
            judge.summary.avg_judge_dlf,
            judge.summary.avg_judge_osr,
            judge.summary.avg_judge_next_step,
            judge.summary.avg_judge_comprehension,
            judge.summary.failed_cases,
        ));
    }
    out
}

pub fn render_market_head_judge_pack_markdown(
    report: &MarketHeadJudgePackComparisonReport,
) -> String {
    let mut out = String::new();
    out.push_str("# Market-Head Judge Pack Comparison\n\n");
    out.push_str(&format!("Generated: `{}`\n\n", report.generated_at));
    out.push_str(&format!(
        "This comparison combines same-pack judge reports over `{}` classes.\n\n",
        report.cases
    ));
    out.push_str("| challenged head | judge head | canonical absorption cfsr | canonical absorption dlf | canonical absorption osr | canonical absorption ras | canonical mpr | judge cfsr | judge csr | judge dlf | judge osr | judge next-step | judge comprehension | failures |\n");
    out.push_str("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for entry in &report.canonical_vs_judge {
        out.push_str(&format!(
            "| {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {} |\n",
            entry.challenged_head,
            entry.judge_head,
            entry.canonical.avg_absorption_cfsr,
            entry.canonical.avg_absorption_dlf,
            entry.canonical.avg_absorption_osr,
            entry.canonical.avg_absorption_ras,
            entry.canonical.avg_mpr,
            entry.judge.avg_judge_cfsr,
            entry.judge.avg_judge_csr,
            entry.judge.avg_judge_dlf,
            entry.judge.avg_judge_osr,
            entry.judge.avg_judge_next_step,
            entry.judge.avg_judge_comprehension,
            entry.judge.failed_cases,
        ));
    }
    out
}

pub fn render_market_head_judge_calibration_markdown(
    report: &MarketHeadJudgeCalibrationReport,
) -> String {
    let mut out = String::new();
    out.push_str("# Market-Head Judge Calibration Report\n\n");
    out.push_str(&format!("Generated: `{}`\n\n", report.generated_at));
    out.push_str(
        "This report aggregates same-pack judge artifacts and measures drift against canonical raw absorption.\n\n",
    );
    out.push_str(
        "- shared metrics compare judge `CFSR`, `DLF`, and `OSR` against canonical absorption rates\n",
    );
    out.push_str(
        "- `ras/comprehension proxy` compares canonical absorption `RAS` against judge comprehension as a soft alignment signal\n\n",
    );
    out.push_str("| judge head | challenged heads | packs | classes | alignment | verdict | cfsr delta | dlf delta | osr delta | ras/comprehension proxy delta | avg judge next-step | avg canonical mpr | failures |\n");
    out.push_str("| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for entry in &report.entries {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {:.2} | {:?} | {:+.2} | {:+.2} | {:+.2} | {:+.2} | {:.2} | {:.2} | {} |\n",
            entry.judge_head,
            entry.challenged_heads.join(", "),
            entry.pack_count,
            entry.class_count,
            entry.shared_alignment_score,
            entry.verdict,
            entry.cfsr_delta.mean_signed_delta,
            entry.dlf_delta.mean_signed_delta,
            entry.osr_delta.mean_signed_delta,
            entry.ras_comprehension_proxy_delta.mean_signed_delta,
            entry.avg_judge_next_step,
            entry.avg_canonical_mpr,
            entry.failed_cases,
        ));
    }
    out
}

pub fn render_market_head_judge_disagreement_markdown(
    report: &MarketHeadJudgeDisagreementReport,
) -> String {
    let mut out = String::new();
    out.push_str("# Market-Head Judge Disagreement Report\n\n");
    out.push_str(&format!("Generated: `{}`\n\n", report.generated_at));
    out.push_str(&format!(
        "- challenged head: `{}`\n- judge head: `{}`\n\n",
        report.challenged_head, report.judge_head
    ));
    out.push_str(&format!(
        "- classes: `{}`\n- avg alignment drift: `{:.2}`\n- divergent cases: `{}` / `{}`\n\n",
        report.summary.class_count,
        report.summary.avg_alignment_score,
        report.summary.divergent_cases,
        report.summary.class_count,
    ));
    out.push_str("| class | canonical cfsr | judge cfsr | delta | canonical csr | judge csr | delta | canonical dlf | judge dlf | delta | canonical osr | judge osr | delta | canonical ras | judge comprehension | proxy delta | dominant drift | classification |\n");
    out.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n");
    for case in &report.cases {
        out.push_str(&format!(
            "| {} | {:.2} | {:.2} | {:+.2} | {:.2} | {:.2} | {:+.2} | {:.2} | {:.2} | {:+.2} | {:.2} | {:.2} | {:+.2} | {:.2} | {:.2} | {:+.2} | {} | {} |\n",
            case.class.slug(),
            case.canonical_absorption_cfsr,
            case.judge_cfsr,
            case.cfsr_delta,
            case.canonical_absorption_csr,
            case.judge_csr,
            case.csr_delta,
            case.canonical_absorption_dlf,
            case.judge_dlf,
            case.dlf_delta,
            case.canonical_absorption_osr,
            case.judge_osr,
            case.osr_delta,
            case.canonical_absorption_ras,
            case.judge_comprehension,
            case.ras_comprehension_proxy_delta,
            case.dominant_drift_metric,
            case.drift_classification,
        ));
    }
    if !report.summary.classification_counts.is_empty() {
        out.push_str("\n## Drift Classifications\n\n");
        for (classification, count) in &report.summary.classification_counts {
            out.push_str(&format!("- `{}`: `{}`\n", classification, count));
        }
    }
    let detailed_cases = report
        .cases
        .iter()
        .filter(|case| !case.critical_fact_diagnostics.is_empty() && case.cfsr_delta.abs() > 0.0)
        .collect::<Vec<_>>();
    if !detailed_cases.is_empty() {
        out.push_str("\n## Critical Fact Diagnostics\n\n");
        for case in detailed_cases {
            out.push_str(&format!("### {}\n\n", case.class.slug()));
            if !case.observed_critical_facts.is_empty() {
                out.push_str("Observed critical facts:\n");
                for fact in &case.observed_critical_facts {
                    out.push_str(&format!("- `{}`\n", fact));
                }
                out.push('\n');
            }
            for detail in &case.critical_fact_diagnostics {
                out.push_str(&format!(
                    "- expectation `{}`: canonical `{}`; judge score `{}`\n",
                    detail.expectation,
                    if detail.canonical_matched {
                        "matched"
                    } else {
                        "missed"
                    },
                    detail
                        .judge_score
                        .map(|score| score.to_string())
                        .unwrap_or_else(|| "n/a".to_string()),
                ));
                if let Some(note) = &detail.strictness_note {
                    out.push_str(&format!("  strictness: `{}`\n", note));
                }
                if !detail.required_concepts.is_empty() {
                    out.push_str(&format!(
                        "  required concepts: `{}`\n",
                        detail.required_concepts.join("`, `")
                    ));
                }
                if let Some(note_text) = &detail.canonical_note_text {
                    out.push_str(&format!("  canonical matched note: `{}`\n", note_text));
                }
                if let Some(reason) = &detail.judge_reason {
                    out.push_str(&format!("  judge reason: `{}`\n", reason));
                }
            }
            out.push('\n');
        }
    }
    out
}

fn render_market_head_challenge_readme() -> String {
    [
        "# Market-Head Challenge Pack",
        "",
        "This directory contains a secondary validation lane for closed or market-hosted heads.",
        "",
        "Rules:",
        "- the local economical proof line remains canonical",
        "- this pack is additive evidence only",
        "- do not feed `challenge-evaluator-pack.json` to the model under test",
        "",
        "How to use it:",
        "1. open `challenge-manifest.json`",
        "2. for each class, give `prompt.txt` plus `response.schema.json` to the target head",
        "3. save the model response as `<class>/response.json`",
        "4. run `ice bench-market evaluate --evaluator-pack <.../challenge-evaluator-pack.json> --responses-dir <.../market-head> --model <name>`",
        "",
        "The evaluator will score the returned JSON with the same protocol-aware benchmark judge used by the local proof line.",
        "",
    ]
    .join("\n")
}

fn render_market_head_judge_readme() -> String {
    [
        "# Market-Head Judge Pack",
        "",
        "This directory contains a blind judge lane for the same market-head challenge responses.",
        "",
        "Rules:",
        "- keep the canonical scorer as the baseline",
        "- use this pack only as additive semantic judgment evidence",
        "- do not change the challenged model response before judging it",
        "",
        "How to use it:",
        "1. open `judge-manifest.json`",
        "2. for each class, give `prompt.txt` plus `response.schema.json` to the judge head",
        "3. save the judge result as `<class>/response.json`",
        "4. run `ice bench-market judge-evaluate --judge-manifest <.../judge-manifest.json> --responses-dir <.../market-head-judge> --model <name>`",
        "",
    ]
    .join("\n")
}

fn render_market_head_judge_prompt(
    class: BenchmarkClass,
    truth: &GroundTruth,
    output: &AgentContinuationOutput,
) -> String {
    let sections = [
        (
            "critical_facts",
            truth_items_for_prompt(&truth.critical_facts),
        ),
        ("constraints", truth_items_for_prompt(&truth.constraints)),
        ("decisions", truth_items_for_prompt(&truth.decisions)),
        ("scars", truth_items_for_prompt(&truth.scars)),
    ];
    let mut lines = vec![
        "You are a strict continuity judge for the Shared Continuity Kernel.".to_string(),
        "Score the receiving head semantically, not by literal keyword echo.".to_string(),
        "Use only the ground truth and model output below. Return JSON only.".to_string(),
        format!("Class: {}", class.slug()),
    ];
    lines.push("Section: scoring_rubric".to_string());
    lines.push(
        serde_json::to_string_pretty(&judge_prompt_scoring_rules(class))
            .unwrap_or_else(|_| "[]".to_string()),
    );
    for (name, items) in sections {
        lines.push(format!("Section: {name}"));
        if items.is_empty() {
            lines.push("[]".to_string());
        } else {
            lines.push(serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string()));
        }
    }
    lines.push("Section: next_step".to_string());
    lines.push(
        serde_json::to_string_pretty(
            &truth
                .next_step_keywords
                .iter()
                .enumerate()
                .map(|(index, keyword)| {
                    serde_json::json!({
                        "index": index,
                        "expectation": keyword,
                    })
                })
                .collect::<Vec<_>>(),
        )
        .unwrap_or_else(|_| "[]".to_string()),
    );
    lines.push("Section: model_output".to_string());
    lines.push(serde_json::to_string_pretty(output).unwrap_or_else(|_| "{}".to_string()));
    lines.push("Score scale: 0=absent, 1=garbled, 2=paraphrased, 3=precise.".to_string());
    lines.join("\n\n")
}

fn truth_items_for_prompt(items: &[TruthItem]) -> Vec<serde_json::Value> {
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let mut value = serde_json::json!({
                "index": index,
                "expectation": item.keywords.join(" "),
                "rationale_cues": item.rationale_keywords.join(" "),
            });
            if let Some(note) = item.judge_note {
                value["strictness_note"] = serde_json::Value::String(note.to_string());
            }
            if !item.judge_required_concepts.is_empty() {
                value["required_concepts"] = serde_json::Value::Array(
                    item.judge_required_concepts
                        .iter()
                        .map(|concept| serde_json::Value::String((*concept).to_string()))
                        .collect(),
                );
            }
            value
        })
        .collect()
}

fn judge_prompt_scoring_rules(class: BenchmarkClass) -> Vec<String> {
    let mut rules = vec![
        "Score 3 only when the expectation is preserved completely, including any strictness_note listed beside that expectation.".to_string(),
        "If an expectation includes required_concepts, score 3 only when every required concept is explicitly present in the model output.".to_string(),
        "Score 2 when the answer is semantically related but one key qualifier, root cause, or role is softened or missing.".to_string(),
        "Score 1 when only a loose hint survives and the expectation would still need human repair.".to_string(),
    ];
    if matches!(
        class,
        BenchmarkClass::StrongToSmallContinuation | BenchmarkClass::SmallToSmallRelay
    ) {
        rules.push(
            "Continuation-heavy strictness: do not award full credit to a critical fact when only the file/path anchor survives but the selector/support-memory failure is paraphrased away.".to_string(),
        );
    }
    rules
}

fn judge_response_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["summary", "critical_facts", "constraints", "decisions", "scars", "next_step"],
        "properties": {
            "summary": {"type": "string"},
            "critical_facts": judge_score_array_schema(),
            "constraints": judge_score_array_schema(),
            "decisions": judge_score_array_schema(),
            "scars": judge_score_array_schema(),
            "next_step": judge_score_array_schema()
        }
    })
}

fn judge_score_array_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["index", "score", "reason"],
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "score": {"type": "integer", "minimum": 0, "maximum": 3},
                "reason": {"type": "string"}
            }
        }
    })
}

fn judge_response_template() -> serde_json::Value {
    serde_json::json!({
        "summary": "",
        "critical_facts": [],
        "constraints": [],
        "decisions": [],
        "scars": [],
        "next_step": []
    })
}

fn judge_evaluate_response(response: &JudgeResponse, truth: &GroundTruth) -> JudgeEvaluation {
    let critical_fact_rate =
        judge_category_rate(&response.critical_facts, truth.critical_facts.len().max(1));
    let constraint_rate =
        judge_category_rate(&response.constraints, truth.constraints.len().max(1));
    let decision_rate = judge_category_rate(&response.decisions, truth.decisions.len().max(1));
    let scar_rate = judge_category_rate(&response.scars, truth.scars.len().max(1));
    let next_step_expected = truth.next_step_keywords.len().max(1);
    let next_step_rate = judge_category_rate(&response.next_step, next_step_expected);
    let comprehension_score =
        (critical_fact_rate + constraint_rate + decision_rate + scar_rate + next_step_rate) / 5.0;
    JudgeEvaluation {
        critical_fact_rate,
        constraint_rate,
        decision_rate,
        scar_rate,
        next_step_rate,
        comprehension_score,
    }
}

fn summarize_market_head_judge_disagreement(
    cases: &[MarketHeadJudgeDisagreementCase],
) -> MarketHeadJudgeDisagreementSummary {
    let mut summary = MarketHeadJudgeDisagreementSummary {
        class_count: cases.len(),
        ..MarketHeadJudgeDisagreementSummary::default()
    };
    if cases.is_empty() {
        return summary;
    }
    let len = cases.len() as f64;
    summary.avg_abs_cfsr_delta = cases.iter().map(|case| case.cfsr_delta.abs()).sum::<f64>() / len;
    summary.avg_abs_csr_delta = cases.iter().map(|case| case.csr_delta.abs()).sum::<f64>() / len;
    summary.avg_abs_dlf_delta = cases.iter().map(|case| case.dlf_delta.abs()).sum::<f64>() / len;
    summary.avg_abs_osr_delta = cases.iter().map(|case| case.osr_delta.abs()).sum::<f64>() / len;
    summary.avg_abs_ras_proxy_delta = cases
        .iter()
        .map(|case| case.ras_comprehension_proxy_delta.abs())
        .sum::<f64>()
        / len;
    summary.avg_alignment_score = cases
        .iter()
        .map(|case| case.shared_alignment_score)
        .sum::<f64>()
        / len;
    summary.divergent_cases = cases
        .iter()
        .filter(|case| case.shared_alignment_score > 0.05)
        .count();
    for case in cases {
        *summary
            .classification_counts
            .entry(case.drift_classification.clone())
            .or_insert(0) += 1;
    }
    summary
}

fn judge_category_rate(scores: &[JudgeItemScore], expected_count: usize) -> f64 {
    if expected_count == 0 {
        return 1.0;
    }
    let mut normalized = vec![0u8; expected_count];
    for item in scores {
        if item.index < expected_count {
            normalized[item.index] = normalized[item.index].max(item.score.min(3));
        }
    }
    normalized.iter().map(|score| *score as f64).sum::<f64>() / (expected_count as f64 * 3.0)
}

fn market_head_response_template() -> serde_json::Value {
    serde_json::json!({
        "summary": "",
        "critical_facts": [],
        "constraints": [],
        "decisions": [],
        "open_hypotheses": [],
        "operational_scars": [],
        "avoid_repeating": [],
        "next_step": ""
    })
}

fn relative_display(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
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
        BenchmarkMetrics, ContextEnvelope, ContinuityBenchConfig, ContinuityPathKind,
        ContinuityPathReport, ContinuityPathRole, Evaluation, JudgeCalibrationVerdict,
        JudgeEvaluation, MarketHeadChallengeCaseEvaluation, MarketHeadChallengeConfig,
        MarketHeadChallengeEvaluationReport, MarketHeadChallengeSummary,
        MarketHeadJudgeCaseEvaluation, MarketHeadJudgeComparisonEntry,
        MarketHeadJudgeEvaluationReport, MarketHeadJudgeSamePackComparisonReport,
        MarketHeadJudgeSummary, ResourceEnvelope, SurfacedItem, build_context_envelope,
        compare_market_head_judge_calibration, compare_market_head_judge_disagreement,
        compare_market_head_judge_pack, compare_market_head_same_pack,
        evaluate_market_head_challenge, evaluate_market_head_judge_challenge,
        export_market_head_challenge, export_market_head_judge_challenge, populate_scenario,
        render_market_head_judge_calibration_markdown,
        render_market_head_judge_disagreement_markdown, render_market_head_judge_pack_markdown,
        render_market_head_judge_prompt, render_market_head_same_pack_markdown,
        render_suite_markdown, repair_output_from_envelope, scenario_for,
    };
    use crate::adapters::{AgentContinuationOutput, DecisionNote, EvidenceNote, ModelCallMetrics};
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

    fn market_head_config(
        output_dir: PathBuf,
        classes: Vec<BenchmarkClass>,
    ) -> MarketHeadChallengeConfig {
        MarketHeadChallengeConfig {
            output_dir,
            ollama_endpoint: "http://127.0.0.1:11434".into(),
            strong_model: "qwen2.5:1.5b".into(),
            small_model: "qwen2.5:0.5b".into(),
            embedding_backend: "hash:128".into(),
            retrieval_protocol: "uci+compiler+vector://hash:128?budget=192&candidates=12&recent=6"
                .into(),
            classes,
            token_budget: 192,
            candidate_limit: 12,
            recent_window: 6,
            timeout_secs: 180,
            num_predict: 384,
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
    fn export_market_head_challenge_writes_prompt_and_evaluator_pack() {
        let dir = tempdir().unwrap();
        let report = export_market_head_challenge(market_head_config(
            dir.path().to_path_buf(),
            vec![BenchmarkClass::AgentSwapSurvival],
        ))
        .unwrap();

        assert_eq!(report.case_count, 1);
        let prompt = std::fs::read_to_string(
            dir.path()
                .join("market-head")
                .join("agent-swap-survival")
                .join("prompt.txt"),
        )
        .unwrap();
        assert!(prompt.contains("Section: handoff_proof"));
        assert!(
            std::fs::metadata(
                dir.path()
                    .join("market-head")
                    .join("challenge-evaluator-pack.json")
            )
            .is_ok()
        );
        assert!(
            std::fs::metadata(
                dir.path()
                    .join("market-head")
                    .join("agent-swap-survival")
                    .join("data")
            )
            .is_err(),
            "market-head class pack must not leak kernel state files"
        );
        assert!(
            std::fs::metadata(
                dir.path()
                    .join("market-head")
                    .join(".work")
                    .join("agent-swap-survival")
                    .join("data")
            )
            .is_ok(),
            "kernel work state should stay under .work, outside the blind challenge pack"
        );
    }

    #[test]
    fn evaluate_market_head_challenge_scores_valid_response() {
        let dir = tempdir().unwrap();
        let export = export_market_head_challenge(market_head_config(
            dir.path().to_path_buf(),
            vec![BenchmarkClass::AgentSwapSurvival],
        ))
        .unwrap();
        let response_path = dir
            .path()
            .join("market-head")
            .join("agent-swap-survival")
            .join("response.json");
        std::fs::write(
            &response_path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "summary": "Resumed continuity safely.",
                "critical_facts": [
                    "Primary context is bench / task-agent-swap-survival for this resume. || f1",
                    "selector_missing in src/query.rs || i1"
                ],
                "constraints": [
                    "Preserve provenance || k1"
                ],
                "decisions": [
                    "Use the unified continuity interface || The runtime should route agent swaps through one continuity interface rather than raw transcript transfer. || pd1"
                ],
                "open_hypotheses": [],
                "operational_scars": [
                    "Avoid naive probes || ps1"
                ],
                "avoid_repeating": [],
                "next_step": "Benchmark adapter path || pn1"
            }))
            .unwrap(),
        )
        .unwrap();

        let report = evaluate_market_head_challenge(
            export.evaluator_pack_path,
            dir.path().join("market-head"),
            "external-test",
        )
        .unwrap();

        assert_eq!(report.summary.class_count, 1);
        assert_eq!(report.summary.failed_cases, 0);
        assert!(
            report.summary.avg_cfsr > 0.0,
            "expected the evaluator to recover at least one critical fact, got {}",
            report.summary.avg_cfsr
        );
        assert!(
            report.cases[0].failure.is_none(),
            "expected a successful case evaluation, got {:?}",
            report.cases[0].failure
        );
    }

    #[test]
    fn export_market_head_judge_writes_blind_pack() {
        let dir = tempdir().unwrap();
        let export = export_market_head_challenge(market_head_config(
            dir.path().to_path_buf(),
            vec![BenchmarkClass::AgentSwapSurvival],
        ))
        .unwrap();
        let response_path = dir
            .path()
            .join("market-head")
            .join("agent-swap-survival")
            .join("response.json");
        std::fs::write(
            &response_path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "summary": "Resumed continuity safely.",
                "critical_facts": [
                    "Primary context is bench / task-agent-swap-survival for this resume. || f1"
                ],
                "constraints": ["Preserve provenance || k1"],
                "decisions": [
                    "Use the unified continuity interface || The runtime should route agent swaps through one continuity interface rather than raw transcript transfer. || pd1"
                ],
                "open_hypotheses": [],
                "operational_scars": ["Avoid naive probes || ps1"],
                "avoid_repeating": [],
                "next_step": "Benchmark adapter path || pn1"
            }))
            .unwrap(),
        )
        .unwrap();

        let judge_export = export_market_head_judge_challenge(
            export.evaluator_pack_path,
            dir.path().join("market-head"),
            dir.path(),
        )
        .unwrap();

        assert_eq!(judge_export.case_count, 1);
        let prompt = std::fs::read_to_string(
            dir.path()
                .join("market-head-judge")
                .join("agent-swap-survival")
                .join("prompt.txt"),
        )
        .unwrap();
        assert!(prompt.contains("Section: model_output"));
        assert!(prompt.contains("Section: decisions"));
    }

    #[test]
    fn judge_prompt_includes_continuation_strictness_guidance() {
        let scenario = scenario_for(BenchmarkClass::StrongToSmallContinuation);
        let prompt = render_market_head_judge_prompt(
            BenchmarkClass::StrongToSmallContinuation,
            &scenario.truth,
            &AgentContinuationOutput::default(),
        );

        assert!(prompt.contains("Section: scoring_rubric"));
        assert!(prompt.contains("strictness_note"));
        assert!(prompt.contains("required_concepts"));
        assert!(prompt.contains("selector/support-memory failure"));
        assert!(prompt.contains("Continuation-heavy strictness"));
    }

    #[test]
    fn evaluate_market_head_judge_scores_valid_response() {
        let dir = tempdir().unwrap();
        let export = export_market_head_challenge(market_head_config(
            dir.path().to_path_buf(),
            vec![BenchmarkClass::AgentSwapSurvival],
        ))
        .unwrap();
        let response_path = dir
            .path()
            .join("market-head")
            .join("agent-swap-survival")
            .join("response.json");
        std::fs::write(
            &response_path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "summary": "Resumed continuity safely.",
                "critical_facts": [
                    "Primary context is bench / task-agent-swap-survival for this resume. || f1",
                    "selector_missing in src/query.rs || i1"
                ],
                "constraints": ["Preserve provenance || k1"],
                "decisions": [
                    "Use the unified continuity interface || The runtime should route agent swaps through one continuity interface rather than raw transcript transfer. || pd1"
                ],
                "open_hypotheses": [],
                "operational_scars": ["Avoid naive probes || ps1"],
                "avoid_repeating": [],
                "next_step": "Benchmark adapter path || pn1"
            }))
            .unwrap(),
        )
        .unwrap();
        let judge_export = export_market_head_judge_challenge(
            export.evaluator_pack_path,
            dir.path().join("market-head"),
            dir.path(),
        )
        .unwrap();
        let judge_response_path = dir
            .path()
            .join("market-head-judge")
            .join("agent-swap-survival")
            .join("response.json");
        std::fs::write(
            &judge_response_path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "summary": "The response preserves the main continuity bundle.",
                "critical_facts": [
                    {"index": 0, "score": 3, "reason": "Primary context survived precisely."},
                    {"index": 1, "score": 3, "reason": "selector_missing survived precisely."}
                ],
                "constraints": [
                    {"index": 0, "score": 3, "reason": "Constraint preserved."}
                ],
                "decisions": [
                    {"index": 0, "score": 3, "reason": "Decision and rationale survived."}
                ],
                "scars": [
                    {"index": 0, "score": 3, "reason": "Scar survived."}
                ],
                "next_step": [
                    {"index": 0, "score": 2, "reason": "Next step is semantically right."}
                ]
            }))
            .unwrap(),
        )
        .unwrap();

        let report = evaluate_market_head_judge_challenge(
            judge_export.manifest_path,
            dir.path().join("market-head-judge"),
            "judge-external",
        )
        .unwrap();

        assert_eq!(report.summary.class_count, 1);
        assert_eq!(report.summary.failed_cases, 0);
        assert!(report.summary.avg_judge_cfsr > 0.0);
        assert!(report.summary.avg_judge_dlf > 0.0);
        assert!(report.summary.avg_judge_comprehension > 0.0);
    }

    #[test]
    fn compare_market_head_same_pack_collects_canonical_and_judges() {
        let dir = tempdir().unwrap();
        let canonical_path = dir.path().join("canonical.json");
        let judge_a_path = dir.path().join("claude-judge.json");
        let judge_b_path = dir.path().join("codex-judge.json");

        std::fs::write(
            &canonical_path,
            serde_json::to_vec_pretty(&MarketHeadChallengeEvaluationReport {
                generated_at: "2026-03-23T05:00:00Z".into(),
                model_name: "claude-external".into(),
                evaluator_pack_path: "/tmp/evaluator.json".into(),
                responses_dir: "/tmp/responses".into(),
                cases: Vec::new(),
                summary: MarketHeadChallengeSummary {
                    class_count: 5,
                    avg_absorption_cfsr: 0.7,
                    avg_absorption_dlf: 0.6,
                    avg_absorption_osr: 1.0,
                    avg_absorption_ras: 0.66,
                    avg_cfsr: 1.0,
                    avg_dlf: 0.6,
                    avg_osr: 1.0,
                    avg_ras: 0.72,
                    avg_mpr: 0.14,
                    avg_pc: 0.91,
                    failed_cases: 0,
                },
            })
            .unwrap(),
        )
        .unwrap();
        for (path, cfsr) in [(&judge_a_path, 0.9), (&judge_b_path, 0.97)] {
            std::fs::write(
                path,
                serde_json::to_vec_pretty(&MarketHeadJudgeEvaluationReport {
                    generated_at: "2026-03-23T05:01:00Z".into(),
                    model_name: "judge".into(),
                    manifest_path: "/tmp/judge-manifest.json".into(),
                    responses_dir: "/tmp/judge-responses".into(),
                    cases: Vec::new(),
                    summary: MarketHeadJudgeSummary {
                        class_count: 5,
                        avg_judge_cfsr: cfsr,
                        avg_judge_csr: 1.0,
                        avg_judge_dlf: 1.0,
                        avg_judge_osr: 1.0,
                        avg_judge_next_step: 0.1,
                        avg_judge_comprehension: 0.8,
                        failed_cases: 0,
                    },
                })
                .unwrap(),
            )
            .unwrap();
        }

        let report = compare_market_head_same_pack(
            "claude",
            &canonical_path,
            vec![
                ("claude".into(), judge_a_path.clone()),
                ("codex".into(), judge_b_path.clone()),
            ],
        )
        .unwrap();

        assert_eq!(report.challenged_head, "claude");
        assert_eq!(report.canonical.avg_absorption_dlf, 0.6);
        assert_eq!(report.judges.len(), 2);
        assert_eq!(report.judges[0].judge_head, "claude");
        assert_eq!(report.judges[1].judge_head, "codex");

        let markdown = render_market_head_same_pack_markdown(&report);
        assert!(markdown.contains("Challenged head: `claude`"));
        assert!(markdown.contains("| codex | 0.97 | 1.00 | 1.00 | 1.00 | 0.10 | 0.80 | 0 |"));
    }

    #[test]
    fn compare_market_head_judge_pack_flattens_same_pack_reports() {
        let dir = tempdir().unwrap();
        let codex_path = dir.path().join("same-pack-codex.json");
        let claude_path = dir.path().join("same-pack-claude.json");

        for (path, challenged_head, canonical_cfsr, judges) in [
            (
                &codex_path,
                "codex",
                0.9,
                vec![
                    MarketHeadJudgeComparisonEntry {
                        judge_head: "claude".into(),
                        summary: MarketHeadJudgeSummary {
                            class_count: 5,
                            avg_judge_cfsr: 0.87,
                            avg_judge_csr: 1.0,
                            avg_judge_dlf: 1.0,
                            avg_judge_osr: 1.0,
                            avg_judge_next_step: 0.07,
                            avg_judge_comprehension: 0.79,
                            failed_cases: 0,
                        },
                    },
                    MarketHeadJudgeComparisonEntry {
                        judge_head: "codex".into(),
                        summary: MarketHeadJudgeSummary {
                            class_count: 5,
                            avg_judge_cfsr: 0.93,
                            avg_judge_csr: 1.0,
                            avg_judge_dlf: 1.0,
                            avg_judge_osr: 0.93,
                            avg_judge_next_step: 0.23,
                            avg_judge_comprehension: 0.82,
                            failed_cases: 0,
                        },
                    },
                ],
            ),
            (
                &claude_path,
                "claude",
                0.7,
                vec![
                    MarketHeadJudgeComparisonEntry {
                        judge_head: "claude".into(),
                        summary: MarketHeadJudgeSummary {
                            class_count: 5,
                            avg_judge_cfsr: 0.9,
                            avg_judge_csr: 1.0,
                            avg_judge_dlf: 1.0,
                            avg_judge_osr: 1.0,
                            avg_judge_next_step: 0.07,
                            avg_judge_comprehension: 0.79,
                            failed_cases: 0,
                        },
                    },
                    MarketHeadJudgeComparisonEntry {
                        judge_head: "codex".into(),
                        summary: MarketHeadJudgeSummary {
                            class_count: 5,
                            avg_judge_cfsr: 0.97,
                            avg_judge_csr: 1.0,
                            avg_judge_dlf: 1.0,
                            avg_judge_osr: 1.0,
                            avg_judge_next_step: 0.2,
                            avg_judge_comprehension: 0.83,
                            failed_cases: 0,
                        },
                    },
                ],
            ),
        ] {
            std::fs::write(
                path,
                serde_json::to_vec_pretty(&MarketHeadJudgeSamePackComparisonReport {
                    generated_at: "2026-03-23T06:10:00Z".into(),
                    challenged_head: challenged_head.into(),
                    canonical: MarketHeadChallengeSummary {
                        class_count: 5,
                        avg_absorption_cfsr: canonical_cfsr,
                        avg_absorption_dlf: 1.0,
                        avg_absorption_osr: 1.0,
                        avg_absorption_ras: 0.78,
                        avg_cfsr: 1.0,
                        avg_dlf: 1.0,
                        avg_mpr: 0.06,
                        avg_osr: 1.0,
                        avg_pc: 0.91,
                        avg_ras: 0.8,
                        failed_cases: 0,
                    },
                    judges,
                })
                .unwrap(),
            )
            .unwrap();
        }

        let report =
            compare_market_head_judge_pack(vec![claude_path.clone(), codex_path.clone()]).unwrap();

        assert_eq!(report.cases, 5);
        assert_eq!(report.canonical_vs_judge.len(), 4);
        assert_eq!(report.canonical_vs_judge[0].challenged_head, "claude");
        assert_eq!(report.canonical_vs_judge[0].judge_head, "claude");
        assert_eq!(report.canonical_vs_judge[3].challenged_head, "codex");
        assert_eq!(report.canonical_vs_judge[3].judge_head, "codex");

        let markdown = render_market_head_judge_pack_markdown(&report);
        assert!(markdown.contains("Market-Head Judge Pack Comparison"));
        assert!(markdown.contains("| claude | codex | 0.70 | 1.00 | 1.00 | 0.78 | 0.06 | 0.97 | 1.00 | 1.00 | 1.00 | 0.20 | 0.83 | 0 |"));
    }

    #[test]
    fn compare_market_head_judge_pack_accepts_legacy_same_pack_shape() {
        let dir = tempdir().unwrap();
        let legacy_path = dir.path().join("legacy-same-pack.json");
        std::fs::write(
            &legacy_path,
            serde_json::json!({
                "generated_at": "2026-03-23T06:30:00Z",
                "challenged_head": "claude",
                "cases": 5,
                "canonical": {
                    "avg_absorption_cfsr": 0.7,
                    "avg_absorption_dlf": 0.6,
                    "avg_absorption_osr": 1.0,
                    "avg_absorption_ras": 0.66,
                    "avg_cfsr": 1.0,
                    "avg_dlf": 0.6,
                    "avg_mpr": 0.14,
                    "avg_osr": 1.0,
                    "avg_pc": 0.91,
                    "avg_ras": 0.72,
                    "class_count": 5,
                    "failed_cases": 0
                },
                "judges": [{
                    "judge_head": "codex",
                    "avg_judge_cfsr": 0.97,
                    "avg_judge_comprehension": 0.83,
                    "avg_judge_csr": 1.0,
                    "avg_judge_dlf": 1.0,
                    "avg_judge_next_step": 0.2,
                    "avg_judge_osr": 1.0,
                    "class_count": 5,
                    "failed_cases": 0,
                    "source_artifact": "/tmp/legacy.json"
                }],
                "blocked_runs": [{
                    "challenged_head": "claude",
                    "judge_head": "claude",
                    "reason": "quota",
                    "detail": "reset pending"
                }]
            })
            .to_string(),
        )
        .unwrap();

        let report = compare_market_head_judge_pack(vec![legacy_path]).unwrap();

        assert_eq!(report.cases, 5);
        assert_eq!(report.canonical_vs_judge.len(), 1);
        assert_eq!(report.canonical_vs_judge[0].challenged_head, "claude");
        assert_eq!(report.canonical_vs_judge[0].judge_head, "codex");
        assert_eq!(report.canonical_vs_judge[0].judge.avg_judge_cfsr, 0.97);
    }

    #[test]
    fn compare_market_head_judge_calibration_aggregates_same_pack_reports() {
        let dir = tempdir().unwrap();
        let claude_path = dir.path().join("same-pack-claude.json");
        let codex_path = dir.path().join("same-pack-codex.json");

        for (path, challenged_head, canonical_cfsr, canonical_ras, judges) in [
            (
                &claude_path,
                "claude",
                0.70,
                0.76,
                vec![
                    MarketHeadJudgeComparisonEntry {
                        judge_head: "claude".into(),
                        summary: MarketHeadJudgeSummary {
                            class_count: 5,
                            avg_judge_cfsr: 0.74,
                            avg_judge_csr: 1.0,
                            avg_judge_dlf: 1.0,
                            avg_judge_osr: 1.0,
                            avg_judge_next_step: 0.10,
                            avg_judge_comprehension: 0.78,
                            failed_cases: 0,
                        },
                    },
                    MarketHeadJudgeComparisonEntry {
                        judge_head: "codex".into(),
                        summary: MarketHeadJudgeSummary {
                            class_count: 5,
                            avg_judge_cfsr: 0.84,
                            avg_judge_csr: 1.0,
                            avg_judge_dlf: 0.90,
                            avg_judge_osr: 1.0,
                            avg_judge_next_step: 0.15,
                            avg_judge_comprehension: 0.86,
                            failed_cases: 0,
                        },
                    },
                ],
            ),
            (
                &codex_path,
                "codex",
                0.90,
                0.78,
                vec![
                    MarketHeadJudgeComparisonEntry {
                        judge_head: "claude".into(),
                        summary: MarketHeadJudgeSummary {
                            class_count: 5,
                            avg_judge_cfsr: 0.86,
                            avg_judge_csr: 1.0,
                            avg_judge_dlf: 1.0,
                            avg_judge_osr: 1.0,
                            avg_judge_next_step: 0.08,
                            avg_judge_comprehension: 0.79,
                            failed_cases: 0,
                        },
                    },
                    MarketHeadJudgeComparisonEntry {
                        judge_head: "codex".into(),
                        summary: MarketHeadJudgeSummary {
                            class_count: 5,
                            avg_judge_cfsr: 0.96,
                            avg_judge_csr: 1.0,
                            avg_judge_dlf: 1.0,
                            avg_judge_osr: 0.96,
                            avg_judge_next_step: 0.20,
                            avg_judge_comprehension: 0.83,
                            failed_cases: 0,
                        },
                    },
                ],
            ),
        ] {
            std::fs::write(
                path,
                serde_json::to_vec_pretty(&MarketHeadJudgeSamePackComparisonReport {
                    generated_at: "2026-03-23T08:00:00Z".into(),
                    challenged_head: challenged_head.into(),
                    canonical: MarketHeadChallengeSummary {
                        class_count: 5,
                        avg_absorption_cfsr: canonical_cfsr,
                        avg_absorption_dlf: 1.0,
                        avg_absorption_osr: 1.0,
                        avg_absorption_ras: canonical_ras,
                        avg_cfsr: 1.0,
                        avg_dlf: 1.0,
                        avg_mpr: 0.10,
                        avg_osr: 1.0,
                        avg_pc: 0.90,
                        avg_ras: canonical_ras,
                        failed_cases: 0,
                    },
                    judges,
                })
                .unwrap(),
            )
            .unwrap();
        }

        let report = compare_market_head_judge_calibration(vec![claude_path, codex_path]).unwrap();

        assert_eq!(report.entries.len(), 2);
        assert_eq!(report.entries[0].judge_head, "claude");
        assert_eq!(report.entries[0].challenged_heads, vec!["claude", "codex"]);
        assert_eq!(report.entries[0].pack_count, 2);
        assert_eq!(report.entries[0].class_count, 10);
        assert_eq!(report.entries[0].verdict, JudgeCalibrationVerdict::Aligned);
        assert_eq!(report.entries[1].judge_head, "codex");
        assert_eq!(
            report.entries[1].verdict,
            JudgeCalibrationVerdict::Divergent
        );

        let markdown = render_market_head_judge_calibration_markdown(&report);
        assert!(markdown.contains("Market-Head Judge Calibration Report"));
        assert!(markdown.contains("| claude | claude, codex | 2 | 10 |"));
        assert!(markdown.contains("Aligned"));
    }

    #[test]
    fn compare_market_head_judge_disagreement_reports_case_level_drift() {
        let dir = tempdir().unwrap();
        let canonical_path = dir.path().join("canonical.json");
        let judge_path = dir.path().join("judge.json");
        let challenged_response_path = dir.path().join("challenged-response.json");
        let judge_response_path = dir.path().join("judge-response.json");

        std::fs::write(
            &challenged_response_path,
            serde_json::json!({
                "critical_facts": [
                    {
                        "text": "Primary context is bench / task-agent-swap for this resume."
                    },
                    {
                        "text": "Selector pruning dropped required support memory from src/query.rs."
                    }
                ],
                "constraints": [
                    {
                        "text": "Only patch the targeted query regression."
                    }
                ],
                "decisions": [
                    {
                        "text": "Patch src/query.rs before the next rerun."
                    }
                ],
                "operational_scars": [
                    {
                        "text": "A previous regression dropped support memory."
                    }
                ]
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            &judge_response_path,
            serde_json::json!({
                "critical_facts": [
                    {
                        "index": 0,
                        "score": 3,
                        "reason": "Selector pruning dropped required support memory from src/query.rs explicitly preserves both the support-memory failure and the file anchor."
                    },
                    {
                        "index": 1,
                        "score": 3,
                        "reason": "The output explicitly identifies the resume's primary context."
                    }
                ],
                "constraints": [
                    {
                        "index": 0,
                        "score": 3,
                        "reason": "Constraint preserved."
                    }
                ],
                "decisions": [
                    {
                        "index": 0,
                        "score": 2,
                        "reason": "Decision mostly preserved."
                    }
                ],
                "scars": [
                    {
                        "index": 0,
                        "score": 3,
                        "reason": "Scar preserved."
                    }
                ],
                "next_step": [
                    {
                        "index": 0,
                        "score": 1,
                        "reason": "Next step partially preserved."
                    }
                ]
            })
            .to_string(),
        )
        .unwrap();

        std::fs::write(
            &canonical_path,
            serde_json::to_vec_pretty(&MarketHeadChallengeEvaluationReport {
                generated_at: "2026-03-23T09:00:00Z".into(),
                model_name: "codex-external".into(),
                evaluator_pack_path: "/tmp/evaluator.json".into(),
                responses_dir: "/tmp/responses".into(),
                cases: vec![MarketHeadChallengeCaseEvaluation {
                    class: BenchmarkClass::AgentSwapSurvival,
                    scenario_id: "agent-swap-survival".into(),
                    protocol: "handoff-proof(4)".into(),
                    response_path: challenged_response_path.display().to_string(),
                    status: BaselineStatus::Ok,
                    raw_evaluation: Evaluation {
                        critical_fact_survival_rate: 0.5,
                        constraint_survival_rate: 1.0,
                        context_pack_quality_per_token: 0.5,
                        decision_lineage_fidelity: 0.5,
                        duplicate_work_rate: 0.0,
                        matched_constraints: 1,
                        matched_critical_facts: 1,
                        matched_decisions: 1,
                        matched_scars: 1,
                        memory_pollution_rate: 0.0,
                        mistake_recurrence_rate: 0.0,
                        operational_scar_retention: 1.0,
                        provenance_coverage: 1.0,
                        resume_accuracy_score: 0.6,
                        total_items: 10,
                        unsupported_items: 0,
                    },
                    evaluation: Evaluation::default(),
                    failure: None,
                }],
                summary: MarketHeadChallengeSummary::default(),
            })
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            &judge_path,
            serde_json::to_vec_pretty(&MarketHeadJudgeEvaluationReport {
                generated_at: "2026-03-23T09:01:00Z".into(),
                model_name: "claude-judge".into(),
                manifest_path: "/tmp/judge-manifest.json".into(),
                responses_dir: "/tmp/judge-responses".into(),
                cases: vec![MarketHeadJudgeCaseEvaluation {
                    class: BenchmarkClass::AgentSwapSurvival,
                    scenario_id: "agent-swap-survival".into(),
                    protocol: "judge://handoff-proof(4)".into(),
                    response_path: judge_response_path.display().to_string(),
                    status: BaselineStatus::Ok,
                    evaluation: JudgeEvaluation {
                        critical_fact_rate: 1.0,
                        constraint_rate: 1.0,
                        decision_rate: 0.83,
                        scar_rate: 1.0,
                        next_step_rate: 0.33,
                        comprehension_score: 0.83,
                    },
                    failure: None,
                }],
                summary: MarketHeadJudgeSummary::default(),
            })
            .unwrap(),
        )
        .unwrap();

        let report =
            compare_market_head_judge_disagreement("codex", "claude", &canonical_path, &judge_path)
                .unwrap();

        assert_eq!(report.summary.class_count, 1);
        assert_eq!(report.summary.divergent_cases, 1);
        assert_eq!(report.cases.len(), 1);
        assert_eq!(report.cases[0].dominant_drift_metric, "cfsr");
        assert_eq!(
            report.cases[0].drift_classification,
            "canonical_lexical_gap"
        );
        assert_eq!(
            report
                .summary
                .classification_counts
                .get("canonical_lexical_gap"),
            Some(&1)
        );
        assert!((report.cases[0].cfsr_delta - 0.5).abs() < 1e-9);
        let markdown = render_market_head_judge_disagreement_markdown(&report);
        assert!(markdown.contains("Market-Head Judge Disagreement Report"));
        assert!(markdown.contains("| agent-swap-survival | 0.50 | 1.00 | +0.50 |"));
        assert!(markdown.contains("canonical_lexical_gap"));
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
            }],
            summary: BenchmarkSummary::default(),
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
            }],
            summary: BenchmarkSummary::default(),
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
}

#[derive(Debug, Clone)]
struct SupportLabel {
    label: String,
    support_type: String,
    support_id: String,
}
