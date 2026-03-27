mod meta_analysis;
mod phase3;
mod runner;
mod survival;

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use anyhow::Result;
use chrono::Utc;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::adapters::{AgentContinuationOutput, DecisionNote, EvidenceNote};
use crate::model::DimensionValue;

// Re-export everything that external modules need.
pub(crate) use meta_analysis::generate_meta_lessons;
pub use meta_analysis::{
    LessonDirection, MetaLesson, MetaLessonEvidence, MetaLessonReport, write_meta_lessons_to_kernel,
};
pub use phase3::{
    HypothesisValidationResult, Phase3Arm, Phase3ClassResult, Phase3ValidationReport,
    write_phase3_outcomes_to_kernel,
};
pub(crate) use runner::{
    analyze_and_write_back, build_context_envelope, format_continuity_path_label,
    populate_scenario, scenario_for,
};
pub use survival::{
    CategoryStats, ItemFeatures, SurvivalOutcome, SurvivalRecord, SurvivalReport, TruthCategory,
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

    pub fn strong_agent(
        &self,
        agent_id: &str,
        role: &str,
        namespace: &str,
    ) -> crate::adapters::AgentAdapterConfig {
        crate::adapters::AgentAdapterConfig {
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

    pub fn small_agent(
        &self,
        agent_id: &str,
        role: &str,
        namespace: &str,
    ) -> crate::adapters::AgentAdapterConfig {
        crate::adapters::AgentAdapterConfig {
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

    pub(crate) fn uses_handoff_proof(self) -> bool {
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

    pub(crate) fn shared_path_role(self) -> ContinuityPathRole {
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hypothesis_injection: Option<Phase3ClassResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineRunReport {
    pub baseline: BaselineKind,
    pub status: BaselineStatus,
    pub model: String,
    pub retrieval_ms: u128,
    pub model_metrics: crate::adapters::ModelCallMetrics,
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
    pub(crate) kind: crate::model::EventKind,
    pub(crate) scope: crate::model::Scope,
    pub(crate) content: String,
    pub(crate) dimensions: Vec<DimensionValue>,
    pub(crate) marks: Vec<ScenarioMark>,
}

#[derive(Debug, Clone)]
pub(crate) struct ScenarioMark {
    pub(crate) kind: crate::continuity::ContinuityKind,
    pub(crate) title: String,
    pub(crate) body: String,
    pub(crate) status: crate::continuity::ContinuityStatus,
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

// ---------------------------------------------------------------------------
// Shared helper functions used across submodules
// ---------------------------------------------------------------------------

pub(crate) fn match_keywords(text: &str, keywords: &[&str]) -> bool {
    let lowered = text.to_lowercase();
    keywords
        .iter()
        .all(|keyword| lowered.contains(&keyword.to_lowercase()))
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

fn flatten_keywords(items: &[TruthItem]) -> Vec<&str> {
    items
        .iter()
        .flat_map(|item| item.keywords.clone())
        .collect()
}

pub(crate) fn estimate_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}

pub(crate) fn trim_text(text: &str, max: usize) -> String {
    if text.len() <= max {
        text.to_string()
    } else {
        format!("{}...", &text[..max])
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

fn normalized_text(text: &str) -> String {
    text.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

// ---------------------------------------------------------------------------
// Top-level suite runner
// ---------------------------------------------------------------------------

pub fn run_continuity_suite(config: ContinuityBenchConfig) -> Result<BenchmarkSuiteReport> {
    std::fs::create_dir_all(&config.output_dir)?;
    let started = Utc::now();

    let prior_hypotheses = phase3::load_prior_hypotheses(&config.output_dir);

    let mut classes = Vec::new();
    for class in config.selected_classes() {
        classes.push(runner::run_class(class, &config, &prior_hypotheses)?);
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

    let meta = generate_meta_lessons(&[report.clone()], 0.05);
    if !meta.lessons.is_empty() {
        let meta_path = config.output_dir.join("meta-lessons.json");
        std::fs::write(&meta_path, serde_json::to_vec_pretty(&meta)?)?;
    }
    report.meta_lessons = Some(meta);

    if !prior_hypotheses.is_empty() {
        let class_results: Vec<Phase3ClassResult> = report
            .classes
            .iter()
            .filter_map(|cr| cr.hypothesis_injection.clone())
            .collect();
        if !class_results.is_empty() {
            let cycle = phase3::detect_validation_cycle(&config.output_dir);
            let phase3_report =
                phase3::generate_phase3_report(&class_results, &prior_hypotheses, cycle);
            let phase3_path = config.output_dir.join("phase3-report.json");
            std::fs::write(&phase3_path, serde_json::to_vec_pretty(&phase3_report)?)?;
            report.phase3 = Some(phase3_report);
        }
    }

    let suite_report_path = config.output_dir.join("suite-report.json");
    std::fs::write(&suite_report_path, serde_json::to_vec_pretty(&report)?)?;
    let summary_path = config.output_dir.join("summary.md");
    std::fs::write(&summary_path, render_suite_markdown(&report))?;
    Ok(report)
}

fn summarize(classes: &[BenchmarkClassReport]) -> BenchmarkSummary {
    let failed_runs = classes
        .iter()
        .flat_map(|item| std::iter::once(&item.continuity).chain(item.baselines.iter()))
        .filter(|run| run.status == BaselineStatus::Failed)
        .count();
    if classes.is_empty() {
        return BenchmarkSummary {
            class_count: 0,
            total_runs: 0,
            failed_runs,
            ..BenchmarkSummary::default()
        };
    }
    let n = classes.len() as f64;
    BenchmarkSummary {
        class_count: classes.len(),
        total_runs: classes.len() * 6,
        failed_runs,
        avg_cfsr: classes.iter().map(|item| item.metrics.cfsr).sum::<f64>() / n,
        avg_csr: classes.iter().map(|item| item.metrics.csr).sum::<f64>() / n,
        avg_dlf: classes.iter().map(|item| item.metrics.dlf).sum::<f64>() / n,
        avg_ras: classes.iter().map(|item| item.metrics.ras).sum::<f64>() / n,
        avg_mpr: classes.iter().map(|item| item.metrics.mpr).sum::<f64>() / n,
        avg_pc: classes.iter().map(|item| item.metrics.pc).sum::<f64>() / n,
        avg_smcl: classes.iter().map(|item| item.metrics.smcl).sum::<f64>() / n,
        avg_sscr: classes.iter().map(|item| item.metrics.sscr).sum::<f64>() / n,
        avg_cgd: classes.iter().map(|item| item.metrics.cgd).sum::<f64>() / n,
    }
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

// ---------------------------------------------------------------------------
// Repair helpers (used by runner and needed by market_head via re-export)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::{AgentContinuationOutput, DecisionNote, EvidenceNote, ModelCallMetrics};
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
    // Evaluation scoring logic
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

        let notes = vec![DecisionNote {
            text: "Use the unified interface".into(),
            rationale: "Better performance".into(),
            evidence: vec!["f1".into()],
        }];
        let score = score_decisions(&notes, &truth, &labels);
        assert_eq!(score.matched, 0);

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
        assert_eq!(runner::section_prefix("constraints"), "k");
        assert_eq!(runner::section_prefix("decisions"), "d");
        assert_eq!(runner::section_prefix("hypotheses"), "h");
        assert_eq!(runner::section_prefix("incidents"), "i");
        assert_eq!(runner::section_prefix("operational_scars"), "s");
        assert_eq!(runner::section_prefix("open_threads"), "t");
        assert_eq!(runner::section_prefix("unknown"), "x");
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
        use BenchmarkClass::*;
        assert!(!BaselineIsolation.uses_handoff_proof());
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
        use ContinuityPathRole::*;
        assert_eq!(
            BenchmarkClass::AgentSwapSurvival.shared_path_role(),
            ProofPath
        );
        assert_eq!(
            BenchmarkClass::BaselineIsolation.shared_path_role(),
            ExplicitControl
        );
    }
}
