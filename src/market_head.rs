use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::adapters::{
    AgentAdapter, AgentContinuationOutput, OllamaAdapter, parse_structured_output,
    render_structured_resume_prompt, structured_output_schema,
};
use crate::benchmark::{
    BaselineKind, BaselineStatus, BenchmarkClass, ContextEnvelope, ContinuityBenchConfig,
    Evaluation, GroundTruth, TruthItem, analyze_and_write_back, build_context_envelope,
    evaluate_output, failed_evaluation, format_continuity_path_label, match_keywords,
    populate_scenario, repair_output_from_envelope, scenario_for,
};
use crate::continuity::{
    AttachAgentInput, OpenContextInput, SharedContinuityKernel, SignalInput, SnapshotInput,
    UnifiedContinuityInterface,
};
use crate::model::{DimensionValue, MemoryLayer, Selector, SnapshotResolution};

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

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

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
    populate_scenario(&kernel, &context.id, &scenario)?;

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

// ---------------------------------------------------------------------------
// Summaries
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::AgentContinuationOutput;
    use crate::benchmark::{BaselineStatus, BenchmarkClass, Evaluation, scenario_for};
    use std::path::PathBuf;
    use tempfile::tempdir;

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
}
