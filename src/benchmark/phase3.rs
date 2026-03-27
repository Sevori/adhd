use std::path::Path;

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::adapters::{AgentAdapter, SurvivalHypothesis};
use crate::continuity::{
    ContinuityItemInput, ContinuityKind, ContinuityStatus, SharedContinuityKernel,
    UnifiedContinuityInterface,
};
use crate::model::{DimensionValue, MemoryLayer, Scope};

use super::meta_analysis::{LessonDirection, MetaLessonReport};
use super::runner::build_context_envelope;
use super::survival::{SurvivalReport, TruthCategory, category_rate};
use super::{BaselineKind, BenchmarkClass, ContinuityBenchConfig, Scenario};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Phase3Arm {
    Treatment,
    Control,
}

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

pub(crate) fn extract_eligible_hypotheses(report: &MetaLessonReport) -> Vec<SurvivalHypothesis> {
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

pub(crate) fn load_prior_hypotheses(output_dir: &Path) -> Vec<SurvivalHypothesis> {
    let meta_path = output_dir.join("meta-lessons.json");
    let Ok(contents) = std::fs::read_to_string(&meta_path) else {
        return Vec::new();
    };
    let Ok(report) = serde_json::from_str::<MetaLessonReport>(&contents) else {
        return Vec::new();
    };
    extract_eligible_hypotheses(&report)
}

pub(crate) fn detect_validation_cycle(output_dir: &Path) -> usize {
    let phase3_path = output_dir.join("phase3-report.json");
    let Ok(contents) = std::fs::read_to_string(&phase3_path) else {
        return 1;
    };
    let Ok(prior) = serde_json::from_str::<Phase3ValidationReport>(&contents) else {
        return 1;
    };
    prior.cycle + 1
}

pub(super) fn run_phase3_injection(
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

    let control_result = adapter.analyze(&scenario.objective, &envelope.text);
    let (control_survival, control_eval) = match control_result {
        Ok((output, _metrics)) => {
            let repaired = super::repair_output_from_envelope(output, &envelope);
            let eval = super::evaluate_output(&repaired, &scenario.truth, &envelope);
            let survival =
                super::survival::benchmark_survival_analysis(&repaired, &scenario.truth, &envelope);
            (Some(survival), eval)
        }
        Err(_) => (None, super::failed_evaluation(&envelope)),
    };

    let treatment_result =
        adapter.analyze_with_hypotheses(&scenario.objective, &envelope.text, hypotheses);
    let (treatment_survival, treatment_eval) = match treatment_result {
        Ok((output, _metrics)) => {
            let repaired = super::repair_output_from_envelope(output, &envelope);
            let eval = super::evaluate_output(&repaired, &scenario.truth, &envelope);
            let survival =
                super::survival::benchmark_survival_analysis(&repaired, &scenario.truth, &envelope);
            (Some(survival), eval)
        }
        Err(_) => (None, super::failed_evaluation(&envelope)),
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

pub(crate) fn generate_phase3_report(
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

#[cfg(test)]
mod tests {
    use super::super::runner::{build_context_envelope, populate_scenario, scenario_for};
    use super::*;
    use crate::adapters::AgentContinuationOutput;
    use crate::adapters::{EvidenceNote, ModelCallMetrics, SurvivalHypothesis};
    use crate::benchmark::meta_analysis::{MetaLesson, MetaLessonEvidence, MetaLessonReport};
    use crate::benchmark::survival::CategoryStats;
    use crate::benchmark::{BaselineKind, BenchmarkClass, ContinuityBenchConfig};
    use crate::continuity::{
        AttachAgentInput, OpenContextInput, SharedContinuityKernel, UnifiedContinuityInterface,
    };
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
        let result = load_prior_hypotheses(dir.path());
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

        let result = load_prior_hypotheses(dir.path());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].feature_name, "file_path");
    }

    #[test]
    fn detect_validation_cycle_returns_1_when_no_prior() {
        let dir = tempdir().unwrap();
        assert_eq!(detect_validation_cycle(dir.path()), 1);
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

        assert_eq!(detect_validation_cycle(dir.path()), 4);
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

        let control_eval =
            super::super::evaluate_output(&control_output, &scenario.truth, &envelope);
        let treatment_eval =
            super::super::evaluate_output(&treatment_output, &scenario.truth, &envelope);
        let control_survival = super::super::survival::benchmark_survival_analysis(
            &control_output,
            &scenario.truth,
            &envelope,
        );
        let treatment_survival = super::super::survival::benchmark_survival_analysis(
            &treatment_output,
            &scenario.truth,
            &envelope,
        );

        assert!(
            treatment_eval.resume_accuracy_score >= control_eval.resume_accuracy_score,
            "treatment ({}) should score >= control ({})",
            treatment_eval.resume_accuracy_score,
            control_eval.resume_accuracy_score
        );

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

        let written = write_phase3_outcomes_to_kernel(&kernel, &context.id, &phase3_report);
        assert!(
            written.is_ok(),
            "kernel write must succeed: {:?}",
            written.err()
        );

        let json = serde_json::to_string_pretty(&phase3_report).unwrap();
        assert!(json.contains("hypotheses_tested"));
        assert!(json.contains("cycle"));
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
}
