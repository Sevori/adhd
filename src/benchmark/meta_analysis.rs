use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::continuity::{
    ContinuityItemInput, ContinuityKind, ContinuityStatus, SharedContinuityKernel,
    UnifiedContinuityInterface,
};
use crate::model::{DimensionValue, MemoryLayer, Scope};

use super::BenchmarkSuiteReport;
use super::survival::{SurvivalOutcome, SurvivalRecord, TruthCategory};

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
    pub p_value: f64,
    pub adjusted_p_value: f64,
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
    pub sample_size: usize,
    pub benchmark_classes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLessonReport {
    pub generated_at: String,
    pub total_records: usize,
    pub lessons: Vec<MetaLesson>,
    pub candidates_tested: usize,
}

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
        let rank = rank_rev + 1;
        let corrected = raw_p * (m as f64) / (rank as f64);
        cumulative_min = cumulative_min.min(corrected).min(1.0);
        adjusted[orig_idx] = cumulative_min;
    }
    adjusted
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
                benchmark_classes: 0,
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

pub(crate) fn generate_meta_lessons(
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

pub fn write_meta_lessons_to_kernel(
    kernel: &SharedContinuityKernel,
    context_id: &str,
    report: &MetaLessonReport,
) -> Result<Vec<crate::continuity::ContinuityItemRecord>> {
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
                    DimensionValue {
                        key: "metacognitive_phase".into(),
                        value: "2".into(),
                        weight: 100,
                    },
                    DimensionValue {
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

#[cfg(test)]
mod tests {
    use super::super::runner::{build_context_envelope, populate_scenario, scenario_for};
    use super::super::survival::{ItemFeatures, category_stats};
    use super::*;
    use crate::adapters::AgentContinuationOutput;
    use crate::adapters::{EvidenceNote, ModelCallMetrics};
    use crate::benchmark::{
        BaselineKind, BaselineRunReport, BaselineStatus, BenchmarkClass, BenchmarkClassReport,
        BenchmarkMetrics, BenchmarkSuiteReport, BenchmarkSummary, ContinuityBenchConfig,
        Evaluation, ResourceEnvelope,
    };
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
        let raw = vec![0.001, 0.01, 0.04, 0.20, 0.80];
        let adj = benjamini_hochberg(&raw);
        for (r, a) in raw.iter().zip(adj.iter()) {
            assert!(*a >= *r, "adjusted {a} must be >= raw {r}");
        }
        assert!(adj[0] <= adj[1]);
        for a in &adj {
            assert!(*a <= 1.0, "adjusted p must be <= 1.0");
        }
        assert!(adj[3] > 0.05);
        assert!(adj[4] > 0.05);
    }

    #[test]
    fn compare_feature_distributions_finds_signal() {
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
        let (lessons, _) = compare_feature_distributions(&records, 0.05, 1, 5);
        let fp_lesson = lessons.iter().find(|l| l.feature_name == "file_path");
        assert!(fp_lesson.is_some(), "should detect file_path signal");
        assert_eq!(fp_lesson.unwrap().direction, LessonDirection::SurvivedMore);
    }

    #[test]
    fn compare_feature_distributions_ignores_small_samples() {
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
        let (lessons, candidates) = compare_feature_distributions(&records, 0.05, 3, 2);
        assert!(
            lessons.is_empty(),
            "should reject with < 3 benchmark classes"
        );
        assert_eq!(candidates, 0, "should not even test candidates");
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

        let survival = super::super::survival::benchmark_survival_analysis(
            &output,
            &scenario.truth,
            &envelope,
        );
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

        let written = write_meta_lessons_to_kernel(&kernel, &context.id, &meta);
        assert!(
            written.is_ok(),
            "kernel write must succeed: {:?}",
            written.err()
        );

        let json = serde_json::to_string_pretty(&meta).unwrap();
        assert!(json.contains("total_records"));
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
        assert!(has_sparse_cells(1, 1, 1, 50));
    }

    #[test]
    fn has_sparse_cells_returns_true_for_unbalanced_margins() {
        assert!(has_sparse_cells(2, 0, 8, 10));
    }

    // -----------------------------------------------------------------------
    // benjamini_hochberg edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn benjamini_hochberg_empty_input() {
        assert!(benjamini_hochberg(&[]).is_empty());
    }

    #[test]
    fn benjamini_hochberg_single_value() {
        let adj = benjamini_hochberg(&[0.03]);
        assert_eq!(adj.len(), 1);
        assert!((adj[0] - 0.03).abs() < f64::EPSILON);
    }

    #[test]
    fn benjamini_hochberg_all_significant() {
        let raw = vec![0.001, 0.002, 0.003];
        let adj = benjamini_hochberg(&raw);
        for a in &adj {
            assert!(*a < 0.05, "all should remain significant, got {a}");
        }
    }

    // -----------------------------------------------------------------------
    // Chi-squared edge cases
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

    // -----------------------------------------------------------------------
    // generate_meta_lessons with multiple suite reports
    // -----------------------------------------------------------------------

    #[test]
    fn generate_meta_lessons_multiple_suites_aggregates_records() {
        let make_survival = |records: Vec<SurvivalRecord>| super::super::survival::SurvivalReport {
            facts: category_stats(&records, TruthCategory::CriticalFact),
            constraints: category_stats(&records, TruthCategory::Constraint),
            decisions: category_stats(&records, TruthCategory::Decision),
            scars: category_stats(&records, TruthCategory::OperationalScar),
            records,
            surfaced_item_count: 0,
            surfaced_with_provenance: 0,
            total_envelope_tokens: 0,
        };

        let make_records = || {
            (0..6)
                .map(|i| SurvivalRecord {
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
        assert_eq!(report.total_records, 18);
    }
}
