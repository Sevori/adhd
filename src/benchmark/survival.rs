use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::adapters::AgentContinuationOutput;

use super::{ContextEnvelope, GroundTruth, TruthItem, match_keywords};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TruthCategory {
    CriticalFact,
    Constraint,
    Decision,
    OperationalScar,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SurvivalOutcome {
    Survived,
    Lost,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemFeatures {
    pub keyword_count: usize,
    pub keyword_char_length: usize,
    pub contains_file_path: bool,
    pub contains_numeric_ref: bool,
    pub matched_note_text: Option<String>,
    pub matched_note_tokens: Option<usize>,
    pub prohibition_framing: Option<bool>,
    pub aspiration_framing: Option<bool>,
    pub evidence_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalRecord {
    pub category: TruthCategory,
    pub outcome: SurvivalOutcome,
    pub keywords: Vec<String>,
    pub features: ItemFeatures,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CategoryStats {
    pub total: usize,
    pub survived: usize,
    pub lost: usize,
    pub rate: f64,
    pub avg_keyword_count_survived: f64,
    pub avg_keyword_count_lost: f64,
    pub file_path_rate_survived: f64,
    pub file_path_rate_lost: f64,
    pub prohibition_rate_survived: f64,
    pub aspiration_rate_survived: f64,
}

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

pub(crate) fn benchmark_survival_analysis(
    output: &AgentContinuationOutput,
    truth: &GroundTruth,
    envelope: &ContextEnvelope,
) -> SurvivalReport {
    let evidence_labels: HashSet<String> =
        envelope.surfaced.iter().map(|s| s.label.clone()).collect();

    let mut records = Vec::new();

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

pub(crate) fn category_stats(records: &[SurvivalRecord], category: TruthCategory) -> CategoryStats {
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

pub(crate) fn category_rate(survival: &SurvivalReport, category: TruthCategory) -> f64 {
    match category {
        TruthCategory::CriticalFact => survival.facts.rate,
        TruthCategory::Constraint => survival.constraints.rate,
        TruthCategory::Decision => survival.decisions.rate,
        TruthCategory::OperationalScar => survival.scars.rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::{AgentContinuationOutput, DecisionNote, EvidenceNote};
    use crate::benchmark::{BaselineKind, ContextEnvelope, GroundTruth, SurfacedItem, TruthItem};

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

        assert_eq!(report.facts.total, 2);
        assert_eq!(report.facts.survived, 1);
        assert_eq!(report.facts.lost, 1);
        assert!((report.facts.rate - 0.5).abs() < f64::EPSILON);
        assert_eq!(report.constraints.total, 1);
        assert_eq!(report.constraints.survived, 1);
        assert_eq!(report.decisions.total, 1);
        assert_eq!(report.decisions.lost, 1);
        assert_eq!(report.scars.total, 1);
        assert_eq!(report.scars.survived, 1);
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
        let output = AgentContinuationOutput {
            critical_facts: vec![EvidenceNote {
                text: "selector_missing in src/query.rs".into(),
                evidence: Vec::new(),
            }],
            ..Default::default()
        };

        let report = benchmark_survival_analysis(&output, &truth, &envelope);

        assert_eq!(report.facts.file_path_rate_survived, 1.0);
        assert_eq!(report.facts.file_path_rate_lost, 0.0);
    }

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
}
