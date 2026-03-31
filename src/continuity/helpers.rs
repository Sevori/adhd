use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Duration, Utc};

use crate::dispatch;
use crate::model::{DimensionValue, EventInput, Scope, Selector};

use super::schema::*;
use super::types::*;

pub(crate) fn compile_handoff_proof(read: &ContextRead) -> HandoffProof {
    let mut registers = Vec::new();

    let primary_fact = read
        .incidents
        .first()
        .map(|item| {
            (
                "pf1",
                "fact",
                item.id.clone(),
                item.title.clone(),
                item.body.clone(),
                !item.supports.is_empty(),
            )
        })
        .or_else(|| {
            read.recall.items.first().map(|item| {
                (
                    "pf1",
                    "fact",
                    item.id.clone(),
                    item.title.clone(),
                    item.preview.clone(),
                    item.support_count > 0,
                )
            })
        })
        .unwrap_or_else(|| {
            (
                "pf1",
                "fact",
                read.context.id.clone(),
                "Primary context".to_string(),
                format!(
                    "Resume inside {} / {}.",
                    read.context.namespace, read.context.task_id
                ),
                true,
            )
        });
    registers.push(HandoffProofRegister {
        label: primary_fact.0.to_string(),
        register_kind: primary_fact.1.to_string(),
        source_id: primary_fact.2,
        title: primary_fact.3,
        body: trim_text(&primary_fact.4, PROOF_TRIM_LIMIT),
        has_provenance: primary_fact.5,
    });

    let push_item_register = |regs: &mut Vec<HandoffProofRegister>,
                              label: &str,
                              kind: &str,
                              item: &ContinuityItemRecord| {
        regs.push(HandoffProofRegister {
            label: label.to_string(),
            register_kind: kind.to_string(),
            source_id: item.id.clone(),
            title: item.title.clone(),
            body: trim_text(&item.body, PROOF_TRIM_LIMIT),
            has_provenance: !item.supports.is_empty(),
        });
    };

    if let Some(item) = read.decisions.first() {
        push_item_register(&mut registers, "pd1", "decision", item);
    }
    if let Some(item) = read.constraints.first() {
        push_item_register(&mut registers, "pk1", "constraint", item);
    }
    if let Some(item) = read.operational_scars.first() {
        push_item_register(&mut registers, "ps1", "scar", item);
    }
    if let Some(item) = read
        .working_state
        .iter()
        .find(|item| {
            item.title == "model-next-step" || item.extra["next_step"].as_bool() == Some(true)
        })
        .or_else(|| read.working_state.first())
    {
        push_item_register(&mut registers, "pn1", "next_step", item);
    }

    let digest = registers
        .iter()
        .map(|item| format!("{}:{}:{}", item.label, item.register_kind, item.title))
        .collect::<Vec<_>>()
        .join(" | ");

    HandoffProof { digest, registers }
}

pub(crate) fn trim_text(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.trim().to_string();
    }
    let mut trimmed = String::new();
    for ch in text.chars().take(limit.saturating_sub(3)) {
        trimmed.push(ch);
    }
    trimmed.push_str("...");
    trimmed.trim().to_string()
}

pub(crate) fn inject_context(event: &mut EventInput, context: &ContextRecord) {
    event.namespace = Some(context.namespace.clone());
    event.task_id = Some(context.task_id.clone());
    event.dimensions = augment_dimensions(
        std::mem::take(&mut event.dimensions),
        vec![
            DimensionValue {
                key: "context".to_string(),
                value: context.id.clone(),
                weight: DEFAULT_DIMENSION_WEIGHT,
            },
            DimensionValue {
                key: "context_namespace".to_string(),
                value: context.namespace.clone(),
                weight: DEFAULT_DIMENSION_WEIGHT,
            },
            DimensionValue {
                key: "context_task".to_string(),
                value: context.task_id.clone(),
                weight: DEFAULT_DIMENSION_WEIGHT,
            },
        ],
    );
}

pub(crate) fn resolve_namespace(
    storage: &crate::storage::Storage,
    namespace: Option<String>,
) -> anyhow::Result<Option<String>> {
    match namespace {
        Some(namespace) => storage.resolve_namespace_alias(Some(namespace.as_str())),
        None => Ok(None),
    }
}

pub(crate) fn resolve_selector_namespace(
    storage: &crate::storage::Storage,
    selector: Option<Selector>,
) -> anyhow::Result<Option<Selector>> {
    let Some(mut selector) = selector else {
        return Ok(None);
    };
    selector.namespace = storage.resolve_namespace_alias(selector.namespace.as_deref())?;
    Ok(Some(selector))
}

pub(crate) fn merge_context_selector(
    context: &ContextRecord,
    selector: Option<Selector>,
) -> Selector {
    let mut selector = selector.unwrap_or_else(|| context.selector.clone());
    selector.namespace = Some(context.namespace.clone());
    selector.all.push(crate::model::DimensionFilter {
        key: "context".to_string(),
        values: vec![context.id.clone()],
    });
    selector
}

pub(crate) fn filter_kind(
    items: &[ContinuityItemRecord],
    kind: ContinuityKind,
) -> Vec<ContinuityItemRecord> {
    items
        .iter()
        .filter(|item| item.kind == kind)
        .cloned()
        .collect()
}

const RECENT_LEARNING_WINDOW_DAYS: i64 = 7;
const RECENT_LEARNING_LIMIT: usize = 6;
const LEARNING_SUMMARY_LIMIT: usize = 3;
const CURRENT_PRACTICE_LIMIT: usize = 5;
const CURRENT_PRACTICE_EVIDENCE_LIMIT: usize = 3;
const CURRENT_PRACTICE_EVIDENCE_SUMMARY_LIMIT: usize = 2;
const OPERATIONAL_STATE_LIMIT: usize = 8;
const RECENT_UPDATE_LIMIT: usize = 6;
const ACTIVE_THREAD_LIMIT: usize = 6;

pub(crate) fn build_learning_view(
    objective: &str,
    items: &[ContinuityItemRecord],
    now: DateTime<Utc>,
) -> LearningView {
    let mut candidates = learning_candidates(items);
    if objective_requests_history_context(objective) {
        candidates.sort_by(|left, right| {
            left.updated_at
                .cmp(&right.updated_at)
                .then_with(|| left.created_at.cmp(&right.created_at))
                .then_with(|| left.title.cmp(&right.title))
        });
        let summary = summarize_learning_line(&candidates, now);
        return LearningView {
            mode: LearningViewMode::Lineage,
            summary,
            items: candidates,
        };
    }

    let recent_cutoff = now - Duration::days(RECENT_LEARNING_WINDOW_DAYS);
    let recent_window_items = candidates
        .iter()
        .filter(|item| item.updated_at >= recent_cutoff)
        .cloned()
        .collect::<Vec<_>>();
    let used_recent_window = !recent_window_items.is_empty();
    let mut recent_items = if used_recent_window {
        recent_window_items
    } else {
        candidates
    };
    recent_items.truncate(RECENT_LEARNING_LIMIT);
    let summary = summarize_recent_learning(&recent_items, used_recent_window, now);

    LearningView {
        mode: LearningViewMode::Recent,
        summary,
        items: recent_items,
    }
}

pub(crate) fn annotate_practice_states(items: &mut [ContinuityItemRecord], now: DateTime<Utc>) {
    let items_by_id = items
        .iter()
        .map(|item| (item.id.clone(), item.clone()))
        .collect::<BTreeMap<_, _>>();
    let support_index = current_practice_support_index(items, &items_by_id);
    let mut strongest_clusters = BTreeMap::<String, (usize, f64, DateTime<Utc>)>::new();
    let mut baseline_states = vec![None; items.len()];
    let mut practice_ranks = vec![0.0; items.len()];

    for (index, item) in items.iter().enumerate() {
        let state = derive_practice_state(item, now);
        baseline_states[index] = state;
        let Some(state) = state else {
            continue;
        };
        let support_signal =
            current_practice_support_signal(item, &items_by_id, &support_index, now);
        practice_ranks[index] = current_practice_rank(item, state, now, support_signal);
        if let Some(cluster_key) = continuity_practice_cluster_key(item) {
            strongest_clusters
                .entry(cluster_key)
                .and_modify(|entry| {
                    let current = (practice_ranks[index], item.updated_at);
                    let best = (entry.1, entry.2);
                    if current.0 > best.0 || (current.0 == best.0 && current.1 > best.1) {
                        *entry = (index, current.0, current.1);
                    }
                })
                .or_insert((index, practice_ranks[index], item.updated_at));
        }
    }

    for (index, item) in items.iter_mut().enumerate() {
        let mut state = baseline_states[index];
        if let (Some(current_state), Some(cluster_key)) =
            (state, continuity_practice_cluster_key(item))
            && let Some((winner_index, winner_rank, winner_updated_at)) =
                strongest_clusters.get(&cluster_key)
            && *winner_index != index
            && current_state != PracticeLifecycleState::Retired
        {
            let rank_gap = *winner_rank - practice_ranks[index];
            if rank_gap >= 0.14 || *winner_updated_at > item.updated_at {
                state = Some(PracticeLifecycleState::Stale);
            } else if current_state == PracticeLifecycleState::Current {
                state = Some(PracticeLifecycleState::Aging);
            }
        }
        item.practice_state = state;
    }
}

pub(crate) fn build_current_practice_view(
    items: &[ContinuityItemRecord],
    now: DateTime<Utc>,
) -> PracticeView {
    let items_by_id = items
        .iter()
        .map(|item| (item.id.clone(), item.clone()))
        .collect::<BTreeMap<_, _>>();
    let support_index = current_practice_support_index(items, &items_by_id);
    let mut practice_items = items
        .iter()
        .filter(|item| is_current_practice_candidate(item))
        .cloned()
        .collect::<Vec<_>>();
    let support_signal_by_id = practice_items
        .iter()
        .map(|item| {
            (
                item.id.clone(),
                current_practice_support_signal(item, &items_by_id, &support_index, now),
            )
        })
        .collect::<BTreeMap<_, _>>();
    practice_items.sort_by(|left, right| {
        practice_state_sort_rank(right.practice_state)
            .cmp(&practice_state_sort_rank(left.practice_state))
            .then_with(|| {
                support_signal_by_id
                    .get(&right.id)
                    .copied()
                    .unwrap_or_default()
                    .partial_cmp(
                        &support_signal_by_id
                            .get(&left.id)
                            .copied()
                            .unwrap_or_default(),
                    )
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| right.updated_at.cmp(&left.updated_at))
            .then_with(|| {
                right
                    .importance
                    .partial_cmp(&left.importance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| right.title.cmp(&left.title))
    });

    let mut seen_clusters = BTreeSet::new();
    let mut selected = Vec::new();
    for item in practice_items {
        if let Some(cluster_key) = continuity_practice_cluster_key(&item)
            && !seen_clusters.insert(cluster_key)
        {
            continue;
        }
        selected.push(item);
        if selected.len() >= CURRENT_PRACTICE_LIMIT {
            break;
        }
    }

    let evidence = selected
        .iter()
        .filter_map(|item| {
            let evidence = current_practice_evidence_items(item, &items_by_id, &support_index, now);
            if evidence.is_empty() {
                return None;
            }
            Some(PracticeEvidenceRecord {
                practice_id: item.id.clone(),
                support_signal: support_signal_by_id
                    .get(&item.id)
                    .copied()
                    .unwrap_or_default(),
                evidence_count: evidence.len(),
                evidence,
            })
        })
        .collect::<Vec<_>>();

    let summary = summarize_current_practice(&selected, &evidence, now);
    PracticeView {
        summary,
        items: selected,
        evidence,
    }
}

pub(crate) fn build_operational_state_view(
    items: &[ContinuityItemRecord],
    now: DateTime<Utc>,
) -> Vec<ContinuityItemRecord> {
    let mut ranked = items
        .iter()
        .filter_map(|item| operational_state_candidate_score(item, now).map(|score| (score, item)))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.1.updated_at.cmp(&left.1.updated_at))
            .then_with(|| right.1.title.cmp(&left.1.title))
    });

    let mut selected = Vec::new();
    let mut seen_clusters = BTreeSet::<String>::new();
    for (_, item) in ranked {
        if let Some(cluster_key) = continuity_practice_cluster_key(item)
            && !seen_clusters.insert(cluster_key)
        {
            continue;
        }
        selected.push(item.clone());
        if selected.len() >= OPERATIONAL_STATE_LIMIT {
            break;
        }
    }

    selected
}

pub(crate) fn build_next_step_view(items: &[ContinuityItemRecord]) -> Vec<ContinuityItemRecord> {
    let mut next_steps = items
        .iter()
        .filter(|item| is_operational_next_step(item))
        .cloned()
        .collect::<Vec<_>>();
    next_steps.sort_by(|left, right| {
        right
            .updated_at
            .cmp(&left.updated_at)
            .then_with(|| {
                right
                    .retention
                    .effective_salience
                    .partial_cmp(&left.retention.effective_salience)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| right.title.cmp(&left.title))
    });
    next_steps.truncate(3);
    next_steps
}

pub(crate) fn build_recent_update_view(
    items: &[ContinuityItemRecord],
    now: DateTime<Utc>,
) -> Vec<ContinuityItemRecord> {
    let recent_cutoff = now - Duration::days(RECENT_LEARNING_WINDOW_DAYS);
    let mut ranked = items
        .iter()
        .filter_map(|item| {
            recent_update_candidate_score(item, recent_cutoff, now).map(|score| (score, item))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.1.updated_at.cmp(&left.1.updated_at))
            .then_with(|| right.1.title.cmp(&left.1.title))
    });

    let mut selected = Vec::new();
    let mut seen_clusters = BTreeSet::<String>::new();
    for (_, item) in ranked {
        if let Some(cluster_key) = continuity_practice_cluster_key(item)
            && !seen_clusters.insert(cluster_key)
        {
            continue;
        }
        selected.push(item.clone());
        if selected.len() >= RECENT_UPDATE_LIMIT {
            break;
        }
    }

    selected
}

pub(crate) fn build_active_thread_view(
    items: &[ContinuityItemRecord],
    now: DateTime<Utc>,
) -> Vec<ContinuityItemRecord> {
    let active_cutoff = now - Duration::days(RECENT_LEARNING_WINDOW_DAYS);
    let mut ranked = items
        .iter()
        .filter_map(|item| {
            active_thread_candidate_score(item, active_cutoff, now).map(|score| (score, item))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.1.updated_at.cmp(&left.1.updated_at))
            .then_with(|| right.1.title.cmp(&left.1.title))
    });

    let mut selected = Vec::new();
    let mut seen_clusters = BTreeSet::<String>::new();
    for (_, item) in ranked {
        if let Some(cluster_key) = continuity_practice_cluster_key(item)
            && !seen_clusters.insert(cluster_key)
        {
            continue;
        }
        selected.push(item.clone());
        if selected.len() >= ACTIVE_THREAD_LIMIT {
            break;
        }
    }

    selected
}

fn learning_candidates(items: &[ContinuityItemRecord]) -> Vec<ContinuityItemRecord> {
    let mut candidates = items
        .iter()
        .filter(|item| is_learning_candidate(item))
        .cloned()
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .updated_at
            .cmp(&left.updated_at)
            .then_with(|| right.created_at.cmp(&left.created_at))
            .then_with(|| {
                right
                    .importance
                    .partial_cmp(&left.importance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| right.title.cmp(&left.title))
    });
    candidates
}

fn is_learning_candidate(item: &ContinuityItemRecord) -> bool {
    if item.status == ContinuityStatus::Rejected {
        return false;
    }
    matches!(
        item.kind,
        ContinuityKind::Lesson
            | ContinuityKind::Outcome
            | ContinuityKind::Decision
            | ContinuityKind::Incident
            | ContinuityKind::OperationalScar
    )
}

pub(crate) fn objective_requests_history_context(objective: &str) -> bool {
    let normalized = normalize_objective_text(objective);
    [
        "history",
        "timeline",
        "lineage",
        "evolution",
        "over time",
        "weekly review",
        "how we learned",
        "what we learned over",
        "historico",
        "histórico",
        "linha",
        "evolucao",
        "evolução",
        "ao longo",
        "semana inteira",
        "linha de aprendizado",
        "full learning line",
        "why did it change",
        "how did it change",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

pub(crate) fn objective_requests_current_state_context(objective: &str) -> bool {
    if objective_requests_history_context(objective) {
        return false;
    }
    let normalized = normalize_objective_text(objective);
    [
        "current state",
        "current live state",
        "live state",
        "latest state",
        "active state",
        "current practice",
        "latest practice",
        "latest guidance",
        "active guidance",
        "right now",
        "as of now",
        "where are we now",
        "what is current",
        "what's current",
        "estado atual",
        "estado corrente",
        "pratica atual",
        "prática atual",
        "guia atual",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

pub(crate) fn objective_requests_operational_state_context(objective: &str) -> bool {
    if objective_requests_history_context(objective) {
        return false;
    }
    if objective_requests_current_state_context(objective) {
        return true;
    }
    let normalized = normalize_objective_text(objective);
    [
        "what should we do",
        "what do we do next",
        "what should i do",
        "what should the agent do",
        "what's the plan",
        "what is the plan",
        "what's our plan",
        "what is our plan",
        "next step",
        "next steps",
        "next move",
        "status update",
        "where do we stand",
        "where are we",
        "what's blocking",
        "what is blocking",
        "what are the blockers",
        "what is the blocker",
        "what should we focus on",
        "what is the focus",
        "what should we prioritize",
        "what are the priorities",
        "how should we proceed",
        "what matters now",
        "o que fazemos agora",
        "qual o proximo passo",
        "qual o próximo passo",
        "como seguimos",
        "onde estamos",
        "o que mudou",
        "qual a prioridade",
        "quais sao os bloqueios",
        "quais são os bloqueios",
        "qual e o plano",
        "qual é o plano",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

pub(crate) fn objective_requests_next_step_context(objective: &str) -> bool {
    if objective_requests_history_context(objective) {
        return false;
    }
    let normalized = normalize_objective_text(objective);
    [
        "what should we do next",
        "what do we do next",
        "what should i do next",
        "what should the agent do next",
        "next step",
        "next steps",
        "next move",
        "what now",
        "where next",
        "qual o proximo passo",
        "qual o próximo passo",
        "proximo passo",
        "próximo passo",
        "o que fazemos agora",
        "como seguimos agora",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

pub(crate) fn objective_requests_recent_update_context(objective: &str) -> bool {
    if objective_requests_history_context(objective) {
        return false;
    }
    let normalized = normalize_objective_text(objective);
    [
        "what changed",
        "what changed recently",
        "recent update",
        "recent updates",
        "latest update",
        "latest updates",
        "latest decision",
        "latest decisions",
        "recent decision",
        "recent decisions",
        "latest lesson",
        "latest lessons",
        "recent lesson",
        "recent lessons",
        "what's new",
        "what is new",
        "new since",
        "progress update",
        "continue from here",
        "pick up from here",
        "where did we land",
        "what did we just ship",
        "what merged",
        "what was merged",
        "what just merged",
        "merged recently",
        "recent merge",
        "latest merge",
        "o que mudou",
        "mudou recentemente",
        "atualizacao recente",
        "atualização recente",
        "ultima decisao",
        "última decisão",
        "ultimo aprendizado",
        "último aprendizado",
        "continue daqui",
        "segue daqui",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

pub(crate) fn objective_requests_active_thread_context(objective: &str) -> bool {
    if objective_requests_history_context(objective) {
        return false;
    }
    if objective_requests_current_state_context(objective)
        || objective_requests_operational_state_context(objective)
        || objective_requests_next_step_context(objective)
        || objective_requests_recent_update_context(objective)
    {
        return false;
    }
    let normalized = normalize_objective_text(objective);
    let tokens = objective_content_tokens(&normalized);
    let phrases = [
        "continue",
        "resume",
        "pick up",
        "follow up",
        "verify this",
        "verify the merged slice",
        "sanity check",
        "where were we",
        "where did we stop",
        "what next here",
        "what is next here",
        "what should land next",
        "next cut",
        "next slice",
        "what is the next cut",
        "toca a proxima",
        "toca a próxima",
        "segue ai",
        "segue aí",
    ];
    if phrases.iter().any(|needle| normalized.contains(needle)) {
        return true;
    }
    let generic_control_tokens = [
        "continue", "resume", "verify", "check", "follow", "proceed", "advance", "merge", "merged",
        "ship", "land", "push", "next", "slice", "cut",
    ];
    let has_control_token = tokens
        .iter()
        .any(|token| generic_control_tokens.contains(&token.as_str()));
    has_control_token && tokens.len() <= 7
}

fn summarize_recent_learning(
    items: &[ContinuityItemRecord],
    used_recent_window: bool,
    now: DateTime<Utc>,
) -> String {
    if items.is_empty() {
        return "No recent learnings are recorded yet.".to_string();
    }
    let focus = items
        .iter()
        .take(LEARNING_SUMMARY_LIMIT)
        .map(|item| {
            format!(
                "{} [{}; {}]",
                item.title,
                item.kind.as_str(),
                relative_time_label(item.updated_at, now)
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    let window_label = if used_recent_window {
        format!("the last {} days", RECENT_LEARNING_WINDOW_DAYS)
    } else {
        "the latest recorded history".to_string()
    };
    format!(
        "Recent learning digest from {}: {}. {} learning signal(s) are surfaced here.",
        window_label,
        focus,
        items.len()
    )
}

fn summarize_learning_line(items: &[ContinuityItemRecord], now: DateTime<Utc>) -> String {
    if items.is_empty() {
        return "No learning line is recorded yet.".to_string();
    }
    if items.len() == 1 {
        let item = &items[0];
        return format!(
            "Learning line has a single recorded pivot: {} [{}; {}].",
            item.title,
            item.kind.as_str(),
            relative_time_label(item.updated_at, now)
        );
    }

    let first = items.first().expect("non-empty learning line");
    let last = items.last().expect("non-empty learning line");
    let span_days = (last.updated_at - first.updated_at).num_days().max(0);
    let mut pivots = vec![first.title.clone()];
    if items.len() > 2 {
        pivots.push(items[items.len() / 2].title.clone());
    }
    pivots.push(last.title.clone());
    pivots.dedup();

    format!(
        "Learning line spans {} signal(s) across {} day(s): {}.",
        items.len(),
        span_days,
        pivots.join(" -> ")
    )
}

fn summarize_current_practice(
    items: &[ContinuityItemRecord],
    evidence: &[PracticeEvidenceRecord],
    now: DateTime<Utc>,
) -> String {
    if items.is_empty() {
        return "No current practice is established yet.".to_string();
    }
    let evidence_by_practice = evidence
        .iter()
        .map(|bundle| (bundle.practice_id.as_str(), bundle))
        .collect::<BTreeMap<_, _>>();
    let focus = items
        .iter()
        .take(LEARNING_SUMMARY_LIMIT)
        .map(|item| {
            let evidence_suffix = evidence_by_practice
                .get(item.id.as_str())
                .map(|bundle| {
                    let evidence_titles = bundle
                        .evidence
                        .iter()
                        .take(CURRENT_PRACTICE_EVIDENCE_SUMMARY_LIMIT)
                        .map(|evidence_item| evidence_item.title.clone())
                        .collect::<Vec<_>>();
                    format!(
                        "; backed by {} signal(s): {}",
                        bundle.evidence_count,
                        evidence_titles.join(", ")
                    )
                })
                .unwrap_or_else(|| "; no live support signals".to_string());
            format!(
                "{} [{}; {}; {}{}]",
                item.title,
                item.kind.as_str(),
                practice_state_label(item.practice_state),
                relative_time_label(item.updated_at, now),
                evidence_suffix
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "Current practice favors {} active guidance signal(s): {}.",
        items.len(),
        focus
    )
}

fn relative_time_label(ts: DateTime<Utc>, now: DateTime<Utc>) -> String {
    let delta = now.signed_duration_since(ts);
    if delta.num_days() >= 1 {
        format!("{}d ago", delta.num_days())
    } else if delta.num_hours() >= 1 {
        format!("{}h ago", delta.num_hours())
    } else if delta.num_minutes() >= 1 {
        format!("{}m ago", delta.num_minutes())
    } else {
        "just now".to_string()
    }
}

fn is_current_practice_candidate(item: &ContinuityItemRecord) -> bool {
    if !is_guidance_like(item.kind) {
        return false;
    }
    matches!(
        item.practice_state,
        Some(PracticeLifecycleState::Current | PracticeLifecycleState::Aging)
    )
}

fn derive_practice_state(
    item: &ContinuityItemRecord,
    now: DateTime<Utc>,
) -> Option<PracticeLifecycleState> {
    if !is_guidance_like(item.kind) {
        return None;
    }
    if matches!(
        item.status,
        ContinuityStatus::Superseded | ContinuityStatus::Rejected
    ) {
        return Some(PracticeLifecycleState::Retired);
    }

    let anchor = continuity_practice_anchor(item).unwrap_or(item.updated_at);
    let age_hours = (now - anchor).num_seconds().max(0) as f64 / 3600.0;
    let fresh_window_hours = (item.retention.half_life_hours * 0.18).clamp(12.0, 24.0 * 14.0);
    let stale_window_hours = (item.retention.half_life_hours * 0.55).clamp(48.0, 24.0 * 45.0);
    let retirement_window_hours = guidance_retirement_window_hours(stale_window_hours);

    if item.status == ContinuityStatus::Resolved {
        if age_hours <= fresh_window_hours * 0.75 {
            Some(PracticeLifecycleState::Aging)
        } else {
            Some(PracticeLifecycleState::Stale)
        }
    } else if item.status.is_open() && age_hours > retirement_window_hours {
        Some(PracticeLifecycleState::Retired)
    } else if age_hours <= fresh_window_hours {
        Some(PracticeLifecycleState::Current)
    } else if age_hours <= stale_window_hours {
        Some(PracticeLifecycleState::Aging)
    } else {
        Some(PracticeLifecycleState::Stale)
    }
}

fn continuity_practice_anchor(item: &ContinuityItemRecord) -> Option<DateTime<Utc>> {
    let plasticity = continuity_plasticity_state(&item.extra);
    let mut anchors = vec![item.updated_at];
    if let Some(plasticity) = plasticity {
        if let Some(value) = plasticity.last_strengthened_at {
            anchors.push(value);
        }
        if let Some(value) = plasticity.last_confirmed_at {
            anchors.push(value);
        }
        if let Some(value) = plasticity.last_reactivated_at {
            anchors.push(value);
        }
    }
    anchors.into_iter().max()
}

fn continuity_plasticity_state(extra: &serde_json::Value) -> Option<ContinuityPlasticityState> {
    extra
        .get("plasticity")
        .cloned()
        .and_then(|value| serde_json::from_value::<ContinuityPlasticityState>(value).ok())
}

fn current_practice_rank(
    item: &ContinuityItemRecord,
    state: PracticeLifecycleState,
    now: DateTime<Utc>,
    support_signal: f64,
) -> f64 {
    let recency_hours = (now - item.updated_at).num_seconds().max(0) as f64 / 3600.0;
    let state_score = match state {
        PracticeLifecycleState::Current => 0.32,
        PracticeLifecycleState::Aging => 0.12,
        PracticeLifecycleState::Stale => -0.18,
        PracticeLifecycleState::Retired => -0.32,
    };
    state_score
        + item.retention.effective_salience
        + item.importance * 0.24
        + item.confidence * 0.16
        + support_signal
        - (recency_hours / 72.0).min(0.22)
}

fn current_practice_support_index(
    items: &[ContinuityItemRecord],
    items_by_id: &BTreeMap<String, ContinuityItemRecord>,
) -> BTreeMap<String, Vec<(SupportRef, ContinuityItemRecord)>> {
    let mut out = BTreeMap::<String, Vec<(SupportRef, ContinuityItemRecord)>>::new();
    for item in items {
        for support in &item.supports {
            if support.support_type != "continuity" {
                continue;
            }
            let Some(target) = items_by_id.get(&support.support_id) else {
                continue;
            };
            out.entry(target.id.clone())
                .or_default()
                .push((support.clone(), item.clone()));
        }
    }
    out
}

fn current_practice_support_signal(
    item: &ContinuityItemRecord,
    items_by_id: &BTreeMap<String, ContinuityItemRecord>,
    support_index: &BTreeMap<String, Vec<(SupportRef, ContinuityItemRecord)>>,
    now: DateTime<Utc>,
) -> f64 {
    current_practice_evidence_candidates(item, items_by_id, support_index, now)
        .iter()
        .enumerate()
        .map(|(index, (_, _, score))| {
            let damp = match index {
                0 => 1.0,
                1 => 0.72,
                _ => 0.48,
            };
            score * damp
        })
        .sum::<f64>()
        .clamp(0.0, 0.34)
}

fn current_practice_evidence_items(
    item: &ContinuityItemRecord,
    items_by_id: &BTreeMap<String, ContinuityItemRecord>,
    support_index: &BTreeMap<String, Vec<(SupportRef, ContinuityItemRecord)>>,
    now: DateTime<Utc>,
) -> Vec<ContinuityItemRecord> {
    current_practice_evidence_candidates(item, items_by_id, support_index, now)
        .into_iter()
        .map(|(_, evidence, _)| evidence)
        .collect()
}

fn current_practice_evidence_candidates(
    item: &ContinuityItemRecord,
    items_by_id: &BTreeMap<String, ContinuityItemRecord>,
    support_index: &BTreeMap<String, Vec<(SupportRef, ContinuityItemRecord)>>,
    now: DateTime<Utc>,
) -> Vec<(SupportRef, ContinuityItemRecord, f64)> {
    let mut ranked = Vec::<(SupportRef, ContinuityItemRecord, f64)>::new();
    let mut seen = BTreeSet::<String>::new();

    for (support_ref, evidence_item) in support_index.get(&item.id).into_iter().flatten().cloned() {
        if seen.insert(evidence_item.id.clone()) {
            let score = current_practice_evidence_rank(&support_ref, &evidence_item, now);
            ranked.push((support_ref, evidence_item, score));
        }
    }

    for support_ref in item
        .supports
        .iter()
        .filter(|support| support.support_type == "continuity")
        .cloned()
    {
        let Some(evidence_item) = items_by_id.get(&support_ref.support_id).cloned() else {
            continue;
        };
        if seen.insert(evidence_item.id.clone()) {
            let score = current_practice_evidence_rank(&support_ref, &evidence_item, now);
            ranked.push((support_ref, evidence_item, score));
        }
    }

    ranked.retain(|(_, evidence_item, score)| {
        *score > 0.0 && evidence_item.status != ContinuityStatus::Rejected
    });
    ranked.sort_by(|left, right| {
        right
            .2
            .partial_cmp(&left.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.1.updated_at.cmp(&left.1.updated_at))
            .then_with(|| right.1.title.cmp(&left.1.title))
    });
    ranked.truncate(CURRENT_PRACTICE_EVIDENCE_LIMIT);
    ranked
}

fn current_practice_evidence_rank(
    support_ref: &SupportRef,
    evidence_item: &ContinuityItemRecord,
    now: DateTime<Utc>,
) -> f64 {
    let kind_score = match evidence_item.kind {
        ContinuityKind::Outcome => 0.13,
        ContinuityKind::Lesson => 0.11,
        ContinuityKind::OperationalScar => 0.1,
        ContinuityKind::Incident => 0.08,
        ContinuityKind::Decision | ContinuityKind::Constraint => 0.06,
        ContinuityKind::Fact | ContinuityKind::Derivation => 0.04,
        _ => 0.02,
    };
    let reason_score = match support_ref.reason.as_deref().map(str::trim) {
        Some("outcome_confirmed") => 0.12,
        Some("belief_update_current") => 0.1,
        Some("outcome_used") => 0.08,
        Some("same runtime path") => 0.06,
        Some(reason) if !reason.is_empty() => 0.04,
        _ => 0.0,
    };
    let status_score = match evidence_item.status {
        ContinuityStatus::Resolved => 0.04,
        ContinuityStatus::Open | ContinuityStatus::Active => 0.02,
        ContinuityStatus::Superseded => -0.04,
        ContinuityStatus::Rejected => -0.12,
    };
    let age_hours = (now - evidence_item.updated_at).num_seconds().max(0) as f64 / 3600.0;
    let recency_score = if age_hours <= 24.0 {
        0.08
    } else if age_hours <= 24.0 * 3.0 {
        0.05
    } else if age_hours <= 24.0 * 7.0 {
        0.03
    } else {
        0.0
    };

    (support_ref.weight.clamp(0.0, 1.5) * 0.08
        + kind_score
        + reason_score
        + status_score
        + recency_score
        + evidence_item.retention.effective_salience * 0.04)
        .clamp(-0.12, 0.22)
}

fn continuity_practice_cluster_key(item: &ContinuityItemRecord) -> Option<String> {
    item_metadata_str(&item.extra, "practice_key")
        .or_else(|| item_metadata_str(&item.extra, "belief_key"))
}

fn item_metadata_str(extra: &serde_json::Value, key: &str) -> Option<String> {
    extra
        .get(key)
        .and_then(|value| value.as_str())
        .or_else(|| {
            extra
                .get("user")
                .and_then(|user| user.get(key))
                .and_then(|value| value.as_str())
        })
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn is_guidance_like(kind: ContinuityKind) -> bool {
    matches!(
        kind,
        ContinuityKind::Decision
            | ContinuityKind::Constraint
            | ContinuityKind::Lesson
            | ContinuityKind::Outcome
    )
}

fn is_operational_next_step(item: &ContinuityItemRecord) -> bool {
    item.kind == ContinuityKind::WorkingState
        && (item.title == "model-next-step" || item.extra["next_step"].as_bool() == Some(true))
}

fn operational_state_candidate_score(
    item: &ContinuityItemRecord,
    now: DateTime<Utc>,
) -> Option<f64> {
    if !counts_as_open_thread(item, now) {
        return None;
    }

    let practice_state = if is_guidance_like(item.kind) {
        item.practice_state
            .or_else(|| derive_practice_state(item, now))
    } else {
        None
    };
    let kind_score = match item.kind {
        ContinuityKind::WorkingState if is_operational_next_step(item) => 0.42,
        ContinuityKind::Constraint => 0.34,
        ContinuityKind::Decision => 0.3,
        ContinuityKind::Outcome => 0.22,
        ContinuityKind::Lesson => 0.18,
        ContinuityKind::Incident => 0.24,
        ContinuityKind::Signal => 0.22,
        ContinuityKind::WorkClaim => 0.14,
        ContinuityKind::WorkingState => 0.12,
        _ => return None,
    };
    let practice_score = match practice_state {
        Some(PracticeLifecycleState::Current) => 0.18,
        Some(PracticeLifecycleState::Aging) => 0.08,
        Some(PracticeLifecycleState::Stale) => -0.12,
        Some(PracticeLifecycleState::Retired) => -0.22,
        None => 0.0,
    };
    let status_score = match item.status {
        ContinuityStatus::Active => 0.1,
        ContinuityStatus::Open => 0.05,
        _ => 0.0,
    };
    let coordination_score = coordination_signal(item)
        .map(|signal| match signal.severity {
            value if value == CoordinationSeverity::Block.as_str() => 0.18,
            value if value == CoordinationSeverity::Warn.as_str() => 0.12,
            value if value == CoordinationSeverity::Info.as_str() => 0.04,
            _ => 0.0,
        })
        .unwrap_or_default();
    let next_step_score = if is_operational_next_step(item) {
        0.26
    } else {
        0.0
    };
    let recency_hours = (now - item.updated_at).num_seconds().max(0) as f64 / 3600.0;
    let recency_penalty = (recency_hours / 96.0).min(0.24);

    Some(
        kind_score
            + practice_score
            + status_score
            + coordination_score
            + next_step_score
            + item.retention.effective_salience * 0.85
            + item.importance * 0.18
            + item.confidence * 0.12
            - recency_penalty,
    )
}

fn recent_update_candidate_score(
    item: &ContinuityItemRecord,
    recent_cutoff: DateTime<Utc>,
    now: DateTime<Utc>,
) -> Option<f64> {
    if item.status == ContinuityStatus::Rejected {
        return None;
    }

    let kind_score = match item.kind {
        ContinuityKind::Decision => 0.34,
        ContinuityKind::Lesson => 0.3,
        ContinuityKind::Outcome => 0.28,
        ContinuityKind::Constraint => 0.24,
        ContinuityKind::Incident => 0.2,
        ContinuityKind::OperationalScar => 0.18,
        ContinuityKind::Fact if !item.supports.is_empty() => 0.14,
        _ => return None,
    };
    let recency_bonus = if item.updated_at >= recent_cutoff {
        0.24
    } else {
        0.0
    };
    let status_score = match item.status {
        ContinuityStatus::Active => 0.14,
        ContinuityStatus::Open => 0.08,
        ContinuityStatus::Resolved => 0.04,
        ContinuityStatus::Superseded => -0.04,
        ContinuityStatus::Rejected => return None,
    };
    let practice_score = if is_guidance_like(item.kind) {
        match item
            .practice_state
            .or_else(|| derive_practice_state(item, now))
        {
            Some(PracticeLifecycleState::Current) => 0.16,
            Some(PracticeLifecycleState::Aging) => 0.08,
            Some(PracticeLifecycleState::Stale) => -0.12,
            Some(PracticeLifecycleState::Retired) => -0.2,
            None => 0.0,
        }
    } else {
        0.0
    };
    let support_signal = (item.supports.len().min(3) as f64) * 0.06;
    let age_hours = (now - item.updated_at).num_seconds().max(0) as f64 / 3600.0;
    let age_penalty = (age_hours / 120.0).min(0.22);

    Some(
        kind_score
            + recency_bonus
            + status_score
            + practice_score
            + support_signal
            + item.retention.effective_salience * 0.7
            + item.importance * 0.14
            + item.confidence * 0.1
            - age_penalty,
    )
}

fn active_thread_candidate_score(
    item: &ContinuityItemRecord,
    active_cutoff: DateTime<Utc>,
    now: DateTime<Utc>,
) -> Option<f64> {
    if item.status == ContinuityStatus::Rejected {
        return None;
    }

    let kind_score = match item.kind {
        ContinuityKind::WorkingState if is_operational_next_step(item) => 0.34,
        ContinuityKind::WorkingState => 0.24,
        ContinuityKind::Decision => 0.32,
        ContinuityKind::Lesson => 0.28,
        ContinuityKind::Outcome => 0.24,
        ContinuityKind::Constraint => 0.18,
        ContinuityKind::Incident => 0.14,
        ContinuityKind::OperationalScar => 0.12,
        ContinuityKind::Fact if !item.supports.is_empty() => 0.1,
        _ => return None,
    };
    let status_score = match item.status {
        ContinuityStatus::Active => 0.18,
        ContinuityStatus::Open => 0.1,
        ContinuityStatus::Resolved => 0.02,
        ContinuityStatus::Superseded => -0.08,
        ContinuityStatus::Rejected => return None,
    };
    let practice_score = if is_guidance_like(item.kind) {
        match item
            .practice_state
            .or_else(|| derive_practice_state(item, now))
        {
            Some(PracticeLifecycleState::Current) => 0.2,
            Some(PracticeLifecycleState::Aging) => 0.1,
            Some(PracticeLifecycleState::Stale) => return None,
            Some(PracticeLifecycleState::Retired) => return None,
            None => 0.0,
        }
    } else {
        0.0
    };
    let support_signal = (item.supports.len().min(3) as f64) * 0.06;
    let freshness_bonus = if item.updated_at >= active_cutoff {
        0.24
    } else {
        0.08
    };
    let age_hours = (now - item.updated_at).num_seconds().max(0) as f64 / 3600.0;
    if age_hours > (RECENT_LEARNING_WINDOW_DAYS as f64 * 24.0 * 3.0)
        && item.kind != ContinuityKind::Constraint
    {
        return None;
    }
    let age_penalty = (age_hours / 168.0).min(0.28);

    Some(
        kind_score
            + status_score
            + practice_score
            + support_signal
            + freshness_bonus
            + item.retention.effective_salience * 0.72
            + item.importance * 0.15
            + item.confidence * 0.1
            - age_penalty,
    )
}

fn normalize_objective_text(objective: &str) -> String {
    let mut normalized = String::with_capacity(objective.len());
    let mut last_was_space = true;
    for ch in objective.chars().flat_map(|ch| ch.to_lowercase()) {
        let mapped = if ch.is_alphanumeric() { ch } else { ' ' };
        if mapped == ' ' {
            if !last_was_space {
                normalized.push(' ');
                last_was_space = true;
            }
        } else {
            normalized.push(mapped);
            last_was_space = false;
        }
    }
    normalized.trim().to_string()
}

fn objective_content_tokens(normalized: &str) -> Vec<String> {
    const STOPWORDS: &[&str] = &[
        "a", "an", "and", "as", "at", "by", "do", "for", "from", "here", "i", "is", "it", "just",
        "me", "now", "of", "on", "our", "the", "this", "to", "up", "we", "what", "where", "with",
        "ai", "o", "os", "a", "as", "da", "de", "do", "dos", "das", "e", "em", "na", "no", "nos",
        "nas", "um", "uma", "para", "por", "que", "isso", "isto", "aqui", "agora", "segue", "toca",
    ];
    normalized
        .split_whitespace()
        .filter(|token| !STOPWORDS.contains(token))
        .map(ToString::to_string)
        .collect()
}

fn guidance_retirement_window_hours(stale_window_hours: f64) -> f64 {
    (stale_window_hours * 2.2).clamp(24.0 * 5.0, 24.0 * 120.0)
}

fn practice_state_sort_rank(state: Option<PracticeLifecycleState>) -> usize {
    match state.unwrap_or(PracticeLifecycleState::Stale) {
        PracticeLifecycleState::Current => 3,
        PracticeLifecycleState::Aging => 2,
        PracticeLifecycleState::Stale => 1,
        PracticeLifecycleState::Retired => 0,
    }
}

fn practice_state_label(state: Option<PracticeLifecycleState>) -> &'static str {
    match state.unwrap_or(PracticeLifecycleState::Stale) {
        PracticeLifecycleState::Current => "current",
        PracticeLifecycleState::Aging => "aging",
        PracticeLifecycleState::Stale => "stale",
        PracticeLifecycleState::Retired => "retired",
    }
}

pub(crate) fn default_work_claim_lease_seconds() -> u64 {
    180
}

pub(crate) fn normalize_work_claim_resources(resources: &[String]) -> Vec<String> {
    let mut normalized = resources
        .iter()
        .map(|resource| resource.trim())
        .filter(|resource| !resource.is_empty())
        .map(|resource| resource.to_ascii_lowercase())
        .collect::<Vec<_>>();
    normalized.sort();
    normalized.dedup();
    normalized
}

pub(crate) fn work_claim_key(
    context_id: &str,
    scope: Scope,
    agent_id: &str,
    title: &str,
    resources: &[String],
) -> String {
    let normalized_title = title
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    let subject = if resources.is_empty() {
        normalized_title
    } else {
        resources.join("|")
    };
    format!("{context_id}:{scope}:{agent_id}:{subject}")
}

pub(crate) fn merge_work_claim_extra(
    extra: serde_json::Value,
    coordination: &WorkClaimCoordination,
) -> serde_json::Value {
    let coordination_value =
        serde_json::to_value(coordination).unwrap_or_else(|_| serde_json::json!({}));
    match extra {
        serde_json::Value::Object(mut map) => {
            map.insert("coordination".to_string(), coordination_value);
            serde_json::Value::Object(map)
        }
        serde_json::Value::Null => serde_json::json!({ "coordination": coordination_value }),
        other => serde_json::json!({
            "coordination": coordination_value,
            "payload": other,
        }),
    }
}

pub(crate) fn work_claim_coordination_from_extra(
    extra: &serde_json::Value,
) -> Option<WorkClaimCoordination> {
    extra
        .get("user")
        .and_then(|value| value.get("coordination"))
        .or_else(|| extra.get("coordination"))
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
}

pub(crate) fn work_claim_coordination(
    item: &ContinuityItemRecord,
) -> Option<WorkClaimCoordination> {
    if item.kind != ContinuityKind::WorkClaim {
        return None;
    }
    work_claim_coordination_from_extra(&item.extra)
}

pub(crate) fn work_claim_is_live(item: &ContinuityItemRecord, now: DateTime<Utc>) -> bool {
    if item.kind != ContinuityKind::WorkClaim || !item.status.is_open() {
        return false;
    }
    work_claim_coordination(item)
        .and_then(|coordination| coordination.lease_expires_at)
        .map(|deadline| deadline > now)
        .unwrap_or(true)
}

pub(crate) fn counts_as_open_thread(item: &ContinuityItemRecord, now: DateTime<Utc>) -> bool {
    if !item.status.is_open() {
        return false;
    }
    if item.kind == ContinuityKind::WorkClaim {
        return work_claim_is_live(item, now);
    }
    if is_guidance_like(item.kind)
        && matches!(
            derive_practice_state(item, now),
            Some(PracticeLifecycleState::Stale | PracticeLifecycleState::Retired)
        )
    {
        return false;
    }
    true
}

pub(crate) fn work_claims_conflict(
    left: &ContinuityItemRecord,
    right: &ContinuityItemRecord,
    now: DateTime<Utc>,
) -> bool {
    if left.id == right.id || !work_claim_is_live(left, now) || !work_claim_is_live(right, now) {
        return false;
    }
    let Some(left_coordination) = work_claim_coordination(left) else {
        return false;
    };
    let Some(right_coordination) = work_claim_coordination(right) else {
        return false;
    };
    if !(left_coordination.exclusive || right_coordination.exclusive) {
        return false;
    }
    left_coordination.resources.iter().any(|resource| {
        right_coordination
            .resources
            .iter()
            .any(|other| other == resource)
    })
}

pub(crate) fn default_coordination_severity(lane: CoordinationLane) -> CoordinationSeverity {
    match lane {
        CoordinationLane::Review | CoordinationLane::Coach => CoordinationSeverity::Info,
        CoordinationLane::Warning | CoordinationLane::Anxiety => CoordinationSeverity::Warn,
        CoordinationLane::Backoff => CoordinationSeverity::Block,
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CoordinationSignalExtraInput {
    pub lane: CoordinationLane,
    pub severity: CoordinationSeverity,
    pub target_agent_id: Option<String>,
    pub target_projected_lane: Option<CoordinationProjectedLane>,
    pub claim_id: Option<String>,
    pub resource: Option<String>,
    pub projection_ids: Vec<String>,
    pub projected_lanes: Vec<CoordinationProjectedLane>,
}

pub(crate) fn merge_coordination_signal_extra(
    extra: serde_json::Value,
    input: CoordinationSignalExtraInput,
) -> serde_json::Value {
    let coordination_value = serde_json::json!({
        "lane": input.lane.as_str(),
        "severity": input.severity.as_str(),
        "target_agent_id": input.target_agent_id,
        "target_projected_lane": input.target_projected_lane,
        "claim_id": input.claim_id,
        "resource": input.resource,
        "projection_ids": input.projection_ids,
        "projected_lanes": input.projected_lanes,
    });
    match extra {
        serde_json::Value::Object(mut map) => {
            map.insert("coordination_signal".to_string(), coordination_value);
            serde_json::Value::Object(map)
        }
        serde_json::Value::Null => {
            serde_json::json!({ "coordination_signal": coordination_value })
        }
        other => serde_json::json!({
            "coordination_signal": coordination_value,
            "payload": other,
        }),
    }
}

pub(crate) fn coordination_signal_from_extra(
    extra: &serde_json::Value,
) -> Option<CoordinationSignalRecord> {
    extra
        .get("user")
        .and_then(|value| value.get("coordination_signal"))
        .or_else(|| extra.get("coordination_signal"))
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
}

pub(crate) fn coordination_signal(item: &ContinuityItemRecord) -> Option<CoordinationSignalRecord> {
    if item.kind != ContinuityKind::Signal || !item.status.is_open() {
        return None;
    }
    coordination_signal_from_extra(&item.extra)
}

pub(crate) fn organism_state(
    items: &[ContinuityItemRecord],
    now: DateTime<Utc>,
    agent_badges: &[AgentBadgeRecord],
    lane_projections: &[LaneProjectionRecord],
) -> serde_json::Value {
    let mut retention_classes = BTreeMap::<String, usize>::new();
    let mut kinds = BTreeMap::<String, usize>::new();
    let mut open_pressure = Vec::new();
    let mut treated_count = 0usize;
    let mut active_claims = Vec::new();
    let mut active_claim_records = Vec::new();
    let mut expired_claims = 0usize;
    let mut coordination_pressure = Vec::new();
    let mut anxiety_pressure = Vec::new();

    for item in items {
        *retention_classes
            .entry(item.retention.class.clone())
            .or_insert(0) += 1;
        *kinds.entry(item.kind.as_str().to_string()).or_insert(0) += 1;
        if item.retention.class.starts_with("treated_") {
            treated_count += 1;
        }
        if item.kind == ContinuityKind::WorkClaim {
            if work_claim_is_live(item, now) {
                if let Some(coordination) = work_claim_coordination(item) {
                    active_claims.push(serde_json::json!({
                        "id": item.id,
                        "agent_id": item.author_agent_id,
                        "title": item.title,
                        "resources": coordination.resources,
                        "exclusive": coordination.exclusive,
                        "lease_expires_at": coordination.lease_expires_at,
                    }));
                    active_claim_records.push(item);
                }
            } else if item.status.is_open() {
                expired_claims += 1;
            }
        }
        if counts_as_open_thread(item, now) {
            let practice_state = if is_guidance_like(item.kind) {
                derive_practice_state(item, now).map(|state| practice_state_label(Some(state)))
            } else {
                None
            };
            open_pressure.push(serde_json::json!({
                "id": item.id,
                "kind": item.kind.as_str(),
                "status": item.status.as_str(),
                "title": item.title,
                "retention_class": item.retention.class,
                "effective_salience": item.retention.effective_salience,
                "practice_state": practice_state,
            }));
        }
        if let Some(signal) = coordination_signal(item) {
            let summary = serde_json::json!({
                "id": item.id,
                "lane": signal.lane,
                "severity": signal.severity,
                "title": item.title,
                "target_agent_id": signal.target_agent_id,
                "target_projected_lane": signal.target_projected_lane,
                "claim_id": signal.claim_id,
                "resource": signal.resource,
                "projection_ids": signal.projection_ids,
                "projected_lanes": signal.projected_lanes,
            });
            if summary["lane"].as_str() == Some(CoordinationLane::Anxiety.as_str()) {
                anxiety_pressure.push(summary.clone());
            }
            coordination_pressure.push(summary);
        }
    }
    open_pressure.truncate(8);
    active_claims.truncate(8);
    coordination_pressure.truncate(8);
    anxiety_pressure.truncate(8);

    let attachment_projection_by_id = attachment_projection_map(agent_badges, lane_projections);
    let mut seen_resources = BTreeMap::<String, ClaimConflictSummary>::new();
    for (index, left) in active_claim_records.iter().enumerate() {
        let Some(left_coordination) = work_claim_coordination(left) else {
            continue;
        };
        for right in active_claim_records.iter().skip(index + 1) {
            if !work_claims_conflict(left, right, now) {
                continue;
            }
            let Some(right_coordination) = work_claim_coordination(right) else {
                continue;
            };
            for resource in left_coordination.resources.iter().filter(|resource| {
                right_coordination
                    .resources
                    .iter()
                    .any(|other| other == *resource)
            }) {
                let conflict = seen_resources
                    .entry(resource.clone())
                    .or_insert_with(|| ClaimConflictSummary::new(resource.clone()));
                conflict.absorb_claim(
                    left,
                    projected_lane_for_claim(
                        left,
                        &left_coordination,
                        &attachment_projection_by_id,
                        lane_projections,
                    ),
                );
                conflict.absorb_claim(
                    right,
                    projected_lane_for_claim(
                        right,
                        &right_coordination,
                        &attachment_projection_by_id,
                        lane_projections,
                    ),
                );
            }
        }
    }
    let claim_conflicts = seen_resources.into_values().collect::<Vec<_>>();

    serde_json::json!({
        "continuity_items": items.len(),
        "treated_items": treated_count,
        "retention_classes": retention_classes,
        "kinds": kinds,
        "open_pressure": open_pressure,
        "active_claim_count": active_claim_records.len(),
        "expired_claim_count": expired_claims,
        "active_claims": active_claims,
        "claim_conflict_count": claim_conflicts.len(),
        "claim_conflicts": claim_conflicts,
        "coordination_signal_count": coordination_pressure.len(),
        "coordination_pressure": coordination_pressure,
        "anxiety_signal_count": anxiety_pressure.len(),
        "anxiety_pressure": anxiety_pressure,
    })
}

pub(crate) fn merge_dispatch_worker_lane_projections(
    lane_projections: &mut Vec<LaneProjectionRecord>,
    dispatch_state: &dispatch::DispatchOrganismSnapshot,
    namespace_fallback: &str,
) {
    if dispatch_state.workers.is_empty() {
        return;
    }
    let mut merged = lane_projections
        .drain(..)
        .map(|projection| (projection.projection_id.clone(), projection))
        .collect::<BTreeMap<_, _>>();
    for worker in &dispatch_state.workers {
        let dispatch_projection = dispatch_worker_lane_projection(worker, namespace_fallback);
        if let Some(existing) = merged.get_mut(&dispatch_projection.projection_id) {
            absorb_dispatch_lane_projection(existing, dispatch_projection);
        } else {
            merged.insert(
                dispatch_projection.projection_id.clone(),
                dispatch_projection,
            );
        }
    }
    let mut values = merged.into_values().collect::<Vec<_>>();
    values.sort_by(|left, right| {
        right
            .updated_at
            .cmp(&left.updated_at)
            .then_with(|| left.projection_kind.cmp(&right.projection_kind))
            .then_with(|| left.label.cmp(&right.label))
    });
    *lane_projections = values;
}

pub(crate) fn merge_dispatch_assignment_pressure(
    lane_projections: &mut Vec<LaneProjectionRecord>,
    dispatch_state: &dispatch::DispatchOrganismSnapshot,
    namespace_fallback: &str,
) {
    if dispatch_state.assignments.is_empty() {
        return;
    }
    let mut merged = lane_projections
        .drain(..)
        .map(|projection| (projection.projection_id.clone(), projection))
        .collect::<BTreeMap<_, _>>();
    for assignment in &dispatch_state.assignments {
        let projection = merged
            .entry(assignment.projected_lane.projection_id.clone())
            .or_insert_with(|| {
                dispatch_assignment_lane_projection(
                    assignment,
                    &assignment.projected_lane,
                    namespace_fallback,
                    true,
                )
            });
        absorb_dispatch_assignment_pressure(projection, assignment, true);
        if let Some(attached_lane) = assignment.attached_projected_lane.as_ref()
            && attached_lane.projection_id != assignment.projected_lane.projection_id
        {
            let projection = merged
                .entry(attached_lane.projection_id.clone())
                .or_insert_with(|| {
                    dispatch_assignment_lane_projection(
                        assignment,
                        attached_lane,
                        namespace_fallback,
                        false,
                    )
                });
            absorb_dispatch_assignment_pressure(projection, assignment, false);
        }
    }
    let mut values = merged.into_values().collect::<Vec<_>>();
    values.sort_by(|left, right| {
        right
            .updated_at
            .cmp(&left.updated_at)
            .then_with(|| left.projection_kind.cmp(&right.projection_kind))
            .then_with(|| left.label.cmp(&right.label))
    });
    *lane_projections = values;
}

fn dispatch_worker_lane_projection(
    worker: &dispatch::DispatchWorkerPresence,
    namespace_fallback: &str,
) -> LaneProjectionRecord {
    let headline = if worker.active_assignment_count > 0 {
        format!(
            "{} is {} with {} active dispatch assignment(s).",
            worker.display_name, worker.status, worker.active_assignment_count
        )
    } else {
        format!(
            "{} is {} via the dispatch spine.",
            worker.display_name, worker.status
        )
    };
    LaneProjectionRecord {
        projection_id: worker.projected_lane.projection_id.clone(),
        namespace: worker
            .namespace
            .clone()
            .unwrap_or_else(|| namespace_fallback.to_string()),
        projection_kind: worker.projected_lane.projection_kind.clone(),
        label: worker.projected_lane.label.clone(),
        resource: worker.projected_lane.resource.clone(),
        repo_root: worker.projected_lane.repo_root.clone(),
        branch: worker.projected_lane.branch.clone(),
        task_id: worker.projected_lane.task_id.clone(),
        connected_agents: 1,
        live_claims: 0,
        claim_conflicts: 0,
        coordination_signal_count: 0,
        blocking_signal_count: 0,
        review_signal_count: 0,
        dispatch_assignment_count: 0,
        dispatch_assignment_anxiety_max: 0.0,
        dispatch_assignment_explicit_cli_count: 0,
        dispatch_assignment_live_badge_opt_in_count: 0,
        coordination_lanes: Vec::new(),
        agent_ids: vec![worker.worker_id.clone()],
        display_names: vec![worker.display_name.clone()],
        focus: if worker.focus.trim().is_empty() {
            "dispatch ready".to_string()
        } else {
            worker.focus.clone()
        },
        headline,
        updated_at: worker.last_seen_at,
    }
}

fn absorb_dispatch_lane_projection(
    existing: &mut LaneProjectionRecord,
    dispatch_projection: LaneProjectionRecord,
) {
    if existing.resource.is_none() {
        existing.resource = dispatch_projection.resource.clone();
    }
    if existing.task_id.is_none() {
        existing.task_id = dispatch_projection.task_id.clone();
    }
    if existing.repo_root.is_none() {
        existing.repo_root = dispatch_projection.repo_root.clone();
    }
    if existing.branch.is_none() {
        existing.branch = dispatch_projection.branch.clone();
    }
    let agent_ids = existing
        .agent_ids
        .iter()
        .cloned()
        .chain(dispatch_projection.agent_ids)
        .collect::<BTreeSet<_>>();
    existing.agent_ids = agent_ids.iter().cloned().collect();
    existing.connected_agents = existing.agent_ids.len();
    let display_names = existing
        .display_names
        .iter()
        .cloned()
        .chain(dispatch_projection.display_names)
        .collect::<BTreeSet<_>>();
    existing.display_names = display_names.iter().cloned().collect();
    if dispatch_projection.updated_at >= existing.updated_at
        || existing.focus.is_empty()
        || existing.headline.is_empty()
    {
        existing.focus = dispatch_projection.focus;
        existing.headline = dispatch_projection.headline;
        existing.updated_at = dispatch_projection.updated_at;
    }
}

fn dispatch_assignment_lane_projection(
    assignment: &dispatch::DispatchAssignmentPresence,
    lane: &CoordinationProjectedLane,
    namespace_fallback: &str,
    carry_worker_identity: bool,
) -> LaneProjectionRecord {
    let connected_agents = usize::from(carry_worker_identity);
    let agent_ids = if carry_worker_identity {
        vec![assignment.worker_id.clone()]
    } else {
        Vec::new()
    };
    let display_names = if carry_worker_identity {
        vec![assignment.worker_id.clone()]
    } else {
        Vec::new()
    };
    LaneProjectionRecord {
        projection_id: lane.projection_id.clone(),
        namespace: assignment
            .namespace
            .clone()
            .if_empty_then(namespace_fallback),
        projection_kind: lane.projection_kind.clone(),
        label: lane.label.clone(),
        resource: lane.resource.clone(),
        repo_root: lane.repo_root.clone(),
        branch: lane.branch.clone(),
        task_id: lane.task_id.clone(),
        connected_agents,
        live_claims: 0,
        claim_conflicts: 0,
        coordination_signal_count: 0,
        blocking_signal_count: 0,
        review_signal_count: 0,
        dispatch_assignment_count: 0,
        dispatch_assignment_anxiety_max: 0.0,
        dispatch_assignment_explicit_cli_count: 0,
        dispatch_assignment_live_badge_opt_in_count: 0,
        coordination_lanes: Vec::new(),
        agent_ids,
        display_names,
        focus: assignment.objective.clone(),
        headline: format!(
            "{} active dispatch assignment(s); max anxiety {:.2}",
            1, assignment.pressure.anxiety
        ),
        updated_at: assignment.created_at,
    }
}

fn absorb_dispatch_assignment_pressure(
    projection: &mut LaneProjectionRecord,
    assignment: &dispatch::DispatchAssignmentPresence,
    carry_worker_identity: bool,
) {
    projection.dispatch_assignment_count += 1;
    projection.dispatch_assignment_anxiety_max = projection
        .dispatch_assignment_anxiety_max
        .max(assignment.pressure.anxiety);
    if !carry_worker_identity {
        match assignment.attached_projected_lane_source {
            Some(dispatch::DispatchAttachedLaneSource::ExplicitCli) => {
                projection.dispatch_assignment_explicit_cli_count += 1;
            }
            Some(dispatch::DispatchAttachedLaneSource::LiveBadgeOptIn) => {
                projection.dispatch_assignment_live_badge_opt_in_count += 1;
            }
            None => {}
        }
    }
    if carry_worker_identity {
        if !projection
            .agent_ids
            .iter()
            .any(|agent_id| agent_id == &assignment.worker_id)
        {
            projection.agent_ids.push(assignment.worker_id.clone());
            projection.connected_agents = projection.agent_ids.len();
        }
        if !projection
            .display_names
            .iter()
            .any(|display_name| display_name == &assignment.worker_id)
        {
            projection.display_names.push(assignment.worker_id.clone());
        }
    }
    if assignment.created_at >= projection.updated_at
        || projection.focus.is_empty()
        || projection.headline.is_empty()
    {
        projection.focus = assignment.objective.clone();
        projection.headline = format!(
            "{} active dispatch assignment(s); max anxiety {:.2}",
            projection.dispatch_assignment_count, projection.dispatch_assignment_anxiety_max
        );
        projection.updated_at = assignment.created_at;
    }
}

pub(super) trait IfEmptyThen {
    fn if_empty_then(self, fallback: &str) -> String;
}

impl IfEmptyThen for String {
    fn if_empty_then(self, fallback: &str) -> String {
        if self.trim().is_empty() {
            fallback.to_string()
        } else {
            self
        }
    }
}

fn attachment_projection_map(
    agent_badges: &[AgentBadgeRecord],
    lane_projections: &[LaneProjectionRecord],
) -> BTreeMap<String, (Option<String>, LaneProjectionRecord)> {
    let mut attachment_projection_by_id = BTreeMap::new();
    for badge in agent_badges.iter().filter(|badge| badge.connected) {
        let Some(projection) = projection_for_badge(badge, lane_projections) else {
            continue;
        };
        attachment_projection_by_id.insert(
            badge.attachment_id.clone(),
            (Some(badge.display_name.clone()), projection.clone()),
        );
    }
    attachment_projection_by_id
}

fn projection_for_badge<'a>(
    badge: &AgentBadgeRecord,
    lane_projections: &'a [LaneProjectionRecord],
) -> Option<&'a LaneProjectionRecord> {
    lane_projections
        .iter()
        .find(|projection| {
            projection.namespace == badge.namespace
                && projection.resource == badge.resource
                && projection.repo_root == badge.repo_root
                && projection.branch == badge.branch
                && projection.task_id == badge.task_id
        })
        .or_else(|| {
            lane_projections.iter().find(|projection| {
                projection.namespace == badge.namespace
                    && projection.repo_root == badge.repo_root
                    && projection.branch == badge.branch
                    && projection.task_id == badge.task_id
            })
        })
        .or_else(|| {
            lane_projections.iter().find(|projection| {
                projection.namespace == badge.namespace
                    && projection.resource == badge.resource
                    && projection.task_id == badge.task_id
            })
        })
}

fn projected_lane_for_claim(
    item: &ContinuityItemRecord,
    coordination: &WorkClaimCoordination,
    attachment_projection_by_id: &BTreeMap<String, (Option<String>, LaneProjectionRecord)>,
    lane_projections: &[LaneProjectionRecord],
) -> Option<ClaimConflictLaneSummary> {
    if let Some((display_name, projection)) = coordination
        .attachment_id
        .as_ref()
        .and_then(|attachment_id| attachment_projection_by_id.get(attachment_id))
    {
        return Some(claim_conflict_lane_summary(
            item,
            display_name.clone(),
            projection,
        ));
    }

    let resource = coordination
        .resources
        .iter()
        .find(|resource| !resource.trim().is_empty())?;
    let projection = lane_projections.iter().find(|projection| {
        projection.namespace == item.namespace
            && projection.resource.as_deref() == Some(resource.as_str())
            && projection.task_id.as_deref() == Some(item.task_id.as_str())
    })?;
    Some(claim_conflict_lane_summary(item, None, projection))
}

fn claim_conflict_lane_summary(
    item: &ContinuityItemRecord,
    display_name: Option<String>,
    projection: &LaneProjectionRecord,
) -> ClaimConflictLaneSummary {
    ClaimConflictLaneSummary {
        claim_id: item.id.clone(),
        agent_id: item.author_agent_id.clone(),
        title: item.title.clone(),
        display_name,
        projection_id: projection.projection_id.clone(),
        projection_kind: projection.projection_kind.clone(),
        label: projection.label.clone(),
        resource: projection.resource.clone(),
        repo_root: projection.repo_root.clone(),
        branch: projection.branch.clone(),
        task_id: projection.task_id.clone(),
    }
}

pub(crate) fn augment_dimensions(
    base: Vec<DimensionValue>,
    extra: Vec<DimensionValue>,
) -> Vec<DimensionValue> {
    let mut merged = base;
    for item in extra {
        if !merged
            .iter()
            .any(|existing| existing.key == item.key && existing.value == item.value)
        {
            merged.push(item);
        }
    }
    merged
}
