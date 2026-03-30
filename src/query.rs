use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::Result;
use chrono::Utc;
use uuid::Uuid;

use crate::embedding::cosine_similarity;
use crate::model::{
    CandidateRecord, ContextPack, ContextPackItem, ContextPackManifest, MemoryRecord, QueryInput,
    RejectedCandidate, ScoreBreakdown,
};
use crate::storage::Storage;

pub fn build_context_pack(storage: &Storage, query: QueryInput) -> Result<ContextPack> {
    let total_start = Instant::now();
    let continuity_start = Instant::now();
    let continuity = if let (Some(namespace), Some(task_id)) =
        (query.namespace.as_deref(), query.task_id.as_deref())
    {
        let objective = query.objective.as_deref().unwrap_or(&query.query_text);
        match storage.resolve_context(None, Some(namespace), Some(task_id)) {
            Ok(context) => {
                let recall = storage.recall_continuity(
                    &context.id,
                    objective,
                    false,
                    query.candidate_limit,
                )?;
                let memory_ids = recall
                    .items
                    .iter()
                    .map(|item| item.memory_id.clone())
                    .collect::<Vec<_>>();
                storage.memories_by_ids(&memory_ids)?
            }
            Err(_) => Vec::new(),
        }
    } else {
        Vec::new()
    };
    let continuity_ms = continuity_start.elapsed().as_millis();

    let lexical_start = Instant::now();
    let lexical = storage.search_lexical(&query.query_text, query.candidate_limit)?;
    let lexical_ms = lexical_start.elapsed().as_millis();

    let selector_start = Instant::now();
    let selector = if let Some(selector) = &query.selector {
        storage.selector_memories(selector, query.candidate_limit)?
    } else {
        Vec::new()
    };
    let selector_ms = selector_start.elapsed().as_millis();

    let view_start = Instant::now();
    let view = if let Some(view_id) = &query.view_id {
        storage.view_memories(view_id)?
    } else {
        Vec::new()
    };
    let view_ms = view_start.elapsed().as_millis();

    let entity_start = Instant::now();
    let entity = storage.entity_memories(&query.query_text, query.candidate_limit)?;
    let entity_ms = entity_start.elapsed().as_millis();

    let temporal_start = Instant::now();
    let temporal = storage.recent_memories(
        query.session_id.as_deref(),
        query.task_id.as_deref(),
        query.agent_id.as_deref(),
        query.candidate_limit,
    )?;
    let temporal_ms = temporal_start.elapsed().as_millis();

    let vector_start = Instant::now();
    let query_vector = storage.embed_query_vector(&query.query_text)?;
    let vector_candidates = storage.vector_memories()?;
    let mut vector_scored = vector_candidates
        .into_iter()
        .map(|(memory, vector)| (memory, cosine_similarity(&query_vector, &vector)))
        .collect::<Vec<_>>();
    vector_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    vector_scored.truncate(query.candidate_limit);
    let vector_ms = vector_start.elapsed().as_millis();

    let mut merged: HashMap<String, CandidateRecord> = HashMap::new();
    merge_ranked(&mut merged, continuity, "continuity", |breakdown, score| {
        breakdown.continuity = score
    });
    merge_ranked(&mut merged, selector, "selector", |breakdown, score| {
        breakdown.selector = score
    });
    merge_ranked(&mut merged, view, "view", |breakdown, score| {
        breakdown.view = score
    });
    merge_ranked(&mut merged, lexical, "lexical", |breakdown, score| {
        breakdown.lexical = score
    });
    merge_ranked(&mut merged, entity, "entity", |breakdown, score| {
        breakdown.entity = score
    });
    merge_ranked(&mut merged, temporal, "temporal", |breakdown, score| {
        breakdown.temporal = score
    });
    for (rank, (memory, similarity)) in vector_scored.into_iter().enumerate() {
        let score = similarity.max(0.0);
        if score <= 0.0 {
            continue;
        }
        let entry = merged
            .entry(memory.id.clone())
            .or_insert_with(|| CandidateRecord {
                memory,
                final_score: 0.0,
                breakdown: ScoreBreakdown::default(),
                why: Vec::new(),
                provenance: serde_json::Value::Null,
            });
        entry.breakdown.vector = entry.breakdown.vector.max(score);
        entry.why.push(format!("vector#{rank}"));
    }

    let lineage_start = Instant::now();
    let seed_ids = merged
        .keys()
        .cloned()
        .take(query.candidate_limit.min(8))
        .collect::<Vec<_>>();
    let lineage = storage.lineage_neighbors(&seed_ids, query.candidate_limit.min(12))?;
    let lineage_ms = lineage_start.elapsed().as_millis();
    merge_ranked(&mut merged, lineage, "lineage", |breakdown, score| {
        breakdown.lineage = score
    });

    let now = Utc::now();
    let mut candidates = merged
        .into_values()
        .map(|mut candidate| {
            candidate.breakdown.continuity_kind = continuity_kind_score(&candidate.memory);
            candidate.breakdown.continuity_status = continuity_status_score(&candidate.memory);
            candidate.breakdown.recency = recency_score(now, candidate.memory.ts);
            candidate.breakdown.salience = candidate.memory.importance.clamp(0.0, 1.0) * 0.3;
            candidate.breakdown.scope = scope_score(&query, &candidate.memory);
            let vector_weight = vector_score_weight(&candidate.memory);
            let source_role_score = continuity_source_role_score(&candidate.memory);
            candidate.final_score = candidate.breakdown.continuity * 1.2
                + candidate.breakdown.continuity_kind
                + candidate.breakdown.continuity_status
                + candidate.breakdown.lexical * 0.95
                + candidate.breakdown.selector * 0.95
                + candidate.breakdown.vector * vector_weight
                + candidate.breakdown.entity * 0.7
                + candidate.breakdown.temporal * 0.5
                + candidate.breakdown.recency * 0.25
                + candidate.breakdown.salience
                + candidate.breakdown.lineage * 0.35
                + candidate.breakdown.view * 0.4
                + candidate.breakdown.scope
                + source_role_score;
            if let Some(kind) = continuity_kind_label(&candidate.memory) {
                candidate.why.push(format!("continuity_kind:{kind}"));
            }
            if let Some(status) = continuity_status_label(&candidate.memory) {
                candidate.why.push(format!("continuity_status:{status}"));
            }
            if let Some(source_role) = continuity_source_role_label(&candidate.memory) {
                candidate.why.push(format!("source_role:{source_role}"));
            }
            candidate.why.sort();
            candidate.why.dedup();
            candidate
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut selected = Vec::new();
    let mut rejected = Vec::new();
    let mut seen_scope_keys = HashSet::new();
    let mut seen_sources = HashSet::new();
    let mut seen_belief_keys = HashMap::<String, (f64, Option<String>)>::new();
    let mut used_tokens = 0usize;

    for candidate in candidates {
        let belief_key = continuity_belief_key_label(&candidate.memory).map(ToString::to_string);
        let provenance = storage.provenance_for_memory(&candidate.memory)?;
        let item = ContextPackItem {
            memory_id: candidate.memory.id.clone(),
            layer: candidate.memory.layer,
            token_estimate: candidate.memory.token_estimate + 6,
            final_score: candidate.final_score,
            why: candidate.why.clone(),
            breakdown: candidate.breakdown.clone(),
            provenance,
            body: candidate.memory.body.clone(),
        };

        if candidate.final_score < 0.18 {
            rejected.push(RejectedCandidate {
                memory_id: candidate.memory.id,
                layer: candidate.memory.layer,
                token_estimate: item.token_estimate,
                final_score: candidate.final_score,
                reason: "score_below_cutoff".to_string(),
            });
            continue;
        }
        if seen_scope_keys.contains(&candidate.memory.scope_key) {
            rejected.push(RejectedCandidate {
                memory_id: candidate.memory.id,
                layer: candidate.memory.layer,
                token_estimate: item.token_estimate,
                final_score: candidate.final_score,
                reason: "duplicate_scope_key".to_string(),
            });
            continue;
        }
        if let Some(source_event_id) = &candidate.memory.source_event_id {
            if seen_sources.contains(source_event_id) {
                rejected.push(RejectedCandidate {
                    memory_id: candidate.memory.id,
                    layer: candidate.memory.layer,
                    token_estimate: item.token_estimate,
                    final_score: candidate.final_score,
                    reason: "duplicate_source_event".to_string(),
                });
                continue;
            }
        }
        if let Some(belief_key) = belief_key.as_deref() {
            if let Some((best_score, best_source_role)) = seen_belief_keys.get(belief_key) {
                let source_role = continuity_source_role_label(&candidate.memory);
                let assistant_shadowing_user = belief_key.starts_with("user.")
                    && best_source_role.as_deref() == Some("user")
                    && source_role == Some("assistant");
                if assistant_shadowing_user || *best_score >= candidate.final_score + 0.05 {
                    rejected.push(RejectedCandidate {
                        memory_id: candidate.memory.id,
                        layer: candidate.memory.layer,
                        token_estimate: item.token_estimate,
                        final_score: candidate.final_score,
                        reason: "belief_key_competitor".to_string(),
                    });
                    continue;
                }
            }
        }
        if used_tokens + item.token_estimate > query.budget_tokens {
            rejected.push(RejectedCandidate {
                memory_id: candidate.memory.id,
                layer: candidate.memory.layer,
                token_estimate: item.token_estimate,
                final_score: candidate.final_score,
                reason: "token_budget_exceeded".to_string(),
            });
            continue;
        }

        used_tokens += item.token_estimate;
        seen_scope_keys.insert(candidate.memory.scope_key.clone());
        if let Some(source_event_id) = candidate.memory.source_event_id.clone() {
            seen_sources.insert(source_event_id);
        }
        if let Some(belief_key) = belief_key {
            seen_belief_keys
                .entry(belief_key)
                .and_modify(|entry| {
                    if candidate.final_score > entry.0 {
                        *entry = (
                            candidate.final_score,
                            continuity_source_role_label(&candidate.memory)
                                .map(ToString::to_string),
                        );
                    }
                })
                .or_insert_with(|| {
                    (
                        candidate.final_score,
                        continuity_source_role_label(&candidate.memory).map(ToString::to_string),
                    )
                });
        }
        selected.push(item);
    }

    let id = Uuid::now_v7().to_string();
    let created_at = Utc::now();
    let manifest_path = storage
        .paths
        .debug_dir
        .join("context-packs")
        .join(format!("{id}.json"))
        .display()
        .to_string();
    let manifest = ContextPackManifest {
        id: id.clone(),
        created_at,
        query: query.clone(),
        used_tokens,
        selected: selected.clone(),
        rejected: rejected.clone(),
        timings_ms: serde_json::json!({
            "continuity": continuity_ms,
            "lexical": lexical_ms,
            "selector": selector_ms,
            "view": view_ms,
            "entity": entity_ms,
            "temporal": temporal_ms,
            "vector": vector_ms,
            "lineage": lineage_ms,
            "total": total_start.elapsed().as_millis(),
        }),
    };
    let pack = ContextPack {
        id: id.clone(),
        created_at,
        query,
        used_tokens,
        items: selected,
        manifest_path,
    };
    storage.persist_context_pack(&pack, &manifest, &rejected)?;
    Ok(pack)
}

fn merge_ranked(
    merged: &mut HashMap<String, CandidateRecord>,
    records: Vec<MemoryRecord>,
    why_prefix: &str,
    apply: impl Fn(&mut ScoreBreakdown, f64),
) {
    let len = records.len().max(1) as f64;
    for (rank, memory) in records.into_iter().enumerate() {
        let score = 1.0 - (rank as f64 / len);
        let entry = merged
            .entry(memory.id.clone())
            .or_insert_with(|| CandidateRecord {
                memory,
                final_score: 0.0,
                breakdown: ScoreBreakdown::default(),
                why: Vec::new(),
                provenance: serde_json::Value::Null,
            });
        apply(&mut entry.breakdown, score);
        entry.why.push(format!("{why_prefix}#{rank}"));
    }
}

fn recency_score(now: chrono::DateTime<Utc>, ts: chrono::DateTime<Utc>) -> f64 {
    let age_hours = (now - ts).num_seconds().max(0) as f64 / 3600.0;
    1.0 / (1.0 + age_hours / 6.0)
}

fn scope_score(query: &QueryInput, memory: &MemoryRecord) -> f64 {
    let mut score = 0.0_f64;
    if query.session_id.as_deref() == Some(memory.session_id.as_str()) {
        score += 0.2;
    }
    if query.task_id.as_deref() == memory.task_id.as_deref() {
        score += 0.15;
    }
    if query.agent_id.as_deref() == Some(memory.agent_id.as_str()) {
        score += 0.08;
    }
    if let Some(namespace) = query.namespace.as_deref() {
        if memory
            .extra
            .get("namespace")
            .and_then(|value| value.as_str())
            == Some(namespace)
        {
            score += 0.12;
        }
    }
    score
}

fn vector_score_weight(memory: &MemoryRecord) -> f64 {
    if memory.id.starts_with("continuity-memory:") {
        0.0
    } else {
        0.75
    }
}

fn continuity_kind_score(memory: &MemoryRecord) -> f64 {
    continuity_kind_label(memory)
        .map(|kind| match kind {
            "operational_scar" => 0.42,
            "decision" => 0.34,
            "constraint" => 0.31,
            "incident" => 0.28,
            "signal" => 0.22,
            "lesson" => 0.2,
            "outcome" => 0.18,
            "fact" => 0.16,
            "hypothesis" => 0.14,
            "derivation" => 0.12,
            "summary" => 0.11,
            "work_claim" => 0.1,
            "working_state" => 0.08,
            _ => 0.0,
        })
        .unwrap_or(0.0)
}

fn continuity_status_score(memory: &MemoryRecord) -> f64 {
    continuity_status_label(memory)
        .map(|status| match status {
            "open" => 0.18,
            "active" => 0.14,
            "resolved" => 0.03,
            "superseded" => 0.01,
            "rejected" => 0.0,
            _ => 0.0,
        })
        .unwrap_or(0.0)
}

fn continuity_kind_label(memory: &MemoryRecord) -> Option<&str> {
    if !memory.id.starts_with("continuity-memory:") {
        return None;
    }
    memory.extra.get("kind").and_then(|value| value.as_str())
}

fn continuity_status_label(memory: &MemoryRecord) -> Option<&str> {
    if !memory.id.starts_with("continuity-memory:") {
        return None;
    }
    memory.extra.get("status").and_then(|value| value.as_str())
}

fn continuity_belief_key_label(memory: &MemoryRecord) -> Option<&str> {
    if !memory.id.starts_with("continuity-memory:") {
        return None;
    }
    continuity_metadata_str(&memory.extra, "belief_key")
}

fn continuity_source_role_label(memory: &MemoryRecord) -> Option<&str> {
    if !memory.id.starts_with("continuity-memory:") {
        return None;
    }
    continuity_metadata_str(&memory.extra, "source_role")
}

fn continuity_source_role_score(memory: &MemoryRecord) -> f64 {
    let belief_key = continuity_belief_key_label(memory);
    let source_role = continuity_source_role_label(memory);
    let kind = continuity_kind_label(memory);
    let is_user_belief = belief_key
        .map(|value| value.starts_with("user."))
        .unwrap_or(false);
    match source_role {
        Some("user") => {
            if is_user_belief {
                0.2
            } else {
                0.08
            }
        }
        Some("assistant") => {
            if is_user_belief {
                -0.18
            } else if matches!(kind, Some("fact" | "derivation" | "lesson")) {
                -0.08
            } else {
                -0.03
            }
        }
        Some("importer") | Some("system") | Some("tool") => {
            if is_user_belief {
                0.03
            } else {
                0.06
            }
        }
        Some(_) | None => 0.0,
    }
}

fn continuity_metadata_str<'a>(extra: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    extra.get(key).and_then(|value| value.as_str()).or_else(|| {
        extra
            .get("user")
            .and_then(|value| value.get(key))
            .and_then(|value| value.as_str())
    })
}

#[cfg(test)]
mod tests {
    use super::{
        continuity_kind_label, continuity_kind_score, continuity_status_label,
        continuity_status_score, vector_score_weight,
    };
    use crate::model::{MemoryLayer, MemoryRecord, Scope};
    use chrono::Utc;

    fn continuity_memory(kind: &str, status: &str) -> MemoryRecord {
        MemoryRecord {
            id: "continuity-memory:test".into(),
            layer: MemoryLayer::Semantic,
            scope: Scope::Project,
            agent_id: "agent".into(),
            session_id: "session".into(),
            task_id: Some("task".into()),
            ts: Utc::now(),
            importance: 0.9,
            confidence: 0.9,
            token_estimate: 12,
            source_event_id: None,
            scope_key: format!("{kind}:{status}"),
            body: "body".into(),
            extra: serde_json::json!({
                "kind": kind,
                "status": status,
                "namespace": "bench",
            }),
        }
    }

    #[test]
    fn continuity_pack_priority_prefers_decision_over_incident() {
        let decision = continuity_memory("decision", "active");
        let incident = continuity_memory("incident", "open");

        let decision_total = continuity_kind_score(&decision) + continuity_status_score(&decision);
        let incident_total = continuity_kind_score(&incident) + continuity_status_score(&incident);

        assert_eq!(continuity_kind_label(&decision), Some("decision"));
        assert_eq!(continuity_status_label(&decision), Some("active"));
        assert!(decision_total > incident_total);
    }

    #[test]
    fn non_continuity_memory_gets_no_continuity_priority() {
        let mut memory = continuity_memory("decision", "active");
        memory.id = "semantic:test".into();

        assert_eq!(continuity_kind_label(&memory), None);
        assert_eq!(continuity_status_label(&memory), None);
        assert_eq!(continuity_kind_score(&memory), 0.0);
        assert_eq!(continuity_status_score(&memory), 0.0);
    }

    #[test]
    fn continuity_memory_drops_direct_vector_weight() {
        let memory = continuity_memory("decision", "active");
        assert_eq!(vector_score_weight(&memory), 0.0);
    }

    #[test]
    fn non_continuity_memory_keeps_vector_weight() {
        let mut memory = continuity_memory("decision", "active");
        memory.id = "semantic:test".into();
        assert_eq!(vector_score_weight(&memory), 0.75);
    }
}
