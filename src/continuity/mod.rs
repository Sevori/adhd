mod helpers;
mod interface;
mod kernel;
mod schema;
mod types;

// Re-export everything that was pub in the original continuity.rs.
// External code uses `crate::continuity::X` — nothing must break.

pub use interface::{UciRequest, UciResponse, UnifiedContinuityInterface};
pub use kernel::SharedContinuityKernel;
pub use schema::{
    AgentAttachmentRecord, AgentBadgeRecord, AttachAgentInput, ClaimWorkInput, ContextRead,
    ContextRecord, ContinuityCompiledChunkRecord, ContinuityCompilerStateRecord,
    ContinuityHandoffInput, ContinuityHandoffRecord, ContinuityItemInput, ContinuityItemRecord,
    ContinuityPlasticityState, ContinuityRecall, ContinuityRecallCompiler, ContinuityRecallItem,
    ContinuityRetentionState, CoordinationProjectedLane, CoordinationSignalInput, ExplainTarget,
    HandoffProof, HandoffProofRegister, HeartbeatInput, LaneProjectionRecord, LearningView,
    LearningViewMode, MachineProfile, OpenContextInput, OutcomeInput, PracticeEvidenceRecord,
    PracticeLifecycleState, PracticeView, ReadContextInput, RecallInput, ResolveOrSupersedeInput,
    ResumeInput, ResumeRecord, SignalInput, SnapshotInput, SnapshotManifest, SnapshotRecord,
    SupportRef, TelemetryEventInput, UpsertAgentBadgeInput, WriteEventInput,
};
pub use types::{
    ContextStatus, ContinuityKind, ContinuityStatus, CoordinationLane, CoordinationSeverity,
    DEFAULT_MACHINE_TASK_ID, MACHINE_NAMESPACE_ALIAS,
};

// Re-export pub(crate) items used by other modules in this crate.
pub(crate) use helpers::{
    annotate_practice_states, build_current_practice_view, coordination_signal,
    default_work_claim_lease_seconds, merge_work_claim_extra, normalize_work_claim_resources,
    objective_requests_current_state_context, objective_requests_history_context,
    work_claim_coordination, work_claim_is_live, work_claim_key, work_claims_conflict,
};
pub(crate) use schema::{CoordinationSignalRecord, WorkClaimConflict, WorkClaimCoordination};

// Used by test code in other modules (storage.rs, dogfood.rs).
#[allow(unused_imports)]
pub(crate) use helpers::{
    CoordinationSignalExtraInput, coordination_signal_from_extra, merge_coordination_signal_extra,
};

#[cfg(test)]
mod tests {
    use chrono::{Duration, Utc};
    use rusqlite::{Connection, params};
    use tempfile::tempdir;

    use crate::engine::Engine;
    use crate::model::{DimensionValue, MemoryLayer, Scope, SnapshotResolution};

    use super::helpers::*;
    use super::schema::*;
    use super::types::*;
    use super::*;

    #[test]
    fn continuity_round_trip_supports_snapshot_and_resume() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let attachment = engine
            .attach_agent(AttachAgentInput {
                agent_id: "planner-a".into(),
                agent_type: "ollama".into(),
                capabilities: vec!["plan".into(), "reason".into()],
                namespace: "demo".into(),
                role: Some("planner".into()),
                metadata: serde_json::json!({"model": "glm-4.7-flash"}),
            })
            .unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "shared-kernel".into(),
                session_id: "session-1".into(),
                objective: "stabilize continuity".into(),
                selector: None,
                agent_id: Some("planner-a".into()),
                attachment_id: Some(attachment.id.clone()),
            })
            .unwrap();

        let decision = engine
            .mark_decision(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "planner-a".into(),
                kind: ContinuityKind::Decision,
                title: "Context identity wins".into(),
                body: "Use context namespace and task lineage as the stable identity, not the current agent.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Active),
                importance: Some(0.95),
                confidence: Some(0.9),
                salience: Some(0.95),
                layer: None,
                supports: Vec::new(),
                dimensions: vec![DimensionValue {
                    key: "decision.identity".into(),
                    value: "context-first".into(),
                    weight: DEFAULT_DIMENSION_WEIGHT,
                }],
                extra: serde_json::json!({}),
            })
            .unwrap();
        let scar = engine
            .mark_operational_scar(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "debugger-a".into(),
                kind: ContinuityKind::OperationalScar,
                title: "Do not probe Ollama blindly".into(),
                body: "Ad hoc model probes hung; use the controlled adapter path and capture telemetry.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.98),
                confidence: Some(0.9),
                salience: Some(0.99),
                layer: None,
                supports: vec![SupportRef {
                    support_type: "continuity".into(),
                    support_id: decision.id.clone(),
                    reason: Some("same runtime path".into()),
                    weight: 0.8,
                }],
                dimensions: vec![DimensionValue {
                    key: "scar.runtime".into(),
                    value: "ollama-probe".into(),
                    weight: DEFAULT_DIMENSION_WEIGHT,
                }],
                extra: serde_json::json!({}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id.clone()),
                namespace: None,
                task_id: None,
                objective: "resume continuity work".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("coder-a".into()),
                session_id: Some("session-2".into()),
                view_id: None,
                include_resolved: false,
                candidate_limit: 16,
            })
            .unwrap();

        assert_eq!(read.decisions.len(), 1);
        assert_eq!(read.operational_scars.len(), 1);
        assert!(
            read.decisions
                .iter()
                .any(|item| item.title.contains("Context identity wins"))
        );

        let snapshot = engine
            .snapshot(SnapshotInput {
                context_id: Some(context.id.clone()),
                namespace: None,
                task_id: None,
                objective: Some("checkpoint before handoff".into()),
                selector: None,
                resolution: SnapshotResolution::Medium,
                token_budget: 256,
                candidate_limit: 16,
                owner_agent_id: Some("planner-a".into()),
            })
            .unwrap();
        let resumed = engine
            .resume(ResumeInput {
                snapshot_id: Some(snapshot.id.clone()),
                context_id: None,
                namespace: None,
                task_id: None,
                objective: "resume from snapshot".into(),
                token_budget: 256,
                candidate_limit: 16,
                agent_id: Some("small-agent".into()),
            })
            .unwrap();

        assert_eq!(resumed.snapshot.unwrap().id, snapshot.id);
        assert_eq!(resumed.context.operational_scars[0].id, scar.id);
    }

    #[test]
    fn heartbeat_keeps_attachment_alive_without_fake_work() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let attachment = engine
            .attach_agent(AttachAgentInput {
                agent_id: "idle-agent".into(),
                agent_type: "local".into(),
                capabilities: vec!["observe".into()],
                namespace: "demo".into(),
                role: Some("observer".into()),
                metadata: serde_json::json!({"mode": "heartbeat-test"}),
            })
            .unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "heartbeat".into(),
                session_id: "session-heartbeat".into(),
                objective: "stay visible while idle".into(),
                selector: None,
                agent_id: Some("idle-agent".into()),
                attachment_id: Some(attachment.id.clone()),
            })
            .unwrap();

        let beat = engine
            .heartbeat(HeartbeatInput {
                attachment_id: Some(attachment.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
            })
            .unwrap();

        assert_eq!(beat.id, attachment.id);
        assert!(beat.tick_count >= 3);
        assert_eq!(beat.context_id.as_deref(), Some(context.id.as_str()));

        let metrics = engine.metrics_snapshot().unwrap().prometheus_text;
        assert!(metrics.contains(
            "ice_agent_active{agent_id=\"idle-agent\",agent_type=\"local\",namespace=\"demo\",role=\"observer\"} 1"
        ));
    }

    #[test]
    fn stale_working_state_fades_below_operational_scar() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "retention-test".into(),
                session_id: "session-1".into(),
                objective: "let scars outlive stale scratch state".into(),
                selector: None,
                agent_id: Some("operator".into()),
                attachment_id: None,
            })
            .unwrap();
        let mut working_items = engine
            .write_derivations(vec![ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "operator".into(),
                kind: ContinuityKind::WorkingState,
                title: "Old scratch".into(),
                body: "Temporary scratch note that should fade.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Active),
                importance: Some(0.95),
                confidence: Some(0.95),
                salience: Some(0.98),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            }])
            .unwrap();
        let working = working_items.remove(0);
        let scar = engine
            .mark_operational_scar(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "operator".into(),
                kind: ContinuityKind::OperationalScar,
                title: "Expensive failure".into(),
                body: "This failure should stay sticky.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.70),
                confidence: Some(0.80),
                salience: Some(0.72),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let sqlite = dir.path().join("data/ice.sqlite");
        let conn = Connection::open(sqlite).unwrap();
        let stale_at = (Utc::now() - Duration::days(21)).to_rfc3339();
        conn.execute(
            "UPDATE continuity_items SET ts = ?2, updated_at = ?2 WHERE id = ?1",
            params![working.id, stale_at],
        )
        .unwrap();
        conn.execute(
            "UPDATE continuity_items SET ts = ?2, updated_at = ?2 WHERE id = ?1",
            params![scar.id, stale_at],
        )
        .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "resume without repeating failure".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("operator".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert_eq!(read.open_threads[0].kind, ContinuityKind::OperationalScar);
        assert!(
            read.operational_scars[0].retention.effective_salience
                > read.working_state[0].retention.effective_salience
        );
    }

    #[test]
    fn resolved_items_enter_treated_retention_and_decay_faster() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "therapy-test".into(),
                session_id: "session-1".into(),
                objective: "treated memories should still exist but interfere less".into(),
                selector: None,
                agent_id: Some("therapist".into()),
                attachment_id: None,
            })
            .unwrap();
        let open_constraint = engine
            .mark_constraint(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "therapist".into(),
                kind: ContinuityKind::Constraint,
                title: "Keep logs portable".into(),
                body: "Observability must stay portable.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.85),
                confidence: Some(0.85),
                salience: Some(0.80),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        let treated_constraint = engine
            .mark_constraint(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "therapist".into(),
                kind: ContinuityKind::Constraint,
                title: "Old workaround".into(),
                body: "No longer required after the fix.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.85),
                confidence: Some(0.85),
                salience: Some(0.80),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .resolve_or_supersede(ResolveOrSupersedeInput {
                continuity_id: treated_constraint.id.clone(),
                actor_agent_id: "therapist".into(),
                new_status: ContinuityStatus::Resolved,
                supersedes_id: None,
                resolution_note: Some("therapy completed".into()),
                extra: serde_json::json!({"mode": "therapy"}),
            })
            .unwrap();

        let sqlite = dir.path().join("data/ice.sqlite");
        let conn = Connection::open(sqlite).unwrap();
        let aged_at = (Utc::now() - Duration::days(14)).to_rfc3339();
        conn.execute(
            "UPDATE continuity_items SET ts = ?2, updated_at = ?2 WHERE id = ?1",
            params![open_constraint.id, aged_at],
        )
        .unwrap();
        conn.execute(
            "UPDATE continuity_items SET ts = ?2, updated_at = ?2, resolved_at = ?2 WHERE id = ?1",
            params![treated_constraint.id, aged_at],
        )
        .unwrap();

        let items = engine
            .with_storage(|storage, _| storage.list_continuity_items(&context.id, true))
            .unwrap();
        let open_item = items
            .iter()
            .find(|item| item.id == open_constraint.id)
            .unwrap();
        let treated_item = items
            .iter()
            .find(|item| item.id == treated_constraint.id)
            .unwrap();

        assert_eq!(treated_item.retention.class, "treated_constraint");
        assert!(open_item.retention.effective_salience > treated_item.retention.effective_salience);
    }

    #[test]
    fn read_context_exposes_organism_state_summary() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "organism-summary".into(),
                session_id: "session-1".into(),
                objective: "summarize organism state".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();
        engine
            .mark_operational_scar(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::OperationalScar,
                title: "Do not repeat the hang".into(),
                body: "A prior hang should remain visible.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.9),
                confidence: Some(0.9),
                salience: Some(0.9),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        let treated = engine
            .mark_constraint(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Constraint,
                title: "Legacy workaround".into(),
                body: "This constraint has been retired.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.8),
                confidence: Some(0.8),
                salience: Some(0.7),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .resolve_or_supersede(ResolveOrSupersedeInput {
                continuity_id: treated.id.clone(),
                actor_agent_id: "observer".into(),
                new_status: ContinuityStatus::Resolved,
                supersedes_id: None,
                resolution_note: Some("recovery complete".into()),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect organism".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert_eq!(
            read.organism["retention_classes"]["operational_scar"].as_u64(),
            Some(1)
        );
        assert_eq!(
            read.organism["retention_classes"]["treated_constraint"].as_u64(),
            Some(1)
        );
        assert_eq!(read.organism["treated_items"].as_u64(), Some(1));
        assert_eq!(
            read.organism["open_pressure"][0]["retention_class"].as_str(),
            Some("operational_scar")
        );
        assert_eq!(read.recall.items[0].kind, ContinuityKind::OperationalScar);
        assert!(read.recall.summary.contains("operational_scar"));
    }

    #[test]
    fn read_context_surfaces_recent_learning_digest_by_default() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "recent-learning".into(),
                session_id: "session-1".into(),
                objective: "surface recent learning".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();

        let lesson = engine
            .write_derivations(vec![ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Lesson,
                title: "Weekly pattern".into(),
                body: "Repeated retries mean the interface needs a stronger guardrail.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.8),
                confidence: Some(0.9),
                salience: Some(0.78),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            }])
            .unwrap()
            .pop()
            .unwrap();
        let outcome = engine
            .write_derivations(vec![ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Outcome,
                title: "Outcome hardening".into(),
                body: "Adding a stronger fallback stopped the drift.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Resolved),
                importance: Some(0.86),
                confidence: Some(0.9),
                salience: Some(0.82),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            }])
            .unwrap()
            .pop()
            .unwrap();
        let decision = engine
            .mark_decision(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Decision,
                title: "Fresh guardrail".into(),
                body: "Default to the newest learnings before expanding the full history.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Active),
                importance: Some(0.92),
                confidence: Some(0.91),
                salience: Some(0.88),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .write_derivations(vec![ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Fact,
                title: "Ignore this fact".into(),
                body: "Facts should not pollute the learning digest.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.95),
                confidence: Some(0.95),
                salience: Some(0.95),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            }])
            .unwrap();

        let sqlite = dir.path().join("data/ice.sqlite");
        let conn = Connection::open(sqlite).unwrap();
        for (id, ts) in [
            (lesson.id.as_str(), Utc::now() - Duration::days(6)),
            (outcome.id.as_str(), Utc::now() - Duration::days(2)),
            (decision.id.as_str(), Utc::now() - Duration::hours(3)),
        ] {
            conn.execute(
                "UPDATE continuity_items SET ts = ?2, updated_at = ?2 WHERE id = ?1",
                params![id, ts.to_rfc3339()],
            )
            .unwrap();
        }

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "What did we learn recently?".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert_eq!(read.lessons.len(), 1);
        assert_eq!(read.learning.mode, LearningViewMode::Recent);
        assert_eq!(
            read.learning
                .items
                .iter()
                .map(|item| item.title.as_str())
                .collect::<Vec<_>>(),
            vec!["Fresh guardrail", "Outcome hardening", "Weekly pattern"]
        );
        assert!(read.learning.summary.contains("Recent learning digest"));
        assert!(read.learning.summary.contains("Fresh guardrail"));
        assert_eq!(read.rationale["learning_mode"].as_str(), Some("recent"));
        assert_eq!(read.rationale["learning_item_count"].as_u64(), Some(3));
    }

    #[test]
    fn read_context_expands_full_learning_line_when_objective_requests_history() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "learning-lineage".into(),
                session_id: "session-1".into(),
                objective: "track learning lineage".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();

        let incident = engine
            .mark_incident(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Incident,
                title: "Initial miss".into(),
                body: "The system kept resurfacing stale context instead of the latest practice."
                    .into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Resolved),
                importance: Some(0.83),
                confidence: Some(0.88),
                salience: Some(0.84),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        let lesson = engine
            .write_derivations(vec![ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Lesson,
                title: "Mid-course correction".into(),
                body: "Tie learning to continuity instead of leaving it inside prompts.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.87),
                confidence: Some(0.9),
                salience: Some(0.83),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            }])
            .unwrap()
            .pop()
            .unwrap();
        let outcome = engine
            .write_derivations(vec![ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Outcome,
                title: "Current operator habit".into(),
                body: "The operator now sees the latest learning first and can ask for the whole line.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Active),
                importance: Some(0.91),
                confidence: Some(0.92),
                salience: Some(0.9),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            }])
            .unwrap()
            .pop()
            .unwrap();

        let sqlite = dir.path().join("data/ice.sqlite");
        let conn = Connection::open(sqlite).unwrap();
        for (id, ts) in [
            (incident.id.as_str(), Utc::now() - Duration::days(10)),
            (lesson.id.as_str(), Utc::now() - Duration::days(4)),
            (outcome.id.as_str(), Utc::now() - Duration::days(1)),
        ] {
            conn.execute(
                "UPDATE continuity_items SET ts = ?2, updated_at = ?2 WHERE id = ?1",
                params![id, ts.to_rfc3339()],
            )
            .unwrap();
        }

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "Show the full learning timeline over time.".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert_eq!(read.learning.mode, LearningViewMode::Lineage);
        assert_eq!(
            read.learning
                .items
                .iter()
                .map(|item| item.title.as_str())
                .collect::<Vec<_>>(),
            vec![
                "Initial miss",
                "Mid-course correction",
                "Current operator habit"
            ]
        );
        assert!(read.learning.summary.contains("Initial miss"));
        assert!(read.learning.summary.contains("Current operator habit"));
        assert_eq!(read.rationale["learning_mode"].as_str(), Some("lineage"));
        assert_eq!(read.rationale["learning_item_count"].as_u64(), Some(3));
    }

    #[test]
    fn read_context_current_practice_prefers_latest_guidance_and_rejects_stale_pack_competitor() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "current-practice".into(),
                session_id: "session-1".into(),
                objective: "track the current review practice".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();

        let stale = engine
            .mark_decision(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Decision,
                title: "Old review flow".into(),
                body: "Review every PR with the old checklist before touching tests.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Active),
                importance: Some(0.8),
                confidence: Some(0.82),
                salience: Some(0.8),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "practice_key": "operator.review.flow",
                }),
            })
            .unwrap();
        let current = engine
            .mark_decision(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Decision,
                title: "Current review flow".into(),
                body: "Start from the current practice first, then expand lineage on demand."
                    .into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Active),
                importance: Some(0.9),
                confidence: Some(0.9),
                salience: Some(0.88),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "practice_key": "operator.review.flow",
                }),
            })
            .unwrap();
        let guardrail = engine
            .mark_constraint(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Constraint,
                title: "Latest guardrail".into(),
                body: "Default to current practice in normal recall mode.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.88),
                confidence: Some(0.9),
                salience: Some(0.86),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "practice_key": "operator.review.guardrail",
                }),
            })
            .unwrap();

        let sqlite = dir.path().join("data/ice.sqlite");
        let conn = Connection::open(sqlite).unwrap();
        for (item_id, memory_id, ts) in [
            (
                stale.id.as_str(),
                stale.memory_id.as_str(),
                Utc::now() - Duration::days(35),
            ),
            (
                current.id.as_str(),
                current.memory_id.as_str(),
                Utc::now() - Duration::hours(4),
            ),
            (
                guardrail.id.as_str(),
                guardrail.memory_id.as_str(),
                Utc::now() - Duration::hours(2),
            ),
        ] {
            conn.execute(
                "UPDATE continuity_items SET ts = ?2, updated_at = ?2 WHERE id = ?1",
                params![item_id, ts.to_rfc3339()],
            )
            .unwrap();
            conn.execute(
                "UPDATE memory_items SET ts = ?2 WHERE id = ?1",
                params![memory_id, ts.to_rfc3339()],
            )
            .unwrap();
        }

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "Resume the current review practice.".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        let stale_decision = read
            .decisions
            .iter()
            .find(|item| item.id == stale.id)
            .unwrap();
        let current_decision = read
            .decisions
            .iter()
            .find(|item| item.id == current.id)
            .unwrap();
        assert_eq!(
            stale_decision.practice_state,
            Some(PracticeLifecycleState::Retired)
        );
        assert_eq!(
            current_decision.practice_state,
            Some(PracticeLifecycleState::Current)
        );
        assert!(
            read.current_practice
                .items
                .iter()
                .any(|item| item.id == current.id)
        );
        assert!(
            read.current_practice
                .items
                .iter()
                .any(|item| item.id == guardrail.id)
        );
        assert!(
            !read
                .current_practice
                .items
                .iter()
                .any(|item| item.id == stale.id)
        );
        assert!(
            read.current_practice
                .summary
                .contains("Current review flow")
        );
        assert!(
            read.pack
                .items
                .iter()
                .any(|item| item.memory_id == current.memory_id)
        );
        assert!(
            !read
                .pack
                .items
                .iter()
                .any(|item| item.memory_id == stale.memory_id)
        );

        let manifest = engine.explain_context_pack(&read.pack.id).unwrap();
        assert!(manifest.rejected.iter().any(|candidate| {
            candidate.memory_id == stale.memory_id && candidate.reason == "practice_key_competitor"
        }));
    }

    #[test]
    fn read_context_current_practice_prefers_corroborated_guidance_over_unproven_competitor() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "current-practice-evidence".into(),
                session_id: "session-1".into(),
                objective: "track the current review practice".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();

        let grounded = engine
            .mark_decision(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Decision,
                title: "Grounded review flow".into(),
                body: "Start from current practice and keep the learning chain attached.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Active),
                importance: Some(0.88),
                confidence: Some(0.9),
                salience: Some(0.88),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "practice_key": "operator.review.flow",
                }),
            })
            .unwrap();
        let unproven = engine
            .mark_decision(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Decision,
                title: "Fresh but unproven review flow".into(),
                body: "Replace the review flow with a brand new shortcut immediately.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Active),
                importance: Some(0.88),
                confidence: Some(0.9),
                salience: Some(0.88),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "practice_key": "operator.review.flow",
                }),
            })
            .unwrap();
        let derivations = engine
            .write_derivations(vec![
                ContinuityItemInput {
                    context_id: context.id.clone(),
                    author_agent_id: "observer".into(),
                    kind: ContinuityKind::Lesson,
                    title: "Review lineage stayed stable".into(),
                    body: "The grounded flow kept continuity intact during the last review pass."
                        .into(),
                    scope: Scope::Project,
                    status: Some(ContinuityStatus::Resolved),
                    importance: Some(0.82),
                    confidence: Some(0.88),
                    salience: Some(0.82),
                    layer: None,
                    supports: vec![SupportRef {
                        support_type: "continuity".into(),
                        support_id: grounded.id.clone(),
                        reason: Some("belief_update_current".into()),
                        weight: 1.1,
                    }],
                    dimensions: Vec::new(),
                    extra: serde_json::json!({}),
                },
                ContinuityItemInput {
                    context_id: context.id.clone(),
                    author_agent_id: "observer".into(),
                    kind: ContinuityKind::Outcome,
                    title: "Review outcome confirmed the grounded flow".into(),
                    body: "The last review outcome confirmed that the grounded flow reduced drift."
                        .into(),
                    scope: Scope::Project,
                    status: Some(ContinuityStatus::Resolved),
                    importance: Some(0.84),
                    confidence: Some(0.9),
                    salience: Some(0.84),
                    layer: None,
                    supports: vec![SupportRef {
                        support_type: "continuity".into(),
                        support_id: grounded.id.clone(),
                        reason: Some("outcome_confirmed".into()),
                        weight: 1.2,
                    }],
                    dimensions: Vec::new(),
                    extra: serde_json::json!({}),
                },
            ])
            .unwrap();
        let lesson = derivations[0].clone();
        let outcome = derivations[1].clone();

        let sqlite = dir.path().join("data/ice.sqlite");
        let conn = Connection::open(sqlite).unwrap();
        for (item_id, memory_id, ts) in [
            (
                grounded.id.as_str(),
                grounded.memory_id.as_str(),
                Utc::now() - Duration::hours(6),
            ),
            (
                unproven.id.as_str(),
                unproven.memory_id.as_str(),
                Utc::now() - Duration::hours(1),
            ),
            (
                lesson.id.as_str(),
                lesson.memory_id.as_str(),
                Utc::now() - Duration::hours(2),
            ),
            (
                outcome.id.as_str(),
                outcome.memory_id.as_str(),
                Utc::now() - Duration::minutes(30),
            ),
        ] {
            conn.execute(
                "UPDATE continuity_items SET ts = ?2, updated_at = ?2 WHERE id = ?1",
                params![item_id, ts.to_rfc3339()],
            )
            .unwrap();
            conn.execute(
                "UPDATE memory_items SET ts = ?2 WHERE id = ?1",
                params![memory_id, ts.to_rfc3339()],
            )
            .unwrap();
        }

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "Resume the current review practice.".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert_eq!(
            read.current_practice
                .items
                .first()
                .map(|item| item.id.as_str()),
            Some(grounded.id.as_str())
        );
        assert!(
            !read
                .current_practice
                .items
                .iter()
                .any(|item| item.id == unproven.id)
        );
        let grounded_evidence = read
            .current_practice
            .evidence
            .iter()
            .find(|bundle| bundle.practice_id == grounded.id)
            .unwrap();
        assert_eq!(grounded_evidence.evidence_count, 2);
        assert!(grounded_evidence.support_signal > 0.15);
        assert!(
            grounded_evidence
                .evidence
                .iter()
                .any(|item| item.id == lesson.id)
        );
        assert!(
            grounded_evidence
                .evidence
                .iter()
                .any(|item| item.id == outcome.id)
        );
        assert!(
            read.current_practice
                .summary
                .contains("Grounded review flow")
        );
        assert!(
            read.current_practice
                .summary
                .contains("Review lineage stayed stable")
        );
        assert!(
            read.current_practice
                .summary
                .contains("Review outcome confirmed the grounded flow")
        );
        assert!(
            read.pack
                .items
                .iter()
                .any(|item| item.memory_id == grounded.memory_id)
        );
        assert!(
            read.pack
                .items
                .iter()
                .any(|item| item.memory_id == lesson.memory_id)
        );
        assert!(
            read.pack
                .items
                .iter()
                .any(|item| item.memory_id == outcome.memory_id)
        );
        assert!(
            read.pack
                .items
                .iter()
                .filter(|item| item.memory_id == lesson.memory_id
                    || item.memory_id == outcome.memory_id)
                .all(|item| item
                    .why
                    .iter()
                    .any(|why| why == "current_practice_evidence"))
        );
    }

    #[test]
    fn read_context_history_mode_keeps_practice_lineage_available() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "practice-history".into(),
                session_id: "session-1".into(),
                objective: "track review-practice evolution".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();

        let stale = engine
            .mark_decision(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Decision,
                title: "Old review flow".into(),
                body: "Older operator habit that should remain available for lineage.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Active),
                importance: Some(0.8),
                confidence: Some(0.82),
                salience: Some(0.8),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "practice_key": "operator.review.flow",
                }),
            })
            .unwrap();
        let current = engine
            .mark_decision(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Decision,
                title: "Current review flow".into(),
                body: "Newer operator habit that should win in normal recall mode.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Active),
                importance: Some(0.9),
                confidence: Some(0.9),
                salience: Some(0.88),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "practice_key": "operator.review.flow",
                }),
            })
            .unwrap();

        let sqlite = dir.path().join("data/ice.sqlite");
        let conn = Connection::open(sqlite).unwrap();
        for (item_id, memory_id, ts) in [
            (
                stale.id.as_str(),
                stale.memory_id.as_str(),
                Utc::now() - Duration::days(28),
            ),
            (
                current.id.as_str(),
                current.memory_id.as_str(),
                Utc::now() - Duration::hours(3),
            ),
        ] {
            conn.execute(
                "UPDATE continuity_items SET ts = ?2, updated_at = ?2 WHERE id = ?1",
                params![item_id, ts.to_rfc3339()],
            )
            .unwrap();
            conn.execute(
                "UPDATE memory_items SET ts = ?2 WHERE id = ?1",
                params![memory_id, ts.to_rfc3339()],
            )
            .unwrap();
        }

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "Show the review practice timeline and why it changed over time.".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert_eq!(read.learning.mode, LearningViewMode::Lineage);
        assert!(read.learning.items.iter().any(|item| item.id == stale.id));
        assert!(read.learning.items.iter().any(|item| item.id == current.id));
        assert!(
            read.current_practice
                .items
                .iter()
                .all(|item| item.id != stale.id)
        );
        assert!(
            read.pack
                .items
                .iter()
                .any(|item| item.memory_id == stale.memory_id)
        );
        assert!(
            read.pack
                .items
                .iter()
                .any(|item| item.memory_id == current.memory_id)
        );
    }

    #[test]
    fn read_context_open_pressure_ignores_stale_open_guidance() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "open-pressure".into(),
                session_id: "session-1".into(),
                objective: "track live review guidance".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();

        let stale = engine
            .mark_decision(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Decision,
                title: "Old review ritual".into(),
                body: "Ask Claude to refresh the old MCP session before every move.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.82),
                confidence: Some(0.84),
                salience: Some(0.82),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        let current = engine
            .mark_constraint(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Constraint,
                title: "Current review guidance".into(),
                body: "Default to current practice in normal recall mode.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.88),
                confidence: Some(0.9),
                salience: Some(0.87),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let sqlite = dir.path().join("data/ice.sqlite");
        let conn = Connection::open(sqlite).unwrap();
        conn.execute(
            "UPDATE continuity_items SET ts = ?2, updated_at = ?2 WHERE id = ?1",
            params![
                stale.id.as_str(),
                (Utc::now() - Duration::days(12)).to_rfc3339()
            ],
        )
        .unwrap();
        conn.execute(
            "UPDATE memory_items SET ts = ?2 WHERE id = ?1",
            params![
                stale.memory_id.as_str(),
                (Utc::now() - Duration::days(12)).to_rfc3339()
            ],
        )
        .unwrap();
        conn.execute(
            "UPDATE continuity_items SET ts = ?2, updated_at = ?2 WHERE id = ?1",
            params![
                current.id.as_str(),
                (Utc::now() - Duration::hours(2)).to_rfc3339()
            ],
        )
        .unwrap();
        conn.execute(
            "UPDATE memory_items SET ts = ?2 WHERE id = ?1",
            params![
                current.memory_id.as_str(),
                (Utc::now() - Duration::hours(2)).to_rfc3339()
            ],
        )
        .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "What guidance is currently live?".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        let open_pressure = read.organism["open_pressure"].as_array().unwrap();
        assert!(open_pressure.iter().any(|item| item["id"] == current.id));
        assert!(!open_pressure.iter().any(|item| item["id"] == stale.id));
    }

    #[test]
    fn read_context_surfaces_derived_belief_update_lessons() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "belief-learning".into(),
                session_id: "session-1".into(),
                objective: "surface belief learning".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();

        let stale = engine
            .write_derivations(vec![ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Fact,
                title: "Alice lives in London".into(),
                body: "Alice currently lives in London.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.8),
                confidence: Some(0.8),
                salience: Some(0.8),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "belief_key": "user.location.city",
                    "source_role": "user",
                }),
            }])
            .unwrap()
            .pop()
            .unwrap();
        let replacement = engine
            .write_derivations(vec![ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "observer".into(),
                kind: ContinuityKind::Fact,
                title: "Alice lives in Berlin".into(),
                body: "Alice currently lives in Berlin.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.82),
                confidence: Some(0.82),
                salience: Some(0.82),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({
                    "belief_key": "user.location.city",
                    "source_role": "user",
                }),
            }])
            .unwrap()
            .pop()
            .unwrap();

        engine
            .resolve_or_supersede(ResolveOrSupersedeInput {
                continuity_id: stale.id.clone(),
                actor_agent_id: "observer".into(),
                new_status: ContinuityStatus::Superseded,
                supersedes_id: Some(replacement.id.clone()),
                resolution_note: Some("The user moved.".into()),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "What did we learn recently?".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert!(
            read.learning
                .items
                .iter()
                .any(|item| item.title.starts_with("Belief update:"))
        );
        assert!(read.learning.summary.contains("Belief update:"));
        assert!(read.lessons.iter().any(|item| {
            item.extra["user"]["learning_trigger"]
                == serde_json::json!("prediction_error_reconsolidation")
        }));
    }

    #[test]
    fn read_context_recall_surfaces_trauma_before_stale_working_state() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "recall-fastpath".into(),
                session_id: "session-1".into(),
                objective: "keep trauma searchable".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();

        for index in 0..64 {
            engine
                .write_derivations(vec![ContinuityItemInput {
                    context_id: context.id.clone(),
                    author_agent_id: "observer".into(),
                    kind: ContinuityKind::WorkingState,
                    title: format!("Scratch note {index}"),
                    body: "low-signal scratch state".into(),
                    scope: Scope::Project,
                    status: Some(ContinuityStatus::Active),
                    importance: Some(0.2),
                    confidence: Some(0.6),
                    salience: Some(0.12),
                    layer: None,
                    supports: Vec::new(),
                    dimensions: Vec::new(),
                    extra: serde_json::json!({}),
                }])
                .unwrap();
        }

        engine
            .mark_operational_scar(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "debugger".into(),
                kind: ContinuityKind::OperationalScar,
                title: "Hot reload is worthless if it forks the brain.".into(),
                body: "Split runtime roots make the dashboard lie.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.98),
                confidence: Some(0.9),
                salience: Some(0.99),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id.clone()),
                namespace: None,
                task_id: None,
                objective: "What trauma matters if observability lies after a restart?".into(),
                token_budget: PROOF_TRIM_LIMIT,
                selector: None,
                agent_id: Some("reader".into()),
                session_id: Some("session-2".into()),
                view_id: None,
                include_resolved: false,
                candidate_limit: 12,
            })
            .unwrap();

        assert_eq!(read.recall.items[0].kind, ContinuityKind::OperationalScar);
        assert!(read.recall.items[0].title.contains("forks the brain"));
        assert!(read.recall.items.len() <= 12);
        assert!(
            read.recall.answer_hint.is_some(),
            "expected a safe answer hint for the top trauma"
        );
    }

    #[test]
    fn read_context_pack_reflex_surfaces_recalled_fact_without_manual_tool_choice() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "machine-demo".into(),
                task_id: "reflex-pack".into(),
                session_id: "session-1".into(),
                objective: "keep machine facts instantly available".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();

        for index in 0..48 {
            engine
                .write_derivations(vec![ContinuityItemInput {
                    context_id: context.id.clone(),
                    author_agent_id: "observer".into(),
                    kind: ContinuityKind::WorkingState,
                    title: format!("Scratch note {index}"),
                    body: "low-signal scratch state".into(),
                    scope: Scope::Project,
                    status: Some(ContinuityStatus::Active),
                    importance: Some(0.2),
                    confidence: Some(0.55),
                    salience: Some(0.1),
                    layer: None,
                    supports: Vec::new(),
                    dimensions: Vec::new(),
                    extra: serde_json::json!({}),
                }])
                .unwrap();
        }

        engine
            .write_derivations(vec![ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "archivist".into(),
                kind: ContinuityKind::Fact,
                title: "Alice smell baseline".into(),
                body: "The smell of Alice is flowers.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.97),
                confidence: Some(0.99),
                salience: Some(0.98),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            }])
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id.clone()),
                namespace: None,
                task_id: None,
                objective: "What does Alice smell like?".into(),
                token_budget: 240,
                selector: None,
                agent_id: Some("reader".into()),
                session_id: Some("session-2".into()),
                view_id: None,
                include_resolved: false,
                candidate_limit: 12,
            })
            .unwrap();

        let reflex_item = read
            .pack
            .items
            .iter()
            .take(3)
            .find(|item| item.body.contains("flowers"))
            .expect("expected recalled fact to surface near the front of the pack");
        assert!(reflex_item.breakdown.continuity > 0.0);
        assert!(
            reflex_item
                .why
                .iter()
                .any(|why| why.starts_with("continuity#"))
        );
        assert_eq!(
            read.recall.answer_hint.as_deref(),
            Some("The smell of Alice is flowers.")
        );
        assert_eq!(read.recall.compiler.dominant_band.as_deref(), Some("hot"));
        assert!(read.recall.compiler.compiled_hit_count > 0);
        assert!(
            read.recall.items[0]
                .why
                .iter()
                .any(|why| why.starts_with("compiled_"))
        );
        assert_eq!(
            read.rationale["continuity_recall_compiler"]["dominant_band"].as_str(),
            Some("hot")
        );
        assert!(
            read.rationale["continuity_recall_compiler"]["compiled_hit_count"]
                .as_u64()
                .unwrap_or_default()
                > 0
        );
        assert_eq!(
            read.rationale["continuity_recall_top_why"][0].as_str(),
            Some("lexical")
        );
    }

    #[test]
    fn read_context_recall_refreshes_compiled_bands_when_memory_changes() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "machine-demo".into(),
                task_id: "compiler-refresh".into(),
                session_id: "session-1".into(),
                objective: "keep durable facts cheap to recall".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();

        engine
            .write_derivations(vec![ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "archivist".into(),
                kind: ContinuityKind::Fact,
                title: "Baseline smell".into(),
                body: "Alice smells like rain.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.92),
                confidence: Some(0.95),
                salience: Some(0.91),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            }])
            .unwrap();

        let first = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id.clone()),
                namespace: None,
                task_id: None,
                objective: "What does Alice smell like?".into(),
                token_budget: PROOF_TRIM_LIMIT,
                selector: None,
                agent_id: Some("reader-a".into()),
                session_id: Some("session-a".into()),
                view_id: None,
                include_resolved: false,
                candidate_limit: 12,
            })
            .unwrap();
        assert_eq!(
            first.recall.answer_hint.as_deref(),
            Some("Alice smells like rain.")
        );

        engine
            .write_derivations(vec![ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "archivist".into(),
                kind: ContinuityKind::Fact,
                title: "Upgraded smell".into(),
                body: "Alice smells like flowers.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.97),
                confidence: Some(0.99),
                salience: Some(0.99),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            }])
            .unwrap();

        let second = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "What does Alice smell like now?".into(),
                token_budget: PROOF_TRIM_LIMIT,
                selector: None,
                agent_id: Some("reader-b".into()),
                session_id: Some("session-b".into()),
                view_id: None,
                include_resolved: false,
                candidate_limit: 12,
            })
            .unwrap();
        assert!(second.recall.items[0].preview.contains("flowers"));
        assert!(
            second.recall.items[0]
                .why
                .iter()
                .any(|why| why.starts_with("compiled_"))
        );
        assert_eq!(second.recall.compiler.dominant_band.as_deref(), Some("hot"));
    }

    #[test]
    fn explain_context_surfaces_continuity_compiler_state_and_chunks() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "machine-demo".into(),
                task_id: "compiler-explain".into(),
                session_id: "session-1".into(),
                objective: "make compiler state explainable".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();

        engine
            .write_derivations(vec![ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "archivist".into(),
                kind: ContinuityKind::Fact,
                title: "Compiler explain smell".into(),
                body: "Alice smells like flowers.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.97),
                confidence: Some(0.99),
                salience: Some(0.99),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            }])
            .unwrap();

        engine
            .read_context(ReadContextInput {
                context_id: Some(context.id.clone()),
                namespace: None,
                task_id: None,
                objective: "What does Alice smell like?".into(),
                token_budget: PROOF_TRIM_LIMIT,
                selector: None,
                agent_id: Some("reader".into()),
                session_id: Some("session-a".into()),
                view_id: None,
                include_resolved: false,
                candidate_limit: 12,
            })
            .unwrap();

        let explained = engine
            .explain(ExplainTarget::Context { id: context.id })
            .unwrap();
        assert_eq!(
            explained["compiler"]["state"]["dirty"].as_bool(),
            Some(false)
        );
        assert_eq!(
            explained["compiler"]["state"]["item_count"].as_u64(),
            Some(1)
        );
        assert_eq!(
            explained["compiler"]["chunks"][0]["band"].as_str(),
            Some("hot")
        );
        assert_eq!(
            explained["compiler"]["chunks"][0]["item_count"].as_u64(),
            Some(1)
        );
    }

    #[test]
    fn read_context_marks_dispatch_as_unconfigured_when_absent() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "dispatch-unconfigured".into(),
                session_id: "session-1".into(),
                objective: "inspect dispatch state".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: None,
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect organism".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert_eq!(
            read.organism["dispatch"]["configured"].as_bool(),
            Some(false)
        );
        assert_eq!(
            read.organism["dispatch"]["workers_active"].as_u64(),
            Some(0)
        );
        assert_eq!(
            read.organism["dispatch"]["assignments_active"].as_u64(),
            Some(0)
        );
    }

    #[test]
    fn read_context_exposes_lane_projections() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let attachment = engine
            .attach_agent(AttachAgentInput {
                agent_id: "observer".into(),
                agent_type: "codex".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "demo".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo", "branch": "main"}),
            })
            .unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: DEFAULT_MACHINE_TASK_ID.into(),
                session_id: "session-projection".into(),
                objective: "project repo lanes".into(),
                selector: None,
                agent_id: Some("observer".into()),
                attachment_id: Some(attachment.id.clone()),
            })
            .unwrap();
        engine
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(attachment.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Observer".into()),
                status: Some("syncing".into()),
                focus: Some("project repo lanes".into()),
                headline: Some("surface repo worktree projections".into()),
                resource: Some("repo/demo/main".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({"source": "test"}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "observer".into(),
                title: "Own repo lane".into(),
                body: "Project the repo lane from the machine brain.".into(),
                scope: Scope::Shared,
                resources: vec!["repo/demo/main".into()],
                exclusive: true,
                attachment_id: Some(attachment.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .publish_coordination_signal(CoordinationSignalInput {
                context_id: context.id.clone(),
                agent_id: "boundary-warden".into(),
                title: "Back off the repo lane".into(),
                body: "The repo lane is under active coordination pressure.".into(),
                lane: CoordinationLane::Backoff,
                target_agent_id: Some("observer".into()),
                target_projected_lane: Some(CoordinationProjectedLane {
                    projection_id: "repo:/tmp/demo:main".into(),
                    projection_kind: "repo".into(),
                    label: "demo @ main".into(),
                    resource: Some("repo/demo/main".into()),
                    repo_root: Some("/tmp/demo".into()),
                    branch: Some("main".into()),
                    task_id: Some(DEFAULT_MACHINE_TASK_ID.into()),
                }),
                claim_id: None,
                resource: Some("repo/demo/main".into()),
                severity: None,
                projection_ids: vec!["repo:/tmp/demo:main".into()],
                projected_lanes: vec![CoordinationProjectedLane {
                    projection_id: "repo:/tmp/demo:main".into(),
                    projection_kind: "repo".into(),
                    label: "demo @ main".into(),
                    resource: Some("repo/demo/main".into()),
                    repo_root: Some("/tmp/demo".into()),
                    branch: Some("main".into()),
                    task_id: Some(DEFAULT_MACHINE_TASK_ID.into()),
                }],
                extra: serde_json::json!({"source": "test"}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect projected lanes".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: false,
                candidate_limit: 32,
            })
            .unwrap();

        let projections = read.organism["lane_projections"].as_array().unwrap();
        assert_eq!(projections.len(), 1);
        assert_eq!(projections[0]["projection_kind"].as_str(), Some("repo"));
        assert_eq!(projections[0]["label"].as_str(), Some("demo @ main"));
        assert_eq!(projections[0]["connected_agents"].as_u64(), Some(1));
        assert_eq!(projections[0]["live_claims"].as_u64(), Some(1));
        assert_eq!(
            projections[0]["coordination_signal_count"].as_u64(),
            Some(1)
        );
        assert_eq!(projections[0]["blocking_signal_count"].as_u64(), Some(1));
        assert_eq!(
            projections[0]["coordination_lanes"][0].as_str(),
            Some("backoff")
        );
        assert_eq!(
            projections[0]["focus"].as_str(),
            Some("Back off the repo lane")
        );
    }

    #[test]
    fn read_context_keeps_same_repo_branch_worktrees_separate() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let attachment_a = engine
            .attach_agent(AttachAgentInput {
                agent_id: "observer-a".into(),
                agent_type: "codex".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "demo".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/worktrees/adhd", "branch": "main"}),
            })
            .unwrap();
        let attachment_b = engine
            .attach_agent(AttachAgentInput {
                agent_id: "observer-b".into(),
                agent_type: "codex".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "demo".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/shadow/adhd", "branch": "main"}),
            })
            .unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: DEFAULT_MACHINE_TASK_ID.into(),
                session_id: "session-worktrees".into(),
                objective: "separate same-branch repo worktrees".into(),
                selector: None,
                agent_id: Some("observer-a".into()),
                attachment_id: Some(attachment_a.id.clone()),
            })
            .unwrap();
        engine
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(attachment_a.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Observer A".into()),
                status: Some("syncing".into()),
                focus: Some("watch first worktree".into()),
                headline: Some("keep worktree A separate".into()),
                resource: Some("repo/adhd/main".into()),
                repo_root: Some("/tmp/worktrees/adhd".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({"source": "test"}),
            })
            .unwrap();
        engine
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(attachment_b.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Observer B".into()),
                status: Some("syncing".into()),
                focus: Some("watch second worktree".into()),
                headline: Some("keep worktree B separate".into()),
                resource: Some("repo/adhd/main".into()),
                repo_root: Some("/tmp/shadow/adhd".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({"source": "test"}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "observer-a".into(),
                title: "Hold first repo lane".into(),
                body: "Project the first worktree lane.".into(),
                scope: Scope::Shared,
                resources: vec!["repo/adhd/main".into()],
                exclusive: true,
                attachment_id: Some(attachment_a.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "observer-b".into(),
                title: "Hold second repo lane".into(),
                body: "Project the second worktree lane.".into(),
                scope: Scope::Shared,
                resources: vec!["repo/adhd/main".into()],
                exclusive: true,
                attachment_id: Some(attachment_b.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect repo worktree lanes".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer-a".into()),
                session_id: None,
                view_id: None,
                include_resolved: false,
                candidate_limit: 32,
            })
            .unwrap();

        let projections = read.organism["lane_projections"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|projection| projection["projection_kind"].as_str() == Some("repo"))
            .collect::<Vec<_>>();
        assert_eq!(projections.len(), 2);
        assert!(projections.iter().any(|projection| {
            projection["projection_id"].as_str() == Some("repo:/tmp/worktrees/adhd:main")
                && projection["repo_root"].as_str() == Some("/tmp/worktrees/adhd")
                && projection["connected_agents"].as_u64() == Some(1)
                && projection["live_claims"].as_u64() == Some(1)
        }));
        assert!(projections.iter().any(|projection| {
            projection["projection_id"].as_str() == Some("repo:/tmp/shadow/adhd:main")
                && projection["repo_root"].as_str() == Some("/tmp/shadow/adhd")
                && projection["connected_agents"].as_u64() == Some(1)
                && projection["live_claims"].as_u64() == Some(1)
        }));
    }

    #[test]
    fn claim_conflicts_include_projected_lane_identity() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let main_attachment = engine
            .attach_agent(AttachAgentInput {
                agent_id: "agent-main".into(),
                agent_type: "local".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "demo".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo", "branch": "main"}),
            })
            .unwrap();
        let shadow_attachment = engine
            .attach_agent(AttachAgentInput {
                agent_id: "agent-shadow".into(),
                agent_type: "local".into(),
                capabilities: vec!["read_context".into(), "claim_work".into()],
                namespace: "demo".into(),
                role: Some("operator".into()),
                metadata: serde_json::json!({"repo_root": "/tmp/demo", "branch": "feature/shadow"}),
            })
            .unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: DEFAULT_MACHINE_TASK_ID.into(),
                session_id: "session-projected-conflict".into(),
                objective: "surface lane-aware conflicts".into(),
                selector: None,
                agent_id: Some("agent-main".into()),
                attachment_id: Some(main_attachment.id.clone()),
            })
            .unwrap();
        engine
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(main_attachment.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Agent Main".into()),
                status: Some("editing".into()),
                focus: Some("touch file".into()),
                headline: Some("main lane".into()),
                resource: Some("repo/demo/main".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("main".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        engine
            .upsert_agent_badge(UpsertAgentBadgeInput {
                attachment_id: Some(shadow_attachment.id.clone()),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
                display_name: Some("Agent Shadow".into()),
                status: Some("editing".into()),
                focus: Some("touch file".into()),
                headline: Some("shadow lane".into()),
                resource: Some("repo/demo/feature/shadow".into()),
                repo_root: Some("/tmp/demo".into()),
                branch: Some("feature/shadow".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "agent-main".into(),
                title: "Edit storage from main".into(),
                body: "Touch file/src/storage.rs from main.".into(),
                scope: Scope::Project,
                resources: vec!["file/src/storage.rs".into()],
                exclusive: true,
                attachment_id: Some(main_attachment.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "agent-shadow".into(),
                title: "Edit storage from shadow".into(),
                body: "Touch file/src/storage.rs from shadow.".into(),
                scope: Scope::Project,
                resources: vec!["file/src/storage.rs".into()],
                exclusive: true,
                attachment_id: Some(shadow_attachment.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect lane-aware conflict state".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("agent-main".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 24,
            })
            .unwrap();

        let conflicts = read.organism["claim_conflicts"].as_array().unwrap();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(
            conflicts[0]["resource"].as_str(),
            Some("file/src/storage.rs")
        );
        let lane_projection_ids = read.organism["lane_projections"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|projection| projection["projection_id"].as_str())
            .collect::<Vec<_>>();
        let projection_ids = conflicts[0]["projection_ids"].as_array().unwrap();
        assert_eq!(projection_ids.len(), 2);
        assert!(projection_ids.iter().all(|projection_id| {
            projection_id
                .as_str()
                .map(|projection_id| lane_projection_ids.contains(&projection_id))
                .unwrap_or(false)
        }));
        let lanes = conflicts[0]["projected_lanes"].as_array().unwrap();
        assert_eq!(lanes.len(), 2);
        assert!(lanes.iter().any(|lane| {
            lane["label"].as_str() == Some("demo @ main")
                && lane["projection_kind"].as_str() == Some("repo")
                && lane["display_name"].as_str() == Some("Agent Main")
        }));
        assert!(lanes.iter().any(|lane| {
            lane["label"].as_str() == Some("demo @ feature/shadow")
                && lane["projection_kind"].as_str() == Some("repo")
                && lane["display_name"].as_str() == Some("Agent Shadow")
        }));
    }

    #[test]
    fn handoff_manifest_persists_organism_state() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "handoff-organism".into(),
                session_id: "session-1".into(),
                objective: "handoff should carry organism state".into(),
                selector: None,
                agent_id: Some("planner-a".into()),
                attachment_id: None,
            })
            .unwrap();
        engine
            .mark_operational_scar(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "planner-a".into(),
                kind: ContinuityKind::OperationalScar,
                title: "Prior failure".into(),
                body: "This scar should survive the handoff manifest.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.9),
                confidence: Some(0.9),
                salience: Some(0.9),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let handoff = engine
            .handoff(ContinuityHandoffInput {
                context_id: Some(context.id),
                namespace: Some("demo".into()),
                task_id: Some("handoff-organism".into()),
                from_agent_id: "planner-a".into(),
                to_agent_id: "coder-b".into(),
                objective: "resume with organism state".into(),
                reason: "swap heads".into(),
                selector: None,
                resolution: SnapshotResolution::Medium,
                token_budget: 256,
                candidate_limit: 16,
            })
            .unwrap();

        let explained = engine
            .explain(ExplainTarget::Handoff {
                id: handoff.handoff.id,
            })
            .unwrap();
        assert_eq!(
            explained["context"]["organism"]["retention_classes"]["operational_scar"].as_u64(),
            Some(1)
        );
        assert_eq!(
            explained["proof"]["registers"]
                .as_array()
                .unwrap()
                .iter()
                .find(|item| item["label"].as_str() == Some("ps1"))
                .and_then(|item| item["register_kind"].as_str()),
            Some("scar")
        );
        assert!(
            handoff
                .proof
                .registers
                .iter()
                .any(|item| item.label == "ps1" && item.register_kind == "scar")
        );
    }

    #[test]
    fn work_claims_surface_conflicts_in_context_reads() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let planner = engine
            .attach_agent(AttachAgentInput {
                agent_id: "planner-a".into(),
                agent_type: "local".into(),
                capabilities: vec!["plan".into()],
                namespace: "demo".into(),
                role: Some("planner".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let debugger = engine
            .attach_agent(AttachAgentInput {
                agent_id: "debugger-b".into(),
                agent_type: "local".into(),
                capabilities: vec!["debug".into()],
                namespace: "demo".into(),
                role: Some("debugger".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "claim-conflicts".into(),
                session_id: "session-1".into(),
                objective: "coordinate files without collisions".into(),
                selector: None,
                agent_id: Some("planner-a".into()),
                attachment_id: Some(planner.id.clone()),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "planner-a".into(),
                title: "Own src/continuity.rs".into(),
                body: "I am rewriting the kernel coordination path.".into(),
                scope: Scope::Project,
                resources: vec!["src/continuity.rs".into()],
                exclusive: true,
                attachment_id: Some(planner.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "debugger-b".into(),
                title: "Also touching src/continuity.rs".into(),
                body: "I think the planner is doing something cursed.".into(),
                scope: Scope::Project,
                resources: vec!["src/continuity.rs".into()],
                exclusive: true,
                attachment_id: Some(debugger.id),
                lease_seconds: Some(120),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "resume the same claim state".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert_eq!(read.work_claims.len(), 2);
        assert_eq!(read.organism["active_claim_count"].as_u64(), Some(2));
        assert_eq!(read.organism["claim_conflict_count"].as_u64(), Some(1));
        assert!(
            read.open_threads
                .iter()
                .any(|item| item.kind == ContinuityKind::WorkClaim)
        );
    }

    #[test]
    fn heartbeat_renews_attached_work_claim_lease() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let attachment = engine
            .attach_agent(AttachAgentInput {
                agent_id: "coder-a".into(),
                agent_type: "local".into(),
                capabilities: vec!["write".into()],
                namespace: "demo".into(),
                role: Some("coder".into()),
                metadata: serde_json::json!({}),
            })
            .unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "claim-heartbeat".into(),
                session_id: "session-1".into(),
                objective: "keep the claim alive while idle".into(),
                selector: None,
                agent_id: Some("coder-a".into()),
                attachment_id: Some(attachment.id.clone()),
            })
            .unwrap();
        let claim = engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "coder-a".into(),
                title: "Own src/mcp.rs".into(),
                body: "Implement MCP plumbing without collisions.".into(),
                scope: Scope::Project,
                resources: vec!["src/mcp.rs".into()],
                exclusive: true,
                attachment_id: Some(attachment.id.clone()),
                lease_seconds: Some(60),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let sqlite = dir.path().join("data/ice.sqlite");
        let conn = Connection::open(sqlite).unwrap();
        let (memory_id, extra_json) = conn
            .query_row(
                "SELECT memory_id, extra_json FROM continuity_items WHERE id = ?1",
                params![claim.id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .unwrap();
        let mut extra: serde_json::Value = serde_json::from_str(&extra_json).unwrap();
        extra["user"]["coordination"]["lease_expires_at"] =
            serde_json::json!((Utc::now() - Duration::seconds(5)).to_rfc3339());
        conn.execute(
            "UPDATE continuity_items SET extra_json = ?2 WHERE id = ?1",
            params![claim.id, extra.to_string()],
        )
        .unwrap();
        conn.execute(
            "UPDATE memory_items SET extra_json = ?2 WHERE id = ?1",
            params![memory_id, extra.to_string()],
        )
        .unwrap();

        let expired = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id.clone()),
                namespace: None,
                task_id: None,
                objective: "before heartbeat".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();
        assert!(expired.work_claims.is_empty());

        engine
            .heartbeat(HeartbeatInput {
                attachment_id: Some(attachment.id),
                agent_id: None,
                namespace: None,
                context_id: Some(context.id.clone()),
            })
            .unwrap();

        let renewed = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "after heartbeat".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();
        assert_eq!(renewed.work_claims.len(), 1);
    }

    #[test]
    fn coordination_anxiety_surfaces_in_organism_pressure() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "anxiety-pressure".into(),
                session_id: "session-1".into(),
                objective: "let the organism worry out loud".into(),
                selector: None,
                agent_id: Some("planner-a".into()),
                attachment_id: None,
            })
            .unwrap();
        let claim = engine
            .claim_work(ClaimWorkInput {
                context_id: context.id.clone(),
                agent_id: "planner-a".into(),
                title: "Own src/storage.rs".into(),
                body: "Metrics surgery in progress.".into(),
                scope: Scope::Project,
                resources: vec!["src/storage.rs".into()],
                exclusive: true,
                attachment_id: None,
                lease_seconds: Some(180),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .publish_coordination_signal(CoordinationSignalInput {
                context_id: context.id.clone(),
                agent_id: "therapist".into(),
                title: "Anxiety spike over storage surgery".into(),
                body: "Watch metrics cardinality before this lands.".into(),
                lane: CoordinationLane::Anxiety,
                target_agent_id: Some("planner-a".into()),
                target_projected_lane: Some(CoordinationProjectedLane {
                    projection_id: "repo:demo:main".into(),
                    projection_kind: "repo".into(),
                    label: "demo @ main".into(),
                    resource: Some("repo/demo/main".into()),
                    repo_root: Some("/tmp/demo".into()),
                    branch: Some("main".into()),
                    task_id: Some("anxiety-pressure".into()),
                }),
                claim_id: Some(claim.id),
                resource: Some("src/storage.rs".into()),
                severity: None,
                projection_ids: vec!["repo:demo:main".into()],
                projected_lanes: vec![CoordinationProjectedLane {
                    projection_id: "repo:demo:main".into(),
                    projection_kind: "repo".into(),
                    label: "demo @ main".into(),
                    resource: Some("repo/demo/main".into()),
                    repo_root: Some("/tmp/demo".into()),
                    branch: Some("main".into()),
                    task_id: Some("anxiety-pressure".into()),
                }],
                extra: serde_json::json!({}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect anxiety".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert_eq!(read.coordination_signals.len(), 1);
        assert_eq!(read.organism["coordination_signal_count"].as_u64(), Some(1));
        assert_eq!(read.organism["anxiety_signal_count"].as_u64(), Some(1));
        assert_eq!(
            read.organism["anxiety_pressure"][0]["target_agent_id"].as_str(),
            Some("planner-a")
        );
        assert_eq!(
            read.organism["anxiety_pressure"][0]["target_projected_lane"]["label"].as_str(),
            Some("demo @ main")
        );
        assert_eq!(
            read.organism["anxiety_pressure"][0]["projection_ids"]
                .as_array()
                .map(Vec::len),
            Some(1)
        );
        assert_eq!(
            read.organism["anxiety_pressure"][0]["projected_lanes"][0]["label"].as_str(),
            Some("demo @ main")
        );
    }

    #[test]
    fn therapy_resolves_anxiety_into_treated_memory() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "therapy-anxiety".into(),
                session_id: "session-1".into(),
                objective: "treat false coordination fear".into(),
                selector: None,
                agent_id: Some("planner-a".into()),
                attachment_id: None,
            })
            .unwrap();
        let signal = engine
            .publish_coordination_signal(CoordinationSignalInput {
                context_id: context.id.clone(),
                agent_id: "therapist".into(),
                title: "Anxiety over phantom risk".into(),
                body: "This might break invariants, but it is still unproven.".into(),
                lane: CoordinationLane::Anxiety,
                target_agent_id: Some("planner-a".into()),
                target_projected_lane: None,
                claim_id: None,
                resource: Some("src/continuity.rs".into()),
                severity: Some(CoordinationSeverity::Warn),
                projection_ids: Vec::new(),
                projected_lanes: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();
        engine
            .resolve_or_supersede(ResolveOrSupersedeInput {
                continuity_id: signal.id.clone(),
                actor_agent_id: "therapist".into(),
                new_status: ContinuityStatus::Resolved,
                supersedes_id: None,
                resolution_note: Some("Therapy proved the fear was not reality.".into()),
                extra: serde_json::json!({"therapy": true}),
            })
            .unwrap();

        let read = engine
            .read_context(ReadContextInput {
                context_id: Some(context.id),
                namespace: None,
                task_id: None,
                objective: "inspect treated anxiety".into(),
                token_budget: 256,
                selector: None,
                agent_id: Some("observer".into()),
                session_id: None,
                view_id: None,
                include_resolved: true,
                candidate_limit: 16,
            })
            .unwrap();

        assert!(read.coordination_signals.is_empty());
        assert_eq!(read.organism["anxiety_signal_count"].as_u64(), Some(0));
        assert_eq!(
            read.organism["retention_classes"]["treated_signal"].as_u64(),
            Some(1)
        );
    }

    #[test]
    fn resolve_updates_status_dimension() {
        let dir = tempdir().unwrap();
        let engine = Engine::open(dir.path()).unwrap();
        let context = engine
            .open_context(OpenContextInput {
                namespace: "demo".into(),
                task_id: "resolve-test".into(),
                session_id: "session-1".into(),
                objective: "track scar status".into(),
                selector: None,
                agent_id: Some("debugger-a".into()),
                attachment_id: None,
            })
            .unwrap();
        let incident = engine
            .mark_incident(ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: "debugger-a".into(),
                kind: ContinuityKind::Incident,
                title: "Regression".into(),
                body: "A known regression reappeared.".into(),
                scope: Scope::Project,
                status: Some(ContinuityStatus::Open),
                importance: Some(0.9),
                confidence: Some(0.8),
                salience: Some(0.9),
                layer: None,
                supports: Vec::new(),
                dimensions: Vec::new(),
                extra: serde_json::json!({}),
            })
            .unwrap();

        let resolved = engine
            .resolve_or_supersede(ResolveOrSupersedeInput {
                continuity_id: incident.id.clone(),
                actor_agent_id: "debugger-a".into(),
                new_status: ContinuityStatus::Resolved,
                supersedes_id: None,
                resolution_note: Some("fixed in kernel".into()),
                extra: serde_json::json!({}),
            })
            .unwrap();

        assert_eq!(resolved.status, ContinuityStatus::Resolved);
        let explained = engine
            .explain(ExplainTarget::ContinuityItem { id: incident.id })
            .unwrap();
        assert_eq!(
            explained["item"]["status"].as_str(),
            Some(ContinuityStatus::Resolved.as_str())
        );
    }

    #[test]
    fn continuity_kind_default_layer_maps_correctly() {
        assert_eq!(
            ContinuityKind::WorkingState.default_layer(),
            MemoryLayer::Hot
        );
        assert_eq!(ContinuityKind::WorkClaim.default_layer(), MemoryLayer::Hot);
        assert_eq!(ContinuityKind::Signal.default_layer(), MemoryLayer::Hot);
        assert_eq!(
            ContinuityKind::Derivation.default_layer(),
            MemoryLayer::Semantic
        );
        assert_eq!(ContinuityKind::Fact.default_layer(), MemoryLayer::Semantic);
        assert_eq!(
            ContinuityKind::Decision.default_layer(),
            MemoryLayer::Semantic
        );
        assert_eq!(
            ContinuityKind::Constraint.default_layer(),
            MemoryLayer::Semantic
        );
        assert_eq!(
            ContinuityKind::OperationalScar.default_layer(),
            MemoryLayer::Semantic
        );
        assert_eq!(
            ContinuityKind::Hypothesis.default_layer(),
            MemoryLayer::Episodic
        );
        assert_eq!(
            ContinuityKind::Incident.default_layer(),
            MemoryLayer::Episodic
        );
        assert_eq!(
            ContinuityKind::Outcome.default_layer(),
            MemoryLayer::Episodic
        );
        assert_eq!(
            ContinuityKind::Lesson.default_layer(),
            MemoryLayer::Episodic
        );
        assert_eq!(
            ContinuityKind::Summary.default_layer(),
            MemoryLayer::Summary
        );
    }

    #[test]
    fn continuity_kind_as_str_round_trips_through_serde() {
        let all_kinds = [
            ContinuityKind::WorkingState,
            ContinuityKind::WorkClaim,
            ContinuityKind::Derivation,
            ContinuityKind::Fact,
            ContinuityKind::Decision,
            ContinuityKind::Constraint,
            ContinuityKind::Hypothesis,
            ContinuityKind::Incident,
            ContinuityKind::OperationalScar,
            ContinuityKind::Outcome,
            ContinuityKind::Signal,
            ContinuityKind::Summary,
            ContinuityKind::Lesson,
        ];
        for kind in all_kinds {
            let json = serde_json::to_string(&kind).unwrap();
            let deserialized: ContinuityKind = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, kind);
            assert_eq!(
                json.trim_matches('"'),
                kind.as_str(),
                "as_str should match serde serialization for {:?}",
                kind
            );
        }
    }

    #[test]
    fn continuity_status_as_str_all_variants() {
        assert_eq!(ContinuityStatus::Open.as_str(), "open");
        assert_eq!(ContinuityStatus::Active.as_str(), "active");
        assert_eq!(ContinuityStatus::Resolved.as_str(), "resolved");
        assert_eq!(ContinuityStatus::Superseded.as_str(), "superseded");
        assert_eq!(ContinuityStatus::Rejected.as_str(), "rejected");
    }

    #[test]
    fn continuity_status_is_open_distinguishes_active_from_closed() {
        assert!(ContinuityStatus::Open.is_open());
        assert!(ContinuityStatus::Active.is_open());
        assert!(!ContinuityStatus::Resolved.is_open());
        assert!(!ContinuityStatus::Superseded.is_open());
        assert!(!ContinuityStatus::Rejected.is_open());
    }

    #[test]
    fn context_status_as_str_all_variants() {
        assert_eq!(ContextStatus::Open.as_str(), "open");
        assert_eq!(ContextStatus::Paused.as_str(), "paused");
        assert_eq!(ContextStatus::Closed.as_str(), "closed");
    }

    #[test]
    fn coordination_lane_as_str_all_variants() {
        assert_eq!(CoordinationLane::Anxiety.as_str(), "anxiety");
        assert_eq!(CoordinationLane::Review.as_str(), "review");
        assert_eq!(CoordinationLane::Warning.as_str(), "warning");
        assert_eq!(CoordinationLane::Backoff.as_str(), "backoff");
        assert_eq!(CoordinationLane::Coach.as_str(), "coach");
    }

    #[test]
    fn coordination_severity_as_str_all_variants() {
        assert_eq!(CoordinationSeverity::Info.as_str(), "info");
        assert_eq!(CoordinationSeverity::Warn.as_str(), "warn");
        assert_eq!(CoordinationSeverity::Block.as_str(), "block");
    }

    #[test]
    fn default_coordination_severity_maps_lanes() {
        assert_eq!(
            default_coordination_severity(CoordinationLane::Review),
            CoordinationSeverity::Info
        );
        assert_eq!(
            default_coordination_severity(CoordinationLane::Coach),
            CoordinationSeverity::Info
        );
        assert_eq!(
            default_coordination_severity(CoordinationLane::Warning),
            CoordinationSeverity::Warn
        );
        assert_eq!(
            default_coordination_severity(CoordinationLane::Anxiety),
            CoordinationSeverity::Warn
        );
        assert_eq!(
            default_coordination_severity(CoordinationLane::Backoff),
            CoordinationSeverity::Block
        );
    }

    #[test]
    fn normalize_work_claim_resources_deduplicates_and_sorts() {
        let input = vec![
            "  SRC/main.rs ".to_string(),
            "src/lib.rs".to_string(),
            "SRC/MAIN.RS".to_string(),
            "".to_string(),
            "  ".to_string(),
        ];
        let result = normalize_work_claim_resources(&input);
        assert_eq!(result, vec!["src/lib.rs", "src/main.rs"]);
    }

    #[test]
    fn normalize_work_claim_resources_empty_input() {
        let result = normalize_work_claim_resources(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn work_claim_key_uses_title_when_no_resources() {
        let key = work_claim_key("ctx-1", Scope::Project, "agent-a", "Fix the bug!", &[]);
        assert!(key.starts_with("ctx-1:"));
        assert!(key.contains("agent-a"));
        assert!(key.contains("fix-the-bug"));
        assert!(!key.contains("!"));
    }

    #[test]
    fn work_claim_key_uses_resources_when_present() {
        let resources = vec!["src/main.rs".to_string(), "src/lib.rs".to_string()];
        let key = work_claim_key("ctx-1", Scope::Shared, "agent-a", "title", &resources);
        assert!(key.contains("src/main.rs|src/lib.rs"));
        assert!(!key.contains("title"));
    }

    #[test]
    fn merge_work_claim_extra_into_null() {
        let coordination = WorkClaimCoordination {
            claim_key: "key".into(),
            resources: vec!["file.rs".into()],
            exclusive: true,
            ..Default::default()
        };
        let result = merge_work_claim_extra(serde_json::Value::Null, &coordination);
        assert!(result["coordination"]["claim_key"].as_str() == Some("key"));
        assert!(result["coordination"]["exclusive"].as_bool() == Some(true));
    }

    #[test]
    fn merge_work_claim_extra_into_object() {
        let coordination = WorkClaimCoordination::default();
        let existing = serde_json::json!({"user_data": 42});
        let result = merge_work_claim_extra(existing, &coordination);
        assert_eq!(result["user_data"].as_u64(), Some(42));
        assert!(result["coordination"].is_object());
    }

    #[test]
    fn merge_work_claim_extra_into_non_object() {
        let coordination = WorkClaimCoordination::default();
        let result = merge_work_claim_extra(serde_json::json!("string payload"), &coordination);
        assert_eq!(result["payload"].as_str(), Some("string payload"));
        assert!(result["coordination"].is_object());
    }

    #[test]
    fn work_claim_coordination_from_extra_finds_nested_and_top_level() {
        let with_user = serde_json::json!({
            "user": { "coordination": { "claim_key": "from-user", "resources": [], "exclusive": false } }
        });
        let result = work_claim_coordination_from_extra(&with_user).unwrap();
        assert_eq!(result.claim_key, "from-user");

        let top_level = serde_json::json!({
            "coordination": { "claim_key": "from-top", "resources": [], "exclusive": true }
        });
        let result = work_claim_coordination_from_extra(&top_level).unwrap();
        assert_eq!(result.claim_key, "from-top");

        let empty = serde_json::json!({});
        assert!(work_claim_coordination_from_extra(&empty).is_none());
    }

    #[test]
    fn merge_coordination_signal_extra_into_null() {
        let result = merge_coordination_signal_extra(
            serde_json::Value::Null,
            CoordinationSignalExtraInput {
                lane: CoordinationLane::Anxiety,
                severity: CoordinationSeverity::Warn,
                target_agent_id: Some("target-agent".into()),
                target_projected_lane: None,
                claim_id: Some("claim-1".into()),
                resource: Some("src/file.rs".into()),
                projection_ids: vec!["proj-1".into()],
                projected_lanes: Vec::new(),
            },
        );
        assert_eq!(
            result["coordination_signal"]["lane"].as_str(),
            Some("anxiety")
        );
        assert_eq!(
            result["coordination_signal"]["severity"].as_str(),
            Some("warn")
        );
        assert_eq!(
            result["coordination_signal"]["target_agent_id"].as_str(),
            Some("target-agent")
        );
        assert_eq!(
            result["coordination_signal"]["claim_id"].as_str(),
            Some("claim-1")
        );
    }

    #[test]
    fn merge_coordination_signal_extra_into_object_preserves_fields() {
        let existing = serde_json::json!({"metadata": "keep"});
        let result = merge_coordination_signal_extra(
            existing,
            CoordinationSignalExtraInput {
                lane: CoordinationLane::Review,
                severity: CoordinationSeverity::Info,
                target_agent_id: None,
                target_projected_lane: None,
                claim_id: None,
                resource: None,
                projection_ids: Vec::new(),
                projected_lanes: Vec::new(),
            },
        );
        assert_eq!(result["metadata"].as_str(), Some("keep"));
        assert_eq!(
            result["coordination_signal"]["lane"].as_str(),
            Some("review")
        );
    }

    #[test]
    fn merge_coordination_signal_extra_wraps_non_object() {
        let result = merge_coordination_signal_extra(
            serde_json::json!(42),
            CoordinationSignalExtraInput {
                lane: CoordinationLane::Backoff,
                severity: CoordinationSeverity::Block,
                target_agent_id: None,
                target_projected_lane: None,
                claim_id: None,
                resource: None,
                projection_ids: Vec::new(),
                projected_lanes: Vec::new(),
            },
        );
        assert_eq!(result["payload"].as_u64(), Some(42));
        assert_eq!(
            result["coordination_signal"]["lane"].as_str(),
            Some("backoff")
        );
    }

    #[test]
    fn coordination_signal_from_extra_finds_nested_and_top_level() {
        let with_user = serde_json::json!({
            "user": { "coordination_signal": { "lane": "anxiety", "severity": "warn" } }
        });
        let result = coordination_signal_from_extra(&with_user).unwrap();
        assert_eq!(result.lane, "anxiety");

        let top_level = serde_json::json!({
            "coordination_signal": { "lane": "review", "severity": "info" }
        });
        let result = coordination_signal_from_extra(&top_level).unwrap();
        assert_eq!(result.lane, "review");

        let empty = serde_json::json!({});
        assert!(coordination_signal_from_extra(&empty).is_none());
    }

    fn make_test_item(
        id: &str,
        kind: ContinuityKind,
        status: ContinuityStatus,
        extra: serde_json::Value,
    ) -> ContinuityItemRecord {
        ContinuityItemRecord {
            id: id.into(),
            memory_id: format!("mem-{id}"),
            context_id: "ctx-1".into(),
            namespace: "ns".into(),
            task_id: "task".into(),
            author_agent_id: "agent-a".into(),
            kind,
            scope: Scope::Project,
            status,
            title: format!("Item {id}"),
            body: "body".into(),
            importance: 0.9,
            confidence: 0.9,
            salience: 0.9,
            retention: ContinuityRetentionState {
                class: kind.as_str().to_string(),
                age_hours: 1.0,
                half_life_hours: 36.0,
                floor: 0.03,
                decay_multiplier: 1.0,
                effective_salience: 0.9,
            },
            created_at: Utc::now(),
            updated_at: Utc::now(),
            supersedes_id: None,
            resolved_at: None,
            supports: Vec::new(),
            practice_state: None,
            extra,
        }
    }

    #[test]
    fn filter_kind_returns_only_matching() {
        let items = vec![
            make_test_item(
                "1",
                ContinuityKind::Fact,
                ContinuityStatus::Open,
                serde_json::json!({}),
            ),
            make_test_item(
                "2",
                ContinuityKind::Decision,
                ContinuityStatus::Open,
                serde_json::json!({}),
            ),
            make_test_item(
                "3",
                ContinuityKind::Fact,
                ContinuityStatus::Active,
                serde_json::json!({}),
            ),
        ];
        let facts = filter_kind(&items, ContinuityKind::Fact);
        assert_eq!(facts.len(), 2);
        assert!(facts.iter().all(|item| item.kind == ContinuityKind::Fact));

        let empty = filter_kind(&items, ContinuityKind::Lesson);
        assert!(empty.is_empty());
    }

    #[test]
    fn work_claim_is_live_requires_kind_and_open_status() {
        let now = Utc::now();
        let non_claim = make_test_item(
            "1",
            ContinuityKind::Fact,
            ContinuityStatus::Open,
            serde_json::json!({}),
        );
        assert!(!work_claim_is_live(&non_claim, now));

        let resolved_claim = make_test_item(
            "2",
            ContinuityKind::WorkClaim,
            ContinuityStatus::Resolved,
            serde_json::json!({}),
        );
        assert!(!work_claim_is_live(&resolved_claim, now));
    }

    #[test]
    fn work_claim_is_live_with_no_coordination_defaults_to_true() {
        let now = Utc::now();
        let claim = make_test_item(
            "1",
            ContinuityKind::WorkClaim,
            ContinuityStatus::Open,
            serde_json::json!({}),
        );
        assert!(work_claim_is_live(&claim, now));
    }

    #[test]
    fn work_claim_is_live_respects_lease_expiry() {
        let now = Utc::now();
        let future = now + Duration::hours(1);
        let past = now - Duration::hours(1);

        let live_claim = make_test_item(
            "1",
            ContinuityKind::WorkClaim,
            ContinuityStatus::Active,
            serde_json::json!({
                "coordination": {
                    "claim_key": "k",
                    "resources": [],
                    "exclusive": false,
                    "lease_expires_at": future.to_rfc3339()
                }
            }),
        );
        assert!(work_claim_is_live(&live_claim, now));

        let expired_claim = make_test_item(
            "2",
            ContinuityKind::WorkClaim,
            ContinuityStatus::Active,
            serde_json::json!({
                "coordination": {
                    "claim_key": "k",
                    "resources": [],
                    "exclusive": false,
                    "lease_expires_at": past.to_rfc3339()
                }
            }),
        );
        assert!(!work_claim_is_live(&expired_claim, now));
    }

    #[test]
    fn counts_as_open_thread_filters_resolved_and_expired_claims() {
        let now = Utc::now();
        let resolved = make_test_item(
            "1",
            ContinuityKind::Fact,
            ContinuityStatus::Resolved,
            serde_json::json!({}),
        );
        assert!(!counts_as_open_thread(&resolved, now));

        let open_fact = make_test_item(
            "2",
            ContinuityKind::Fact,
            ContinuityStatus::Open,
            serde_json::json!({}),
        );
        assert!(counts_as_open_thread(&open_fact, now));

        let open_claim_no_coord = make_test_item(
            "3",
            ContinuityKind::WorkClaim,
            ContinuityStatus::Open,
            serde_json::json!({}),
        );
        assert!(counts_as_open_thread(&open_claim_no_coord, now));

        let past = now - Duration::hours(1);
        let expired_claim = make_test_item(
            "4",
            ContinuityKind::WorkClaim,
            ContinuityStatus::Open,
            serde_json::json!({
                "coordination": {
                    "claim_key": "k",
                    "resources": [],
                    "exclusive": false,
                    "lease_expires_at": past.to_rfc3339()
                }
            }),
        );
        assert!(!counts_as_open_thread(&expired_claim, now));

        let mut stale_open_decision = make_test_item(
            "5",
            ContinuityKind::Decision,
            ContinuityStatus::Open,
            serde_json::json!({}),
        );
        stale_open_decision.updated_at = now - Duration::days(12);
        stale_open_decision.retention.half_life_hours = 72.0;
        assert!(!counts_as_open_thread(&stale_open_decision, now));

        let mut fresh_open_decision = make_test_item(
            "6",
            ContinuityKind::Decision,
            ContinuityStatus::Open,
            serde_json::json!({}),
        );
        fresh_open_decision.updated_at = now - Duration::hours(3);
        fresh_open_decision.retention.half_life_hours = 72.0;
        assert!(counts_as_open_thread(&fresh_open_decision, now));
    }

    #[test]
    fn work_claims_conflict_requires_exclusive_and_shared_resource() {
        let now = Utc::now();
        let future = now + Duration::hours(1);

        let make_claim = |id: &str, resources: Vec<&str>, exclusive: bool| {
            make_test_item(
                id,
                ContinuityKind::WorkClaim,
                ContinuityStatus::Active,
                serde_json::json!({
                    "coordination": {
                        "claim_key": format!("k-{id}"),
                        "resources": resources,
                        "exclusive": exclusive,
                        "lease_expires_at": future.to_rfc3339()
                    }
                }),
            )
        };

        let claim_a = make_claim("a", vec!["file.rs"], true);
        let claim_b = make_claim("b", vec!["file.rs"], false);
        assert!(work_claims_conflict(&claim_a, &claim_b, now));

        let claim_c = make_claim("c", vec!["file.rs"], false);
        let claim_d = make_claim("d", vec!["file.rs"], false);
        assert!(!work_claims_conflict(&claim_c, &claim_d, now));

        let claim_e = make_claim("e", vec!["a.rs"], true);
        let claim_f = make_claim("f", vec!["b.rs"], true);
        assert!(!work_claims_conflict(&claim_e, &claim_f, now));

        assert!(!work_claims_conflict(&claim_a, &claim_a, now));
    }

    #[test]
    fn coordination_signal_requires_signal_kind_and_open_status() {
        let signal_item = make_test_item(
            "1",
            ContinuityKind::Signal,
            ContinuityStatus::Active,
            serde_json::json!({
                "coordination_signal": {
                    "lane": "anxiety",
                    "severity": "warn"
                }
            }),
        );
        assert!(coordination_signal(&signal_item).is_some());

        let resolved_signal = make_test_item(
            "2",
            ContinuityKind::Signal,
            ContinuityStatus::Resolved,
            serde_json::json!({
                "coordination_signal": { "lane": "anxiety", "severity": "warn" }
            }),
        );
        assert!(coordination_signal(&resolved_signal).is_none());

        let wrong_kind = make_test_item(
            "3",
            ContinuityKind::Fact,
            ContinuityStatus::Open,
            serde_json::json!({
                "coordination_signal": { "lane": "anxiety", "severity": "warn" }
            }),
        );
        assert!(coordination_signal(&wrong_kind).is_none());
    }

    #[test]
    fn trim_text_within_limit_returns_trimmed() {
        assert_eq!(trim_text("  hello  ", 100), "hello");
    }

    #[test]
    fn trim_text_beyond_limit_truncates_with_ellipsis() {
        let long = "a".repeat(300);
        let trimmed = trim_text(&long, 10);
        assert!(trimmed.ends_with("..."));
        assert!(trimmed.chars().count() <= 10);
    }

    #[test]
    fn augment_dimensions_avoids_duplicates() {
        let base = vec![DimensionValue {
            key: "k1".into(),
            value: "v1".into(),
            weight: DEFAULT_DIMENSION_WEIGHT,
        }];
        let extra = vec![
            DimensionValue {
                key: "k1".into(),
                value: "v1".into(),
                weight: DEFAULT_DIMENSION_WEIGHT,
            },
            DimensionValue {
                key: "k2".into(),
                value: "v2".into(),
                weight: 50,
            },
        ];
        let merged = augment_dimensions(base, extra);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].key, "k1");
        assert_eq!(merged[1].key, "k2");
    }

    #[test]
    fn default_functions_return_expected_values() {
        assert_eq!(super::schema::default_candidate_limit(), 24);
        assert_eq!(super::schema::default_token_budget(), 384);
        assert_eq!(
            super::schema::default_snapshot_resolution(),
            SnapshotResolution::Medium
        );
        assert_eq!(default_work_claim_lease_seconds(), 180);
    }

    #[test]
    fn if_empty_then_uses_fallback_for_blank() {
        assert_eq!("".to_string().if_empty_then("fallback"), "fallback");
        assert_eq!("  ".to_string().if_empty_then("fallback"), "fallback");
        assert_eq!("value".to_string().if_empty_then("fallback"), "value");
    }

    #[test]
    fn continuity_kind_serde_roundtrip() {
        let kind = ContinuityKind::OperationalScar;
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(json, "\"operational_scar\"");
        let back: ContinuityKind = serde_json::from_str(&json).unwrap();
        assert_eq!(back, kind);
    }

    #[test]
    fn continuity_status_serde_roundtrip() {
        for status in [
            ContinuityStatus::Open,
            ContinuityStatus::Active,
            ContinuityStatus::Resolved,
            ContinuityStatus::Superseded,
            ContinuityStatus::Rejected,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: ContinuityStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status);
            assert_eq!(json.trim_matches('"'), status.as_str());
        }
    }

    #[test]
    fn coordination_lane_serde_roundtrip() {
        for lane in [
            CoordinationLane::Anxiety,
            CoordinationLane::Review,
            CoordinationLane::Warning,
            CoordinationLane::Backoff,
            CoordinationLane::Coach,
        ] {
            let json = serde_json::to_string(&lane).unwrap();
            let back: CoordinationLane = serde_json::from_str(&json).unwrap();
            assert_eq!(back, lane);
        }
    }

    #[test]
    fn coordination_severity_serde_roundtrip() {
        for severity in [
            CoordinationSeverity::Info,
            CoordinationSeverity::Warn,
            CoordinationSeverity::Block,
        ] {
            let json = serde_json::to_string(&severity).unwrap();
            let back: CoordinationSeverity = serde_json::from_str(&json).unwrap();
            assert_eq!(back, severity);
        }
    }
}
