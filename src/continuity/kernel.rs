use anyhow::Result;
use chrono::Utc;
use uuid::Uuid;

use crate::dispatch;
use crate::engine::Engine;
use crate::model::{
    DimensionValue, EventInput, EventKind, IngestManifest, MemoryLayer, QueryInput, ReplayRow,
    Scope, Selector, SubscriptionInput, SubscriptionRecord, ViewInput, ViewOp,
};
use crate::query::build_context_pack;

use super::helpers::*;
use super::interface::UnifiedContinuityInterface;
use super::schema::*;
use super::types::*;

pub type SharedContinuityKernel = Engine;

impl UnifiedContinuityInterface for Engine {
    fn identify_machine(&self) -> Result<MachineProfile> {
        self.with_storage(|storage, _| storage.machine_profile())
    }

    fn attach_agent(&self, input: AttachAgentInput) -> Result<AgentAttachmentRecord> {
        self.with_storage(|storage, _| {
            let mut input = input;
            input.namespace = resolve_namespace(storage, Some(input.namespace))?
                .expect("namespace should remain present");
            storage.attach_agent(input)
        })
    }

    fn upsert_agent_badge(&self, input: UpsertAgentBadgeInput) -> Result<AgentBadgeRecord> {
        self.with_storage(|storage, _| {
            let mut input = input;
            input.namespace = resolve_namespace(storage, input.namespace)?;
            storage.upsert_agent_badge(input)
        })
    }

    fn heartbeat(&self, input: HeartbeatInput) -> Result<AgentAttachmentRecord> {
        self.with_storage(|storage, _| {
            let mut input = input;
            input.namespace = resolve_namespace(storage, input.namespace)?;
            storage.heartbeat_attachment(input)
        })
    }

    fn open_context(&self, input: OpenContextInput) -> Result<ContextRecord> {
        self.with_storage(|storage, _| {
            let mut input = input;
            input.namespace = resolve_namespace(storage, Some(input.namespace))?
                .expect("namespace should remain present");
            input.selector = resolve_selector_namespace(storage, input.selector)?;
            storage.open_context(input)
        })
    }

    fn read_context(&self, input: ReadContextInput) -> Result<ContextRead> {
        self.with_storage(|storage, _| {
            let namespace = resolve_namespace(storage, input.namespace.clone())?;
            let context = storage.resolve_context(
                input.context_id.as_deref(),
                namespace.as_deref(),
                input.task_id.as_deref(),
            )?;
            let selector = merge_context_selector(
                &context,
                resolve_selector_namespace(storage, input.selector.clone())?,
            );
            let continuity = storage.list_continuity_items(&context.id, input.include_resolved)?;
            let recall = storage.recall_continuity(
                &context.id,
                &input.objective,
                input.include_resolved,
                input.candidate_limit.min(8).max(4),
            )?;
            let pack = build_context_pack(
                storage,
                QueryInput {
                    agent_id: input.agent_id.clone(),
                    session_id: input
                        .session_id
                        .clone()
                        .or_else(|| Some(context.session_id.clone())),
                    task_id: Some(context.task_id.clone()),
                    namespace: Some(context.namespace.clone()),
                    objective: Some(input.objective.clone()),
                    selector: Some(selector.clone()),
                    view_id: input.view_id.clone(),
                    query_text: input.objective.clone(),
                    budget_tokens: input.token_budget,
                    candidate_limit: input.candidate_limit,
                },
            )?;
            if let Some(agent_id) = input.agent_id.as_deref() {
                storage.touch_active_attachment(
                    agent_id,
                    Some(context.namespace.as_str()),
                    Some(context.id.as_str()),
                )?;
            }
            let latest_snapshot_id = storage.latest_snapshot_id(&context.id)?;
            let pack_item_count = pack.items.len();
            let now = Utc::now();
            let agent_badges =
                storage.list_agent_badges(Some(context.namespace.as_str()), None)?;
            let mut lane_projections =
                storage.list_lane_projections(Some(context.namespace.as_str()), None)?;
            let dispatch_state =
                dispatch::organism_snapshot(&storage.config.root, Some(context.namespace.as_str()));
            merge_dispatch_worker_lane_projections(
                &mut lane_projections,
                &dispatch_state,
                context.namespace.as_str(),
            );
            merge_dispatch_assignment_pressure(
                &mut lane_projections,
                &dispatch_state,
                context.namespace.as_str(),
            );
            let mut organism =
                organism_state(&continuity, now, &agent_badges, &lane_projections);
            if let Some(object) = organism.as_object_mut() {
                object.insert(
                    "agent_badges".to_string(),
                    serde_json::to_value(&agent_badges)?,
                );
                object.insert(
                    "lane_projections".to_string(),
                    serde_json::to_value(&lane_projections)?,
                );
                object.insert(
                    "dispatch".to_string(),
                    serde_json::to_value(&dispatch_state)?,
                );
            }
            Ok(ContextRead {
                context,
                objective: input.objective,
                pack,
                latest_snapshot_id,
                organism,
                recall: recall.clone(),
                working_state: filter_kind(&continuity, ContinuityKind::WorkingState),
                work_claims: continuity
                    .iter()
                    .filter(|item| work_claim_is_live(item, now))
                    .cloned()
                    .collect(),
                coordination_signals: continuity
                    .iter()
                    .filter(|item| coordination_signal(item).is_some())
                    .cloned()
                    .collect(),
                decisions: filter_kind(&continuity, ContinuityKind::Decision),
                constraints: filter_kind(&continuity, ContinuityKind::Constraint),
                hypotheses: filter_kind(&continuity, ContinuityKind::Hypothesis),
                incidents: filter_kind(&continuity, ContinuityKind::Incident),
                operational_scars: filter_kind(&continuity, ContinuityKind::OperationalScar),
                outcomes: filter_kind(&continuity, ContinuityKind::Outcome),
                signals: filter_kind(&continuity, ContinuityKind::Signal),
                open_threads: continuity
                    .iter()
                    .filter(|item| counts_as_open_thread(item, now))
                    .cloned()
                    .collect(),
                rationale: serde_json::json!({
                    "selector": selector,
                    "continuity_count": continuity.len(),
                    "pack_item_count": pack_item_count,
                    "continuity_recall_count": recall.items.len(),
                    "continuity_recall_timings_ms": recall.timings_ms,
                    "continuity_recall_compiler": recall.compiler,
                    "continuity_recall_top_why": recall.items.first().map(|item| item.why.clone()).unwrap_or_default(),
                }),
            })
        })
    }

    fn recall(&self, input: RecallInput) -> Result<ContinuityRecall> {
        self.with_storage(|storage, _| {
            let namespace = resolve_namespace(storage, input.namespace.clone())?;
            let context = storage.resolve_context(
                input.context_id.as_deref(),
                namespace.as_deref(),
                input.task_id.as_deref(),
            )?;
            storage.recall_continuity(
                &context.id,
                &input.objective,
                input.include_resolved,
                input.candidate_limit,
            )
        })
    }

    fn write_events(&self, inputs: Vec<WriteEventInput>) -> Result<Vec<IngestManifest>> {
        self.with_storage_mut(|storage, telemetry| {
            let mut manifests = Vec::new();
            for mut input in inputs {
                if let Some(context_id) = &input.context_id {
                    let context = storage.get_context(context_id)?;
                    inject_context(&mut input.event, &context);
                } else {
                    input.event.namespace =
                        resolve_namespace(storage, input.event.namespace.clone())?;
                }
                manifests.push(storage.ingest(input.event, telemetry)?);
            }
            Ok(manifests)
        })
    }

    fn write_derivations(
        &self,
        inputs: Vec<ContinuityItemInput>,
    ) -> Result<Vec<ContinuityItemRecord>> {
        self.with_storage(|storage, _| {
            inputs
                .into_iter()
                .map(|input| storage.persist_continuity_item(input))
                .collect()
        })
    }

    fn claim_work(&self, input: ClaimWorkInput) -> Result<ContinuityItemRecord> {
        self.with_storage(|storage, _| storage.claim_work(input))
    }

    fn publish_coordination_signal(
        &self,
        input: CoordinationSignalInput,
    ) -> Result<ContinuityItemRecord> {
        let severity = input
            .severity
            .unwrap_or_else(|| default_coordination_severity(input.lane));
        let mut dimensions = vec![
            DimensionValue {
                key: "signal.lane".to_string(),
                value: input.lane.as_str().to_string(),
                weight: DEFAULT_DIMENSION_WEIGHT,
            },
            DimensionValue {
                key: "signal.severity".to_string(),
                value: severity.as_str().to_string(),
                weight: DEFAULT_DIMENSION_WEIGHT,
            },
        ];
        if let Some(target_agent_id) = &input.target_agent_id {
            dimensions.push(DimensionValue {
                key: "signal.target_agent".to_string(),
                value: target_agent_id.clone(),
                weight: DEFAULT_DIMENSION_WEIGHT,
            });
        }
        if let Some(claim_id) = &input.claim_id {
            dimensions.push(DimensionValue {
                key: "claim.id".to_string(),
                value: claim_id.clone(),
                weight: DEFAULT_DIMENSION_WEIGHT,
            });
        }
        if let Some(resource) = &input.resource {
            dimensions.push(DimensionValue {
                key: "claim.resource".to_string(),
                value: resource.clone(),
                weight: DEFAULT_DIMENSION_WEIGHT,
            });
        }
        self.publish_signal(SignalInput {
            context_id: input.context_id,
            agent_id: input.agent_id,
            title: input.title,
            body: input.body,
            dimensions,
            extra: merge_coordination_signal_extra(
                input.extra,
                input.lane,
                severity,
                input.target_agent_id,
                input.target_projected_lane,
                input.claim_id,
                input.resource,
                input.projection_ids,
                input.projected_lanes,
            ),
        })
    }

    fn publish_signal(&self, input: SignalInput) -> Result<ContinuityItemRecord> {
        self.with_storage(|storage, _| {
            storage.persist_continuity_item(ContinuityItemInput {
                context_id: input.context_id,
                author_agent_id: input.agent_id,
                kind: ContinuityKind::Signal,
                title: input.title,
                body: input.body,
                scope: Scope::Shared,
                status: Some(ContinuityStatus::Active),
                importance: Some(0.9),
                confidence: Some(0.9),
                salience: Some(0.95),
                layer: Some(MemoryLayer::Hot),
                supports: Vec::new(),
                dimensions: input.dimensions,
                extra: input.extra,
            })
        })
    }

    fn subscribe(&self, input: SubscriptionInput) -> Result<SubscriptionRecord> {
        self.with_storage(|storage, _| storage.create_subscription(input))
    }

    fn handoff(&self, input: ContinuityHandoffInput) -> Result<ContinuityHandoffRecord> {
        let machine = self.identify_machine()?;
        let namespace = match input.namespace.clone() {
            Some(namespace) => {
                self.with_storage(|storage, _| resolve_namespace(storage, Some(namespace)))?
            }
            None => Some(machine.namespace.clone()),
        };
        let task_id = input
            .task_id
            .clone()
            .unwrap_or_else(|| machine.default_task_id.clone());
        let selector = self.with_storage(|storage, _| {
            resolve_selector_namespace(storage, input.selector.clone())
        })?;
        let context = self.open_context(OpenContextInput {
            namespace: namespace
                .clone()
                .expect("handoff namespace should stay present"),
            task_id,
            session_id: format!("handoff-{}", Uuid::now_v7()),
            objective: input.objective.clone(),
            selector: selector.clone(),
            agent_id: Some(input.from_agent_id.clone()),
            attachment_id: None,
        })?;
        let snapshot = self.snapshot(SnapshotInput {
            context_id: Some(
                input
                    .context_id
                    .clone()
                    .unwrap_or_else(|| context.id.clone()),
            ),
            namespace: None,
            task_id: None,
            objective: Some(input.objective.clone()),
            selector: selector.clone(),
            resolution: input.resolution,
            token_budget: input.token_budget,
            candidate_limit: input.candidate_limit,
            owner_agent_id: Some(input.from_agent_id.clone()),
        })?;
        let context_read = self.read_context(ReadContextInput {
            context_id: Some(snapshot.context_id.clone()),
            namespace: None,
            task_id: None,
            objective: input.objective.clone(),
            token_budget: input.token_budget,
            selector,
            agent_id: Some(input.to_agent_id.clone()),
            session_id: None,
            view_id: Some(snapshot.view_id.clone()),
            include_resolved: false,
            candidate_limit: input.candidate_limit,
        })?;
        let proof = compile_handoff_proof(&context_read);
        let handoff = self.with_storage(|storage, _| {
            let record = crate::model::HandoffRecord {
                id: format!("handoff:{}", Uuid::now_v7()),
                created_at: Utc::now(),
                from_agent_id: input.from_agent_id.clone(),
                to_agent_id: input.to_agent_id.clone(),
                reason: input.reason.clone(),
                view_id: snapshot.view_id.clone(),
                pack_id: snapshot.pack_id.clone(),
                conflict_count: 0,
                manifest_path: storage
                    .paths
                    .debug_dir
                    .join("handoffs")
                    .join(format!("{}.json", Uuid::now_v7()))
                    .display()
                    .to_string(),
            };
            let pack_manifest = storage.explain_context_pack(&snapshot.pack_id)?;
            let snapshot_manifest = storage.explain_snapshot(&snapshot.id)?;
            storage.persist_handoff(
                &record,
                &serde_json::json!({
                    "handoff": record.clone(),
                    "reason": input.reason,
                    "snapshot": snapshot_manifest,
                    "pack": pack_manifest,
                    "context": context_read,
                    "proof": proof,
                }),
                &input.objective,
            )?;
            storage.get_handoff(&record.id)
        })?;
        Ok(ContinuityHandoffRecord {
            handoff,
            snapshot,
            context: context_read,
            proof,
        })
    }

    fn snapshot(&self, input: SnapshotInput) -> Result<SnapshotRecord> {
        self.with_storage(|storage, _| {
            let namespace = resolve_namespace(storage, input.namespace.clone())?;
            let context = storage.resolve_context(
                input.context_id.as_deref(),
                namespace.as_deref(),
                input.task_id.as_deref(),
            )?;
            let objective = input
                .objective
                .clone()
                .unwrap_or_else(|| context.objective.clone());
            let selector = merge_context_selector(
                &context,
                resolve_selector_namespace(storage, input.selector.clone())?,
            );
            let view = storage.materialize_view(ViewInput {
                op: ViewOp::Snapshot,
                owner_agent_id: input.owner_agent_id.clone(),
                namespace: Some(context.namespace.clone()),
                objective: Some(objective.clone()),
                selectors: vec![selector.clone()],
                source_view_ids: Vec::new(),
                resolution: Some(input.resolution),
                limit: Some(input.candidate_limit.max(1)),
            })?;
            let pack = build_context_pack(
                storage,
                QueryInput {
                    agent_id: input.owner_agent_id.clone(),
                    session_id: Some(context.session_id.clone()),
                    task_id: Some(context.task_id.clone()),
                    namespace: Some(context.namespace.clone()),
                    objective: Some(objective.clone()),
                    selector: Some(selector.clone()),
                    view_id: Some(view.id.clone()),
                    query_text: objective.clone(),
                    budget_tokens: input.token_budget,
                    candidate_limit: input.candidate_limit,
                },
            )?;
            storage.persist_snapshot(
                &context,
                selector,
                objective,
                input.resolution,
                &view.id,
                &pack.id,
            )
        })
    }

    fn resume(&self, input: ResumeInput) -> Result<ResumeRecord> {
        let snapshot = if let Some(snapshot_id) = &input.snapshot_id {
            Some(self.with_storage(|storage, _| storage.get_snapshot(snapshot_id))?)
        } else {
            None
        };
        let namespace =
            self.with_storage(|storage, _| resolve_namespace(storage, input.namespace.clone()))?;
        let context = self.read_context(ReadContextInput {
            context_id: input
                .context_id
                .clone()
                .or_else(|| snapshot.as_ref().map(|item| item.context_id.clone())),
            namespace,
            task_id: input.task_id.clone(),
            objective: input.objective,
            token_budget: input.token_budget,
            selector: snapshot.as_ref().map(|item| item.selector.clone()),
            agent_id: input.agent_id,
            session_id: None,
            view_id: snapshot.as_ref().map(|item| item.view_id.clone()),
            include_resolved: false,
            candidate_limit: input.candidate_limit,
        })?;
        Ok(ResumeRecord { snapshot, context })
    }

    fn explain(&self, target: ExplainTarget) -> Result<serde_json::Value> {
        self.with_storage(|storage, _| match target {
            ExplainTarget::Context { id } => {
                Ok(serde_json::to_value(storage.explain_context(&id)?)?)
            }
            ExplainTarget::ContinuityItem { id } => {
                Ok(serde_json::to_value(storage.explain_continuity_item(&id)?)?)
            }
            ExplainTarget::Snapshot { id } => Ok(storage.explain_snapshot(&id)?),
            ExplainTarget::Handoff { id } => storage.explain_handoff(&id),
            ExplainTarget::Pack { id } => {
                Ok(serde_json::to_value(storage.explain_context_pack(&id)?)?)
            }
            ExplainTarget::View { id } => Ok(serde_json::to_value(storage.explain_view(&id)?)?),
        })
    }

    fn replay_selector(&self, selector: Selector, limit: usize) -> Result<Vec<ReplayRow>> {
        self.with_storage(|storage, telemetry| {
            storage.replay_by_selector(telemetry, &selector, limit)
        })
    }

    fn record_outcome(&self, input: OutcomeInput) -> Result<ContinuityItemRecord> {
        self.with_storage(|storage, _| storage.record_outcome(input))
    }

    fn mark_decision(&self, mut input: ContinuityItemInput) -> Result<ContinuityItemRecord> {
        input.kind = ContinuityKind::Decision;
        self.with_storage(|storage, _| storage.persist_continuity_item(input))
    }

    fn mark_constraint(&self, mut input: ContinuityItemInput) -> Result<ContinuityItemRecord> {
        input.kind = ContinuityKind::Constraint;
        self.with_storage(|storage, _| storage.persist_continuity_item(input))
    }

    fn mark_hypothesis(&self, mut input: ContinuityItemInput) -> Result<ContinuityItemRecord> {
        input.kind = ContinuityKind::Hypothesis;
        self.with_storage(|storage, _| storage.persist_continuity_item(input))
    }

    fn mark_incident(&self, mut input: ContinuityItemInput) -> Result<ContinuityItemRecord> {
        input.kind = ContinuityKind::Incident;
        self.with_storage(|storage, _| storage.persist_continuity_item(input))
    }

    fn mark_operational_scar(
        &self,
        mut input: ContinuityItemInput,
    ) -> Result<ContinuityItemRecord> {
        input.kind = ContinuityKind::OperationalScar;
        input.salience = Some(input.salience.unwrap_or(0.95));
        input.importance = Some(input.importance.unwrap_or(0.95));
        self.with_storage(|storage, _| storage.persist_continuity_item(input))
    }

    fn resolve_or_supersede(&self, input: ResolveOrSupersedeInput) -> Result<ContinuityItemRecord> {
        self.with_storage(|storage, _| storage.resolve_continuity_item(input))
    }

    fn emit_telemetry(&self, input: TelemetryEventInput) -> Result<serde_json::Value> {
        let namespace =
            self.with_storage(|storage, _| resolve_namespace(storage, input.namespace.clone()))?;
        let event = WriteEventInput {
            context_id: input.context_id.clone(),
            event: EventInput {
                kind: EventKind::Trace,
                agent_id: input.agent_id,
                agent_role: Some("telemetry".to_string()),
                timestamp: None,
                session_id: format!("telemetry-{}", Uuid::now_v7()),
                task_id: input.task_id,
                project_id: None,
                goal_id: None,
                run_id: None,
                namespace,
                environment: Some("local".to_string()),
                source: "uci".to_string(),
                scope: Scope::Shared,
                tags: vec![format!("level:{}", input.level)],
                dimensions: Vec::new(),
                content: input.message,
                attributes: input.attributes,
            },
        };
        let manifests = self.write_events(vec![event])?;
        Ok(serde_json::json!({
            "ingested": manifests.len(),
            "event_id": manifests.first().map(|item| item.event.id.clone()),
        }))
    }
}
