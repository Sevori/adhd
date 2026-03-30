use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Result, anyhow};
use sysinfo::{Pid, ProcessesToUpdate, System};

use crate::adapters::{
    AgentAdapter, AgentContinuationOutput, DecisionNote, EvidenceNote, ModelCallMetrics,
    OllamaAdapter, SurvivalHypothesis,
};
use crate::continuity::{
    AttachAgentInput, ContextRead, ContinuityHandoffInput, ContinuityItemInput, ContinuityKind,
    ContinuityStatus, HandoffProof, OpenContextInput, ReadContextInput, SharedContinuityKernel,
    SignalInput, SnapshotInput, UnifiedContinuityInterface,
};
use crate::model::{
    DimensionValue, EventInput, EventKind, MemoryLayer, Scope, Selector, SnapshotResolution,
};

use super::survival::SurvivalReport;
use super::{
    BaselineKind, BaselineRunReport, BaselineStatus, BenchmarkClass, BenchmarkClassReport,
    BenchmarkMetrics, ContextEnvelope, ContinuityBenchConfig, ContinuityPathKind,
    ContinuityPathReport, ContinuityPathRole, Evaluation, GroundTruth, ResourceEnvelope, Scenario,
    ScenarioMark, ScenarioPhase, SurfacedItem, TruthItem,
};

pub(crate) fn run_class(
    class: BenchmarkClass,
    config: &ContinuityBenchConfig,
    hypotheses: &[SurvivalHypothesis],
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

    populate_scenario(&kernel, &context.id, &scenario)?;

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
        super::ratio_or_zero(
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
        hrt: super::inferred_hrt(continuity.evaluation.resume_accuracy_score),
        crl_ms: continuity.retrieval_ms as f64,
        smcl: super::ratio_or_zero(
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
    let hypothesis_injection = if !hypotheses.is_empty() {
        super::phase3::run_phase3_injection(
            &class_root,
            class,
            &context.id,
            &scenario,
            &small_b,
            config,
            hypotheses,
        )
        .ok()
    } else {
        None
    };

    let report = BenchmarkClassReport {
        class,
        scenario_id: scenario.id.clone(),
        continuity,
        baselines,
        metrics,
        resource,
        artifacts: vec![full_transcript.model, config.small_model.clone()],
        hypothesis_injection,
    };
    let report_path = class_root.join("report.json");
    std::fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub(crate) fn populate_scenario(
    kernel: &SharedContinuityKernel,
    context_id: &str,
    scenario: &Scenario,
) -> Result<()> {
    let mut counter = 0usize;
    for phase in &scenario.phases {
        counter += 1;
        let manifests = kernel.write_events(vec![crate::continuity::WriteEventInput {
            context_id: Some(context_id.to_string()),
            event: EventInput {
                kind: phase.kind.clone(),
                agent_id: phase.actor_id.clone(),
                agent_role: Some(phase.actor_role.clone()),
                timestamp: None,
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
        for mark in &phase.marks {
            kernel.write_derivations(vec![ContinuityItemInput {
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
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn analyze_and_write_back(
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
        class_root, class, provider, context_id, scenario, adapter, config,
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
        survival: context.survival,
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
) -> Result<BaselineRunReport> {
    let run = if matches!(class, BenchmarkClass::CrashRecovery)
        && baseline == BaselineKind::SharedContinuity
    {
        let reopened = SharedContinuityKernel::open(class_root)?;
        run_baseline_inner(
            class_root, class, baseline, context_id, scenario, adapter, config,
        )?
        .with_kernel(reopened)
    } else {
        run_baseline_inner(
            class_root, class, baseline, context_id, scenario, adapter, config,
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
        survival: run.survival,
        failure: run.failure,
        artifacts: run.artifacts,
        continuity_path: run.continuity_path,
    })
}

pub(super) struct BaselineExecution {
    pub(super) envelope: ContextEnvelope,
    pub(super) output: AgentContinuationOutput,
    pub(super) model_metrics: ModelCallMetrics,
    pub(super) evaluation: Evaluation,
    pub(super) survival: Option<SurvivalReport>,
    pub(super) status: BaselineStatus,
    pub(super) failure: Option<String>,
    pub(super) artifacts: Vec<String>,
    pub(super) continuity_path: Option<ContinuityPathReport>,
}

impl BaselineExecution {
    pub(super) fn with_kernel(self, _kernel: SharedContinuityKernel) -> Self {
        self
    }
}

pub(super) fn run_baseline_inner(
    class_root: &Path,
    class: BenchmarkClass,
    baseline: BaselineKind,
    context_id: &str,
    scenario: &Scenario,
    adapter: &impl AgentAdapter,
    config: &ContinuityBenchConfig,
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
    let (output, model_metrics, evaluation, survival, status, failure) = match adapter
        .analyze(&scenario.objective, &envelope.text)
    {
        Ok((output, model_metrics)) => {
            let repaired = super::repair_output_from_envelope(output, &envelope);
            let evaluation = super::evaluate_output(&repaired, &scenario.truth, &envelope);
            let survival =
                super::survival::benchmark_survival_analysis(&repaired, &scenario.truth, &envelope);
            (
                repaired,
                model_metrics,
                evaluation,
                Some(survival),
                BaselineStatus::Ok,
                None,
            )
        }
        Err(error) => (
            AgentContinuationOutput::default(),
            ModelCallMetrics::default(),
            super::failed_evaluation(&envelope),
            None,
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
        survival,
        status,
        failure,
        artifacts,
        continuity_path,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_context_envelope(
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
                token_estimate: super::estimate_tokens(&scenario.objective),
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
                    super::trim_text(&row.event.input.content, 280)
                ));
            }
            let text = format!("Objective: {}\n{}", scenario.objective, lines.join("\n"));
            Ok((
                ContextEnvelope {
                    provider: baseline,
                    retrieval_ms: started.elapsed().as_millis(),
                    token_estimate: super::estimate_tokens(&text),
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
            format!(
                "{} :: {}",
                incident.title,
                super::trim_text(&incident.body, 220)
            ),
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
            format!(
                "{} :: {}",
                decision.title,
                super::trim_text(&decision.body, 220)
            ),
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
                super::trim_text(&item.preview, 220)
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
            super::trim_text(&item.body, 280)
        ));
    }
    let text = lines.join("\n");
    ContextEnvelope {
        provider: BaselineKind::SharedContinuity,
        retrieval_ms,
        token_estimate: super::estimate_tokens(&text),
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
            super::trim_text(&item.body, 220)
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
            super::trim_text(&item.body, 220)
        ));
    }
}

pub(crate) fn section_prefix(name: &str) -> &'static str {
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
        lines.push(format!(
            "[{label}][memory] {}",
            super::trim_text(&body, 260)
        ));
    }
    let text = lines.join("\n");
    ContextEnvelope {
        provider,
        retrieval_ms,
        token_estimate: super::estimate_tokens(&text),
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
            super::trim_text(&row.event.input.content, 240)
        ));
    }
    let text = lines.join("\n");
    ContextEnvelope {
        provider,
        retrieval_ms,
        token_estimate: super::estimate_tokens(&text),
        text,
        surfaced,
    }
}

pub(crate) fn format_continuity_path_label(path: &ContinuityPathReport) -> String {
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
        title: super::trim_text(&item.text, 80),
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
        title: super::trim_text(&item.text, 80),
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

pub(crate) fn scenario_for(class: BenchmarkClass) -> Scenario {
    let base_truth = GroundTruth {
        critical_facts: vec![
            TruthItem {
                keywords: vec!["selector_missing", "src/query.rs"],
                rationale_keywords: Vec::new(),
                judge_note: Some(
                    "Full credit requires both the selector/support-memory failure and the src/query.rs anchor; file mention alone is partial.",
                ),
                judge_required_concepts: vec!["selector/support-memory failure", "src/query.rs"],
            },
            TruthItem {
                keywords: vec!["context", "primary"],
                rationale_keywords: Vec::new(),
                judge_note: Some(
                    "Full credit requires explicit identification that the active resume context is the primary context, not just a task mention.",
                ),
                judge_required_concepts: vec!["active resume context", "primary context"],
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
        avoid_repeating: vec![TruthItem {
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

fn gpu_report(config: &ContinuityBenchConfig) -> Option<String> {
    std::process::Command::new("ollama")
        .arg("ps")
        .env("OLLAMA_HOST", config.ollama_endpoint.replace("http://", ""))
        .output()
        .ok()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|text| !text.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenario_for_all_classes_produces_valid_scenarios() {
        for class in BenchmarkClass::all() {
            let scenario = scenario_for(class);
            assert!(
                !scenario.id.is_empty(),
                "scenario id should not be empty for {class:?}"
            );
            assert!(
                !scenario.truth.critical_facts.is_empty(),
                "scenario should have critical facts for {class:?}"
            );
            assert!(
                !scenario.phases.is_empty(),
                "scenario should have phases for {class:?}"
            );
        }
    }
}
