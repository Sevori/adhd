use anyhow::Result;

use crate::model::{IngestManifest, ReplayRow, Selector, SubscriptionInput, SubscriptionRecord};

use super::schema::*;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum UciRequest {
    IdentifyMachine,
    AttachAgent { input: AttachAgentInput },
    UpsertAgentBadge { input: UpsertAgentBadgeInput },
    Heartbeat { input: HeartbeatInput },
    OpenContext { input: OpenContextInput },
    ReadContext { input: ReadContextInput },
    Recall { input: RecallInput },
    WriteEvents { inputs: Vec<WriteEventInput> },
    WriteDerivations { inputs: Vec<ContinuityItemInput> },
    ClaimWork { input: ClaimWorkInput },
    PublishCoordinationSignal { input: CoordinationSignalInput },
    PublishSignal { input: SignalInput },
    Subscribe { input: SubscriptionInput },
    Handoff { input: ContinuityHandoffInput },
    Snapshot { input: SnapshotInput },
    Resume { input: ResumeInput },
    Explain { target: ExplainTarget },
    Replay { selector: Selector, limit: usize },
    RecordOutcome { input: OutcomeInput },
    MarkDecision { input: ContinuityItemInput },
    MarkConstraint { input: ContinuityItemInput },
    MarkHypothesis { input: ContinuityItemInput },
    MarkIncident { input: ContinuityItemInput },
    MarkOperationalScar { input: ContinuityItemInput },
    ResolveOrSupersede { input: ResolveOrSupersedeInput },
    EmitTelemetry { input: TelemetryEventInput },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "op", content = "data", rename_all = "snake_case")]
pub enum UciResponse {
    MachineProfile(MachineProfile),
    AgentAttachment(AgentAttachmentRecord),
    AgentBadge(AgentBadgeRecord),
    Context(ContextRecord),
    ContextRead(ContextRead),
    Recall(ContinuityRecall),
    IngestManifests(Vec<IngestManifest>),
    ContinuityItems(Vec<ContinuityItemRecord>),
    Subscription(SubscriptionRecord),
    Handoff(ContinuityHandoffRecord),
    Snapshot(SnapshotRecord),
    Resume(ResumeRecord),
    Explain(serde_json::Value),
    Replay(Vec<ReplayRow>),
    Telemetry(serde_json::Value),
}

pub trait UnifiedContinuityInterface {
    fn identify_machine(&self) -> Result<MachineProfile>;
    fn attach_agent(&self, input: AttachAgentInput) -> Result<AgentAttachmentRecord>;
    fn upsert_agent_badge(&self, input: UpsertAgentBadgeInput) -> Result<AgentBadgeRecord>;
    fn heartbeat(&self, input: HeartbeatInput) -> Result<AgentAttachmentRecord>;
    fn open_context(&self, input: OpenContextInput) -> Result<ContextRecord>;
    fn read_context(&self, input: ReadContextInput) -> Result<ContextRead>;
    fn recall(&self, input: RecallInput) -> Result<ContinuityRecall>;
    fn write_events(&self, inputs: Vec<WriteEventInput>) -> Result<Vec<IngestManifest>>;
    fn write_derivations(
        &self,
        inputs: Vec<ContinuityItemInput>,
    ) -> Result<Vec<ContinuityItemRecord>>;
    fn claim_work(&self, input: ClaimWorkInput) -> Result<ContinuityItemRecord>;
    fn publish_coordination_signal(
        &self,
        input: CoordinationSignalInput,
    ) -> Result<ContinuityItemRecord>;
    fn publish_signal(&self, input: SignalInput) -> Result<ContinuityItemRecord>;
    fn subscribe(&self, input: SubscriptionInput) -> Result<SubscriptionRecord>;
    fn handoff(&self, input: ContinuityHandoffInput) -> Result<ContinuityHandoffRecord>;
    fn snapshot(&self, input: SnapshotInput) -> Result<SnapshotRecord>;
    fn resume(&self, input: ResumeInput) -> Result<ResumeRecord>;
    fn explain(&self, target: ExplainTarget) -> Result<serde_json::Value>;
    fn replay_selector(&self, selector: Selector, limit: usize) -> Result<Vec<ReplayRow>>;
    fn record_outcome(&self, input: OutcomeInput) -> Result<ContinuityItemRecord>;
    fn mark_decision(&self, input: ContinuityItemInput) -> Result<ContinuityItemRecord>;
    fn mark_constraint(&self, input: ContinuityItemInput) -> Result<ContinuityItemRecord>;
    fn mark_hypothesis(&self, input: ContinuityItemInput) -> Result<ContinuityItemRecord>;
    fn mark_incident(&self, input: ContinuityItemInput) -> Result<ContinuityItemRecord>;
    fn mark_operational_scar(&self, input: ContinuityItemInput) -> Result<ContinuityItemRecord>;
    fn resolve_or_supersede(&self, input: ResolveOrSupersedeInput) -> Result<ContinuityItemRecord>;
    fn emit_telemetry(&self, input: TelemetryEventInput) -> Result<serde_json::Value>;

    fn handle_request(&self, request: UciRequest) -> Result<UciResponse> {
        Ok(match request {
            UciRequest::IdentifyMachine => UciResponse::MachineProfile(self.identify_machine()?),
            UciRequest::AttachAgent { input } => {
                UciResponse::AgentAttachment(self.attach_agent(input)?)
            }
            UciRequest::UpsertAgentBadge { input } => {
                UciResponse::AgentBadge(self.upsert_agent_badge(input)?)
            }
            UciRequest::Heartbeat { input } => UciResponse::AgentAttachment(self.heartbeat(input)?),
            UciRequest::OpenContext { input } => UciResponse::Context(self.open_context(input)?),
            UciRequest::ReadContext { input } => {
                UciResponse::ContextRead(self.read_context(input)?)
            }
            UciRequest::Recall { input } => UciResponse::Recall(self.recall(input)?),
            UciRequest::WriteEvents { inputs } => {
                UciResponse::IngestManifests(self.write_events(inputs)?)
            }
            UciRequest::WriteDerivations { inputs } => {
                UciResponse::ContinuityItems(self.write_derivations(inputs)?)
            }
            UciRequest::ClaimWork { input } => {
                UciResponse::ContinuityItems(vec![self.claim_work(input)?])
            }
            UciRequest::PublishCoordinationSignal { input } => {
                UciResponse::ContinuityItems(vec![self.publish_coordination_signal(input)?])
            }
            UciRequest::PublishSignal { input } => {
                UciResponse::ContinuityItems(vec![self.publish_signal(input)?])
            }
            UciRequest::Subscribe { input } => UciResponse::Subscription(self.subscribe(input)?),
            UciRequest::Handoff { input } => UciResponse::Handoff(self.handoff(input)?),
            UciRequest::Snapshot { input } => UciResponse::Snapshot(self.snapshot(input)?),
            UciRequest::Resume { input } => UciResponse::Resume(self.resume(input)?),
            UciRequest::Explain { target } => UciResponse::Explain(self.explain(target)?),
            UciRequest::Replay { selector, limit } => {
                UciResponse::Replay(self.replay_selector(selector, limit)?)
            }
            UciRequest::RecordOutcome { input } => {
                UciResponse::ContinuityItems(vec![self.record_outcome(input)?])
            }
            UciRequest::MarkDecision { input } => {
                UciResponse::ContinuityItems(vec![self.mark_decision(input)?])
            }
            UciRequest::MarkConstraint { input } => {
                UciResponse::ContinuityItems(vec![self.mark_constraint(input)?])
            }
            UciRequest::MarkHypothesis { input } => {
                UciResponse::ContinuityItems(vec![self.mark_hypothesis(input)?])
            }
            UciRequest::MarkIncident { input } => {
                UciResponse::ContinuityItems(vec![self.mark_incident(input)?])
            }
            UciRequest::MarkOperationalScar { input } => {
                UciResponse::ContinuityItems(vec![self.mark_operational_scar(input)?])
            }
            UciRequest::ResolveOrSupersede { input } => {
                UciResponse::ContinuityItems(vec![self.resolve_or_supersede(input)?])
            }
            UciRequest::EmitTelemetry { input } => {
                UciResponse::Telemetry(self.emit_telemetry(input)?)
            }
        })
    }
}
