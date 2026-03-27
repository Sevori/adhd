use serde::{Deserialize, Serialize};

use crate::model::MemoryLayer;

pub const MACHINE_NAMESPACE_ALIAS: &str = "@machine";
pub const DEFAULT_MACHINE_TASK_ID: &str = "machine-organism";

pub(crate) const PROOF_TRIM_LIMIT: usize = 220;
pub(crate) const DEFAULT_DIMENSION_WEIGHT: i32 = 100;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContinuityKind {
    WorkingState,
    WorkClaim,
    Derivation,
    Fact,
    Decision,
    Constraint,
    Hypothesis,
    Incident,
    OperationalScar,
    Outcome,
    Signal,
    Summary,
    Lesson,
}

impl ContinuityKind {
    pub fn default_layer(self) -> MemoryLayer {
        match self {
            Self::WorkingState | Self::WorkClaim | Self::Signal => MemoryLayer::Hot,
            Self::Derivation | Self::Fact | Self::Decision | Self::Constraint => {
                MemoryLayer::Semantic
            }
            Self::Hypothesis | Self::Incident | Self::Outcome | Self::Lesson => {
                MemoryLayer::Episodic
            }
            Self::OperationalScar => MemoryLayer::Semantic,
            Self::Summary => MemoryLayer::Summary,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::WorkingState => "working_state",
            Self::WorkClaim => "work_claim",
            Self::Derivation => "derivation",
            Self::Fact => "fact",
            Self::Decision => "decision",
            Self::Constraint => "constraint",
            Self::Hypothesis => "hypothesis",
            Self::Incident => "incident",
            Self::OperationalScar => "operational_scar",
            Self::Outcome => "outcome",
            Self::Signal => "signal",
            Self::Summary => "summary",
            Self::Lesson => "lesson",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContinuityStatus {
    Open,
    Active,
    Resolved,
    Superseded,
    Rejected,
}

impl ContinuityStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Active => "active",
            Self::Resolved => "resolved",
            Self::Superseded => "superseded",
            Self::Rejected => "rejected",
        }
    }

    pub fn is_open(self) -> bool {
        matches!(self, Self::Open | Self::Active)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextStatus {
    Open,
    Paused,
    Closed,
}

impl ContextStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Paused => "paused",
            Self::Closed => "closed",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CoordinationLane {
    Anxiety,
    Review,
    Warning,
    Backoff,
    Coach,
}

impl CoordinationLane {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Anxiety => "anxiety",
            Self::Review => "review",
            Self::Warning => "warning",
            Self::Backoff => "backoff",
            Self::Coach => "coach",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CoordinationSeverity {
    Info,
    Warn,
    Block,
}

impl CoordinationSeverity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Block => "block",
        }
    }
}
