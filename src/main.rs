use std::collections::BTreeMap;
use std::fs;
use std::future::Future;
use std::io::{self, Read};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Command as StdCommand, Stdio};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::Serialize;
use serde_json::Value;
use tempfile::TempDir;
use tokio::process::Command as TokioCommand;
use tokio::time::{Duration, sleep};
use uuid::Uuid;

use ice::Engine;
use ice::benchmark::{BenchmarkClass, ContinuityBenchConfig, run_continuity_suite};
use ice::claude_install::{
    ClaudeCodeInstallRequest, ClaudeCodeStatusRequest, ClaudeCodeUninstallRequest,
    DEFAULT_CLAUDE_SERVER_NAME, claude_code_status, install_claude_code, uninstall_claude_code,
};
use ice::codex::{CodexGlobalInstallRequest, DEFAULT_CODEX_GLOBAL_SERVER_NAME, install_global_mcp};
use ice::continuity::{
    AgentBadgeRecord, AttachAgentInput, ClaimWorkInput, ContinuityItemInput, ContinuityKind,
    ContinuityStatus, CoordinationProjectedLane, DEFAULT_MACHINE_TASK_ID, HeartbeatInput,
    MACHINE_NAMESPACE_ALIAS, OpenContextInput, ReadContextInput, ResolveOrSupersedeInput,
    SnapshotInput, SupportRef, UciRequest, UnifiedContinuityInterface, UpsertAgentBadgeInput,
    WriteEventInput,
};
use ice::dispatch::{
    CompleteAssignmentInput, DEFAULT_DISPATCH_NOTIFY_CHANNEL, DispatchAttachedLaneSource,
    DispatchConfig, DispatchSignalKind, DispatchSpine, DispatchWorkerTier,
    DispatchWorkerUpsertInput, PublishDispatchSignalInput,
};
use ice::dogfood::{OrganismChorusConfig, run_organism_choir};
use ice::goose_install::{
    DEFAULT_GOOSE_SERVER_NAME, GooseInstallRequest, GooseStatusRequest, GooseUninstallRequest,
    goose_status, install_goose, uninstall_goose,
};
use ice::http::serve;
use ice::longmemeval::{
    LongMemEvalEvaluateConfig, LongMemEvalReaderMethod, LongMemEvalReaderProvider,
    LongMemEvalRunConfig, evaluate_longmemeval, run_longmemeval,
};
use ice::market_head::{
    MarketHeadChallengeConfig, MarketHeadChallengeManifest, compare_market_head_judge_calibration,
    compare_market_head_judge_disagreement, compare_market_head_judge_pack,
    compare_market_head_same_pack, evaluate_market_head_challenge,
    evaluate_market_head_judge_challenge, export_market_head_challenge,
    export_market_head_judge_challenge, render_market_head_judge_calibration_markdown,
    render_market_head_judge_disagreement_markdown, render_market_head_judge_pack_markdown,
    render_market_head_same_pack_markdown,
};
use ice::mcp::serve_stdio;
use ice::model::{
    DimensionValue, EventInput, EventKind, HandoffInput, MemoryLayer, QueryInput, Scope, Selector,
    SnapshotResolution, SubscriptionInput, ViewInput, ViewOp,
};
use ice::opencode_install::{
    DEFAULT_OPENCODE_SERVER_NAME, OpenCodeInstallRequest, OpenCodeStatusRequest,
    OpenCodeUninstallRequest, install_opencode, opencode_status, uninstall_opencode,
};
use ice::openhands_install::{
    DEFAULT_OPENHANDS_SERVER_NAME, OpenHandsInstallRequest, OpenHandsStatusRequest,
    OpenHandsUninstallRequest, install_openhands, openhands_status, uninstall_openhands,
};
use ice::telemetry::EngineTelemetry;

#[derive(Debug, Parser)]
#[command(name = "ice")]
#[command(about = "Infinite Context Engine prototype")]
struct Cli {
    #[arg(long, default_value = ".ice")]
    root: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Ingest(IngestArgs),
    Replay(ReplayArgs),
    Memory(MemoryArgs),
    Query(QueryArgs),
    Annotate(AnnotateArgs),
    Relate(RelateArgs),
    View(ViewArgs),
    Handoff(HandoffArgs),
    Subscribe(SubscribeArgs),
    Explain(ExplainArgs),
    Metrics,
    #[command(name = "longmemeval", alias = "long-mem-eval")]
    LongMemEval(LongMemEvalArgs),
    Serve(ServeArgs),
    Mcp,
    Wrap(WrapArgs),
    Demo(DemoArgs),
    Bench(BenchArgs),
    BenchMarket(BenchMarketArgs),
    Uci(UciArgs),
    Dogfood(DogfoodArgs),
    Dispatch(DispatchArgs),
    Codex(CodexArgs),
    Claude(ClaudeArgs),
    Openhands(OpenHandsArgs),
    Opencode(OpenCodeArgs),
    Goose(GooseArgs),
}

#[derive(Debug, Args)]
struct IngestArgs {
    #[arg(long, value_enum)]
    kind: EventKind,
    #[arg(long)]
    agent: String,
    #[arg(long)]
    timestamp: Option<DateTime<Utc>>,
    #[arg(long)]
    session: String,
    #[arg(long)]
    task: Option<String>,
    #[arg(long)]
    project: Option<String>,
    #[arg(long)]
    goal: Option<String>,
    #[arg(long)]
    run: Option<String>,
    #[arg(long)]
    namespace: Option<String>,
    #[arg(long)]
    environment: Option<String>,
    #[arg(long)]
    agent_role: Option<String>,
    #[arg(long, default_value = "cli")]
    source: String,
    #[arg(long, value_enum, default_value_t = Scope::Shared)]
    scope: Scope,
    #[arg(long, value_delimiter = ',')]
    tags: Vec<String>,
    #[arg(long)]
    text: Option<String>,
    #[arg(long, default_value = "[]")]
    dimensions_json: String,
    #[arg(long, default_value = "{}")]
    attrs_json: String,
}

#[derive(Debug, Args)]
struct ReplayArgs {
    #[arg(long)]
    session: Option<String>,
    #[arg(long)]
    selector_json: Option<String>,
    #[arg(long, default_value_t = 50)]
    limit: usize,
}

#[derive(Debug, Args)]
struct MemoryArgs {
    #[command(subcommand)]
    command: MemoryCommand,
}

#[derive(Debug, Args)]
struct QueryArgs {
    #[arg(long)]
    text: Option<String>,
    #[arg(long)]
    agent: Option<String>,
    #[arg(long)]
    session: Option<String>,
    #[arg(long)]
    task: Option<String>,
    #[arg(long)]
    namespace: Option<String>,
    #[arg(long)]
    objective: Option<String>,
    #[arg(long)]
    view: Option<String>,
    #[arg(long)]
    selector_json: Option<String>,
    #[arg(long, default_value_t = 512)]
    budget_tokens: usize,
    #[arg(long, default_value_t = 24)]
    candidate_limit: usize,
}

#[derive(Debug, Args)]
struct AnnotateArgs {
    #[arg(long)]
    item_type: String,
    #[arg(long)]
    item_id: String,
    #[arg(long, default_value = "[]")]
    dimensions_json: String,
}

#[derive(Debug, Args)]
struct RelateArgs {
    #[arg(long)]
    source_id: String,
    #[arg(long)]
    target_id: String,
    #[arg(long)]
    relation: String,
    #[arg(long, default_value_t = 1.0)]
    weight: f64,
    #[arg(long, default_value = "{}")]
    attrs_json: String,
}

#[derive(Debug, Args)]
struct ViewArgs {
    #[command(subcommand)]
    command: ViewCommand,
}

#[derive(Debug, Subcommand)]
enum ViewCommand {
    Create {
        #[arg(long, value_enum, default_value_t = ViewOp::Slice)]
        op: ViewOp,
        #[arg(long)]
        owner_agent: Option<String>,
        #[arg(long)]
        namespace: Option<String>,
        #[arg(long)]
        objective: Option<String>,
        #[arg(long, default_value = "[]")]
        selector_json: String,
        #[arg(long)]
        source_view: Vec<String>,
        #[arg(long, value_enum)]
        resolution: Option<SnapshotResolution>,
        #[arg(long, default_value_t = 48)]
        limit: usize,
    },
    Get {
        #[arg(long)]
        id: String,
    },
    Explain {
        #[arg(long)]
        id: String,
    },
    Fork {
        #[arg(long)]
        id: String,
        #[arg(long)]
        owner_agent: Option<String>,
    },
}

#[derive(Debug, Args)]
struct HandoffArgs {
    #[arg(long)]
    from_agent: String,
    #[arg(long)]
    to_agent: String,
    #[arg(long)]
    reason: String,
    #[arg(long)]
    view: Option<String>,
    #[arg(long)]
    selector_json: Option<String>,
    #[arg(long)]
    namespace: Option<String>,
    #[arg(long)]
    objective: Option<String>,
    #[arg(long)]
    text: Option<String>,
    #[arg(long, default_value_t = 192)]
    budget_tokens: usize,
}

#[derive(Debug, Args)]
struct SubscribeArgs {
    #[command(subcommand)]
    command: SubscribeCommand,
}

#[derive(Debug, Subcommand)]
enum SubscribeCommand {
    Create {
        #[arg(long)]
        agent: String,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        selector_json: String,
    },
    Poll {
        #[arg(long)]
        id: String,
        #[arg(long, default_value_t = 25)]
        limit: usize,
    },
}

#[derive(Debug, Args)]
struct ExplainArgs {
    #[command(subcommand)]
    command: ExplainCommand,
}

#[derive(Debug, Subcommand)]
enum ExplainCommand {
    Pack {
        #[arg(long)]
        id: String,
    },
    Handoff {
        #[arg(long)]
        id: String,
    },
}

#[derive(Debug, Args)]
struct ServeArgs {
    #[arg(long, default_value = "127.0.0.1:4040")]
    addr: SocketAddr,
}

#[derive(Debug, Args)]
#[command(trailing_var_arg = true)]
struct WrapArgs {
    #[arg(long)]
    agent: String,
    #[arg(long)]
    session: String,
    #[arg(long)]
    task: Option<String>,
    #[arg(long, default_value = "wrap")]
    source: String,
    #[arg(long, value_enum, default_value_t = Scope::Shared)]
    scope: Scope,
    #[arg(long, value_delimiter = ',')]
    tags: Vec<String>,
    #[arg(required = true)]
    command: Vec<String>,
}

#[derive(Debug, Args)]
struct DemoArgs {
    #[arg(long, default_value_t = 160)]
    budget_tokens: usize,
}

#[derive(Debug, Args)]
struct BenchArgs {
    #[arg(long, value_enum, default_value_t = BenchMode::Continuity)]
    mode: BenchMode,
    #[arg(long = "class", value_enum)]
    classes: Vec<BenchmarkClass>,
    #[arg(long, default_value_t = 12)]
    cases: usize,
    #[arg(long, default_value_t = 24)]
    distractors: usize,
    #[arg(long, default_value_t = 8)]
    recent_window: usize,
    #[arg(long, default_value_t = 160)]
    budget_tokens: usize,
    #[arg(long, default_value_t = 24)]
    candidate_limit: usize,
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    ollama_endpoint: String,
    #[arg(long, default_value = "glm-4.7-flash:latest")]
    strong_model: String,
    #[arg(long, default_value = "qwen2.5:0.5b")]
    small_model: String,
    #[arg(long, default_value_t = 180)]
    timeout_secs: u64,
    #[arg(long, default_value_t = 192)]
    num_predict: usize,
}

#[derive(Debug, Args)]
struct UciArgs {
    #[arg(long)]
    json: Option<String>,
}

#[derive(Debug, Args)]
struct LongMemEvalArgs {
    #[command(subcommand)]
    command: LongMemEvalCommand,
}

#[derive(Debug, Subcommand)]
enum LongMemEvalCommand {
    Run(LongMemEvalRunArgs),
    Evaluate(LongMemEvalEvaluateArgs),
}

#[derive(Debug, Args, Clone)]
struct LongMemEvalRunArgs {
    #[arg(long)]
    dataset: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    work_dir: Option<PathBuf>,
    #[arg(long, default_value = "longmemeval")]
    namespace_prefix: String,
    #[arg(long, value_enum, default_value_t = LongMemEvalReaderProviderArg::Ollama)]
    reader_provider: LongMemEvalReaderProviderArg,
    #[arg(long, value_enum, default_value_t = LongMemEvalReaderMethodArg::ConSeparate)]
    reader_method: LongMemEvalReaderMethodArg,
    #[arg(long)]
    reader_endpoint: Option<String>,
    #[arg(long, default_value = "qwen2.5:14b")]
    reader_model: String,
    #[arg(long)]
    reader_api_key_env: Option<String>,
    #[arg(long, default_value_t = 180)]
    reader_timeout_secs: u64,
    #[arg(long, default_value_t = 256)]
    reader_num_predict: usize,
    #[arg(long, default_value_t = 4)]
    reader_max_retries: usize,
    #[arg(long, default_value_t = 2)]
    reader_retry_backoff_secs: u64,
    #[arg(long, default_value_t = 512)]
    budget_tokens: usize,
    #[arg(long, default_value_t = 24)]
    candidate_limit: usize,
    #[arg(long, default_value_t = 0)]
    offset: usize,
    #[arg(long)]
    max_cases: Option<usize>,
    #[arg(long = "question-id", value_delimiter = ',')]
    question_ids: Vec<String>,
    #[arg(long = "question-type", value_delimiter = ',')]
    question_types: Vec<String>,
}

#[derive(Debug, Args, Clone)]
struct LongMemEvalEvaluateArgs {
    #[arg(long)]
    repo: PathBuf,
    #[arg(long)]
    predictions: PathBuf,
    #[arg(long)]
    dataset: PathBuf,
    #[arg(long, default_value = "python3")]
    python_bin: String,
    #[arg(long, default_value = "gpt-4o")]
    judge_model: String,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum LongMemEvalReaderProviderArg {
    Ollama,
    OpenaiCompatible,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum LongMemEvalReaderMethodArg {
    Direct,
    ConSeparate,
}

impl From<LongMemEvalReaderProviderArg> for LongMemEvalReaderProvider {
    fn from(value: LongMemEvalReaderProviderArg) -> Self {
        match value {
            LongMemEvalReaderProviderArg::Ollama => Self::Ollama,
            LongMemEvalReaderProviderArg::OpenaiCompatible => Self::OpenAiCompatible,
        }
    }
}

impl From<LongMemEvalReaderMethodArg> for LongMemEvalReaderMethod {
    fn from(value: LongMemEvalReaderMethodArg) -> Self {
        match value {
            LongMemEvalReaderMethodArg::Direct => Self::Direct,
            LongMemEvalReaderMethodArg::ConSeparate => Self::ConSeparate,
        }
    }
}

fn default_longmemeval_reader_endpoint(provider: LongMemEvalReaderProviderArg) -> &'static str {
    match provider {
        LongMemEvalReaderProviderArg::Ollama => "http://127.0.0.1:11434",
        LongMemEvalReaderProviderArg::OpenaiCompatible => "https://api.openai.com/v1",
    }
}

fn default_longmemeval_reader_api_key_env(
    provider: LongMemEvalReaderProviderArg,
) -> Option<&'static str> {
    match provider {
        LongMemEvalReaderProviderArg::Ollama => None,
        LongMemEvalReaderProviderArg::OpenaiCompatible => Some("OPENAI_API_KEY"),
    }
}

#[derive(Debug, Args)]
struct BenchMarketArgs {
    #[command(subcommand)]
    command: BenchMarketCommand,
}

#[derive(Debug, Subcommand)]
enum BenchMarketCommand {
    Export(BenchMarketExportArgs),
    Evaluate(BenchMarketEvaluateArgs),
    Answer(BenchMarketAnswerArgs),
    JudgeExport(BenchMarketJudgeExportArgs),
    JudgeEvaluate(BenchMarketJudgeEvaluateArgs),
    JudgeCompare(BenchMarketJudgeCompareArgs),
    JudgePackCompare(BenchMarketJudgePackCompareArgs),
    JudgeCalibrate(BenchMarketJudgeCalibrateArgs),
    JudgeDisagreement(BenchMarketJudgeDisagreementArgs),
}

#[derive(Debug, Args, Clone)]
struct BenchMarketExportArgs {
    #[arg(long = "class", value_enum)]
    classes: Vec<BenchmarkClass>,
    #[arg(long, default_value_t = 192)]
    budget_tokens: usize,
    #[arg(long, default_value_t = 12)]
    candidate_limit: usize,
    #[arg(long, default_value_t = 6)]
    recent_window: usize,
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    ollama_endpoint: String,
    #[arg(long, default_value = "qwen2.5:1.5b")]
    strong_model: String,
    #[arg(long, default_value = "qwen2.5:0.5b")]
    small_model: String,
    #[arg(long, default_value_t = 180)]
    timeout_secs: u64,
    #[arg(long, default_value_t = 384)]
    num_predict: usize,
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct BenchMarketEvaluateArgs {
    #[arg(long)]
    evaluator_pack: PathBuf,
    #[arg(long)]
    responses_dir: PathBuf,
    #[arg(long)]
    model: String,
}

#[derive(Debug, Args, Clone)]
struct BenchMarketJudgeExportArgs {
    #[arg(long)]
    evaluator_pack: PathBuf,
    #[arg(long)]
    responses_dir: PathBuf,
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct BenchMarketJudgeEvaluateArgs {
    #[arg(long)]
    judge_manifest: PathBuf,
    #[arg(long)]
    responses_dir: PathBuf,
    #[arg(long)]
    model: String,
}

#[derive(Debug, Clone)]
struct NamedJudgeReportArg {
    judge_head: String,
    report_path: PathBuf,
}

#[derive(Debug, Args, Clone)]
struct BenchMarketJudgeCompareArgs {
    #[arg(long)]
    challenged_head: String,
    #[arg(long)]
    canonical_report: PathBuf,
    #[arg(long = "judge-report", value_parser = parse_named_judge_report, required = true)]
    judge_reports: Vec<NamedJudgeReportArg>,
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct BenchMarketJudgePackCompareArgs {
    #[arg(long = "same-pack-report", required = true)]
    same_pack_reports: Vec<PathBuf>,
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct BenchMarketJudgeCalibrateArgs {
    #[arg(long = "same-pack-report", required = true)]
    same_pack_reports: Vec<PathBuf>,
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct BenchMarketJudgeDisagreementArgs {
    #[arg(long)]
    challenged_head: String,
    #[arg(long)]
    judge_head: String,
    #[arg(long)]
    canonical_report: PathBuf,
    #[arg(long)]
    judge_report: PathBuf,
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum BenchMarketProvider {
    Codex,
    Claude,
}

#[derive(Debug, Args, Clone)]
struct BenchMarketAnswerArgs {
    #[arg(long)]
    responses_dir: PathBuf,
    #[arg(long)]
    manifest: Option<PathBuf>,
    #[arg(long, value_enum)]
    provider: BenchMarketProvider,
    #[arg(long)]
    model: Option<String>,
}

#[derive(Debug, Serialize)]
struct BenchMarketAnswerReport {
    generated_at: String,
    provider: BenchMarketProvider,
    model: String,
    responses_dir: String,
    manifest_path: String,
    case_count: usize,
    cases: Vec<BenchMarketAnswerCaseReport>,
}

#[derive(Debug, Serialize)]
struct BenchMarketAnswerCaseReport {
    class: BenchmarkClass,
    scenario_id: String,
    response_path: String,
}

#[derive(Debug, Serialize)]
struct BenchMarketJudgeCompareReport {
    challenged_head: String,
    canonical_report_path: String,
    judge_report_paths: Vec<BenchMarketJudgeCompareReportInput>,
    json_path: String,
    markdown_path: String,
}

#[derive(Debug, Serialize)]
struct BenchMarketJudgeCompareReportInput {
    judge_head: String,
    report_path: String,
}

#[derive(Debug, Serialize)]
struct BenchMarketJudgePackCompareReport {
    same_pack_report_paths: Vec<String>,
    json_path: String,
    markdown_path: String,
}

#[derive(Debug, Serialize)]
struct BenchMarketJudgeCalibrateReport {
    same_pack_report_paths: Vec<String>,
    json_path: String,
    markdown_path: String,
}

#[derive(Debug, Serialize)]
struct BenchMarketJudgeDisagreementCliReport {
    challenged_head: String,
    judge_head: String,
    canonical_report_path: String,
    judge_report_path: String,
    json_path: String,
    markdown_path: String,
}

#[derive(Debug, Args)]
struct DogfoodArgs {
    #[command(subcommand)]
    command: DogfoodCommand,
}

#[derive(Debug, Args)]
struct DispatchArgs {
    #[command(subcommand)]
    command: DispatchCommand,
}

#[derive(Debug, Args)]
struct CodexArgs {
    #[command(subcommand)]
    command: CodexCommand,
}

#[derive(Debug, Args)]
struct ClaudeArgs {
    #[command(subcommand)]
    command: ClaudeCommand,
}

#[derive(Debug, Args)]
struct OpenHandsArgs {
    #[command(subcommand)]
    command: OpenHandsCommand,
}

#[derive(Debug, Args)]
struct OpenCodeArgs {
    #[command(subcommand)]
    command: OpenCodeCommand,
}

#[derive(Debug, Args)]
struct GooseArgs {
    #[command(subcommand)]
    command: GooseCommand,
}

#[derive(Debug, Subcommand)]
enum DogfoodCommand {
    SyncRepo(RepoSyncArgs),
    Heartbeat(HeartbeatArgs),
    Organism(OrganismArgs),
}

#[derive(Debug, Subcommand)]
enum DispatchCommand {
    Init(DispatchInitArgs),
    Complete(DispatchCompleteArgsCli),
    Heartbeat(DispatchWorkerArgs),
    Claim(DispatchClaimArgs),
    Listen(DispatchListenArgs),
    Ack(DispatchAckArgs),
    Stats(DispatchStatsArgs),
}

#[derive(Debug, Subcommand)]
enum CodexCommand {
    InstallGlobal(CodexInstallGlobalArgs),
}

#[derive(Debug, Subcommand)]
enum ClaudeCommand {
    InstallGlobal(ClaudeInstallGlobalArgs),
    Status(ClaudeStatusArgs),
    Uninstall(ClaudeUninstallArgs),
}

#[derive(Debug, Subcommand)]
enum OpenHandsCommand {
    InstallGlobal(OpenHandsInstallGlobalArgs),
    Status(OpenHandsStatusArgs),
    Uninstall(OpenHandsUninstallArgs),
}

#[derive(Debug, Subcommand)]
enum OpenCodeCommand {
    InstallGlobal(OpenCodeInstallGlobalArgs),
    Status(OpenCodeStatusArgs),
    Uninstall(OpenCodeUninstallArgs),
}

#[derive(Debug, Subcommand)]
enum GooseCommand {
    InstallGlobal(GooseInstallGlobalArgs),
    Status(GooseStatusArgs),
    Uninstall(GooseUninstallArgs),
}

#[derive(Debug, Args, Clone)]
struct ClaudeInstallGlobalArgs {
    #[arg(long)]
    root: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_CLAUDE_SERVER_NAME)]
    server_name: String,
    #[arg(long, hide = true)]
    code_config: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct ClaudeStatusArgs {
    #[arg(long)]
    root: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_CLAUDE_SERVER_NAME)]
    server_name: String,
    #[arg(long, hide = true)]
    code_config: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct ClaudeUninstallArgs {
    #[arg(long, default_value = DEFAULT_CLAUDE_SERVER_NAME)]
    server_name: String,
    #[arg(long, hide = true)]
    code_config: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct OpenHandsInstallGlobalArgs {
    #[arg(long)]
    root: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_OPENHANDS_SERVER_NAME)]
    server_name: String,
    #[arg(long, hide = true)]
    mcp_config: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct OpenHandsStatusArgs {
    #[arg(long)]
    root: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_OPENHANDS_SERVER_NAME)]
    server_name: String,
    #[arg(long, hide = true)]
    mcp_config: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct OpenHandsUninstallArgs {
    #[arg(long, default_value = DEFAULT_OPENHANDS_SERVER_NAME)]
    server_name: String,
    #[arg(long, hide = true)]
    mcp_config: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct OpenCodeInstallGlobalArgs {
    #[arg(long)]
    root: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_OPENCODE_SERVER_NAME)]
    server_name: String,
    #[arg(long, hide = true)]
    config: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct OpenCodeStatusArgs {
    #[arg(long)]
    root: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_OPENCODE_SERVER_NAME)]
    server_name: String,
    #[arg(long, hide = true)]
    config: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct OpenCodeUninstallArgs {
    #[arg(long, default_value = DEFAULT_OPENCODE_SERVER_NAME)]
    server_name: String,
    #[arg(long, hide = true)]
    config: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct GooseInstallGlobalArgs {
    #[arg(long)]
    root: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_GOOSE_SERVER_NAME)]
    server_name: String,
    #[arg(long, hide = true)]
    config: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct GooseStatusArgs {
    #[arg(long)]
    root: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_GOOSE_SERVER_NAME)]
    server_name: String,
    #[arg(long, hide = true)]
    config: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct GooseUninstallArgs {
    #[arg(long, default_value = DEFAULT_GOOSE_SERVER_NAME)]
    server_name: String,
    #[arg(long, hide = true)]
    config: Option<PathBuf>,
}

#[derive(Debug, Args, Clone)]
struct RepoSyncArgs {
    #[arg(long)]
    repo_root: Option<PathBuf>,
    #[arg(long)]
    branch: Option<String>,
    #[arg(long)]
    head: Option<String>,
    #[arg(long, default_value = "codex-repo-sync")]
    agent: String,
    #[arg(long, default_value = "operator")]
    agent_type: String,
    #[arg(long, default_value = "@machine")]
    namespace: String,
    #[arg(long, default_value = "machine-organism")]
    task: String,
    #[arg(long)]
    session: Option<String>,
    #[arg(
        long,
        default_value = "Resume repository execution from compact continuity state and explicit scars."
    )]
    objective: String,
    #[arg(long, default_value_t = 192)]
    budget_tokens: usize,
    #[arg(long, default_value_t = 16)]
    candidate_limit: usize,
    #[arg(long, value_enum, default_value_t = SnapshotResolution::Medium)]
    resolution: SnapshotResolution,
    #[arg(long)]
    watch_secs: Option<u64>,
    #[arg(long)]
    heartbeat_secs: Option<u64>,
    #[arg(long)]
    decision: Vec<String>,
    #[arg(long)]
    constraint: Vec<String>,
    #[arg(long)]
    incident: Vec<String>,
    #[arg(long)]
    scar: Vec<String>,
}

#[derive(Debug, Args, Clone)]
struct HeartbeatArgs {
    #[arg(long)]
    attachment_id: Option<String>,
    #[arg(long)]
    agent: Option<String>,
    #[arg(long)]
    namespace: Option<String>,
    #[arg(long)]
    context_id: Option<String>,
    #[arg(long)]
    every_secs: Option<u64>,
}

#[derive(Debug, Args, Clone)]
struct OrganismArgs {
    #[arg(long, default_value = "@machine")]
    namespace: String,
    #[arg(long, default_value = "machine-organism")]
    task: String,
    #[arg(long)]
    session: Option<String>,
    #[arg(
        long,
        default_value = "Keep lightweight organism agents attached to the shared brain and translate pressure into continuity state."
    )]
    objective: String,
    #[arg(long)]
    pulse_secs: Option<u64>,
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    pulse_count: Option<u64>,
    #[arg(long, default_value_t = 180)]
    lease_secs: u64,
    #[arg(long, default_value_t = 256)]
    budget_tokens: usize,
    #[arg(long, default_value_t = 24)]
    candidate_limit: usize,
}

#[derive(Debug, Args, Clone)]
struct DispatchInitArgs {
    #[arg(long)]
    database_url: Option<String>,
    #[arg(long, default_value = DEFAULT_DISPATCH_NOTIFY_CHANNEL)]
    notify_channel: String,
    #[arg(long, default_value_t = 120)]
    worker_stale_secs: u64,
}

#[derive(Debug, Args, Clone)]
struct DispatchCompleteArgsCli {
    #[arg(long, value_enum, default_value_t = DispatchSignalKind::TaskComplete)]
    kind: DispatchSignalKind,
    #[arg(long)]
    context_id: Option<String>,
    #[arg(long)]
    namespace: Option<String>,
    #[arg(long)]
    task: Option<String>,
    #[arg(long)]
    agent: String,
    #[arg(long)]
    title: String,
    #[arg(long)]
    result: String,
    #[arg(long)]
    objective: Option<String>,
    #[arg(long, default_value_t = 0.75)]
    quality: f64,
    #[arg(long, value_delimiter = ',')]
    failures: Vec<String>,
    #[arg(long)]
    selector_json: Option<String>,
    #[arg(long)]
    target_role: Option<String>,
    #[arg(long, value_enum)]
    preferred_tier: Option<DispatchWorkerTier>,
    #[arg(long)]
    reason: Option<String>,
    #[arg(long, default_value = "{}")]
    extra_json: String,
    #[arg(long, value_enum, default_value_t = SnapshotResolution::Medium)]
    resolution: SnapshotResolution,
    #[arg(long, default_value_t = 256)]
    token_budget: usize,
    #[arg(long, default_value_t = 24)]
    candidate_limit: usize,
}

#[derive(Debug, Args, Clone)]
struct DispatchWorkerArgs {
    #[arg(long)]
    worker_id: String,
    #[arg(long)]
    display_name: String,
    #[arg(long)]
    role: String,
    #[arg(long)]
    agent_type: String,
    #[arg(long, value_enum)]
    tier: DispatchWorkerTier,
    #[arg(long)]
    model: String,
    #[arg(long, value_delimiter = ',')]
    capabilities: Vec<String>,
    #[arg(long, default_value_t = 1)]
    max_parallelism: usize,
    #[arg(long)]
    focus: Option<String>,
    #[arg(long)]
    namespace: Option<String>,
    #[arg(long)]
    task: Option<String>,
    #[arg(long, default_value = "listening")]
    status: String,
    #[arg(long, default_value = "{}")]
    metadata_json: String,
    #[arg(long)]
    attached_repo_root: Option<String>,
    #[arg(long)]
    attached_branch: Option<String>,
    #[arg(long)]
    attached_label: Option<String>,
    #[arg(long)]
    attached_resource: Option<String>,
    #[arg(long, default_value_t = false)]
    derive_attached_lane_from_badge: bool,
}

#[derive(Debug, Args, Clone)]
struct DispatchClaimArgs {
    #[command(flatten)]
    worker: DispatchWorkerArgs,
    #[arg(long, default_value_t = 256)]
    token_budget: usize,
    #[arg(long, default_value_t = 24)]
    candidate_limit: usize,
}

#[derive(Debug, Args, Clone)]
struct DispatchListenArgs {
    #[command(flatten)]
    claim: DispatchClaimArgs,
    #[arg(long, default_value_t = 60)]
    timeout_secs: u64,
}

#[derive(Debug, Args, Clone)]
struct DispatchAckArgs {
    #[arg(long)]
    worker_id: String,
    #[arg(long)]
    assignment_id: String,
    #[arg(long, default_value_t = false)]
    failed: bool,
}

#[derive(Debug, Args, Clone)]
struct DispatchStatsArgs {
    #[arg(long, default_value_t = false)]
    no_color: bool,
}

#[derive(Debug, Args, Clone)]
struct CodexInstallGlobalArgs {
    #[arg(long)]
    codex_home: Option<PathBuf>,
    #[arg(long)]
    config_path: Option<PathBuf>,
    #[arg(long)]
    machine_root: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_CODEX_GLOBAL_SERVER_NAME)]
    server_name: String,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BenchMode {
    Continuity,
    Legacy,
}

#[derive(Debug, Subcommand)]
enum MemoryCommand {
    List {
        #[arg(long, value_enum)]
        layer: Option<MemoryLayer>,
        #[arg(long, default_value_t = 25)]
        limit: usize,
    },
}

fn main() -> Result<()> {
    EngineTelemetry::init_logging();
    let cli = Cli::parse();
    let root = cli.root.clone();

    match cli.command {
        Command::Ingest(args) => {
            let engine = open_engine(&root)?;
            let content = read_payload(args.text)?;
            let attributes = serde_json::from_str(&args.attrs_json)
                .with_context(|| format!("parsing attrs-json: {}", args.attrs_json))?;
            let dimensions = parse_dimensions(&args.dimensions_json)?;
            let manifest = engine.ingest(EventInput {
                kind: args.kind,
                agent_id: args.agent,
                agent_role: args.agent_role,
                timestamp: args.timestamp,
                session_id: args.session,
                task_id: args.task,
                project_id: args.project,
                goal_id: args.goal,
                run_id: args.run,
                namespace: args.namespace,
                environment: args.environment,
                source: args.source,
                scope: args.scope,
                tags: args.tags,
                dimensions,
                content,
                attributes,
            })?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        Command::Replay(args) => {
            let engine = open_engine(&root)?;
            let rows = if let Some(selector_json) = args.selector_json {
                let selector = parse_selector(&selector_json)?;
                engine.replay_by_selector(&selector, args.limit)?
            } else {
                engine.replay(args.session.as_deref(), args.limit)?
            };
            println!("{}", serde_json::to_string_pretty(&rows)?);
        }
        Command::Memory(args) => match args.command {
            MemoryCommand::List { layer, limit } => {
                let engine = open_engine(&root)?;
                let rows = engine.list_memory(layer, limit)?;
                println!("{}", serde_json::to_string_pretty(&rows)?);
            }
        },
        Command::Query(args) => {
            let engine = open_engine(&root)?;
            let query_text = read_payload(args.text)?;
            let pack = engine.build_context_pack(QueryInput {
                agent_id: args.agent,
                session_id: args.session,
                task_id: args.task,
                namespace: args.namespace,
                objective: args.objective,
                selector: args
                    .selector_json
                    .as_deref()
                    .map(parse_selector)
                    .transpose()?,
                view_id: args.view,
                query_text,
                budget_tokens: args.budget_tokens,
                candidate_limit: args.candidate_limit,
            })?;
            println!("{}", serde_json::to_string_pretty(&pack)?);
        }
        Command::Annotate(args) => {
            let engine = open_engine(&root)?;
            let dimensions = parse_dimensions(&args.dimensions_json)?;
            let result = engine.annotate_item(&args.item_type, &args.item_id, &dimensions)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::Relate(args) => {
            let engine = open_engine(&root)?;
            let relation = engine.relate_items(
                &args.source_id,
                &args.target_id,
                &args.relation,
                args.weight,
                serde_json::from_str(&args.attrs_json)?,
            )?;
            println!("{}", serde_json::to_string_pretty(&relation)?);
        }
        Command::View(args) => match args.command {
            ViewCommand::Create {
                op,
                owner_agent,
                namespace,
                objective,
                selector_json,
                source_view,
                resolution,
                limit,
            } => {
                let engine = open_engine(&root)?;
                let selectors = serde_json::from_str(&selector_json)
                    .with_context(|| format!("parsing selector-json: {selector_json}"))?;
                let view = engine.materialize_view(ViewInput {
                    op,
                    owner_agent_id: owner_agent,
                    namespace,
                    objective,
                    selectors,
                    source_view_ids: source_view,
                    resolution,
                    limit: Some(limit),
                })?;
                println!("{}", serde_json::to_string_pretty(&view)?);
            }
            ViewCommand::Get { id } => {
                let engine = open_engine(&root)?;
                let view = engine.get_view(&id)?;
                println!("{}", serde_json::to_string_pretty(&view)?);
            }
            ViewCommand::Explain { id } => {
                let engine = open_engine(&root)?;
                let manifest = engine.explain_view(&id)?;
                println!("{}", serde_json::to_string_pretty(&manifest)?);
            }
            ViewCommand::Fork { id, owner_agent } => {
                let engine = open_engine(&root)?;
                let fork = engine.fork_view(&id, owner_agent)?;
                println!("{}", serde_json::to_string_pretty(&fork)?);
            }
        },
        Command::Handoff(args) => {
            let engine = open_engine(&root)?;
            let query_text = read_payload(args.text)?;
            let handoff = engine.create_handoff(HandoffInput {
                from_agent_id: args.from_agent,
                to_agent_id: args.to_agent,
                reason: args.reason,
                query_text,
                budget_tokens: args.budget_tokens,
                view_id: args.view,
                selector: args
                    .selector_json
                    .as_deref()
                    .map(parse_selector)
                    .transpose()?,
                objective: args.objective,
                namespace: args.namespace,
            })?;
            println!("{}", serde_json::to_string_pretty(&handoff)?);
        }
        Command::Subscribe(args) => match args.command {
            SubscribeCommand::Create {
                agent,
                name,
                selector_json,
            } => {
                let engine = open_engine(&root)?;
                let selector = parse_selector(&selector_json)?;
                let subscription = engine.create_subscription(SubscriptionInput {
                    agent_id: agent,
                    name,
                    selector,
                })?;
                println!("{}", serde_json::to_string_pretty(&subscription)?);
            }
            SubscribeCommand::Poll { id, limit } => {
                let engine = open_engine(&root)?;
                let poll = engine.poll_subscription(&id, limit)?;
                println!("{}", serde_json::to_string_pretty(&poll)?);
            }
        },
        Command::Explain(args) => match args.command {
            ExplainCommand::Pack { id } => {
                let engine = open_engine(&root)?;
                let manifest = engine.explain_context_pack(&id)?;
                println!("{}", serde_json::to_string_pretty(&manifest)?);
            }
            ExplainCommand::Handoff { id } => {
                let engine = open_engine(&root)?;
                let manifest = engine.explain_handoff(&id)?;
                println!("{}", serde_json::to_string_pretty(&manifest)?);
            }
        },
        Command::Metrics => {
            let engine = open_engine(&root)?;
            let snapshot = engine.metrics_snapshot()?;
            print!("{}", snapshot.prometheus_text);
        }
        Command::LongMemEval(args) => match args.command {
            LongMemEvalCommand::Run(args) => {
                let work_dir = args
                    .work_dir
                    .clone()
                    .unwrap_or_else(|| root.join("longmemeval").join("work"));
                let reader_provider: LongMemEvalReaderProvider = args.reader_provider.into();
                let reader_method: LongMemEvalReaderMethod = args.reader_method.into();
                let reader_endpoint = args.reader_endpoint.clone().unwrap_or_else(|| {
                    default_longmemeval_reader_endpoint(args.reader_provider).to_string()
                });
                let report = run_longmemeval(LongMemEvalRunConfig {
                    dataset_path: args.dataset,
                    output_path: args.output,
                    work_dir,
                    namespace_prefix: args.namespace_prefix,
                    reader_provider,
                    reader_method,
                    reader_endpoint,
                    reader_model: args.reader_model,
                    reader_api_key_env: args.reader_api_key_env.or_else(|| {
                        default_longmemeval_reader_api_key_env(args.reader_provider)
                            .map(str::to_string)
                    }),
                    reader_timeout_secs: args.reader_timeout_secs,
                    reader_num_predict: args.reader_num_predict,
                    reader_max_retries: args.reader_max_retries,
                    reader_retry_backoff_secs: args.reader_retry_backoff_secs,
                    budget_tokens: args.budget_tokens,
                    candidate_limit: args.candidate_limit,
                    offset: args.offset,
                    max_cases: args.max_cases,
                    question_ids: args.question_ids,
                    question_types: args.question_types,
                })?;
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
            LongMemEvalCommand::Evaluate(args) => {
                let report = evaluate_longmemeval(LongMemEvalEvaluateConfig {
                    repo_path: args.repo,
                    predictions_path: args.predictions,
                    dataset_path: args.dataset,
                    python_bin: args.python_bin,
                    judge_model: args.judge_model,
                })?;
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
        },
        Command::Serve(args) => {
            let engine = open_engine(&root)?;
            block_on(serve(engine.clone(), args.addr))?;
        }
        Command::Mcp => {
            let engine = open_engine(&root)?;
            serve_stdio(engine.clone())?;
        }
        Command::Wrap(args) => {
            let engine = open_engine(&root)?;
            let exit_code = block_on(run_wrapped(engine.clone(), args))?;
            std::process::exit(exit_code);
        }
        Command::Demo(args) => {
            let engine = open_engine(&root)?;
            let result = block_on(run_demo(engine.clone(), args))?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::Bench(args) => {
            let engine = open_engine(&root)?;
            let result = block_on(run_bench(engine.clone(), cli.root.clone(), args))?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::BenchMarket(args) => match args.command {
            BenchMarketCommand::Export(args) => {
                let engine = open_engine(&root)?;
                let result = block_on(run_bench_market_export(
                    engine.clone(),
                    cli.root.clone(),
                    args,
                ))?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            BenchMarketCommand::Evaluate(args) => {
                let result = run_bench_market_evaluate(args)?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            BenchMarketCommand::Answer(args) => {
                let result = run_bench_market_answer(args)?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            BenchMarketCommand::JudgeExport(args) => {
                let result = run_bench_market_judge_export(cli.root.clone(), args)?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            BenchMarketCommand::JudgeEvaluate(args) => {
                let result = run_bench_market_judge_evaluate(args)?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            BenchMarketCommand::JudgeCompare(args) => {
                let result = run_bench_market_judge_compare(args)?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            BenchMarketCommand::JudgePackCompare(args) => {
                let result = run_bench_market_judge_pack_compare(args)?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            BenchMarketCommand::JudgeCalibrate(args) => {
                let result = run_bench_market_judge_calibrate(args)?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            BenchMarketCommand::JudgeDisagreement(args) => {
                let result = run_bench_market_judge_disagreement(args)?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
        },
        Command::Uci(args) => {
            let engine = open_engine(&root)?;
            let request_text = read_payload(args.json)?;
            let request: UciRequest = serde_json::from_str(&request_text)
                .with_context(|| format!("parsing uci request: {request_text}"))?;
            let response = engine.handle_request(request)?;
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
        Command::Dogfood(args) => match args.command {
            DogfoodCommand::SyncRepo(args) => {
                let engine = open_engine(&root)?;
                let result = block_on(run_repo_sync(
                    engine.clone(),
                    resolve_repo_root(args.repo_root.clone())?,
                    args,
                ))?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            DogfoodCommand::Heartbeat(args) => {
                let engine = open_engine(&root)?;
                let result = block_on(run_attachment_heartbeat(engine.clone(), args))?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            DogfoodCommand::Organism(args) => {
                let engine = open_engine(&root)?;
                let session_id = args
                    .session
                    .clone()
                    .unwrap_or_else(|| format!("organism-{}", Uuid::now_v7()));
                let result = block_on(run_organism_choir(
                    engine.clone(),
                    OrganismChorusConfig {
                        namespace: args.namespace,
                        task_id: args.task,
                        objective: args.objective,
                        session_id,
                        pulse_secs: args.pulse_secs,
                        pulse_count: args.pulse_count,
                        lease_secs: args.lease_secs,
                        token_budget: args.budget_tokens,
                        candidate_limit: args.candidate_limit,
                    },
                ))?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
        },
        Command::Dispatch(args) => match args.command {
            DispatchCommand::Init(args) => {
                let database_url = resolve_dispatch_database_url(args.database_url)?;
                let notify_channel = args.notify_channel.clone();
                let _spine = DispatchSpine::init(
                    &root,
                    DispatchConfig {
                        database_url,
                        notify_channel: notify_channel.clone(),
                        worker_stale_secs: args.worker_stale_secs,
                    },
                )?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "config_path": root.join("data/dispatch-config.json"),
                        "metrics_preview": DispatchSpine::render_metrics(&root),
                        "notify_channel": notify_channel,
                    }))?
                );
            }
            DispatchCommand::Complete(args) => {
                let engine = open_engine(&root)?;
                let spine = DispatchSpine::from_root_required(&root)?;
                let context = engine.read_context(ReadContextInput {
                    context_id: args.context_id,
                    namespace: args
                        .namespace
                        .clone()
                        .or_else(|| Some(MACHINE_NAMESPACE_ALIAS.to_string())),
                    task_id: args
                        .task
                        .clone()
                        .or_else(|| Some(DEFAULT_MACHINE_TASK_ID.to_string())),
                    objective: args
                        .objective
                        .clone()
                        .unwrap_or_else(|| "publish dispatch completion signal".to_string()),
                    token_budget: args.token_budget.max(64),
                    selector: args
                        .selector_json
                        .as_deref()
                        .map(parse_selector)
                        .transpose()?,
                    agent_id: Some(args.agent.clone()),
                    session_id: None,
                    view_id: None,
                    include_resolved: true,
                    candidate_limit: args.candidate_limit.max(8),
                })?;
                let objective = args
                    .objective
                    .clone()
                    .unwrap_or_else(|| context.context.objective.clone());
                let outcome = engine.record_outcome(ice::continuity::OutcomeInput {
                    context_id: context.context.id.clone(),
                    agent_id: args.agent.clone(),
                    title: args.title,
                    result: args.result.clone(),
                    quality: args.quality.clamp(0.0, 1.0),
                    pack_id: Some(context.pack.id.clone()),
                    used_memory_ids: context
                        .pack
                        .items
                        .iter()
                        .map(|item| item.memory_id.clone())
                        .collect(),
                    confirmed_memory_ids: Vec::new(),
                    contradicted_memory_ids: Vec::new(),
                    failures: args.failures.clone(),
                    dimensions: Vec::new(),
                    extra: serde_json::json!({
                        "dispatch": true,
                        "reason": args.reason,
                    }),
                })?;
                let snapshot = engine.snapshot(SnapshotInput {
                    context_id: Some(context.context.id.clone()),
                    namespace: Some(context.context.namespace.clone()),
                    task_id: Some(context.context.task_id.clone()),
                    objective: Some(objective.clone()),
                    selector: args
                        .selector_json
                        .as_deref()
                        .map(parse_selector)
                        .transpose()?,
                    resolution: args.resolution,
                    token_budget: args.token_budget,
                    candidate_limit: args.candidate_limit,
                    owner_agent_id: Some(args.agent.clone()),
                })?;
                let machine = engine.identify_machine()?;
                let signal = spine.publish_signal(
                    &machine,
                    PublishDispatchSignalInput {
                        kind: args.kind,
                        from_agent_id: args.agent,
                        context_id: context.context.id,
                        namespace: context.context.namespace,
                        task_id: context.context.task_id,
                        snapshot_id: Some(snapshot.id.clone()),
                        objective,
                        target_role: args.target_role,
                        preferred_tier: args.preferred_tier,
                        reason: args.reason,
                        extra: serde_json::json!({
                            "outcome_id": outcome.id,
                            "quality": args.quality,
                            "failures": args.failures,
                            "extra": serde_json::from_str::<serde_json::Value>(&args.extra_json)
                                .with_context(|| format!("parsing extra-json: {}", args.extra_json))?,
                        }),
                    },
                )?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "signal": signal,
                        "snapshot_id": snapshot.id,
                        "outcome_id": outcome.id,
                    }))?
                );
            }
            DispatchCommand::Heartbeat(args) => {
                let engine = open_engine(&root)?;
                let spine = DispatchSpine::from_root_required(&root)?;
                let derive_attached_lane_from_badge = args.derive_attached_lane_from_badge;
                let worker = enrich_dispatch_worker_from_badges(
                    engine.as_ref(),
                    parse_dispatch_worker(args)?,
                    derive_attached_lane_from_badge,
                )?;
                let result = spine.upsert_worker(worker)?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            DispatchCommand::Claim(args) => {
                let engine = open_engine(&root)?;
                let spine = DispatchSpine::from_root_required(&root)?;
                let derive_attached_lane_from_badge = args.worker.derive_attached_lane_from_badge;
                let worker = enrich_dispatch_worker_from_badges(
                    engine.as_ref(),
                    parse_dispatch_worker(args.worker)?,
                    derive_attached_lane_from_badge,
                )?;
                let result = spine.claim_for_worker(
                    engine.as_ref(),
                    worker,
                    args.token_budget,
                    args.candidate_limit,
                )?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            DispatchCommand::Listen(args) => {
                let engine = open_engine(&root)?;
                let spine = DispatchSpine::from_root_required(&root)?;
                let derive_attached_lane_from_badge =
                    args.claim.worker.derive_attached_lane_from_badge;
                let worker = enrich_dispatch_worker_from_badges(
                    engine.as_ref(),
                    parse_dispatch_worker(args.claim.worker)?,
                    derive_attached_lane_from_badge,
                )?;
                let result = spine.wait_and_claim_for_worker(
                    engine.as_ref(),
                    worker,
                    args.claim.token_budget,
                    args.claim.candidate_limit,
                    args.timeout_secs,
                )?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            DispatchCommand::Ack(args) => {
                let spine = DispatchSpine::from_root_required(&root)?;
                let result = spine.complete_assignment(CompleteAssignmentInput {
                    assignment_id: args.assignment_id,
                    worker_id: Some(args.worker_id),
                    failed: args.failed,
                })?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            DispatchCommand::Stats(_) => {
                print!("{}", DispatchSpine::render_metrics(&root));
            }
        },
        Command::Codex(args) => match args.command {
            CodexCommand::InstallGlobal(args) => {
                let result = install_global_mcp(CodexGlobalInstallRequest {
                    codex_home: args.codex_home,
                    config_path: args.config_path,
                    machine_root: args.machine_root,
                    server_name: args.server_name,
                })?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
        },
        Command::Claude(args) => match args.command {
            ClaudeCommand::InstallGlobal(args) => {
                let result = install_claude_code(ClaudeCodeInstallRequest {
                    organism_root: args.root,
                    server_name: args.server_name,
                    code_config_path: args.code_config,
                })?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            ClaudeCommand::Status(args) => {
                let result = claude_code_status(ClaudeCodeStatusRequest {
                    server_name: args.server_name,
                    code_config_path: args.code_config,
                    organism_root: args.root,
                })?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            ClaudeCommand::Uninstall(args) => {
                let result = uninstall_claude_code(ClaudeCodeUninstallRequest {
                    server_name: args.server_name,
                    code_config_path: args.code_config,
                })?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
        },
        Command::Openhands(args) => match args.command {
            OpenHandsCommand::InstallGlobal(args) => {
                let result = install_openhands(OpenHandsInstallRequest {
                    organism_root: args.root,
                    server_name: args.server_name,
                    mcp_config_path: args.mcp_config,
                })?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            OpenHandsCommand::Status(args) => {
                let result = openhands_status(OpenHandsStatusRequest {
                    server_name: args.server_name,
                    mcp_config_path: args.mcp_config,
                    organism_root: args.root,
                })?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            OpenHandsCommand::Uninstall(args) => {
                let result = uninstall_openhands(OpenHandsUninstallRequest {
                    server_name: args.server_name,
                    mcp_config_path: args.mcp_config,
                })?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
        },
        Command::Opencode(args) => match args.command {
            OpenCodeCommand::InstallGlobal(args) => {
                let result = install_opencode(OpenCodeInstallRequest {
                    organism_root: args.root,
                    server_name: args.server_name,
                    config_path: args.config,
                })?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            OpenCodeCommand::Status(args) => {
                let result = opencode_status(OpenCodeStatusRequest {
                    server_name: args.server_name,
                    config_path: args.config,
                    organism_root: args.root,
                })?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            OpenCodeCommand::Uninstall(args) => {
                let result = uninstall_opencode(OpenCodeUninstallRequest {
                    server_name: args.server_name,
                    config_path: args.config,
                })?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
        },
        Command::Goose(args) => match args.command {
            GooseCommand::InstallGlobal(args) => {
                let result = install_goose(GooseInstallRequest {
                    organism_root: args.root,
                    server_name: args.server_name,
                    config_path: args.config,
                })?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            GooseCommand::Status(args) => {
                let result = goose_status(GooseStatusRequest {
                    server_name: args.server_name,
                    config_path: args.config,
                    organism_root: args.root,
                })?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            GooseCommand::Uninstall(args) => {
                let result = uninstall_goose(GooseUninstallRequest {
                    server_name: args.server_name,
                    config_path: args.config,
                })?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
        },
    }

    Ok(())
}

fn open_engine(root: &Path) -> Result<Arc<Engine>> {
    Ok(Arc::new(Engine::open(root)?))
}

fn block_on<F, T>(future: F) -> Result<T>
where
    F: Future<Output = Result<T>>,
{
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building Tokio runtime")?
        .block_on(future)
}

fn read_payload(text: Option<String>) -> Result<String> {
    if let Some(text) = text {
        return Ok(text);
    }
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    if buffer.trim().is_empty() {
        anyhow::bail!("event content must be provided with --text or stdin");
    }
    Ok(buffer)
}

fn parse_dimensions(json: &str) -> Result<Vec<DimensionValue>> {
    serde_json::from_str(json).with_context(|| format!("parsing dimensions-json: {json}"))
}

fn parse_selector(json: &str) -> Result<Selector> {
    serde_json::from_str(json).with_context(|| format!("parsing selector-json: {json}"))
}

fn resolve_repo_root(path: Option<PathBuf>) -> Result<PathBuf> {
    let repo_root = match path {
        Some(path) if path.is_absolute() => path,
        Some(path) => std::env::current_dir()?.join(path),
        None => std::env::current_dir()?,
    };
    repo_root
        .canonicalize()
        .with_context(|| format!("canonicalizing repo root {}", repo_root.display()))
}

fn resolve_dispatch_database_url(value: Option<String>) -> Result<String> {
    if let Some(value) = value.filter(|item| !item.trim().is_empty()) {
        return Ok(value);
    }
    std::env::var("ICE_DISPATCH_DATABASE_URL").context(
        "dispatch database URL must be provided with --database-url or ICE_DISPATCH_DATABASE_URL",
    )
}

fn parse_dispatch_worker(args: DispatchWorkerArgs) -> Result<DispatchWorkerUpsertInput> {
    let mut metadata: serde_json::Value = serde_json::from_str(&args.metadata_json)
        .with_context(|| format!("parsing metadata-json: {}", args.metadata_json))?;
    apply_dispatch_attached_lane_args(&mut metadata, &args)?;
    Ok(DispatchWorkerUpsertInput {
        worker_id: args.worker_id,
        display_name: args.display_name,
        role: args.role,
        tier: args.tier,
        agent_type: args.agent_type,
        model: args.model,
        capabilities: args.capabilities,
        max_parallelism: args.max_parallelism,
        status: args.status,
        focus: args.focus.unwrap_or_default(),
        namespace: args.namespace,
        task_id: args.task,
        metadata,
    })
}

fn apply_dispatch_attached_lane_args(
    metadata: &mut serde_json::Value,
    args: &DispatchWorkerArgs,
) -> Result<()> {
    let has_metadata_lane = metadata.get("attached_lane").is_some();
    match (
        args.attached_repo_root.as_deref(),
        args.attached_branch.as_deref(),
    ) {
        (None, None) => Ok(()),
        (Some(_), None) | (None, Some(_)) => {
            anyhow::bail!("--attached-repo-root and --attached-branch must be provided together")
        }
        (Some(repo_root), Some(branch)) => {
            if has_metadata_lane {
                anyhow::bail!(
                    "attached lane cannot be supplied via both --metadata-json and --attached-* flags"
                );
            }
            let metadata_object = metadata.as_object_mut().context(
                "metadata-json must decode to a JSON object when using --attached-* flags",
            )?;
            let repo_name = std::path::Path::new(repo_root)
                .file_name()
                .and_then(|value| value.to_str())
                .filter(|value| !value.trim().is_empty())
                .unwrap_or(repo_root);
            let label = args
                .attached_label
                .clone()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| format!("{repo_name} @ {branch}"));
            let resource = args
                .attached_resource
                .clone()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| format!("repo/{repo_name}/{branch}"));
            metadata_object.insert(
                "attached_lane".to_string(),
                serde_json::json!({
                    "projection_id": format!("repo:{repo_root}:{branch}"),
                    "projection_kind": "repo",
                    "label": label,
                    "resource": resource,
                    "repo_root": repo_root,
                    "branch": branch,
                    "task_id": args.task,
                }),
            );
            metadata_object.insert(
                "attached_lane_source".to_string(),
                serde_json::to_value(DispatchAttachedLaneSource::ExplicitCli)?,
            );
            Ok(())
        }
    }
}

fn enrich_dispatch_worker_from_badges(
    engine: &Engine,
    mut worker: DispatchWorkerUpsertInput,
    derive_attached_lane_from_badge: bool,
) -> Result<DispatchWorkerUpsertInput> {
    if !derive_attached_lane_from_badge {
        return Ok(worker);
    }
    if worker.metadata.get("attached_lane").is_some() || !worker.metadata.is_object() {
        return Ok(worker);
    }
    let Some(namespace) = worker.namespace.as_deref() else {
        return Ok(worker);
    };
    let badges = engine.list_agent_badges(Some(namespace), worker.task_id.as_deref())?;
    let inferred_lane = derive_dispatch_attached_lane_from_badges(&worker, &badges);
    if let Some(lane) = inferred_lane {
        worker
            .metadata
            .as_object_mut()
            .expect("dispatch worker metadata object checked above")
            .insert("attached_lane".to_string(), serde_json::to_value(lane)?);
        worker
            .metadata
            .as_object_mut()
            .expect("dispatch worker metadata object checked above")
            .insert(
                "attached_lane_source".to_string(),
                serde_json::to_value(DispatchAttachedLaneSource::LiveBadgeOptIn)?,
            );
    }
    Ok(worker)
}

fn derive_dispatch_attached_lane_from_badges(
    worker: &DispatchWorkerUpsertInput,
    badges: &[AgentBadgeRecord],
) -> Option<CoordinationProjectedLane> {
    let mut lanes = BTreeMap::new();
    for badge in badges.iter().filter(|badge| badge.connected) {
        if badge.agent_id != worker.worker_id {
            continue;
        }
        if let Some(task_id) = worker.task_id.as_deref() {
            if badge.task_id.as_deref() != Some(task_id) {
                continue;
            }
        }
        let Some(lane) = derive_dispatch_attached_lane_from_badge(badge) else {
            continue;
        };
        lanes.entry(lane.projection_id.clone()).or_insert(lane);
    }
    if lanes.len() == 1 {
        lanes.into_values().next()
    } else {
        None
    }
}

fn derive_dispatch_attached_lane_from_badge(
    badge: &AgentBadgeRecord,
) -> Option<CoordinationProjectedLane> {
    let repo_root = badge
        .repo_root
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    let branch = badge
        .branch
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .or_else(|| {
            badge
                .resource
                .as_deref()
                .and_then(dispatch_branch_from_repo_resource)
        })?;
    let repo_name = std::path::Path::new(repo_root)
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(repo_root);
    let resource = badge
        .resource
        .clone()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| format!("repo/{repo_name}/{branch}"));
    Some(CoordinationProjectedLane {
        projection_id: format!("repo:{repo_root}:{branch}"),
        projection_kind: "repo".to_string(),
        label: format!("{repo_name} @ {branch}"),
        resource: Some(resource),
        repo_root: Some(repo_root.to_string()),
        branch: Some(branch.to_string()),
        task_id: badge.task_id.clone(),
    })
}

fn dispatch_branch_from_repo_resource(resource: &str) -> Option<&str> {
    let rest = resource.trim().strip_prefix("repo/")?;
    let (_, branch) = rest.split_once('/')?;
    let branch = branch.trim();
    if branch.is_empty() {
        None
    } else {
        Some(branch)
    }
}

#[derive(Debug, Serialize)]
struct RepoSyncFile {
    path: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct RepoSyncState {
    repo_root: String,
    branch: String,
    head: String,
    status: String,
    recent_log: String,
    files: Vec<RepoSyncFile>,
}

#[derive(Debug, Clone, Copy)]
enum RepoSyncGitRequirement {
    Required,
    Optional,
}

fn upsert_repo_sync_badge(
    engine: &Engine,
    attachment_id: &str,
    context_id: &str,
    agent_id: &str,
    state: &RepoSyncState,
    status: &str,
) -> Result<()> {
    let focus = repo_sync_focus(state);
    let headline = format!(
        "{} on {} @ {}",
        state.branch,
        repo_root_name(PathBuf::from(&state.repo_root).as_path()),
        short_head(&state.head)
    );
    engine.upsert_agent_badge(UpsertAgentBadgeInput {
        attachment_id: Some(attachment_id.to_string()),
        agent_id: None,
        namespace: None,
        context_id: Some(context_id.to_string()),
        display_name: Some(agent_id.to_string()),
        status: Some(status.to_string()),
        focus: Some(focus),
        headline: Some(headline),
        resource: Some(repo_sync_lane_resource(state)),
        repo_root: Some(state.repo_root.clone()),
        branch: Some(state.branch.clone()),
        metadata: serde_json::json!({
            "source": "dogfood_sync_repo",
            "head": state.head,
        }),
    })?;
    Ok(())
}

fn claim_repo_sync_lane(
    engine: &Engine,
    context_id: &str,
    attachment_id: &str,
    agent_id: &str,
    state: &RepoSyncState,
    lease_seconds: Option<u64>,
) -> Result<String> {
    let resource = repo_sync_lane_resource(state);
    let claim = engine.claim_work(ClaimWorkInput {
        context_id: context_id.to_string(),
        agent_id: agent_id.to_string(),
        title: format!(
            "Hold repo lane {} @ {}",
            repo_root_name(PathBuf::from(&state.repo_root).as_path()),
            state.branch
        ),
        body: format!(
            "Keep the projected repo lane {} owned while {} is syncing compact state for {}.",
            resource, agent_id, state.repo_root
        ),
        scope: Scope::Shared,
        resources: vec![resource],
        exclusive: true,
        attachment_id: Some(attachment_id.to_string()),
        lease_seconds,
        extra: serde_json::json!({
            "source": "dogfood_sync_repo",
            "repo_root": state.repo_root.clone(),
            "branch": state.branch.clone(),
            "head": state.head.clone(),
        }),
    })?;
    Ok(claim.id)
}

fn repo_sync_lane_resource(state: &RepoSyncState) -> String {
    format!(
        "repo/{}/{}",
        repo_root_name(PathBuf::from(&state.repo_root).as_path()),
        state.branch
    )
}

fn repo_sync_focus(state: &RepoSyncState) -> String {
    let current_cycle = state
        .files
        .iter()
        .find(|item| item.path == "docs/current-cycle.md")
        .map(|item| item.content.as_str())
        .unwrap_or_default();
    let current_slice_lines = markdown_section_lines(current_cycle, "## Current Slice");
    let current_slice_is_complete = current_slice_lines
        .iter()
        .any(|line| line.trim().to_ascii_lowercase().contains("proved locally"));
    if !current_slice_is_complete {
        if let Some(current) = extract_markdown_section_lead(&current_slice_lines) {
            return current;
        }
    }
    let next_slice_lines = markdown_section_lines(current_cycle, "## Next Slice");
    extract_markdown_section_lead(&next_slice_lines)
        .unwrap_or_else(|| format!("watching {} @ {}", state.branch, short_head(&state.head)))
}

fn markdown_section_lines<'a>(markdown: &'a str, heading: &str) -> Vec<&'a str> {
    let mut in_section = false;
    let mut lines = Vec::new();
    for line in markdown.lines() {
        let trimmed = line.trim();
        if trimmed == heading {
            in_section = true;
            continue;
        }
        if in_section {
            if trimmed.starts_with("## ") {
                break;
            }
            lines.push(line);
        }
    }
    lines
}

fn extract_markdown_section_lead(lines: &[&str]) -> Option<String> {
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || !is_actionable_markdown_line(trimmed) {
            continue;
        }
        return Some(strip_markdown_list_prefix(trimmed).to_string());
    }
    None
}

fn is_actionable_markdown_line(line: &str) -> bool {
    line.starts_with("- ") || line.starts_with("* ") || numbered_markdown_prefix_len(line).is_some()
}

fn strip_markdown_list_prefix(line: &str) -> &str {
    if let Some(rest) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")) {
        return rest.trim();
    }
    if let Some(prefix_len) = numbered_markdown_prefix_len(line) {
        return line[prefix_len..].trim();
    }
    line.trim()
}

fn numbered_markdown_prefix_len(line: &str) -> Option<usize> {
    let digit_count = line.chars().take_while(|ch| ch.is_ascii_digit()).count();
    if digit_count == 0 {
        return None;
    }
    let rest = &line[digit_count..];
    if let Some(stripped) = rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") ")) {
        return Some(line.len() - stripped.len());
    }
    None
}

fn short_head(head: &str) -> String {
    head.chars().take(8).collect()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::process::ExitStatusExt;
    use std::path::PathBuf;

    use super::{
        BenchMarketJudgeCalibrateArgs, BenchMarketJudgeCompareArgs,
        BenchMarketJudgeDisagreementArgs, BenchMarketJudgePackCompareArgs, BenchMarketProvider,
        DispatchWorkerArgs, DispatchWorkerTier, NamedJudgeReportArg, RepoSyncFile,
        RepoSyncGitRequirement, RepoSyncState, build_market_head_answer_prompt,
        default_market_head_model, derive_dispatch_attached_lane_from_badges,
        describe_claude_failure, extract_claude_structured_output, extract_markdown_section_lead,
        markdown_section_lines, parse_dispatch_worker, parse_named_judge_report, repo_sync_focus,
        repo_sync_lane_resource, resolve_market_head_manifest_path, resolve_repo_sync_git_text,
        run_bench_market_judge_calibrate, run_bench_market_judge_compare,
        run_bench_market_judge_disagreement, run_bench_market_judge_pack_compare,
    };
    use chrono::Utc;
    use ice::continuity::AgentBadgeRecord;
    use ice::market_head::{
        MarketHeadChallengeEvaluationReport, MarketHeadChallengeSummary,
        MarketHeadJudgeComparisonEntry, MarketHeadJudgeEvaluationReport,
        MarketHeadJudgeSamePackComparisonReport, MarketHeadJudgeSummary,
    };
    use serde_json::json;
    use uuid::Uuid;

    #[test]
    fn extract_markdown_section_lead_skips_narrative_prefixes() {
        let markdown = r#"
## Next Slice

After the live agent badge registry:

1. teach the brain to project multiple repo and task lanes beneath the machine identity without making the machine organism itself noisy
2. make the badge registry speak in those projected repo/task lanes
"#;

        let lines = markdown_section_lines(markdown, "## Next Slice");
        assert_eq!(
            extract_markdown_section_lead(&lines).as_deref(),
            Some(
                "teach the brain to project multiple repo and task lanes beneath the machine identity without making the machine organism itself noisy"
            )
        );
    }

    #[test]
    fn build_market_head_answer_prompt_keeps_prompt_and_template_visible() {
        let prompt = build_market_head_answer_prompt(
            "Operator prompt body",
            "{\"critical_facts\":[],\"next_step\":\"\"}",
        );

        assert!(prompt.contains("Operator prompt body"));
        assert!(prompt.contains("Template JSON shape:"));
        assert!(prompt.contains("\"critical_facts\""));
        assert!(prompt.contains("Return exactly one JSON object"));
    }

    #[test]
    fn extract_claude_structured_output_returns_valid_payload() {
        let stdout = json!({
            "is_error": false,
            "structured_output": {
                "critical_facts": ["pf1"],
                "decision_summary": "Keep the proof path alive",
                "next_step": "Run the evaluator",
            }
        })
        .to_string();

        let parsed = extract_claude_structured_output(&stdout).expect("parse Claude output");

        assert_eq!(parsed["critical_facts"][0].as_str(), Some("pf1"));
        assert_eq!(
            parsed["decision_summary"].as_str(),
            Some("Keep the proof path alive")
        );
    }

    #[test]
    fn extract_claude_structured_output_rejects_error_envelopes() {
        let stdout = json!({
            "is_error": true,
            "result": "session expired"
        })
        .to_string();

        let error = extract_claude_structured_output(&stdout).expect_err("reject Claude error");
        assert!(error.to_string().contains("session expired"));
    }

    #[test]
    fn describe_claude_failure_reads_result_when_stderr_is_empty() {
        let output = std::process::Output {
            status: std::process::ExitStatus::from_raw(256),
            stdout: json!({
                "is_error": true,
                "result": "You've hit your limit · resets 2am (Europe/London)"
            })
            .to_string()
            .into_bytes(),
            stderr: Vec::new(),
        };

        assert_eq!(
            describe_claude_failure(&output),
            "You've hit your limit · resets 2am (Europe/London)"
        );
    }

    #[test]
    fn resolve_market_head_manifest_prefers_nested_judge_manifest_when_present() {
        let root = std::env::temp_dir().join(format!("ice-judge-manifest-{}", Uuid::now_v7()));
        let judge_root = root.join("market-head-judge");
        fs::create_dir_all(&judge_root).expect("create judge root");
        let judge_manifest = judge_root.join("judge-manifest.json");
        fs::write(&judge_manifest, "{}").expect("write judge manifest");

        let resolved =
            resolve_market_head_manifest_path(&root, None).expect("resolve nested judge manifest");

        assert_eq!(resolved, judge_manifest);

        fs::remove_dir_all(&root).expect("cleanup judge root");
    }

    #[test]
    fn parse_named_judge_report_accepts_head_equals_path() {
        let parsed =
            parse_named_judge_report("codex=/tmp/judge.json").expect("parse named judge report");
        assert_eq!(parsed.judge_head, "codex");
        assert_eq!(parsed.report_path, PathBuf::from("/tmp/judge.json"));
    }

    #[test]
    fn run_bench_market_judge_compare_writes_artifacts() {
        let root = std::env::temp_dir().join(format!("ice-judge-compare-{}", Uuid::now_v7()));
        fs::create_dir_all(&root).expect("create compare root");
        let canonical_path = root.join("canonical.json");
        let claude_judge_path = root.join("claude-judge.json");
        let codex_judge_path = root.join("codex-judge.json");
        let output_dir = root.join("out");

        fs::write(
            &canonical_path,
            serde_json::to_vec_pretty(&MarketHeadChallengeEvaluationReport {
                generated_at: "2026-03-23T06:00:00Z".into(),
                model_name: "claude-external".into(),
                evaluator_pack_path: "/tmp/evaluator.json".into(),
                responses_dir: "/tmp/responses".into(),
                cases: Vec::new(),
                summary: MarketHeadChallengeSummary {
                    class_count: 5,
                    avg_absorption_cfsr: 0.7,
                    avg_absorption_dlf: 0.6,
                    avg_absorption_osr: 1.0,
                    avg_absorption_ras: 0.66,
                    avg_cfsr: 1.0,
                    avg_dlf: 0.6,
                    avg_mpr: 0.14,
                    avg_osr: 1.0,
                    avg_pc: 0.91,
                    avg_ras: 0.72,
                    failed_cases: 0,
                },
            })
            .expect("serialize canonical report"),
        )
        .expect("write canonical report");

        for (path, head, cfsr) in [
            (&claude_judge_path, "claude", 0.9),
            (&codex_judge_path, "codex", 0.97),
        ] {
            fs::write(
                path,
                serde_json::to_vec_pretty(&MarketHeadJudgeEvaluationReport {
                    generated_at: "2026-03-23T06:01:00Z".into(),
                    model_name: format!("{head}-judge"),
                    manifest_path: "/tmp/judge-manifest.json".into(),
                    responses_dir: "/tmp/judge-responses".into(),
                    cases: Vec::new(),
                    summary: MarketHeadJudgeSummary {
                        class_count: 5,
                        avg_judge_cfsr: cfsr,
                        avg_judge_csr: 1.0,
                        avg_judge_dlf: 1.0,
                        avg_judge_osr: 1.0,
                        avg_judge_next_step: 0.1,
                        avg_judge_comprehension: 0.8,
                        failed_cases: 0,
                    },
                })
                .expect("serialize judge report"),
            )
            .expect("write judge report");
        }

        let value = run_bench_market_judge_compare(BenchMarketJudgeCompareArgs {
            challenged_head: "claude".into(),
            canonical_report: canonical_path.clone(),
            judge_reports: vec![
                NamedJudgeReportArg {
                    judge_head: "claude".into(),
                    report_path: claude_judge_path.clone(),
                },
                NamedJudgeReportArg {
                    judge_head: "codex".into(),
                    report_path: codex_judge_path.clone(),
                },
            ],
            output_dir: Some(output_dir.clone()),
        })
        .expect("run same-pack comparison");

        let json_path = output_dir.join("market-head-judge-same-pack-claude.json");
        let markdown_path = output_dir.join("market-head-judge-same-pack-claude.md");
        assert!(json_path.exists());
        assert!(markdown_path.exists());
        let json_path_text = json_path.display().to_string();
        assert_eq!(value["json_path"].as_str(), Some(json_path_text.as_str()));

        let markdown = fs::read_to_string(markdown_path).expect("read markdown");
        assert!(markdown.contains("Challenged head: `claude`"));
        assert!(markdown.contains("| codex | 0.97 | 1.00 | 1.00 | 1.00 | 0.10 | 0.80 | 0 |"));

        fs::remove_dir_all(root).expect("cleanup compare root");
    }

    #[test]
    fn run_bench_market_judge_pack_compare_writes_artifacts() {
        let root = std::env::temp_dir().join(format!("ice-judge-pack-compare-{}", Uuid::now_v7()));
        fs::create_dir_all(&root).expect("create pack compare root");
        let claude_path = root.join("same-pack-claude.json");
        let codex_path = root.join("same-pack-codex.json");
        let output_dir = root.join("out");

        for (path, challenged_head, canonical_cfsr, judges) in [
            (
                &claude_path,
                "claude",
                0.7,
                vec![
                    MarketHeadJudgeComparisonEntry {
                        judge_head: "claude".into(),
                        summary: MarketHeadJudgeSummary {
                            class_count: 5,
                            avg_judge_cfsr: 0.9,
                            avg_judge_csr: 1.0,
                            avg_judge_dlf: 1.0,
                            avg_judge_osr: 1.0,
                            avg_judge_next_step: 0.07,
                            avg_judge_comprehension: 0.79,
                            failed_cases: 0,
                        },
                    },
                    MarketHeadJudgeComparisonEntry {
                        judge_head: "codex".into(),
                        summary: MarketHeadJudgeSummary {
                            class_count: 5,
                            avg_judge_cfsr: 0.97,
                            avg_judge_csr: 1.0,
                            avg_judge_dlf: 1.0,
                            avg_judge_osr: 1.0,
                            avg_judge_next_step: 0.2,
                            avg_judge_comprehension: 0.83,
                            failed_cases: 0,
                        },
                    },
                ],
            ),
            (
                &codex_path,
                "codex",
                0.9,
                vec![MarketHeadJudgeComparisonEntry {
                    judge_head: "claude".into(),
                    summary: MarketHeadJudgeSummary {
                        class_count: 5,
                        avg_judge_cfsr: 0.87,
                        avg_judge_csr: 1.0,
                        avg_judge_dlf: 1.0,
                        avg_judge_osr: 1.0,
                        avg_judge_next_step: 0.07,
                        avg_judge_comprehension: 0.79,
                        failed_cases: 0,
                    },
                }],
            ),
        ] {
            fs::write(
                path,
                serde_json::to_vec_pretty(&MarketHeadJudgeSamePackComparisonReport {
                    generated_at: "2026-03-23T06:20:00Z".into(),
                    challenged_head: challenged_head.into(),
                    canonical: MarketHeadChallengeSummary {
                        class_count: 5,
                        avg_absorption_cfsr: canonical_cfsr,
                        avg_absorption_dlf: 1.0,
                        avg_absorption_osr: 1.0,
                        avg_absorption_ras: 0.78,
                        avg_cfsr: 1.0,
                        avg_dlf: 1.0,
                        avg_mpr: 0.06,
                        avg_osr: 1.0,
                        avg_pc: 0.91,
                        avg_ras: 0.8,
                        failed_cases: 0,
                    },
                    judges,
                })
                .expect("serialize same-pack report"),
            )
            .expect("write same-pack report");
        }

        let value = run_bench_market_judge_pack_compare(BenchMarketJudgePackCompareArgs {
            same_pack_reports: vec![claude_path.clone(), codex_path.clone()],
            output_dir: Some(output_dir.clone()),
        })
        .expect("run pack compare");

        let json_path = output_dir.join("market-head-judge-pack-comparison.json");
        let markdown_path = output_dir.join("market-head-judge-pack-comparison.md");
        assert!(json_path.exists());
        assert!(markdown_path.exists());
        let json_path_text = json_path.display().to_string();
        assert_eq!(value["json_path"].as_str(), Some(json_path_text.as_str()));

        let markdown = fs::read_to_string(markdown_path).expect("read pack markdown");
        assert!(markdown.contains("Market-Head Judge Pack Comparison"));
        assert!(markdown.contains("| claude | codex | 0.70 | 1.00 | 1.00 | 0.78 | 0.06 | 0.97 | 1.00 | 1.00 | 1.00 | 0.20 | 0.83 | 0 |"));

        fs::remove_dir_all(root).expect("cleanup pack compare root");
    }

    #[test]
    fn run_bench_market_judge_calibrate_writes_artifacts() {
        let root = std::env::temp_dir().join(format!("ice-judge-calibrate-{}", Uuid::now_v7()));
        fs::create_dir_all(&root).expect("create calibration root");
        let claude_path = root.join("same-pack-claude.json");
        let codex_path = root.join("same-pack-codex.json");
        let output_dir = root.join("out");

        for (path, challenged_head, canonical_cfsr, canonical_ras, judges) in [
            (
                &claude_path,
                "claude",
                0.70,
                0.76,
                vec![
                    serde_json::json!({
                        "judge_head": "claude",
                        "summary": {
                            "class_count": 5,
                            "avg_judge_cfsr": 0.74,
                            "avg_judge_csr": 1.0,
                            "avg_judge_dlf": 1.0,
                            "avg_judge_osr": 1.0,
                            "avg_judge_next_step": 0.10,
                            "avg_judge_comprehension": 0.78,
                            "failed_cases": 0
                        }
                    }),
                    serde_json::json!({
                        "judge_head": "codex",
                        "summary": {
                            "class_count": 5,
                            "avg_judge_cfsr": 0.84,
                            "avg_judge_csr": 1.0,
                            "avg_judge_dlf": 0.90,
                            "avg_judge_osr": 1.0,
                            "avg_judge_next_step": 0.15,
                            "avg_judge_comprehension": 0.86,
                            "failed_cases": 0
                        }
                    }),
                ],
            ),
            (
                &codex_path,
                "codex",
                0.90,
                0.78,
                vec![
                    serde_json::json!({
                        "judge_head": "claude",
                        "summary": {
                            "class_count": 5,
                            "avg_judge_cfsr": 0.86,
                            "avg_judge_csr": 1.0,
                            "avg_judge_dlf": 1.0,
                            "avg_judge_osr": 1.0,
                            "avg_judge_next_step": 0.08,
                            "avg_judge_comprehension": 0.79,
                            "failed_cases": 0
                        }
                    }),
                    serde_json::json!({
                        "judge_head": "codex",
                        "summary": {
                            "class_count": 5,
                            "avg_judge_cfsr": 0.96,
                            "avg_judge_csr": 1.0,
                            "avg_judge_dlf": 1.0,
                            "avg_judge_osr": 0.96,
                            "avg_judge_next_step": 0.20,
                            "avg_judge_comprehension": 0.83,
                            "failed_cases": 0
                        }
                    }),
                ],
            ),
        ] {
            fs::write(
                path,
                serde_json::json!({
                    "generated_at": "2026-03-23T08:30:00Z",
                    "challenged_head": challenged_head,
                    "canonical": {
                        "class_count": 5,
                        "avg_absorption_cfsr": canonical_cfsr,
                        "avg_absorption_dlf": 1.0,
                        "avg_absorption_osr": 1.0,
                        "avg_absorption_ras": canonical_ras,
                        "avg_cfsr": 1.0,
                        "avg_dlf": 1.0,
                        "avg_mpr": 0.10,
                        "avg_pc": 0.90,
                        "avg_osr": 1.0,
                        "avg_ras": canonical_ras,
                        "failed_cases": 0
                    },
                    "judges": judges
                })
                .to_string(),
            )
            .expect("write same-pack report");
        }

        let value = run_bench_market_judge_calibrate(BenchMarketJudgeCalibrateArgs {
            same_pack_reports: vec![claude_path.clone(), codex_path.clone()],
            output_dir: Some(output_dir.clone()),
        })
        .expect("run judge calibration");

        let json_path = output_dir.join("market-head-judge-calibration.json");
        let markdown_path = output_dir.join("market-head-judge-calibration.md");
        assert!(json_path.exists());
        assert!(markdown_path.exists());
        let json_path_text = json_path.display().to_string();
        assert_eq!(value["json_path"].as_str(), Some(json_path_text.as_str()));

        let markdown = fs::read_to_string(markdown_path).expect("read judge calibration markdown");
        assert!(markdown.contains("Market-Head Judge Calibration Report"));
        assert!(markdown.contains("| claude | claude, codex | 2 | 10 |"));
        assert!(markdown.contains("Aligned"));

        fs::remove_dir_all(root).expect("cleanup calibration root");
    }

    #[test]
    fn run_bench_market_judge_disagreement_writes_artifacts() {
        let root = std::env::temp_dir().join(format!("ice-judge-disagreement-{}", Uuid::now_v7()));
        fs::create_dir_all(&root).expect("create disagreement root");
        let canonical_path = root.join("canonical.json");
        let judge_path = root.join("judge.json");
        let output_dir = root.join("out");
        let challenged_response_path = root.join("challenged-response.json");
        let judge_response_path = root.join("judge-response.json");

        fs::write(
            &challenged_response_path,
            serde_json::json!({
                "summary": "Resumed continuity safely.",
                "critical_facts": [
                    "Primary context is bench / task-agent-swap-survival for this resume.",
                    "Selector pruning dropped required support memory from src/query.rs."
                ],
                "constraints": ["Preserve provenance"],
                "decisions": [{
                    "text": "Use the unified continuity interface",
                    "rationale": "agent swaps route through one continuity interface",
                    "evidence": ["pd1"]
                }],
                "open_hypotheses": [],
                "operational_scars": ["Avoid naive probes"],
                "avoid_repeating": [],
                "next_step": "Benchmark adapter path"
            })
            .to_string(),
        )
        .expect("write challenged response");
        fs::write(
            &judge_response_path,
            serde_json::json!({
                "summary": "The response preserved the main continuity bundle but softened one fact.",
                "critical_facts": [
                    {"index": 0, "score": 3, "reason": "The response preserves the selector/support-memory failure in src/query.rs semantically."},
                    {"index": 1, "score": 2, "reason": "The response says primary context, but the active primary role is softened."}
                ],
                "constraints": [{"index": 0, "score": 3, "reason": "Constraint preserved."}],
                "decisions": [{"index": 0, "score": 2, "reason": "Decision survived."}],
                "scars": [{"index": 0, "score": 3, "reason": "Scar preserved."}],
                "next_step": [{"index": 0, "score": 1, "reason": "Benchmark mention is weak."}]
            })
            .to_string(),
        )
        .expect("write judge response");

        fs::write(
            &canonical_path,
            serde_json::json!({
                "generated_at": "2026-03-23T09:30:00Z",
                "model_name": "codex-external",
                "evaluator_pack_path": "/tmp/evaluator.json",
                "responses_dir": "/tmp/responses",
                "cases": [{
                    "class": "agent_swap_survival",
                    "scenario_id": "agent-swap-survival",
                    "protocol": "handoff-proof(4)",
                    "response_path": challenged_response_path,
                    "status": "ok",
                    "raw_evaluation": {
                        "critical_fact_survival_rate": 0.5,
                        "constraint_survival_rate": 1.0,
                        "context_pack_quality_per_token": 0.5,
                        "decision_lineage_fidelity": 0.5,
                        "duplicate_work_rate": 0.0,
                        "matched_constraints": 1,
                        "matched_critical_facts": 1,
                        "matched_decisions": 1,
                        "matched_scars": 1,
                        "memory_pollution_rate": 0.0,
                        "mistake_recurrence_rate": 0.0,
                        "operational_scar_retention": 1.0,
                        "provenance_coverage": 1.0,
                        "resume_accuracy_score": 0.6,
                        "total_items": 10,
                        "unsupported_items": 0
                    },
                    "evaluation": {
                        "critical_fact_survival_rate": 0.5,
                        "constraint_survival_rate": 1.0,
                        "context_pack_quality_per_token": 0.5,
                        "decision_lineage_fidelity": 0.5,
                        "duplicate_work_rate": 0.0,
                        "matched_constraints": 1,
                        "matched_critical_facts": 1,
                        "matched_decisions": 1,
                        "matched_scars": 1,
                        "memory_pollution_rate": 0.0,
                        "mistake_recurrence_rate": 0.0,
                        "operational_scar_retention": 1.0,
                        "provenance_coverage": 1.0,
                        "resume_accuracy_score": 0.6,
                        "total_items": 10,
                        "unsupported_items": 0
                    },
                    "failure": null
                }],
                "summary": {
                    "class_count": 1,
                    "avg_absorption_cfsr": 0.5,
                    "avg_absorption_dlf": 0.5,
                    "avg_absorption_osr": 1.0,
                    "avg_absorption_ras": 0.6,
                    "avg_cfsr": 0.5,
                    "avg_dlf": 0.5,
                    "avg_osr": 1.0,
                    "avg_ras": 0.6,
                    "avg_mpr": 0.0,
                    "avg_pc": 1.0,
                    "failed_cases": 0
                }
            })
            .to_string(),
        )
        .expect("write canonical report");
        fs::write(
            &judge_path,
            serde_json::json!({
                "generated_at": "2026-03-23T09:31:00Z",
                "model_name": "claude-judge",
                "manifest_path": "/tmp/judge-manifest.json",
                "responses_dir": "/tmp/judge-responses",
                "cases": [{
                    "class": "agent_swap_survival",
                    "scenario_id": "agent-swap-survival",
                    "protocol": "judge://handoff-proof(4)",
                    "response_path": judge_response_path,
                    "status": "ok",
                    "evaluation": {
                        "critical_fact_rate": 1.0,
                        "constraint_rate": 1.0,
                        "decision_rate": 0.83,
                        "scar_rate": 1.0,
                        "next_step_rate": 0.33,
                        "comprehension_score": 0.83
                    },
                    "failure": null
                }],
                "summary": {
                    "class_count": 1,
                    "avg_judge_cfsr": 1.0,
                    "avg_judge_csr": 1.0,
                    "avg_judge_dlf": 0.83,
                    "avg_judge_osr": 1.0,
                    "avg_judge_next_step": 0.33,
                    "avg_judge_comprehension": 0.83,
                    "failed_cases": 0
                }
            })
            .to_string(),
        )
        .expect("write judge report");

        let value = run_bench_market_judge_disagreement(BenchMarketJudgeDisagreementArgs {
            challenged_head: "codex".into(),
            judge_head: "claude".into(),
            canonical_report: canonical_path.clone(),
            judge_report: judge_path.clone(),
            output_dir: Some(output_dir.clone()),
        })
        .expect("run judge disagreement");

        let json_path = output_dir.join("market-head-judge-disagreement-codex-claude.json");
        let markdown_path = output_dir.join("market-head-judge-disagreement-codex-claude.md");
        assert!(json_path.exists());
        assert!(markdown_path.exists());
        let json_path_text = json_path.display().to_string();
        assert_eq!(value["json_path"].as_str(), Some(json_path_text.as_str()));

        let markdown = fs::read_to_string(markdown_path).expect("read disagreement markdown");
        assert!(markdown.contains("Market-Head Judge Disagreement Report"));
        assert!(markdown.contains("| agent-swap-survival | 0.50 | 1.00 | +0.50 |"));
        assert!(markdown.contains("| dominant drift | classification |"));
        assert!(markdown.contains("## Drift Classifications"));
        assert!(markdown.contains("## Critical Fact Diagnostics"));
        assert!(
            markdown
                .contains("Selector pruning dropped required support memory from src/query.rs.")
        );
        assert!(markdown.contains("required concepts"));

        fs::remove_dir_all(root).expect("cleanup disagreement root");
    }

    #[test]
    fn default_market_head_model_matches_supported_provider_defaults() {
        assert_eq!(
            default_market_head_model(&BenchMarketProvider::Codex),
            "gpt-5.4"
        );
        assert_eq!(
            default_market_head_model(&BenchMarketProvider::Claude),
            "opus"
        );
    }

    #[test]
    fn repo_sync_focus_prefers_actionable_next_slice_when_current_slice_is_complete() {
        let state = RepoSyncState {
            repo_root: "/tmp/adhd".to_string(),
            branch: "task/observability-stack".to_string(),
            head: "77b33fa2deadbeef".to_string(),
            status: "watching".to_string(),
            recent_log: "77b33fa feat: rebase observability runtime onto machine organism"
                .to_string(),
            files: vec![RepoSyncFile {
                path: "docs/current-cycle.md".to_string(),
                content: r#"
## Current Slice

The current slice was proved locally on 2026-03-21:

1. this should not win once the slice is complete

## Next Slice

After the live agent badge registry:

1. make the badge point at the real next action
"#
                .to_string(),
            }],
        };

        assert_eq!(
            repo_sync_focus(&state),
            "make the badge point at the real next action"
        );
    }

    #[test]
    fn repo_sync_lane_resource_tracks_repo_and_branch() {
        let state = RepoSyncState {
            repo_root: "/tmp/demo".to_string(),
            branch: "feature/proof".to_string(),
            head: "77b33fa2deadbeef".to_string(),
            status: "watching".to_string(),
            recent_log: "77b33fa feat: rebase observability runtime onto machine organism"
                .to_string(),
            files: Vec::new(),
        };

        assert_eq!(repo_sync_lane_resource(&state), "repo/demo/feature/proof");
    }

    #[test]
    fn resolve_repo_sync_git_text_uses_explicit_branch_fallback_for_non_git_worktree_mounts() {
        let repo_root = std::env::temp_dir().join(format!("ice-sync-fallback-{}", Uuid::now_v7()));
        fs::create_dir_all(&repo_root).expect("create temp repo root");

        let result = resolve_repo_sync_git_text(
            &repo_root,
            &["branch", "--show-current"],
            Some("task/multi-worktree-shadow"),
            RepoSyncGitRequirement::Required,
        )
        .expect("resolve branch fallback");

        assert_eq!(result, "task/multi-worktree-shadow");

        fs::remove_dir_all(&repo_root).expect("cleanup temp repo root");
    }

    #[test]
    fn resolve_repo_sync_git_text_marks_optional_git_fields_unavailable_when_git_is_missing() {
        let repo_root = std::env::temp_dir().join(format!("ice-sync-optional-{}", Uuid::now_v7()));
        fs::create_dir_all(&repo_root).expect("create temp repo root");

        let result = resolve_repo_sync_git_text(
            &repo_root,
            &["log", "--oneline", "--decorate", "-5"],
            None,
            RepoSyncGitRequirement::Optional,
        )
        .expect("resolve optional fallback");

        assert!(result.starts_with("git log --oneline --decorate -5 unavailable:"));

        fs::remove_dir_all(&repo_root).expect("cleanup temp repo root");
    }

    #[test]
    fn parse_dispatch_worker_injects_attached_repo_lane_metadata() {
        let worker = parse_dispatch_worker(DispatchWorkerArgs {
            worker_id: "worker-a".to_string(),
            display_name: "Worker A".to_string(),
            role: "coder".to_string(),
            agent_type: "ollama".to_string(),
            tier: DispatchWorkerTier::Small,
            model: "qwen2.5:0.5b".to_string(),
            capabilities: vec!["read".to_string(), "write".to_string()],
            max_parallelism: 1,
            focus: Some("Stay on the repo lane".to_string()),
            namespace: Some("@machine".to_string()),
            task: Some("machine-organism".to_string()),
            status: "listening".to_string(),
            metadata_json: "{}".to_string(),
            attached_repo_root: Some("/tmp/demo".to_string()),
            attached_branch: Some("main".to_string()),
            attached_label: None,
            attached_resource: None,
            derive_attached_lane_from_badge: false,
        })
        .expect("parse worker with attached lane");

        assert_eq!(
            worker.metadata["attached_lane"]["projection_id"].as_str(),
            Some("repo:/tmp/demo:main")
        );
        assert_eq!(
            worker.metadata["attached_lane"]["label"].as_str(),
            Some("demo @ main")
        );
        assert_eq!(
            worker.metadata["attached_lane"]["resource"].as_str(),
            Some("repo/demo/main")
        );
        assert_eq!(
            worker.metadata["attached_lane"]["task_id"].as_str(),
            Some("machine-organism")
        );
        assert_eq!(
            worker.metadata["attached_lane_source"].as_str(),
            Some("explicit_cli")
        );
    }

    #[test]
    fn parse_dispatch_worker_rejects_conflicting_attached_lane_sources() {
        let error = parse_dispatch_worker(DispatchWorkerArgs {
            worker_id: "worker-a".to_string(),
            display_name: "Worker A".to_string(),
            role: "coder".to_string(),
            agent_type: "ollama".to_string(),
            tier: DispatchWorkerTier::Small,
            model: "qwen2.5:0.5b".to_string(),
            capabilities: vec!["read".to_string()],
            max_parallelism: 1,
            focus: None,
            namespace: Some("@machine".to_string()),
            task: Some("machine-organism".to_string()),
            status: "listening".to_string(),
            metadata_json: json!({
                "attached_lane": {
                    "projection_id": "repo:/tmp/demo:main",
                    "projection_kind": "repo",
                    "label": "demo @ main"
                }
            })
            .to_string(),
            attached_repo_root: Some("/tmp/demo".to_string()),
            attached_branch: Some("main".to_string()),
            attached_label: None,
            attached_resource: None,
            derive_attached_lane_from_badge: false,
        })
        .expect_err("conflicting attached lane sources should fail");

        assert!(
            error
                .to_string()
                .contains("attached lane cannot be supplied via both"),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn derive_dispatch_attached_lane_from_badges_uses_single_matching_repo_badge() {
        let worker = parse_dispatch_worker(DispatchWorkerArgs {
            worker_id: "worker-a".to_string(),
            display_name: "Worker A".to_string(),
            role: "coder".to_string(),
            agent_type: "ollama".to_string(),
            tier: DispatchWorkerTier::Small,
            model: "qwen2.5:0.5b".to_string(),
            capabilities: vec!["read".to_string()],
            max_parallelism: 1,
            focus: None,
            namespace: Some("@machine".to_string()),
            task: Some("machine-organism".to_string()),
            status: "listening".to_string(),
            metadata_json: "{}".to_string(),
            attached_repo_root: None,
            attached_branch: None,
            attached_label: None,
            attached_resource: None,
            derive_attached_lane_from_badge: false,
        })
        .expect("parse worker");
        let badge = AgentBadgeRecord {
            attachment_id: "attach:worker-a".to_string(),
            agent_id: "worker-a".to_string(),
            agent_type: "codex".to_string(),
            namespace: "@machine".to_string(),
            task_id: Some("machine-organism".to_string()),
            context_id: Some("ctx-1".to_string()),
            role: Some("coder".to_string()),
            display_name: "Worker A".to_string(),
            status: "watching".to_string(),
            focus: "Stay inside the repo lane".to_string(),
            headline: "repo lane badge".to_string(),
            resource: Some("repo/demo/main".to_string()),
            repo_root: Some("/tmp/demo".to_string()),
            branch: Some("main".to_string()),
            metadata: json!({}),
            updated_at: Utc::now(),
            last_seen_at: Utc::now(),
            tick_count: 1,
            connected: true,
        };

        let lane = derive_dispatch_attached_lane_from_badges(&worker, &[badge])
            .expect("single matching badge should derive attached lane");
        assert_eq!(lane.projection_id, "repo:/tmp/demo:main");
        assert_eq!(lane.label, "demo @ main");
    }

    #[test]
    fn derive_dispatch_attached_lane_from_badges_rejects_ambiguous_repo_badges() {
        let worker = parse_dispatch_worker(DispatchWorkerArgs {
            worker_id: "worker-a".to_string(),
            display_name: "Worker A".to_string(),
            role: "coder".to_string(),
            agent_type: "ollama".to_string(),
            tier: DispatchWorkerTier::Small,
            model: "qwen2.5:0.5b".to_string(),
            capabilities: vec!["read".to_string()],
            max_parallelism: 1,
            focus: None,
            namespace: Some("@machine".to_string()),
            task: Some("machine-organism".to_string()),
            status: "listening".to_string(),
            metadata_json: "{}".to_string(),
            attached_repo_root: None,
            attached_branch: None,
            attached_label: None,
            attached_resource: None,
            derive_attached_lane_from_badge: false,
        })
        .expect("parse worker");
        let now = Utc::now();
        let badge_a = AgentBadgeRecord {
            attachment_id: "attach:worker-a:1".to_string(),
            agent_id: "worker-a".to_string(),
            agent_type: "codex".to_string(),
            namespace: "@machine".to_string(),
            task_id: Some("machine-organism".to_string()),
            context_id: Some("ctx-1".to_string()),
            role: Some("coder".to_string()),
            display_name: "Worker A".to_string(),
            status: "watching".to_string(),
            focus: "Lane A".to_string(),
            headline: "repo lane badge".to_string(),
            resource: Some("repo/demo/main".to_string()),
            repo_root: Some("/tmp/demo".to_string()),
            branch: Some("main".to_string()),
            metadata: json!({}),
            updated_at: now,
            last_seen_at: now,
            tick_count: 1,
            connected: true,
        };
        let badge_b = AgentBadgeRecord {
            attachment_id: "attach:worker-a:2".to_string(),
            agent_id: "worker-a".to_string(),
            agent_type: "codex".to_string(),
            namespace: "@machine".to_string(),
            task_id: Some("machine-organism".to_string()),
            context_id: Some("ctx-2".to_string()),
            role: Some("coder".to_string()),
            display_name: "Worker A".to_string(),
            status: "watching".to_string(),
            focus: "Lane B".to_string(),
            headline: "repo lane badge".to_string(),
            resource: Some("repo/other/main".to_string()),
            repo_root: Some("/tmp/other".to_string()),
            branch: Some("main".to_string()),
            metadata: json!({}),
            updated_at: now,
            last_seen_at: now,
            tick_count: 1,
            connected: true,
        };

        assert!(
            derive_dispatch_attached_lane_from_badges(&worker, &[badge_a, badge_b]).is_none(),
            "ambiguous live badges should not auto-derive an attached lane"
        );
    }
}

async fn run_repo_sync(
    engine: Arc<Engine>,
    repo_root: PathBuf,
    args: RepoSyncArgs,
) -> Result<serde_json::Value> {
    let session_id = args
        .session
        .clone()
        .unwrap_or_else(|| format!("repo-sync-{}", Uuid::now_v7()));
    let attachment = engine.attach_agent(AttachAgentInput {
        agent_id: args.agent.clone(),
        agent_type: args.agent_type.clone(),
        capabilities: vec![
            "repo_sync".to_string(),
            "snapshot".to_string(),
            "continuity_dogfood".to_string(),
        ],
        namespace: args.namespace.clone(),
        role: Some("operator".to_string()),
        metadata: serde_json::json!({
            "repo_root": repo_root.display().to_string(),
            "source": "dogfood_sync_repo",
        }),
    })?;
    let context = engine.open_context(OpenContextInput {
        namespace: args.namespace.clone(),
        task_id: args.task.clone(),
        session_id: session_id.clone(),
        objective: args.objective.clone(),
        selector: None,
        agent_id: Some(args.agent.clone()),
        attachment_id: Some(attachment.id.clone()),
    })?;
    let initial_state = collect_repo_sync_state(&repo_root, &args)?;
    upsert_repo_sync_badge(
        engine.as_ref(),
        &attachment.id,
        &context.id,
        &args.agent,
        &initial_state,
        "watching",
    )?;

    let mut last_payload = None;
    let mut latest_snapshot_id = context.last_snapshot_id.clone();
    let mut last_heartbeat_at = Instant::now();
    let mut latest_claim_id = None;
    let mut last_result = serde_json::json!({
        "context_id": context.id.clone(),
        "changed": false,
    });

    loop {
        let state = collect_repo_sync_state(&repo_root, &args)?;
        let payload_text = serde_json::to_string_pretty(&state)?;
        if last_payload.as_ref() != Some(&payload_text) {
            upsert_repo_sync_badge(
                engine.as_ref(),
                &attachment.id,
                &context.id,
                &args.agent,
                &state,
                "syncing",
            )?;
            latest_claim_id = Some(claim_repo_sync_lane(
                engine.as_ref(),
                &context.id,
                &attachment.id,
                &args.agent,
                &state,
                args.heartbeat_secs.or(args.watch_secs),
            )?);
            let before = engine.read_context(ReadContextInput {
                context_id: Some(context.id.clone()),
                namespace: Some(args.namespace.clone()),
                task_id: Some(args.task.clone()),
                objective: args.objective.clone(),
                token_budget: args.budget_tokens,
                selector: None,
                agent_id: Some(args.agent.clone()),
                session_id: Some(session_id.clone()),
                view_id: None,
                include_resolved: true,
                candidate_limit: args.candidate_limit,
            })?;

            let state_event = engine.write_events(vec![WriteEventInput {
                context_id: Some(context.id.clone()),
                event: EventInput {
                    kind: EventKind::Document,
                    agent_id: args.agent.clone(),
                    agent_role: Some("repo_sync".to_string()),
                    timestamp: None,
                    session_id: session_id.clone(),
                    task_id: Some(args.task.clone()),
                    project_id: Some(repo_root_name(&repo_root)),
                    goal_id: Some("dogfood".to_string()),
                    run_id: Some(format!("repo-sync-{}", Uuid::now_v7())),
                    namespace: Some(args.namespace.clone()),
                    environment: Some("local".to_string()),
                    source: "dogfood_sync_repo".to_string(),
                    scope: Scope::Project,
                    tags: vec![
                        "dogfood".to_string(),
                        "repo_state".to_string(),
                        format!("branch:{}", state.branch),
                    ],
                    dimensions: repo_state_dimensions(&state),
                    content: payload_text.clone(),
                    attributes: serde_json::to_value(&state)?,
                },
            }])?;
            let support = state_event
                .first()
                .map(|manifest| SupportRef {
                    support_type: "event".to_string(),
                    support_id: manifest.event.id.clone(),
                    reason: Some("repo compact state".to_string()),
                    weight: 1.0,
                })
                .into_iter()
                .collect::<Vec<_>>();

            let current_cycle = state
                .files
                .iter()
                .find(|item| item.path == "docs/current-cycle.md")
                .map(|item| item.content.as_str())
                .unwrap_or("");
            let summary_body = format!(
                "Branch: {}\nHead: {}\nStatus:\n{}\n\nRecent log:\n{}\n\nCurrent cycle:\n{}",
                state.branch, state.head, state.status, state.recent_log, current_cycle
            );
            let summary = engine.write_derivations(vec![ContinuityItemInput {
                context_id: context.id.clone(),
                author_agent_id: args.agent.clone(),
                kind: ContinuityKind::WorkingState,
                title: "Repo execution compact state".to_string(),
                body: summary_body,
                scope: Scope::Project,
                status: Some(ContinuityStatus::Active),
                importance: Some(0.98),
                confidence: Some(0.98),
                salience: Some(0.98),
                layer: Some(MemoryLayer::Hot),
                supports: support.clone(),
                dimensions: repo_state_dimensions(&state),
                extra: serde_json::json!({
                    "repo_root": state.repo_root,
                    "files": state.files.iter().map(|item| item.path.clone()).collect::<Vec<_>>(),
                }),
            }])?;
            let summary = summary
                .into_iter()
                .next()
                .context("missing repo sync working-state item")?;
            supersede_previous_repo_sync(&engine, &before, &summary.id, &args.agent)?;

            let mut recorded = Vec::new();
            recorded.extend(record_explicit_notes(
                &engine,
                &before,
                &context.id,
                &args.agent,
                ContinuityKind::Decision,
                &args.decision,
                &support,
            )?);
            recorded.extend(record_explicit_notes(
                &engine,
                &before,
                &context.id,
                &args.agent,
                ContinuityKind::Constraint,
                &args.constraint,
                &support,
            )?);
            recorded.extend(record_explicit_notes(
                &engine,
                &before,
                &context.id,
                &args.agent,
                ContinuityKind::Incident,
                &args.incident,
                &support,
            )?);
            recorded.extend(record_explicit_notes(
                &engine,
                &before,
                &context.id,
                &args.agent,
                ContinuityKind::OperationalScar,
                &args.scar,
                &support,
            )?);

            let snapshot = engine.snapshot(SnapshotInput {
                context_id: Some(context.id.clone()),
                namespace: Some(args.namespace.clone()),
                task_id: Some(args.task.clone()),
                objective: Some(args.objective.clone()),
                selector: None,
                resolution: args.resolution,
                token_budget: args.budget_tokens,
                candidate_limit: args.candidate_limit,
                owner_agent_id: Some(args.agent.clone()),
            })?;
            latest_snapshot_id = Some(snapshot.id.clone());

            last_result = serde_json::json!({
                "context_id": context.id.clone(),
                "attachment_id": attachment.id.clone(),
                "changed": true,
                "event_id": state_event.first().map(|item| item.event.id.clone()),
                "repo_lane_claim_id": latest_claim_id,
                "working_state_id": summary.id,
                "snapshot_id": snapshot.id,
                "recorded_titles": recorded,
            });
            last_payload = Some(payload_text);
            last_heartbeat_at = Instant::now();
        } else {
            let heartbeat_due = args
                .heartbeat_secs
                .filter(|secs| args.watch_secs.is_some() && *secs > 0)
                .map(|secs| last_heartbeat_at.elapsed() >= Duration::from_secs(secs))
                .unwrap_or(false);
            if heartbeat_due {
                upsert_repo_sync_badge(
                    engine.as_ref(),
                    &attachment.id,
                    &context.id,
                    &args.agent,
                    &state,
                    "watching",
                )?;
                latest_claim_id = Some(claim_repo_sync_lane(
                    engine.as_ref(),
                    &context.id,
                    &attachment.id,
                    &args.agent,
                    &state,
                    args.heartbeat_secs.or(args.watch_secs),
                )?);
                let heartbeat = engine.write_events(vec![WriteEventInput {
                    context_id: Some(context.id.clone()),
                    event: EventInput {
                        kind: EventKind::Trace,
                        agent_id: args.agent.clone(),
                        agent_role: Some("repo_sync_heartbeat".to_string()),
                        timestamp: None,
                        session_id: session_id.clone(),
                        task_id: Some(args.task.clone()),
                        project_id: Some(repo_root_name(&repo_root)),
                        goal_id: Some("dogfood".to_string()),
                        run_id: Some(format!("repo-sync-heartbeat-{}", Uuid::now_v7())),
                        namespace: Some(args.namespace.clone()),
                        environment: Some("local".to_string()),
                        source: "dogfood_sync_repo_heartbeat".to_string(),
                        scope: Scope::Project,
                        tags: vec![
                            "dogfood".to_string(),
                            "heartbeat".to_string(),
                            format!("branch:{}", state.branch),
                        ],
                        dimensions: repo_state_dimensions(&state),
                        content: format!(
                            "Repo sync heartbeat for branch {} at head {}.",
                            state.branch, state.head
                        ),
                        attributes: serde_json::json!({
                            "repo_root": state.repo_root,
                            "branch": state.branch,
                            "head": state.head,
                            "status": state.status,
                        }),
                    },
                }])?;
                last_heartbeat_at = Instant::now();
                last_result = serde_json::json!({
                    "context_id": context.id.clone(),
                    "attachment_id": attachment.id.clone(),
                    "changed": false,
                    "heartbeat_event_id": heartbeat.first().map(|item| item.event.id.clone()),
                    "repo_lane_claim_id": latest_claim_id,
                    "snapshot_id": latest_snapshot_id,
                });
            } else {
                last_result = serde_json::json!({
                    "context_id": context.id.clone(),
                    "attachment_id": attachment.id.clone(),
                    "changed": false,
                    "repo_lane_claim_id": latest_claim_id,
                    "snapshot_id": latest_snapshot_id,
                });
            }
        }

        if let Some(watch_secs) = args.watch_secs {
            println!("{}", serde_json::to_string(&last_result)?);
            sleep(Duration::from_secs(watch_secs)).await;
            continue;
        }

        break;
    }

    Ok(last_result)
}

async fn run_attachment_heartbeat(
    engine: Arc<Engine>,
    args: HeartbeatArgs,
) -> Result<serde_json::Value> {
    let latest = loop {
        let attachment = engine.heartbeat(HeartbeatInput {
            attachment_id: args.attachment_id.clone(),
            agent_id: args.agent.clone(),
            namespace: args.namespace.clone(),
            context_id: args.context_id.clone(),
        })?;
        let current = serde_json::json!({
            "attachment_id": attachment.id,
            "agent_id": attachment.input.agent_id,
            "namespace": attachment.input.namespace,
            "context_id": attachment.context_id,
            "tick_count": attachment.tick_count,
            "last_seen_at": attachment.last_seen_at,
            "active": attachment.active,
        });
        if let Some(every_secs) = args.every_secs {
            println!("{}", serde_json::to_string(&current)?);
            sleep(Duration::from_secs(every_secs)).await;
            continue;
        }
        break current;
    };
    Ok(latest)
}

fn collect_repo_sync_state(
    repo_root: &std::path::Path,
    args: &RepoSyncArgs,
) -> Result<RepoSyncState> {
    let files = [
        "CONTEXT.md",
        "docs/current-cycle.md",
        "docs/agent-handoff.md",
    ]
    .into_iter()
    .map(|path| {
        let full_path = repo_root.join(path);
        let content = fs::read_to_string(&full_path)
            .with_context(|| format!("reading {}", full_path.display()))?;
        Ok(RepoSyncFile {
            path: path.to_string(),
            content,
        })
    })
    .collect::<Result<Vec<_>>>()?;

    Ok(RepoSyncState {
        repo_root: repo_root.display().to_string(),
        branch: resolve_repo_sync_git_text(
            repo_root,
            &["branch", "--show-current"],
            args.branch.as_deref(),
            RepoSyncGitRequirement::Required,
        )?,
        head: resolve_repo_sync_git_text(
            repo_root,
            &["rev-parse", "HEAD"],
            args.head.as_deref(),
            RepoSyncGitRequirement::Optional,
        )?,
        status: resolve_repo_sync_git_text(
            repo_root,
            &["status", "--short", "--branch"],
            None,
            RepoSyncGitRequirement::Optional,
        )?,
        recent_log: resolve_repo_sync_git_text(
            repo_root,
            &["log", "--oneline", "--decorate", "-5"],
            None,
            RepoSyncGitRequirement::Optional,
        )?,
        files,
    })
}

fn resolve_repo_sync_git_text(
    repo_root: &std::path::Path,
    git_args: &[&str],
    fallback: Option<&str>,
    requirement: RepoSyncGitRequirement,
) -> Result<String> {
    match git_output(repo_root, git_args) {
        Ok(output) if !output.trim().is_empty() => Ok(output),
        Ok(_) => match fallback {
            Some(value) if !value.trim().is_empty() => Ok(value.trim().to_string()),
            _ => match requirement {
                RepoSyncGitRequirement::Required => anyhow::bail!(
                    "git {} returned empty output and no explicit fallback was provided; pass --{} when syncing a containerized worktree without git metadata",
                    git_args.join(" "),
                    git_arg_name(git_args)
                ),
                RepoSyncGitRequirement::Optional => Ok(format!(
                    "git {} unavailable: empty output",
                    git_args.join(" ")
                )),
            },
        },
        Err(error) => match fallback {
            Some(value) if !value.trim().is_empty() => Ok(value.trim().to_string()),
            _ => match requirement {
                RepoSyncGitRequirement::Required => anyhow::bail!(
                    "{error:#}; pass --{} when syncing a containerized worktree without git metadata",
                    git_arg_name(git_args)
                ),
                RepoSyncGitRequirement::Optional => Ok(format!(
                    "git {} unavailable: {:#}",
                    git_args.join(" "),
                    error
                )),
            },
        },
    }
}

fn git_arg_name(git_args: &[&str]) -> &'static str {
    match git_args {
        ["branch", "--show-current"] => "branch",
        ["rev-parse", "HEAD"] => "head",
        _ => "git-fallback",
    }
}

fn git_output(repo_root: &std::path::Path, args: &[&str]) -> Result<String> {
    let output = StdCommand::new("git")
        .arg("-c")
        .arg(format!("safe.directory={}", repo_root.display()))
        .args(args)
        .current_dir(repo_root)
        .output()
        .with_context(|| format!("running git {}", args.join(" ")))?;
    if !output.status.success() {
        anyhow::bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn repo_state_dimensions(state: &RepoSyncState) -> Vec<DimensionValue> {
    vec![
        DimensionValue {
            key: "repo_branch".to_string(),
            value: state.branch.clone(),
            weight: 100,
        },
        DimensionValue {
            key: "repo_head".to_string(),
            value: state.head.clone(),
            weight: 100,
        },
        DimensionValue {
            key: "repo_sync".to_string(),
            value: "compact_state".to_string(),
            weight: 100,
        },
    ]
}

fn repo_root_name(repo_root: &std::path::Path) -> String {
    repo_root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("repo")
        .to_string()
}

fn supersede_previous_repo_sync(
    engine: &Engine,
    current: &ice::continuity::ContextRead,
    replacement_id: &str,
    agent_id: &str,
) -> Result<()> {
    for item in &current.working_state {
        if item.title == "Repo execution compact state"
            && item.id != replacement_id
            && item.status.is_open()
        {
            let _ = engine.resolve_or_supersede(ResolveOrSupersedeInput {
                continuity_id: item.id.clone(),
                actor_agent_id: agent_id.to_string(),
                new_status: ContinuityStatus::Superseded,
                supersedes_id: Some(replacement_id.to_string()),
                resolution_note: Some("replaced by fresher repo sync".to_string()),
                extra: serde_json::json!({"source": "dogfood_sync_repo"}),
            })?;
        }
    }
    Ok(())
}

fn record_explicit_notes(
    engine: &Engine,
    current: &ice::continuity::ContextRead,
    context_id: &str,
    agent_id: &str,
    kind: ContinuityKind,
    notes: &[String],
    supports: &[SupportRef],
) -> Result<Vec<String>> {
    let existing = match kind {
        ContinuityKind::Decision => &current.decisions,
        ContinuityKind::Constraint => &current.constraints,
        ContinuityKind::Incident => &current.incidents,
        ContinuityKind::OperationalScar => &current.operational_scars,
        _ => return Ok(Vec::new()),
    };
    let mut recorded = Vec::new();
    for note in notes {
        let title = titled_note(kind, note);
        if existing
            .iter()
            .any(|item| item.title == title && item.body == *note && item.status.is_open())
        {
            continue;
        }
        let input = ContinuityItemInput {
            context_id: context_id.to_string(),
            author_agent_id: agent_id.to_string(),
            kind,
            title: title.clone(),
            body: note.clone(),
            scope: Scope::Project,
            status: Some(ContinuityStatus::Active),
            importance: Some(match kind {
                ContinuityKind::OperationalScar => 0.99,
                ContinuityKind::Decision | ContinuityKind::Constraint => 0.97,
                _ => 0.94,
            }),
            confidence: Some(0.98),
            salience: Some(match kind {
                ContinuityKind::OperationalScar => 0.99,
                _ => 0.95,
            }),
            layer: Some(match kind {
                ContinuityKind::Incident => MemoryLayer::Episodic,
                ContinuityKind::OperationalScar => MemoryLayer::Semantic,
                _ => MemoryLayer::Semantic,
            }),
            supports: supports.to_vec(),
            dimensions: Vec::new(),
            extra: serde_json::json!({"source": "dogfood_sync_repo"}),
        };
        match kind {
            ContinuityKind::Decision => {
                engine.mark_decision(input)?;
            }
            ContinuityKind::Constraint => {
                engine.mark_constraint(input)?;
            }
            ContinuityKind::Incident => {
                engine.mark_incident(input)?;
            }
            ContinuityKind::OperationalScar => {
                engine.mark_operational_scar(input)?;
            }
            _ => {}
        }
        recorded.push(title);
    }
    Ok(recorded)
}

fn titled_note(kind: ContinuityKind, text: &str) -> String {
    let prefix = match kind {
        ContinuityKind::Decision => "Decision",
        ContinuityKind::Constraint => "Constraint",
        ContinuityKind::Incident => "Incident",
        ContinuityKind::OperationalScar => "Scar",
        _ => "Note",
    };
    let first_line = text.lines().next().unwrap_or(text).trim();
    let clipped = if first_line.chars().count() > 72 {
        let trimmed = first_line.chars().take(69).collect::<String>();
        format!("{trimmed}...")
    } else {
        first_line.to_string()
    };
    format!("{prefix}: {clipped}")
}

async fn run_wrapped(engine: Arc<Engine>, args: WrapArgs) -> Result<i32> {
    let joined = args.command.join(" ");
    engine.ingest(EventInput {
        kind: EventKind::ShellCommand,
        agent_id: args.agent.clone(),
        agent_role: None,
        timestamp: None,
        session_id: args.session.clone(),
        task_id: args.task.clone(),
        project_id: None,
        goal_id: None,
        run_id: None,
        namespace: None,
        environment: None,
        source: args.source.clone(),
        scope: args.scope.clone(),
        tags: args.tags.clone(),
        dimensions: Vec::new(),
        content: joined.clone(),
        attributes: serde_json::json!({"argv": args.command}),
    })?;

    let mut child = TokioCommand::new(&args.command[0]);
    child.args(&args.command[1..]);
    child.stdout(Stdio::piped());
    child.stderr(Stdio::piped());
    let output = child.output().await?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code().unwrap_or(1);

    if !stdout.trim().is_empty() {
        engine.ingest(EventInput {
            kind: EventKind::ShellOutput,
            agent_id: args.agent.clone(),
            agent_role: None,
            timestamp: None,
            session_id: args.session.clone(),
            task_id: args.task.clone(),
            project_id: None,
            goal_id: None,
            run_id: None,
            namespace: None,
            environment: None,
            source: args.source.clone(),
            scope: args.scope.clone(),
            tags: args.tags.clone(),
            dimensions: Vec::new(),
            content: stdout.clone(),
            attributes: serde_json::json!({"stream": "stdout", "command": joined}),
        })?;
        print!("{stdout}");
    }

    if !stderr.trim().is_empty() {
        engine.ingest(EventInput {
            kind: if exit_code == 0 {
                EventKind::ShellOutput
            } else {
                EventKind::Error
            },
            agent_id: args.agent.clone(),
            agent_role: None,
            timestamp: None,
            session_id: args.session.clone(),
            task_id: args.task.clone(),
            project_id: None,
            goal_id: None,
            run_id: None,
            namespace: None,
            environment: None,
            source: args.source.clone(),
            scope: args.scope.clone(),
            tags: args.tags.clone(),
            dimensions: Vec::new(),
            content: stderr.clone(),
            attributes: serde_json::json!({"stream": "stderr", "command": joined, "exit_code": exit_code}),
        })?;
        eprint!("{stderr}");
    }

    if exit_code != 0 {
        engine.ingest(EventInput {
            kind: EventKind::Error,
            agent_id: args.agent,
            agent_role: None,
            timestamp: None,
            session_id: args.session,
            task_id: args.task,
            project_id: None,
            goal_id: None,
            run_id: None,
            namespace: None,
            environment: None,
            source: args.source,
            scope: args.scope,
            tags: args.tags,
            dimensions: Vec::new(),
            content: format!("wrapped command exited with status {exit_code}: {joined}"),
            attributes: serde_json::json!({"exit_code": exit_code}),
        })?;
    }

    Ok(exit_code)
}

async fn run_demo(engine: Arc<Engine>, args: DemoArgs) -> Result<serde_json::Value> {
    let namespace = "fabric-demo";
    let project = "demo-project";
    let session = "fabric-session";
    let task = "api-failure";
    let endpoint = "/v1/context-pack";
    let planner = "planner-agent";
    let debugger = "debugger-agent";

    let planner_dimensions = vec![
        DimensionValue {
            key: "endpoint".into(),
            value: endpoint.into(),
            weight: 100,
        },
        DimensionValue {
            key: "file".into(),
            value: "src/http.rs".into(),
            weight: 100,
        },
        DimensionValue {
            key: "claim.api_status".into(),
            value: "healthy".into(),
            weight: 100,
        },
    ];
    engine.ingest(EventInput {
        kind: EventKind::Prompt,
        agent_id: planner.to_string(),
        agent_role: Some("planner".to_string()),
        timestamp: None,
        session_id: session.to_string(),
        task_id: Some(task.to_string()),
        project_id: Some(project.to_string()),
        goal_id: Some("stabilize-api".to_string()),
        run_id: Some("demo-run".to_string()),
        namespace: Some(namespace.to_string()),
        environment: Some("local".to_string()),
        source: "demo".to_string(),
        scope: Scope::Project,
        tags: vec!["demo".to_string(), "planner".to_string()],
        dimensions: planner_dimensions,
        content: "Planner assumption: the endpoint should stay healthy if provenance and selector constraints are preserved.".to_string(),
        attributes: serde_json::json!({"turn": 1}),
    })?;
    engine.ingest(EventInput {
        kind: EventKind::Error,
        agent_id: "api-agent".to_string(),
        agent_role: Some("executor".to_string()),
        timestamp: None,
        session_id: session.to_string(),
        task_id: Some(task.to_string()),
        project_id: Some(project.to_string()),
        goal_id: Some("stabilize-api".to_string()),
        run_id: Some("demo-run".to_string()),
        namespace: Some(namespace.to_string()),
        environment: Some("local".to_string()),
        source: "demo".to_string(),
        scope: Scope::Project,
        tags: vec!["demo".to_string(), "incident".to_string()],
        dimensions: vec![
            DimensionValue { key: "endpoint".into(), value: endpoint.into(), weight: 100 },
            DimensionValue { key: "file".into(), value: "src/http.rs".into(), weight: 100 },
            DimensionValue { key: "error".into(), value: "TimeoutError".into(), weight: 100 },
        ],
        content: "TimeoutError while serving /v1/context-pack after selector pruning dropped the original invariant.".to_string(),
        attributes: serde_json::json!({"turn": 2}),
    })?;
    engine.ingest(EventInput {
        kind: EventKind::Note,
        agent_id: planner.to_string(),
        agent_role: Some("planner".to_string()),
        timestamp: None,
        session_id: session.to_string(),
        task_id: Some(task.to_string()),
        project_id: Some(project.to_string()),
        goal_id: Some("stabilize-api".to_string()),
        run_id: Some("demo-run".to_string()),
        namespace: Some(namespace.to_string()),
        environment: Some("local".to_string()),
        source: "demo".to_string(),
        scope: Scope::Project,
        tags: vec!["demo".to_string(), "constraint".to_string()],
        dimensions: vec![
            DimensionValue { key: "file".into(), value: "src/storage.rs".into(), weight: 100 },
            DimensionValue { key: "constraint.provenance".into(), value: "must-preserve".into(), weight: 100 },
        ],
        content: "Constraint: hand-offs must preserve provenance, unresolved contradictions, and replay anchors.".to_string(),
        attributes: serde_json::json!({"turn": 3}),
    })?;
    engine.ingest(EventInput {
        kind: EventKind::Note,
        agent_id: debugger.to_string(),
        agent_role: Some("debugger".to_string()),
        timestamp: None,
        session_id: session.to_string(),
        task_id: Some(task.to_string()),
        project_id: Some(project.to_string()),
        goal_id: Some("stabilize-api".to_string()),
        run_id: Some("demo-run".to_string()),
        namespace: Some(namespace.to_string()),
        environment: Some("local".to_string()),
        source: "demo".to_string(),
        scope: Scope::Project,
        tags: vec!["demo".to_string(), "debugger".to_string()],
        dimensions: vec![
            DimensionValue { key: "endpoint".into(), value: endpoint.into(), weight: 100 },
            DimensionValue { key: "file".into(), value: "src/query.rs".into(), weight: 100 },
            DimensionValue { key: "claim.api_status".into(), value: "degraded".into(), weight: 100 },
            DimensionValue { key: "hypothesis.root_cause".into(), value: "selector_missing".into(), weight: 100 },
        ],
        content: "Debugger finding: selector_missing in src/query.rs causes the degraded endpoint state and contradicts the planner assumption.".to_string(),
        attributes: serde_json::json!({"turn": 4}),
    })?;
    for turn in 0..6 {
        engine.ingest(EventInput {
            kind: EventKind::Note,
            agent_id: "noise-agent".to_string(),
            agent_role: Some("noise".to_string()),
            timestamp: None,
            session_id: format!("noise-session-{turn}"),
            task_id: Some("noise".to_string()),
            project_id: Some("other-project".to_string()),
            goal_id: Some("noise".to_string()),
            run_id: Some("noise-run".to_string()),
            namespace: Some(namespace.to_string()),
            environment: Some("local".to_string()),
            source: "demo".to_string(),
            scope: Scope::Shared,
            tags: vec!["noise".to_string()],
            dimensions: vec![DimensionValue {
                key: "subsystem".into(),
                value: format!("noise-{turn}"),
                weight: 100,
            }],
            content: format!(
                "unrelated noise event {turn} with queue={}, retries={}",
                turn * 11,
                turn % 3
            ),
            attributes: serde_json::json!({"turn": turn + 10}),
        })?;
    }

    let selector = Selector {
        all: vec![
            ice::model::DimensionFilter {
                key: "project".into(),
                values: vec![project.into()],
            },
            ice::model::DimensionFilter {
                key: "endpoint".into(),
                values: vec![endpoint.into()],
            },
        ],
        any: vec![ice::model::DimensionFilter {
            key: "file".into(),
            values: vec![
                "src/http.rs".into(),
                "src/query.rs".into(),
                "src/storage.rs".into(),
            ],
        }],
        exclude: Vec::new(),
        layers: Vec::new(),
        start_ts: None,
        end_ts: None,
        limit: Some(24),
        namespace: Some(namespace.into()),
    };
    let planner_view = engine.materialize_view(ViewInput {
        op: ViewOp::Slice,
        owner_agent_id: Some(planner.to_string()),
        namespace: Some(namespace.into()),
        objective: Some("planner incident slice".into()),
        selectors: vec![selector.clone()],
        source_view_ids: Vec::new(),
        resolution: Some(SnapshotResolution::Medium),
        limit: Some(24),
    })?;
    let debugger_fork = engine.fork_view(&planner_view.id, Some(debugger.to_string()))?;
    let merged_view = engine.materialize_view(ViewInput {
        op: ViewOp::Merge,
        owner_agent_id: Some(planner.to_string()),
        namespace: Some(namespace.into()),
        objective: Some("merge planner and debugger evidence".into()),
        selectors: vec![selector.clone()],
        source_view_ids: vec![planner_view.id.clone(), debugger_fork.id.clone()],
        resolution: Some(SnapshotResolution::Medium),
        limit: Some(24),
    })?;
    let handoff = engine.create_handoff(HandoffInput {
        from_agent_id: planner.to_string(),
        to_agent_id: "coder-agent".to_string(),
        reason: "send the smallest high-signal failure packet to the coder".to_string(),
        query_text: "What is the minimum context needed to fix the /v1/context-pack failure without losing provenance?".to_string(),
        budget_tokens: args.budget_tokens,
        view_id: Some(merged_view.id.clone()),
        selector: Some(selector.clone()),
        objective: Some("fix the failing endpoint".to_string()),
        namespace: Some(namespace.into()),
    })?;
    let replay = engine.replay_by_selector(&selector, 12)?;
    let handoff_manifest = engine.explain_handoff(&handoff.id)?;
    let merged_manifest = engine.explain_view(&merged_view.id)?;
    let metrics_snapshot = engine.metrics_snapshot()?;
    let metrics_lines = metrics_snapshot
        .prometheus_text
        .lines()
        .filter(|line| {
            line.starts_with("ice_views_persisted")
                || line.starts_with("ice_handoffs_persisted")
                || line.starts_with("ice_item_dimensions_persisted")
                || line.starts_with("ice_fabric_relations_persisted")
        })
        .collect::<Vec<_>>();

    Ok(serde_json::json!({
        "selector": selector,
        "planner_view": planner_view,
        "debugger_fork": debugger_fork,
        "merged_view": merged_view,
        "handoff": handoff,
        "replay_count": replay.len(),
        "merged_conflicts": merged_manifest.conflicts,
        "handoff_manifest": handoff_manifest,
        "metrics_excerpt": metrics_lines,
    }))
}

#[derive(Debug, Serialize)]
struct BenchResult {
    cases: usize,
    distractors_per_case: usize,
    recent_window: usize,
    fabric_hits: usize,
    fabric_support_hits: usize,
    baseline_recent_hits: usize,
    baseline_recent_support_hits: usize,
    baseline_vector_hits: usize,
    baseline_vector_support_hits: usize,
    baseline_summary_hits: usize,
    baseline_summary_support_hits: usize,
    fabric_recall: f64,
    fabric_support_recall: f64,
    baseline_recent_recall: f64,
    baseline_recent_support_recall: f64,
    baseline_vector_recall: f64,
    baseline_vector_support_recall: f64,
    baseline_summary_recall: f64,
    baseline_summary_support_recall: f64,
    avg_fabric_query_ms: f64,
    avg_fabric_pack_tokens: f64,
    avg_view_conflicts: f64,
}

async fn run_bench(
    engine: Arc<Engine>,
    root: PathBuf,
    args: BenchArgs,
) -> Result<serde_json::Value> {
    if matches!(args.mode, BenchMode::Continuity) {
        let output_dir = root.join("benchmarks");
        let embedding_backend = engine.embedding_backend_key();
        let retrieval_protocol = format!(
            "uci+compiler+vector://{}?budget={}&candidates={}&recent={}",
            embedding_backend, args.budget_tokens, args.candidate_limit, args.recent_window
        );
        let config = ContinuityBenchConfig {
            output_dir,
            ollama_endpoint: args.ollama_endpoint,
            strong_model: args.strong_model,
            small_model: args.small_model,
            embedding_backend,
            retrieval_protocol,
            classes: args.classes,
            token_budget: args.budget_tokens,
            candidate_limit: args.candidate_limit,
            recent_window: args.recent_window,
            timeout_secs: args.timeout_secs,
            num_predict: args.num_predict,
        };
        let report = tokio::task::spawn_blocking(move || run_continuity_suite(config))
            .await
            .context("joining continuity benchmark task")??;
        return Ok(serde_json::to_value(report)?);
    }
    let result = run_legacy_bench(engine, args).await?;
    Ok(serde_json::to_value(result)?)
}

async fn run_bench_market_export(
    engine: Arc<Engine>,
    root: PathBuf,
    args: BenchMarketExportArgs,
) -> Result<serde_json::Value> {
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| root.join("benchmarks"));
    let embedding_backend = engine.embedding_backend_key();
    let retrieval_protocol = format!(
        "uci+compiler+vector://{}?budget={}&candidates={}&recent={}",
        embedding_backend, args.budget_tokens, args.candidate_limit, args.recent_window
    );
    let config = MarketHeadChallengeConfig {
        output_dir,
        ollama_endpoint: args.ollama_endpoint,
        strong_model: args.strong_model,
        small_model: args.small_model,
        embedding_backend,
        retrieval_protocol,
        classes: args.classes,
        token_budget: args.budget_tokens,
        candidate_limit: args.candidate_limit,
        recent_window: args.recent_window,
        timeout_secs: args.timeout_secs,
        num_predict: args.num_predict,
    };
    let report = tokio::task::spawn_blocking(move || export_market_head_challenge(config))
        .await
        .context("joining market-head export task")??;
    Ok(serde_json::to_value(report)?)
}

fn run_bench_market_evaluate(args: BenchMarketEvaluateArgs) -> Result<serde_json::Value> {
    let report =
        evaluate_market_head_challenge(args.evaluator_pack, args.responses_dir, &args.model)?;
    Ok(serde_json::to_value(report)?)
}

fn run_bench_market_judge_export(
    root: PathBuf,
    args: BenchMarketJudgeExportArgs,
) -> Result<serde_json::Value> {
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| root.join("benchmarks"));
    let report =
        export_market_head_judge_challenge(args.evaluator_pack, args.responses_dir, output_dir)?;
    Ok(serde_json::to_value(report)?)
}

fn run_bench_market_judge_evaluate(
    args: BenchMarketJudgeEvaluateArgs,
) -> Result<serde_json::Value> {
    let report =
        evaluate_market_head_judge_challenge(args.judge_manifest, args.responses_dir, &args.model)?;
    Ok(serde_json::to_value(report)?)
}

fn run_bench_market_judge_compare(args: BenchMarketJudgeCompareArgs) -> Result<serde_json::Value> {
    let judge_report_paths: Vec<BenchMarketJudgeCompareReportInput> = args
        .judge_reports
        .iter()
        .map(|entry| BenchMarketJudgeCompareReportInput {
            judge_head: entry.judge_head.clone(),
            report_path: entry.report_path.display().to_string(),
        })
        .collect();
    let judge_reports = args
        .judge_reports
        .into_iter()
        .map(|entry| (entry.judge_head, entry.report_path))
        .collect();
    let report = compare_market_head_same_pack(
        &args.challenged_head,
        &args.canonical_report,
        judge_reports,
    )?;
    let markdown = render_market_head_same_pack_markdown(&report);
    let output_dir = args.output_dir.unwrap_or_else(|| {
        args.canonical_report
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    });
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("creating output dir {}", output_dir.display()))?;
    let artifact_stem = format!(
        "market-head-judge-same-pack-{}",
        sanitize_market_head_artifact_component(&args.challenged_head)
    );
    let json_path = output_dir.join(format!("{artifact_stem}.json"));
    let markdown_path = output_dir.join(format!("{artifact_stem}.md"));
    fs::write(&json_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("writing same-pack JSON {}", json_path.display()))?;
    fs::write(&markdown_path, markdown)
        .with_context(|| format!("writing same-pack markdown {}", markdown_path.display()))?;

    Ok(serde_json::to_value(BenchMarketJudgeCompareReport {
        challenged_head: args.challenged_head,
        canonical_report_path: args.canonical_report.display().to_string(),
        judge_report_paths,
        json_path: json_path.display().to_string(),
        markdown_path: markdown_path.display().to_string(),
    })?)
}

fn run_bench_market_judge_pack_compare(
    args: BenchMarketJudgePackCompareArgs,
) -> Result<serde_json::Value> {
    let report = compare_market_head_judge_pack(args.same_pack_reports.clone())?;
    let markdown = render_market_head_judge_pack_markdown(&report);
    let output_dir = args.output_dir.unwrap_or_else(|| {
        args.same_pack_reports
            .first()
            .and_then(|path| path.parent())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    });
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("creating output dir {}", output_dir.display()))?;
    let json_path = output_dir.join("market-head-judge-pack-comparison.json");
    let markdown_path = output_dir.join("market-head-judge-pack-comparison.md");
    fs::write(&json_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("writing pack compare JSON {}", json_path.display()))?;
    fs::write(&markdown_path, markdown)
        .with_context(|| format!("writing pack compare markdown {}", markdown_path.display()))?;

    Ok(serde_json::to_value(BenchMarketJudgePackCompareReport {
        same_pack_report_paths: args
            .same_pack_reports
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        json_path: json_path.display().to_string(),
        markdown_path: markdown_path.display().to_string(),
    })?)
}

fn run_bench_market_answer(args: BenchMarketAnswerArgs) -> Result<serde_json::Value> {
    let manifest_path =
        resolve_market_head_manifest_path(&args.responses_dir, args.manifest.as_deref())?;
    let content_root = manifest_path
        .parent()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "manifest {} has no parent directory",
                manifest_path.display()
            )
        })?
        .to_path_buf();
    let manifest: MarketHeadChallengeManifest = serde_json::from_slice(
        &fs::read(&manifest_path)
            .with_context(|| format!("reading manifest {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parsing manifest {}", manifest_path.display()))?;
    let model = args
        .model
        .clone()
        .unwrap_or_else(|| default_market_head_model(&args.provider).to_string());
    let codex_home = match args.provider {
        BenchMarketProvider::Codex => Some(prepare_bare_codex_home()?),
        BenchMarketProvider::Claude => None,
    };

    let mut cases = Vec::with_capacity(manifest.cases.len());
    for case in &manifest.cases {
        let prompt_text = fs::read_to_string(content_root.join(&case.prompt_path))
            .with_context(|| format!("reading prompt for {}", case.scenario_id))?;
        let template_text = fs::read_to_string(content_root.join(&case.template_path))
            .with_context(|| format!("reading template for {}", case.scenario_id))?;
        let schema_text = fs::read_to_string(content_root.join(&case.schema_path))
            .with_context(|| format!("reading schema for {}", case.scenario_id))?;
        let response_path = content_root.join(&case.response_path);
        let class_dir = response_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("missing class directory for {}", case.scenario_id))?;
        let prompt = build_market_head_answer_prompt(&prompt_text, &template_text);

        match args.provider {
            BenchMarketProvider::Codex => run_codex_market_head_answer(
                codex_home
                    .as_ref()
                    .expect("codex provider requires prepared bare home"),
                class_dir,
                &response_path,
                &schema_text,
                &model,
                &prompt,
            )?,
            BenchMarketProvider::Claude => run_claude_market_head_answer(
                class_dir,
                &response_path,
                &schema_text,
                &model,
                &prompt,
            )?,
        }

        cases.push(BenchMarketAnswerCaseReport {
            class: case.class,
            scenario_id: case.scenario_id.clone(),
            response_path: response_path.display().to_string(),
        });
    }

    let report = BenchMarketAnswerReport {
        generated_at: Utc::now().to_rfc3339(),
        provider: args.provider,
        model,
        responses_dir: content_root.display().to_string(),
        manifest_path: manifest_path.display().to_string(),
        case_count: cases.len(),
        cases,
    };
    Ok(serde_json::to_value(report)?)
}

fn run_bench_market_judge_calibrate(
    args: BenchMarketJudgeCalibrateArgs,
) -> Result<serde_json::Value> {
    let report = compare_market_head_judge_calibration(args.same_pack_reports.clone())?;
    let markdown = render_market_head_judge_calibration_markdown(&report);
    let output_dir = args.output_dir.unwrap_or_else(|| {
        args.same_pack_reports
            .first()
            .and_then(|path| path.parent().map(Path::to_path_buf))
            .unwrap_or_else(|| PathBuf::from("."))
    });
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("creating calibration output dir {}", output_dir.display()))?;

    let json_path = output_dir.join("market-head-judge-calibration.json");
    let markdown_path = output_dir.join("market-head-judge-calibration.md");
    fs::write(&json_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("writing judge calibration JSON {}", json_path.display()))?;
    fs::write(&markdown_path, markdown).with_context(|| {
        format!(
            "writing judge calibration markdown {}",
            markdown_path.display()
        )
    })?;

    let result = BenchMarketJudgeCalibrateReport {
        same_pack_report_paths: args
            .same_pack_reports
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        json_path: json_path.display().to_string(),
        markdown_path: markdown_path.display().to_string(),
    };
    Ok(serde_json::to_value(result)?)
}

fn run_bench_market_judge_disagreement(
    args: BenchMarketJudgeDisagreementArgs,
) -> Result<serde_json::Value> {
    let report = compare_market_head_judge_disagreement(
        args.challenged_head.clone(),
        args.judge_head.clone(),
        &args.canonical_report,
        &args.judge_report,
    )?;
    let markdown = render_market_head_judge_disagreement_markdown(&report);
    let output_dir = args.output_dir.unwrap_or_else(|| {
        args.canonical_report
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    });
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("creating disagreement output dir {}", output_dir.display()))?;
    let stem = format!(
        "market-head-judge-disagreement-{}-{}",
        sanitize_market_head_artifact_component(&args.challenged_head),
        sanitize_market_head_artifact_component(&args.judge_head),
    );
    let json_path = output_dir.join(format!("{stem}.json"));
    let markdown_path = output_dir.join(format!("{stem}.md"));
    fs::write(&json_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("writing disagreement JSON {}", json_path.display()))?;
    fs::write(&markdown_path, markdown)
        .with_context(|| format!("writing disagreement markdown {}", markdown_path.display()))?;

    let result = BenchMarketJudgeDisagreementCliReport {
        challenged_head: args.challenged_head,
        judge_head: args.judge_head,
        canonical_report_path: args.canonical_report.display().to_string(),
        judge_report_path: args.judge_report.display().to_string(),
        json_path: json_path.display().to_string(),
        markdown_path: markdown_path.display().to_string(),
    };
    Ok(serde_json::to_value(result)?)
}

fn default_market_head_model(provider: &BenchMarketProvider) -> &'static str {
    match provider {
        BenchMarketProvider::Codex => "gpt-5.4",
        BenchMarketProvider::Claude => "opus",
    }
}

fn build_market_head_answer_prompt(prompt_text: &str, template_text: &str) -> String {
    format!(
        "{prompt_text}\n\nTemplate JSON shape:\n{template_text}\n\nReturn exactly one JSON object matching the provided schema. Use only information present above. No markdown. No commentary."
    )
}

fn resolve_market_head_manifest_path(
    responses_dir: &Path,
    explicit_manifest: Option<&Path>,
) -> Result<PathBuf> {
    if let Some(path) = explicit_manifest {
        return Ok(path.to_path_buf());
    }

    let candidates = [
        responses_dir.join("challenge-manifest.json"),
        responses_dir.join("judge-manifest.json"),
        responses_dir
            .join("market-head-judge")
            .join("judge-manifest.json"),
    ];
    let existing: Vec<PathBuf> = candidates
        .into_iter()
        .filter(|path| path.exists())
        .collect();

    match existing.as_slice() {
        [only] => Ok(only.clone()),
        [] => anyhow::bail!(
            "no market-head manifest found under {}; expected challenge-manifest.json, judge-manifest.json, or market-head-judge/judge-manifest.json",
            responses_dir.display()
        ),
        [challenge, ..] if challenge.ends_with("challenge-manifest.json") => Ok(challenge.clone()),
        [first, ..] => Ok(first.clone()),
    }
}

fn parse_named_judge_report(raw: &str) -> std::result::Result<NamedJudgeReportArg, String> {
    let (judge_head, report_path) = raw
        .split_once('=')
        .ok_or_else(|| format!("expected <judge-head>=<path>, got `{raw}`"))?;
    let judge_head = judge_head.trim();
    let report_path = report_path.trim();
    if judge_head.is_empty() {
        return Err(format!("missing judge head in `{raw}`"));
    }
    if report_path.is_empty() {
        return Err(format!("missing report path in `{raw}`"));
    }
    Ok(NamedJudgeReportArg {
        judge_head: judge_head.to_string(),
        report_path: PathBuf::from(report_path),
    })
}

fn sanitize_market_head_artifact_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut previous_dash = false;
    for ch in value.chars() {
        let mapped = if ch.is_ascii_alphanumeric() {
            previous_dash = false;
            ch.to_ascii_lowercase()
        } else {
            if previous_dash {
                continue;
            }
            previous_dash = true;
            '-'
        };
        out.push(mapped);
    }
    let cleaned = out.trim_matches('-').to_string();
    if cleaned.is_empty() {
        "pack".to_string()
    } else {
        cleaned
    }
}

fn run_codex_market_head_answer(
    bare_home: &TempDir,
    class_dir: &Path,
    response_path: &Path,
    schema_text: &str,
    model: &str,
    prompt: &str,
) -> Result<()> {
    let schema_path = class_dir.join("response.schema.json");
    fs::write(&schema_path, schema_text)
        .with_context(|| format!("writing schema {}", schema_path.display()))?;
    let output = StdCommand::new("codex")
        .current_dir(class_dir)
        .env("CODEX_HOME", bare_home.path())
        .arg("exec")
        .arg("--skip-git-repo-check")
        .arg("--ephemeral")
        .arg("--dangerously-bypass-approvals-and-sandbox")
        .arg("-m")
        .arg(model)
        .arg("--output-schema")
        .arg(&schema_path)
        .arg("-o")
        .arg(response_path)
        .arg(prompt)
        .output()
        .context("running codex market-head answer")?;
    if !output.status.success() {
        anyhow::bail!(
            "codex market-head answer failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

fn run_claude_market_head_answer(
    class_dir: &Path,
    response_path: &Path,
    schema_text: &str,
    model: &str,
    prompt: &str,
) -> Result<()> {
    let output = StdCommand::new("claude")
        .current_dir(class_dir)
        .arg("-p")
        .arg("--model")
        .arg(model)
        .arg("--permission-mode")
        .arg("bypassPermissions")
        .arg("--output-format")
        .arg("json")
        .arg("--json-schema")
        .arg(schema_text)
        .arg(prompt)
        .output()
        .context("running claude market-head answer")?;
    if !output.status.success() {
        anyhow::bail!(
            "claude market-head answer failed: {}",
            describe_claude_failure(&output)
        );
    }
    let structured = extract_claude_structured_output(&String::from_utf8_lossy(&output.stdout))?;
    fs::write(response_path, serde_json::to_vec_pretty(&structured)?)
        .with_context(|| format!("writing response {}", response_path.display()))?;
    Ok(())
}

fn describe_claude_failure(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !stderr.is_empty() {
        return stderr;
    }

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if stdout.is_empty() {
        return "unknown Claude failure".to_string();
    }

    serde_json::from_str::<Value>(&stdout)
        .ok()
        .and_then(|parsed| {
            parsed
                .get("result")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .filter(|message| !message.is_empty())
        .unwrap_or(stdout)
}

fn extract_claude_structured_output(stdout: &str) -> Result<Value> {
    let parsed: Value = serde_json::from_str(stdout).context("parsing Claude JSON output")?;
    if parsed
        .get("is_error")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        anyhow::bail!(
            "Claude returned an error: {}",
            parsed
                .get("result")
                .and_then(Value::as_str)
                .unwrap_or("unknown Claude error")
        );
    }
    parsed
        .get("structured_output")
        .cloned()
        .context("Claude output did not include structured_output")
}

fn prepare_bare_codex_home() -> Result<TempDir> {
    let source_home = std::env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".codex")))
        .context("HOME or CODEX_HOME must be set to prepare bare Codex home")?;
    let source_auth = source_home.join("auth.json");
    if !source_auth.is_file() {
        anyhow::bail!(
            "missing Codex auth at {}; run `codex login` first",
            source_auth.display()
        );
    }
    let bare_home = tempfile::tempdir().context("creating temporary bare Codex home")?;
    fs::copy(&source_auth, bare_home.path().join("auth.json"))
        .with_context(|| format!("copying Codex auth from {}", source_auth.display()))?;
    Ok(bare_home)
}

async fn run_legacy_bench(engine: Arc<Engine>, args: BenchArgs) -> Result<BenchResult> {
    let mut fabric_hits = 0usize;
    let mut fabric_support_hits = 0usize;
    let mut recent_hits = 0usize;
    let mut recent_support_hits = 0usize;
    let mut vector_hits = 0usize;
    let mut vector_support_hits = 0usize;
    let mut summary_hits = 0usize;
    let mut summary_support_hits = 0usize;
    let mut total_query_ms = 0u128;
    let mut total_pack_tokens = 0usize;
    let mut total_view_conflicts = 0usize;

    for case in 0..args.cases {
        let session = format!("bench-session-{case}");
        let task = format!("bench-task-{case}");
        let planner = format!("bench-planner-{}", case % 3);
        let debugger = format!("bench-debugger-{}", case % 3);
        let answer = format!("checkpoint-{:03}", case);
        let subsystem = format!("subsystem-{:03}", case);
        let endpoint = format!("/v1/bench/{case}");
        let support_marker = "src/query.rs";

        engine.ingest(EventInput {
            kind: EventKind::Note,
            agent_id: planner.clone(),
            agent_role: Some("planner".to_string()),
            timestamp: None,
            session_id: session.clone(),
            task_id: Some(task.clone()),
            project_id: Some("bench-project".to_string()),
            goal_id: Some("benchmark".to_string()),
            run_id: Some(format!("bench-run-{case}")),
            namespace: Some("bench".to_string()),
            environment: Some("local".to_string()),
            source: "bench".to_string(),
            scope: Scope::Project,
            tags: vec!["fact".to_string()],
            dimensions: vec![
                DimensionValue {
                    key: "subsystem".into(),
                    value: subsystem.clone(),
                    weight: 100,
                },
                DimensionValue {
                    key: "endpoint".into(),
                    value: endpoint.clone(),
                    weight: 100,
                },
                DimensionValue {
                    key: "claim.checkpoint".into(),
                    value: answer.clone(),
                    weight: 100,
                },
            ],
            content: format!("{subsystem} must use {answer} when replaying archived context."),
            attributes: serde_json::json!({"case": case, "type": "fact"}),
        })?;
        engine.ingest(EventInput {
            kind: EventKind::Note,
            agent_id: debugger.clone(),
            agent_role: Some("debugger".to_string()),
            timestamp: None,
            session_id: format!("bench-shared-session-{case}"),
            task_id: Some(format!("bench-shared-task-{case}")),
            project_id: Some("bench-project".to_string()),
            goal_id: Some("benchmark".to_string()),
            run_id: Some(format!("bench-run-{case}")),
            namespace: Some("bench".to_string()),
            environment: Some("local".to_string()),
            source: "bench".to_string(),
            scope: Scope::Project,
            tags: vec!["incident".to_string()],
            dimensions: vec![
                DimensionValue { key: "subsystem".into(), value: subsystem.clone(), weight: 100 },
                DimensionValue { key: "endpoint".into(), value: endpoint.clone(), weight: 100 },
                DimensionValue { key: "file".into(), value: support_marker.into(), weight: 100 },
            ],
            content: format!("Debugger notes from another session: {subsystem} fails unless {answer} is replayed before distractors, and {support_marker} exposed the failure."),
            attributes: serde_json::json!({"case": case, "type": "support"}),
        })?;

        for step in 0..args.distractors {
            engine.ingest(EventInput {
                kind: EventKind::Note,
                agent_id: if step % 2 == 0 {
                    planner.clone()
                } else {
                    debugger.clone()
                },
                agent_role: Some("bench".to_string()),
                timestamp: None,
                session_id: session.clone(),
                task_id: Some(task.clone()),
                project_id: Some("bench-project".to_string()),
                goal_id: Some("benchmark".to_string()),
                run_id: Some(format!("bench-run-{case}")),
                namespace: Some("bench".to_string()),
                environment: Some("local".to_string()),
                source: "bench".to_string(),
                scope: Scope::Shared,
                tags: vec!["noise".to_string()],
                dimensions: vec![
                    DimensionValue {
                        key: "subsystem".into(),
                        value: format!("noise-{case}-{step}"),
                        weight: 100,
                    },
                    DimensionValue {
                        key: "endpoint".into(),
                        value: format!("/noise/{step}"),
                        weight: 100,
                    },
                ],
                content: format!(
                    "distractor step {step} for {subsystem}: metrics={}, retries={}, queue=stable",
                    step * 3,
                    step % 5
                ),
                attributes: serde_json::json!({"case": case, "type": "noise", "step": step}),
            })?;
        }

        let query = format!(
            "What checkpoint should {subsystem} use when replaying archived context, and which file exposed the failure?"
        );
        let recent = engine.replay(Some(&session), args.recent_window)?;
        if recent
            .iter()
            .any(|row| row.event.input.content.contains(&answer))
        {
            recent_hits += 1;
        }
        if recent
            .iter()
            .any(|row| row.event.input.content.contains(support_marker))
        {
            recent_support_hits += 1;
        }
        if engine
            .vector_baseline(
                &query,
                Some(&session),
                Some(&task),
                None,
                args.recent_window,
            )?
            .iter()
            .any(|memory| memory.body.contains(&answer))
        {
            vector_hits += 1;
        }
        if engine
            .vector_baseline(
                &query,
                Some(&session),
                Some(&task),
                None,
                args.recent_window,
            )?
            .iter()
            .any(|memory| memory.body.contains(support_marker))
        {
            vector_support_hits += 1;
        }
        if engine
            .summary_baseline(Some(&session), Some(&task), None, args.recent_window)?
            .iter()
            .any(|memory| memory.body.contains(&answer))
        {
            summary_hits += 1;
        }
        if engine
            .summary_baseline(Some(&session), Some(&task), None, args.recent_window)?
            .iter()
            .any(|memory| memory.body.contains(support_marker))
        {
            summary_support_hits += 1;
        }

        let selector = Selector {
            all: vec![
                ice::model::DimensionFilter {
                    key: "project".into(),
                    values: vec!["bench-project".into()],
                },
                ice::model::DimensionFilter {
                    key: "subsystem".into(),
                    values: vec![subsystem.clone()],
                },
            ],
            any: vec![ice::model::DimensionFilter {
                key: "endpoint".into(),
                values: vec![endpoint.clone()],
            }],
            exclude: Vec::new(),
            layers: Vec::new(),
            start_ts: None,
            end_ts: None,
            limit: Some(16),
            namespace: Some("bench".into()),
        };
        let view = engine.materialize_view(ViewInput {
            op: ViewOp::Snapshot,
            owner_agent_id: Some(planner.clone()),
            namespace: Some("bench".into()),
            objective: Some("benchmark fabric slice".into()),
            selectors: vec![selector.clone()],
            source_view_ids: Vec::new(),
            resolution: Some(SnapshotResolution::Medium),
            limit: Some(16),
        })?;
        total_view_conflicts += view.conflict_count;

        let started = Instant::now();
        let pack = engine.build_context_pack(QueryInput {
            agent_id: Some(planner),
            session_id: Some(session),
            task_id: Some(task),
            namespace: Some("bench".to_string()),
            objective: Some("benchmark recall".to_string()),
            selector: Some(selector),
            view_id: Some(view.id),
            query_text: query,
            budget_tokens: args.budget_tokens,
            candidate_limit: args.candidate_limit,
        })?;
        total_query_ms += started.elapsed().as_millis();
        total_pack_tokens += pack.used_tokens;
        if pack.items.iter().any(|item| item.body.contains(&answer)) {
            fabric_hits += 1;
        }
        if pack
            .items
            .iter()
            .any(|item| item.body.contains(support_marker))
        {
            fabric_support_hits += 1;
        }
    }

    Ok(BenchResult {
        cases: args.cases,
        distractors_per_case: args.distractors,
        recent_window: args.recent_window,
        fabric_hits,
        fabric_support_hits,
        baseline_recent_hits: recent_hits,
        baseline_recent_support_hits: recent_support_hits,
        baseline_vector_hits: vector_hits,
        baseline_vector_support_hits: vector_support_hits,
        baseline_summary_hits: summary_hits,
        baseline_summary_support_hits: summary_support_hits,
        fabric_recall: fabric_hits as f64 / args.cases as f64,
        fabric_support_recall: fabric_support_hits as f64 / args.cases as f64,
        baseline_recent_recall: recent_hits as f64 / args.cases as f64,
        baseline_recent_support_recall: recent_support_hits as f64 / args.cases as f64,
        baseline_vector_recall: vector_hits as f64 / args.cases as f64,
        baseline_vector_support_recall: vector_support_hits as f64 / args.cases as f64,
        baseline_summary_recall: summary_hits as f64 / args.cases as f64,
        baseline_summary_support_recall: summary_support_hits as f64 / args.cases as f64,
        avg_fabric_query_ms: total_query_ms as f64 / args.cases as f64,
        avg_fabric_pack_tokens: total_pack_tokens as f64 / args.cases as f64,
        avg_view_conflicts: total_view_conflicts as f64 / args.cases as f64,
    })
}
