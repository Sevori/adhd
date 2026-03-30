use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};
use std::{cmp::Ordering, collections::BTreeSet};

use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, NaiveDateTime, Utc};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::Engine;
use crate::model::{ContextPack, EventInput, EventKind, QueryInput, Scope};

#[derive(Debug, Clone)]
pub struct LongMemEvalRunConfig {
    pub dataset_path: PathBuf,
    pub output_path: PathBuf,
    pub work_dir: PathBuf,
    pub namespace_prefix: String,
    pub reader_provider: LongMemEvalReaderProvider,
    pub reader_endpoint: String,
    pub reader_model: String,
    pub reader_api_key_env: Option<String>,
    pub reader_timeout_secs: u64,
    pub reader_num_predict: usize,
    pub reader_max_retries: usize,
    pub reader_retry_backoff_secs: u64,
    pub budget_tokens: usize,
    pub candidate_limit: usize,
    pub offset: usize,
    pub max_cases: Option<usize>,
    pub question_ids: Vec<String>,
    pub question_types: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LongMemEvalReaderProvider {
    Ollama,
    OpenAiCompatible,
}

#[derive(Debug, Clone)]
pub struct LongMemEvalEvaluateConfig {
    pub repo_path: PathBuf,
    pub predictions_path: PathBuf,
    pub dataset_path: PathBuf,
    pub python_bin: String,
    pub judge_model: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LongMemEvalRunReport {
    pub generated_at: String,
    pub dataset_path: String,
    pub output_path: String,
    pub report_path: String,
    pub debug_root: String,
    pub work_dir: String,
    pub reader_provider: String,
    pub reader_model: String,
    pub reader_endpoint: String,
    pub reader_max_retries: usize,
    pub budget_tokens: usize,
    pub candidate_limit: usize,
    pub total_cases: usize,
    pub selected_cases: usize,
    pub executed_cases: usize,
    pub case_reports: Vec<LongMemEvalCaseReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LongMemEvalCaseReport {
    pub question_id: String,
    pub question_type: String,
    pub question: String,
    pub question_date: String,
    pub hypothesis: String,
    pub duration_ms: u128,
    pub case_root: String,
    pub context_manifest_path: String,
    pub prompt_path: String,
    pub response_path: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LongMemEvalEvaluateReport {
    pub generated_at: String,
    pub repo_path: String,
    pub predictions_path: String,
    pub dataset_path: String,
    pub judge_model: String,
    pub result_path: String,
    pub evaluate_stdout: String,
    pub metrics_stdout: Option<String>,
    pub metrics_skipped_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct LongMemEvalCase {
    question_id: String,
    question_type: String,
    question: String,
    question_date: String,
    haystack_dates: Vec<String>,
    haystack_session_ids: Vec<String>,
    haystack_sessions: Vec<Vec<LongMemEvalTurn>>,
}

#[derive(Debug, Clone, Deserialize)]
struct LongMemEvalTurn {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: OllamaGenerateOptions,
}

#[derive(Debug, Clone, Serialize)]
struct OllamaGenerateOptions {
    temperature: f64,
    num_predict: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct OllamaGenerateResponse {
    response: String,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAiChatCompletionRequest {
    model: String,
    messages: Vec<OpenAiChatMessage>,
    temperature: f64,
    max_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OpenAiChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAiChatCompletionResponse {
    choices: Vec<OpenAiChatChoice>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAiChatChoice {
    message: OpenAiChatResponseMessage,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAiChatResponseMessage {
    content: OpenAiMessageContent,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum OpenAiMessageContent {
    Text(String),
    Parts(Vec<OpenAiContentPart>),
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAiContentPart {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

pub fn run_longmemeval(config: LongMemEvalRunConfig) -> Result<LongMemEvalRunReport> {
    fs::create_dir_all(&config.work_dir).with_context(|| {
        format!(
            "creating LongMemEval work dir {}",
            config.work_dir.display()
        )
    })?;
    if let Some(parent) = config.output_path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "creating LongMemEval output dir {}",
                parent.as_os_str().to_string_lossy()
            )
        })?;
    }

    let dataset = read_dataset(&config.dataset_path)?;
    let total_cases = dataset.len();
    let filtered = filter_cases(dataset, &config.question_ids, &config.question_types);
    let selected_cases = filtered.len();
    let selected = filtered
        .into_iter()
        .skip(config.offset)
        .take(config.max_cases.unwrap_or(usize::MAX))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        bail!("LongMemEval selection is empty after filters/offset");
    }

    let report_path = append_path_suffix(&config.output_path, ".report.json");
    let debug_root = append_path_suffix(&config.output_path, ".debug");
    if debug_root.exists() {
        fs::remove_dir_all(&debug_root)
            .with_context(|| format!("removing previous debug root {}", debug_root.display()))?;
    }
    fs::create_dir_all(&debug_root)
        .with_context(|| format!("creating debug root {}", debug_root.display()))?;

    let client = Client::builder()
        .timeout(Duration::from_secs(config.reader_timeout_secs.max(5)))
        .build()
        .context("building LongMemEval reader client")?;
    let mut output = File::create(&config.output_path)
        .with_context(|| format!("creating predictions file {}", config.output_path.display()))?;
    let dataset_label = dataset_label(&config.dataset_path);
    let mut case_reports = Vec::with_capacity(selected.len());

    for case in selected {
        let started = Instant::now();
        let question_slug = sanitize_fs_component(&case.question_id);
        let case_root = config.work_dir.join(&question_slug);
        if case_root.exists() {
            fs::remove_dir_all(&case_root)
                .with_context(|| format!("removing previous case root {}", case_root.display()))?;
        }
        fs::create_dir_all(&case_root)
            .with_context(|| format!("creating case root {}", case_root.display()))?;
        let engine = Engine::open(&case_root)
            .with_context(|| format!("opening ICE engine for {}", case.question_id))?;

        replay_case(&engine, &config.namespace_prefix, &dataset_label, &case)?;

        let namespace = case_namespace(&config.namespace_prefix, &dataset_label, &case.question_id);
        let pack = engine.build_context_pack(QueryInput {
            agent_id: Some("longmemeval-reader".to_string()),
            session_id: None,
            task_id: None,
            namespace: Some(namespace),
            objective: Some(format!(
                "Answer the benchmark question using only recalled memory. Question date: {}.",
                case.question_date
            )),
            selector: None,
            view_id: None,
            query_text: case.question.clone(),
            budget_tokens: config.budget_tokens,
            candidate_limit: config.candidate_limit,
        })?;
        let context_text = render_context_pack_text(&pack, &case.question);
        let prompt = render_answer_prompt(&case, &context_text);
        let hypothesis = generate_answer(
            &client,
            config.reader_provider,
            &config.reader_endpoint,
            &config.reader_model,
            config.reader_api_key_env.as_deref(),
            config.reader_num_predict,
            config.reader_max_retries,
            config.reader_retry_backoff_secs,
            &prompt,
        )
        .with_context(|| format!("running reader for {}", case.question_id))?;

        writeln!(
            output,
            "{}",
            serde_json::to_string(&json!({
                "question_id": case.question_id,
                "hypothesis": hypothesis
            }))?
        )
        .with_context(|| format!("writing prediction for {}", case.question_id))?;

        let case_debug_root = debug_root.join(&question_slug);
        fs::create_dir_all(&case_debug_root)
            .with_context(|| format!("creating case debug root {}", case_debug_root.display()))?;
        let prompt_path = case_debug_root.join("prompt.txt");
        let response_path = case_debug_root.join("response.txt");
        let pack_path = case_debug_root.join("context-pack.json");
        fs::write(&prompt_path, prompt.as_bytes())
            .with_context(|| format!("writing prompt {}", prompt_path.display()))?;
        fs::write(&response_path, hypothesis.as_bytes())
            .with_context(|| format!("writing response {}", response_path.display()))?;
        fs::write(&pack_path, serde_json::to_vec_pretty(&pack)?)
            .with_context(|| format!("writing context pack {}", pack_path.display()))?;

        case_reports.push(LongMemEvalCaseReport {
            question_id: case.question_id,
            question_type: case.question_type,
            question: case.question,
            question_date: case.question_date,
            hypothesis,
            duration_ms: started.elapsed().as_millis(),
            case_root: case_root.display().to_string(),
            context_manifest_path: pack.manifest_path,
            prompt_path: prompt_path.display().to_string(),
            response_path: response_path.display().to_string(),
        });
    }

    let report = LongMemEvalRunReport {
        generated_at: Utc::now().to_rfc3339(),
        dataset_path: config.dataset_path.display().to_string(),
        output_path: config.output_path.display().to_string(),
        report_path: report_path.display().to_string(),
        debug_root: debug_root.display().to_string(),
        work_dir: config.work_dir.display().to_string(),
        reader_provider: reader_provider_label(config.reader_provider).to_string(),
        reader_model: config.reader_model,
        reader_endpoint: config.reader_endpoint,
        reader_max_retries: config.reader_max_retries,
        budget_tokens: config.budget_tokens,
        candidate_limit: config.candidate_limit,
        total_cases,
        selected_cases,
        executed_cases: case_reports.len(),
        case_reports,
    };

    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("writing LongMemEval report {}", report_path.display()))?;
    Ok(report)
}

pub fn evaluate_longmemeval(
    config: LongMemEvalEvaluateConfig,
) -> Result<LongMemEvalEvaluateReport> {
    let eval_dir = config.repo_path.join("src").join("evaluation");
    let evaluate_script = eval_dir.join("evaluate_qa.py");
    let metrics_script = eval_dir.join("print_qa_metrics.py");
    if !evaluate_script.exists() {
        bail!(
            "official LongMemEval evaluator not found at {}",
            evaluate_script.display()
        );
    }
    if !metrics_script.exists() {
        bail!(
            "official LongMemEval metrics script not found at {}",
            metrics_script.display()
        );
    }

    let result_path = PathBuf::from(format!(
        "{}.eval-results-{}",
        config.predictions_path.display(),
        config.judge_model
    ));
    let evaluate_output = Command::new(&config.python_bin)
        .arg(&evaluate_script)
        .arg(&config.judge_model)
        .arg(&config.predictions_path)
        .arg(&config.dataset_path)
        .current_dir(&eval_dir)
        .output()
        .with_context(|| format!("running {}", evaluate_script.display()))?;
    if !evaluate_output.status.success() {
        bail!(
            "official LongMemEval evaluation failed with status {}\nstdout:\n{}\nstderr:\n{}",
            evaluate_output.status,
            String::from_utf8_lossy(&evaluate_output.stdout),
            String::from_utf8_lossy(&evaluate_output.stderr),
        );
    }

    let evaluate_stdout = String::from_utf8(evaluate_output.stdout)
        .context("official LongMemEval evaluator stdout was not valid UTF-8")?;

    let (metrics_stdout, metrics_skipped_reason) = if config.judge_model == "gpt-4o" {
        let metrics_output = Command::new(&config.python_bin)
            .arg(&metrics_script)
            .arg(&result_path)
            .arg(&config.dataset_path)
            .current_dir(&eval_dir)
            .output()
            .with_context(|| format!("running {}", metrics_script.display()))?;
        if !metrics_output.status.success() {
            bail!(
                "official LongMemEval metrics failed with status {}\nstdout:\n{}\nstderr:\n{}",
                metrics_output.status,
                String::from_utf8_lossy(&metrics_output.stdout),
                String::from_utf8_lossy(&metrics_output.stderr),
            );
        }
        (
            Some(
                String::from_utf8(metrics_output.stdout)
                    .context("official LongMemEval metrics stdout was not valid UTF-8")?,
            ),
            None,
        )
    } else {
        (
            None,
            Some(
                "Skipped print_qa_metrics.py because the official script hardcodes gpt-4o-only labels."
                    .to_string(),
            ),
        )
    };

    Ok(LongMemEvalEvaluateReport {
        generated_at: Utc::now().to_rfc3339(),
        repo_path: config.repo_path.display().to_string(),
        predictions_path: config.predictions_path.display().to_string(),
        dataset_path: config.dataset_path.display().to_string(),
        judge_model: config.judge_model,
        result_path: result_path.display().to_string(),
        evaluate_stdout,
        metrics_stdout,
        metrics_skipped_reason,
    })
}

fn read_dataset(path: &Path) -> Result<Vec<LongMemEvalCase>> {
    serde_json::from_slice(
        &fs::read(path)
            .with_context(|| format!("reading LongMemEval dataset {}", path.display()))?,
    )
    .with_context(|| format!("parsing LongMemEval dataset {}", path.display()))
}

fn filter_cases(
    dataset: Vec<LongMemEvalCase>,
    question_ids: &[String],
    question_types: &[String],
) -> Vec<LongMemEvalCase> {
    let wanted_ids = question_ids
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .collect::<std::collections::BTreeSet<_>>();
    let wanted_types = question_types
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .collect::<std::collections::BTreeSet<_>>();
    dataset
        .into_iter()
        .filter(|case| {
            (wanted_ids.is_empty() || wanted_ids.contains(case.question_id.as_str()))
                && (wanted_types.is_empty() || wanted_types.contains(case.question_type.as_str()))
        })
        .collect()
}

fn replay_case(
    engine: &Engine,
    namespace_prefix: &str,
    dataset_label: &str,
    case: &LongMemEvalCase,
) -> Result<()> {
    if case.haystack_dates.len() != case.haystack_session_ids.len()
        || case.haystack_dates.len() != case.haystack_sessions.len()
    {
        bail!(
            "LongMemEval case {} has mismatched haystack lengths",
            case.question_id
        );
    }

    let mut sessions = case
        .haystack_dates
        .iter()
        .zip(case.haystack_session_ids.iter())
        .zip(case.haystack_sessions.iter())
        .enumerate()
        .map(|(index, ((raw_date, session_id), turns))| {
            Ok(ReplaySession {
                index,
                timestamp: parse_longmemeval_datetime(raw_date)
                    .with_context(|| format!("parsing haystack date {raw_date}"))?,
                raw_date: raw_date.clone(),
                session_id: session_id.clone(),
                content: format_session_document(session_id, raw_date, turns),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    sessions.sort_by_key(|session| session.timestamp);

    let namespace = case_namespace(namespace_prefix, dataset_label, &case.question_id);
    for session in sessions {
        engine.ingest(EventInput {
            kind: EventKind::Document,
            agent_id: "longmemeval-importer".to_string(),
            agent_role: Some("importer".to_string()),
            timestamp: Some(session.timestamp),
            session_id: session.session_id.clone(),
            task_id: None,
            project_id: Some("longmemeval".to_string()),
            goal_id: Some(case.question_id.clone()),
            run_id: None,
            namespace: Some(namespace.clone()),
            environment: Some("longmemeval".to_string()),
            source: "longmemeval".to_string(),
            scope: Scope::Shared,
            tags: vec![
                "longmemeval".to_string(),
                dataset_label.to_string(),
                case.question_type.clone(),
            ],
            dimensions: vec![],
            content: session.content,
            attributes: json!({
                "dataset": dataset_label,
                "question_id": case.question_id,
                "question_date": case.question_date,
                "question_type": case.question_type,
                "session_id": session.session_id,
                "session_date": session.raw_date,
                "session_index": session.index,
            }),
        })?;
    }
    Ok(())
}

fn render_context_pack_text(pack: &ContextPack, question: &str) -> String {
    if pack.items.is_empty() {
        return "No continuity items were retrieved.".to_string();
    }
    let mut rendered = pack
        .items
        .iter()
        .enumerate()
        .map(|(index, item)| render_context_item(index, item, question))
        .collect::<Vec<_>>();
    rendered.sort_by(|left, right| {
        left.sort_key
            .cmp(&right.sort_key)
            .then_with(|| {
                right
                    .final_score
                    .partial_cmp(&left.final_score)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| left.index.cmp(&right.index))
    });
    rendered
        .into_iter()
        .map(|item| item.rendered)
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn render_answer_prompt(case: &LongMemEvalCase, context_text: &str) -> String {
    format!(
        "You are answering a LongMemEval benchmark question with an ICE continuity pack.\n\
Return only the final answer as plain text.\n\
Use only the retrieved continuity evidence below.\n\
If the evidence does not support a specific answer, reply exactly: I don't know\n\
Think silently before answering.\n\
Resolve the answer from the retrieved sessions, not from generic world knowledge.\n\
Pay attention to who said each fact: `User:` lines describe the human, while `Assistant:` lines describe prior recommendations or answers.\n\
For preference questions, infer what the human would likely want from their stated interests, goals, and prior choices.\n\
For counting questions, count distinct items or tasks across all retrieved sessions before answering.\n\
For temporal or update questions, use the session dates and prefer the latest evidence that predates the question date.\n\
Do not mention ICE, context packs, memories, provenance, or benchmark metadata.\n\
\n\
Question date: {question_date}\n\
Question: {question}\n\
\n\
Retrieved continuity:\n\
{context_text}\n",
        question_date = case.question_date,
        question = case.question,
        context_text = context_text,
    )
}

#[derive(Debug, Clone)]
struct RenderedContextItem {
    index: usize,
    sort_key: Option<String>,
    final_score: f64,
    rendered: String,
}

fn render_context_item(
    index: usize,
    item: &crate::model::ContextPackItem,
    question: &str,
) -> RenderedContextItem {
    let transcript = extract_longmemeval_transcript(&item.body);
    let session_date = extract_session_date_line(&transcript).map(str::to_string);
    let excerpt = focus_transcript_excerpt(&transcript, question, 1100);
    let header = match session_date.as_deref() {
        Some(date) => format!(
            "[m{}][{}][score={:.3}][session_date={}]",
            index + 1,
            item.layer,
            item.final_score,
            date
        ),
        None => format!(
            "[m{}][{}][score={:.3}]",
            index + 1,
            item.layer,
            item.final_score
        ),
    };

    RenderedContextItem {
        index,
        sort_key: session_date,
        final_score: item.final_score,
        rendered: format!("{header}\n{excerpt}"),
    }
}

fn extract_longmemeval_transcript(body: &str) -> String {
    let body = body.trim();
    let without_metadata = body
        .split_once("content=")
        .map(|(_, content)| content)
        .unwrap_or(body);
    let anchored = without_metadata
        .find("LongMemEval session:")
        .map(|index| &without_metadata[index..])
        .unwrap_or(without_metadata);
    anchored
        .lines()
        .take_while(|line| {
            let trimmed = line.trim_start();
            !trimmed.starts_with("| entities:") && !trimmed.contains(" | entities:")
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn extract_session_date_line(transcript: &str) -> Option<&str> {
    transcript
        .lines()
        .find_map(|line| line.trim().strip_prefix("Session date: "))
}

fn focus_transcript_excerpt(transcript: &str, question: &str, max_chars: usize) -> String {
    let lines = transcript
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return "No retrieved transcript content.".to_string();
    }

    let keywords = question_keywords(question);
    let header_len = lines
        .iter()
        .take(2)
        .filter(|line| {
            line.starts_with("LongMemEval session:") || line.starts_with("Session date:")
        })
        .count()
        .max(1);

    let mut scored = lines
        .iter()
        .enumerate()
        .skip(header_len)
        .map(|(index, line)| (index, score_line_against_question(line, &keywords)))
        .filter(|(_, score)| *score > 0)
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));

    if scored.is_empty() {
        return head_tail_excerpt(&lines, max_chars);
    }

    let mut included = BTreeSet::new();
    for index in 0..header_len.min(lines.len()) {
        included.insert(index);
    }
    for (index, _) in scored.into_iter().take(3) {
        for line_index in index.saturating_sub(1)..=(index + 2).min(lines.len().saturating_sub(1)) {
            included.insert(line_index);
        }
    }

    let ordered = included.into_iter().collect::<Vec<_>>();
    let mut rendered = Vec::new();
    let mut previous = None;
    for index in ordered {
        if previous.is_some_and(|last| index > last + 1) {
            rendered.push("...".to_string());
        }
        rendered.push(lines[index].to_string());
        previous = Some(index);
    }
    trim_text(&rendered.join("\n"), max_chars)
}

fn head_tail_excerpt(lines: &[&str], max_chars: usize) -> String {
    if lines.len() <= 8 {
        return trim_text(&lines.join("\n"), max_chars);
    }
    let mut rendered = lines
        .iter()
        .take(5)
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    rendered.push("...".to_string());
    let mut text = rendered.join("\n");
    text.push('\n');
    text.push_str(&lines[lines.len().saturating_sub(3)..].join("\n"));
    trim_text(&text, max_chars)
}

fn question_keywords(question: &str) -> Vec<String> {
    const STOPWORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "can", "did", "do", "does", "for", "from",
        "had", "has", "have", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "our",
        "please", "previous", "that", "the", "their", "this", "to", "was", "were", "what", "when",
        "where", "which", "who", "with", "you", "your",
    ];

    let mut keywords = question
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .map(|token| token.trim().to_ascii_lowercase())
        .filter(|token| token.len() >= 3)
        .filter(|token| !STOPWORDS.contains(&token.as_str()))
        .collect::<Vec<_>>();
    keywords.sort();
    keywords.dedup();
    keywords
}

fn score_line_against_question(line: &str, keywords: &[String]) -> usize {
    if keywords.is_empty() {
        return 0;
    }
    let line_lower = line.to_ascii_lowercase();
    keywords
        .iter()
        .filter(|keyword| line_lower.contains(keyword.as_str()))
        .count()
}

fn generate_answer(
    client: &Client,
    provider: LongMemEvalReaderProvider,
    endpoint: &str,
    model: &str,
    api_key_env: Option<&str>,
    num_predict: usize,
    max_retries: usize,
    retry_backoff_secs: u64,
    prompt: &str,
) -> Result<String> {
    let attempts = max_retries.max(1);
    let backoff = retry_backoff_secs.max(1);
    let mut last_error = None;

    for attempt in 1..=attempts {
        let result = match provider {
            LongMemEvalReaderProvider::Ollama => {
                generate_ollama_answer(client, endpoint, model, num_predict, prompt)
            }
            LongMemEvalReaderProvider::OpenAiCompatible => generate_openai_compatible_answer(
                client,
                endpoint,
                model,
                api_key_env,
                num_predict,
                prompt,
            ),
        };

        match result {
            Ok(answer) => return Ok(answer),
            Err(error) => {
                let retryable = is_retryable_reader_error(&error);
                if attempt >= attempts || !retryable {
                    return Err(error);
                }
                last_error = Some(error);
                std::thread::sleep(Duration::from_secs(backoff.saturating_mul(attempt as u64)));
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow!("LongMemEval reader failed without an error payload")))
}

fn generate_ollama_answer(
    client: &Client,
    endpoint: &str,
    model: &str,
    num_predict: usize,
    prompt: &str,
) -> Result<String> {
    let response = client
        .post(format!("{}/api/generate", endpoint.trim_end_matches('/')))
        .json(&OllamaGenerateRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            stream: false,
            options: OllamaGenerateOptions {
                temperature: 0.0,
                num_predict,
            },
        })
        .send()
        .with_context(|| format!("requesting LongMemEval answer from {model}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .unwrap_or_else(|error| format!("unable to read ollama error body: {error}"));
        return Err(anyhow!(
            "LongMemEval reader request failed for {} with {}: {}",
            model,
            status,
            body
        ));
    }
    let payload: OllamaGenerateResponse = response.json()?;
    Ok(normalize_hypothesis(&payload.response))
}

fn generate_openai_compatible_answer(
    client: &Client,
    endpoint: &str,
    model: &str,
    api_key_env: Option<&str>,
    num_predict: usize,
    prompt: &str,
) -> Result<String> {
    let endpoint = endpoint.trim_end_matches('/');
    let api_key = resolve_reader_api_key(api_key_env);
    if endpoint.contains("api.openai.com") && api_key.is_none() {
        bail!(
            "LongMemEval openai-compatible reader requires an API key; set {} or pass a different endpoint",
            api_key_env.unwrap_or("OPENAI_API_KEY")
        );
    }

    let mut request =
        client
            .post(format!("{endpoint}/chat/completions"))
            .json(&OpenAiChatCompletionRequest {
                model: model.to_string(),
                messages: vec![OpenAiChatMessage {
                    role: "user".to_string(),
                    content: prompt.to_string(),
                }],
                temperature: 0.0,
                max_tokens: num_predict,
            });
    if let Some(api_key) = api_key {
        request = request.bearer_auth(api_key);
    }

    let response = request
        .send()
        .with_context(|| format!("requesting openai-compatible LongMemEval answer from {model}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_else(|error| {
            format!("unable to read openai-compatible error body: {error}")
        });
        return Err(anyhow!(
            "LongMemEval openai-compatible reader request failed for {} with {}: {}",
            model,
            status,
            body
        ));
    }
    let payload: OpenAiChatCompletionResponse = response.json()?;
    let content = payload
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("openai-compatible chat completion returned no choices"))?
        .message
        .content
        .into_text();
    Ok(normalize_hypothesis(&content))
}

fn normalize_hypothesis(raw: &str) -> String {
    let trimmed = raw.trim().trim_matches('`').trim();
    if trimmed.is_empty() {
        return "I don't know".to_string();
    }
    let single_line = trimmed.lines().next().unwrap_or(trimmed).trim();
    let unquoted = single_line
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or(single_line)
        .trim();
    if unquoted.is_empty() {
        "I don't know".to_string()
    } else {
        unquoted.to_string()
    }
}

impl OpenAiMessageContent {
    fn into_text(self) -> String {
        match self {
            Self::Text(text) => text,
            Self::Parts(parts) => parts
                .into_iter()
                .filter_map(|part| if part.kind == "text" { part.text } else { None })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

fn resolve_reader_api_key(env_name: Option<&str>) -> Option<String> {
    env_name
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .and_then(|name| std::env::var(name).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn reader_provider_label(provider: LongMemEvalReaderProvider) -> &'static str {
    match provider {
        LongMemEvalReaderProvider::Ollama => "ollama",
        LongMemEvalReaderProvider::OpenAiCompatible => "openai-compatible",
    }
}

fn is_retryable_reader_error(error: &anyhow::Error) -> bool {
    let text = error.to_string();
    text.contains("429")
        || text.contains(" 500")
        || text.contains(" 502")
        || text.contains(" 503")
        || text.contains(" 504")
        || error.chain().any(|cause| {
            cause
                .downcast_ref::<reqwest::Error>()
                .is_some_and(|err| err.is_timeout() || err.is_connect() || err.is_request())
        })
}

fn parse_longmemeval_datetime(raw: &str) -> Result<DateTime<Utc>> {
    if let Ok(value) = DateTime::parse_from_rfc3339(raw) {
        return Ok(value.with_timezone(&Utc));
    }
    const FORMATS: &[&str] = &[
        "%Y/%m/%d (%a) %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ];
    for format in FORMATS {
        if let Ok(value) = NaiveDateTime::parse_from_str(raw, format) {
            return Ok(DateTime::<Utc>::from_naive_utc_and_offset(value, Utc));
        }
    }
    bail!("unsupported LongMemEval date format: {raw}")
}

fn format_session_document(session_id: &str, raw_date: &str, turns: &[LongMemEvalTurn]) -> String {
    let mut lines = vec![
        format!("LongMemEval session: {session_id}"),
        format!("Session date: {raw_date}"),
    ];
    for (index, turn) in turns.iter().enumerate() {
        let role = match turn.role.trim().to_ascii_lowercase().as_str() {
            "assistant" => "Assistant",
            _ => "User",
        };
        lines.push(format!(
            "Turn {} {}: {}",
            index + 1,
            role,
            turn.content.trim()
        ));
    }
    lines.join("\n")
}

fn case_namespace(namespace_prefix: &str, dataset_label: &str, question_id: &str) -> String {
    format!("{namespace_prefix}:{dataset_label}:{question_id}")
}

fn dataset_label(path: &Path) -> String {
    path.file_stem()
        .and_then(|value| value.to_str())
        .map(sanitize_fs_component)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "longmemeval".to_string())
}

fn append_path_suffix(path: &Path, suffix: &str) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("artifact");
    path.with_file_name(format!("{file_name}{suffix}"))
}

fn sanitize_fs_component(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    sanitized.trim_matches('_').to_string()
}

fn trim_text(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let trimmed = value
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    format!("{trimmed}…")
}

#[derive(Debug, Clone)]
struct ReplaySession {
    index: usize,
    timestamp: DateTime<Utc>,
    raw_date: String,
    session_id: String,
    content: String,
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::thread;

    use tempfile::tempdir;

    use super::*;

    fn spawn_fake_ollama(response_body: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake ollama");
        let address = listener.local_addr().expect("fake ollama address");
        let body = response_body.to_string();
        thread::spawn(move || {
            for _ in 0..1 {
                let (mut stream, _) = listener.accept().expect("accept fake ollama request");
                let mut request = vec![0_u8; 4096];
                let read = stream.read(&mut request).expect("read fake request");
                let request = String::from_utf8_lossy(&request[..read]);
                assert!(request.contains("POST /api/generate"));
                let payload = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream
                    .write_all(payload.as_bytes())
                    .expect("write fake ollama response");
            }
        });
        format!("http://{}", address)
    }

    fn spawn_fake_openai_compatible(response_body: &str) -> String {
        let listener =
            TcpListener::bind("127.0.0.1:0").expect("bind fake openai-compatible server");
        let address = listener
            .local_addr()
            .expect("fake openai-compatible address");
        let body = response_body.to_string();
        thread::spawn(move || {
            for _ in 0..1 {
                let (mut stream, _) = listener
                    .accept()
                    .expect("accept fake openai-compatible request");
                let mut request = vec![0_u8; 8192];
                let read = stream
                    .read(&mut request)
                    .expect("read fake openai-compatible request");
                let request = String::from_utf8_lossy(&request[..read]);
                assert!(request.contains("POST /v1/chat/completions"));
                let payload = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream
                    .write_all(payload.as_bytes())
                    .expect("write fake openai-compatible response");
            }
        });
        format!("http://{}/v1", address)
    }

    #[test]
    fn parse_longmemeval_datetime_supports_benchmark_format() {
        let parsed =
            parse_longmemeval_datetime("2023/04/10 (Mon) 23:07").expect("parse benchmark date");
        assert_eq!(parsed.to_rfc3339(), "2023-04-10T23:07:00+00:00");
    }

    #[test]
    fn extract_longmemeval_transcript_strips_storage_metadata() {
        let body = "kind=document\nsource=longmemeval\ncontent=LongMemEval session: sess-1\nSession date: 2023/05/23 (Tue) 18:02\nTurn 1 User: I graduated with a physics degree.\nTurn 2 Assistant: Nice.\n| entities: path:2023/05/23";
        let transcript = extract_longmemeval_transcript(body);
        assert!(transcript.starts_with("LongMemEval session: sess-1"));
        assert!(transcript.contains("I graduated with a physics degree."));
        assert!(!transcript.contains("kind=document"));
        assert!(!transcript.contains("| entities:"));
    }

    #[test]
    fn focus_transcript_excerpt_prefers_lines_matching_question_keywords() {
        let transcript = "LongMemEval session: sess-1\n\
Session date: 2023/05/23 (Tue) 18:02\n\
Turn 1 User: I enjoy audiobooks during my commute.\n\
Turn 2 Assistant: That's a nice habit.\n\
Turn 3 User: My daily commute to work takes about 45 minutes each way.\n\
Turn 4 Assistant: That's enough time for a few chapters.\n\
Turn 5 User: I usually listen to fiction on the train.";
        let excerpt =
            focus_transcript_excerpt(transcript, "How long is my daily commute to work?", 500);
        assert!(excerpt.contains("45 minutes each way"));
        assert!(excerpt.contains("Session date: 2023/05/23 (Tue) 18:02"));
    }

    #[test]
    fn head_tail_excerpt_keeps_tail_when_no_question_match_exists() {
        let lines = [
            "LongMemEval session: sess-1",
            "Session date: 2023/05/23 (Tue) 18:02",
            "Turn 1 User: Intro line.",
            "Turn 2 Assistant: More setup.",
            "Turn 3 User: Still setup.",
            "Turn 4 Assistant: More setup.",
            "Turn 5 User: Useful tail fact.",
            "Turn 6 Assistant: Tail follow-up.",
            "Turn 7 User: Final tail fact.",
        ];
        let excerpt = head_tail_excerpt(&lines, 500);
        assert!(excerpt.contains("LongMemEval session: sess-1"));
        assert!(excerpt.contains("Useful tail fact."));
        assert!(excerpt.contains("Final tail fact."));
        assert!(excerpt.contains("..."));
    }

    #[test]
    fn run_longmemeval_writes_predictions_and_debug_artifacts() -> Result<()> {
        let dir = tempdir()?;
        let dataset_path = dir.path().join("oracle.json");
        fs::write(
            &dataset_path,
            serde_json::to_vec_pretty(&json!([
                {
                    "question_id": "sample_1",
                    "question_type": "single-session-user",
                    "question": "What color is the bike?",
                    "answer": "blue",
                    "question_date": "2023/04/10 (Mon) 23:07",
                    "haystack_dates": ["2023/04/01 (Sat) 09:00"],
                    "haystack_session_ids": ["sess-1"],
                    "haystack_sessions": [[
                        {"role": "user", "content": "My new bike is blue.", "has_answer": true},
                        {"role": "assistant", "content": "Nice bike!", "has_answer": false}
                    ]],
                    "answer_session_ids": ["sess-1"]
                }
            ]))?,
        )?;
        let output_path = dir.path().join("predictions.jsonl");
        let fake_endpoint = spawn_fake_ollama(r#"{"response":"blue"}"#);

        let report = run_longmemeval(LongMemEvalRunConfig {
            dataset_path: dataset_path.clone(),
            output_path: output_path.clone(),
            work_dir: dir.path().join("work"),
            namespace_prefix: "longmemeval".to_string(),
            reader_provider: LongMemEvalReaderProvider::Ollama,
            reader_endpoint: fake_endpoint,
            reader_model: "fake-reader".to_string(),
            reader_api_key_env: None,
            reader_timeout_secs: 30,
            reader_num_predict: 32,
            reader_max_retries: 2,
            reader_retry_backoff_secs: 1,
            budget_tokens: 256,
            candidate_limit: 12,
            offset: 0,
            max_cases: None,
            question_ids: Vec::new(),
            question_types: Vec::new(),
        })?;

        let predictions = fs::read_to_string(&output_path)?;
        assert!(predictions.contains("\"question_id\":\"sample_1\""));
        assert!(predictions.contains("\"hypothesis\":\"blue\""));
        assert_eq!(report.executed_cases, 1);
        assert!(
            Path::new(&report.case_reports[0].prompt_path).exists(),
            "expected prompt artifact to exist"
        );
        assert!(
            Path::new(&report.case_reports[0].response_path).exists(),
            "expected response artifact to exist"
        );
        Ok(())
    }

    #[test]
    fn run_longmemeval_supports_openai_compatible_reader() -> Result<()> {
        let dir = tempdir()?;
        let dataset_path = dir.path().join("oracle.json");
        fs::write(
            &dataset_path,
            serde_json::to_vec_pretty(&json!([
                {
                    "question_id": "sample_2",
                    "question_type": "single-session-user",
                    "question": "What snack did I mention?",
                    "answer": "almonds",
                    "question_date": "2023/04/11 (Tue) 09:15",
                    "haystack_dates": ["2023/04/10 (Mon) 18:30"],
                    "haystack_session_ids": ["sess-2"],
                    "haystack_sessions": [[
                        {"role": "user", "content": "I packed almonds for the train ride.", "has_answer": true},
                        {"role": "assistant", "content": "That is a solid snack.", "has_answer": false}
                    ]],
                    "answer_session_ids": ["sess-2"]
                }
            ]))?,
        )?;
        let output_path = dir.path().join("predictions.jsonl");
        let fake_endpoint =
            spawn_fake_openai_compatible(r#"{"choices":[{"message":{"content":"almonds"}}]}"#);

        let report = run_longmemeval(LongMemEvalRunConfig {
            dataset_path,
            output_path: output_path.clone(),
            work_dir: dir.path().join("work"),
            namespace_prefix: "longmemeval".to_string(),
            reader_provider: LongMemEvalReaderProvider::OpenAiCompatible,
            reader_endpoint: fake_endpoint,
            reader_model: "fake-chat".to_string(),
            reader_api_key_env: None,
            reader_timeout_secs: 30,
            reader_num_predict: 32,
            reader_max_retries: 2,
            reader_retry_backoff_secs: 1,
            budget_tokens: 256,
            candidate_limit: 12,
            offset: 0,
            max_cases: None,
            question_ids: Vec::new(),
            question_types: Vec::new(),
        })?;

        let predictions = fs::read_to_string(&output_path)?;
        assert!(predictions.contains("\"question_id\":\"sample_2\""));
        assert!(predictions.contains("\"hypothesis\":\"almonds\""));
        assert_eq!(report.reader_provider, "openai-compatible");
        Ok(())
    }

    #[test]
    fn run_longmemeval_retries_transient_openai_compatible_failures() -> Result<()> {
        let dir = tempdir()?;
        let dataset_path = dir.path().join("oracle.json");
        fs::write(
            &dataset_path,
            serde_json::to_vec_pretty(&json!([
                {
                    "question_id": "sample_retry",
                    "question_type": "single-session-user",
                    "question": "What fruit did I buy?",
                    "answer": "pear",
                    "question_date": "2023/04/12 (Wed) 10:00",
                    "haystack_dates": ["2023/04/11 (Tue) 08:30"],
                    "haystack_session_ids": ["sess-retry"],
                    "haystack_sessions": [[
                        {"role": "user", "content": "I bought a pear at the market.", "has_answer": true}
                    ]],
                    "answer_session_ids": ["sess-retry"]
                }
            ]))?,
        )?;
        let attempts = Arc::new(AtomicUsize::new(0));
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let counter = attempts.clone();
        thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().expect("accept retry test request");
                let mut request = vec![0_u8; 8192];
                let read = stream.read(&mut request).expect("read retry test request");
                let request = String::from_utf8_lossy(&request[..read]);
                assert!(request.contains("POST /v1/chat/completions"));
                let attempt = counter.fetch_add(1, Ordering::SeqCst);
                let payload = if attempt == 0 {
                    "HTTP/1.1 429 Too Many Requests\r\ncontent-type: application/json\r\ncontent-length: 17\r\n\r\n{\"error\":\"slow\"}".to_string()
                } else {
                    let body = r#"{"choices":[{"message":{"content":"pear"}}]}"#;
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                        body.len(),
                        body
                    )
                };
                stream
                    .write_all(payload.as_bytes())
                    .expect("write retry test response");
            }
        });

        let report = run_longmemeval(LongMemEvalRunConfig {
            dataset_path,
            output_path: dir.path().join("predictions.jsonl"),
            work_dir: dir.path().join("work"),
            namespace_prefix: "longmemeval".to_string(),
            reader_provider: LongMemEvalReaderProvider::OpenAiCompatible,
            reader_endpoint: format!("http://{address}/v1"),
            reader_model: "fake-chat".to_string(),
            reader_api_key_env: None,
            reader_timeout_secs: 30,
            reader_num_predict: 32,
            reader_max_retries: 2,
            reader_retry_backoff_secs: 1,
            budget_tokens: 256,
            candidate_limit: 12,
            offset: 0,
            max_cases: None,
            question_ids: Vec::new(),
            question_types: Vec::new(),
        })?;

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(report.case_reports[0].hypothesis, "pear");
        Ok(())
    }

    #[test]
    fn evaluate_longmemeval_runs_official_scripts() -> Result<()> {
        let dir = tempdir()?;
        let repo_path = dir.path().join("LongMemEval");
        let eval_dir = repo_path.join("src").join("evaluation");
        fs::create_dir_all(&eval_dir)?;
        let predictions_path = dir.path().join("predictions.jsonl");
        let dataset_path = dir.path().join("oracle.json");
        fs::write(
            &predictions_path,
            "{\"question_id\":\"q1\",\"hypothesis\":\"blue\"}\n",
        )?;
        fs::write(&dataset_path, "[]")?;
        fs::write(
            eval_dir.join("evaluate_qa.py"),
            r#"import sys
open(sys.argv[2] + '.eval-results-' + sys.argv[1], 'w').write('{"question_id":"q1","autoeval_label":{"model":"gpt-4o-2024-08-06","label": true}}\n')
print("eval ok")
"#,
        )?;
        fs::write(
            eval_dir.join("print_qa_metrics.py"),
            r#"print("metrics ok")"#,
        )?;

        let report = evaluate_longmemeval(LongMemEvalEvaluateConfig {
            repo_path: repo_path.clone(),
            predictions_path: predictions_path.clone(),
            dataset_path: dataset_path.clone(),
            python_bin: "python3".to_string(),
            judge_model: "gpt-4o".to_string(),
        })?;

        assert!(report.evaluate_stdout.contains("eval ok"));
        assert_eq!(report.metrics_stdout.as_deref(), Some("metrics ok\n"));
        assert!(Path::new(&report.result_path).exists());
        Ok(())
    }
}
