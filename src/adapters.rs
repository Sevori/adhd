use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAdapterConfig {
    pub agent_id: String,
    pub agent_type: String,
    pub model: String,
    pub endpoint: String,
    pub namespace: String,
    pub role: String,
    pub timeout_secs: u64,
    pub num_predict: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelCallMetrics {
    pub total_ms: u128,
    pub load_ms: u128,
    pub eval_count: u64,
    pub prompt_eval_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(from = "LooseEvidenceNote")]
pub struct EvidenceNote {
    pub text: String,
    #[serde(default)]
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(from = "LooseDecisionNote")]
pub struct DecisionNote {
    pub text: String,
    pub rationale: String,
    #[serde(default)]
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(from = "LooseActionNote")]
pub struct ActionNote {
    pub text: String,
    #[serde(default)]
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentContinuationOutput {
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub critical_facts: Vec<EvidenceNote>,
    #[serde(default)]
    pub constraints: Vec<EvidenceNote>,
    #[serde(default)]
    pub decisions: Vec<DecisionNote>,
    #[serde(default)]
    pub open_hypotheses: Vec<EvidenceNote>,
    #[serde(default)]
    pub operational_scars: Vec<EvidenceNote>,
    #[serde(default)]
    pub avoid_repeating: Vec<EvidenceNote>,
    #[serde(default)]
    pub next_step: ActionNote,
}

/// A survival hypothesis extracted from Phase 2 meta-analysis, ready for
/// injection into the extraction prompt (Phase 3 closed-loop validation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalHypothesis {
    pub feature_name: String,
    pub category: String,
    pub direction: String,
    pub hint: String,
}

pub trait AgentAdapter {
    fn config(&self) -> &AgentAdapterConfig;
    fn analyze(
        &self,
        objective: &str,
        context_text: &str,
    ) -> Result<(AgentContinuationOutput, ModelCallMetrics)>;
    /// Analyse with optional survival hypothesis injection (Phase 3).
    /// Default ignores hypotheses; concrete adapters override.
    fn analyze_with_hypotheses(
        &self,
        objective: &str,
        context_text: &str,
        _hypotheses: &[SurvivalHypothesis],
    ) -> Result<(AgentContinuationOutput, ModelCallMetrics)> {
        self.analyze(objective, context_text)
    }
}

#[derive(Debug, Clone)]
pub struct OllamaAdapter {
    config: AgentAdapterConfig,
    client: Client,
}

impl OllamaAdapter {
    pub fn new(config: AgentAdapterConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs.max(30)))
            .build()?;
        Ok(Self { config, client })
    }

    fn prompt(&self, objective: &str, context_text: &str) -> String {
        render_structured_resume_prompt(self.config.role.as_str(), objective, context_text)
    }

    fn prompt_with_hypotheses(
        &self,
        objective: &str,
        context_text: &str,
        hypotheses: &[SurvivalHypothesis],
    ) -> String {
        render_structured_resume_prompt_with_hypotheses(
            self.config.role.as_str(),
            objective,
            context_text,
            hypotheses,
        )
    }

    fn ensure_success(&self) -> Result<()> {
        let response = self
            .client
            .get(format!("{}/api/tags", self.config.endpoint))
            .send()
            .with_context(|| format!("connecting to Ollama at {}", self.config.endpoint))?;
        if !response.status().is_success() {
            return Err(anyhow!(
                "ollama tags request failed with {}",
                response.status()
            ));
        }
        Ok(())
    }

    fn generate_with_prompt(
        &self,
        prompt: String,
    ) -> Result<(AgentContinuationOutput, ModelCallMetrics)> {
        self.ensure_success()?;
        let request = OllamaGenerateRequest {
            model: self.config.model.clone(),
            prompt,
            format: structured_output_schema(),
            stream: false,
            options: OllamaOptions {
                temperature: 0.0,
                num_predict: self.config.num_predict,
            },
        };
        let response = self
            .client
            .post(format!("{}/api/generate", self.config.endpoint))
            .json(&request)
            .send()
            .with_context(|| format!("requesting generation from {}", self.config.model))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .unwrap_or_else(|error| format!("unable to read ollama error body: {error}"));
            return Err(anyhow!(
                "ollama generation failed for {} with {}: {}",
                self.config.model,
                status,
                body
            ));
        }
        let payload: OllamaGenerateResponse = response.json()?;
        let parsed = parse_structured_output(&payload.response)?;
        Ok((
            parsed,
            ModelCallMetrics {
                total_ms: nanos_to_ms(payload.total_duration),
                load_ms: nanos_to_ms(payload.load_duration),
                eval_count: payload.eval_count.unwrap_or_default(),
                prompt_eval_count: payload.prompt_eval_count.unwrap_or_default(),
            },
        ))
    }
}

impl AgentAdapter for OllamaAdapter {
    fn config(&self) -> &AgentAdapterConfig {
        &self.config
    }

    fn analyze(
        &self,
        objective: &str,
        context_text: &str,
    ) -> Result<(AgentContinuationOutput, ModelCallMetrics)> {
        self.generate_with_prompt(self.prompt(objective, context_text))
    }

    fn analyze_with_hypotheses(
        &self,
        objective: &str,
        context_text: &str,
        hypotheses: &[SurvivalHypothesis],
    ) -> Result<(AgentContinuationOutput, ModelCallMetrics)> {
        self.generate_with_prompt(self.prompt_with_hypotheses(objective, context_text, hypotheses))
    }
}

#[derive(Debug, Clone, Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    format: Value,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Debug, Clone, Serialize)]
struct OllamaOptions {
    temperature: f64,
    num_predict: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct OllamaGenerateResponse {
    response: String,
    total_duration: Option<u128>,
    load_duration: Option<u128>,
    eval_count: Option<u64>,
    prompt_eval_count: Option<u64>,
}

pub fn parse_structured_output(response: &str) -> Result<AgentContinuationOutput> {
    let trimmed = response.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("model returned empty response"));
    }
    let candidate = if trimmed.starts_with('{') && trimmed.ends_with('}') {
        trimmed.to_string()
    } else {
        let start = trimmed
            .find('{')
            .ok_or_else(|| anyhow!("model response did not contain json object"))?;
        let end = trimmed
            .rfind('}')
            .ok_or_else(|| anyhow!("model response did not contain json terminator"))?;
        trimmed[start..=end].to_string()
    };
    let value: Value = serde_json::from_str(&candidate)
        .with_context(|| format!("parsing model json response: {candidate}"))?;
    normalize_structured_output(value)
}

fn nanos_to_ms(value: Option<u128>) -> u128 {
    value.unwrap_or_default() / 1_000_000
}

pub fn structured_output_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "maxLength": 160
            },
            "critical_facts": string_array_schema(128),
            "constraints": string_array_schema(128),
            "decisions": string_array_schema(160),
            "open_hypotheses": string_array_schema(128),
            "operational_scars": string_array_schema(128),
            "avoid_repeating": string_array_schema(128),
            "next_step": {
                "type": "string",
                "maxLength": 96
            }
        },
        "required": [
            "summary",
            "critical_facts",
            "constraints",
            "decisions",
            "open_hypotheses",
            "operational_scars",
            "avoid_repeating",
            "next_step"
        ],
        "additionalProperties": false
    })
}

pub fn render_structured_resume_prompt(role: &str, objective: &str, context_text: &str) -> String {
    render_structured_resume_prompt_with_hypotheses(role, objective, context_text, &[])
}

pub fn render_structured_resume_prompt_with_hypotheses(
    role: &str,
    objective: &str,
    context_text: &str,
    hypotheses: &[SurvivalHypothesis],
) -> String {
    let hints = render_hypothesis_hints(hypotheses);
    format!(
        "You are a {role} agent resuming work inside a Shared Continuity Kernel.\n\
Return JSON only with keys summary, critical_facts, constraints, decisions, open_hypotheses, operational_scars, avoid_repeating, next_step.\n\
Use natural language phrases copied from the context. Do not output snake_case identifiers unless the context literally uses them.\n\
Do not answer with placeholder references like `decision/d1`, `incidents/i1`, `rationale/k1`, `p1`, or `r1`; copy the natural-language phrase after the label instead.\n\
Never copy the literal scaffolding words `decision text`, `rationale text`, or `evidence ids` into the JSON output.\n\
Classify carefully:\n\
- critical_facts are concrete incidents, files, selectors, or state facts\n\
- constraints are rules or must/avoid requirements\n\
- decisions are chosen approaches plus why they were chosen\n\
- operational_scars are prior failures or warnings that should change behavior\n\
When `resumption_core`, incidents, or answer hints are present, prefer those for `critical_facts` before hypotheses or scars.\n\
For decisions, copy the human decision title and the human rationale phrase after `::`, not the label or the instruction text.\n\
For note arrays, each item must be a single string in the form `text || e1,e2` when you have evidence, or just `text` when you do not.\n\
For decisions, each item must be a single string in the form `decision text || rationale text || e1,e2`.\n\
Evidence ids must be labels from the context such as a1, e1, d1, f1, k1, h1, i1, s1, t1, p1, or r1.\n\
When a matching label exists in the context, include at least one evidence id. Prefer the most specific high-signal label such as f1, a1, r1, d1, k1, i1, s1, or p1.\n\
Examples:\n\
- `Primary context is bench / task-strong-to-small for this resume. || f1`\n\
- `selector_missing in src/query.rs || i1`\n\
- `selector_missing in src/query.rs || a1,r1`\n\
- `Preserve provenance || k1`\n\
- `Use the unified continuity interface || agent swaps share one context namespace || d1`\n\
- `Avoid naive probes || s1`\n\
Do not invent unsupported facts. Omit weak claims instead of guessing.\n\
If a list has no good item, return an empty array []. If next_step is unknown, return an empty string.\n\
Keep the output compact: summary under 30 words, next_step under 20 words, each list with at most 2 items, and each item text under 18 words.\n\
{hints}\
Objective: {objective}\n\
\n\
Context:\n{context_text}\n",
        role = role,
        hints = hints,
        objective = objective,
        context_text = context_text,
    )
}

/// Render survival hypothesis hints as prompt directives.
///
/// When hypotheses are present, they become `[SURVIVAL HINTS]` directives
/// instructing the model to emphasise features correlated with survival.
pub fn render_hypothesis_hints(hypotheses: &[SurvivalHypothesis]) -> String {
    if hypotheses.is_empty() {
        return String::new();
    }
    let mut lines = vec!["[SURVIVAL HINTS] The following patterns are statistically correlated with better information survival. Apply them when extracting items:".to_string()];
    for (i, h) in hypotheses.iter().enumerate() {
        lines.push(format!("{}. [{}] {}", i + 1, h.category, h.hint));
    }
    lines.push(String::new());
    lines.join("\n")
}

/// Extract eligible survival hypotheses from Phase 2 MetaLessons.
/// Only hypotheses with `sparse_cells: false` and `adjusted_p_value < 0.05` qualify.
pub fn hypotheses_from_meta_lessons(
    lessons: &[crate::benchmark::MetaLesson],
) -> Vec<SurvivalHypothesis> {
    lessons
        .iter()
        .filter(|l| !l.evidence.sparse_cells && l.evidence.adjusted_p_value < 0.05)
        .map(|l| SurvivalHypothesis {
            feature_name: l.feature_name.clone(),
            category: format!("{:?}", l.category),
            direction: format!("{:?}", l.direction),
            hint: l.pattern.clone(),
        })
        .collect()
}

/// Extract survival hypotheses from kernel Hypothesis items written by Phase 2.
///
/// Filters for items that have `requires_validation: true` in extra and
/// are not flagged with `sparse_cells: true` in their evidence.
pub fn hypotheses_from_kernel_items(
    items: &[crate::continuity::ContinuityItemRecord],
) -> Vec<SurvivalHypothesis> {
    items
        .iter()
        .filter(|item| {
            item.kind == crate::continuity::ContinuityKind::Hypothesis
                && item.status.is_open()
                && item
                    .extra
                    .get("requires_validation")
                    .and_then(|v| v.as_bool())
                    == Some(true)
                && item
                    .extra
                    .get("evidence")
                    .and_then(|e| e.get("sparse_cells"))
                    .and_then(|v| v.as_bool())
                    != Some(true)
        })
        .map(|item| {
            let feature_name = item
                .extra
                .get("feature_name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let category = item
                .extra
                .get("category")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let direction = item
                .extra
                .get("direction")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            SurvivalHypothesis {
                feature_name,
                category,
                direction,
                hint: item.title.clone(),
            }
        })
        .collect()
}

fn string_array_schema(max_length: usize) -> Value {
    serde_json::json!({
        "type": "array",
        "items": {
            "type": "string",
            "maxLength": max_length
        },
        "maxItems": 2
    })
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum LooseEvidenceNote {
    Text(String),
    Full {
        text: String,
        #[serde(default)]
        evidence: Vec<String>,
    },
}

impl From<LooseEvidenceNote> for EvidenceNote {
    fn from(value: LooseEvidenceNote) -> Self {
        match value {
            LooseEvidenceNote::Text(text) => parse_evidence_text(&text),
            LooseEvidenceNote::Full { text, evidence } => Self { text, evidence },
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum LooseDecisionNote {
    Text(String),
    Full {
        text: String,
        #[serde(default)]
        rationale: String,
        #[serde(default)]
        evidence: Vec<String>,
    },
}

impl From<LooseDecisionNote> for DecisionNote {
    fn from(value: LooseDecisionNote) -> Self {
        match value {
            LooseDecisionNote::Text(text) => parse_decision_text(&text),
            LooseDecisionNote::Full {
                text,
                rationale,
                evidence,
            } => Self {
                text,
                rationale,
                evidence,
            },
        }
    }
}

fn parse_evidence_text(text: &str) -> EvidenceNote {
    let parts = split_fields(text);
    if parts.len() >= 2 {
        let evidence = parse_evidence_ids(parts.last().unwrap_or(&String::new()));
        if !evidence.is_empty() {
            return EvidenceNote {
                text: parts[..parts.len() - 1].join(" || "),
                evidence,
            };
        }
    }
    EvidenceNote {
        text: text.trim().to_string(),
        evidence: Vec::new(),
    }
}

fn parse_decision_text(text: &str) -> DecisionNote {
    let parts = split_fields(text);
    if parts.len() >= 3 {
        let evidence = parse_evidence_ids(parts.last().unwrap_or(&String::new()));
        if !evidence.is_empty() {
            return DecisionNote {
                text: parts[0].clone(),
                rationale: parts[1..parts.len() - 1].join(" || "),
                evidence,
            };
        }
    }
    if parts.len() >= 2 {
        return DecisionNote {
            text: parts[0].clone(),
            rationale: parts[1..].join(" || "),
            evidence: Vec::new(),
        };
    }
    DecisionNote {
        text: text.trim().to_string(),
        rationale: String::new(),
        evidence: Vec::new(),
    }
}

fn split_fields(text: &str) -> Vec<String> {
    text.split("||")
        .map(|part| part.trim().to_string())
        .filter(|part| !part.is_empty())
        .collect()
}

fn parse_evidence_ids(text: &str) -> Vec<String> {
    text.split(',')
        .map(|part| part.trim())
        .filter(|part| !part.is_empty())
        .filter(|part| {
            let mut chars = part.chars();
            matches!(chars.next(), Some(prefix) if prefix.is_ascii_lowercase())
                && chars.all(|ch| ch.is_ascii_digit())
        })
        .map(|part| part.to_string())
        .collect()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum LooseActionNote {
    Text(String),
    Full {
        text: String,
        #[serde(default)]
        evidence: Vec<String>,
    },
}

impl From<LooseActionNote> for ActionNote {
    fn from(value: LooseActionNote) -> Self {
        match value {
            LooseActionNote::Text(text) => Self {
                text,
                evidence: Vec::new(),
            },
            LooseActionNote::Full { text, evidence } => Self { text, evidence },
        }
    }
}

fn normalize_structured_output(value: Value) -> Result<AgentContinuationOutput> {
    let Value::Object(map) = value else {
        return Err(anyhow!("model json response was not an object"));
    };
    Ok(AgentContinuationOutput {
        summary: string_field(map.get("summary")),
        critical_facts: parse_evidence_list(map.get("critical_facts"))?,
        constraints: parse_evidence_list(map.get("constraints"))?,
        decisions: parse_decision_list(map.get("decisions"))?,
        open_hypotheses: parse_evidence_list(map.get("open_hypotheses"))?,
        operational_scars: parse_evidence_list(map.get("operational_scars"))?,
        avoid_repeating: parse_evidence_list(map.get("avoid_repeating"))?,
        next_step: parse_action(map.get("next_step"))?,
    })
}

fn parse_evidence_list(value: Option<&Value>) -> Result<Vec<EvidenceNote>> {
    parse_note_list(value, parse_evidence_note)
}

fn parse_decision_list(value: Option<&Value>) -> Result<Vec<DecisionNote>> {
    parse_note_list(value, parse_decision_note)
}

fn parse_note_list<T, F>(value: Option<&Value>, mut parse_one: F) -> Result<Vec<T>>
where
    F: FnMut(&Value) -> Result<Option<T>>,
{
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    match value {
        Value::Null => Ok(Vec::new()),
        Value::Bool(false) => Ok(Vec::new()),
        Value::Array(items) => {
            let mut parsed = Vec::new();
            for item in items {
                if let Some(note) = parse_one(item)? {
                    parsed.push(note);
                }
            }
            Ok(parsed)
        }
        other => Ok(parse_one(other)?.into_iter().collect()),
    }
}

fn parse_evidence_note(value: &Value) -> Result<Option<EvidenceNote>> {
    if is_empty_placeholder(value) {
        return Ok(None);
    }
    serde_json::from_value::<EvidenceNote>(value.clone())
        .map(Some)
        .with_context(|| format!("parsing evidence note: {value}"))
}

fn parse_decision_note(value: &Value) -> Result<Option<DecisionNote>> {
    if is_empty_placeholder(value) {
        return Ok(None);
    }
    serde_json::from_value::<DecisionNote>(value.clone())
        .map(Some)
        .with_context(|| format!("parsing decision note: {value}"))
}

fn parse_action(value: Option<&Value>) -> Result<ActionNote> {
    let Some(value) = value else {
        return Ok(ActionNote::default());
    };
    if is_empty_placeholder(value) {
        return Ok(ActionNote::default());
    }
    serde_json::from_value::<ActionNote>(value.clone())
        .with_context(|| format!("parsing next step: {value}"))
}

fn string_field(value: Option<&Value>) -> String {
    value.and_then(value_to_text).unwrap_or_default()
}

fn value_to_text(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(text) => Some(text.trim().to_string()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(flag) => Some(flag.to_string()),
        Value::Object(object) => object
            .get("text")
            .and_then(value_to_text)
            .or_else(|| object.get("summary").and_then(value_to_text)),
        _ => None,
    }
    .filter(|text| !text.is_empty())
}

fn is_empty_placeholder(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(false) => true,
        Value::String(text) => {
            let normalized = text.trim().to_ascii_lowercase();
            normalized.is_empty()
                || matches!(normalized.as_str(), "none" | "n/a" | "null" | "false")
        }
        Value::Array(items) => items.is_empty(),
        Value::Object(object) => object.is_empty(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        SurvivalHypothesis, parse_structured_output, render_hypothesis_hints,
        render_structured_resume_prompt_with_hypotheses,
    };

    #[test]
    fn render_hypothesis_hints_empty_produces_empty_string() {
        assert_eq!(render_hypothesis_hints(&[]), "");
    }

    #[test]
    fn render_hypothesis_hints_formats_directives() {
        let hypotheses = vec![
            SurvivalHypothesis {
                feature_name: "file_path".into(),
                category: "CriticalFact".into(),
                direction: "survived_more".into(),
                hint: "Include file paths in critical facts — items with file paths survive 90% vs 10% without".into(),
            },
            SurvivalHypothesis {
                feature_name: "prohibition_framing".into(),
                category: "Constraint".into(),
                direction: "survived_more".into(),
                hint: "Use prohibition framing (never/avoid/must not) for constraints".into(),
            },
        ];
        let hints = render_hypothesis_hints(&hypotheses);
        assert!(hints.contains("[SURVIVAL HINTS]"));
        assert!(hints.contains("1. [CriticalFact]"));
        assert!(hints.contains("2. [Constraint]"));
        assert!(hints.contains("file paths survive 90%"));
        assert!(hints.contains("prohibition framing"));
    }

    #[test]
    fn prompt_with_hypotheses_includes_hints_section() {
        let hypotheses = vec![SurvivalHypothesis {
            feature_name: "file_path".into(),
            category: "CriticalFact".into(),
            direction: "survived_more".into(),
            hint: "Include file paths".into(),
        }];
        let prompt = render_structured_resume_prompt_with_hypotheses(
            "tester",
            "test objective",
            "some context",
            &hypotheses,
        );
        assert!(prompt.contains("[SURVIVAL HINTS]"));
        assert!(prompt.contains("Include file paths"));
        assert!(prompt.contains("test objective"));
        assert!(prompt.contains("some context"));
    }

    #[test]
    fn prompt_without_hypotheses_matches_original() {
        let original = super::render_structured_resume_prompt("tester", "obj", "ctx");
        let with_empty =
            render_structured_resume_prompt_with_hypotheses("tester", "obj", "ctx", &[]);
        assert_eq!(original, with_empty);
    }

    #[test]
    fn parser_accepts_scalar_and_boolean_fallbacks() {
        let output = parse_structured_output(
            r#"{
                "summary":"resume benchmark",
                "critical_facts":"selector_missing in src/query.rs",
                "constraints":false,
                "decisions":{"text":"Use unified continuity interface","rationale":"agent swap continuity","evidence":["e1"]},
                "open_hypotheses":"adapter timeout remains possible",
                "operational_scars":"naive probe hung ollama",
                "avoid_repeating":false,
                "next_step":"run continuity benchmark"
            }"#,
        )
        .expect("parser should normalize scalar and boolean fields");

        assert_eq!(output.summary, "resume benchmark");
        assert_eq!(output.critical_facts.len(), 1);
        assert_eq!(output.constraints.len(), 0);
        assert_eq!(output.decisions.len(), 1);
        assert_eq!(output.open_hypotheses.len(), 1);
        assert_eq!(output.operational_scars.len(), 1);
        assert_eq!(output.avoid_repeating.len(), 0);
        assert_eq!(output.next_step.text, "run continuity benchmark");
    }

    #[test]
    fn parser_treats_empty_placeholders_as_missing() {
        let output = parse_structured_output(
            r#"{
                "summary":{"summary":"keep going"},
                "critical_facts":["selector_missing in src/query.rs"],
                "constraints":"none",
                "decisions":[],
                "open_hypotheses":null,
                "operational_scars":"",
                "avoid_repeating":"false",
                "next_step":{"text":"benchmark adapters","evidence":["e2"]}
            }"#,
        )
        .expect("parser should drop empty placeholders");

        assert_eq!(output.summary, "keep going");
        assert_eq!(output.critical_facts.len(), 1);
        assert!(output.constraints.is_empty());
        assert!(output.open_hypotheses.is_empty());
        assert!(output.operational_scars.is_empty());
        assert!(output.avoid_repeating.is_empty());
        assert_eq!(output.next_step.evidence, vec!["e2"]);
    }

    #[test]
    fn parser_extracts_rationale_and_evidence_from_flat_strings() {
        let output = parse_structured_output(
            r#"{
                "summary":"resume",
                "critical_facts":["selector_missing in src/query.rs || e1,e2"],
                "constraints":["preserve provenance || k1"],
                "decisions":["Use unified continuity interface || preserve swap continuity || d1,e3"],
                "open_hypotheses":["adapter timeout || h1"],
                "operational_scars":["naive probe hung ollama || s1"],
                "avoid_repeating":["manual probe || s1"],
                "next_step":"run shared continuity benchmark"
            }"#,
        )
        .expect("parser should decode flat-string schema");

        assert_eq!(output.critical_facts[0].evidence, vec!["e1", "e2"]);
        assert_eq!(output.constraints[0].evidence, vec!["k1"]);
        assert_eq!(output.decisions[0].text, "Use unified continuity interface");
        assert_eq!(output.decisions[0].rationale, "preserve swap continuity");
        assert_eq!(output.decisions[0].evidence, vec!["d1", "e3"]);
        assert_eq!(output.operational_scars[0].evidence, vec!["s1"]);
        assert_eq!(output.next_step.text, "run shared continuity benchmark");
    }
}
