#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ICE_BIN="${ICE_BIN:-cargo run --release --}"

ARTIFACTS_ROOT="${LONGMEMEVAL_ARTIFACTS_ROOT:-$ROOT_DIR/.artifacts/longmemeval}"
DATA_DIR="${LONGMEMEVAL_DATA_DIR:-$ARTIFACTS_ROOT/data}"
WORK_DIR="${LONGMEMEVAL_WORK_DIR:-$ARTIFACTS_ROOT/work}"
REPO_DIR="${LONGMEMEVAL_REPO_DIR:-$ARTIFACTS_ROOT/LongMemEval}"
VENV_DIR="${LONGMEMEVAL_VENV_DIR:-$ARTIFACTS_ROOT/.venv312}"
SPLIT="${LONGMEMEVAL_SPLIT:-oracle}"
READER_PROVIDER="${LONGMEMEVAL_READER_PROVIDER:-openai-compatible}"
READER_ENDPOINT="${LONGMEMEVAL_READER_ENDPOINT:-https://api.openai.com/v1}"
READER_MODEL="${LONGMEMEVAL_READER_MODEL:-gpt-4.1-mini}"
JUDGE_MODEL="${LONGMEMEVAL_JUDGE_MODEL:-gpt-4o}"
PYTHON_BIN="${LONGMEMEVAL_PYTHON_BIN:-python3.12}"
READER_NUM_PREDICT="${LONGMEMEVAL_READER_NUM_PREDICT:-96}"
READER_MAX_RETRIES="${LONGMEMEVAL_READER_MAX_RETRIES:-4}"
READER_RETRY_BACKOFF_SECS="${LONGMEMEVAL_READER_RETRY_BACKOFF_SECS:-2}"
BUDGET_TOKENS="${LONGMEMEVAL_BUDGET_TOKENS:-512}"
CANDIDATE_LIMIT="${LONGMEMEVAL_CANDIDATE_LIMIT:-24}"
MAX_CASES="${LONGMEMEVAL_MAX_CASES:-}"

mkdir -p "$DATA_DIR" "$WORK_DIR"

case "$SPLIT" in
  oracle)
    DATASET_URL="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"
    DATASET_PATH="$DATA_DIR/longmemeval_oracle.json"
    ;;
  s)
    DATASET_URL="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
    DATASET_PATH="$DATA_DIR/longmemeval_s_cleaned.json"
    ;;
  m)
    DATASET_URL="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json"
    DATASET_PATH="$DATA_DIR/longmemeval_m_cleaned.json"
    ;;
  *)
    echo "Unsupported LONGMEMEVAL_SPLIT: $SPLIT" >&2
    exit 1
    ;;
esac

PREDICTIONS_PATH="$ARTIFACTS_ROOT/${SPLIT}-predictions.jsonl"
RUN_REPORT_PATH="${PREDICTIONS_PATH}.report.json"
EVAL_REPORT_PATH="$ARTIFACTS_ROOT/${SPLIT}-evaluation-report.json"
SUMMARY_PATH="$ARTIFACTS_ROOT/${SPLIT}-summary.md"

echo "==> Downloading dataset: $DATASET_URL"
curl -L --fail --silent --show-error "$DATASET_URL" -o "$DATASET_PATH"

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "==> Updating LongMemEval repo"
  git -C "$REPO_DIR" pull --ff-only
else
  echo "==> Cloning LongMemEval repo"
  rm -rf "$REPO_DIR"
  git clone --depth 1 https://github.com/xiaowu0162/LongMemEval.git "$REPO_DIR"
fi

echo "==> Preparing evaluator venv with $PYTHON_BIN"
rm -rf "$VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install -r "$REPO_DIR/requirements-lite.txt"
"$VENV_DIR/bin/pip" install 'httpx<0.28'

RUN_CMD=(
  $ICE_BIN
  --root "$WORK_DIR/ice-root"
  longmemeval run
  --dataset "$DATASET_PATH"
  --output "$PREDICTIONS_PATH"
  --reader-provider "$READER_PROVIDER"
  --reader-endpoint "$READER_ENDPOINT"
  --reader-model "$READER_MODEL"
  --reader-num-predict "$READER_NUM_PREDICT"
  --reader-max-retries "$READER_MAX_RETRIES"
  --reader-retry-backoff-secs "$READER_RETRY_BACKOFF_SECS"
  --budget-tokens "$BUDGET_TOKENS"
  --candidate-limit "$CANDIDATE_LIMIT"
)

if [[ -n "$MAX_CASES" ]]; then
  RUN_CMD+=(--max-cases "$MAX_CASES")
fi

echo "==> Running LongMemEval generation"
"${RUN_CMD[@]}" > "$ARTIFACTS_ROOT/${SPLIT}-run.json"

echo "==> Running official evaluator"
$ICE_BIN longmemeval evaluate \
  --repo "$REPO_DIR" \
  --predictions "$PREDICTIONS_PATH" \
  --dataset "$DATASET_PATH" \
  --python-bin "$VENV_DIR/bin/python" \
  --judge-model "$JUDGE_MODEL" > "$EVAL_REPORT_PATH"

RUN_REPORT_PATH="$RUN_REPORT_PATH" \
EVAL_REPORT_PATH="$EVAL_REPORT_PATH" \
SUMMARY_PATH="$SUMMARY_PATH" \
DATASET_PATH="$DATASET_PATH" \
PREDICTIONS_PATH="$PREDICTIONS_PATH" \
python3 <<'PY'
import json
import os
import re
from pathlib import Path

run_report_path = Path(os.environ["RUN_REPORT_PATH"])
eval_report_path = Path(os.environ["EVAL_REPORT_PATH"])
summary_path = Path(os.environ["SUMMARY_PATH"])
dataset_path = Path(os.environ["DATASET_PATH"])
predictions_path = Path(os.environ["PREDICTIONS_PATH"])

run_report = json.loads(run_report_path.read_text())
eval_report = json.loads(eval_report_path.read_text())
metrics_stdout = eval_report.get("metrics_stdout") or ""
evaluate_stdout = eval_report.get("evaluate_stdout") or ""

overall_match = re.search(r"Overall Accuracy:\s*([0-9.]+)", metrics_stdout)
task_match = re.search(r"\ttemporal-reasoning:\s*([0-9.]+)\s*\((\d+)\)", metrics_stdout)
overall_accuracy = overall_match.group(1) if overall_match else "unknown"
temporal_accuracy = task_match.group(1) if task_match else "unknown"
temporal_n = task_match.group(2) if task_match else "0"

summary = "\n".join([
    "# LongMemEval Summary",
    "",
    f"- Dataset: `{dataset_path}`",
    f"- Predictions: `{predictions_path}`",
    f"- Reader provider: `{run_report['reader_provider']}`",
    f"- Reader model: `{run_report['reader_model']}`",
    f"- Judge model: `{eval_report['judge_model']}`",
    f"- Executed cases: `{run_report['executed_cases']}`",
    f"- Overall accuracy: `{overall_accuracy}`",
    f"- Temporal-reasoning accuracy: `{temporal_accuracy}` over `{temporal_n}` cases",
    "",
    "## Metrics",
    "",
    "```text",
    metrics_stdout.strip(),
    "```",
    "",
    "## Evaluation Output",
    "",
    "```text",
    evaluate_stdout.strip(),
    "```",
])

summary_path.write_text(summary + "\n")
print(summary)
PY

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  cat "$SUMMARY_PATH" >> "$GITHUB_STEP_SUMMARY"
fi
