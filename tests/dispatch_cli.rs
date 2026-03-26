use std::path::Path;
use std::process::{Child, Command, Output, Stdio};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use postgres::{Client, NoTls};
use serde_json::{Value, json};
use tempfile::tempdir;
use uuid::Uuid;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_ice")
}

fn run_output(root: &Path, args: &[String], envs: &[(&str, &str)]) -> Result<Output> {
    let output = Command::new(bin())
        .arg("--root")
        .arg(root)
        .args(args)
        .envs(envs.iter().copied())
        .output()
        .with_context(|| format!("running ice {}", args.join(" ")))?;
    if !output.status.success() {
        anyhow::bail!(
            "command failed: ice {}\nstdout:\n{}\nstderr:\n{}",
            args.join(" "),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(output)
}

fn run_json(root: &Path, args: &[String], envs: &[(&str, &str)]) -> Result<Value> {
    let output = run_output(root, args, envs)?;
    serde_json::from_slice(&output.stdout)
        .with_context(|| format!("parsing JSON output from ice {}", args.join(" ")))
}

fn run_text(root: &Path, args: &[String], envs: &[(&str, &str)]) -> Result<String> {
    let output = run_output(root, args, envs)?;
    String::from_utf8(output.stdout).context("metrics output should be valid UTF-8")
}

fn spawn_listener(root: &Path, args: &[String], envs: &[(&str, &str)]) -> Result<Child> {
    Command::new(bin())
        .arg("--root")
        .arg(root)
        .args(args)
        .envs(envs.iter().copied())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawning listener: ice {}", args.join(" ")))
}

struct TestDatabase {
    url: String,
    admin_url: String,
    dbname: String,
}

impl Drop for TestDatabase {
    fn drop(&mut self) {
        if let Ok(mut admin) = Client::connect(&self.admin_url, NoTls) {
            let _ = admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1 AND pid <> pg_backend_pid()",
                &[&self.dbname],
            );
            let _ = admin.batch_execute(&format!("DROP DATABASE IF EXISTS \"{}\"", self.dbname));
        }
    }
}

fn provision_test_database(base_url: &str) -> Result<TestDatabase> {
    let dbname = format!("ice_dispatch_{}", Uuid::now_v7().simple());
    let admin_url = database_url_with_name(base_url, "postgres")?;
    let mut admin = Client::connect(&admin_url, NoTls)?;
    admin.batch_execute(&format!("DROP DATABASE IF EXISTS \"{}\"", dbname))?;
    admin.batch_execute(&format!("CREATE DATABASE \"{}\"", dbname))?;
    let url = database_url_with_name(base_url, &dbname)?;
    Ok(TestDatabase {
        url,
        admin_url,
        dbname,
    })
}

fn database_url_with_name(base_url: &str, dbname: &str) -> Result<String> {
    let (before_query, query) = match base_url.split_once('?') {
        Some((left, right)) => (left, Some(right)),
        None => (base_url, None),
    };
    let slash = before_query
        .rfind('/')
        .context("PostgreSQL URL must include a database path")?;
    let mut url = format!("{}{dbname}", &before_query[..=slash]);
    if let Some(query) = query {
        url.push('?');
        url.push_str(query);
    }
    Ok(url)
}

#[test]
fn dispatch_cli_routes_completion_to_waiting_worker_over_postgres() -> Result<()> {
    let Some(database_url) = std::env::var_os("ICE_TEST_POSTGRES_URL") else {
        eprintln!("skipping dispatch CLI integration test: ICE_TEST_POSTGRES_URL not set");
        return Ok(());
    };
    let database_url = database_url
        .into_string()
        .map_err(|_| anyhow::anyhow!("ICE_TEST_POSTGRES_URL must be valid UTF-8"))?;
    let database = provision_test_database(&database_url)?;
    let dir = tempdir()?;
    let root = dir.path();
    let notify_channel = format!(
        "ice_dispatch_{}",
        &Uuid::now_v7().simple().to_string()[..12]
    );
    let worker_id = format!("listener-{}", &Uuid::now_v7().simple().to_string()[..8]);
    let objective = "Wake the next head from durable continuity";

    run_json(
        root,
        &[
            "dispatch".to_string(),
            "init".to_string(),
            "--database-url".to_string(),
            database.url.clone(),
            "--notify-channel".to_string(),
            notify_channel.clone(),
        ],
        &[],
    )?;

    let context = run_json(
        root,
        &[
            "uci".to_string(),
            "--json".to_string(),
            serde_json::to_string(&json!({
                "op": "open_context",
                "input": {
                    "namespace": "@machine",
                    "task_id": "machine-organism",
                    "session_id": "dispatch-proof",
                    "objective": objective,
                    "agent_id": "planner-a",
                    "attachment_id": serde_json::Value::Null,
                    "selector": serde_json::Value::Null
                }
            }))?,
        ],
        &[],
    )?;
    let context_id = context["data"]["id"]
        .as_str()
        .context("open_context response missing context id")?
        .to_string();

    for (op, title, body) in [
        (
            "mark_constraint",
            "Keep SQLite as the continuity source of truth",
            "Dispatch must wake the next head without bypassing the continuity kernel or leaking raw transcript state.",
        ),
        (
            "mark_operational_scar",
            "Nested dispatch payloads break tiny workers",
            "Small heads need a compact assignment envelope with visible constraints and scars, not a giant hidden blob.",
        ),
    ] {
        run_json(
            root,
            &[
                "uci".to_string(),
                "--json".to_string(),
                serde_json::to_string(&json!({
                    "op": op,
                    "input": {
                        "context_id": context_id,
                        "author_agent_id": "planner-a",
                        "kind": op.strip_prefix("mark_").unwrap(),
                        "title": title,
                        "body": body,
                        "scope": "shared",
                        "status": serde_json::Value::Null,
                        "importance": 0.99,
                        "confidence": 0.97,
                        "salience": serde_json::Value::Null,
                        "layer": serde_json::Value::Null,
                        "supports": [],
                        "dimensions": [],
                        "extra": {}
                    }
                }))?,
            ],
            &[],
        )?;
    }

    let listen_args = vec![
        "dispatch".to_string(),
        "listen".to_string(),
        "--worker-id".to_string(),
        worker_id.clone(),
        "--display-name".to_string(),
        "Listener Small".to_string(),
        "--role".to_string(),
        "coder".to_string(),
        "--agent-type".to_string(),
        "ollama".to_string(),
        "--tier".to_string(),
        "small".to_string(),
        "--model".to_string(),
        "qwen2.5:0.5b".to_string(),
        "--capabilities".to_string(),
        "read,write,claim".to_string(),
        "--max-parallelism".to_string(),
        "1".to_string(),
        "--namespace".to_string(),
        "@machine".to_string(),
        "--task".to_string(),
        "machine-organism".to_string(),
        "--timeout-secs".to_string(),
        "10".to_string(),
    ];
    let listener = spawn_listener(root, &listen_args, &[])?;
    thread::sleep(Duration::from_millis(750));

    let completion = run_json(
        root,
        &[
            "dispatch".to_string(),
            "complete".to_string(),
            "--context-id".to_string(),
            context_id.clone(),
            "--agent".to_string(),
            "planner-a".to_string(),
            "--title".to_string(),
            "Router completed the planning slice".to_string(),
            "--result".to_string(),
            "The next head should implement the dispatch proof from the bounded continuity pack."
                .to_string(),
            "--objective".to_string(),
            objective.to_string(),
            "--quality".to_string(),
            "0.88".to_string(),
            "--target-role".to_string(),
            "coder".to_string(),
            "--preferred-tier".to_string(),
            "small".to_string(),
            "--reason".to_string(),
            "Wake a small local coder".to_string(),
        ],
        &[],
    )?;
    let signal_id = completion["signal"]["id"]
        .as_str()
        .context("dispatch complete did not return a signal id")?
        .to_string();

    let listen_output = listener
        .wait_with_output()
        .context("waiting for dispatch listener to exit")?;
    if !listen_output.status.success() {
        anyhow::bail!(
            "listener failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&listen_output.stdout),
            String::from_utf8_lossy(&listen_output.stderr)
        );
    }
    let assignment: Value = serde_json::from_slice(&listen_output.stdout)
        .context("parsing dispatch listener output as JSON")?;
    let assignment_id = assignment["id"]
        .as_str()
        .context("listener output missing assignment id")?
        .to_string();

    assert_eq!(assignment["signal_id"].as_str(), Some(signal_id.as_str()));
    assert_eq!(assignment["worker_id"].as_str(), Some(worker_id.as_str()));
    assert_eq!(assignment["worker_role"].as_str(), Some("coder"));
    assert_eq!(
        assignment["envelope"]["resume"]["context_id"].as_str(),
        Some(context_id.as_str())
    );
    assert_eq!(
        assignment["envelope"]["context_preview"]["constraints"][0].as_str(),
        Some("Keep SQLite as the continuity source of truth")
    );
    assert_eq!(
        assignment["envelope"]["context_preview"]["operational_scars"][0].as_str(),
        Some("Nested dispatch payloads break tiny workers")
    );
    let anxiety = assignment["pressure"]["anxiety"]
        .as_f64()
        .context("assignment pressure missing anxiety")?;
    assert!(
        anxiety >= 0.7,
        "small-tier worker should inherit visible anxiety"
    );

    let acknowledged = run_json(
        root,
        &[
            "dispatch".to_string(),
            "ack".to_string(),
            "--worker-id".to_string(),
            worker_id.clone(),
            "--assignment-id".to_string(),
            assignment_id.clone(),
        ],
        &[],
    )?;
    assert_eq!(acknowledged["status"].as_str(), Some("completed"));

    let metrics = run_text(root, &["metrics".to_string()], &[])?;
    assert!(metrics.contains("ice_dispatch_up 1"));
    assert!(metrics.contains(&format!(
        "ice_dispatch_worker_connected{{worker_id=\"{worker_id}\""
    )));
    assert!(metrics.contains(&format!(
        "ice_dispatch_assignments{{worker_id=\"{worker_id}\",role=\"coder\",tier=\"small\",status=\"completed\"}} 1"
    )));
    assert!(metrics.contains(&format!("projection_id=\"dispatch:worker:{worker_id}\"")));
    assert!(metrics.contains("ice_dispatch_projection_workers{"));
    assert!(metrics.contains("ice_dispatch_projection_assignments{"));

    Ok(())
}

#[test]
fn read_context_surfaces_dispatch_presence_inside_organism() -> Result<()> {
    let Some(database_url) = std::env::var_os("ICE_TEST_POSTGRES_URL") else {
        eprintln!("skipping organism dispatch integration test: ICE_TEST_POSTGRES_URL not set");
        return Ok(());
    };
    let database_url = database_url
        .into_string()
        .map_err(|_| anyhow::anyhow!("ICE_TEST_POSTGRES_URL must be valid UTF-8"))?;
    let database = provision_test_database(&database_url)?;
    let dir = tempdir()?;
    let root = dir.path();
    let notify_channel = format!(
        "ice_dispatch_{}",
        &Uuid::now_v7().simple().to_string()[..12]
    );
    let worker_id = format!("observer-{}", &Uuid::now_v7().simple().to_string()[..8]);

    run_json(
        root,
        &[
            "dispatch".to_string(),
            "init".to_string(),
            "--database-url".to_string(),
            database.url.clone(),
            "--notify-channel".to_string(),
            notify_channel,
        ],
        &[],
    )?;

    let context = run_json(
        root,
        &[
            "uci".to_string(),
            "--json".to_string(),
            serde_json::to_string(&json!({
                "op": "open_context",
                "input": {
                    "namespace": "@machine",
                    "task_id": "machine-organism",
                    "session_id": "dispatch-organism",
                    "objective": "Watch dispatch enter the organism",
                    "agent_id": "planner-a",
                    "attachment_id": serde_json::Value::Null,
                    "selector": serde_json::Value::Null
                }
            }))?,
        ],
        &[],
    )?;
    let context_id = context["data"]["id"]
        .as_str()
        .context("open_context response missing context id")?
        .to_string();

    run_json(
        root,
        &[
            "dispatch".to_string(),
            "complete".to_string(),
            "--context-id".to_string(),
            context_id.clone(),
            "--agent".to_string(),
            "planner-a".to_string(),
            "--title".to_string(),
            "Planner finished the routing slice".to_string(),
            "--result".to_string(),
            "The next worker should continue from the dispatch projection proof.".to_string(),
            "--objective".to_string(),
            "Surface dispatch inside the organism".to_string(),
            "--quality".to_string(),
            "0.91".to_string(),
            "--target-role".to_string(),
            "coder".to_string(),
            "--preferred-tier".to_string(),
            "small".to_string(),
            "--reason".to_string(),
            "Wake one visible small worker".to_string(),
        ],
        &[],
    )?;

    let assignment = run_json(
        root,
        &[
            "dispatch".to_string(),
            "claim".to_string(),
            "--worker-id".to_string(),
            worker_id.clone(),
            "--display-name".to_string(),
            "Observer Small".to_string(),
            "--role".to_string(),
            "coder".to_string(),
            "--agent-type".to_string(),
            "ollama".to_string(),
            "--tier".to_string(),
            "small".to_string(),
            "--model".to_string(),
            "qwen2.5:0.5b".to_string(),
            "--capabilities".to_string(),
            "read,write,claim".to_string(),
            "--max-parallelism".to_string(),
            "1".to_string(),
            "--namespace".to_string(),
            "@machine".to_string(),
            "--task".to_string(),
            "machine-organism".to_string(),
            "--focus".to_string(),
            "Continue from dispatch projection proof".to_string(),
        ],
        &[],
    )?;
    let assignment_id = assignment["id"]
        .as_str()
        .context("dispatch claim did not return assignment id")?
        .to_string();

    let read = run_json(
        root,
        &[
            "uci".to_string(),
            "--json".to_string(),
            serde_json::to_string(&json!({
                "op": "read_context",
                "input": {
                    "context_id": context_id,
                    "namespace": serde_json::Value::Null,
                    "task_id": serde_json::Value::Null,
                    "objective": "Inspect the machine organism",
                    "token_budget": 256,
                    "selector": serde_json::Value::Null,
                    "agent_id": "planner-a",
                    "session_id": serde_json::Value::Null,
                    "view_id": serde_json::Value::Null,
                    "include_resolved": true,
                    "candidate_limit": 16
                }
            }))?,
        ],
        &[],
    )?;

    assert_eq!(
        read["data"]["organism"]["dispatch"]["configured"].as_bool(),
        Some(true)
    );
    assert_eq!(
        read["data"]["organism"]["dispatch"]["reachable"].as_bool(),
        Some(true)
    );
    assert_eq!(
        read["data"]["organism"]["dispatch"]["workers_active"].as_u64(),
        Some(1)
    );
    assert_eq!(
        read["data"]["organism"]["dispatch"]["assignments_active"].as_u64(),
        Some(1)
    );
    assert_eq!(
        read["data"]["organism"]["dispatch"]["workers"][0]["worker_id"].as_str(),
        Some(worker_id.as_str())
    );
    assert_eq!(
        read["data"]["organism"]["dispatch"]["assignments"][0]["assignment_id"].as_str(),
        Some(assignment_id.as_str())
    );
    let projection_id = format!("dispatch:worker:{worker_id}");
    let projection = read["data"]["organism"]["lane_projections"]
        .as_array()
        .context("lane projections should be present")?
        .iter()
        .find(|projection| projection["projection_id"].as_str() == Some(projection_id.as_str()))
        .context("dispatch worker lane should appear inside organism lane projections")?;
    assert_eq!(projection["dispatch_assignment_count"].as_u64(), Some(1));
    let projected_anxiety = projection["dispatch_assignment_anxiety_max"]
        .as_f64()
        .context("dispatch worker lane should expose assignment anxiety")?;
    assert!(
        projected_anxiety >= 0.7,
        "dispatch worker lane should carry the small-tier anxiety profile"
    );
    let projected_non_dispatch = read["data"]["organism"]["lane_projections"]
        .as_array()
        .context("lane projections should be present")?
        .iter()
        .filter(|projection| projection["projection_kind"].as_str() != Some("dispatch_worker"))
        .any(|projection| {
            projection["dispatch_assignment_count"]
                .as_u64()
                .unwrap_or(0)
                > 0
        });
    assert!(
        !projected_non_dispatch,
        "dispatch pressure should stay on dispatch worker lanes when no attached_lane metadata is present"
    );

    Ok(())
}

#[test]
fn read_context_rolls_dispatch_pressure_into_explicit_attached_lane() -> Result<()> {
    let Some(database_url) = std::env::var_os("ICE_TEST_POSTGRES_URL") else {
        eprintln!(
            "skipping attached-lane dispatch integration test: ICE_TEST_POSTGRES_URL not set"
        );
        return Ok(());
    };
    let database_url = database_url
        .into_string()
        .map_err(|_| anyhow::anyhow!("ICE_TEST_POSTGRES_URL must be valid UTF-8"))?;
    let database = provision_test_database(&database_url)?;
    let dir = tempdir()?;
    let root = dir.path();
    let notify_channel = format!(
        "ice_dispatch_{}",
        &Uuid::now_v7().simple().to_string()[..12]
    );
    let worker_id = format!("repo-bound-{}", &Uuid::now_v7().simple().to_string()[..8]);
    let repo_root = root.join("demo-repo");
    let repo_projection_id = format!("repo:{}:main", repo_root.display());

    run_json(
        root,
        &[
            "dispatch".to_string(),
            "init".to_string(),
            "--database-url".to_string(),
            database.url.clone(),
            "--notify-channel".to_string(),
            notify_channel,
        ],
        &[],
    )?;

    let context = run_json(
        root,
        &[
            "uci".to_string(),
            "--json".to_string(),
            serde_json::to_string(&json!({
                "op": "open_context",
                "input": {
                    "namespace": "@machine",
                    "task_id": "machine-organism",
                    "session_id": "dispatch-attached-lane",
                    "objective": "Roll dispatch pressure into an explicit repo lane",
                    "agent_id": "planner-a",
                    "attachment_id": serde_json::Value::Null,
                    "selector": serde_json::Value::Null
                }
            }))?,
        ],
        &[],
    )?;
    let context_id = context["data"]["id"]
        .as_str()
        .context("open_context response missing context id")?
        .to_string();

    run_json(
        root,
        &[
            "dispatch".to_string(),
            "complete".to_string(),
            "--context-id".to_string(),
            context_id.clone(),
            "--agent".to_string(),
            "planner-a".to_string(),
            "--title".to_string(),
            "Planner finished the repo-attached routing slice".to_string(),
            "--result".to_string(),
            "The next worker should continue inside the explicitly attached repo lane.".to_string(),
            "--objective".to_string(),
            "Surface dispatch pressure in the attached repo lane".to_string(),
            "--quality".to_string(),
            "0.93".to_string(),
            "--target-role".to_string(),
            "coder".to_string(),
            "--preferred-tier".to_string(),
            "small".to_string(),
            "--reason".to_string(),
            "Wake one repo-attached small worker".to_string(),
        ],
        &[],
    )?;

    let assignment = run_json(
        root,
        &[
            "dispatch".to_string(),
            "claim".to_string(),
            "--worker-id".to_string(),
            worker_id.clone(),
            "--display-name".to_string(),
            "Repo Bound Small".to_string(),
            "--role".to_string(),
            "coder".to_string(),
            "--agent-type".to_string(),
            "ollama".to_string(),
            "--tier".to_string(),
            "small".to_string(),
            "--model".to_string(),
            "qwen2.5:0.5b".to_string(),
            "--capabilities".to_string(),
            "read,write,claim".to_string(),
            "--max-parallelism".to_string(),
            "1".to_string(),
            "--namespace".to_string(),
            "@machine".to_string(),
            "--task".to_string(),
            "machine-organism".to_string(),
            "--focus".to_string(),
            "Continue from attached repo dispatch proof".to_string(),
            "--attached-repo-root".to_string(),
            repo_root.display().to_string(),
            "--attached-branch".to_string(),
            "main".to_string(),
        ],
        &[],
    )?;

    assert_eq!(
        assignment["envelope"]["attached_projected_lane"]["projection_id"].as_str(),
        Some(repo_projection_id.as_str())
    );
    assert_eq!(
        assignment["envelope"]["attached_projected_lane_source"].as_str(),
        Some("explicit_cli")
    );

    let read = run_json(
        root,
        &[
            "uci".to_string(),
            "--json".to_string(),
            serde_json::to_string(&json!({
                "op": "read_context",
                "input": {
                    "context_id": context_id,
                    "namespace": serde_json::Value::Null,
                    "task_id": serde_json::Value::Null,
                    "objective": "Inspect the machine organism after explicit attached lane routing",
                    "token_budget": 256,
                    "selector": serde_json::Value::Null,
                    "agent_id": "planner-a",
                    "session_id": serde_json::Value::Null,
                    "view_id": serde_json::Value::Null,
                    "include_resolved": true,
                    "candidate_limit": 16
                }
            }))?,
        ],
        &[],
    )?;

    let dispatch_projection_id = format!("dispatch:worker:{worker_id}");
    let lane_projections = read["data"]["organism"]["lane_projections"]
        .as_array()
        .context("lane projections should be present")?;
    let dispatch_lane = lane_projections
        .iter()
        .find(|projection| {
            projection["projection_id"].as_str() == Some(dispatch_projection_id.as_str())
        })
        .context("dispatch worker lane should still be present")?;
    assert_eq!(dispatch_lane["dispatch_assignment_count"].as_u64(), Some(1));
    assert_eq!(
        dispatch_lane["dispatch_assignment_explicit_cli_count"].as_u64(),
        Some(0)
    );
    assert_eq!(
        dispatch_lane["dispatch_assignment_live_badge_opt_in_count"].as_u64(),
        Some(0)
    );

    let repo_lane = lane_projections
        .iter()
        .find(|projection| {
            projection["projection_id"].as_str() == Some(repo_projection_id.as_str())
        })
        .context("explicitly attached repo lane should receive dispatch pressure")?;
    assert_eq!(repo_lane["projection_kind"].as_str(), Some("repo"));
    assert_eq!(repo_lane["dispatch_assignment_count"].as_u64(), Some(1));
    assert_eq!(
        repo_lane["dispatch_assignment_explicit_cli_count"].as_u64(),
        Some(1)
    );
    assert_eq!(
        repo_lane["dispatch_assignment_live_badge_opt_in_count"].as_u64(),
        Some(0)
    );
    assert_eq!(repo_lane["connected_agents"].as_u64(), Some(0));
    assert_eq!(
        repo_lane["agent_ids"].as_array().map(Vec::len),
        Some(0),
        "repo lane should inherit pressure without inventing ownership"
    );
    let repo_anxiety = repo_lane["dispatch_assignment_anxiety_max"]
        .as_f64()
        .context("repo lane should expose rolled-up dispatch anxiety")?;
    assert!(repo_anxiety >= 0.7);
    let dispatch_state = &read["data"]["organism"]["dispatch"];
    let worker_presence = dispatch_state["workers"]
        .as_array()
        .context("dispatch workers should be present")?
        .iter()
        .find(|worker| worker["worker_id"].as_str() == Some(worker_id.as_str()))
        .context("repo-attached worker should be present in dispatch snapshot")?;
    assert_eq!(
        worker_presence["attached_projected_lane_source"].as_str(),
        Some("explicit_cli")
    );

    let metrics = run_text(root, &["metrics".to_string()], &[])?;
    assert!(metrics.contains(&format!("projection_id=\"{}\"", repo_projection_id)));
    assert!(metrics.contains("projection_kind=\"repo\""));
    assert!(metrics.contains("ice_dispatch_projection_assignments{"));
    assert!(metrics.contains("ice_dispatch_projection_assignment_sources{"));
    assert!(metrics.contains("source=\"explicit_cli\""));

    Ok(())
}

#[test]
fn dispatch_claim_derives_attached_lane_from_live_badge_state_only_when_opted_in() -> Result<()> {
    let Some(database_url) = std::env::var_os("ICE_TEST_POSTGRES_URL") else {
        eprintln!(
            "skipping badge-derived dispatch integration test: ICE_TEST_POSTGRES_URL not set"
        );
        return Ok(());
    };
    let database_url = database_url
        .into_string()
        .map_err(|_| anyhow::anyhow!("ICE_TEST_POSTGRES_URL must be valid UTF-8"))?;
    let database = provision_test_database(&database_url)?;
    let dir = tempdir()?;
    let root = dir.path();
    let notify_channel = format!(
        "ice_dispatch_{}",
        &Uuid::now_v7().simple().to_string()[..12]
    );
    let worker_id = format!("badge-bound-{}", &Uuid::now_v7().simple().to_string()[..8]);
    let repo_root = root.join("badge-demo-repo");
    let repo_projection_id = format!("repo:{}:main", repo_root.display());

    run_json(
        root,
        &[
            "dispatch".to_string(),
            "init".to_string(),
            "--database-url".to_string(),
            database.url.clone(),
            "--notify-channel".to_string(),
            notify_channel,
        ],
        &[],
    )?;

    let attachment = run_json(
        root,
        &[
            "uci".to_string(),
            "--json".to_string(),
            serde_json::to_string(&json!({
                "op": "attach_agent",
                "input": {
                    "agent_id": worker_id,
                    "agent_type": "codex",
                    "capabilities": ["read", "write", "claim"],
                    "namespace": "@machine",
                    "role": "coder",
                    "metadata": {}
                }
            }))?,
        ],
        &[],
    )?;
    let attachment_id = attachment["data"]["id"]
        .as_str()
        .context("attach_agent response missing attachment id")?
        .to_string();

    let context = run_json(
        root,
        &[
            "uci".to_string(),
            "--json".to_string(),
            serde_json::to_string(&json!({
                "op": "open_context",
                "input": {
                    "namespace": "@machine",
                    "task_id": "machine-organism",
                    "session_id": "dispatch-badge-derived",
                    "objective": "Derive attached repo lane from live badge state",
                    "agent_id": "planner-a",
                    "attachment_id": serde_json::Value::Null,
                    "selector": serde_json::Value::Null
                }
            }))?,
        ],
        &[],
    )?;
    let context_id = context["data"]["id"]
        .as_str()
        .context("open_context response missing context id")?
        .to_string();

    run_json(
        root,
        &[
            "uci".to_string(),
            "--json".to_string(),
            serde_json::to_string(&json!({
                "op": "upsert_agent_badge",
                "input": {
                    "attachment_id": attachment_id,
                    "agent_id": serde_json::Value::Null,
                    "namespace": serde_json::Value::Null,
                    "context_id": context_id,
                    "display_name": "Badge Bound Worker",
                    "status": "watching",
                    "focus": "Stay inside the badge-derived repo lane",
                    "headline": "repo lane badge",
                    "resource": "repo/badge-demo-repo/main",
                    "repo_root": repo_root.display().to_string(),
                    "branch": "main",
                    "metadata": {}
                }
            }))?,
        ],
        &[],
    )?;

    run_json(
        root,
        &[
            "dispatch".to_string(),
            "complete".to_string(),
            "--context-id".to_string(),
            context_id.clone(),
            "--agent".to_string(),
            "planner-a".to_string(),
            "--title".to_string(),
            "Planner finished the badge-derived routing slice".to_string(),
            "--result".to_string(),
            "The next worker should derive its repo lane from the live badge state.".to_string(),
            "--objective".to_string(),
            "Surface dispatch pressure from a badge-derived repo lane".to_string(),
            "--quality".to_string(),
            "0.94".to_string(),
            "--target-role".to_string(),
            "coder".to_string(),
            "--preferred-tier".to_string(),
            "small".to_string(),
            "--reason".to_string(),
            "Wake one badge-bound small worker".to_string(),
        ],
        &[],
    )?;

    let assignment = run_json(
        root,
        &[
            "dispatch".to_string(),
            "claim".to_string(),
            "--worker-id".to_string(),
            worker_id.clone(),
            "--display-name".to_string(),
            "Badge Bound Worker".to_string(),
            "--role".to_string(),
            "coder".to_string(),
            "--agent-type".to_string(),
            "ollama".to_string(),
            "--tier".to_string(),
            "small".to_string(),
            "--model".to_string(),
            "qwen2.5:0.5b".to_string(),
            "--capabilities".to_string(),
            "read,write,claim".to_string(),
            "--max-parallelism".to_string(),
            "1".to_string(),
            "--namespace".to_string(),
            "@machine".to_string(),
            "--task".to_string(),
            "machine-organism".to_string(),
            "--focus".to_string(),
            "Continue from badge-derived repo dispatch proof".to_string(),
            "--derive-attached-lane-from-badge".to_string(),
        ],
        &[],
    )?;

    assert_eq!(
        assignment["envelope"]["attached_projected_lane"]["projection_id"].as_str(),
        Some(repo_projection_id.as_str())
    );
    assert_eq!(
        assignment["envelope"]["attached_projected_lane_source"].as_str(),
        Some("live_badge_opt_in")
    );

    let read = run_json(
        root,
        &[
            "uci".to_string(),
            "--json".to_string(),
            serde_json::to_string(&json!({
                "op": "read_context",
                "input": {
                    "context_id": context_id,
                    "namespace": serde_json::Value::Null,
                    "task_id": serde_json::Value::Null,
                    "objective": "Inspect the badge-derived dispatch lane",
                    "token_budget": 256,
                    "selector": serde_json::Value::Null,
                    "agent_id": "planner-a",
                    "session_id": serde_json::Value::Null,
                    "view_id": serde_json::Value::Null,
                    "include_resolved": true,
                    "candidate_limit": 16
                }
            }))?,
        ],
        &[],
    )?;
    let repo_lane = read["data"]["organism"]["lane_projections"]
        .as_array()
        .context("lane projections should be present")?
        .iter()
        .find(|projection| {
            projection["projection_id"].as_str() == Some(repo_projection_id.as_str())
        })
        .context("badge-derived repo lane should receive dispatch pressure")?;
    assert_eq!(repo_lane["dispatch_assignment_count"].as_u64(), Some(1));
    assert_eq!(
        repo_lane["dispatch_assignment_explicit_cli_count"].as_u64(),
        Some(0)
    );
    assert_eq!(
        repo_lane["dispatch_assignment_live_badge_opt_in_count"].as_u64(),
        Some(1)
    );
    assert_eq!(repo_lane["connected_agents"].as_u64(), Some(1));
    let assignment_presence = read["data"]["organism"]["dispatch"]["assignments"]
        .as_array()
        .context("dispatch assignments should be present")?
        .iter()
        .find(|item| item["worker_id"].as_str() == Some(worker_id.as_str()))
        .context("badge-derived worker assignment should be present in dispatch snapshot")?;
    assert_eq!(
        assignment_presence["attached_projected_lane_source"].as_str(),
        Some("live_badge_opt_in")
    );

    Ok(())
}

#[test]
fn dispatch_claim_ignores_live_badge_state_without_opt_in() -> Result<()> {
    let Some(database_url) = std::env::var_os("ICE_TEST_POSTGRES_URL") else {
        eprintln!(
            "skipping badge-derived dispatch integration test: ICE_TEST_POSTGRES_URL not set"
        );
        return Ok(());
    };
    let database_url = database_url
        .into_string()
        .map_err(|_| anyhow::anyhow!("ICE_TEST_POSTGRES_URL must be valid UTF-8"))?;
    let database = provision_test_database(&database_url)?;
    let dir = tempdir()?;
    let root = dir.path();
    let notify_channel = format!(
        "ice_dispatch_{}",
        &Uuid::now_v7().simple().to_string()[..12]
    );
    let worker_id = format!(
        "badge-passive-{}",
        &Uuid::now_v7().simple().to_string()[..8]
    );
    let repo_root = root.join("badge-passive-repo");
    let repo_projection_id = format!("repo:{}:main", repo_root.display());

    run_json(
        root,
        &[
            "dispatch".to_string(),
            "init".to_string(),
            "--database-url".to_string(),
            database.url.clone(),
            "--notify-channel".to_string(),
            notify_channel,
        ],
        &[],
    )?;

    let attachment = run_json(
        root,
        &[
            "uci".to_string(),
            "--json".to_string(),
            serde_json::to_string(&json!({
                "op": "attach_agent",
                "input": {
                    "agent_id": worker_id,
                    "agent_type": "codex",
                    "capabilities": ["read", "write", "claim"],
                    "namespace": "@machine",
                    "role": "coder",
                    "metadata": {}
                }
            }))?,
        ],
        &[],
    )?;
    let attachment_id = attachment["data"]["id"]
        .as_str()
        .context("attach_agent response missing attachment id")?
        .to_string();

    let context = run_json(
        root,
        &[
            "uci".to_string(),
            "--json".to_string(),
            serde_json::to_string(&json!({
                "op": "open_context",
                "input": {
                    "namespace": "@machine",
                    "task_id": "machine-organism",
                    "session_id": "dispatch-badge-passive",
                    "objective": "Refuse attached repo lane inference without explicit opt-in",
                    "agent_id": "planner-a",
                    "attachment_id": serde_json::Value::Null,
                    "selector": serde_json::Value::Null
                }
            }))?,
        ],
        &[],
    )?;
    let context_id = context["data"]["id"]
        .as_str()
        .context("open_context response missing context id")?
        .to_string();

    run_json(
        root,
        &[
            "uci".to_string(),
            "--json".to_string(),
            serde_json::to_string(&json!({
                "op": "upsert_agent_badge",
                "input": {
                    "attachment_id": attachment_id,
                    "agent_id": serde_json::Value::Null,
                    "namespace": serde_json::Value::Null,
                    "context_id": context_id,
                    "display_name": "Badge Passive Worker",
                    "status": "watching",
                    "focus": "Stay inside the badge-derived repo lane",
                    "headline": "repo lane badge",
                    "resource": "repo/badge-passive-repo/main",
                    "repo_root": repo_root.display().to_string(),
                    "branch": "main",
                    "metadata": {}
                }
            }))?,
        ],
        &[],
    )?;

    run_json(
        root,
        &[
            "dispatch".to_string(),
            "complete".to_string(),
            "--context-id".to_string(),
            context_id.clone(),
            "--agent".to_string(),
            "planner-a".to_string(),
            "--title".to_string(),
            "Planner finished the non-inferred routing slice".to_string(),
            "--result".to_string(),
            "The next worker should not inherit a repo lane without explicit opt-in.".to_string(),
            "--objective".to_string(),
            "Keep dispatch pressure off repo lanes unless opt-in is explicit".to_string(),
            "--quality".to_string(),
            "0.91".to_string(),
            "--target-role".to_string(),
            "coder".to_string(),
            "--preferred-tier".to_string(),
            "small".to_string(),
            "--reason".to_string(),
            "Wake one worker without badge-derived opt-in".to_string(),
        ],
        &[],
    )?;

    let assignment = run_json(
        root,
        &[
            "dispatch".to_string(),
            "claim".to_string(),
            "--worker-id".to_string(),
            worker_id.clone(),
            "--display-name".to_string(),
            "Badge Passive Worker".to_string(),
            "--role".to_string(),
            "coder".to_string(),
            "--agent-type".to_string(),
            "ollama".to_string(),
            "--tier".to_string(),
            "small".to_string(),
            "--model".to_string(),
            "qwen2.5:0.5b".to_string(),
            "--capabilities".to_string(),
            "read,write,claim".to_string(),
            "--max-parallelism".to_string(),
            "1".to_string(),
            "--namespace".to_string(),
            "@machine".to_string(),
            "--task".to_string(),
            "machine-organism".to_string(),
            "--focus".to_string(),
            "Continue from passive badge dispatch proof".to_string(),
        ],
        &[],
    )?;

    assert!(
        assignment["envelope"]["attached_projected_lane"].is_null(),
        "attached lane should stay unset without explicit badge-derived opt-in"
    );
    assert!(
        assignment["envelope"]["attached_projected_lane_source"].is_null(),
        "attached lane source should stay unset without explicit badge-derived opt-in"
    );

    let read = run_json(
        root,
        &[
            "uci".to_string(),
            "--json".to_string(),
            serde_json::to_string(&json!({
                "op": "read_context",
                "input": {
                    "context_id": context_id,
                    "namespace": serde_json::Value::Null,
                    "task_id": serde_json::Value::Null,
                    "objective": "Inspect the non-inferred dispatch lane",
                    "token_budget": 256,
                    "selector": serde_json::Value::Null,
                    "agent_id": "planner-a",
                    "session_id": serde_json::Value::Null,
                    "view_id": serde_json::Value::Null,
                    "include_resolved": true,
                    "candidate_limit": 16
                }
            }))?,
        ],
        &[],
    )?;
    let repo_lane = read["data"]["organism"]["lane_projections"]
        .as_array()
        .context("lane projections should be present")?
        .iter()
        .find(|projection| {
            projection["projection_id"].as_str() == Some(repo_projection_id.as_str())
        })
        .context("repo lane should still exist from the live badge")?;
    assert_eq!(
        repo_lane["dispatch_assignment_count"].as_u64().unwrap_or(0),
        0,
        "badge presence alone should not roll dispatch pressure into the repo lane"
    );
    assert_eq!(
        repo_lane["dispatch_assignment_explicit_cli_count"].as_u64(),
        Some(0)
    );
    assert_eq!(
        repo_lane["dispatch_assignment_live_badge_opt_in_count"].as_u64(),
        Some(0)
    );
    assert_eq!(
        repo_lane["connected_agents"].as_u64(),
        Some(1),
        "the live badge should remain visible as operator state"
    );

    Ok(())
}
