use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use serde_json::Value;
use tempfile::tempdir;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_ice")
}

fn run_install(current_dir: &Path, args: &[&str]) -> Result<Value> {
    let output = Command::new(bin())
        .current_dir(current_dir)
        .args(args)
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
    serde_json::from_slice(&output.stdout)
        .with_context(|| format!("parsing JSON output from ice {}", args.join(" ")))
}

#[test]
fn codex_install_global_writes_managed_config_without_creating_repo_root() -> Result<()> {
    let codex_home = tempdir()?;
    let scratch = tempdir()?;
    let config_path = codex_home.path().join("config.toml");
    let machine_root = codex_home.path().join("organisms/brain");

    let first = run_install(
        scratch.path(),
        &[
            "codex",
            "install-global",
            "--config-path",
            config_path.to_str().context("config path utf-8")?,
            "--machine-root",
            machine_root.to_str().context("machine root utf-8")?,
        ],
    )?;
    assert_eq!(first["changed"].as_bool(), Some(true));
    assert_eq!(first["server_name"].as_str(), Some("ice_machine"));
    assert!(config_path.exists());
    assert!(machine_root.join("data").exists());
    assert!(
        !scratch.path().join(".ice").exists(),
        "codex install should not eagerly create a repo-local .ice root"
    );

    let config = fs::read_to_string(&config_path)?;
    assert!(config.contains("[mcp_servers.ice_machine]"));
    assert!(config.contains("mcp"));
    assert!(config.contains(machine_root.to_str().context("machine root utf-8")?));

    let second = run_install(
        scratch.path(),
        &[
            "codex",
            "install-global",
            "--config-path",
            config_path.to_str().context("config path utf-8")?,
            "--machine-root",
            machine_root.to_str().context("machine root utf-8")?,
        ],
    )?;
    assert_eq!(second["changed"].as_bool(), Some(false));

    Ok(())
}
