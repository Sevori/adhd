use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use serde_json::Value;
use tempfile::tempdir;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_ice")
}

fn run_ice(current_dir: &Path, args: &[&str]) -> Result<Value> {
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
fn claude_install_status_uninstall_roundtrip_is_hermetic() -> Result<()> {
    let dir = tempdir()?;
    let root = dir.path().join("organisms/ice");
    let code_config = dir.path().join(".claude.json");

    let install = run_ice(
        dir.path(),
        &[
            "claude",
            "install-global",
            "--root",
            root.to_str().context("root utf-8")?,
            "--code-config",
            code_config.to_str().context("code config utf-8")?,
        ],
    )?;

    assert_eq!(install["changed"].as_bool(), Some(true));
    assert_eq!(install["restart_required"].as_bool(), Some(true));
    assert_eq!(
        install["code_config_path"].as_str(),
        Some(code_config.to_str().unwrap())
    );
    assert_eq!(
        install["organism_root"].as_str(),
        Some(root.to_str().unwrap())
    );

    let status = run_ice(
        dir.path(),
        &[
            "claude",
            "status",
            "--root",
            root.to_str().context("root utf-8")?,
            "--code-config",
            code_config.to_str().context("code config utf-8")?,
        ],
    )?;

    assert_eq!(status["binary_exists"].as_bool(), Some(true));
    assert_eq!(status["organism_root_exists"].as_bool(), Some(true));
    assert_eq!(status["code_config_found"].as_bool(), Some(true));
    assert_eq!(status["code_has_ice_entry"].as_bool(), Some(true));

    let second_install = run_ice(
        dir.path(),
        &[
            "claude",
            "install-global",
            "--root",
            root.to_str().context("root utf-8")?,
            "--code-config",
            code_config.to_str().context("code config utf-8")?,
        ],
    )?;

    assert_eq!(second_install["changed"].as_bool(), Some(false));

    let uninstall = run_ice(
        dir.path(),
        &[
            "claude",
            "uninstall",
            "--code-config",
            code_config.to_str().context("code config utf-8")?,
        ],
    )?;

    assert_eq!(uninstall["changed"].as_bool(), Some(true));

    let status_after = run_ice(
        dir.path(),
        &[
            "claude",
            "status",
            "--root",
            root.to_str().context("root utf-8")?,
            "--code-config",
            code_config.to_str().context("code config utf-8")?,
        ],
    )?;

    assert_eq!(status_after["code_has_ice_entry"].as_bool(), Some(false));

    Ok(())
}
