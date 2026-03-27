use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{Value, json};

pub const DEFAULT_CLAUDE_SERVER_NAME: &str = "ice-shared-continuity-kernel";
pub const DEFAULT_CLAUDE_MACHINE_ROOT_DIR: &str = ".claude/organisms/ice";

const ICE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone)]
pub struct ClaudeCodeInstallRequest {
    pub organism_root: Option<PathBuf>,
    pub server_name: String,
    pub code_config_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ClaudeCodeInstallResult {
    pub changed: bool,
    pub code_config_path: String,
    pub organism_root: String,
    pub server_name: String,
    pub command: String,
    pub args: Vec<String>,
    pub restart_required: bool,
    pub skipped_malformed: bool,
}

#[derive(Debug, Clone)]
pub struct ClaudeCodeUninstallRequest {
    pub server_name: String,
    pub code_config_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ClaudeCodeUninstallResult {
    pub changed: bool,
    pub code_config_path: String,
    pub skipped_malformed: bool,
}

#[derive(Debug, Clone)]
pub struct ClaudeCodeStatusRequest {
    pub server_name: String,
    pub code_config_path: Option<PathBuf>,
    pub organism_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ClaudeCodeStatusResult {
    pub binary_path: String,
    pub binary_exists: bool,
    pub organism_root: String,
    pub organism_root_exists: bool,
    pub code_config_path: String,
    pub code_config_found: bool,
    pub code_has_ice_entry: bool,
    pub ice_version: String,
}

pub fn install_claude_code(request: ClaudeCodeInstallRequest) -> Result<ClaudeCodeInstallResult> {
    validate_server_name(&request.server_name)?;

    let binary = resolve_binary()?;
    let organism_root = resolve_organism_root(request.organism_root.as_ref())?;
    let code_config_path = resolve_code_config(request.code_config_path.as_deref())?;

    fs::create_dir_all(&organism_root)
        .with_context(|| format!("creating organism root {}", organism_root.display()))?;

    let args = vec![
        "--root".to_string(),
        organism_root.display().to_string(),
        "mcp".to_string(),
    ];
    let entry = build_server_entry(&binary, &args);
    let (changed, skipped_malformed) =
        upsert_code_entry(&code_config_path, &request.server_name, &entry)?;

    Ok(ClaudeCodeInstallResult {
        changed,
        code_config_path: code_config_path.display().to_string(),
        organism_root: organism_root.display().to_string(),
        server_name: request.server_name,
        command: binary.display().to_string(),
        args,
        restart_required: true,
        skipped_malformed,
    })
}

pub fn uninstall_claude_code(
    request: ClaudeCodeUninstallRequest,
) -> Result<ClaudeCodeUninstallResult> {
    validate_server_name(&request.server_name)?;

    let code_config_path = resolve_code_config(request.code_config_path.as_deref())?;
    let (changed, skipped_malformed) =
        remove_managed_entry(&code_config_path, &request.server_name)?;

    Ok(ClaudeCodeUninstallResult {
        changed,
        code_config_path: code_config_path.display().to_string(),
        skipped_malformed,
    })
}

pub fn claude_code_status(request: ClaudeCodeStatusRequest) -> Result<ClaudeCodeStatusResult> {
    validate_server_name(&request.server_name)?;

    let binary = resolve_binary()?;
    let organism_root = resolve_organism_root(request.organism_root.as_ref())?;
    let code_config_path = resolve_code_config(request.code_config_path.as_deref())?;

    Ok(ClaudeCodeStatusResult {
        binary_path: binary.display().to_string(),
        binary_exists: binary.exists(),
        organism_root: organism_root.display().to_string(),
        organism_root_exists: organism_root.exists(),
        code_config_path: code_config_path.display().to_string(),
        code_config_found: code_config_path.exists(),
        code_has_ice_entry: has_ice_entry(&code_config_path, &request.server_name),
        ice_version: ICE_VERSION.to_string(),
    })
}

fn resolve_binary() -> Result<PathBuf> {
    env::current_exe().context("resolving current executable path")
}

fn resolve_home() -> Result<PathBuf> {
    #[cfg(windows)]
    if let Some(val) = env::var_os("USERPROFILE") {
        return Ok(PathBuf::from(val));
    }
    if let Some(val) = env::var_os("HOME") {
        return Ok(PathBuf::from(val));
    }
    anyhow::bail!("HOME environment variable not set; cannot resolve Claude Code paths")
}

fn resolve_absolute(path: impl AsRef<Path>) -> Result<PathBuf> {
    let path = path.as_ref();
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    Ok(env::current_dir()
        .context("resolving current working directory")?
        .join(path))
}

fn resolve_organism_root(requested: Option<&PathBuf>) -> Result<PathBuf> {
    if let Some(path) = requested {
        return resolve_absolute(path);
    }
    let home = resolve_home()?;
    Ok(home.join(DEFAULT_CLAUDE_MACHINE_ROOT_DIR))
}

fn resolve_code_config(override_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        return resolve_absolute(path);
    }
    let home = resolve_home()?;
    Ok(home.join(".claude.json"))
}

fn validate_server_name(server_name: &str) -> Result<()> {
    if server_name.is_empty()
        || !server_name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
    {
        anyhow::bail!(
            "server name must contain only ASCII letters, digits, underscores, or hyphens"
        );
    }
    Ok(())
}

fn build_server_entry(command: &Path, args: &[String]) -> Value {
    json!({
        "type": "stdio",
        "command": command.display().to_string(),
        "args": args,
        "env": {},
        "_ice_managed": true,
        "_ice_version": ICE_VERSION,
    })
}

fn read_config(path: &Path) -> (Value, bool) {
    if !path.exists() {
        return (json!({}), false);
    }
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!(
                "warning: could not read {}: {err}; skipping",
                path.display()
            );
            return (json!({}), true);
        }
    };
    match serde_json::from_str::<Value>(&text) {
        Ok(value) if value.is_object() => (value, false),
        _ => {
            eprintln!(
                "warning: {} contains malformed or non-object JSON; skipping",
                path.display()
            );
            (json!({}), true)
        }
    }
}

fn write_config(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating config directory {}", parent.display()))?;
    }
    let rendered = serde_json::to_string_pretty(value).context("rendering Claude config JSON")?;
    fs::write(path, rendered).with_context(|| format!("writing Claude config {}", path.display()))
}

fn upsert_code_entry(config_path: &Path, server_name: &str, entry: &Value) -> Result<(bool, bool)> {
    let (mut config, malformed) = read_config(config_path);
    if malformed {
        return Ok((false, true));
    }

    let servers = config
        .as_object_mut()
        .expect("config must be object")
        .entry("mcpServers")
        .or_insert_with(|| json!({}));
    let servers_obj = servers
        .as_object_mut()
        .context("Claude Code config `mcpServers` is not a JSON object")?;

    if let Some(existing) = servers_obj.get(server_name) {
        let managed = existing
            .get("_ice_managed")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !managed {
            anyhow::bail!(
                "Claude Code config already contains unmanaged mcpServers.{server_name}; rename it or remove it before installing the managed ICE entry"
            );
        }
        if existing == entry {
            return Ok((false, false));
        }
    }

    servers_obj.insert(server_name.to_string(), entry.clone());
    write_config(config_path, &config)?;
    Ok((true, false))
}

fn remove_managed_entry(config_path: &Path, server_name: &str) -> Result<(bool, bool)> {
    if !config_path.exists() {
        return Ok((false, false));
    }

    let (mut config, malformed) = read_config(config_path);
    if malformed {
        return Ok((false, true));
    }

    let changed = config
        .get_mut("mcpServers")
        .and_then(Value::as_object_mut)
        .map(|servers| {
            let managed = servers
                .get(server_name)
                .and_then(|value| value.get("_ice_managed"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if managed {
                servers.remove(server_name);
                true
            } else {
                false
            }
        })
        .unwrap_or(false);

    if changed {
        write_config(config_path, &config)?;
    }

    Ok((changed, false))
}

fn has_ice_entry(config_path: &Path, server_name: &str) -> bool {
    let (config, malformed) = read_config(config_path);
    if malformed {
        return false;
    }
    config
        .get("mcpServers")
        .and_then(|value| value.get(server_name))
        .is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_entry() -> Value {
        build_server_entry(
            &PathBuf::from("/tmp/ice"),
            &["--root".into(), "/tmp/organism".into(), "mcp".into()],
        )
    }

    #[test]
    fn upsert_creates_managed_entry() {
        let dir = tempdir().unwrap();
        let config = dir.path().join(".claude.json");

        let (changed, skipped) =
            upsert_code_entry(&config, DEFAULT_CLAUDE_SERVER_NAME, &test_entry()).unwrap();
        assert!(changed);
        assert!(!skipped);

        let written: Value = serde_json::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert!(
            written["mcpServers"][DEFAULT_CLAUDE_SERVER_NAME]["_ice_managed"]
                .as_bool()
                .unwrap()
        );
    }

    #[test]
    fn upsert_is_idempotent_for_same_managed_entry() {
        let dir = tempdir().unwrap();
        let config = dir.path().join(".claude.json");

        let first = upsert_code_entry(&config, DEFAULT_CLAUDE_SERVER_NAME, &test_entry()).unwrap();
        let second = upsert_code_entry(&config, DEFAULT_CLAUDE_SERVER_NAME, &test_entry()).unwrap();
        assert_eq!(first, (true, false));
        assert_eq!(second, (false, false));
    }

    #[test]
    fn upsert_preserves_other_servers() {
        let dir = tempdir().unwrap();
        let config = dir.path().join(".claude.json");
        let initial = json!({
            "mcpServers": {
                "other": { "command": "other", "args": [] }
            }
        });
        fs::write(&config, serde_json::to_string_pretty(&initial).unwrap()).unwrap();

        upsert_code_entry(&config, DEFAULT_CLAUDE_SERVER_NAME, &test_entry()).unwrap();

        let written: Value = serde_json::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert_eq!(
            written["mcpServers"]["other"]["command"].as_str(),
            Some("other")
        );
        assert!(written["mcpServers"][DEFAULT_CLAUDE_SERVER_NAME].is_object());
    }

    #[test]
    fn upsert_rejects_unmanaged_collision() {
        let dir = tempdir().unwrap();
        let config = dir.path().join(".claude.json");
        let initial = json!({
            "mcpServers": {
                "ice-shared-continuity-kernel": { "command": "other", "args": [] }
            }
        });
        fs::write(&config, serde_json::to_string_pretty(&initial).unwrap()).unwrap();

        let err =
            upsert_code_entry(&config, DEFAULT_CLAUDE_SERVER_NAME, &test_entry()).unwrap_err();
        assert!(
            err.to_string()
                .contains("already contains unmanaged mcpServers.ice-shared-continuity-kernel")
        );
    }

    #[test]
    fn remove_only_removes_managed_entry() {
        let dir = tempdir().unwrap();
        let config = dir.path().join(".claude.json");
        let initial = json!({
            "mcpServers": {
                "ice-shared-continuity-kernel": {
                    "command": "/tmp/ice",
                    "args": [],
                    "_ice_managed": true
                },
                "other": { "command": "other", "args": [] }
            }
        });
        fs::write(&config, serde_json::to_string_pretty(&initial).unwrap()).unwrap();

        let (changed, skipped) = remove_managed_entry(&config, DEFAULT_CLAUDE_SERVER_NAME).unwrap();
        assert!(changed);
        assert!(!skipped);

        let written: Value = serde_json::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert!(
            written["mcpServers"]
                .get(DEFAULT_CLAUDE_SERVER_NAME)
                .is_none()
        );
        assert_eq!(
            written["mcpServers"]["other"]["command"].as_str(),
            Some("other")
        );
    }

    #[test]
    fn malformed_config_is_skipped_without_overwrite() {
        let dir = tempdir().unwrap();
        let config = dir.path().join(".claude.json");
        fs::write(&config, "not-json").unwrap();

        let (changed, skipped) =
            upsert_code_entry(&config, DEFAULT_CLAUDE_SERVER_NAME, &test_entry()).unwrap();
        assert!(!changed);
        assert!(skipped);
        assert_eq!(fs::read_to_string(&config).unwrap(), "not-json");
    }

    #[test]
    fn validate_server_name_rejects_empty() {
        assert!(validate_server_name("").is_err());
    }

    #[test]
    fn validate_server_name_rejects_special_chars() {
        assert!(validate_server_name("name with spaces").is_err());
        assert!(validate_server_name("name/slash").is_err());
        assert!(validate_server_name("name.dot").is_err());
    }

    #[test]
    fn validate_server_name_accepts_valid() {
        assert!(validate_server_name("ice-shared-continuity-kernel").is_ok());
        assert!(validate_server_name("my_server_123").is_ok());
        assert!(validate_server_name("ABC").is_ok());
    }

    #[test]
    fn remove_managed_entry_nonexistent_file() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("does-not-exist.json");
        let (changed, skipped) = remove_managed_entry(&config, DEFAULT_CLAUDE_SERVER_NAME).unwrap();
        assert!(!changed);
        assert!(!skipped);
    }

    #[test]
    fn remove_managed_entry_no_mcp_servers_key() {
        let dir = tempdir().unwrap();
        let config = dir.path().join(".claude.json");
        fs::write(&config, r#"{"other": "data"}"#).unwrap();
        let (changed, skipped) = remove_managed_entry(&config, DEFAULT_CLAUDE_SERVER_NAME).unwrap();
        assert!(!changed);
        assert!(!skipped);
    }

    #[test]
    fn remove_unmanaged_entry_is_not_removed() {
        let dir = tempdir().unwrap();
        let config = dir.path().join(".claude.json");
        let initial = json!({
            "mcpServers": {
                "ice-shared-continuity-kernel": {
                    "command": "other",
                    "args": []
                }
            }
        });
        fs::write(&config, serde_json::to_string_pretty(&initial).unwrap()).unwrap();
        let (changed, skipped) = remove_managed_entry(&config, DEFAULT_CLAUDE_SERVER_NAME).unwrap();
        assert!(!changed);
        assert!(!skipped);
    }

    #[test]
    fn has_ice_entry_returns_true_when_present() {
        let dir = tempdir().unwrap();
        let config = dir.path().join(".claude.json");
        upsert_code_entry(&config, DEFAULT_CLAUDE_SERVER_NAME, &test_entry()).unwrap();
        assert!(has_ice_entry(&config, DEFAULT_CLAUDE_SERVER_NAME));
    }

    #[test]
    fn has_ice_entry_returns_false_when_absent() {
        let dir = tempdir().unwrap();
        let config = dir.path().join(".claude.json");
        fs::write(&config, r#"{"mcpServers": {}}"#).unwrap();
        assert!(!has_ice_entry(&config, DEFAULT_CLAUDE_SERVER_NAME));
    }

    #[test]
    fn has_ice_entry_returns_false_for_malformed_config() {
        let dir = tempdir().unwrap();
        let config = dir.path().join(".claude.json");
        fs::write(&config, "not json at all").unwrap();
        assert!(!has_ice_entry(&config, DEFAULT_CLAUDE_SERVER_NAME));
    }

    #[test]
    fn has_ice_entry_returns_false_for_missing_file() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("nonexistent.json");
        assert!(!has_ice_entry(&config, DEFAULT_CLAUDE_SERVER_NAME));
    }

    #[test]
    fn build_server_entry_contains_expected_fields() {
        let entry = build_server_entry(
            &PathBuf::from("/usr/bin/ice"),
            &["--root".into(), "/data".into(), "mcp".into()],
        );
        assert_eq!(entry["type"], "stdio");
        assert_eq!(entry["command"], "/usr/bin/ice");
        assert_eq!(entry["args"][0], "--root");
        assert_eq!(entry["_ice_managed"], true);
        assert!(entry["_ice_version"].as_str().is_some());
    }

    #[test]
    fn read_config_returns_empty_for_nonexistent_file() {
        let (config, malformed) = read_config(Path::new("/nonexistent/path/.claude.json"));
        assert_eq!(config, json!({}));
        assert!(!malformed);
    }

    #[test]
    fn read_config_returns_malformed_for_non_object_json() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("array.json");
        fs::write(&config, "[1, 2, 3]").unwrap();
        let (_, malformed) = read_config(&config);
        assert!(malformed);
    }

    #[test]
    fn resolve_absolute_handles_absolute_path() {
        let result = resolve_absolute("/absolute/path").unwrap();
        assert_eq!(result, PathBuf::from("/absolute/path"));
    }

    #[test]
    fn upsert_updates_managed_entry_when_different() {
        let dir = tempdir().unwrap();
        let config = dir.path().join(".claude.json");

        let entry1 = build_server_entry(
            &PathBuf::from("/tmp/ice-v1"),
            &["--root".into(), "/tmp/organism".into(), "mcp".into()],
        );
        let entry2 = build_server_entry(
            &PathBuf::from("/tmp/ice-v2"),
            &["--root".into(), "/tmp/organism".into(), "mcp".into()],
        );

        let (changed1, _) =
            upsert_code_entry(&config, DEFAULT_CLAUDE_SERVER_NAME, &entry1).unwrap();
        assert!(changed1);

        let (changed2, _) =
            upsert_code_entry(&config, DEFAULT_CLAUDE_SERVER_NAME, &entry2).unwrap();
        assert!(changed2);

        let written: Value = serde_json::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert_eq!(
            written["mcpServers"][DEFAULT_CLAUDE_SERVER_NAME]["command"],
            "/tmp/ice-v2"
        );
    }

    #[test]
    fn remove_malformed_config_skips() {
        let dir = tempdir().unwrap();
        let config = dir.path().join(".claude.json");
        fs::write(&config, "not-json").unwrap();
        let (changed, skipped) = remove_managed_entry(&config, DEFAULT_CLAUDE_SERVER_NAME).unwrap();
        assert!(!changed);
        assert!(skipped);
    }
}
