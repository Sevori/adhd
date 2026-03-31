use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

use crate::config::EngineConfig;

pub const DEFAULT_CODEX_GLOBAL_SERVER_NAME: &str = "adhd_machine";
pub const DEFAULT_CODEX_MACHINE_ROOT_DIR: &str = "organisms/ice";

const MANAGED_BLOCK_PREFIX: &str = "# BEGIN ICE MANAGED MCP SERVER ";
const MANAGED_BLOCK_SUFFIX: &str = "# END ICE MANAGED MCP SERVER ";

#[derive(Debug, Clone)]
pub struct CodexGlobalInstallRequest {
    pub codex_home: Option<PathBuf>,
    pub config_path: Option<PathBuf>,
    pub machine_root: Option<PathBuf>,
    pub server_name: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CodexGlobalInstallResult {
    pub changed: bool,
    pub codex_home: String,
    pub config_path: String,
    pub machine_root: String,
    pub server_name: String,
    pub command: String,
    pub args: Vec<String>,
    pub restart_required: bool,
}

pub fn install_global_mcp(request: CodexGlobalInstallRequest) -> Result<CodexGlobalInstallResult> {
    validate_server_name(&request.server_name)?;

    let config_path = request
        .config_path
        .as_ref()
        .map(resolve_absolute_path)
        .transpose()?;
    let codex_home = resolve_codex_home(request.codex_home.as_ref(), config_path.as_deref())?;
    let config_path = config_path.unwrap_or_else(|| codex_home.join("config.toml"));
    let machine_root = request
        .machine_root
        .as_ref()
        .map(resolve_absolute_path)
        .transpose()?
        .unwrap_or_else(|| codex_home.join(DEFAULT_CODEX_MACHINE_ROOT_DIR));
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let command = find_executable_on_path("cargo").unwrap_or_else(|| PathBuf::from("cargo"));
    let args = vec![
        "run".to_string(),
        "--quiet".to_string(),
        "--manifest-path".to_string(),
        manifest_path.display().to_string(),
        "--".to_string(),
        "--root".to_string(),
        machine_root.display().to_string(),
        "mcp".to_string(),
    ];

    fs::create_dir_all(&codex_home)
        .with_context(|| format!("creating Codex home {}", codex_home.display()))?;
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating config parent {}", parent.display()))?;
    }
    EngineConfig::with_root(&machine_root)
        .ensure_dirs()
        .with_context(|| format!("ensuring machine root {}", machine_root.display()))?;

    let existing = if config_path.exists() {
        fs::read_to_string(&config_path)
            .with_context(|| format!("reading Codex config {}", config_path.display()))?
    } else {
        String::new()
    };
    let block = render_managed_mcp_block(&request.server_name, &command, &args)?;
    let update = upsert_managed_mcp_block(&existing, &request.server_name, &block)?;
    if update.changed {
        fs::write(&config_path, update.content)
            .with_context(|| format!("writing Codex config {}", config_path.display()))?;
    }

    Ok(CodexGlobalInstallResult {
        changed: update.changed,
        codex_home: codex_home.display().to_string(),
        config_path: config_path.display().to_string(),
        machine_root: machine_root.display().to_string(),
        server_name: request.server_name,
        command: command.display().to_string(),
        args,
        restart_required: true,
    })
}

#[derive(Debug)]
struct ManagedBlockUpdate {
    content: String,
    changed: bool,
}

fn resolve_codex_home(
    requested_home: Option<&PathBuf>,
    config_path: Option<&Path>,
) -> Result<PathBuf> {
    if let Some(home) = requested_home {
        return resolve_absolute_path(home);
    }
    if let Some(config_path) = config_path
        && let Some(parent) = config_path.parent()
    {
        return Ok(parent.to_path_buf());
    }
    if let Some(value) = env::var_os("CODEX_HOME") {
        let path = PathBuf::from(value);
        if !path.as_os_str().is_empty() {
            return resolve_absolute_path(&path);
        }
    }
    let home =
        env::var_os("HOME").context("HOME or CODEX_HOME must be set to resolve Codex home")?;
    Ok(PathBuf::from(home).join(".codex"))
}

fn resolve_absolute_path(path: impl AsRef<Path>) -> Result<PathBuf> {
    let path = path.as_ref();
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    Ok(env::current_dir()
        .context("resolving current working directory")?
        .join(path))
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

fn find_executable_on_path(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

fn render_managed_mcp_block(server_name: &str, command: &Path, args: &[String]) -> Result<String> {
    let rendered_args = args
        .iter()
        .map(|value| toml_string(value))
        .collect::<Result<Vec<_>>>()?
        .join(", ");
    Ok(format!(
        "{}{}\n[mcp_servers.{}]\ncommand = {}\nargs = [{}]\n{}{}\n",
        MANAGED_BLOCK_PREFIX,
        server_name,
        server_name,
        toml_string(&command.display().to_string())?,
        rendered_args,
        MANAGED_BLOCK_SUFFIX,
        server_name
    ))
}

fn toml_string(value: &str) -> Result<String> {
    serde_json::to_string(value).context("encoding TOML string")
}

fn upsert_managed_mcp_block(
    existing: &str,
    server_name: &str,
    block: &str,
) -> Result<ManagedBlockUpdate> {
    let begin_marker = managed_begin_marker(server_name);
    let end_marker = managed_end_marker(server_name);
    let managed_range = managed_block_range(existing, &begin_marker, &end_marker)?;
    let unmanaged = match managed_range {
        Some((start, end)) => {
            let mut stripped = String::with_capacity(existing.len() - (end - start));
            stripped.push_str(&existing[..start]);
            stripped.push_str(&existing[end..]);
            stripped
        }
        None => existing.to_string(),
    };
    ensure_no_unmanaged_server_block(&unmanaged, server_name)?;
    let normalized_block = normalize_managed_block(block);
    let next = match managed_range {
        Some((start, end)) => {
            let mut updated = String::with_capacity(existing.len() + normalized_block.len());
            updated.push_str(&existing[..start]);
            updated.push_str(&normalized_block);
            updated.push_str(&existing[end..]);
            updated
        }
        None => append_managed_block(existing, &normalized_block),
    };
    Ok(ManagedBlockUpdate {
        changed: next != existing,
        content: next,
    })
}

fn managed_begin_marker(server_name: &str) -> String {
    format!("{MANAGED_BLOCK_PREFIX}{server_name}")
}

fn managed_end_marker(server_name: &str) -> String {
    format!("{MANAGED_BLOCK_SUFFIX}{server_name}")
}

fn managed_block_range(
    existing: &str,
    begin_marker: &str,
    end_marker: &str,
) -> Result<Option<(usize, usize)>> {
    let begin = existing.find(begin_marker);
    let end = existing.find(end_marker);
    match (begin, end) {
        (None, None) => Ok(None),
        (Some(_), None) | (None, Some(_)) => anyhow::bail!(
            "Codex config contains an incomplete managed ICE MCP block; repair or remove it first"
        ),
        (Some(begin), Some(end)) if end < begin => {
            anyhow::bail!("Codex config contains a malformed managed ICE MCP block ordering")
        }
        (Some(begin), Some(end)) => {
            let end_line = existing[end..]
                .find('\n')
                .map(|offset| end + offset + 1)
                .unwrap_or(existing.len());
            Ok(Some((begin, end_line)))
        }
    }
}

fn ensure_no_unmanaged_server_block(existing: &str, server_name: &str) -> Result<()> {
    let header = format!("[mcp_servers.{server_name}]");
    if existing.lines().any(|line| line.trim() == header) {
        anyhow::bail!(
            "Codex config already contains unmanaged {header}; rename it or remove it before installing the managed ICE block"
        );
    }
    Ok(())
}

fn normalize_managed_block(block: &str) -> String {
    let trimmed = block.trim_matches('\n');
    format!("{trimmed}\n")
}

fn append_managed_block(existing: &str, block: &str) -> String {
    if existing.trim().is_empty() {
        return block.to_string();
    }
    let trimmed = existing.trim_end_matches('\n');
    format!("{trimmed}\n\n{block}")
}

#[cfg(test)]
mod tests {
    use super::{
        CodexGlobalInstallRequest, DEFAULT_CODEX_GLOBAL_SERVER_NAME, append_managed_block,
        install_global_mcp, managed_begin_marker, managed_end_marker, render_managed_mcp_block,
        upsert_managed_mcp_block,
    };
    use tempfile::tempdir;

    #[test]
    fn append_managed_block_separates_existing_config_cleanly() {
        let appended = append_managed_block("model = \"gpt-5.4\"\n", "# block\n");
        assert_eq!(appended, "model = \"gpt-5.4\"\n\n# block\n");
    }

    #[test]
    fn managed_block_upsert_is_idempotent() {
        let block = render_managed_mcp_block(
            DEFAULT_CODEX_GLOBAL_SERVER_NAME,
            "/opt/homebrew/bin/cargo".as_ref(),
            &[
                "run".to_string(),
                "--quiet".to_string(),
                "--manifest-path".to_string(),
                "/tmp/adhd/Cargo.toml".to_string(),
                "--".to_string(),
                "--root".to_string(),
                "/tmp/.codex/organisms/ice".to_string(),
                "mcp".to_string(),
            ],
        )
        .expect("render managed block");
        let first = upsert_managed_mcp_block("model = \"gpt-5.4\"\n", "adhd_machine", &block)
            .expect("append managed block");
        assert!(first.changed);
        let second = upsert_managed_mcp_block(&first.content, "adhd_machine", &block)
            .expect("replace managed block idempotently");
        assert!(!second.changed);
        assert_eq!(first.content, second.content);
        assert!(
            second
                .content
                .contains(&managed_begin_marker("adhd_machine"))
        );
        assert!(second.content.contains(&managed_end_marker("adhd_machine")));
    }

    #[test]
    fn managed_block_upsert_rejects_unmanaged_duplicate_server() {
        let block = render_managed_mcp_block(
            DEFAULT_CODEX_GLOBAL_SERVER_NAME,
            "/opt/homebrew/bin/cargo".as_ref(),
            &[
                "run".to_string(),
                "--quiet".to_string(),
                "--manifest-path".to_string(),
                "/tmp/adhd/Cargo.toml".to_string(),
                "--".to_string(),
                "--root".to_string(),
                "/tmp/.codex/organisms/ice".to_string(),
                "mcp".to_string(),
            ],
        )
        .expect("render managed block");
        let err = upsert_managed_mcp_block(
            "[mcp_servers.adhd_machine]\ncommand = \"cargo\"\nargs = [\"run\"]\n",
            "adhd_machine",
            &block,
        )
        .expect_err("reject unmanaged duplicate");
        assert!(err.to_string().contains("unmanaged"));
    }

    #[test]
    fn install_global_mcp_creates_machine_root_and_config() {
        let dir = tempdir().expect("create temp dir");
        let codex_home = dir.path().join(".codex");
        let machine_root = dir.path().join("brain");
        let config_path = codex_home.join("config.toml");

        let result = install_global_mcp(CodexGlobalInstallRequest {
            codex_home: Some(codex_home.clone()),
            config_path: Some(config_path.clone()),
            machine_root: Some(machine_root.clone()),
            server_name: DEFAULT_CODEX_GLOBAL_SERVER_NAME.to_string(),
        })
        .expect("install global mcp");

        assert!(result.changed);
        assert!(config_path.exists());
        assert!(machine_root.join("data").exists());
    }
}
