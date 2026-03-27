use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{Value, json};

use crate::config::EngineConfig;
use crate::managed_clients::{
    ICE_VERSION, has_json_entry, remove_json_managed_entry, resolve_absolute, resolve_binary,
    resolve_home, upsert_json_entry, validate_server_name,
};

pub const DEFAULT_OPENHANDS_SERVER_NAME: &str = "ice-shared-continuity-kernel";
pub const DEFAULT_OPENHANDS_MACHINE_ROOT_DIR: &str = ".openhands/organisms/ice";

#[derive(Debug, Clone)]
pub struct OpenHandsInstallRequest {
    pub organism_root: Option<PathBuf>,
    pub server_name: String,
    pub mcp_config_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct OpenHandsInstallResult {
    pub changed: bool,
    pub mcp_config_path: String,
    pub organism_root: String,
    pub server_name: String,
    pub command: String,
    pub args: Vec<String>,
    pub restart_required: bool,
    pub skipped_malformed: bool,
}

#[derive(Debug, Clone)]
pub struct OpenHandsUninstallRequest {
    pub server_name: String,
    pub mcp_config_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct OpenHandsUninstallResult {
    pub changed: bool,
    pub mcp_config_path: String,
    pub skipped_malformed: bool,
}

#[derive(Debug, Clone)]
pub struct OpenHandsStatusRequest {
    pub server_name: String,
    pub mcp_config_path: Option<PathBuf>,
    pub organism_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct OpenHandsStatusResult {
    pub binary_path: String,
    pub binary_exists: bool,
    pub organism_root: String,
    pub organism_root_exists: bool,
    pub mcp_config_path: String,
    pub mcp_config_found: bool,
    pub mcp_has_ice_entry: bool,
    pub ice_version: String,
}

pub fn install_openhands(request: OpenHandsInstallRequest) -> Result<OpenHandsInstallResult> {
    validate_server_name(&request.server_name)?;

    let binary = resolve_binary()?;
    let organism_root = resolve_organism_root(request.organism_root.as_ref())?;
    let mcp_config_path = resolve_mcp_config(request.mcp_config_path.as_deref())?;

    EngineConfig::with_root(&organism_root)
        .ensure_dirs()
        .with_context(|| format!("ensuring organism root {}", organism_root.display()))?;

    let args = vec![
        "--root".to_string(),
        organism_root.display().to_string(),
        "mcp".to_string(),
    ];
    let entry = build_server_entry(&binary, &args);
    let (changed, skipped_malformed) =
        upsert_mcp_entry(&mcp_config_path, &request.server_name, &entry)?;

    Ok(OpenHandsInstallResult {
        changed,
        mcp_config_path: mcp_config_path.display().to_string(),
        organism_root: organism_root.display().to_string(),
        server_name: request.server_name,
        command: binary.display().to_string(),
        args,
        restart_required: true,
        skipped_malformed,
    })
}

pub fn uninstall_openhands(request: OpenHandsUninstallRequest) -> Result<OpenHandsUninstallResult> {
    validate_server_name(&request.server_name)?;

    let mcp_config_path = resolve_mcp_config(request.mcp_config_path.as_deref())?;
    let (changed, skipped_malformed) =
        remove_managed_entry(&mcp_config_path, &request.server_name)?;

    Ok(OpenHandsUninstallResult {
        changed,
        mcp_config_path: mcp_config_path.display().to_string(),
        skipped_malformed,
    })
}

pub fn openhands_status(request: OpenHandsStatusRequest) -> Result<OpenHandsStatusResult> {
    validate_server_name(&request.server_name)?;

    let binary = resolve_binary()?;
    let organism_root = resolve_organism_root(request.organism_root.as_ref())?;
    let mcp_config_path = resolve_mcp_config(request.mcp_config_path.as_deref())?;

    Ok(OpenHandsStatusResult {
        binary_path: binary.display().to_string(),
        binary_exists: binary.exists(),
        organism_root: organism_root.display().to_string(),
        organism_root_exists: organism_root.exists(),
        mcp_config_path: mcp_config_path.display().to_string(),
        mcp_config_found: mcp_config_path.exists(),
        mcp_has_ice_entry: has_ice_entry(&mcp_config_path, &request.server_name),
        ice_version: ICE_VERSION.to_string(),
    })
}

fn resolve_organism_root(requested: Option<&PathBuf>) -> Result<PathBuf> {
    if let Some(path) = requested {
        return resolve_absolute(path.as_path());
    }
    Ok(resolve_home("OpenHands")?.join(DEFAULT_OPENHANDS_MACHINE_ROOT_DIR))
}

fn resolve_mcp_config(override_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        return resolve_absolute(path);
    }
    Ok(resolve_home("OpenHands")?.join(".openhands/mcp.json"))
}

fn build_server_entry(command: &Path, args: &[String]) -> Value {
    json!({
        "transport": "stdio",
        "command": command.display().to_string(),
        "args": args,
        "env": {},
        "enabled": true,
        "_ice_managed": true,
        "_ice_version": ICE_VERSION,
    })
}

fn upsert_mcp_entry(config_path: &Path, server_name: &str, entry: &Value) -> Result<(bool, bool)> {
    upsert_json_entry(config_path, "mcpServers", server_name, entry, "OpenHands")
}

fn remove_managed_entry(config_path: &Path, server_name: &str) -> Result<(bool, bool)> {
    remove_json_managed_entry(config_path, "mcpServers", server_name, "OpenHands")
}

fn has_ice_entry(config_path: &Path, server_name: &str) -> bool {
    has_json_entry(config_path, "mcpServers", server_name, "OpenHands")
}

#[cfg(test)]
mod tests {
    use std::{ffi::OsString, fs};

    use super::*;
    use tempfile::tempdir;

    fn test_entry() -> Value {
        build_server_entry(
            &PathBuf::from("/tmp/ice"),
            &["--root".into(), "/tmp/organism".into(), "mcp".into()],
        )
    }

    fn restore_home(home: Option<OsString>) {
        match home {
            Some(value) => unsafe { std::env::set_var("HOME", value) },
            None => unsafe { std::env::remove_var("HOME") },
        }
    }

    #[test]
    fn upsert_creates_managed_entry() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("mcp.json");

        let (changed, skipped) =
            upsert_mcp_entry(&config, DEFAULT_OPENHANDS_SERVER_NAME, &test_entry()).unwrap();
        assert!(changed);
        assert!(!skipped);

        let written: Value = serde_json::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert!(
            written["mcpServers"][DEFAULT_OPENHANDS_SERVER_NAME]["_ice_managed"]
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            written["mcpServers"][DEFAULT_OPENHANDS_SERVER_NAME]["transport"].as_str(),
            Some("stdio")
        );
    }

    #[test]
    fn upsert_is_idempotent_for_same_managed_entry() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("mcp.json");

        let first =
            upsert_mcp_entry(&config, DEFAULT_OPENHANDS_SERVER_NAME, &test_entry()).unwrap();
        let second =
            upsert_mcp_entry(&config, DEFAULT_OPENHANDS_SERVER_NAME, &test_entry()).unwrap();
        assert_eq!(first, (true, false));
        assert_eq!(second, (false, false));
    }

    #[test]
    fn upsert_preserves_other_servers() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("mcp.json");
        let initial = json!({
            "mcpServers": {
                "other": { "transport": "stdio", "command": "other", "args": [] }
            }
        });
        fs::write(&config, serde_json::to_string_pretty(&initial).unwrap()).unwrap();

        upsert_mcp_entry(&config, DEFAULT_OPENHANDS_SERVER_NAME, &test_entry()).unwrap();

        let written: Value = serde_json::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert_eq!(
            written["mcpServers"]["other"]["command"].as_str(),
            Some("other")
        );
        assert!(written["mcpServers"][DEFAULT_OPENHANDS_SERVER_NAME].is_object());
    }

    #[test]
    fn upsert_rejects_unmanaged_collision() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("mcp.json");
        let initial = json!({
            "mcpServers": {
                "ice-shared-continuity-kernel": { "transport": "stdio", "command": "other", "args": [] }
            }
        });
        fs::write(&config, serde_json::to_string_pretty(&initial).unwrap()).unwrap();

        let err =
            upsert_mcp_entry(&config, DEFAULT_OPENHANDS_SERVER_NAME, &test_entry()).unwrap_err();
        assert!(
            err.to_string()
                .contains("already contains unmanaged mcpServers.ice-shared-continuity-kernel")
        );
    }

    #[test]
    fn remove_only_removes_managed_entry() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("mcp.json");
        let initial = json!({
            "mcpServers": {
                "ice-shared-continuity-kernel": {
                    "transport": "stdio",
                    "command": "/tmp/ice",
                    "args": [],
                    "_ice_managed": true
                },
                "other": { "transport": "stdio", "command": "other", "args": [] }
            }
        });
        fs::write(&config, serde_json::to_string_pretty(&initial).unwrap()).unwrap();

        let (changed, skipped) =
            remove_managed_entry(&config, DEFAULT_OPENHANDS_SERVER_NAME).unwrap();
        assert!(changed);
        assert!(!skipped);

        let written: Value = serde_json::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert!(
            written["mcpServers"]
                .get(DEFAULT_OPENHANDS_SERVER_NAME)
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
        let config = dir.path().join("mcp.json");
        fs::write(&config, "not-json").unwrap();

        let (changed, skipped) =
            upsert_mcp_entry(&config, DEFAULT_OPENHANDS_SERVER_NAME, &test_entry()).unwrap();
        assert!(!changed);
        assert!(skipped);
        assert_eq!(fs::read_to_string(&config).unwrap(), "not-json");
    }

    #[test]
    fn install_status_uninstall_roundtrip_with_explicit_paths() {
        let dir = tempdir().unwrap();
        let organism_root = dir.path().join("organism");
        let config = dir.path().join("mcp.json");

        let install = install_openhands(OpenHandsInstallRequest {
            organism_root: Some(organism_root.clone()),
            server_name: DEFAULT_OPENHANDS_SERVER_NAME.to_string(),
            mcp_config_path: Some(config.clone()),
        })
        .unwrap();
        assert!(install.changed);

        let status = openhands_status(OpenHandsStatusRequest {
            server_name: DEFAULT_OPENHANDS_SERVER_NAME.to_string(),
            mcp_config_path: Some(config.clone()),
            organism_root: Some(organism_root.clone()),
        })
        .unwrap();
        assert!(status.binary_exists);
        assert!(status.organism_root_exists);
        assert!(status.mcp_config_found);
        assert!(status.mcp_has_ice_entry);

        let uninstall = uninstall_openhands(OpenHandsUninstallRequest {
            server_name: DEFAULT_OPENHANDS_SERVER_NAME.to_string(),
            mcp_config_path: Some(config.clone()),
        })
        .unwrap();
        assert!(uninstall.changed);

        let status_after = openhands_status(OpenHandsStatusRequest {
            server_name: DEFAULT_OPENHANDS_SERVER_NAME.to_string(),
            mcp_config_path: Some(config),
            organism_root: Some(organism_root),
        })
        .unwrap();
        assert!(!status_after.mcp_has_ice_entry);
    }

    #[test]
    fn status_works_before_install_and_uninstall_is_noop() {
        let dir = tempdir().unwrap();
        let organism_root = dir.path().join("organism");
        let config = dir.path().join("mcp.json");

        let status = openhands_status(OpenHandsStatusRequest {
            server_name: DEFAULT_OPENHANDS_SERVER_NAME.to_string(),
            mcp_config_path: Some(config.clone()),
            organism_root: Some(organism_root.clone()),
        })
        .unwrap();
        assert!(!status.organism_root_exists);
        assert!(!status.mcp_config_found);
        assert!(!status.mcp_has_ice_entry);

        let uninstall = uninstall_openhands(OpenHandsUninstallRequest {
            server_name: DEFAULT_OPENHANDS_SERVER_NAME.to_string(),
            mcp_config_path: Some(config),
        })
        .unwrap();
        assert!(!uninstall.changed);
    }

    #[test]
    fn install_rejects_invalid_server_name() {
        let dir = tempdir().unwrap();
        let err = install_openhands(OpenHandsInstallRequest {
            organism_root: Some(dir.path().join("organism")),
            server_name: "bad name".to_string(),
            mcp_config_path: Some(dir.path().join("mcp.json")),
        })
        .unwrap_err();
        assert!(err.to_string().contains("ASCII letters"));
    }

    #[test]
    fn install_uses_home_defaults_when_paths_are_omitted() {
        let _guard = crate::managed_clients::TEST_ENV_MUTEX.lock().unwrap();
        let dir = tempdir().unwrap();
        let previous_home = std::env::var_os("HOME");
        unsafe {
            std::env::set_var("HOME", dir.path());
        }

        let install = install_openhands(OpenHandsInstallRequest {
            organism_root: None,
            server_name: DEFAULT_OPENHANDS_SERVER_NAME.to_string(),
            mcp_config_path: None,
        })
        .unwrap();
        assert!(install.organism_root.ends_with(".openhands/organisms/ice"));
        assert!(install.mcp_config_path.ends_with(".openhands/mcp.json"));

        restore_home(previous_home);
    }

    #[test]
    fn install_uses_home_defaults_when_home_was_initially_missing() {
        let _guard = crate::managed_clients::TEST_ENV_MUTEX.lock().unwrap();
        let dir = tempdir().unwrap();
        let original_home = std::env::var_os("HOME");
        unsafe {
            std::env::remove_var("HOME");
        }
        let previous_home = std::env::var_os("HOME");
        assert!(previous_home.is_none());
        unsafe {
            std::env::set_var("HOME", dir.path());
        }

        let install = install_openhands(OpenHandsInstallRequest {
            organism_root: None,
            server_name: DEFAULT_OPENHANDS_SERVER_NAME.to_string(),
            mcp_config_path: None,
        })
        .unwrap();
        assert!(install.organism_root.ends_with(".openhands/organisms/ice"));
        assert!(install.mcp_config_path.ends_with(".openhands/mcp.json"));

        restore_home(previous_home);
        restore_home(original_home);
    }
}
