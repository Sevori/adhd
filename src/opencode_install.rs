use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{Value, json};

use crate::config::EngineConfig;
use crate::managed_clients::{
    ICE_VERSION, has_json_entry, remove_json_managed_entry, resolve_absolute, resolve_binary,
    resolve_home, upsert_json_entry, validate_server_name,
};

pub const DEFAULT_OPENCODE_SERVER_NAME: &str = "ice-shared-continuity-kernel";
pub const DEFAULT_OPENCODE_MACHINE_ROOT_DIR: &str = ".config/opencode/organisms/ice";

#[derive(Debug, Clone)]
pub struct OpenCodeInstallRequest {
    pub organism_root: Option<PathBuf>,
    pub server_name: String,
    pub config_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct OpenCodeInstallResult {
    pub changed: bool,
    pub config_path: String,
    pub organism_root: String,
    pub server_name: String,
    pub command: Vec<String>,
    pub restart_required: bool,
    pub skipped_malformed: bool,
}

#[derive(Debug, Clone)]
pub struct OpenCodeUninstallRequest {
    pub server_name: String,
    pub config_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct OpenCodeUninstallResult {
    pub changed: bool,
    pub config_path: String,
    pub skipped_malformed: bool,
}

#[derive(Debug, Clone)]
pub struct OpenCodeStatusRequest {
    pub server_name: String,
    pub config_path: Option<PathBuf>,
    pub organism_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct OpenCodeStatusResult {
    pub binary_path: String,
    pub binary_exists: bool,
    pub organism_root: String,
    pub organism_root_exists: bool,
    pub config_path: String,
    pub config_found: bool,
    pub has_ice_entry: bool,
    pub ice_version: String,
}

pub fn install_opencode(request: OpenCodeInstallRequest) -> Result<OpenCodeInstallResult> {
    validate_server_name(&request.server_name)?;

    let binary = resolve_binary()?;
    let organism_root = resolve_organism_root(request.organism_root.as_ref())?;
    let config_path = resolve_config(request.config_path.as_deref())?;

    EngineConfig::with_root(&organism_root)
        .ensure_dirs()
        .with_context(|| format!("ensuring organism root {}", organism_root.display()))?;

    let command = vec![
        binary.display().to_string(),
        "--root".to_string(),
        organism_root.display().to_string(),
        "mcp".to_string(),
    ];
    let entry = build_server_entry(&command);
    let update = upsert_json_entry(
        &config_path,
        "mcp",
        &request.server_name,
        &entry,
        "OpenCode",
    );
    let (changed, skipped_malformed) = update?;

    Ok(OpenCodeInstallResult {
        changed,
        config_path: config_path.display().to_string(),
        organism_root: organism_root.display().to_string(),
        server_name: request.server_name,
        command,
        restart_required: true,
        skipped_malformed,
    })
}

pub fn uninstall_opencode(request: OpenCodeUninstallRequest) -> Result<OpenCodeUninstallResult> {
    validate_server_name(&request.server_name)?;

    let config_path = resolve_config(request.config_path.as_deref())?;
    let (changed, skipped_malformed) =
        remove_json_managed_entry(&config_path, "mcp", &request.server_name, "OpenCode")?;

    Ok(OpenCodeUninstallResult {
        changed,
        config_path: config_path.display().to_string(),
        skipped_malformed,
    })
}

pub fn opencode_status(request: OpenCodeStatusRequest) -> Result<OpenCodeStatusResult> {
    validate_server_name(&request.server_name)?;

    let binary = resolve_binary()?;
    let organism_root = resolve_organism_root(request.organism_root.as_ref())?;
    let config_path = resolve_config(request.config_path.as_deref())?;

    Ok(OpenCodeStatusResult {
        binary_path: binary.display().to_string(),
        binary_exists: binary.exists(),
        organism_root: organism_root.display().to_string(),
        organism_root_exists: organism_root.exists(),
        config_path: config_path.display().to_string(),
        config_found: config_path.exists(),
        has_ice_entry: has_json_entry(&config_path, "mcp", &request.server_name, "OpenCode"),
        ice_version: ICE_VERSION.to_string(),
    })
}

fn resolve_organism_root(requested: Option<&PathBuf>) -> Result<PathBuf> {
    if let Some(path) = requested {
        return resolve_absolute(path.as_path());
    }
    Ok(resolve_home("OpenCode")?.join(DEFAULT_OPENCODE_MACHINE_ROOT_DIR))
}

fn resolve_config(override_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        return resolve_absolute(path);
    }
    Ok(resolve_home("OpenCode")?.join(".config/opencode/opencode.json"))
}

fn build_server_entry(command: &[String]) -> Value {
    json!({
        "type": "local",
        "command": command,
        "enabled": true,
        "environment": {},
        "_ice_managed": true,
        "_ice_version": ICE_VERSION,
    })
}

#[cfg(test)]
mod tests {
    use std::{ffi::OsString, fs};

    use super::*;
    use tempfile::tempdir;

    fn test_entry() -> Value {
        build_server_entry(&[
            "/tmp/ice".into(),
            "--root".into(),
            "/tmp/organism".into(),
            "mcp".into(),
        ])
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
        let config = dir.path().join("opencode.json");

        let (changed, skipped) = upsert_json_entry(
            &config,
            "mcp",
            DEFAULT_OPENCODE_SERVER_NAME,
            &test_entry(),
            "OpenCode",
        )
        .unwrap();
        assert!(changed);
        assert!(!skipped);

        let written: Value = serde_json::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert!(
            written["mcp"][DEFAULT_OPENCODE_SERVER_NAME]["_ice_managed"]
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            written["mcp"][DEFAULT_OPENCODE_SERVER_NAME]["type"].as_str(),
            Some("local")
        );
    }

    #[test]
    fn upsert_is_idempotent_for_same_managed_entry() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("opencode.json");

        let first = upsert_json_entry(
            &config,
            "mcp",
            DEFAULT_OPENCODE_SERVER_NAME,
            &test_entry(),
            "OpenCode",
        )
        .unwrap();
        let second = upsert_json_entry(
            &config,
            "mcp",
            DEFAULT_OPENCODE_SERVER_NAME,
            &test_entry(),
            "OpenCode",
        )
        .unwrap();
        assert_eq!(first, (true, false));
        assert_eq!(second, (false, false));
    }

    #[test]
    fn upsert_preserves_other_servers() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("opencode.json");
        let initial = json!({
            "mcp": {
                "other": {
                    "type": "local",
                    "command": ["other"],
                    "enabled": true
                }
            }
        });
        fs::write(&config, serde_json::to_string_pretty(&initial).unwrap()).unwrap();

        upsert_json_entry(
            &config,
            "mcp",
            DEFAULT_OPENCODE_SERVER_NAME,
            &test_entry(),
            "OpenCode",
        )
        .unwrap();

        let written: Value = serde_json::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert_eq!(
            written["mcp"]["other"]["command"][0].as_str(),
            Some("other")
        );
        assert!(written["mcp"][DEFAULT_OPENCODE_SERVER_NAME].is_object());
    }

    #[test]
    fn upsert_accepts_jsonc_and_rewrites_as_json() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("opencode.json");
        fs::write(
            &config,
            r#"
            {
              // existing config
              "model": "ollama/qwen2.5:14b",
              "mcp": {
                "other": {
                  "type": "local",
                  "command": ["other"],
                },
              },
            }
            "#,
        )
        .unwrap();

        let (changed, skipped) = upsert_json_entry(
            &config,
            "mcp",
            DEFAULT_OPENCODE_SERVER_NAME,
            &test_entry(),
            "OpenCode",
        )
        .unwrap();
        assert!(changed);
        assert!(!skipped);

        let written: Value = serde_json::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert_eq!(written["model"].as_str(), Some("ollama/qwen2.5:14b"));
        assert!(written["mcp"][DEFAULT_OPENCODE_SERVER_NAME].is_object());
    }

    #[test]
    fn upsert_rejects_unmanaged_collision() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("opencode.json");
        let initial = json!({
            "mcp": {
                "ice-shared-continuity-kernel": {
                    "type": "local",
                    "command": ["other"]
                }
            }
        });
        fs::write(&config, serde_json::to_string_pretty(&initial).unwrap()).unwrap();

        let err = upsert_json_entry(
            &config,
            "mcp",
            DEFAULT_OPENCODE_SERVER_NAME,
            &test_entry(),
            "OpenCode",
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("already contains unmanaged mcp.ice-shared-continuity-kernel")
        );
    }

    #[test]
    fn remove_only_removes_managed_entry() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("opencode.json");
        let initial = json!({
            "mcp": {
                "ice-shared-continuity-kernel": {
                    "type": "local",
                    "command": ["/tmp/ice"],
                    "_ice_managed": true
                },
                "other": {
                    "type": "local",
                    "command": ["other"]
                }
            }
        });
        fs::write(&config, serde_json::to_string_pretty(&initial).unwrap()).unwrap();

        let (changed, skipped) =
            remove_json_managed_entry(&config, "mcp", DEFAULT_OPENCODE_SERVER_NAME, "OpenCode")
                .unwrap();
        assert!(changed);
        assert!(!skipped);

        let written: Value = serde_json::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert!(written["mcp"].get(DEFAULT_OPENCODE_SERVER_NAME).is_none());
        assert_eq!(
            written["mcp"]["other"]["command"][0].as_str(),
            Some("other")
        );
    }

    #[test]
    fn malformed_config_is_skipped_without_overwrite() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("opencode.json");
        fs::write(&config, "not-json").unwrap();

        let (changed, skipped) = upsert_json_entry(
            &config,
            "mcp",
            DEFAULT_OPENCODE_SERVER_NAME,
            &test_entry(),
            "OpenCode",
        )
        .unwrap();
        assert!(!changed);
        assert!(skipped);
        assert_eq!(fs::read_to_string(&config).unwrap(), "not-json");
    }

    #[test]
    fn install_status_uninstall_roundtrip_with_explicit_paths() {
        let dir = tempdir().unwrap();
        let organism_root = dir.path().join("organism");
        let config = dir.path().join("opencode.json");

        let install = install_opencode(OpenCodeInstallRequest {
            organism_root: Some(organism_root.clone()),
            server_name: DEFAULT_OPENCODE_SERVER_NAME.to_string(),
            config_path: Some(config.clone()),
        })
        .unwrap();
        assert!(install.changed);

        let status = opencode_status(OpenCodeStatusRequest {
            server_name: DEFAULT_OPENCODE_SERVER_NAME.to_string(),
            config_path: Some(config.clone()),
            organism_root: Some(organism_root.clone()),
        })
        .unwrap();
        assert!(status.binary_exists);
        assert!(status.organism_root_exists);
        assert!(status.config_found);
        assert!(status.has_ice_entry);

        let uninstall = uninstall_opencode(OpenCodeUninstallRequest {
            server_name: DEFAULT_OPENCODE_SERVER_NAME.to_string(),
            config_path: Some(config.clone()),
        })
        .unwrap();
        assert!(uninstall.changed);

        let status_after = opencode_status(OpenCodeStatusRequest {
            server_name: DEFAULT_OPENCODE_SERVER_NAME.to_string(),
            config_path: Some(config),
            organism_root: Some(organism_root),
        })
        .unwrap();
        assert!(!status_after.has_ice_entry);
    }

    #[test]
    fn status_works_before_install_and_uninstall_is_noop() {
        let dir = tempdir().unwrap();
        let organism_root = dir.path().join("organism");
        let config = dir.path().join("opencode.json");

        let status = opencode_status(OpenCodeStatusRequest {
            server_name: DEFAULT_OPENCODE_SERVER_NAME.to_string(),
            config_path: Some(config.clone()),
            organism_root: Some(organism_root),
        })
        .unwrap();
        assert!(!status.config_found);
        assert!(!status.has_ice_entry);

        let uninstall = uninstall_opencode(OpenCodeUninstallRequest {
            server_name: DEFAULT_OPENCODE_SERVER_NAME.to_string(),
            config_path: Some(config),
        })
        .unwrap();
        assert!(!uninstall.changed);
    }

    #[test]
    fn install_rejects_invalid_server_name() {
        let dir = tempdir().unwrap();
        let err = install_opencode(OpenCodeInstallRequest {
            organism_root: Some(dir.path().join("organism")),
            server_name: "bad name".to_string(),
            config_path: Some(dir.path().join("opencode.json")),
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

        let install = install_opencode(OpenCodeInstallRequest {
            organism_root: None,
            server_name: DEFAULT_OPENCODE_SERVER_NAME.to_string(),
            config_path: None,
        })
        .unwrap();
        assert!(
            install
                .organism_root
                .ends_with(".config/opencode/organisms/ice")
        );
        assert!(
            install
                .config_path
                .ends_with(".config/opencode/opencode.json")
        );

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

        let install = install_opencode(OpenCodeInstallRequest {
            organism_root: None,
            server_name: DEFAULT_OPENCODE_SERVER_NAME.to_string(),
            config_path: None,
        })
        .unwrap();
        assert!(
            install
                .organism_root
                .ends_with(".config/opencode/organisms/ice")
        );
        assert!(
            install
                .config_path
                .ends_with(".config/opencode/opencode.json")
        );

        restore_home(previous_home);
        restore_home(original_home);
    }
}
