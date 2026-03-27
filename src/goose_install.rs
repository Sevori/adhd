use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;
use serde_yaml::{Mapping, Value};

use crate::config::EngineConfig;
use crate::managed_clients::{
    ICE_VERSION, has_yaml_entry, remove_yaml_managed_entry, resolve_absolute, resolve_binary,
    resolve_home, upsert_yaml_entry, validate_server_name,
};

pub const DEFAULT_GOOSE_SERVER_NAME: &str = "ice-shared-continuity-kernel";
pub const DEFAULT_GOOSE_MACHINE_ROOT_DIR: &str = ".config/goose/organisms/ice";
const DEFAULT_GOOSE_TIMEOUT_SECS: u64 = 300;

#[derive(Debug, Clone)]
pub struct GooseInstallRequest {
    pub organism_root: Option<PathBuf>,
    pub server_name: String,
    pub config_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GooseInstallResult {
    pub changed: bool,
    pub config_path: String,
    pub organism_root: String,
    pub server_name: String,
    pub command: String,
    pub args: Vec<String>,
    pub restart_required: bool,
    pub skipped_malformed: bool,
}

#[derive(Debug, Clone)]
pub struct GooseUninstallRequest {
    pub server_name: String,
    pub config_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GooseUninstallResult {
    pub changed: bool,
    pub config_path: String,
    pub skipped_malformed: bool,
}

#[derive(Debug, Clone)]
pub struct GooseStatusRequest {
    pub server_name: String,
    pub config_path: Option<PathBuf>,
    pub organism_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GooseStatusResult {
    pub binary_path: String,
    pub binary_exists: bool,
    pub organism_root: String,
    pub organism_root_exists: bool,
    pub config_path: String,
    pub config_found: bool,
    pub has_ice_entry: bool,
    pub ice_version: String,
}

pub fn install_goose(request: GooseInstallRequest) -> Result<GooseInstallResult> {
    validate_server_name(&request.server_name)?;

    let binary = resolve_binary()?;
    let organism_root = resolve_organism_root(request.organism_root.as_ref())?;
    let config_path = resolve_config(request.config_path.as_deref())?;

    EngineConfig::with_root(&organism_root)
        .ensure_dirs()
        .with_context(|| format!("ensuring organism root {}", organism_root.display()))?;

    let args = vec![
        "--root".to_string(),
        organism_root.display().to_string(),
        "mcp".to_string(),
    ];
    let entry = build_server_entry(&request.server_name, &binary.display().to_string(), &args);
    let update = upsert_yaml_entry(
        &config_path,
        "extensions",
        &request.server_name,
        &entry,
        "Goose",
    );
    let (changed, skipped_malformed) = update?;

    Ok(GooseInstallResult {
        changed,
        config_path: config_path.display().to_string(),
        organism_root: organism_root.display().to_string(),
        server_name: request.server_name,
        command: binary.display().to_string(),
        args,
        restart_required: true,
        skipped_malformed,
    })
}

pub fn uninstall_goose(request: GooseUninstallRequest) -> Result<GooseUninstallResult> {
    validate_server_name(&request.server_name)?;

    let config_path = resolve_config(request.config_path.as_deref())?;
    let (changed, skipped_malformed) =
        remove_yaml_managed_entry(&config_path, "extensions", &request.server_name, "Goose")?;

    Ok(GooseUninstallResult {
        changed,
        config_path: config_path.display().to_string(),
        skipped_malformed,
    })
}

pub fn goose_status(request: GooseStatusRequest) -> Result<GooseStatusResult> {
    validate_server_name(&request.server_name)?;

    let binary = resolve_binary()?;
    let organism_root = resolve_organism_root(request.organism_root.as_ref())?;
    let config_path = resolve_config(request.config_path.as_deref())?;

    Ok(GooseStatusResult {
        binary_path: binary.display().to_string(),
        binary_exists: binary.exists(),
        organism_root: organism_root.display().to_string(),
        organism_root_exists: organism_root.exists(),
        config_path: config_path.display().to_string(),
        config_found: config_path.exists(),
        has_ice_entry: has_yaml_entry(&config_path, "extensions", &request.server_name, "Goose"),
        ice_version: ICE_VERSION.to_string(),
    })
}

fn resolve_organism_root(requested: Option<&PathBuf>) -> Result<PathBuf> {
    if let Some(path) = requested {
        return resolve_absolute(path.as_path());
    }
    Ok(resolve_home("Goose")?.join(DEFAULT_GOOSE_MACHINE_ROOT_DIR))
}

fn resolve_config(override_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        return resolve_absolute(path);
    }
    Ok(resolve_home("Goose")?.join(".config/goose/config.yaml"))
}

fn build_server_entry(server_name: &str, command: &str, args: &[String]) -> Value {
    let mut mapping = Mapping::new();
    mapping.insert(Value::String("enabled".to_string()), Value::Bool(true));
    mapping.insert(
        Value::String("type".to_string()),
        Value::String("stdio".to_string()),
    );
    mapping.insert(
        Value::String("name".to_string()),
        Value::String(server_name.to_string()),
    );
    mapping.insert(
        Value::String("description".to_string()),
        Value::String("ICE shared continuity kernel".to_string()),
    );
    mapping.insert(
        Value::String("cmd".to_string()),
        Value::String(command.to_string()),
    );
    mapping.insert(
        Value::String("args".to_string()),
        Value::Sequence(args.iter().cloned().map(Value::String).collect()),
    );
    mapping.insert(
        Value::String("envs".to_string()),
        Value::Mapping(Mapping::new()),
    );
    mapping.insert(
        Value::String("env_keys".to_string()),
        Value::Sequence(Vec::new()),
    );
    mapping.insert(
        Value::String("timeout".to_string()),
        Value::Number(DEFAULT_GOOSE_TIMEOUT_SECS.into()),
    );
    mapping.insert(Value::String("bundled".to_string()), Value::Bool(false));
    mapping.insert(
        Value::String("available_tools".to_string()),
        Value::Sequence(Vec::new()),
    );
    mapping.insert(Value::String("_ice_managed".to_string()), Value::Bool(true));
    mapping.insert(
        Value::String("_ice_version".to_string()),
        Value::String(ICE_VERSION.to_string()),
    );
    Value::Mapping(mapping)
}

#[cfg(test)]
mod tests {
    use std::{ffi::OsString, fs};

    use super::*;
    use tempfile::tempdir;

    fn test_entry() -> Value {
        build_server_entry(
            DEFAULT_GOOSE_SERVER_NAME,
            "/tmp/ice",
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
        let config = dir.path().join("config.yaml");

        let (changed, skipped) = upsert_yaml_entry(
            &config,
            "extensions",
            DEFAULT_GOOSE_SERVER_NAME,
            &test_entry(),
            "Goose",
        )
        .unwrap();
        assert!(changed);
        assert!(!skipped);

        let written: Value = serde_yaml::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert!(
            written["extensions"][DEFAULT_GOOSE_SERVER_NAME]["_ice_managed"]
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            written["extensions"][DEFAULT_GOOSE_SERVER_NAME]["type"].as_str(),
            Some("stdio")
        );
    }

    #[test]
    fn upsert_is_idempotent_for_same_managed_entry() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("config.yaml");

        let first = upsert_yaml_entry(
            &config,
            "extensions",
            DEFAULT_GOOSE_SERVER_NAME,
            &test_entry(),
            "Goose",
        )
        .unwrap();
        let second = upsert_yaml_entry(
            &config,
            "extensions",
            DEFAULT_GOOSE_SERVER_NAME,
            &test_entry(),
            "Goose",
        )
        .unwrap();
        assert_eq!(first, (true, false));
        assert_eq!(second, (false, false));
    }

    #[test]
    fn upsert_preserves_other_extensions() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("config.yaml");
        fs::write(
            &config,
            r#"
extensions:
  other:
    enabled: true
    type: builtin
    name: developer
"#,
        )
        .unwrap();

        upsert_yaml_entry(
            &config,
            "extensions",
            DEFAULT_GOOSE_SERVER_NAME,
            &test_entry(),
            "Goose",
        )
        .unwrap();

        let written: Value = serde_yaml::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert_eq!(
            written["extensions"]["other"]["type"].as_str(),
            Some("builtin")
        );
        assert!(written["extensions"][DEFAULT_GOOSE_SERVER_NAME].is_mapping());
    }

    #[test]
    fn upsert_rejects_unmanaged_collision() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("config.yaml");
        fs::write(
            &config,
            r#"
extensions:
  ice-shared-continuity-kernel:
    enabled: true
    type: stdio
    name: ice-shared-continuity-kernel
    cmd: other
    args: []
"#,
        )
        .unwrap();

        let err = upsert_yaml_entry(
            &config,
            "extensions",
            DEFAULT_GOOSE_SERVER_NAME,
            &test_entry(),
            "Goose",
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("already contains unmanaged extensions.ice-shared-continuity-kernel")
        );
    }

    #[test]
    fn remove_only_removes_managed_entry() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("config.yaml");
        fs::write(
            &config,
            r#"
extensions:
  ice-shared-continuity-kernel:
    enabled: true
    type: stdio
    name: ice-shared-continuity-kernel
    cmd: /tmp/ice
    args: []
    _ice_managed: true
  other:
    enabled: true
    type: builtin
    name: developer
"#,
        )
        .unwrap();

        let (changed, skipped) =
            remove_yaml_managed_entry(&config, "extensions", DEFAULT_GOOSE_SERVER_NAME, "Goose")
                .unwrap();
        assert!(changed);
        assert!(!skipped);

        let written: Value = serde_yaml::from_str(&fs::read_to_string(&config).unwrap()).unwrap();
        assert!(
            written["extensions"]
                .get(DEFAULT_GOOSE_SERVER_NAME)
                .is_none()
        );
        assert_eq!(
            written["extensions"]["other"]["type"].as_str(),
            Some("builtin")
        );
    }

    #[test]
    fn malformed_config_is_skipped_without_overwrite() {
        let dir = tempdir().unwrap();
        let config = dir.path().join("config.yaml");
        fs::write(&config, "extensions: [").unwrap();

        let (changed, skipped) = upsert_yaml_entry(
            &config,
            "extensions",
            DEFAULT_GOOSE_SERVER_NAME,
            &test_entry(),
            "Goose",
        )
        .unwrap();
        assert!(!changed);
        assert!(skipped);
        assert_eq!(fs::read_to_string(&config).unwrap(), "extensions: [");
    }

    #[test]
    fn install_status_uninstall_roundtrip_with_explicit_paths() {
        let dir = tempdir().unwrap();
        let organism_root = dir.path().join("organism");
        let config = dir.path().join("config.yaml");

        let install = install_goose(GooseInstallRequest {
            organism_root: Some(organism_root.clone()),
            server_name: DEFAULT_GOOSE_SERVER_NAME.to_string(),
            config_path: Some(config.clone()),
        })
        .unwrap();
        assert!(install.changed);

        let status = goose_status(GooseStatusRequest {
            server_name: DEFAULT_GOOSE_SERVER_NAME.to_string(),
            config_path: Some(config.clone()),
            organism_root: Some(organism_root.clone()),
        })
        .unwrap();
        assert!(status.binary_exists);
        assert!(status.organism_root_exists);
        assert!(status.config_found);
        assert!(status.has_ice_entry);

        let uninstall = uninstall_goose(GooseUninstallRequest {
            server_name: DEFAULT_GOOSE_SERVER_NAME.to_string(),
            config_path: Some(config.clone()),
        })
        .unwrap();
        assert!(uninstall.changed);

        let status_after = goose_status(GooseStatusRequest {
            server_name: DEFAULT_GOOSE_SERVER_NAME.to_string(),
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
        let config = dir.path().join("config.yaml");

        let status = goose_status(GooseStatusRequest {
            server_name: DEFAULT_GOOSE_SERVER_NAME.to_string(),
            config_path: Some(config.clone()),
            organism_root: Some(organism_root),
        })
        .unwrap();
        assert!(!status.config_found);
        assert!(!status.has_ice_entry);

        let uninstall = uninstall_goose(GooseUninstallRequest {
            server_name: DEFAULT_GOOSE_SERVER_NAME.to_string(),
            config_path: Some(config),
        })
        .unwrap();
        assert!(!uninstall.changed);
    }

    #[test]
    fn install_rejects_invalid_server_name() {
        let dir = tempdir().unwrap();
        let err = install_goose(GooseInstallRequest {
            organism_root: Some(dir.path().join("organism")),
            server_name: "bad name".to_string(),
            config_path: Some(dir.path().join("config.yaml")),
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

        let install = install_goose(GooseInstallRequest {
            organism_root: None,
            server_name: DEFAULT_GOOSE_SERVER_NAME.to_string(),
            config_path: None,
        })
        .unwrap();
        assert!(
            install
                .organism_root
                .ends_with(".config/goose/organisms/ice")
        );
        assert!(install.config_path.ends_with(".config/goose/config.yaml"));

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

        let install = install_goose(GooseInstallRequest {
            organism_root: None,
            server_name: DEFAULT_GOOSE_SERVER_NAME.to_string(),
            config_path: None,
        })
        .unwrap();
        assert!(
            install
                .organism_root
                .ends_with(".config/goose/organisms/ice")
        );
        assert!(install.config_path.ends_with(".config/goose/config.yaml"));

        restore_home(previous_home);
        restore_home(original_home);
    }
}
