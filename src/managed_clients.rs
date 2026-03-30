use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::{Value as JsonValue, json};
use serde_yaml::{Mapping as YamlMapping, Value as YamlValue};

pub const ICE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
pub(crate) static TEST_ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn resolve_binary() -> Result<PathBuf> {
    env::current_exe().context("resolving current executable path")
}

pub fn resolve_home(client_name: &str) -> Result<PathBuf> {
    #[cfg(windows)]
    if let Some(val) = env::var_os("USERPROFILE") {
        return Ok(PathBuf::from(val));
    }
    if let Some(val) = env::var_os("HOME") {
        return Ok(PathBuf::from(val));
    }
    anyhow::bail!("HOME environment variable not set; cannot resolve {client_name} paths")
}

pub fn resolve_absolute(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    Ok(env::current_dir()
        .context("resolving current working directory")?
        .join(path))
}

pub fn validate_server_name(server_name: &str) -> Result<()> {
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

pub fn read_json_config(path: &Path, owner_label: &str) -> (JsonValue, bool) {
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
    match serde_json::from_str::<JsonValue>(&text).or_else(|_| json5::from_str::<JsonValue>(&text))
    {
        Ok(value) if value.is_object() => (value, false),
        _ => {
            eprintln!(
                "warning: {} contains malformed or non-object {} JSON; skipping",
                path.display(),
                owner_label
            );
            (json!({}), true)
        }
    }
}

pub fn write_json_config(path: &Path, value: &JsonValue, owner_label: &str) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating config directory {}", parent.display()))?;
    }
    let rendered = serde_json::to_string_pretty(value)
        .with_context(|| format!("rendering {owner_label} config JSON"))?;
    fs::write(path, rendered)
        .with_context(|| format!("writing {owner_label} config {}", path.display()))
}

pub fn upsert_json_entry(
    config_path: &Path,
    container_key: &str,
    server_name: &str,
    entry: &JsonValue,
    owner_label: &str,
) -> Result<(bool, bool)> {
    let (mut config, malformed) = read_json_config(config_path, owner_label);
    if malformed {
        return Ok((false, true));
    }

    let container = config
        .as_object_mut()
        .expect("config must be object")
        .entry(container_key)
        .or_insert_with(|| json!({}));
    let container_obj = container
        .as_object_mut()
        .with_context(|| format!("{owner_label} config `{container_key}` is not a JSON object"))?;

    let existing_matches = if let Some(existing) = container_obj.get(server_name) {
        let managed = existing
            .get("_ice_managed")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false);
        if !managed {
            anyhow::bail!(
                "{owner_label} config already contains unmanaged {container_key}.{server_name}; rename it or remove it before installing the managed ICE entry"
            );
        }
        existing == entry
    } else {
        false
    };

    if existing_matches {
        return Ok((false, false));
    }

    container_obj.insert(server_name.to_string(), entry.clone());
    write_json_config(config_path, &config, owner_label)?;
    Ok((true, false))
}

pub fn remove_json_managed_entry(
    config_path: &Path,
    container_key: &str,
    server_name: &str,
    owner_label: &str,
) -> Result<(bool, bool)> {
    if !config_path.exists() {
        return Ok((false, false));
    }

    let (mut config, malformed) = read_json_config(config_path, owner_label);
    if malformed {
        return Ok((false, true));
    }

    let changed = config
        .get_mut(container_key)
        .and_then(JsonValue::as_object_mut)
        .map(|container| {
            let managed = container
                .get(server_name)
                .and_then(|value| value.get("_ice_managed"))
                .and_then(JsonValue::as_bool)
                .unwrap_or(false);
            if managed {
                container.remove(server_name);
                true
            } else {
                false
            }
        })
        .unwrap_or(false);

    if changed {
        write_json_config(config_path, &config, owner_label)?;
    }

    Ok((changed, false))
}

pub fn has_json_entry(
    config_path: &Path,
    container_key: &str,
    server_name: &str,
    owner_label: &str,
) -> bool {
    let (config, malformed) = read_json_config(config_path, owner_label);
    if malformed {
        return false;
    }
    config
        .get(container_key)
        .and_then(|value| value.get(server_name))
        .is_some()
}

pub fn read_yaml_config(path: &Path, owner_label: &str) -> (YamlValue, bool) {
    if !path.exists() {
        return (YamlValue::Mapping(YamlMapping::new()), false);
    }
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!(
                "warning: could not read {}: {err}; skipping",
                path.display()
            );
            return (YamlValue::Mapping(YamlMapping::new()), true);
        }
    };
    match serde_yaml::from_str::<YamlValue>(&text) {
        Ok(value) if value.is_mapping() => (value, false),
        _ => {
            eprintln!(
                "warning: {} contains malformed or non-mapping {} YAML; skipping",
                path.display(),
                owner_label
            );
            (YamlValue::Mapping(YamlMapping::new()), true)
        }
    }
}

pub fn write_yaml_config(path: &Path, value: &YamlValue, owner_label: &str) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating config directory {}", parent.display()))?;
    }
    let rendered = serde_yaml::to_string(value)
        .with_context(|| format!("rendering {owner_label} config YAML"))?;
    fs::write(path, rendered)
        .with_context(|| format!("writing {owner_label} config {}", path.display()))
}

pub fn upsert_yaml_entry(
    config_path: &Path,
    container_key: &str,
    server_name: &str,
    entry: &YamlValue,
    owner_label: &str,
) -> Result<(bool, bool)> {
    let (mut config, malformed) = read_yaml_config(config_path, owner_label);
    if malformed {
        return Ok((false, true));
    }

    let container = config
        .as_mapping_mut()
        .expect("config must be mapping")
        .entry(YamlValue::String(container_key.to_string()))
        .or_insert_with(|| YamlValue::Mapping(YamlMapping::new()));
    let container_map = container
        .as_mapping_mut()
        .with_context(|| format!("{owner_label} config `{container_key}` is not a YAML mapping"))?;

    let server_key = YamlValue::String(server_name.to_string());
    let existing_matches = if let Some(existing) = container_map.get(&server_key) {
        let managed = existing
            .as_mapping()
            .and_then(|mapping| mapping.get(YamlValue::String("_ice_managed".to_string())))
            .and_then(YamlValue::as_bool)
            .unwrap_or(false);
        if !managed {
            anyhow::bail!(
                "{owner_label} config already contains unmanaged {container_key}.{server_name}; rename it or remove it before installing the managed ICE entry"
            );
        }
        existing == entry
    } else {
        false
    };

    if existing_matches {
        return Ok((false, false));
    }

    container_map.insert(server_key, entry.clone());
    write_yaml_config(config_path, &config, owner_label)?;
    Ok((true, false))
}

pub fn remove_yaml_managed_entry(
    config_path: &Path,
    container_key: &str,
    server_name: &str,
    owner_label: &str,
) -> Result<(bool, bool)> {
    if !config_path.exists() {
        return Ok((false, false));
    }

    let (mut config, malformed) = read_yaml_config(config_path, owner_label);
    if malformed {
        return Ok((false, true));
    }

    let changed = config
        .get_mut(container_key)
        .and_then(YamlValue::as_mapping_mut)
        .map(|container| {
            let server_key = YamlValue::String(server_name.to_string());
            let managed = container
                .get(&server_key)
                .and_then(YamlValue::as_mapping)
                .and_then(|mapping| mapping.get(YamlValue::String("_ice_managed".to_string())))
                .and_then(YamlValue::as_bool)
                .unwrap_or(false);
            if managed {
                container.remove(&server_key);
                true
            } else {
                false
            }
        })
        .unwrap_or(false);

    if changed {
        write_yaml_config(config_path, &config, owner_label)?;
    }

    Ok((changed, false))
}

pub fn has_yaml_entry(
    config_path: &Path,
    container_key: &str,
    server_name: &str,
    owner_label: &str,
) -> bool {
    let (config, malformed) = read_yaml_config(config_path, owner_label);
    if malformed {
        return false;
    }
    let server_key = YamlValue::String(server_name.to_string());
    config
        .get(container_key)
        .and_then(YamlValue::as_mapping)
        .and_then(|mapping| mapping.get(&server_key))
        .is_some()
}

#[cfg(test)]
mod tests {
    use std::{env, fs};
    use std::{ffi::OsString, path::Path};

    use super::*;

    fn json_entry() -> JsonValue {
        json!({
            "type": "local",
            "command": ["ice", "mcp"],
            "_ice_managed": true,
        })
    }

    fn yaml_entry() -> YamlValue {
        let mut entry = YamlMapping::new();
        entry.insert(
            YamlValue::String("type".to_string()),
            YamlValue::String("stdio".to_string()),
        );
        entry.insert(
            YamlValue::String("_ice_managed".to_string()),
            YamlValue::Bool(true),
        );
        YamlValue::Mapping(entry)
    }

    fn restore_home(home: Option<OsString>) {
        match home {
            Some(value) => unsafe { env::set_var("HOME", value) },
            None => unsafe { env::remove_var("HOME") },
        }
    }

    #[test]
    fn resolve_binary_returns_existing_executable() {
        let binary = resolve_binary().unwrap();
        assert!(binary.exists());
    }

    #[test]
    fn resolve_home_returns_absolute_path() {
        let _guard = TEST_ENV_MUTEX.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let previous_home = env::var_os("HOME");
        unsafe {
            env::set_var("HOME", dir.path());
        }

        let home = resolve_home("test-client").unwrap();
        assert!(home.is_absolute());
        assert_eq!(home, dir.path());

        restore_home(previous_home);
    }

    #[test]
    fn resolve_home_errors_without_home_env() {
        let _guard = TEST_ENV_MUTEX.lock().unwrap();
        let original_home = env::var_os("HOME");
        unsafe {
            env::remove_var("HOME");
        }
        let previous_home = env::var_os("HOME");
        assert!(previous_home.is_none());

        let err = resolve_home("test-client").unwrap_err();
        assert!(
            err.to_string()
                .contains("HOME environment variable not set")
        );

        restore_home(previous_home);
        restore_home(original_home);
    }

    #[test]
    fn resolve_absolute_expands_relative_paths() {
        let resolved = resolve_absolute(Path::new("relative/path")).unwrap();
        assert!(resolved.is_absolute());
        assert!(resolved.ends_with("relative/path"));
    }

    #[test]
    fn validate_server_name_rejects_invalid_characters() {
        let err = validate_server_name("bad name").unwrap_err();
        assert!(err.to_string().contains("ASCII letters"));
    }

    #[test]
    fn resolve_absolute_preserves_absolute_paths() {
        let absolute = PathBuf::from("/tmp/already-absolute");
        assert_eq!(resolve_absolute(absolute.as_path()).unwrap(), absolute);
    }

    #[test]
    fn validate_server_name_accepts_safe_names() {
        validate_server_name("ice-shared_continuity-kernel").unwrap();
    }

    #[test]
    fn json_helpers_roundtrip_and_preserve_unmanaged_entries() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        fs::write(
            &config,
            serde_json::to_string_pretty(&json!({
                "mcp": {
                    "other": { "type": "local", "command": ["other"] }
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let (changed, skipped) =
            upsert_json_entry(&config, "mcp", "ice", &json_entry(), "Test").unwrap();
        assert_eq!((changed, skipped), (true, false));
        assert!(has_json_entry(&config, "mcp", "ice", "Test"));

        let written = fs::read_to_string(&config).unwrap();
        assert!(written.contains("\"other\""));

        let (changed, skipped) = remove_json_managed_entry(&config, "mcp", "ice", "Test").unwrap();
        assert_eq!((changed, skipped), (true, false));
        assert!(!has_json_entry(&config, "mcp", "ice", "Test"));
    }

    #[test]
    fn json_helpers_skip_malformed_json5_or_non_object_files() {
        let dir = tempfile::tempdir().unwrap();
        let malformed = dir.path().join("bad.json");
        fs::write(&malformed, "[]").unwrap();
        assert_eq!(read_json_config(&malformed, "Test"), (json!({}), true));

        let jsonc = dir.path().join("jsonc.json");
        fs::write(
            &jsonc,
            r#"
            {
              // comment
              "mcp": {
                "other": { "type": "local", "command": ["other"] },
              },
            }
            "#,
        )
        .unwrap();
        let (parsed, skipped) = read_json_config(&jsonc, "Test");
        assert!(!skipped);
        assert_eq!(parsed["mcp"]["other"]["type"].as_str(), Some("local"));
    }

    #[cfg(unix)]
    #[test]
    fn json_helpers_skip_unreadable_files() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        fs::write(&config, "{}").unwrap();

        let mut unreadable = fs::metadata(&config).unwrap().permissions();
        unreadable.set_mode(0o000);
        fs::set_permissions(&config, unreadable).unwrap();

        assert_eq!(read_json_config(&config, "Test"), (json!({}), true));

        let mut readable = fs::metadata(&config).unwrap().permissions();
        readable.set_mode(0o600);
        fs::set_permissions(&config, readable).unwrap();
    }

    #[test]
    fn json_helpers_reject_unmanaged_collisions() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        fs::write(
            &config,
            serde_json::to_string_pretty(&json!({
                "mcp": {
                    "ice": { "type": "local", "command": ["other"] }
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let err = upsert_json_entry(&config, "mcp", "ice", &json_entry(), "Test").unwrap_err();
        assert!(err.to_string().contains("unmanaged mcp.ice"));
    }

    #[test]
    fn json_helpers_reject_non_object_container() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        fs::write(
            &config,
            serde_json::to_string_pretty(&json!({
                "mcp": []
            }))
            .unwrap(),
        )
        .unwrap();

        let err = upsert_json_entry(&config, "mcp", "ice", &json_entry(), "Test").unwrap_err();
        assert!(err.to_string().contains("`mcp` is not a JSON object"));
    }

    #[test]
    fn write_json_config_creates_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/config.json");
        write_json_config(&path, &json!({"ok": true}), "Test").unwrap();
        assert_eq!(
            serde_json::from_str::<JsonValue>(&fs::read_to_string(path).unwrap()).unwrap()["ok"]
                .as_bool(),
            Some(true)
        );
    }

    #[test]
    fn write_json_config_supports_relative_path_without_parent_directory() {
        let _guard = TEST_ENV_MUTEX.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let previous_dir = env::current_dir().unwrap();
        env::set_current_dir(dir.path()).unwrap();

        write_json_config(Path::new("config.json"), &json!({"ok": true}), "Test").unwrap();

        let written = fs::read_to_string(dir.path().join("config.json")).unwrap();
        assert_eq!(
            serde_json::from_str::<JsonValue>(&written).unwrap()["ok"].as_bool(),
            Some(true)
        );

        env::set_current_dir(previous_dir).unwrap();
    }

    #[test]
    fn remove_json_managed_entry_is_noop_for_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("missing.json");
        assert_eq!(
            remove_json_managed_entry(&config, "mcp", "ice", "Test").unwrap(),
            (false, false)
        );
        assert!(!has_json_entry(&config, "mcp", "ice", "Test"));
    }

    #[test]
    fn remove_json_managed_entry_skips_malformed_config() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("bad.json");
        fs::write(&config, "not-json").unwrap();

        assert_eq!(
            remove_json_managed_entry(&config, "mcp", "ice", "Test").unwrap(),
            (false, true)
        );
    }

    #[test]
    fn remove_json_managed_entry_leaves_unmanaged_entry() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        fs::write(
            &config,
            serde_json::to_string_pretty(&json!({
                "mcp": {
                    "ice": { "type": "local", "command": ["other"] }
                }
            }))
            .unwrap(),
        )
        .unwrap();

        assert_eq!(
            remove_json_managed_entry(&config, "mcp", "ice", "Test").unwrap(),
            (false, false)
        );
        assert!(has_json_entry(&config, "mcp", "ice", "Test"));
    }

    #[test]
    fn has_json_entry_returns_false_for_malformed_config() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("bad.json");
        fs::write(&config, "not-json").unwrap();

        assert!(!has_json_entry(&config, "mcp", "ice", "Test"));
    }

    #[test]
    fn yaml_helpers_roundtrip_and_preserve_unmanaged_entries() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.yaml");
        fs::write(
            &config,
            r#"
extensions:
  other:
    type: builtin
"#,
        )
        .unwrap();

        let (changed, skipped) =
            upsert_yaml_entry(&config, "extensions", "ice", &yaml_entry(), "Test").unwrap();
        assert_eq!((changed, skipped), (true, false));
        assert!(has_yaml_entry(&config, "extensions", "ice", "Test"));

        let written = fs::read_to_string(&config).unwrap();
        assert!(written.contains("other"));

        let (changed, skipped) =
            remove_yaml_managed_entry(&config, "extensions", "ice", "Test").unwrap();
        assert_eq!((changed, skipped), (true, false));
        assert!(!has_yaml_entry(&config, "extensions", "ice", "Test"));
    }

    #[test]
    fn yaml_helpers_skip_malformed_or_non_mapping_files() {
        let dir = tempfile::tempdir().unwrap();
        let malformed = dir.path().join("bad.yaml");
        fs::write(&malformed, "- just\n- a\n- list\n").unwrap();
        assert!(read_yaml_config(&malformed, "Test").1);
    }

    #[cfg(unix)]
    #[test]
    fn yaml_helpers_skip_unreadable_files() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.yaml");
        fs::write(&config, "extensions: {}\n").unwrap();

        let mut unreadable = fs::metadata(&config).unwrap().permissions();
        unreadable.set_mode(0o000);
        fs::set_permissions(&config, unreadable).unwrap();

        assert!(read_yaml_config(&config, "Test").1);

        let mut readable = fs::metadata(&config).unwrap().permissions();
        readable.set_mode(0o600);
        fs::set_permissions(&config, readable).unwrap();
    }

    #[test]
    fn yaml_helpers_reject_unmanaged_collisions() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.yaml");
        fs::write(
            &config,
            r#"
extensions:
  ice:
    type: stdio
"#,
        )
        .unwrap();

        let err =
            upsert_yaml_entry(&config, "extensions", "ice", &yaml_entry(), "Test").unwrap_err();
        assert!(err.to_string().contains("unmanaged extensions.ice"));
    }

    #[test]
    fn yaml_helpers_reject_non_mapping_container() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.yaml");
        fs::write(
            &config,
            r#"
extensions: []
"#,
        )
        .unwrap();

        let err =
            upsert_yaml_entry(&config, "extensions", "ice", &yaml_entry(), "Test").unwrap_err();
        assert!(
            err.to_string()
                .contains("`extensions` is not a YAML mapping")
        );
    }

    #[test]
    fn write_yaml_config_creates_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/config.yaml");
        write_yaml_config(
            &path,
            &YamlValue::Mapping(YamlMapping::from_iter([(
                YamlValue::String("ok".to_string()),
                YamlValue::Bool(true),
            )])),
            "Test",
        )
        .unwrap();
        let parsed: YamlValue = serde_yaml::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(parsed["ok"].as_bool(), Some(true));
    }

    #[test]
    fn write_yaml_config_supports_relative_path_without_parent_directory() {
        let _guard = TEST_ENV_MUTEX.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let previous_dir = env::current_dir().unwrap();
        env::set_current_dir(dir.path()).unwrap();

        write_yaml_config(
            Path::new("config.yaml"),
            &YamlValue::Mapping(YamlMapping::from_iter([(
                YamlValue::String("ok".to_string()),
                YamlValue::Bool(true),
            )])),
            "Test",
        )
        .unwrap();

        let written = fs::read_to_string(dir.path().join("config.yaml")).unwrap();
        let parsed: YamlValue = serde_yaml::from_str(&written).unwrap();
        assert_eq!(parsed["ok"].as_bool(), Some(true));

        env::set_current_dir(previous_dir).unwrap();
    }

    #[test]
    fn remove_yaml_managed_entry_is_noop_for_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("missing.yaml");
        assert_eq!(
            remove_yaml_managed_entry(&config, "extensions", "ice", "Test").unwrap(),
            (false, false)
        );
        assert!(!has_yaml_entry(&config, "extensions", "ice", "Test"));
    }

    #[test]
    fn remove_yaml_managed_entry_skips_malformed_config() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("bad.yaml");
        fs::write(&config, "extensions: [").unwrap();

        assert_eq!(
            remove_yaml_managed_entry(&config, "extensions", "ice", "Test").unwrap(),
            (false, true)
        );
    }

    #[test]
    fn remove_yaml_managed_entry_leaves_unmanaged_entry() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.yaml");
        fs::write(
            &config,
            r#"
extensions:
  ice:
    type: stdio
"#,
        )
        .unwrap();

        assert_eq!(
            remove_yaml_managed_entry(&config, "extensions", "ice", "Test").unwrap(),
            (false, false)
        );
        assert!(has_yaml_entry(&config, "extensions", "ice", "Test"));
    }

    #[test]
    fn has_yaml_entry_returns_false_for_malformed_config() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("bad.yaml");
        fs::write(&config, "extensions: [").unwrap();

        assert!(!has_yaml_entry(&config, "extensions", "ice", "Test"));
    }
}
