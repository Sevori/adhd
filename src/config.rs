use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::embedding::EmbeddingBackendConfig;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub root: PathBuf,
    pub segment_target_bytes: u64,
    pub episode_max_chars: usize,
    pub embedding_backend: EmbeddingBackendConfig,
}

#[derive(Debug, Clone)]
pub struct EnginePaths {
    pub root: PathBuf,
    pub data_dir: PathBuf,
    pub log_dir: PathBuf,
    pub debug_dir: PathBuf,
    pub sqlite_path: PathBuf,
    pub dispatch_config_path: PathBuf,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            root: PathBuf::from(".ice"),
            segment_target_bytes: 16 * 1024 * 1024,
            episode_max_chars: 8_192,
            embedding_backend: EmbeddingBackendConfig::default(),
        }
    }
}

impl EngineConfig {
    pub fn with_root(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            ..Self::default()
        }
    }

    pub fn with_embedding_backend(mut self, embedding_backend: EmbeddingBackendConfig) -> Self {
        self.embedding_backend = embedding_backend;
        self
    }

    pub fn paths(&self) -> EnginePaths {
        EnginePaths {
            root: self.root.clone(),
            data_dir: self.root.join("data"),
            log_dir: self.root.join("data/log"),
            debug_dir: self.root.join("data/debug"),
            sqlite_path: self.root.join("data/ice.sqlite"),
            dispatch_config_path: self.root.join("data/dispatch-config.json"),
        }
    }

    pub fn ensure_dirs(&self) -> Result<EnginePaths> {
        let paths = self.paths();
        fs::create_dir_all(&paths.root)?;
        fs::create_dir_all(&paths.data_dir)?;
        fs::create_dir_all(&paths.log_dir)?;
        fs::create_dir_all(paths.debug_dir.join("ingest"))?;
        fs::create_dir_all(paths.debug_dir.join("context-packs"))?;
        fs::create_dir_all(paths.debug_dir.join("views"))?;
        fs::create_dir_all(paths.debug_dir.join("handoffs"))?;
        fs::create_dir_all(paths.debug_dir.join("snapshots"))?;
        Ok(paths)
    }
}
