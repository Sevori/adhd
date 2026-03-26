use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingBackendConfig {
    Hash {
        dim: usize,
    },
    Ollama {
        endpoint: String,
        model: String,
        timeout_secs: u64,
    },
}

impl Default for EmbeddingBackendConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl EmbeddingBackendConfig {
    pub fn from_env() -> Self {
        match std::env::var("ICE_EMBEDDING_PROVIDER")
            .unwrap_or_else(|_| "hash".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "ollama" => Self::Ollama {
                endpoint: std::env::var("ICE_EMBEDDING_ENDPOINT")
                    .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string()),
                model: std::env::var("ICE_EMBEDDING_MODEL")
                    .unwrap_or_else(|_| "all-minilm".to_string()),
                timeout_secs: std::env::var("ICE_EMBEDDING_TIMEOUT_SECS")
                    .ok()
                    .and_then(|value| value.parse::<u64>().ok())
                    .unwrap_or(30),
            },
            _ => Self::Hash {
                dim: std::env::var("ICE_HASH_EMBED_DIM")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .filter(|value| *value > 0)
                    .unwrap_or(128),
            },
        }
    }

    pub fn backend_key(&self) -> String {
        match self {
            Self::Hash { dim } => format!("hash:{dim}"),
            Self::Ollama { model, .. } => format!("ollama:{model}"),
        }
    }
}

pub trait EmbeddingAdapter {
    fn backend_key(&self) -> String;
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

pub enum EmbeddingRuntime {
    Hash(HashEmbeddingAdapter),
    Ollama(OllamaEmbeddingAdapter),
}

impl EmbeddingRuntime {
    pub fn from_config(config: &EmbeddingBackendConfig) -> Result<Self> {
        match config {
            EmbeddingBackendConfig::Hash { dim } => {
                Ok(Self::Hash(HashEmbeddingAdapter { dim: *dim }))
            }
            EmbeddingBackendConfig::Ollama {
                endpoint,
                model,
                timeout_secs,
            } => Ok(Self::Ollama(OllamaEmbeddingAdapter::new(
                endpoint.clone(),
                model.clone(),
                *timeout_secs,
            )?)),
        }
    }

    pub fn backend_key(&self) -> String {
        match self {
            Self::Hash(adapter) => adapter.backend_key(),
            Self::Ollama(adapter) => adapter.backend_key(),
        }
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        match self {
            Self::Hash(adapter) => adapter.embed(text),
            Self::Ollama(adapter) => adapter.embed(text),
        }
    }
}

pub struct HashEmbeddingAdapter {
    dim: usize,
}

impl EmbeddingAdapter for HashEmbeddingAdapter {
    fn backend_key(&self) -> String {
        format!("hash:{}", self.dim)
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(hash_embed(text, self.dim))
    }
}

pub struct OllamaEmbeddingAdapter {
    endpoint: String,
    model: String,
    client: Client,
}

impl OllamaEmbeddingAdapter {
    pub fn new(endpoint: String, model: String, timeout_secs: u64) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs.max(5)))
            .build()?;
        Ok(Self {
            endpoint,
            model,
            client,
        })
    }
}

impl EmbeddingAdapter for OllamaEmbeddingAdapter {
    fn backend_key(&self) -> String {
        format!("ollama:{}", self.model)
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let response = self
            .client
            .post(format!("{}/api/embed", self.endpoint))
            .json(&OllamaEmbedRequest {
                model: self.model.clone(),
                input: text.to_string(),
            })
            .send()
            .with_context(|| format!("requesting embeddings from {}", self.endpoint))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .unwrap_or_else(|error| format!("unable to read ollama error body: {error}"));
            return Err(anyhow!(
                "ollama embed failed for {} with {}: {}",
                self.model,
                status,
                body
            ));
        }
        let payload: OllamaEmbedResponse = response.json()?;
        let vector = payload
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("ollama embed returned no embeddings"))?;
        if vector.is_empty() {
            return Err(anyhow!("ollama embed returned an empty vector"));
        }
        Ok(vector)
    }
}

#[derive(Debug, Clone, Serialize)]
struct OllamaEmbedRequest {
    model: String,
    input: String,
}

#[derive(Debug, Clone, Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

pub fn hash_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut vector = vec![0.0_f32; dim];
    for token in text
        .split(|c: char| !(c.is_alphanumeric() || matches!(c, '/' | '.' | '_' | '-')))
        .filter(|token| !token.is_empty())
    {
        let lowered = token.to_lowercase();
        accumulate_feature(&mut vector, lowered.as_bytes());
        if lowered.len() >= 3 {
            for window in lowered.as_bytes().windows(3) {
                accumulate_feature(&mut vector, window);
            }
        }
    }
    let norm = l2_norm(&vector) as f32;
    if norm > 0.0 {
        for item in &mut vector {
            *item /= norm;
        }
    }
    vector
}

fn accumulate_feature(vector: &mut [f32], feature: &[u8]) {
    let hash = blake3::hash(feature);
    let bytes = hash.as_bytes();
    let index = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize % vector.len();
    let sign = if bytes[4] % 2 == 0 { 1.0 } else { -1.0 };
    vector[index] += sign;
}

pub fn l2_norm(vector: &[f32]) -> f64 {
    let sum = vector
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum::<f64>();
    sum.sqrt()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let dot = (0..len)
        .map(|index| a[index] as f64 * b[index] as f64)
        .sum::<f64>();
    let denom = l2_norm(a) * l2_norm(b);
    if denom == 0.0 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use super::*;

    #[test]
    fn hash_embedding_backend_key_includes_dimension() {
        let runtime = EmbeddingRuntime::from_config(&EmbeddingBackendConfig::Hash { dim: 64 })
            .expect("hash runtime");
        assert_eq!(runtime.backend_key(), "hash:64");
        let vector = runtime
            .embed("database crashed at 3am")
            .expect("hash embed");
        assert_eq!(vector.len(), 64);
    }

    #[test]
    fn ollama_embedding_runtime_calls_embed_endpoint() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake ollama");
        let address = listener.local_addr().expect("fake ollama addr");
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept request");
            let mut buffer = [0_u8; 4096];
            let read = stream.read(&mut buffer).expect("read request");
            let request = String::from_utf8_lossy(&buffer[..read]);
            assert!(request.starts_with("POST /api/embed HTTP/1.1"));
            assert!(request.contains("\"model\":\"embeddinggemma\""));
            let body = r#"{"embeddings":[[0.0,1.0,0.0]]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream
                .write_all(response.as_bytes())
                .expect("write response");
        });

        let runtime = EmbeddingRuntime::from_config(&EmbeddingBackendConfig::Ollama {
            endpoint: format!("http://{}", address),
            model: "embeddinggemma".into(),
            timeout_secs: 5,
        })
        .expect("ollama runtime");
        let vector = runtime
            .embed("database crashed at 3am")
            .expect("ollama embed");
        assert_eq!(runtime.backend_key(), "ollama:embeddinggemma");
        assert_eq!(vector, vec![0.0, 1.0, 0.0]);

        server.join().expect("server join");
    }

    #[test]
    fn hash_embed_empty_text_returns_zero_vector() {
        let vector = hash_embed("", 64);
        assert_eq!(vector.len(), 64);
        assert!(vector.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn hash_embed_normalised_to_unit_length() {
        let vector = hash_embed("database crashed at 3am during deploy", 128);
        let norm = l2_norm(&vector);
        assert!((norm - 1.0).abs() < 1e-6, "expected unit norm, got {norm}");
    }

    #[test]
    fn hash_embed_deterministic() {
        let a = hash_embed("same input text", 64);
        let b = hash_embed("same input text", 64);
        assert_eq!(a, b);
    }

    #[test]
    fn hash_embed_different_texts_differ() {
        let a = hash_embed("database crash", 64);
        let b = hash_embed("network timeout", 64);
        assert_ne!(a, b);
    }

    #[test]
    fn cosine_similarity_identical_vectors() {
        let a = hash_embed("test text", 64);
        let similarity = cosine_similarity(&a, &a);
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_zero() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let similarity = cosine_similarity(&a, &b);
        assert!((similarity - 0.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_empty_vectors() {
        let similarity = cosine_similarity(&[], &[]);
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn cosine_similarity_zero_vectors() {
        let a = vec![0.0, 0.0, 0.0];
        let similarity = cosine_similarity(&a, &a);
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn cosine_similarity_different_lengths_uses_min_for_dot() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.5, 0.5];
        let similarity = cosine_similarity(&a, &b);
        // dot product uses min(2,4)=2 elements -> dot=1.0
        // norms use full vectors: norm_a=1.0, norm_b=sqrt(1.5)
        let expected = 1.0 / (1.0 * (1.5_f64).sqrt());
        assert!((similarity - expected).abs() < 1e-6);
    }

    #[test]
    fn l2_norm_zero_vector() {
        assert_eq!(l2_norm(&[0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn l2_norm_unit_vector() {
        let norm = l2_norm(&[1.0, 0.0, 0.0]);
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn embedding_backend_config_backend_key() {
        let hash_config = EmbeddingBackendConfig::Hash { dim: 256 };
        assert_eq!(hash_config.backend_key(), "hash:256");

        let ollama_config = EmbeddingBackendConfig::Ollama {
            endpoint: "http://localhost:11434".into(),
            model: "nomic-embed-text".into(),
            timeout_secs: 30,
        };
        assert_eq!(ollama_config.backend_key(), "ollama:nomic-embed-text");
    }

    #[test]
    fn embedding_runtime_from_hash_config() {
        let config = EmbeddingBackendConfig::Hash { dim: 32 };
        let runtime = EmbeddingRuntime::from_config(&config).unwrap();
        assert_eq!(runtime.backend_key(), "hash:32");
        let vec = runtime.embed("test").unwrap();
        assert_eq!(vec.len(), 32);
    }

    #[test]
    fn ollama_adapter_backend_key() {
        let adapter =
            OllamaEmbeddingAdapter::new("http://127.0.0.1:11434".into(), "test-model".into(), 10)
                .unwrap();
        assert_eq!(adapter.backend_key(), "ollama:test-model");
    }

    #[test]
    fn ollama_adapter_connection_refused_returns_error() {
        let adapter =
            OllamaEmbeddingAdapter::new("http://127.0.0.1:1".into(), "fake-model".into(), 1)
                .unwrap();
        let err = adapter.embed("test").unwrap_err();
        assert!(err.to_string().contains("requesting embeddings"));
    }

    #[test]
    fn ollama_embed_non_success_status_returns_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("addr");
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buffer = [0_u8; 4096];
            let _ = stream.read(&mut buffer).expect("read");
            let body = "model not found";
            let response = format!(
                "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).expect("write");
        });
        let adapter =
            OllamaEmbeddingAdapter::new(format!("http://{}", address), "bad-model".into(), 5)
                .unwrap();
        let err = adapter.embed("test").unwrap_err();
        assert!(err.to_string().contains("ollama embed failed"));
        server.join().unwrap();
    }

    #[test]
    fn ollama_embed_empty_embeddings_returns_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("addr");
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buffer = [0_u8; 4096];
            let _ = stream.read(&mut buffer).expect("read");
            let body = r#"{"embeddings":[]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).expect("write");
        });
        let adapter =
            OllamaEmbeddingAdapter::new(format!("http://{}", address), "empty-model".into(), 5)
                .unwrap();
        let err = adapter.embed("test").unwrap_err();
        assert!(err.to_string().contains("no embeddings"));
        server.join().unwrap();
    }

    #[test]
    fn ollama_embed_empty_vector_returns_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("addr");
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buffer = [0_u8; 4096];
            let _ = stream.read(&mut buffer).expect("read");
            let body = r#"{"embeddings":[[]]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).expect("write");
        });
        let adapter =
            OllamaEmbeddingAdapter::new(format!("http://{}", address), "empty-vec-model".into(), 5)
                .unwrap();
        let err = adapter.embed("test").unwrap_err();
        assert!(err.to_string().contains("empty vector"));
        server.join().unwrap();
    }

    #[test]
    fn hash_embed_token_splitting() {
        let vector = hash_embed("hello/world.rs foo_bar-baz", 64);
        assert_eq!(vector.len(), 64);
        let norm = l2_norm(&vector);
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hash_embed_short_tokens_skip_trigrams() {
        let a = hash_embed("ab", 64);
        let b = hash_embed("ab", 64);
        assert_eq!(a, b);
    }

    #[test]
    fn ollama_timeout_clamped_to_minimum_five() {
        let adapter =
            OllamaEmbeddingAdapter::new("http://127.0.0.1:11434".into(), "test".into(), 1).unwrap();
        assert_eq!(adapter.backend_key(), "ollama:test");
    }
}
