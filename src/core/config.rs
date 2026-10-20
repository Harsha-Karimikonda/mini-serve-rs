use clap::Parser;

#[derive(Parser, Clone)]
#[command(
    author,
    version,
    about = "RLSE: High-throughput Rust LLM inference engine (Rust LLM Serving Engine)"
)]
pub struct Settings {
    #[arg(long, env = "RLSE_HOST", default_value = "0.0.0.0")]
    pub host: String,

    #[arg(long, env = "RLSE_PORT", default_value_t = 8000)]
    pub port: u16,

    #[arg(long, env = "RLSE_MODEL", default_value = "mock")]
    pub model: String,

    #[arg(long, env = "RLSE_DEVICE", default_value = "auto")]
    pub device: String,

    #[arg(long, env = "RLSE_MAX_BATCH_SIZE", default_value_t = 8)]
    pub max_batch_size: usize,

    #[arg(long, env = "RLSE_MAX_WAITING_REQUESTS", default_value_t = 256)]
    pub max_waiting_requests: usize,

    #[arg(long, env = "RLSE_KV_CACHE_BLOCKS", default_value_t = 1024)]
    pub kv_cache_blocks: usize,

    #[arg(long, env = "RLSE_BLOCK_SIZE", default_value_t = 16)]
    pub block_size: usize,

    /// Bytes reserved for KV tensors, prefix activations, and estimated execution workspaces; excludes weights.
    #[arg(long, env = "RLSE_TENSOR_MEMORY_BUDGET_BYTES", default_value_t = crate::cache::tensor_memory::DEFAULT_TENSOR_MEMORY_BUDGET)]
    pub tensor_memory_budget_bytes: usize,

    #[arg(long, env = "RLSE_EXECUTION_HEADROOM_BYTES", default_value_t = crate::cache::tensor_memory::DEFAULT_EXECUTION_HEADROOM)]
    pub execution_headroom_bytes: usize,

    #[arg(long, env = "RLSE_NUM_WORKERS", default_value_t = 1)]
    pub num_workers: usize,

    #[arg(long, env = "RLSE_SHUTDOWN_TIMEOUT_MS", default_value_t = 30_000)]
    pub shutdown_timeout_ms: u64,

    #[arg(long, env = "RLSE_ALLOW_MOCK_FALLBACK", default_value_t = false)]
    pub allow_mock_fallback: bool,

    #[arg(long, env = "RLSE_LOG_LEVEL", default_value = "info")]
    pub log_level: String,

    /// Optional shared bearer key for protected HTTP routes. Prefer the environment variable.
    #[arg(long, env = "RLSE_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,

    /// Additional independent client credentials (comma-separated in the environment).
    #[arg(
        long,
        env = "RLSE_API_KEYS",
        value_delimiter = ',',
        hide_env_values = true
    )]
    pub api_keys: Vec<String>,

    /// Accepted generation attempts per client per fixed 60-second window; zero disables.
    #[arg(long, env = "RLSE_CLIENT_REQUESTS_PER_MINUTE", default_value_t = 0)]
    pub client_requests_per_minute: usize,

    /// Queued and active generations per client; zero disables.
    #[arg(long, env = "RLSE_CLIENT_MAX_CONCURRENT", default_value_t = 0)]
    pub client_max_concurrent: usize,

    #[arg(long, env = "HF_TOKEN", hide_env_values = true)]
    pub hf_token: Option<String>,
}

use crate::core::constants::*;

impl Default for Settings {
    fn default() -> Self {
        Self {
            host: DEFAULT_HOST.to_string(),
            port: DEFAULT_PORT,
            model: DEFAULT_MODEL.to_string(),
            device: DEFAULT_DEVICE.to_string(),
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            max_waiting_requests: DEFAULT_MAX_WAITING_REQUESTS,
            kv_cache_blocks: DEFAULT_KV_CACHE_BLOCKS,
            block_size: DEFAULT_BLOCK_SIZE,
            num_workers: 1,
            shutdown_timeout_ms: 30_000,
            tensor_memory_budget_bytes: crate::cache::tensor_memory::DEFAULT_TENSOR_MEMORY_BUDGET,
            execution_headroom_bytes: crate::cache::tensor_memory::DEFAULT_EXECUTION_HEADROOM,
            allow_mock_fallback: false,
            log_level: "info".to_string(),
            hf_token: None,
            api_key: None,
            api_keys: Vec::new(),
            client_requests_per_minute: 0,
            client_max_concurrent: 0,
        }
    }
}

impl Settings {
    /// Validate before constructing caches, loading weights, or spawning workers.
    pub fn validate(&self) -> Result<(), crate::core::errors::EngineError> {
        use crate::core::errors::EngineError;
        for (name, value) in [
            ("block_size", self.block_size),
            ("kv_cache_blocks", self.kv_cache_blocks),
            ("max_batch_size", self.max_batch_size),
            ("max_waiting_requests", self.max_waiting_requests),
            ("num_workers", self.num_workers),
        ] {
            if value == 0 {
                return Err(EngineError::InvalidRequest(format!(
                    "{name} must be greater than zero"
                )));
            }
        }
        let capacity = self
            .kv_cache_blocks
            .checked_mul(self.block_size)
            .ok_or_else(|| {
                EngineError::InvalidRequest("KV token capacity overflows usize".into())
            })?;
        if capacity < 2 {
            return Err(EngineError::InvalidRequest(
                "KV capacity must hold a prompt and at least one output token".into(),
            ));
        }
        crate::cache::tensor_memory::TensorMemoryBudget::new(
            self.tensor_memory_budget_bytes,
            self.execution_headroom_bytes,
        )?;
        if self.api_keys.len() + usize::from(self.api_key.is_some()) > 128 {
            return Err(EngineError::InvalidRequest(
                "At most 128 API keys are supported".into(),
            ));
        }
        if self
            .api_key
            .iter()
            .chain(&self.api_keys)
            .any(|key| key.is_empty() || !key.bytes().all(|b| b.is_ascii_graphic()))
        {
            return Err(EngineError::InvalidRequest(
                "api_key must contain only nonempty printable ASCII without whitespace".into(),
            ));
        }
        validate_device(&self.device)
    }
}

pub fn validate_device(device: &str) -> Result<(), crate::core::errors::EngineError> {
    let supported = match device {
        "auto" | "cpu" => true,
        "metal" => cfg!(feature = "metal"),
        "cuda" => cfg!(feature = "cuda"),
        _ => false,
    };
    if supported {
        Ok(())
    } else {
        Err(crate::core::errors::EngineError::InvalidRequest(format!(
            "Unsupported device '{device}'; use auto, cpu, or an accelerator enabled at build time"
        )))
    }
}

// Credentials must never appear in diagnostics, including Debug output.
impl std::fmt::Debug for Settings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Settings")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("model", &self.model)
            .field("device", &self.device)
            .field(
                "authentication_enabled",
                &(self.api_key.is_some() || !self.api_keys.is_empty()),
            )
            .finish_non_exhaustive()
    }
}
