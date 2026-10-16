use clap::Parser;

#[derive(Parser, Debug, Clone)]
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

    #[arg(long, env = "RLSE_ALLOW_MOCK_FALLBACK", default_value_t = false)]
    pub allow_mock_fallback: bool,

    #[arg(long, env = "RLSE_LOG_LEVEL", default_value = "info")]
    pub log_level: String,

    #[arg(long, env = "HF_TOKEN")]
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
            tensor_memory_budget_bytes: crate::cache::tensor_memory::DEFAULT_TENSOR_MEMORY_BUDGET,
            execution_headroom_bytes: crate::cache::tensor_memory::DEFAULT_EXECUTION_HEADROOM,
            allow_mock_fallback: false,
            log_level: "info".to_string(),
            hf_token: None,
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
