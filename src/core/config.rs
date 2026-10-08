use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(
    author,
    version,
    about = "Mini-Serve: High-throughput Rust LLM inference engine"
)]
pub struct Settings {
    #[arg(long, env = "MINI_HOST", default_value = "0.0.0.0")]
    pub host: String,

    #[arg(long, env = "MINI_PORT", default_value_t = 8000)]
    pub port: u16,

    #[arg(long, env = "MINI_MODEL", default_value = "mock")]
    pub model: String,

    #[arg(long, env = "MINI_DEVICE", default_value = "auto")]
    pub device: String,

    #[arg(long, env = "MINI_MAX_BATCH_SIZE", default_value_t = 8)]
    pub max_batch_size: usize,

    #[arg(long, env = "MINI_MAX_WAITING_REQUESTS", default_value_t = 256)]
    pub max_waiting_requests: usize,

    #[arg(long, env = "MINI_KV_CACHE_BLOCKS", default_value_t = 1024)]
    pub kv_cache_blocks: usize,

    #[arg(long, env = "MINI_BLOCK_SIZE", default_value_t = 16)]
    pub block_size: usize,

    #[arg(long, env = "MINI_NUM_WORKERS", default_value_t = 2)]
    pub num_workers: usize,

    #[arg(long, env = "MINI_LOG_LEVEL", default_value = "info")]
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
            num_workers: 2,
            log_level: "info".to_string(),
            hf_token: None,
        }
    }
}
