use async_trait::async_trait;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::info;

use crate::backends::qwen2::{Config as QwenConfig, ModelForCausalLM as QwenModel};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::backends::traits::{ModelBackend, StepToken};
use crate::core::errors::EngineError;
use crate::core::types::{ActiveSequence, RequestId};

pub struct CandleBackend {
    name: String,
    device: Device,
    tokenizer: Tokenizer,
    model_template: Arc<QwenModel>,
    active_models: Mutex<HashMap<RequestId, QwenModel>>,
    eos_token_id: u32,
}

impl CandleBackend {
    pub fn load_hf(model_id: &str, device_preference: &str) -> Result<Self, EngineError> {
        info!("Initializing CandleBackend for model: {}", model_id);

        let local_path = std::path::Path::new(model_id);
        let normalized_slug = model_id.replace('/', "-");
        let local_slug_path = std::path::Path::new("models").join(&normalized_slug);
        let local_models_path = std::path::Path::new("models").join(model_id);

        let (tokenizer_path, config_path, model_path) = if local_path.is_dir()
            && local_path.join("model.safetensors").exists()
        {
            info!("Loading model from local path: {:?}", local_path);
            (
                local_path.join("tokenizer.json"),
                local_path.join("config.json"),
                local_path.join("model.safetensors"),
            )
        } else if local_slug_path.is_dir() && local_slug_path.join("model.safetensors").exists() {
            info!("Loading model from local directory: {:?}", local_slug_path);
            (
                local_slug_path.join("tokenizer.json"),
                local_slug_path.join("config.json"),
                local_slug_path.join("model.safetensors"),
            )
        } else if local_models_path.is_dir() && local_models_path.join("model.safetensors").exists()
        {
            info!(
                "Loading model from local directory: {:?}",
                local_models_path
            );
            (
                local_models_path.join("tokenizer.json"),
                local_models_path.join("config.json"),
                local_models_path.join("model.safetensors"),
            )
        } else {
            info!("Fetching model {} from Hugging Face hub...", model_id);
            let token = std::env::var("HF_TOKEN")
                .or_else(|_| std::env::var("hf_token"))
                .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
                .ok();

            let mut builder = hf_hub::api::sync::ApiBuilder::from_env();
            if let Some(ref t) = token {
                let mask_len = t.len().min(8);
                info!(
                    "Authenticated Hugging Face request with token ({}...)",
                    &t[..mask_len]
                );
                builder = builder.with_token(Some(t.clone()));
            }

            let api = builder
                .build()
                .map_err(|e| EngineError::BackendError(format!("Failed to init HF API: {}", e)))?;
            let repo = api.model(model_id.to_string());

            let tok = repo.get("tokenizer.json").map_err(|e| {
                EngineError::BackendError(format!("Failed to fetch tokenizer: {}", e))
            })?;
            let cfg = repo
                .get("config.json")
                .map_err(|e| EngineError::BackendError(format!("Failed to fetch config: {}", e)))?;
            let w = repo.get("model.safetensors").map_err(|e| {
                EngineError::BackendError(format!("Failed to fetch model weights: {}", e))
            })?;
            (tok, cfg, w)
        };

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EngineError::BackendError(format!("Failed to load tokenizer: {}", e)))?;

        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| EngineError::BackendError(format!("Failed to read config: {}", e)))?;
        let config: QwenConfig = serde_json::from_str(&config_str)
            .map_err(|e| EngineError::BackendError(format!("Failed to parse config: {}", e)))?;

        // Determine device
        let device = match device_preference {
            #[cfg(feature = "metal")]
            "metal" | "auto" => Device::new_metal(0).unwrap_or(Device::Cpu),
            #[cfg(feature = "cuda")]
            "cuda" => Device::new_cuda(0).unwrap_or(Device::Cpu),
            _ => Device::Cpu,
        };
        info!("Using device: {:?}", device);

        let dtype = match device {
            Device::Cpu => DType::F32,
            _ => DType::F16,
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device).map_err(|e| {
                EngineError::BackendError(format!("Failed to mmap safetensors: {}", e))
            })?
        };

        let mut model = QwenModel::new(&config, vb).map_err(|e| {
            EngineError::BackendError(format!("Failed to construct Qwen model: {}", e))
        })?;
        model.clear_kv_cache();

        let eos_token_id = tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(151643);

        Ok(Self {
            name: model_id.to_string(),
            device,
            tokenizer,
            model_template: Arc::new(model),
            active_models: Mutex::new(HashMap::new()),
            eos_token_id,
        })
    }
}

#[async_trait]
impl ModelBackend for CandleBackend {
    fn name(&self) -> &str {
        &self.name
    }

    async fn init_sequence(&self, prompt: &str) -> Result<Vec<u32>, EngineError> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| EngineError::InvalidRequest(format!("Tokenization error: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    async fn step_batch(
        &self,
        sequences: &mut [ActiveSequence],
    ) -> Result<Vec<StepToken>, EngineError> {
        if sequences.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(sequences.len());
        let mut active_models_guard = self.active_models.lock();

        // Prune any stale models for sequences that are no longer active
        active_models_guard.retain(|req_id, _| sequences.iter().any(|s| &s.request_id == req_id));

        for seq in sequences.iter_mut() {
            // Get or clone an isolated model instance for this sequence (zero-copy weight tensors)
            let model = active_models_guard
                .entry(seq.request_id.clone())
                .or_insert_with(|| {
                    let mut cloned = self.model_template.as_ref().clone();
                    cloned.clear_kv_cache();
                    cloned
                });

            let input_ids = if !seq.is_prefilled {
                // Prefill pass: feed entire prompt token sequence
                seq.is_prefilled = true;
                seq.prompt_tokens.clone()
            } else {
                // Decode pass: feed only the latest generated token
                vec![*seq.output_tokens.last().unwrap_or(&1)]
            };

            let input_tensor = Tensor::new(input_ids.as_slice(), &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| EngineError::BackendError(format!("Tensor creation failed: {}", e)))?;

            let pos = if !seq.output_tokens.is_empty() {
                seq.prompt_tokens.len() + seq.output_tokens.len() - 1
            } else {
                0
            };

            let step_start = std::time::Instant::now();
            let logits = model
                .forward(&input_tensor, pos)
                .map_err(|e| EngineError::BackendError(format!("Forward pass failed: {}", e)))?;
            let forward_dur = step_start.elapsed();

            // candle's Qwen2 forward returns [batch_size, 1, vocab_size] narrowed to the last token
            let last_logits = logits.squeeze(0).and_then(|t| t.squeeze(0)).map_err(|e| {
                EngineError::BackendError(format!("Logits extraction failed: {}", e))
            })?;

            // Greedy argmax sampling
            let next_token_id = last_logits
                .argmax(0)
                .and_then(|t| t.to_scalar::<u32>())
                .map_err(|e| EngineError::BackendError(format!("Argmax failed: {}", e)))?;

            let total_dur = step_start.elapsed();
            tracing::debug!(
                "Token step for {}: forward={:?}, total={:?} (pos={})",
                seq.request_id,
                forward_dur,
                total_dur,
                pos
            );

            let is_eos = next_token_id == self.eos_token_id
                || seq.output_tokens.len() + 1 >= seq.sampling.max_tokens;

            let token_text = self
                .tokenizer
                .decode(&[next_token_id], false)
                .unwrap_or_else(|_| " ".to_string());

            results.push(StepToken {
                token_id: next_token_id,
                text: token_text,
                is_eos,
            });
        }

        // Clean up models for any sequences that just finished
        for (i, seq) in sequences.iter().enumerate() {
            if results[i].is_eos {
                active_models_guard.remove(&seq.request_id);
            }
        }

        Ok(results)
    }
}
