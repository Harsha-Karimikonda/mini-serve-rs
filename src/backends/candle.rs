use async_trait::async_trait;
use parking_lot::Mutex;
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
    active_models: Mutex<Vec<(RequestId, QwenModel)>>,
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
            active_models: Mutex::new(Vec::new()),
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

        let mut active_models_guard = self.active_models.lock();

        // Prune any stale models for sequences that are no longer active
        active_models_guard.retain(|(req_id, _)| sequences.iter().any(|s| &s.request_id == req_id));

        // Ensure every active sequence has an initialized model
        for seq in sequences.iter() {
            if !active_models_guard
                .iter()
                .any(|(id, _)| id == &seq.request_id)
            {
                let mut cloned = self.model_template.as_ref().clone();
                cloned.clear_kv_cache();
                active_models_guard.push((seq.request_id.clone(), cloned));
            }
        }

        // Align active_models_guard to the exact sequence ordering
        active_models_guard.sort_by_key(|(id, _)| {
            sequences
                .iter()
                .position(|s| &s.request_id == id)
                .unwrap_or(usize::MAX)
        });

        let mut results: Vec<Option<StepToken>> = vec![None; sequences.len()];

        // 1. Prefill pass: Execute any un-prefilled sequences individually
        for (i, seq) in sequences.iter_mut().enumerate() {
            if !seq.is_prefilled {
                seq.is_prefilled = true;
                let model = &mut active_models_guard[i].1;
                let input_tensor = Tensor::new(seq.prompt_tokens.as_slice(), &self.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| {
                        EngineError::BackendError(format!("Prefill tensor creation failed: {}", e))
                    })?;

                let pos = 0;
                let logits = model.forward(&input_tensor, pos).map_err(|e| {
                    EngineError::BackendError(format!("Prefill forward pass failed: {}", e))
                })?;

                let last_logits = logits.squeeze(0).and_then(|t| t.squeeze(0)).map_err(|e| {
                    EngineError::BackendError(format!("Prefill logits extraction failed: {}", e))
                })?;

                let next_token_id = last_logits
                    .argmax(0)
                    .and_then(|t| t.to_scalar::<u32>())
                    .map_err(|e| EngineError::BackendError(format!("Argmax failed: {}", e)))?;

                let is_eos = next_token_id == self.eos_token_id
                    || seq.output_tokens.len() + 1 >= seq.sampling.max_tokens;

                let token_text = self
                    .tokenizer
                    .decode(&[next_token_id], false)
                    .unwrap_or_else(|_| " ".to_string());

                results[i] = Some(StepToken {
                    token_id: next_token_id,
                    text: token_text,
                    is_eos,
                });
            }
        }

        // 2. Decode pass: Batch all remaining decode sequences into a single unified GPU forward pass
        let decode_indices: Vec<usize> = (0..sequences.len())
            .filter(|&i| results[i].is_none())
            .collect();

        if !decode_indices.is_empty() {
            let mut decode_models: Vec<&mut QwenModel> = Vec::with_capacity(decode_indices.len());
            let mut decode_input_ids: Vec<u32> = Vec::with_capacity(decode_indices.len());
            let mut decode_positions: Vec<usize> = Vec::with_capacity(decode_indices.len());

            let mut models_iter = active_models_guard.iter_mut();
            let mut prev_idx = 0;
            for &idx in &decode_indices {
                let skip = idx - prev_idx;
                let (_, model) = models_iter.nth(skip).unwrap();
                decode_models.push(model);
                prev_idx = idx + 1;

                let seq = &sequences[idx];
                let token_id = *seq.output_tokens.last().unwrap_or(&1);
                decode_input_ids.push(token_id);
                let pos = if !seq.output_tokens.is_empty() {
                    seq.prompt_tokens.len() + seq.output_tokens.len() - 1
                } else {
                    0
                };
                decode_positions.push(pos);
            }

            let next_tokens = QwenModel::step_decode_batch(
                &mut decode_models[..],
                &decode_input_ids,
                &decode_positions,
                &self.device,
            )
            .map_err(|e| EngineError::BackendError(format!("Batched decode step failed: {}", e)))?;

            for (k, &idx) in decode_indices.iter().enumerate() {
                let next_token_id = next_tokens[k];
                let seq = &sequences[idx];
                let is_eos = next_token_id == self.eos_token_id
                    || seq.output_tokens.len() + 1 >= seq.sampling.max_tokens;

                let token_text = self
                    .tokenizer
                    .decode(&[next_token_id], false)
                    .unwrap_or_else(|_| " ".to_string());

                results[idx] = Some(StepToken {
                    token_id: next_token_id,
                    text: token_text,
                    is_eos,
                });
            }
        }

        let final_results: Vec<StepToken> = results.into_iter().map(|r| r.unwrap()).collect();

        // Clean up models for any sequences that just finished
        for (i, seq) in sequences.iter().enumerate() {
            if final_results[i].is_eos {
                active_models_guard.retain(|(id, _)| id != &seq.request_id);
            }
        }

        Ok(final_results)
    }
}
