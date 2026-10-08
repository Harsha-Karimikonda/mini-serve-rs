use async_trait::async_trait;
use parking_lot::Mutex;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::info;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config as QwenConfig, ModelForCausalLM as QwenModel};

use crate::backends::traits::{ModelBackend, StepToken};
use crate::core::errors::EngineError;
use crate::core::types::ActiveSequence;

pub struct CandleBackend {
    name: String,
    device: Device,
    tokenizer: Tokenizer,
    model: Arc<Mutex<QwenModel>>,
    eos_token_id: u32,
}

impl CandleBackend {
    pub fn load_hf(model_id: &str, device_preference: &str) -> Result<Self, EngineError> {
        info!("Loading model {} from Hugging Face hub...", model_id);
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| EngineError::BackendError(format!("Failed to init HF API: {}", e)))?;
        let repo = api.model(model_id.to_string());

        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| EngineError::BackendError(format!("Failed to fetch tokenizer: {}", e)))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| EngineError::BackendError(format!("Failed to load tokenizer: {}", e)))?;

        let config_path = repo
            .get("config.json")
            .map_err(|e| EngineError::BackendError(format!("Failed to fetch config: {}", e)))?;
        let config_str = std::fs::read_to_string(config_path)
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

        let model_path = repo.get("model.safetensors").map_err(|e| {
            EngineError::BackendError(format!("Failed to fetch model weights: {}", e))
        })?;

        let dtype = match device {
            Device::Cpu => DType::F32,
            _ => DType::F16,
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device).map_err(|e| {
                EngineError::BackendError(format!("Failed to mmap safetensors: {}", e))
            })?
        };

        let model = QwenModel::new(&config, vb).map_err(|e| {
            EngineError::BackendError(format!("Failed to construct Qwen model: {}", e))
        })?;

        let eos_token_id = tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(151643);

        Ok(Self {
            name: model_id.to_string(),
            device,
            tokenizer,
            model: Arc::new(Mutex::new(model)),
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

        // For each active sequence, perform forward step
        let mut model_lock = self.model.lock();

        for seq in sequences.iter_mut() {
            let input_ids = if !seq.is_prefilled {
                // Prefill pass: all prompt tokens
                seq.is_prefilled = true;
                seq.prompt_tokens.clone()
            } else {
                // Decode pass: latest generated token
                vec![*seq.output_tokens.last().unwrap_or(&1)]
            };

            let seq_len = input_ids.len();
            let input_tensor = Tensor::new(input_ids.as_slice(), &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| EngineError::BackendError(format!("Tensor creation failed: {}", e)))?;

            let pos = if seq.is_prefilled && !seq.output_tokens.is_empty() {
                seq.prompt_tokens.len() + seq.output_tokens.len() - 1
            } else {
                0
            };

            let logits = model_lock
                .forward(&input_tensor, pos)
                .map_err(|e| EngineError::BackendError(format!("Forward pass failed: {}", e)))?;

            // Take last token logits [1, seq_len, vocab_size] -> [vocab_size]
            let last_logits = logits
                .squeeze(0)
                .and_then(|t| t.get(seq_len - 1))
                .map_err(|e| {
                    EngineError::BackendError(format!("Logits extraction failed: {}", e))
                })?;

            // Greedy argmax
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

            results.push(StepToken {
                token_id: next_token_id,
                text: token_text,
                is_eos,
            });
        }

        Ok(results)
    }
}
