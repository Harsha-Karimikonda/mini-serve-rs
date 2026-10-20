use crate::cache::tensor_memory::{
    MemoryCharge, MemoryReservation, TensorMemoryBudget, DEFAULT_EXECUTION_HEADROOM,
    DEFAULT_TENSOR_MEMORY_BUDGET,
};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
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
    healthy: AtomicBool,
    device: Device,
    tokenizer: Tokenizer,
    chat_template: Result<crate::backends::chat::ChatTemplate, String>,
    model_template: Arc<QwenModel>,
    active_models: Mutex<Vec<(RequestId, QwenModel)>>,
    eos_token_id: u32,
    context_length: usize,
    memory_config: QwenConfig,
    memory_budget: Arc<TensorMemoryBudget>,
    reservations: Mutex<HashMap<RequestId, Arc<MemoryReservation>>>,
    prefix_cache: Option<crate::cache::SharedPrefixCache>,
}

impl CandleBackend {
    pub fn load_hf(model_id: &str, device_preference: &str) -> Result<Self, EngineError> {
        crate::core::config::validate_device(device_preference)?;
        info!("Initializing CandleBackend for model: {}", model_id);

        let local_path = std::path::Path::new(model_id);
        let normalized_slug = model_id.replace('/', "-");
        let local_slug_path = std::path::Path::new("models").join(&normalized_slug);
        let local_models_path = std::path::Path::new("models").join(model_id);

        let (tokenizer_path, config_path, model_paths, chat_config_path) = if local_path.is_dir()
            && (local_path.join("model.safetensors").exists()
                || local_path.join("model.safetensors.index.json").exists())
        {
            info!("Loading model from local path: {:?}", local_path);
            let weights = resolve_model_weights_local(local_path).ok_or_else(|| {
                EngineError::BackendError(format!(
                    "Failed to find valid model weights in local path: {:?}",
                    local_path
                ))
            })?;
            (
                local_path.join("tokenizer.json"),
                local_path.join("config.json"),
                weights,
                Some(local_path.join("tokenizer_config.json")),
            )
        } else if local_slug_path.is_dir()
            && (local_slug_path.join("model.safetensors").exists()
                || local_slug_path
                    .join("model.safetensors.index.json")
                    .exists())
        {
            info!("Loading model from local directory: {:?}", local_slug_path);
            let weights = resolve_model_weights_local(&local_slug_path).ok_or_else(|| {
                EngineError::BackendError(format!(
                    "Failed to find valid model weights in slug path: {:?}",
                    local_slug_path
                ))
            })?;
            (
                local_slug_path.join("tokenizer.json"),
                local_slug_path.join("config.json"),
                weights,
                Some(local_slug_path.join("tokenizer_config.json")),
            )
        } else if local_models_path.is_dir()
            && (local_models_path.join("model.safetensors").exists()
                || local_models_path
                    .join("model.safetensors.index.json")
                    .exists())
        {
            info!(
                "Loading model from local directory: {:?}",
                local_models_path
            );
            let weights = resolve_model_weights_local(&local_models_path).ok_or_else(|| {
                EngineError::BackendError(format!(
                    "Failed to find valid model weights in models path: {:?}",
                    local_models_path
                ))
            })?;
            (
                local_models_path.join("tokenizer.json"),
                local_models_path.join("config.json"),
                weights,
                Some(local_models_path.join("tokenizer_config.json")),
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
            let weights = resolve_model_weights_hf(&repo)?;
            let chat = repo
                .get("tokenizer_config.json")
                .inspect_err(|e| {
                    tracing::warn!("Chat metadata unavailable: {e}; completions remain available");
                })
                .ok();
            (tok, cfg, weights, chat)
        };

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EngineError::BackendError(format!("Failed to load tokenizer: {}", e)))?;

        let chat_template = chat_config_path
            .ok_or_else(|| "tokenizer_config.json unavailable".to_owned())
            .and_then(|path| std::fs::read_to_string(path).map_err(|e| e.to_string()))
            .and_then(|source| crate::backends::chat::ChatTemplate::from_config(&source));
        if let Err(error) = &chat_template {
            tracing::warn!("Chat unavailable: {error}; completions remain available");
        }

        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| EngineError::BackendError(format!("Failed to read config: {}", e)))?;
        let config: QwenConfig = serde_json::from_str(&config_str)
            .map_err(|e| EngineError::BackendError(format!("Failed to parse config: {}", e)))?;

        // Determine device
        let device = match device_preference {
            #[cfg(feature = "metal")]
            "metal" => Device::new_metal(0).map_err(|e| {
                EngineError::BackendError(format!("Metal device initialization failed: {e}"))
            })?,
            #[cfg(feature = "metal")]
            "auto" => Device::new_metal(0).unwrap_or(Device::Cpu),
            #[cfg(feature = "cuda")]
            "cuda" => Device::new_cuda(0).map_err(|e| {
                EngineError::BackendError(format!("CUDA device initialization failed: {}", e))
            })?,
            #[cfg(not(feature = "cuda"))]
            "cuda" => {
                return Err(EngineError::BackendError(
                    "CUDA device requested but 'cuda' feature is not compiled into rlse"
                        .to_string(),
                ));
            }
            _ => Device::Cpu,
        };
        info!("Using device: {:?}", device);

        let dtype = match device {
            Device::Cpu => DType::F32,
            _ => DType::F16,
        };

        // SAFETY: Loading safetensors weights via mmap is memory-efficient for LLM inference.
        // The backing files are guaranteed to remain unmodified while loaded.
        #[allow(unsafe_code)]
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&model_paths, dtype, &device).map_err(|e| {
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
            healthy: AtomicBool::new(true),
            device,
            tokenizer,
            chat_template,
            model_template: Arc::new(model),
            active_models: Mutex::new(Vec::new()),
            eos_token_id,
            context_length: config.max_position_embeddings,
            memory_config: config,
            memory_budget: TensorMemoryBudget::new(
                DEFAULT_TENSOR_MEMORY_BUDGET,
                DEFAULT_EXECUTION_HEADROOM,
            )
            .expect("valid defaults"),
            reservations: Mutex::new(HashMap::new()),
            prefix_cache: None,
        })
    }

    /// The actual execution device, rather than the requested device preference.
    pub fn device(&self) -> &Device {
        &self.device
    }

    fn is_eos_token(&self, id: u32) -> bool {
        id == self.eos_token_id
            || ["<|im_end|>", "<|endoftext|>"]
                .iter()
                .any(|token| self.tokenizer.token_to_id(token) == Some(id))
    }

    /// Upper-bound the request's contiguous KV capacity plus estimated workspaces.
    /// Allocator metadata, kernels' private workspaces, and weights are excluded.
    fn memory_charge(&self, prompt: usize, max_tokens: usize) -> Result<MemoryCharge, EngineError> {
        let cfg = &self.memory_config;
        let overflow =
            || EngineError::CachePressure("Tensor memory estimate overflows usize".into());
        let total = prompt.checked_add(max_tokens).ok_or_else(overflow)?;
        if prompt == 0 || max_tokens == 0 || total > self.context_length {
            return Err(EngineError::TokenLimitExceeded(
                "Request exceeds model context or has zero tokens".into(),
            ));
        }
        let product = |values: &[usize]| {
            values
                .iter()
                .try_fold(1usize, |a, b| a.checked_mul(*b))
                .ok_or_else(overflow)
        };
        let bytes = self.model_template.base_model.dtype.size_in_bytes();
        let kv = product(&[
            2,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim(),
            total,
            bytes,
        ])?;
        let attention = product(&[8, cfg.num_attention_heads, prompt, total, bytes.max(4)])?;
        let hidden = product(&[16, cfg.hidden_size, prompt, bytes])?;
        let mlp = product(&[8, cfg.intermediate_size, prompt, bytes])?;
        let decode = product(&[8, cfg.num_attention_heads, total, cfg.head_dim(), bytes])?;
        let logits = product(&[16, cfg.vocab_size, bytes.max(4)])?;
        let execution = [kv, attention, hidden, mlp, decode, logits]
            .into_iter()
            .try_fold(0usize, |a, b| a.checked_add(b))
            .ok_or_else(overflow)?;
        Ok(MemoryCharge {
            active_kv: kv,
            execution,
            prefix: 0,
        })
    }

    /// Configure a shared budget before admitting any requests.
    pub fn try_set_prefix_cache(
        &mut self,
        prefix_cache: crate::cache::SharedPrefixCache,
    ) -> Result<(), EngineError> {
        if !self.reservations.lock().is_empty() || !self.active_models.lock().is_empty() {
            return Err(EngineError::InvalidRequest(
                "Cannot replace prefix memory budget while requests are active".into(),
            ));
        }
        self.set_prefix_cache(prefix_cache);
        Ok(())
    }

    pub fn set_prefix_cache(&mut self, prefix_cache: crate::cache::SharedPrefixCache) {
        self.memory_budget = prefix_cache.lock().memory_budget();
        self.prefix_cache = Some(prefix_cache);
    }
}

#[async_trait]
impl ModelBackend for CandleBackend {
    fn format_chat(
        &self,
        messages: &[crate::core::types::ChatMessage],
    ) -> Result<String, EngineError> {
        self.chat_template
            .as_ref()
            .map_err(|reason| {
                EngineError::InvalidRequest(format!(
                    "Chat template unavailable: {reason}; use /v1/completions"
                ))
            })?
            .render(messages)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    fn mark_unhealthy(&self) {
        self.healthy.store(false, Ordering::Release);
        self.active_models.lock().clear();
        self.reservations.lock().clear();
        if let Some(prefix) = &self.prefix_cache {
            prefix.lock().clear();
        }
    }

    fn context_length(&self) -> Option<usize> {
        Some(self.context_length)
    }

    async fn init_sequence(&self, prompt: &str) -> Result<Vec<u32>, EngineError> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| EngineError::InvalidRequest(format!("Tokenization error: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn reserve_sequence(
        &self,
        id: &RequestId,
        prompt: usize,
        max_tokens: usize,
    ) -> Result<(), EngineError> {
        let mut reservations = self.reservations.lock();
        if !self.is_healthy() {
            return Err(EngineError::NoHealthyWorkers(
                "Backend quarantined; restart required".into(),
            ));
        }
        if reservations.contains_key(id) {
            return Ok(());
        }
        let charge = self.memory_charge(prompt, max_tokens)?;
        let reservation = if let Some(pc) = &self.prefix_cache {
            pc.lock().reserve_memory(charge)?
        } else {
            self.memory_budget.try_reserve(charge).ok_or_else(|| {
                self.memory_budget.record_rejection();
                EngineError::CachePressure("Tensor memory budget exhausted".into())
            })?
        };
        reservations.insert(id.clone(), reservation);
        Ok(())
    }

    async fn step_batch(
        &self,
        sequences: &mut [ActiveSequence],
    ) -> Result<Vec<StepToken>, EngineError> {
        if sequences.is_empty() {
            return Ok(Vec::new());
        }

        let mut active_models_guard = self.active_models.lock();
        if !self.is_healthy() {
            return Err(EngineError::NoHealthyWorkers(
                "Backend quarantined; restart required".into(),
            ));
        }
        // Keep the execution mutex until health is published so another worker
        // cannot launch a forward after a failing device operation.
        let result = (|| {
            for seq in sequences.iter() {
                self.reserve_sequence(
                    &seq.request_id,
                    seq.prompt_tokens.len(),
                    seq.sampling.max_tokens,
                )?;
            }

            // Other schedulers share this backend. Only explicit request completion or
            // cancellation may remove their state; this batch is not a global live set.

            // Ensure every active sequence has an initialized model
            for seq in sequences.iter() {
                if !active_models_guard
                    .iter()
                    .any(|(id, _)| id == &seq.request_id)
                {
                    let mut cloned = self.model_template.as_ref().clone();
                    cloned.clear_kv_cache();
                    cloned
                        .set_kv_capacity(seq.prompt_tokens.len() + seq.sampling.max_tokens)
                        .map_err(|e| EngineError::BackendError(e.to_string()))?;
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

                    // Check prefix cache for reusable KV activations
                    let matched_kv = if let Some(ref pc) = self.prefix_cache {
                        pc.lock().lookup_kv(&seq.prompt_tokens)
                    } else {
                        None
                    };

                    let last_logits = if let Some(ref snapshot) = matched_kv {
                        let matched_tokens = snapshot.matched_tokens();
                        let cached_logits = snapshot.logits().cloned();
                        // An exact hit without logits must replay its final token from
                        // the preceding KV position, never append it to a complete cache.
                        let restored_tokens = if matched_tokens == seq.prompt_tokens.len()
                            && cached_logits.is_none()
                        {
                            matched_tokens - 1
                        } else {
                            matched_tokens
                        };
                        let snapshot_pairs: Vec<(Tensor, Tensor)> = snapshot
                            .layers()
                            .iter()
                            .map(|pair| {
                                if pair.k.dim(2)? != matched_tokens
                                    || pair.v.dim(2)? != matched_tokens
                                {
                                    candle_core::bail!(
                                        "Prefix snapshot length does not match cache key"
                                    )
                                }
                                Ok((
                                    pair.k.narrow(2, 0, restored_tokens)?,
                                    pair.v.narrow(2, 0, restored_tokens)?,
                                ))
                            })
                            .collect::<candle_core::Result<_>>()
                            .map_err(|e| EngineError::BackendError(e.to_string()))?;
                        model.set_kv_snapshot(&snapshot_pairs).map_err(|e| {
                            EngineError::BackendError(format!("Failed to set KV snapshot: {}", e))
                        })?;

                        if matched_tokens < seq.prompt_tokens.len() {
                            let remaining_tokens = &seq.prompt_tokens[matched_tokens..];
                            let input_tensor = Tensor::new(remaining_tokens, &self.device)
                                .and_then(|t| t.unsqueeze(0))
                                .map_err(|e| {
                                    EngineError::BackendError(format!(
                                        "Prefill remaining tensor creation failed: {}",
                                        e
                                    ))
                                })?;

                            let logits =
                                model.forward(&input_tensor, matched_tokens).map_err(|e| {
                                    EngineError::BackendError(format!(
                                        "Prefill remaining forward pass failed: {}",
                                        e
                                    ))
                                })?;
                            logits.squeeze(0).and_then(|t| t.squeeze(0)).map_err(|e| {
                                EngineError::BackendError(format!(
                                    "Prefill logits extraction failed: {}",
                                    e
                                ))
                            })?
                        } else if let Some(logits) = cached_logits {
                            logits.clone()
                        } else {
                            let last_tok = *seq.prompt_tokens.last().unwrap_or(&1);
                            let input_tensor = Tensor::new(&[last_tok], &self.device)
                                .and_then(|t| t.unsqueeze(0))
                                .map_err(|e| {
                                    EngineError::BackendError(format!(
                                        "Prefill last-token tensor creation failed: {}",
                                        e
                                    ))
                                })?;
                            let logits = model
                                .forward(&input_tensor, seq.prompt_tokens.len() - 1)
                                .map_err(|e| {
                                    EngineError::BackendError(format!(
                                        "Prefill last-token forward pass failed: {}",
                                        e
                                    ))
                                })?;
                            logits.squeeze(0).and_then(|t| t.squeeze(0)).map_err(|e| {
                                EngineError::BackendError(format!(
                                    "Prefill logits extraction failed: {}",
                                    e
                                ))
                            })?
                        }
                    } else {
                        let input_tensor = Tensor::new(seq.prompt_tokens.as_slice(), &self.device)
                            .and_then(|t| t.unsqueeze(0))
                            .map_err(|e| {
                                EngineError::BackendError(format!(
                                    "Prefill tensor creation failed: {}",
                                    e
                                ))
                            })?;

                        let pos = 0;
                        let logits = model.forward(&input_tensor, pos).map_err(|e| {
                            EngineError::BackendError(format!("Prefill forward pass failed: {}", e))
                        })?;

                        logits.squeeze(0).and_then(|t| t.squeeze(0)).map_err(|e| {
                            EngineError::BackendError(format!(
                                "Prefill logits extraction failed: {}",
                                e
                            ))
                        })?
                    };

                    // Insert the completed prompt's KV into prefix_cache for future requests
                    if let Some(ref pc) = self.prefix_cache {
                        let snapshot = model.get_kv_snapshot();
                        let full_kv: Option<Vec<crate::cache::prefix_cache::LayerKVPair>> =
                            snapshot
                                .into_iter()
                                .map(|opt| {
                                    opt.map(|(k, v)| crate::cache::prefix_cache::LayerKVPair {
                                        k,
                                        v,
                                    })
                                })
                                .collect();
                        if let Some(layers) = full_kv {
                            pc.lock()
                                .insert_kv(&seq.prompt_tokens, &layers, Some(&last_logits));
                        }
                    };

                    let seed = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as u64;
                    let next_token_id = sample_token(&last_logits, &seq.sampling, seed)?;

                    let is_eos = self.is_eos_token(next_token_id);
                    let token_text = if is_eos {
                        String::new()
                    } else {
                        decode_next_token(
                            &self.tokenizer,
                            &seq.output_tokens,
                            next_token_id,
                            &seq.output_text,
                        )
                    };

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
                let mut decode_models: Vec<&mut QwenModel> =
                    Vec::with_capacity(decode_indices.len());
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

                let batched_logits = QwenModel::step_decode_batch(
                    &mut decode_models[..],
                    &decode_input_ids,
                    &decode_positions,
                    &self.device,
                )
                .map_err(|e| {
                    EngineError::BackendError(format!("Batched decode step failed: {}", e))
                })?;

                for (k, &idx) in decode_indices.iter().enumerate() {
                    let seq = &sequences[idx];
                    let seq_logits = batched_logits.get(k).map_err(|e| {
                        EngineError::BackendError(format!("Logits extraction failed: {}", e))
                    })?;

                    let seed = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as u64
                        + k as u64;

                    let next_token_id = sample_token(&seq_logits, &seq.sampling, seed)?;

                    let is_eos = self.is_eos_token(next_token_id);
                    let token_text = if is_eos {
                        String::new()
                    } else {
                        decode_next_token(
                            &self.tokenizer,
                            &seq.output_tokens,
                            next_token_id,
                            &seq.output_text,
                        )
                    };

                    results[idx] = Some(StepToken {
                        token_id: next_token_id,
                        text: token_text,
                        is_eos,
                    });
                }
            }

            let final_results: Vec<StepToken> = results.into_iter().map(|r| r.unwrap()).collect();

            // Clean up models for any sequences that just finished (EOS or max_tokens reached)
            for (i, seq) in sequences.iter().enumerate() {
                if final_results[i].is_eos || seq.output_tokens.len() + 1 >= seq.sampling.max_tokens
                {
                    active_models_guard.retain(|(id, _)| id != &seq.request_id);
                    self.reservations.lock().remove(&seq.request_id);
                }
            }

            Ok(final_results)
        })();
        if matches!(&result, Err(EngineError::BackendError(_))) {
            self.healthy.store(false, Ordering::Release);
            active_models_guard.clear();
            self.reservations.lock().clear();
            if let Some(prefix) = &self.prefix_cache {
                prefix.lock().clear();
            }
        }
        result
    }

    fn cancel_sequence(&self, request_id: &RequestId) {
        let mut active_models_guard = self.active_models.lock();
        active_models_guard.retain(|(id, _)| id != request_id);
        self.reservations.lock().remove(request_id);
    }
}

fn decode_next_token(
    tokenizer: &Tokenizer,
    current_output_tokens: &[u32],
    next_token_id: u32,
    current_text: &str,
) -> String {
    let mut all_tokens = Vec::with_capacity(current_output_tokens.len() + 1);
    all_tokens.extend_from_slice(current_output_tokens);
    all_tokens.push(next_token_id);
    match tokenizer.decode(&all_tokens, false) {
        Ok(new_full_text) => {
            let clean_text = if new_full_text.ends_with('\u{FFFD}') {
                new_full_text.trim_end_matches('\u{FFFD}').to_string()
            } else {
                new_full_text
            };
            if clean_text.len() >= current_text.len() && clean_text.starts_with(current_text) {
                clean_text[current_text.len()..].to_string()
            } else {
                tokenizer
                    .decode(&[next_token_id], false)
                    .unwrap_or_else(|_| " ".to_string())
            }
        }
        Err(_) => tokenizer
            .decode(&[next_token_id], false)
            .unwrap_or_else(|_| " ".to_string()),
    }
}

fn sample_token(
    logits: &Tensor,
    sampling: &crate::core::types::SamplingParams,
    seed: u64,
) -> Result<u32, EngineError> {
    let logits = logits
        .to_device(&Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F32))
        .map_err(|e| EngineError::BackendError(format!("Moving logits to CPU failed: {}", e)))?;

    if sampling.temperature <= 0.0 {
        logits
            .argmax(0)
            .and_then(|t| t.to_scalar::<u32>())
            .map_err(|e| EngineError::BackendError(format!("Argmax failed: {}", e)))
    } else {
        let temp = Some(sampling.temperature as f64);
        let top_p = if sampling.top_p > 0.0 && sampling.top_p < 1.0 {
            Some(sampling.top_p as f64)
        } else {
            None
        };
        let mut lp = candle_transformers::generation::LogitsProcessor::new(seed, temp, top_p);
        lp.sample(&logits)
            .map_err(|e| EngineError::BackendError(format!("Sampling failed: {}", e)))
    }
}

fn resolve_model_weights_local(dir: &std::path::Path) -> Option<Vec<std::path::PathBuf>> {
    let single_file = dir.join("model.safetensors");
    if single_file.exists() {
        return Some(vec![single_file]);
    }
    let index_file = dir.join("model.safetensors.index.json");
    if index_file.exists() {
        if let Ok(content) = std::fs::read_to_string(&index_file) {
            if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(weight_map) = json_val.get("weight_map").and_then(|v| v.as_object()) {
                    let mut filenames: Vec<String> = weight_map
                        .values()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect::<std::collections::HashSet<_>>()
                        .into_iter()
                        .collect();
                    filenames.sort();
                    let paths: Vec<std::path::PathBuf> =
                        filenames.into_iter().map(|f| dir.join(f)).collect();
                    if !paths.is_empty() && paths.iter().all(|p| p.exists()) {
                        return Some(paths);
                    }
                }
            }
        }
    }
    None
}

fn resolve_model_weights_hf(
    repo: &hf_hub::api::sync::ApiRepo,
) -> Result<Vec<std::path::PathBuf>, EngineError> {
    if let Ok(w) = repo.get("model.safetensors") {
        return Ok(vec![w]);
    }
    if let Ok(idx_path) = repo.get("model.safetensors.index.json") {
        let content = std::fs::read_to_string(&idx_path).map_err(|e| {
            EngineError::BackendError(format!("Failed to read safetensors index: {}", e))
        })?;
        let json_val: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
            EngineError::BackendError(format!("Failed to parse safetensors index: {}", e))
        })?;
        if let Some(weight_map) = json_val.get("weight_map").and_then(|v| v.as_object()) {
            let mut filenames: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            filenames.sort();
            let mut paths = Vec::with_capacity(filenames.len());
            for f in filenames {
                let p = repo.get(&f).map_err(|e| {
                    EngineError::BackendError(format!("Failed to fetch weight shard {}: {}", f, e))
                })?;
                paths.push(p);
            }
            if !paths.is_empty() {
                return Ok(paths);
            }
        }
    }
    Err(EngineError::BackendError(
        "Could not find model.safetensors or model.safetensors.index.json in model repository"
            .to_string(),
    ))
}

#[cfg(test)]
mod correctness_tests {
    use super::*;
    use crate::cache::create_shared_prefix_cache;
    use crate::core::types::{SamplingParams, TokenEvent};
    use std::time::Duration;

    fn stop_backend() -> CandleBackend {
        let config: QwenConfig = serde_json::from_value(serde_json::json!({
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "max_position_embeddings": 128,
            "hidden_act": "silu", "tie_word_embeddings": true
        }))
        .unwrap();
        let mut model =
            QwenModel::new(&config, VarBuilder::zeros(DType::F32, &Device::Cpu)).unwrap();
        // Equal logits deterministically select token 0, decoded as "User".
        model.lm_head = candle_nn::Linear::new(
            Tensor::zeros((64, 32), DType::F32, &Device::Cpu).unwrap(),
            None,
        );
        let vocab = (0..64)
            .map(|id| {
                (
                    if id == 0 {
                        "User".into()
                    } else {
                        format!("t{id}")
                    },
                    id,
                )
            })
            .collect();
        let tokenizer = Tokenizer::new(
            tokenizers::models::wordlevel::WordLevel::builder()
                .vocab(vocab)
                .unk_token("t1".into())
                .build()
                .unwrap(),
        );
        CandleBackend {
            name: "tiny-stop-test".into(),
            chat_template: Err("CPU fixture".into()),
            healthy: AtomicBool::new(true),
            device: Device::Cpu,
            tokenizer,
            model_template: Arc::new(model),
            active_models: Mutex::new(Vec::new()),
            eos_token_id: u32::MAX,
            context_length: config.max_position_embeddings,
            memory_config: config,
            memory_budget: TensorMemoryBudget::new(
                DEFAULT_TENSOR_MEMORY_BUDGET,
                DEFAULT_EXECUTION_HEADROOM,
            )
            .unwrap(),
            reservations: Mutex::new(HashMap::new()),
            prefix_cache: None,
        }
    }

    async fn complete_with_stop(stop: &str) -> (String, String) {
        let scheduler = Arc::new(crate::scheduler::Scheduler::new(
            4,
            32,
            crate::cache::create_shared_cache(256, 16),
            create_shared_prefix_cache(16, 128),
            Arc::new(stop_backend()),
        ));
        let mut rx = scheduler
            .submit(
                RequestId("stop-regression".into()),
                "t1".into(),
                SamplingParams {
                    temperature: 0.0,
                    top_p: 1.0,
                    max_tokens: 4,
                    stop: vec![stop.into()],
                },
            )
            .await
            .unwrap();
        let task = scheduler.start_loop();
        let mut text = String::new();
        let reason = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                match rx.recv().await.expect("missing terminal event") {
                    TokenEvent::Token(token) => text.push_str(&token),
                    TokenEvent::Done { finish_reason, .. } => break finish_reason,
                    TokenEvent::Error(error) => panic!("{error}"),
                }
            }
        })
        .await
        .unwrap();
        scheduler.shutdown();
        task.await.unwrap();
        (text, reason)
    }

    #[tokio::test]
    async fn stop_spanning_tokens_is_withheld() {
        let (text, reason) = complete_with_stop("User User").await;
        eprintln!("C: multi-token stop='User User', output={text:?}, finish={reason}");
        assert!(
            !text.contains("User User"),
            "multi-token stop leaked: {text:?}"
        );
        assert_eq!(reason, "stop");
    }

    #[tokio::test]
    async fn matched_stop_text_is_withheld() {
        let (text, reason) = complete_with_stop("User").await;
        eprintln!("C: single-token stop='User', output={text:?}, finish={reason}");
        assert_eq!(text, "", "matched stop text must be stripped");
        assert_eq!(reason, "stop");
    }

    fn backend() -> CandleBackend {
        let config: QwenConfig = serde_json::from_value(serde_json::json!({
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "max_position_embeddings": 128,
            "hidden_act": "silu", "tie_word_embeddings": true
        }))
        .unwrap();
        let vars = candle_nn::VarMap::new();
        let model = QwenModel::new(
            &config,
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
        )
        .unwrap();
        CandleBackend {
            name: "tiny-test".into(),
            chat_template: Err("CPU fixture".into()),
            healthy: AtomicBool::new(true),
            device: Device::Cpu,
            tokenizer: Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default()),
            model_template: Arc::new(model),
            active_models: Mutex::new(Vec::new()),
            eos_token_id: u32::MAX,
            context_length: config.max_position_embeddings,
            memory_config: config,
            memory_budget: TensorMemoryBudget::new(
                DEFAULT_TENSOR_MEMORY_BUDGET,
                DEFAULT_EXECUTION_HEADROOM,
            )
            .expect("valid defaults"),
            reservations: Mutex::new(HashMap::new()),
            prefix_cache: None,
        }
    }

    fn sequence(id: &str, tokens: Vec<u32>) -> ActiveSequence {
        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        let mut seq = ActiveSequence::new(
            RequestId(id.into()),
            String::new(),
            SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                max_tokens: 64,
                stop: vec![],
            },
            tx,
        );
        seq.prompt_tokens = tokens;
        seq
    }

    async fn step(backend: &CandleBackend, seq: &mut ActiveSequence) {
        let token = backend
            .step_batch(std::slice::from_mut(seq))
            .await
            .unwrap()
            .remove(0);
        seq.output_tokens.push(token.token_id);
        seq.output_text.push_str(&token.text);
        let models = backend.active_models.lock();
        let model = &models
            .iter()
            .find(|(id, _)| id == &seq.request_id)
            .unwrap()
            .1;
        for pair in model.get_kv_snapshot() {
            let (k, v) = pair.unwrap();
            assert_eq!(
                k.dim(2).unwrap(),
                seq.prompt_tokens.len() + seq.output_tokens.len() - 1
            );
            assert_eq!(k.dims(), v.dims());
        }
    }

    #[tokio::test]
    async fn interleaved_worker_batches_preserve_request_state() {
        let shared = backend();
        let isolated = CandleBackend {
            model_template: shared.model_template.clone(),
            ..backend()
        };
        let mut a = sequence("worker-a", vec![1, 2, 3]);
        let mut b = sequence("worker-b", vec![4, 5, 6, 7, 8]);
        let mut reference = sequence("reference", b.prompt_tokens.clone());
        for _ in 0..3 {
            step(&shared, &mut a).await;
            step(&shared, &mut b).await;
            step(&isolated, &mut reference).await;
        }
        shared.cancel_sequence(&a.request_id);
        for _ in 0..3 {
            step(&shared, &mut b).await;
            step(&isolated, &mut reference).await;
        }
        assert_eq!(b.output_tokens, reference.output_tokens);
        assert_eq!(shared.active_models.lock().len(), 1);
        shared.cancel_sequence(&b.request_id);
        assert!(shared.active_models.lock().is_empty());
    }

    struct YieldingBackend(
        Arc<CandleBackend>,
        Mutex<std::collections::HashMap<RequestId, Vec<u32>>>,
    );

    #[async_trait]
    impl ModelBackend for YieldingBackend {
        fn name(&self) -> &str {
            "tiny-test"
        }
        fn context_length(&self) -> Option<usize> {
            self.0.context_length()
        }
        fn reserve_sequence(
            &self,
            id: &RequestId,
            prompt: usize,
            max_tokens: usize,
        ) -> Result<(), EngineError> {
            self.0.reserve_sequence(id, prompt, max_tokens)
        }
        async fn init_sequence(&self, prompt: &str) -> Result<Vec<u32>, EngineError> {
            Ok(prompt
                .split_whitespace()
                .map(|token| token.parse().unwrap())
                .collect())
        }
        async fn step_batch(
            &self,
            sequences: &mut [ActiveSequence],
        ) -> Result<Vec<StepToken>, EngineError> {
            let result = self.0.step_batch(sequences).await;
            if let Ok(tokens) = &result {
                let mut recorded = self.1.lock();
                for (seq, token) in sequences.iter().zip(tokens) {
                    recorded
                        .entry(seq.request_id.clone())
                        .or_default()
                        .push(token.token_id);
                }
            }
            // Force the two scheduler tasks to interleave on a single CPU thread.
            tokio::task::yield_now().await;
            result
        }
        fn cancel_sequence(&self, id: &RequestId) {
            self.0.cancel_sequence(id);
        }
    }

    #[tokio::test]
    async fn two_schedulers_share_weights_through_completion_and_cancellation() {
        use crate::cache::create_shared_cache;
        use crate::core::types::TokenEvent;
        use crate::scheduler::scheduler::Scheduler;
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            for cancel in [false, true] {
                let shared = Arc::new(backend());
                let isolated = CandleBackend {
                    model_template: shared.model_template.clone(),
                    ..backend()
                };
                let mut expected = sequence("reference", vec![4, 5, 6, 7, 8]);
                for _ in 0..6 {
                    step(&isolated, &mut expected).await;
                }
                let backend = Arc::new(YieldingBackend(
                    shared.clone(),
                    Mutex::new(std::collections::HashMap::new()),
                ));
                let cache = create_shared_cache(256, 16);
                let prefix = create_shared_prefix_cache(16, 128);
                let workers: Vec<_> = (0..2)
                    .map(|_| {
                        Arc::new(Scheduler::new(
                            2,
                            32,
                            cache.clone(),
                            prefix.clone(),
                            backend.clone(),
                        ))
                    })
                    .collect();
                let mut sampling = expected.sampling.clone();
                sampling.max_tokens = if cancel { 100 } else { 2 };
                let mut a = workers[0]
                    .submit(RequestId("a".into()), "1 2 3".into(), sampling.clone())
                    .await
                    .unwrap();
                sampling.max_tokens = 6;
                let mut b = workers[1]
                    .submit(RequestId("b".into()), "4 5 6 7 8".into(), sampling)
                    .await
                    .unwrap();
                let handles: Vec<_> = workers.iter().map(Scheduler::start_loop).collect();
                if cancel {
                    assert!(matches!(a.recv().await, Some(TokenEvent::Token(_))));
                    drop(a);
                } else {
                    loop {
                        match a.recv().await.unwrap() {
                            TokenEvent::Done { .. } => break,
                            TokenEvent::Error(e) => panic!("{e}"),
                            TokenEvent::Token(_) => {}
                        }
                    }
                }
                let mut count = 0;
                loop {
                    match b.recv().await.unwrap() {
                        TokenEvent::Token(_) => count += 1,
                        TokenEvent::Done { usage, .. } => {
                            assert_eq!(usage.completion_tokens, 6);
                            break;
                        }
                        TokenEvent::Error(e) => panic!("{e}"),
                    }
                }
                assert_eq!(count, 6);
                assert_eq!(
                    backend.1.lock()[&RequestId("b".into())],
                    expected.output_tokens
                );
                // Each worker must reclaim its own state without touching its peer.
                while workers.iter().any(|w| w.active_count() != 0) {
                    tokio::task::yield_now().await;
                }
                assert_eq!(cache.lock().allocated_count(), 0);
                assert!(shared.active_models.lock().is_empty());
                for worker in workers {
                    worker.shutdown();
                }
                for handle in handles {
                    handle.await.unwrap();
                }
            }
        })
        .await
        .expect("interleaved schedulers failed to terminate");
    }

    #[tokio::test]
    async fn cached_prefill_matches_reference_at_chunk_boundaries() {
        let mut cached = backend();
        let pc = create_shared_prefix_cache(16, 128);
        cached.set_prefix_cache(pc.clone());
        let plain = CandleBackend {
            model_template: cached.model_template.clone(),
            ..backend()
        };
        let tokens: Vec<u32> = (0..33).map(|i| (i % 61 + 1) as u32).collect();
        let mut prime = sequence("prime", tokens.clone());
        step(&cached, &mut prime).await;
        cached.cancel_sequence(&prime.request_id);
        let before_snapshot = pc.lock().lookup_kv(&tokens).unwrap();
        let before = before_snapshot.layers();
        let before_values: Vec<_> = before
            .iter()
            .map(|p| {
                (
                    p.k.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    p.v.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                )
            })
            .collect();
        // The 16- and 32-token exact matches initially have no stored logits.
        for len in [16, 32, 15, 17, 31, 33, 16, 32] {
            let mut actual = sequence("cached", tokens[..len].to_vec());
            let mut expected = sequence("plain", tokens[..len].to_vec());
            for _ in 0..5 {
                step(&cached, &mut actual).await;
                step(&plain, &mut expected).await;
            }
            assert_eq!(actual.output_tokens, expected.output_tokens, "length {len}");
            // Compare final prompt logits numerically as well as greedy tokens.
            if len % 16 == 0 {
                let snapshot = pc.lock().lookup_kv(&tokens[..len]).unwrap();
                let logits = snapshot.logits();
                let mut model = plain.model_template.as_ref().clone();
                let expected = model
                    .forward(
                        &Tensor::new(&tokens[..len], &Device::Cpu)
                            .unwrap()
                            .unsqueeze(0)
                            .unwrap(),
                        0,
                    )
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .squeeze(0)
                    .unwrap();
                let diff = (logits.unwrap() - expected)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                assert!(diff < 1e-4, "logit difference {diff} at length {len}");
            }
            cached.cancel_sequence(&actual.request_id);
            plain.cancel_sequence(&expected.request_id);
        }
        let after_snapshot = pc.lock().lookup_kv(&tokens).unwrap();
        let after = after_snapshot.layers();
        for (i, pair) in after.iter().enumerate() {
            assert_eq!(
                pair.k.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                before_values[i].0
            );
            assert_eq!(
                pair.v.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                before_values[i].1
            );
        }
    }

    #[test]
    fn malformed_snapshots_are_rejected_before_mutation() {
        let backend = backend();
        let mut model = backend.model_template.as_ref().clone();
        model
            .forward(&Tensor::new(&[[1u32, 2, 3]], &Device::Cpu).unwrap(), 0)
            .unwrap();
        let snapshot: Vec<_> = model.get_kv_snapshot().into_iter().flatten().collect();
        assert!(model.set_kv_snapshot(&snapshot[..1]).is_err());
        let mut invalid = snapshot.clone();
        invalid[1].0 = invalid[1].0.narrow(2, 0, 2).unwrap();
        assert!(model.set_kv_snapshot(&invalid).is_err());
        for pair in model.get_kv_snapshot() {
            assert_eq!(pair.unwrap().0.dim(2).unwrap(), 3);
        }
    }
    #[tokio::test]
    async fn tensor_reservations_follow_physical_capacity_eviction_and_cancellation() {
        use crate::cache::prefix_cache::PrefixCache;
        let mut backend = backend();
        let a_charge = backend.memory_charge(16, 8).unwrap();
        let b_charge = backend.memory_charge(3, 7).unwrap();
        let a_bytes = a_charge.active_kv + a_charge.execution;
        let b_bytes = b_charge.active_kv + b_charge.execution;
        let budget = TensorMemoryBudget::new(64 + a_bytes + b_bytes, 64).unwrap();
        let pc = Arc::new(Mutex::new(PrefixCache::with_memory_budget(
            16,
            8,
            budget.clone(),
        )));
        backend.try_set_prefix_cache(pc.clone()).unwrap();
        let mut a = sequence("a", vec![1; 16]);
        a.sampling.max_tokens = 8;
        let mut b = sequence("b", vec![2; 3]);
        b.sampling.max_tokens = 7;
        backend.reserve_sequence(&a.request_id, 16, 8).unwrap();
        backend.reserve_sequence(&a.request_id, 16, 8).unwrap();
        assert_eq!(budget.stats().active_kv_reserved_bytes, a_charge.active_kv);
        step(&backend, &mut a).await;
        assert!(budget.stats().prefix_retained_bytes > 0);
        backend.reserve_sequence(&b.request_id, 3, 7).unwrap();
        assert_eq!(budget.stats().prefix_retained_bytes, 0);
        assert_eq!(pc.lock().tensor_evictions(), 1);
        step(&backend, &mut b).await;
        assert!(backend
            .reserve_sequence(&RequestId("rejected".into()), 16, 8)
            .is_err());
        assert_eq!(backend.reservations.lock().len(), 2);
        assert_eq!(
            budget.stats().active_kv_reserved_bytes,
            a_charge.active_kv + b_charge.active_kv
        );
        // Inspect underlying buffers, not the smaller visible snapshot shapes.
        for (id, model) in backend.active_models.lock().iter() {
            let capacity = if id == &a.request_id { 24 } else { 10 };
            for pair in model.get_kv_snapshot() {
                let (k, v) = pair.unwrap();
                for tensor in [k, v] {
                    let (storage, _) = tensor.storage_and_layout();
                    match &*storage {
                        candle_core::Storage::Cpu(candle_core::CpuStorage::F32(values)) => {
                            assert_eq!(values.len(), 2 * 8 * capacity)
                        }
                        _ => panic!("expected CPU f32 fixture"),
                    }
                }
            }
        }
        assert!(backend.try_set_prefix_cache(pc.clone()).is_err());
        backend.cancel_sequence(&a.request_id);
        assert_eq!(budget.stats().active_kv_reserved_bytes, b_charge.active_kv);
        backend.cancel_sequence(&b.request_id);
        assert_eq!(budget.stats().reserved_bytes, 64);
        assert!(budget.stats().peak_reserved_bytes <= budget.stats().budget_bytes);
        assert!(backend.active_models.lock().is_empty());
    }

    #[tokio::test]
    async fn scheduler_backend_failure_reclaims_tensor_reservations() {
        use crate::cache::create_shared_cache;
        use crate::core::types::TokenEvent;
        use crate::scheduler::scheduler::Scheduler;
        let shared = Arc::new(backend());
        let wrapped = Arc::new(YieldingBackend(shared.clone(), Mutex::new(HashMap::new())));
        let cache = create_shared_cache(16, 16);
        let scheduler = Arc::new(Scheduler::new(
            1,
            8,
            cache.clone(),
            create_shared_prefix_cache(16, 8),
            wrapped,
        ));
        let mut rx = scheduler
            .submit(
                RequestId("invalid-token".into()),
                "999".into(),
                SamplingParams {
                    max_tokens: 2,
                    ..SamplingParams::default()
                },
            )
            .await
            .unwrap();
        let handle = scheduler.start_loop();
        let event = tokio::time::timeout(std::time::Duration::from_secs(3), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(event, TokenEvent::Error(_)));
        scheduler.shutdown();
        handle.await.unwrap();
        assert_eq!(shared.memory_budget.stats().active_kv_reserved_bytes, 0);
        assert_eq!(shared.memory_budget.stats().execution_reserved_bytes, 0);
        assert!(shared.reservations.lock().is_empty());
        assert!(shared.active_models.lock().is_empty());
        assert_eq!(cache.lock().allocated_count(), 0);
        assert!(!shared.is_healthy());
        assert!(!scheduler.is_healthy());
        assert!(shared
            .reserve_sequence(&RequestId("retry".into()), 1, 2)
            .is_err());
        let mut retry = sequence("retry", vec![1]);
        assert!(matches!(
            shared.step_batch(std::slice::from_mut(&mut retry)).await,
            Err(EngineError::NoHealthyWorkers(_))
        ));
    }
    #[tokio::test]
    async fn qwen_eos_markers_are_suppressed_in_prefill_and_decode() {
        let mut backend = backend();
        let mut probe = sequence("probe", vec![1, 2, 3]);
        for _ in 0..2 {
            step(&backend, &mut probe).await;
        }
        let first = probe.output_tokens[0];
        let second = probe.output_tokens[1];
        backend.cancel_sequence(&probe.request_id);
        for (decode, marker) in [(false, "<|im_end|>"), (true, "<|endoftext|>")] {
            backend.tokenizer = Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default());
            backend.eos_token_id = u32::MAX;
            let mut seq = sequence("eos", vec![1, 2, 3]);
            if decode {
                step(&backend, &mut seq).await;
            }
            let id = if decode { second } else { first };
            let tokenizer = tokenizers::models::wordlevel::WordLevel::builder()
                .vocab(
                    [(marker.to_owned(), id), ("<unk>".into(), 64)]
                        .into_iter()
                        .collect(),
                )
                .unk_token("<unk>".into())
                .build()
                .unwrap();
            backend.tokenizer = Tokenizer::new(tokenizer);
            assert_eq!(backend.tokenizer.decode(&[id], false).unwrap(), marker);
            let output = backend
                .step_batch(std::slice::from_mut(&mut seq))
                .await
                .unwrap()
                .remove(0);
            assert!(output.is_eos);
            assert!(output.text.is_empty(), "leaked {marker}");
            assert!(backend.active_models.lock().is_empty());
            assert_eq!(backend.memory_budget.stats().active_kv_reserved_bytes, 0);
        }
    }
}
