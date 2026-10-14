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
    prefix_cache: Option<crate::cache::SharedPrefixCache>,
}

impl CandleBackend {
    pub fn load_hf(model_id: &str, device_preference: &str) -> Result<Self, EngineError> {
        info!("Initializing CandleBackend for model: {}", model_id);

        let local_path = std::path::Path::new(model_id);
        let normalized_slug = model_id.replace('/', "-");
        let local_slug_path = std::path::Path::new("models").join(&normalized_slug);
        let local_models_path = std::path::Path::new("models").join(model_id);

        let (tokenizer_path, config_path, model_paths) = if local_path.is_dir()
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
            )
        } else if local_slug_path.is_dir()
            && (local_slug_path.join("model.safetensors").exists()
                || local_slug_path.join("model.safetensors.index.json").exists())
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
            )
        } else if local_models_path.is_dir()
            && (local_models_path.join("model.safetensors").exists()
                || local_models_path.join("model.safetensors.index.json").exists())
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
            (tok, cfg, weights)
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
            "cuda" => Device::new_cuda(0).map_err(|e| {
                EngineError::BackendError(format!("CUDA device initialization failed: {}", e))
            })?,
            #[cfg(not(feature = "cuda"))]
            "cuda" => {
                return Err(EngineError::BackendError(
                    "CUDA device requested but 'cuda' feature is not compiled into mini-serve".to_string(),
                ));
            }
            _ => Device::Cpu,
        };
        info!("Using device: {:?}", device);

        let dtype = match device {
            Device::Cpu => DType::F32,
            _ => DType::F16,
        };

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
            device,
            tokenizer,
            model_template: Arc::new(model),
            active_models: Mutex::new(Vec::new()),
            eos_token_id,
            prefix_cache: None,
        })
    }

    pub fn set_prefix_cache(&mut self, prefix_cache: crate::cache::SharedPrefixCache) {
        self.prefix_cache = Some(prefix_cache);
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

                // Check prefix cache for reusable KV activations
                let matched_kv = if let Some(ref pc) = self.prefix_cache {
                    pc.lock().lookup_kv(&seq.prompt_tokens)
                } else {
                    None
                };

                let last_logits = if let Some((matched_tokens, kv_layers, cached_logits)) = matched_kv {
                    let snapshot_pairs: Vec<(Tensor, Tensor)> = kv_layers
                        .into_iter()
                        .map(|pair| (pair.k, pair.v))
                        .collect();
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

                        let logits = model
                            .forward(&input_tensor, matched_tokens)
                            .map_err(|e| {
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
                        logits
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
                        EngineError::BackendError(format!("Prefill logits extraction failed: {}", e))
                    })?
                };

                // Insert the completed prompt's KV into prefix_cache for future requests
                if let Some(ref pc) = self.prefix_cache {
                    let snapshot = model.get_kv_snapshot();
                    let full_kv: Option<Vec<crate::cache::prefix_cache::LayerKVPair>> = snapshot
                        .into_iter()
                        .map(|opt| opt.map(|(k, v)| crate::cache::prefix_cache::LayerKVPair { k, v }))
                        .collect();
                    if let Some(layers) = full_kv {
                        pc.lock().insert_kv(&seq.prompt_tokens, &layers, Some(&last_logits));
                    }
                };

                let seed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;
                let next_token_id = sample_token(&last_logits, &seq.sampling, seed)?;

                let token_text = decode_next_token(
                    &self.tokenizer,
                    &seq.output_tokens,
                    next_token_id,
                    &seq.output_text,
                );

                let is_stop_token = seq
                    .sampling
                    .stop
                    .iter()
                    .any(|s| !s.is_empty() && (token_text == *s || token_text.contains(s)));
                let is_eos = next_token_id == self.eos_token_id || is_stop_token;

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

            let batched_logits = QwenModel::step_decode_batch(
                &mut decode_models[..],
                &decode_input_ids,
                &decode_positions,
                &self.device,
            )
            .map_err(|e| EngineError::BackendError(format!("Batched decode step failed: {}", e)))?;

            for (k, &idx) in decode_indices.iter().enumerate() {
                let seq = &sequences[idx];
                let seq_logits = batched_logits
                    .get(k)
                    .map_err(|e| EngineError::BackendError(format!("Logits extraction failed: {}", e)))?;

                let seed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64
                    + k as u64;

                let next_token_id = sample_token(&seq_logits, &seq.sampling, seed)?;

                let token_text = decode_next_token(
                    &self.tokenizer,
                    &seq.output_tokens,
                    next_token_id,
                    &seq.output_text,
                );

                let is_stop_token = seq
                    .sampling
                    .stop
                    .iter()
                    .any(|s| !s.is_empty() && (token_text == *s || token_text.contains(s)));
                let is_eos = next_token_id == self.eos_token_id || is_stop_token;

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
            if final_results[i].is_eos || seq.output_tokens.len() + 1 >= seq.sampling.max_tokens {
                active_models_guard.retain(|(id, _)| id != &seq.request_id);
            }
        }

        Ok(final_results)
    }

    fn cancel_sequence(&self, request_id: &RequestId) {
        let mut active_models_guard = self.active_models.lock();
        active_models_guard.retain(|(id, _)| id != request_id);
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
