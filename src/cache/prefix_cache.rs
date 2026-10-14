use candle_core::Tensor;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};

use crate::core::constants::{DEFAULT_BLOCK_SIZE, DEFAULT_PREFIX_CACHE_MAX_CHUNKS};

pub const DEFAULT_CHUNK_SIZE: usize = DEFAULT_BLOCK_SIZE;
pub const DEFAULT_MAX_CHUNKS: usize = DEFAULT_PREFIX_CACHE_MAX_CHUNKS;

#[derive(Debug, Clone)]
pub struct LayerKVPair {
    pub k: Tensor,
    pub v: Tensor,
}

#[derive(Debug, Clone)]
pub struct CachedChunk {
    pub tokens: Vec<u32>,
    pub kv_layers: Option<Vec<LayerKVPair>>,
    pub last_logits: Option<Tensor>,
}

#[derive(Debug, Clone)]
pub struct PrefixMatch {
    pub matched_tokens: usize,
    pub chunk_hashes: Vec<String>,
}

#[derive(Debug)]
pub struct PrefixCache {
    chunk_size: usize,
    max_chunks: usize,
    chunks: HashMap<String, CachedChunk>,
    lru_order: VecDeque<String>,
    hits: usize,
    misses: usize,
    tokens_saved: usize,
}

impl PrefixCache {
    pub fn new(chunk_size: usize, max_chunks: usize) -> Self {
        Self {
            chunk_size,
            max_chunks,
            chunks: HashMap::new(),
            lru_order: VecDeque::new(),
            hits: 0,
            misses: 0,
            tokens_saved: 0,
        }
    }

    pub fn compute_chunk_hashes(tokens: &[u32], chunk_size: usize) -> Vec<(String, Vec<u32>)> {
        let num_chunks = tokens.len() / chunk_size;
        let mut result = Vec::with_capacity(num_chunks);
        let mut prev_hash = String::new();

        for i in 0..num_chunks {
            let start = i * chunk_size;
            let chunk_tokens = &tokens[start..start + chunk_size];

            let mut hasher = Sha256::new();
            if !prev_hash.is_empty() {
                hasher.update(prev_hash.as_bytes());
            }
            for &t in chunk_tokens {
                hasher.update(t.to_le_bytes());
            }
            let hash_str = hex::encode(hasher.finalize());
            result.push((hash_str.clone(), chunk_tokens.to_vec()));
            prev_hash = hash_str;
        }

        result
    }

    pub fn lookup(&mut self, tokens: &[u32]) -> PrefixMatch {
        let chunk_pairs = Self::compute_chunk_hashes(tokens, self.chunk_size);
        if chunk_pairs.is_empty() {
            self.misses += 1;
            return PrefixMatch {
                matched_tokens: 0,
                chunk_hashes: Vec::new(),
            };
        }

        let mut matched_chunks = 0;
        let mut matched_hashes = Vec::new();

        for (hash, _) in &chunk_pairs {
            if self.chunks.contains_key(hash) {
                matched_chunks += 1;
                matched_hashes.push(hash.clone());
                // Refresh LRU
                if let Some(pos) = self.lru_order.iter().position(|h| h == hash) {
                    self.lru_order.remove(pos);
                }
                self.lru_order.push_back(hash.clone());
            } else {
                break;
            }
        }

        let matched_tokens = matched_chunks * self.chunk_size;
        if matched_tokens > 0 {
            self.hits += 1;
            self.tokens_saved += matched_tokens;
        } else {
            self.misses += 1;
        }

        PrefixMatch {
            matched_tokens,
            chunk_hashes: matched_hashes,
        }
    }

    pub fn lookup_kv(
        &mut self,
        tokens: &[u32],
    ) -> Option<(usize, Vec<LayerKVPair>, Option<Tensor>)> {
        let chunk_pairs = Self::compute_chunk_hashes(tokens, self.chunk_size);
        if chunk_pairs.is_empty() {
            self.misses += 1;
            return None;
        }

        let mut longest_match: Option<(usize, Vec<LayerKVPair>, Option<Tensor>)> = None;

        for (i, (hash, _)) in chunk_pairs.iter().enumerate() {
            if let Some(chunk) = self.chunks.get(hash) {
                if let Some(ref kv) = chunk.kv_layers {
                    let token_count = (i + 1) * self.chunk_size;
                    let logits = if token_count == tokens.len() {
                        chunk.last_logits.clone()
                    } else {
                        None
                    };
                    longest_match = Some((token_count, kv.clone(), logits));
                }
                // Refresh LRU
                if let Some(pos) = self.lru_order.iter().position(|h| h == hash) {
                    self.lru_order.remove(pos);
                }
                self.lru_order.push_back(hash.clone());
            } else {
                break;
            }
        }

        if let Some((matched_tokens, _, _)) = &longest_match {
            self.hits += 1;
            self.tokens_saved += *matched_tokens;
        } else {
            self.misses += 1;
        }

        longest_match
    }

    pub fn insert(&mut self, tokens: &[u32]) {
        let chunk_pairs = Self::compute_chunk_hashes(tokens, self.chunk_size);
        for (hash, chunk_tokens) in chunk_pairs {
            self.insert_chunk(hash, chunk_tokens, None, None);
        }
    }

    pub fn insert_kv(
        &mut self,
        tokens: &[u32],
        full_kv_layers: &[LayerKVPair],
        last_logits: Option<&Tensor>,
    ) {
        let chunk_pairs = Self::compute_chunk_hashes(tokens, self.chunk_size);
        let num_pairs = chunk_pairs.len();
        for (i, (hash, chunk_tokens)) in chunk_pairs.into_iter().enumerate() {
            let chunk_end = (i + 1) * self.chunk_size;

            let mut sliced_layers = Vec::with_capacity(full_kv_layers.len());
            let mut slice_ok = true;
            for layer in full_kv_layers {
                let k_len = layer.k.dim(2).unwrap_or(0);
                let v_len = layer.v.dim(2).unwrap_or(0);
                if k_len >= chunk_end && v_len >= chunk_end {
                    if let (Ok(k), Ok(v)) = (
                        layer.k.narrow(2, 0, chunk_end).and_then(|t| t.contiguous()),
                        layer.v.narrow(2, 0, chunk_end).and_then(|t| t.contiguous()),
                    ) {
                        sliced_layers.push(LayerKVPair { k, v });
                    } else {
                        slice_ok = false;
                        break;
                    }
                } else {
                    slice_ok = false;
                    break;
                }
            }

            if slice_ok && sliced_layers.len() == full_kv_layers.len() {
                let chunk_logits = if i + 1 == num_pairs && chunk_end == tokens.len() {
                    last_logits.cloned()
                } else {
                    None
                };
                self.insert_chunk(hash, chunk_tokens, Some(sliced_layers), chunk_logits);
            }
        }
    }

    fn insert_chunk(
        &mut self,
        hash: String,
        tokens: Vec<u32>,
        kv: Option<Vec<LayerKVPair>>,
        logits: Option<Tensor>,
    ) {
        if !self.chunks.contains_key(&hash) {
            while self.chunks.len() >= self.max_chunks && !self.lru_order.is_empty() {
                if let Some(evicted) = self.lru_order.pop_front() {
                    self.chunks.remove(&evicted);
                }
            }
            self.chunks.insert(
                hash.clone(),
                CachedChunk {
                    tokens,
                    kv_layers: kv,
                    last_logits: logits,
                },
            );
            self.lru_order.push_back(hash);
        } else if kv.is_some() {
            if let Some(chunk) = self.chunks.get_mut(&hash) {
                if chunk.kv_layers.is_none() {
                    chunk.kv_layers = kv;
                }
                if chunk.last_logits.is_none() && logits.is_some() {
                    chunk.last_logits = logits;
                }
            }
        }
    }

    pub fn stats(&self) -> (usize, usize, usize) {
        (self.hits, self.misses, self.tokens_saved)
    }

    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total > 0 {
            (self.hits as f32 / total as f32) * 100.0
        } else {
            0.0
        }
    }
}

pub type SharedPrefixCache = std::sync::Arc<Mutex<PrefixCache>>;

pub fn create_shared_prefix_cache(chunk_size: usize, max_chunks: usize) -> SharedPrefixCache {
    std::sync::Arc::new(Mutex::new(PrefixCache::new(chunk_size, max_chunks)))
}
