use crate::cache::tensor_memory::{
    MemoryCharge, MemoryReservation, TensorMemoryBudget, DEFAULT_EXECUTION_HEADROOM,
    DEFAULT_TENSOR_MEMORY_BUDGET,
};
use crate::core::errors::EngineError;
use candle_core::Tensor;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

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
    reservation: Option<Arc<MemoryReservation>>,
}

/// A borrowed snapshot keeps its storage charged even after cache eviction.
#[derive(Debug, Clone)]
pub struct PrefixSnapshot {
    matched_tokens: usize,
    layers: Vec<LayerKVPair>,
    logits: Option<Tensor>,
    _reservation: Arc<MemoryReservation>,
}

impl PrefixSnapshot {
    pub fn matched_tokens(&self) -> usize {
        self.matched_tokens
    }
    pub fn layers(&self) -> &[LayerKVPair] {
        &self.layers
    }
    pub fn logits(&self) -> Option<&Tensor> {
        self.logits.as_ref()
    }
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
    memory: Arc<TensorMemoryBudget>,
    tensor_evictions: usize,
}

impl PrefixCache {
    pub fn new(chunk_size: usize, max_chunks: usize) -> Self {
        Self::with_memory_budget(
            chunk_size,
            max_chunks,
            TensorMemoryBudget::new(DEFAULT_TENSOR_MEMORY_BUDGET, DEFAULT_EXECUTION_HEADROOM)
                .expect("valid defaults"),
        )
    }

    pub fn with_memory_budget(
        chunk_size: usize,
        max_chunks: usize,
        memory: Arc<TensorMemoryBudget>,
    ) -> Self {
        Self {
            chunk_size,
            max_chunks,
            chunks: HashMap::new(),
            lru_order: VecDeque::new(),
            hits: 0,
            misses: 0,
            tokens_saved: 0,
            memory,
            tensor_evictions: 0,
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

        PrefixMatch {
            matched_tokens,
            chunk_hashes: matched_hashes,
        }
    }

    pub fn lookup_kv(&mut self, tokens: &[u32]) -> Option<PrefixSnapshot> {
        let chunk_pairs = Self::compute_chunk_hashes(tokens, self.chunk_size);
        if chunk_pairs.is_empty() {
            return None;
        }

        let mut longest_match = None;

        for (i, (hash, _)) in chunk_pairs.iter().enumerate() {
            if let Some(chunk) = self.chunks.get(hash) {
                if let Some(ref kv) = chunk.kv_layers {
                    let token_count = (i + 1) * self.chunk_size;
                    let logits = if token_count == tokens.len() {
                        chunk.last_logits.clone()
                    } else {
                        None
                    };
                    longest_match = Some(PrefixSnapshot {
                        matched_tokens: token_count,
                        layers: kv.clone(),
                        logits,
                        _reservation: chunk
                            .reservation
                            .as_ref()
                            .expect("KV storage is charged")
                            .clone(),
                    });
                }
                // Refresh LRU
                if let Some(pos) = self.lru_order.iter().position(|h| h == hash) {
                    self.lru_order.remove(pos);
                }
                self.lru_order.push_back(hash.clone());
            }
            // Each chained key authenticates its complete prefix. Intermediate
            // snapshots may have been evicted while a longer one remains.
        }

        longest_match
    }

    pub fn insert(&mut self, tokens: &[u32]) {
        let chunk_pairs = Self::compute_chunk_hashes(tokens, self.chunk_size);
        for (hash, chunk_tokens) in chunk_pairs {
            self.insert_chunk(hash, chunk_tokens);
        }
    }

    pub fn memory_budget(&self) -> Arc<TensorMemoryBudget> {
        self.memory.clone()
    }

    pub fn tensor_evictions(&self) -> usize {
        self.tensor_evictions
    }

    fn evict_oldest(&mut self) -> bool {
        if let Some(hash) = self.lru_order.pop_front() {
            if self
                .chunks
                .remove(&hash)
                .is_some_and(|c| c.kv_layers.is_some())
            {
                self.tensor_evictions += 1;
            }
            true
        } else {
            false
        }
    }

    /// Admission and cache insertion share a budget across all workers.
    pub fn reserve_memory(
        &mut self,
        charge: MemoryCharge,
    ) -> Result<Arc<MemoryReservation>, EngineError> {
        loop {
            if let Some(reservation) = self.memory.try_reserve(charge) {
                return Ok(reservation);
            }
            if !self.evict_oldest() {
                self.memory.record_rejection();
                return Err(EngineError::CachePressure(
                    "Tensor memory budget exhausted".into(),
                ));
            }
        }
    }

    pub fn clear(&mut self) {
        while self.evict_oldest() {}
    }

    pub fn insert_kv(
        &mut self,
        tokens: &[u32],
        full_kv_layers: &[LayerKVPair],
        last_logits: Option<&Tensor>,
    ) {
        if self.max_chunks == 0 || full_kv_layers.is_empty() {
            return;
        }
        let chunk_pairs = Self::compute_chunk_hashes(tokens, self.chunk_size);
        for (i, (hash, chunk_tokens)) in chunk_pairs.into_iter().enumerate() {
            let chunk_end = (i + 1) * self.chunk_size;
            let logits = if chunk_end == tokens.len() {
                last_logits
            } else {
                None
            };
            if self.chunks.get(&hash).is_some_and(|chunk| {
                chunk.kv_layers.is_some() && (logits.is_none() || chunk.last_logits.is_some())
            }) {
                continue;
            }
            // Reserve before making owned copies. A narrow/contiguous view can
            // retain a much larger active KV buffer, including for one KV head.
            let bytes = (|| -> Option<usize> {
                let mut bytes = 0usize;
                for layer in full_kv_layers {
                    for t in [&layer.k, &layer.v] {
                        let (batch, heads, len, dim) = t.dims4().ok()?;
                        if len < chunk_end {
                            return None;
                        }
                        let size = batch
                            .checked_mul(heads)?
                            .checked_mul(chunk_end)?
                            .checked_mul(dim)?
                            .checked_mul(t.dtype().size_in_bytes())?;
                        bytes = bytes.checked_add(size)?;
                    }
                }
                if let Some(t) = logits {
                    bytes = bytes
                        .checked_add(t.elem_count().checked_mul(t.dtype().size_in_bytes())?)?;
                }
                Some(bytes)
            })();
            let Some(bytes) = bytes else {
                continue;
            };
            let Ok(reservation) = self.reserve_memory(MemoryCharge {
                prefix: bytes,
                ..MemoryCharge::default()
            }) else {
                continue;
            };
            let copied = (|| -> candle_core::Result<_> {
                let mut layers = Vec::with_capacity(full_kv_layers.len());
                for layer in full_kv_layers {
                    layers.push(LayerKVPair {
                        k: layer
                            .k
                            .narrow(2, 0, chunk_end)?
                            .force_contiguous()?
                            .detach(),
                        v: layer
                            .v
                            .narrow(2, 0, chunk_end)?
                            .force_contiguous()?
                            .detach(),
                    });
                }
                let logits = logits
                    .map(|t| t.force_contiguous().map(|t| t.detach()))
                    .transpose()?;
                Ok((layers, logits))
            })();
            if let Ok((layers, logits)) = copied {
                // Replace an older metadata-only entry or snapshot lacking logits.
                self.chunks.remove(&hash);
                self.lru_order.retain(|h| h != &hash);
                while self.chunks.len() >= self.max_chunks {
                    self.evict_oldest();
                }
                self.chunks.insert(
                    hash.clone(),
                    CachedChunk {
                        tokens: chunk_tokens,
                        kv_layers: Some(layers),
                        last_logits: logits,
                        reservation: Some(reservation),
                    },
                );
                self.lru_order.push_back(hash);
            }
        }
    }

    fn insert_chunk(&mut self, hash: String, tokens: Vec<u32>) {
        if self.max_chunks == 0 || self.chunks.contains_key(&hash) {
            return;
        }
        while self.chunks.len() >= self.max_chunks {
            self.evict_oldest();
        }
        self.chunks.insert(
            hash.clone(),
            CachedChunk {
                tokens,
                kv_layers: None,
                last_logits: None,
                reservation: None,
            },
        );
        self.lru_order.push_back(hash);
    }

    /// Record successful execution reuse, never speculative metadata lookups.
    pub fn record_reuse(&mut self, skipped_tokens: usize) {
        if skipped_tokens == 0 {
            self.misses += 1;
        } else {
            self.hits += 1;
            self.tokens_saved += skipped_tokens;
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
