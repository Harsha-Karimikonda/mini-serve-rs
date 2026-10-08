use crate::core::errors::EngineError;
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_blocks: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub utilization: f32,
    pub fragmentation: f32,
}

#[derive(Debug)]
pub struct KVCache {
    total_blocks: usize,
    block_size: usize,
    free_blocks: Vec<usize>,
    allocations: HashMap<String, Vec<usize>>,
}

impl KVCache {
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        let mut free = Vec::with_capacity(total_blocks);
        // Push in reverse so pop() yields 0, 1, 2, ...
        for i in (0..total_blocks).rev() {
            free.push(i);
        }
        Self {
            total_blocks,
            block_size,
            free_blocks: free,
            allocations: HashMap::new(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    pub fn free_count(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn allocated_count(&self) -> usize {
        self.total_blocks - self.free_blocks.len()
    }

    pub fn blocks_needed(&self, tokens: usize) -> usize {
        if tokens == 0 {
            1
        } else {
            tokens.div_ceil(self.block_size)
        }
    }

    pub fn allocate(
        &mut self,
        request_id: &str,
        num_tokens: usize,
    ) -> Result<Vec<usize>, EngineError> {
        let needed = self.blocks_needed(num_tokens);
        if needed > self.free_blocks.len() {
            return Err(EngineError::CachePressure(format!(
                "Requested {} blocks ({} tokens), but only {} free blocks remain",
                needed,
                num_tokens,
                self.free_blocks.len()
            )));
        }

        let mut blocks = Vec::with_capacity(needed);
        for _ in 0..needed {
            blocks.push(self.free_blocks.pop().unwrap());
        }

        self.allocations
            .insert(request_id.to_string(), blocks.clone());
        Ok(blocks)
    }

    pub fn release(&mut self, request_id: &str) -> usize {
        if let Some(blocks) = self.allocations.remove(request_id) {
            let count = blocks.len();
            for b in blocks {
                self.free_blocks.push(b);
            }
            count
        } else {
            0
        }
    }

    pub fn stats(&self) -> CacheStats {
        let free = self.free_blocks.len();
        let allocated = self.total_blocks - free;
        let utilization = if self.total_blocks > 0 {
            allocated as f32 / self.total_blocks as f32
        } else {
            0.0
        };

        // Fragmentation: measure contiguous free blocks vs isolated blocks
        let free_set: HashSet<usize> = self.free_blocks.iter().copied().collect();
        let mut contiguous_runs = 0usize;
        for i in 0..self.total_blocks {
            if free_set.contains(&i) && (i == 0 || !free_set.contains(&(i - 1))) {
                contiguous_runs += 1;
            }
        }

        let fragmentation = if free > 0 && contiguous_runs > 1 {
            (contiguous_runs as f32 - 1.0) / (free as f32)
        } else {
            0.0
        };

        CacheStats {
            total_blocks: self.total_blocks,
            allocated_blocks: allocated,
            free_blocks: free,
            utilization: (utilization * 1000.0).round() / 1000.0,
            fragmentation: (fragmentation.min(1.0) * 1000.0).round() / 1000.0,
        }
    }
}

pub type SharedKVCache = std::sync::Arc<Mutex<KVCache>>;

pub fn create_shared_cache(total_blocks: usize, block_size: usize) -> SharedKVCache {
    std::sync::Arc::new(Mutex::new(KVCache::new(total_blocks, block_size)))
}
