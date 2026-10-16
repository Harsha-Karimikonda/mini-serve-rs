//! Conservative reservations, not device allocator measurements. Weights are excluded.
use std::sync::Arc;

use parking_lot::Mutex;
use serde::Serialize;

use crate::core::errors::EngineError;

pub const DEFAULT_TENSOR_MEMORY_BUDGET: usize = 8 * 1024 * 1024 * 1024;
pub const DEFAULT_EXECUTION_HEADROOM: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryCharge {
    pub active_kv: usize,
    pub execution: usize,
    pub prefix: usize,
}

impl MemoryCharge {
    fn total(self) -> Option<usize> {
        self.active_kv
            .checked_add(self.execution)?
            .checked_add(self.prefix)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorMemoryStats {
    pub budget_bytes: usize,
    pub headroom_bytes: usize,
    pub active_kv_reserved_bytes: usize,
    pub execution_reserved_bytes: usize,
    pub prefix_retained_bytes: usize,
    pub reserved_bytes: usize,
    pub available_bytes: usize,
    pub peak_reserved_bytes: usize,
    pub rejections: usize,
}

#[derive(Debug)]
pub struct TensorMemoryBudget {
    stats: Mutex<TensorMemoryStats>,
}

impl TensorMemoryBudget {
    pub fn new(budget_bytes: usize, headroom_bytes: usize) -> Result<Arc<Self>, EngineError> {
        if budget_bytes == 0 || headroom_bytes >= budget_bytes {
            return Err(EngineError::InvalidRequest(
                "tensor memory budget must be positive and greater than execution headroom".into(),
            ));
        }
        Ok(Arc::new(Self {
            stats: Mutex::new(TensorMemoryStats {
                budget_bytes,
                headroom_bytes,
                active_kv_reserved_bytes: 0,
                execution_reserved_bytes: 0,
                prefix_retained_bytes: 0,
                reserved_bytes: headroom_bytes,
                available_bytes: budget_bytes - headroom_bytes,
                peak_reserved_bytes: headroom_bytes,
                rejections: 0,
            }),
        }))
    }

    pub fn stats(&self) -> TensorMemoryStats {
        self.stats.lock().clone()
    }

    /// One lease per owned allocation/reservation; cloned leases are not new charges.
    pub fn try_reserve(self: &Arc<Self>, charge: MemoryCharge) -> Option<Arc<MemoryReservation>> {
        let total = charge.total()?;
        let mut stats = self.stats.lock();
        if total > stats.available_bytes {
            return None;
        }
        stats.active_kv_reserved_bytes += charge.active_kv;
        stats.execution_reserved_bytes += charge.execution;
        stats.prefix_retained_bytes += charge.prefix;
        stats.reserved_bytes += total;
        stats.available_bytes -= total;
        stats.peak_reserved_bytes = stats.peak_reserved_bytes.max(stats.reserved_bytes);
        Some(Arc::new(MemoryReservation {
            budget: self.clone(),
            charge,
        }))
    }

    pub fn record_rejection(&self) {
        self.stats.lock().rejections += 1;
    }
}

#[derive(Debug)]
pub struct MemoryReservation {
    budget: Arc<TensorMemoryBudget>,
    charge: MemoryCharge,
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        let mut stats = self.budget.stats.lock();
        let total = self.charge.total().expect("validated charge");
        stats.active_kv_reserved_bytes -= self.charge.active_kv;
        stats.execution_reserved_bytes -= self.charge.execution;
        stats.prefix_retained_bytes -= self.charge.prefix;
        stats.reserved_bytes -= total;
        stats.available_bytes += total;
    }
}
