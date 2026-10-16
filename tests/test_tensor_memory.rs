use std::sync::Arc;

use candle_core::{CpuStorage, Device, Storage, Tensor};
use rlse::cache::{
    prefix_cache::{LayerKVPair, PrefixCache},
    tensor_memory::{MemoryCharge, TensorMemoryBudget},
};

#[test]
fn shared_leases_and_concurrent_reservations_stay_within_budget() {
    assert!(TensorMemoryBudget::new(0, 0).is_err());
    assert!(TensorMemoryBudget::new(100, 100).is_err());
    let budget = TensorMemoryBudget::new(108, 8).unwrap();
    let barrier = Arc::new(std::sync::Barrier::new(16));
    let threads: Vec<_> = (0..16)
        .map(|_| {
            let budget = budget.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let lease = budget.try_reserve(MemoryCharge {
                    active_kv: 20,
                    execution: 5,
                    prefix: 0,
                });
                barrier.wait();
                assert_eq!(budget.stats().reserved_bytes, 108);
                // Keep all successful leases alive until each thread observed the peak.
                barrier.wait();
                lease.is_some()
            })
        })
        .collect();
    assert_eq!(
        threads
            .into_iter()
            .map(|t| t.join().unwrap())
            .filter(|ok| *ok)
            .count(),
        4
    );
    assert_eq!(budget.stats().reserved_bytes, 8);
    let lease = budget
        .try_reserve(MemoryCharge {
            prefix: 32,
            ..MemoryCharge::default()
        })
        .unwrap();
    let shared = lease.clone();
    drop(lease);
    assert_eq!(budget.stats().prefix_retained_bytes, 32);
    drop(shared);
    assert_eq!(budget.stats().available_bytes, 100);
    assert!(budget
        .try_reserve(MemoryCharge {
            active_kv: usize::MAX,
            execution: 1,
            prefix: 0
        })
        .is_none());
}

fn layers() -> Vec<LayerKVPair> {
    vec![LayerKVPair {
        k: Tensor::ones((1, 1, 128, 2), candle_core::DType::F32, &Device::Cpu).unwrap(),
        v: Tensor::zeros((1, 1, 128, 2), candle_core::DType::F32, &Device::Cpu).unwrap(),
    }]
}

#[test]
fn prefix_copies_are_compact_independent_and_charged_until_last_borrow_drops() {
    let budget = TensorMemoryBudget::new(900, 64).unwrap();
    let mut cache = PrefixCache::with_memory_budget(16, 8, budget.clone());
    let layers = layers();
    let prompt: Vec<u32> = (0..32).collect();
    let logits = Tensor::ones(4, candle_core::DType::F32, &Device::Cpu).unwrap();
    cache.insert_kv(&prompt, &layers, Some(&logits));
    assert_eq!(budget.stats().prefix_retained_bytes, 256 + 512 + 16);
    let snapshot = cache.lookup_kv(&prompt).unwrap();
    let borrowed = snapshot.clone();
    assert_eq!(budget.stats().prefix_retained_bytes, 784);
    for tensor in [&snapshot.layers()[0].k, &snapshot.layers()[0].v] {
        let (storage, _) = tensor.storage_and_layout();
        match &*storage {
            Storage::Cpu(CpuStorage::F32(values)) => assert_eq!(values.len(), 32 * 2),
            _ => panic!("expected compact CPU f32 storage"),
        }
    }
    // The original 128-token buffer can change without modifying the cached prefix.
    layers[0]
        .k
        .slice_set(
            &Tensor::zeros((1, 1, 128, 2), candle_core::DType::F32, &Device::Cpu).unwrap(),
            2,
            0,
        )
        .unwrap();
    assert!(snapshot.layers()[0]
        .k
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .all(|v| *v == 1.0));
    cache.clear();
    assert_eq!(budget.stats().prefix_retained_bytes, 528);
    drop(snapshot);
    assert_eq!(budget.stats().prefix_retained_bytes, 528);
    drop(borrowed);
    assert_eq!(budget.stats().reserved_bytes, 64);
}

#[test]
fn admission_evicts_prefixes_and_cannot_reclaim_borrowed_storage() {
    let budget = TensorMemoryBudget::new(1100, 64).unwrap();
    let mut cache = PrefixCache::with_memory_budget(16, 8, budget.clone());
    let prompt: Vec<u32> = (0..32).collect();
    cache.insert_kv(&prompt, &layers(), None);
    let snapshot = cache.lookup_kv(&prompt).unwrap();
    assert!(cache
        .reserve_memory(MemoryCharge {
            active_kv: 600,
            ..MemoryCharge::default()
        })
        .is_err());
    assert_eq!(cache.tensor_evictions(), 2);
    assert_eq!(budget.stats().prefix_retained_bytes, 512);
    assert_eq!(budget.stats().rejections, 1);
    drop(snapshot);
    let active = cache
        .reserve_memory(MemoryCharge {
            active_kv: 600,
            execution: 100,
            prefix: 0,
        })
        .unwrap();
    assert_eq!(budget.stats().active_kv_reserved_bytes, 600);
    cache.insert_kv(&prompt, &layers(), None);
    assert!(budget.stats().reserved_bytes <= 1100);
    drop(active);
    cache.clear();
    assert_eq!(budget.stats().reserved_bytes, 64);
    assert!(budget.stats().peak_reserved_bytes <= 1100);
}
