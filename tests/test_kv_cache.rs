use mini_serve::cache::KVCache;
use mini_serve::core::errors::EngineError;

#[test]
fn test_cache_allocate_and_release() {
    let mut cache = KVCache::new(100, 16);
    assert_eq!(cache.free_count(), 100);
    assert_eq!(cache.allocated_count(), 0);

    // 32 tokens requires 2 blocks
    let blocks = cache
        .allocate("req-1", 32)
        .expect("allocation should succeed");
    assert_eq!(blocks.len(), 2);
    assert_eq!(cache.free_count(), 98);
    assert_eq!(cache.allocated_count(), 2);

    // Releasing req-1 returns blocks
    let released = cache.release("req-1");
    assert_eq!(released, 2);
    assert_eq!(cache.free_count(), 100);
    assert_eq!(cache.allocated_count(), 0);
}

#[test]
fn test_cache_pressure_error() {
    let mut cache = KVCache::new(4, 16);
    // 5 blocks needed (5 * 16 = 80 tokens)
    let err = cache
        .allocate("req-over", 80)
        .expect_err("should trigger cache pressure");
    match err {
        EngineError::CachePressure(msg) => {
            assert!(msg.contains("Requested 5 blocks"));
        }
        _ => panic!("Expected CachePressure error"),
    }
}

#[test]
fn test_fragmentation_stats() {
    let mut cache = KVCache::new(10, 16);
    let _ = cache.allocate("req-1", 16).unwrap(); // block 0
    let _ = cache.allocate("req-2", 16).unwrap(); // block 1
    let _ = cache.allocate("req-3", 16).unwrap(); // block 2

    // Release middle block to create fragment
    cache.release("req-2");
    let stats = cache.stats();
    assert_eq!(stats.total_blocks, 10);
    assert_eq!(stats.free_blocks, 8);
    assert!(stats.utilization > 0.0);
}
