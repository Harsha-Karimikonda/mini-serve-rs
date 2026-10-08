use mini_serve::cache::PrefixCache;

#[test]
fn test_prefix_chunk_hashing() {
    let tokens: Vec<u32> = (0..32).collect();
    let chunks = PrefixCache::compute_chunk_hashes(&tokens, 16);
    assert_eq!(chunks.len(), 2);
    assert_ne!(chunks[0].0, chunks[1].0);
}

#[test]
fn test_prefix_cache_insert_and_lookup() {
    let mut cache = PrefixCache::new(16, 100);
    let prompt: Vec<u32> = (0..48).collect(); // 3 chunks

    // Initial miss
    let initial_lookup = cache.lookup(&prompt);
    assert_eq!(initial_lookup.matched_tokens, 0);

    // Insert
    cache.insert(&prompt);

    // Lookup should match 48 tokens
    let hit_lookup = cache.lookup(&prompt);
    assert_eq!(hit_lookup.matched_tokens, 48);
    assert_eq!(hit_lookup.chunk_hashes.len(), 3);

    let (hits, misses, tokens_saved) = cache.stats();
    assert_eq!(hits, 1);
    assert_eq!(misses, 1);
    assert_eq!(tokens_saved, 48);
}

#[test]
fn test_prefix_cache_lru_eviction() {
    let mut cache = PrefixCache::new(16, 2); // Max 2 chunks
    let seq_a: Vec<u32> = (0..32).collect(); // 2 chunks (fills cache)
    cache.insert(&seq_a);

    let lookup_a = cache.lookup(&seq_a);
    assert_eq!(lookup_a.matched_tokens, 32);

    // Insert seq_b (2 new chunks) -> should evict seq_a chunks
    let seq_b: Vec<u32> = (100..132).collect();
    cache.insert(&seq_b);

    let lookup_b = cache.lookup(&seq_b);
    assert_eq!(lookup_b.matched_tokens, 32);

    // seq_a should now be evicted
    let lookup_a_after = cache.lookup(&seq_a);
    assert_eq!(lookup_a_after.matched_tokens, 0);
}
