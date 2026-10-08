use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

use mini_serve::backends::mock::MockBackend;
use mini_serve::cache::{create_shared_cache, create_shared_prefix_cache};
use mini_serve::core::types::{RequestId, SamplingParams};
use mini_serve::scheduler::scheduler::Scheduler;
use mini_serve::scheduler::sequence::TokenEvent;

#[tokio::test]
async fn test_continuous_batching_dynamic_admission_and_early_eos() {
    let cache = create_shared_cache(256, 16);
    let prefix_cache = create_shared_prefix_cache(16, 128);
    let backend = Arc::new(MockBackend::new("mock", 5)); // 5ms per step

    let scheduler = Arc::new(Scheduler::new(4, 32, cache.clone(), prefix_cache, backend));
    scheduler.start_loop();

    // Submit Sequence A (short: 3 tokens)
    let sampling_short = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_tokens: 3,
        stop: vec![],
    };
    let mut rx_a = scheduler
        .submit(
            RequestId("req-a".to_string()),
            "Query A".to_string(),
            sampling_short,
        )
        .await
        .unwrap();

    // Submit Sequence B (long: 15 tokens)
    let sampling_long = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_tokens: 15,
        stop: vec![],
    };
    let mut rx_b = scheduler
        .submit(
            RequestId("req-b".to_string()),
            "Query B".to_string(),
            sampling_long,
        )
        .await
        .unwrap();

    // Sequence A should complete first in ~3 steps (~15-20ms)
    let mut tokens_a = Vec::new();
    while let Some(evt) = rx_a.recv().await {
        match evt {
            TokenEvent::Token(t) => tokens_a.push(t),
            TokenEvent::Done(_) => break,
            TokenEvent::Error(e) => panic!("Error in Seq A: {}", e),
        }
    }
    assert_eq!(tokens_a.len(), 3);

    // After Seq A completes, Seq B should still be streaming
    let mut tokens_b = Vec::new();
    while let Some(evt) = rx_b.recv().await {
        match evt {
            TokenEvent::Token(t) => tokens_b.push(t),
            TokenEvent::Done(_) => break,
            TokenEvent::Error(e) => panic!("Error in Seq B: {}", e),
        }
    }
    assert_eq!(tokens_b.len(), 15);

    // After both finish, cache should fully drain
    sleep(Duration::from_millis(20)).await;
    assert_eq!(cache.lock().allocated_count(), 0);
}

#[tokio::test]
async fn test_continuous_batching_client_cancellation() {
    let cache = create_shared_cache(256, 16);
    let prefix_cache = create_shared_prefix_cache(16, 128);
    let backend = Arc::new(MockBackend::new("mock", 10)); // 10ms per step

    let scheduler = Arc::new(Scheduler::new(4, 32, cache.clone(), prefix_cache, backend));
    scheduler.start_loop();

    let sampling = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_tokens: 50,
        stop: vec![],
    };
    let rx = scheduler
        .submit(
            RequestId("req-abort".to_string()),
            "Query to abort".to_string(),
            sampling,
        )
        .await
        .unwrap();

    // Wait for initial admission and 1 token
    let mut rx = rx;
    let _first_tok = rx.recv().await.unwrap();
    assert!(cache.lock().allocated_count() > 0);

    // Client disconnect: drop receiver!
    drop(rx);

    // Allow scheduler to sweep cancellation on next step
    sleep(Duration::from_millis(40)).await;

    // Cache blocks must be recovered immediately!
    assert_eq!(cache.lock().allocated_count(), 0);
}
