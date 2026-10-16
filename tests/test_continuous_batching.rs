use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

use rlse::backends::mock::MockBackend;
use rlse::cache::{create_shared_cache, create_shared_prefix_cache};
use rlse::core::types::{RequestId, SamplingParams};
use rlse::scheduler::scheduler::Scheduler;
use rlse::scheduler::sequence::TokenEvent;

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
            TokenEvent::Done { .. } => break,
            TokenEvent::Error(e) => panic!("Error in Seq A: {}", e),
        }
    }
    assert_eq!(tokens_a.len(), 3);

    // After Seq A completes, Seq B should still be streaming
    let mut tokens_b = Vec::new();
    while let Some(evt) = rx_b.recv().await {
        match evt {
            TokenEvent::Token(t) => tokens_b.push(t),
            TokenEvent::Done { .. } => break,
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

#[tokio::test]
async fn test_slow_client_does_not_stall_batch() {
    let cache = create_shared_cache(256, 16);
    let prefix_cache = create_shared_prefix_cache(16, 128);
    let backend = Arc::new(EndlessBackend { error_after: None });

    let scheduler = Arc::new(Scheduler::new(4, 32, cache.clone(), prefix_cache, backend));
    scheduler.start_loop();

    // Sequence 1: Slow client with max_tokens: 100, but we NEVER read from its rx channel!
    // Since channel buffer is 64, it will fill up quickly.
    let sampling_slow = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_tokens: 100,
        stop: vec![],
    };
    let mut rx_slow = scheduler
        .submit(
            RequestId("slow-client".to_string()),
            "Slow prompt".to_string(),
            sampling_slow,
        )
        .await
        .unwrap();

    // Sequence 2 keeps consuming while the slow receiver saturates.
    let sampling_fast = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_tokens: 100,
        stop: vec![],
    };
    let mut rx_fast = scheduler
        .submit(
            RequestId("fast-client".to_string()),
            "Fast prompt".to_string(),
            sampling_fast,
        )
        .await
        .unwrap();

    tokio::time::timeout(Duration::from_secs(3), async {
        let mut fast_tokens = 0;
        loop {
            match rx_fast.recv().await.expect("missing fast terminal event") {
                TokenEvent::Token(_) => fast_tokens += 1,
                TokenEvent::Done { finish_reason, .. } => {
                    assert_eq!(finish_reason, "length");
                    break;
                }
                TokenEvent::Error(e) => panic!("Error in fast client: {e}"),
            }
        }
        assert_eq!(fast_tokens, 100);
        let mut slow_tokens = 0;
        loop {
            match rx_slow.recv().await.expect("missing slow terminal event") {
                TokenEvent::Token(_) => slow_tokens += 1,
                TokenEvent::Error(e) => {
                    assert!(e.contains("too slow"));
                    break;
                }
                TokenEvent::Done { .. } => panic!("Truncated response reported success"),
            }
        }
        assert_eq!(slow_tokens, 64);
        while scheduler.active_count() != 0 {
            tokio::task::yield_now().await;
        }
        assert_eq!(cache.lock().allocated_count(), 0);
    })
    .await
    .expect("slow client stalled batch");
    scheduler.shutdown();
}

struct EndlessBackend {
    error_after: Option<usize>,
}

#[async_trait::async_trait]
impl rlse::backends::ModelBackend for EndlessBackend {
    fn name(&self) -> &str {
        "endless"
    }
    async fn init_sequence(&self, _: &str) -> Result<Vec<u32>, rlse::core::errors::EngineError> {
        Ok(vec![1])
    }
    async fn step_batch(
        &self,
        sequences: &mut [rlse::core::types::ActiveSequence],
    ) -> Result<Vec<rlse::backends::StepToken>, rlse::core::errors::EngineError> {
        sleep(Duration::from_millis(1)).await;
        if self
            .error_after
            .is_some_and(|limit| sequences.iter().any(|seq| seq.output_tokens.len() >= limit))
        {
            return Err(rlse::core::errors::EngineError::BackendError(
                "injected failure".into(),
            ));
        }
        Ok(sequences
            .iter()
            .map(|_| rlse::backends::StepToken {
                token_id: 1,
                text: "x".into(),
                is_eos: false,
            })
            .collect())
    }
}

#[tokio::test]
async fn terminal_slot_survives_full_token_buffer() {
    for error_after in [None, Some(64)] {
        let cache = create_shared_cache(256, 16);
        let scheduler = Arc::new(Scheduler::new(
            1,
            32,
            cache.clone(),
            create_shared_prefix_cache(16, 128),
            Arc::new(EndlessBackend { error_after }),
        ));
        let mut rx = scheduler
            .submit(
                RequestId("full".into()),
                "prompt".into(),
                SamplingParams {
                    temperature: 0.0,
                    top_p: 1.0,
                    max_tokens: if error_after.is_some() { 100 } else { 64 },
                    stop: vec![],
                },
            )
            .await
            .unwrap();
        scheduler.start_loop();
        tokio::time::timeout(Duration::from_secs(3), async {
            // Wait until the producer has terminated without consuming any tokens.
            while !rx.is_closed() {
                tokio::task::yield_now().await;
            }
            for _ in 0..64 {
                assert!(matches!(rx.recv().await, Some(TokenEvent::Token(_))));
            }
            match rx.recv().await.expect("reserved terminal missing") {
                TokenEvent::Done {
                    usage,
                    finish_reason,
                } => {
                    assert!(error_after.is_none());
                    assert_eq!(usage.completion_tokens, 64);
                    assert_eq!(finish_reason, "length");
                }
                TokenEvent::Error(e) => {
                    assert!(error_after.is_some());
                    assert!(e.contains("injected failure"));
                }
                TokenEvent::Token(_) => panic!("unexpected token"),
            }
            assert_eq!(cache.lock().allocated_count(), 0);
        })
        .await
        .expect("full buffer lost termination");
        scheduler.shutdown();
    }
}
