use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;

use async_trait::async_trait;
use rlse::{
    backends::{ModelBackend, StepToken},
    cache::{create_shared_cache, create_shared_prefix_cache},
    core::{
        config::Settings,
        errors::EngineError,
        types::{ActiveSequence, RequestId, SamplingParams, TokenEvent},
    },
    scheduler::scheduler::Scheduler,
};

#[test]
fn configuration_boundaries() {
    assert!(Settings::default().validate().is_ok());
    let invalid = [
        Settings {
            block_size: 0,
            ..Settings::default()
        },
        Settings {
            kv_cache_blocks: 0,
            ..Settings::default()
        },
        Settings {
            max_batch_size: 0,
            ..Settings::default()
        },
        Settings {
            max_waiting_requests: 0,
            ..Settings::default()
        },
        Settings {
            num_workers: 0,
            ..Settings::default()
        },
        Settings {
            block_size: 1,
            kv_cache_blocks: 1,
            ..Settings::default()
        },
        Settings {
            block_size: usize::MAX,
            kv_cache_blocks: 2,
            ..Settings::default()
        },
        Settings {
            tensor_memory_budget_bytes: 0,
            ..Settings::default()
        },
        Settings {
            tensor_memory_budget_bytes: 100,
            execution_headroom_bytes: 100,
            ..Settings::default()
        },
        Settings {
            device: "metla".into(),
            ..Settings::default()
        },
    ];
    for settings in invalid {
        assert!(settings.validate().is_err(), "accepted {settings:?}");
    }
    for device in ["cpu", "auto", "metal", "cuda"] {
        let expected = matches!(device, "cpu" | "auto")
            || (device == "metal" && cfg!(feature = "metal"))
            || (device == "cuda" && cfg!(feature = "cuda"));
        assert_eq!(
            Settings {
                device: device.into(),
                ..Settings::default()
            }
            .validate()
            .is_ok(),
            expected
        );
    }
}

#[test]
fn sampling_boundaries() {
    for temperature in [0.0, 1.0, 2.0] {
        for top_p in [f32::MIN_POSITIVE, 0.5, 1.0] {
            assert!(SamplingParams {
                temperature,
                top_p,
                ..SamplingParams::default()
            }
            .validate()
            .is_ok());
        }
    }
    for temperature in [-0.1, 2.1, f32::NAN, f32::INFINITY] {
        assert!(SamplingParams {
            temperature,
            ..SamplingParams::default()
        }
        .validate()
        .is_err());
    }
    for top_p in [-0.1, 0.0, 1.1, f32::NAN, f32::INFINITY] {
        assert!(SamplingParams {
            top_p,
            ..SamplingParams::default()
        }
        .validate()
        .is_err());
    }
    assert!(SamplingParams {
        max_tokens: 0,
        ..SamplingParams::default()
    }
    .validate()
    .is_err());
    assert!(SamplingParams {
        stop: vec![String::new()],
        ..SamplingParams::default()
    }
    .validate()
    .is_err());
}

struct LimitedBackend {
    steps: AtomicUsize,
    cancelled: AtomicUsize,
}

#[async_trait]
impl ModelBackend for LimitedBackend {
    fn name(&self) -> &str {
        "limited"
    }
    fn context_length(&self) -> Option<usize> {
        Some(8)
    }
    async fn init_sequence(&self, prompt: &str) -> Result<Vec<u32>, EngineError> {
        Ok(vec![1; prompt.split_whitespace().count()])
    }
    async fn step_batch(
        &self,
        sequences: &mut [ActiveSequence],
    ) -> Result<Vec<StepToken>, EngineError> {
        self.steps.fetch_add(sequences.len(), Ordering::SeqCst);
        tokio::task::yield_now().await;
        Ok(sequences
            .iter()
            .map(|_| StepToken {
                token_id: 1,
                text: "x".into(),
                is_eos: false,
            })
            .collect())
    }
    fn cancel_sequence(&self, _: &RequestId) {
        self.cancelled.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn reject_before_queue_or_execution_and_accept_exact_limit() {
    let backend = Arc::new(LimitedBackend {
        steps: AtomicUsize::new(0),
        cancelled: AtomicUsize::new(0),
    });
    let cache = create_shared_cache(4, 2);
    let scheduler = Arc::new(Scheduler::new(
        2,
        32,
        cache.clone(),
        create_shared_prefix_cache(2, 8),
        backend.clone(),
    ));
    for (prompt, max_tokens) in [
        ("one", 0),
        ("one", usize::MAX),
        ("one two three four five six seven eight nine", 1),
        ("one two", 7),
        (" ", 1),
    ] {
        assert!(scheduler
            .submit(
                RequestId::new_completion(),
                prompt.into(),
                SamplingParams {
                    max_tokens,
                    ..SamplingParams::default()
                }
            )
            .await
            .is_err());
        assert_eq!(scheduler.queue_depth(), 0);
        assert_eq!(cache.lock().allocated_count(), 0);
        assert_eq!(backend.steps.load(Ordering::SeqCst), 0);
    }
    let mut rx = scheduler
        .submit(
            RequestId::new_completion(),
            "one two three four".into(),
            SamplingParams {
                max_tokens: 4,
                ..SamplingParams::default()
            },
        )
        .await
        .unwrap();
    let handle = scheduler.start_loop();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            match rx.recv().await.expect("missing terminal") {
                TokenEvent::Token(_) => {}
                TokenEvent::Done {
                    usage,
                    finish_reason,
                } => {
                    assert_eq!(usage.total_tokens, 8);
                    assert_eq!(finish_reason, "length");
                    break;
                }
                TokenEvent::Error(e) => panic!("{e}"),
            }
        }
    })
    .await
    .unwrap();
    scheduler.shutdown();
    handle.await.unwrap();
    assert_eq!(cache.lock().allocated_count(), 0);
}

#[tokio::test]
async fn combined_length_exceeds_cache_and_contention_reports_error() {
    let backend = Arc::new(LimitedBackend {
        steps: AtomicUsize::new(0),
        cancelled: AtomicUsize::new(0),
    });
    let cache = create_shared_cache(2, 2);
    let scheduler = Arc::new(Scheduler::new(
        2,
        32,
        cache.clone(),
        create_shared_prefix_cache(2, 8),
        backend.clone(),
    ));
    assert!(matches!(
        scheduler
            .submit(
                RequestId::new_completion(),
                "one two".into(),
                SamplingParams {
                    max_tokens: 3,
                    ..SamplingParams::default()
                }
            )
            .await,
        Err(EngineError::CachePressure(_))
    ));
    let mut receivers = Vec::new();
    for _ in 0..2 {
        receivers.push(
            scheduler
                .submit(
                    RequestId::new_completion(),
                    "one".into(),
                    SamplingParams {
                        max_tokens: 3,
                        ..SamplingParams::default()
                    },
                )
                .await
                .unwrap(),
        );
    }
    let handle = scheduler.start_loop();
    tokio::time::timeout(Duration::from_secs(2), async {
        let mut errors = 0;
        for mut rx in receivers {
            loop {
                match rx.recv().await.expect("missing terminal") {
                    TokenEvent::Token(_) => {}
                    TokenEvent::Error(e) => {
                        assert!(e.contains("KV cache"));
                        errors += 1;
                        break;
                    }
                    TokenEvent::Done { usage, .. } => {
                        assert_eq!(usage.completion_tokens, 3);
                        break;
                    }
                }
            }
        }
        assert!(errors >= 1);
    })
    .await
    .unwrap();
    scheduler.shutdown();
    handle.await.unwrap();
    assert_eq!(cache.lock().allocated_count(), 0);
    assert_eq!(backend.cancelled.load(Ordering::SeqCst), 2);
    assert!(backend.steps.load(Ordering::SeqCst) < 6);
}

#[tokio::test]
async fn invalid_scheduler_settings_cannot_queue_forever() {
    for (batch_size, queue_size, blocks, block_size) in
        [(0, 1, 1, 2), (1, 0, 1, 2), (1, 1, 0, 2), (1, 1, 1, 0)]
    {
        let cache = create_shared_cache(blocks, block_size);
        let scheduler = Scheduler::new(
            batch_size,
            queue_size,
            cache.clone(),
            create_shared_prefix_cache(2, 8),
            Arc::new(LimitedBackend {
                steps: AtomicUsize::new(0),
                cancelled: AtomicUsize::new(0),
            }),
        );
        assert!(scheduler
            .submit(
                RequestId::new_completion(),
                "one".into(),
                SamplingParams {
                    max_tokens: 1,
                    ..SamplingParams::default()
                }
            )
            .await
            .is_err());
        assert_eq!(scheduler.queue_depth(), 0);
        assert_eq!(cache.lock().allocated_count(), 0);
    }
}
