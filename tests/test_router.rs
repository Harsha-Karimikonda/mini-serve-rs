use rlse::backends::mock::MockBackend;
use rlse::cache::{create_shared_cache, create_shared_prefix_cache};
use rlse::core::types::{RequestId, SamplingParams};
use rlse::routing::router::{Router, WorkerHandle};
use rlse::scheduler::scheduler::Scheduler;
use std::sync::Arc;

#[tokio::test]
async fn test_latency_aware_routing_balances_traffic() {
    let cache = create_shared_cache(256, 16);
    let prefix_cache = create_shared_prefix_cache(16, 128);
    let backend = Arc::new(MockBackend::new("mock", 1));

    let sched_1 = Arc::new(Scheduler::new(
        4,
        32,
        cache.clone(),
        prefix_cache.clone(),
        backend.clone(),
    ));
    let sched_2 = Arc::new(Scheduler::new(
        4,
        32,
        cache.clone(),
        prefix_cache.clone(),
        backend.clone(),
    ));

    sched_1.start_loop();
    sched_2.start_loop();

    let router = Router::new(vec![
        WorkerHandle::new("worker-1", sched_1.clone()),
        WorkerHandle::new("worker-2", sched_2.clone()),
    ]);

    // Choose worker when both empty -> worker-1
    let chosen = router.choose_worker().unwrap();
    assert_eq!(chosen.id, "worker-1");

    // Artificially enqueue into worker-1
    let sampling = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_tokens: 10,
        stop: vec![],
    };
    let _rx = sched_1
        .submit(
            RequestId("req-1".to_string()),
            "prompt".to_string(),
            sampling,
        )
        .await
        .unwrap();

    // Now worker-1 has queue/active > 0, so next request routes to worker-2!
    let chosen_next = router.choose_worker().unwrap();
    assert_eq!(chosen_next.id, "worker-2");
}

#[tokio::test]
async fn model_routing_never_selects_a_different_loaded_model() {
    let workers: Vec<_> = ["alpha", "beta", "alpha"]
        .into_iter()
        .map(|name| {
            Arc::new(Scheduler::new(
                4,
                32,
                create_shared_cache(64, 16),
                create_shared_prefix_cache(16, 8),
                Arc::new(MockBackend::new(name, 0)),
            ))
        })
        .collect();
    let router = Router::new(
        workers
            .iter()
            .enumerate()
            .map(|(i, w)| WorkerHandle::new(i.to_string(), w.clone()))
            .collect(),
    );
    assert_eq!(router.model_names(), ["alpha", "beta"]);
    // Leave loops stopped to make queue admission observable.
    let _first = router
        .submit_model(
            "beta",
            RequestId("first".into()),
            "hello".into(),
            SamplingParams::default(),
        )
        .await
        .unwrap();
    let messages = vec![rlse::core::types::ChatMessage {
        role: "user".into(),
        content: "hello".into(),
    }];
    let _second = router
        .submit_chat(
            "beta",
            RequestId("second".into()),
            &messages,
            SamplingParams::default(),
        )
        .await
        .unwrap();
    assert_eq!(workers[0].queue_depth(), 0);
    assert_eq!(workers[1].queue_depth(), 2);
    assert_eq!(workers[2].queue_depth(), 0);
    assert!(router
        .submit_model(
            "missing",
            RequestId("bad".into()),
            "hello".into(),
            SamplingParams::default()
        )
        .await
        .is_err());
}
