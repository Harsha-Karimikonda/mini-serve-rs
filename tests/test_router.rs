use mini_serve::backends::mock::MockBackend;
use mini_serve::cache::{create_shared_cache, create_shared_prefix_cache};
use mini_serve::core::types::{RequestId, SamplingParams};
use mini_serve::routing::router::{Router, WorkerHandle};
use mini_serve::scheduler::scheduler::Scheduler;
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
