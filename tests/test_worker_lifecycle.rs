use async_trait::async_trait;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use rlse::{
    api::{create_app, AppState},
    backends::{ModelBackend, StepToken},
    cache::{create_shared_cache, create_shared_prefix_cache},
    core::{
        config::Settings,
        errors::EngineError,
        types::{ActiveSequence, RequestId, SamplingParams},
    },
    routing::router::{Router, WorkerHandle},
    scheduler::{scheduler::Scheduler, sequence::TokenEvent},
    telemetry::create_telemetry,
};
use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};
use tower::ServiceExt;

struct BlockingBackend {
    entered: AtomicBool,
    allocations: AtomicUsize,
    panic: bool,
}
#[async_trait]
impl ModelBackend for BlockingBackend {
    fn name(&self) -> &str {
        "blocked"
    }
    async fn init_sequence(&self, _: &str) -> Result<Vec<u32>, EngineError> {
        Ok(vec![1])
    }
    fn reserve_sequence(&self, _: &RequestId, _: usize, _: usize) -> Result<(), EngineError> {
        self.allocations.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn cancel_sequence(&self, _: &RequestId) {
        self.allocations.fetch_sub(1, Ordering::SeqCst);
    }
    async fn step_batch(&self, seq: &mut [ActiveSequence]) -> Result<Vec<StepToken>, EngineError> {
        self.entered.store(true, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(300));
        assert!(!self.panic, "injected worker panic");
        Ok(seq
            .iter()
            .map(|_| StepToken {
                token_id: 1,
                text: "x".into(),
                is_eos: false,
            })
            .collect())
    }
}

#[tokio::test(flavor = "current_thread")]
async fn blocking_forward_keeps_http_responsive_and_shutdown_cleans_up() {
    exercise(false).await;
}
#[tokio::test(flavor = "current_thread")]
async fn panic_removes_worker_and_terminates_active_and_queued_requests() {
    exercise(true).await;
}
async fn exercise(panic: bool) {
    let backend = Arc::new(BlockingBackend {
        entered: AtomicBool::new(false),
        allocations: AtomicUsize::new(0),
        panic,
    });
    let cache = create_shared_cache(64, 16);
    let prefix_cache = create_shared_prefix_cache(16, 8);
    let worker = Arc::new(Scheduler::new(
        1,
        2,
        cache.clone(),
        prefix_cache.clone(),
        backend.clone(),
    ));
    let task = worker.start_loop();
    let router = Arc::new(Router::new(vec![WorkerHandle::new("one", worker.clone())]));
    let app = create_app(AppState {
        router: router.clone(),
        cache: cache.clone(),
        prefix_cache,
        telemetry: create_telemetry(),
        settings: Settings::default(),
    });
    let sampling = SamplingParams {
        max_tokens: 30,
        ..SamplingParams::default()
    };
    let mut active = worker
        .submit(RequestId("active".into()), "hello".into(), sampling.clone())
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while !backend.entered.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
    let mut queued = worker
        .submit(RequestId("queued".into()), "hello".into(), sampling)
        .await
        .unwrap();
    for path in ["/health", "/api/status"] {
        let start = std::time::Instant::now();
        let response = tokio::time::timeout(
            Duration::from_millis(100),
            app.clone()
                .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap()),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(start.elapsed() < Duration::from_millis(100));
    }
    if !panic {
        worker.begin_drain(Duration::from_millis(10));
        assert!(worker
            .submit(
                RequestId("late".into()),
                "hello".into(),
                SamplingParams::default()
            )
            .await
            .is_err());
    }
    tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
    for rx in [&mut active, &mut queued] {
        let mut error = false;
        while let Some(event) = rx.recv().await {
            if matches!(event, TokenEvent::Error(_)) {
                error = true;
            }
        }
        assert!(error);
    }
    assert!(!worker.is_healthy());
    assert!(router.choose_worker().is_err());
    assert_eq!(worker.active_count(), 0);
    assert_eq!(worker.queue_depth(), 0);
    assert_eq!(backend.allocations.load(Ordering::SeqCst), 0);
    assert_eq!(cache.lock().allocated_count(), 0);
}

#[tokio::test]
async fn saturated_worker_routes_to_available_capacity() {
    let make = || {
        Arc::new(Scheduler::new(
            1,
            1,
            create_shared_cache(64, 16),
            create_shared_prefix_cache(16, 8),
            Arc::new(rlse::backends::mock::MockBackend::new("mock", 0)),
        ))
    };
    let first = make();
    let second = make();
    let router = Router::new(vec![
        WorkerHandle::new("full", first.clone()),
        WorkerHandle::new("free", second.clone()),
    ]);
    let _one = first
        .submit(
            RequestId("one".into()),
            "hello".into(),
            SamplingParams::default(),
        )
        .await
        .unwrap();
    assert_eq!(router.choose_worker().unwrap().id, "free");
    let _two = router
        .submit_model(
            "mock",
            RequestId("two".into()),
            "hello".into(),
            SamplingParams::default(),
        )
        .await
        .unwrap();
    assert_eq!(second.queue_depth(), 1);
}

#[tokio::test]
async fn drain_finishes_accepted_work_and_joins_worker() {
    let worker = Arc::new(Scheduler::new(
        1,
        2,
        create_shared_cache(64, 16),
        create_shared_prefix_cache(16, 8),
        Arc::new(rlse::backends::mock::MockBackend::new("mock", 1)),
    ));
    let mut rx = worker
        .submit(
            RequestId("drain".into()),
            "hello".into(),
            SamplingParams {
                max_tokens: 2,
                ..SamplingParams::default()
            },
        )
        .await
        .unwrap();
    let task = worker.start_loop();
    worker.begin_drain(Duration::from_secs(1));
    tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
    let mut done = false;
    while let Some(event) = rx.recv().await {
        match event {
            TokenEvent::Done { usage, .. } => {
                assert_eq!(usage.completion_tokens, 2);
                done = true;
            }
            TokenEvent::Error(err) => panic!("{err}"),
            _ => {}
        }
    }
    assert!(done);
    assert!(!worker.is_healthy());
    assert_eq!(worker.active_count(), 0);
}
