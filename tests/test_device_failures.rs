use async_trait::async_trait;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use parking_lot::Mutex;
use rlse::{
    api::{create_app, AppState},
    backends::{MockBackend, ModelBackend, StepToken},
    cache::{create_shared_cache, create_shared_prefix_cache},
    core::{
        config::Settings,
        errors::EngineError,
        types::{ActiveSequence, RequestId, SamplingParams, TokenEvent},
    },
    routing::router::{Router, WorkerHandle},
    scheduler::Scheduler,
    telemetry::create_telemetry,
};
use serde_json::{json, Value};
use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};
use tower::ServiceExt;

// Deliberately retain partial state on failure; the serving layer must cancel it.
struct FaultBackend {
    healthy: AtomicBool,
    mode: AtomicUsize,
    forwards: AtomicUsize,
    tokenizations: AtomicUsize,
    live: Mutex<HashSet<RequestId>>,
    release_failure: tokio::sync::Notify,
}
impl FaultBackend {
    fn new(mode: usize) -> Self {
        Self {
            healthy: AtomicBool::new(true),
            mode: AtomicUsize::new(mode),
            forwards: AtomicUsize::new(0),
            tokenizations: AtomicUsize::new(0),
            live: Mutex::new(HashSet::new()),
            release_failure: tokio::sync::Notify::new(),
        }
    }
}
#[async_trait]
impl ModelBackend for FaultBackend {
    fn name(&self) -> &str {
        "fault"
    }
    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::SeqCst)
    }
    fn mark_unhealthy(&self) {
        self.healthy.store(false, Ordering::SeqCst);
    }
    fn format_chat(
        &self,
        messages: &[rlse::core::types::ChatMessage],
    ) -> Result<String, EngineError> {
        MockBackend::new("mock", 0).format_chat(messages)
    }
    async fn init_sequence(&self, _: &str) -> Result<Vec<u32>, EngineError> {
        self.tokenizations.fetch_add(1, Ordering::SeqCst);
        Ok(vec![1])
    }
    fn reserve_sequence(&self, id: &RequestId, _: usize, _: usize) -> Result<(), EngineError> {
        self.live.lock().insert(id.clone());
        match self.mode.load(Ordering::SeqCst) {
            1 => Err(EngineError::CachePressure(
                "injected reservation pressure".into(),
            )),
            2 => Err(EngineError::BackendError(
                "injected allocation failure".into(),
            )),
            _ => Ok(()),
        }
    }
    async fn step_batch(&self, seqs: &mut [ActiveSequence]) -> Result<Vec<StepToken>, EngineError> {
        self.forwards.fetch_add(1, Ordering::SeqCst);
        if self.mode.load(Ordering::SeqCst) == 4 {
            self.release_failure.notified().await;
            return Err(EngineError::BackendError(
                "injected shared device failure".into(),
            ));
        }
        if self.mode.load(Ordering::SeqCst) == 3 && !seqs[0].output_tokens.is_empty() {
            return Err(EngineError::BackendError("injected device failure".into()));
        }
        Ok(seqs
            .iter()
            .map(|_| StepToken {
                token_id: 1,
                text: "token".into(),
                is_eos: false,
            })
            .collect())
    }
    fn cancel_sequence(&self, id: &RequestId) {
        self.live.lock().remove(id);
    }
}
fn params() -> SamplingParams {
    SamplingParams {
        max_tokens: 4,
        temperature: 0.,
        ..SamplingParams::default()
    }
}
async fn request(
    app: axum::Router,
    path: &str,
    payload: Option<Value>,
) -> axum::response::Response {
    let mut req = Request::builder().uri(path);
    let body = match payload {
        Some(p) => {
            req = req
                .method("POST")
                .header("content-type", "application/json");
            Body::from(p.to_string())
        }
        None => Body::empty(),
    };
    app.oneshot(req.body(body).unwrap()).await.unwrap()
}
async fn bytes(res: axum::response::Response) -> String {
    String::from_utf8(res.into_body().collect().await.unwrap().to_bytes().to_vec()).unwrap()
}

#[tokio::test]
async fn allocation_and_device_failures_quarantine_and_report_readiness() {
    tokio::time::timeout(Duration::from_secs(5), async {
        for mode in [2, 3] {
            for path in ["/v1/completions", "/v1/chat/completions"] {
                for stream in [false, true] {
                    let backend = Arc::new(FaultBackend::new(mode));
                    let cache = create_shared_cache(32, 16);
                    let prefix = create_shared_prefix_cache(16, 8);
                    let workers: Vec<_> = (0..2).map(|_| Arc::new(Scheduler::new(1, 8, cache.clone(), prefix.clone(), backend.clone()))).collect();
                    let router = Arc::new(Router::new(workers.iter().enumerate().map(|(i,w)| WorkerHandle::new(i.to_string(), w.clone())).collect()));
                    let app = create_app(AppState { router: router.clone(), cache: cache.clone(), prefix_cache: prefix, telemetry: create_telemetry(), settings: Settings::default() });
                    assert_eq!(request(app.clone(), "/ready", None).await.status(), StatusCode::OK);
                    // Both workers have queued work when the first execution path fails.
                    let mut queued = workers[1].submit(RequestId("queued".into()), "hello".into(), params()).await.unwrap();
                    let handle = workers[0].start_loop();
                    let payload = json!({"model":"fault", "prompt":"hello", "messages":[{"role":"user","content":"hello"}], "max_tokens":4, "stream":stream});
                    let response = request(app.clone(), path, Some(payload.clone())).await;
                    assert_eq!(response.status(), if stream { StatusCode::OK } else { StatusCode::INTERNAL_SERVER_ERROR });
                    let body = bytes(response).await;
                    assert!(body.contains(if mode == 2 { "injected allocation failure" } else { "injected device failure" }), "{body}");
                    assert!(!body.contains("[DONE]"));
                    if stream && mode == 3 { assert!(body.contains("token")); }
                    handle.await.unwrap();
                    // Sibling starts only after quarantine; it must drain without forwarding.
                    workers[1].start_loop().await.unwrap();
                    assert!(matches!(queued.recv().await, Some(TokenEvent::Error(_))));
                    assert!(queued.recv().await.is_none());
                    assert!(backend.live.lock().is_empty());
                    assert_eq!(cache.lock().allocated_count(), 0);
                    assert!(router.choose_worker().is_err());
                    assert!(workers.iter().all(|w| !w.is_healthy() && w.queue_depth() == 0 && w.active_count() == 0));
                    assert_eq!(request(app.clone(), "/health", None).await.status(), StatusCode::OK);
                    assert_eq!(request(app.clone(), "/ready", None).await.status(), StatusCode::SERVICE_UNAVAILABLE);
                    let status: Value = serde_json::from_str(&bytes(request(app.clone(), "/api/status", None).await).await).unwrap();
                    assert_eq!(status["status"], "unavailable");
                    assert!(status["workers"].as_array().unwrap().iter().all(|w| w["healthy"] == false));
                    assert!(bytes(request(app.clone(), "/metrics", None).await).await.contains("mini_ready_workers 0\n"));
                    let tokenizations = backend.tokenizations.load(Ordering::SeqCst);
                    assert_eq!(request(app.clone(), path, Some(payload)).await.status(), StatusCode::SERVICE_UNAVAILABLE);
                    assert!(workers[0].submit(RequestId("direct".into()), "hello".into(), params()).await.is_err());
                    assert_eq!(backend.tokenizations.load(Ordering::SeqCst), tokenizations);
                    assert_eq!(backend.forwards.load(Ordering::SeqCst), if mode == 2 { 0 } else { 2 });
                    // An independent backend remains eligible.
                    let good = Arc::new(Scheduler::new(1, 8, create_shared_cache(32,16), create_shared_prefix_cache(16,8), Arc::new(MockBackend::new("mock", 0))));
                    let mixed = Router::new(vec![WorkerHandle::new("bad", workers[0].clone()), WorkerHandle::new("good", good)]);
                    assert_eq!(mixed.choose_worker().unwrap().id, "good");
                }
            }
        }
    }).await.expect("failure handling hung");
}

#[tokio::test]
async fn reservation_pressure_reclaims_partial_state_and_allows_later_work() {
    tokio::time::timeout(Duration::from_secs(3), async {
        let backend = Arc::new(FaultBackend::new(1));
        let cache = create_shared_cache(32,16);
        let worker = Arc::new(Scheduler::new(1,8,cache.clone(),create_shared_prefix_cache(16,8),backend.clone()));
        let handle = worker.start_loop();
        let mut rx = worker.submit(RequestId("pressure".into()), "hello".into(), params()).await.unwrap();
        assert!(matches!(rx.recv().await, Some(TokenEvent::Error(e)) if e.contains("reservation pressure")));
        assert!(backend.live.lock().is_empty());
        assert_eq!(backend.forwards.load(Ordering::SeqCst), 0);
        assert!(worker.is_healthy());
        backend.mode.store(0, Ordering::SeqCst);
        let mut rx = worker.submit(RequestId("recovered".into()), "hello".into(), params()).await.unwrap();
        let mut tokens = 0;
        while let Some(event) = rx.recv().await {
            match event { TokenEvent::Token(_) => tokens += 1, TokenEvent::Done { .. } => break, TokenEvent::Error(e) => panic!("{e}") }
        }
        assert_eq!(tokens, 4);
        worker.shutdown();
        handle.await.unwrap();
        assert!(backend.live.lock().is_empty());
        assert_eq!(cache.lock().allocated_count(), 0);
    }).await.expect("reservation recovery hung");
}

#[tokio::test]
async fn shared_failure_drains_active_and_queued_work_on_both_workers() {
    tokio::time::timeout(Duration::from_secs(3), async {
        let backend = Arc::new(FaultBackend::new(4));
        let cache = create_shared_cache(32, 16);
        let prefix = create_shared_prefix_cache(16, 8);
        let workers: Vec<_> = (0..2)
            .map(|_| {
                Arc::new(Scheduler::new(
                    1,
                    8,
                    cache.clone(),
                    prefix.clone(),
                    backend.clone(),
                ))
            })
            .collect();
        let mut receivers = Vec::new();
        for (i, worker) in workers.iter().enumerate() {
            for j in 0..2 {
                receivers.push(
                    worker
                        .submit(RequestId(format!("{i}-{j}")), "hello".into(), params())
                        .await
                        .unwrap(),
                );
            }
        }
        let handles: Vec<_> = workers.iter().map(Scheduler::start_loop).collect();
        while backend.live.lock().len() != 2 {
            tokio::task::yield_now().await;
        }
        assert_eq!(cache.lock().allocated_count(), 2);
        assert!(workers
            .iter()
            .all(|w| w.active_count() == 1 && w.queue_depth() == 1));
        backend.release_failure.notify_waiters();
        for handle in handles {
            handle.await.unwrap();
        }
        for mut rx in receivers {
            assert!(matches!(rx.recv().await, Some(TokenEvent::Error(_))));
            assert!(rx.recv().await.is_none());
        }
        assert!(backend.live.lock().is_empty());
        assert_eq!(cache.lock().allocated_count(), 0);
        assert_eq!(backend.forwards.load(Ordering::SeqCst), 2);
        assert!(workers
            .iter()
            .all(|w| !w.is_healthy() && w.active_count() == 0 && w.queue_depth() == 0));
    })
    .await
    .expect("shared quarantine did not drain");
}
