use async_trait::async_trait;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use rlse::{
    api::{create_app, AppState},
    backends::{ModelBackend, StepToken},
    cache::{create_shared_cache, create_shared_prefix_cache},
    core::{config::Settings, errors::EngineError, types::ActiveSequence},
    routing::router::{Router, WorkerHandle},
    scheduler::Scheduler,
    telemetry::create_telemetry,
};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;
use tower::ServiceExt;

struct Gated {
    gate: tokio::sync::Semaphore,
    entered: AtomicUsize,
    tokenized: AtomicUsize,
    fail: AtomicBool,
}
#[async_trait]
impl ModelBackend for Gated {
    fn name(&self) -> &str {
        "gated"
    }
    async fn init_sequence(&self, _: &str) -> Result<Vec<u32>, EngineError> {
        self.tokenized.fetch_add(1, Ordering::SeqCst);
        Ok(vec![1])
    }
    async fn step_batch(&self, seq: &mut [ActiveSequence]) -> Result<Vec<StepToken>, EngineError> {
        self.entered.fetch_add(1, Ordering::SeqCst);
        self.gate.acquire().await.unwrap().forget();
        if self.fail.load(Ordering::SeqCst) {
            return Err(EngineError::CachePressure("injected failure".into()));
        }
        Ok(seq
            .iter()
            .map(|_| StepToken {
                token_id: 1,
                text: "ok".into(),
                is_eos: true,
            })
            .collect())
    }
}
fn setup(settings: Settings) -> (axum::Router, Arc<Scheduler>, Arc<Gated>) {
    let backend = Arc::new(Gated {
        gate: tokio::sync::Semaphore::new(0),
        entered: AtomicUsize::new(0),
        tokenized: AtomicUsize::new(0),
        fail: AtomicBool::new(false),
    });
    let cache = create_shared_cache(64, 16);
    let prefix = create_shared_prefix_cache(16, 8);
    let worker = Arc::new(Scheduler::new(
        2,
        8,
        cache.clone(),
        prefix.clone(),
        backend.clone(),
    ));
    let app = create_app(AppState {
        router: Arc::new(Router::new(vec![WorkerHandle::new(
            "limits",
            worker.clone(),
        )])),
        cache,
        prefix_cache: prefix,
        telemetry: create_telemetry(),
        settings,
    });
    (app, worker, backend)
}
fn request(key: &str, body: &str) -> Request<Body> {
    Request::builder()
        .uri("/v1/completions")
        .method("POST")
        .header("authorization", format!("Bearer {key}"))
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap()
}
const STREAM: &str = r#"{"model":"gated","prompt":"hello","max_tokens":1,"stream":true}"#;
const INVALID: &str = r#"{"model":"gated","prompt":"hello","max_tokens":0}"#;
async fn released(app: &axum::Router, key: &str, expected: StatusCode) {
    loop {
        let res = app.clone().oneshot(request(key, INVALID)).await.unwrap();
        if res.status() != StatusCode::TOO_MANY_REQUESTS {
            assert_eq!(res.status(), expected);
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}
#[tokio::test]
async fn client_isolation_and_permits_survive_disconnect_until_worker_cleanup() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (app, worker, backend) = setup(Settings {
            api_keys: vec!["a".into(), "b".into()],
            client_max_concurrent: 1,
            ..Settings::default()
        });
        let task = worker.start_loop();
        let a = app.clone().oneshot(request("a", STREAM)).await.unwrap();
        assert_eq!(a.status(), StatusCode::OK);
        while backend.entered.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
        let rejected = app.clone().oneshot(request("a", STREAM)).await.unwrap();
        assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(rejected.headers()["retry-after"], "1");
        assert_eq!(backend.tokenized.load(Ordering::SeqCst), 1);
        let b = app.clone().oneshot(request("b", STREAM)).await.unwrap();
        assert_eq!(b.status(), StatusCode::OK);
        drop(a);
        // A running forward still owns A's permit, even after its HTTP body drops.
        assert_eq!(
            app.clone()
                .oneshot(request("a", STREAM))
                .await
                .unwrap()
                .status(),
            StatusCode::TOO_MANY_REQUESTS
        );
        backend.gate.add_permits(2);
        let bytes = b.into_body().collect().await.unwrap().to_bytes();
        assert!(std::str::from_utf8(&bytes)
            .unwrap()
            .ends_with("data: [DONE]\n\n"));
        released(&app, "a", StatusCode::BAD_REQUEST).await;
        released(&app, "b", StatusCode::BAD_REQUEST).await;
        // Validation failure also releases its permit immediately.
        released(&app, "a", StatusCode::BAD_REQUEST).await;
        worker.shutdown();
        task.await.unwrap();
    })
    .await
    .expect("client isolation/cleanup timed out");
}
#[tokio::test]
async fn failures_and_shutdown_release_generation_permits() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (app, worker, backend) = setup(Settings {
            api_key: Some("a".into()),
            client_max_concurrent: 1,
            ..Settings::default()
        });
        let task = worker.start_loop();
        backend.fail.store(true, Ordering::SeqCst);
        let res = app.clone().oneshot(request("a", STREAM)).await.unwrap();
        backend.gate.add_permits(1);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        assert!(std::str::from_utf8(&bytes)
            .unwrap()
            .contains("injected failure"));
        released(&app, "a", StatusCode::BAD_REQUEST).await;
        backend.fail.store(false, Ordering::SeqCst);
        let queued = app.clone().oneshot(request("a", STREAM)).await.unwrap();
        worker.shutdown();
        backend.gate.add_permits(1);
        task.await.unwrap();
        let _ = queued.into_body().collect().await.unwrap();
        // Health is checked before sampling; a leaked permit would instead give 429.
        released(&app, "a", StatusCode::SERVICE_UNAVAILABLE).await;
    })
    .await
    .expect("failure/shutdown quota cleanup timed out");
}
#[tokio::test]
async fn rate_limits_are_per_credential_and_anonymous_identity_cannot_be_spoofed() {
    for authenticated in [true, false] {
        let (app, _, backend) = setup(Settings {
            api_keys: if authenticated {
                vec!["a".into(), "b".into()]
            } else {
                vec![]
            },
            client_requests_per_minute: 1,
            ..Settings::default()
        });
        assert_eq!(
            app.clone()
                .oneshot(request("a", "invalid JSON"))
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
        let rejected = app.clone().oneshot(request("a", STREAM)).await.unwrap();
        assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!((1..=60).contains(
            &rejected.headers()["retry-after"]
                .to_str()
                .unwrap()
                .parse::<u64>()
                .unwrap()
        ));
        let second = app.oneshot(request("b", "invalid JSON")).await.unwrap();
        assert_eq!(
            second.status(),
            if authenticated {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::TOO_MANY_REQUESTS
            }
        );
        assert_eq!(backend.tokenized.load(Ordering::SeqCst), 0);
    }
}
#[test]
fn configured_client_bookkeeping_is_bounded_and_credentials_redacted() {
    let settings = Settings {
        api_keys: vec!["secret-client".into(); 129],
        ..Settings::default()
    };
    assert!(!format!("{settings:?}").contains("secret-client"));
    assert!(settings.validate().unwrap_err().to_string().contains("128"));
}
