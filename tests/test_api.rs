use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use std::sync::Arc;
use tower::ServiceExt;

use rlse::api::{create_app, AppState};
use rlse::backends::mock::MockBackend;
use rlse::backends::{ModelBackend, StepToken};
use rlse::cache::{create_shared_cache, create_shared_prefix_cache};
use rlse::core::config::Settings;
use rlse::core::{errors::EngineError, types::ActiveSequence};
use rlse::routing::router::{Router, WorkerHandle};
use rlse::scheduler::scheduler::Scheduler;
use rlse::telemetry::create_telemetry;

fn setup_test_app() -> axum::Router {
    // Simulate explicit fallback: configured identity differs from the loaded mock.
    let settings = Settings {
        model: "unloaded-model".into(),
        ..Settings::default()
    };
    let cache = create_shared_cache(256, 16);
    let prefix_cache = create_shared_prefix_cache(16, 128);
    let telemetry = create_telemetry();
    let backend = Arc::new(MockBackend::new("mock", 1));

    let scheduler = Arc::new(Scheduler::new(
        4,
        32,
        cache.clone(),
        prefix_cache.clone(),
        backend,
    ));
    scheduler.start_loop();

    let router = Arc::new(Router::new(vec![WorkerHandle::new("worker-1", scheduler)]));

    let state = AppState {
        router,
        cache,
        prefix_cache,
        telemetry,
        settings,
    };

    create_app(state)
}

#[tokio::test]
async fn test_health_and_models_endpoints() {
    let app = setup_test_app();

    // GET /health
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // GET /v1/models
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "list");
    assert_eq!(json["data"].as_array().unwrap().len(), 1);
    assert_eq!(json["data"][0]["id"], "mock");
}

#[tokio::test]
async fn test_completion_non_streaming() {
    let app = setup_test_app();

    let req_body = serde_json::json!({
        "model": "mock",
        "prompt": "Test prompt",
        "max_tokens": 5,
        "stream": false
    });

    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "text_completion");
    assert_eq!(json["choices"][0]["finish_reason"], "length");
    assert_eq!(json["usage"]["completion_tokens"].as_u64().unwrap(), 5);
}

#[tokio::test]
async fn test_chat_completion_non_streaming() {
    let app = setup_test_app();

    let req_body = serde_json::json!({
        "model": "mock",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 4,
        "stream": false
    });

    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
}

#[tokio::test]
async fn test_dashboard_and_status() {
    let app = setup_test_app();

    // GET /dashboard
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/dashboard")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // GET /api/status
    let res = app
        .oneshot(
            Request::builder()
                .uri("/api/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "healthy");
    assert_eq!(json["engine"], "rlse");
}

#[tokio::test]
async fn streaming_endpoints_include_finish_reason_and_done() {
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        for path in ["/v1/completions", "/v1/chat/completions"] {
            let app = setup_test_app();
            let res = app.oneshot(Request::builder().method("POST").uri(path)
                .header("content-type", "application/json")
                .body(Body::from(serde_json::json!({
                    "model": "mock", "prompt": "Hello", "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 4, "stream": true
                }).to_string())).unwrap()).await.unwrap();
            assert_eq!(res.status(), StatusCode::OK);
            let bytes = res.into_body().collect().await.unwrap().to_bytes();
            let text = std::str::from_utf8(&bytes).unwrap();
            assert!(text.contains("\"finish_reason\":\"length\""), "{text}");
            assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
        }
    }).await.expect("SSE did not terminate");
}

struct FailingBackend;

#[async_trait::async_trait]
impl rlse::backends::ModelBackend for FailingBackend {
    fn name(&self) -> &str {
        "mock"
    }
    fn format_chat(
        &self,
        messages: &[rlse::core::types::ChatMessage],
    ) -> Result<String, EngineError> {
        MockBackend::new("mock", 0).format_chat(messages)
    }
    async fn init_sequence(&self, _: &str) -> Result<Vec<u32>, rlse::core::errors::EngineError> {
        Ok(vec![1])
    }
    async fn step_batch(
        &self,
        _: &mut [rlse::core::types::ActiveSequence],
    ) -> Result<Vec<rlse::backends::StepToken>, rlse::core::errors::EngineError> {
        Err(rlse::core::errors::EngineError::BackendError(
            "injected backend failure".into(),
        ))
    }
}

#[tokio::test]
async fn backend_failure_is_explicit_for_both_apis_and_releases_cache() {
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        for path in ["/v1/completions", "/v1/chat/completions"] {
            for stream in [false, true] {
                let cache = create_shared_cache(256, 16);
                let prefix_cache = create_shared_prefix_cache(16, 128);
                let scheduler = Arc::new(Scheduler::new(4, 32, cache.clone(), prefix_cache.clone(), Arc::new(FailingBackend)));
                let handle = scheduler.start_loop();
                let app = create_app(AppState {
                    router: Arc::new(Router::new(vec![WorkerHandle::new("failing", scheduler.clone())])),
                    cache: cache.clone(), prefix_cache, telemetry: create_telemetry(), settings: Settings::default(),
                });
                let res = app.oneshot(Request::builder().method("POST").uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::json!({
                        "model": "mock", "prompt": "Hello", "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 4, "stream": stream
                    }).to_string())).unwrap()).await.unwrap();
                assert_eq!(res.status(), if stream { StatusCode::OK } else { StatusCode::INTERNAL_SERVER_ERROR });
                let bytes = res.into_body().collect().await.unwrap().to_bytes();
                let text = std::str::from_utf8(&bytes).unwrap();
                assert!(text.contains("injected backend failure"), "{text}");
                assert!(!text.contains("[DONE]"));
                assert_eq!(cache.lock().allocated_count(), 0);
                scheduler.shutdown();
                handle.await.unwrap();
            }
        }
    }).await.expect("error response did not terminate");
}

#[tokio::test]
async fn invalid_limits_fail_before_json_or_sse_generation() {
    let app = setup_test_app();
    for route in ["/v1/completions", "/v1/chat/completions"] {
        for stream in [false, true] {
            for (field, value, message, status) in [
                (
                    "max_tokens",
                    serde_json::json!(0),
                    "max_tokens",
                    StatusCode::BAD_REQUEST,
                ),
                (
                    "temperature",
                    serde_json::json!(-0.1),
                    "temperature",
                    StatusCode::BAD_REQUEST,
                ),
                (
                    "temperature",
                    serde_json::json!(2.1),
                    "temperature",
                    StatusCode::BAD_REQUEST,
                ),
                (
                    "top_p",
                    serde_json::json!(0),
                    "top_p",
                    StatusCode::BAD_REQUEST,
                ),
                (
                    "top_p",
                    serde_json::json!(1.1),
                    "top_p",
                    StatusCode::BAD_REQUEST,
                ),
                (
                    "max_tokens",
                    serde_json::json!(4096),
                    "KV capacity",
                    StatusCode::TOO_MANY_REQUESTS,
                ),
            ] {
                let mut payload = serde_json::json!({
                    "model": "mock", "prompt": "one", "messages": [{"role": "user", "content": "one"}],
                    "max_tokens": 1, "stream": stream,
                });
                payload[field] = value;
                let response = tokio::time::timeout(
                    std::time::Duration::from_secs(2),
                    app.clone().oneshot(
                        Request::builder()
                            .method("POST")
                            .uri(route)
                            .header("content-type", "application/json")
                            .body(Body::from(payload.to_string()))
                            .unwrap(),
                    ),
                )
                .await
                .unwrap()
                .unwrap();
                assert_eq!(response.status(), status, "{route} {payload}");
                if status == StatusCode::TOO_MANY_REQUESTS {
                    assert_eq!(response.headers()["retry-after"], "1");
                }
                let body = response.into_body().collect().await.unwrap().to_bytes();
                let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
                assert!(
                    error["error"]["message"]
                        .as_str()
                        .unwrap()
                        .contains(message),
                    "{error}"
                );
            }
        }
    }
}

#[tokio::test]
async fn tensor_memory_telemetry_is_separate_from_logical_blocks() {
    let app = setup_test_app();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let status: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let memory = &status["tensor_memory"];
    assert_eq!(status["cache"]["allocated_blocks"], 0);
    assert_eq!(memory["active_kv_reserved_bytes"], 0);
    assert_eq!(memory["prefix_retained_bytes"], 0);
    assert_eq!(memory["reserved_bytes"], memory["headroom_bytes"]);
    assert_eq!(
        memory["reserved_bytes"].as_u64().unwrap() + memory["available_bytes"].as_u64().unwrap(),
        memory["budget_bytes"].as_u64().unwrap()
    );
    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let text = std::str::from_utf8(&body).unwrap();
    for name in [
        "budget",
        "headroom",
        "active_kv_reserved",
        "execution_reserved",
        "prefix_retained",
        "reserved",
        "available",
        "peak_reserved",
    ] {
        assert!(text.contains(&format!("# TYPE mini_tensor_memory_{name}_bytes gauge")));
        assert!(text.contains(&format!(
            "mini_tensor_memory_{name}_bytes {}\n",
            memory[format!("{name}_bytes")]
        )));
    }
    assert!(text.contains("mini_tensor_memory_rejections_total 0\n"));
    assert!(text.contains("mini_prefix_tensor_evictions_total 0\n"));
}

struct CapturePrompt(parking_lot::Mutex<Option<String>>);

#[async_trait::async_trait]
impl ModelBackend for CapturePrompt {
    fn name(&self) -> &str {
        "capture"
    }
    fn format_chat(
        &self,
        messages: &[rlse::core::types::ChatMessage],
    ) -> Result<String, EngineError> {
        rlse::backends::chat::ChatTemplate::from_config(include_str!(
            "fixtures/chat/qwen3-tokenizer_config.json"
        ))
        .unwrap()
        .render(messages)
    }
    async fn init_sequence(&self, prompt: &str) -> Result<Vec<u32>, EngineError> {
        *self.0.lock() = Some(prompt.into());
        Ok(vec![1])
    }
    async fn step_batch(
        &self,
        sequences: &mut [ActiveSequence],
    ) -> Result<Vec<StepToken>, EngineError> {
        Ok(sequences
            .iter()
            .map(|_| StepToken {
                token_id: 1,
                text: "ok".into(),
                is_eos: true,
            })
            .collect())
    }
}

#[tokio::test]
async fn chat_route_applies_role_delimiters() {
    let backend = Arc::new(CapturePrompt(parking_lot::Mutex::new(None)));
    let cache = create_shared_cache(256, 16);
    let prefix_cache = create_shared_prefix_cache(16, 128);
    let scheduler = Arc::new(Scheduler::new(
        4,
        32,
        cache.clone(),
        prefix_cache.clone(),
        backend.clone(),
    ));
    let router = Arc::new(Router::new(vec![WorkerHandle::new(
        "chat-template-test",
        scheduler.clone(),
    )]));
    let app = create_app(AppState {
        router,
        cache,
        prefix_cache,
        telemetry: create_telemetry(),
        settings: Settings::default(),
    });
    let task = scheduler.start_loop();
    let response = app.oneshot(Request::builder()
        .method("POST").uri("/v1/chat/completions").header("content-type", "application/json")
        .body(Body::from(serde_json::json!({"model":"capture", "messages":[{"role":"user", "content":"Hello"}], "max_tokens":1}).to_string()))
        .unwrap()).await.unwrap();
    let status = response.status();
    let _ = response.into_body().collect().await.unwrap();
    scheduler.shutdown();
    task.await.unwrap();
    assert!(status.is_success());
    let actual = backend.0.lock().clone().unwrap();
    eprintln!("C: chat prompt handed to backend={actual:?}");
    // This checks route formatting only, not pretrained response quality or
    // the additional thinking-mode options supported by some Qwen templates.
    assert_eq!(
        actual,
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[tokio::test]
async fn unknown_models_and_roles_are_rejected_before_admission() {
    let backend = Arc::new(CapturePrompt(parking_lot::Mutex::new(None)));
    let cache = create_shared_cache(256, 16);
    let prefix = create_shared_prefix_cache(16, 128);
    let scheduler = Arc::new(Scheduler::new(
        4,
        32,
        cache.clone(),
        prefix.clone(),
        backend.clone(),
    ));
    // No loop: valid admissions would queue, invalid requests must not reach it.
    let app = create_app(AppState {
        router: Arc::new(Router::new(vec![WorkerHandle::new(
            "capture",
            scheduler.clone(),
        )])),
        cache: cache.clone(),
        prefix_cache: prefix,
        telemetry: create_telemetry(),
        settings: Settings::default(),
    });
    for stream in [false, true] {
        for (route, model, role, message) in [
            ("/v1/completions", "missing", "user", "Unknown model"),
            ("/v1/chat/completions", "missing", "user", "Unknown model"),
            (
                "/v1/chat/completions",
                "capture",
                "tool",
                "Unsupported chat role",
            ),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(route)
                        .header("content-type", "application/json")
                        .body(Body::from(
                            serde_json::json!({"model":model,"prompt":"hello",
                    "messages":[{"role":role,"content":"hello"}],"stream":stream})
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body = response.into_body().collect().await.unwrap().to_bytes();
            assert!(std::str::from_utf8(&body).unwrap().contains(message));
            assert!(backend.0.lock().is_none());
            assert_eq!(scheduler.queue_depth(), 0);
            assert_eq!(cache.lock().allocated_count(), 0);
        }
    }
}

#[tokio::test]
async fn loaded_identity_is_consistent_after_mock_fallback() {
    let app = setup_test_app();
    for route in ["/health", "/api/status"] {
        let response = app
            .clone()
            .oneshot(Request::builder().uri(route).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["model"], "mock");
    }
    for route in ["/v1/completions", "/v1/chat/completions"] {
        for stream in [false, true] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(route)
                        .header("content-type", "application/json")
                        .body(Body::from(
                            serde_json::json!({"model":"mock","prompt":"hello",
                    "messages":[{"role":"user","content":"hello"}],"max_tokens":1,"stream":stream})
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = response.into_body().collect().await.unwrap().to_bytes();
            let text = std::str::from_utf8(&body).unwrap();
            if stream {
                for data in text
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter(|data| *data != "[DONE]")
                {
                    let value: serde_json::Value = serde_json::from_str(data).unwrap();
                    assert_eq!(value["model"], "mock");
                }
            } else {
                let value: serde_json::Value = serde_json::from_str(text).unwrap();
                assert_eq!(value["model"], "mock");
            }
        }
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(route)
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({"model":"unloaded-model","prompt":"hello",
                "messages":[{"role":"user","content":"hello"}]})
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}

struct CompletionOnly(CapturePrompt);
#[async_trait::async_trait]
impl ModelBackend for CompletionOnly {
    fn name(&self) -> &str {
        "completion-only"
    }
    async fn init_sequence(&self, prompt: &str) -> Result<Vec<u32>, EngineError> {
        self.0.init_sequence(prompt).await
    }
    async fn step_batch(
        &self,
        sequences: &mut [ActiveSequence],
    ) -> Result<Vec<StepToken>, EngineError> {
        self.0.step_batch(sequences).await
    }
}

#[tokio::test]
async fn unavailable_chat_template_leaves_completions_usable() {
    let backend = Arc::new(CompletionOnly(CapturePrompt(parking_lot::Mutex::new(None))));
    let cache = create_shared_cache(256, 16);
    let prefix = create_shared_prefix_cache(16, 128);
    let scheduler = Arc::new(Scheduler::new(
        4,
        32,
        cache.clone(),
        prefix.clone(),
        backend.clone(),
    ));
    let app = create_app(AppState {
        router: Arc::new(Router::new(vec![WorkerHandle::new(
            "completion-only",
            scheduler.clone(),
        )])),
        cache,
        prefix_cache: prefix,
        telemetry: create_telemetry(),
        settings: Settings::default(),
    });
    for stream in [false, true] {
        let response = app.clone().oneshot(Request::builder().method("POST").uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::json!({"model":"completion-only", "messages":[{"role":"user","content":"hello"}], "stream":stream}).to_string()))
            .unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(backend.0 .0.lock().is_none());
        assert_eq!(scheduler.queue_depth(), 0);
    }
    let handle = scheduler.start_loop();
    let response = app.oneshot(Request::builder().method("POST").uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::json!({"model":"completion-only", "prompt":"unchanged prompt", "max_tokens":1}).to_string()))
        .unwrap()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.into_body().collect().await.unwrap();
    assert_eq!(backend.0 .0.lock().as_deref(), Some("unchanged prompt"));
    scheduler.shutdown();
    handle.await.unwrap();
}
