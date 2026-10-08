use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use std::sync::Arc;
use tower::ServiceExt;

use mini_serve::api::{create_app, AppState};
use mini_serve::backends::mock::MockBackend;
use mini_serve::cache::{create_shared_cache, create_shared_prefix_cache};
use mini_serve::core::config::Settings;
use mini_serve::routing::router::{Router, WorkerHandle};
use mini_serve::scheduler::scheduler::Scheduler;
use mini_serve::telemetry::create_telemetry;

fn setup_test_app() -> axum::Router {
    let settings = Settings::default();
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
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert!(json["usage"]["completion_tokens"].as_u64().unwrap() > 0);
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
    assert_eq!(json["engine"], "mini-serve-rs");
}
