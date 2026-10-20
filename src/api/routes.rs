use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        Html, IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use serde_json::json;
use std::convert::Infallible;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::api::dashboard::DASHBOARD_HTML;
use crate::cache::{SharedKVCache, SharedPrefixCache};
use crate::core::config::Settings;
use crate::core::errors::EngineError;
use crate::core::types::*;
use crate::routing::router::SharedRouter;
use crate::scheduler::sequence::TokenEvent;
use crate::telemetry::{log_request, SharedTelemetry};

#[derive(Clone)]
pub struct AppState {
    pub router: SharedRouter,
    pub cache: SharedKVCache,
    pub prefix_cache: SharedPrefixCache,
    pub telemetry: SharedTelemetry,
    pub settings: Settings,
}

pub fn create_app(state: AppState) -> Router {
    let protected = Router::new()
        .route("/v1/models", get(models_handler))
        .route("/v1/completions", post(completions_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/api/status", get(status_handler))
        .route("/metrics", get(metrics_handler))
        .route_layer(axum::middleware::from_fn_with_state(
            crate::api::auth::ApiAuth::new(&state.settings),
            crate::api::auth::authenticate,
        ));
    Router::new()
        .merge(protected)
        .route("/", get(dashboard_handler))
        .route("/dashboard", get(dashboard_handler))
        .route("/health", get(health_handler))
        .route("/ready", get(readiness_handler))
        .with_state(state)
}

/// Turn unexpected worker/channel closure into an explicit SSE error.
fn response_events(
    rx: tokio::sync::mpsc::Receiver<TokenEvent>,
) -> impl futures_util::Stream<Item = TokenEvent> {
    futures_util::stream::unfold((rx, false), |(mut rx, ended)| async move {
        if ended {
            return None;
        }
        let event = rx.recv().await.unwrap_or_else(|| {
            TokenEvent::Error("Generation channel closed without terminal state".to_string())
        });
        let ended = !matches!(event, TokenEvent::Token(_));
        Some((event, (rx, ended)))
    })
}

async fn collect_response(
    mut rx: tokio::sync::mpsc::Receiver<TokenEvent>,
) -> Result<(String, TokenUsage, String), EngineError> {
    let mut text = String::new();
    while let Some(event) = rx.recv().await {
        match event {
            TokenEvent::Token(token) => text.push_str(&token),
            TokenEvent::Done {
                usage,
                finish_reason,
            } => return Ok((text, usage, finish_reason)),
            TokenEvent::Error(error) => return Err(EngineError::BackendError(error)),
        }
    }
    Err(EngineError::BackendError(
        "Generation channel closed without terminal state".to_string(),
    ))
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

async fn dashboard_handler(State(state): State<AppState>) -> Html<&'static str> {
    if state.settings.api_key.is_some() || !state.settings.api_keys.is_empty() {
        Html("<!doctype html><title>RLSE</title><p>API authentication is enabled. Use an API client with an Authorization bearer header. The browser dashboard is disabled.</p>")
    } else {
        Html(DASHBOARD_HTML)
    }
}

async fn health_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "model": state.router.model_names().first(),
        "models": state.router.model_names(),
        "device": state.settings.device,
        "workers": state.router.workers().len(),
    }))
}

async fn readiness_handler(State(state): State<AppState>) -> impl IntoResponse {
    let ready = state.router.choose_worker().is_ok();
    (
        if ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(json!({"ready": ready})),
    )
}

async fn models_handler(State(state): State<AppState>) -> Json<ModelListResponse> {
    let now = now_secs();
    Json(ModelListResponse {
        object: "list".to_string(),
        data: state
            .router
            .model_names()
            .into_iter()
            .map(|id| ModelCard {
                id,
                object: "model".into(),
                created: now,
                owned_by: "rlse".into(),
            })
            .collect(),
    })
}

async fn completions_handler(
    State(state): State<AppState>,
    permit: Option<axum::Extension<std::sync::Arc<crate::core::client_limits::ClientPermit>>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, EngineError> {
    let t0 = Instant::now();
    state.telemetry.record_request();

    if req.prompt.is_empty() {
        return Err(EngineError::InvalidRequest(
            "Prompt cannot be empty".to_string(),
        ));
    }

    let request_id = RequestId::new_completion();
    let sampling = SamplingParams {
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        stop: req.stop,
    };

    let rx = state
        .router
        .submit_model_with_permit(
            &req.model,
            request_id.clone(),
            req.prompt,
            sampling,
            permit.map(|p| p.0),
        )
        .await?;

    if req.stream {
        let stream_req_id = request_id.0.clone();
        let stream_model = req.model.clone();

        let sse_stream = response_events(rx).flat_map(move |event| match event {
            TokenEvent::Token(token_text) => {
                let chunk = CompletionChunk {
                    id: stream_req_id.clone(),
                    usage: None,
                    object: "text_completion.chunk".to_string(),
                    created: now_secs(),
                    model: stream_model.clone(),
                    choices: vec![CompletionChunkChoice {
                        text: token_text,
                        index: 0,
                        finish_reason: None,
                    }],
                };
                futures_util::stream::iter(vec![Ok::<Event, Infallible>(
                    Event::default().data(serde_json::to_string(&chunk).unwrap()),
                )])
            }
            TokenEvent::Done {
                finish_reason,
                usage,
            } => {
                let final_chunk = CompletionChunk {
                    id: stream_req_id.clone(),
                    usage: Some(usage),
                    object: "text_completion.chunk".to_string(),
                    created: now_secs(),
                    model: stream_model.clone(),
                    choices: vec![CompletionChunkChoice {
                        text: String::new(),
                        index: 0,
                        finish_reason: Some(finish_reason),
                    }],
                };
                futures_util::stream::iter(vec![
                    Ok::<Event, Infallible>(
                        Event::default().data(serde_json::to_string(&final_chunk).unwrap()),
                    ),
                    Ok::<Event, Infallible>(Event::default().data("[DONE]")),
                ])
            }
            TokenEvent::Error(err_msg) => {
                let err_json = json!({
                    "error": {
                        "message": err_msg,
                        "type": "server_error",
                        "code": 500
                    }
                });
                futures_util::stream::iter(vec![Ok::<Event, Infallible>(
                    Event::default().data(serde_json::to_string(&err_json).unwrap()),
                )])
            }
        });

        Ok(Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let (full_text, final_usage, final_finish_reason) = collect_response(rx).await?;

        let response = CompletionResponse {
            id: request_id.0,
            object: "text_completion".to_string(),
            created: now_secs(),
            model: req.model,
            choices: vec![CompletionChoice {
                text: full_text,
                index: 0,
                logprobs: None,
                finish_reason: Some(final_finish_reason),
            }],
            usage: final_usage,
        };

        log_request(
            "POST",
            "/v1/completions",
            &response.id,
            200,
            (t0.elapsed().as_secs_f64() * 1000.0).round(),
        );

        Ok(Json(response).into_response())
    }
}

async fn chat_completions_handler(
    State(state): State<AppState>,
    permit: Option<axum::Extension<std::sync::Arc<crate::core::client_limits::ClientPermit>>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, EngineError> {
    let t0 = Instant::now();
    state.telemetry.record_request();

    if req.messages.is_empty() {
        return Err(EngineError::InvalidRequest(
            "Messages cannot be empty".to_string(),
        ));
    }

    let request_id = RequestId::new_chat();
    let sampling = SamplingParams {
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        stop: req.stop,
    };

    let rx = state
        .router
        .submit_chat_with_permit(
            &req.model,
            request_id.clone(),
            &req.messages,
            sampling,
            permit.map(|p| p.0),
        )
        .await?;

    if req.stream {
        let stream_req_id = request_id.0.clone();
        let stream_model = req.model.clone();

        let sse_stream = response_events(rx).flat_map(move |event| match event {
            TokenEvent::Token(token_text) => {
                let chunk = ChatCompletionChunk {
                    id: stream_req_id.clone(),
                    usage: None,
                    object: "chat.completion.chunk".to_string(),
                    created: now_secs(),
                    model: stream_model.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatChunkDelta {
                            role: None,
                            content: Some(token_text),
                        },
                        finish_reason: None,
                    }],
                };
                futures_util::stream::iter(vec![Ok::<Event, Infallible>(
                    Event::default().data(serde_json::to_string(&chunk).unwrap()),
                )])
            }
            TokenEvent::Done {
                finish_reason,
                usage,
            } => {
                let final_chunk = ChatCompletionChunk {
                    id: stream_req_id.clone(),
                    usage: Some(usage),
                    object: "chat.completion.chunk".to_string(),
                    created: now_secs(),
                    model: stream_model.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatChunkDelta {
                            role: None,
                            content: None,
                        },
                        finish_reason: Some(finish_reason),
                    }],
                };
                futures_util::stream::iter(vec![
                    Ok::<Event, Infallible>(
                        Event::default().data(serde_json::to_string(&final_chunk).unwrap()),
                    ),
                    Ok::<Event, Infallible>(Event::default().data("[DONE]")),
                ])
            }
            TokenEvent::Error(err_msg) => {
                let err_json = json!({
                    "error": {
                        "message": err_msg,
                        "type": "server_error",
                        "code": 500
                    }
                });
                futures_util::stream::iter(vec![Ok::<Event, Infallible>(
                    Event::default().data(serde_json::to_string(&err_json).unwrap()),
                )])
            }
        });

        Ok(Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let (full_text, final_usage, final_finish_reason) = collect_response(rx).await?;

        let response = ChatCompletionResponse {
            id: request_id.0,
            object: "chat.completion".to_string(),
            created: now_secs(),
            model: req.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: full_text,
                },
                finish_reason: Some(final_finish_reason),
            }],
            usage: final_usage,
        };

        log_request(
            "POST",
            "/v1/chat/completions",
            &response.id,
            200,
            (t0.elapsed().as_secs_f64() * 1000.0).round(),
        );

        Ok(Json(response).into_response())
    }
}

async fn status_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let cache_stats = state.cache.lock().stats();
    let (hits, misses, tokens_saved) = state.prefix_cache.lock().stats();
    let hit_rate = state.prefix_cache.lock().hit_rate();
    let memory = state.prefix_cache.lock().memory_budget().stats();
    let prefix_evictions = state.prefix_cache.lock().tensor_evictions();

    let workers_info: Vec<serde_json::Value> = state
        .router
        .workers()
        .iter()
        .map(|w| {
            let m = w.scheduler.metrics();
            json!({
                "id": w.id,
                "healthy": w.scheduler.is_healthy(),
                "queue_depth": m.queue_depth,
                "active_count": m.active_count,
                "avg_latency_ms": m.avg_latency_ms,
                "execution": m.execution,
                "score": (w.score() * 10.0).round() / 10.0
            })
        })
        .collect();

    Json(json!({
        "status": if state.router.choose_worker().is_ok() { "healthy" } else { "unavailable" },
        "engine": "rlse",
        "version": "0.2.0",
        "model": state.router.model_names().first(),
        "models": state.router.model_names(),
        "tokens_per_sec": state.router.tokens_per_second(),
        "total_tokens": state.router.execution_stats().generated_tokens,
        "execution": state.router.execution_stats(),
        "total_requests": state.telemetry.total_requests(),
        "workers": workers_info,
        "cache": {
            "total_blocks": cache_stats.total_blocks,
            "allocated_blocks": cache_stats.allocated_blocks,
            "free_blocks": cache_stats.free_blocks,
            "utilization": cache_stats.utilization,
            "fragmentation": cache_stats.fragmentation
        },
        "tensor_memory": memory,
        "prefix_cache": {
            "tensor_evictions": prefix_evictions,
            "hits": hits,
            "misses": misses,
            "tokens_saved": tokens_saved,
            "hit_rate_pct": (hit_rate * 10.0).round() / 10.0
        }
    }))
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    let tps = state.router.tokens_per_second();
    let execution = state.router.execution_stats();
    let total_tokens = execution.generated_tokens;
    let total_requests = state.telemetry.total_requests();
    let cache = state.cache.lock().stats();

    let mut body = String::new();
    body.push_str("# HELP mini_ready_workers Workers with usable execution paths\n# TYPE mini_ready_workers gauge\n");
    body.push_str(&format!(
        "mini_ready_workers {}\n",
        state
            .router
            .workers()
            .iter()
            .filter(|w| w.scheduler.is_healthy())
            .count()
    ));
    for (name, value) in [
        ("accepted_total", execution.accepted),
        ("completed_total", execution.completed),
        ("errors_total", execution.errors),
        ("cancellations_total", execution.cancellations),
        ("queue_delay_count", execution.queue_delay_count),
        ("queue_delay_microseconds_total", execution.queue_delay_us),
        ("ttft_count", execution.ttft_count),
        ("ttft_microseconds_total", execution.ttft_us),
    ] {
        body.push_str(&format!("# HELP mini_execution_{name} Scheduler execution accounting\n# TYPE mini_execution_{name} counter\nmini_execution_{name} {value}\n"));
    }
    body.push_str("# HELP mini_tokens_per_second Current token generation throughput\n");
    body.push_str("# TYPE mini_tokens_per_second gauge\n");
    body.push_str(&format!("mini_tokens_per_second {}\n\n", tps));

    body.push_str("# HELP mini_tokens_total Cumulative tokens generated\n");
    body.push_str("# TYPE mini_tokens_total counter\n");
    body.push_str(&format!("mini_tokens_total {}\n\n", total_tokens));

    body.push_str("# HELP mini_requests_total Cumulative inference requests\n");
    body.push_str("# TYPE mini_requests_total counter\n");
    body.push_str(&format!("mini_requests_total {}\n\n", total_requests));

    body.push_str("# HELP mini_cache_utilization KV cache utilization fraction\n");
    body.push_str("# TYPE mini_cache_utilization gauge\n");
    body.push_str(&format!("mini_cache_utilization {}\n\n", cache.utilization));

    body.push_str("# HELP mini_cache_fragmentation KV cache fragmentation fraction\n");
    body.push_str("# TYPE mini_cache_fragmentation gauge\n");
    body.push_str(&format!(
        "mini_cache_fragmentation {}\n",
        cache.fragmentation
    ));

    let memory = state.prefix_cache.lock().memory_budget().stats();
    for (name, value) in [
        ("budget", memory.budget_bytes),
        ("headroom", memory.headroom_bytes),
        ("active_kv_reserved", memory.active_kv_reserved_bytes),
        ("execution_reserved", memory.execution_reserved_bytes),
        ("prefix_retained", memory.prefix_retained_bytes),
        ("reserved", memory.reserved_bytes),
        ("available", memory.available_bytes),
        ("peak_reserved", memory.peak_reserved_bytes),
    ] {
        body.push_str(&format!("# HELP mini_tensor_memory_{name}_bytes Tensor memory reservation accounting in bytes (excludes weights)\n# TYPE mini_tensor_memory_{name}_bytes gauge\nmini_tensor_memory_{name}_bytes {value}\n"));
    }
    body.push_str(&format!("# HELP mini_tensor_memory_rejections_total Failed tensor memory reservations\n# TYPE mini_tensor_memory_rejections_total counter\nmini_tensor_memory_rejections_total {}\n", memory.rejections));
    body.push_str(&format!("# HELP mini_prefix_tensor_evictions_total Evicted prefix tensor entries\n# TYPE mini_prefix_tensor_evictions_total counter\nmini_prefix_tensor_evictions_total {}\n", state.prefix_cache.lock().tensor_evictions()));

    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        body,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn channel_closure_is_an_error_for_json_and_sse() {
        for streaming in [false, true] {
            let (tx, rx) = tokio::sync::mpsc::channel(2);
            tx.send(TokenEvent::Token("partial".into())).await.unwrap();
            drop(tx);
            if streaming {
                let events: Vec<_> = response_events(rx).collect().await;
                assert_eq!(events.len(), 2);
                assert!(
                    matches!(&events[1], TokenEvent::Error(e) if e.contains("without terminal"))
                );
            } else {
                assert!(collect_response(rx)
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("without terminal"));
            }
        }
    }
}
