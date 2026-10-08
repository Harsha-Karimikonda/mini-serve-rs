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
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::wrappers::ReceiverStream;

use crate::api::dashboard::DASHBOARD_HTML;
use crate::cache::{SharedKVCache, SharedPrefixCache};
use crate::core::config::Settings;
use crate::core::errors::EngineError;
use crate::core::types::*;
use crate::routing::router::SharedRouter;
use crate::scheduler::sequence::TokenEvent;
use crate::telemetry::metrics::SharedTelemetry;

#[derive(Clone)]
pub struct AppState {
    pub router: SharedRouter,
    pub cache: SharedKVCache,
    pub prefix_cache: SharedPrefixCache,
    pub telemetry: SharedTelemetry,
    pub settings: Settings,
}

pub fn create_app(state: AppState) -> Router {
    Router::new()
        .route("/", get(dashboard_handler))
        .route("/dashboard", get(dashboard_handler))
        .route("/health", get(health_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/completions", post(completions_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/api/status", get(status_handler))
        .route("/metrics", get(metrics_handler))
        .with_state(state)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

async fn dashboard_handler() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

async fn health_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "model": state.settings.model,
        "device": state.settings.device,
        "workers": state.router.workers().len(),
    }))
}

async fn models_handler(State(state): State<AppState>) -> Json<ModelListResponse> {
    let now = now_secs();
    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![
            ModelCard {
                id: state.settings.model.clone(),
                object: "model".to_string(),
                created: now,
                owned_by: "mini-serve-rs".to_string(),
            },
            ModelCard {
                id: "mock".to_string(),
                object: "model".to_string(),
                created: now,
                owned_by: "system".to_string(),
            },
        ],
    })
}

async fn completions_handler(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, EngineError> {
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
        .submit(request_id.clone(), req.prompt, sampling)
        .await?;

    if req.stream {
        let stream_req_id = request_id.0.clone();
        let stream_model = req.model.clone();
        let telemetry = state.telemetry.clone();

        let sse_stream = ReceiverStream::new(rx).map(move |event| match event {
            TokenEvent::Token(token_text) => {
                telemetry.record_tokens(1);
                let chunk = CompletionChunk {
                    id: stream_req_id.clone(),
                    object: "text_completion.chunk".to_string(),
                    created: now_secs(),
                    model: stream_model.clone(),
                    choices: vec![CompletionChunkChoice {
                        text: token_text,
                        index: 0,
                        finish_reason: None,
                    }],
                };
                Ok::<Event, Infallible>(
                    Event::default().data(serde_json::to_string(&chunk).unwrap()),
                )
            }
            TokenEvent::Done(_) => Ok::<Event, Infallible>(Event::default().data("[DONE]")),
            TokenEvent::Error(err_msg) => {
                let err_json = json!({
                    "error": {
                        "message": err_msg,
                        "type": "server_error",
                        "code": 500
                    }
                });
                Ok::<Event, Infallible>(
                    Event::default().data(serde_json::to_string(&err_json).unwrap()),
                )
            }
        });

        Ok(Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let mut rx = rx;
        let mut full_text = String::new();
        let mut final_usage = TokenUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        };

        while let Some(event) = rx.recv().await {
            match event {
                TokenEvent::Token(text) => {
                    full_text.push_str(&text);
                }
                TokenEvent::Done(usage) => {
                    final_usage = usage;
                    break;
                }
                TokenEvent::Error(err) => {
                    return Err(EngineError::BackendError(err));
                }
            }
        }

        state.telemetry.record_tokens(final_usage.completion_tokens);

        let response = CompletionResponse {
            id: request_id.0,
            object: "text_completion".to_string(),
            created: now_secs(),
            model: req.model,
            choices: vec![CompletionChoice {
                text: full_text,
                index: 0,
                logprobs: None,
                finish_reason: Some("stop".to_string()),
            }],
            usage: final_usage,
        };

        Ok(Json(response).into_response())
    }
}

async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, EngineError> {
    state.telemetry.record_request();

    if req.messages.is_empty() {
        return Err(EngineError::InvalidRequest(
            "Messages cannot be empty".to_string(),
        ));
    }

    // Format chat messages into prompt
    let mut prompt = String::new();
    for msg in &req.messages {
        prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
    }
    prompt.push_str("assistant: ");

    let request_id = RequestId::new_chat();
    let sampling = SamplingParams {
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        stop: req.stop,
    };

    let rx = state
        .router
        .submit(request_id.clone(), prompt, sampling)
        .await?;

    if req.stream {
        let stream_req_id = request_id.0.clone();
        let stream_model = req.model.clone();
        let telemetry = state.telemetry.clone();

        let sse_stream = ReceiverStream::new(rx).map(move |event| match event {
            TokenEvent::Token(token_text) => {
                telemetry.record_tokens(1);
                let chunk = ChatCompletionChunk {
                    id: stream_req_id.clone(),
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
                Ok::<Event, Infallible>(
                    Event::default().data(serde_json::to_string(&chunk).unwrap()),
                )
            }
            TokenEvent::Done(_) => Ok::<Event, Infallible>(Event::default().data("[DONE]")),
            TokenEvent::Error(err_msg) => {
                let err_json = json!({
                    "error": {
                        "message": err_msg,
                        "type": "server_error",
                        "code": 500
                    }
                });
                Ok::<Event, Infallible>(
                    Event::default().data(serde_json::to_string(&err_json).unwrap()),
                )
            }
        });

        Ok(Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let mut rx = rx;
        let mut full_text = String::new();
        let mut final_usage = TokenUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        };

        while let Some(event) = rx.recv().await {
            match event {
                TokenEvent::Token(text) => {
                    full_text.push_str(&text);
                }
                TokenEvent::Done(usage) => {
                    final_usage = usage;
                    break;
                }
                TokenEvent::Error(err) => {
                    return Err(EngineError::BackendError(err));
                }
            }
        }

        state.telemetry.record_tokens(final_usage.completion_tokens);

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
                finish_reason: Some("stop".to_string()),
            }],
            usage: final_usage,
        };

        Ok(Json(response).into_response())
    }
}

async fn status_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let cache_stats = state.cache.lock().stats();
    let (hits, misses, tokens_saved) = state.prefix_cache.lock().stats();
    let hit_rate = state.prefix_cache.lock().hit_rate();

    let workers_info: Vec<serde_json::Value> = state
        .router
        .workers()
        .iter()
        .map(|w| {
            let m = w.scheduler.metrics();
            json!({
                "id": w.id,
                "healthy": true,
                "queue_depth": m.queue_depth,
                "active_count": m.active_count,
                "avg_latency_ms": m.avg_latency_ms,
                "score": (w.score() * 10.0).round() / 10.0
            })
        })
        .collect();

    Json(json!({
        "status": "healthy",
        "engine": "mini-serve-rs",
        "version": "0.2.0",
        "model": state.settings.model,
        "tokens_per_sec": state.telemetry.tokens_per_second(),
        "total_tokens": state.telemetry.total_tokens(),
        "total_requests": state.telemetry.total_requests(),
        "workers": workers_info,
        "cache": {
            "total_blocks": cache_stats.total_blocks,
            "allocated_blocks": cache_stats.allocated_blocks,
            "free_blocks": cache_stats.free_blocks,
            "utilization": cache_stats.utilization,
            "fragmentation": cache_stats.fragmentation
        },
        "prefix_cache": {
            "hits": hits,
            "misses": misses,
            "tokens_saved": tokens_saved,
            "hit_rate_pct": (hit_rate * 10.0).round() / 10.0
        }
    }))
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    let tps = state.telemetry.tokens_per_second();
    let total_tokens = state.telemetry.total_tokens();
    let total_requests = state.telemetry.total_requests();
    let cache = state.cache.lock().stats();

    let mut body = String::new();
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

    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        body,
    )
}
