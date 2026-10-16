use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{body::Body, http::Request};
use http_body_util::BodyExt;
use rlse::{
    api::{create_app, AppState},
    backends::{mock::MockBackend, ModelBackend, StepToken},
    cache::{create_shared_cache, create_shared_prefix_cache},
    core::{config::Settings, errors::EngineError, types::ActiveSequence},
    routing::router::{Router, WorkerHandle},
    scheduler::Scheduler,
    telemetry::create_telemetry,
};
use serde_json::{json, Value};
use tower::ServiceExt;

struct Script {
    fragments: Vec<&'static str>,
    eos: bool,
}

#[async_trait]
impl ModelBackend for Script {
    fn name(&self) -> &str {
        "mock"
    }
    async fn init_sequence(&self, _: &str) -> Result<Vec<u32>, EngineError> {
        Ok(vec![1])
    }
    async fn step_batch(
        &self,
        sequences: &mut [ActiveSequence],
    ) -> Result<Vec<StepToken>, EngineError> {
        Ok(sequences
            .iter()
            .map(|seq| {
                let index = seq.output_tokens.len();
                StepToken {
                    token_id: index as u32,
                    text: self.fragments[index].into(),
                    is_eos: self.eos && index + 1 == self.fragments.len(),
                }
            })
            .collect())
    }
}

async fn assert_responses(
    backend: Arc<dyn ModelBackend>,
    stop: Vec<&str>,
    max_tokens: usize,
    expected: &str,
    reason: &str,
    generated: usize,
) {
    for chat in [false, true] {
        for streaming in [false, true] {
            let cache = create_shared_cache(64, 16);
            let prefix_cache = create_shared_prefix_cache(16, 16);
            let telemetry = create_telemetry();
            let scheduler = Arc::new(Scheduler::new(
                1,
                8,
                cache.clone(),
                prefix_cache.clone(),
                backend.clone(),
            ));
            let handle = scheduler.start_loop();
            let app = create_app(AppState {
                router: Arc::new(Router::new(vec![WorkerHandle::new(
                    "test",
                    scheduler.clone(),
                )])),
                cache: cache.clone(),
                prefix_cache,
                telemetry: telemetry.clone(),
                settings: Settings::default(),
            });
            let payload = json!({ "model": "mock", "prompt": "one", "messages": [{"role": "user", "content": "one"}],
                "max_tokens": max_tokens, "stop": stop, "stream": streaming });
            let route = if chat {
                "/v1/chat/completions"
            } else {
                "/v1/completions"
            };
            let response = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(route)
                        .header("content-type", "application/json")
                        .body(Body::from(payload.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), axum::http::StatusCode::OK);
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            let mut text = String::new();
            let finish_reason;
            if streaming {
                let body = std::str::from_utf8(&bytes).unwrap();
                assert!(body.contains("data: [DONE]"));
                let mut finish = None;
                for line in body.lines().filter_map(|line| line.strip_prefix("data: ")) {
                    if line == "[DONE]" {
                        continue;
                    }
                    let chunk: Value = serde_json::from_str(line).unwrap();
                    assert!(chunk.get("error").is_none(), "{chunk}");
                    let choice = &chunk["choices"][0];
                    let fragment = if chat {
                        &choice["delta"]["content"]
                    } else {
                        &choice["text"]
                    };
                    text.push_str(fragment.as_str().unwrap_or_default());
                    if let Some(reason) = choice["finish_reason"].as_str() {
                        finish = Some(reason.to_owned());
                    }
                    // Even intermediate SSE text must not reveal a stop marker.
                    for stop in &stop {
                        assert!(!text.contains(stop), "{text:?} leaked {stop:?}");
                    }
                }
                finish_reason = finish.unwrap();
            } else {
                let body: Value = serde_json::from_slice(&bytes).unwrap();
                let choice = &body["choices"][0];
                text = if chat {
                    choice["message"]["content"].as_str().unwrap()
                } else {
                    choice["text"].as_str().unwrap()
                }
                .to_owned();
                finish_reason = choice["finish_reason"].as_str().unwrap().to_owned();
                assert_eq!(body["usage"]["completion_tokens"], generated);
            }
            assert_eq!(text, expected, "chat={chat}, streaming={streaming}");
            assert_eq!(finish_reason, reason);
            assert_eq!(telemetry.total_tokens(), generated);
            scheduler.shutdown();
            handle.await.unwrap();
            assert_eq!(cache.lock().allocated_count(), 0);
        }
    }
}

#[tokio::test]
async fn stop_markers_are_withheld_for_both_endpoints_and_response_modes() {
    tokio::time::timeout(Duration::from_secs(5), async {
        for (fragments, stop, expected, generated) in [
            (vec!["Hi STOP tail"], vec!["STOP"], "Hi ", 1),
            (vec!["Hi S", "T", "OP tail"], vec!["STOP"], "Hi ", 3),
            (vec!["a", "b", "cd tail"], vec!["abcd", "bc"], "a", 3),
            (vec!["café 終", "わ", "り tail"], vec!["終わり"], "café ", 3),
            (vec!["STOPPED"], vec!["STOPPED", "STOP"], "", 1),
        ] {
            let limit = fragments.len();
            assert_responses(
                Arc::new(Script {
                    fragments,
                    eos: false,
                }),
                stop,
                limit,
                expected,
                "stop",
                generated,
            )
            .await;
        }
    })
    .await
    .expect("stop matching stalled generation");
}

#[tokio::test]
async fn normal_termination_flushes_partial_stops_and_counts_empty_decodes() {
    tokio::time::timeout(Duration::from_secs(5), async {
        assert_responses(
            Arc::new(Script {
                fragments: vec!["Hi ST", ""],
                eos: true,
            }),
            vec!["STOP"],
            10,
            "Hi ST",
            "stop",
            2,
        )
        .await;
        assert_responses(
            Arc::new(Script {
                fragments: vec!["Hi ST"],
                eos: false,
            }),
            vec!["STOP"],
            1,
            "Hi ST",
            "length",
            1,
        )
        .await;
        assert_responses(
            Arc::new(Script {
                fragments: vec!["", "hello", ""],
                eos: true,
            }),
            vec![],
            10,
            "hello",
            "stop",
            3,
        )
        .await;
    })
    .await
    .expect("normal termination stalled");
}

#[tokio::test]
async fn mock_uses_shared_stop_matching_and_flush_contract() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let mock = Arc::new(MockBackend::new("mock", 0));
        assert_responses(
            mock.clone(),
            vec!["batching in"],
            10,
            "Continuous ",
            "stop",
            3,
        )
        .await;
        assert_responses(
            mock,
            vec!["batching in"],
            2,
            "Continuous batching",
            "length",
            2,
        )
        .await;
    })
    .await
    .expect("mock stop matching stalled");
}
