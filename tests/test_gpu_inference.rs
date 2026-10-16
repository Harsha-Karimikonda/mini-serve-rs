//! Explicit GPU completion gate; fails rather than falling back to CPU or skipping its body.
#![cfg(feature = "metal")]

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{body::Body, http::Request};
use http_body_util::BodyExt;
use rlse::{
    api::{create_app, AppState},
    backends::{candle::CandleBackend, ModelBackend},
    cache::{create_shared_cache, prefix_cache::PrefixCache, tensor_memory::TensorMemoryBudget},
    core::{
        config::Settings,
        types::{RequestId, SamplingParams, TokenEvent, TokenUsage},
    },
    routing::router::{Router, WorkerHandle},
    scheduler::Scheduler,
    telemetry::create_telemetry,
};
use serde_json::{json, Value};
use tokio::sync::mpsc::Receiver;
use tower::ServiceExt;

const MODEL: &str = "models/Qwen3-0.6B";
const PROMPT: &str = "The capital of France is";

// Inject a returned error only after a successful real-model GPU forward.
// This tests serving cleanup and quarantine, not driver OOM/recovery.
struct FaultAfterForward {
    model: Arc<CandleBackend>,
    remaining: AtomicUsize,
}
#[async_trait::async_trait]
impl ModelBackend for FaultAfterForward {
    fn name(&self) -> &str {
        self.model.name()
    }
    fn context_length(&self) -> Option<usize> {
        self.model.context_length()
    }
    fn is_healthy(&self) -> bool {
        self.model.is_healthy()
    }
    fn mark_unhealthy(&self) {
        self.model.mark_unhealthy();
    }
    async fn init_sequence(
        &self,
        prompt: &str,
    ) -> Result<Vec<u32>, rlse::core::errors::EngineError> {
        self.model.init_sequence(prompt).await
    }
    fn reserve_sequence(
        &self,
        id: &RequestId,
        prompt: usize,
        output: usize,
    ) -> Result<(), rlse::core::errors::EngineError> {
        self.model.reserve_sequence(id, prompt, output)
    }
    fn cancel_sequence(&self, id: &RequestId) {
        self.model.cancel_sequence(id);
    }
    async fn step_batch(
        &self,
        seqs: &mut [rlse::core::types::ActiveSequence],
    ) -> Result<Vec<rlse::backends::StepToken>, rlse::core::errors::EngineError> {
        let tokens = self.model.step_batch(seqs).await?;
        if self
            .remaining
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .ok()
            == Some(1)
        {
            return Err(rlse::core::errors::EngineError::BackendError(
                "injected failure after real GPU forward".into(),
            ));
        }
        Ok(tokens)
    }
}

struct Output {
    text: String,
    fragments: Vec<String>,
    usage: TokenUsage,
    reason: String,
}

fn sampling(max_tokens: usize) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        max_tokens,
        stop: vec![],
    }
}

async fn collect(mut rx: Receiver<TokenEvent>) -> Output {
    tokio::time::timeout(Duration::from_secs(60), async {
        let mut fragments = Vec::new();
        loop {
            match rx
                .recv()
                .await
                .expect("generation closed without terminal state")
            {
                TokenEvent::Token(text) => fragments.push(text),
                TokenEvent::Done {
                    usage,
                    finish_reason,
                } => {
                    assert_eq!(usage.completion_tokens, fragments.len());
                    assert_eq!(
                        usage.total_tokens,
                        usage.prompt_tokens + usage.completion_tokens
                    );
                    let text = fragments.concat();
                    assert!(!text.contains("<|im_end|>") && !text.contains("<|endoftext|>"));
                    return Output {
                        text,
                        fragments,
                        usage,
                        reason: finish_reason,
                    };
                }
                TokenEvent::Error(e) => panic!("GPU generation failed: {e}"),
            }
        }
    })
    .await
    .expect("GPU generation timed out")
}

async fn generate(worker: &Scheduler, prompt: &str, max_tokens: usize) -> Output {
    let result = collect(
        worker
            .submit(
                RequestId::new_completion(),
                prompt.into(),
                sampling(max_tokens),
            )
            .await
            .unwrap(),
    )
    .await;
    println!(
        "prompt={prompt:?}, output={:?}, finish={}, usage={:?}",
        result.text, result.reason, result.usage
    );
    result
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires host Metal GPU access and local Qwen3-0.6B pretrained weights"]
async fn pretrained_gpu_memory_prefix_stops_and_workers() {
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
    let started = std::time::Instant::now();
    assert!(
        std::path::Path::new(MODEL)
            .join("model.safetensors")
            .is_file(),
        "local pretrained weights are required"
    );
    let budget = TensorMemoryBudget::new(512 * 1024 * 1024, 32 * 1024 * 1024).unwrap();
    let prefix = Arc::new(parking_lot::Mutex::new(PrefixCache::with_memory_budget(
        16,
        128,
        budget.clone(),
    )));
    let mut model = CandleBackend::load_hf(MODEL, "metal").unwrap();
    assert!(
        matches!(model.device(), candle_core::Device::Metal(_)),
        "GPU gate cannot use CPU fallback"
    );
    println!(
        "ACTUAL_DEVICE={:?}; model={MODEL}; greedy temperature=0 top_p=1",
        model.device()
    );
    model.try_set_prefix_cache(prefix.clone()).unwrap();
    let model = Arc::new(model);
    let fault = Arc::new(FaultAfterForward {
        model: model.clone(),
        remaining: AtomicUsize::new(0),
    });
    let cache = create_shared_cache(1024, 16);
    let workers: Vec<_> = (0..2)
        .map(|_| {
            Arc::new(Scheduler::new(
                4,
                32,
                cache.clone(),
                prefix.clone(),
                fault.clone(),
            ))
        })
        .collect();
    let handles: Vec<_> = workers.iter().map(Scheduler::start_loop).collect();
    let router = Arc::new(Router::new(
        workers
            .iter()
            .enumerate()
            .map(|(i, w)| WorkerHandle::new(format!("gpu-{i}"), w.clone()))
            .collect(),
    ));
    let app = create_app(AppState {
        router,
        cache: cache.clone(),
        prefix_cache: prefix.clone(),
        telemetry: create_telemetry(),
        settings: Settings {
            model: MODEL.into(),
            device: "metal".into(),
            tensor_memory_budget_bytes: 512 * 1024 * 1024,
            execution_headroom_bytes: 32 * 1024 * 1024,
            ..Settings::default()
        },
    });

    let plain = generate(&workers[0], PROMPT, 12).await;
    assert!(plain.text.to_lowercase().contains("paris"));
    let mut aligned = PROMPT.to_owned();
    while model.init_sequence(&aligned).await.unwrap().len() < 32 {
        aligned.push_str(" France");
    }
    assert_eq!(model.init_sequence(&aligned).await.unwrap().len(), 32);
    prefix.lock().clear();
    let reference = generate(&workers[0], &aligned, 12).await;
    assert!(budget.stats().prefix_retained_bytes > 0);
    let saved = prefix.lock().stats().2;
    let exact = generate(&workers[0], &aligned, 12).await;
    assert_eq!(exact.text, reference.text, "exact cached output diverged");
    assert!(prefix.lock().stats().2 >= saved + 32);
    let extended = format!("{aligned} and Italy");
    prefix.lock().clear();
    let extended_reference = generate(&workers[0], &extended, 12).await;
    // The 32-token snapshot from a longer prompt has no saved logits.
    let replay = generate(&workers[0], &aligned, 12).await;
    assert_eq!(replay.text, reference.text, "missing-logit replay diverged");
    prefix.lock().clear();
    generate(&workers[0], &aligned, 12).await;
    let partial = generate(&workers[0], &extended, 12).await;
    assert_eq!(
        partial.text, extended_reference.text,
        "partial prefix output diverged"
    );
    println!("PASS: exact prefix with logits, exact prefix without logits, partial prefix outputs match uncached reference");

    let nonempty: Vec<_> = plain.fragments.iter().filter(|s| !s.is_empty()).collect();
    assert!(nonempty.len() >= 3);
    let single = nonempty[0].to_string();
    let spanning = format!("{}{}", nonempty[1], nonempty[2]);
    assert!(
        !plain.fragments.iter().any(|s| s.contains(&spanning)),
        "selected stop must span token fragments"
    );
    let incomplete = format!("{} impossible sentinel", plain.text);
    for (stop, expected, reason) in [
        (single, String::new(), "stop"),
        (
            spanning.clone(),
            plain.text[..plain.text.find(&spanning).unwrap()].into(),
            "stop",
        ),
        (incomplete, plain.text.clone(), "length"),
    ] {
        for streaming in [false, true] {
            let response = app.clone().oneshot(Request::builder().method("POST").uri("/v1/completions").header("content-type", "application/json")
                .body(Body::from(json!({"model": MODEL, "prompt": PROMPT, "temperature": 0, "max_tokens": 12, "stop": [stop], "stream": streaming}).to_string())).unwrap()).await.unwrap();
            assert!(response.status().is_success());
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            let mut output = String::new();
            let mut finish = None;
            if streaming {
                let sse = std::str::from_utf8(&bytes).unwrap();
                assert!(sse.contains("data: [DONE]"));
                for frame in sse
                    .lines()
                    .filter_map(|l| l.strip_prefix("data: "))
                    .filter(|l| *l != "[DONE]")
                {
                    let chunk: Value = serde_json::from_str(frame).unwrap();
                    assert!(chunk.get("error").is_none(), "{chunk}");
                    output.push_str(chunk["choices"][0]["text"].as_str().unwrap());
                    assert!(!output.contains(&stop), "SSE stop leaked");
                    if let Some(r) = chunk["choices"][0]["finish_reason"].as_str() {
                        finish = Some(r.to_owned());
                    }
                }
            } else {
                let body: Value = serde_json::from_slice(&bytes).unwrap();
                output = body["choices"][0]["text"].as_str().unwrap().into();
                finish = Some(body["choices"][0]["finish_reason"].as_str().unwrap().into());
                assert!(body["usage"]["completion_tokens"].as_u64().unwrap() <= 12);
            }
            assert_eq!(output, expected);
            assert_eq!(finish.as_deref(), Some(reason));
            println!(
                "PASS: GPU {} stop={stop:?}, output={output:?}, finish={reason}",
                if streaming { "SSE" } else { "JSON" }
            );
        }
    }

    let eos = generate(&workers[0], "<|im_start|>user\nReply with only the word Paris.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", 32).await;
    assert_eq!(
        eos.reason, "stop",
        "EOS probe did not reach a natural stop within 32 tokens"
    );
    let italy = "The capital of Italy is";
    let italy_reference = generate(&workers[0], italy, 12).await;
    let a = workers[0]
        .submit(RequestId::new_completion(), PROMPT.into(), sampling(12))
        .await
        .unwrap();
    let b = workers[1]
        .submit(RequestId::new_completion(), italy.into(), sampling(12))
        .await
        .unwrap();
    let (a, b) = tokio::join!(collect(a), collect(b));
    assert_eq!(a.text, plain.text);
    assert_eq!(b.text, italy_reference.text);
    println!("PASS: two shared-weight workers match isolated output");

    let mut cancelled = workers[0]
        .submit(RequestId::new_completion(), PROMPT.into(), sampling(512))
        .await
        .unwrap();
    assert!(matches!(cancelled.recv().await, Some(TokenEvent::Token(_))));
    drop(cancelled);
    tokio::time::timeout(Duration::from_secs(10), async {
        while workers.iter().any(|w| w.active_count() != 0) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(budget.stats().active_kv_reserved_bytes, 0);
    assert_eq!(budget.stats().execution_reserved_bytes, 0);
    assert_eq!(cache.lock().allocated_count(), 0);
    println!("PASS: cancellation releases active tensor reservations and logical KV blocks");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model": MODEL, "prompt": PROMPT, "temperature": 0, "max_tokens": 4000})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        axum::http::StatusCode::INTERNAL_SERVER_ERROR
    );
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert!(std::str::from_utf8(&body)
        .unwrap()
        .contains("Tensor memory budget exhausted"));
    assert_eq!(budget.stats().prefix_retained_bytes, 0);
    assert_eq!(budget.stats().reserved_bytes, budget.stats().headroom_bytes);
    assert!(budget.stats().rejections > 0);
    assert!(budget.stats().peak_reserved_bytes <= budget.stats().budget_bytes);
    let recovered = generate(&workers[0], PROMPT, 12).await;
    assert_eq!(recovered.text, plain.text);
    assert_eq!(cache.lock().allocated_count(), 0);
    assert!(workers.iter().all(|w| w.is_healthy()));
    fault.remaining.store(2, Ordering::SeqCst);
    let response = app.clone().oneshot(Request::builder().method("POST").uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(json!({"model":MODEL, "prompt":PROMPT, "temperature":0, "max_tokens":128, "stream":true}).to_string())).unwrap()).await.unwrap();
    assert!(response.status().is_success());
    let body = tokio::time::timeout(Duration::from_secs(30), response.into_body().collect())
        .await
        .unwrap()
        .unwrap()
        .to_bytes();
    let text = std::str::from_utf8(&body).unwrap();
    assert!(
        text.contains("injected failure after real GPU forward"),
        "{text}"
    );
    assert!(
        text.contains(" Paris"),
        "the first real GPU token must precede the error: {text}"
    );
    assert!(!text.contains("[DONE]"));
    assert!(!model.is_healthy());
    assert!(workers.iter().all(|w| !w.is_healthy()));
    assert_eq!(budget.stats().active_kv_reserved_bytes, 0);
    assert_eq!(budget.stats().execution_reserved_bytes, 0);
    assert_eq!(budget.stats().prefix_retained_bytes, 0);
    assert_eq!(cache.lock().allocated_count(), 0);
    let ready = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ready.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
    let rejected = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"model":MODEL, "prompt":PROMPT, "max_tokens":12}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        rejected.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );
    println!("PASS: injected returned failure after real Metal prefill/decode emits SSE error, quarantines both shared workers, clears active/prefix reservations, makes readiness and subsequent admission return 503; no automatic retry");
    for worker in &workers {
        worker.shutdown();
    }
    for handle in handles {
        handle.await.unwrap();
    }
    prefix.lock().clear();
    assert_eq!(budget.stats().reserved_bytes, budget.stats().headroom_bytes);
    println!("PASS: budget rejection evicts prefixes, reclaims state, and subsequent GPU inference succeeds; stats={:?}; elapsed={:?}", budget.stats(), started.elapsed());
}
