//! Manual diagnostics. Known-bug tests assert the desired behavior and may fail.
//! Run with `cargo test --lib production_reproductions -- --include-ignored --nocapture
//! --test-threads=1 --skip real_model`.

use super::*;
use crate::api::{create_app, AppState};
use crate::cache::{create_shared_cache, create_shared_prefix_cache};
use crate::core::config::Settings;
use crate::core::types::{SamplingParams, TokenEvent};
use crate::routing::router::{Router, WorkerHandle};
use crate::scheduler::scheduler::Scheduler;
use crate::telemetry::create_telemetry;
use axum::{body::Body, http::Request};
use http_body_util::BodyExt;
use std::sync::mpsc as sync_mpsc;
use std::time::{Duration, Instant};
use tower::ServiceExt;

fn fixture() -> CandleBackend {
    let config: QwenConfig = serde_json::from_value(serde_json::json!({
        "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "max_position_embeddings": 128,
        "hidden_act": "silu", "tie_word_embeddings": true
    }))
    .unwrap();
    let mut model = QwenModel::new(&config, VarBuilder::zeros(DType::F32, &Device::Cpu)).unwrap();
    // Equal logits deterministically select token 0, decoded as "User".
    model.lm_head = candle_nn::Linear::new(
        Tensor::zeros((64, 32), DType::F32, &Device::Cpu).unwrap(),
        None,
    );
    let vocab = (0..64)
        .map(|id| {
            (
                if id == 0 {
                    "User".into()
                } else {
                    format!("t{id}")
                },
                id,
            )
        })
        .collect();
    let tokenizer = Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .unk_token("t1".into())
            .build()
            .unwrap(),
    );
    CandleBackend {
        name: "tiny-reproduction".into(),
        device: Device::Cpu,
        tokenizer,
        model_template: Arc::new(model),
        active_models: Mutex::new(Vec::new()),
        eos_token_id: u32::MAX,
        context_length: config.max_position_embeddings,
        memory_config: config,
        memory_budget: TensorMemoryBudget::new(
            DEFAULT_TENSOR_MEMORY_BUDGET,
            DEFAULT_EXECUTION_HEADROOM,
        )
        .unwrap(),
        reservations: Mutex::new(HashMap::new()),
        prefix_cache: None,
    }
}

fn app(backend: Arc<dyn ModelBackend>) -> (axum::Router, Arc<Scheduler>) {
    let cache = create_shared_cache(256, 16);
    let prefix_cache = create_shared_prefix_cache(16, 128);
    let scheduler = Arc::new(Scheduler::new(
        4,
        32,
        cache.clone(),
        prefix_cache.clone(),
        backend,
    ));
    let router = Arc::new(Router::new(vec![WorkerHandle::new(
        "reproduction",
        scheduler.clone(),
    )]));
    let state = AppState {
        router,
        cache,
        prefix_cache,
        telemetry: create_telemetry(),
        settings: Settings::default(),
    };
    (create_app(state), scheduler)
}

async fn complete(stop: &str) -> (String, String) {
    let (_app, scheduler) = app(Arc::new(fixture()));
    let mut rx = scheduler
        .submit(
            RequestId("stop-reproduction".into()),
            "t1".into(),
            SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                max_tokens: 4,
                stop: vec![stop.into()],
            },
        )
        .await
        .unwrap();
    let task = scheduler.start_loop();
    let mut text = String::new();
    let reason = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match rx.recv().await.expect("missing terminal event") {
                TokenEvent::Token(token) => text.push_str(&token),
                TokenEvent::Done { finish_reason, .. } => break finish_reason,
                TokenEvent::Error(error) => panic!("{error}"),
            }
        }
    })
    .await
    .unwrap();
    scheduler.shutdown();
    task.await.unwrap();
    (text, reason)
}

#[tokio::test]
async fn stop_spanning_tokens_is_withheld() {
    let (text, reason) = complete("User User").await;
    eprintln!("C: multi-token stop='User User', output={text:?}, finish={reason}");
    assert!(
        !text.contains("User User"),
        "multi-token stop leaked: {text:?}"
    );
    assert_eq!(reason, "stop");
}

#[tokio::test]
async fn matched_stop_text_is_withheld() {
    let (text, reason) = complete("User").await;
    eprintln!("C: single-token stop='User', output={text:?}, finish={reason}");
    assert_eq!(text, "", "matched stop text must be stripped");
}

struct CapturePrompt(Mutex<Option<String>>);

#[async_trait]
impl ModelBackend for CapturePrompt {
    fn name(&self) -> &str {
        "capture"
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
#[ignore = "known gap: route does not apply the Qwen-style chat template"]
async fn chat_route_applies_role_delimiters() {
    let backend = Arc::new(CapturePrompt(Mutex::new(None)));
    let (app, scheduler) = app(backend.clone());
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

struct SynchronousForward;

#[async_trait]
impl ModelBackend for SynchronousForward {
    fn name(&self) -> &str {
        "controlled-blocking"
    }
    async fn init_sequence(&self, _: &str) -> Result<Vec<u32>, EngineError> {
        Ok(vec![1])
    }
    async fn step_batch(
        &self,
        sequences: &mut [ActiveSequence],
    ) -> Result<Vec<StepToken>, EngineError> {
        std::thread::sleep(Duration::from_millis(250));
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

struct ObserveForward {
    inner: Arc<dyn ModelBackend>,
    entered: Mutex<Option<sync_mpsc::Sender<()>>>,
    released: Mutex<sync_mpsc::Receiver<()>>,
}

#[async_trait]
impl ModelBackend for ObserveForward {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn context_length(&self) -> Option<usize> {
        self.inner.context_length()
    }
    async fn init_sequence(&self, prompt: &str) -> Result<Vec<u32>, EngineError> {
        self.inner.init_sequence(prompt).await
    }
    fn reserve_sequence(
        &self,
        id: &RequestId,
        prompt: usize,
        max: usize,
    ) -> Result<(), EngineError> {
        self.inner.reserve_sequence(id, prompt, max)
    }
    async fn step_batch(
        &self,
        sequences: &mut [ActiveSequence],
    ) -> Result<Vec<StepToken>, EngineError> {
        if let Some(sender) = self.entered.lock().take() {
            sender.send(()).unwrap();
            self.released
                .lock()
                .recv_timeout(Duration::from_secs(5))
                .unwrap();
        }
        self.inner.step_batch(sequences).await
    }
    fn cancel_sequence(&self, id: &RequestId) {
        self.inner.cancel_sequence(id);
    }
}

async fn health_delay(inner: Arc<dyn ModelBackend>, prompt: String) -> (Duration, Duration) {
    let (entered_tx, entered_rx) = sync_mpsc::channel();
    let (release_tx, release_rx) = sync_mpsc::channel();
    let (sent_tx, sent_rx) = tokio::sync::oneshot::channel();
    let observed = Arc::new(ObserveForward {
        inner,
        entered: Mutex::new(Some(entered_tx)),
        released: Mutex::new(release_rx),
    });
    let (app, scheduler) = app(observed);
    let baseline_start = Instant::now();
    let idle = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(idle.status().is_success());
    let idle_elapsed = baseline_start.elapsed();
    let coordinator = std::thread::spawn(move || {
        entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        sent_tx.send(Instant::now()).unwrap();
        release_tx.send(()).unwrap();
    });
    let mut rx = scheduler
        .submit(
            RequestId("blocking-probe".into()),
            prompt,
            SamplingParams {
                temperature: 0.0,
                max_tokens: 1,
                ..SamplingParams::default()
            },
        )
        .await
        .unwrap();
    let task = scheduler.start_loop();
    // The external thread makes this probe ready before releasing the forward.
    // On this current-thread runtime it cannot be polled until compute yields.
    let ready_at = sent_rx.await.unwrap();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let elapsed = ready_at.elapsed();
    assert!(response.status().is_success());
    loop {
        match rx.recv().await.unwrap() {
            TokenEvent::Done { .. } => break,
            TokenEvent::Error(error) => panic!("{error}"),
            TokenEvent::Token(_) => {}
        }
    }
    scheduler.shutdown();
    task.await.unwrap();
    coordinator.join().unwrap();
    (idle_elapsed, elapsed)
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "known gap: controlled synchronous forward blocks the HTTP runtime worker"]
async fn health_remains_responsive_during_synchronous_forward() {
    let (idle, loaded) = health_delay(Arc::new(SynchronousForward), "probe".into()).await;
    eprintln!("B: controlled 250 ms forward: idle health={idle:?}, ready-to-health={loaded:?}");
    assert!(
        loaded < Duration::from_millis(100),
        "health was blocked for {loaded:?}"
    );
}

#[test]
#[ignore = "manual contention diagnostic; uses real Candle backend on a tiny CPU model"]
fn shared_registry_lock_blocks_an_independent_forward() {
    let backend = Arc::new(fixture());
    let guard = backend.active_models.lock();
    let (entered_tx, entered_rx) = sync_mpsc::channel();
    let (done_tx, done_rx) = sync_mpsc::channel();
    let worker_backend = backend.clone();
    let worker = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let (tx, _rx) = tokio::sync::mpsc::channel(65);
        let mut seq = ActiveSequence::new(
            RequestId("independent".into()),
            "t1".into(),
            SamplingParams {
                temperature: 0.0,
                max_tokens: 4,
                ..SamplingParams::default()
            },
            tx,
        );
        seq.prompt_tokens = vec![1];
        entered_tx.send(()).unwrap();
        let start = Instant::now();
        runtime
            .block_on(worker_backend.step_batch(std::slice::from_mut(&mut seq)))
            .unwrap();
        worker_backend.cancel_sequence(&seq.request_id);
        done_tx.send(start.elapsed()).unwrap();
    });
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let blocked = done_rx.recv_timeout(Duration::from_millis(150));
    drop(guard);
    let elapsed = done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    worker.join().unwrap();
    eprintln!("D: independent Candle call blocked while registry held for 150 ms; completed in {elapsed:?}");
    assert!(matches!(blocked, Err(sync_mpsc::RecvTimeoutError::Timeout)));
    assert_eq!(backend.memory_budget.stats().active_kv_reserved_bytes, 0);
    // This establishes lock contention, not GPU throughput scaling.
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires local Qwen3 weights; use MINI_REPRO_DEVICE=metal with --features metal for GPU"]
async fn real_model_health_latency() {
    let device = std::env::var("MINI_REPRO_DEVICE").unwrap_or_else(|_| "cpu".into());
    let backend = CandleBackend::load_hf("models/Qwen3-0.6B", &device).unwrap();
    eprintln!("B: real model actual device={:?}", backend.device);
    let prompt =
        "A continuous batching engine admits waiting requests between decoding steps. ".repeat(32);
    let tokens = backend
        .tokenizer
        .encode(prompt.clone(), true)
        .unwrap()
        .get_ids()
        .to_vec();
    let backend = Arc::new(backend);
    let (idle, loaded) = health_delay(backend.clone(), prompt).await;
    eprintln!(
        "B: real-model single-runtime-worker: idle health={idle:?}, ready-to-health={loaded:?}"
    );
    // Diagnostic timing only: no hardware-independent latency threshold.

    let mut serial = Vec::new();
    for id in ["serial-a", "serial-b"] {
        let backend = backend.clone();
        let tokens = tokens.clone();
        serial.push(
            std::thread::spawn(move || timed_prefill(backend, id, tokens, None))
                .join()
                .unwrap(),
        );
    }
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let start = Instant::now();
    let workers: Vec<_> = ["concurrent-a", "concurrent-b"]
        .into_iter()
        .map(|id| {
            let backend = backend.clone();
            let tokens = tokens.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || timed_prefill(backend, id, tokens, Some(barrier)))
        })
        .collect();
    let concurrent: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect();
    let wall = start.elapsed();
    eprintln!("D: real model {}-token prefill: two sequential calls={:?}; two simultaneous calls wall={wall:?}, individual latencies={:?}",
        tokens.len(), serial.iter().map(|(elapsed, _)| *elapsed).sum::<Duration>(),
        concurrent.iter().map(|(elapsed, _)| *elapsed).collect::<Vec<_>>());
    assert!(serial
        .iter()
        .chain(&concurrent)
        .all(|(_, token)| *token == serial[0].1));
    assert!(backend.active_models.lock().is_empty());
    assert_eq!(backend.memory_budget.stats().active_kv_reserved_bytes, 0);
}

fn timed_prefill(
    backend: Arc<CandleBackend>,
    id: &str,
    tokens: Vec<u32>,
    barrier: Option<Arc<std::sync::Barrier>>,
) -> (Duration, u32) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (tx, _rx) = tokio::sync::mpsc::channel(65);
    let mut sequence = ActiveSequence::new(
        RequestId(id.into()),
        String::new(),
        SamplingParams {
            temperature: 0.0,
            max_tokens: 2,
            ..SamplingParams::default()
        },
        tx,
    );
    sequence.prompt_tokens = tokens;
    if let Some(barrier) = barrier {
        barrier.wait();
    }
    let start = Instant::now();
    let output = runtime
        .block_on(backend.step_batch(std::slice::from_mut(&mut sequence)))
        .unwrap();
    let elapsed = start.elapsed();
    backend.cancel_sequence(&sequence.request_id);
    (elapsed, output[0].token_id)
}
