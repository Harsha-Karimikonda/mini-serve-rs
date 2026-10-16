use rlse::backends::candle::CandleBackend;
use rlse::cache::{create_shared_cache, create_shared_prefix_cache};
use rlse::core::types::{RequestId, SamplingParams};
use rlse::scheduler::scheduler::Scheduler;
use rlse::scheduler::sequence::TokenEvent;
use std::path::Path;
use std::sync::Arc;

#[tokio::test]
#[ignore = "requires local Qwen3-0.6B weights in models/Qwen3-0.6B"]
async fn test_qwen3_load_and_generate() {
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
    let started = std::time::Instant::now();
    let model_path = "models/Qwen3-0.6B";
    assert!(
        Path::new(model_path).is_dir(),
        "Required model weights are missing: {model_path}"
    );
    let device = if cfg!(feature = "metal") {
        "metal"
    } else {
        "cpu"
    };
    println!("Initializing CandleBackend with Qwen3-0.6B on {device}...");
    let backend = Arc::new(
        CandleBackend::load_hf(model_path, device)
            .expect("Failed to load Qwen3-0.6B model via CandleBackend"),
    );

    let cache = create_shared_cache(256, 16);
    let prefix_cache = create_shared_prefix_cache(16, 128);
    let scheduler = Arc::new(Scheduler::new(4, 32, cache.clone(), prefix_cache, backend));
    let worker = scheduler.start_loop();

    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        max_tokens: 12,
        stop: vec![],
    };

    println!("Submitting prompt: 'The capital of France is'...");
    let mut rx = scheduler
        .submit(
            RequestId("qwen3-test-1".to_string()),
            "The capital of France is".to_string(),
            sampling,
        )
        .await
        .expect("Failed to submit request to scheduler");

    let mut generated_tokens = Vec::new();
    let mut terminal = None;
    while let Some(evt) = rx.recv().await {
        match evt {
            TokenEvent::Token(token) => {
                print!("{}", token);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                generated_tokens.push(token);
            }
            TokenEvent::Done {
                usage,
                finish_reason,
            } => {
                println!(
                    "\n[Sequence completed: {} prompt tokens, {} completion tokens]",
                    usage.prompt_tokens, usage.completion_tokens
                );
                terminal = Some((usage, finish_reason));
                break;
            }
            TokenEvent::Error(err) => {
                panic!("Scheduler error during generation: {}", err);
            }
        }
    }

    let (usage, finish_reason) = terminal.expect("Inference closed without a terminal event");
    assert_eq!(usage.completion_tokens, generated_tokens.len());
    assert_eq!(
        usage.total_tokens,
        usage.prompt_tokens + usage.completion_tokens
    );
    assert!(matches!(finish_reason.as_str(), "stop" | "length"));
    assert_eq!(
        cache.lock().allocated_count(),
        0,
        "Inference leaked logical KV blocks"
    );
    assert_eq!(scheduler.active_count(), 0);
    scheduler.shutdown();
    worker.await.expect("Scheduler task failed");
    println!(
        "Verified terminal={finish_reason}, usage={usage:?}, cache released; elapsed={:?}",
        started.elapsed()
    );
    assert!(
        !generated_tokens.is_empty(),
        "Qwen3 should successfully generate tokens"
    );
    let full_text: String = generated_tokens.concat();
    println!("Full output text: '{}'", full_text);
    assert!(
        full_text.to_lowercase().contains("paris"),
        "Generated text should contain Paris! Got: {}",
        full_text
    );
}
