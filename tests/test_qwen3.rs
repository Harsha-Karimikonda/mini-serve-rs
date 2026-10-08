use mini_serve::backends::candle::CandleBackend;
use mini_serve::cache::{create_shared_cache, create_shared_prefix_cache};
use mini_serve::core::types::{RequestId, SamplingParams};
use mini_serve::scheduler::scheduler::Scheduler;
use mini_serve::scheduler::sequence::TokenEvent;
use std::path::Path;
use std::sync::Arc;

#[tokio::test]
async fn test_qwen3_load_and_generate() {
    let model_path = "models/Qwen3-0.6B";
    if !Path::new(model_path).exists() {
        eprintln!("Skipping test_qwen3_load_and_generate: model directory not found");
        return;
    }

    println!("Initializing CandleBackend with Qwen3-0.6B on Metal...");
    let backend = Arc::new(
        CandleBackend::load_hf(model_path, "metal")
            .expect("Failed to load Qwen3-0.6B model via CandleBackend"),
    );

    let cache = create_shared_cache(256, 16);
    let prefix_cache = create_shared_prefix_cache(16, 128);
    let scheduler = Arc::new(Scheduler::new(4, 32, cache, prefix_cache, backend));
    scheduler.start_loop();

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
    while let Some(evt) = rx.recv().await {
        match evt {
            TokenEvent::Token(token) => {
                print!("{}", token);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                generated_tokens.push(token);
            }
            TokenEvent::Done(usage) => {
                println!(
                    "\n[Sequence completed: {} prompt tokens, {} completion tokens]",
                    usage.prompt_tokens, usage.completion_tokens
                );
                break;
            }
            TokenEvent::Error(err) => {
                panic!("Scheduler error during generation: {}", err);
            }
        }
    }

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

