//! Concurrent Continuous Batching Demonstration for Mini-Serve
//!
//! Demonstrates iteration-level continuous batching:
//! Multiple concurrent requests with heterogeneous lengths are admitted dynamically.
//! Short sequences complete and exit immediately without being blocked by longer sequences.
//!
//! # How to run:
//! ```bash
//! cargo run --example concurrent_batching
//! cargo run --example concurrent_batching -- --url http://127.0.0.1:8000 --concurrency 6
//! ```

use clap::Parser;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

#[derive(Parser, Debug)]
#[command(
    name = "concurrent-batching",
    about = "Concurrent continuous batching test"
)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    url: String,

    #[arg(long, default_value = "mock")]
    model: String,

    #[arg(long, default_value_t = 6)]
    concurrency: usize,
}

struct RequestSummary {
    id: usize,
    requested_tokens: usize,
    generated_tokens: usize,
    duration_ms: f64,
    tok_per_sec: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let client = reqwest::Client::new();

    println!("===============================================================");
    println!(" ⚡ Mini-Serve Continuous Batching Simulation");
    println!(" Target URL:   {}", args.url);
    println!(" Model:        {}", args.model);
    println!(" Concurrency:  {} simultaneous streams", args.concurrency);
    println!("===============================================================\n");

    let prompts = [
        ("Summarize quantum computing in 10 words", 15),
        (
            "Write a paragraph about distributed consensus and Paxos algorithms",
            35,
        ),
        (
            "Explain how continuous batching eliminates head-of-line blocking in LLM serving",
            50,
        ),
        (
            "List five key benefits of systems programming in pure Rust",
            25,
        ),
        (
            "Draft a short technical memo on zero-copy grouped-query attention",
            40,
        ),
        (
            "Describe the life cycle of an HTTP streaming connection using SSE",
            30,
        ),
        (
            "Explain the difference between L1 cache and main memory latency",
            20,
        ),
        (
            "Write a haiku about GPU memory management and page tables",
            15,
        ),
    ];

    let summaries = Arc::new(Mutex::new(Vec::new()));
    let global_start = Instant::now();
    let mut handles = Vec::new();

    for i in 0..args.concurrency {
        let (prompt_text, max_tokens) = prompts[i % prompts.len()];
        let client = client.clone();
        let url = args.url.clone();
        let model = args.model.clone();
        let summaries = Arc::clone(&summaries);

        let handle = tokio::spawn(async move {
            let req_start = Instant::now();
            let payload = json!({
                "model": model,
                "prompt": prompt_text,
                "max_tokens": max_tokens,
                "stream": true
            });

            let res = match client
                .post(format!("{}/v1/completions", url))
                .json(&payload)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[Req #{}] Failed to send: {}", i + 1, e);
                    return;
                }
            };

            if !res.status().is_success() {
                eprintln!("[Req #{}] HTTP Error: {}", i + 1, res.status());
                return;
            }

            let mut stream = res.bytes_stream();
            let mut tokens = 0usize;
            let mut buffer = String::new();

            while let Some(chunk_res) = stream.next().await {
                if let Ok(chunk) = chunk_res {
                    buffer.push_str(&String::from_utf8_lossy(&chunk));
                    while let Some(pos) = buffer.find('\n') {
                        let line = buffer[..pos].trim().to_string();
                        buffer.drain(..=pos);
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data == "[DONE]" {
                                break;
                            }
                            if let Ok(val) = serde_json::from_str::<Value>(data) {
                                if let Some(choices) = val.get("choices").and_then(|c| c.as_array())
                                {
                                    if let Some(first) = choices.first() {
                                        if let Some(text) =
                                            first.get("text").and_then(|t| t.as_str())
                                        {
                                            if !text.is_empty() {
                                                tokens += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let duration = req_start.elapsed();
            let duration_ms = duration.as_secs_f64() * 1000.0;
            let tok_per_sec = if duration.as_secs_f64() > 0.0 {
                tokens as f64 / duration.as_secs_f64()
            } else {
                0.0
            };

            println!(
                "✓ [Req #{:02}] FINISHED in {:>6.1} ms | Gen: {:>2}/{} toks | {:>5.1} tok/s",
                i + 1,
                duration_ms,
                tokens,
                max_tokens,
                tok_per_sec
            );

            summaries.lock().await.push(RequestSummary {
                id: i + 1,
                requested_tokens: max_tokens,
                generated_tokens: tokens,
                duration_ms,
                tok_per_sec,
            });
        });

        handles.push(handle);
    }

    println!(
        "All {} requests dispatched concurrently...\n",
        args.concurrency
    );
    for h in handles {
        let _ = h.await;
    }

    let total_elapsed = global_start.elapsed();
    let results = summaries.lock().await;
    let total_tokens: usize = results.iter().map(|r| r.generated_tokens).sum();
    let aggregate_throughput = total_tokens as f64 / total_elapsed.as_secs_f64();

    println!("\n===============================================================");
    println!(" 📊 Continuous Batching Execution Summary:");
    println!(
        " Concurrency Level:       {} parallel streams",
        args.concurrency
    );
    println!(" Total Tokens Generated:  {} tokens", total_tokens);
    println!(
        " Total Elapsed Wall Time: {:.2} s",
        total_elapsed.as_secs_f64()
    );
    println!(
        " Aggregate Throughput:    {:.1} tokens/sec 🚀",
        aggregate_throughput
    );
    println!("---------------------------------------------------------------");
    println!(" Req ID | Target Tokens | Generated | Duration (ms) | Speed (tok/s)");
    println!("---------------------------------------------------------------");
    for r in results.iter() {
        println!(
            " #{:02}   | {:>13} | {:>9} | {:>13.1} | {:>12.1}",
            r.id, r.requested_tokens, r.generated_tokens, r.duration_ms, r.tok_per_sec
        );
    }
    println!("===============================================================");
    println!("Notice how short requests finished first and immediately yielded");
    println!("their batch slots, demonstrating zero head-of-line blocking!");

    Ok(())
}
