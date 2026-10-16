//! Prefix Cache & RAG Acceleration Example for Mini-Serve
//!
//! Demonstrates how Mini-Serve's 16-token chunk-hashed prefix cache
//! dramatically accelerates Retrieval-Augmented Generation (RAG) by bypassing
//! the prefill forward pass for recurring document context.
//!
//! # How to run:
//! ```bash
//! cargo run --example prefix_cache_rag
//! cargo run --example prefix_cache_rag -- --url http://127.0.0.1:8000 --model Qwen/Qwen2.5-0.5B-Instruct
//! ```

use clap::Parser;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::time::Instant;

const SYSTEM_DOCUMENT_PREFIX: &str = "\
[DOCUMENT CONTEXT: Mini-Serve High-Throughput Inference Engine]
Mini-Serve is a high-throughput, fault-tolerant Large Language Model serving engine \
and continuous batching control plane written in pure Rust. It delivers deterministic, \
low-latency token streaming with zero Python GIL contention, zero model weight duplication, \
and a lightweight native binary footprint.

1. Multi-Threading Architecture:
Mini-Serve serves requests across lightweight worker threads sharing model weights in memory \
via Arc<ModelWeights> without requiring out-of-process subprocesses or IPC serialization overhead.

2. Grouped-Query Attention (GQA):
Per-KV-head 2D grouped matmul eliminates full-cache cloning (repeat_kv), maintaining high decode \
speeds with zero context degradation across long contexts.

3. Iteration-Level Continuous Batching:
Dynamic per-step admission admits waiting requests into active batches on every forward pass. \
Sequences emitting EOS or reaching token limits are evicted instantly with zero batch starvation. \
Client disconnect cooperative cancellation reclaims physical KV blocks in real time.

4. Paged KV-Cache & 16-Token Chained Prefix Caching:
Logical and physical block management (16 tokens/block) dynamically allocates KV memory. \
Chunks of 16 tokens are hashed using chained SHA-256 digests. If a prefix chunk matches a previous \
request, prefill computation is bypassed and activations are directly restored.
[END OF CONTEXT]
";

#[derive(Parser, Debug)]
#[command(name = "prefix-cache-rag", about = "RAG prefix caching demonstration")]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    url: String,

    #[arg(long, default_value = "mock")]
    model: String,

    #[arg(long, default_value_t = 30)]
    max_tokens: usize,
}

struct QueryResult {
    ttft_ms: f64,
    total_ms: f64,
    token_count: usize,
    output_text: String,
}

async fn run_query(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    prompt: &str,
    max_tokens: usize,
) -> Result<QueryResult, Box<dyn std::error::Error>> {
    let payload = json!({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": true
    });

    let start = Instant::now();
    let mut first_token_time: Option<Instant> = None;
    let mut token_count = 0usize;
    let mut output_text = String::new();

    let res = client
        .post(format!("{}/v1/completions", url))
        .json(&payload)
        .send()
        .await?;

    if !res.status().is_success() {
        let err = res.text().await?;
        return Err(format!("Server returned error: {}", err).into());
    }

    let mut stream = res.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk_res) = stream.next().await {
        let chunk = chunk_res?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].trim().to_string();
            buffer.drain(..=pos);

            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    break;
                }
                if let Ok(val) = serde_json::from_str::<Value>(data) {
                    if let Some(choices) = val.get("choices").and_then(|c| c.as_array()) {
                        if let Some(first) = choices.first() {
                            if let Some(text) = first.get("text").and_then(|t| t.as_str()) {
                                if !text.is_empty() {
                                    if first_token_time.is_none() {
                                        first_token_time = Some(Instant::now());
                                    }
                                    token_count += 1;
                                    output_text.push_str(text);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = first_token_time
        .map(|t| (t - start).as_secs_f64() * 1000.0)
        .unwrap_or(total_ms);

    Ok(QueryResult {
        ttft_ms,
        total_ms,
        token_count,
        output_text,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let client = reqwest::Client::new();

    println!("===============================================================");
    println!(" ⚡ Mini-Serve Prefix Cache & RAG Acceleration Demo");
    println!(" Target URL: {}", args.url);
    println!(" Model:      {}", args.model);
    println!("===============================================================\n");

    println!(
        "Context document size: ~{} characters",
        SYSTEM_DOCUMENT_PREFIX.len()
    );
    println!("Scenario: Two distinct user queries sharing the same document context.\n");

    // 1. Cold Request
    let prompt_cold = format!(
        "{}\nQuestion: What is continuous batching?",
        SYSTEM_DOCUMENT_PREFIX
    );
    println!("--- [Step 1] Cold Query (Prefix Cache MISS) ---");
    println!("Sending Query 1: 'What is continuous batching?'");
    let cold_result = run_query(
        &client,
        &args.url,
        &args.model,
        &prompt_cold,
        args.max_tokens,
    )
    .await?;
    println!("Generated text: \"{}\"", cold_result.output_text.trim());
    println!(
        ">> Cold TTFT (Time to First Token): {:.2} ms ({} tokens generated)",
        cold_result.ttft_ms, cold_result.token_count
    );
    println!(
        ">> Cold Total Time:                {:.2} ms\n",
        cold_result.total_ms
    );

    // 2. Warm Request
    let prompt_warm = format!(
        "{}\nQuestion: How does the paged KV cache avoid memory fragmentation?",
        SYSTEM_DOCUMENT_PREFIX
    );
    println!("--- [Step 2] Warm Query (Prefix Cache HIT) ---");
    println!("Sending Query 2 with identical document context: 'How does the paged KV cache avoid memory fragmentation?'");
    let warm_result = run_query(
        &client,
        &args.url,
        &args.model,
        &prompt_warm,
        args.max_tokens,
    )
    .await?;
    println!("Generated text: \"{}\"", warm_result.output_text.trim());
    println!(
        ">> Warm TTFT (Time to First Token): {:.2} ms ({} tokens generated)",
        warm_result.ttft_ms, warm_result.token_count
    );
    println!(
        ">> Warm Total Time:                {:.2} ms\n",
        warm_result.total_ms
    );

    // 3. Comparison
    println!("===============================================================");
    println!(" 📊 Performance Analysis:");
    println!(
        " Cold TTFT (Prefill required):  {:.2} ms",
        cold_result.ttft_ms
    );
    println!(
        " Warm TTFT (Prefill bypassed):  {:.2} ms",
        warm_result.ttft_ms
    );
    if warm_result.ttft_ms < cold_result.ttft_ms {
        let speedup = cold_result.ttft_ms / warm_result.ttft_ms.max(0.1);
        println!(" TTFT Latency Reduction:        {:.1}x FASTER! 🚀", speedup);
    }
    println!("===============================================================");
    println!("Note: With real weights (Qwen), prefill forward passes for the");
    println!("cached prefix are skipped, reducing TTFT from O(prompt_len) to O(1).");

    Ok(())
}
