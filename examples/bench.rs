//! Pure Rust Benchmark Harness for Mini-Serve
//!
//! Run with:
//! cargo run --example bench --features metal -- [OPTIONS]
//!
//! Example:
//! cargo run --example bench --features metal -- --url http://127.0.0.1:8002 --model Qwen/Qwen2.5-0.5B-Instruct

use clap::Parser;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::time::Instant;

const PROMPT_SHORT: &str = "Explain the theory of relativity in one short sentence.";

const PROMPT_MEDIUM: &str = "\
In computer systems and distributed architectures, memory consistency models define \
the rules under which memory operations (reads and writes) appear to occur from the perspective \
of concurrent threads or nodes. Strict consistency requires that any read receives the most recent \
write, but this imposes severe synchronization costs across hardware cores or networks. Sequential \
consistency, formalized by Leslie Lamport in 1979, guarantees that the execution produces the same \
results as if all memory accesses were executed in some sequential order, and the operations of each \
individual processor appear in this sequence in the order specified by its program. Modern out-of-order \
processors such as Apple Silicon (ARMv8.5-A) and x86 implement relaxed memory models, where store buffers, \
speculative execution, and memory fences like DMB (Data Memory Barrier) dictate visibility. Please summarize \
how modern hardware memory models trade off sequential consistency for instruction-level parallelism.";

const PROMPT_LONG: &str = "\
A continuous batching inference engine is designed to maximize GPU utilization when serving Large Language Models. \
Traditional batching models wait for a fixed number of requests to arrive, pack them into a rectangular matrix tensor, \
and execute forward passes until all sequences finish. This introduces substantial inefficiencies known as the 'head-of-line blocking' \
and 'tail latency penalty', where shorter sequences sit idle while the longest sequence in the batch completes its generation. \
Continuous batching (or iteration-level scheduling), pioneered by Orca and popularized by vLLM, transforms inference into a dynamic, \
step-by-step iteration loop. At every decoding iteration, completed sequences that encounter an EOS token or reach their max_tokens \
limit are immediately evicted, free KV-cache slots are reclaimed, and newly arrived waiting requests in the scheduling queue are admitted \
dynamically without waiting for other streams to finish. \
Paged KV-cache architectures further solve GPU memory fragmentation. Standard KV-caching pre-allocates contiguous virtual memory buffers \
sized to the maximum possible sequence length (e.g. 2048 or 4096 tokens). In real-world workloads, actual output lengths exhibit high variance, \
leading to internal fragmentation where 60% to 80% of VRAM is reserved for tokens that are never generated. By segmenting the Key and Value cache \
into non-contiguous, fixed-size physical blocks (e.g. 16 tokens per block), the engine dynamically maps logical tokens to physical block IDs through \
a page table, mirroring OS virtual memory management. \
Prefix caching builds on this abstraction: when multiple prompts share common prefixes (such as system instructions, few-shot examples, \
or document contexts in Retrieval-Augmented Generation), the engine hashes 16-token chunks using cryptographic hashing (such as SHA-256) \
and stores the computed Key-Value activations in an LRU prefix cache. Subsequent requests sharing that prefix can bypass prefill computation \
entirely, jumping directly to decoding. \
Furthermore, in production systems, cooperative client disconnect cancellation is vital: if an HTTP client drops its TCP connection, \
the engine must immediately cease forward computation for that sequence, reclaim GPU blocks, and avoid wasting precious FLOPs. \
Please analyze the architectural synergy between continuous batching, paged memory allocation, and zero-overhead systems programming.";

#[derive(Parser, Debug)]
#[command(name = "mini-serve-bench", about = "Pure Rust benchmark harness")]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8002")]
    url: String,

    #[arg(long, default_value = "Qwen/Qwen2.5-0.5B-Instruct")]
    model: String,
}

struct StreamResult {
    total_tokens: usize,
    ttft_ms: f64,
    total_latency_ms: f64,
    tok_per_sec: f64,
    sample_text: String,
}

async fn measure_pure_ttft(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    prompt: &str,
) -> f64 {
    let t0 = Instant::now();
    let payload = json!({
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "stream": false
    });

    let res = client
        .post(format!("{}/v1/completions", base_url))
        .json(&payload)
        .send()
        .await;

    match res {
        Ok(resp) => {
            let _ = resp.text().await;
            t0.elapsed().as_secs_f64() * 1000.0
        }
        Err(e) => {
            eprintln!("Error measuring TTFT: {}", e);
            0.0
        }
    }
}

async fn measure_streaming(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    prompt: &str,
    max_tokens: usize,
) -> StreamResult {
    let t0 = Instant::now();
    let mut first_token_time = None;
    let mut token_count = 0;
    let mut text_acc = String::new();

    let payload = json!({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": true
    });

    let resp = client
        .post(format!("{}/v1/completions", base_url))
        .json(&payload)
        .send()
        .await;

    if let Ok(response) = resp {
        let mut stream = response.bytes_stream();
        while let Some(chunk_res) = stream.next().await {
            if let Ok(chunk) = chunk_res {
                let text = String::from_utf8_lossy(&chunk);
                for line in text.lines() {
                    let line = line.trim();
                    if !line.starts_with("data:") {
                        continue;
                    }
                    let data = line[5..].trim();
                    if data == "[DONE]" {
                        break;
                    }
                    if let Ok(v) = serde_json::from_str::<Value>(data) {
                        if let Some(txt) = v["choices"][0]["text"].as_str() {
                            if first_token_time.is_none() {
                                first_token_time = Some(t0.elapsed());
                            }
                            token_count += 1;
                            text_acc.push_str(txt);
                        }
                    }
                }
            }
        }
    }

    let total_latency = t0.elapsed();
    let total_latency_ms = total_latency.as_secs_f64() * 1000.0;
    let ttft_ms = first_token_time.map_or(total_latency_ms, |d| d.as_secs_f64() * 1000.0);
    let tok_per_sec = if total_latency.as_secs_f64() > 0.0 {
        token_count as f64 / total_latency.as_secs_f64()
    } else {
        0.0
    };

    StreamResult {
        total_tokens: token_count,
        ttft_ms,
        total_latency_ms,
        tok_per_sec,
        sample_text: text_acc,
    }
}

async fn measure_concurrency(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    prompt: &str,
    concurrency: usize,
    max_tokens: usize,
) -> (f64, f64, usize) {
    let t0 = Instant::now();
    let mut handles = Vec::with_capacity(concurrency);

    for _ in 0..concurrency {
        let cl = client.clone();
        let url = base_url.to_string();
        let mdl = model.to_string();
        let pr = prompt.to_string();
        handles.push(tokio::spawn(async move {
            measure_streaming(&cl, &url, &mdl, &pr, max_tokens).await
        }));
    }

    let mut total_tokens = 0;
    let mut sum_ttft = 0.0;
    for h in handles {
        if let Ok(res) = h.await {
            total_tokens += res.total_tokens;
            sum_ttft += res.ttft_ms;
        }
    }

    let elapsed_s = t0.elapsed().as_secs_f64();
    let agg_tok_per_sec = if elapsed_s > 0.0 {
        total_tokens as f64 / elapsed_s
    } else {
        0.0
    };
    let avg_ttft = sum_ttft / concurrency as f64;

    (agg_tok_per_sec, avg_ttft, total_tokens)
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .expect("Failed to build client");

    println!("========================================================");
    println!("PURE RUST BENCHMARK HARNESS (Mini-Serve)");
    println!("Target URL : {}", args.url);
    println!("Model ID   : {}", args.model);
    println!("========================================================");

    // Warmup
    print!("[*] Warming up server... ");
    let _ = measure_streaming(&client, &args.url, &args.model, "Hello!", 5).await;
    println!("done.");

    // 1. Short prompt
    println!("\n[1] Short Prompt (~10 tokens prefill, 25 tokens max decode):");
    let ttft_short = measure_pure_ttft(&client, &args.url, &args.model, PROMPT_SHORT).await;
    let res_short = measure_streaming(&client, &args.url, &args.model, PROMPT_SHORT, 25).await;
    println!("    - Pure Prefill TTFT : {:.2} ms", ttft_short);
    println!(
        "    - End-to-End Latency: {:.2} ms",
        res_short.total_latency_ms
    );
    println!(
        "    - Decode Speed      : {:.1} tok/s",
        res_short.tok_per_sec
    );
    println!("    - Tokens Produced   : {}", res_short.total_tokens);
    let sample: String = res_short.sample_text.chars().take(60).collect();
    println!("    - Sample Output     : \"{}...\"", sample.trim());

    // 2. Medium prompt
    println!("\n[2] Medium Prompt (~200 tokens prefill, 50 tokens max decode):");
    let ttft_med = measure_pure_ttft(&client, &args.url, &args.model, PROMPT_MEDIUM).await;
    let res_med = measure_streaming(&client, &args.url, &args.model, PROMPT_MEDIUM, 50).await;
    println!("    - Pure Prefill TTFT : {:.2} ms", ttft_med);
    println!(
        "    - End-to-End Latency: {:.2} ms",
        res_med.total_latency_ms
    );
    println!("    - Decode Speed      : {:.1} tok/s", res_med.tok_per_sec);
    println!("    - Tokens Produced   : {}", res_med.total_tokens);

    // 3. Long prompt
    println!("\n[3] Long Prompt (~450 tokens prefill, 50 tokens max decode):");
    let ttft_long = measure_pure_ttft(&client, &args.url, &args.model, PROMPT_LONG).await;
    let res_long = measure_streaming(&client, &args.url, &args.model, PROMPT_LONG, 50).await;
    println!("    - Pure Prefill TTFT : {:.2} ms", ttft_long);
    println!(
        "    - End-to-End Latency: {:.2} ms",
        res_long.total_latency_ms
    );
    println!(
        "    - Decode Speed      : {:.1} tok/s",
        res_long.tok_per_sec
    );
    println!("    - Tokens Produced   : {}", res_long.total_tokens);

    // 4. Concurrent Continuous Batching
    println!("\n[4] Concurrent Continuous Batching (C=4 streams, 30 tokens each):");
    let (c4_tok_s, c4_avg_ttft, c4_tokens) =
        measure_concurrency(&client, &args.url, &args.model, PROMPT_SHORT, 4, 30).await;
    println!("    - Aggregated Throughput: {:.1} tok/s", c4_tok_s);
    println!("    - Average TTFT         : {:.2} ms", c4_avg_ttft);
    println!("    - Total Tokens         : {}", c4_tokens);

    println!("\n========================================================");
    println!("BENCHMARK RUN COMPLETED");
    println!("========================================================");
}
