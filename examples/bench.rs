//! Repeatable HTTP completion benchmark with strict incremental SSE parsing.
use clap::{Parser, ValueEnum};
use futures_util::StreamExt;
use rlse::sse::CompletionStream;
use serde_json::json;
use std::time::{Duration, Instant};

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

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PrefixMode {
    Warm,
    Cold,
}

#[derive(Parser)]
#[command(
    name = "rlse-bench",
    about = "Repeated HTTP benchmark; client timings include network and scheduling"
)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8002")]
    url: String,
    #[arg(long, default_value = "mock")]
    model: String,
    #[arg(long, default_value = PROMPT_MEDIUM)]
    prompt: String,
    #[arg(long, default_value_t = 32)]
    max_tokens: usize,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    #[arg(long, default_value_t = 1)]
    warmups: usize,
    #[arg(long, default_value_t = 1)]
    concurrency: usize,
    #[arg(long, value_enum, default_value_t = PrefixMode::Warm)]
    prefix_mode: PrefixMode,
    #[arg(long, env = "RLSE_API_KEY", hide_env_values = true)]
    api_key: Option<String>,
}

struct StreamResult {
    tokens: usize,
    ttft_ms: f64,
    latency_ms: f64,
    end_to_end_tokens_per_sec: f64,
    observed_decode_tokens_per_sec: Option<f64>,
    authoritative_usage: bool,
}

async fn measure(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    prompt: &str,
    max_tokens: usize,
) -> Result<StreamResult, String> {
    let started = Instant::now();
    let response = client.post(format!("{url}/v1/completions"))
        .json(&json!({"model":model,"prompt":prompt,"max_tokens":max_tokens,"temperature":0,"top_p":1,"stream":true}))
        .send().await.map_err(|e|e.to_string())?.error_for_status().map_err(|e|e.to_string())?;
    if !response
        .headers()
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .is_some_and(|h| h.starts_with("text/event-stream"))
    {
        return Err("Expected text/event-stream".into());
    }
    let mut stream = response.bytes_stream();
    let mut parser = CompletionStream::default();
    let mut first = None;
    let mut last = None;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| e.to_string())?;
        let previous = parser.token_events;
        parser.push(&chunk)?;
        if parser.token_events > previous {
            let now = started.elapsed();
            first.get_or_insert(now);
            last = Some(now);
        }
    }
    let tokens = parser.finish()?;
    let elapsed = started.elapsed();
    let ttft = first.ok_or("No generated-token event in completed stream")?;
    let decode_duration = last.unwrap().saturating_sub(ttft).as_secs_f64();
    Ok(StreamResult {
        tokens,
        ttft_ms: ttft.as_secs_f64() * 1000.,
        latency_ms: elapsed.as_secs_f64() * 1000.,
        end_to_end_tokens_per_sec: tokens as f64 / elapsed.as_secs_f64(),
        observed_decode_tokens_per_sec: if tokens > 1 && decode_duration > 0. {
            Some((tokens - 1) as f64 / decode_duration)
        } else {
            None
        },
        authoritative_usage: parser.completion_tokens.is_some(),
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.rounds == 0 || args.concurrency == 0 || args.max_tokens == 0 {
        return Err("rounds, concurrency, and max-tokens must be positive".into());
    }
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(key) = args.api_key {
        let mut value = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
            .map_err(|_| "Invalid API credential header")?;
        value.set_sensitive(true);
        headers.insert(reqwest::header::AUTHORIZATION, value);
    }
    let client = reqwest::Client::builder()
        .default_headers(headers)
        .timeout(Duration::from_secs(120))
        .build()?;
    println!("model={} prefix_mode={:?} warmups={} rounds={} concurrency={} max_tokens={} temperature=0 top_p=1",args.model,args.prefix_mode,args.warmups,args.rounds,args.concurrency,args.max_tokens);
    println!("Timings are client observations including network, queueing, and prefill; decode rate uses arrivals after the first token. Cold mode adds a unique leading nonce per request to prevent prefix reuse.");
    for _ in 0..args.warmups {
        measure(
            &client,
            &args.url,
            &args.model,
            &args.prompt,
            args.max_tokens,
        )
        .await?;
    }
    for round in 0..args.rounds {
        let before: serde_json::Value = client
            .get(format!("{}/api/status", args.url))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let started = Instant::now();
        let results = futures_util::future::join_all((0..args.concurrency).map(|_| {
            let prompt = match args.prefix_mode {
                PrefixMode::Warm => args.prompt.clone(),
                PrefixMode::Cold => format!("{}\n{}", uuid::Uuid::new_v4(), args.prompt),
            };
            let client = &client;
            let url = &args.url;
            let model = &args.model;
            async move { measure(client, url, model, &prompt, args.max_tokens).await }
        }))
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        let elapsed = started.elapsed().as_secs_f64();
        let total: usize = results.iter().map(|r| r.tokens).sum();
        let after: serde_json::Value = client
            .get(format!("{}/api/status", args.url))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let delta = |name: &str| -> Result<u64, String> {
            let a = after["prefix_cache"][name]
                .as_u64()
                .ok_or("Missing prefix metric")?;
            let b = before["prefix_cache"][name]
                .as_u64()
                .ok_or("Missing prefix metric")?;
            a.checked_sub(b)
                .ok_or_else(|| "Server counters reset during measurement".into())
        };
        println!(
            "  server_prefix_hits={} server_prefix_misses={} server_prefix_tokens_saved={}",
            delta("hits")?,
            delta("misses")?,
            delta("tokens_saved")?
        );
        println!(
            "round={} aggregate_end_to_end_tokens_per_sec={:.2} tokens={} elapsed_ms={:.2}",
            round + 1,
            total as f64 / elapsed,
            total,
            elapsed * 1000.
        );
        for (i, result) in results.iter().enumerate() {
            println!("  request={} tokens={} count_source={} ttft_ms={:.2} latency_ms={:.2} end_to_end_tokens_per_sec={:.2} observed_decode_tokens_per_sec={}",i+1,result.tokens,if result.authoritative_usage {"server_usage"}else{"token_events_estimate"},result.ttft_ms,result.latency_ms,result.end_to_end_tokens_per_sec,result.observed_decode_tokens_per_sec.map_or_else(||"unmeasurable".into(),|n|format!("{n:.2}")));
        }
    }
    Ok(())
}
