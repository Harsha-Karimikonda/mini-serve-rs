//! Interactive Terminal Chat (REPL) using RLSE
//!
//! Demonstrates streaming chat completions (/v1/chat/completions) with live token
//! rendering, time-to-first-token (TTFT) tracking, and multi-turn context.
//!
//! # How to run:
//! ```bash
//! # Against mock backend:
//! cargo run --example cli_chat
//!
//! # Against real model on port 8000:
//! cargo run --example cli_chat -- --url http://127.0.0.1:8000 --model Qwen/Qwen2.5-0.5B-Instruct
//! ```

use clap::Parser;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{self, Write};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "rlse-chat",
    about = "Interactive terminal chat client for RLSE"
)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    url: String,

    #[arg(long, default_value = "mock")]
    model: String,

    #[arg(long, default_value = "You are a helpful, concise AI assistant.")]
    system: String,

    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: usize,
    stream: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let client = reqwest::Client::new();

    println!("=======================================================");
    println!(" ⚡ Mini-Serve Interactive Chat CLI");
    println!(" Connected to: {}", args.url);
    println!(" Model:        {}", args.model);
    println!(" Commands:     Type '/clear' to reset, '/exit' to quit");
    println!("=======================================================\n");

    let mut messages: Vec<ChatMessage> = vec![ChatMessage {
        role: "system".to_string(),
        content: args.system.clone(),
    }];

    let stdin = io::stdin();
    loop {
        print!("You > ");
        io::stdout().flush()?;

        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        if input.eq_ignore_ascii_case("/exit") || input.eq_ignore_ascii_case("/quit") {
            println!("Goodbye!");
            break;
        }

        if input.eq_ignore_ascii_case("/clear") {
            messages.truncate(1);
            println!("\n--- Conversation cleared ---\n");
            continue;
        }

        messages.push(ChatMessage {
            role: "user".to_string(),
            content: input.to_string(),
        });

        let payload = ChatRequest {
            model: args.model.clone(),
            messages: messages.clone(),
            temperature: args.temperature,
            max_tokens: args.max_tokens,
            stream: true,
        };

        print!("Assistant > ");
        io::stdout().flush()?;

        let start_time = Instant::now();
        let mut first_token_time: Option<Instant> = None;
        let mut token_count = 0usize;
        let mut assistant_response = String::new();

        let res = client
            .post(format!("{}/v1/chat/completions", args.url))
            .json(&payload)
            .send()
            .await;

        let response = match res {
            Ok(r) => {
                if !r.status().is_success() {
                    let err_text = r.text().await.unwrap_or_default();
                    eprintln!("\nError from server: {}", err_text);
                    continue;
                }
                r
            }
            Err(e) => {
                eprintln!(
                    "\nFailed to connect to server: {}. Is mini-serve running?",
                    e
                );
                continue;
            }
        };

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk_res) = stream.next().await {
            let chunk = match chunk_res {
                Ok(bytes) => bytes,
                Err(e) => {
                    eprintln!("\nStream error: {}", e);
                    break;
                }
            };

            let text = String::from_utf8_lossy(&chunk);
            buffer.push_str(&text);

            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer.drain(..=pos);

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        break;
                    }

                    if let Ok(val) = serde_json::from_str::<Value>(data) {
                        if let Some(choices) = val.get("choices").and_then(|c| c.as_array()) {
                            if let Some(first) = choices.first() {
                                if let Some(delta) = first.get("delta") {
                                    if let Some(content) =
                                        delta.get("content").and_then(|c| c.as_str())
                                    {
                                        if first_token_time.is_none() {
                                            first_token_time = Some(Instant::now());
                                        }
                                        token_count += 1;
                                        print!("{}", content);
                                        io::stdout().flush()?;
                                        assistant_response.push_str(content);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let total_duration = start_time.elapsed();
        let ttft_ms = first_token_time
            .map(|t| (t - start_time).as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let decode_tok_per_sec = if token_count > 1 && total_duration.as_secs_f64() > 0.0 {
            token_count as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        println!();
        println!(
            "\x1b[90m[Stats: {} tokens | TTFT: {:.1}ms | Speed: {:.1} tok/s]\x1b[0m\n",
            token_count, ttft_ms, decode_tok_per_sec
        );

        messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: assistant_response,
        });
    }

    Ok(())
}
