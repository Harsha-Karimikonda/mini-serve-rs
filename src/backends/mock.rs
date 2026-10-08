use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

use crate::core::errors::EngineError;
use crate::scheduler::sequence::ActiveSequence;

pub struct StepToken {
    pub token_id: u32,
    pub text: String,
    pub is_eos: bool,
}

#[async_trait]
pub trait ModelBackend: Send + Sync {
    fn name(&self) -> &str;
    async fn init_sequence(&self, prompt: &str) -> Result<Vec<u32>, EngineError>;
    async fn step_batch(
        &self,
        sequences: &mut [ActiveSequence],
    ) -> Result<Vec<StepToken>, EngineError>;
}

pub struct MockBackend {
    name: String,
    step_latency: Duration,
    vocabulary: Vec<&'static str>,
}

impl MockBackend {
    pub fn new(name: impl Into<String>, step_latency_ms: u64) -> Self {
        let vocabulary = vec![
            "Continuous",
            "batching",
            "in",
            "Rust",
            "delivers",
            "deterministic",
            "low-latency",
            "token",
            "generation",
            "without",
            "Python",
            "GIL",
            "contention.",
            "Paged",
            "KV-cache",
            "allocates",
            "fixed-size",
            "memory",
            "blocks",
            "dynamically,",
            "while",
            "client-disconnect",
            "cancellation",
            "instantly",
            "reclaims",
            "GPU",
            "resources.",
            "High",
            "throughput",
            "is",
            "sustained",
            "across",
            "concurrent",
            "inference",
            "streams",
            "safely.",
        ];

        Self {
            name: name.into(),
            step_latency: Duration::from_millis(step_latency_ms),
            vocabulary,
        }
    }
}

#[async_trait]
impl ModelBackend for MockBackend {
    fn name(&self) -> &str {
        &self.name
    }

    async fn init_sequence(&self, prompt: &str) -> Result<Vec<u32>, EngineError> {
        // Approximate token count by whitespace splitting
        let words: Vec<&str> = prompt.split_whitespace().collect();
        let count = words.len().max(1);
        let tokens: Vec<u32> = (0..count as u32).collect();
        Ok(tokens)
    }

    async fn step_batch(
        &self,
        sequences: &mut [ActiveSequence],
    ) -> Result<Vec<StepToken>, EngineError> {
        if !self.step_latency.is_zero() {
            sleep(self.step_latency).await;
        }

        let mut results = Vec::with_capacity(sequences.len());

        for seq in sequences.iter_mut() {
            let next_idx = seq.output_tokens.len();
            let is_eos = next_idx + 1 >= seq.sampling.max_tokens;

            let word = self.vocabulary[next_idx % self.vocabulary.len()];
            let token_text = if next_idx == 0 {
                word.to_string()
            } else {
                format!(" {}", word)
            };

            let token_id = (next_idx % self.vocabulary.len()) as u32 + 100;

            results.push(StepToken {
                token_id,
                text: token_text,
                is_eos,
            });
        }

        Ok(results)
    }
}

pub type SharedBackend = Arc<dyn ModelBackend>;
