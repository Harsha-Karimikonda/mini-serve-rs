use crate::core::types::{RequestId, SamplingParams, TokenUsage};
use std::time::Instant;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub enum TokenEvent {
    Token(String),
    Done(TokenUsage),
    Error(String),
}

#[derive(Debug)]
pub struct ActiveSequence {
    pub request_id: RequestId,
    pub prompt: String,
    pub prompt_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub output_text: String,
    pub sampling: SamplingParams,
    pub is_prefilled: bool,
    pub is_finished: bool,
    pub finish_reason: Option<String>,
    pub kv_blocks: Vec<usize>,
    pub created_at: Instant,
    pub ttft_measured: bool,
    pub ttft_duration: Option<std::time::Duration>,
    pub tx: mpsc::Sender<TokenEvent>,
}

impl ActiveSequence {
    pub fn new(
        request_id: RequestId,
        prompt: String,
        sampling: SamplingParams,
        tx: mpsc::Sender<TokenEvent>,
    ) -> Self {
        Self {
            request_id,
            prompt,
            prompt_tokens: Vec::new(),
            output_tokens: Vec::new(),
            output_text: String::new(),
            sampling,
            is_prefilled: false,
            is_finished: false,
            finish_reason: None,
            kv_blocks: Vec::new(),
            created_at: Instant::now(),
            ttft_measured: false,
            ttft_duration: None,
            tx,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.tx.is_closed()
    }

    pub fn usage(&self) -> TokenUsage {
        TokenUsage {
            prompt_tokens: self.prompt_tokens.len(),
            completion_tokens: self.output_tokens.len(),
            total_tokens: self.prompt_tokens.len() + self.output_tokens.len(),
        }
    }
}
