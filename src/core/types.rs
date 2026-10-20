use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(pub String);

impl RequestId {
    pub fn new_completion() -> Self {
        Self(format!("cmpl-{}", Uuid::new_v4()))
    }

    pub fn new_chat() -> Self {
        Self(format!("chatcmpl-{}", Uuid::new_v4()))
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub stop: Vec<String>,
}

fn default_temperature() -> f32 {
    1.0
}
fn default_top_p() -> f32 {
    1.0
}
fn default_max_tokens() -> usize {
    128
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            max_tokens: 128,
            stop: Vec::new(),
        }
    }
}

impl SamplingParams {
    pub fn validate(&self) -> Result<(), crate::core::errors::EngineError> {
        use crate::core::errors::EngineError;
        if self.max_tokens == 0 {
            return Err(EngineError::InvalidRequest(
                "max_tokens must be greater than zero".into(),
            ));
        }
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) {
            return Err(EngineError::InvalidRequest(
                "temperature must be finite and in [0, 2]".into(),
            ));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(EngineError::InvalidRequest(
                "top_p must be finite and in (0, 1]".into(),
            ));
        }
        if self.stop.iter().any(String::is_empty) {
            return Err(EngineError::InvalidRequest(
                "stop strings must not be empty".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: TokenUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChunkChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: TokenUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Clone)]
pub enum TokenEvent {
    Token(String),
    Done {
        usage: TokenUsage,
        finish_reason: String,
    },
    Error(String),
}

#[derive(Debug)]
pub struct ActiveSequence {
    pub request_id: RequestId,
    pub prompt: String,
    pub prompt_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    /// Raw decoded text, including withheld stop prefixes, for incremental decoding.
    pub output_text: String,
    pub stop_matcher: crate::core::stop::StopMatcher,
    pub sampling: SamplingParams,
    pub is_prefilled: bool,
    pub is_finished: bool,
    pub finish_reason: Option<String>,
    pub kv_blocks: Vec<usize>,
    pub created_at: std::time::Instant,
    pub ttft_measured: bool,
    pub ttft_duration: Option<std::time::Duration>,
    pub tx: tokio::sync::mpsc::Sender<TokenEvent>,
    /// Capacity reserved at admission so a terminal event cannot be dropped.
    pub terminal_permit: Option<tokio::sync::mpsc::OwnedPermit<TokenEvent>>,
    pub terminal_error: Option<String>,
    pub trace: Option<crate::telemetry::metrics::RequestTrace>,
    pub client_permit: Option<std::sync::Arc<crate::core::client_limits::ClientPermit>>,
}

impl ActiveSequence {
    pub fn new(
        request_id: RequestId,
        prompt: String,
        sampling: SamplingParams,
        tx: tokio::sync::mpsc::Sender<TokenEvent>,
    ) -> Self {
        let stop_matcher = crate::core::stop::StopMatcher::new(sampling.stop.clone());
        Self {
            request_id,
            prompt,
            prompt_tokens: Vec::new(),
            output_tokens: Vec::new(),
            output_text: String::new(),
            stop_matcher,
            sampling,
            is_prefilled: false,
            is_finished: false,
            finish_reason: None,
            kv_blocks: Vec::new(),
            created_at: std::time::Instant::now(),
            ttft_measured: false,
            ttft_duration: None,
            tx,
            terminal_permit: None,
            terminal_error: None,
            client_permit: None,
            trace: None,
        }
    }

    pub fn send_terminal(&mut self, event: TokenEvent) {
        if let Some(trace) = &mut self.trace {
            use crate::telemetry::metrics::Outcome;
            trace.finish(if self.tx.is_closed() {
                Outcome::Cancelled
            } else {
                match &event {
                    TokenEvent::Done { finish_reason, .. } if finish_reason == "abort" => {
                        Outcome::Cancelled
                    }
                    TokenEvent::Done { .. } => Outcome::Completed,
                    TokenEvent::Error(_) => Outcome::Error,
                    TokenEvent::Token(_) => return,
                }
            });
        }
        if let Some(permit) = self.terminal_permit.take() {
            permit.send(event);
        } else {
            // Manually constructed sequences may not have a reserved slot.
            let _ = self.tx.try_send(event);
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
