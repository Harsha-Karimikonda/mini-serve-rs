use crate::core::errors::EngineError;
use crate::core::types::{ActiveSequence, ChatMessage, RequestId};
use async_trait::async_trait;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct StepToken {
    pub token_id: u32,
    /// Raw decoded fragment. Special EOS markers must produce empty text.
    /// Shared scheduling applies user-defined stops before delivery.
    pub text: String,
    pub is_eos: bool,
}

#[async_trait]
pub trait ModelBackend: Send + Sync {
    fn name(&self) -> &str;
    /// Render the loaded model's chat template before tokenization and caching.
    fn format_chat(&self, _messages: &[ChatMessage]) -> Result<String, EngineError> {
        Err(EngineError::InvalidRequest(
            "Chat template unavailable for this backend; use /v1/completions".into(),
        ))
    }
    /// Shared execution health. Stateful backends must publish quarantine to all workers.
    fn is_healthy(&self) -> bool {
        true
    }
    /// Permanently quarantine execution after a returned backend failure. No automatic retry.
    fn mark_unhealthy(&self) {}
    /// Maximum supported prompt plus output length, when the model has a context limit.
    fn context_length(&self) -> Option<usize> {
        None
    }
    /// Tokenize only: admission calls this before allocating request execution state.
    async fn init_sequence(&self, prompt: &str) -> Result<Vec<u32>, EngineError>;
    /// Reserve execution resources before logical allocation and the first forward.
    fn reserve_sequence(
        &self,
        _request_id: &RequestId,
        _prompt_tokens: usize,
        _max_tokens: usize,
    ) -> Result<(), EngineError> {
        Ok(())
    }
    async fn step_batch(
        &self,
        sequences: &mut [ActiveSequence],
    ) -> Result<Vec<StepToken>, EngineError>;
    fn cancel_sequence(&self, _request_id: &RequestId) {}
}

pub type SharedBackend = Arc<dyn ModelBackend>;
