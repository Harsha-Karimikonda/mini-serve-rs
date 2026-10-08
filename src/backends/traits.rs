use crate::core::errors::EngineError;
use crate::core::types::ActiveSequence;
use async_trait::async_trait;
use std::sync::Arc;

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

pub type SharedBackend = Arc<dyn ModelBackend>;
