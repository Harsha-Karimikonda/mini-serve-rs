use std::sync::Arc;
use tokio::sync::mpsc;

use crate::core::errors::EngineError;
use crate::core::types::{RequestId, SamplingParams};
use crate::scheduler::scheduler::{SchedulerMetrics, SharedScheduler};
use crate::scheduler::sequence::TokenEvent;

#[derive(Clone)]
pub struct WorkerHandle {
    pub id: String,
    pub scheduler: SharedScheduler,
}

impl WorkerHandle {
    pub fn new(id: impl Into<String>, scheduler: SharedScheduler) -> Self {
        Self {
            id: id.into(),
            scheduler,
        }
    }

    pub fn score(&self) -> f64 {
        let metrics = self.scheduler.metrics();
        let queue = metrics.queue_depth as f64;
        let active = metrics.active_count as f64;
        let latency = metrics.avg_latency_ms.max(1.0); // minimum 1ms

        (queue + active + 1.0) * latency
    }
}

pub struct Router {
    workers: Vec<WorkerHandle>,
}

impl Router {
    pub fn new(workers: Vec<WorkerHandle>) -> Self {
        Self { workers }
    }

    pub fn workers(&self) -> &[WorkerHandle] {
        &self.workers
    }

    pub fn choose_worker(&self) -> Result<&WorkerHandle, EngineError> {
        self.workers
            .iter()
            .filter(|worker| worker.scheduler.has_capacity())
            .min_by(|a, b| a.score().total_cmp(&b.score()))
            .ok_or_else(|| EngineError::NoHealthyWorkers("No ready execution paths".into()))
    }

    pub async fn submit(
        &self,
        request_id: RequestId,
        prompt: String,
        sampling: SamplingParams,
    ) -> Result<mpsc::Receiver<TokenEvent>, EngineError> {
        self.submit_candidates(None, request_id, prompt, sampling, None)
            .await
    }

    /// Loaded identities remain discoverable even when execution is quarantined.
    pub fn model_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self
            .workers
            .iter()
            .map(|w| w.scheduler.backend().name().to_owned())
            .collect();
        names.sort();
        names.dedup();
        names
    }

    fn worker_for_model(&self, model: &str) -> Result<&WorkerHandle, EngineError> {
        if !self
            .workers
            .iter()
            .any(|w| w.scheduler.backend().name() == model)
        {
            return Err(EngineError::InvalidRequest(format!(
                "Unknown model '{model}'"
            )));
        }
        self.workers
            .iter()
            .filter(|w| w.scheduler.backend().name() == model && w.scheduler.is_healthy())
            .min_by(|a, b| a.score().total_cmp(&b.score()))
            .ok_or_else(|| {
                EngineError::NoHealthyWorkers("No ready execution paths for model".into())
            })
    }

    pub async fn submit_model(
        &self,
        model: &str,
        request_id: RequestId,
        prompt: String,
        sampling: SamplingParams,
    ) -> Result<mpsc::Receiver<TokenEvent>, EngineError> {
        self.submit_model_with_permit(model, request_id, prompt, sampling, None)
            .await
    }

    pub async fn submit_model_with_permit(
        &self,
        model: &str,
        request_id: RequestId,
        prompt: String,
        sampling: SamplingParams,
        permit: Option<Arc<crate::core::client_limits::ClientPermit>>,
    ) -> Result<mpsc::Receiver<TokenEvent>, EngineError> {
        self.worker_for_model(model)?;
        self.submit_candidates(Some(model), request_id, prompt, sampling, permit)
            .await
    }

    async fn submit_candidates(
        &self,
        model: Option<&str>,
        request_id: RequestId,
        prompt: String,
        sampling: SamplingParams,
        permit: Option<Arc<crate::core::client_limits::ClientPermit>>,
    ) -> Result<mpsc::Receiver<TokenEvent>, EngineError> {
        let mut workers: Vec<_> = self
            .workers
            .iter()
            .filter(|w| model.is_none_or(|m| w.scheduler.backend().name() == m))
            .filter(|w| w.scheduler.is_healthy())
            .collect();
        workers.sort_by(|a, b| a.score().total_cmp(&b.score()));
        let mut error = EngineError::NoHealthyWorkers("No ready execution paths".into());
        for worker in workers {
            match worker
                .scheduler
                .submit_with_permit(
                    request_id.clone(),
                    prompt.clone(),
                    sampling.clone(),
                    permit.clone(),
                )
                .await
            {
                Ok(rx) => return Ok(rx),
                Err(err @ (EngineError::AdmissionError(_) | EngineError::NoHealthyWorkers(_))) => {
                    error = err
                }
                Err(err) => return Err(err),
            }
        }
        Err(error)
    }

    pub async fn submit_chat(
        &self,
        model: &str,
        request_id: RequestId,
        messages: &[crate::core::types::ChatMessage],
        sampling: SamplingParams,
    ) -> Result<mpsc::Receiver<TokenEvent>, EngineError> {
        self.submit_chat_with_permit(model, request_id, messages, sampling, None)
            .await
    }

    pub async fn submit_chat_with_permit(
        &self,
        model: &str,
        request_id: RequestId,
        messages: &[crate::core::types::ChatMessage],
        sampling: SamplingParams,
        permit: Option<Arc<crate::core::client_limits::ClientPermit>>,
    ) -> Result<mpsc::Receiver<TokenEvent>, EngineError> {
        crate::backends::chat::validate_messages(messages)?;
        let worker = self.worker_for_model(model)?;
        let prompt = worker.scheduler.backend().format_chat(messages)?;
        self.submit_candidates(Some(model), request_id, prompt, sampling, permit)
            .await
    }

    pub fn execution_stats(&self) -> crate::telemetry::metrics::ExecutionStats {
        let mut stats = crate::telemetry::metrics::ExecutionStats::default();
        for worker in &self.workers {
            stats += worker.scheduler.metrics().execution;
        }
        stats
    }

    pub fn tokens_per_second(&self) -> f64 {
        self.workers
            .iter()
            .map(|w| w.scheduler.tokens_per_second())
            .sum()
    }

    pub fn all_metrics(&self) -> Vec<(String, SchedulerMetrics)> {
        self.workers
            .iter()
            .map(|w| (w.id.clone(), w.scheduler.metrics()))
            .collect()
    }
}

pub type SharedRouter = Arc<Router>;
