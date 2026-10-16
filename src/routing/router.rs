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
            .filter(|worker| worker.scheduler.is_healthy())
            .min_by(|a, b| a.score().total_cmp(&b.score()))
            .ok_or_else(|| EngineError::NoHealthyWorkers("No ready execution paths".into()))
    }

    pub async fn submit(
        &self,
        request_id: RequestId,
        prompt: String,
        sampling: SamplingParams,
    ) -> Result<mpsc::Receiver<TokenEvent>, EngineError> {
        let worker = self.choose_worker()?;
        worker.scheduler.submit(request_id, prompt, sampling).await
    }

    pub fn all_metrics(&self) -> Vec<(String, SchedulerMetrics)> {
        self.workers
            .iter()
            .map(|w| (w.id.clone(), w.scheduler.metrics()))
            .collect()
    }
}

pub type SharedRouter = Arc<Router>;
