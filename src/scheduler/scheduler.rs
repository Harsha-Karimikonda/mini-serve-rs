use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Notify};
use tracing::{debug, info, warn};

use crate::backends::{ModelBackend, StepToken};
use crate::cache::{SharedKVCache, SharedPrefixCache};
use crate::core::errors::EngineError;
use crate::core::types::{RequestId, SamplingParams};
use crate::scheduler::sequence::{ActiveSequence, TokenEvent};

struct QueueItem {
    request_id: RequestId,
    prompt: String,
    sampling: SamplingParams,
    tx: mpsc::Sender<TokenEvent>,
    enqueued_at: Instant,
}

#[derive(Debug, Clone)]
pub struct SchedulerMetrics {
    pub queue_depth: usize,
    pub active_count: usize,
    pub avg_latency_ms: f64,
}

pub struct Scheduler {
    max_batch_size: usize,
    max_waiting_requests: usize,
    cache: SharedKVCache,
    prefix_cache: SharedPrefixCache,
    backend: Arc<dyn ModelBackend>,
    queue: Arc<Mutex<VecDeque<QueueItem>>>,
    queue_notify: Arc<Notify>,
    active_count: Arc<AtomicUsize>,
    latencies: Arc<Mutex<VecDeque<Duration>>>,
    shutdown: Arc<Notify>,
}

impl Scheduler {
    pub fn new(
        max_batch_size: usize,
        max_waiting_requests: usize,
        cache: SharedKVCache,
        prefix_cache: SharedPrefixCache,
        backend: Arc<dyn ModelBackend>,
    ) -> Self {
        Self {
            max_batch_size,
            max_waiting_requests,
            cache,
            prefix_cache,
            backend,
            queue: Arc::new(Mutex::new(VecDeque::new())),
            queue_notify: Arc::new(Notify::new()),
            active_count: Arc::new(AtomicUsize::new(0)),
            latencies: Arc::new(Mutex::new(VecDeque::with_capacity(64))),
            shutdown: Arc::new(Notify::new()),
        }
    }

    pub fn queue_depth(&self) -> usize {
        self.queue.lock().len()
    }

    pub fn active_count(&self) -> usize {
        self.active_count.load(Ordering::Relaxed)
    }

    pub fn avg_latency_ms(&self) -> f64 {
        let lock = self.latencies.lock();
        if lock.is_empty() {
            0.0
        } else {
            let total_ms: f64 = lock.iter().map(|d| d.as_secs_f64() * 1000.0).sum();
            (total_ms / lock.len() as f64 * 10.0).round() / 10.0
        }
    }

    pub fn metrics(&self) -> SchedulerMetrics {
        SchedulerMetrics {
            queue_depth: self.queue_depth(),
            active_count: self.active_count(),
            avg_latency_ms: self.avg_latency_ms(),
        }
    }

    pub async fn submit(
        &self,
        request_id: RequestId,
        prompt: String,
        sampling: SamplingParams,
    ) -> Result<mpsc::Receiver<TokenEvent>, EngineError> {
        let mut queue = self.queue.lock();
        if queue.len() >= self.max_waiting_requests {
            return Err(EngineError::AdmissionError(format!(
                "Queue saturated: {} requests pending",
                queue.len()
            )));
        }

        // Fast-fail check against total cache capacity
        let needed_blocks = self.cache.lock().blocks_needed(sampling.max_tokens);
        let total_blocks = self.cache.lock().total_blocks();
        if needed_blocks > total_blocks {
            return Err(EngineError::CachePressure(format!(
                "Requested {} blocks for max_tokens={}, which exceeds total cluster cache size {}",
                needed_blocks, sampling.max_tokens, total_blocks
            )));
        }

        let (tx, rx) = mpsc::channel(64);
        queue.push_back(QueueItem {
            request_id,
            prompt,
            sampling,
            tx,
            enqueued_at: Instant::now(),
        });
        drop(queue);

        self.queue_notify.notify_one();
        Ok(rx)
    }

    pub fn start_loop(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let scheduler = Arc::clone(self);
        tokio::spawn(async move {
            scheduler.run_continuous_loop().await;
        })
    }

    async fn run_continuous_loop(&self) {
        info!(
            "Continuous batching scheduler loop started (max_batch_size={})",
            self.max_batch_size
        );
        let mut active: Vec<ActiveSequence> = Vec::with_capacity(self.max_batch_size);

        loop {
            // Dynamic Admission: Admit waiting requests up to max_batch_size
            while active.len() < self.max_batch_size {
                let item = {
                    let mut q = self.queue.lock();
                    q.pop_front()
                };

                match item {
                    Some(item) => {
                        if item.tx.is_closed() {
                            debug!("Skipping cancelled request {}", item.request_id);
                            continue;
                        }

                        let prompt_tokens = match self.backend.init_sequence(&item.prompt).await {
                            Ok(tokens) => tokens,
                            Err(err) => {
                                let _ = item.tx.send(TokenEvent::Error(err.to_string())).await;
                                continue;
                            }
                        };

                        // Check prefix cache for reuse
                        {
                            let mut pc = self.prefix_cache.lock();
                            let matched = pc.lookup(&prompt_tokens);
                            if matched.matched_tokens > 0 {
                                debug!(
                                    "Prefix cache hit for {}: {} tokens reused",
                                    item.request_id, matched.matched_tokens
                                );
                            }
                        }

                        // Allocate KV cache blocks
                        let total_tokens_est = prompt_tokens.len() + item.sampling.max_tokens;
                        let blocks = {
                            let mut cache = self.cache.lock();
                            cache.allocate(&item.request_id.0, total_tokens_est)
                        };

                        match blocks {
                            Ok(blocks) => {
                                let mut seq = ActiveSequence::new(
                                    item.request_id,
                                    item.prompt,
                                    item.sampling,
                                    item.tx,
                                );
                                seq.created_at = item.enqueued_at;
                                seq.prompt_tokens = prompt_tokens;
                                seq.kv_blocks = blocks;
                                active.push(seq);
                            }
                            Err(err) => {
                                warn!("Cache allocation failed for {}: {}", item.request_id, err);
                                let _ = item.tx.send(TokenEvent::Error(err.to_string())).await;
                            }
                        }
                    }
                    None => break,
                }
            }

            self.active_count.store(active.len(), Ordering::Relaxed);

            // If no sequences active, sleep until a new request arrives
            if active.is_empty() {
                tokio::select! {
                    _ = self.queue_notify.notified() => continue,
                    _ = self.shutdown.notified() => break,
                }
            }

            // Client Disconnect / Cancellation Sweep
            let mut i = 0;
            while i < active.len() {
                if active[i].is_cancelled() {
                    debug!(
                        "Client disconnected mid-stream: cancelling {}",
                        active[i].request_id
                    );
                    let cancelled = active.swap_remove(i);
                    self.cache.lock().release(&cancelled.request_id.0);
                } else {
                    i += 1;
                }
            }
            self.active_count.store(active.len(), Ordering::Relaxed);

            if active.is_empty() {
                continue;
            }

            // Step Batch Forward Pass
            let step_tokens: Vec<StepToken> = match self.backend.step_batch(&mut active).await {
                Ok(toks) => toks,
                Err(err) => {
                    warn!("Backend step_batch error: {}", err);
                    for seq in active.drain(..) {
                        let _ = seq.tx.send(TokenEvent::Error(err.to_string())).await;
                        self.cache.lock().release(&seq.request_id.0);
                    }
                    self.active_count.store(0, Ordering::Relaxed);
                    continue;
                }
            };

            // Demultiplex tokens back to client response streams
            for (seq, step) in active.iter_mut().zip(step_tokens) {
                if !seq.ttft_measured {
                    seq.ttft_measured = true;
                    seq.ttft_duration = Some(seq.created_at.elapsed());
                }

                seq.output_tokens.push(step.token_id);
                seq.output_text.push_str(&step.text);

                // Stream token to client
                let _ = seq.tx.send(TokenEvent::Token(step.text)).await;

                if step.is_eos || seq.output_tokens.len() >= seq.sampling.max_tokens {
                    seq.is_finished = true;
                    seq.finish_reason = if step.is_eos {
                        Some("stop".to_string())
                    } else {
                        Some("length".to_string())
                    };
                }
            }

            // Evict finished sequences and reclaim KV-cache blocks immediately
            let mut j = 0;
            while j < active.len() {
                if active[j].is_finished || active[j].is_cancelled() {
                    let finished = active.swap_remove(j);

                    if finished.is_finished {
                        let usage = finished.usage();
                        let _ = finished.tx.send(TokenEvent::Done(usage)).await;

                        // Insert prompt into prefix cache for future requests
                        self.prefix_cache.lock().insert(&finished.prompt_tokens);

                        // Record latency
                        let duration = finished.created_at.elapsed();
                        let mut lats = self.latencies.lock();
                        if lats.len() >= 64 {
                            lats.pop_front();
                        }
                        lats.push_back(duration);
                    }

                    // Instant KV-cache block recovery!
                    self.cache.lock().release(&finished.request_id.0);
                } else {
                    j += 1;
                }
            }
            self.active_count.store(active.len(), Ordering::Relaxed);
        }

        info!("Continuous batching scheduler loop exited cleanly");
    }

    pub fn shutdown(&self) {
        self.shutdown.notify_one();
    }
}

pub type SharedScheduler = Arc<Scheduler>;
