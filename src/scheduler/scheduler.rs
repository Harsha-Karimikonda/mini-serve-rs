use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
    prompt_tokens: Vec<u32>,
    request_id: RequestId,
    prompt: String,
    sampling: SamplingParams,
    tx: mpsc::Sender<TokenEvent>,
    terminal_permit: mpsc::OwnedPermit<TokenEvent>,
    enqueued_at: Instant,
}

#[derive(Debug, Clone)]
pub struct SchedulerMetrics {
    pub queue_depth: usize,
    pub active_count: usize,
    pub avg_latency_ms: f64,
}

pub struct Scheduler {
    accepting: AtomicBool,
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
            accepting: AtomicBool::new(true),
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

    pub fn is_healthy(&self) -> bool {
        self.accepting.load(Ordering::Acquire) && self.backend.is_healthy()
    }

    fn check_health(&self) -> Result<(), EngineError> {
        if self.is_healthy() {
            Ok(())
        } else {
            Err(EngineError::NoHealthyWorkers(
                "Worker unavailable; restart required after backend failure".into(),
            ))
        }
    }

    fn record_failure(&self, error: &EngineError) {
        if matches!(error, EngineError::BackendError(_)) {
            self.accepting.store(false, Ordering::Release);
            self.backend.mark_unhealthy();
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
        self.check_health()?;
        sampling.validate()?;
        if prompt.trim().is_empty() {
            return Err(EngineError::InvalidRequest("Prompt cannot be empty".into()));
        }
        if self.max_batch_size == 0 || self.max_waiting_requests == 0 {
            return Err(EngineError::InvalidRequest(
                "Scheduler batch and queue sizes must be greater than zero".into(),
            ));
        }
        // Tokenization must not allocate request-scoped execution state. Validate
        // the complete request before queuing or allocating any KV blocks.
        let prompt_tokens = self.backend.init_sequence(&prompt).await?;
        if prompt_tokens.is_empty() {
            return Err(EngineError::InvalidRequest(
                "Prompt must encode at least one token".into(),
            ));
        }
        let total_tokens = prompt_tokens
            .len()
            .checked_add(sampling.max_tokens)
            .ok_or_else(|| {
                EngineError::TokenLimitExceeded("Prompt plus max_tokens overflows usize".into())
            })?;
        if let Some(limit) = self.backend.context_length() {
            if total_tokens > limit {
                return Err(EngineError::TokenLimitExceeded(format!(
                    "Prompt ({} tokens) plus max_tokens={} exceeds model context {limit}",
                    prompt_tokens.len(),
                    sampling.max_tokens
                )));
            }
        }
        {
            let cache = self.cache.lock();
            if cache.block_size() == 0 || cache.total_blocks() == 0 {
                return Err(EngineError::InvalidRequest(
                    "KV block size and capacity must be greater than zero".into(),
                ));
            }
            let capacity = cache
                .total_blocks()
                .checked_mul(cache.block_size())
                .ok_or_else(|| {
                    EngineError::InvalidRequest("KV token capacity overflows usize".into())
                })?;
            if total_tokens > capacity {
                return Err(EngineError::CachePressure(format!(
                    "Prompt plus max_tokens requires {total_tokens} tokens, exceeding logical KV capacity {capacity}"
                )));
            }
        }
        let mut queue = self.queue.lock();
        self.check_health()?;
        if queue.len() >= self.max_waiting_requests {
            return Err(EngineError::AdmissionError(format!(
                "Queue saturated: {} requests pending",
                queue.len()
            )));
        }

        // Keep 64 token slots plus one slot reserved for completion or error.
        let (tx, rx) = mpsc::channel(65);
        let terminal_permit = tx
            .clone()
            .try_reserve_owned()
            .expect("new channel has capacity");
        queue.push_back(QueueItem {
            prompt_tokens,
            request_id,
            prompt,
            sampling,
            tx,
            terminal_permit,
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
            if !self.is_healthy() {
                for mut seq in active.drain(..) {
                    self.backend.cancel_sequence(&seq.request_id);
                    self.cache.lock().release(&seq.request_id.0);
                    seq.send_terminal(TokenEvent::Error(
                        "Worker unavailable; restart required".into(),
                    ));
                }
                for item in self.queue.lock().drain(..) {
                    item.terminal_permit.send(TokenEvent::Error(
                        "Worker unavailable; restart required".into(),
                    ));
                }
                self.active_count.store(0, Ordering::Relaxed);
                break;
            }
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

                        let prompt_tokens = item.prompt_tokens;

                        if let Err(err) = self.backend.reserve_sequence(
                            &item.request_id,
                            prompt_tokens.len(),
                            item.sampling.max_tokens,
                        ) {
                            self.record_failure(&err);
                            // Backends may have partially prepared request state.
                            self.backend.cancel_sequence(&item.request_id);
                            item.terminal_permit
                                .send(TokenEvent::Error(err.to_string()));
                            if !self.is_healthy() {
                                break;
                            }
                            continue;
                        }

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

                        // Allocate initial KV cache blocks for prompt tokens
                        let initial_tokens = prompt_tokens.len().max(1);
                        let blocks = {
                            let mut cache = self.cache.lock();
                            cache.allocate(&item.request_id.0, initial_tokens)
                        };

                        match blocks {
                            Ok(blocks) => {
                                let mut seq = ActiveSequence::new(
                                    item.request_id,
                                    item.prompt,
                                    item.sampling,
                                    item.tx,
                                );
                                seq.terminal_permit = Some(item.terminal_permit);
                                seq.created_at = item.enqueued_at;
                                seq.prompt_tokens = prompt_tokens;
                                seq.kv_blocks = blocks;
                                active.push(seq);
                            }
                            Err(err) => {
                                self.backend.cancel_sequence(&item.request_id);
                                warn!("Cache allocation failed for {}: {}", item.request_id, err);
                                item.terminal_permit
                                    .send(TokenEvent::Error(err.to_string()));
                            }
                        }
                    }
                    None => break,
                }
            }

            if !self.is_healthy() {
                continue;
            }
            self.active_count.store(active.len(), Ordering::Relaxed);

            // If no sequences active, sleep until a new request arrives
            if active.is_empty() {
                tokio::select! {
                    _ = self.queue_notify.notified() => continue,
                    _ = self.shutdown.notified() => continue,
                    // Observe quarantine published by another worker sharing this backend.
                    _ = tokio::time::sleep(Duration::from_millis(25)) => continue,
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
                    self.backend.cancel_sequence(&cancelled.request_id);
                    self.cache.lock().release(&cancelled.request_id.0);
                } else {
                    i += 1;
                }
            }
            self.active_count.store(active.len(), Ordering::Relaxed);

            if active.is_empty() {
                continue;
            }

            // Reserve logical space before executing the next step. Contending
            // requests can exhaust the shared pool after initial admission.
            let mut i = 0;
            while i < active.len() {
                let seq = &mut active[i];
                let allocation = {
                    let mut cache = self.cache.lock();
                    let next_len = seq.prompt_tokens.len() + seq.output_tokens.len() + 1;
                    if cache.blocks_needed(next_len) > seq.kv_blocks.len() {
                        cache.allocate_block(&seq.request_id.0).map(Some)
                    } else {
                        Ok(None)
                    }
                };
                match allocation {
                    Ok(Some(block)) => {
                        seq.kv_blocks.push(block);
                        i += 1;
                    }
                    Ok(None) => i += 1,
                    Err(err) => {
                        let mut failed = active.swap_remove(i);
                        failed.send_terminal(TokenEvent::Error(err.to_string()));
                        self.backend.cancel_sequence(&failed.request_id);
                        self.cache.lock().release(&failed.request_id.0);
                    }
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
                    self.record_failure(&err);
                    warn!("Backend step_batch error: {}", err);
                    for mut seq in active.drain(..) {
                        self.backend.cancel_sequence(&seq.request_id);
                        self.cache.lock().release(&seq.request_id.0);
                        seq.send_terminal(TokenEvent::Error(err.to_string()));
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

                let hit_max = seq.output_tokens.len() >= seq.sampling.max_tokens;
                let (mut visible_text, matched_stop) = seq.stop_matcher.push(&step.text);
                if !matched_stop && (step.is_eos || hit_max) {
                    visible_text.push_str(&seq.stop_matcher.finish());
                }

                // Keep one event per generated token, including empty decoded
                // fragments. Usage counts generated IDs, not visible strings.
                match seq.tx.try_send(TokenEvent::Token(visible_text)) {
                    Ok(_) => {}
                    Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                        warn!(
                            "Client channel full for request {}, dropping slow client to avoid stalling batch",
                            seq.request_id
                        );
                        seq.is_finished = true;
                        seq.terminal_error =
                            Some("Client too slow: token channel capacity exceeded".to_string());
                    }
                    Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                        seq.is_finished = true;
                        seq.finish_reason = Some("abort".to_string());
                    }
                }

                if !seq.is_finished && (step.is_eos || matched_stop || hit_max) {
                    seq.is_finished = true;
                    seq.finish_reason = Some(if step.is_eos || matched_stop {
                        "stop".to_string()
                    } else {
                        "length".to_string()
                    });
                }
            }

            // Evict finished sequences and reclaim KV-cache blocks immediately
            let mut j = 0;
            while j < active.len() {
                if active[j].is_finished || active[j].is_cancelled() {
                    let mut finished = active.swap_remove(j);

                    if finished.is_finished {
                        let usage = finished.usage();
                        let finish_reason = finished
                            .finish_reason
                            .clone()
                            .unwrap_or_else(|| "stop".to_string());
                        let terminal = match finished.terminal_error.take() {
                            Some(error) => TokenEvent::Error(error),
                            None => TokenEvent::Done {
                                usage,
                                finish_reason,
                            },
                        };
                        finished.send_terminal(terminal);

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

                    // Instant backend model slot and KV-cache block recovery!
                    self.backend.cancel_sequence(&finished.request_id);
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
        self.accepting.store(false, Ordering::Release);
        self.shutdown.notify_one();
    }
}

pub type SharedScheduler = Arc<Scheduler>;
