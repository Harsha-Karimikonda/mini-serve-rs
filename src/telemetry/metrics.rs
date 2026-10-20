use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct Telemetry {
    total_requests: Arc<AtomicUsize>,
    total_tokens: Arc<AtomicUsize>,
    token_timestamps: Arc<Mutex<VecDeque<(Instant, usize)>>>,
    window_duration: Duration,
}

impl Telemetry {
    pub fn new(window_seconds: u64) -> Self {
        Self {
            total_requests: Arc::new(AtomicUsize::new(0)),
            total_tokens: Arc::new(AtomicUsize::new(0)),
            token_timestamps: Arc::new(Mutex::new(VecDeque::new())),
            window_duration: Duration::from_secs(window_seconds),
        }
    }

    pub fn record_request(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_tokens(&self, count: usize) {
        if count == 0 {
            return;
        }
        self.total_tokens.fetch_add(count, Ordering::Relaxed);

        let now = Instant::now();
        let mut list = self.token_timestamps.lock();
        list.push_back((now, count));
        self.prune(&mut list, now);
    }

    fn prune(&self, list: &mut VecDeque<(Instant, usize)>, now: Instant) {
        while let Some(&(time, _)) = list.front() {
            if now.duration_since(time) > self.window_duration {
                list.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn tokens_per_second(&self) -> f64 {
        let now = Instant::now();
        let mut list = self.token_timestamps.lock();
        self.prune(&mut list, now);

        if list.is_empty() {
            return 0.0;
        }

        let total_in_window: usize = list.iter().map(|(_, count)| count).sum();
        let oldest = list.front().unwrap().0;
        let elapsed = now.duration_since(oldest).as_secs_f64();

        if elapsed <= 0.05 {
            total_in_window as f64
        } else {
            (total_in_window as f64 / elapsed * 10.0).round() / 10.0
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.total_tokens.load(Ordering::Relaxed)
    }

    pub fn total_requests(&self) -> usize {
        self.total_requests.load(Ordering::Relaxed)
    }
}

pub type SharedTelemetry = Arc<Telemetry>;

pub fn create_telemetry() -> SharedTelemetry {
    Arc::new(Telemetry::new(5))
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ExecutionStats {
    pub accepted: usize,
    pub completed: usize,
    pub errors: usize,
    pub cancellations: usize,
    pub generated_tokens: usize,
    pub queue_delay_count: usize,
    pub queue_delay_us: usize,
    pub ttft_count: usize,
    pub ttft_us: usize,
}

impl std::ops::AddAssign for ExecutionStats {
    fn add_assign(&mut self, rhs: Self) {
        self.accepted += rhs.accepted;
        self.completed += rhs.completed;
        self.errors += rhs.errors;
        self.cancellations += rhs.cancellations;
        self.generated_tokens += rhs.generated_tokens;
        self.queue_delay_count += rhs.queue_delay_count;
        self.queue_delay_us += rhs.queue_delay_us;
        self.ttft_count += rhs.ttft_count;
        self.ttft_us += rhs.ttft_us;
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Outcome {
    Completed,
    Error,
    Cancelled,
}

/// Every accepted request records exactly one outcome, including unwind and drop.
#[derive(Debug)]
pub struct RequestTrace {
    counters: Arc<Mutex<ExecutionStats>>,
    started: Instant,
    outcome: Option<Outcome>,
    first_token: bool,
}

impl RequestTrace {
    pub fn new(counters: Arc<Mutex<ExecutionStats>>) -> Self {
        counters.lock().accepted += 1;
        Self {
            counters,
            started: Instant::now(),
            outcome: None,
            first_token: false,
        }
    }
    pub fn admitted(&self) {
        let mut stats = self.counters.lock();
        stats.queue_delay_count += 1;
        stats.queue_delay_us += self.started.elapsed().as_micros().min(usize::MAX as u128) as usize;
    }
    pub fn token(&mut self) {
        let mut stats = self.counters.lock();
        stats.generated_tokens += 1;
        if !self.first_token {
            self.first_token = true;
            stats.ttft_count += 1;
            stats.ttft_us += self.started.elapsed().as_micros().min(usize::MAX as u128) as usize;
        }
    }
    pub fn finish(&mut self, outcome: Outcome) {
        if self.outcome.is_none() {
            self.outcome = Some(outcome);
            let mut stats = self.counters.lock();
            match outcome {
                Outcome::Completed => stats.completed += 1,
                Outcome::Error => stats.errors += 1,
                Outcome::Cancelled => stats.cancellations += 1,
            }
        }
    }
}
impl Drop for RequestTrace {
    fn drop(&mut self) {
        self.finish(Outcome::Cancelled);
    }
}
