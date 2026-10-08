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
