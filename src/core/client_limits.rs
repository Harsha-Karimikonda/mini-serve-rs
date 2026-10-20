use parking_lot::Mutex;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug)]
struct Usage {
    window: Instant,
    requests: usize,
    concurrent: usize,
}

/// One bounded record per configured credential (or one anonymous record).
#[derive(Debug)]
pub struct ClientLimits {
    rate: usize,
    concurrency: usize,
    usage: Mutex<Usage>,
}

#[derive(Debug)]
pub struct ClientPermit(Arc<ClientLimits>);

impl Drop for ClientPermit {
    fn drop(&mut self) {
        self.0.usage.lock().concurrent -= 1;
    }
}

impl ClientLimits {
    pub fn new(rate: usize, concurrency: usize) -> Arc<Self> {
        Arc::new(Self {
            rate,
            concurrency,
            usage: Mutex::new(Usage {
                window: Instant::now(),
                requests: 0,
                concurrent: 0,
            }),
        })
    }

    /// Returns a Retry-After duration on rejection. Rejected attempts do not
    /// consume the rate budget; accepted attempts include invalid request bodies.
    pub fn acquire(self: &Arc<Self>) -> Result<Arc<ClientPermit>, u64> {
        self.acquire_at(Instant::now())
    }

    fn acquire_at(self: &Arc<Self>, now: Instant) -> Result<Arc<ClientPermit>, u64> {
        let mut usage = self.usage.lock();
        let elapsed = now.duration_since(usage.window);
        if elapsed >= Duration::from_secs(60) {
            usage.window = now;
            usage.requests = 0;
        }
        if self.rate != 0 && usage.requests >= self.rate {
            let remaining =
                Duration::from_secs(60).saturating_sub(now.duration_since(usage.window));
            return Err(remaining.as_secs() + u64::from(remaining.subsec_nanos() != 0));
        }
        if self.concurrency != 0 && usage.concurrent >= self.concurrency {
            return Err(1);
        }
        usage.requests = usage.requests.saturating_add(1);
        usage.concurrent += 1;
        Ok(Arc::new(ClientPermit(self.clone())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_window_and_shared_lease_are_exact() {
        let client = ClientLimits::new(2, 1);
        let start = client.usage.lock().window;
        let permit = client.acquire_at(start).unwrap();
        let clone = permit.clone();
        drop(permit);
        assert_eq!(client.acquire_at(start).unwrap_err(), 1);
        drop(clone);
        drop(client.acquire_at(start + Duration::from_secs(1)).unwrap());
        assert_eq!(
            client
                .acquire_at(start + Duration::from_millis(1500))
                .unwrap_err(),
            59
        );
        drop(client.acquire_at(start + Duration::from_secs(60)).unwrap());
    }

    #[test]
    fn simultaneous_admissions_cannot_bypass_concurrency() {
        let client = ClientLimits::new(0, 1);
        let barrier = Arc::new(std::sync::Barrier::new(9));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let client = client.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    client.acquire()
                })
            })
            .collect();
        barrier.wait();
        let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
        drop(results);
        assert!(client.acquire().is_ok());
    }
}
