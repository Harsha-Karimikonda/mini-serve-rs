use clap::Parser;
use std::sync::Arc;
use tracing::info;

use rlse::api::{create_app, AppState};
use rlse::backends::mock::MockBackend;
use rlse::cache::tensor_memory::TensorMemoryBudget;
use rlse::cache::{create_shared_cache, PrefixCache};
use rlse::core::config::Settings;
use rlse::routing::router::{Router, WorkerHandle};
use rlse::scheduler::scheduler::Scheduler;
use rlse::telemetry::{create_telemetry, init_logging};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let settings = Settings::parse();
    settings.validate()?;
    init_logging(&settings.log_level);

    info!(
        "Starting RLSE (Rust LLM Serving Engine) v0.2.0 on {}:{} (model={}, workers={})",
        settings.host, settings.port, settings.model, settings.num_workers
    );

    let cache = create_shared_cache(settings.kv_cache_blocks, settings.block_size);
    let memory_budget = TensorMemoryBudget::new(
        settings.tensor_memory_budget_bytes,
        settings.execution_headroom_bytes,
    )?;
    let prefix_cache = Arc::new(parking_lot::Mutex::new(PrefixCache::with_memory_budget(
        settings.block_size,
        512,
        memory_budget,
    )));
    let telemetry = create_telemetry();

    if let Some(ref token) = settings.hf_token {
        std::env::set_var("HF_TOKEN", token);
    }

    // Initialize backend (Mock if requested, or Candle on Metal/CPU)
    let backend: Arc<dyn rlse::backends::ModelBackend> = if settings.model == "mock" {
        Arc::new(MockBackend::new(&settings.model, 3))
    } else {
        info!(
            "Loading real Hugging Face model {} via Candle backend...",
            settings.model
        );
        match rlse::backends::CandleBackend::load_hf(&settings.model, &settings.device) {
            Ok(mut candle_backend) => {
                candle_backend.try_set_prefix_cache(prefix_cache.clone())?;
                Arc::new(candle_backend)
            }
            Err(e) => {
                if settings.allow_mock_fallback {
                    tracing::warn!(
                        "Failed to load Candle model: {}. Falling back to MockBackend (--allow-mock-fallback enabled).",
                        e
                    );
                    Arc::new(MockBackend::new(&settings.model, 3))
                } else {
                    tracing::error!(
                        "Failed to load Candle model: {}. Terminating (pass --allow-mock-fallback to override).",
                        e
                    );
                    return Err(e.into());
                }
            }
        }
    };

    // Spawn multi-worker schedulers with shared memory weights and caches
    let mut worker_handles = Vec::with_capacity(settings.num_workers);
    for i in 0..settings.num_workers {
        let worker_id = format!("worker-{}", i + 1);
        let scheduler = Arc::new(Scheduler::new(
            settings.max_batch_size,
            settings.max_waiting_requests,
            Arc::clone(&cache),
            Arc::clone(&prefix_cache),
            Arc::clone(&backend),
        ));

        // Start worker continuous batching loop
        scheduler.start_loop();
        worker_handles.push(WorkerHandle::new(worker_id, scheduler));
    }

    let router = Arc::new(Router::new(worker_handles));

    let state = AppState {
        router: Arc::clone(&router),
        cache,
        prefix_cache,
        telemetry,
        settings: settings.clone(),
    };

    let app = create_app(state);
    let addr = format!("{}:{}", settings.host, settings.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("Server listening on http://{}", addr);
    info!("Dashboard available at http://{}/dashboard", addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Shutting down continuous batching schedulers...");
    for worker in router.workers() {
        worker.scheduler.shutdown();
    }

    info!("RLSE gracefully terminated");
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    info!("Shutdown signal received, draining in-flight requests...");
}
