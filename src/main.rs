use clap::Parser;
use std::sync::Arc;
use tracing::info;

use mini_serve::api::{create_app, AppState};
use mini_serve::backends::mock::MockBackend;
use mini_serve::cache::{create_shared_cache, create_shared_prefix_cache};
use mini_serve::core::config::Settings;
use mini_serve::routing::router::{Router, WorkerHandle};
use mini_serve::scheduler::scheduler::Scheduler;
use mini_serve::telemetry::{create_telemetry, init_logging};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let settings = Settings::parse();
    init_logging(&settings.log_level);

    info!(
        "Starting Mini-Serve Rust Core v0.2.0 on {}:{} (model={}, workers={})",
        settings.host, settings.port, settings.model, settings.num_workers
    );

    let cache = create_shared_cache(settings.kv_cache_blocks, settings.block_size);
    let prefix_cache = create_shared_prefix_cache(settings.block_size, 512);
    let telemetry = create_telemetry();

    if let Some(ref token) = settings.hf_token {
        std::env::set_var("HF_TOKEN", token);
    }

    // Initialize backend (Mock if requested, or Candle on Metal/CPU)
    let backend: Arc<dyn mini_serve::backends::ModelBackend> = if settings.model == "mock" {
        Arc::new(MockBackend::new(&settings.model, 3))
    } else {
        info!(
            "Loading real Hugging Face model {} via Candle backend...",
            settings.model
        );
        match mini_serve::backends::CandleBackend::load_hf(&settings.model, &settings.device) {
            Ok(candle_backend) => Arc::new(candle_backend),
            Err(e) => {
                tracing::error!(
                    "Failed to load Candle model: {}. Falling back to MockBackend.",
                    e
                );
                Arc::new(MockBackend::new(&settings.model, 3))
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
        router,
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

    info!("Mini-Serve gracefully terminated");
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    info!("Shutdown signal received, draining in-flight requests...");
}
