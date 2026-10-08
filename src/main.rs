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

    // Initialize backend (Mock by default, Candle on GPU/Metal)
    let backend = Arc::new(MockBackend::new(&settings.model, 3));

    // Spawn multi-worker schedulers with shared memory weights and caches
    let mut worker_handles = Vec::with_capacity(settings.num_workers);
    for i in 0..settings.num_workers {
        let worker_id = format!("worker-{}", i + 1);
        let scheduler = Arc::new(Scheduler::new(
            settings.max_batch_size,
            settings.max_waiting_requests,
            Arc::clone(&cache),
            Arc::clone(&prefix_cache),
            Arc::clone(&backend) as Arc<dyn mini_serve::backends::ModelBackend>,
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
