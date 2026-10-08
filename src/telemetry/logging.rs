use std::env;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

pub fn init_logging(log_level: &str) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("mini_serve={},tower_http=info", log_level)));

    let format = env::var("MINI_LOG_FORMAT").unwrap_or_else(|_| "text".to_string());

    if format.to_lowercase() == "json" {
        tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer().json().with_target(false))
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer().with_target(false))
            .init();
    }
}

/// Centralized request audit logger. Prompts and generated text are never logged.
pub fn log_request(method: &str, path: &str, request_id: &str, status: u16, duration_ms: f64) {
    tracing::info!(
        method = %method,
        path = %path,
        request_id = %request_id,
        status = status,
        duration_ms = duration_ms,
        "Request processed"
    );
}
