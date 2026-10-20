use crate::core::{client_limits::ClientLimits, config::Settings};
use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use sha2::{Digest, Sha256};
use std::sync::Arc;

#[derive(Clone)]
pub(super) struct ApiAuth {
    clients: Vec<([u8; 32], Arc<ClientLimits>)>,
    anonymous: Arc<ClientLimits>,
}

impl ApiAuth {
    pub(super) fn new(settings: &Settings) -> Self {
        let mut clients = Vec::new();
        for key in settings.api_key.iter().chain(&settings.api_keys) {
            let digest: [u8; 32] = Sha256::digest(key.as_bytes()).into();
            if !clients.iter().any(|(existing, _)| *existing == digest) {
                clients.push((
                    digest,
                    ClientLimits::new(
                        settings.client_requests_per_minute,
                        settings.client_max_concurrent,
                    ),
                ));
            }
        }
        Self {
            clients,
            anonymous: ClientLimits::new(
                settings.client_requests_per_minute,
                settings.client_max_concurrent,
            ),
        }
    }
}

pub(super) async fn authenticate(
    State(auth): State<ApiAuth>,
    mut request: Request,
    next: Next,
) -> Response {
    let mut values = request.headers().get_all(header::AUTHORIZATION).iter();
    let supplied = values.next().and_then(|value| value.to_str().ok());
    let digest: Option<[u8; 32]> = if values.next().is_none() {
        supplied
            .and_then(|value| value.split_once(' '))
            .filter(|(scheme, key)| {
                scheme.eq_ignore_ascii_case("bearer")
                    && !key.is_empty()
                    && key.bytes().all(|b| b.is_ascii_graphic())
            })
            .map(|(_, key)| Sha256::digest(key.as_bytes()).into())
    } else {
        None
    };
    let client = if auth.clients.is_empty() {
        Some(&auth.anonymous)
    } else {
        digest.as_ref().and_then(|actual| {
            auth.clients.iter().find_map(|(expected, client)| {
                let equal = actual
                    .iter()
                    .zip(expected)
                    .fold(0u8, |diff, (a, b)| diff | (a ^ b))
                    == 0;
                equal.then_some(client)
            })
        })
    };
    if let Some(client) = client {
        if request.method() == axum::http::Method::POST
            && matches!(
                request.uri().path(),
                "/v1/completions" | "/v1/chat/completions"
            )
        {
            match client.acquire() {
                Ok(permit) => { request.extensions_mut().insert(permit); }
                Err(retry) => return (
                    StatusCode::TOO_MANY_REQUESTS,
                    [(header::RETRY_AFTER, retry.to_string())],
                    Json(serde_json::json!({"error": {"message": "Client generation quota exceeded", "type": "rate_limit_error", "code": 429}})),
                ).into_response(),
            }
        }
        next.run(request).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            Json(serde_json::json!({"error": {
                "message": "Invalid or missing API key",
                "type": "authentication_error",
                "code": 401
            }})),
        )
            .into_response()
    }
}
