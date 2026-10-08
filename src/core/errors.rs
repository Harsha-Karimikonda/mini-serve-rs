use axum::{
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

#[derive(Debug, Clone)]
pub enum EngineError {
    AdmissionError(String),
    CachePressure(String),
    NoHealthyWorkers(String),
    RequestCancelled(String),
    TokenLimitExceeded(String),
    InvalidRequest(String),
    BackendError(String),
}

// Since thiserror is not in Cargo.toml yet, implement std::fmt::Display manually or add thiserror.
// Writing standard Display & Error implementations keeps zero extra dependencies!

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AdmissionError(msg) => write!(f, "Queue saturated: {}", msg),
            Self::CachePressure(msg) => write!(f, "KV cache capacity exceeded: {}", msg),
            Self::NoHealthyWorkers(msg) => write!(f, "No healthy workers available: {}", msg),
            Self::RequestCancelled(msg) => write!(f, "Request cancelled: {}", msg),
            Self::TokenLimitExceeded(msg) => write!(f, "Token limit exceeded: {}", msg),
            Self::InvalidRequest(msg) => write!(f, "Invalid request: {}", msg),
            Self::BackendError(msg) => write!(f, "Backend error: {}", msg),
        }
    }
}

impl std::error::Error for EngineError {}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: u16,
}

#[derive(Debug, Serialize)]
pub struct ErrorPayload {
    pub error: ErrorDetail,
}

impl IntoResponse for EngineError {
    fn into_response(self) -> Response {
        let (status, err_type, retry_after) = match &self {
            Self::AdmissionError(_) => (
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_exceeded",
                Some("1"),
            ),
            Self::CachePressure(_) => (
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_exceeded",
                Some("1"),
            ),
            Self::NoHealthyWorkers(_) => (
                StatusCode::SERVICE_UNAVAILABLE,
                "service_unavailable",
                Some("2"),
            ),
            Self::RequestCancelled(_) => (StatusCode::BAD_REQUEST, "client_closed_request", None),
            Self::TokenLimitExceeded(_) => (StatusCode::BAD_REQUEST, "invalid_request_error", None),
            Self::InvalidRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error", None),
            Self::BackendError(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_server_error",
                None,
            ),
        };

        let payload = ErrorPayload {
            error: ErrorDetail {
                message: self.to_string(),
                error_type: err_type.to_string(),
                param: None,
                code: status.as_u16(),
            },
        };

        let mut res = (status, Json(payload)).into_response();
        if let Some(secs) = retry_after {
            if let Ok(val) = HeaderValue::from_str(secs) {
                res.headers_mut().insert(header::RETRY_AFTER, val);
            }
        }
        res
    }
}
