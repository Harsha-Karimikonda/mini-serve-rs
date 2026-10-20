use async_trait::async_trait;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use rlse::{
    api::{create_app, AppState},
    backends::{ModelBackend, StepToken},
    cache::{create_shared_cache, create_shared_prefix_cache},
    core::{
        config::Settings,
        errors::EngineError,
        types::{ActiveSequence, RequestId, SamplingParams},
    },
    routing::router::{Router, WorkerHandle},
    scheduler::Scheduler,
    telemetry::create_telemetry,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tower::ServiceExt;
struct EmptyTokens(AtomicBool);
#[async_trait]
impl ModelBackend for EmptyTokens {
    fn name(&self) -> &str {
        "empty"
    }
    async fn init_sequence(&self, _: &str) -> Result<Vec<u32>, EngineError> {
        Ok(vec![1])
    }
    async fn step_batch(&self, seq: &mut [ActiveSequence]) -> Result<Vec<StepToken>, EngineError> {
        if self.0.load(Ordering::SeqCst) {
            return Err(EngineError::CachePressure("test execution error".into()));
        }
        Ok(seq
            .iter()
            .map(|_| StepToken {
                token_id: 1,
                text: String::new(),
                is_eos: false,
            })
            .collect())
    }
}
#[tokio::test]
async fn execution_counters_count_empty_tokens_and_exact_outcomes_for_json_and_sse() {
    tokio::time::timeout(std::time::Duration::from_secs(3),async{
        let backend=Arc::new(EmptyTokens(AtomicBool::new(false)));
        let cache=create_shared_cache(64,16);
        let prefix=create_shared_prefix_cache(16,8);
        let worker=Arc::new(Scheduler::new(2,8,cache.clone(),prefix.clone(),backend.clone()));
        let router=Arc::new(Router::new(vec![WorkerHandle::new("metered",worker.clone())]));
        let app=create_app(AppState{router:router.clone(),cache,prefix_cache:prefix,telemetry:create_telemetry(),settings:Settings::default()});
        let cancelled=worker.submit(RequestId("cancelled".into()),"hello".into(),SamplingParams::default()).await.unwrap();
        drop(cancelled);
        let task=worker.start_loop();
        for stream in [false,true]{
            let res=app.clone().oneshot(Request::builder().method("POST").uri("/v1/completions").header("content-type","application/json").body(Body::from(serde_json::json!({"model":"empty","prompt":"hello","max_tokens":3,"stream":stream}).to_string())).unwrap()).await.unwrap();
            assert_eq!(res.status(),StatusCode::OK);
            let bytes=res.into_body().collect().await.unwrap().to_bytes();
            if stream {
                let terminal=std::str::from_utf8(&bytes).unwrap().lines().filter_map(|line|line.strip_prefix("data: ")).filter(|data|*data!="[DONE]").map(|data|serde_json::from_str::<serde_json::Value>(data).unwrap()).find(|v|v.get("usage").is_some()).unwrap();
                assert_eq!(terminal["usage"]["completion_tokens"],3);
            } else {assert_eq!(serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()["usage"]["completion_tokens"],3);}
        }
        backend.0.store(true,Ordering::SeqCst);
        let res=app.clone().oneshot(Request::builder().method("POST").uri("/v1/completions").header("content-type","application/json").body(Body::from(r#"{"model":"empty","prompt":"hello","max_tokens":3}"#)).unwrap()).await.unwrap();
        assert_eq!(res.status(),StatusCode::INTERNAL_SERVER_ERROR);
        worker.shutdown();task.await.unwrap();
        let stats=router.execution_stats();
        assert_eq!((stats.accepted,stats.completed,stats.errors,stats.cancellations,stats.generated_tokens),(4,2,1,1,6));
        assert_eq!(stats.queue_delay_count,3);
        assert_eq!(stats.ttft_count,2);

        for path in ["/api/status","/metrics"] {
            let res=app.clone().oneshot(Request::builder().uri(path).body(Body::empty()).unwrap()).await.unwrap();
            let bytes=res.into_body().collect().await.unwrap().to_bytes();
            let text=std::str::from_utf8(&bytes).unwrap();
            if path=="/api/status"{let v:serde_json::Value=serde_json::from_str(text).unwrap();assert_eq!(v["total_tokens"],6);assert_eq!(v["execution"]["cancellations"],1);assert_eq!(v["workers"][0]["healthy"],false);}
            else {assert!(text.contains("mini_tokens_total 6\n"));assert!(text.contains("mini_execution_errors_total 1\n"));assert!(text.contains("mini_ready_workers 0\n"));}
        }
    }).await.expect("execution telemetry hung");
}
