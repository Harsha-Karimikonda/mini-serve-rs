#![cfg(unix)]
use futures_util::StreamExt;
use rlse::sse::CompletionStream;
use std::{
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

struct Server(Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
async fn start(timeout: u64) -> (Server, String, reqwest::Client) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let child = Command::new(env!("CARGO_BIN_EXE_rlse"))
        .args([
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--model",
            "mock",
            "--device",
            "cpu",
            "--shutdown-timeout-ms",
            &timeout.to_string(),
        ])
        .env_remove("RLSE_API_KEY")
        .env_remove("RLSE_API_KEYS")
        .env_remove("RLSE_CLIENT_REQUESTS_PER_MINUTE")
        .env_remove("RLSE_CLIENT_MAX_CONCURRENT")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut server = Server(child);
    let url = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            assert!(
                server.0.try_wait().unwrap().is_none(),
                "server exited before ready"
            );
            if client
                .get(format!("{url}/ready"))
                .send()
                .await
                .is_ok_and(|r| r.status().is_success())
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("server did not start");
    (server, url, client)
}
fn interrupt(server: &Server) {
    assert!(Command::new("/bin/kill")
        .args(["-INT", &server.0.id().to_string()])
        .status()
        .unwrap()
        .success());
}
async fn exit(server: &mut Server) -> std::process::ExitStatus {
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if let Some(status) = server.0.try_wait().unwrap() {
                return status;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("server failed to exit within test bound")
}
#[tokio::test]
async fn ctrl_c_exits_idle_process_cleanly() {
    let (mut server, _, _) = start(1000).await;
    interrupt(&server);
    assert!(exit(&mut server).await.success());
}
#[tokio::test]
async fn ctrl_c_drains_actual_http_stream_and_joins_workers() {
    let (mut server, url, client) = start(2000).await;
    let response=client.post(format!("{url}/v1/completions")).json(&serde_json::json!({"model":"mock","prompt":"hello","stream":true,"temperature":0,"max_tokens":32})).send().await.unwrap();
    assert!(response.status().is_success());
    let mut stream = response.bytes_stream();
    let first = stream.next().await.unwrap().unwrap();
    interrupt(&server);
    let mut parser = CompletionStream::default();
    parser.push(&first).unwrap();
    while let Some(chunk) = stream.next().await {
        parser.push(&chunk.unwrap()).unwrap();
    }
    assert_eq!(parser.finish().unwrap(), 32);
    assert!(exit(&mut server).await.success());
}
#[tokio::test]
async fn zero_deadline_bounds_process_shutdown_during_generation() {
    let (mut server, url, client) = start(0).await;
    let response = client
        .post(format!("{url}/v1/completions"))
        .json(&serde_json::json!({"model":"mock","prompt":"hello","stream":true,"max_tokens":32}))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let mut stream = response.bytes_stream();
    let _ = stream.next().await.unwrap().unwrap();
    let started = Instant::now();
    interrupt(&server);
    let status = exit(&mut server).await;
    assert!(matches!(status.code(), Some(0 | 1)), "{status}");
    assert!(started.elapsed() < Duration::from_secs(3));
    // Deadline expiry may terminate HTTP delivery. No success assertion on this body.
    drop(stream);
}
