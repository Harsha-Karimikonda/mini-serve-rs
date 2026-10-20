#![cfg(any(feature = "metal", feature = "cuda"))]
use rlse::{
    backends::{CandleBackend, ModelBackend},
    core::types::{ActiveSequence, RequestId, SamplingParams},
};
use serde_json::Value;

#[tokio::test]
#[ignore = "requires explicit GPU and pinned local pretrained weights; compares independent Transformers CPU reference"]
async fn pretrained_gpu_matches_independent_generation_reference() {
    let model_dir =
        std::env::var("RLSE_GPU_MODEL_DIR").unwrap_or_else(|_| "models/Qwen3-0.6B".into());
    let device = std::env::var("RLSE_GPU_DEVICE").unwrap_or_else(|_| "metal".into());
    assert!(
        matches!(device.as_str(), "metal" | "cuda"),
        "explicit GPU required"
    );
    assert!(
        std::path::Path::new(&model_dir)
            .join("model.safetensors")
            .is_file(),
        "local pretrained weights required"
    );
    let backend = CandleBackend::load_hf(&model_dir, &device).unwrap();
    assert!(matches!(
        (device.as_str(), backend.device()),
        ("metal", candle_core::Device::Metal(_)) | ("cuda", candle_core::Device::Cuda(_))
    ));
    println!("ACTUAL_DEVICE={:?}; model={model_dir}; temperature=0 top_p=1; independent CPU/F32 reference",backend.device());
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/generation/qwen3-reference.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let messages: Vec<rlse::core::types::ChatMessage> =
            serde_json::from_value(case["messages"].clone()).unwrap();
        let prompt = backend.format_chat(&messages).unwrap();
        assert_eq!(prompt, case["rendered"].as_str().unwrap());
        let prompt_ids = backend.init_sequence(&prompt).await.unwrap();
        assert_eq!(
            prompt_ids,
            serde_json::from_value::<Vec<u32>>(case["prompt_token_ids"].clone()).unwrap()
        );
        let (tx, _rx) = tokio::sync::mpsc::channel(65);
        let mut seq = ActiveSequence::new(
            RequestId(case["name"].as_str().unwrap().into()),
            prompt,
            SamplingParams {
                temperature: 0.,
                top_p: 1.,
                max_tokens: 32,
                stop: vec![],
            },
            tx,
        );
        seq.prompt_tokens = prompt_ids;
        backend
            .reserve_sequence(&seq.request_id, seq.prompt_tokens.len(), 32)
            .unwrap();
        for _ in 0..32 {
            let steps = backend
                .step_batch(std::slice::from_mut(&mut seq))
                .await
                .unwrap();
            let token = &steps[0];
            seq.output_tokens.push(token.token_id);
            seq.output_text.push_str(&token.text);
            if token.is_eos {
                break;
            }
        }
        backend.cancel_sequence(&seq.request_id);
        assert_eq!(
            seq.output_tokens,
            serde_json::from_value::<Vec<u32>>(case["token_ids"].clone()).unwrap(),
            "{} greedy tokens differ from independent reference",
            case["name"]
        );
        assert_eq!(seq.output_text, case["text"].as_str().unwrap());
        // Independently generated token IDs/text are compared before applying
        // the common stop contract, which has separate JSON/SSE regressions.
        let stops: Vec<String> = serde_json::from_value(case["stop"].clone()).unwrap();
        let mut matcher = rlse::core::stop::StopMatcher::new(stops.clone());
        let (mut text, matched) = matcher.push(&seq.output_text);
        if !matched {
            text.push_str(&matcher.finish());
        }
        let reference = case["text"].as_str().unwrap();
        let end = stops
            .iter()
            .filter_map(|stop| reference.find(stop))
            .min()
            .unwrap_or(reference.len());
        assert_eq!(text, &reference[..end]);
        println!(
            "PASS: {} tokens={} reference={:?} visible_after_stop={text:?}",
            case["name"],
            seq.output_tokens.len(),
            seq.output_text
        );
    }
}
