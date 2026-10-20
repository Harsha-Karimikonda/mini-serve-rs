use rlse::{backends::chat::ChatTemplate, core::types::ChatMessage};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const FIXTURES: [(&str, &str); 2] = [
    (
        include_str!("fixtures/chat/qwen25-tokenizer_config.json"),
        include_str!("fixtures/chat/qwen25-reference.json"),
    ),
    (
        include_str!("fixtures/chat/qwen3-tokenizer_config.json"),
        include_str!("fixtures/chat/qwen3-reference.json"),
    ),
];

#[test]
fn pinned_qwen_templates_match_transformers_reference() {
    for (config, reference) in FIXTURES {
        let reference: Value = serde_json::from_str(reference).unwrap();
        assert_eq!(
            hex::encode(Sha256::digest(config.as_bytes())),
            reference["config_sha256"]
        );
        let template = ChatTemplate::from_config(config).unwrap();
        for case in reference["cases"].as_array().unwrap() {
            let messages: Vec<ChatMessage> =
                serde_json::from_value(case["messages"].clone()).unwrap();
            assert_eq!(
                template.render(&messages).unwrap(),
                case["rendered"],
                "{}",
                reference["model"]
            );
        }
    }
}

#[test]
fn invalid_roles_and_template_metadata_fail_explicitly() {
    for config in [
        "{",
        "{}",
        r#"{"chat_template":""}"#,
        r#"{"chat_template":[]}"#,
        r#"{"chat_template":"{% invalid %}"}"#,
    ] {
        assert!(ChatTemplate::from_config(config).is_err(), "{config}");
    }
    let template = ChatTemplate::from_config(FIXTURES[0].0).unwrap();
    assert!(template
        .render(&[])
        .unwrap_err()
        .to_string()
        .contains("Messages"));
    let invalid = vec![ChatMessage {
        role: "tool".into(),
        content: "text".into(),
    }];
    assert!(template
        .render(&invalid)
        .unwrap_err()
        .to_string()
        .contains("Unsupported chat role"));
    for source in [
        "{{ raise_exception('unsupported request') }}",
        "{{ unknown_function() }}",
    ] {
        let config = json!({"chat_template":source}).to_string();
        let template = ChatTemplate::from_config(&config).unwrap();
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hello".into(),
        }];
        assert!(template
            .render(&messages)
            .unwrap_err()
            .to_string()
            .contains("Unsupported chat template"));
    }
}

#[test]
#[ignore = "requires pinned tokenizer.json files in RLSE_CHAT_ARTIFACTS/qwen25 and qwen3"]
fn pinned_token_ids_match_transformers_reference() {
    let root = std::env::var("RLSE_CHAT_ARTIFACTS").expect("set RLSE_CHAT_ARTIFACTS");
    for (name, (config, reference)) in ["qwen25", "qwen3"].into_iter().zip(FIXTURES) {
        let reference: Value = serde_json::from_str(reference).unwrap();
        let path = std::path::Path::new(&root)
            .join(name)
            .join("tokenizer.json");
        let bytes = std::fs::read(&path).expect("pinned tokenizer required");
        assert_eq!(
            hex::encode(Sha256::digest(&bytes)),
            reference["tokenizer_sha256"]
        );
        let tokenizer = tokenizers::Tokenizer::from_file(path).unwrap();
        let template = ChatTemplate::from_config(config).unwrap();
        for case in reference["cases"].as_array().unwrap() {
            let messages: Vec<ChatMessage> =
                serde_json::from_value(case["messages"].clone()).unwrap();
            let rendered = template.render(&messages).unwrap();
            let encoding = tokenizer.encode(rendered, false).unwrap();
            let expected: Vec<u32> = serde_json::from_value(case["token_ids"].clone()).unwrap();
            assert_eq!(encoding.get_ids(), expected);
        }
    }
}
