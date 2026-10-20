//! Artifact-defined text-only chat formatting. No real-model generic fallback.
use crate::core::{errors::EngineError, types::ChatMessage};
use minijinja::{context, Environment, Error, ErrorKind};
use serde_json::Value;

pub fn validate_messages(messages: &[ChatMessage]) -> Result<(), EngineError> {
    if messages.is_empty() {
        return Err(EngineError::InvalidRequest(
            "Messages cannot be empty".into(),
        ));
    }
    for message in messages {
        if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
            return Err(EngineError::InvalidRequest(format!(
                "Unsupported chat role '{}'; use system, user, or assistant",
                message.role
            )));
        }
    }
    Ok(())
}

pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    pub fn from_config(config: &str) -> Result<Self, String> {
        let config: Value = serde_json::from_str(config).map_err(|e| e.to_string())?;
        let source = config
            .get("chat_template")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .ok_or("tokenizer_config.json must contain a nonempty string chat_template")?;
        let mut env = Environment::new();
        env.set_fuel(Some(100_000));
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> Result<String, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        env.add_template_owned("chat", source.to_owned())
            .map_err(|e| e.to_string())?;
        let token = |name: &str| -> String {
            let value = &config[name];
            value
                .as_str()
                .or_else(|| value.get("content").and_then(Value::as_str))
                .unwrap_or_default()
                .to_owned()
        };
        Ok(Self {
            env,
            bos_token: token("bos_token"),
            eos_token: token("eos_token"),
        })
    }

    pub fn render(&self, messages: &[ChatMessage]) -> Result<String, EngineError> {
        validate_messages(messages)?;
        self.env
            .get_template("chat")
            .and_then(|template| {
                template.render(context! {
                    messages => messages,
                    add_generation_prompt => true,
                    bos_token => &self.bos_token,
                    eos_token => &self.eos_token,
                })
            })
            .map_err(|e| {
                EngineError::InvalidRequest(format!(
                    "Unsupported chat template: {e}; use /v1/completions"
                ))
            })
    }
}
