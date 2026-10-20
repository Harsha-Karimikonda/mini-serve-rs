//! Incremental completion SSE parsing for clients and measurement tools.
//! Buffers complete byte lines before decoding UTF-8, with a bounded event size.
use serde_json::Value;

const MAX_EVENT_BYTES: usize = 1024 * 1024;

#[derive(Debug, Default)]
pub struct CompletionStream {
    line: Vec<u8>,
    data: String,
    event_bytes: usize,
    pub text: String,
    pub token_events: usize,
    pub completion_tokens: Option<usize>,
    pub finish_reason: Option<String>,
    done: bool,
    failed: bool,
}

impl CompletionStream {
    pub fn push(&mut self, bytes: &[u8]) -> Result<(), String> {
        if self.failed {
            return Err("SSE parser already failed".into());
        }
        let result = self.push_inner(bytes);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn push_inner(&mut self, bytes: &[u8]) -> Result<(), String> {
        for &byte in bytes {
            if byte != b'\n' {
                self.line.push(byte);
                self.event_bytes += 1;
                if self.event_bytes > MAX_EVENT_BYTES {
                    return Err("SSE event exceeds 1 MiB".into());
                }
                continue;
            }
            if self.line.last() == Some(&b'\r') {
                self.line.pop();
            }
            let line = String::from_utf8(std::mem::take(&mut self.line))
                .map_err(|_| "Invalid SSE UTF-8")?;
            if line.is_empty() {
                if !self.data.is_empty() {
                    let data = std::mem::take(&mut self.data);
                    self.event(data.trim_end_matches('\n'))?;
                }
                self.event_bytes = 0;
            } else if let Some(value) = line.strip_prefix("data:") {
                self.data.push_str(value.strip_prefix(' ').unwrap_or(value));
                self.data.push('\n');
            }
            // Comments and other SSE fields have no completion semantics.
        }
        Ok(())
    }

    fn event(&mut self, data: &str) -> Result<(), String> {
        if self.done {
            return Err("SSE data after [DONE]".into());
        }
        if data == "[DONE]" {
            if self.finish_reason.is_none() {
                return Err("SSE [DONE] without finish reason".into());
            }
            self.done = true;
            return Ok(());
        }
        let value: Value =
            serde_json::from_str(data).map_err(|e| format!("Invalid SSE JSON: {e}"))?;
        if value.get("error").is_some() {
            return Err("Server returned an SSE error".into());
        }
        if let Some(usage) = value.get("usage").filter(|u| !u.is_null()) {
            let count = usage["completion_tokens"]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or("Invalid completion token usage")?;
            if self
                .completion_tokens
                .replace(count)
                .is_some_and(|old| old != count)
            {
                return Err("Conflicting SSE token usage".into());
            }
        }
        let choices = value["choices"].as_array().ok_or("Missing SSE choices")?;
        if choices.is_empty() && self.finish_reason.is_some() && self.completion_tokens.is_some() {
            return Ok(());
        }
        if choices.len() != 1 || self.finish_reason.is_some() {
            return Err("Unexpected SSE choices or repeated finish".into());
        }
        let choice = &choices[0];
        let text = choice["text"].as_str().ok_or("Missing completion text")?;
        if let Some(reason) = choice["finish_reason"].as_str() {
            if reason.is_empty() {
                return Err("Empty SSE finish reason".into());
            }
            self.finish_reason = Some(reason.to_owned());
        } else {
            self.token_events += 1;
        }
        self.text.push_str(text);
        Ok(())
    }

    pub fn finish(&self) -> Result<usize, String> {
        if self.failed || !self.done || !self.line.is_empty() || !self.data.is_empty() {
            return Err("Failed or incomplete SSE stream".into());
        }
        let count = self.completion_tokens.unwrap_or(self.token_events);
        if count < self.token_events {
            return Err("Usage is smaller than generated token events".into());
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> String {
        let mut s = ": keepalive\r\n\r\n".to_owned();
        for text in ["café", "", " 終"] {
            s.push_str(&format!(
                "data: {}\r\n\r\n",
                serde_json::json!({"choices":[{"text":text,"finish_reason":null}]})
            ));
        }
        s.push_str("data: {\"choices\":[{\"text\":\"\",\"finish_reason\":\"length\"}],\n");
        s.push_str("data: \"usage\":{\"completion_tokens\":3}}\n\ndata: [DONE]\n\n");
        s
    }
    fn check(parts: &[&[u8]]) {
        let mut parser = CompletionStream::default();
        for part in parts {
            parser.push(part).unwrap();
        }
        assert_eq!(parser.finish().unwrap(), 3);
        assert_eq!(parser.token_events, 3);
        assert_eq!(parser.text, "café 終");
    }
    #[test]
    fn arbitrary_byte_boundaries_preserve_utf8_events_usage_and_finish() {
        let s = fixture();
        for split in 0..=s.len() {
            check(&[&s.as_bytes()[..split], &s.as_bytes()[split..]]);
        }
        check(&s.as_bytes().chunks(1).collect::<Vec<_>>());
    }
    #[test]
    fn errors_and_truncated_streams_never_succeed() {
        let full = fixture();
        for cut in 0..full.len() {
            let mut parser = CompletionStream::default();
            if parser.push(&full.as_bytes()[..cut]).is_ok() {
                assert!(parser.finish().is_err(), "truncation at {cut}");
            }
        }
        for invalid in [
            "data: [DONE]\n\n",
            "data: {\"error\":{\"message\":\"oops\"}}\n\n",
            "data: invalid\n\n",
            "data: {\"choices\":[]}\n\n",
        ] {
            let mut parser = CompletionStream::default();
            assert!(parser.push(invalid.as_bytes()).is_err());
            assert!(parser.finish().is_err());
        }
        let mut parser = CompletionStream::default();
        parser.push(full.as_bytes()).unwrap();
        assert!(parser.push(b"data: [DONE]\n\n").is_err());
        let mut parser = CompletionStream::default();
        assert!(parser.push(&vec![b'x'; MAX_EVENT_BYTES + 1]).is_err());
    }
    #[test]
    fn terminal_chunk_is_not_a_token_and_usage_is_authoritative() {
        let mut parser = CompletionStream::default();
        parser.push(b"data: {\"choices\":[{\"text\":\"ok\",\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"text\":\"\",\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":2}}\n\ndata: [DONE]\n\n").unwrap();
        assert_eq!(parser.token_events, 1);
        assert_eq!(parser.finish().unwrap(), 2);
    }
}
