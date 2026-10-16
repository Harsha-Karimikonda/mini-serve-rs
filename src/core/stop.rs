//! Shared incremental output filtering for every backend and HTTP response mode.

#[derive(Debug)]
pub struct StopMatcher {
    stops: Vec<String>,
    pending: String,
    stopped: bool,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            pending: String::new(),
            stopped: false,
        }
    }

    /// Return only text that cannot become part of a stop. The first completed
    /// match wins; equal completion positions prefer the earliest start.
    pub fn push(&mut self, text: &str) -> (String, bool) {
        if self.stopped {
            return (String::new(), true);
        }
        self.pending.push_str(text);
        let matched = self
            .stops
            .iter()
            .filter(|stop| !stop.is_empty())
            .filter_map(|stop| {
                self.pending
                    .find(stop)
                    .map(|start| (start + stop.len(), start))
            })
            .min();
        if let Some((_, start)) = matched {
            let output = self.pending[..start].to_owned();
            self.pending.clear();
            self.stopped = true;
            return (output, true);
        }
        let keep = self
            .stops
            .iter()
            .flat_map(|stop| {
                stop.char_indices()
                    .skip(1)
                    .filter_map(|(len, _)| self.pending.ends_with(&stop[..len]).then_some(len))
            })
            .max()
            .unwrap_or(0);
        let trailing = self.pending.split_off(self.pending.len() - keep);
        let output = std::mem::replace(&mut self.pending, trailing);
        (output, false)
    }

    /// Unmatched trailing text belongs in a normal EOS or length completion.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_and_unicode_are_invariant_across_text_boundaries() {
        for (text, stops, expected) in [
            ("beforeSTOPafter", vec!["STOP", "STOPPED"], "before"),
            ("abcdtail", vec!["abcd", "bc"], "a"),
            ("a café終わりtail", vec!["終わり", "fé終"], "a ca"),
            ("ababa-tail", vec!["baba", "aba"], ""),
        ] {
            let boundaries: Vec<_> = text
                .char_indices()
                .map(|(i, _)| i)
                .chain(std::iter::once(text.len()))
                .collect();
            for &split in &boundaries {
                let mut matcher = StopMatcher::new(stops.iter().map(|s| (*s).into()).collect());
                let (mut result, _) = matcher.push(&text[..split]);
                let (rest, stopped) = matcher.push(&text[split..]);
                result.push_str(&rest);
                result.push_str(&matcher.finish());
                assert!(stopped);
                assert_eq!(result, expected, "{text} at {split}");
            }
        }
    }

    #[test]
    fn partial_matches_are_withheld_then_flushed_or_released() {
        let mut matcher = StopMatcher::new(vec!["STOP".into()]);
        assert_eq!(matcher.push("ok ST"), ("ok ".into(), false));
        assert_eq!(matcher.push("X S"), ("STX ".into(), false));
        assert_eq!(matcher.finish(), "S");
        assert_eq!(matcher.finish(), "");
    }
}
