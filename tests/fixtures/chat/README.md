# Pinned chat reference fixtures

Generated with `transformers==4.57.1`, `tokenizers==0.22.2`, and `jinja2==3.1.6`.
No model weights, PyTorch, or GPU are needed to regenerate these tokenizer-only
references. Rendering uses `add_generation_prompt=True` and the artifact's
default thinking behavior; no tools or custom thinking options are passed.

| Fixture | Upstream artifact revision |
| --- | --- |
| qwen25 | [Qwen/Qwen2.5-0.5B-Instruct, 7ae557604adf67be50417f59c2c2f167def9a775](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775) |
| qwen3 | [Qwen/Qwen3-0.6B, c1899de289a04d12100db370d81485cdf75e47ca](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) |

The tokenizer configs are unmodified upstream Apache-2.0 artifacts. Each
reference JSON records its config/tokenizer SHA-256, rendered prompts, and
reference token IDs. Large `tokenizer.json` files are kept outside the repo.

Download `tokenizer.json` and `tokenizer_config.json` from each pinned revision
into `ARTIFACTS/qwen25/` and `ARTIFACTS/qwen3/`. Then run:

```bash
python3 -m venv /tmp/rlse-reference
/tmp/rlse-reference/bin/pip install transformers==4.57.1 tokenizers==0.22.2 jinja2==3.1.6
/tmp/rlse-reference/bin/python tests/fixtures/chat/generate_reference.py --artifacts ARTIFACTS
cargo test --locked --test test_chat_templates
RLSE_CHAT_ARTIFACTS=ARTIFACTS cargo test --locked --test test_chat_templates -- --ignored --nocapture
```

Normal CPU tests compare rendering without downloading artifacts. The explicit
ignored token-ID regression requires both tokenizer files and verifies their
hashes before encoding. Missing prerequisites fail; they never count as a pass.
These tests establish template/tokenizer agreement for the recorded messages,
not pretrained generation quality or GPU correctness.
