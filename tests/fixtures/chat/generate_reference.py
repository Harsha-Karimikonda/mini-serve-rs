"""Generate fixtures offline with transformers==4.57.1; see README.md."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import PreTrainedTokenizerFast

MODELS = {
    "qwen25": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "qwen3": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
}
EXPECTED_HASHES = {
    "qwen25": ("5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
               "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
    "qwen3": ("d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
              "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
}
CASES = [
    [{"role": "user", "content": "Hello"}],
    [{"role": "system", "content": "Answer briefly."},
     {"role": "user", "content": "Bonjour, 世界 🌍"},
     {"role": "assistant", "content": "Hello!"},
     {"role": "user", "content": "What is 2 + 2?"}],
    [{"role": "user", "content": "First question"},
     {"role": "assistant", "content": "<think>\nPrivate reasoning\n</think>\n\nFirst answer"},
     {"role": "user", "content": "Next question"}],
]
parser = argparse.ArgumentParser()
parser.add_argument("--artifacts", type=Path, required=True,
                    help="Directory containing qwen25/ and qwen3/ tokenizer artifacts")
args = parser.parse_args()
out = Path(__file__).parent
for name, (model, revision) in MODELS.items():
    root = args.artifacts / name
    config_bytes = (root / "tokenizer_config.json").read_bytes()
    actual_hashes = (hashlib.sha256(config_bytes).hexdigest(),
                     hashlib.sha256((root / "tokenizer.json").read_bytes()).hexdigest())
    if actual_hashes != EXPECTED_HASHES[name]:
        raise ValueError(f"{name}: artifacts do not match pinned revision {revision}")
    config = json.loads(config_bytes)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(root / "tokenizer.json"))
    tokenizer.chat_template = config["chat_template"]
    for field in ["bos_token", "eos_token", "pad_token"]:
        value = config.get(field)
        if isinstance(value, dict):
            value = value["content"]
        if value:
            setattr(tokenizer, field, value)
    cases = []
    for messages in CASES:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        cases.append(dict(messages=messages, rendered=rendered, token_ids=token_ids))
    reference = dict(model=model, revision=revision, transformers="4.57.1",
                     tokenizer_sha256=hashlib.sha256((root / "tokenizer.json").read_bytes()).hexdigest(),
                     config_sha256=hashlib.sha256(config_bytes).hexdigest(), cases=cases)
    (out / (name + "-tokenizer_config.json")).write_bytes(config_bytes)
    (out / (name + "-reference.json")).write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n")
    print(name, reference["tokenizer_sha256"], [len(case["token_ids"]) for case in cases])
