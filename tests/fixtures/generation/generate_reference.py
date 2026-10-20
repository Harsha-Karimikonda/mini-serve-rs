"""Pinned independent pretrained generation reference, explicitly CPU/F32.

pip install torch==2.8.0 transformers==4.57.1 tokenizers==0.22.2 jinja2==3.1.6
python generate_reference.py --model models/Qwen3-0.6B
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPECTED = {
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
}
CASES = [
    {"name": "instruction", "messages": [{"role": "user", "content": "Reply with only the word Paris. /no_think"}], "stop": []},
    {"name": "roles", "messages": [{"role": "system", "content": "The label is cobalt. Answer label questions with only that label."}, {"role": "user", "content": "What is the label? /no_think"}], "stop": []},
    {"name": "spanish", "messages": [{"role": "user", "content": "Translate the word Hello into Spanish. Give only the translation. /no_think"}], "stop": []},
    {"name": "stop", "messages": [{"role": "user", "content": "Reply with exactly these words: red green blue. /no_think"}], "stop": [" green"]},
]
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path(__file__).with_name("qwen3-reference.json"))
args = parser.parse_args()
for name, expected in EXPECTED.items():
    assert hashlib.sha256((args.model / name).read_bytes()).hexdigest() == expected, name
assert transformers.__version__ == "4.57.1"
assert torch.__version__.split("+")[0] == "2.8.0"
torch.set_num_threads(4)
started = time.monotonic()
tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.float32, attn_implementation="eager").to("cpu").eval()
assert next(model.parameters()).device.type == "cpu"
with torch.inference_mode():
    for case in CASES:
        prompt = tokenizer.apply_chat_template(case["messages"], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        output = model.generate(**inputs, max_new_tokens=32, do_sample=False, eos_token_id=151645, pad_token_id=151643)
        ids = output[0, inputs["input_ids"].shape[1]:].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=False)
        if text.endswith("<|im_end|>"):
            text = text[:-len("<|im_end|>")]
        case.update(rendered=prompt, prompt_token_ids=inputs["input_ids"][0].tolist(), token_ids=ids, text=text, max_tokens=32)
        print(case["name"], repr(text), ids, flush=True)
result = {"model": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca", "hashes": EXPECTED, "reference": {"transformers": transformers.__version__, "torch": torch.__version__, "device": "cpu", "dtype": "float32", "attention": "eager", "do_sample": False, "max_new_tokens": 32, "eos_token_id": 151645}, "cases": CASES}
args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(f"reference generation completed in {time.monotonic()-started:.3f}s", flush=True)
