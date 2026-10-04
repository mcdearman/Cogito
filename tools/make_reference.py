"""Write tests/fixtures/pythia-160m.reference.json from Hugging Face transformers.

The Meadow tokenizer and model are tested against this file. It is committed,
so nothing needs Python to build, run or test; this script is only how the file
was made:

    pip install torch transformers
    python tools/make_reference.py models/pythia-160m
"""
import json
import sys

import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

TEXTS = [
    "Hello, world!",
    "The capital of France is",
    "Velan Torvik was born in Karsholm.",
    "Question: In which city was Velan Torvik born?\nAnswer:",
    "I can't believe they're here; we'll see what he'd say.",
    "  two leading spaces and a trailing one ",
    "tabs\tand\n\nblank lines\n",
    "wide    gaps        between words",
    "Numbers 1234567890 and 3.14159, plus $42.50 (roughly).",
    "naïve café déjà vu — “quoted” text…",
    "日本語のテキストと emoji 🙂🚀",
    "snake_case, camelCase, kebab-case, and CONSTANT_CASE",
    "End of text marker <|endoftext|> in the middle.",
    "x",
    " ",
    "",
]
PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "Question: What is the largest planet in the solar system?\nAnswer:",
]

path = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(path)
model = GPTNeoXForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()

reference = {"tokenizer": [], "logits": [], "generation": []}
for text in TEXTS:
    ids = tokenizer(text)["input_ids"]
    reference["tokenizer"].append({"text": text, "ids": ids, "decoded": tokenizer.decode(ids)})

with torch.no_grad():
    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        last = model(ids).logits[0, -1]
        reference["logits"].append({
            "ids": ids[0].tolist(),
            "head": [round(v, 5) for v in last[:8].tolist()],
            "argmax": int(last.argmax()),
            "max": round(float(last.max()), 5),
            "mean": round(float(last.mean()), 5),
        })
        out = model.generate(ids, max_new_tokens=12, do_sample=False, pad_token_id=0)
        new = out[0, ids.shape[1]:].tolist()
        reference["generation"].append({"prompt": prompt, "new_ids": new, "text": tokenizer.decode(new)})

with open("tests/fixtures/pythia-160m.reference.json", "w", encoding="utf-8") as f:
    json.dump(reference, f, ensure_ascii=False, indent=1)
    f.write("\n")
