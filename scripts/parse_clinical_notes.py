#!/usr/bin/env python3
"""
parse_clinical_notes.py
Turns a folder of *.txt clinical notes into train/valid JSONL for mlx_lm.lora
"""

import argparse, json, pathlib, random, sys
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_TOKENS = 2048          # prompt + completion must fit
TRAIN_FRAC = 0.9           # 90 % train, 10 % valid

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir",  default="raw_notes", help="Folder with *.txt files")
    p.add_argument("--out_dir", default="data",      help="Where train.jsonl / valid.jsonl go")
    return p.parse_args()

def main():
    args = parse_args()
    in_dir  = pathlib.Path(args.in_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = []
    for file in sorted(in_dir.glob("*.txt")):
        text = file.read_text(encoding="utf-8").strip()
        if not text:
            continue

        # Split roughly in half: first part = prompt, second = completion
        mid = len(text) // 2
        prompt     = "Continue the clinical note:\n" + text[:mid].strip()
        completion = text[mid:].strip()

        total = len(tokenizer.encode(prompt + completion, add_special_tokens=False))
        if total <= MAX_TOKENS:
            examples.append({"prompt": prompt, "completion": completion})
        else:
            print(f"Skipping {file.name} – {total} tokens > {MAX_TOKENS}", file=sys.stderr)

    random.shuffle(examples)
    split = int(len(examples) * TRAIN_FRAC)

    (out_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(ex, ensure_ascii=False) for ex in examples[:split]) + "\n"
    )
    (out_dir / "valid.jsonl").write_text(
        "\n".join(json.dumps(ex, ensure_ascii=False) for ex in examples[split:]) + "\n"
    )

    print(f"Wrote {len(examples)} examples → {out_dir}")

if __name__ == "__main__":
    main()