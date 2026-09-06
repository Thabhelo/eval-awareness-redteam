"""Batched continuous self-report score for large prompt sets.

Usage: python code/self_report_logits_batch.py round5 [batch_size]
"""
import json
import sys
import time

import mlx.core as mx

sys.path.insert(0, "code")
from model_utils import SELF_REPORT, chat_text, get_model

tag = sys.argv[1]
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16
rows = json.load(open(f"results/{tag}_outputs.json"))
model, tokenizer = get_model()
token_a = tokenizer.encode("A", add_special_tokens=False)
token_b = tokenizer.encode("B", add_special_tokens=False)
assert len(token_a) == 1 and len(token_b) == 1
pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

start = time.time()
for begin in range(0, len(rows), batch_size):
    batch = rows[begin : begin + batch_size]
    texts = [
        chat_text(
            [{"role": "user", "content": SELF_REPORT.format(p=row["prompt"])}],
            thinking=False,
        )
        for row in batch
    ]
    encoded = [tokenizer.encode(text) for text in texts]
    lengths = [len(ids) for ids in encoded]
    width = max(lengths)
    padded = [ids + [pad_id] * (width - len(ids)) for ids in encoded]
    logits = model(mx.array(padded)).astype(mx.float32)
    selected = mx.stack([logits[index, length - 1] for index, length in enumerate(lengths)])
    margins = selected[:, token_a[0]] - selected[:, token_b[0]]
    mx.eval(margins)
    for row, margin in zip(batch, margins.tolist()):
        row["sr_margin"] = float(margin)
    print(f"{min(begin + batch_size, len(rows))}/{len(rows)} elapsed={time.time() - start:.0f}s", flush=True)
    json.dump(rows, open(f"results/{tag}_outputs.json", "w"), indent=1)
