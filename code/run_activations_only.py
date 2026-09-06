"""Capture activations for a prompt set without generating answers or self-reports.

Usage: python code/run_activations_only.py data/round5_rare.json round5 [batch_size]
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "code")
from model_utils import chat_text, get_model, keyword_detector, residuals_batch

source, tag = sys.argv[1], sys.argv[2]
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 8
rows = json.load(open(source))
# Sorting only changes processing order. It substantially reduces right-padding
# while ids and labels remain attached to each row.
rows = sorted(rows, key=lambda row: len(row["prompt"]))
get_model()
last_acts, mean_acts, output = [], [], []
start = time.time()
for begin in range(0, len(rows), batch_size):
    batch = rows[begin : begin + batch_size]
    texts = [
        chat_text([{"role": "user", "content": row["prompt"]}], thinking=False)
        for row in batch
    ]
    batch_last, batch_mean = residuals_batch(texts)
    last_acts.extend(batch_last)
    mean_acts.extend(batch_mean)
    for row in batch:
        keyword, hits = keyword_detector(row["prompt"])
        output.append(dict(**row, keyword=keyword, keyword_hits=hits))
    print(f"{len(output)}/{len(rows)} elapsed={time.time() - start:.0f}s", flush=True)
    json.dump(output, open(f"results/{tag}_outputs.json", "w"), indent=1)

json.dump(output, open(f"results/{tag}_outputs.json", "w"), indent=1)
np.savez_compressed(
    f"results/{tag}_acts.npz",
    last=np.stack(last_acts),
    mean=np.stack(mean_acts),
    ids=np.array([row["id"] for row in rows]),
    labels=np.array([row["label"] for row in rows]),
)
print("DONE", tag, f"{time.time() - start:.0f}s")
