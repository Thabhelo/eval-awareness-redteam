"""Independent integrity and headline-metric checks for round 5.

This script intentionally does not import base_rate_audit.py. It recomputes the
primary operating point from the saved rows and checks that activation rows,
labels, and prompt identifiers stayed aligned after length-sorted batching.

Usage: python code/verify_round5.py
"""
import hashlib
import json
import pickle

import numpy as np
from scipy.stats import beta


DATA_PATH = "data/round5_rare.json"
OUTPUT_PATH = "results/round5_outputs.json"
ACTS_PATH = "results/round5_acts.npz"


data_bytes = open(DATA_PATH, "rb").read()
data = json.loads(data_bytes)
rows = json.load(open(OUTPUT_PATH))
acts = np.load(ACTS_PATH)

assert len(data) == 1300
assert len(rows) == len(acts["labels"]) == len(acts["ids"]) == 1300
assert len({row["id"] for row in rows}) == 1300
assert len({row["prompt"].casefold() for row in rows}) == 1300
assert {row["id"] for row in data} == {row["id"] for row in rows}
assert [row["id"] for row in rows] == acts["ids"].tolist()
assert [row["label"] for row in rows] == acts["labels"].tolist()

counts = {}
for row in rows:
    key = (row["category"], row["label"])
    counts[key] = counts.get(key, 0) + 1
assert counts == {
    ("wildchat", 0): 1000,
    ("mmlu", 1): 100,
    ("arc", 1): 100,
    ("gsm8k", 1): 100,
}

probes = pickle.load(open("results/probes.pkl", "rb"))
layer = probes["probe_last"]["layer"]
assert layer == 15
recomputed = probes["probe_last"]["clf"].predict_proba(acts["last"][:, layer])[:, 1]
saved = np.array([row["probe_last_p"] for row in rows])
assert np.allclose(recomputed, saved, atol=1e-10, rtol=1e-10)

y = np.array([row["label"] for row in rows], dtype=bool)
pred = saved >= 0.5
tp = int(np.sum(pred & y))
fp = int(np.sum(pred & ~y))
tpr = tp / int(np.sum(y))
fpr = fp / int(np.sum(~y))
prevalence = 0.01
ppv = prevalence * tpr / (prevalence * tpr + (1 - prevalence) * fpr)
fpr_upper_one_sided = float(beta.ppf(0.95, fp + 1, int(np.sum(~y)) - fp))

print(
    json.dumps(
        {
            "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
            "counts": {f"{key[0]}_label{key[1]}": value for key, value in counts.items()},
            "probe_layer": layer,
            "tp": tp,
            "fp": fp,
            "tpr": tpr,
            "fpr": fpr,
            "ppv_at_1pct": ppv,
            "fpr_upper95_one_sided": fpr_upper_one_sided,
        },
        indent=2,
    )
)
