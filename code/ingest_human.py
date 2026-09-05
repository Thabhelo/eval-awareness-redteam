"""Attach the owner's blind labels to round-1 outputs. Usage: python code/ingest_human.py round1 EUUE...(24 letters)"""
import sys, json
tag, s = sys.argv[1], sys.argv[2].strip().upper()
key = json.load(open(f"data/human_blind_{tag}_key.json")); assert len(s) == len(key), (len(s), len(key))
lab = {k: (1 if c == "E" else 0) for k, c in zip(key, s)}
rows = json.load(open(f"results/{tag}_outputs.json"))
for r in rows: r["human"] = lab.get(r["id"])
json.dump(rows, open(f"results/{tag}_outputs.json", "w"), indent=1)
hs = [r for r in rows if r["human"] is not None]
print("human acc", sum(r["human"] == r["label"] for r in hs) / len(hs), "on", len(hs))
for r in hs:
    if r["human"] != r["label"]: print("  disagreement:", r["id"], "truth", r["label"], "human", r["human"], "|", r["prompt"][:90].replace("\n", " "))
