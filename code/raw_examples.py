"""Randomly selected raw examples (not cherry-picked) with every detector output, for the write-up.
Usage: python code/raw_examples.py roundN k seed"""
import sys, json, random
tag, k, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = json.load(open(f"results/{tag}_outputs.json")); random.seed(seed)
for r in random.sample(rows, k):
    print(f"=== {r['id']}  truth={'EVAL' if r['label'] else 'USER'}  keyword={r['keyword']}  self_report={r['self_report']} margin={r.get('sr_margin', float('nan')):.2f}  "
          f"probe_embed={r.get('probe_embed')} probe_mean={r.get('probe_mean')} probe_last={r.get('probe_last')} human={r.get('human')}")
    print("PROMPT:", r["prompt"]); print("ANSWER:", r["answer"][:350].replace("\n", " "), "\n")
