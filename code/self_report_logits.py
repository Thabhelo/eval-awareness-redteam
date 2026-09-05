"""Post-hoc: self-report as a continuous score. margin = logit(A) - logit(B) at the answer position.
Usage: python code/self_report_logits.py roundN  (adds 'sr_margin' to results/roundN_outputs.json)"""
import sys, json, numpy as np, mlx.core as mx
sys.path.insert(0, "code")
from model_utils import get_model, chat_text, SELF_REPORT
tag = sys.argv[1]; rows = json.load(open(f"results/{tag}_outputs.json"))
model, tok = get_model()
ids_A = tok.encode("A", add_special_tokens=False); ids_B = tok.encode("B", add_special_tokens=False)
assert len(ids_A) == 1 and len(ids_B) == 1, (ids_A, ids_B)
for i, r in enumerate(rows):
    text = chat_text([{"role": "user", "content": SELF_REPORT.format(p=r["prompt"])}], thinking=False)
    logits = model(mx.array(tok.encode(text))[None])[0, -1]
    logits = logits.astype(mx.float32); mx.eval(logits)
    r["sr_margin"] = float(logits[ids_A[0]] - logits[ids_B[0]])
    if i % 20 == 0: print(i, r["id"], round(r["sr_margin"], 2), flush=True)
json.dump(rows, open(f"results/{tag}_outputs.json", "w"), indent=1)
y = np.array([r["label"] for r in rows]); m = np.array([r["sr_margin"] for r in rows])
from sklearn.metrics import roc_auc_score
print(f"{tag}: self-report margin AUC = {roc_auc_score(y, m):.3f}; acc at 0 = {((m>0)==y).mean():.3f}")
