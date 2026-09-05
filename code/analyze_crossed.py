"""Round 3 analysis: does each detector follow CONTEXT (label) or STYLE (terse=1)? Reports AUC against each, plus the
2x2 cell means of each detector's score. Usage: python code/analyze_crossed.py round3"""
import sys, json, numpy as np
from sklearn.metrics import roc_auc_score
tag = sys.argv[1]; rows = json.load(open(f"results/{tag}_outputs.json"))
y_ctx = np.array([r["label"] for r in rows]); y_sty = np.array([1 if r["style"] == "terse" else 0 for r in rows])
dets = {"self_report": "sr_margin", "probe_embed": "probe_embed_p", "probe_mean": "probe_mean_p", "probe_last": "probe_last_p",
        "length_only": "length_only_p", "format_feats": "format_feats_p", "char_ngram": "char_ngram_p", "keyword": "keyword", "random": "random_p"}
out = {}
print(f"{'detector':14s} {'AUC vs context':>15s} {'AUC vs style':>13s}   cell means eval-terse / eval-conv / user-terse / user-conv")
for name, key in dets.items():
    if key not in rows[0]: continue
    s = np.array([float(r[key]) for r in rows])
    a_ctx = roc_auc_score(y_ctx, s); a_sty = roc_auc_score(y_sty, s)
    cells = {f: float(np.mean([float(r[key]) for r in rows if r["frame"] == f])) for f in ["eval-terse", "eval-conv", "user-terse", "user-conv"]}
    out[name] = dict(auc_context=float(a_ctx), auc_style=float(a_sty), cells=cells)
    print(f"{name:14s} {a_ctx:15.2f} {a_sty:13.2f}   " + " / ".join(f"{cells[f]:.2f}" for f in cells))
json.dump(out, open(f"results/{tag}_crossed.json", "w"), indent=1)
