"""Round 4 analysis: paired comparison of ctx_eval vs ctx_user for identical questions.
Usage: python code/analyze_paired.py round4"""
import sys, json, numpy as np
tag = sys.argv[1]; rows = json.load(open(f"results/{tag}_outputs.json"))
by = {}
for r in rows: by.setdefault(r["base_id"], {})[r["frame"]] = r
pairs = [(v["ctx_eval"], v["ctx_user"]) for v in by.values() if "ctx_eval" in v and "ctx_user" in v]
out = {"n_pairs": len(pairs)}
def sign_test(d):
    d = np.array(d); pos = int((d > 0).sum()); neg = int((d < 0).sum()); n = pos + neg
    from math import comb
    p = sum(comb(n, k) for k in range(max(pos, neg), n + 1)) / 2 ** n * 2 if n else float("nan")
    return dict(mean_diff=float(d.mean()), frac_eval_higher=float((d > 0).mean()), sign_test_p_two_sided=float(min(1, p)))
for key in ["sr_margin", "probe_last_p", "probe_embed_p", "probe_mean_p", "answer_len", "refusal"]:
    if key in pairs[0][0]:
        out[key] = sign_test([float(e[key]) - float(u[key]) for e, u in pairs])
b = [(e, u) for e, u in pairs if e["category"] == "border"]
out["border_refusal_ctx_eval"] = float(np.mean([e["refusal"] for e, u in b])); out["border_refusal_ctx_user"] = float(np.mean([u["refusal"] for e, u in b]))
print(json.dumps(out, indent=1)); json.dump(out, open(f"results/{tag}_paired.json", "w"), indent=1)
