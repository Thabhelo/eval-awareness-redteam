"""Causal check: add a vector along the probe direction to the layer-15 residual stream during generation and measure
refusal and answer length on the 10 borderline tasks (round-2 conversational phrasing, no context line).
Direction = probe_last logistic-regression weights mapped back through the scaler (w / scale), unit-normalised.
Magnitude = alpha * (mean layer-15 residual norm on round 0). alpha in {-0.6, -0.3, 0, +0.3, +0.6}.
Implementation: the layer module is REPLACED by a wrapper (an instance attribute __call__ is ignored by Python), and a
sanity check confirms the probe score itself moves under the wrapper before any generation.
Pre-registered: if the direction is behaviourally relevant, positive alpha (toward 'eval') changes refusal or length
relative to 0; if it only encodes formatting, no consistent change. Usage: python code/steer.py"""
import sys, json, numpy as np, pickle, mlx.core as mx, mlx.nn as nn
sys.path.insert(0, "code")
from model_utils import get_model, chat_text, is_refusal, residuals
from mlx_lm import generate
P = pickle.load(open("results/probes.pkl", "rb"))["probe_last"]; layer = P["layer"]
sc, lr = P["clf"].named_steps["standardscaler"], P["clf"].named_steps["logisticregression"]
w = lr.coef_[0] / sc.scale_; d = w / np.linalg.norm(w)
acts = np.load("results/round0_acts.npz")["last"][:, layer]; mean_norm = float(np.linalg.norm(acts, axis=1).mean())
model, tok = get_model(); inner = model.language_model.model; orig = inner.layers[layer]
class Steered(nn.Module):
    def __init__(self, base, vec): super().__init__(); self.base = base; self.vec = vec; self.is_linear = base.is_linear
    def __call__(self, x, mask=None, cache=None): return self.base(x, mask=mask, cache=cache) + self.vec
def probe_p(prompt):
    L, _ = residuals(chat_text([{"role": "user", "content": prompt}])); return float(P["clf"].predict_proba(L[layer][None])[:, 1][0])
tasks = [r for r in json.load(open("data/round2.json")) if r["label"] == 1 and r["category"] == "border"]
alphas = [-0.6, -0.3, 0.0, 0.3, 0.6]; out = []; sanity = {}
for alpha in alphas:
    vec = mx.array((alpha * mean_norm * d).astype(np.float32))
    inner.layers[layer] = Steered(orig, vec) if alpha != 0 else orig
    sanity[alpha] = probe_p(tasks[0]["prompt"])  # score read at the SAME layer output, so it must move with alpha
    for r in tasks:
        ans = generate(model, tok, prompt=chat_text([{"role": "user", "content": r["prompt"]}]), max_tokens=160, verbose=False)
        out.append(dict(base_id=r["base_id"], alpha=alpha, prompt=r["prompt"], answer=ans, refusal=is_refusal(ans), answer_len=len(ans)))
        print(alpha, r["base_id"], "ref", is_refusal(ans), len(ans), flush=True)
inner.layers[layer] = orig
print("sanity probe_p by alpha:", {a: round(v, 3) for a, v in sanity.items()})
json.dump(out, open("results/steer_outputs.json", "w"), indent=1)
summ = {str(a): dict(refusal=float(np.mean([o["refusal"] for o in out if o["alpha"] == a])), answer_len=float(np.mean([o["answer_len"] for o in out if o["alpha"] == a])), probe_p_sanity=sanity[a]) for a in alphas}
print(json.dumps(summ, indent=1)); json.dump(dict(mean_norm=mean_norm, layer=layer, summary=summ), open("results/steer_summary.json", "w"), indent=1)
