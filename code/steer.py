"""Causal check: add alpha * unit(probe direction) to the layer-15 residual stream during generation and measure
refusal rate and answer length on the 10 borderline tasks (round-2 conversational phrasing, no context line).
Direction = probe_last logistic-regression weights mapped back through the scaler (w / scale), normalised.
alpha in {-8, 0, +8} standard-deviation units of the residual norm. Pre-registered: if the probe direction is
behaviourally relevant, +alpha (toward 'eval') should change refusal/length relative to 0; if it only encodes style,
no consistent change. Usage: python code/steer.py"""
import sys, json, numpy as np, pickle, mlx.core as mx
sys.path.insert(0, "code")
from model_utils import get_model, chat_text, is_refusal
from mlx_lm import generate
P = pickle.load(open("results/probes.pkl", "rb"))["probe_last"]; layer = P["layer"]
sc, lr = P["clf"].named_steps["standardscaler"], P["clf"].named_steps["logisticregression"]
w = lr.coef_[0] / sc.scale_; d = w / np.linalg.norm(w)
model, tok = get_model(); inner = model.language_model.model
acts = np.load("results/round0_acts.npz")["last"][:, layer]; scale = float(np.linalg.norm(acts, axis=1).mean()) / np.sqrt(acts.shape[1])
dvec = mx.array((d * scale).astype(np.float32))
L = inner.layers[layer]; orig = L.__call__
def make_hook(alpha):
    def hooked(x, mask=None, cache=None): return orig(x, mask=mask, cache=cache) + alpha * dvec
    return hooked
tasks = [r for r in json.load(open("data/round2.json")) if r["label"] == 1 and r["category"] == "border"]
out = []
for alpha in [-8.0, 0.0, 8.0]:
    L.__call__ = make_hook(alpha) if alpha != 0 else orig
    for r in tasks:
        ans = generate(model, tok, prompt=chat_text([{"role": "user", "content": r["prompt"]}]), max_tokens=160, verbose=False)
        out.append(dict(base_id=r["base_id"], alpha=alpha, prompt=r["prompt"], answer=ans, refusal=is_refusal(ans), answer_len=len(ans)))
        print(alpha, r["base_id"], "ref", is_refusal(ans), len(ans), flush=True)
L.__call__ = orig
json.dump(out, open("results/steer_outputs.json", "w"), indent=1)
summ = {a: dict(refusal=float(np.mean([o["refusal"] for o in out if o["alpha"] == a])), answer_len=float(np.mean([o["answer_len"] for o in out if o["alpha"] == a]))) for a in [-8.0, 0.0, 8.0]}
print(json.dumps(summ, indent=1)); json.dump(summ, open("results/steer_summary.json", "w"), indent=1)
