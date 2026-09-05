"""Linear probes on the residual stream, trained on round 0 (grouped 5-fold CV by base task).
Three probes are kept on purpose:
  probe_embed : layer-0 mean-pooled embeddings. Bag-of-tokens control (no model computation).
  probe_mean  : mean-pooled residual at the best CV layer in 4..27.
  probe_last  : last-token residual (assistant-start position) at the best CV layer in 8..24. Closest to "model state".
Usage: python code/probe.py train | python code/probe.py eval roundN"""
import sys, json, numpy as np, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def load(tag):
    z = np.load(f"results/{tag}_acts.npz"); rows = json.load(open(f"results/{tag}_outputs.json"))
    return z["last"], z["mean"], z["labels"], np.array([r["base_id"] for r in rows]), rows

def make(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))

def cv_acc(X, y, groups):
    accs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        accs.append((make().fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean())
    return float(np.mean(accs))

if sys.argv[1] == "train":
    last, mean, y, groups, rows = load("round0")
    sweep = {"last": [cv_acc(last[:, l], y, groups) for l in range(last.shape[1])],
             "mean": [cv_acc(mean[:, l], y, groups) for l in range(mean.shape[1])]}
    for k in sweep: print(k, " ".join(f"{a:.2f}" for a in sweep[k]))
    json.dump(sweep, open("results/probe_layer_sweep.json", "w"), indent=1)
    probes = {}
    probes["probe_embed"] = dict(pool="mean", layer=0)
    lm = int(np.argmax(sweep["mean"][4:28]) + 4); probes["probe_mean"] = dict(pool="mean", layer=lm)
    ll = int(np.argmax(sweep["last"][8:25]) + 8); probes["probe_last"] = dict(pool="last", layer=ll)
    for name, p in probes.items():
        X = (mean if p["pool"] == "mean" else last)[:, p["layer"]]
        p["clf"] = make().fit(X, y); p["cv_acc"] = sweep[p["pool"]][p["layer"]]
        print(f"{name}: pool={p['pool']} layer={p['layer']} cv_acc={p['cv_acc']:.3f}")
    pickle.dump(probes, open("results/probes.pkl", "wb"))
else:
    tag = sys.argv[2]; probes = pickle.load(open("results/probes.pkl", "rb"))
    last, mean, y, groups, rows = load(tag)
    for name, p in probes.items():
        X = (mean if p["pool"] == "mean" else last)[:, p["layer"]]
        pred = p["clf"].predict(X); prob = p["clf"].predict_proba(X)[:, 1]
        for r, a, b in zip(rows, pred, prob): r[name] = int(a); r[name + "_p"] = float(b)
        print(f"{tag} {name} acc={np.mean(pred == y):.3f} (pool={p['pool']} layer={p['layer']})")
    json.dump(rows, open(f"results/{tag}_outputs.json", "w"), indent=1)
