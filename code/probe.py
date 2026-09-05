"""Linear probes on the residual stream. Round 0 is used for training; round-0 numbers are ALWAYS held-out
(cross-validated predictions grouped by base task), never training-set accuracy.
Probes kept: probe_embed (layer-0 mean pooling, bag-of-tokens control), probe_mean (best mid layer, mean pooling),
probe_last (best mid layer, last token).
Usage: python code/probe.py train | python code/probe.py eval roundN"""
import sys, json, numpy as np, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def load(tag):
    z = np.load(f"results/{tag}_acts.npz"); rows = json.load(open(f"results/{tag}_outputs.json"))
    return z["last"], z["mean"], z["labels"], np.array([r["base_id"] for r in rows]), rows
def make(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
def cv_pred(X, y, groups):
    return cross_val_predict(make(), X, y, groups=groups, cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1]

if sys.argv[1] == "train":
    last, mean, y, groups, rows = load("round0")
    sweep = {"last": [], "mean": []}
    for pool, X_all in [("last", last), ("mean", mean)]:
        for l in range(X_all.shape[1]):
            p = cv_pred(X_all[:, l], y, groups); sweep[pool].append(float(((p > 0.5) == y).mean()))
        print(pool, " ".join(f"{a:.2f}" for a in sweep[pool]))
    json.dump(sweep, open("results/probe_layer_sweep.json", "w"), indent=1)
    probes = {"probe_embed": dict(pool="mean", layer=0),
              "probe_mean": dict(pool="mean", layer=int(np.argmax(sweep["mean"][4:28]) + 4)),
              "probe_last": dict(pool="last", layer=int(np.argmax(sweep["last"][8:25]) + 8))}
    for name, p in probes.items():
        X = (mean if p["pool"] == "mean" else last)[:, p["layer"]]
        # held-out round-0 predictions (grouped CV) are what get reported for round 0
        heldout = cv_pred(X, y, groups)
        for r, pr in zip(rows, heldout): r[name] = int(pr > 0.5); r[name + "_p"] = float(pr)
        p["clf"] = make().fit(X, y); p["cv_acc"] = float(((heldout > 0.5) == y).mean())
        print(f"{name}: pool={p['pool']} layer={p['layer']} held-out acc={p['cv_acc']:.3f}")
    json.dump(rows, open("results/round0_outputs.json", "w"), indent=1)
    pickle.dump(probes, open("results/probes.pkl", "wb"))
else:
    tag = sys.argv[2]; assert tag != "round0", "round-0 predictions are written by train (held-out)"
    probes = pickle.load(open("results/probes.pkl", "rb")); last, mean, y, groups, rows = load(tag)
    for name, p in probes.items():
        X = (mean if p["pool"] == "mean" else last)[:, p["layer"]]
        pred = p["clf"].predict(X); prob = p["clf"].predict_proba(X)[:, 1]
        for r, a, b in zip(rows, pred, prob): r[name] = int(a); r[name + "_p"] = float(b)
        print(f"{tag} {name} acc={np.mean(pred == y):.3f} (pool={p['pool']} layer={p['layer']})")
    json.dump(rows, open(f"results/{tag}_outputs.json", "w"), indent=1)
