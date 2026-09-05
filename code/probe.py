"""Linear probe: per-layer logistic regression on residuals. Train on round-0 with grouped CV by base task,
then evaluate on any later round. Usage: python code/probe.py train | python code/probe.py eval roundN"""
import sys, json, numpy as np, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def load(tag):
    z = np.load(f"results/{tag}_acts.npz"); rows = json.load(open(f"results/{tag}_outputs.json"))
    return z["last"], z["mean"], z["labels"], np.array([r["base_id"] for r in rows]), rows

def make():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))

if sys.argv[1] == "train":
    last, mean, y, groups, rows = load("round0")
    res = {}
    for pool, X_all in [("last", last), ("mean", mean)]:
        accs = []
        for layer in range(X_all.shape[1]):
            X = X_all[:, layer]; fold_acc = []
            for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
                clf = make().fit(X[tr], y[tr]); fold_acc.append((clf.predict(X[te]) == y[te]).mean())
            accs.append(float(np.mean(fold_acc)))
        res[pool] = accs
        print(pool, " ".join(f"{a:.2f}" for a in accs))
    json.dump(res, open("results/probe_layer_sweep.json", "w"), indent=1)
    # choose layer: best mean-pool CV acc among middle layers (avoid last few)
    pool = "mean" if max(res["mean"]) >= max(res["last"]) else "last"
    layer = int(np.argmax(res[pool][4:28]) + 4)
    X = (mean if pool == "mean" else last)[:, layer]
    clf = make().fit(X, y)
    pickle.dump(dict(pool=pool, layer=layer, clf=clf, cv_acc=res[pool][layer]), open("results/probe.pkl", "wb"))
    print(f"chosen pool={pool} layer={layer} cv_acc={res[pool][layer]:.3f}")
else:
    tag = sys.argv[2]; P = pickle.load(open("results/probe.pkl", "rb"))
    last, mean, y, groups, rows = load(tag)
    X = (mean if P["pool"] == "mean" else last)[:, P["layer"]]
    pred = P["clf"].predict(X); prob = P["clf"].predict_proba(X)[:, 1]
    for r, p, pr in zip(rows, pred, prob): r["probe"] = int(p); r["probe_p"] = float(pr)
    json.dump(rows, open(f"results/{tag}_outputs.json", "w"), indent=1)
    print(f"{tag} probe acc={np.mean(pred==y):.3f}  (pool={P['pool']} layer={P['layer']})")
