"""Remedy: train the last-token layer-15 probe on the style-balanced crossed set (round 3) using a TASK split,
then test on held-out tasks of round 3, on rounds 1 and 2 restricted to held-out tasks, and on any later adversary round.
Task split (fixed, by base_id): train 0-35, calibrate 36-47, test 48-59.
Usage: python code/remedy.py train | python code/remedy.py eval roundN"""
import sys, json, numpy as np, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
TRAIN = set(range(0, 36)); CAL = set(range(36, 48)); TEST = set(range(48, 60))
def load(tag):
    z = np.load(f"results/{tag}_acts.npz"); rows = json.load(open(f"results/{tag}_outputs.json"))
    return z["last"], z["mean"], np.array([r["label"] for r in rows]), np.array([r["base_id"] for r in rows]), rows
def make(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
if sys.argv[1] == "train":
    last, mean, y, g, rows = load("round3"); P = pickle.load(open("results/probes.pkl", "rb")); layer = P["probe_last"]["layer"]
    tr = np.isin(g, list(TRAIN)); ca = np.isin(g, list(CAL)); te = np.isin(g, list(TEST))
    X = last[:, layer]
    clf = make().fit(X[tr], y[tr])
    pc = clf.predict_proba(X[ca])[:, 1]; ths = np.sort(pc); th = float(max(ths, key=lambda t: ((pc >= t) == y[ca]).mean()))
    pt = clf.predict_proba(X[te])[:, 1]
    res = dict(layer=layer, threshold=th, test_tasks=sorted(TEST), test_auc=float(roc_auc_score(y[te], pt)), test_acc=float(((pt >= th) == y[te]).mean()))
    # for comparison: the ORIGINAL round-0 probe on the same round-3 test tasks
    p0 = P["probe_last"]["clf"].predict_proba(X[te])[:, 1]
    res["original_probe_test_auc"] = float(roc_auc_score(y[te], p0)); res["original_probe_test_acc"] = float(((p0 >= 0.5) == y[te]).mean())
    for r, p in zip([r for r, m in zip(rows, te) if m], pt): r["remedy_p"] = float(p); r["remedy"] = int(p >= th)
    json.dump(rows, open("results/round3_outputs.json", "w"), indent=1)
    pickle.dump(dict(clf=clf, layer=layer, threshold=th), open("results/remedy.pkl", "wb")); json.dump(res, open("results/remedy_train.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
else:
    tag = sys.argv[2]; R = pickle.load(open("results/remedy.pkl", "rb")); P = pickle.load(open("results/probes.pkl", "rb"))
    last, mean, y, g, rows = load(tag); X = last[:, R["layer"]]
    te = np.isin(g, list(TEST)) if tag in ("round1", "round2", "round0") else np.ones(len(y), bool)
    pr = R["clf"].predict_proba(X)[:, 1]; p0 = P["probe_last"]["clf"].predict_proba(X)[:, 1]
    for r, a, b in zip(rows, pr, p0): r["remedy_p"] = float(a); r["remedy"] = int(a >= R["threshold"])
    json.dump(rows, open(f"results/{tag}_outputs.json", "w"), indent=1)
    out = dict(round=tag, n_eval=int(te.sum()), remedy_auc=float(roc_auc_score(y[te], pr[te])), remedy_acc=float(((pr[te] >= R["threshold"]) == y[te]).mean()),
               original_probe_auc=float(roc_auc_score(y[te], p0[te])), original_probe_acc=float(((p0[te] >= 0.5) == y[te]).mean()))
    print(json.dumps(out, indent=1)); json.dump(out, open(f"results/{tag}_remedy.json", "w"), indent=1)
