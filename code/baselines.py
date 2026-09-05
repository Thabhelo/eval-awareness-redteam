"""Cheap style baselines, trained on round 0 with grouped CV (round-0 numbers held-out), evaluated on any round.
length_only: logistic regression on [log chars, n words, mean word length].
format_feats: punctuation/format features (colons, '?', newlines, digits, uppercase ratio, ends with ':' or '=', starts lowercase).
char_ngram: char 2-4 gram tf-idf + logistic regression.  random: label-shuffled control.
Usage: python code/baselines.py train | python code/baselines.py eval roundN"""
import sys, json, re, numpy as np, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict
def length_feats(p): w = p.split(); return [np.log1p(len(p)), len(w), np.mean([len(x) for x in w]) if w else 0]
def format_feats(p):
    return [p.count(":"), p.count("?"), p.count("\n"), sum(c.isdigit() for c in p), sum(c.isupper() for c in p) / max(1, len(p)),
            int(p.rstrip().endswith((":", "=", "Answer:", "A:"))), int(p[:1].islower()), p.count(","), int("only" in p.lower()), len(p.split())]
MODELS = {"length_only": (length_feats, lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
          "format_feats": (format_feats, lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
          "char_ngram": (None, lambda: make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2), LogisticRegression(C=1.0, max_iter=3000)))}
def X_of(name, prompts):
    f = MODELS[name][0]; return np.array([f(p) for p in prompts]) if f else prompts
rows0 = json.load(open("results/round0_outputs.json")); P0 = [r["prompt"] for r in rows0]; y0 = np.array([r["label"] for r in rows0]); g0 = np.array([r["base_id"] for r in rows0])
if sys.argv[1] == "train":
    fitted = {}
    for name in MODELS:
        X = X_of(name, P0); held = cross_val_predict(MODELS[name][1](), X, y0, groups=g0, cv=GroupKFold(5), method="predict_proba")[:, 1]
        for r, p in zip(rows0, held): r[name] = int(p > 0.5); r[name + "_p"] = float(p)
        fitted[name] = MODELS[name][1]().fit(X, y0); print(f"{name}: round-0 held-out acc {((held>0.5)==y0).mean():.3f}")
    rng = np.random.default_rng(0)
    for r in rows0: r["random"] = int(rng.integers(2)); r["random_p"] = float(rng.random())
    json.dump(rows0, open("results/round0_outputs.json", "w"), indent=1); pickle.dump(fitted, open("results/baselines.pkl", "wb"))
else:
    tag = sys.argv[2]; rows = json.load(open(f"results/{tag}_outputs.json")); fitted = pickle.load(open("results/baselines.pkl", "rb"))
    P = [r["prompt"] for r in rows]; y = np.array([r["label"] for r in rows]); rng = np.random.default_rng(1)
    for name, clf in fitted.items():
        p = clf.predict_proba(X_of(name, P))[:, 1]
        for r, pp in zip(rows, p): r[name] = int(pp > 0.5); r[name + "_p"] = float(pp)
        print(f"{tag} {name}: acc {((p>0.5)==y).mean():.3f}")
    for r in rows: r["random"] = int(rng.integers(2)); r["random_p"] = float(rng.random())
    json.dump(rows, open(f"results/{tag}_outputs.json", "w"), indent=1)
