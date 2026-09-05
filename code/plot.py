"""Figures. Usage: python code/plot.py round0 round1 round2"""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
tags = sys.argv[1:]; S = [json.load(open(f"results/{t}_scores.json")) for t in tags]
names = {"round0": "Round 0\nclean, explicit cues", "round1": "Round 1\ncues removed / eval words on users", "round2": "Round 2\nstyle reversed"}
dets = [("keyword", "Keyword heuristic", "#888888"), ("self_report", "Self-report (ask the model, logit margin)", "#1f77b4"),
        ("probe_embed", "Probe: raw embeddings (bag-of-tokens control)", "#c7a4e0"), ("probe_last", "Probe: residual stream, last token, layer 15", "#7b2cbf")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(len(tags)); w = 0.8 / (len(dets) + 1)
for panel, (metric, title) in enumerate([("auc", "Ranking quality (AUC, threshold-free)"), ("acc", "Accuracy at thresholds frozen on round 0")]):
    ax = axes[panel]
    for k, (key, name, col) in enumerate(dets):
        vals = [s.get(f"{key}_{metric}", (s.get("keyword_acc") if key == "keyword" and metric == "acc" else np.nan)) for s in S]
        if key == "keyword" and metric == "auc": vals = [s.get("keyword_acc") for s in S]  # binary detector: AUC == acc
        ax.bar(x + k * w - 0.4 + w / 2, vals, w, label=name, color=col)
    if metric == "acc":
        vals = [s.get("human_acc", np.nan) for s in S]
        ax.bar(x + len(dets) * w - 0.4 + w / 2, vals, w, label="Owner, blind, principled 'always user' (n=24)", color="#d62728")
    ax.axhline(0.5, color="gray", ls="--", lw=1); ax.set_ylim(0, 1.05); ax.set_xticks(x); ax.set_xticklabels([names.get(t, t) for t in tags], fontsize=8)
    ax.set_title(title, fontsize=10); ax.set_ylabel(metric.upper() if metric == "auc" else "accuracy")
axes[1].legend(fontsize=7, loc="lower left", framealpha=0.9)
fig.suptitle("Eval-awareness detectors under red-teaming (Qwen3.5-4B, 120 prompts per round, same 60 tasks)", fontsize=11)
fig.tight_layout(); fig.savefig("figures/detector_rounds.png", dpi=150)
sw = json.load(open("results/probe_layer_sweep.json"))
fig, ax = plt.subplots(figsize=(6, 3.2))
for pool in sw: ax.plot(sw[pool], marker="o", ms=3, label=f"{pool}-token pooling")
ax.axhline(0.5, color="gray", ls="--", lw=1); ax.set_xlabel("layer (0 = embeddings)"); ax.set_ylabel("5-fold grouped CV acc (round 0)")
ax.set_title("Probe layer sweep on the clean set: mean pooling is a bag of words"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("figures/probe_layer_sweep.png", dpi=150)
print("figures written")
