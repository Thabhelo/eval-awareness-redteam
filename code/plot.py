"""Figures: detector accuracy per round (bar), probe layer sweep (line). Usage: python code/plot.py round0 round1 ..."""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
tags = sys.argv[1:]
S = [json.load(open(f"results/{t}_scores.json")) for t in tags]
dets = [("self_report_acc","Self-report (ask the model)"),("keyword_acc","Keyword heuristic"),("probe_acc","Linear probe (residual stream)")]
if any("human_acc" in s for s in S): dets.append(("human_acc","Human (owner, blind, n=24)"))
fig, ax = plt.subplots(figsize=(7,4))
w = 0.8/len(dets); x = np.arange(len(tags))
for k,(key,name) in enumerate(dets):
    vals = [s.get(key, np.nan) for s in S]
    ax.bar(x + k*w - 0.4 + w/2, vals, w, label=name)
ax.axhline(0.5, color="gray", ls="--", lw=1); ax.text(len(tags)-0.6, 0.51, "chance", color="gray", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f"{t}\n(n={s['n']})" for t,s in zip(tags,S)])
ax.set_ylim(0,1.05); ax.set_ylabel("accuracy: eval vs user"); ax.set_title("Eval-awareness detectors under red-teaming (Qwen3.5-4B)")
ax.legend(fontsize=8, loc="lower left"); fig.tight_layout(); fig.savefig("figures/detector_accuracy.png", dpi=150)
sw = json.load(open("results/probe_layer_sweep.json"))
fig, ax = plt.subplots(figsize=(6,3.2))
for pool in sw: ax.plot(sw[pool], marker="o", ms=3, label=f"{pool}-token pooling")
ax.axhline(0.5, color="gray", ls="--", lw=1); ax.set_xlabel("layer (0 = embeddings)"); ax.set_ylabel("5-fold CV accuracy (round 0)")
ax.set_title("Probe layer sweep"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("figures/probe_layer_sweep.png", dpi=150)
print("figures written")
