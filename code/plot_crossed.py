"""Round-3 figures: (a) AUC vs context and vs style per detector; (b) 2x2 cell means for probe_last and format_feats.
Usage: python code/plot_crossed.py"""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
C = json.load(open("results/round3_crossed.json"))
order = [d for d in ["self_report", "probe_last", "probe_embed", "format_feats", "char_ngram", "length_only", "keyword", "random"] if d in C]
labels = {"self_report": "self-report", "probe_last": "probe L15 last", "probe_embed": "probe embeddings", "format_feats": "format features", "char_ngram": "char n-gram", "length_only": "length only", "keyword": "keyword", "random": "random"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.4, 1]})
x = np.arange(len(order)); w = 0.38
axes[0].bar(x - w/2, [C[d]["auc_context"] for d in order], w, label="AUC vs context (eval vs user)", color="#1f77b4")
axes[0].bar(x + w/2, [C[d]["auc_style"] for d in order], w, label="AUC vs style (terse vs conversational)", color="#ff7f0e")
axes[0].axhline(0.5, color="gray", ls="--", lw=1); axes[0].set_xticks(x); axes[0].set_xticklabels([labels[d] for d in order], rotation=30, ha="right", fontsize=8)
axes[0].set_ylim(0, 1.05); axes[0].set_title("Round 3: what does each detector track?", fontsize=10); axes[0].legend(fontsize=8, loc="lower left")
cells = ["eval-terse", "eval-conv", "user-terse", "user-conv"]
M = np.array([[C[d]["cells"][c] for c in cells] for d in ["probe_last", "format_feats", "char_ngram"] if d in C])
im = axes[1].imshow(M, vmin=0, vmax=1, cmap="Purples", aspect="auto")
axes[1].set_xticks(range(4)); axes[1].set_xticklabels(cells, rotation=30, ha="right", fontsize=8); axes[1].set_yticks(range(M.shape[0])); axes[1].set_yticklabels(["probe L15 last", "format features", "char n-gram"], fontsize=8)
for i in range(M.shape[0]):
    for j in range(4): axes[1].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8, color="black" if M[i,j] < 0.6 else "white")
axes[1].set_title("Mean detector score per cell (1 = 'eval')", fontsize=10)
fig.tight_layout(); fig.savefig("figures/round3_crossed.png", dpi=150); print("figures/round3_crossed.png")
