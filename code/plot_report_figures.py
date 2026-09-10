"""Report figures for the MATS write-up, drawn only from saved outputs.

Reads:
  results/round5_outputs.json   per-row scores (probe_last_p, remedy_p, label, category, source_index)
  results/base_rate_audit.json  summary block written by code/base_rate_audit.py
  results/remedy.pkl            frozen threshold of the style-balanced probe

Nothing is re-scored. Every number drawn on a figure is recomputed from the rows
above and asserted against the values quoted in the report; a mismatch raises.

Usage: python code/plot_report_figures.py
"""
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# Validated categorical slots (dataviz reference palette, light surface).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.dpi": 300, "figure.dpi": 100, "savefig.facecolor": "white",
})
WIDTH = 6.5  # inches, Google Doc column width


def ppv(tpr, fpr, prevalence):
    return prevalence * tpr / (prevalence * tpr + (1 - prevalence) * fpr)


def check(name, got, expected, tol=0.0):
    ok = abs(got - expected) <= tol if isinstance(expected, float) else got == expected
    flag = "OK " if ok else "MISMATCH"
    print(f"  [{flag}] {name}: recomputed {got} vs report {expected}")
    if not ok:
        raise SystemExit(f"STOP: {name} disagrees with the report (recomputed {got}, report {expected})")


# ---------------------------------------------------------------- load rows
rows = json.load(open("results/round5_outputs.json"))
audit = json.load(open("results/base_rate_audit.json"))["round5"]["detectors"]
y = np.array([int(r["label"]) for r in rows])
cat = np.array([r["category"] for r in rows])
src = np.array([r["source_index"] for r in rows])
p_frozen = np.array([float(r["probe_last_p"]) for r in rows])
p_remedy = np.array([float(r["remedy_p"]) for r in rows])
remedy_thr = float(pickle.load(open("results/remedy.pkl", "rb"))["threshold"])

# Calibration split exactly as in code/base_rate_audit.py::external_calibration
calib = np.zeros(len(rows), dtype=bool)
for c, n in [("wildchat", 300), ("mmlu", 33), ("arc", 33), ("gsm8k", 33)]:
    idx = np.where(cat == c)[0]
    calib[idx[np.argsort(src[idx])][:n]] = True
test = ~calib
calib_thr = float(np.nextafter(np.max(p_frozen[calib & (y == 0)]), np.inf))

neg, pos = p_frozen[y == 0], p_frozen[y == 1]

print("== Recomputed from results/round5_outputs.json ==")
check("n negatives (WildChat)", int(len(neg)), 1000)
check("n positives (benchmarks)", int(len(pos)), 300)
fp_frozen = int((neg >= 0.5).sum()); tp_frozen = int((pos >= 0.5).sum())
check("frozen FP at 0.5", fp_frozen, 331)
check("frozen TP at 0.5", tp_frozen, 300)
check("frozen FPR", round(fp_frozen / 1000, 4), 0.331, 1e-9)
auc_full = roc_auc_score(y, p_frozen)
check("frozen AUC (all 1,300)", round(auc_full, 3), 0.983, 1e-9)
fp_rem = int((p_remedy[y == 0] >= remedy_thr).sum()); tp_rem = int((p_remedy[y == 1] >= remedy_thr).sum())
check("style-balanced FP", fp_rem, 147)
check("style-balanced TP", tp_rem, 299)
check("style-balanced AUC", round(roc_auc_score(y, p_remedy), 3), 0.988, 1e-9)
check("calibration negatives", int((calib & (y == 0)).sum()), 300)
check("calibration threshold == audit", calib_thr, audit["probe_last"]["deployment_calibrated"]["threshold"], 1e-12)
n_test_neg = int((test & (y == 0)).sum()); n_test_pos = int((test & (y == 1)).sum())
check("test negatives", n_test_neg, 700)
check("test positives", n_test_pos, 201)
fp_cal = int((p_frozen[test & (y == 0)] >= calib_thr).sum())
tp_cal = int((p_frozen[test & (y == 1)] >= calib_thr).sum())
check("calibrated FP (test)", fp_cal, 3)
check("calibrated TP (test)", tp_cal, 64)
tpr_f, fpr_f = tp_frozen / 300, fp_frozen / 1000
tpr_r, fpr_r = tp_rem / 300, fp_rem / 1000
tpr_c, fpr_c = tp_cal / 201, fp_cal / 700
check("projected precision @1%, frozen (%)", round(100 * ppv(tpr_f, fpr_f, 0.01), 2), 2.96, 1e-9)
check("projected precision @1%, style-balanced (%)", round(100 * ppv(tpr_r, fpr_r, 0.01), 2), 6.41, 1e-9)
check("projected precision @1%, calibrated (%)", round(100 * ppv(tpr_c, fpr_c, 0.01), 1), 42.9, 1e-9)
check("calibrated TPR (%)", round(100 * tpr_c, 1), 31.8, 1e-9)
check("calibrated FPR (%)", round(100 * fpr_c, 2), 0.43, 1e-9)
print(f"  calibrated threshold value: {calib_thr:.6f}")
print(f"  style-balanced frozen threshold: {remedy_thr:.6f}")
print(f"  AUC on held-out test split: {roc_auc_score(y[test], p_frozen[test]):.4f}")

# ================================================================ Figure 1
# Logit-spaced bins so the 0.90-0.998 positive mass and both thresholds are legible.
edges = np.linspace(-9.5, 9.5, 39)
logit = lambda q: np.log(q / (1 - q))
h_neg, _ = np.histogram(logit(neg), bins=edges)
h_pos, _ = np.histogram(logit(pos), bins=edges)
centers = (edges[:-1] + edges[1:]) / 2
w = edges[1] - edges[0]
fp_bins = centers >= logit(0.5)  # first edge above 0.5 is exactly 0 -> clean split
assert h_neg[fp_bins].sum() == fp_frozen and h_pos[fp_bins].sum() == tp_frozen

fig, ax = plt.subplots(figsize=(WIDTH, 4.3))
ax.bar(centers, h_neg, width=w, color=BLUE, alpha=0.32, edgecolor="white", linewidth=0.6, label="WildChat users (n = 1,000)")
ax.bar(centers[fp_bins], h_neg[fp_bins], width=w, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.6)
ax.bar(centers, h_pos, width=w, color=ORANGE, alpha=0.80, edgecolor="white", linewidth=0.6, label="Benchmark prompts (n = 300)")
ax.axvline(0, color=INK, linewidth=1.4, linestyle="--")
ax.axvline(logit(calib_thr), color=INK, linewidth=1.4, linestyle=":")
ax.grid(axis="x", visible=False)
ticks = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
ax.set_xticks([logit(t) for t in ticks], [f"{t:g}" for t in ticks])
ax.set_xlim(edges[0], edges[-1])
ax.set_xlabel("Probe score  P(evaluation)  [logit-spaced axis]")
ax.set_ylabel("Prompts per bin (count)")
ax.set_title("Frozen layer-15 probe: 331 of 1,000 real-user prompts score above 0.5", loc="left")
ymax = max(h_neg.max(), h_pos.max())
top = ymax * 1.48
ax.set_ylim(0, top)
ax.set_xlim(edges[0], 10.2)
ax.set_yticks(np.arange(0, ymax + 1, 20))
ax.text(0.3, top * 0.99, "frozen\nthreshold 0.5", ha="left", va="top", fontsize=11, color=INK, linespacing=1.1)
ax.text(logit(calib_thr) + 0.3, top * 0.99, f"calibrated\nthreshold {calib_thr:.3f}", ha="left", va="top", fontsize=11, color=INK, linespacing=1.1)
ax.annotate(f"{fp_frozen} false positives\n({fp_frozen}/1,000 = 33.1% FPR)", xy=(logit(0.66), 68), xytext=(-0.6, ymax * 1.14), fontsize=11, color=INK, ha="right", va="top", linespacing=1.1,
            arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8, shrinkB=2))
ax.text(5.7, ymax * 0.66, f"{tp_frozen}/300 benchmark\nprompts above 0.5", ha="left", va="center", fontsize=11, color=INK, linespacing=1.1)
ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 1.0), handlelength=1.4)
fig.tight_layout()
fig.savefig("figures/fig1_score_distributions.png", bbox_inches="tight")
print("\n== Figure 1 (figures/fig1_score_distributions.png) values ==")
print(f"  negatives: n={len(neg)}, >=0.5: {fp_frozen} (shaded), >={calib_thr:.4f}: {int((neg>=calib_thr).sum())}")
print(f"  positives: n={len(pos)}, >=0.5: {tp_frozen}, >={calib_thr:.4f}: {int((pos>=calib_thr).sum())}")
print(f"  vertical lines: 0.5 and {calib_thr:.6f}")
print(f"  bins: {len(edges)-1} equal-width in logit space over ({edges[0]}, {edges[-1]}); score range neg [{neg.min():.4f}, {neg.max():.4f}], pos [{pos.min():.4f}, {pos.max():.4f}]")

# ================================================================ Figure 2
prev = np.logspace(np.log10(0.001), np.log10(0.5), 400)
series = [
    ("Frozen probe, threshold 0.5\n(TPR 100%, FPR 33.1%)", tpr_f, fpr_f, BLUE, (0.0155, 14.0)),
    ("Style-balanced probe\n(TPR 99.7%, FPR 14.7%)", tpr_r, fpr_r, ORANGE, (0.0155, 23.0)),
    ("Calibrated threshold\n(TPR 31.8%, FPR 0.43%)", tpr_c, fpr_c, AQUA, (0.0118, 40.5)),
]
fig, ax = plt.subplots(figsize=(WIDTH, 4.3))
ax.axhline(50, color=INK2, linewidth=1.0, linestyle="--")
ax.text(0.022, 51.5, "50% precision bar", fontsize=11, color=INK2, va="bottom", ha="left")
ax.axvline(0.01, color=GRID, linewidth=1.0)
print("\n== Figure 2 (figures/fig2_precision_vs_prevalence.png) values ==")
for label, tpr, fpr, color, txt in series:
    ax.plot(prev, 100 * ppv(tpr, fpr, prev), color=color, linewidth=2, label=label)
    v = 100 * ppv(tpr, fpr, 0.01)
    ax.plot([0.01], [v], marker="o", markersize=8, color=color, markeredgecolor="white", markeredgewidth=1.5, zorder=5)
    ax.annotate(f"{v:.2f}%" if v < 10 else f"{v:.1f}%", xy=(0.01, v), xytext=txt, fontsize=11, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.7, shrinkB=5), zorder=6)
    print(f"  {label.splitlines()[0]}: TPR={tpr:.6f}, FPR={fpr:.6f}, precision@1% = {v:.4f}%")
ax.set_xscale("log")
ax.set_xlim(0.001, 0.5)
ax.set_xticks([0.001, 0.01, 0.1, 0.5], ["0.1%", "1%", "10%", "50%"])
ax.set_ylim(0, 100)
ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Prevalence of evaluation prompts in traffic (base rate, log scale)")
ax.set_ylabel("Projected precision (% of alerts real)")
ax.set_title("AUC 0.983 still yields 2.96% precision at a 1% base rate", loc="left")
ax.text(0.0107, 98, "1% operating point", fontsize=11, color=INK2, va="top", ha="left")
ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.99), handlelength=1.4, labelspacing=0.45)
fig.tight_layout()
fig.savefig("figures/fig2_precision_vs_prevalence.png", bbox_inches="tight")
print("  curve: precision = pi*TPR / (pi*TPR + (1-pi)*FPR), pi from 0.001 to 0.5; horizontal line at 50%")

# ================================================================ Figure 3
fpr_all, tpr_all, _ = roc_curve(y, p_frozen)
fpr_t, tpr_t, _ = roc_curve(y[test], p_frozen[test])
auc_test = roc_auc_score(y[test], p_frozen[test])
fig, ax = plt.subplots(figsize=(WIDTH, 4.6))
ax.plot([0, 1], [0, 1], color=GRID, linewidth=1, linestyle="--")
ax.plot(fpr_all, tpr_all, color=BLUE, linewidth=2, label=f"All 1,300 prompts (AUC {auc_full:.3f})")
ax.plot(fpr_t, tpr_t, color=AQUA, linewidth=2, linestyle=(0, (3, 2)), label=f"Held-out test split, 700 neg / 201 pos (AUC {auc_test:.3f})")
ax.plot([fpr_f], [tpr_f], "o", markersize=9, color=BLUE, markeredgecolor="white", markeredgewidth=1.5, zorder=5)
ax.annotate(f"Frozen 0.5 threshold\n(FPR {100*fpr_f:.1f}%, TPR {100*tpr_f:.0f}%)", xy=(fpr_f, tpr_f), xytext=(fpr_f + 0.05, tpr_f - 0.14), fontsize=11, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
ax.plot([fpr_c], [tpr_c], "o", markersize=9, color=AQUA, markeredgecolor="white", markeredgewidth=1.5, zorder=5)
ax.annotate(f"Calibrated threshold {calib_thr:.3f}\n(FPR {100*fpr_c:.2f}%, TPR {100*tpr_c:.1f}%)", xy=(fpr_c, tpr_c), xytext=(0.10, tpr_c - 0.03), fontsize=11, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
ax.set_xlim(-0.01, 1.0); ax.set_ylim(0, 1.02)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0%", "20%", "40%", "60%", "80%", "100%"])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0%", "20%", "40%", "60%", "80%", "100%"])
ax.set_xlabel("False-positive rate (real-user prompts flagged)")
ax.set_ylabel("True-positive rate (benchmark prompts flagged)")
ax.set_title("ROC of the frozen layer-15 probe with both operating points", loc="left")
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig("figures/fig3_roc_operating_points.png", bbox_inches="tight")
print("\n== Figure 3 (figures/fig3_roc_operating_points.png) values ==")
print(f"  full-set ROC: n={len(y)}, AUC={auc_full:.4f}; frozen point FPR={fpr_f:.4f} ({fp_frozen}/1000), TPR={tpr_f:.4f} ({tp_frozen}/300)")
print(f"  test-split ROC: n={int(test.sum())}, AUC={auc_test:.4f}; calibrated point FPR={fpr_c:.5f} ({fp_cal}/700), TPR={tpr_c:.5f} ({tp_cal}/201)")
