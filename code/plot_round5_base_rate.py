"""Application figure for the round-5 rare-event result."""
import json

import matplotlib.pyplot as plt


audit = json.load(open("results/base_rate_audit.json"))["round5"]["detectors"]
order = [
    ("probe_last", "Residual probe", "#b33a3a"),
    ("remedy", "Style-balanced probe", "#7a5195"),
    ("self_report", "Self-report score", "#009e73"),
    ("probe_embed", "Embedding probe", "#2f6f9f"),
    ("probe_mean", "Mean-pool probe", "#5b8e7d"),
    ("format_feats", "Format baseline", "#cc8b3c"),
    ("char_ngram", "Character baseline", "#777777"),
]

fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})

ax = axes[0]
for key, label, color in order:
    point = audit[key]
    auc = point["auc"]
    precision = point["ppv_by_prevalence"]["0.01"]
    size = 105 if key == "probe_last" else 70
    ax.scatter(
        auc,
        precision,
        s=size,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label=label,
    )
ax.axhline(0.5, color="#222222", linestyle="--", linewidth=1)
ax.text(0.635, 0.515, "minimum precision target", fontsize=8.5, color="#333333")
ax.set_xlim(0.60, 1.005)
ax.set_ylim(0, 0.56)
ax.set_xlabel("ROC AUC on the external set")
ax.set_ylabel("Precision if only 1% of cases are positive")
ax.set_title("High AUC does not make a usable monitor", loc="left", fontweight="bold")
ax.grid(alpha=0.18)
ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.02, 0.88),
    ncol=2,
    frameon=False,
    fontsize=8.2,
    handletextpad=0.35,
    columnspacing=0.7,
)

ax = axes[1]
primary = audit["probe_last"]
cal = primary["deployment_calibrated"]
bars = [
    (
        "Frozen threshold",
        primary["true_alerts_per_10000_at_1pct"],
        primary["false_alerts_per_10000_at_1pct"],
        primary["ppv_by_prevalence"]["0.01"],
        primary["tpr"],
    ),
    (
        "Calibrated on 300\nreal-user prompts",
        100 * cal["tpr"],
        9900 * cal["fpr"],
        cal["ppv_at_1pct"],
        cal["tpr"],
    ),
]
for index, (name, true_count, false_count, precision, recall) in enumerate(bars):
    total = true_count + false_count
    true_share = true_count / total if total else 0
    ax.barh(index, true_share, color="#2b7bba", height=0.56)
    ax.barh(index, 1 - true_share, left=true_share, color="#d55e5e", height=0.56)
    ax.text(0.02, index, f"{true_count:.0f} true", va="center", ha="left", color="white", fontweight="bold", fontsize=9)
    ax.text(0.98, index, f"{false_count:.0f} false", va="center", ha="right", color="white", fontweight="bold", fontsize=9)
    ax.text(1.02, index, f"{precision:.1%} precision\n{recall:.1%} recall", va="center", fontsize=9)
ax.axvline(0.5, color="#222222", linestyle="--", linewidth=1)
ax.set_yticks(range(len(bars)), [item[0] for item in bars])
ax.invert_yaxis()
ax.set_xlim(0, 1.32)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Share of alerts")
ax.set_title("What the alerts contain at a 1% base rate", loc="left", fontweight="bold")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)

fig.suptitle("The 1% stress test: a strong ranking score still mostly cries wolf", fontsize=14, fontweight="bold", y=1.02)
fig.text(0.5, -0.01, "External set: 1,000 WildChat first-user turns and 300 unseen benchmark prompts. Thresholds were not fit on the final test split.", ha="center", fontsize=8.5, color="#444444")
fig.tight_layout()
fig.savefig("figures/round5_base_rate.png", dpi=240, bbox_inches="tight", facecolor="white")
