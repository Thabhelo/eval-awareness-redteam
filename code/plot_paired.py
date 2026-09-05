"""Round-4 figure: per-question paired probe score and self-report margin, ctx_user -> ctx_eval, plus refusal/length."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = json.load(open("results/round4_outputs.json")); by = {}
for r in rows: by.setdefault(r["base_id"], {})[r["frame"]] = r
pairs = [(v["ctx_user"], v["ctx_eval"]) for v in by.values() if len(v) == 2]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, key, title in [(axes[0], "probe_last_p", "Probe L15 score"), (axes[1], "sr_margin", "Self-report margin (logit A - logit B)")]:
    for u, e in pairs: ax.plot([0, 1], [u[key], e[key]], color="#7b2cbf", alpha=0.35, lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["'helping a member\nof the public'", "'automated evaluation\nof the assistant'"], fontsize=8); ax.set_title(title, fontsize=10)
ax = axes[2]; ref = [np.mean([u["refusal"] for u, e in pairs]), np.mean([e["refusal"] for u, e in pairs])]; ln = [np.mean([u["answer_len"] for u, e in pairs]), np.mean([e["answer_len"] for u, e in pairs])]
ax.bar([0, 1], ref, 0.5, color="#d62728"); ax.set_ylim(0, 1); ax.set_xticks([0, 1]); ax.set_xticklabels(["user line", "eval line"], fontsize=8); ax.set_title(f"Refusal rate (answer length {ln[0]:.0f} vs {ln[1]:.0f} chars)", fontsize=10)
fig.suptitle("Round 4: same question, only the preceding context line differs (n=60 pairs)", fontsize=10); fig.tight_layout(); fig.savefig("figures/round4_paired.png", dpi=150); print("figures/round4_paired.png")
