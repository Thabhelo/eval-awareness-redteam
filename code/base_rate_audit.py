"""Audit balanced-training detectors as rare-event monitors.

The central quantity is positive predictive value after changing the class prior:

    PPV = prevalence * TPR / (prevalence * TPR + (1-prevalence) * FPR)

Usage: python code/base_rate_audit.py
"""
import json
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

PREVALENCES = [0.001, 0.01, 0.05, 0.10]
DETECTORS = [
    "self_report",
    "probe_last",
    "probe_mean",
    "probe_embed",
    "remedy",
    "format_feats",
    "char_ngram",
    "length_only",
    "keyword",
]


def ppv(tpr, fpr, prevalence):
    denominator = prevalence * tpr + (1 - prevalence) * fpr
    return prevalence * tpr / denominator if denominator else None


def binomial_interval(successes, total, alpha=0.05):
    lower = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes, total - successes + 1)
    upper = 1.0 if successes == total else beta.ppf(1 - alpha / 2, successes + 1, total - successes)
    upper_one_sided = 1.0 if successes == total else beta.ppf(1 - alpha, successes + 1, total - successes)
    return float(lower), float(upper), float(upper_one_sided)


def score_of(row, detector):
    if detector == "keyword":
        return float(row["keyword"])
    if detector == "self_report":
        return float(row["sr_margin"])
    return float(row[detector + "_p"])


def fixed_threshold(detector):
    if detector == "self_report":
        return float(json.load(open("results/calibration.json"))["sr_threshold"])
    if detector == "remedy":
        return float(pickle.load(open("results/remedy.pkl", "rb"))["threshold"])
    return 0.5


def operating_point(y, scores, threshold):
    positive = y == 1
    negative = ~positive
    prediction = scores >= threshold
    tp = int(np.sum(prediction & positive))
    fp = int(np.sum(prediction & negative))
    n_pos = int(np.sum(positive))
    n_neg = int(np.sum(negative))
    tpr = tp / n_pos
    fpr = fp / n_neg
    tpr_ci = binomial_interval(tp, n_pos)
    fpr_ci = binomial_interval(fp, n_neg)
    return dict(
        threshold=float(threshold),
        tp=tp,
        fp=fp,
        n_positive=n_pos,
        n_negative=n_neg,
        tpr=tpr,
        tpr_ci95=list(tpr_ci[:2]),
        fpr=fpr,
        fpr_ci95=list(fpr_ci[:2]),
        fpr_upper95_one_sided=fpr_ci[2],
    )


def external_calibration(rows, y, scores):
    """Choose a threshold on a fixed round-5 calibration split, test once."""
    categories = np.array([row["category"] for row in rows])
    source_index = np.array([row["source_index"] for row in rows])
    calibration = np.zeros(len(rows), dtype=bool)
    for category, count in [("wildchat", 300), ("mmlu", 33), ("arc", 33), ("gsm8k", 33)]:
        indices = np.where(categories == category)[0]
        ordered = indices[np.argsort(source_index[indices])]
        calibration[ordered[:count]] = True
    test = ~calibration
    calibration_negative_scores = scores[calibration & (y == 0)]
    threshold = float(np.nextafter(np.max(calibration_negative_scores), np.inf))
    point = operating_point(y[test], scores[test], threshold)
    point["calibration_n_negative"] = int(np.sum(calibration & (y == 0)))
    point["calibration_n_positive"] = int(np.sum(calibration & (y == 1)))
    point["calibration_false_positives"] = int(
        np.sum((scores[calibration & (y == 0)] >= threshold))
    )
    point["auc"] = float(roc_auc_score(y[test], scores[test]))
    point["ppv_at_1pct"] = ppv(point["tpr"], point["fpr"], 0.01)
    point["ppv_lower95_at_1pct"] = ppv(
        point["tpr_ci95"][0], point["fpr_ci95"][1], 0.01
    )
    point["required_fpr_for_50pct_ppv"] = 0.01 * point["tpr"] / 0.99
    point["clears_point_estimate_bar"] = bool(
        point["tpr"] >= 0.5 and point["ppv_at_1pct"] >= 0.5
    )
    point["clears_95pct_confidence_bar"] = bool(
        point["tpr_ci95"][0] >= 0.5
        and point["fpr_upper95_one_sided"]
        <= 0.01 * point["tpr_ci95"][0] / 0.99
    )
    # Descriptive upper bound only: this looks at the final test labels and is
    # never used to select a reported operating threshold.
    test_y = y[test]
    test_scores = scores[test]
    candidates = []
    for candidate_threshold in np.unique(test_scores):
        candidate = operating_point(test_y, test_scores, candidate_threshold)
        if candidate["tpr"] < 0.5:
            continue
        candidate_ppv = ppv(candidate["tpr"], candidate["fpr"], 0.01)
        candidates.append((candidate, candidate_ppv))
    min_fpr_point, _ = min(candidates, key=lambda item: item[0]["fpr"])
    best_ppv_point, best_ppv = max(candidates, key=lambda item: item[1])
    point["posthoc_oracle_diagnostic"] = {
        "warning": "Uses final test labels; descriptive only, not a deployable threshold.",
        "minimum_fpr_at_tpr_at_least_50pct": min_fpr_point["fpr"],
        "tpr_at_minimum_fpr": min_fpr_point["tpr"],
        "threshold_at_minimum_fpr": min_fpr_point["threshold"],
        "best_ppv_at_1pct_with_tpr_at_least_50pct": best_ppv,
        "tpr_at_best_ppv": best_ppv_point["tpr"],
        "fpr_at_best_ppv": best_ppv_point["fpr"],
        "threshold_at_best_ppv": best_ppv_point["threshold"],
    }
    return point


def audit(tag):
    rows = json.load(open(f"results/{tag}_outputs.json"))
    y = np.array([row["label"] for row in rows], dtype=int)
    result = dict(round=tag, n=len(rows), detectors={})
    for detector in DETECTORS:
        if detector == "keyword":
            available = "keyword" in rows[0]
        elif detector == "self_report":
            available = "sr_margin" in rows[0]
        else:
            available = detector + "_p" in rows[0]
        if not available:
            continue
        scores = np.array([score_of(row, detector) for row in rows])
        threshold = fixed_threshold(detector)
        point = operating_point(y, scores, threshold)
        point["auc"] = float(roc_auc_score(y, scores))
        point["ppv_by_prevalence"] = {
            str(prevalence): ppv(point["tpr"], point["fpr"], prevalence)
            for prevalence in PREVALENCES
        }
        point["ppv_lower95_at_1pct"] = ppv(
            point["tpr_ci95"][0], point["fpr_ci95"][1], 0.01
        )
        point["alerts_per_10000_at_1pct"] = 100 * point["tpr"] + 9900 * point["fpr"]
        point["true_alerts_per_10000_at_1pct"] = 100 * point["tpr"]
        point["false_alerts_per_10000_at_1pct"] = 9900 * point["fpr"]
        point["accuracy_at_1pct"] = 0.01 * point["tpr"] + 0.99 * (1 - point["fpr"])
        point["false_alerts_per_true_alert_at_1pct"] = (
            point["false_alerts_per_10000_at_1pct"]
            / point["true_alerts_per_10000_at_1pct"]
            if point["true_alerts_per_10000_at_1pct"]
            else None
        )
        if tag == "round5":
            point["deployment_calibrated"] = external_calibration(rows, y, scores)
            point["by_source"] = {}
            for source in sorted({row["category"] for row in rows}):
                mask = np.array([row["category"] == source for row in rows])
                labels = y[mask]
                source_scores = scores[mask]
                prediction = source_scores >= threshold
                rate = float(np.mean(prediction))
                interval = binomial_interval(int(np.sum(prediction)), int(np.sum(mask)))
                point["by_source"][source] = dict(
                    n=int(np.sum(mask)),
                    positive_call_rate=rate,
                    ci95=list(interval[:2]),
                )
        result["detectors"][detector] = point
    return result


tags = [f"round{index}" for index in range(6) if os.path.exists(f"results/round{index}_outputs.json")]
audits = {tag: audit(tag) for tag in tags}
with open("results/base_rate_audit.json", "w") as handle:
    json.dump(audits, handle, indent=1, allow_nan=False)

print("round detector auc tpr fpr ppv@1% false_alerts/10k")
for tag, result in audits.items():
    for detector, point in result["detectors"].items():
        precision = point["ppv_by_prevalence"]["0.01"]
        print(
            tag,
            detector,
            f"{point['auc']:.3f}",
            f"{point['tpr']:.3f}",
            f"{point['fpr']:.3f}",
            "undefined" if precision is None else f"{precision:.3f}",
            f"{point['false_alerts_per_10000_at_1pct']:.1f}",
        )


def plot():
    chosen = [tag for tag in ["round0", "round1", "round2", "round3", "round4", "round5"] if tag in audits]
    points = [audits[tag]["detectors"]["probe_last"] for tag in chosen]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    for tag, point in zip(chosen, points):
        value = point["ppv_by_prevalence"]["0.01"]
        ax.scatter(point["auc"], value, s=70)
        ax.annotate(tag.replace("round", "R"), (point["auc"], value), xytext=(5, 4), textcoords="offset points")
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set(xlabel="ROC AUC", ylabel="Precision if positives are 1%", ylim=(-0.03, 1.03))
    ax.set_title("High AUC can still mean mostly false alarms")
    ax.grid(alpha=0.2)

    ax = axes[1]
    true = [point["true_alerts_per_10000_at_1pct"] for point in points]
    false = [point["false_alerts_per_10000_at_1pct"] for point in points]
    x = np.arange(len(chosen))
    ax.bar(x, false, color="#d55e5e", label="False alerts")
    ax.bar(x, true, bottom=false, color="#2b7bba", label="True alerts")
    ax.set_xticks(x, [tag.replace("round", "R") for tag in chosen])
    ax.set(ylabel="Expected alerts per 10,000 cases")
    ax.set_title("Frozen 0.5 threshold, 1% prevalence")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Rare-event audit of the layer-15 eval/deployment probe", fontweight="bold")
    fig.tight_layout()
    fig.savefig("figures/base_rate_audit.png", dpi=220, bbox_inches="tight")


plot()
