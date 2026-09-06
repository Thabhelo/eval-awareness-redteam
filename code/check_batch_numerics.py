"""Check whether batched MLX numerics change the primary threshold decisions.

Recomputes, one prompt at a time, every row close to the frozen threshold and
the rows that could determine or cross the deployment-calibrated threshold.
"""
import json
import pickle
import sys

import numpy as np

sys.path.insert(0, "code")
from model_utils import chat_text, residuals


rows = json.load(open("results/round5_outputs.json"))
audit = json.load(open("results/base_rate_audit.json"))["round5"]["detectors"]["probe_last"]
probe = pickle.load(open("results/probes.pkl", "rb"))["probe_last"]
layer = probe["layer"]
calibrated_threshold = audit["deployment_calibrated"]["threshold"]

wild = [row for row in rows if row["category"] == "wildchat"]
calibration_negative_ids = set(
    row["id"] for row in sorted(wild, key=lambda row: row["source_index"])[:300]
)
calibration_ids = set(calibration_negative_ids)
for category in ("mmlu", "arc", "gsm8k"):
    category_rows = sorted(
        (row for row in rows if row["category"] == category),
        key=lambda row: row["source_index"],
    )
    calibration_ids.update(row["id"] for row in category_rows[:33])
cal_neg = [row for row in wild if row["id"] in calibration_negative_ids]
test = [row for row in rows if row["id"] not in calibration_ids]

candidates = {}
for row in rows:
    if abs(row["probe_last_p"] - 0.5) <= 0.02:
        candidates[row["id"]] = row
for row in sorted(cal_neg, key=lambda row: row["probe_last_p"], reverse=True)[:20]:
    candidates[row["id"]] = row
for row in test:
    if abs(row["probe_last_p"] - calibrated_threshold) <= 0.02:
        candidates[row["id"]] = row

checks = []
for index, row in enumerate(candidates.values(), 1):
    text = chat_text([{"role": "user", "content": row["prompt"]}], thinking=False)
    last, _ = residuals(text)
    p_single = float(probe["clf"].predict_proba(last[layer][None])[:, 1][0])
    checks.append(
        {
            "id": row["id"],
            "category": row["category"],
            "source_index": row["source_index"],
            "p_batch": row["probe_last_p"],
            "p_single": p_single,
            "absolute_difference": abs(p_single - row["probe_last_p"]),
            "fixed_batch_call": row["probe_last_p"] >= 0.5,
            "fixed_single_call": p_single >= 0.5,
            "is_calibration_negative": row["id"] in calibration_negative_ids,
        }
    )
    print(f"{index}/{len(candidates)} {row['id']}", flush=True)

calibration_checks = [item for item in checks if item["is_calibration_negative"]]
single_calibrated_threshold = float(
    np.nextafter(max(item["p_single"] for item in calibration_checks), np.inf)
)
for item in checks:
    item["calibrated_batch_call"] = item["p_batch"] >= calibrated_threshold
    item["calibrated_single_call"] = item["p_single"] >= single_calibrated_threshold

summary = {
    "n_checked": len(checks),
    "candidate_rule": "all rows within 0.02 of 0.5; top 20 calibration negatives; all test rows within 0.02 of the calibrated threshold",
    "max_absolute_probability_difference": max(item["absolute_difference"] for item in checks),
    "fixed_threshold_flips": sum(item["fixed_batch_call"] != item["fixed_single_call"] for item in checks),
    "batch_calibrated_threshold": calibrated_threshold,
    "single_prompt_calibrated_threshold_from_checked_top20": single_calibrated_threshold,
    "calibrated_call_flips_among_checked_rows": sum(
        item["calibrated_batch_call"] != item["calibrated_single_call"] for item in checks
    ),
}

# Sensitivity operating points: substitute single-prompt scores for every row
# inside the stated decision bands, while retaining batched scores elsewhere.
# These are not replacements for the pre-registered batched implementation;
# they show whether the conclusions depend on batching numerics near a boundary.
checked_by_id = {item["id"]: item for item in checks}


def sensitivity_point(threshold, calibrated=False):
    calls = []
    for row in rows:
        checked = checked_by_id.get(row["id"])
        score = checked["p_single"] if checked is not None else row["probe_last_p"]
        calls.append(score >= threshold)
    labels = np.array([row["label"] for row in rows], dtype=bool)
    calls = np.array(calls, dtype=bool)
    if calibrated:
        test_mask = np.array([row["id"] not in calibration_ids for row in rows], dtype=bool)
        labels = labels[test_mask]
        calls = calls[test_mask]
    tp = int(np.sum(calls & labels))
    fp = int(np.sum(calls & ~labels))
    n_positive = int(np.sum(labels))
    n_negative = int(np.sum(~labels))
    tpr = tp / n_positive
    fpr = fp / n_negative
    prevalence = 0.01
    denominator = prevalence * tpr + (1 - prevalence) * fpr
    return {
        "status": "targeted sensitivity; checked single-prompt scores substituted inside the documented decision bands",
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "tpr": tpr,
        "fpr": fpr,
        "ppv_at_1pct": prevalence * tpr / denominator if denominator else None,
    }


summary["fixed_threshold_sensitivity"] = sensitivity_point(0.5)
summary["calibrated_threshold_sensitivity"] = sensitivity_point(
    single_calibrated_threshold, calibrated=True
)
json.dump({"summary": summary, "checks": checks}, open("results/round5_batch_check.json", "w"), indent=1)
print(json.dumps(summary, indent=2))
