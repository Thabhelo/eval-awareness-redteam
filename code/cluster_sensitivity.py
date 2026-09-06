"""Post-hoc sensitivity analysis for WildChat page-level dependence.

Round 5 fetched 100-row source pages, then sampled unique prompts from their
union. Rows from the same page may be correlated, so the exact binomial
interval is not the only uncertainty view. This script bootstraps source pages
as clusters and reports leave-one-page-out ranges. It does not change scores or
thresholds.
"""
import json

import numpy as np


SEED = 20260905
BOOTSTRAPS = 20000
rows = json.load(open("results/round5_outputs.json"))
metadata = json.load(open("data/round5_sources.json"))
audit = json.load(open("results/base_rate_audit.json"))["round5"]["detectors"]["probe_last"]
offsets = metadata["negatives"]["page_offsets"]


def page_of(source_index):
    matches = [offset for offset in offsets if offset <= source_index < offset + 100]
    if len(matches) != 1:
        raise ValueError(f"Expected one page for source row {source_index}, got {matches}")
    return matches[0]


wild = [dict(row, source_page_offset=page_of(row["source_index"])) for row in rows if row["label"] == 0]
calibration_ids = {
    row["id"] for row in sorted(wild, key=lambda row: row["source_index"])[:300]
}


def analyse(selected, threshold):
    clusters = []
    for page in sorted({row["source_page_offset"] for row in selected}):
        page_rows = [row for row in selected if row["source_page_offset"] == page]
        clusters.append(
            {
                "page_offset": page,
                "n": len(page_rows),
                "fp": sum(row["probe_last_p"] >= threshold for row in page_rows),
            }
        )
    rng = np.random.default_rng(SEED)
    rates = []
    for _ in range(BOOTSTRAPS):
        sample = rng.integers(0, len(clusters), len(clusters))
        n = sum(clusters[index]["n"] for index in sample)
        fp = sum(clusters[index]["fp"] for index in sample)
        rates.append(fp / n)
    leave_one_out = []
    total_n = sum(cluster["n"] for cluster in clusters)
    total_fp = sum(cluster["fp"] for cluster in clusters)
    for cluster in clusters:
        leave_one_out.append(
            (total_fp - cluster["fp"]) / (total_n - cluster["n"])
        )
    return {
        "n_rows": total_n,
        "n_page_clusters": len(clusters),
        "false_positives": total_fp,
        "fpr": total_fp / total_n,
        "page_cluster_bootstrap_ci95": [float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))],
        "leave_one_page_out_fpr_range": [min(leave_one_out), max(leave_one_out)],
        "clusters": clusters,
    }


result = {
    "status": "post-hoc sensitivity analysis; not pre-registered",
    "iid_warning": "Rows fetched from the same 100-row source page may be correlated.",
    "fixed_threshold": analyse(wild, 0.5),
    "deployment_calibrated_test": analyse(
        [row for row in wild if row["id"] not in calibration_ids],
        audit["deployment_calibrated"]["threshold"],
    ),
}
json.dump(result, open("results/round5_cluster_sensitivity.json", "w"), indent=1)
print(json.dumps({key: value for key, value in result.items() if key not in ("iid_warning",)}, indent=2))
