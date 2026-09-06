"""Write deterministic qualitative samples for the round-5 audit."""
import json
import random

SEED = 20260905
rows = json.load(open("results/round5_outputs.json"))
audit = json.load(open("results/base_rate_audit.json"))["round5"]["detectors"]["probe_last"]


def excerpt(text, limit=360):
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def line(row):
    return (
        f"- {row['id']} | declared={row['category']} | "
        f"probe={row['probe_last_p']:.3f} | {excerpt(row['prompt'])}"
    )


rng = random.Random(SEED)
negative = [row for row in rows if row["label"] == 0]
positive = [row for row in rows if row["label"] == 1]
rng.shuffle(negative)
rng.shuffle(positive)
random_sample = negative[:10] + positive[:3]

fixed_false_positives = sorted(
    [row for row in rows if row["label"] == 0 and row["probe_last_p"] >= 0.5],
    key=lambda row: row["probe_last_p"],
    reverse=True,
)
near_threshold = sorted(rows, key=lambda row: abs(row["probe_last_p"] - 0.5))[:12]

calibrated_threshold = audit["deployment_calibrated"]["threshold"]
calibration_negatives = sorted(negative, key=lambda row: row["source_index"])[:300]
calibration_ids = {row["id"] for row in calibration_negatives}
calibrated_test_false_positives = sorted(
    [
        row
        for row in negative
        if row["id"] not in calibration_ids and row["probe_last_p"] >= calibrated_threshold
    ],
    key=lambda row: row["probe_last_p"],
    reverse=True,
)

with open("notes/round5_qualitative_audit.md", "w") as handle:
    handle.write("# Round-5 qualitative audit\n\n")
    handle.write("Generated deterministically with seed 20260905. No item was selected for looking persuasive.\n\n")
    handle.write("## Random sample chosen without detector scores\n\n")
    handle.write("\n".join(line(row) for row in random_sample) + "\n\n")
    handle.write(f"## Highest-scoring WildChat false positives at frozen 0.5 threshold ({len(fixed_false_positives)} total)\n\n")
    handle.write("\n".join(line(row) for row in fixed_false_positives[:30]) + "\n\n")
    handle.write("## Items nearest the frozen threshold\n\n")
    handle.write("\n".join(line(row) for row in near_threshold) + "\n\n")
    handle.write(
        f"## Held-out WildChat false positives after deployment calibration at {calibrated_threshold:.6f} "
        f"({len(calibrated_test_false_positives)} total)\n\n"
    )
    handle.write("\n".join(line(row) for row in calibrated_test_false_positives[:30]) + "\n")

print(
    f"fixed false positives={len(fixed_false_positives)}; "
    f"deployment-calibrated test false positives={len(calibrated_test_false_positives)}"
)
