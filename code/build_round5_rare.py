"""Build the preregistered rare-event audit set from public Hugging Face datasets.

Negative class: 1,000 authentic English first-user turns from WildChat-1M.
Positive class: 100 prompts each from MMLU, ARC-Challenge, and GSM8K.

Rows are sampled deterministically by source row index. No model output or detector
score is consulted during construction.
"""
import json
import random
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError

SEED = 20260905


def rows(dataset, config, split, offset, length=100):
    query = urllib.parse.urlencode(
        dict(dataset=dataset, config=config, split=split, offset=offset, length=length)
    )
    url = "https://datasets-server.huggingface.co/rows?" + query
    request = urllib.request.Request(url, headers={"User-Agent": "eval-awareness-research/1.0"})
    for attempt in range(7):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
            time.sleep(1.25)
            return result
        except HTTPError as error:
            if error.code != 429 or attempt == 6:
                raise
            time.sleep(5 * (attempt + 1))


def sample_pages(dataset, config, split, target, page_count, seed):
    first = rows(dataset, config, split, 0, 1)
    total = first["num_rows_total"]
    rng = random.Random(seed)
    offsets = sorted(rng.sample(range(0, total - 100), page_count))
    candidates = []
    for offset in offsets:
        candidates.extend(rows(dataset, config, split, offset, 100)["rows"])
    rng.shuffle(candidates)
    return candidates, total, offsets


def wildchat_prompts():
    candidates, total, offsets = sample_pages(
        "allenai/WildChat-1M", "default", "train", 1000, 36, SEED
    )
    out, seen = [], set()
    for item in candidates:
        row = item["row"]
        conversation = row.get("conversation") or []
        if not conversation:
            continue
        first = conversation[0]
        prompt = (first.get("content") or "").strip()
        if first.get("role") != "user" or first.get("redacted"):
            continue
        if (row.get("language") or first.get("language")) != "English":
            continue
        if not (20 <= len(prompt) <= 1200):
            continue
        key = prompt.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(
            dict(
                id=f"r5_wild_{item['row_idx']}",
                round=5,
                base_id=len(out),
                category="wildchat",
                label=0,
                source="WildChat-1M",
                source_index=item["row_idx"],
                prompt=prompt,
            )
        )
        if len(out) == 1000:
            break
    if len(out) != 1000:
        raise RuntimeError(f"Only found {len(out)} valid WildChat prompts")
    return out, dict(
        total=total,
        page_offsets=offsets,
        selected_row_indices=sorted(row["source_index"] for row in out),
    )


def format_mmlu(row):
    letters = "ABCD"
    choices = "\n".join(f"{a}. {c}" for a, c in zip(letters, row["choices"]))
    return f"{row['question']}\n{choices}\nAnswer:"


def format_arc(row):
    choices = "\n".join(
        f"{a}. {c}" for a, c in zip(row["choices"]["label"], row["choices"]["text"])
    )
    return f"{row['question']}\n{choices}\nAnswer:"


def format_gsm8k(row):
    return f"{row['question']}\nAnswer:"


def eval_prompts():
    specs = [
        ("cais/mmlu", "all", "test", "mmlu", format_mmlu, SEED + 1),
        ("allenai/ai2_arc", "ARC-Challenge", "test", "arc", format_arc, SEED + 2),
        ("openai/gsm8k", "main", "test", "gsm8k", format_gsm8k, SEED + 3),
    ]
    out, metadata = [], {}
    for dataset, config, split, short, formatter, seed in specs:
        candidates, total, offsets = sample_pages(
            dataset, config, split, 100, 3, seed
        )
        selected, seen = [], set()
        for item in candidates:
            prompt = formatter(item["row"]).strip()
            if prompt.casefold() in seen or len(prompt) > 1600:
                continue
            seen.add(prompt.casefold())
            selected.append(
                dict(
                    id=f"r5_{short}_{item['row_idx']}",
                    round=5,
                    base_id=1000 + len(out) + len(selected),
                    category=short,
                    label=1,
                    source=dataset,
                    source_index=item["row_idx"],
                    prompt=prompt,
                )
            )
            if len(selected) == 100:
                break
        if len(selected) != 100:
            raise RuntimeError(f"Only found {len(selected)} valid {short} prompts")
        out.extend(selected)
        metadata[short] = dict(
            total=total,
            page_offsets=offsets,
            selected_row_indices=sorted(row["source_index"] for row in selected),
        )
    return out, metadata


negatives, wild_meta = wildchat_prompts()
positives, eval_meta = eval_prompts()
dataset = negatives + positives
random.Random(SEED).shuffle(dataset)

with open("data/round5_rare.json", "w") as handle:
    json.dump(dataset, handle, ensure_ascii=False, indent=1)
with open("data/round5_sources.json", "w") as handle:
    json.dump(
        dict(seed=SEED, negatives=wild_meta, positives=eval_meta),
        handle,
        indent=1,
    )
print(f"Wrote {len(dataset)} rows: {len(negatives)} deployment, {len(positives)} evaluation")
