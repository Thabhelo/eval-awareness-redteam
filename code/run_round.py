"""Run all detectors + answers on a round file. Usage: python code/run_round.py data/roundN.json roundN"""
import sys, json, time, numpy as np
sys.path.insert(0, "code")
from model_utils import *

src, tag = sys.argv[1], sys.argv[2]
rows = json.load(open(src))
model, tok = get_model()
last_acts, mean_acts, out = [], [], []
t0 = time.time()
for i, r in enumerate(rows):
    p = r["prompt"]
    L, M = residuals(chat_text([{"role": "user", "content": p}], thinking=False))
    last_acts.append(L); mean_acts.append(M)
    ans = answer(p, max_tokens=160)
    sr, sr_raw = self_report(p)
    kw, hits = keyword_detector(p)
    out.append(dict(**r, answer=ans, refusal=is_refusal(ans), answer_len=len(ans),
                    self_report=sr, self_report_raw=sr_raw, keyword=kw, keyword_hits=hits))
    if i % 10 == 0:
        print(f"{i}/{len(rows)} {time.time()-t0:.0f}s id={r['id']} sr={sr} kw={kw} ref={is_refusal(ans)}", flush=True)
        json.dump(out, open(f"results/{tag}_outputs.json", "w"), indent=1)
json.dump(out, open(f"results/{tag}_outputs.json", "w"), indent=1)
np.savez_compressed(f"results/{tag}_acts.npz", last=np.stack(last_acts), mean=np.stack(mean_acts),
                    ids=np.array([r["id"] for r in rows]), labels=np.array([r["label"] for r in rows]))
print("DONE", tag, f"{time.time()-t0:.0f}s")
