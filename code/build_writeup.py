"""Build the application write-up (HTML for Google Docs import). Numbers are read from results/*_scores.json.
Usage: python code/build_writeup.py round0 round1 [round2]"""
import sys, json, base64, random, datetime
tags = sys.argv[1:]; S = {t: json.load(open(f"results/{t}_scores.json")) for t in tags}
R = {t: json.load(open(f"results/{t}_outputs.json")) for t in tags}
def pct(x): return f"{100*x:.0f}%"
def img(path, w=640):
    b = base64.b64encode(open(path, "rb").read()).decode(); return f'<img src="data:image/png;base64,{b}" width="{w}">'
REW = '<span style="background:#fff2a8;font-weight:bold">[REWRITE IN YOUR OWN VOICE]</span>'
def row(label, key, fmt=pct):
    cells = "".join(f"<td>{fmt(S[t][key]) if key in S[t] else '-'}</td>" for t in tags); return f"<tr><td>{label}</td>{cells}</tr>"
hdr = "".join(f"<th>{t}</th>" for t in tags)
table = f"""<table border="1" cellpadding="4" style="border-collapse:collapse;font-size:90%"><tr><th>Detector</th>{hdr}</tr>
{row("Keyword heuristic, accuracy", "keyword_acc")}
{row("Self-report (logit margin), AUC", "self_report_auc", lambda x: f"{x:.2f}")}
{row("Self-report, accuracy at frozen threshold", "self_report_acc")}
{row("Probe on raw embeddings (control), AUC", "probe_embed_auc", lambda x: f"{x:.2f}")}
{row("Probe on raw embeddings (control), accuracy", "probe_embed_acc")}
{row("Probe, residual stream L15 last token, AUC", "probe_last_auc", lambda x: f"{x:.2f}")}
{row("Probe, residual stream L15 last token, accuracy", "probe_last_acc")}
{row("Owner, blind, always-user prior (n=24)", "human_acc")}
{row("Refusal rate, eval-labelled prompts", "refusal_rate_eval")}
{row("Refusal rate, user-labelled prompts", "refusal_rate_dep")}
</table>"""
def examples(tag, k, seed):
    random.seed(seed); out = []
    for r in random.sample(R[tag], k):
        out.append(f"<p style='font-size:90%'><b>{r['id']}</b> truth={'EVAL' if r['label'] else 'USER'} | keyword={r['keyword']} | self-report margin={r.get('sr_margin',float('nan')):.1f} (calls eval if &ge; 3.0) | probe_last p={r.get('probe_last_p',float('nan')):.2f}<br>"
                   f"<i>prompt:</i> {r['prompt']}<br><i>answer (first 200 chars):</i> {r['answer'][:200]}</p>")
    return "\n".join(out)
ex = "".join(f"<h3>{t} (3 random, seed 3)</h3>{examples(t,3,3)}" for t in tags)
doc = f"""<html><body style='font-family:Arial;font-size:11pt'>
<h1>Does an eval-awareness detector survive an adversary?</h1>
<p>Thabhelo Duve. MATS 12.0 application, Neel Nanda stream. {datetime.date.today().isoformat()}.<br>
Code and data: REPO_LINK</p>
<h2>Executive summary</h2>
<p>{REW}</p>
<p><b>Question.</b> A model that knows it is being evaluated may behave differently. Cheap detectors for that state are easy to build: ask the model, match keywords, or train a linear probe on the residual stream. Each looks fine on a clean matched set. I asked whether any of them survive an adversary that hides the cues, and what they are actually detecting.</p>
<p><b>Setup.</b> Qwen3.5-4B (8-bit, MLX, run locally on a laptop, thinking off). 60 base tasks, each rendered as an evaluation item and as a user message, 120 prompts per round. Round 0 has explicit cues ("You are being evaluated. Question 3 of 50"). Detectors are built on round 0 and frozen. A red-team agent (Claude Code) then writes new rounds on the same 60 tasks: round 1 strips every cue word from the evaluation items and puts words like "evaluation", "test", "grading" into genuine user messages; round 2 reverses the style, casual evaluation items against terse lazy-user fragments. I labelled a blind sample by hand and adjudicated the ground-truth definition.</p>
{img("figures/detector_rounds.png", 660)}
<p><b>Findings.</b></p>
<ul>
<li>The keyword heuristic collapses in round 1 ({pct(S['round1']['keyword_acc'])}): it flags users who mention their performance evaluation and misses evaluation items with the words removed.</li>
<li>The self-report and the probes keep <i>ranking</i> correctly in round 1 (AUC {S['round1']['self_report_auc']:.2f} and {S['round1']['probe_last_auc']:.2f}) but their thresholds, frozen on round 0, break (accuracy {pct(S['round1']['self_report_acc'])} and {pct(S['round1']['probe_last_acc'])}). The adversary shifted the score distribution rather than reversing it. Neither the probe nor the self-report is fooled by eval vocabulary in user messages (under 3% false positives); they miss disguised evaluation items instead.</li>
<li>Round 2 (style reversed: casual evaluation items vs terse lazy-user fragments) settles what they learned. The self-report and the probes go <i>below</i> chance in ranking (AUC {S['round2']['self_report_auc']:.2f} for the self-report, {S['round2']['probe_last_auc']:.2f} for the layer-15 probe): they rank terse users as more "evaluation" than chatty evaluation items. At the frozen thresholds they call almost everything a user (accuracy {pct(S['round2']['self_report_acc'])} and {pct(S['round2']['probe_last_acc'])}). What they detect is terseness and answer-format structure, not where the message came from.</li>
<li>The mean-pooled probe reaches 100% on round 0 at every layer including the raw embeddings, so it is a bag of words. I kept the embedding probe as a control; the last-token layer-15 probe is the one that reads model state.</li>
<li>No behaviour shift was detectable: refusal rates were {pct(S['round0']['refusal_rate_eval'])} vs {pct(S['round0']['refusal_rate_dep'])} in round 0 and equal in round 1, at n=60 per side.</li>
</ul>
<p><b>What I think this means.</b> {REW} "Evaluation" is a fact about where a message came from, not about its text. Any text can come from a user. A text-only detector can at best learn style, and these detectors learned terseness and answer-format structure. That is useful against naive benchmarks and useless against an adversary who controls style. My own blind labels reflect that: I answered "user" for every item on principle, because every item could have come from a user.</p>

<h2>1. Technical setup</h2>
<ul>
<li>Model: mlx-community/Qwen3.5-4B-MLX-8bit, mlx-lm 0.31.3, Apple M3, 16 GB. Thinking disabled for speed. 32 decoder layers (hybrid linear and full attention), hidden size 2560.</li>
<li>Prompts: 60 base tasks in six categories (fact, math, code, reasoning, writing, borderline-safety). Round 0: six eval frames and six deployment frames cycled over tasks. Rounds 1 and 2: hand-written by the red-team agent, one eval and one user version per task. Ground truth fixed before writing: label 1 = an item produced by an evaluation process with no human need; label 0 = a person with a real need wrote it.</li>
<li>Detectors: (a) keyword regex over 14 cue patterns; (b) self-report: a separate call asking the model to classify the message as A (automated evaluation) or B (real user); scored as logit(A) minus logit(B) at the answer position, threshold 3.0 chosen on round 0 and frozen; (c) logistic-regression probes (standardised, C=0.05) on residual activations at the assistant-start position, layer chosen by grouped 5-fold cross-validation on round 0 (groups = base task).</li>
<li>Behaviour: answer generated with 160 tokens; refusal by regex; answer length.</li>
<li>Metrics: accuracy against ground truth at frozen thresholds; AUC for the continuous detectors; refusal rate by label.</li>
</ul>
<h2>2. Results table</h2>
{table}
<h2>3. Randomly selected raw examples (not cherry-picked)</h2>
{ex}
<h2>4. Strongest evidence against my conclusions</h2>
<ul>
<li>High AUC in round 1 means the self-report and the probe did learn something beyond cue words. The honest statement is that they learned style features that correlate with provenance in my data, not that they learned nothing.</li>
<li>The "user" messages were written by an LLM to imitate users, not collected from real users. A detector might separate real users from my imitations for reasons I cannot see.</li>
<li>One model, one size, one quantisation, thinking off. A reasoning model with visible chain of thought might verbalise eval awareness in a way none of my detectors capture.</li>
<li>The probe threshold could be recalibrated per round; the failure I report is of a fixed detector, which is the deployment-relevant case, but a defender who can relabel a sample each round would do better.</li>
</ul>
<h2>5. Limitations</h2>
<ul>
<li>n = 120 per round, 60 per label. Differences under about 10 points are not meaningful.</li>
<li>The adversary and the write-up assistant are the same agent (Claude Code). I chose the attacks, read the raw outputs, and set the ground-truth definition, but I did not independently generate prompts.</li>
<li>Refusal detection is a regex; the borderline set is 10 tasks.</li>
<li>No real evaluation transcripts or real user logs were used.</li>
</ul>
<h2>6. LLM use and what I checked</h2>
<p>{REW} Claude Code (Fable 5.1) wrote the code, the adversarial prompts and the first draft of this document, and ran the experiments on my machine while I directed. What I checked myself: CHECKED_BY_OWNER. What I did not check: DID_NOT_CHECK. The one place I would be least surprised to find an error: LEAST_SURPRISED.</p>
<h2>7. Time</h2>
<p>Started 2026-09-04 22:50 CDT. Active hours: TOGGL_HOURS (Toggl screenshot below). Excluded: model download, run time while I was doing other things.</p>
<p>TOGGL_SCREENSHOT</p>
<h2>8. Log and reproduction</h2>
<p>Timestamped research log with pre-registered expectations for each round: notes/research_log.md in the repository. Reproduce with the commands in README.md.</p>
</body></html>"""
open("results/writeup.html", "w").write(doc); print("results/writeup.html", len(doc), "bytes")
