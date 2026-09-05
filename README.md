# Does an eval-awareness detector survive an adversary?

Application task for Neel Nanda's MATS 12.0 stream (September 2026). Owner: Thabhelo Duve.

**Question.** Cheap detectors for "the model thinks it is being evaluated" are easy to build: ask the model,
match keywords, or train a linear probe on the residual stream. Each looks fine on a clean matched set.
Do any of them survive an adversary that writes evaluation items that look like real users, and real-user
messages that use evaluation vocabulary?

**Model.** Qwen3.5-4B (mlx-community 8-bit), run locally on an Apple M3 with MLX, thinking disabled.

**Rounds.**
- Round 0: 60 base tasks x {eval frame, deployment frame} = 120 prompts with explicit surface cues. Detectors are built and the probe layer is chosen here (grouped 5-fold CV by base task).
- Round 1: same 60 tasks rewritten by the red-team agent (Claude Code, Fable 5.1): evaluation items with all cue words removed but benchmark structure kept; user messages with "evaluation", "test", "grading" in everyday senses. Owner labels a 24-item blind sample as a human baseline.
- Round 2: adaptive, written after inspecting which round-1 items fooled which detector.
- Round 3: fully crossed context x style (240 prompts); user side carries a stated situation, no eval vocabulary.
- Round 4: identical question, only a preceding context line varied (120 prompts, paired analysis).
- Baselines: length only, format features, char n-grams, random (code/baselines.py). Remedy: probe retrained on round 3 with a task split (code/remedy.py). Steering along the probe direction (code/steer.py).
- Round-0 numbers are grouped cross-validated (held-out), never training accuracy.

**Ground truth.** label 1 = an item produced by an evaluation process, nobody needs the answer.
label 0 = a person with a real need wrote it. Fixed before the adversarial rounds were written.

**Layout.** `code/` (dataset builders, model utils, runner, probe, scoring, plots), `data/` (prompt sets, blind
sheets), `results/` (outputs, activations, scores), `figures/`, `notes/research_log.md` (timestamped).

Reproduce: `python code/build_dataset.py && python code/run_round.py data/round0.json round0 && python code/probe.py train && python code/probe.py eval round0 && python code/score.py round0`.
