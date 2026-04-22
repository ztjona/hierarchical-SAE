# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Where to read what

- **`readme.md`** — project overview, two-headed CNN architecture, experience tuple schema, reward functions, loss approaches, training loop, quick-start commands, project layout. **Read this first.**
- **`Research-status.md`** — every experiment, its hypothesis, fixed params, result, and post-mortem; the "Key Takeaways" and "Current Open Problem" sections are the source of truth for what has already been tried and why it failed. **Consult before proposing new experiments or interpreting results.**

Do not restate content from those files here — link to the relevant section instead.

## Operational rules

These are non-obvious contracts that span files or that have already bitten us. They live here because they're load-bearing for code changes, not because they're useful prose for a human reader.

- **Don't reintroduce an adversarial sign flip in the Bellman step while `REWARD_FUNCTION="propagate"`.** The rewards already encode player perspective; double-negation diverges Q as `R/(1−γ)`. See `Research-status.md` → `FA_Bellman` post-mortem.
- **Q_select saturation is the standing open problem.** Across every experiment to date the select head collapses to −1; Q_place carries performance. Before "fixing" select-head behaviour, read `Research-status.md` → `Current Open Problem` and the candidate-fix table.
- **Curriculum fine-tuning from N=2 → larger N catastrophically forgets.** Don't assume a gentler LR/TAU will save it — the root cause is replay-buffer distribution shift. See `Research-status.md` → `Ac_fineShallow`.
- **The `-1` sentinels in the experience tuple (`action_place=-1` first move, `action_sel=-1` terminal) are load-bearing.** `DQN_training_step` asserts they never co-occur and that `action_sel=-1` implies `done=True`. Don't silently filter them out of the buffer.
- **Place and select are independent action spaces.** Take `max` over each head independently, never over averaged logits (this was the correct half of commit `51b59ba`).
- **`mode_2x2=True` is the default across training and evaluation.** Keep it consistent — a bot trained with 2×2 wins against a no-2×2 baseline isn't a valid comparison.
- **Pickled results may contain CUDA tensors.** Use the `CPUUnpickler` pattern in `tools/view_qv.py` when loading `CHECKPOINTS/<exp>/<exp>.pkl` on a CPU-only machine.

## Adding a new experiment

- Follow the naming scheme in `Research-status.md` → `Experiment Naming Convention`: first letter bumps on code/algorithm changes, second letter on hyperparameter sweeps within that code version. `run_trains.py` appends the `(<idx>)<MMDD>_<PARAM>_<VALUE>` suffix.
- When a sweep concludes, append a new section to `Research-status.md` (Question → table → Fixed → Result → Conclusion). If it shifts the research direction, also update `readme.md`'s pointer to the current status.
