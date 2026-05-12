# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Where to read what

- **`readme.md`** — project overview, two-headed CNN architecture, experience tuple schema, reward functions, loss approaches, training loop, quick-start commands, project layout. **Read this first.**
- **`Research-status.md`** — *slim* ledger (~120 lines): champion, current open problem, code-version letter index, last few results, forward queue. **Read first when deciding what to run next or interpreting results.**
- **`docs/diary/series-<letter>.md`** — per-series long-form: every experiment in that code-version letter, its hypothesis, fixed params, result, and post-mortem. Linked from `Research-status.md`'s series index. **Consult when reasoning about a specific experiment family.**
- **`docs/diary/<date>_<topic>.md`** — design notes (new schemas, new model classes, new auxiliary losses). Index in `docs/diary/README.md`. **Consult before modifying a `TRANSITION_SCHEMA`, model class, or bot inference rule.**
- **`tools/results_compare.py`** — CLI over the JSONL summaries at `results/<series>/<exp>.summary.jsonl` (`CHECKPOINTS/` stays binary-only). Use `list`, `show <exp>`, `diff <a> <b>`, `rank --metric KEY`. Authoritative source for cross-experiment numbers (the markdown files describe *interpretation*, the JSONL stores *values*).

Do not restate content from those files here — link to the relevant section instead.

## Operational rules

These are non-obvious contracts that span files or that have already bitten us. They live here because they're load-bearing for code changes, not because they're useful prose for a human reader.

- **Don't reintroduce an adversarial sign flip in the Bellman step while `REWARD_FUNCTION="propagate"`.** The rewards already encode player perspective; double-negation diverges Q as `R/(1−γ)`. See `docs/diary/series-FG.md` → `FA_Bellman` post-mortem.
- **Q_select saturation is the standing open problem.** Across every experiment to date the select head collapses to −1; Q_place carries performance. Before "fixing" select-head behaviour, read `Research-status.md` → `Current Open Problem` and `docs/diary/series-P.md` / `series-Q.md` for the latest triangulation.
- **Curriculum fine-tuning from N=2 → larger N catastrophically forgets.** Don't assume a gentler LR/TAU will save it — the root cause is replay-buffer distribution shift. See `docs/diary/series-A.md` → `Ac_fineShallow`.
- **The `-1` sentinels in the experience tuple (`action_place=-1` first move, `action_sel=-1` terminal) are load-bearing.** `DQN_training_step` asserts they never co-occur and that `action_sel=-1` implies `done=True`. Don't silently filter them out of the buffer.
- **Place and select are independent action spaces.** Take `max` over each head independently, never over averaged logits (this was the correct half of commit `51b59ba`).
- **`mode_2x2=True` is the default across training and evaluation.** Keep it consistent — a bot trained with 2×2 wins against a no-2×2 baseline isn't a valid comparison.
- **Pickled results may contain CUDA tensors.** Use the `CPUUnpickler` pattern in `tools/view_qv.py` when loading `CHECKPOINTS/<exp>/<exp>.pkl` on a CPU-only machine.
- **`TRANSITION_SCHEMA` is a triplet contract** (schema name × model class × bot class). The valid combinations are documented at the top of `trainRL.py`. `joint` uses `QuartoCNN*` + `Quarto_bot`; `decoupled_autoreg` uses `QuartoCNNAutoreg*` + `Quarto_autoreg_bot`; `unified_autoreg` uses either (`QuartoCNNAutoregUnified*` + `Quarto_unified_bot`) for the OA-series mask-enabled inference, OR (`QuartoCNNUnifiedNoMask` + `Quarto_unified_nomask_bot`) for the QC-series no-mask inference + aux legality head. Mixing a model from one row with a bot/schema from another will silently produce wrong aux semantics — no exception, just bad gradients. The `unified_autoreg` schema reuses the `DQN_training_step_decoupled_autoreg` target machinery, so target rules (`DECOUPLED_TARGET_STYLE`) apply to all autoreg schemas. The QC training loop additionally computes a per-batch `BCEWithLogits` legality loss via `policy_net.legality_logits(state_board, state_aux)` against `legality_target_from_board(state_board)` and adds `λ_legality · L_legality` to `L_DQN` before `backward()`.

## Adding a new experiment

- Follow the naming scheme in `Research-status.md` → `Experiment Naming Convention`: first letter bumps on code/algorithm changes, second letter on hyperparameter sweeps within that code version. `run_trains.py` appends the `(<idx>)<MMDD>_<PARAM>_<VALUE>` suffix.
- When a sweep concludes:
  1. The JSONL summary at `results/<series>/<exp>.summary.jsonl` is written automatically by `trainRL.py`. Backfill older runs with `tools/pkl_to_jsonl.py` (binaries stay under `CHECKPOINTS/<exp>/`; JSONL + plots + per-series notes live under `results/<series>/`).
  2. Append a "Result" subsection to the relevant `docs/diary/series-<letter>.md` (Hypothesis → Fixed params → Decision gate → Result → Conclusion).
  3. Update `Research-status.md` if the result changes the champion, the open problem, or the forward queue. Keep that file under ~150 lines.
