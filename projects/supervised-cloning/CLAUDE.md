# CLAUDE.md — projects/supervised-cloning

This file provides guidance to Claude Code (claude.ai/code) for the
**supervised-cloning** sub-project. The objective is to imitate
`bot/minimax_bot.py::MinimaxBot` (αβ minimax teacher) with a neural
network via supervised cross-entropy on collected teacher decisions.

The parent repo is RL-focused (DQN self-play). This sub-project is
**disjoint in scope** — no Q-values, no replay buffer, no Bellman target —
just `(board, piece) → (place_label, select_label)` supervised
classification. Only `bot/minimax_bot.py` and the architecture in
`models/CNN1.py` are shared with the parent.

## Where to read what

- **`train.py` docstring** — CLI source of truth: flags, augmentation
  policy, loss composition, train/val split policy. Read this before
  changing the training script.
- **`collect_data.py` docstring** — data collection: opponent mix, teacher
  depth, output schema. Read this before changing the dataset format.
- **`experiments/<name>/summary.md`** — per-run results (config, dataset
  sizes, top-1 / top-3 accuracies, training curves). The trainer writes one
  automatically when `--exp <name>` is passed.
- **Parent `CLAUDE.md`** (repo root) — shared infrastructure (model
  classes, BotAI contract, mode_2x2). Conventions there apply *only when
  shared code is touched*; supervised cloning has its own naming and
  output conventions documented below.

## Operational rules

These are non-obvious contracts that span files or that have already
bitten us. They live here because they're load-bearing for code changes.

- **Symmetry augmentation needs separate forward and inverse permutation
  tables.** `_POS_PERMS_FWD` (old_pos → new_pos) is for PLACE labels;
  `_POS_PERMS_INV` (new_pos → old_pos) is for PLACE legal masks consumed
  via gather indexing (`mask[:, perm_inv]`). Conflating them silently
  mislabels 7/8 of every augmented copy. The pre-2026-05 single-table
  version of this code was buggy; do not revert.
- **Augmentation inverse order matters: flip-then-CW, not CW-then-flip.**
  The forward pipeline applies `rot90_CCW^k` then `flip_cols`. The
  inverse is `rot90_CW^k ∘ flip` — i.e. invert the flip *first*, then
  invert the rotations. Reversing the order silently mislabels 4/8 of
  every augmented copy (transforms t ∈ {4,5,6,7}). Pre-2026-05-13 code
  had this bug; A1/B1 experiments inherit conservative PLACE numbers as
  a result. Guarded by
  `tests/test_symmetry_augmentation.py::test_place_label_follows_board_rotation`.
- **Mask transformation applies only to PLACE samples.** SELECT legal
  masks are 16-d piece-availability masks and are *rotation-invariant*.
  Applying a board-position permutation to a SELECT mask scrambles which
  pieces are marked legal and corrupts the CE loss for the affected rows.
  `augment_symmetries` must filter on `actions == ACTION_PLACE` before
  applying the perm.
- **Train/val split is at the GAME level, not the SAMPLE level.** Adjacent
  positions in the same game are highly correlated; sample-level splits
  leak ~all of train into val. `load_split` already does this — preserve
  it.
- **Illegal moves are masked from the logits before CE loss.** Set logits
  at illegal positions to a large negative value before `log_softmax`.
  Without this, gradients flow through invalid actions and the model
  learns to put non-zero probability on illegal moves.
- **`mode_2x2=True` is the default** when collecting data, matching the
  parent repo's evaluation convention. Mixing 2×2 and non-2×2 datasets is
  a silent inconsistency.
- **The teacher is `MinimaxBot(depth=2)`.** Verified at 98% WR vs the
  parent repo's `bot_loss-BT` baseline (commit `b788e47`). Higher depths
  exist but are slower; lower depths are weaker. Don't change the default
  teacher without re-validating the WR.
- **Data files generated before any train.py augmentation fixes are still
  valid.** `collect_data.py` does not touch augmentation — the data on
  disk is correct. After a train-script bug fix, re-train, do not
  re-collect.
- **CLI scripts `chdir` to the project root** (the dir containing `bot/`)
  before parsing paths. Output paths in flags are therefore relative to
  the project root, not to `projects/supervised-cloning/`. Don't break
  this when refactoring path handling.

## Run vs experiment — output convention

The `train.py` CLI distinguishes two output modes:

| Mode | Trigger | Output dir | What's written |
|---|---|---|---|
| **Ad-hoc / smoke run** | default (no `--exp`) | `checkpoints/` (or `--out <path>`) | `best.pt`, `final.pt`, `training_curves.png` only |
| **Tracked experiment** | `--exp <name>` | `experiments/<name>/` | the above **plus** auto-generated `summary.md` |

Use ad-hoc runs for debugging, smoke tests, and exploratory tweaks —
output is overwritten on the next run. **Treat `checkpoints/` as scratch.**
Promote a run to a named experiment as soon as the result is meant to be
compared against another run or referenced from elsewhere.

`*.pt` is gitignored repo-wide, so checkpoint binaries never make it into
git by default. Force-add (`git add -f`) only the experiment artifacts
that document a published result; never force-add ad-hoc `checkpoints/`
binaries.

## Experiment naming

`<series_letter><index>_<descriptor>` — e.g. `A1_baseline_cnn`,
`A2_lambda_sweep`, `B1_uncoupled`.

- **Series letter** bumps when a *condition* changes: architecture, loss
  formulation, teacher type, dataset source. Sweeping a hyperparameter
  does *not* advance the letter.
- **Index** advances within a series (1, 2, 3, ...).
- **Descriptor** identifies the variable being swept or the distinguishing
  feature of the run.

Note: the parent repo's RL series uses a two-letter scheme (`Aa`, `Ab`,
...). This sub-project deliberately uses letter-plus-digit to make the
two namespaces visually distinct in commit messages and registry tables.

# Regenerate the standard 5k-game supervised dataset (deterministic given seed):
```bash
python projects/supervised-cloning/collect_data.py -g 5000 -d 2 --seed 42 \
    -o projects/supervised-cloning/data/collected_5k.npz
```
## Adding a new experiment

1. Make sure `train.py` is **committed at the state you want to use**. If
   you have uncommitted changes, commit them first — experiment artifacts
   must reference reproducible code.
2. Pick a name following the convention above.
3. Run with `python projects/supervised-cloning/train.py --exp <name> …`.
4. After the run, edit `experiments/<name>/summary.md` to add a
   `## Notes` section: what hypothesis the run tested, what you learned,
   whether the numbers warrant a follow-up. The auto-generated tables are
   fine; the notes are what give the artifact context months later.
5. If the run shifts the research direction (new SOTA, a hypothesis
   confirmed/falsified, a bug found), also note it in a project-level
   status doc — to be created at `projects/supervised-cloning/Research-status.md`
   when the third or fourth experiment lands; for now the per-run
   `summary.md` is the canonical record.

## Things that have bitten this sub-project before

- **Symmetry-augmentation forward/inverse confusion** (fixed 2026-05-08).
  See operational rule above. The pre-fix code corrupted PLACE labels
  *and* SELECT legal masks under 7/8 of the augmentations; experiments
  produced before the fix should be considered untrusted and re-run.
- **Symmetry-augmentation flip-vs-rotation inverse order** (fixed
  2026-05-13). Independent of the 2026-05-08 fix above. `_pos_inv`
  composed flip *after* CW rotation when inverting transforms
  t ∈ {4,5,6,7}; the correct order is flip first, then CW. The bug
  mislabeled PLACE targets, PLACE legal masks, and (once added) PLACE
  soft-target distributions in 4/8 of every augmented copy. A1, B1_bc,
  and B1_dagger_diverse were all trained with the buggy table; their
  PLACE-head metrics understate what the same data + corrected
  augmentation would have produced. SELECT samples are unaffected.
- **`checkpoints/` as a tracked artifact area** (cleaned 2026-05-08). The
  initial layout used `checkpoints/` for both ad-hoc runs and the only
  named run, blurring the run/experiment distinction. The current
  convention restricts `checkpoints/` to scratch and routes named runs to
  `experiments/<name>/`.

## Style

- The CLI scripts in this subproject use **docopt-style module
  docstrings** as the source of truth for arguments — keep the docstring
  in sync if you add or rename options. The trainer also reads `--exp`
  / `--out` from the docstring; renaming or restructuring those flags
  requires updating both the docstring and the dispatch in `main()`.
