# Supervised cloning of MinimaxBot

Sub-project that trains a CNN to imitate `MinimaxBot` (the project's
ground-truth strong baseline) via behavior cloning (BC), with an optional
DAgger iteration to close the BC distribution-shift gap.

The architecture and game representation are the same as the rest of the
repo — see the project root `readme.md` for the two-headed CNN, board
encoding, and 2×2 win mode. This README only covers what's specific to
supervised cloning.

## Pipeline

```
                 +-----------------+    .npz    +----------+   .pt   +-----------------+
  opponent mix → | collect_data.py | ─────────> | train.py | ──────> | clone_benchmark |
                 +-----------------+            +----------+         +-----------------+
                          ▲                          │
                          │            best.pt       │
                          │  +-----------------+     │
                          └──| dagger_collect  |<────┘   (DAgger iteration: optional)
                             +-----------------+
```

1. **`collect_data.py`** — plays games between `MinimaxBot(depth=d)` and a
   mix of opponents (`random`, `minimax_d1..d3`, `loss_BT`). Records every
   teacher decision into an `.npz` with the schema:
   - `boards (N,16,4,4) f32` — board one-hot
   - `pieces (N,16) f32` — selected-piece one-hot (zeros for SELECT turns)
   - `labels (N,) i16` — teacher's chosen index (board pos for PLACE, piece
     idx for SELECT)
   - `actions (N,) u8` — `0=PLACE`, `1=SELECT`
   - `legal_masks (N,16) bool` — legal-action mask at the recorded state
   - `game_ids (N,) i32` — per-game id, used for game-level train/val split

2. **`train.py`** — masked cross-entropy on both heads, joint loss
   `L = L_place + λ·L_select`. Train/val split is *by game_id* (not sample),
   and 8× dihedral symmetry augmentation runs on the training split only.
   Accepts one or more `.npz` files via `--data a.npz,b.npz` — game ids are
   offset across files so concatenation is safe.

3. **`clone_benchmark.py`** — loads `best.pt` (or `final.pt`) from an
   experiment dir and plays 100 games against `random`, `minimax_d2`, and
   `loss-BT`. Win rate is `(wins + 0.5·draws)/total`, split P1 / P2.

4. **`dagger_collect.py`** — closes the BC coverage gap. Plays games where a
   trained clone is one side and an opponent (loss-BT, minimax_d2, ...) is
   the other, then records every state the *clone* visited, labeling it
   with the move `MinimaxBot(depth=d)` would have chosen at that state.
   Output schema matches `collect_data.py`, so the file is a drop-in second
   `--data` argument for `train.py`.

## Quick start

```bash
# 1) Behavior-clone a teacher on the default opponent mix.
python projects/supervised-cloning/collect_data.py -g 5000 -d 2 \
    -o projects/supervised-cloning/data/collected_5k.npz
python projects/supervised-cloning/train.py --exp A1_baseline_cnn --epochs 150

# 2) Benchmark.
python projects/supervised-cloning/clone_benchmark.py A1_baseline_cnn --matches 100
```

## Experiments

### Naming convention

`<letter><digit>_<short_description>` — first letter bumps on a code or
algorithm change, digit on a hyperparameter sweep within that version.
Experiment outputs land under `experiments/<name>/`:

- `best.pt`, `final.pt` — checkpoints (best = highest avg val acc)
- `training_curves.png` — loss and per-head top-1 / top-3 accuracy
- `summary.md` — config, dataset stats, final metrics, win rates

### Adding an experiment

1. Either re-use `collected_5k.npz` or generate a new data file (different
   teacher depth, different mix).
2. Decide on the experiment name following the convention above.
3. Run `train.py --exp <name>`.
4. Append the result to `Research-status.md` if it shifts the research
   direction.

### A1_baseline_cnn

Vanilla BC against `MinimaxBot(depth=2)` with the default mix
(`random:0.3, minimax_d1:0.3, minimax_d2:0.4`), `QuartoCNN` (coupled
place→select), CE loss, `λ=1.0`. **Findings:** the model reaches ~27 % val
PLACE top-1 and ~14 % val SELECT top-1 — low only because most teacher
labels are tied (minimax_d2 returns score = 0 on most non-terminal
positions and tie-breaks by iteration order). In play the clone agrees
with the teacher ~94 % of the time *on canonical d2-vs-d2 lines*, but only
~56 % on states induced by loss-BT. Win rates with this checkpoint:
`random ≈ 54 %, minimax_d2 ≈ 0 %, loss-BT ≈ 40 %`. The 0 % vs minimax_d2 is
structural (deterministic self-play with a deterministic punisher); the 40 %
vs loss-BT reveals a **distribution-shift / coverage** problem, since
minimax_d2 itself beats loss-BT 96 %.

### B1_dagger_diverse

Single-iteration DAgger on top of a diversified BC dataset. Two changes
relative to A1:

1. **Diversified collection mix** — adds `loss_BT` (the strong CNN
   baseline) to the opponents the teacher plays against, so the BC
   training distribution covers states loss-BT induces.
2. **One DAgger pass** — train an interim clone, run it against the
   benchmark-time opponents, record every state the clone visits, label
   each with `MinimaxBot(depth=2)`'s move, and retrain on the union.

Run the four stages manually:

```bash
# (1) Diversified BC collection (slow)
python projects/supervised-cloning/collect_data.py -g 5000 -d 2 \
    --opponent-mix "random:0.15,minimax_d1:0.2,minimax_d2:0.35,loss_BT:0.3" \
    -o projects/supervised-cloning/data/B1_diverse.npz

# (2) Interim BC clone (slow)
python projects/supervised-cloning/train.py --exp B1_dagger_diverse_bc \
    --data projects/supervised-cloning/data/B1_diverse.npz \
    --epochs 150 --no-eval

# (3) DAgger collection using the interim clone (slow)
python projects/supervised-cloning/dagger_collect.py B1_dagger_diverse_bc -g 2000 -d 2 \
    --opponent-mix "minimax_d2:0.4,loss_BT:0.4,minimax_d1:0.1,random:0.1" \
    -o projects/supervised-cloning/data/B1_dagger.npz

# (4) Final training on combined data (slow)
python projects/supervised-cloning/train.py --exp B1_dagger_diverse \
    --data projects/supervised-cloning/data/B1_diverse.npz,projects/supervised-cloning/data/B1_dagger.npz \
    --epochs 150

# (5) Benchmark
python projects/supervised-cloning/clone_benchmark.py B1_dagger_diverse --matches 100
```

## Operational notes

- `mode_2x2=True` is the project default; keep it consistent across
  collection, training, and benchmark.
- `Quarto_bot(deterministic=False, temperature=0.1)` is *effectively
  deterministic* — `softmax(tanh(x)/0.1)` gives the argmax ≈ 99.99 % mass.
  If you want real exploration, use `T ≥ 1.0`.
- The teacher's labels at depth 2 are mostly tied (score = 0 for any
  non-terminal move). Low val top-1 doesn't mean a bad model — track top-3
  as well, or evaluate via win rate.

### Known issues / retroactive fixes

- **Symmetry-augmentation flip-ordering bug (fixed 2026-05-13).** The
  `_pos_inv(idx, t)` table in `train.py` previously composed the column
  flip *after* the CW rotation when inverting transforms t ∈ {4,5,6,7}.
  The forward pipeline applies `rot90_CCW^k` then `flip_cols`, so the
  inverse must apply *flip first*, then CW^k. The buggy order silently
  mislabeled PLACE targets, PLACE legal masks, and (in C-series and
  beyond) PLACE soft-target distributions in 4 of every 8 augmented
  copies of every PLACE sample. SELECT samples are unaffected
  (piece-indexed). This was not the same bug as the pre-2026-05-08
  "forward/inverse confusion" fix noted in `CLAUDE.md`; both flaws
  coexisted until now.
  - **Impact on prior experiments.** `A1_baseline_cnn`,
    `B1_dagger_diverse_bc`, and `B1_dagger_diverse` were all trained
    with the buggy table. Their PLACE-head numbers are conservative;
    SELECT-head numbers stand. Headline win-rate deltas from `B1` vs
    `A1` should be re-evaluated alongside C-series runs (which use the
    corrected table) before being treated as a clean DAgger effect.
  - **Detection.** A unit test
    (`tests/test_symmetry_augmentation.py::test_place_label_follows_board_rotation`)
    now asserts that after each of the 8 D4 transforms, the post-rotation
    PLACE label points to the cell where the piece actually ended up on
    the rotated board.

## File map

| File | Purpose |
|---|---|
| `collect_data.py` | Generate BC dataset from teacher vs mixed opponents |
| `dagger_collect.py` | Generate DAgger dataset from clone vs mixed opponents |
| `train.py` | Train clone (masked CE, joint loss, symmetry-augmented) |
| `clone_benchmark.py` | Evaluate a checkpoint against random / minimax_d2 / loss-BT |
| `experiments/<name>/` | Per-experiment checkpoints, curves, and `summary.md` |
| `data/*.npz` | Collected datasets |
