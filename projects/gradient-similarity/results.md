# Gradient-similarity diagnostic — results

Tracks per-checkpoint measurements of `cos(grad(L_place), grad(L_select))` and per-head
gradient norms on the shared trunk. Motivated by Nexus (arXiv:2604.09258): if the
two heads' gradients are misaligned at convergence, the "distant minima" geometry
the paper warns about is present and a Nexus-style regularizer is in scope. If
gradients are already aligned — or if one of them has near-zero norm — Nexus is
not the right fix.

See [`measure.py`](./measure.py) for the script and CLI.

## Methodology (constant across runs unless noted)

- Generate a fresh self-play experience pool from the loaded policy.
- Sample `n_batches` independent batches; on each, compute `L_place` and
  `L_select` separately via `DQN_training_step` (`mc_select` for joint schema:
  Bellman target for place, Monte Carlo return for select).
- For each batch, compute `grad(L_place)` and `grad(L_select)` w.r.t. policy
  parameters; report cosine and norms broken down by parameter group:
  - **trunk**: `fc_in_piece`, `conv1`, `conv2`, `fc1` — the shared backbone
    where the multi-task gradient story actually plays out.
  - **all**: full parameter vector.
  - **place_head** / **select_head**: the independent final layers. With
    `QuartoCNN_uncoupled` each head receives gradient from only its own loss,
    so cross-cosine on a head subset is identically `null` (sanity check).
- Aggregate as mean ± std across batches.

## Runs

### baseline (loss_BT) — `LOSS_APPROACHs_1212-2_only_select` E=1034

**Note on label.** The folder name says `only_select` but per project owner this
checkpoint is mislabeled — it was actually trained with `combined` or `only_place`
(the `only_select` line of work is the open Q_select-collapse problem, not a
shippable baseline; and `place` is the verified-working head).

- Architecture: `QuartoCNN_uncoupled`, schema: `joint`
- Diagnostic loss-approach: `mc_select` (Bellman for place, MC return for select)
- `--matches 64 --n-last-states 6 --n-batches 16 --batch-size 64`
- Experience pool: 383 transitions

| metric                      | mean   | std    |
| --------------------------- | -----: | -----: |
| `cosine` — trunk            | +0.252 | 0.342  |
| `cosine` — all params       | +0.231 | 0.318  |
| `L_place`                   |  0.661 |   —    |
| `L_select`                  |  0.426 |   —    |
| `grad_norm_place` — trunk   |  2.208 | 0.887  |
| `grad_norm_select` — trunk  |  0.0079| 0.0091 |
| `grad_norm_select` — head   |  0.0018| 0.0022 |

Raw JSON: [`results/baseline_loss_BT.json`](./results/baseline_loss_BT.json).

#### Findings

1. **Trunk cosine is mildly positive, not antagonistic** (+0.25). The two heads
   are not pulling the trunk in opposite directions — there is no "sum-of-minima"
   pathology in the Nexus sense. Cosine is "moderate-close" rather than
   "intersection-close", so in principle there is *some* room for a Nexus-style
   regularizer to tighten alignment, but the gap is not the dominant problem.

2. **`grad(L_select)` is ~280× smaller than `grad(L_place)` on the trunk**
   (0.0079 vs 2.21). This is the headline. The select head's tanh outputs are
   saturated, so its loss has near-vanishing derivative w.r.t. the trunk. Even
   if cosine were perfect, the select-side update would be ~0 — Nexus's
   regularizer cannot align a vector whose norm is essentially noise.

3. **Cosine variance dominates the mean** (std 0.34 vs mean 0.25). With one of
   the gradients this small, the cosine *direction* is dominated by noise. The
   cosine measurement on this checkpoint should be read with low confidence; the
   gradient-norm asymmetry is the more reliable signal.

#### Interpretation w.r.t. the Nexus question

- The Q_select collapse on this checkpoint is a **gradient-magnitude problem,
  not a gradient-direction problem**. Nexus operates on direction (cosine), so
  it is not on the critical path for fixing the saturation.
- **Anything that revives select-head gradient magnitude is upstream of Nexus.**
  The `mc_select` line of work (Monte Carlo target, no self-bootstrap through
  Q_select) is the right kind of intervention — it directly attacks the
  saturation feedback loop. Once select grad norm reaches the same order of
  magnitude as place, the cosine measurement becomes meaningful and Nexus
  becomes worth considering.
- Practical pre-condition for retesting Nexus: a checkpoint where
  `grad_norm_select / grad_norm_place > ~0.1` on the trunk. Below that, cosine
  is in the noise floor.

#### Caveats

- Single checkpoint, late-training (E=1034). Says nothing about cosine dynamics
  over training — early epochs may look very different. Sweeping a full
  trajectory (`--dir`) would clarify whether trunk cosine drifts as Q_select
  saturates.
- `mc_select` was used as the diagnostic loss. With `separate_bellman` instead,
  `L_select` would bootstrap through the saturated Q_select itself and would
  produce even smaller gradients — the asymmetry would be larger, not smaller,
  so the qualitative conclusion is robust.
- Cosine is computed on `SmoothL1` per-batch losses, not population-averaged.
  The Nexus paper reports population-averaged cosine across discrete data
  sources; the analogue here is "across mini-batches", which is what Algorithm 3
  of the paper uses, so this matches.

## Open follow-ups

- Sweep across an `mc_select` training trajectory (e.g. `LB_mcSelect(*)` once
  per-epoch checkpoints exist) to see whether select grad norms recover and at
  what point trunk cosine becomes informative.
- Compare against an `only_place` baseline of similar training duration: cosine
  should be undefined (select head untrained), and place-side norms set the
  upper bound for what "healthy" gradient magnitude looks like.
- Decoupled-autoreg variant: re-run with `--schema decoupled_autoreg --arch
  QuartoCNNAutoreg` on a recent autoreg checkpoint. The phase-separated batches
  should give a cleaner cosine measurement (no masking inside the batch).
