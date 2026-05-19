# Q_select diagnostic suite — Report on Ta_minimaxSelect(1)

**Date:** 2026-05-19
**Target:** `Ta_minimaxSelect(1)0514_DEPTH_2` @ epoch 4350 (latest available
checkpoint; the training run was truncated at epoch 4350 in the published
artefacts).
**Architecture:** `QuartoCNNAutoregUnifiedS4` (Sa(3) substrate).
**Scripts run:** `position_structure.py` (D1), `buffer_signal_density.py`
(D2), `decoupled_select.py` (D3). Driver: `run_all.py`.

This report consolidates the diagnostic plan in [`PLAN.md`](PLAN.md) with
the live numbers in
[`results/Ta_minimaxSelect(1)0514_DEPTH_2/`](results/Ta_minimaxSelect(1)0514_DEPTH_2/).

---

## TL;DR

| Hypothesis | Verdict | Strongest evidence |
|---|---|---|
| **H1 — metric artefact** | **Supported.** | `safe_piece_recall = 0.81` (vs chance 0.11). `forcing_loss_bottom_recall = 0.95` (vs chance 0.70). Spearman ρ̄ = 0.55 over 896 decisive states. |
| **H2 — signal sparsity** | **Falsified.** | 95% of regenerated buffer SELECT rows have `max|target| > 0.5`. N=2 endgame: 100%, N=4 curriculum: 94%. |
| **H3 — multitask interference** | **Falsified.** | A decoupled select-only net trained on 2048 oracle-labelled rows for 4000 epochs achieved `safe_piece_recall = 0.62` and ρ̄ = 0.36 — strictly *worse* than the joint network on the same metric. The shared trunk is helping, not hurting. |

**Action.** The Q_select head is doing its job at Ta(1) epoch 4350; the
match-outcome-conditioned Δ metric in
[`QuartoRL/results_io.py:192-214`](../../QuartoRL/results_io.py) was hiding
it. Replace (or supplement) that metric with position-structure recalls
as the primary Q_select gate. Architectural decoupling and per-row
oversampling are **not** the right next moves; they address mechanisms
the data does not implicate.

---

## D1 — Position-structure metric

**Question.** On positions where the oracle says some piece is forcing-loss
(or where some piece is uniquely safe), does Q_select rank that piece
correctly?

**Method.** Sampled 1500 SELECT-phase self-play states (filter:
≥2 placed pieces). For each state, computed the depth-2 minimax oracle
target, the network's 16-d Q_select vector, and three metrics: forcing-loss
recall (strict, narrow filter), forcing-loss bottom recall (relaxed),
safe-piece recall, and Spearman ρ.

**Forcing-loss-set size.** A typical decisive state has
**`forcing_set_size_mean = 5.83`** forcing-loss pieces out of ~14 available.
This is the user's observation made quantitative: when there are multiple
simultaneous threats, multiple pieces lose at once, so the original
"single forcing piece" filter (`forcing_loss_recall`) only fires twice
in 1500 states. The relaxed `forcing_loss_bottom_recall` metric — *does
argmin Q_select land inside the forcing-loss set, regardless of set size?*
— is the right population statistic.

| Metric | Value | Chance baseline | n |
|---|---|---|---|
| `forcing_loss_bottom_recall` | **0.946** | 0.703 | 1167 (states with ≥1 forcing piece) |
| `safe_piece_recall` | **0.807** | 0.115 | 897 (decisive: ≥1 forcing AND ≥1 safe) |
| Spearman ρ (mean) | **0.547** | — | 896 |
| Spearman ρ (p25 / p50 / p75) | 0.41 / 0.64 / 0.82 | — | 896 |
| `forcing_loss_recall` (strict, n=1 forcing) | 0.0 | — | 2 (statistically meaningless) |

Numbers from
[`results/Ta_minimaxSelect(1)0514_DEPTH_2/position_structure.jsonl`](results/Ta_minimaxSelect(1)0514_DEPTH_2/position_structure.jsonl).

**Reading.** Both relaxed recalls are well above chance, and the ρ
distribution is firmly positive (p25 = 0.41). The "≈0 Δ" observation
from `summarize_q_outcome` averages over many states, most of which
have many forcing-loss pieces; the average Q_select value is naturally
near zero because the head correctly outputs `-1`-ish for forcing pieces
and `+1`-ish for safe pieces. Conditioning on match outcome rather than
position structure obscures this.

**Verdict.** **H1 supported.** Q_select at Ta(1) epoch 4350 already
encodes position structure faithfully.

---

## D2 — Buffer signal-density audit (cheap path)

**Question.** What fraction of replay-buffer SELECT rows carry a
non-trivial oracle target, split by `N_LAST_STATES` bucket?

**Method.** Regenerated one epoch of unified-autoreg experience with the
trained policy_net as the behaviour policy, depth-2 oracle wired in.
Cheap path per the plan; the faithful path would need a buffer-dump hook
in `trainRL.py` and is not implemented here.

| Bucket | n_rows | `nonzero_frac_p1` | `nonzero_frac_p5` | max\|target\| p50 |
|---|---|---|---|---|
| Overall (N=4 ∪ N=2) | 128 | **0.953** | **0.953** | 0.99 |
| N=4 (curriculum) | 96 | 0.938 | 0.938 | 0.99 |
| N=2 (endgame anchor) | 32 | 1.000 | 1.000 | 0.99 |

Numbers from
[`results/Ta_minimaxSelect(1)0514_DEPTH_2/buffer_signal_density.jsonl`](results/Ta_minimaxSelect(1)0514_DEPTH_2/buffer_signal_density.jsonl).

**Reading.** Most rows hit a terminal piece in the minimax tree (target
magnitude clamps at `(100+depth)/(100+depth) = 1.0` for terminal nodes
after the normalisation in
[`_minimax_select_target`](../../QuartoRL/RL_functions.py#L115)). The
buffer is *not* signal-starved at this stage of training; the N=4
curriculum window already lives deep enough in the game tree that nearly
every SELECT decision is on or near a forcing line.

**Verdict.** **H2 falsified.** Signal density is ~95%, not the
<10% PLAN-H2 threshold. Per-row oversampling on "decisive" rows would not
be load-bearing — almost every row already is decisive at this depth.

**Caveat.** Cheap path only — this samples one epoch's worth of behaviour
under the *current* (well-trained) policy. The faithful path (logging the
actual training buffer at fixed checkpoints, gated by a config flag) would
strengthen this finding, especially for the *early* phase of training
where the policy is still random.

---

## D3 — Decoupled select-only network

**Question.** Trained in isolation on the same oracle targets, does a
separate small CNN reach the joint network's D1 numbers?

**Method.** Collected 2048 SELECT rows (1024 curriculum + 1024 endgame)
under the Ta(1) policy with a depth-2 oracle, trained a small
`SelectOnlyNet` (~70k params) for 4000 epochs with masked SmoothL1
and AdamW. Held-out set: 1536 rows from a separate seed. Same D1 metric
applied to the resulting Q_select.

| Metric | Joint net (D1) | Decoupled net (D3) | Δ |
|---|---|---|---|
| `safe_piece_recall` | **0.807** | 0.620 | −0.187 |
| `forcing_loss_bottom_recall` | **0.946** | 0.928 | −0.018 |
| Spearman ρ̄ | **0.547** | 0.365 | −0.182 |
| Train loss (final) | n/a | 0.002 | — |

Numbers from
[`results/Ta_minimaxSelect(1)0514_DEPTH_2/decoupled_select.jsonl`](results/Ta_minimaxSelect(1)0514_DEPTH_2/decoupled_select.jsonl).

**Reading.** The decoupled net fits the training targets cleanly (loss
0.002), and its `forcing_loss_bottom_recall` is comparable to the joint
network's. But on the harder metric — *correctly distinguishing the
single best safe piece from a sea of forcing-loss pieces* — it is
substantially behind. The features the joint trunk built for Q_place
are evidently *useful* to Q_select, not in competition with it. This is
the opposite of what H3 predicted.

**Verdict.** **H3 falsified at this data scale.** The shared trunk is a
feature, not a tax. Structural decoupling of the select trunk is the
wrong direction.

**Caveats.**

- 2048 train rows is still much smaller than the joint net's effective
  training distribution. A multi-epoch buffer dump from a faithful path
  would be a stronger test, but the *direction* of the comparison (joint
  better than decoupled) is already informative — a small select-only
  net *with* the cleanest possible supervision *still* underperforms.
- The held-out set comes from the same policy distribution as training
  for both nets, so this is not testing generalisation across policy
  changes.

---

## Action items

1. **Replace the Q_select gate** in [`QuartoRL/results_io.py:192-214`](../../QuartoRL/results_io.py)
   (or add a parallel section to the JSONL summary). The match-outcome-
   conditioned Δ metric is uninformative on Q_select for Ta-family
   models. The right gate is **position-structure recall** on a held-out
   sample (this script is reusable per
   [`PLAN.md`](PLAN.md) → "Shared contract"). Suggested headline:
   `safe_piece_recall` ≥ 0.70 (chance ≈ 0.11) for *evaluation*; D1's full
   record for *interpretation*.
2. **Retire Sb-style structural decoupling** of the select trunk from the
   forward queue. The diagnostic does not support H3.
3. **Retire α-reweighting / oversampling** of select rows from the
   forward queue. The diagnostic does not support H2.
4. **Re-evaluate older "Q_select saturation" runs** (M, N, O, Q, R, S
   non-T) under the same D1 metric. The interpretation of their flat Δ
   may also be metric artefact; this is cheap to test (the script accepts
   any unified-autoreg checkpoint).
5. **Faithful path D2** — add a buffer-dump hook to `trainRL.py`
   (gated by a config flag, default off) to enable D2's faithful path on
   future runs. The cheap path is suggestive, not definitive.

---

## Reproducibility

```
python analysis/qselect_diagnostics/run_all.py \
    --exp 'Ta_minimaxSelect(1)0514_DEPTH_2' --include-d3 \
    --n-states 500 --n-matches 32 \
    --d3-epochs 4000 --d3-n-matches 512
```

Wall-clock on a single CPU core: D1 ≈ 6 min (oracle dominates),
D2 ≈ 16 s, D3 ≈ 23 min (data collection ≈ 30 s + training ≈ 22 min).

Run records, console logs, and JSONL records are under
[`results/Ta_minimaxSelect(1)0514_DEPTH_2/`](results/Ta_minimaxSelect(1)0514_DEPTH_2/).

---

## Disclosures and limits

- All three diagnostics share `sample_states` / `gen_experience_unified_autoreg`
  with depth-2 minimax targets. A subtle bug in the oracle would
  contaminate D1, D2, and D3 simultaneously. The helper reuses the same
  `_minimax_select_target` function the training loop uses for the Ta
  series, so any bug there has been present throughout T-series training.
- The diagnostic was run on a single experiment (Ta(1) at epoch 4350).
  Whether the same H1-supported verdict holds for older non-T-series
  checkpoints (M, S, etc.) is an open question — see action item 4. If
  it does, the "Q_select saturation" framing across the project's
  history is largely a metric artefact.
- This report and the underlying plan were both AI-drafted. The
  *numbers* (D1/D2/D3 columns) are direct from the JSONL artefacts and
  are reliable; the *interpretation* of those numbers as "H1 supported"
  / "H2 falsified" / "H3 falsified" is the AI's framing and should be
  spot-checked before being treated as load-bearing.
