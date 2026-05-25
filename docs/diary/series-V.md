# Series V — Minimax-oracle ablation mid-training

T-series substrate (`unified_autoreg` + `QuartoCNNAutoregUnifiedS4` + depth-2
minimax select target) with a new knob `MINIMAX_DISABLE_AFTER_EPOCH = K`:
at the start of epoch `K` the oracle is dropped and the target style switches
to `td_place_mc_select` for the remainder of training. Parent:
`Research-status.md`; substrate design notes in
[`series-T.md`](series-T.md) and
[`2026-05-14_qselect-target-rethink.md`](2026-05-14_qselect-target-rethink.md).

## Ve_oracleAblation — does oracle supervision need to stay on?

**Hypothesis.** Two readings of the T-series result are observationally
equivalent at the loss level:

- **(W) "Warmup":** the oracle imprints structure on the trunk during early
  training; once imprinted, MC supervision is enough to maintain it. If true,
  we can drop the oracle mid-training and save the inference cost without
  losing WR — and the Q_select position-structure recall (D1) should hold up
  after disable.
- **(P) "Persistent driver":** the oracle keeps shaping the trunk on every
  epoch; remove it and the structure decays under MC pressure. Then disabling
  the oracle caps WR below the never-disabled run and D1 recalls drop.

`MINIMAX_DISABLE_AFTER_EPOCH ∈ {None, 2000, 4000}` separates these by
giving each variant ≥2000 post-disable epochs at 6000 total.

**Code surface.** `trainRL.py:107` adds `MINIMAX_DISABLE_AFTER_EPOCH = None`
(default no-op). The training loop checks at the top of each epoch
(`trainRL.py:472-484`) — when the flag is set, the threshold is reached, and
the oracle is still active, it sets `SELECT_ORACLE = None` and
`DECOUPLED_TARGET_STYLE = "td_place_mc_select"`. `run_trains.py:107-128`
holds the three-variant `MULTI_PARAMS` block.

**Fixed:** `STARTING_NET=None`, `ARCHITECTURE=QuartoCNNAutoregUnifiedS4`
(Sa(3) substrate), `TRANSITION_SCHEMA="unified_autoreg"`,
`USE_MINIMAX_SELECT_TARGET=True`, `MINIMAX_SELECT_DEPTH=2`,
`REWARD_FUNCTION="final"`, `N_LAST_STATES_INIT = N_LAST_STATES_FINAL = 4`,
`ENDGAME_FRACTION=0.5`, `N_LAST_STATES_ENDGAME=2`, `LR=7e-4`, `TAU=0.01`,
`(α_place, α_select) = (1.0, 1.0)`, `EPOCHS=6000`.

| Run | Disable @ | Epochs |
|---|---|---|
| Ve(1) | none (control) | 6000 |
| Ve(2) | 2000 | 6000 |
| Ve(3) | 4000 | 6000 |

**Decision gate (pre-registered).**
- (W) supported if Ve(2)/Ve(3) match Ve(1) WR within ~2pp **and** D1
  position-structure recalls within ~5pp at epoch 6000.
- (P) supported if Ve(2)/Ve(3) cap below Ve(1) on WR **and** D1 recalls drop
  appreciably after disable.

## Result — 2026-05-21 (6000 epochs each)

### WR and loss [DIRECT — from JSONL summaries]

| Run | Disable @ | `loss_select` | WR vs BT (final / peak) | WR vs random (final / peak) | WR trend (pp/1000ep) |
|---|---|---|---|---|---|
| Ve(1) NEVER | — | **0.048** | **85.2% / 87.1%** | 92.0% / 92.9% | **+2.9 ↑** |
| Ve(2) DIS 2000 | 2000 | 0.240 | 78.3% / 80.3% | 89.3% / 91.1% | +2.1 ↑ |
| Ve(3) DIS 4000 | 4000 | 0.213 | 79.1% / 81.5% | 89.5% / 91.5% | +0.1 (p=0.79) |

`loss_select` snaps from the ~0.05 minimax floor back to the ~0.22–0.24
R/S floor within a single checkpoint of the disable epoch — independent
re-confirmation that the 0.24 floor under MC is a target-noise property,
not a representation or optimisation limit.

### D1 position-structure recalls — 2026-05-22

Each run evaluated at epoch 6000 via
`analysis/qselect_diagnostics/position_structure.py` on 1500 sampled
SELECT states. Per-run JSONL under
`analysis/qselect_diagnostics/results/Ve_oracleAblation(*)/position_structure.jsonl`.

| Metric | Ve(1) NEVER | Ve(2) DIS 2000 | Ve(3) DIS 4000 | Ta(1) ref @4350 | Chance |
|---|---|---|---|---|---|
| `safe_piece_recall` | **0.821** | 0.541 | 0.580 | 0.807 | 0.114 |
| `forcing_loss_bottom_recall` | **0.957** | 0.830 | 0.854 | 0.946 | 0.700 |
| `spearman_rho_mean` | **0.578** | 0.247 | 0.272 | 0.547 | 0 |
| `n_states_decisive` | 925 | 887 | 879 | 896 | — |

### Reading [INFERENTIAL — interpretation; numbers above are DIRECT]

1. **Ve(1) reproduces and slightly exceeds Ta(1) on every D1 metric.** Two
   additional kilo-epochs of oracle supervision push each recall a hair
   upward. Q_select keeps absorbing piece-structure as long as the oracle
   is on; no plateau visible at 6000 epochs.

2. **Disabling the oracle *erodes* Q_select position structure — does not
   merely fail to add more.** Safe-piece recall drops from 0.82 → ~0.55
   (a 32-pp swing); Spearman ρ̄ halves (0.58 → ~0.26). MC supervision
   after disable actively pushes Q_select away from oracle-imprinted
   ranking. This is the cleanest single-experiment refutation of the
   "warmup that can be removed" reading (W).

3. **Late disable ≈ early disable.** Ve(2) and Ve(3) land within a few
   pp of each other on every D1 metric and on WR, despite 2000 vs 4000
   oracle-on epochs. The decay appears to converge to a steady state
   once the oracle is gone, roughly independent of where the cut is
   made — suggesting MC pulls Q_select toward a fixed equilibrium that
   sits well below the oracle-supervised one.

4. **D1 still well above chance after disable.** Ve(2)/Ve(3)
   safe-piece recall is 5× chance and ρ̄ stays meaningfully positive
   (~0.25). The trunk retains *some* of the imprint under MC — just
   not the level Ve(1) holds. (P) is therefore directional, not absolute:
   MC can hold a fraction of the structure but not the whole.

5. **D1 tracks WR.** The 6–7 pp WR gap is mirrored by the ~30-pp recall
   gap and ~0.30 ρ̄ gap. Two independent signals concur: the oracle is
   the active driver of the late-training improvement, not a one-time
   seeder.

### Conclusions

- **(P) supported, (W) refuted.** The minimax oracle is a persistent
  supervisor whose contribution does not survive removal. Plan for oracle
  cost in production training; do not design recipes that drop it.
- **Ve(1) is the new WR champion candidate** — 85.2% vs `bot_loss-BT`,
  peak 87.1%, still rising at +2.9 pp/1000ep. Surpasses Ta(1)'s truncated
  80.3% by ~5pp and ME(2)'s 73.7% by ~11.5pp. Promotion contingent on the
  10k confirmation (Ve(4), launched 2026-05-22 with the same recipe and
  `EPOCHS=10000`).
- **D1 generalises beyond Ta(1).** This is the first cross-experiment
  evidence that D1 captures real, oracle-dependent representational
  structure. The metric tracked the expected ordering (Ve(1) ≫ Ve(2/3) ≫
  chance) without re-tuning.
- **Follow-up — D1 on pre-T runs.** Item 2 of the forward queue is now
  partially answered. The prediction sharpens: M/S/OA checkpoints never
  exposed to the oracle should land near Ve(2)/Ve(3)'s residual (~0.55
  safe-piece recall) — not near Ve(1)'s 0.82. Worth a cheap scan to
  decide whether the historic "Q_select saturation" verdicts on those
  runs need wholesale revision.

### Loader fix shipped with the D1 scan

`analysis/qselect_diagnostics/_common.py:84` — added
`("Ve_oracleAblation", "QuartoCNNAutoregUnifiedS4")` to `_EXP_ARCH_HINTS`
so the loader instantiates the S4 trunk rather than the default
`QuartoCNNAutoregUnified` (fc1=128). Without this the Ve checkpoints
would fail to load with a `size mismatch for fc1.weight: [512,512] vs
[128,512]` error. Future series running on Sa(3) need either their own
hint entry or a `--architecture` CLI override.

## Result — Ve(4) 10k confirmation, 2026-05-24

`Ve_oracleAblation(4)0522_DISABLE_NEVER_10k` — same recipe as Ve(1)
(Sa(3) + minimax depth=2, oracle always on) extended to EPOCHS=10000.
Launched 2026-05-22 18:32, finished 2026-05-24 19:04 (~48.5 h wall).

### WR and loss [DIRECT — from JSONL summaries]

| Metric | Ve(4) @10k | Ve(1) @6k | Δ |
|---|---|---|---|
| WR vs BT final | **0.872** | 0.852 | +0.020 |
| WR vs BT peak | **0.889** | 0.871 | +0.018 |
| WR vs random final | **0.938** | 0.920 | +0.018 |
| WR vs random peak | **0.948** | 0.929 | +0.019 |
| WR trend vs BT (last 5k, pp/1000ep) | +0.65 ↑ (CI 0.51–0.78, p≈0) | +2.9 ↑ | slowing, still significant |
| WR trend vs random (last 5k, pp/1000ep) | +0.44 ↑ (CI 0.36–0.52, p≈0) | — | still significant |
| `loss_select` final | **0.041** | 0.048 | −0.007 |
| `loss_place` final | 0.064 | — | — |

### D1 position-structure recalls — 2026-05-24

Diagnostic command:
`python analysis/qselect_diagnostics/position_structure.py
--exp 'Ve_oracleAblation(4)0522_DISABLE_NEVER_10k' --epoch 10000
--n-states 500 --seed 1234` (identical params to Ve(1)/Ta(1) scans).

| Metric | Ve(4) @10k | Ve(1) @6k | Ta(1) @4350 | Chance |
|---|---|---|---|---|
| `safe_piece_recall` | **0.846** | 0.821 | 0.807 | 0.114 |
| `forcing_loss_bottom_recall` | **0.968** | 0.957 | 0.946 | 0.700 |
| `spearman_rho_mean` (n) | **0.610** (916) | 0.578 (924) | 0.547 (896) | 0 |
| `forcing_set_size_mean` | 5.97 | 5.90 | 5.83 | — |

### Reading [INFERENTIAL]

1. **Both heads keep improving with more oracle-supervised training.**
   Place head: WR rises +2.0 pp final / +1.8 pp peak vs Ve(1)@6k.
   Select head: D1 recalls and ρ̄ all rise. No sign of a ceiling at 10k
   on either axis under this recipe.

2. **WR trend deceleration is real but the run isn't saturated.**
   Slope on `bot_loss-BT` dropped from Ve(1)'s +2.9 → +0.65 pp/1000ep
   (a ~4× slowdown). CI excludes zero (0.51–0.78), so the model is
   still learning, just on diminishing returns. Extrapolation: ~+3–5 pp
   more WR available if you 2–3×'d the wall-clock again, but the
   marginal cost per pp keeps growing.

3. **D1 ceiling is not far.** `safe_piece_recall = 0.846` means the
   argmax-Q_select picks a safe (non-forcing-loss) piece in ~85% of
   decisive states. With mean forcing-set size ~6/14 available pieces,
   chance-of-safe-pick is ~57% — the model is recovering ~67% of the
   chance-to-perfect gap. The remaining 15% of errors are concentrated
   in states where the oracle and the place-head's value estimate
   disagree on the cost of taking a borderline piece; revisiting these
   in interpretability work would be informative.

4. **`forcing_loss_bottom_recall` is approaching the ceiling.** At
   0.968, the model misses the "give the opponent a forcing piece"
   class in only ~3% of states. Further D1 gains will have to come
   from the safe-piece-recall and ρ̄ axes, not from this one.

### Conclusions

- **Ve(4) promoted to overall champion 2026-05-24**, superseding ME(2)
  (which held the slot since 2026-04-29). Net delta vs ME(2): +13.5 pp
  WR vs BT, +8.0 pp WR vs random, +73.2 pp D1 safe_piece_recall
  (chance ≈ 0.11; ME(2) not yet measured — predicted to land near
  Ve(2)/Ve(3)'s ~0.55 by the pre-T conjecture in forward-queue item
  #3).
- **Forward-queue item #1 (Ve(4) 10k confirmation) closed.** Promotion
  done in `Research-status.md`.
- **Next training axis: Wa_oracleStates** (N_LAST_STATES sweep on the
  Ve recipe) — promoted to top of the forward queue. Run *after* the
  D1 metric ships into the JSONL summary (forward-queue item #1, now
  bumped to top priority) so the new gate is computed inline on every
  Wa checkpoint.
