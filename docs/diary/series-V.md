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
