# Series R — Per-head loss reweighting

OA-family substrate (`QuartoCNNAutoregUnified`, `unified_autoreg`,
`td_place_mc_select`), N=4 constant, ME(2) recipe in every other respect.
Sweep over `(LOSS_ALPHA_PLACE, LOSS_ALPHA_SELECT)`. See parent:
`Research-status.md`.

## Ra_lossWeight — Static (α_place, α_select) sweep

**Hypothesis:** Q_select saturation could be a *gradient-starvation* artefact
of the joint loss `L = (α_p · L_place + α_s · L_select) / (α_p + α_s)`. If the
select head is under-supervised relative to place, raising α_select (or
lowering α_place) should reduce `loss_select`, push Q_select winners-mean
upward, and — if the saturation is the only thing holding WR back — leave or
improve WR vs `bot_loss-BT`.

**Fixed:** `STARTING_NET=None`, `ARCHITECTURE=QuartoCNNAutoregUnified`,
`TRANSITION_SCHEMA="unified_autoreg"`,
`DECOUPLED_TARGET_STYLE="td_place_mc_select"`,
`REWARD_FUNCTION="final"`, `N_LAST_STATES_INIT = N_LAST_STATES_FINAL = 4`,
`ENDGAME_FRACTION=0.5`, `N_LAST_STATES_ENDGAME=2`, `LR=7e-4`, `TAU=0.01`,
`EPOCHS=5000`.

| Run | α_place | α_select | Epochs |
|-----|---------|----------|--------|
| Ra(1) | 1.0 | 1.0 | 5000 |
| Ra(2) | 1.0 | 3.0 | 5000 |
| Ra(3) | 0.3 | 1.0 | 5000 |
| Ra(4) | 0.1 | 1.0 | 5000 |

**Compare to:** `ME_endgame(2)_E_5000` (champion), `OA_unifiedAux(1)`
(prior unified-aux baseline), `bot_random`.

**Decision gate (pre-registered):**
- Q_select winners-minus-losers Δ ≥ +0.40 on at least one cell of the grid, AND
- WR vs `bot_loss-BT` not collapsed (≥ 60%) on the cell that achieves it.

**Result (2026-05-12, 5000 epochs each):**

| Run | α_p / α_s | loss_select | Q_select Δ (w−l) | Q_place Δ | WR vs BT (final) | WR vs BT (peak) |
|---|---|---|---|---|---|---|
| Ra(1) | 1.0 / 1.0 | 0.246 | +0.010 | +0.548 | 71.3% | 72.7% |
| Ra(2) | 1.0 / 3.0 | 0.245 | +0.030 | +0.302 | 64.2% | 66.9% |
| Ra(3) | 0.3 / 1.0 | 0.256 | +0.033 | +0.303 | 65.1% | 66.8% |
| Ra(4) | 0.1 / 1.0 | 0.242 | +0.044 | +0.122 | 48.1% | 49.7% |

Three signals, all negative for the gradient-starvation hypothesis:

1. **`loss_select` is flat at ~0.24 across the entire grid.** A 30× swing in
   relative weight (Ra(1) → Ra(4)) moves it by less than 0.02. The select
   target has a representational floor that the optimiser cannot cross by
   pushing harder on it. This is the cleanest version of the
   "loss-floor-not-gradient-floor" diagnosis we have.
2. **Q_select Δ inches up monotonically with α_select / α_place, but
   asymptotes near +0.04** — still ~10× below the +0.40 gate. The select head
   responds *slightly* to more weight, but the response is bounded.
3. **WR collapses monotonically as α_place drops.** Q_place Δ falls from
   +0.548 at Ra(1) to +0.122 at Ra(4); WR vs BT tracks it from 71.3% → 48.1%.
   The trade is one-way: each unit of α_select buys a fraction of a Q_select
   point at the direct cost of place-head separation and WR.

**Trend note.** WR curves for Ra(2)–Ra(4) are still rising at epoch 5000;
peak ≈ final + 1–3 pp on those three. The ranking does not change with the
peak metric — Ra(1) still wins WR, Ra(4) still wins Q_select Δ. Extending to
10k epochs would refine the numbers but not the conclusion.

**Conclusion.** Static reweighting fails its decision gate on every cell.
Gradient-starvation is **rejected** as the mechanism behind Q_select
saturation: the select head is not under-supervised, it is mis-targeted —
`loss_select`'s floor is in the *targets*, not the gradient. Loss-weight
asymmetry only converts WR into a marginal Q_select gain at a poor exchange
rate. The next two threads pick this up:

- **Sa_archScan** — `series-S.md`. Does a different trunk structure expose a
  representation that lets the select head learn against the same MC return
  target?
- **`2026-05-14_qselect-target-rethink.md`** — design note. If `loss_select`
  has a floor at the *target* level, the noisy MC return on select actions
  (player-perspective ±1 propagated back over many steps from a piece-give
  whose decisiveness is buried in 5–10 subsequent moves) is the candidate.
  Proposes a minimax-oracle label for select.

**Rb queued (not run).** Time-varying schedule on (α_place, α_select):
start at (1.0, 1.0) — the only Ra cell that recovers Q_place — for the first
~2000 epochs, then ramp α_select up to 3.0 once Q_place has stabilised. Ra
ran every cell with the asymmetric weighting *from epoch 0*; the question of
whether a warm Q_place trunk can absorb select pressure without losing place
fidelity is genuinely untested. Cheap to run; ranks behind the target-rethink
because the Ra evidence points at targets, not optimisation.
