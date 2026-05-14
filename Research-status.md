# Quarto RL — Research status

Slim ledger. Long-form per-series content lives in [`docs/diary/`](docs/diary/).
Add a new series entry there when starting a new code-version letter; append
result paragraphs here under "Recent results" when a sweep concludes.

---

## Champion

- **`ME_endgame(2)0429_ENDGAME_FRACTION_0.5`** — 73.7% WR vs `bot_loss-BT`,
  85.8% WR vs `bot_random`, loss 0.207. Uses the M-series decoupled-autoreg
  trunk (`QuartoCNNAutoreg`) with `td_place_mc_select` targets,
  N_LAST_STATES_INIT=2 curriculum to N=4, ENDGAME_FRACTION=0.5,
  N_LAST_STATES_ENDGAME=2, LR=7e-4, TAU=0.01, 5000 epochs.
- **Interpretability substrate of record:** `OA_unifiedAux(1)0509_N_LAST_STATES_INIT_2`
  — 66.3% WR vs `bot_loss-BT`. Same loss / schema family as ME(2) but with the
  unified 32-d phase-stable aux (`[offered_one_hot ; available_mask]`) and no
  phase embedding. 7pp behind ME(2) on WR; standing trade for SAE-able activations.
- **Candidate replacement (pending confirmation):**
  `Sa_archScan(3)0512_ARCH_S4_uniform512` — 73.5% WR vs `bot_loss-BT`, 86.4%
  vs `bot_random` (curve still rising at 5000 epochs). Matches ME(2) on WR
  with the unified-aux substrate; closes the 7pp interp tax. Promote after
  a 10k-epoch confirmation. Full write-up: `docs/diary/series-S.md`.

## Current Open Problem — Q_select saturation

`Q_select` collapses near −1 across every architecture and target style we
have tried (M, N, O, Q, R, S) — the place head carries WR while the select
head contributes ~0 to value separation. Triangulation as of 2026-05-14:

- **Head only (Pa_frozenTrunkSelect):** freezing the ME(2) trunk and retraining
  only `fc2_select` lifts the winners-mean by ≈ +0.20, not the ≥ +0.40 the gate
  required. **Head retraining alone is insufficient.**
- **Asymmetric LR (Nb_asymLR):** 10× LR on the select head produced no separation.
- **Wider `fc1` + aux legality + no inference mask (QC_unifiedNoMask):** the
  aux BCE collapsed (0.69 → 0.15) but `invalid_argmax_rate` stayed at 0.32 and
  `Q_select` reproduced the same −0.73 saturation.
- **Static per-head loss reweighting (Ra_lossWeight):** `loss_select` is
  pinned at ~0.24 across a 30× swing in α_select / α_place. Reweighting
  buys a marginal Q_select Δ (≤ +0.04) at a one-way cost in WR.
  **Gradient-starvation rejected.**
- **Structural trunk variants (Sa_archScan):** deeper conv (Sa(1)) produced
  the largest Q_select Δ on record (+0.170), but as a numeric statistic on
  outcome means — *not* visible plate separation in the qv heatmaps — and
  with a 30pp WR cost (curve still rising at 5000 epochs). Wider/uniform
  FC (Sa(3)) matched ME(2) WR with the unified-aux substrate but produced
  no Q_select gain.

Two triangulated conclusions:

1. The select head responds slightly to trunk capacity changes but the
   `loss_select` floor is invariant to weighting, architecture, and head-only
   retraining. The floor lives in the **targets**, not the gradient or the
   architecture.
2. Under `td_place_mc_select` with `REWARD_FUNCTION="final"`, the Q_select
   target is the MC return from a select action — player-perspective ±1
   discounted from terminal. `Q_select=−1` correctly labels piece-gives that
   directly caused a loss; `Q_select=+1` labels selects up to 6+ moves
   before an eventual win, where the credit is buried under many subsequent
   decisions. The label distribution conflates decisive and incidental
   selects.

Next direction: **minimax-oracle labels for the select head**
(supervised distillation of `Q_select`, RL unchanged for `Q_place`). See
[`docs/diary/2026-05-14_qselect-target-rethink.md`](docs/diary/2026-05-14_qselect-target-rethink.md).

## Code-version letters (active and queued)

| Letter | Series | Architecture / Schema | Status |
|---|---|---|---|
| A–E | early / combined_avg baseline | joint | done — see `docs/diary/series-A.md` |
| F | adversarial sign flip | joint | failed/reverted — `series-FG.md` |
| G | separate_bellman | joint | done — `series-FG.md` |
| H–K | terminal mask / unbound / final / coupled | joint | done — `series-HIJK.md` |
| L | mc_select | joint | done — `series-L.md` |
| M | decoupled_autoreg | `QuartoCNNAutoreg*` + `Quarto_autoreg_bot` | **champion** — `series-M.md` |
| N | shared-trunk diagnostics | M-series | done — `series-N.md` |
| O | unified_autoreg | `QuartoCNNAutoregUnified*` + `Quarto_unified_bot` | done — `series-O.md`, `docs/diary/2026-05-08_unified-aux-trunk.md` |
| P | frozen-trunk select | M-series | done (negative) — `series-P.md` |
| Q | unified + aux legality + no mask | `QuartoCNNUnifiedNoMask` + `Quarto_unified_nomask_bot` | done (gate failed) — `series-Q.md`, `docs/diary/2026-05-11_qc-no-mask.md` |
| R | per-head loss weighting | OA-family | done (gate failed) — `series-R.md` |
| S | structural trunk variants | OA-family | done (mixed) — `series-S.md`; Sa(3) candidate new interp champion |
| **T** | **minimax-oracle select target** | OA-family / Sa(3) | **proposed** — `docs/diary/2026-05-14_qselect-target-rethink.md` |

## Recent results

### Sa_archScan — 2026-05-12 (5000 epochs each, 3 runs)

| Run | Trunk | Q_select Δ | Q_place Δ | WR vs BT (final / peak) |
|---|---|---|---|---|
| Sa(1) | deepConv | **+0.170** (largest on record, numeric only — no plate separation in heatmaps) | +0.231 | 44.3% / 47.8% (still rising) |
| Sa(2) | wideFC | +0.072 | +0.332 | 63.7% / 66.6% |
| Sa(3) | uniform512 | +0.037 | +0.457 | **73.5% / 75.2%** |

Sa(3) matches the M-series champion on WR while keeping the unified-aux
substrate (+7pp over OA(1) interp champion at no WR cost) — promoted to
**interpretability substrate of record (pending confirmation run)**. Sa(1)
moved Q_select Δ numerically but not visibly; WR curve still rising at
5000 epochs — direction not retired. Full write-up in
[`docs/diary/series-S.md`](docs/diary/series-S.md).

### Ra_lossWeight — 2026-05-12 (5000 epochs each, 4 runs)

| α_place / α_select | loss_select | Q_select Δ | WR vs BT (final / peak) |
|---|---|---|---|
| 1.0 / 1.0 | 0.246 | +0.010 | 71.3% / 72.7% |
| 1.0 / 3.0 | 0.245 | +0.030 | 64.2% / 66.9% (rising) |
| 0.3 / 1.0 | 0.256 | +0.033 | 65.1% / 66.8% (rising) |
| 0.1 / 1.0 | 0.242 | +0.044 | 48.1% / 49.7% (rising) |

`loss_select` floor flat at ~0.24 across the grid; WR trades 1:1 against
α_select. **Gradient starvation rejected** as the mechanism. Full write-up
in [`docs/diary/series-R.md`](docs/diary/series-R.md).

### QC_unifiedNoMask(1) — 2026-05-12 (5000 epochs)

| Metric | Gate | Observed | Pass? |
|---|---|---|---|
| WR vs `bot_random` | ≥ 95% **hard** | 85.0% (peak 86.1%) | **no** |
| WR vs `ME(2)_E_5000` | ≥ 50% | 46.2% (peak 48.7%) | **no** |
| `invalid_argmax_rate` (last 100) | < 0.05 | 0.324 (start 0.32, no movement) | **no** |
| `legality_loss` (BCE) | qualitative | 0.69 → 0.15 | aux head fits cleanly |

Aux head decoupled from the Q heads at λ_legality=0.05; do not iterate on Q3/Q4
from QC. Full write-up in [`docs/diary/series-Q.md`](docs/diary/series-Q.md).

### Pa_frozenTrunkSelect(1) — 2026-05-12 (1000 epochs)

| Metric | Gate | Observed | Pass? |
|---|---|---|---|
| Q_select winners Δ | ≥ +0.40 | +0.20 (−0.71 → −0.51) | **no** |
| WR vs `bot_random` | ≥ 90% | 87.0% | marginal |
| WR vs `ME(2)` | ≥ 45% | 53.9% | yes |

Saturation is **not** a head-level pathology. Full write-up in
[`docs/diary/series-P.md`](docs/diary/series-P.md).

## Forward queue

1. **Ta_minimaxSelect** — *new direction.* Supervised distillation of
   `Q_select` from a minimax-oracle label (depth 4 starting point);
   `Q_place` continues to learn from `td_place_mc_select` Bellman targets.
   Substrate: OA-family or Sa(3) post-confirmation. Gate: `loss_select` ≤
   0.10, Q_select Δ ≥ +0.40, WR vs `bot_loss-BT` ≥ 70%. Design note:
   [`docs/diary/2026-05-14_qselect-target-rethink.md`](docs/diary/2026-05-14_qselect-target-rethink.md).
2. **Sa(3) confirmation run** — re-run `Sa_archScan(3)0512_ARCH_S4_uniform512`
   at 10k epochs (curve still rising at 5k) to lock the interpretability-
   champion promotion. Cheap.
3. **Sb_hybridTrunk** — Sa(3) uniform-512 FC with Sa(1) deeper conv stack on
   top. Tests whether the WR carrier (Sa(3) via Q_place) and the numeric
   Q_select gain (Sa(1)) compose. Lower priority than Ta — orthogonal axis.
4. **Rb_schedAlpha** — time-varying (α_place, α_select). Start (1.0, 1.0)
   for ~2000 epochs, then ramp α_select. Tests whether a warm Q_place trunk
   can absorb select-loss pressure that the cold-start Ra grid could not.
   Lowest priority — Ra's `loss_select` floor is targets-side evidence; Rb
   is the optimisation-side hedge.

All runs use the per-head loss / grad logging and JSONL emission from
2026-05-12.

---

## Experiment Naming Convention

Two-part name: **`XY_description`**.

- **First letter (X)** = code version / significant algorithm change. Currently
  in use: A–N, O (unified aux), P (frozen trunk), Q (aux legality + no mask),
  R (loss weighting — queued), S (structural — queued).
- **Second letter (Y)** = hyperparameter sweep within the same code version:
  `a`, `b`, `c`, ...

`run_trains.py` appends `(<idx>)<MMDD>_<PARAM>_<VALUE>` to produce the per-run name (e.g. `MB_final(3)0426_N_LAST_STATES_INIT_4`).

## Operational notes (load-bearing)

- **`mode_2x2=True` is the default** across training and evaluation.
- **`TRANSITION_SCHEMA` is a triplet contract** (schema × model class × bot class). See `CLAUDE.md` for the valid combinations.
- **Pickles may contain CUDA tensors** — use `CPUUnpickler` (see `tools/view_qv.py` or `QuartoRL/results_io.py`) on CPU-only machines.
- **`-1` sentinels in the experience tuple are load-bearing** (`action_place=-1` first move, `action_sel=-1` terminal). Don't filter them out.
- **JSONL results format** is the new source of truth for cross-experiment comparison; see `tools/results_compare.py`. Legacy pickles are backfilled via `tools/pkl_to_jsonl.py`.

## Key takeaways (consolidated)

1. **N=2 starves Q_select of positive samples.** The decoupled-autoreg N=2 window contains only `{loser_place, loser_select, winner_terminal_place}` — zero winner-side select transitions. N≥3 fixes availability, not learning.
2. **Q_select saturation persists across joint/decoupled/unified schemas and across head-only / trunk-wider / aux-legality fixes.** See Open Problem above.
3. **Curriculum jumps from N=2 to higher N catastrophically forget** (Ac_fineShallow). Replay-buffer distribution shift, not LR.
4. **Endgame anchor buffer (ENDGAME_FRACTION=0.5, N_LAST_STATES_ENDGAME=2)** is the single change that unlocked 73% WR (M-series → ME).
5. **Loss reweighting (R) is rejected as a fix for Q_select saturation.** `loss_select` has a floor near 0.24 invariant to a 30× swing in α_select / α_place; reweighting buys Q_select Δ ≤ +0.04 at a one-way WR cost. Gradient starvation is not the mechanism.
6. **Structural trunk changes (S) close the unified-aux WR tax but not Q_select saturation.** Sa(3) matches ME(2) WR with the unified-aux substrate (+7pp over OA(1)) — new interpretability champion candidate. Sa(1) deepConv produced the largest numeric Q_select Δ on record (+0.170) but with no visible plate separation and a WR collapse. The select-side gap is not where deeper trunks alone close it.
7. **The next candidate is target noise on Q_select.** Under `td_place_mc_select` + `REWARD_FUNCTION="final"`, Q_select=+1 is the MC return label on selects that occurred up to 6+ moves before the eventual win — credit assignment buried under subsequent decisions. Proposed remedy: minimax-oracle labels for the select head (T-series). Design note: `docs/diary/2026-05-14_qselect-target-rethink.md`.

For each of these, the originating series file in `docs/diary/` carries the full evidence.
